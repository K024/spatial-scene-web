/**
 * meshing 模块入口：**LayeredRGBD -> MeshScene**。
 *
 * 阶段：步骤②（`image -> splat -> layered RGBAD -> mesh 场景`）的最后一环。
 * 产物是**与渲染器无关的内存数据**，交付给 web 端 WebGL2 / three.js。
 *
 * ── 模块边界（硬约束）──
 * - 本模块只出内存数据，**不落盘、不打包图集、不压缩、不序列化元信息**；
 *   那些归下游 packing / export 模块。
 * - `src/**` 内不得出现 WebGPU 依赖；交付渲染器是 WebGL2 / three.js。
 *   node 侧 WebGPU / CPU 光栅器只作 `scripts/` 下的测试基准，不是交付路径。
 * - 默认不新增 npm 依赖；数学复用 `sharp/linalg.ts` 与 `wsplat/camera.ts`。
 *
 * ── 每层的流水线（顺序不可换）──
 * ```
 * 深度 z --(ndcDepthFromZ)--> 视差 n
 *   1. computeSupportMask        : 迟滞(α)支撑掩码 + 小岛剔除       -> support
 *   2. expandSupportWithMargin   : 外缘余量 + 源像素映射             -> mask / source
 *   3. computeTornEdges          : ★ 视差域断层撕裂 + despeckle      -> 撕裂边表
 *   4. buildLayerRelief          : 反投影 + 出四边形 + 裙边断壁      -> LayerMesh
 * ```
 * 撕裂必须在余量**之后**：余量环也要按源深度参与撕裂，否则外扩会把断层糊上。
 * 裙边在**最后**：它消费「哪些四边形发射了」，才能认出网格边界边。
 *
 * ── 全场收尾 ──
 * 所有层之后追加**背衬平面**：L 层 back-to-front 合成 → α 加权降采样 → 最远深度 quad。
 *
 * ── 文件划分 ──
 * `types.ts` 契约 / `tolerance.ts` 支撑 / `margin.ts` 余量 / `tears.ts` 撕裂 /
 * `relief.ts` 网格 / `backfill.ts` 回填。
 *
 * ── 硬要求 ──
 * 第 3 步是唯一不可简化的正确性要求：深度断层必须被正确撕裂，否则斜视角下出现连接
 * 近远面的「橡皮布」大三角。详见 `tears.ts` / `types.ts` 头部。
 */

import { buildBackingPlane } from "./backfill.ts"
import { expandSupportWithMargin } from "./margin.ts"
import { buildLayerRelief, resolveSkirtDisparity } from "./relief.ts"
import { computeDisparityField, computeTornEdges } from "./tears.ts"
import { computeSupportMask } from "./tolerance.ts"
import type {
  LayerMesh,
  LayerMeshReport,
  MeshingInput,
  MeshingOptions,
  MeshScene,
} from "./types.ts"

export type { BackfillFrame } from "./backfill.ts"
export {
  buildBackingPlane,
  buildBackingPlaneMesh,
  compositeLayersBackToFront,
  downscaleAlphaWeighted,
} from "./backfill.ts"
export { toMeshingInput } from "./input.ts"
export type { MarginResult } from "./margin.ts"
export { expandSupportWithMargin } from "./margin.ts"
export type { LayerReliefStats, ReliefFrame } from "./relief.ts"
export { buildLayerRelief, resolveSkirtDisparity } from "./relief.ts"
export type { TornEdges } from "./tears.ts"
export {
  computeDisparityField,
  computeTornEdges,
  resolveTearEps,
} from "./tears.ts"
export type { SupportMaskFrame, SupportMaskResult } from "./tolerance.ts"
export { computeSupportMask } from "./tolerance.ts"
export type {
  AlphaToleranceOptions,
  BackfillOptions,
  LayerMesh,
  LayerMeshReport,
  LayerReliefContext,
  MarginOptions,
  MeshingInput,
  MeshingOptions,
  MeshReport,
  MeshScene,
  ReliefOptions,
  RgbaTexture,
  SkirtOptions,
  TearOptions,
} from "./types.ts"

/**
 * 把分层 RGBAD 变成 mesh 场景。
 *
 * 纯 CPU、确定性：同一输入 + 同一选项两次调用**逐位相同**（`layering` 的同一约定）。
 */
export function buildMeshScene(
  input: MeshingInput,
  options: MeshingOptions = {},
): MeshScene {
  const { width, height, near, far, placement, frames, ranges, stats, camera } =
    input
  const L = frames.length
  if (placement.L !== L) {
    throw new Error(
      `meshing：placement.L(${placement.L}) != frames.length(${L})`,
    )
  }
  if (placement.boundaries.length !== L + 1) {
    throw new Error(
      `meshing：boundaries 长度应为 ${L + 1}，收到 ${placement.boundaries.length}`,
    )
  }
  if (placement.boundariesZ.length !== L + 1) {
    throw new Error(
      `meshing：boundariesZ 长度应为 ${L + 1}，收到 ${placement.boundariesZ.length}`,
    )
  }
  if (camera.width !== width || camera.height !== height) {
    throw new Error(
      `meshing：相机尺寸 ${camera.width}x${camera.height} 与场景 ${width}x${height} 不一致`,
    )
  }

  const marginRadius = Math.max(
    0,
    Math.floor(options.margin?.radiusPixels ?? 0),
  )
  const skirt = options.skirt === false ? null : (options.skirt ?? {})

  const layers = new Array<LayerMeshReport>(L)
  const meshes = new Array<LayerMesh>(L)
  const pixels = width * height

  for (let k = 0; k < L; k++) {
    const frame = frames[k]
    if (frame.width !== width || frame.height !== height) {
      throw new Error(
        `meshing：第 ${k} 层帧尺寸 ${frame.width}x${frame.height} 与场景 ${width}x${height} 不一致`,
      )
    }
    const bandWidth = placement.boundaries[k + 1] - placement.boundaries[k]

    // 1) α 迟滞支撑 + 小岛剔除
    const support = computeSupportMask(frame, options.alpha)
    // 2) 余量：向外膨胀，并给出逐像素源（uv / 深度来源）
    const margin = expandSupportWithMargin(
      support.support,
      width,
      height,
      marginRadius > 0 ? { radiusPixels: marginRadius } : {},
    )
    // 源深度场：环像素取最近支撑像素的深度（identity 时就是自己的深度）
    const sourceDepth = new Float32Array(pixels)
    for (let i = 0; i < pixels; i++) {
      const s = margin.source[i]
      sourceDepth[i] = s >= 0 ? frame.depth[s] : 0
    }
    // 3) 撕裂：视差域阈值 + despeckle
    const disparity = computeDisparityField(sourceDepth, near, far)
    const torn = computeTornEdges(
      disparity,
      margin.mask,
      width,
      height,
      bandWidth,
      options.tears,
    )
    // 4) 顶点 / 四边形 / 裙边
    const skirtDisparity = skirt ? resolveSkirtDisparity(bandWidth, skirt) : 0
    const { mesh, stats: reliefStats } = buildLayerRelief(
      frame,
      camera,
      margin.mask,
      torn,
      k,
      [placement.boundaries[k], placement.boundaries[k + 1]],
      [placement.boundariesZ[k], placement.boundariesZ[k + 1]],
      options.relief,
      {
        source: margin.source,
        disparity,
        near,
        far,
        skirtDisparity,
      },
    )
    meshes[k] = mesh
    layers[k] = {
      layerIndex: k,
      supportPixels: support.supportPixels,
      removedIslandPixels: support.removedIslandPixels,
      marginPixels: margin.marginPixels,
      tearEps: torn.tearEps,
      tornEdges: torn.tornCount,
      despeckledEdges: torn.despeckledCount,
      quadsEmitted: reliefStats.quadsEmitted,
      quadsSkipped: reliefStats.quadsSkipped,
      boundaryEdges: reliefStats.boundaryEdges,
      wallTriangles: reliefStats.wallTriangleCount,
      skirtDisparity,
    }
  }

  // 全场收尾：背衬平面（所有层之后）
  const backingPlane = buildBackingPlane(
    frames,
    camera,
    placement.boundariesZ[L],
    [0, 1],
    [near, far],
    options.backing,
  )

  return {
    layers: meshes,
    backingPlane,
    // 以下均**引用输入**，不拷贝（见 types.ts 的内存复用约定）。
    layerDepths: placement.layerDepthsZ,
    layerRanges: ranges,
    near,
    far,
    verticalFOV: camera.fovY,
    aspectRatio: width / height,
    stats,
    premultipliedAlpha: false,
    report: { layers },
  }
}
