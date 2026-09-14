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
 *   2. computeTornEdges          : ★ 视差域断层撕裂 + 去噪 + despeckle + 小面片门 -> 撕裂边表
 *   3. buildLayerRelief          : 反投影 + 出四边形 + 裙边断壁      -> LayerMesh
 * ```
 * 裙边在**最后**：它消费「哪些四边形发射了」，才能认出网格边界边。
 *
 * ── 层外缘 / 遮挡区的几何补齐去哪了 ──
 * 旧版的 `margin.ts`（固定半径屏幕空间膨胀 + 最近源深度拷贝）已**移除**。
 * 它现在由 `layering/refine.ts` 的几何补齐（own-gap 回填 + hidden 有界外推）承担：
 * 产出的是**真实 RGBD**（含深度），meshing 只需消费补齐后的层。
 *
 * ── LOD 出面（压面数）──
 * `options.lod` 打开时，第 2/3 步换成 `lod.ts` 的**受限四叉树**自适应出面：
 * 支撑 / 撕裂仍是硬约束（严格只在本层 α 可见范围内出面），但支撑边界与撕裂处的
 * 最小格子可粗到 `minCellPx`，平滑内部按视差误差 `maxError` 合并成大四边形。
 * 这条路**不生成裙边**（裙边属视角兜缝），面数从百万级降到 10–100k 量级。
 * 缺省关闭 = 现有逐像素出面（golden 基线）。
 *
 * ── 全场收尾 ──
 * 所有层之后追加**背衬平面**：L 层 back-to-front 合成 → α 加权降采样 → 最远深度 quad。
 *
 * ── 文件划分 ──
 * `types.ts` 契约 / `tolerance.ts` 支撑 / `tears.ts` 撕裂 /
 * `relief.ts` 网格 / `backfill.ts` 回填。
 *
 * ── 硬要求 ──
 * 第 3 步是唯一不可简化的正确性要求：深度断层必须被正确撕裂，否则斜视角下出现连接
 * 近远面的「橡皮布」大三角。详见 `tears.ts` / `types.ts` 头部。
 */

import { buildBackingPlane } from "./backfill.ts"
import { buildLayerReliefLod } from "./lod.ts"
import { buildLayerRelief, resolveSkirtDisparity } from "./relief.ts"
import {
  computeDisparityField,
  computeTornEdges,
  resolveLayerTearOptions,
} from "./tears.ts"
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
export type { LodFrame, LodMeshResult, LodOptions, LodStats } from "./lod.ts"
export { buildLayerReliefLod } from "./lod.ts"
export type { LayerReliefStats, ReliefFrame } from "./relief.ts"
export { buildLayerRelief, resolveSkirtDisparity } from "./relief.ts"
export type { TornEdges } from "./tears.ts"
export {
  computeDisparityField,
  computeTornEdges,
  denoiseDisparity,
  resolveLayerTearOptions,
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

    // 1) α 迟滞支撑 + 小岛剔除（几何补齐已在 layering/refine.ts 做完；meshing 不再做外缘膨胀）
    const support = computeSupportMask(frame, options.alpha)
    // 2) 撕裂：视差域阈值 + despeckle
    const disparity = computeDisparityField(frame.depth, near, far)
    const torn = computeTornEdges(
      disparity,
      support.support,
      width,
      height,
      bandWidth,
      resolveLayerTearOptions(options.tears, k, L),
    )
    // 3) 顶点 / 四边形 / 裙边
    const skirtDisparity = skirt ? resolveSkirtDisparity(bandWidth, skirt) : 0
    const lodOptions = options.lod === false ? undefined : options.lod
    if (lodOptions) {
      // LOD 出面：受限四叉树自适应（无裙边；裙边属视角兜缝，见 lod.ts）。
      const lod = buildLayerReliefLod(
        frame,
        camera,
        support.support,
        disparity,
        torn,
        lodOptions,
      )
      const opaqueTriangleCount = countOpaqueTriangles(
        lod,
        frame.alpha,
        width,
        height,
      )
      meshes[k] = {
        layerIndex: k,
        width,
        height,
        vertexCount: lod.stats.vertexCount,
        triangleCount: lod.stats.triangleCount,
        wallTriangleCount: 0,
        opaqueTriangleCount,
        positions: lod.positions,
        uvs: lod.uvs,
        indices: lod.indices,
        texture: { width, height, rgb: frame.rgb, alpha: frame.alpha },
        disparityRange: [
          placement.boundaries[k],
          placement.boundaries[k + 1],
        ] as [number, number],
        depthRange: [
          placement.boundariesZ[k],
          placement.boundariesZ[k + 1],
        ] as [number, number],
      }
      layers[k] = {
        layerIndex: k,
        supportPixels: support.supportPixels,
        removedIslandPixels: support.removedIslandPixels,
        tearEps: torn.tearEps,
        tornEdges: torn.tornCount,
        despeckledEdges: torn.despeckledCount,
        quadsEmitted: lod.stats.leaves,
        quadsSkipped: 0,
        boundaryEdges: 0,
        wallTriangles: 0,
        skirtDisparity: 0,
        maxTriangleError: lod.stats.maxTriangleError,
        minCellViolations: lod.stats.minCellViolations,
        errorSkippedLeaves: lod.stats.errorSkippedLeaves,
      }
      continue
    }
    const { mesh, stats: reliefStats } = buildLayerRelief(
      frame,
      camera,
      support.support,
      torn,
      k,
      [placement.boundaries[k], placement.boundaries[k + 1]],
      [placement.boundariesZ[k], placement.boundariesZ[k + 1]],
      options.relief,
      {
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

/** 三个角纹理 α 都 ≥ `opaqueAlpha` 的三角形数（供 `SeparateOpaqueGeometry`）。 */
function countOpaqueTriangles(
  mesh: {
    readonly uvs: Float32Array
    readonly indices: Uint32Array
  },
  alpha: Float32Array,
  width: number,
  height: number,
  opaqueAlpha = 0.99,
): number {
  const triCount = mesh.indices.length / 3
  const texel = (v: number): number => {
    const x = Math.min(
      width - 1,
      Math.max(0, Math.round(mesh.uvs[v * 2] * width - 0.5)),
    )
    const y = Math.min(
      height - 1,
      Math.max(0, Math.round(mesh.uvs[v * 2 + 1] * height - 0.5)),
    )
    return y * width + x
  }
  let count = 0
  for (let t = 0; t < triCount; t++) {
    if (
      alpha[texel(mesh.indices[t * 3])] >= opaqueAlpha &&
      alpha[texel(mesh.indices[t * 3 + 1])] >= opaqueAlpha &&
      alpha[texel(mesh.indices[t * 3 + 2])] >= opaqueAlpha
    ) {
      count++
    }
  }
  return count
}
