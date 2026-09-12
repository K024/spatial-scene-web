/**
 * layering -> meshing 的**进程内接线**（node 侧）。
 *
 * ── 为什么需要它 ──
 * 本链路约定**不落盘**：层 RGBAD 在进程内从 wsplat 渲染器流到 meshing。
 * 所以「跑一遍完整上游」这件事必须有个唯一入口，否则 `meshing-golden` 与将来的
 * `meshing-render-views` / `meshing-export` 会各抄一份参数（`wsplat-scene.ts` 的同一教训）。
 *
 * ── 做什么 ──
 * ```
 * WSplatScene -> computeViewDepths -> 排序 -> DisparityStats -> LayerPlacement
 *             -> 逐层 drawLayer/readback -> frames[] -> (调用方) buildMeshScene
 * ```
 * 与 `layering-export-layers.ts` 走的是同一条路径（同一个排列、同一份统计），
 * 只是不写 PLY/PNG。
 */

import {
  buildLayerPermutation,
  computeLayerRanges,
  permuteNdcDepths,
} from "../../src/spatial-scene/layering/bands.ts"
import { computeDisparityStats } from "../../src/spatial-scene/layering/disparity-stats.ts"
import { computeLayerPlacement } from "../../src/spatial-scene/layering/placement.ts"
import type {
  DisparityStats,
  LayerPlacement,
  LayerSamplingMethod,
} from "../../src/spatial-scene/layering/types.ts"
import type { MeshingInput } from "../../src/spatial-scene/meshing/types.ts"
import { createWSplatRenderer } from "../../src/spatial-scene/wsplat/index.ts"
import {
  computeViewDepths,
  sortSplatsBackToFront,
} from "../../src/spatial-scene/wsplat/sort.ts"
import type { WSplatFrame } from "../../src/spatial-scene/wsplat/types.ts"
import type { WSplatScene } from "./wsplat-scene.ts"

/** 一次分层渲染的产物。 */
export interface LayeredStack {
  readonly placement: LayerPlacement
  readonly stats: DisparityStats
  /** 逐层 RGBAD，索引与 `placement.layerDepths` 对齐（0 = 最近）。 */
  readonly frames: WSplatFrame[]
  /** `[L*2]` 每层视差域范围（含 overlap），= `MXISceneBuilder.getLayerRange(i)` 语义。 */
  readonly ranges: Float32Array
}

export interface RenderLayerStackOptions {
  /** 层数。默认 `8`（出货上界）。 */
  layers?: number
  /** 层边界放置方法。默认 `"quantile"`。 */
  method?: LayerSamplingMethod
  /** 直方图箱数（须 `>= 2L`）。默认 `256`。 */
  binCount?: number
}

/** 跑一遍「splat -> L 层 RGBAD」，返回帧与元数据（进程内）。 */
export async function renderLayerStack(
  device: GPUDevice,
  scene: WSplatScene,
  options: RenderLayerStackOptions = {},
): Promise<LayeredStack> {
  const L = options.layers ?? 8
  const method = options.method ?? "quantile"
  const { width, height } = scene
  const count = scene.gaussians.opacities.length

  const depth = computeViewDepths(
    scene.gaussians.meanVectors,
    scene.camera.viewMatrix,
    count,
  )
  const order = sortSplatsBackToFront(depth, count)
  const sortedNdc = permuteNdcDepths(depth, order, scene.near, scene.far)
  const stats = computeDisparityStats(
    depth,
    { near: scene.near, far: scene.far, binCount: options.binCount ?? 256 },
    scene.gaussians.opacities,
  )
  const placement = computeLayerPlacement(stats, {
    L,
    method,
    near: scene.near,
    far: scene.far,
  })
  const perm = buildLayerPermutation(order, sortedNdc, placement.boundaries)

  const renderer = await createWSplatRenderer(device, {
    size: { width, height },
  })
  renderer.setGaussians(scene.gaussians)
  renderer.setCamera(scene.camera)
  renderer.setSplatOrder(perm.permutation)
  const frames: WSplatFrame[] = []
  for (let k = 0; k < L; k++) {
    renderer.drawLayer(perm.table[k])
    frames.push(await renderer.readback())
  }
  renderer.destroy()

  return {
    placement,
    stats,
    frames,
    ranges: computeLayerRanges(placement.boundaries, 0),
  }
}

/** 把上面的产物装成 `buildMeshScene` 要的输入契约。 */
export function toMeshingInput(
  scene: WSplatScene,
  stack: LayeredStack,
  overlap = 0,
): MeshingInput {
  return {
    L: stack.frames.length,
    width: scene.width,
    height: scene.height,
    near: scene.near,
    far: scene.far,
    placement: stack.placement,
    ranges:
      overlap === 0
        ? stack.ranges
        : computeLayerRanges(stack.placement.boundaries, overlap),
    frames: stack.frames,
    stats: stack.stats,
    camera: scene.camera,
  }
}
