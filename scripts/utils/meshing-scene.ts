/**
 * node 侧分层渲染接线（**测试 / golden 专用**，不是交付路径）。
 *
 * ── 做什么 ──
 * ```
 * WSplatScene -> computeViewDepths -> 排序 -> DisparityStats -> LayerPlacement
 *             -> 逐层 drawLayer/readback -> LayeredRGBD
 * ```
 * 这是「跑一遍完整上游」的唯一实现，否则 golden / 目测脚本会各抄一份参数
 *（`wsplat-scene.ts` 的同一教训）。产物 `LayeredRGBD` 交给
 * `toMeshingInput()`（`src/spatial-scene/meshing/input.ts`）再进 meshing。
 *
 * ── 为什么不放 `src/` ──
 * 它需要装配 PLY 场景（`WSplatScene`，带 `node:fs`）并驱动 WebGPU 渲染器，
 * 属**测试基准**而非可交付模块。`src/spatial-scene/layering/` 只放平台无关的算法与契约；
 * 将来补 web 端生成时，若确有共享价值再把它提升到 layering 的公共入口。
 */

import {
  buildLayerPermutation,
  computeDisparityStats,
  computeLayerPlacement,
  computeLayerRanges,
  type LayeredRGBD,
  type LayerSamplingMethod,
  permuteNdcDepths,
} from "../../src/spatial-scene/layering/index.ts"
import { createWSplatRenderer } from "../../src/spatial-scene/wsplat/index.ts"
import {
  computeViewDepths,
  sortSplatsBackToFront,
} from "../../src/spatial-scene/wsplat/sort.ts"
import type { WSplatFrame } from "../../src/spatial-scene/wsplat/types.ts"
import type { WSplatScene } from "./wsplat-scene.ts"

/** `renderLayerStack` 的旋钮。 */
export interface RenderLayerStackOptions {
  /** 层数。默认 `8`（出货上界）。 */
  layers?: number
  /** 层边界放置方法。默认 `"quantile"`。 */
  method?: LayerSamplingMethod
  /** 直方图箱数（须 `>= 2L`）。默认 `256`。 */
  binCount?: number
}

/**
 * 跑一遍「splat -> L 层 RGBAD」，返回 `LayeredRGBD`（进程内，不落盘）。
 *
 * 渲染器在本函数内创建 / 销毁；`ranges` 用 `overlap = 0`（层间重叠只体现在
 * `toMeshingInput` 的显式 `overlap` 参数上）。
 */
export async function renderLayerStack(
  device: GPUDevice,
  scene: WSplatScene,
  options: RenderLayerStackOptions = {},
): Promise<LayeredRGBD> {
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
    L,
    width,
    height,
    near: scene.near,
    far: scene.far,
    placement,
    ranges: computeLayerRanges(placement.boundaries, 0),
    frames,
    stats,
  }
}
