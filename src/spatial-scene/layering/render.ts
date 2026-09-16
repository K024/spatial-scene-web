/**
 * 分层渲染驱动：**splat 场 -> L 层 RGBAD**（WebGPU）。
 *
 * 阶段：`image -> splat 参数场 -> [layered RGBAD] -> mesh`。本文件是中间那一环，
 * 纯 CPU 的分带 / 放置算法在 `placement.ts` / `bands.ts`，这里只做接线：
 * ```
 * 1. 参考相机 -> 扩视角视图（view.ts）-> 渲染相机
 * 2. computeViewDepths -> sortSplatsBackToFront（wsplat 的确定性 radix）
 * 3. 视差域放置（placement）-> 分带 + 排列 + 层表（bands）
 * 4. setGaussians / setCamera / setSplatOrder
 * 5. for k 远..近: drawLayer(table[k]) -> readback() -> countCulls() -> readSplatStats()
 * 6. （可选）整场直接渲染一帧作对照
 * ```
 *
 * ── 每一步为什么是这样 ──
 * - **扩视角**在相机上做（`viewScale` 扩画布、焦距不变），见 `view.ts` 的说明；
 * - **分带必须用「排序序列的连续区间」**（`over` 结合律），见 `bands.ts`；
 * - **逐层各自 `clear`**（`drawLayer` 的语义）才有 LDI 语义：每层持有自己原本的
 *   颜色/α/ED，不被更近的层衰减。合成口径见 `composite.ts`；
 * - **每层都收剔除统计**（`countCulls`）：`minPixelSize` 是像素量纲，低分辨率下会
 *   **静默**剔掉大量高斯（实测 384 宽下 46.72%），不读统计就不知道某层为什么空了。
 *
 * ── 内存口径 ──
 * 一帧 `WSplatFrame` 在 `w*h` 上约 `(4+3+1+1+1+1) = 11` 字节/像素（preview 4、
 * rgb 12、alpha/depth/transmission/accumulated 各 4 ⇒ 实际 ~28 字节/像素）。
 * 1024×768 时约 25 MB/层，L=10 约 250 MB —— 这是默认 `maxRenderSide = 1024` 的
 * 主要原因；要更高纹理分辨率（更大 GLB）就显式抬它。
 */

import type { Gaussians3D } from "../sharp/types.ts"
import { createWSplatRenderer } from "../wsplat/index.ts"
import { computeViewDepths, sortSplatsBackToFront } from "../wsplat/sort.ts"
import type { WSplatFrame } from "../wsplat/types.ts"
import {
  buildLayerPermutation,
  computeLayerRanges,
  permuteNdcDepths,
} from "./bands.ts"
import { computeLayerPlacement } from "./placement.ts"
import {
  DEFAULT_LAYERS,
  type LayeredRgbd,
  type LayerPermutation,
  type LayerPlacement,
  type LayerRenderOptions,
  type LayerSplatStats,
  MAX_LAYERS,
} from "./types.ts"
import { expandedLayerCamera, resolveLayerView } from "./view.ts"

/** 纯 CPU 的分层计划（可在没有 GPU 的环境里单测）。 */
export interface LayerPlan {
  /** 全局 back-to-front 排列（远 -> 近）。 */
  readonly order: Uint32Array
  /** 与 `order` 同序的视差域深度（递减）。 */
  readonly sortedNdcDepths: Float32Array
  readonly placement: LayerPlacement
  /** 按层分组后的排列 + 层表（交给 `setSplatOrder` / `drawLayer`）。 */
  readonly permutation: LayerPermutation
}

/** `planLayerStack` 的旋钮（`renderLayerStack` 的子集）。 */
export interface LayerPlanOptions {
  readonly layers?: number
  readonly method?: LayerRenderOptions["method"]
  readonly binCount?: number
  readonly near: number
  readonly far: number
  /**
   * 是否用 opacities 作权重（默认 `true`）。`false` 时按样本数等分，
   * 用于对照「按墨量分」与「按个数分」的差别。
   */
  readonly weightByOpacity?: boolean
}

/**
 * 只跑 CPU 部分：排序 -> 放置 -> 分带 -> 排列。
 *
 * @param depths 视图空间 z（米），长度 = 高斯数。
 * @param opacities 不透明度（权重），长度同上。
 */
export function planLayerStack(
  depths: Float32Array,
  opacities: ArrayLike<number> | undefined,
  options: LayerPlanOptions,
): LayerPlan {
  const count = depths.length
  if (opacities && opacities.length !== count) {
    throw new Error(
      `opacities 长度 ${opacities.length} != depths 长度 ${count}`,
    )
  }
  const L = resolveLayerCount(options.layers)
  const order = sortSplatsBackToFront(depths, count)
  const sortedNdcDepths = permuteNdcDepths(
    depths,
    order,
    options.near,
    options.far,
  )
  const placement = computeLayerPlacement(depths, {
    L,
    method: options.method,
    near: options.near,
    far: options.far,
    binCount: options.binCount,
    weights: options.weightByOpacity === false ? undefined : opacities,
  })
  const permutation = buildLayerPermutation(
    order,
    sortedNdcDepths,
    placement.boundaries,
  )
  return { order, sortedNdcDepths, placement, permutation }
}

/**
 * 跑一遍「splat -> L 层 RGBAD」。
 *
 * 渲染器在本函数内创建 / 销毁；相机、渲染参数由 `options` 决定。
 * 产物是**内存数据**（线性 RGBAD），不落盘。
 */
export async function renderLayerStack(
  device: GPUDevice,
  gaussians: Gaussians3D,
  options: LayerRenderOptions,
): Promise<LayeredRgbd> {
  const referenceCamera = options.camera
  const L = resolveLayerCount(options.layers)
  const view = resolveLayerView(
    {
      width: referenceCamera.width,
      height: referenceCamera.height,
      focalLengthPx:
        referenceCamera.width / (2 * Math.tan(referenceCamera.fovX / 2)),
    },
    options,
  )
  const camera = expandedLayerCamera(referenceCamera, view)
  const count = gaussians.opacities.length

  const depths = computeViewDepths(
    gaussians.meanVectors,
    camera.viewMatrix,
    count,
  )
  const plan = planLayerStack(depths, gaussians.opacities, {
    layers: L,
    method: options.method,
    binCount: options.binCount,
    near: camera.near,
    far: camera.far,
  })

  const renderer = await createWSplatRenderer(device, {
    size: { width: view.width, height: view.height },
    resolve: options.resolve,
    minPixelSize: options.minPixelSize,
    alphaClip: options.alphaClip,
    eps2d: options.eps2d,
    antialias: options.antialias,
  })

  try {
    renderer.setGaussians(gaussians, {
      colorSpace: options.colorSpace ?? "linearRGB",
    })
    renderer.setCamera(camera)
    renderer.setSplatOrder(plan.permutation.permutation)

    const frames = new Array<WSplatFrame>(L)
    const stats = new Array<LayerSplatStats>(L)
    // 绘制顺序：远 -> 近（排列本来就是按这个顺序分组的）。
    for (let k = L - 1; k >= 0; k--) {
      renderer.drawLayer(plan.permutation.table[k])
      frames[k] = await renderer.readback(options.resolve)
      renderer.countCulls()
      stats[k] = {
        layerIndex: k,
        ...(await renderer.readSplatStats()),
      }
    }

    let direct: WSplatFrame | null = null
    if (options.includeDirect) {
      // 与分层**同一个相机、同一批高斯**，只是不分层 —— 这就是对照物。
      renderer.setSplatOrder(plan.order)
      renderer.renderSplats()
      direct = await renderer.readback(options.resolve)
    }

    return {
      L,
      width: view.width,
      height: view.height,
      near: camera.near,
      far: camera.far,
      view,
      camera,
      placement: plan.placement,
      ranges: computeLayerRanges(
        plan.placement.boundaries,
        options.rangeOverlap ?? 0,
      ),
      frames,
      stats,
      direct,
    }
  } finally {
    renderer.destroy()
  }
}

/** 层数校验：默认 10，硬范围 `[1, 64]`（推荐 4~16，见 `types.ts`）。 */
export function resolveLayerCount(layers: number | undefined): number {
  const value = layers ?? DEFAULT_LAYERS
  if (!Number.isInteger(value) || value < 1 || value > MAX_LAYERS) {
    throw new Error(
      `层数必须是 [1, ${MAX_LAYERS}] 的整数，收到 ${value}` +
        `（推荐 4~16，默认 ${DEFAULT_LAYERS}）`,
    )
  }
  return value
}
