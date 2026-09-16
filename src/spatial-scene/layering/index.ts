/**
 * layering 模块的**公共入口**（barrel）：splat 场 -> L 层 RGBAD（线性）。
 *
 * ── 提供什么 ──
 * | 文件 | 内容 |
 * |---|---|
 * | `types.ts` | 数据契约（`LayeredRgbd` / `LayerPlacement` / `LayerView` …） |
 * | `view.ts` | 扩视角视图（参考相机 -> 渲染画布 + 相机），meshing 的坐标来源 |
 * | `placement.ts` | 层边界放置（视差域等质量分位 / 等距） |
 * | `bands.ts` | 排序序列 -> 层区间 / 排列 / 层表 |
 * | `render.ts` | WebGPU 驱动：逐层 `drawLayer` + `readback` |
 * | `composite.ts` | 线性空间合成 + 「分层 ≡ 整场」对照 |
 *
 * ── 不提供什么（刻意的）──
 * 不做原图回写 / 逐层几何补齐 / 补视角外推：本阶段只保证**参考视角数值正确**
 *（`compositeLayerFrames(frames) == singleFrameToComposited(direct)`）；
 * 其他视角的优化补充、断壁、洞的修补全部不在本模块。
 *
 * ── 最小用法 ──
 * ```ts
 * const layered = await renderLayerStack(device, gaussiansMetric, {
 *   camera,            // 参考视角（原图）
 *   layers: 10,
 *   viewScale: 1.2,    // 扩视角（给 mesh 留边缘余量）
 *   includeDirect: true, // 顺便拿一帧整场渲染做对照
 * })
 * const diff = compareComposited(
 *   compositeLayerFrames(layered.frames),
 *   singleFrameToComposited(layered.direct!),
 * )
 * ```
 */

export {
  buildLayerPermutation,
  computeLayeredBands,
  computeLayerRanges,
  permuteNdcDepths,
} from "./bands.ts"
export type { CompositedDiff, CompositedFrame } from "./composite.ts"
export {
  compareComposited,
  compositeLayerFrames,
  singleFrameToComposited,
} from "./composite.ts"
export type { LayerPlacementOptions } from "./placement.ts"
export {
  computeLayerPlacement,
  exactBandMass,
  midpointBoundaries,
  midpointTargets,
  ndcDepthFromZ,
  repairStrictMonotonic,
  uniformDepths,
  zFromNdcDepth,
} from "./placement.ts"
export type { LayerPlan, LayerPlanOptions } from "./render.ts"
export {
  planLayerStack,
  renderLayerStack,
  resolveLayerCount,
} from "./render.ts"
export type {
  LayeredRgbd,
  LayerPermutation,
  LayerPlacement,
  LayerPlacementMethod,
  LayerRenderOptions,
  LayerSplatStats,
  LayerTableEntry,
  LayerView,
  LayerViewRect,
  NdcDepth,
  ShortSideOption,
} from "./types.ts"
export {
  DEFAULT_LAYERS,
  DEFAULT_MAX_RENDER_SIDE,
  DEFAULT_SHORT_SIDE,
  DEFAULT_VIEW_SCALE,
  MAX_LAYERS,
  MAX_RECOMMENDED_LAYERS,
  MIN_RECOMMENDED_LAYERS,
} from "./types.ts"
export type { LayerViewOptions, ReferenceView } from "./view.ts"
export {
  cameraRotationRows,
  expandedLayerCamera,
  referenceToRenderPixel,
  renderPixelToCamera,
  renderToReferencePixel,
  resolveLayerView,
} from "./view.ts"
