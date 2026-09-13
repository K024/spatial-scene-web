/**
 * layering 模块的**公共入口**（barrel）。
 *
 * ── 提供什么 ──
 * 纯算法 + 纯数据契约：
 * - `types.ts`        数据契约（`LayeredRGBD` / `LayerPlacement` / `DisparityStats` …）
 * - `bands.ts`        排序序列 -> 层区间 / 排列
 * - `disparity-stats.ts` 视差域统计（直方图 / 分位数 / 矩）
 * - `placement.ts`    层边界放置（7 族 target 生成器）
 * - `refine.ts`       原图回写（权重 / 混合）+ **逐层几何补齐**（own-gap / hidden 外推）
 *
 * ── 不提供什么（刻意的）──
 * **分层渲染（`renderLayerStack`）不在 `src` 里**：它需要装配 PLY 场景并驱动 WebGPU
 * 渲染器，当前只作为 **node 侧测试 / golden 的接线**（`scripts/utils/meshing-scene.ts`），
 * 不是交付路径的一部分。将来补 web 端生成时再决定是否提升，届时本文件是它的落点。
 *
 * ── 为什么可以放心 `import *` 这层 ──
 * 本 barrel 只聚合**平台无关**模块（无 WebGPU / DOM / node / three）；
 * `refine.ts` 对 `sharp/preprocess.ts` 的依赖是**类型**（`SourceImage`），编译期即擦除。
 */

export type { LayeredBandsOptions } from "./bands.ts"
export {
  buildLayerPermutation,
  buildLayerPermutationFromAssignment,
  computeLayeredBands,
  computeLayerRanges,
  permuteNdcDepths,
  validateLayerPermutation,
} from "./bands.ts"
export type {
  DisparityStatsAccumulator,
  DisparityStatsAddOptions,
  DisparityStatsOptions,
} from "./disparity-stats.ts"
export {
  computeDisparityStats,
  createDisparityStatsAccumulator,
  DEFAULT_QUANTILE_PROBS,
  disparityFromZ,
  ndcDepthFromZ,
  quantilesFromBins,
  quantilesFromStats,
  zFromNdcDepth,
} from "./disparity-stats.ts"
export {
  bandMasses,
  bandMassFromSamples,
  computeLayerPlacement,
  frontWeightedTargets,
  hybridTargets,
  midpointBoundaries,
  midpointTargets,
  quantizationError,
  uniformDepths,
} from "./placement.ts"
export type {
  HiddenExtendOptions,
  LayerCompletionOptions,
  RefineLayersOptions,
  RefineWeightOptions,
} from "./refine.ts"
export {
  blendImageWriteback,
  completeLayerGeometry,
  compositeAlphaDepth,
  computeLayerOwnership,
  computeOcclusionMasks,
  computeRefineWeight,
  layerPixelToImagePixel,
  refineLayers,
  resampleImageToLinear,
} from "./refine.ts"
export type {
  DisparityStats,
  LayerBands,
  LayeredRGBD,
  LayerPermutation,
  LayerPlacement,
  LayerPlacementOptions,
  LayerSamplingMethod,
  LayerTableEntry,
  NdcDepth,
} from "./types.ts"
