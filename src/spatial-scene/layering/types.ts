/**
 * layering 的**纯数据类型层**。
 *
 * 本文件**不得** import 任何 WebGPU / node 类型（对照 `wsplat/types.ts` 的定位）：
 * 它是 web / node 共用的数据契约层。GPU 相关类型留在 `index.ts`。
 *
 * ── 在全链路里的位置 ──
 * ```
 * image (+EXIF focal) -> splat 参数场 -> [layered RGBAD] -> mesh（交付物）
 * ```
 * 本模块只做中间那一环：**splat 场 -> L 层 RGBAD**。splat 与 RGBAD 都是中间表示，
 * splat 不出生成管线；渲染侧只光栅化多层 mesh。
 *
 * ── 为什么所有层深 / 边界都活在「视差域」 ──
 * `n(z) = far/(far−near) · (1 − near/z)`，即 **WebGPU 的 NDC 深度**（0 = near，1 = far）。
 * 它是 `1/z` 的**严格仿射**变换，所以在 `n` 上等距 ≡ 在视差上等距。
 * 选它的理由：视差才是视差 / 遮挡的驱动量，且上游 `DisparityStatsEstimator` /
 * `maxRelativeDisparity` / `zNormalized` 全在这个域 —— 一条域贯穿
 * 统计 -> 放置 -> 渲染 -> 元数据，避免每换一个环节就换一次单位。
 */

import type { WSplatFrame } from "../wsplat/types.ts"

/** 视差域（= WebGPU NDC 深度）里的一个值。`0` = near，`1` = far。 */
export type NdcDepth = number

/**
 * 视差分布的统计量（流式累积；对应上游 `DisparityStatsEstimator`）。
 *
 * 直方图建在**视差域** `[0,1]` 上，等宽 `binCount` 个箱。所有分位数都由直方图反解，
 * **不排序**——这是它能在 1.18M 个高斯上瞬间出结果的原因，也是确定性的来源。
 *
 * 与 `MXIScene.attributes`（`MXISceneAttributeDisparity{Mean,Variance,Skewness,Kurtosis,
 * Quantiles,Score}`）一一对应：`disparityMean/Variance/Skewness/Kurtosis/quantiles`。
 */
export interface DisparityStats {
  /** 参与统计的样本数（未加权；跳过了非有限 / 非正的 z）。 */
  readonly sampleSize: number
  /** 权重之和（默认权重为 1，此时 == `sampleSize`）。 */
  readonly weightSum: number
  /** 观测到的**最小视差域值**（= 最近处）。 */
  readonly minimum: NdcDepth
  /** 观测到的**最大视差域值**（= 最远处）。 */
  readonly maximum: NdcDepth
  /** 最近处的**真实度量深度**（米）= `z(minimum)`。 */
  readonly minDepth: number
  /** 最远处的**真实度量深度**（米）= `z(maximum)`。 */
  readonly maxDepth: number
  readonly disparityMean: number
  readonly disparityVariance: number
  /** 三阶中心矩 / σ³（正态 = 0）。 */
  readonly disparitySkewness: number
  /** 四阶中心矩 / σ⁴（**非超额**，正态 = 3）。 */
  readonly disparityKurtosis: number
  readonly binCount: number
  /** 视差域箱宽 = `1 / binCount`。 */
  readonly binWidth: number
  /** 每箱的**权重质量**，长度 `binCount`；第 k 箱覆盖 `[k·binWidth, (k+1)·binWidth)`。 */
  readonly bins: Float32Array
  /** 请求的分位概率，长度 `Q`。 */
  readonly quantileProbs: Float32Array
  /** 与 `quantileProbs` 对齐的分位值（视差域）。 */
  readonly quantiles: Float32Array
}

/**
 * 层边界放置方法。
 *
 * 这**不是** Apple `layerSamplingMethod` 枚举值的复刻（不反汇编、不猜符号数值），
 * 而是把「层边界放哪」当成一个可测量的近似误差问题后，归纳出的几族 **target 生成器**。
 * 前六族是启发式基线，`errorDriven` 是自研的目标驱动解（见 `placement.ts`）。
 */
export type LayerSamplingMethod =
  /** 在视差域 `[0,1]` 上几何等距。基线；**允许空层**（数据集中时必然出现）。 */
  | "uniform"
  /** 几何等距，但限制在**经验支撑** `[minimum, maximum]` 内，并修掉空层。 */
  | "uniformNonEmpty"
  /** 等质量（= 视差分位）。 */
  | "quantile"
  /** 先按 `w ∝ count^p` 重加权直方图，再取等质量分位 —— 密度加权。 */
  | "importance"
  /** 前 `foregroundLayers` 层走 `frontWeighted`，其余走 `uniform`（前后分治）。 */
  | "hybrid"
  /** `t_i = u_i^p`（`u_i = (i+.5)/L`，`p > 1`）→ 前密后疏。 */
  | "frontWeighted"
  /** **自研**：在加权视差直方图上最小化 `Σ w·(n − s_k)²`（加权 1D k-means，DP 精确解）。 */
  | "errorDriven"

export interface LayerPlacementOptions {
  /** 层数。 */
  L: number
  method: LayerSamplingMethod
  /** 投影近平面（米）；与渲染时**必须**一致，否则视差域换算全错。 */
  near: number
  /** 投影远平面（米）。 */
  far: number
  /** `importance`：重加权指数 `p`（`w ∝ count^p`），默认 1。 */
  importanceExponent?: number
  /** `frontWeighted` / `hybrid`：前密指数 `p`（`t = u^p`），默认 2。 */
  frontExponent?: number
  /** `hybrid`：走 `frontWeighted` 的前景层数，默认 `ceil(L/2)`。 */
  foregroundLayers?: number
}

/**
 * 分带结果：每层在 **back-to-front 排序序列**里的连续下标区间。
 *
 * 之所以强调「排序序列里的连续」：`over` 是结合的，所以「层 = 全局排序的一段连续区间」
 * 是**数学恒等**，而不是实现上的方便。切一段渲染只需要改 `draw` 的
 * `firstInstance` / `instanceCount`，WGSL 不用动。
 *
 * ⚠ 排序序列是 back-to-front（`n` **递减**），所以序列最前面的那一段是**最远层**：
 * `first[L-1] == 0`。层索引 0 = 最近。
 */
export interface LayerBands {
  readonly L: number
  /** 参与分带的高斯总数。 */
  readonly total: number
  /** `[L]` 每层在排序序列里的起始下标。 */
  readonly first: Int32Array
  /** `[L]` 每层的高斯个数（`overlap > 0` 时各层之和会大于 `total`）。 */
  readonly count: Int32Array
}

/**
 * 层表的一项：层 `k` 占据**排列**里的 `[base, base + count)`。
 *
 * 把它单独拿出来（而不是隐式靠 `base = Σcount` 推）是因为一旦允许层间重叠 /
 * 密度补偿，“层在排列里是首尾相接的”就不再成立。
 */
export interface LayerTableEntry {
  readonly base: number
  readonly count: number
}

/**
 * 绘制排列 + 层表。层索引 `k` 一律 **0 = 最近**。
 *
 * ── 为什么不直接用「层 = 全局排序的一段区间」 ──
 * 那是**深度分带**的特例，会在两处碎掉：
 * 1. 层间重叠 / 密度补偿让同一个高斯出现在多个层，排列长度 > 高斯数；
 * 2. 非深度分带的层（语义 / 物体 / 前景区块）在全局深度序里根本不连续。
 * 所以排列的所有权归 layering（它知道层语义），渲染器只负责「按你给的顺序画」。
 *
 * ── 合法性（必须当门来验，不能靠“构造上总对” ）──
 * `concatenate(table[k] 的区间)` 必须是一个**合法绘制顺序**：
 * - 层间：`k = L-1`（最远）到 `0`（最近）依次相接；
 * - 层内：`n` 递减（back-to-front）。
 * 两条合起来就等于全局 back-to-front 序（重叠时是它的一份超集，且每项落在正确位置）。
 * 破了任一条件，画面会静默错（`over` 是不可交换的）。
 */
export interface LayerPermutation {
  /** 长度 `length`；每项是一个高斯下标（`< 高斯总数`）。 */
  readonly permutation: Uint32Array
  /** 排列长度。重叠 / 密度补偿时可 `> 高斯总数`。 */
  readonly length: number
  /** `[L]` 层表（0 = 最近）。 */
  readonly table: readonly LayerTableEntry[]
}

/**
 * 一次放置的结果。层索引 `k` 一律 **0 = 最近**。
 *
 * 语义：层 `k` 拥有落进 `[boundaries[k], boundaries[k+1])` 的高斯，
 * 用 `layerDepths[k]` 这一个深度代表整层（这就是「L 个平面」的含义）。
 */
export interface LayerPlacement {
  readonly L: number
  readonly method: LayerSamplingMethod
  readonly near: number
  readonly far: number
  /** `[L]` 每层的代表深度，**视差域**，严格递增。 */
  readonly layerDepths: Float32Array
  /** `[L]` 同上，但换成**真实度量深度**（米），仅供下游 / 调试阅读。 */
  readonly layerDepthsZ: Float32Array
  /** `[L+1]` 层边界，视差域；`[0] == 0`、`[L] == 1`，严格递增。 */
  readonly boundaries: Float32Array
  /** `[L+1]` 同上，米。 */
  readonly boundariesZ: Float32Array
  /** `[L]` 每层占据的质量占比（由直方图求得，`Σ == 1`）。 */
  readonly layerMass: Float32Array
}

/**
 * 分层结果：L 层 RGBAD + 元数据，是 **mesh 阶段的唯一输入**。
 *
 * `frames[k]` 复用 `WSplatFrame` 契约（真实度量深度 / 透射率 T / 累积深度 ED），
 * 所以下游不需要第二套像素格式。
 *
 * ⚠ 下游**不做任何层混合**：每层各自转 mesh，靠视差去看底层露出的像素。所以
 * 「逐层合成 ≡ 全量渲染」（`layering-golden.ts` 的 B 段，阈值级无损）**不免除单层责任** ——
 * 每层自己还得高质量（半 α / 可修缝，见 `layering-perlayer-quality.ts`）。
 * 出货 `L` 上界 = **8**（层数即 mesh 数与代价）。
 *
 * ⚠ 合成多层时用 `rgba`（线性 f32）+ `alpha` + `accumulatedDepth`，**不要**用 `preview`；
 * 跨层合并深度必须在 `(ED, A)` 空间做：`D_total = Σ ED_k / Σ A_k`。
 */
export interface LayeredRGBD {
  readonly L: number
  readonly width: number
  readonly height: number
  /** 投影近平面（米），同时是 `ranges[0]` 的下界。 */
  readonly near: number
  /** 投影远平面（米）。 */
  readonly far: number
  readonly placement: LayerPlacement
  /**
   * `[L*2]` 每层的视差域深度范围 `[x0, x1]`（升序），已含层间重叠。
   * 语义对齐上游 `MXISceneBuilder.getLayerRange(i)`：供渲染侧按深度范围剔除。
   */
  readonly ranges: Float32Array
  /** `[L]` 层像素，索引与 `placement.layerDepths` 对齐（0 = 最近）。 */
  readonly frames: readonly WSplatFrame[]
  /** 供 `MXIScene.attributes` 使用的统计量。 */
  readonly stats: DisparityStats
}
