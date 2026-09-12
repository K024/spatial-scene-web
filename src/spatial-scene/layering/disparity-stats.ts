/**
 * 视差域变换 + 流式视差统计（对应上游 `DisparityStatsEstimator`）。
 *
 * ── 为什么把「域变换」放在统计文件里 ──
 * 这个域不是随便挑的：`n(z) = far/(far−near)·(1 − near/z)` 就是 WebGPU 的 NDC 深度，
 * 同时也是 `1/z` 的严格仿射变换。统计、放置、渲染、元数据四者用的是同一个域，
 * 换算函数必须只有一份实现，否则「统计在 n 上、放置以为在 1/z 上」这类错会静默传播。
 * 所以它与「在这个域上做统计」放在一起，且全部是纯函数。
 *
 * ── 直方图为什么比排序好 ──
 * 1.18M 个高斯排序要几十~几百毫秒且结果依赖比较器细节；等宽直方图是 O(N) 且
 * 分位数反解只依赖箱计数 —— 确定性、可缓存（上游同样缓存 `cachedCumulativeCounts`）。
 * 代价是分位数精度被 `binWidth` 限制，而 `binWidth` 是我们自己选的旋钮。
 *
 * ── 权重 ──
 * `add()` 可以带逐样本权重（默认 1）。分层里权重就是 α：一个几乎全透明的高斯
 * 不应该产生和实心高斯一样强的「这一层该放哪」的投票。
 * 注意 `sampleSize` 数的是**样本个数**，而直方图 / 分位数用的是**权重质量**，
 * 两者在带权时不同，不要混用。
 */

import type { DisparityStats, NdcDepth } from "./types.ts"

/** 默认请求的分位概率（低/中/高分位，够画分布形状）。 */
export const DEFAULT_QUANTILE_PROBS: readonly number[] = [
  0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99,
]

/**
 * 真实度量深度 z（米）-> 视差域 n。`z == near` 给 0，`z == far` 给 1。
 *
 * 越界（`z < near` / `z > far`）**不**在这里夹紧：这里只做数学。
 * 采集时怎么处理越界由 `add()` 决定（见那里的说明）。
 */
export function ndcDepthFromZ(z: number, near: number, far: number): NdcDepth {
  if (!(z > 0) || !(far > near)) return 0
  return (far / (far - near)) * (1 - near / z)
}

/** `ndcDepthFromZ` 的逆：视差域 n -> 真实度量深度 z（米）。 */
export function zFromNdcDepth(n: number, near: number, far: number): number {
  const denom = far - n * (far - near)
  if (!(denom > 0)) return far
  return (near * far) / denom
}

/** 视差（= 逆深度）。它才是遮挡 / 外推误差的驱动量。 */
export function disparityFromZ(z: number): number {
  return z > 0 ? 1 / z : 0
}

/**
 * 从一组直方图权重反解分位数（线性插值于箱内）。
 *
 * `reweightExponent != 1` 时先做 `w_k ← w_k^p`：这是 `importance` 档的全部机制
 * （密度加权），不需要重建第二个直方图对象。
 *
 * 空箱区域的语义：CDF 是平的，二分查找会落到「质量真正开始的那一箱」，
 * 于是多个 target 可能解出**同一个值** —— 这是真实的（该处确实没有样本），
 * 由调用方（`placement.ts`）负责做严格单调修复，不在这里偷偷抖动。
 */
export function quantilesFromBins(
  bins: Float32Array,
  binWidth: number,
  targets: ArrayLike<number>,
  reweightExponent = 1,
  out: Float32Array = new Float32Array(targets.length),
): Float32Array {
  const K = bins.length
  // 累积用 f64：1.18M 个样本 / 256 箱，f32 累积的舍入会让高分位漂移。
  const cumulative = new Float64Array(K)
  let total = 0
  for (let k = 0; k < K; k++) {
    const h = reweightExponent === 1 ? bins[k] : bins[k] ** reweightExponent
    total += h
    cumulative[k] = total
  }
  if (!(total > 0)) {
    out.fill(0)
    return out
  }
  for (let i = 0; i < targets.length; i++) {
    const t = clamp01(targets[i]) * total
    // 第一个使 cumulative[k] >= t 的箱
    let lo = 0
    let hi = K - 1
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if (cumulative[mid] >= t) hi = mid
      else lo = mid + 1
    }
    const below = lo > 0 ? cumulative[lo - 1] : 0
    const mass = cumulative[lo] - below
    const fraction = mass > 0 ? (t - below) / mass : 0.5
    out[i] = clamp01((lo + clamp01(fraction)) * binWidth)
  }
  return out
}

/** `quantilesFromBins` 的便捷版：直接用统计对象里的直方图。 */
export function quantilesFromStats(
  stats: DisparityStats,
  targets: ArrayLike<number>,
  reweightExponent = 1,
  out?: Float32Array,
): Float32Array {
  return quantilesFromBins(
    stats.bins,
    stats.binWidth,
    targets,
    reweightExponent,
    out,
  )
}

export interface DisparityStatsOptions {
  near: number
  far: number
  /** 直方图箱数。默认 256；使用方需保证 `binCount >= 2·L`（见 `placement.ts`）。 */
  binCount?: number
  /** 随 `stats` 一起算出来的分位概率。 */
  quantileProbs?: readonly number[]
}

export interface DisparityStatsAddOptions {
  start?: number
  count?: number
  /** 逐样本权重（默认 1）。长度须覆盖 `[start, start+count)`。 */
  weights?: ArrayLike<number>
}

export interface DisparityStatsAccumulator {
  /**
   * 累加一批真实度量深度 z（米）。跳过非有限 / 非正的 z（它们是无效深度，
   * 不是「很远」）；越界到 `[near, far]` 外的 n 会被**夹紧**到 `[0,1]`——
   * 夹紧而不是丢弃，是为了让 `minimum`/`maximum` 反映真实的极端而非样本缺口。
   */
  add(z: ArrayLike<number>, options?: DisparityStatsAddOptions): void
  /** 出结果。可重复调用（幂等）；之后仍可继续 `add()`。 */
  finish(): DisparityStats
}

/** 建一个流式统计累加器。 */
export function createDisparityStatsAccumulator(
  options: DisparityStatsOptions,
): DisparityStatsAccumulator {
  const { near, far } = options
  const binCount = Math.max(2, Math.floor(options.binCount ?? 256))
  const binWidth = 1 / binCount
  const quantileProbs = Float32Array.from(
    options.quantileProbs ?? DEFAULT_QUANTILE_PROBS,
  )
  const bins = new Float32Array(binCount)

  let sampleSize = 0
  let weightSum = 0
  // 加权原始矩（f64），由它们导出中心矩。
  let s1 = 0
  let s2 = 0
  let s3 = 0
  let s4 = 0
  let minimum = Number.POSITIVE_INFINITY
  let maximum = Number.NEGATIVE_INFINITY

  function add(z: ArrayLike<number>, o: DisparityStatsAddOptions = {}): void {
    const start = o.start ?? 0
    const count = o.count ?? z.length - start
    const weights = o.weights
    for (let i = 0; i < count; i++) {
      const value = z[start + i]
      if (!Number.isFinite(value) || !(value > 0)) continue
      const w = weights ? weights[start + i] : 1
      if (!Number.isFinite(w) || w <= 0) continue
      const n = clamp01(ndcDepthFromZ(value, near, far))
      sampleSize++
      weightSum += w
      s1 += w * n
      const n2 = n * n
      s2 += w * n2
      s3 += w * n2 * n
      s4 += w * n2 * n2
      if (n < minimum) minimum = n
      if (n > maximum) maximum = n
      // 箱下标：n == 1 时落到最后一箱（不能越界）
      const k = Math.min(binCount - 1, Math.floor(n * binCount))
      bins[k] += w
    }
  }

  function finish(): DisparityStats {
    const empty = weightSum <= 0
    const mean = empty ? 0 : s1 / weightSum
    const m2 = empty ? 0 : Math.max(0, s2 / weightSum - mean * mean)
    const m3 = empty
      ? 0
      : s3 / weightSum - 3 * mean * (s2 / weightSum) + 2 * mean ** 3
    const m4 = empty
      ? 0
      : s4 / weightSum -
        4 * mean * (s3 / weightSum) +
        6 * mean ** 2 * (s2 / weightSum) -
        3 * mean ** 4
    const sigma2 = m2
    const sigma = Math.sqrt(sigma2)
    const lo = empty ? 0 : minimum
    const hi = empty ? 0 : maximum
    return {
      sampleSize,
      weightSum,
      minimum: lo,
      maximum: hi,
      minDepth: zFromNdcDepth(lo, near, far),
      maxDepth: empty ? far : zFromNdcDepth(hi, near, far),
      disparityMean: mean,
      disparityVariance: sigma2,
      disparitySkewness: sigma > 0 ? m3 / sigma ** 3 : 0,
      disparityKurtosis: sigma2 > 0 ? m4 / sigma2 ** 2 : 0,
      binCount,
      binWidth,
      bins,
      quantileProbs,
      quantiles: quantilesFromBins(bins, binWidth, quantileProbs),
    }
  }

  return { add, finish }
}

/** 一次性统计（`createDisparityStatsAccumulator` 的便捷封装）。 */
export function computeDisparityStats(
  z: ArrayLike<number>,
  options: DisparityStatsOptions,
  weights?: ArrayLike<number>,
): DisparityStats {
  const accumulator = createDisparityStatsAccumulator(options)
  accumulator.add(z, weights ? { weights } : undefined)
  return accumulator.finish()
}

function clamp01(value: number): number {
  return value < 0 ? 0 : value > 1 ? 1 : value
}
