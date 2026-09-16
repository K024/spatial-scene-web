/**
 * 层边界放置：把「一整片高斯」切成 L 个深度带，并给出每层的代表深度。
 *
 * ── 形状 ──
 * ```
 * 逐样本视差 n_i（+ 权重 w_i） -> 加权直方图 -> CDF
 *   -> targets t_i = (i+.5)/L -> s_i = CDF⁻¹(t_i)     （等质量分位）
 *   -> 严格单调修复 -> 边界 = 相邻层深的中点
 * ```
 * `uniform` 档只是把第一步换成 `s_i = (i+.5)/L`（几何等距），其余完全相同 ——
 * 两档共用同一条流水线，差异必须能用一行公式说清。
 *
 * ── 为什么按「质量」而不是按「深度」等分 ──
 * 等深度带宽在近处浪费层、在远处（`n` 被 `1/z` 压扁）丢层，而**视差误差才决定
 * 重投影位移**。以权重（默认 `opacities`，即该高斯贡献的「墨量」）的等质量分位切，
 * 每层拿到的可见贡献相当，是「L 个深度代表整场」这个目标下最直接的无参选择。
 *
 * ── 边界为什么是「层深的中点」──
 * 层深 `s_k` 是这一层的量化代表，那么把它当量化器的一级，最优归属划分就是最近代表，
 * 即 1D Voronoi：切在中点。两端钉死到 `n(near) = 0` / `n(far) = 1`。
 *
 * ── 严格单调修复为什么必要 ──
 * 分位反解在**空箱段**上是平的：两个 target 可以解出同一个 `n`（该处确实没样本），
 * 于是产生空层 + 退化边界。修复用 `binWidth` 作最小间距（那是直方图自己的分辨率，
 * 不引入假精度）；`binCount >= 2L` 时一定能在 `[0,1]` 内完成。
 */

import type { LayerPlacement, LayerPlacementMethod, NdcDepth } from "./types.ts"

/** `[L]` 等质量 target：`t_i = (i + 0.5) / L`。 */
export function midpointTargets(L: number): Float32Array {
  const out = new Float32Array(L)
  for (let i = 0; i < L; i++) out[i] = (i + 0.5) / L
  return out
}

/** `[L]` 视差域几何等距代表深度。 */
export function uniformDepths(L: number, lo = 0, hi = 1): Float32Array {
  const out = new Float32Array(L)
  for (let i = 0; i < L; i++) out[i] = lo + ((i + 0.5) / L) * (hi - lo)
  return out
}

/** 真实度量深度 z（米）-> 视差域 `n`（`z = near -> 0`，`z = far -> 1`）。 */
export function ndcDepthFromZ(z: number, near: number, far: number): NdcDepth {
  return (far / (far - near)) * (1 - near / z)
}

/** `ndcDepthFromZ` 的逆。 */
export function zFromNdcDepth(n: number, near: number, far: number): number {
  return (near * far) / (far - n * (far - near))
}

/**
 * 中点法补 `L+1` 个边界：`out[0] = 0`、`out[L] = 1`。
 *
 * `layerDepths` 必须已严格递增，否则中点也会撞在一起。
 */
export function midpointBoundaries(
  layerDepths: ArrayLike<number>,
  out: Float32Array = new Float32Array(layerDepths.length + 1),
): Float32Array {
  const L = layerDepths.length
  out[0] = 0
  for (let i = 1; i < L; i++) {
    out[i] = 0.5 * (layerDepths[i - 1] + layerDepths[i])
  }
  out[L] = 1
  return out
}

export interface LayerPlacementOptions {
  /** 层数（>= 1）。 */
  readonly L: number
  /** 投影近平面（米，> 0）。与渲染时**必须**一致，否则视差域换算全错。 */
  readonly near: number
  /** 投影远平面（米，> near）。 */
  readonly far: number
  /** 放置方法，默认 `"quantile"`。 */
  readonly method?: LayerPlacementMethod
  /** 直方图箱数，默认 256（自动抬到 `>= 2L`）。 */
  readonly binCount?: number
  /** 逐样本权重（默认全 1；`quantile` 档通常传 `opacities`）。 */
  readonly weights?: ArrayLike<number>
}

/**
 * 跑一遍放置流水线。
 *
 * @param depths 逐高斯**视图空间 z**（米），长度 `N`；非有限 / 非正值被跳过。
 */
export function computeLayerPlacement(
  depths: ArrayLike<number>,
  options: LayerPlacementOptions,
): LayerPlacement {
  const { L, near, far } = options
  if (!Number.isInteger(L) || L < 1) {
    throw new Error(`L 必须是 >= 1 的整数，收到 ${L}`)
  }
  if (!(near > 0) || !(far > near)) {
    throw new Error(`需要 0 < near < far，收到 near=${near} far=${far}`)
  }
  const method = options.method ?? "quantile"
  const binCount = Math.max(Math.ceil(options.binCount ?? 256), 2 * L, 2)
  const weights = options.weights
  if (weights && weights.length !== depths.length) {
    throw new Error(
      `weights 长度 ${weights.length} != depths 长度 ${depths.length}`,
    )
  }

  // ── 1. 加权直方图（视差域 [0,1]，等宽 binCount 箱）──
  const bins = new Float64Array(binCount)
  const binWidth = 1 / binCount
  const fillBins = (useWeights: boolean): number => {
    bins.fill(0)
    let total = 0
    for (let i = 0; i < depths.length; i++) {
      const z = depths[i]
      if (!(z > 0) || !Number.isFinite(z)) continue
      const w = useWeights ? (weights as ArrayLike<number>)[i] : 1
      if (!Number.isFinite(w) || w <= 0) continue
      const n = clamp01(ndcDepthFromZ(z, near, far))
      const bin = Math.min(binCount - 1, Math.floor(n * binCount))
      bins[bin] += w
      total += w
    }
    return total
  }
  let totalWeight = fillBins(weights !== undefined)
  // 权重全无效（例如全 0 opacity）时退回无权，而不是产出一堆空层。
  if (!(totalWeight > 0) && weights !== undefined) {
    totalWeight = fillBins(false)
  }

  // ── 2. 代表深度 ──
  let layerDepths: Float32Array
  if (method === "uniform" || !(totalWeight > 0)) {
    layerDepths = uniformDepths(L)
  } else {
    layerDepths = quantilesFromBins(
      bins,
      binWidth,
      midpointTargets(L),
      totalWeight,
    )
  }
  repairStrictMonotonic(layerDepths, binWidth)

  // ── 3. 边界 + 米制换算 + 逐层质量 ──
  const boundaries = midpointBoundaries(layerDepths)
  const layerDepthsZ = new Float32Array(L)
  const boundariesZ = new Float32Array(L + 1)
  for (let i = 0; i < L; i++)
    layerDepthsZ[i] = zFromNdcDepth(layerDepths[i], near, far)
  for (let i = 0; i <= L; i++)
    boundariesZ[i] = zFromNdcDepth(boundaries[i], near, far)

  return {
    L,
    method,
    near,
    far,
    layerDepths,
    layerDepthsZ,
    boundaries,
    boundariesZ,
    layerMass: exactBandMass(depths, weights, near, far, boundaries),
  }
}

/**
 * 直方图 CDF 的反解：给定 target（质量比例），返回对应箱中心作为代表深度。
 *
 * 用**箱中心**而不是线性插值：直方图本身的分辨率就是 `binWidth`，插值只会
 * 造出不存在的精度，且会让「空箱段」解出相邻的空层。
 */
function quantilesFromBins(
  bins: Float64Array,
  binWidth: number,
  targets: ArrayLike<number>,
  totalWeight: number,
): Float32Array {
  const out = new Float32Array(targets.length)
  let bin = 0
  let cumulative = 0
  for (let i = 0; i < targets.length; i++) {
    const target = targets[i] * totalWeight
    while (bin < bins.length - 1 && cumulative + bins[bin] < target) {
      cumulative += bins[bin]
      bin++
    }
    out[i] = (bin + 0.5) * binWidth
  }
  return out
}

/**
 * 把 `values` 修成 `[0,1]` 内严格递增、间距至少 `minGap`。
 *
 * 两趟（先从后往前压回上界、再从前在后抬升）。只要 `(n-1)·minGap <= 1`，
 * 结果必定落在 `[0,1]`：第二趟后 `v[i] <= hi - (n-1-i)·minGap <= hi`。
 * 装不下（`binCount < L-1`，构造上不会发生）时退化为等距。
 */
export function repairStrictMonotonic(
  values: Float32Array,
  minGap: number,
): void {
  const n = values.length
  if (n === 0) return
  if (n === 1) {
    values[0] = clamp01(values[0])
    return
  }
  let gap = Math.min(minGap, 1 / (n - 1))
  if (!(gap > 0)) gap = 1 / (n - 1)
  for (let i = 0; i < n; i++) values[i] = clamp01(values[i])
  values[n - 1] = Math.min(values[n - 1], 1)
  for (let i = n - 2; i >= 0; i--) {
    values[i] = Math.min(values[i], values[i + 1] - gap)
  }
  values[0] = Math.max(values[0], 0)
  for (let i = 1; i < n; i++) {
    values[i] = Math.max(values[i], values[i - 1] + gap)
  }
}

/**
 * 逐层质量的**精确**值：逐样本按边界归属（不经直方图）。
 *
 * N = 1.18M 时一遍循环几毫秒，换来「质量报告与实际划分一致」这条不变量
 * （`Σ layerMass == 1`，空层严格为 0）。
 */
export function exactBandMass(
  depths: ArrayLike<number>,
  weights: ArrayLike<number> | undefined,
  near: number,
  far: number,
  boundaries: ArrayLike<number>,
  out: Float32Array = new Float32Array(boundaries.length - 1),
): Float32Array {
  const L = out.length
  // 用 f64 累加再落到 f32：百万级样本在 f32 里累加会把 `Σmass == 1` 破坏到 1e-5 量级。
  const acc = new Float64Array(L)
  let total = 0
  for (let i = 0; i < depths.length; i++) {
    const z = depths[i]
    if (!(z > 0) || !Number.isFinite(z)) continue
    const w = weights ? weights[i] : 1
    if (!Number.isFinite(w) || w <= 0) continue
    const n = clamp01(ndcDepthFromZ(z, near, far))
    let band = upperBound(boundaries, n) - 1
    if (band < 0) band = 0
    if (band > L - 1) band = L - 1
    acc[band] += w
    total += w
  }
  if (total > 0) {
    for (let i = 0; i < L; i++) out[i] = acc[i] / total
  } else {
    out.fill(0)
  }
  return out
}

/**
 * 递增序列里第一个 `> value` 的下标（= `value` 所属半开区间 `[b[k], b[k+1])` 的上界）。
 * 用二分：L ≤ 64，但样本数可能上百万。
 */
function upperBound(sorted: ArrayLike<number>, value: number): number {
  let lo = 0
  let hi = sorted.length
  while (lo < hi) {
    const mid = (lo + hi) >>> 1
    if (sorted[mid] <= value) lo = mid + 1
    else hi = mid
  }
  return lo
}

function clamp01(value: number): number {
  return value < 0 ? 0 : value > 1 ? 1 : value
}
