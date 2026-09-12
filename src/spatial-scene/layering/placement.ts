/**
 * 层边界放置：六族启发式 **target 生成器** + 一个自研的**目标驱动解**。
 *
 * ── 形状 ──
 * 除 `uniform*` 与 `errorDriven` 外，全族共享同一条流水线：
 * ```
 * 直方图（可选重加权） -> CDF F(n) -> targets t_i ∈ [0,1] -> s_i = F⁻¹(t_i) -> 严格单调修复 -> 中点法补边界
 * ```
 * 各档**只差 target 序列**（`importance` 另外改了 CDF，`errorDriven` 干脆不用 CDF）。
 * 这条设计是刻意的：档位之间的差异必须能用一行公式说清，才可能在 E3 上解释胜负。
 *
 * ── 边界为什么是「层深的中点」 ──
 * 层深 `s_k` 是「这一层的代表深度」，那么把它当作量化器的一级，最优的归属划分
 * 就是**最近代表**，即 1D Voronoi：切在中点上。所以
 * `b_k = (s_{k-1} + s_k)/2`，两端分别钉死到 `n(near) = 0` / `n(far) = 1`。
 * `errorDriven` 的 k-means 解天然就是这个结构（最近质心划分 = 中点划分），
 * 于是六族启发式与自研解在**同一个边界约定**下可比。
 *
 * ── `errorDriven` 为什么是加权 1D k-means ──
 * 目标是最小化「用 L 个深度代表整场」的加权平方误差
 * `Σ_i w_i · (n_i − s_{k(i)})²`（`w_i` = 直方图质量，`k(i)` = 最近代表）。
 * 在 1D 上用 DP 可求**精确全局最优**：把 K 个箱切成 L 段，段代价用前缀和 O(1) 求出，
 * 总复杂度 `O(L·K²)`，`K = 256 / L = 32` 时约 2M 次运算 —— 离线秒级。
 * 它对应「重投影误差 ∝ 视差偏差」的线性化，是唯一**由目标函数导出**而非由符号名猜出的档位。
 * 因此判据很干净：它能不能在同样 `L` 下打赢六个启发式。实测（`pier.ply` @768，
 * `layering-perlayer-quality --compare`）：`L=8` 上它**双赢** —— 最坏层半 α
 * `13.07% → 10.62%`、可修缝 `0.479% → 0.330%`（quantile → errorDriven）；但 `L=4` 上相反
 * （`7.96%` vs `7.17%`），只有可修缝更低。⇒ 优势随深度结构翻转，**无全局赢家**；
 * 出货默认仍取 `quantile` + `L=8`。另：前端偏置 `z_mean − k·σ_z` 实测**无效**
 * （σ_z 是 1e-4~1e-2 m，而层带宽 0.06~58 m，k∈{0,0.5,1} 只动 ±0.02pp）。
 *
 * ── 为什么必须做严格单调修复 ──
 * 分位反解在**空箱段**上是平的：两个 target 可以解出同一个 `n`（该处确实没样本）。
 * 那样会产生「空层 + 退化边界」，而空层在 mesh 阶段是纯粹的浪费（一个没有三角形的切片）。
 * 修复用 `binWidth` 作为最小间距（那是直方图自己的分辨率），所以它**不引入假的精度**：
 * 只要 `binCount >= 2L`，修复总能在 `[0,1]` 内完成。
 *
 * ⚠ 层索引一律 **0 = 最近**。注意 back-to-front 排序序列里最前面的那一段是**最远层**，
 * 索引换算见 `bands.ts`。
 */

import {
  ndcDepthFromZ,
  quantilesFromStats,
  zFromNdcDepth,
} from "./disparity-stats.ts"
import type {
  DisparityStats,
  LayerPlacement,
  LayerPlacementOptions,
  NdcDepth,
} from "./types.ts"

/**
 * 中点法补 `L+1` 个边界。`out[0] = 0`、`out[L] = 1`（视差域的 near / far）。
 *
 * `layerDepths` 必须已严格递增，否则中点也会撞在一起。
 */
export function midpointBoundaries(
  layerDepths: ArrayLike<number>,
  near: number,
  far: number,
  out: Float32Array = new Float32Array(layerDepths.length + 1),
): Float32Array {
  const L = layerDepths.length
  out[0] = ndcDepthFromZ(near, near, far)
  for (let i = 1; i < L; i++) {
    out[i] = 0.5 * (layerDepths[i - 1] + layerDepths[i])
  }
  out[L] = ndcDepthFromZ(far, near, far)
  return out
}

/** 等质量 target：`t_i = (i + 0.5) / L`。 */
export function midpointTargets(L: number): Float32Array {
  const out = new Float32Array(L)
  for (let i = 0; i < L; i++) out[i] = (i + 0.5) / L
  return out
}

/** 前密 target：`t_i = u_i^p`（`u_i = (i + 0.5) / L`）。`p = 1` 退化为等质量。 */
export function frontWeightedTargets(L: number, p: number): Float32Array {
  const out = new Float32Array(L)
  for (let i = 0; i < L; i++) out[i] = ((i + 0.5) / L) ** p
  return out
}

/**
 * 前后分治 target：前 `foregroundLayers` 层用 `u_i^p`（前密），其余用 `u_i`（等质量）。
 *
 * 单调性：`u^p < u`（`p > 1`），所以交接处一定向上跳，不会破坏递增。
 */
export function hybridTargets(
  L: number,
  foregroundLayers: number,
  p: number,
): Float32Array {
  const fg = Math.max(1, Math.min(L, Math.floor(foregroundLayers)))
  const out = new Float32Array(L)
  for (let i = 0; i < L; i++) {
    const u = (i + 0.5) / L
    out[i] = i < fg ? u ** p : u
  }
  return out
}

/** 视差域 `[0,1]` 上几何等距。 */
export function uniformDepths(L: number, lo = 0, hi = 1): Float32Array {
  const out = new Float32Array(L)
  for (let i = 0; i < L; i++) out[i] = lo + ((i + 0.5) / L) * (hi - lo)
  return out
}

/**
 * 用直方图统计「每个箱落进哪一层」，得到每层的质量占比。
 *
 * ── 为什么不能按箱中心整箱归属 ──
 * 层带可以比一个箱还窄，而这在**远景被 n 压缩**时必然发生：
 * `n = far/(far−near)·(1 − near/z)` 对 `1/z` 仿射，
 * 所以 `z` 从 28 m 到 38 m（跨度 1.35 倍）只占一个箱（`binWidth = 1/256`）。
 * 那时整箱质量会被记到隔壁层上——实测 `pier.ply` 的 L=16 `quantile` 第 14 层
 * **精确质量 24.17% 被报成 0.00%**（该箱中心落在边界另一侧）。
 *
 * 正确做法：把每箱质量**按与层带的重叠长度**摊开（箱内均匀的直方图假设，
 * 与建直方图时的假设一致）。这样误差上界从「一整箱」降到「箱内不均匀度」。
 *
 * 精确值另见 `bandMassFromSamples`；两者之差是 `layering-golden.ts` 的一条真实数据门。
 */
export function bandMasses(
  stats: DisparityStats,
  boundaries: ArrayLike<number>,
  out: Float32Array = new Float32Array(boundaries.length - 1),
): Float32Array {
  const L = out.length
  out.fill(0)
  const { bins, binWidth } = stats
  let total = 0
  // 箱与层带都按 n 递增，所以起始层号可以单调前推，整体 O(K + L)。
  let firstBand = 0
  for (let k = 0; k < bins.length; k++) {
    const mass = bins[k]
    if (mass <= 0) continue
    total += mass
    const lo = k * binWidth
    const hi = lo + binWidth
    let band = firstBand
    while (band < L - 1 && boundaries[band + 1] <= lo) band++
    firstBand = band
    for (let j = band; j < L; j++) {
      const bandLo = boundaries[j]
      const bandHi = boundaries[j + 1]
      const overlap = Math.min(hi, bandHi) - Math.max(lo, bandLo)
      if (overlap > 0) out[j] += (mass * overlap) / binWidth
      if (bandHi >= hi) {
        firstBand = j
        break
      }
    }
  }
  if (total > 0) {
    for (let i = 0; i < L; i++) out[i] /= total
  }
  return out
}

/**
 * 每层质量的**精确**值：直接逐样本按边界归属，不经直方图。
 *
 * `bandMasses` 是直方图近似（快、只要 `stats`）；这个函数要全量样本（慢、精确）。
 * 两者的差就是直方图分辨率带来的误差，用一条真实数据门量着（见 golden 的 `exactMass`）。
 *
 * `ndc` 必须是**已按 `near`/`far` 换算好**的 NDC 深度序列（与 `stats` 同域）。
 * 边界仍按 `[b[k], b[k+1])` 半开区间判定，最后一个带含右端点，与 `bandMasses` 口径一致。
 */
export function bandMassFromSamples(
  ndc: ArrayLike<number>,
  weights: ArrayLike<number> | undefined,
  boundaries: ArrayLike<number>,
  out: Float32Array = new Float32Array(boundaries.length - 1),
): Float32Array {
  const L = out.length
  out.fill(0)
  let total = 0
  for (let i = 0; i < ndc.length; i++) {
    const w = weights ? weights[i] : 1
    if (!Number.isFinite(w) || w <= 0) continue
    const n = ndc[i]
    if (!Number.isFinite(n)) continue
    let band = L - 1
    while (band > 0 && n < boundaries[band]) band--
    out[band] += w
    total += w
  }
  if (total > 0) {
    for (let i = 0; i < L; i++) out[i] /= total
  }
  return out
}

/**
 * 「用这些代表深度重建整场」的加权平方误差 `Σ w·(n − 最近代表)²`。
 *
 * 这就是 `errorDriven` 的目标函数。导出它是为了让「自研解确实更优」变成一条**可断言的不变量**
 * （见 `scripts/layering-golden.ts`），而不是靠肉眼比较曲线。
 *
 * 返回值已按总质量归一化，因此可跨 `L` 之外的不同数据比较。
 */
export function quantizationError(
  stats: DisparityStats,
  layerDepths: ArrayLike<number>,
): number {
  const L = layerDepths.length
  if (L === 0) return 0
  const { bins, binWidth } = stats
  let total = 0
  let error = 0
  const first = layerDepths[0]
  const last = layerDepths[L - 1]
  for (let k = 0; k < bins.length; k++) {
    const mass = bins[k]
    if (mass <= 0) continue
    const n = (k + 0.5) * binWidth
    // 最近代表：层深已严格递增，可以夹紧到两端再线性扫描
    let depth: number
    if (n <= first) depth = first
    else if (n >= last) depth = last
    else {
      let i = 1
      while (i < L && layerDepths[i] < n) i++
      const a = layerDepths[i - 1]
      const b = layerDepths[i]
      depth = n - a <= b - n ? a : b
    }
    error += mass * (n - depth) ** 2
    total += mass
  }
  return total > 0 ? error / total : 0
}

/** 跑一遍放置流水线。 */
export function computeLayerPlacement(
  stats: DisparityStats,
  options: LayerPlacementOptions,
): LayerPlacement {
  const { L, method, near, far } = options
  if (!Number.isInteger(L) || L < 1) {
    throw new Error(`L 必须是 >= 1 的整数，收到 ${L}`)
  }
  if (stats.binCount < 2 * L) {
    throw new Error(
      `stats.binCount(${stats.binCount}) 必须 >= 2L(${2 * L})：` +
        "低于此值时严格单调修复会吃掉整个视差支撑。请用更大的 binCount 建统计。",
    )
  }
  if (!(far > near)) {
    throw new Error(`需要 far > near，收到 near=${near} far=${far}`)
  }

  const lo = clamp01(stats.minimum)
  const hi = clamp01(Math.max(stats.maximum, stats.minimum))
  // 分位族的层深应当落在**数据支撑**内；`uniform` 刻意跨满 `[0,1]`（它就是「浪费层」的基线）。
  const supportLo = method === "uniform" ? 0 : lo
  const supportHi = method === "uniform" ? 1 : hi

  let depths: Float32Array
  switch (method) {
    case "uniform":
      depths = uniformDepths(L, 0, 1)
      break
    case "uniformNonEmpty":
      depths = uniformDepths(L, lo, hi)
      break
    case "quantile":
      depths = quantilesFromStats(stats, midpointTargets(L))
      break
    case "importance":
      depths = quantilesFromStats(
        stats,
        midpointTargets(L),
        options.importanceExponent ?? 1,
      )
      break
    case "frontWeighted":
      depths = quantilesFromStats(
        stats,
        frontWeightedTargets(L, options.frontExponent ?? 2),
      )
      break
    case "hybrid":
      depths = quantilesFromStats(
        stats,
        hybridTargets(
          L,
          options.foregroundLayers ?? Math.ceil(L / 2),
          options.frontExponent ?? 2,
        ),
      )
      break
    case "errorDriven":
      depths = errorDrivenDepths(stats, L)
      break
  }

  repairStrictMonotonic(depths, stats.binWidth, supportLo, supportHi)
  if (method === "uniformNonEmpty") snapToDistinctOccupiedBins(depths, stats)

  const boundaries = midpointBoundaries(depths, near, far)
  const layerDepthsZ = new Float32Array(L)
  const boundariesZ = new Float32Array(L + 1)
  for (let i = 0; i < L; i++)
    layerDepthsZ[i] = zFromNdcDepth(depths[i], near, far)
  for (let i = 0; i <= L; i++)
    boundariesZ[i] = zFromNdcDepth(boundaries[i], near, far)

  return {
    L,
    method,
    near,
    far,
    layerDepths: depths,
    layerDepthsZ,
    boundaries,
    boundariesZ,
    layerMass: bandMasses(stats, boundaries),
  }
}

/**
 * 加权 1D k-means 的 DP 精确解（只在**非空箱**上切分，因此天然无空层）。
 *
 * 非空箱少于层数时无法给每层都分到质量，退化为等质量分位（保证非空）。
 */
function errorDrivenDepths(stats: DisparityStats, L: number): Float32Array {
  const { bins, binWidth } = stats
  const occupied: number[] = []
  for (let k = 0; k < bins.length; k++) {
    if (bins[k] > 0) occupied.push(k)
  }
  const P = occupied.length
  if (P === 0) return uniformDepths(L, 0, 1)
  if (P < L) return quantilesFromStats(stats, midpointTargets(L))

  // 非空箱上的前缀和：c0 = 质量，c1 = Σw·n，c2 = Σw·n²
  const c0 = new Float64Array(P + 1)
  const c1 = new Float64Array(P + 1)
  const c2 = new Float64Array(P + 1)
  for (let i = 0; i < P; i++) {
    const n = (occupied[i] + 0.5) * binWidth
    const w = bins[occupied[i]]
    c0[i + 1] = c0[i] + w
    c1[i + 1] = c1[i] + w * n
    c2[i + 1] = c2[i] + w * n * n
  }

  /** 段 `[a, b]`（含两端，非空箱下标）的平方误差；无质量时为 0。 */
  const segmentCost = (a: number, b: number): number => {
    const mass = c0[b + 1] - c0[a]
    if (!(mass > 0)) return 0
    const sum = c1[b + 1] - c1[a]
    return c2[b + 1] - c2[a] - (sum * sum) / mass
  }
  const segmentMean = (a: number, b: number): number => {
    const mass = c0[b + 1] - c0[a]
    return mass > 0
      ? (c1[b + 1] - c1[a]) / mass
      : (occupied[a] + 0.5) * binWidth
  }

  // dp[l][j] = 把前 j 个非空箱切成 l 段的最小代价；cut[l][j] = 最后一段的起点
  const width = P + 1
  const dp = new Float64Array((L + 1) * width).fill(Number.POSITIVE_INFINITY)
  const cut = new Int32Array((L + 1) * width)
  dp[0] = 0
  for (let l = 1; l <= L; l++) {
    for (let j = l; j <= P; j++) {
      let best = Number.POSITIVE_INFINITY
      let bestStart = -1
      for (let i = l - 1; i < j; i++) {
        const previous = dp[(l - 1) * width + i]
        if (!Number.isFinite(previous)) continue
        const cost = previous + segmentCost(i, j - 1)
        if (cost < best) {
          best = cost
          bestStart = i
        }
      }
      dp[l * width + j] = best
      cut[l * width + j] = bestStart
    }
  }

  // 回溯：从后往前取每段的加权质心
  const depths = new Float32Array(L)
  let j = P
  for (let l = L; l >= 1; l--) {
    const i = cut[l * width + j]
    depths[l - 1] = segmentMean(i, j - 1)
    j = i
  }
  return depths
}

/**
 * 把 `values` 修成 `[lo,hi]` 内严格递增、间距至少 `minGap`。
 *
 * ── 为什么不能「正向抬升 + 整体下移」──
 * 旧实现是：需要多少就抬多少，然后把**整组**下移 overflow。
 * 当前景/背景是两个相距很远的原子团时，分位反解把最后几层都解在**同一个箱内部**
 *（原子团在 `n` 上只占一个箱），间距远小于 `minGap`，于是抬升把最后几层推出数据支撑，
 * 再整体下移又拉不回来——最后层落在 `n = 1`，而数据最大只到 0.9978。
 * 实测 `pier.ply` L=16 `quantile` 因此产生一个**真空白层**，
 * 而它隔壁那层拿到 24.17% 的质量。
 *
 * ── 改成两趟：先从后往前压回支撑，再从前在后抬升 ──
 * 两趟都是「只收紧不放开」，所以只要 `(n-1)·minGap <= hi-lo`，结果必定落在 `[lo,hi]`：
 * 第二趟后 `v[i] <= max(init[i], v[i-1]+minGap) <= hi - (n-1-i)·minGap <= hi`。
 * 支撑真的装不下时（数据几乎单层）把 `minGap` 降到 `(hi-lo)/(n-1)`，
 * 宁可层变窄也不要出界——出界就是空层。
 *
 * `lo`/`hi` 对分位族应当是**数据的实际支撑** `[stats.minimum, stats.maximum]`，
 * 不是全 `[0,1]`：后者包含两段无数据区，把层推出去就白拿。
 * 但支撑太窄（退化到单点，如 `spike` 分布）时不能就地塌成一堆同值——
 * 那就退化成「L−1 个空带」。这时以支撑中心为心**向两侧扩到刚好装得下**，
 * 再夹回 `[0,1]`：层仍然贴着数据，且保持严格递增。
 */
function repairStrictMonotonic(
  values: Float32Array,
  minGap: number,
  supportLo = 0,
  supportHi = 1,
): void {
  const n = values.length
  if (n === 0) return
  let lo = supportLo
  let hi = supportHi
  if (!(hi > lo)) {
    lo = supportLo
    hi = supportLo
  }
  const need = n > 1 ? (n - 1) * minGap : 0
  if (hi - lo < need) {
    const center = 0.5 * (lo + hi)
    let a = center - need / 2
    let b = center + need / 2
    if (a < 0) {
      b -= a
      a = 0
    }
    if (b > 1) {
      a -= b - 1
      b = 1
    }
    lo = Math.max(0, a)
    hi = Math.min(1, b)
    if (hi - lo < need) {
      lo = 0
      hi = 1
    }
  }
  const gap = Math.min(minGap, n > 1 ? (hi - lo) / (n - 1) : hi - lo)
  const clamp = (v: number): number => (v < lo ? lo : v > hi ? hi : v)
  for (let i = 0; i < n; i++) values[i] = clamp(values[i])
  // 第一趟：从后往前，把顶部压回 `hi`，并保证相邻间距
  values[n - 1] = Math.min(values[n - 1], hi)
  for (let i = n - 2; i >= 0; i--) {
    values[i] = Math.min(clamp(values[i]), values[i + 1] - gap)
  }
  // 第二趟：从前往后抬升（只增不减，所以不会重新越界）
  values[0] = Math.max(clamp(values[0]), lo)
  for (let i = 1; i < n; i++) {
    values[i] = Math.max(clamp(values[i]), values[i - 1] + gap)
  }
}

/**
 * 把几何等距的平面搬到**互不相同的非空箱**上（`uniformNonEmpty` 的收尾）。
 *
 * ── 为什么不「先搬空层再修复单调」 ──
 * 那个做法在双峰分布上会自我打架：空层被搬到模式边缘后，单调修复又把它推回空谷，
 * 几十个空层永远收敛不了（实测 L=32 时仍有 7 个空层）。
 *
 * 这里换成**构造性**保证：给每层分配**不同**的非空箱，层深就是该箱中心。
 * 箱下标严格递增 ⇒ 层深间距 >= `binWidth` ⇒ 单调修复不会动它；
 * 又因为层深恰在箱中心，中点边界必然把它围在自己的带内 ⇒ **无空层是构造出来的**，
 * 不需要迭代。
 *
 * 非空箱数少于层数时（例如全场深度几乎相同）无法做到，退化到等质量分位。
 */
function snapToDistinctOccupiedBins(
  depths: Float32Array,
  stats: DisparityStats,
): void {
  const { bins, binWidth } = stats
  const L = depths.length
  const occupied: number[] = []
  for (let k = 0; k < bins.length; k++) {
    if (bins[k] > 0) occupied.push(k)
  }
  const P = occupied.length
  if (P === 0) return
  if (P < L) {
    depths.set(quantilesFromStats(stats, midpointTargets(L)))
    return
  }

  const center = (index: number): number => (occupied[index] + 0.5) * binWidth

  // 1) 单调游标找最近的非空箱（`depths` 已递增，游标只前进）
  const candidate = new Int32Array(L)
  let cursor = 0
  for (let i = 0; i < L; i++) {
    while (
      cursor + 1 < P &&
      Math.abs(center(cursor + 1) - depths[i]) <=
        Math.abs(center(cursor) - depths[i])
    ) {
      cursor++
    }
    candidate[i] = cursor
  }

  // 2) 夹成严格递增且互不相同：第 i 层至少留 i 个空位给前面、`L-1-i` 个给后面。
  //    因为 P >= L，这两个夹紧一定相容（`previous + 1 <= P-L+i`），不需要回溯。
  let previous = -1
  for (let i = 0; i < L; i++) {
    const maxAllowed = P - L + i
    let chosen = Math.min(maxAllowed, candidate[i])
    if (chosen < i) chosen = i
    if (chosen <= previous) chosen = previous + 1
    depths[i] = center(chosen)
    previous = chosen
  }
}

function clamp01(value: number): NdcDepth {
  return value < 0 ? 0 : value > 1 ? 1 : value
}
