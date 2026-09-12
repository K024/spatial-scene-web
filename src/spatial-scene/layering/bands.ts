/**
 * 分带：把「全局 back-to-front 排序序列」按层边界切成 L 段**连续区间**。
 *
 * ── 为什么是连续区间而不是「按深度范围挑」 ──
 * `over` 是结合的，所以「层 = 全局排序的一段连续区间」与「按层深度范围筛选高斯」
 * 在**参考视角上恒等**（唯一差别是相等深度的 tie-break 归属，两端都一致）。
 * 前者只要改 `draw` 的 `firstInstance` / `instanceCount` 就能渲染，且天然不重不漏。
 * 所以分带在**排序序列的下标空间**里做，而不是在深度空间里反复筛。
 *
 * ── back-to-front 意味着 `n` 递减 ──
 * 视差域 `n` 随 z 单调**增**（`z = near -> n = 0`），而渲染必须先画远的，
 * 所以排序序列里 `n` 是**递减**的，最前面的那一段是**最远层**（`first[L-1] == 0`）。
 * 层索引本身仍按 0 = 最近，两者之间的换算只在这一处发生 —— 集中在一个函数里，
 * 避免「同一份代码里两种方向索引并存」这类静默错误。
 *
 * ── `overlap` 不碰分配 ──
 * `overlap` 只影响**报告的 depthRange**（`computeLayerRanges()`），给下游 mesh / 剔除留余量。
 * 早期版本把它做成了“让边界附近的高斯同时属于两层”，那是**错的**：`over` 不幂等，
 * 重复绘制会把 α 累加两次（`A over B` vs `(A over B) over A` 不等），
 * 重叠区会变得比原图更不透明，参考视角无损直接没了。
 * 消带缝要用别的手段（前端偏置分配 / 网格域回填），不能用重复绘制。
 *
 * ── 渲染器并不需要「层 = 连续区间」这个前提 ──
 * 渲染器只认一张排列（见 `LayerPermutation`）。本模块从全局序**抽取**出按层分组的排列，
 * 所以“合法绘制顺序”是构造保证的；真需要引任意下标集合（密度补偿）时，
 * 这个保证就没了，得靠 `validateLayerPermutation()` 重新验。
 */

import { ndcDepthFromZ } from "./disparity-stats.ts"
import type { LayerBands, LayerPermutation, LayerTableEntry } from "./types.ts"

export interface LayeredBandsOptions {
  /**
   * 层间重叠，单位是**视差域比例**（`0.05` = 全视差范围的 5%）。默认 0。
   *
   * ⚠ 它只影响 `computeLayerRanges()` 报告的 depthRange，**不影响分配**。
   */
  overlap?: number
}

/**
 * 把真实度量深度 z 按排序序列重排，并换算到视差域。
 *
 * `order` 就是 `sortSplatsBackToFront()` 的输出（back-to-front 下标序列）。
 */
export function permuteNdcDepths(
  z: ArrayLike<number>,
  order: ArrayLike<number>,
  near: number,
  far: number,
  out: Float32Array = new Float32Array(order.length),
): Float32Array {
  for (let i = 0; i < order.length; i++) {
    out[i] = ndcDepthFromZ(z[order[i]], near, far)
  }
  return out
}

/**
 * 按边界切出每层的连续下标区间。
 *
 * ── 分配永远是**划分** ──
 * 同一个高斯**只能**进一层。理由是 `over` 不幂等：把边界附近的高斯同时给相邻两层，
 * 会在合成时把它的 α 算两次（`((A over B) over A) ≠ (A over B)`），
 * 重叠区变得比原图更暗 / 更不透明，参考视角无损（验收 A）直接没了。
 * 所以这里**不接受** overlap；层间重叠只存在于报告的 depthRange 里。
 *
 * @param sortedNdcDepths 视差域深度，**已按 back-to-front 顺序重排**（即递减）。
 *   见 `permuteNdcDepths()`。
 * @param boundaries `[L+1]` 层边界（视差域，递增，`[0]==0`、`[L]==1`）。
 */
export function computeLayeredBands(
  sortedNdcDepths: ArrayLike<number>,
  boundaries: ArrayLike<number>,
): LayerBands {
  const L = boundaries.length - 1
  if (L < 1) throw new Error("boundaries 至少要有 2 个元素")
  const total = sortedNdcDepths.length

  const first = new Int32Array(L)
  const count = new Int32Array(L)
  for (let band = 0; band < L; band++) {
    // 层 0（最近）的下界是 -Inf、层 L-1（最远）的上界是 +Inf，
    // 这样两端一定被完整覆盖，不需要特判夹紧。
    const upper =
      band === L - 1 ? Number.POSITIVE_INFINITY : boundaries[band + 1]
    const lower = band === 0 ? Number.NEGATIVE_INFINITY : boundaries[band]
    const start = countAtOrAbove(sortedNdcDepths, upper)
    const end = countAtOrAbove(sortedNdcDepths, lower)
    first[band] = start
    count[band] = Math.max(0, end - start)
  }

  return { L, total, first, count }
}

/**
 * `[L*2]` 每层的视差域深度范围 `[x0, x1]`（升序），含层间重叠。
 *
 * 语义对齐上游 `MXISceneBuilder.getLayerRange(i)`：供渲染侧按层深度范围剔除
 * （每层只需要自己的 near/far，不必用全局的）。
 */
export function computeLayerRanges(
  boundaries: ArrayLike<number>,
  overlap = 0,
  out: Float32Array = new Float32Array((boundaries.length - 1) * 2),
): Float32Array {
  const L = boundaries.length - 1
  for (let band = 0; band < L; band++) {
    out[band * 2] = Math.max(0, boundaries[band] - overlap)
    out[band * 2 + 1] = Math.min(1, boundaries[band + 1] + overlap)
  }
  return out
}

/**
 * 非递增序列里「有多少个元素 >= value」= 第一个 `< value` 的下标。
 *
 * 用二分而不是线性扫描：1.18M 个元素 × 2L 次边界查找，线性是 7.5e7 次比较。
 */
function countAtOrAbove(sorted: ArrayLike<number>, value: number): number {
  let lo = 0
  let hi = sorted.length
  while (lo < hi) {
    const mid = (lo + hi) >>> 1
    if (sorted[mid] >= value) lo = mid + 1
    else hi = mid
  }
  return lo
}

/**
 * 从**全局 back-to-front 排序**出发，造出「按层分组」的排列 + 层表。
 *
 * 这是分层交给渲染器的东西（见 `LayerPermutation` 的契约）：
 * - 层序：`L-1`（最远）到 `0`（最近）—— 因为 `over` 必须先画远的；
 * - 层内：沿用全局序（已是 back-to-front）；
 * - 因为分配是划分，各层首尾相接后**逐位等于**全局序（`length == 高斯总数`）。
 *
 * ── 为什么从全局序推导而不是各自排序 ──
 * 全局序已经是一次 O(N) 确定性 radix 的结果。按层**抽取**只能得到它的子序列，
 * 所以“按层分组后整体仍是一个合法 back-to-front 序”是**构造上成立**的，
 * 不需要重新排序，也不需要再证明一次有序性。
 * （真需要引任意下标集合时——比如密度补偿插进去的高斯——就得重新造排列，
 * 那时“逐位等于全局序”这个强性质就没了，只能靠 `validateLayerPermutation()` 的弱不变量。）
 *
 * @param globalOrder `sortSplatsBackToFront()` 的输出（back-to-front 下标序列）。
 * @param sortedNdcDepths 与 `globalOrder` 同序的视差域值（递减）；见 `permuteNdcDepths()`。
 * @param boundaries `[L+1]` 层边界（视差域，递增）。
 */
export function buildLayerPermutation(
  globalOrder: ArrayLike<number>,
  sortedNdcDepths: ArrayLike<number>,
  boundaries: ArrayLike<number>,
): LayerPermutation {
  const bands = computeLayeredBands(sortedNdcDepths, boundaries)
  const L = bands.L

  // 先定层表。**顺序很关键**：排列是绘制顺序，而 `over` 必须先画远的，
  // 所以占据排列开头的是**最远层** `L-1`，而不是最近的层。
  // （曾把 `base` 按 `k=0..L-1` 累加，结果是近层先画 → 整个排列逆序，
  //   “逆序”门直接报出 L-1 处递增，就是抓着它的。）
  const table: LayerTableEntry[] = new Array<LayerTableEntry>(L)
  let cursor = 0
  for (let k = L - 1; k >= 0; k--) {
    table[k] = { base: cursor, count: bands.count[k] }
    cursor += bands.count[k]
  }
  const length = cursor

  // 同一层内部沿用全局序（已 back-to-front），所以直接按区间拷。
  const permutation = new Uint32Array(length)
  for (let k = 0; k < L; k++) {
    const { base, count } = table[k]
    const start = bands.first[k]
    for (let i = 0; i < count; i++) {
      permutation[base + i] = globalOrder[start + i]
    }
  }
  return { permutation, length, table }
}

/**
 * 从「每个实例的层归属」造排列 + 层表 —— `buildLayerPermutation` 的**通用版**。
 *
 * ── 为什么需要它（不能用连续区间版）──
 * `buildLayerPermutation` 假设「层 = 全局序里的一段连续区间」，那只在
 * **按排序用的同一个深度**分带时成立。E4a 的前端偏置 `z_mean − k·σ_z`
 * 用**另一个**深度决定归属，于是同一层的高斯在全局序里不再连续，
 * 只能按归属标签**筛选**（filter）而不是**切段**（slice）。
 * 语义层 / 任意下标集合同理。
 *
 * ── 仍然保证的合法绘制顺序 ──
 * - 层内沿用 `globalOrder`（已是 back-to-front）⇒ 层内 back-to-front 是构造保证；
 * - 层序 `L-1`（最远）→ `0`（最近），与 `buildLayerPermutation` 一致。
 * 因为用另一个深度做了归属，**跨层**的「远→近」不再严格成立 —— 这正是偏置的
 * 代价（会改写参考视角），所以它只用于 E4a 实验，默认路径仍走连续区间版。
 *
 * @param globalOrder `sortSplatsBackToFront()` 的输出（长度 = 高斯总数）。
 * @param bandOfPosition 长度 = `globalOrder.length`；`bandOfPosition[i]` = 排列位置 `i`
 *   对应的高斯属于哪一层（`0` = 最近）。**必须每个位置都被分到** `[0,L)` 内，
 *   否则结果不是全量严格排列（`length < 高斯总数`）并在末尾抛错。
 * @param L 层数。
 */
export function buildLayerPermutationFromAssignment(
  globalOrder: ArrayLike<number>,
  bandOfPosition: ArrayLike<number>,
  L: number,
): LayerPermutation {
  const total = globalOrder.length
  if (bandOfPosition.length !== total) {
    throw new Error(
      `bandOfPosition 长度 ${bandOfPosition.length} != 全局序长度 ${total}`,
    )
  }
  const count = new Int32Array(L)
  for (let i = 0; i < total; i++) {
    const k = bandOfPosition[i]
    if (!Number.isInteger(k) || k < 0 || k >= L) {
      throw new Error(
        `位置 ${i} 的层归属 ${k} 不在 [0,${L})：每个高斯必须恰好属于一层`,
      )
    }
    count[k]++
  }

  // 层序：排列开头是**最远层**（`over` 必须先画远的），与连续区间版同一约定。
  const table: LayerTableEntry[] = new Array<LayerTableEntry>(L)
  let cursor = 0
  for (let k = L - 1; k >= 0; k--) {
    table[k] = { base: cursor, count: count[k] }
    cursor += count[k]
  }

  const permutation = new Uint32Array(cursor)
  const write = new Int32Array(L)
  for (let k = 0; k < L; k++) write[k] = table[k].base
  for (let i = 0; i < total; i++) {
    const k = bandOfPosition[i]
    permutation[write[k]++] = globalOrder[i]
  }
  return { permutation, length: cursor, table }
}

/**
 * 验一条排列是否是合法的绘制顺序（配合 `LayerPermutation` 的契约）。
 *
 * 返回所有**破约**点（空数组 = 合法）：
 * - `outOfRange`：下标 >= `splatCount`；
 * - `descent`：排列里出现 `n` 变小（`n` 递减才是 back-to-front）。
 *
 * @param ndcDepthBySplat 是**高斯下标 -> 视差域值**（长度 = 高斯总数），
 * 所以它不是排列的某种重排，而是原顺序。
 *
 * 注意这是**弱**不变量：它允许重复下标、允许有缺项。
 * 当排列是由全局序抽取而来时，应该用更强的「逐位等于全局序」去测。
 */
export function validateLayerPermutation(
  permutation: ArrayLike<number>,
  ndcDepthBySplat: ArrayLike<number>,
): { outOfRange: number[]; descent: number[] } {
  const splatCount = ndcDepthBySplat.length
  const outOfRange: number[] = []
  const descent: number[] = []
  let previous = Number.POSITIVE_INFINITY
  for (let i = 0; i < permutation.length; i++) {
    const index = permutation[i]
    if (index >= splatCount) {
      if (outOfRange.length < 8) outOfRange.push(i)
      continue
    }
    const n = ndcDepthBySplat[index]
    if (n > previous) {
      if (descent.length < 8) descent.push(i)
    }
    previous = n
  }
  return { outOfRange, descent }
}
