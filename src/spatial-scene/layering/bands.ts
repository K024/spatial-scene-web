/**
 * 分带：把「全局 back-to-front 排序序列」按层边界切成 L 段**连续区间**。
 *
 * ── 为什么是连续区间而不是「按深度范围挑」──
 * `over` 是结合的，所以「层 = 全局排序的一段连续区间」与「按层深度范围筛选高斯」
 * 在参考视角上**恒等**（唯一差别是边界上相等深度的 tie-break，两端也一致）。
 * 前者只要改 `draw` 的 `firstInstance` / `instanceCount` 就能渲染，且天然不重不漏。
 *
 * ── back-to-front 意味着 `n` 递减 ──
 * 视差 `n` 随 z 单调**增**（`z = near -> n = 0`），而渲染必须先画远的，
 * 所以排序序列里 `n` 是**递减**的，序列最前面的那一段是**最远层**（`first[L-1] == 0`）。
 * 层索引本身仍按 0 = 最近，两者之间的换算只发生在这里 —— 集中在一个函数里，
 * 避免「同一份代码里两种方向索引并存」这类静默错误。
 */

import { ndcDepthFromZ } from "./placement.ts"
import type { LayerBands, LayerPermutation, LayerTableEntry } from "./types.ts"

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
 * 会在合成时把它的 α 算两次（`((A over B) over A) ≠ (A over B)`），重叠区变得比原图
 * 更暗 / 更不透明，参考视角无损（本模块的验收口径）直接没了。
 *
 * @param sortedNdcDepths 视差域深度，**已按 back-to-front 顺序重排**（即递减）。
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
 * `[L*2]` 每层的视差域深度范围 `[x0, x1]`（升序），可含层间重叠。
 *
 * 语义对齐上游 `MXISceneBuilder.getLayerRange(i)`：渲染侧按层深度范围剔除时用。
 * `overlap` **只影响这里的报告值**，不参与分配（见 `computeLayeredBands`）。
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
 * 从**全局 back-to-front 排序**出发，造出「按层分组」的排列 + 层表。
 *
 * 这是分层交给渲染器的东西（见 `LayerPermutation`）：
 * - 层序：`L-1`（最远）到 `0`（最近）—— 因为 `over` 必须先画远的；
 * - 层内：沿用全局序（已是 back-to-front）；
 * - 因为分配是划分，各层首尾相接后**逐位等于**全局序（`length == 高斯总数`）。
 *
 * 从全局序**抽取**（而不是各自排序）意味着「按层分组后整体仍是合法 back-to-front
 * 序」是**构造上成立**的，不需要重新排序，也不需要再证明一次有序性。
 *
 * @param globalOrder `sortSplatsBackToFront()` 的输出。
 * @param sortedNdcDepths 与 `globalOrder` 同序的视差域值（递减）。
 * @param boundaries `[L+1]` 层边界（视差域，递增）。
 */
export function buildLayerPermutation(
  globalOrder: ArrayLike<number>,
  sortedNdcDepths: ArrayLike<number>,
  boundaries: ArrayLike<number>,
): LayerPermutation {
  if (sortedNdcDepths.length !== globalOrder.length) {
    throw new Error(
      `sortedNdcDepths 长度 ${sortedNdcDepths.length} != globalOrder 长度 ${globalOrder.length}`,
    )
  }
  const bands = computeLayeredBands(sortedNdcDepths, boundaries)
  const L = bands.L

  // 先定层表。**顺序很关键**：排列是绘制顺序，而 `over` 必须先画远的，
  // 所以占据排列开头的是**最远层** `L-1`。
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
