/**
 * 排序：按**视图空间 z 从大到小**（远 -> 近 / back-to-front）给出 splat 下标排列。
 *
 * 排序是深度正确的**唯一**来源：输出用的是 `over` 算子，而 WebGPU 的 blend 让
 * **后画的**压在**先画的**上面，所以必须先远后近。
 *
 * 实现：**CPU 端确定性 LSD radix**（32-bit key，8 bit × 4 pass，**间接排序下标**）。
 * - 确定性：同样输入一定给同样排列；深度相等时靠稳定性 tie-break 成原下标升序。
 * - 不走 `Array#sort`：118 万元素的比较排序要几秒，radix 是几十毫秒级，
 *   且行为与将来的 GPU radix 可逐位对拍（替换前提是**逐位一致**，否则宁可用慢的 CPU 版）。
 *
 * 产出的下标序列同时是**分层的依据**：`over` 是结合的，所以「层 = 全局排序的一段
 * 连续区间」，且拓扑上属于哪一层只需要看这个序列（每 splat 单属一层、不重不漏）。
 */

const RADIX_BITS = 8
const RADIX = 1 << RADIX_BITS
const RADIX_MASK = RADIX - 1
const PASSES = 4

const scratchF32 = new Float32Array(1)
const scratchU32 = new Uint32Array(scratchF32.buffer)

/** 正浮点数的位模式；非正数一律当 0（会被排到最后）。 */
function positiveFloatBits(value: number): number {
  scratchF32[0] = value > 0 ? value : 0
  return scratchU32[0]
}

/**
 * 计算视图空间 z（米）。`viewMatrix` 为列主序 16 元素。
 *
 * 只需第三行（相机 z 轴那行）点积，省掉两次乘加。
 */
export function computeViewDepths(
  means: Float32Array,
  viewMatrix: ArrayLike<number>,
  count: number,
  out: Float32Array = new Float32Array(count),
): Float32Array {
  const f0 = viewMatrix[2]
  const f1 = viewMatrix[6]
  const f2 = viewMatrix[10]
  const t = viewMatrix[14]
  for (let i = 0; i < count; i++) {
    out[i] =
      f0 * means[i * 3] + f1 * means[i * 3 + 1] + f2 * means[i * 3 + 2] + t
  }
  return out
}

/**
 * back-to-front：深度**降序**。
 *
 * @param depths 视图空间 z（米）
 * @returns `Uint32Array`，`order[0]` 是最远的高斯
 */
export function sortSplatsBackToFront(
  depths: Float32Array,
  count: number,
): Uint32Array {
  const keys = new Uint32Array(count)
  // 取反后「升序排 key」= 「降序排深度」
  for (let i = 0; i < count; i++) keys[i] = ~positiveFloatBits(depths[i])
  return radixSortIndices(keys, count)
}

/**
 * 间接 LSD radix：排的是「下标」，比较的是 `keys[index]`。稳定。
 *
 * 稳定性正是 tie-break 的来源：key 相同时保持输入顺序（= 原下标升序）。
 */
export function radixSortIndices(
  keys: Uint32Array,
  count: number,
): Uint32Array {
  let src = new Uint32Array(count)
  for (let i = 0; i < count; i++) src[i] = i
  let dst = new Uint32Array(count)
  const counts = new Uint32Array(RADIX)

  for (let pass = 0; pass < PASSES; pass++) {
    const shift = pass * RADIX_BITS
    counts.fill(0)
    for (let i = 0; i < count; i++) {
      counts[(keys[src[i]] >>> shift) & RADIX_MASK]++
    }
    let sum = 0
    for (let b = 0; b < RADIX; b++) {
      const c = counts[b]
      counts[b] = sum
      sum += c
    }
    for (let i = 0; i < count; i++) {
      const idx = src[i]
      dst[counts[(keys[idx] >>> shift) & RADIX_MASK]++] = idx
    }
    const tmp = src
    src = dst
    dst = tmp
  }
  return src
}
