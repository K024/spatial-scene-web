/**
 * 高斯排序与顶点打包（**纯函数，可在 worker / 主线程复用**）。
 *
 * ── 为什么只排一次 ──
 * 3DGS 的正确合成顺序是「远的先画、近的后画」（painter's order + 标准 alpha
 * 混合），严格做法是每帧按当前相机重排。但 spatial scene 的观看范围很小
 * （基本围着参考相机转），于是这里做一个取舍：**用参考相机排一次序，
 * 之后固定不变**。代价是相机离开参考位姿后遮挡/融合顺序会出错，收益是彻底
 * 省掉每帧 1.18M 元素的排序，而且整场只需一次 draw call、不需要深度缓冲。
 *
 * ── 为什么用基数排序 ──
 * 排序键是单精度浮点视深。把 float32 位模式映射成「有序 uint32」后，
 * 就能用 4 趟 8 位 LSD 基数排序在 O(4N) 内完成，没有比较排序的比较器
 * 调用开销（JS 里 1.18M 元素 `Array#sort` 约 1~2 s，基数排序约 0.1 s）。
 */

import type { PackedSplats, SortCamera, SplatSoA } from "./types.ts"
import { SPLAT_LAYOUT, SPLAT_STRIDE } from "./types.ts"

/** 借用同一块内存读写 f32 位模式（避免 `Math.fround` + 手写位运算）。 */
const BITS_F32 = new Float32Array(1)
const BITS_U32 = new Uint32Array(BITS_F32.buffer)

/**
 * float32 位模式 -> 单调递增的 uint32。
 *
 * 正数翻转符号位、负数整体取反；于是「无符号整数序」= 「浮点序」。
 * 调用方还会再取反（`~`）以获得**降序**（远 -> 近）。
 */
function flipFloatBits(bits: number): number {
  return (bits ^ (((bits >>> 31) | 0x80000000) >>> 0)) >>> 0
}

/**
 * 按视深**从远到近**排序，返回 `[N]` 置换表（`order[i]` = 第 i 个绘制的高斯）。
 *
 * 深度取「相机视线方向上的投影」`dot(p - eye, forward)`，而非欧氏距离：
 * 这才是 painter's order 需要的前后关系，也与屏幕投影尺度一致
 * （投影雅可比只依赖视深）。
 */
export function sortByViewDepth(
  soa: SplatSoA,
  camera: SortCamera,
): { order: Uint32Array; ms: number } {
  const t0 = performance.now()
  const n = soa.count
  const { center } = soa
  const [ex, ey, ez] = camera.eye
  const [fx, fy, fz] = camera.forward

  // 1) 深度 -> 可排序 uint32（取反 = 降序）
  let keys = new Uint32Array(n)
  let order = new Uint32Array(n)
  for (let i = 0; i < n; i++) {
    const depth =
      (center[i * 3] - ex) * fx +
      (center[i * 3 + 1] - ey) * fy +
      (center[i * 3 + 2] - ez) * fz
    BITS_F32[0] = depth
    keys[i] = ~flipFloatBits(BITS_U32[0]) >>> 0
    order[i] = i
  }

  // 2) 4 趟 8 位 LSD 基数排序：键与置换表同步搬运
  let nextKeys = new Uint32Array(n)
  let nextOrder = new Uint32Array(n)
  const bucket = new Uint32Array(256)

  for (let shift = 0; shift < 32; shift += 8) {
    bucket.fill(0)
    for (let i = 0; i < n; i++) bucket[(keys[i] >>> shift) & 0xff]++
    let sum = 0
    for (let b = 0; b < 256; b++) {
      const c = bucket[b]
      bucket[b] = sum
      sum += c
    }
    for (let i = 0; i < n; i++) {
      const k = keys[i]
      const p = bucket[(k >>> shift) & 0xff]++
      nextKeys[p] = k
      nextOrder[p] = order[i]
    }
    const tk = keys
    keys = nextKeys
    nextKeys = tk
    const to = order
    order = nextOrder
    nextOrder = to
  }

  return { order, ms: performance.now() - t0 }
}

/** 按置换表把结构数组打包成交错顶点缓冲（布局见 `SPLAT_LAYOUT`）。 */
export function packSplats(
  soa: SplatSoA,
  order: Uint32Array | null,
  sortMs = 0,
): PackedSplats {
  const t0 = performance.now()
  const n = soa.count
  const out = new Float32Array(n * SPLAT_STRIDE)
  const { center, scaleLog, quat, colorLinear, opacityLogit } = soa
  const oC = SPLAT_LAYOUT.center
  const oS = SPLAT_LAYOUT.scaleLog
  const oQ = SPLAT_LAYOUT.quat
  const oCol = SPLAT_LAYOUT.colorLinear
  const oO = SPLAT_LAYOUT.opacityLogit

  for (let i = 0; i < n; i++) {
    const s = order === null ? i : order[i]
    const o = i * SPLAT_STRIDE
    const s3 = s * 3
    const s4 = s * 4
    out[o + oC] = center[s3]
    out[o + oC + 1] = center[s3 + 1]
    out[o + oC + 2] = center[s3 + 2]
    out[o + oS] = scaleLog[s3]
    out[o + oS + 1] = scaleLog[s3 + 1]
    out[o + oS + 2] = scaleLog[s3 + 2]
    out[o + oQ] = quat[s4]
    out[o + oQ + 1] = quat[s4 + 1]
    out[o + oQ + 2] = quat[s4 + 2]
    out[o + oQ + 3] = quat[s4 + 3]
    out[o + oCol] = colorLinear[s3]
    out[o + oCol + 1] = colorLinear[s3 + 1]
    out[o + oCol + 2] = colorLinear[s3 + 2]
    out[o + oO] = opacityLogit[s]
  }
  return { count: n, data: out, sortMs, packMs: performance.now() - t0 }
}
