/**
 * 支撑场：从一层的 RGBAD 里算出「哪里出几何 / 深度取多少」，
 * 并把深度**向外扩**一圈以支撑半透明外围。
 *
 * ── 1. 支撑 = 非全透明 ──
 * `seed` 的判据只有三条：`visible`、`depth > 0`、`alpha > alphaCutoff`（默认 `1/255`，
 * 即**只剔全透明**）。不做迟滞双阈值、不做小岛门（除 `minIslandPixels` 这一项可关的
 * 清理）：本阶段的目标是「尽可能简单 + 参考视角正确」，软边由纹理 α 自己表达。
 *
 * ── 2. 深度为什么要向外扩（`dilatePx`）──
 * 层的 α 是软的：越靠剪影越小，到网格边界处已经接近 0。如果几何严格停在
 * `alpha > cutoff` 的地方，那么：
 * - 斜视角下网格边界会「切进」半透明区，把本该由 α 渐隐的部分**硬切**掉；
 * - 相邻层之间、层与背景之间会出现细缝（露底）。
 * 所以把支撑按 4-邻域向外扩 `dilatePx`（默认 2）像素，外扩像素的深度取
 * **最近 seed 的深度**（BFS 多源扩散，`nearestSeed`）。外扩环采样到的几乎都是
 * 低 α 纹素，正面看不可见，斜视角下正好挡住缝 —— 这就是「深度外扩支撑透明外围」。
 *
 * ── 3. `nearestSeed` 是全局的 ──
 * BFS 不截断在 `dilatePx`：`nearestSeed` / `distance` 对整个图像有效。LOD 的
 * 边界贴合（`snapBoundary`）要用它把粗 grid 的角点贴到最近的 seed 上，
 * 截断会让「支撑外一圈」的角点无法贴合、剪影处整块丢格子。
 */

import type { WSplatFrame } from "../wsplat/types.ts"
import type { SupportOptions } from "./types.ts"

/** 只借支撑判定要用的字段，便于单测造最小对象。 */
export type SupportFrame = Pick<
  WSplatFrame,
  "width" | "height" | "alpha" | "depth" | "visible"
>

/** `computeSupportField` 的产物。 */
export interface SupportField {
  readonly width: number
  readonly height: number
  /** `1` = 该像素出几何（含外扩环）。 */
  readonly support: Uint8Array
  /** `1` = 真实可见（未外扩）。 */
  readonly seed: Uint8Array
  /** 每个像素的深度：seed 取自身，外扩处取**最近 seed** 的深度；无 seed 为 0。 */
  readonly depth: Float32Array
  /** 每个像素最近 seed 的下标（`-1` = 图内没有任何 seed）。 */
  readonly nearestSeed: Int32Array
  /** 到最近 seed 的 4-邻域步数（`0` = 自身是 seed；`-1` = 无 seed）。 */
  readonly distance: Int32Array
  readonly stats: SupportStats
}

/** `computeSupportField` 的诊断计数。 */
export interface SupportStats {
  /** 真实可见（非全透明）的像素数。 */
  readonly seedPixels: number
  /** 外扩进来的像素数（`support − seed`）。 */
  readonly dilatedPixels: number
  /** 外扩后、小岛清理前的支撑像素数。 */
  readonly hysteresisPixels: number
  /** 最终支撑像素数。 */
  readonly supportPixels: number
  /** 被小岛清理掉的像素数 / 块数。 */
  readonly removedIslandPixels: number
  readonly removedIslands: number
}

const DEFAULT_ALPHA_CUTOFF = 1 / 255
const DEFAULT_DILATE_PX = 2
const DEFAULT_MIN_ISLAND_PIXELS = 8

/**
 * 算一层的支撑场。纯 CPU、确定性。
 *
 * 步骤：seed 判据 -> 多源 BFS（最近 seed + 距离）-> 深度扩散 -> 小岛清理。
 */
export function computeSupportField(
  frame: SupportFrame,
  options: SupportOptions = {},
): SupportField {
  const { width, height, alpha, depth, visible } = frame
  const pixels = width * height
  if (alpha.length < pixels || depth.length < pixels) {
    throw new Error(
      `支撑场：缓冲长度不足（需要 ${pixels}，alpha=${alpha.length}, depth=${depth.length}）`,
    )
  }
  const alphaCutoff = Math.max(0, options.alphaCutoff ?? DEFAULT_ALPHA_CUTOFF)
  const dilatePx = Math.max(
    0,
    Math.floor(options.dilatePx ?? DEFAULT_DILATE_PX),
  )
  const minIsland = Math.max(
    0,
    Math.floor(options.minIslandPixels ?? DEFAULT_MIN_ISLAND_PIXELS),
  )

  // ── 1. seed ──
  const seed = new Uint8Array(pixels)
  const seedDepth = new Float32Array(pixels)
  let seedPixels = 0
  for (let i = 0; i < pixels; i++) {
    if (!visible[i] || !(depth[i] > 0) || !(alpha[i] > alphaCutoff)) continue
    seed[i] = 1
    seedDepth[i] = depth[i]
    seedPixels++
  }

  // ── 2. 多源 BFS：最近 seed + 距离（不截断，全图有效）──
  const nearestSeed = new Int32Array(pixels).fill(-1)
  const distance = new Int32Array(pixels).fill(-1)
  const queue = new Int32Array(pixels)
  let head = 0
  let tail = 0
  for (let i = 0; i < pixels; i++) {
    if (seed[i]) {
      nearestSeed[i] = i
      distance[i] = 0
      queue[tail++] = i
    }
  }
  while (head < tail) {
    const i = queue[head++]
    const x = i % width
    const y = (i - x) / width
    const src = nearestSeed[i]
    const d = distance[i] + 1
    if (x > 0) visit(i - 1, src, d)
    if (x + 1 < width) visit(i + 1, src, d)
    if (y > 0) visit(i - width, src, d)
    if (y + 1 < height) visit(i + width, src, d)
  }
  function visit(j: number, src: number, d: number): void {
    if (distance[j] !== -1) return
    distance[j] = d
    nearestSeed[j] = src
    queue[tail++] = j
  }

  // ── 3. 支撑 = seed + 半径内扩环；深度 = 最近 seed 的深度 ──
  const support = new Uint8Array(pixels)
  const outDepth = new Float32Array(pixels)
  let supportPixels = 0
  let dilatedPixels = 0
  for (let i = 0; i < pixels; i++) {
    const src = nearestSeed[i]
    if (src >= 0) outDepth[i] = seedDepth[src]
    if (!seed[i]) {
      if (dilatePx === 0 || distance[i] < 0 || distance[i] > dilatePx) continue
      dilatedPixels++
    }
    support[i] = 1
    supportPixels++
  }
  const hysteresisPixels = supportPixels

  // ── 4. 小岛清理（4-连通分量小于阈值整块丢）──
  let removedIslandPixels = 0
  let removedIslands = 0
  if (minIsland > 1 && supportPixels > 0) {
    const label = new Int32Array(pixels).fill(-1)
    const component = new Int32Array(pixels)
    let nextLabel = 0
    for (let start = 0; start < pixels; start++) {
      if (support[start] === 0 || label[start] !== -1) continue
      let size = 0
      head = 0
      tail = 0
      queue[tail++] = start
      label[start] = nextLabel
      while (head < tail) {
        const i = queue[head++]
        component[size++] = i
        const x = i % width
        const y = (i - x) / width
        if (x > 0) visitSupport(i - 1)
        if (x + 1 < width) visitSupport(i + 1)
        if (y > 0) visitSupport(i - width)
        if (y + 1 < height) visitSupport(i + width)
      }
      if (size < minIsland) {
        for (let k = 0; k < size; k++) support[component[k]] = 0
        removedIslandPixels += size
        removedIslands++
      }
      nextLabel++
    }
    function visitSupport(j: number): void {
      if (support[j] === 1 && label[j] === -1) {
        label[j] = nextLabel
        queue[tail++] = j
      }
    }
  }

  let finalSupportPixels = 0
  for (let i = 0; i < pixels; i++) finalSupportPixels += support[i]

  return {
    width,
    height,
    support,
    seed,
    depth: outDepth,
    nearestSeed,
    distance,
    stats: {
      seedPixels,
      dilatedPixels,
      hysteresisPixels,
      supportPixels: finalSupportPixels,
      removedIslandPixels,
      removedIslands,
    },
  }
}
