/**
 * α 迟滞支撑掩码 + 小岛剔除。
 *
 * 阶段：meshing 的第一步（在撕裂之前）。输出 `support: Uint8Array`，
 * 决定「这一层哪些像素出几何」。
 *
 * ── 为什么是迟滞（hysteresis），不是单阈值 ──
 * 单阈值 `α > τ` 有两头失败：
 * 1. **层带交界软边**（`0 < α < τ`）被整片丢掉 → 相邻层之间能量缺口 → 接缝变暗 / 露底
 *    （`layering-perlayer-quality` 实测单层半 α 最坏 13.07%，不能一概丢掉）；
 * 2. **背景里的孤立低 α 噪点**被保留 → 幽灵面。
 * 迟滞把两者分开：`α >= aHi` 是**强**（确定表面）；`α >= aLo` 是**弱**（候选）；
 * 只有**与强像素 4-连通**的弱像素才进支撑。代价 O(N)，就是两趟泛洪。
 *
 * ── 为什么还要小岛剔除 ──
 * 迟滞之后仍可能留下「孤立的强岛」（例如稀疏 splat 在背景里碰巧叠出 α>0.5 的斑点）。
 * 它们在 mesh 阶段是纯浪费（一个没有邻接的碎片），而且斜视角下会变成漂在空中的小块。
 * 阈值 `minIslandPixels` 按像素数剔。取大了会吃掉真实细结构（栏杆、发丝），所以默认很小。
 *
 * ── 边界 ──
 * - `strongAlpha` 会被夹到 `>= weakAlpha`（配置颠倒时不让语义崩）。
 * - 无强像素时支撑为空（不是「全图」）—— 空层在 golden 里是允许的（`uniform` 基线）。
 * - 无效像素（`visible=0` / `depth<=0`）一律排除，即使 α 很高。
 * - 小岛剔除在**支撑**上做 4-连通；连通用显式栈，不用递归（大图会爆栈）。
 *
 * @see SOLIDI `mesh.py:generate_init_node`（`min_node_in_cc` 小岛剔除）——
 *   vt-vl-lab/3d-photo-inpainting @ de04467 (MIT)。本模块把它的「图连通域」换成数组泛洪。
 */

import type { WSplatFrame } from "../wsplat/types.ts"
import type { AlphaToleranceOptions } from "./types.ts"

/** `computeSupportMask` 的结果与诊断。 */
export interface SupportMaskResult {
  /** `width*height`，1 = 该像素出几何。 */
  readonly support: Uint8Array
  /** 强像素数（`α >= aHi` 且有效）。 */
  readonly strongPixels: number
  /** 弱候选像素数（`aLo <= α < aHi` 且有效）。 */
  readonly weakPixels: number
  /** 迟滞后的支撑像素数（小岛剔除前）。 */
  readonly hysteresisPixels: number
  /** 最终支撑像素数。 */
  readonly supportPixels: number
  /** 被小岛剔除的像素数。 */
  readonly removedIslandPixels: number
  /** 被剔除的小岛个数。 */
  readonly removedIslands: number
}

/** 只借支撑判定要用的字段，便于单测造最小对象。 */
export type SupportMaskFrame = Pick<
  WSplatFrame,
  "alpha" | "visible" | "depth" | "width" | "height"
>

const DEFAULT_STRONG_ALPHA = 0.5
const DEFAULT_WEAK_ALPHA = 0.05
const DEFAULT_MIN_ISLAND_PIXELS = 32

/**
 * 算一层的支撑掩码。
 *
 * 步骤：有效性过滤 -> 强/弱分类 -> 从强像素 4-连通泛洪进弱像素（迟滞）-> 小岛剔除。
 */
export function computeSupportMask(
  frame: SupportMaskFrame,
  options: AlphaToleranceOptions = {},
): SupportMaskResult {
  const { alpha, visible, depth, width, height } = frame
  const pixels = width * height
  if (alpha.length < pixels || depth.length < pixels) {
    throw new Error(
      `支撑掩码：缓冲长度不足（需要 ${pixels}，alpha=${alpha.length}, depth=${depth.length}）`,
    )
  }
  const weakAlpha = clamp01(options.weakAlpha ?? DEFAULT_WEAK_ALPHA)
  const strongAlpha = Math.max(
    weakAlpha,
    clamp01(options.strongAlpha ?? DEFAULT_STRONG_ALPHA),
  )
  const minIsland = Math.max(
    0,
    Math.floor(options.minIslandPixels ?? DEFAULT_MIN_ISLAND_PIXELS),
  )

  // 0 = 无效；1 = 弱候选；2 = 强
  const state = new Uint8Array(pixels)
  let strongPixels = 0
  let weakPixels = 0
  for (let i = 0; i < pixels; i++) {
    if (!visible[i] || !(depth[i] > 0)) continue
    const a = alpha[i]
    if (!(a >= weakAlpha)) continue
    if (a >= strongAlpha) {
      state[i] = 2
      strongPixels++
    } else {
      state[i] = 1
      weakPixels++
    }
  }

  // 迟滞泛洪：把与强像素 4-连通的弱像素提升为支撑。
  // 用显式栈（Int32Array 当栈），避免大图递归爆栈。
  const support = new Uint8Array(pixels)
  const stack = new Int32Array(pixels)
  let stackTop = 0
  for (let i = 0; i < pixels; i++) {
    if (state[i] === 2) {
      support[i] = 1
      stack[stackTop++] = i
    }
  }
  while (stackTop > 0) {
    const i = stack[--stackTop]
    const x = i % width
    const y = (i - x) / width
    if (x > 0) pushWeak(i - 1)
    if (x + 1 < width) pushWeak(i + 1)
    if (y > 0) pushWeak(i - width)
    if (y + 1 < height) pushWeak(i + width)
  }
  function pushWeak(j: number): void {
    // 只吞弱像素；强像素已经进过栈（且 support 已置 1）。
    if (state[j] === 1 && support[j] === 0) {
      support[j] = 1
      stack[stackTop++] = j
    }
  }

  let hysteresisPixels = 0
  for (let i = 0; i < pixels; i++) hysteresisPixels += support[i]

  // 小岛剔除：对支撑做 4-连通标记，统计每个分量的像素数，小于阈值则整块清掉。
  let removedIslandPixels = 0
  let removedIslands = 0
  if (minIsland > 1) {
    const label = new Int32Array(pixels).fill(-1)
    const component: number[] = []
    let componentId = 0
    for (let seed = 0; seed < pixels; seed++) {
      if (support[seed] === 0 || label[seed] !== -1) continue
      component.length = 0
      stackTop = 0
      stack[stackTop++] = seed
      label[seed] = componentId
      while (stackTop > 0) {
        const i = stack[--stackTop]
        component.push(i)
        const x = i % width
        const y = (i - x) / width
        if (x > 0) visit(i - 1)
        if (x + 1 < width) visit(i + 1)
        if (y > 0) visit(i - width)
        if (y + 1 < height) visit(i + width)
      }
      if (component.length < minIsland) {
        for (const i of component) support[i] = 0
        removedIslandPixels += component.length
        removedIslands++
      }
      componentId++
      function visit(j: number): void {
        if (support[j] === 1 && label[j] === -1) {
          label[j] = componentId
          stack[stackTop++] = j
        }
      }
    }
  }

  let supportPixels = 0
  for (let i = 0; i < pixels; i++) supportPixels += support[i]

  return {
    support,
    strongPixels,
    weakPixels,
    hysteresisPixels,
    supportPixels,
    removedIslandPixels,
    removedIslands,
  }
}

function clamp01(value: number): number {
  return value < 0 ? 0 : value > 1 ? 1 : value
}
