/**
 * 层外缘余量（margin）—— `extendedTexture` 的**几何侧**（M2）。
 *
 * 阶段：meshing 的**支撑掩码之后、撕裂之前**。
 *
 * ── 作用 ──
 * 把支撑掩码向外膨胀 `radiusPixels`，环上像素用**最近支撑像素**的 uv / 深度：
 * 几何上把每层的表面延伸到剪影之外一小圈，盖住斜视角下近面与后层之间露出的细缝。
 * 它与裙边（`relief.ts` 的 skirt）互补：skirt 沿**视线向后**挤出，margin 在**屏幕平面向外**铺开。
 *
 * ── 为什么输出 `source` 映射，而不是只给一张更大的掩码 ──
 * 环上像素本身的颜色/深度是背景（低 α），直接用会把背景色拉出来。所以每个环像素要带着
 * **最近支撑像素**的 `uv`（采样边缘纹理）与深度。`source[i]` 就是这个「谁是它的源」的映射。
 * GL 的 `CLAMP_TO_EDGE` 只能在**纹理边界**做这件事，做不了剪影（剪影在纹理内部）。
 *
 * ── 算法 ──
 * 多源 BFS（最近源距离变换）：所有支撑像素是源，逐层向外扩，每个非源像素继承扩展它的
 * 那个像素的源。O(N)，不排序、不迭代。膨胀不会跨过**已被支撑覆盖**的像素去“穿透”它们。
 *
 * @see SOLIDI `mesh_tools.py:enlarge_border` / `fill_dummy_bord`（`extrapolation_thickness=60`）
 *   —— vt-vl-lab/3d-photo-inpainting @ de04467 (MIT)。SOLIDI 用 inpainting 发明外扩内容；
 *   我们只**拉伸边缘**（不发明），颜色/深度的真实度交给后层与回填。
 */

import type { MarginOptions } from "./types.ts"

/** `expandSupportWithMargin` 的结果。 */
export interface MarginResult {
  /** `width*height`，1 = 该像素出几何（支撑 ∪ 环）。 */
  readonly mask: Uint8Array
  /** `width*height`，每个 mask 像素的**源像素**下标（支撑像素指向自己）。 */
  readonly source: Int32Array
  /** 支撑像素数。 */
  readonly supportPixels: number
  /** 新增的环像素数。 */
  readonly marginPixels: number
}

/**
 * 把支撑掩码向外膨胀 `radiusPixels`，并给出逐像素源映射。
 *
 * `radiusPixels <= 0` 时原样返回（`source` 为恒等映射）。
 */
export function expandSupportWithMargin(
  support: Uint8Array,
  width: number,
  height: number,
  options: MarginOptions = {},
): MarginResult {
  const radius = Math.max(0, Math.floor(options.radiusPixels ?? 0))
  const pixels = width * height
  const source = new Int32Array(pixels).fill(-1)
  let supportPixels = 0
  for (let i = 0; i < pixels; i++) {
    if (support[i]) {
      source[i] = i
      supportPixels++
    }
  }
  if (radius <= 0) {
    return {
      mask: support,
      source,
      supportPixels,
      marginPixels: 0,
    }
  }

  const mask = Uint8Array.from(support)
  const distance = new Int32Array(pixels).fill(-1)
  const queue = new Int32Array(pixels)
  let head = 0
  let tail = 0
  for (let i = 0; i < pixels; i++) {
    if (support[i]) {
      distance[i] = 0
      queue[tail++] = i
    }
  }

  const relax = (from: number, to: number): void => {
    if (distance[to] !== -1) return
    if (distance[from] >= radius) return
    distance[to] = distance[from] + 1
    source[to] = source[from]
    mask[to] = 1
    queue[tail++] = to
  }

  while (head < tail) {
    const i = queue[head++]
    const x = i % width
    const y = (i - x) / width
    if (x > 0) relax(i, i - 1)
    if (x + 1 < width) relax(i, i + 1)
    if (y > 0) relax(i, i - width)
    if (y + 1 < height) relax(i, i + width)
  }

  let total = 0
  for (let i = 0; i < pixels; i++) total += mask[i]
  return {
    mask,
    source,
    supportPixels,
    marginPixels: total - supportPixels,
  }
}
