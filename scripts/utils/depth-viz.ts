/**
 * 深度图 / 视差图 / 透射率图的独立可视化（node 侧，写 PNG）。
 *
 * ── 为什么不叠在颜色上（用户要求）──
 * 深度必须是**独立**的一张图：一旦 alpha 混到颜色上，人眼就无法区分
 * 「深度错了」还是「颜色错了」。所以这里每个量各出一张 PNG，不合成、不叠加。
 *
 * ── 映射约定（写在文件名与日志里，避免"看图猜标定"）──
 *   - `depth`：真实度量深度 D（米），**归一化到 [zMin, zMax]，近 = 白、远 = 黑**。
 *     zMin/zMax 取可见像素的稳健分位数（默认 1% / 99%），并打印出来；
 *     文件名带 `_z{min}-{max}m` 后缀（保留 2 位小数）以固定标定。
 *   - `disparity`：**归一化视差** `d = (1/z - 1/zMax) / (1/zMin - 1/zMax)`，
 *     0 = 最远、1 = 最近，用彩色 ramp。这就是下游 `DisparityStats` 那套量化方式
 *     （下游 `DisparityStats` 用的就是这套量化），所以比看 D 更容易发现分层是否合理。
 *   - `transmission`：T = 1 - A，线性灰度（1 = 完全透明/背景）。
 *   - 背景（`visible = 0`）一律画成洋红（magenta）——一眼可辨"没有数据"与"很远"。
 */

/** 归一化区间。 */
export interface DepthRange {
  min: number
  max: number
}

/** 背景（无数据）像素的颜色：洋红。 */
const BACKGROUND_RGBA: readonly [number, number, number, number] = [
  255, 0, 255, 255,
]

/** 取可见像素的稳健分位数作为归一化区间。 */
export function estimateDepthRange(
  depth: Float32Array,
  visible: Uint8Array,
  lowPercentile = 0.01,
  highPercentile = 0.99,
): DepthRange {
  const values: number[] = []
  for (let i = 0; i < visible.length; i++) {
    if (visible[i] && depth[i] > 0) values.push(depth[i])
  }
  if (values.length === 0) return { min: 0, max: 1 }
  values.sort((a, b) => a - b)
  const lo =
    values[
      Math.min(values.length - 1, Math.floor(values.length * lowPercentile))
    ]
  const hi =
    values[
      Math.min(values.length - 1, Math.floor(values.length * highPercentile))
    ]
  return hi > lo ? { min: lo, max: hi } : { min: lo, max: lo + 1e-6 }
}

/**
 * 深度图（灰度，近 = 白）。`range` 之外的深度被夹住。
 *
 * 传入同一个 `range` 就能把「我们的 D」与「SHARP 的深度估计」放在同一标定下直接比。
 */
export function depthToGrayPng(
  depth: Float32Array,
  visible: Uint8Array,
  range: DepthRange,
): Uint8Array {
  const count = visible.length
  const out = new Uint8Array(count * 4)
  const span = Math.max(range.max - range.min, 1e-9)
  for (let i = 0; i < count; i++) {
    if (!visible[i] || !(depth[i] > 0)) {
      writePixel(out, i, BACKGROUND_RGBA)
      continue
    }
    // 近 -> 白：先用 (z - min)/span 得到 0..1（近=0），再取反
    const t = clamp01((depth[i] - range.min) / span)
    const v = Math.round((1 - t) * 255)
    writePixel(out, i, [v, v, v, 255])
  }
  return out
}

/**
 * 归一化视差图（彩色 ramp）。`d = (1/z - 1/zMax) / (1/zMin - 1/zMax)`。
 *
 * 用逆深度而不是线性深度，是因为分层的层边界/分位数统计都在视差域做
 * （下游 `DisparityStats` 的量化域）。
 */
export function disparityToColorPng(
  depth: Float32Array,
  visible: Uint8Array,
  range: DepthRange,
): Uint8Array {
  const count = visible.length
  const out = new Uint8Array(count * 4)
  const invMin = 1 / range.min
  const invMax = 1 / range.max
  const invSpan = invMin - invMax
  for (let i = 0; i < count; i++) {
    if (!visible[i] || !(depth[i] > 0)) {
      writePixel(out, i, BACKGROUND_RGBA)
      continue
    }
    const d = invSpan > 0 ? (1 / depth[i] - invMax) / invSpan : 0
    const [r, g, b] = turboRamp(clamp01(d))
    writePixel(out, i, [r, g, b, 255])
  }
  return out
}

/** 透射率图（灰度）：1 = 完全透明（背景方向），0 = 完全不透明。 */
export function transmissionToGrayPng(transmission: Float32Array): Uint8Array {
  const count = transmission.length
  const out = new Uint8Array(count * 4)
  for (let i = 0; i < count; i++) {
    const v = Math.round(clamp01(transmission[i]) * 255)
    writePixel(out, i, [v, v, v, 255])
  }
  return out
}

/** alpha 覆盖率图（灰度）：本阶段主要是排查"整屏都被半透明高斯糊住"。 */
export function alphaToGrayPng(alpha: Float32Array): Uint8Array {
  const count = alpha.length
  const out = new Uint8Array(count * 4)
  for (let i = 0; i < count; i++) {
    const v = Math.round(clamp01(alpha[i]) * 255)
    writePixel(out, i, [v, v, v, 255])
  }
  return out
}

/**
 * 简化 turbo 风格颜色 ramp（Frederic 版权的 turbo 太长，这里用 5 段线性插值）。
 *
 * 只求"视差高的地方一眼能看出来"，不追求色觉友好/科学配色。
 */
function turboRamp(t: number): [number, number, number] {
  const stops: [number, [number, number, number]][] = [
    [0.0, [48, 18, 59]],
    [0.25, [33, 144, 240]],
    [0.5, [74, 220, 133]],
    [0.75, [244, 214, 73]],
    [1.0, [180, 40, 20]],
  ]
  for (let i = 0; i + 1 < stops.length; i++) {
    const [t0, c0] = stops[i]
    const [t1, c1] = stops[i + 1]
    if (t <= t1) {
      const f = (t - t0) / (t1 - t0)
      return [
        Math.round(c0[0] + (c1[0] - c0[0]) * f),
        Math.round(c0[1] + (c1[1] - c0[1]) * f),
        Math.round(c0[2] + (c1[2] - c0[2]) * f),
      ]
    }
  }
  return stops[stops.length - 1][1]
}

function writePixel(
  out: Uint8Array,
  index: number,
  rgba: readonly [number, number, number, number],
): void {
  out[index * 4] = rgba[0]
  out[index * 4 + 1] = rgba[1]
  out[index * 4 + 2] = rgba[2]
  out[index * 4 + 3] = rgba[3]
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x
}
