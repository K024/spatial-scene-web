/**
 * 颜色空间转换。
 *
 * 严格对照 `ml-sharp/src/sharp/utils/color_space.py`：
 *   - `sRGB2linearRGB` / `linearRGB2sRGB`（阈值分段，非纯 gamma）
 *   - `encode_color_space` / `decode_color_space`
 *
 * 关于 `robust_where` 的语义：原实现为了避免 backward 的 NaN，会先把
 * **非选中分支**的输入 clamp 到安全值，再求值该分支。前向数值上，
 * 由于 `torch.where` 只取选中分支，clamp 不影响前向结果——
 * 因此 JS 侧只需实现**前向**等价逻辑（阈值分段），无需复刻 safe-value 技巧。
 * 本实现显式注明这一点，避免后人误以为漏了东西。
 */

/** 对照 `color_space.py: encode_color_space`：sRGB -> 0，linearRGB -> 1。 */
export function encodeColorSpace(colorSpace: "sRGB" | "linearRGB"): number {
  return colorSpace === "sRGB" ? 0 : 1
}

/** 对照 `color_space.py: decode_color_space`。注意：非 0 一律解释为 linearRGB。 */
export function decodeColorSpace(index: number): "sRGB" | "linearRGB" {
  return index === 0 ? "sRGB" : "linearRGB"
}

/** `sRGB2linearRGB` 的阈值。对照 `color_space.py`。 */
const SRGB_THRESHOLD = 0.04045

/** `linearRGB2sRGB` 的阈值。对照 `color_space.py`。 */
const LINEAR_THRESHOLD = 0.0031308

/**
 * sRGB -> linearRGB，逐元素，原地写入 `out`（若提供）以避免分配。
 *
 * 对照 `color_space.py: sRGB2linearRGB`：
 *   `x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4`
 */
export function sRGB2linearRGB(
  input: Float32Array,
  out?: Float32Array,
): Float32Array {
  const dst = out ?? new Float32Array(input.length)
  for (let i = 0; i < input.length; i++) {
    const x = input[i]
    dst[i] = x <= SRGB_THRESHOLD ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4
  }
  return dst
}

/**
 * linearRGB -> sRGB，逐元素，原地写入 `out`（若提供）以避免分配。
 *
 * 对照 `color_space.py: linearRGB2sRGB`：
 *   `x <= 0.0031308 ? x * 12.92 : 1.055 * x ** (1/2.4) - 0.055`
 *
 * 注意指数是 `1/2.4`（不是 1/2.2）；base 用 `**` 而非 `Math.pow` 的差别可忽略。
 */
export function linearRGB2sRGB(
  input: Float32Array,
  out?: Float32Array,
): Float32Array {
  const dst = out ?? new Float32Array(input.length)
  for (let i = 0; i < input.length; i++) {
    const x = input[i]
    dst[i] = x <= LINEAR_THRESHOLD ? x * 12.92 : 1.055 * x ** (1 / 2.4) - 0.055
  }
  return dst
}
