/**
 * 焦距 / FOV 处理。
 *
 * 严格对照 `ml-sharp/src/sharp/utils/io.py`：
 *   - `convert_focallength`            mm -> px
 *   - `load_rgb` 中的 EXIF 焦距提取与 fallback 链
 *   - `predict.py: predict_image` 里 `disparity_factor = f_px / width` 的构造
 *
 * 为什么这个模块重要（见调研结论）：FOV 是模型输出的**唯一**度量锚定通道。
 * `disparity_factor` 同时进入网络特征输入、度量深度换算、NDC 反投影三处，
 * 给错焦距会整体改变场景尺度与视差分布。
 */

/** 35mm 胶片对角线（用于 mm -> px 换算）。对照 `io.py: convert_focallength`。 */
const FILM_DIAGONAL = Math.sqrt(36 ** 2 + 24 ** 2)

/**
 * 把以 mm 为单位的焦距换算为像素。
 *
 * 逐字复刻 `io.py: convert_focallength`：
 *   `f_mm * sqrt(w^2 + h^2) / sqrt(36^2 + 24^2)`
 *
 * 注意：用的是**对角线**换算（不是按宽），因此 f_px 与宽高比无关的直觉在这里不成立。
 */
export function convertFocallength(
  width: number,
  height: number,
  fMm = 30,
): number {
  return (fMm * Math.sqrt(width ** 2 + height ** 2)) / FILM_DIAGONAL
}

/**
 * EXIF 焦距提取的 fallback 链，复刻 `io.py: load_rgb`。
 *
 * 原逻辑：
 *   1. `f_35mm = FocalLengthIn35mmFilm ?? FocalLenIn35mmFilm`
 *   2. 若为 None 或 < 1 → `f_35mm = FocalLength`
 *   3. 若仍为 None → warn，取 30.0
 *   4. 若 `f_35mm < 10.0` → 认为是真实物理焦距而非等效焦距，`f_35mm *= 8.4`（crude）
 *
 * @param exifTags 已解析的 EXIF tag 字典（键为 tag 名，如 "FocalLengthIn35mmFilm"）。
 *                 允许 undefined，表示无 EXIF。
 * @returns 用于换算的 35mm 等效焦距（mm）。恒为有限正数。
 */
export function resolveFocalLength35mm(
  exifTags: Record<string, unknown> | undefined,
): number {
  let f35: number | null | undefined

  const t = exifTags ?? {}
  const fromTag =
    (t["FocalLengthIn35mmFilm"] as number | undefined) ??
    (t["FocalLenIn35mmFilm"] as number | undefined)

  if (fromTag === undefined || fromTag === null || fromTag < 1) {
    f35 = t["FocalLength"] as number | undefined
    if (f35 === undefined || f35 === null) {
      // 对照原实现：找不到焦距时默认 30mm。
      return 30.0
    }
  } else {
    f35 = fromTag
  }

  // 对照原实现：< 10mm 视为非 35mm 等效，乘 8.4 粗略修正。
  if (f35 < 10.0) {
    f35 = f35 * 8.4
  }
  return f35
}

/**
 * 由图像尺寸与 EXIF 焦距得到像素焦距 `f_px`。
 *
 * 复刻 `load_rgb` 的尾部：`f_px = convert_focallength(width, height, f_35mm)`。
 */
export function focalLengthPxFromExif(
  width: number,
  height: number,
  exifTags?: Record<string, unknown>,
): number {
  return convertFocallength(width, height, resolveFocalLength35mm(exifTags))
}

/**
 * 构造 `disparity_factor` 标量。
 *
 * 逐字复刻 `predict.py: predict_image`：
 *   `disparity_factor = torch.tensor([f_px / width])`
 *
 * **注意分母是 width（原始图像宽），不是 1536。** 这是最容易搞错的一处 bias：
 * 内部推理在 1536×1536 上做，但 disparity_factor 用原始宽归一化。
 */
export function disparityFactor(fPx: number, width: number): number {
  return fPx / width
}

/**
 * 计算 NDC→metric 反投影所需的 `intrinsics_resized`。
 *
 * 逐字复刻 `predict.py: predict_image`：
 * ```
 * intrinsics = [[f_px, 0, width/2, 0],
 *               [0, f_px, height/2, 0],
 *               [0, 0, 1, 0],
 *               [0, 0, 0, 1]]
 * intrinsics_resized[0] *= 1536 / width
 * intrinsics_resized[1] *= 1536 / height
 * ```
 * 即第 0 行整体乘 `1536/width`，第 1 行整体乘 `1536/height`（含主点与 0 项）。
 *
 * 返回 4x4 行主序矩阵（Array(16)）。
 */
export function intrinsicsResized(
  fPx: number,
  width: number,
  height: number,
  internalResolution = 1536,
): number[] {
  const sx = internalResolution / width
  const sy = internalResolution / height

  // 原始 intrinsics（4x4，行主序）
  const k = [
    fPx,
    0,
    width / 2,
    0,
    0,
    fPx,
    height / 2,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    1,
  ]

  // 第 0 行乘 sx，第 1 行乘 sy（与 torch 的 `intrinsics_resized[0] *= ...` 一致）
  for (let c = 0; c < 4; c++) {
    k[0 * 4 + c] *= sx
    k[1 * 4 + c] *= sy
  }
  return k
}
