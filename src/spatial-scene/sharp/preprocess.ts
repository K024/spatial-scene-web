/**
 * 输入预处理：图像 -> [1,3,1536,1536] + disparity_factor。
 *
 * 严格对照 `predict.py: predict_image`：
 * ```
 * image_pt = image.float().permute(2,0,1) / 255.0       # HWC uint8 -> CHW [0,1]
 * disparity_factor = f_px / width
 * image_resized_pt = F.interpolate(image_pt[None], size=(1536,1536),
 *                                  mode="bilinear", align_corners=True)
 * ```
 *
 * ── 关键 bias 1：`align_corners=True` ──
 * PyTorch 的 bilinear + align_corners=True 映射为：
 *   `src = dst * (srcSize - 1) / (dstSize - 1)`
 * 而**不是** align_corners=False 的 `(dst + 0.5) * scale - 0.5`。
 * 两种映射在边缘的行为不同，直接复现时必须用前者。
 *
 * ── 关键 bias 2：通道顺序与值域 ──
 * 输入是 HWC uint8 RGB，先除 255 再 permute 成 CHW；本实现直接产出
 * CHW 布局的 Float32Array（NCHW 展平），并在注释中标明 stride 顺序。
 */

import { INTERNAL_RESOLUTION } from "./types.ts"

/** 源图像（HWC，RGB，uint8 或已归一化 float）。 */
export interface SourceImage {
  /** 像素数据，长度 = width*height*channels，行主序 HWC。 */
  data: Uint8Array | Float32Array
  width: number
  height: number
  /** 通道数；SHARP 只用前 3 个（RGB）。 */
  channels: number
}

/**
 * 双线性缩放，`align_corners=True` 语义（复刻 `F.interpolate`）。
 *
 * 对每个目标像素 `(dy, dx)`：
 *   `sy = dy * (H-1)/(H'-1)`, `sx = dx * (W-1)/(W'-1)`
 * 然后对 4 邻域加权。边界用 clamp（PyTorch 对越界索引的等价处理）。
 *
 * @param src 源像素访问器：`(y, x, c) => number`
 */
function bilinearSample(
  srcH: number,
  srcW: number,
  dstH: number,
  dstW: number,
  get: (y: number, x: number, c: number) => number,
  channels: number,
): Float32Array {
  const out = new Float32Array(channels * dstH * dstW)
  const scaleY = dstH > 1 ? (srcH - 1) / (dstH - 1) : 0
  const scaleX = dstW > 1 ? (srcW - 1) / (dstW - 1) : 0

  for (let dy = 0; dy < dstH; dy++) {
    const fy = dy * scaleY
    const y0 = Math.floor(fy)
    const y1 = Math.min(y0 + 1, srcH - 1)
    const wy = fy - y0
    for (let dx = 0; dx < dstW; dx++) {
      const fx = dx * scaleX
      const x0 = Math.floor(fx)
      const x1 = Math.min(x0 + 1, srcW - 1)
      const wx = fx - x0
      for (let c = 0; c < channels; c++) {
        const v00 = get(y0, x0, c)
        const v01 = get(y0, x1, c)
        const v10 = get(y1, x0, c)
        const v11 = get(y1, x1, c)
        const top = v00 + (v01 - v00) * wx
        const bottom = v10 + (v11 - v10) * wx
        // CHW 布局
        out[c * dstH * dstW + dy * dstW + dx] = top + (bottom - top) * wy
      }
    }
  }
  return out
}

/**
 * 对 CHW 展平的 float32 图像做双线性缩放（`align_corners=True`）。
 *
 * 这是 `preprocessImage` 内部用的同一段数学，单独导出供数值验证使用
 * （见 `scripts/compare-fixtures.ts` 阶段 1a：用确定性合成图与
 * `F.interpolate(align_corners=True)` 对拍，隔离掉 JPEG 解码器差异）。
 *
 * @param src CHW 展平源像素，长度 `channels*srcW*srcH`。
 * @returns CHW 展平结果，长度 `channels*dstW*dstH`。
 */
export function resizeBilinearChw(
  src: ArrayLike<number>,
  srcW: number,
  srcH: number,
  channels: number,
  dstW: number,
  dstH: number,
): Float32Array {
  return bilinearSample(
    srcH,
    srcW,
    dstH,
    dstW,
    (y, x, c) => src[c * srcH * srcW + y * srcW + x],
    channels,
  )
}

/**
 * 把源图像缩放到 1536×1536 并归一化到 [0,1]，返回 **CHW** 展平的 Float32Array。
 *
 * 对照 `predict.py: predict_image` 的 `image_pt -> image_resized_pt`。
 * 输出长度 = 3 * 1536 * 1536，布局为 [c][y][x]（即 NCHW 去掉 N）。
 *
 * 注意：若源图已是 Float32Array，则**不再除 255**（假定调用方已归一化）。
 * 该约定与 PyTorch 侧不同（那里恒除 255，因为 `load_rgb` 返回 uint8）。
 * 为消除歧义，本函数以 `data instanceof Uint8Array` 判定：uint8 才除 255。
 */
export function preprocessImage(
  image: SourceImage,
  internalResolution = INTERNAL_RESOLUTION,
): Float32Array {
  const { data, width, height, channels } = image
  const isU8 = data instanceof Uint8Array
  const inv = isU8 ? 1 / 255 : 1

  const get = (y: number, x: number, c: number): number => {
    const idx = (y * width + x) * channels + c
    return data[idx] * inv
  }

  // SHARP 只用前 3 通道；不足 3 通道的灰度图在 `io.load_rgb` 里已被复制成 3 通道。
  const useChannels = Math.min(channels, 3)
  const resized = bilinearSample(
    height,
    width,
    internalResolution,
    internalResolution,
    get,
    useChannels,
  )

  if (useChannels === 3) return resized

  // 不足 3 通道时补齐（复制到 3 通道），与 `io.load_rgb` 的 dstack 语义一致。
  const out = new Float32Array(3 * internalResolution * internalResolution)
  const plane = internalResolution * internalResolution
  for (let i = 0; i < plane; i++) {
    const v = resized[i]
    out[i] = v
    out[plane + i] = useChannels > 1 ? resized[plane + i] : v
    out[2 * plane + i] = useChannels > 2 ? resized[2 * plane + i] : v
  }
  return out
}
