/**
 * Node 侧图像加载 + EXIF 焦距提取。
 *
 * 目标：复刻 `ml-sharp/src/sharp/utils/io.py: load_rgb` 的行为，
 * 供 JS 推理脚本使用。
 *
 * 复刻点：
 *   1. EXIF 自动旋转（Orientation 3/6/8）；
 *   2. 焦距 fallback 链（见 `sharp/fov.ts: resolveFocalLength35mm`）；
 *   3. 返回 HWC RGB uint8 + f_px。
 *
 * 依赖 `sharp`（libvips 绑定，自带 EXIF 解析）。
 */

import { createRequire } from "node:module"
import {
  convertFocallength,
  resolveFocalLength35mm,
} from "../../src/spatial-scene/sharp/fov.ts"
import type { SourceImage } from "../../src/spatial-scene/sharp/preprocess.ts"

const require = createRequire(import.meta.url)

/** sharp 的最小接口（避免引入其类型依赖）。 */
interface SharpInstance {
  metadata(): Promise<{
    width?: number
    height?: number
    orientation?: number
    exif?: Buffer
  }>
  rotate(): SharpInstance
  extract(opts: {
    left: number
    top: number
    width: number
    height: number
  }): SharpInstance
  removeAlpha(): SharpInstance
  toColourspace(space: string): SharpInstance
  raw(): SharpInstance
  toBuffer(opts: { resolveWithObject: true }): Promise<{
    data: Buffer
    info: { width: number; height: number; channels: number }
  }>
}

type SharpFactory = (input: string | Buffer) => SharpInstance

const sharp = require("sharp") as SharpFactory

/**
 * 导出脚本用的最小 sharp 管线接口。
 *
 * 放这里（而不是让每个脚本各自 `createRequire`）是为了让「写 PNG」只有一份实现 ——
 * 之前 `wsplat-render.ts` 与 `wsplat-check-blit.ts` 各抄了一份 `sharpFromRgba`。
 */
export interface SharpPipeline {
  composite(
    items: readonly { input: Buffer; left: number; top: number }[],
  ): SharpPipeline
  resize(opts: {
    width?: number
    height?: number
    kernel?: string
  }): SharpPipeline
  png(): SharpPipeline
  jpeg(opts?: { quality?: number }): SharpPipeline
  toBuffer(): Promise<Buffer>
  toFile(path: string): Promise<unknown>
}

/** RGBA8 裸像素 -> sharp 管线（可 `.png().toFile(...)`）。 */
export function sharpFromRgba(
  rgba: Uint8Array,
  width: number,
  height: number,
): SharpPipeline {
  const factory = require("sharp") as (
    input: Buffer,
    opts: { raw: { width: number; height: number; channels: number } },
  ) => SharpPipeline
  return factory(Buffer.from(rgba.buffer, rgba.byteOffset, rgba.byteLength), {
    raw: { width, height, channels: 4 },
  })
}

/** 建一张纯色画布（拼图用）。 */
export function createSharpCanvas(
  width: number,
  height: number,
  rgba: readonly [number, number, number, number] = [0, 0, 0, 255],
): SharpPipeline {
  const factory = require("sharp") as (opts: {
    create: {
      width: number
      height: number
      channels: number
      background: { r: number; g: number; b: number; alpha: number }
    }
  }) => SharpPipeline
  const [r, g, b, a] = rgba
  return factory({
    create: {
      width,
      height,
      channels: 4,
      background: { r, g, b, alpha: a / 255 },
    },
  })
}

/** RGBA8 -> PNG Buffer（不落盘，供拼接）。 */
export async function rgbaToPngBuffer(
  rgba: Uint8Array,
  width: number,
  height: number,
): Promise<Buffer> {
  return sharpFromRgba(rgba, width, height).png().toBuffer()
}

/** 已加载的图像 + 元数据。 */
export interface LoadedImage {
  /** HWC RGB uint8 像素。 */
  image: SourceImage
  /** 原始图像域的像素焦距。 */
  fPx: number
  /** 用于换算的 35mm 等效焦距（mm），便于日志核对。 */
  focal35mm: number
  /** 是否来自 EXIF（false 表示使用了默认 30mm）。 */
  focalFromExif: boolean
}

/**
 * 从 EXIF buffer 解析出焦距相关的 tag 字典。
 *
 * 这里刻意**不**做完整 EXIF 解析：只需要 `FocalLength` /
 * `FocalLengthIn35mmFilm` / `FocalLenIn35mmFilm` 三个 tag。
 * 用 sharp 已解析的 `exif` buffer 做尽力而为的提取。
 *
 * 注意：sharp 的 `metadata().exif` 是 TIFF header 起始的原始块，
 * 解析成本高于收益；实践上 sharp 提供 `metadata()` 的 `exif` 但我们
 * 直接依赖 `sharp` 的 `withExif` 并不稳定。因此这里采用
 * **保守策略**：优先使用 EXIF（若能解析），否则落到默认 30mm，
 * 并在返回值里标注来源，让调用方显式决策（例如传入显式焦距覆盖）。
 */
function parseFocalFromExifBuffer(
  exif: Buffer | undefined,
): Record<string, unknown> | undefined {
  if (!exif || exif.length < 14) return undefined

  const tags: Record<string, unknown> = {}
  try {
    const le = exif.readUInt16LE(0) === 0x4949 // "II" = little endian
    const read16 = (o: number): number =>
      le ? exif.readUInt16LE(o) : exif.readUInt16BE(o)
    const read32 = (o: number): number =>
      le ? exif.readUInt32LE(o) : exif.readUInt32BE(o)

    // TIFF header: [byteOrder(2), 42(2), ifdOffset(4)]
    let ifd = read32(4)
    // 只扫第一层 IFD + Exif 子 IFD（0x8769）
    for (let pass = 0; pass < 2 && ifd > 0 && ifd + 2 < exif.length; pass++) {
      const count = read16(ifd)
      let next = 0
      for (let i = 0; i < count; i++) {
        const entry = ifd + 2 + i * 12
        if (entry + 12 > exif.length) break
        const tag = read16(entry)
        const type = read16(entry + 2)
        const num = read32(entry + 4)
        // 只关心 RATIONAL(5) 且 count=1 的焦距，与 SHORT(3) 的 35mm 等效
        if (tag === 0x829a || tag === 0x920a) {
          // FocalLength(0x829a) / FocalLengthIn35mmFilm(0xa405 实为 SHORT)
          if (type === 5 && num === 1) {
            const valOff = entry + 8
            if (valOff + 8 <= exif.length) {
              const num2 = read32(valOff)
              const den = read32(valOff + 4)
              if (den !== 0) {
                const v = num2 / den
                tags[tag === 0x829a ? "FocalLength" : "FocalLengthIn35mmFilm"] =
                  v
              }
            }
          }
        } else if (tag === 0xa405) {
          // FocalLengthIn35mmFilm (SHORT)
          if (type === 3) tags["FocalLengthIn35mmFilm"] = read16(entry + 8)
        } else if (tag === 0x8769) {
          next = read32(entry + 8)
        }
      }
      ifd = next
    }
  } catch {
    return undefined
  }
  return Object.keys(tags).length > 0 ? tags : undefined
}

/**
 * 加载图像，复刻 `io.load_rgb`。
 *
 * @param path 图像路径。
 * @param explicitFocal35mm 显式指定 35mm 等效焦距；给出时**跳过 EXIF**
 *   （便于做「同一张图不同 FOV」的受控实验）。
 * @param crop 可选取图区域 `[x, y, w, h]`（像素）。用于从拼图中裁出单张照片。
 */
export async function loadImage(
  path: string,
  explicitFocal35mm?: number,
  crop?: [number, number, number, number],
): Promise<LoadedImage> {
  const meta = await sharp(path).metadata()

  // 1. 自动旋转（对照 io.load_rgb 的 Orientation 3/6/8 分支）
  let pipeline = sharp(path)
  const orientation = meta.orientation ?? 1
  if (orientation !== 1) {
    pipeline = pipeline.rotate()
  }
  // 2. 取图（若有），再去除 alpha + 保证 RGB（对照 remove_alpha 与 dstack 灰度复制）
  if (crop) {
    pipeline = pipeline.extract({
      left: crop[0],
      top: crop[1],
      width: crop[2],
      height: crop[3],
    })
  }
  pipeline = pipeline.removeAlpha().toColourspace("srgb")

  const { data, info } = await pipeline
    .raw()
    .toBuffer({ resolveWithObject: true })
  const width = info.width
  const height = info.height
  const channels = info.channels

  // 3. 焦距解析
  let focal35mm: number
  let focalFromExif: boolean
  if (explicitFocal35mm !== undefined) {
    focal35mm = explicitFocal35mm
    focalFromExif = false
  } else {
    const tags = parseFocalFromExifBuffer(meta.exif)
    focal35mm = resolveFocalLength35mm(tags)
    focalFromExif = tags !== undefined
  }

  const fPx = convertFocallength(width, height, focal35mm)

  return {
    image: {
      data: new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
      width,
      height,
      channels,
    },
    fPx,
    focal35mm,
    focalFromExif,
  }
}
