/**
 * 极简 PNG 编码器（RGBA8，filter 0，**stored** deflate）。
 *
 * ── 为什么自己写 ──
 * meshing 的产物要能「任何 three.js 场景直接吃」，纹理必须是 PNG（glTF 只认 PNG/JPEG）。
 * 平台相关的编码器（sharp / canvas）会把 `src/` 拖进 node 或 DOM 依赖；引第三方
 * 纯 JS 编码器（pako 等）又要新增依赖。glTF 的 ZIP 语义不需要压缩率 —— 本仓库
 * 明确「暂不做体积优化」—— 所以用 zlib 的 **stored（不压缩）块**：一个文件、
 * 零依赖、浏览器 / node 都能跑。
 *
 * 代价是文件大（约 4 字节/像素 + 行开销），10 层 1024×768 约 30 MB。
 * 要缩体积时把 `encodePngRgba8` 换成带 deflate 的实现即可（接口不变）。
 *
 * 参考：PNG spec (ISO/IEC 15948) §4.1 签名、§11.2.4 IHDR、§10.2 IDAT、
 * RFC 1951 §3.2.4 stored block、RFC 1950 §9 adler32。
 */

const PNG_SIGNATURE = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10])

/** zlib stored 块的最大原始字节数。 */
const MAX_STORED_BLOCK = 0xffff

/**
 * RGBA8 像素 -> PNG 字节。
 *
 * @param rgba 长度必须 `>= width * height * 4`（行主序，左上原点）。
 */
export function encodePngRgba8(
  rgba: Uint8Array,
  width: number,
  height: number,
): Uint8Array {
  if (
    !Number.isInteger(width) ||
    !Number.isInteger(height) ||
    width < 1 ||
    height < 1
  ) {
    throw new Error(`PNG 尺寸非法: ${width}x${height}`)
  }
  const rowBytes = width * 4
  if (rgba.length < rowBytes * height) {
    throw new Error(
      `PNG 像素不足：需要 ${rowBytes * height}，收到 ${rgba.length}`,
    )
  }

  // ── 原始扫描线：每行前面加一个 filter 字节（0 = None）──
  const raw = new Uint8Array((rowBytes + 1) * height)
  for (let y = 0; y < height; y++) {
    raw[y * (rowBytes + 1)] = 0
    raw.set(
      rgba.subarray(y * rowBytes, y * rowBytes + rowBytes),
      y * (rowBytes + 1) + 1,
    )
  }

  const idat = zlibStored(raw)
  const ihdr = new Uint8Array(13)
  writeU32(ihdr, 0, width)
  writeU32(ihdr, 4, height)
  ihdr[8] = 8 // bit depth
  ihdr[9] = 6 // color type: truecolor + alpha
  ihdr[10] = 0 // compression
  ihdr[11] = 0 // filter
  ihdr[12] = 0 // interlace

  const chunks = [
    PNG_SIGNATURE,
    chunk("IHDR", ihdr),
    chunk("IDAT", idat),
    chunk("IEND", new Uint8Array(0)),
  ]
  let total = 0
  for (const c of chunks) total += c.length
  const out = new Uint8Array(total)
  let offset = 0
  for (const c of chunks) {
    out.set(c, offset)
    offset += c.length
  }
  return out
}

/** zlib 流（RFC 1950）：header + stored deflate 块 + adler32。 */
function zlibStored(data: Uint8Array): Uint8Array {
  const blocks = Math.max(1, Math.ceil(data.length / MAX_STORED_BLOCK))
  const deflate = new Uint8Array(data.length + blocks * 5)
  let cursor = 0
  for (let i = 0; i < blocks; i++) {
    const start = i * MAX_STORED_BLOCK
    const len = Math.min(MAX_STORED_BLOCK, data.length - start)
    const final = i === blocks - 1 ? 1 : 0
    deflate[cursor++] = final // BFINAL + BTYPE=00
    deflate[cursor++] = len & 0xff
    deflate[cursor++] = (len >>> 8) & 0xff
    deflate[cursor++] = ~len & 0xff
    deflate[cursor++] = (~len >>> 8) & 0xff
    deflate.set(data.subarray(start, start + len), cursor)
    cursor += len
  }

  const out = new Uint8Array(2 + cursor + 4)
  out[0] = 0x78 // CM=8, CINFO=7
  out[1] = 0x01 // FCHECK 使 (0x78<<8|0x01) % 31 == 0，无预设字典
  out.set(deflate.subarray(0, cursor), 2)
  writeU32(out, 2 + cursor, adler32(data))
  return out
}

/** 一个 PNG chunk：`length | type | data | crc(type+data)`。 */
function chunk(type: string, data: Uint8Array): Uint8Array {
  const out = new Uint8Array(12 + data.length)
  writeU32(out, 0, data.length)
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i)
  out.set(data, 8)
  writeU32(out, 8 + data.length, crc32(out.subarray(4, 8 + data.length)))
  return out
}

function writeU32(target: Uint8Array, offset: number, value: number): void {
  target[offset] = (value >>> 24) & 0xff
  target[offset + 1] = (value >>> 16) & 0xff
  target[offset + 2] = (value >>> 8) & 0xff
  target[offset + 3] = value & 0xff
}

/** CRC-32（IEEE 802.3，PNG 用）。查表懒建。 */
let crcTable: Int32Array | undefined
function crc32(data: Uint8Array): number {
  if (!crcTable) {
    crcTable = new Int32Array(256)
    for (let n = 0; n < 256; n++) {
      let c = n
      for (let k = 0; k < 8; k++) {
        c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
      }
      crcTable[n] = c
    }
  }
  let crc = -1
  for (let i = 0; i < data.length; i++) {
    crc = crcTable[(crc ^ data[i]) & 0xff] ^ (crc >>> 8)
  }
  return (crc ^ -1) >>> 0
}

/** adler32（RFC 1950 §9）。 */
function adler32(data: Uint8Array): number {
  const MOD = 65521
  let a = 1
  let b = 0
  // 每 5552 字节取模一次，避免 32 位溢出（a,b 的上界推导见 zlib）。
  for (let i = 0; i < data.length; ) {
    const end = Math.min(i + 5552, data.length)
    for (; i < end; i++) {
      a += data[i]
      b += a
    }
    a %= MOD
    b %= MOD
  }
  return ((b << 16) | a) >>> 0
}
