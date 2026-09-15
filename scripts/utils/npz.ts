/**
 * 极简 NPZ 读取器（只支持本仓库用到的子集）。
 *
 * NPZ = ZIP（本仓库由 numpy.savez 产生，条目为 **stored** 未压缩）。
 * 每个条目是 .npy v1.0 格式：magic + header dict + 原始数据。
 *
 * ⚠ ZIP64：numpy 的 `savez` 用 `force_zip64=True` 打开条目，所以**即使文件很小**，
 * local header 里的 `compressed_size` / `uncompressed_size` 也写成 `0xFFFFFFFF`，
 * 真实 8 字节值放在 extra field `id=0x0001` 里。这里必须解 ZIP64，否则会把
 * 「读到文件尾」当成解压结果（表现为长度不符）。
 *
 * 支持 dtype：<f4 / <f8 / <i4 / <u4 / <i8 / |u1 / <u2。
 * 不支持：Fortran order（本仓库不用）、结构化 dtype。
 *
 * 之所以手写而不引依赖：只需要读几个固定文件，且希望 web/node 都能用。
 */

import { readFileSync } from "node:fs"
import { inflateRawSync } from "node:zlib"

export interface NpyArray {
  data: Float32Array
  shape: number[]
  dtype: string
}

/** 读取 npy 文件体（从 header 之后开始）。 */
function parseNpy(buf: Buffer): NpyArray {
  if (buf.readUInt8(0) !== 0x93 || buf.toString("latin1", 1, 6) !== "NUMPY") {
    throw new Error("不是合法的 .npy")
  }
  const major = buf.readUInt8(6)
  if (major !== 1) throw new Error(`只支持 .npy v1.0，收到 v${major}`)

  const headerLen = buf.readUInt16LE(8)
  const headerStr = buf.toString("latin1", 10, 10 + headerLen)
  const bodyStart = 10 + headerLen

  const descrMatch = headerStr.match(/'descr':\s*'([^']+)'/)
  const fortranMatch = headerStr.match(/'fortran_order':\s*(True|False)/)
  const shapeMatch = headerStr.match(/'shape':\s*\(([^)]*)\)/)
  if (!descrMatch || !shapeMatch)
    throw new Error(`无法解析 npy header: ${headerStr}`)

  const descr = descrMatch[1]
  const isFortran = fortranMatch?.[1] === "True"

  const shape = shapeMatch[1]
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map((s) => Number.parseInt(s, 10))

  const count = shape.reduce((a, b) => a * b, 1)
  const body = buf.subarray(bodyStart)

  // 读原始元素（先不管顺序）
  const readAt = (i: number): number => {
    switch (descr) {
      case "<f4":
        return body.readFloatLE(i * 4)
      case "<f8":
        return body.readDoubleLE(i * 8)
      case "<i4":
        return body.readInt32LE(i * 4)
      case "<u4":
        return body.readUInt32LE(i * 4)
      case "|u1":
        return body.readUInt8(i)
      default:
        throw new Error(`不支持的 dtype: ${descr}`)
    }
  }

  let data: Float32Array
  if (!isFortran || shape.length <= 1) {
    data = new Float32Array(count)
    for (let i = 0; i < count; i++) data[i] = readAt(i)
  } else {
    // Fortran order：**第一维变化最快**。
    //
    // 本仓库的 fixture 里只有 [1,N,C] 形状会出现 F-order（由 torch 的
    // `flatten(0,1)` + `.numpy()` 产生）。目标是把它们重排成与 C-order 相同的
    // 行主序语义，这样 JS 侧读到的扁平 [N,C] 与 C-order 一致。
    //
    // 通用做法：对每个 C-order 线性下标 dst，求出其多维索引，再算 F-order 的源下标。
    const ndim = shape.length
    const cStrides = new Array<number>(ndim) // C-order 行主序步长
    const fStrides = new Array<number>(ndim) // Fortran 列主序步长
    cStrides[ndim - 1] = 1
    for (let d = ndim - 2; d >= 0; d--)
      cStrides[d] = cStrides[d + 1] * shape[d + 1]
    fStrides[0] = 1
    for (let d = 1; d < ndim; d++) fStrides[d] = fStrides[d - 1] * shape[d - 1]

    const idx = new Array<number>(ndim).fill(0)
    data = new Float32Array(count)
    for (let dst = 0; dst < count; dst++) {
      // dst (C-order) -> 多维索引
      let rem = dst
      for (let d = 0; d < ndim; d++) {
        idx[d] = Math.floor(rem / cStrides[d])
        rem -= idx[d] * cStrides[d]
      }
      // 多维索引 -> F-order 源下标
      let src = 0
      for (let d = 0; d < ndim; d++) src += idx[d] * fStrides[d]
      data[dst] = readAt(src)
    }
  }

  return { data, shape, dtype: descr }
}

/**
 * 解出条目的真实 `compressed_size` / `uncompressed_size`。
 *
 * local header 里这两个字段是 32 位；值为 `0xFFFFFFFF` 时表示「真值在 ZIP64
 * extra field（id=0x0001）里」，且 extra field 内**只含被置位的那些字段**，
 * 顺序固定为 original size（未压缩）→ compressed size。
 */
function resolveSizes(input: {
  buf: Buffer
  extraStart: number
  extraLen: number
  compSizeRaw: number
  uncompSizeRaw: number
  name: string
}): { compSize: number; uncompSize: number } {
  const { buf, extraStart, extraLen, compSizeRaw, uncompSizeRaw, name } = input
  let compSize = compSizeRaw
  let uncompSize = uncompSizeRaw
  if (compSizeRaw !== 0xffffffff && uncompSizeRaw !== 0xffffffff) {
    return { compSize, uncompSize }
  }

  const extraEnd = extraStart + extraLen
  for (let e = extraStart; e + 4 <= extraEnd; ) {
    const id = buf.readUInt16LE(e)
    const size = buf.readUInt16LE(e + 2)
    let p = e + 4
    if (id === 0x0001) {
      const fieldEnd = Math.min(p + size, extraEnd)
      if (uncompSizeRaw === 0xffffffff && p + 8 <= fieldEnd) {
        uncompSize = Number(buf.readBigUInt64LE(p))
        p += 8
      }
      if (compSizeRaw === 0xffffffff && p + 8 <= fieldEnd) {
        compSize = Number(buf.readBigUInt64LE(p))
      }
      break
    }
    e += 4 + size
  }

  if (compSize === 0xffffffff || uncompSize === 0xffffffff) {
    throw new Error(`entry ${name} 声明了 ZIP64 但 extra field 里没有尺寸`)
  }
  return { compSize, uncompSize }
}

/** 读取整个 .npz，返回 name -> array。 */
export function loadNpz(path: string): Record<string, NpyArray> {
  const buf = readFileSync(path)
  const out: Record<string, NpyArray> = {}

  // 线性扫描 local file header（本仓库的 npz 不含 zip64 / 加密）
  for (let i = 0; i + 30 <= buf.length; ) {
    if (buf.readUInt32LE(i) !== 0x04034b50) break
    const method = buf.readUInt16LE(i + 8)
    const compSizeRaw = buf.readUInt32LE(i + 18)
    const uncompSizeRaw = buf.readUInt32LE(i + 22)
    const nameLen = buf.readUInt16LE(i + 26)
    const extraLen = buf.readUInt16LE(i + 28)
    const name = buf.toString("latin1", i + 30, i + 30 + nameLen)
    const dataStart = i + 30 + nameLen + extraLen

    const { compSize, uncompSize } = resolveSizes({
      buf,
      extraStart: i + 30 + nameLen,
      extraLen,
      compSizeRaw,
      uncompSizeRaw,
      name,
    })
    const raw = buf.subarray(dataStart, dataStart + compSize)

    let body: Buffer
    if (method === 0) {
      body = raw
    } else if (method === 8) {
      body = inflateRawSync(raw)
    } else {
      throw new Error(`不支持的压缩方法 ${method}（entry=${name}）`)
    }
    if (body.length !== uncompSize) {
      throw new Error(
        `entry ${name} 解压后长度不符: ${body.length} != ${uncompSize}`,
      )
    }

    if (name.endsWith(".npy")) {
      out[name.replace(/\.npy$/, "")] = parseNpy(body)
    }
    i = dataStart + compSize
  }

  return out
}
