/**
 * 浏览器侧 PLY 解析（binary_little_endian）。
 *
 * 与 `scripts/utils/ply.ts` 的分工：
 * - `scripts/utils/ply.ts` 是**对拍脚本**用的解析器，目标是把自产 PLY
 *   逐字段读回来做数值核对，因此保留完整 element/property 结构与 Float64 列。
 * - 本文件是**运行时**解析器，目标是把 63 MB 顶点流尽快变成 GPU 能吃的
 *   结构数组，因此只认运行时会用到的字段，且优先少分配。
 * 两者故意不合并：一个偏「完整可诊断」，一个偏「快」，合并会两边都变差。
 *
 * 支持范围：`binary_little_endian 1.0` + 标量 property。ascii / 大端 /
 * list property（面片）一律抛错——我们的产物永远不含这些。
 */

import type { PlyAuxData, SceneMeta, SplatSoA } from "./types.ts"

/** PLY 标量类型 -> 字节数。 */
const TYPE_SIZES: Record<string, number> = {
  char: 1,
  int8: 1,
  uchar: 1,
  uint8: 1,
  short: 2,
  int16: 2,
  ushort: 2,
  uint16: 2,
  int: 4,
  int32: 4,
  uint: 4,
  uint32: 4,
  float: 4,
  float32: 4,
  double: 8,
  float64: 8,
}

interface PlyProperty {
  name: string
  type: string
  size: number
  /** 行内字节偏移。 */
  offset: number
}

interface PlyElement {
  name: string
  count: number
  stride: number
  properties: PlyProperty[]
  /** 数据区在文件中的起始字节偏移。 */
  offset: number
}

interface PlyHeader {
  /** 头长度（含 `end_header\n`）。 */
  bytes: number
  elements: PlyElement[]
}

/** 解析 ASCII 头。 */
export function parsePlyHeader(bytes: Uint8Array): PlyHeader {
  // 头一定是 ASCII，按字节扫到 `end_header` 那一行为止
  const limit = Math.min(bytes.length, 65536)
  let end = -1
  for (let i = 0; i + 10 <= limit; i++) {
    if (
      bytes[i] === 0x65 && // e
      bytes[i + 1] === 0x6e && // n
      bytes[i + 2] === 0x64 && // d
      bytes[i + 3] === 0x5f && // _
      bytes[i + 4] === 0x68 && // h
      bytes[i + 5] === 0x65 && // e
      bytes[i + 6] === 0x61 && // a
      bytes[i + 7] === 0x64 && // d
      bytes[i + 8] === 0x65 && // e
      bytes[i + 9] === 0x72 // r
    ) {
      end = i
      break
    }
  }
  if (end < 0) throw new Error("PLY: 找不到 end_header")

  const headerEnd = bytes.indexOf(0x0a, end)
  if (headerEnd < 0) throw new Error("PLY: end_header 后缺少换行")
  const headerBytes = headerEnd + 1

  const text = new TextDecoder("ascii").decode(bytes.subarray(0, headerEnd))
  const lines = text.split(/\r?\n/).map((l) => l.trim())
  const format = lines.find((l) => l.startsWith("format "))
  if (format !== "format binary_little_endian 1.0") {
    throw new Error(`PLY: 只支持 binary_little_endian 1.0，收到 "${format}"`)
  }

  const elements: PlyElement[] = []
  let current: PlyElement | undefined
  for (const line of lines) {
    const el = /^element\s+(\S+)\s+(\d+)$/.exec(line)
    if (el) {
      current = {
        name: el[1],
        count: Number(el[2]),
        stride: 0,
        properties: [],
        offset: 0,
      }
      elements.push(current)
      continue
    }
    const prop = /^property\s+(\S+)\s+(\S+)$/.exec(line)
    if (prop) {
      if (!current) throw new Error("PLY: property 出现在 element 之前")
      const size = TYPE_SIZES[prop[1]]
      if (size === undefined) {
        throw new Error(
          `PLY: 不支持的 property 类型 "${prop[1]}"（list 不支持）`,
        )
      }
      current.properties.push({
        name: prop[2],
        type: prop[1],
        size,
        offset: current.stride,
      })
      current.stride += size
      continue
    }
    if (/^property\s+list\b/.test(line)) {
      throw new Error("PLY: 不支持 list property")
    }
  }

  // 依次累加各 element 的数据区偏移
  let offset = headerBytes
  for (const el of elements) {
    el.offset = offset
    offset += el.stride * el.count
  }
  return { bytes: headerBytes, elements }
}

/** 逐行读取某个 element 的单个标量列（用于小 element）。 */
function readScalarColumn(
  view: DataView,
  el: PlyElement,
  prop: PlyProperty,
): number[] {
  const out = new Array<number>(el.count)
  for (let i = 0; i < el.count; i++) {
    out[i] = readScalar(
      view,
      el.offset + i * el.stride + prop.offset,
      prop.type,
    )
  }
  return out
}

function readScalar(view: DataView, offset: number, type: string): number {
  switch (type) {
    case "float":
    case "float32":
      return view.getFloat32(offset, true)
    case "double":
    case "float64":
      return view.getFloat64(offset, true)
    case "char":
    case "int8":
      return view.getInt8(offset)
    case "uchar":
    case "uint8":
      return view.getUint8(offset)
    case "short":
    case "int16":
      return view.getInt16(offset, true)
    case "ushort":
    case "uint16":
      return view.getUint16(offset, true)
    case "int":
    case "int32":
      return view.getInt32(offset, true)
    case "uint":
    case "uint32":
      return view.getUint32(offset, true)
    default:
      throw new Error(`PLY: 未知类型 "${type}"`)
  }
}

/** 顶点 element 需要的 14 个属性（与 3DGS / ml-sharp 产物一致）。 */
const VERTEX_FIELDS = [
  "x",
  "y",
  "z",
  "f_dc_0",
  "f_dc_1",
  "f_dc_2",
  "opacity",
  "scale_0",
  "scale_1",
  "scale_2",
  "rot_0",
  "rot_1",
  "rot_2",
  "rot_3",
] as const

/** 各字段缺省值（属性缺失时的兜底，保证渲染不炸）。 */
const FIELD_DEFAULTS: Record<string, number> = {
  x: 0,
  y: 0,
  z: 0,
  f_dc_0: 0,
  f_dc_1: 0,
  f_dc_2: 0,
  opacity: 0,
  scale_0: -12,
  scale_1: -12,
  scale_2: -12,
  rot_0: 1,
  rot_1: 0,
  rot_2: 0,
  rot_3: 0,
}

/**
 * 把 `vertex` element 读成结构数组（**尚未做颜色空间转换**：
 * `colorLinear` 这里填的是 PLY 里的原始 `f_dc_*`，由调用方接着转换）。
 */
export function readVertexSoA(
  bytes: Uint8Array,
  header: PlyHeader,
): SplatSoA & { shDc: Float32Array } {
  const el = header.elements.find((e) => e.name === "vertex")
  if (!el) throw new Error("PLY: 缺少 vertex element")

  const n = el.count
  const center = new Float32Array(n * 3)
  const scaleLog = new Float32Array(n * 3)
  const quat = new Float32Array(n * 4)
  const shDc = new Float32Array(n * 3)
  const opacityLogit = new Float32Array(n)

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.length)
  // 预解析每个字段的「行内字节偏移」；-1 表示 PLY 里没有该字段（用兜底值）。
  // 3DGS 族的这几个字段一律是 f4，遇到别的类型直接报错
  // （免得静默按 f4 读出一个垃圾数）。
  const offsets = new Int32Array(VERTEX_FIELDS.length).fill(-1)
  for (let f = 0; f < VERTEX_FIELDS.length; f++) {
    const prop = el.properties.find((p) => p.name === VERTEX_FIELDS[f])
    if (!prop) continue
    if (prop.type !== "float" && prop.type !== "float32") {
      throw new Error(
        `PLY: 字段 ${prop.name} 的类型是 ${prop.type}，只支持 float`,
      )
    }
    offsets[f] = prop.offset
  }

  let row = el.offset
  for (let i = 0; i < n; i++) {
    for (let f = 0; f < 14; f++) {
      const off = offsets[f]
      const v =
        off < 0
          ? FIELD_DEFAULTS[VERTEX_FIELDS[f]]
          : view.getFloat32(row + off, true)
      switch (f) {
        case 0:
          center[i * 3] = v
          break
        case 1:
          center[i * 3 + 1] = v
          break
        case 2:
          center[i * 3 + 2] = v
          break
        case 3:
          shDc[i * 3] = v
          break
        case 4:
          shDc[i * 3 + 1] = v
          break
        case 5:
          shDc[i * 3 + 2] = v
          break
        case 6:
          opacityLogit[i] = v
          break
        case 7:
          scaleLog[i * 3] = v
          break
        case 8:
          scaleLog[i * 3 + 1] = v
          break
        case 9:
          scaleLog[i * 3 + 2] = v
          break
        case 10:
          quat[i * 4] = v
          break
        case 11:
          quat[i * 4 + 1] = v
          break
        case 12:
          quat[i * 4 + 2] = v
          break
        case 13:
          quat[i * 4 + 3] = v
          break
      }
    }
    row += el.stride
  }

  return {
    count: n,
    center,
    scaleLog,
    quat,
    shDc,
    // 占位：调用方用 shDc 转出真正的线性 RGB
    colorLinear: new Float32Array(n * 3),
    opacityLogit,
  }
}

/** 读取所有小 element（intrinsic / image_size / ...）与包围盒。 */
export function readAuxAndMeta(
  bytes: Uint8Array,
  header: PlyHeader,
  soa: SplatSoA,
): SceneMeta {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.length)
  const aux: PlyAuxData = {}

  for (const el of header.elements) {
    const prop = el.properties[0]
    if (!prop) continue
    const values = readScalarColumn(view, el, prop)
    switch (el.name) {
      case "intrinsic":
        aux.intrinsic = values
        break
      case "extrinsic":
        aux.extrinsic = values
        break
      case "image_size":
        aux.imageSize = [values[0], values[1]]
        break
      case "disparity":
        aux.disparity = values
        break
      case "color_space":
        aux.colorSpace = values[0]
        break
      case "version":
        aux.version = values
        break
      default:
        break
    }
  }

  const min: [number, number, number] = [Infinity, Infinity, Infinity]
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity]
  for (let i = 0; i < soa.count; i++) {
    for (let c = 0; c < 3; c++) {
      const v = soa.center[i * 3 + c]
      if (v < min[c]) min[c] = v
      if (v > max[c]) max[c] = v
    }
  }

  return {
    count: soa.count,
    bytes: bytes.length,
    elements: header.elements.map((e) => ({ name: e.name, count: e.count })),
    bounds: { min, max },
    aux,
  }
}
