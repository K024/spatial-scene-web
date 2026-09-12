/**
 * 极简 PLY 读取（只支持本仓库写出的 `binary_little_endian`）。
 *
 * 用途：给离线校验脚本（`check-camera.ts`）读回导出的 PLY，
 * 不必依赖外部工具，也不会因为「文件写坏了」而静默出错。
 * 与 `check-ply.ts` 里的解析相比，这里只取渲染需要的字段
 * （xyz / f_dc / opacity），并且按 header 里的属性名定位偏移，
 * 因此 `--full` 模式（多出 7 个 element）也能直接读。
 */

import { readFileSync } from "node:fs"

/** PLY 标量类型的字节数。 */
const TYPE_SIZE: Record<string, number> = {
  char: 1,
  uchar: 1,
  int8: 1,
  uint8: 1,
  short: 2,
  ushort: 2,
  int16: 2,
  uint16: 2,
  int: 4,
  uint: 4,
  int32: 4,
  uint32: 4,
  float: 4,
  float32: 4,
  double: 8,
  float64: 8,
}

export interface PlyProperty {
  /** 属性名，如 `f_dc_0`。 */
  name: string
  /** 类型名，如 `float`。 */
  type: string
  /** 在本 element 记录内的字节偏移。 */
  offset: number
}

export interface PlyElement {
  name: string
  count: number
  props: PlyProperty[]
  /** 本 element 单条记录的字节数。 */
  stride: number
}

export interface PlyHeader {
  /** body 起始字节（header 末字节之后）。 */
  bodyOffset: number
  elements: PlyElement[]
}

/** 解析 header。只支持 `format binary_little_endian 1.0`。 */
export function parsePlyHeader(buf: Buffer): PlyHeader {
  const text = buf.subarray(0, Math.min(buf.length, 1 << 16)).toString("latin1")
  const endIdx = text.indexOf("end_header\n")
  if (endIdx < 0) throw new Error("未找到 end_header")
  const bodyOffset = endIdx + "end_header\n".length

  const lines = text.slice(0, endIdx).split("\n")
  if (lines[0].trim() !== "ply") throw new Error("不是 PLY 文件")
  if (!lines[1].includes("binary_little_endian")) {
    throw new Error(`不支持的格式: ${lines[1]}`)
  }

  const elements: PlyElement[] = []
  for (const line of lines.slice(2)) {
    const parts = line.trim().split(/\s+/)
    if (parts[0] === "element") {
      elements.push({
        name: parts[1],
        count: Number.parseInt(parts[2], 10),
        props: [],
        stride: 0,
      })
    } else if (parts[0] === "property") {
      const el = elements[elements.length - 1]
      if (!el) throw new Error("property 出现在 element 之前")
      const type = parts[1]
      const size = TYPE_SIZE[type]
      if (size === undefined) throw new Error(`未知属性类型: ${type}`)
      el.props.push({ name: parts[2], type, offset: el.stride })
      el.stride += size
    }
  }
  return { bodyOffset, elements }
}

/** 渲染需要的高斯字段。 */
export interface PlyGaussians {
  count: number
  /** 均值，PLY（度量）坐标系。 */
  x: Float32Array
  y: Float32Array
  z: Float32Array
  /** degree-0 SH 系数（3N），已含 `save_ply` 的 linearRGB->sRGB 处理。 */
  fdc: Float32Array
  /** 不透明度 alpha = sigmoid(opacity) ∈ [0,1]。 */
  alpha: Float32Array
}

/**
 * 读回 vertex element 中的 xyz / f_dc_* / opacity。
 *
 * @throws 缺少必需属性时抛错（比返回 NaN 更容易定位问题）。
 */
export function readPlyGaussians(path: string): PlyGaussians {
  const buf = readFileSync(path)
  const { bodyOffset, elements } = parsePlyHeader(buf)
  const vertex = elements.find((e) => e.name === "vertex")
  if (!vertex) throw new Error("缺少 vertex element")

  const need = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"]
  const offsets = new Map<string, number>()
  for (const p of vertex.props) offsets.set(p.name, p.offset)
  for (const name of need) {
    if (!offsets.has(name)) throw new Error(`vertex 缺少属性: ${name}`)
  }

  const n = vertex.count
  const dv = new DataView(buf.buffer, buf.byteOffset + bodyOffset)
  const f32 = (i: number, name: string): number =>
    dv.getFloat32(i * vertex.stride + (offsets.get(name) as number), true)

  const x = new Float32Array(n)
  const y = new Float32Array(n)
  const z = new Float32Array(n)
  const fdc = new Float32Array(n * 3)
  const alpha = new Float32Array(n)

  for (let i = 0; i < n; i++) {
    x[i] = f32(i, "x")
    y[i] = f32(i, "y")
    z[i] = f32(i, "z")
    fdc[i * 3 + 0] = f32(i, "f_dc_0")
    fdc[i * 3 + 1] = f32(i, "f_dc_1")
    fdc[i * 3 + 2] = f32(i, "f_dc_2")
    alpha[i] = 1 / (1 + Math.exp(-f32(i, "opacity")))
  }

  return { count: n, x, y, z, fdc, alpha }
}
