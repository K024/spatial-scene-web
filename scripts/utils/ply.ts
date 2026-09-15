/**
 * 二进制 PLY 解析（只覆盖本仓库产物用到的子集）。
 *
 * 用途：`scripts/sharp-compare-fixtures.ts` 要把 JS 侧 `serializePly()` 的输出
 * 与 PyTorch 侧 `save_ply()` 的 `reference.ply` **逐 element / 逐 property**
 * 对比。手写解析器而不是引依赖，理由有二：
 *   1. 只需要读本仓库自己产出的格式，依赖收益为负；
 *   2. 断言失败时要能指出**具体是哪个 element 的哪个 property**错了，
 *      通用库（如 plyfile 的 JS 移植）反而不方便。
 *
 * 支持：`format binary_little_endian 1.0` + 标量 property
 *       （char/uchar/short/ushort/int/uint/float/double）。
 * 不支持：ascii、大端、list property。
 * 遇到不支持的情形**直接抛错**：静默读歪比报错危险得多。
 */

/** PLY 标量类型名 -> 字节数。同时接受短名（`int32`）与 PLY 长名（`int`）。 */
const TYPE_SIZE: Record<string, number> = {
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

/** 单个 element（如 `vertex`）解析后的列数据。 */
export interface PlyElement {
  name: string
  count: number
  /** property 名，按 header 中声明的顺序。 */
  properties: string[]
  /** property 名 -> 列（长度恒为 `count`）。 */
  columns: Map<string, Float64Array>
}

export interface ParsedPly {
  /** header 文本（不含 `end_header` 之前之外的内容；用于逐字节对比）。 */
  header: string
  /** header 字节数（含 `end_header\n`），即 body 起始偏移。 */
  headerBytes: number
  elements: PlyElement[]
  element(name: string): PlyElement | undefined
  /** 便捷取列；element 或 property 不存在时返回 undefined。 */
  column(element: string, property: string): Float64Array | undefined
}

/** 从 DataView 读一个标量。 */
function readScalar(
  view: DataView,
  offset: number,
  type: string,
  littleEndian: boolean,
): number {
  switch (type) {
    case "char":
    case "int8":
      return view.getInt8(offset)
    case "uchar":
    case "uint8":
      return view.getUint8(offset)
    case "short":
    case "int16":
      return view.getInt16(offset, littleEndian)
    case "ushort":
    case "uint16":
      return view.getUint16(offset, littleEndian)
    case "int":
    case "int32":
      return view.getInt32(offset, littleEndian)
    case "uint":
    case "uint32":
      return view.getUint32(offset, littleEndian)
    case "float":
    case "float32":
      return view.getFloat32(offset, littleEndian)
    case "double":
    case "float64":
      return view.getFloat64(offset, littleEndian)
    default:
      throw new Error(`parsePly: 不支持的 property 类型 "${type}"`)
  }
}

/**
 * 解析一个二进制 PLY。
 *
 * @param buf 完整文件字节（`readFileSync` 的结果即可）。
 */
export function parsePly(buf: Uint8Array): ParsedPly {
  // header 一定在文件开头；本仓库的 header < 1KB，取 64KB 足够且不必解码整个 body。
  const probeLen = Math.min(buf.length, 1 << 16)
  const probe = Buffer.from(buf.buffer, buf.byteOffset, probeLen).toString(
    "latin1",
  )
  const endMatch = /end_header\r?\n/.exec(probe)
  if (!endMatch) {
    throw new Error(
      "parsePly: 找不到 end_header（不是文本 header 开头的 PLY？）",
    )
  }
  const headerBytes = endMatch.index + endMatch[0].length
  const header = probe.slice(0, headerBytes)

  const lines = header.split(/\r?\n/).filter((l) => l.length > 0)
  if (lines[0] !== "ply") throw new Error("parsePly: 首行不是 `ply`")
  const format = lines.find((l) => l.startsWith("format "))
  if (format !== "format binary_little_endian 1.0") {
    throw new Error(
      `parsePly: 只支持 binary_little_endian 1.0，收到 "${format}"`,
    )
  }

  // ── header 解析 ──
  interface Decl {
    name: string
    properties: string[]
    types: string[]
    count: number
  }
  const decls: Decl[] = []
  for (const line of lines) {
    const el = /^element (\S+) (\d+)$/.exec(line)
    if (el) {
      decls.push({
        name: el[1],
        properties: [],
        types: [],
        count: Number(el[2]),
      })
      continue
    }
    const prop = /^property (\S+) (\S+)$/.exec(line)
    if (!prop) continue // comment / obj_info / format / ply
    const type = prop[1]
    if (type === "list") {
      throw new Error("parsePly: 不支持 list property")
    }
    if (!(type in TYPE_SIZE)) {
      throw new Error(`parsePly: 未知 property 类型 "${type}"`)
    }
    const cur = decls[decls.length - 1]
    if (!cur) throw new Error("parsePly: property 出现在 element 之前")
    cur.properties.push(prop[2])
    cur.types.push(type)
  }

  // ── body 解析 ──
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength)
  const elements: PlyElement[] = []
  let offset = headerBytes

  for (const decl of decls) {
    const sizes = decl.types.map((t) => TYPE_SIZE[t])
    const stride = sizes.reduce((a, b) => a + b, 0)
    const columns = new Map<string, Float64Array>()
    for (const p of decl.properties)
      columns.set(p, new Float64Array(decl.count))

    const need = offset + stride * decl.count
    if (need > buf.length) {
      throw new Error(
        `parsePly: body 长度不足（element ${decl.name} 需要到 ${need}，文件长 ${buf.length}）`,
      )
    }

    for (let row = 0; row < decl.count; row++) {
      let off = offset + row * stride
      for (let p = 0; p < decl.properties.length; p++) {
        columns.get(decl.properties[p])![row] = readScalar(
          view,
          off,
          decl.types[p],
          true,
        )
        off += sizes[p]
      }
    }

    elements.push({
      name: decl.name,
      count: decl.count,
      properties: decl.properties,
      columns,
    })
    offset += stride * decl.count
  }

  const byName = new Map(elements.map((e) => [e.name, e] as const))
  return {
    header,
    headerBytes,
    elements,
    element: (name) => byName.get(name),
    column: (el, prop) => byName.get(el)?.columns.get(prop),
  }
}
