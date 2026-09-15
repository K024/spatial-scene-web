/**
 * PLY 导出（3DGS 格式）。
 *
 * 严格对照 `ml-sharp/src/sharp/utils/gaussians.py: save_ply`。
 *
 * ── 保真要点（三条 bias）──
 * 1. `scale_{i} = log(singular_values)`        —— 存 log 尺度，不是尺度本身
 * 2. `opacity   = logit(opacities)`            —— 存 inverse sigmoid
 * 3. `f_dc_{i}  = rgbToSphericalHarmonics(linearRGB2sRGB(colors))`
 *    —— 先做**线性->sRGB**（`save_ply` 明确注释：为了让公开渲染器不开 gamma 也能看），
 *       再转 degree-0 SH。
 *
 * ── 两种输出模式（决策 C2）──
 * - **默认（compact）**：只写 `vertex` element。SuperSplat / 通用 3DGS 查看器可直接打开。
 * - **full**：在 vertex 之外追加 ml-sharp 的 7 个补充 element
 *   （extrinsic / intrinsic / image_size / frame / disparity / color_space / version），
 *   与官方 `save_ply` 产物逐字段对齐，用于与本仓库 Python 侧互操作。
 *
 * 二进制 little-endian，与 plyfile 默认一致。
 */

import { linearRGB2sRGB } from "../sharp/colorspace.ts"
import { rgbToSphericalHarmonics } from "../sharp/linalg.ts"
import type { Gaussians3D, SceneMetaData } from "../sharp/types.ts"

/** `save_ply` 的版本号：`[1, 5, 0]`（u1 × 3）。 */
export const PLY_VERSION: [number, number, number] = [1, 5, 0]

export interface PlyExportOptions {
  /** 高斯（**度量空间**，即已做过 NDC->metric 反投影）。 */
  gaussians: Gaussians3D
  /** 原始图像域的焦距（px）。对照 `save_ply(f_px=...)`。 */
  fPx: number
  /** 原始图像尺寸 `[width, height]`。 */
  imageShape: [number, number]
  /**
   * 是否追加 ml-sharp 的全部补充 element。
   * 默认 false（仅 vertex，SuperSplat 兼容）。
   */
  full?: boolean
}

/** 只读游标写入器，自动扩容。 */
class ByteWriter {
  private buf: Uint8Array
  private view: DataView
  private pos = 0

  constructor(initial = 1 << 20) {
    this.buf = new Uint8Array(initial)
    this.view = new DataView(this.buf.buffer)
  }

  private ensure(extra: number): void {
    if (this.pos + extra <= this.buf.length) return
    let cap = this.buf.length * 2
    while (cap < this.pos + extra) cap *= 2
    const next = new Uint8Array(cap)
    next.set(this.buf.subarray(0, this.pos))
    this.buf = next
    this.view = new DataView(this.buf.buffer)
  }

  f32(v: number): void {
    this.ensure(4)
    this.view.setFloat32(this.pos, v, true)
    this.pos += 4
  }

  u32(v: number): void {
    this.ensure(4)
    this.view.setUint32(this.pos, v, true)
    this.pos += 4
  }

  i32(v: number): void {
    this.ensure(4)
    this.view.setInt32(this.pos, v, true)
    this.pos += 4
  }

  u8(v: number): void {
    this.ensure(1)
    this.buf[this.pos++] = v & 0xff
  }

  bytes(): Uint8Array {
    return this.buf.subarray(0, this.pos)
  }
}

/** element 描述（用于拼 header）。 */
interface PlyElementDesc {
  name: string
  /** property 列表：`[name, type]`，按写入顺序。 */
  properties: [string, PlyPropType][]
  count: number
}

/**
 * PLY 类型名。
 *
 * 注意用 plyfile 的**文字形式**（`float`/`uint`/`int`/`uchar`）而非
 * 短名（`f4`/`u4`/`i4`/`u1`）：两者都是合法的 PLY，但为了与 ml-sharp 的
 * `save_ply` 产物逐字节一致（便于互操作与对拍），这里照搬 plyfile 的写法。
 * 映射见 plyfile 的 `_data_type_reverse_dict`：f4->float, u4->uint, i4->int, u1->uchar。
 */
type PlyPropType = "float" | "uint" | "int" | "uchar"

function buildHeader(elements: PlyElementDesc[]): string {
  const lines: string[] = ["ply", "format binary_little_endian 1.0"]
  for (const el of elements) {
    lines.push(`element ${el.name} ${el.count}`)
    for (const [pname, ptype] of el.properties)
      lines.push(`property ${ptype} ${pname}`)
  }
  lines.push("end_header")
  return `${lines.join("\n")}\n`
}

const VERTEX_PROPS: [string, PlyPropType][] = [
  ["x", "float"],
  ["y", "float"],
  ["z", "float"],
  ["f_dc_0", "float"],
  ["f_dc_1", "float"],
  ["f_dc_2", "float"],
  ["opacity", "float"],
  ["scale_0", "float"],
  ["scale_1", "float"],
  ["scale_2", "float"],
  ["rot_0", "float"],
  ["rot_1", "float"],
  ["rot_2", "float"],
  ["rot_3", "float"],
]

/**
 * 序列化为 PLY 字节流。
 *
 * @returns 完整 PLY（header + 二进制 body）。
 */
export function serializePly(opts: PlyExportOptions): Uint8Array {
  const { gaussians, fPx, imageShape } = opts
  const full = opts.full ?? false
  const [width, height] = imageShape

  const n = gaussians.opacities.length

  // ── 预计算派生字段 ──
  // 颜色：linearRGB -> sRGB -> degree-0 SH（顺序不可颠倒，见文件头注释 bias 3）
  const srgb = linearRGB2sRGB(gaussians.colors)
  const sh = rgbToSphericalHarmonics(srgb)

  // ── 补充 element 数据（full 模式）──
  // disparity element：1/z 的 10% 与 90% 分位数。对照 save_ply 的 torch.quantile。
  let disparityQuantiles: [number, number] = [0, 0]
  if (full) {
    const invZ = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      const z = gaussians.meanVectors[i * 3 + 2]
      invZ[i] = z !== 0 ? 1.0 / z : 0
    }
    const sorted = Float64Array.from(invZ).sort()
    disparityQuantiles = [quantile(sorted, 0.1), quantile(sorted, 0.9)]
  }

  // ── header ──
  const elements: PlyElementDesc[] = [
    { name: "vertex", properties: VERTEX_PROPS, count: n },
  ]
  if (full) {
    elements.push(
      { name: "extrinsic", properties: [["extrinsic", "float"]], count: 16 },
      { name: "intrinsic", properties: [["intrinsic", "float"]], count: 9 },
      { name: "image_size", properties: [["image_size", "uint"]], count: 2 },
      { name: "frame", properties: [["frame", "int"]], count: 2 },
      { name: "disparity", properties: [["disparity", "float"]], count: 2 },
      { name: "color_space", properties: [["color_space", "uchar"]], count: 1 },
      { name: "version", properties: [["version", "uchar"]], count: 3 },
    )
  }

  const header = buildHeader(elements)
  const headerBytes = new TextEncoder().encode(header)

  // 估算容量：vertex 14 floats + 补充 element 的字节数
  const bodyEstimate = n * 14 * 4 + (full ? (16 + 9 + 2 + 2 + 1 + 3) * 4 : 0)
  const w = new ByteWriter(headerBytes.length + bodyEstimate)
  // 直接写入 header 字节
  for (let i = 0; i < headerBytes.length; i++) w.u8(headerBytes[i])

  // ── vertex body ──
  for (let i = 0; i < n; i++) {
    w.f32(gaussians.meanVectors[i * 3 + 0])
    w.f32(gaussians.meanVectors[i * 3 + 1])
    w.f32(gaussians.meanVectors[i * 3 + 2])

    w.f32(sh[i * 3 + 0])
    w.f32(sh[i * 3 + 1])
    w.f32(sh[i * 3 + 2])

    // opacity logit：log(p / (1-p))
    const p = gaussians.opacities[i]
    w.f32(inverseSigmoid(p))

    // scale log
    w.f32(Math.log(gaussians.singularValues[i * 3 + 0]))
    w.f32(Math.log(gaussians.singularValues[i * 3 + 1]))
    w.f32(Math.log(gaussians.singularValues[i * 3 + 2]))

    // w-first 四元数，原样写出
    w.f32(gaussians.quaternions[i * 4 + 0])
    w.f32(gaussians.quaternions[i * 4 + 1])
    w.f32(gaussians.quaternions[i * 4 + 2])
    w.f32(gaussians.quaternions[i * 4 + 3])
  }

  // ── 补充 element body（full 模式）──
  if (full) {
    // extrinsic：单位阵（对照 save_ply 的 dummy extrinsics）
    for (let r = 0; r < 4; r++)
      for (let c = 0; c < 4; c++) w.f32(r === c ? 1 : 0)
    // intrinsic：3x3 行主序 [f,0,w/2, 0,f,h/2, 0,0,1]
    w.f32(fPx)
    w.f32(0)
    w.f32(width * 0.5)
    w.f32(0)
    w.f32(fPx)
    w.f32(height * 0.5)
    w.f32(0)
    w.f32(0)
    w.f32(1)
    // image_size：[width, height] u4
    w.u32(width)
    w.u32(height)
    // frame：[1, numGaussians] i4
    w.i32(1)
    w.i32(n)
    // disparity：10% / 90% 分位数
    w.f32(disparityQuantiles[0])
    w.f32(disparityQuantiles[1])
    // color_space：sRGB = 0
    w.u8(0)
    // version
    w.u8(PLY_VERSION[0])
    w.u8(PLY_VERSION[1])
    w.u8(PLY_VERSION[2])
  }

  return w.bytes()
}

/**
 * 反 sigmoid：`log(p / (1-p))`。对照 `save_ply` 内的 `_inverse_sigmoid`。
 *
 * p 会在网络输出后被 sigmoid 压到 [0,1]，极端值需要夹紧以避免 ±Inf。
 */
function inverseSigmoid(p: number): number {
  const EPS = 1e-7
  const q = Math.min(Math.max(p, EPS), 1 - EPS)
  return Math.log(q / (1 - q))
}

/**
 * 线性插值分位数，复刻 `torch.quantile` 的默认（linear）插值。
 *
 * 对照 `save_ply`：`torch.quantile(disparity, q=[0.1, 0.9])`。
 *
 * @param sorted 已升序排序的数组。
 */
export function quantile(sorted: ArrayLike<number>, q: number): number {
  const n = sorted.length
  if (n === 0) return NaN
  if (n === 1) return sorted[0]
  const pos = q * (n - 1)
  const lo = Math.floor(pos)
  const hi = Math.min(lo + 1, n - 1)
  const frac = pos - lo
  return sorted[lo] + (sorted[hi] - sorted[lo]) * frac
}

/**
 * 便捷包装：由高斯 + 元数据生成 PLY 字节。
 *
 * 提供与 `save_ply(gaussians, f_px, image_shape, path)` 等价的入参形态，
 * 便于逐字段对照。
 */
export function gaussiansToPly(
  gaussians: Gaussians3D,
  meta: SceneMetaData,
  opts: { full?: boolean } = {},
): Uint8Array {
  return serializePly({
    gaussians,
    fPx: meta.focalLengthPx,
    imageShape: meta.resolutionPx,
    full: opts.full,
  })
}
