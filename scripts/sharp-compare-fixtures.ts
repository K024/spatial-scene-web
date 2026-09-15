/**
 * 数值对拍：`src/spatial-scene/**` 的 JS 复刻 vs PyTorch 侧 fixtures。
 *
 * Fixtures 由 `py-models/scripts/01_smoke_test.py` 的**同一次**前向产生，
 * 落在 `py-models/out/fixtures/`：
 *
 *   resize.npz     平滑合成图 3024x2268 -> 1536²，用于**单独**验证 bilinear
 *                  `align_corners=True`（隔离 JPEG 解码器差异）
 *   input.npz      torch 侧预处理结果 image[3,1536,1536] + 原图尺寸 + f_px
 *                  + disparity_factor
 *   ndc.npz        模型输出的 NDC 空间 5 个张量 + disparity（**PyTorch** 输出）
 *   metric.npz     `unproject_gaussians` 之后 + intrinsics_resized
 *                  + unprojection_matrix
 *   reference.ply  官方 `save_ply` 产物，用于 PLY 逐 element / 逐 property 对拍
 *
 * 本脚本只做数值/字节层面的对比，**不跑 ONNX**：ONNX 相对 PyTorch 的偏差是
 * 另一条独立链路，由 `py-models/scripts/04_evaluate.py` 负责。这里回答的问题是
 * 「JS 是否忠实复刻了 PyTorch 语义」。
 *
 * 用法:
 *   npx tsx scripts/sharp-compare-fixtures.ts
 *   npx tsx scripts/sharp-compare-fixtures.ts --only resize,ply --no-image
 *   npx tsx scripts/sharp-compare-fixtures.ts --list
 *
 * 退出码：0 = 全部通过；1 = 有失败项。
 */

import { existsSync, readFileSync } from "node:fs"
import { resolve } from "node:path"
import { parseArgs } from "node:util"

import { quantile, serializePly } from "../src/spatial-scene/export/ply.ts"
import { f16BitsToF32, f32ToF16 } from "../src/spatial-scene/infer/session.ts"
import {
  linearRGB2sRGB,
  sRGB2linearRGB,
} from "../src/spatial-scene/sharp/colorspace.ts"
import {
  convertFocallength,
  disparityFactor,
  intrinsicsResized,
  resolveFocalLength35mm,
} from "../src/spatial-scene/sharp/fov.ts"
import {
  composeCovarianceMatrix,
  rgbToSphericalHarmonics,
  rotationMatrixFromQuaternion,
  rotationMatrixToQuaternion,
  sphericalHarmonicsToRgb,
  symmetricEigen3,
} from "../src/spatial-scene/sharp/linalg.ts"
import {
  preprocessImage,
  resizeBilinearChw,
} from "../src/spatial-scene/sharp/preprocess.ts"
import {
  type Gaussians3D,
  INTERNAL_RESOLUTION,
  type SceneMetaData,
} from "../src/spatial-scene/sharp/types.ts"
import {
  applyTransform,
  getUnprojectionMatrix,
  identity4,
} from "../src/spatial-scene/sharp/unproject.ts"
import { DEFAULT_IMAGE, humanSize, REPO_ROOT } from "./utils/common.ts"
import { loadImage } from "./utils/image.ts"
import { loadNpz, type NpyArray } from "./utils/npz.ts"
import { type ParsedPly, parsePly } from "./utils/ply.ts"

// --------------------------------------------------------------------------- #
// 阶段与断言框架
// --------------------------------------------------------------------------- #

const STAGES = [
  ["resize", "bilinear align_corners=True（resize.npz）"],
  ["fov", "焦距 / disparity_factor / intrinsics（input.npz + metric.npz）"],
  ["unproject", "NDC -> 度量空间（ndc.npz -> metric.npz）"],
  ["unit", "纯函数自洽性（quantile / SH / 四元数 / 特征分解 / fp16）"],
  ["colorspace", "linearRGB->sRGB + degree-0 SH（metric.npz + reference.ply）"],
  ["ply", "PLY 序列化逐字段对拍（metric.npz -> reference.ply）"],
  ["image", "整图预处理（含 JPEG 解码器差异，部分为信息性）"],
] as const

type StageName = (typeof STAGES)[number][0]

interface Line {
  stage: StageName
  label: string
  status: "pass" | "fail" | "info"
  detail: string
}

const lines: Line[] = []
let currentStage: StageName = "resize"

function record(status: Line["status"], label: string, detail: string): void {
  lines.push({ stage: currentStage, label, status, detail })
  const tag = status === "pass" ? "PASS" : status === "fail" ? "FAIL" : "INFO"
  console.log(`  ${tag}  ${label}${detail ? `  — ${detail}` : ""}`)
}

/** 硬断言：不满足即计入失败。 */
function check(label: string, ok: boolean, detail = ""): void {
  record(ok ? "pass" : "fail", label, detail)
}

/** 信息性输出：仅记录，不影响退出码。 */
function info(label: string, detail = ""): void {
  record("info", label, detail)
}

// --------------------------------------------------------------------------- #
// 数值工具
// --------------------------------------------------------------------------- #

interface Diff {
  max: number
  at: number
  mean: number
  nan: number
  count: number
}

/** 逐元素绝对差统计。`skip` 可跳过已知的、刻意不同的下标。 */
function diffAbs(
  a: ArrayLike<number>,
  b: ArrayLike<number>,
  skip?: (i: number) => boolean,
): Diff {
  let max = 0
  let at = -1
  let sum = 0
  let count = 0
  let nan = 0
  for (let i = 0; i < a.length; i++) {
    if (skip?.(i)) continue
    const x = a[i]
    const y = b[i]
    if (Number.isNaN(x) || Number.isNaN(y)) {
      nan++
      continue
    }
    const d = Math.abs(x - y)
    if (d > max) {
      max = d
      at = i
    }
    sum += d
    count++
  }
  return { max, at, mean: count > 0 ? sum / count : 0, nan, count }
}

function psnr(
  a: ArrayLike<number>,
  b: ArrayLike<number>,
  range: number,
): number {
  let sum = 0
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i]
    sum += d * d
  }
  const mse = sum / a.length
  return mse <= 0 ? 99 : 10 * Math.log10((range * range) / mse)
}

/** 皮尔逊相关系数（可抽样加速）。 */
function correlation(
  a: ArrayLike<number>,
  b: ArrayLike<number>,
  stride = 1,
): number {
  let n = 0
  let sa = 0
  let sb = 0
  for (let i = 0; i < a.length; i += stride) {
    sa += a[i]
    sb += b[i]
    n++
  }
  const ma = sa / n
  const mb = sb / n
  let cov = 0
  let va = 0
  let vb = 0
  for (let i = 0; i < a.length; i += stride) {
    const da = a[i] - ma
    const db = b[i] - mb
    cov += da * db
    va += da * da
    vb += db * db
  }
  return cov / Math.sqrt(va * vb)
}

/** 格式化浮点误差。 */
function e(v: number): string {
  return v.toExponential(3)
}

/** 3x3 行主序矩阵乘法（脚本内自用，避免为对拍引入依赖）。 */
function mat3Mul(a: ArrayLike<number>, b: ArrayLike<number>): Float64Array {
  const c = new Float64Array(9)
  for (let r = 0; r < 3; r++) {
    for (let col = 0; col < 3; col++) {
      let acc = 0
      for (let k = 0; k < 3; k++) acc += a[r * 3 + k] * b[k * 3 + col]
      c[r * 3 + col] = acc
    }
  }
  return c
}

/** 由 (四元数, 奇异值) 重建单个高斯的协方差 Σ = R diag(s²) Rᵀ。 */
function covarianceOf(
  quaternions: ArrayLike<number>,
  singularValues: ArrayLike<number>,
  i: number,
  out: Float64Array,
): Float64Array {
  const r = rotationMatrixFromQuaternion(
    quaternions[i * 4],
    quaternions[i * 4 + 1],
    quaternions[i * 4 + 2],
    quaternions[i * 4 + 3],
  )
  return composeCovarianceMatrix(
    r,
    singularValues[i * 3],
    singularValues[i * 3 + 1],
    singularValues[i * 3 + 2],
    out,
  )
}

// --------------------------------------------------------------------------- #
// fixture 访问
// --------------------------------------------------------------------------- #

const npzCache = new Map<string, Record<string, NpyArray>>()
let fixturesDir = resolve(REPO_ROOT, "py-models", "out", "fixtures")

function fixturePath(name: string): string {
  const p = resolve(fixturesDir, name)
  if (!existsSync(p)) {
    throw new Error(
      `缺少 fixture: ${p}\n` +
        "请先在 py-models 下生成： python scripts/01_smoke_test.py\n" +
        "（需要 ml-sharp 权重与 venv，见 py-models/prepare.sh）",
    )
  }
  return p
}

function npz(name: string): Record<string, NpyArray> {
  const cached = npzCache.get(name)
  if (cached) return cached
  const z = loadNpz(fixturePath(name))
  npzCache.set(name, z)
  return z
}

/** 高斯（展平 [N, C]）从 npy 条目组装，供 `ply.ts` / `unproject.ts` 消费。 */
function gaussiansFrom(z: Record<string, NpyArray>): Gaussians3D {
  return {
    meanVectors: z.mean_vectors.data,
    singularValues: z.singular_values.data,
    quaternions: z.quaternions.data,
    colors: z.colors.data,
    opacities: z.opacities.data,
  }
}

// --------------------------------------------------------------------------- #
// 阶段 1：resize（bilinear + align_corners=True）
// --------------------------------------------------------------------------- #

function stageResize(): void {
  const z = npz("resize.npz")
  const [gw, gh] = z.src_size.data
  const [dw, dh] = z.dst_size.data
  const alignCorners = z.align_corners.data[0]
  check("align_corners 声明为 1", alignCorners === 1)

  const out = resizeBilinearChw(z.src.data, gw, gh, 3, dw, dh)
  const d = diffAbs(out, z.dst.data)
  info("src/dst", `${gw}x${gh} -> ${dw}x${dh}，合成图（正弦叠加 + 确定性纹理）`)
  check(
    "resizeBilinearChw 与 F.interpolate 一致",
    d.max <= 1e-5,
    `max=${e(d.max)} mean=${e(d.mean)} PSNR=${psnr(out, z.dst.data, 2).toFixed(2)}dB`,
  )

  // 该 fixture 是本脚本里最大的中间物，用完即弃
  npzCache.delete("resize.npz")
}

// --------------------------------------------------------------------------- #
// 阶段 2：焦距 / disparity_factor / intrinsics
// --------------------------------------------------------------------------- #

function stageFov(): void {
  const inp = npz("input.npz")
  const metric = npz("metric.npz")
  const w = inp.orig_width.data[0]
  const h = inp.orig_height.data[0]
  const fPx = inp.f_px.data[0]
  const df = inp.disparity_factor.data[0]
  info("原图", `${w}x${h}，f_px=${fPx.toFixed(4)}`)

  // convertFocallength：用 f_px 反推 35mm 等效焦距，再正推回去。
  // 这同时校验了 FILM_DIAGONAL 常量（若写成 36 或 43 都会立刻暴露）。
  const filmDiagonal = Math.sqrt(36 ** 2 + 24 ** 2)
  const f35 = (fPx * filmDiagonal) / Math.hypot(w, h)
  const fPxBack = convertFocallength(w, h, f35)
  check(
    "convertFocallength 可逆（35mm 等效焦距）",
    Math.abs(fPxBack - fPx) <= 1e-6 * fPx,
    `f_35mm=${f35.toFixed(4)}mm，回代 ${fPxBack.toFixed(4)}px`,
  )

  // fallback 链（io.py: load_rgb）
  const noExif = resolveFocalLength35mm(undefined)
  const exif35 = resolveFocalLength35mm({ FocalLengthIn35mmFilm: 35 })
  const exifLow = resolveFocalLength35mm({
    FocalLengthIn35mmFilm: 0.5,
    FocalLength: 4.5,
  })
  const exifPhys = resolveFocalLength35mm({ FocalLength: 5 })
  const exifAlt = resolveFocalLength35mm({ FocalLenIn35mmFilm: 24 })
  const chainOk =
    noExif === 30 &&
    exif35 === 35 &&
    // <1 时回退到 FocalLength（4.5），再因 <10 乘 8.4
    Math.abs(exifLow - 4.5 * 8.4) < 1e-12 &&
    Math.abs(exifPhys - 42) < 1e-12 &&
    exifAlt === 24
  check(
    "resolveFocalLength35mm fallback 链",
    chainOk,
    `无 EXIF=${noExif}, 35mm tag=${exif35}, 0.5->${exifLow}, 5mm->${exifPhys}, FocalLenIn35mmFilm=${exifAlt}`,
  )

  // 本用例的 f_px 应恰好等于默认 30mm（example.jpg 无可用 EXIF）
  check(
    "本用例走默认 30mm 分支",
    Math.abs(convertFocallength(w, h, 30) - fPx) <= 1e-3,
    `convertFocallength(30mm)=${convertFocallength(w, h, 30).toFixed(4)} vs f_px=${fPx.toFixed(4)}`,
  )

  // disparity_factor = f_px / width（**分母是原图宽，不是 1536**）
  //
  // fixture 里是 **float32**（`torch.tensor([...], dtype=torch.float32)`），
  // 我们在 float64 里算完再交给 ort 转 fp16 ⇒ 断言前先 fround，比「绝对值差
  // 1e-9」更准确地表达「同一个 float32」。
  const dfJs = disparityFactor(fPx, w)
  check(
    "disparityFactor = f_px / width",
    Math.fround(dfJs) === df,
    `js=${dfJs} fround=${Math.fround(dfJs)} fixture=${df}`,
  )

  // intrinsics_resized：第 0 行乘 1536/W、第 1 行乘 1536/H（**含主点，整行缩放**）
  const k = intrinsicsResized(fPx, w, h, INTERNAL_RESOLUTION)
  const kRef = metric.intrinsics_resized.data
  const kDiff = diffAbs(k, kRef)
  check(
    "intrinsicsResized（整行缩放，含主点）",
    kDiff.max <= 1e-2,
    `max=${e(kDiff.max)}（fx=${k[0].toFixed(3)}, fy=${k[5].toFixed(3)}, cx=${k[2].toFixed(3)}, cy=${k[6].toFixed(3)}）`,
  )

  // unprojection 矩阵：inv(ndc @ K @ I)，内部域 1536²
  const m = getUnprojectionMatrix(identity4(), k, [
    INTERNAL_RESOLUTION,
    INTERNAL_RESOLUTION,
  ])
  const mDiff = diffAbs(m, metric.unprojection_matrix.data)
  check(
    "getUnprojectionMatrix = inv(ndc @ K)",
    mDiff.max <= 1e-6,
    `max=${e(mDiff.max)} diag=[${m[0].toFixed(6)}, ${m[5].toFixed(6)}, ${m[10].toFixed(6)}]`,
  )
}

// --------------------------------------------------------------------------- #
// 阶段 3：NDC -> 度量空间
// --------------------------------------------------------------------------- #

function stageUnproject(): void {
  const ndc = npz("ndc.npz")
  const metric = npz("metric.npz")
  const n = ndc.opacities.data.length

  const inp = npz("input.npz")
  const w = inp.orig_width.data[0]
  const h = inp.orig_height.data[0]
  const fPx = inp.f_px.data[0]

  const k = intrinsicsResized(fPx, w, h, INTERNAL_RESOLUTION)
  const m = getUnprojectionMatrix(identity4(), k, [
    INTERNAL_RESOLUTION,
    INTERNAL_RESOLUTION,
  ])

  const t0 = Date.now()
  const out = applyTransform(gaussiansFrom(ndc), m)
  const ms = Date.now() - t0
  info("applyTransform", `${n} 个高斯，${ms}ms`)

  const meanD = diffAbs(out.meanVectors, metric.mean_vectors.data)
  const sceneScale = maxAbs(metric.mean_vectors.data)
  check(
    "mean_vectors（度量空间坐标）",
    meanD.max <= 1e-5,
    `max=${e(meanD.max)} mean=${e(meanD.mean)}（场景量程 ${sceneScale.toFixed(2)}）`,
  )

  const sD = diffAbs(out.singularValues, metric.singular_values.data)
  check(
    "singular_values（sqrt(特征值)）",
    sD.max <= 1e-6,
    `max=${e(sD.max)} mean=${e(sD.mean)}`,
  )

  // 颜色/不透明度按引用透传：必须**逐位**相同
  const cD = diffAbs(out.colors, metric.colors.data)
  const oD = diffAbs(out.opacities, metric.opacities.data)
  check("colors 原样透传", cD.max === 0, `max=${e(cD.max)}`)
  check("opacities 原样透传", oD.max === 0, `max=${e(oD.max)}`)

  // ── 姿态：不能直接比四元数分量 ──
  //
  // (q, s) 与协方差 Σ = R diag(s²) Rᵀ 的关系是多对一的：
  // 把 R 的两列同时取反（D = diag(±1)，det D = +1）得到另一个**合法**分解，
  // 且 Σ 不变。torch.linalg.svd（LAPACK）与我们的 Jacobi 迭代对特征向量
  // 的符号选择本就不同 ⇒ 直接比 rot_i 会看到大量"翻转"，但渲染结果完全一致。
  // 因此这里比**渲染真正依赖的量** Σ，这是唯一有意义的判据。
  const covD = new Float64Array(n)
  const covA = new Float64Array(9)
  const covB = new Float64Array(9)
  let covMax = 0
  let flipCount = 0
  for (let i = 0; i < n; i++) {
    covarianceOf(out.quaternions, out.singularValues, i, covA)
    covarianceOf(metric.quaternions.data, metric.singular_values.data, i, covB)
    let local = 0
    for (let k = 0; k < 9; k++) {
      const d = Math.abs(covA[k] - covB[k])
      if (d > local) local = d
    }
    covD[i] = local
    if (local > covMax) covMax = local
    // 顺带统计四元数符号翻转的规模（信息性，不是错误）
    const dot = Math.abs(
      out.quaternions[i * 4] * metric.quaternions.data[i * 4] +
        out.quaternions[i * 4 + 1] * metric.quaternions.data[i * 4 + 1] +
        out.quaternions[i * 4 + 2] * metric.quaternions.data[i * 4 + 2] +
        out.quaternions[i * 4 + 3] * metric.quaternions.data[i * 4 + 3],
    )
    if (dot < 0.99) flipCount++
  }
  check(
    "协方差 Σ（姿态+尺度的渲染唯一判据）",
    covMax <= 1e-7,
    `max=${e(covMax)} mean=${e(meanOf(covD))}`,
  )
  info(
    "四元数分量与 torch SVD 有符号差（等价表示）",
    `${flipCount}/${n}（${((flipCount / n) * 100).toFixed(2)}%）存在列取反，Σ 仍一致`,
  )
}

// --------------------------------------------------------------------------- #
// 阶段 4：纯函数自洽性
// --------------------------------------------------------------------------- #

function stageUnit(): void {
  // quantile（对照 torch.quantile 的 linear 插值）
  const q1 = quantile([1, 2, 3, 4], 0.1)
  const q2 = quantile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.9)
  const q3 = quantile([5], 0.5)
  const q4 = quantile([1, 2, 3, 4], 0)
  const q5 = quantile([1, 2, 3, 4], 1)
  check(
    "quantile（linear 插值）",
    Math.abs(q1 - 1.3) < 1e-12 &&
      Math.abs(q2 - 9.1) < 1e-12 &&
      q3 === 5 &&
      q4 === 1 &&
      q5 === 4,
    `q(.1)=${q1} q(.9)=${q2} 单元素=${q3} 端点=[${q4},${q5}]`,
  )

  // SH <-> RGB
  const rgb = new Float32Array(300)
  for (let i = 0; i < rgb.length; i++) rgb[i] = ((i * 37) % 100) / 100
  const sh = rgbToSphericalHarmonics(rgb)
  const shBack = sphericalHarmonicsToRgb(sh)
  const shD = diffAbs(shBack, rgb)
  check("degree-0 SH 往返", shD.max <= 1e-6, `max=${e(shD.max)}`)

  // 四元数 <-> 旋转矩阵（用 fixture 的真实四元数抽样）
  const metric = npz("metric.npz")
  const n = metric.opacities.data.length
  let maxRot = 0
  let maxQuad = 0
  for (let i = 0; i < n; i += 97) {
    const q = metric.quaternions.data
    const r = rotationMatrixFromQuaternion(
      q[i * 4],
      q[i * 4 + 1],
      q[i * 4 + 2],
      q[i * 4 + 3],
    )
    const q2 = rotationMatrixToQuaternion(r)
    const r2 = rotationMatrixFromQuaternion(q2[0], q2[1], q2[2], q2[3])
    const d = diffAbs(r2, r).max
    if (d > maxRot) maxRot = d
    // 四元数单位性（w-first）
    const norm = Math.hypot(q2[0], q2[1], q2[2], q2[3])
    if (Math.abs(norm - 1) > maxQuad) maxQuad = Math.abs(norm - 1)
  }
  check(
    "R -> q -> R 往返（fixture 抽样）",
    maxRot <= 1e-9,
    `max|ΔR|=${e(maxRot)}，|q| 偏差 ${e(maxQuad)}`,
  )

  // 对称特征分解：构造 R diag(9,4,1) Rᵀ
  const axis = [1, 2, 3]
  const axisNorm = Math.hypot(axis[0], axis[1], axis[2])
  const ang = Math.PI / 6
  const rKnown = rotationMatrixFromQuaternion(
    Math.cos(ang / 2),
    (axis[0] / axisNorm) * Math.sin(ang / 2),
    (axis[1] / axisNorm) * Math.sin(ang / 2),
    (axis[2] / axisNorm) * Math.sin(ang / 2),
  )
  const diag = new Float64Array([9, 0, 0, 0, 4, 0, 0, 0, 1])
  const mKnown = mat3Mul(mat3Mul(rKnown, diag), transpose3(rKnown))
  const eig = symmetricEigen3(mKnown)
  const eigValsOk = diffAbs(eig.values, [9, 4, 1]).max <= 1e-9
  // 重建 Σ = V Λ Vᵀ
  const lam = new Float64Array([
    eig.values[0],
    0,
    0,
    0,
    eig.values[1],
    0,
    0,
    0,
    eig.values[2],
  ])
  const rebuilt = mat3Mul(mat3Mul(eig.vectors, lam), transpose3(eig.vectors))
  const rebuildDiff = diffAbs(rebuilt, mKnown).max
  check(
    "symmetricEigen3（降序 + 可重建）",
    eigValsOk && rebuildDiff <= 1e-9,
    `λ=[${eig.values.map((v) => v.toFixed(6)).join(", ")}] 重建误差=${e(rebuildDiff)}`,
  )

  // fp16 位模式转换（platform.node.ts / platform.web.ts 的传输路径）
  //
  // 分三档，而不是「全部相对误差 <= eps」：
  //   * 可精确表示的值（2 的幂、整数、0、±Inf）必须**逐位**还原；
  //   * 一般正规值允许 fp16 的舍入（<= 2^-10）；
  //   * 小于最小 subnormal 一半（2^-25）的值必然下溢成 0，不能按相对误差算。
  const exactValues = [
    0,
    -0,
    1,
    -1,
    0.5,
    -2.5,
    0.25,
    1024,
    65504, // fp16 最大正规数
    6.103515625e-5, // 最小正规数 2^-14
    2 ** -24, // 最小 subnormal 2^-24（写成字面量会丢精度，触发 noPrecisionLoss）
    Number.POSITIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
  ]
  const roundedValues = [0.1, -0.1, 1 / 3, 1234.5678, -0.0001234]
  const underflowValues = [1e-9, -1e-9, 1e-30]
  let exactOk = true
  for (const v of exactValues) {
    const back = f16BitsToF32(Uint16Array.from([f32ToF16(v)]))[0]
    if (!Object.is(back, v)) {
      exactOk = false
      info("fp16 精确性", `期望 ${v}，得到 ${back}`)
    }
  }
  let maxRel = 0
  for (const v of roundedValues) {
    const back = f16BitsToF32(Uint16Array.from([f32ToF16(v)]))[0]
    const rel = Math.abs(back - v) / Math.abs(v)
    if (rel > maxRel) maxRel = rel
  }
  let underflowOk = true
  for (const v of underflowValues) {
    const back = f16BitsToF32(Uint16Array.from([f32ToF16(v)]))[0]
    if (Math.abs(back) > 2 ** -24) underflowOk = false
  }
  check(
    "f32 <-> fp16 位模式往返",
    exactOk && maxRel <= 2 ** -10 && underflowOk,
    `精确值逐位还原；一般值最大相对误差=${e(maxRel)}（fp16 eps≈9.8e-4）；2^-25 以下下溢为 0`,
  )
}

// --------------------------------------------------------------------------- #
// 阶段 5：颜色空间 + degree-0 SH
// --------------------------------------------------------------------------- #

function stageColorspace(): void {
  const metric = npz("metric.npz")
  const ref = referencePly()

  // 分段函数边界（对照 color_space.py 的阈值）
  //
  // 关键在于「阈值处取线性分支」，两条分支在设计上连续（0.0031308*12.92
  // 与 0.04045 是同一设计点的两侧）。输入是 Float32Array，故容差取 1e-6。
  const linearLo = linearRGB2sRGB(Float32Array.from([0.0031308]))[0]
  const srgbLo = sRGB2linearRGB(Float32Array.from([0.04045]))[0]
  check(
    "两个阈值分支在边界处连续",
    Math.abs(linearLo - 0.0031308 * 12.92) < 1e-6 &&
      Math.abs(srgbLo - 0.04045 / 12.92) < 1e-6 &&
      Math.abs(linearRGB2sRGB(Float32Array.from([1]))[0] - 1) < 1e-6,
    `linear2srgb(0.0031308)=${linearLo.toFixed(6)} srgb2linear(0.04045)=${srgbLo.toFixed(6)}`,
  )

  const rt = sRGB2linearRGB(linearRGB2sRGB(metric.colors.data))
  const rtD = diffAbs(rt, metric.colors.data)
  check(
    "linearRGB -> sRGB -> linearRGB 往返",
    rtD.max <= 1e-6,
    `max=${e(rtD.max)}`,
  )

  // save_ply 的颜色链：colors(linear) -> sRGB -> SH -> f_dc_*
  const sh = rgbToSphericalHarmonics(linearRGB2sRGB(metric.colors.data))
  const n = metric.opacities.data.length
  const fields: [string, number][] = [
    ["f_dc_0", 0],
    ["f_dc_1", 1],
    ["f_dc_2", 2],
  ]
  let worst = 0
  for (const [name, c] of fields) {
    const col = ref.column("vertex", name)
    if (!col) throw new Error(`reference.ply 缺少 vertex.${name}`)
    let max = 0
    for (let i = 0; i < n; i++) {
      const d = Math.abs(sh[i * 3 + c] - col[i])
      if (d > max) max = d
    }
    worst = Math.max(worst, max)
    info(`vertex.${name}`, `max=${e(max)}`)
  }
  check("f_dc_*（linearRGB->sRGB->SH）", worst <= 1e-6, `max=${e(worst)}`)
}

// --------------------------------------------------------------------------- #
// 阶段 6：PLY 序列化
// --------------------------------------------------------------------------- #

let referenceCache: ParsedPly | undefined

function referencePly(): ParsedPly {
  if (referenceCache) return referenceCache
  const buf = readFileSync(fixturePath("reference.ply"))
  referenceCache = parsePly(buf)
  return referenceCache
}

function stagePly(): void {
  const metric = npz("metric.npz")
  const inp = npz("input.npz")
  const w = inp.orig_width.data[0]
  const h = inp.orig_height.data[0]
  const fPx = inp.f_px.data[0]
  const n = metric.opacities.data.length

  const g = gaussiansFrom(metric)
  const meta: SceneMetaData = {
    focalLengthPx: fPx,
    resolutionPx: [w, h],
    colorSpace: "linearRGB",
  }

  const compact = serializePly({
    gaussians: g,
    fPx,
    imageShape: meta.resolutionPx,
    full: false,
  })
  const full = serializePly({
    gaussians: g,
    fPx,
    imageShape: meta.resolutionPx,
    full: true,
  })
  info(
    "输出体积",
    `compact=${humanSize(compact.byteLength)} full=${humanSize(full.byteLength)} reference=${humanSize(readFileSync(fixturePath("reference.ply")).byteLength)}`,
  )

  const ours = parsePly(compact)
  const oursFull = parsePly(full)
  const ref = referencePly()

  // ── 结构 ──
  check(
    "compact 只含 vertex element",
    ours.elements.length === 1 && ours.elements[0].name === "vertex",
    `elements=[${ours.elements.map((el) => el.name).join(", ")}]`,
  )
  check(
    "vertex count = N",
    ours.element("vertex")?.count === n,
    `${ours.element("vertex")?.count} vs ${n}`,
  )
  check(
    "compact body 字节数 = header + N*14*4",
    compact.byteLength === ours.headerBytes + n * 14 * 4,
    `实际=${compact.byteLength} 期望=${ours.headerBytes + n * 14 * 4}`,
  )
  check(
    "full header 与 reference.ply 逐字节一致",
    oursFull.header === ref.header,
    `${oursFull.headerBytes}B vs ${ref.headerBytes}B`,
  )

  // ── vertex 各列 ──
  const refV = ref.element("vertex")
  if (!refV) throw new Error("reference.ply 缺少 vertex element")
  check(
    "vertex property 列表一致",
    ours
      .element("vertex")!
      .properties.every((p, i) => p === refV.properties[i]) &&
      ours.element("vertex")!.properties.length === refV.properties.length,
    `${ours.element("vertex")!.properties.join(",")}`,
  )

  const tolerance: Record<string, number> = {
    x: 1e-5,
    y: 1e-5,
    z: 1e-5,
    f_dc_0: 1e-5,
    f_dc_1: 1e-5,
    f_dc_2: 1e-5,
    opacity: 1e-3,
    scale_0: 1e-4,
    scale_1: 1e-4,
    scale_2: 1e-4,
  }
  const refOpacity = ref.column("vertex", "opacity")!
  // 参考实现对 opacity==1 的单个高斯写出 +Inf（`log(p/(1-p))` 无 clamp），
  // 我们刻意 clamp 到 1-1e-7（写成 16.118）——公开渲染器读到 Inf 容易出 NaN。
  // 这是**唯一**一处有意的数值偏离，单独计数。
  let clamped = 0
  for (let i = 0; i < n; i++) if (!Number.isFinite(refOpacity[i])) clamped++

  let allOk = true
  for (const [name, tol] of Object.entries(tolerance)) {
    const a = ours.column("vertex", name)!
    const b = ref.column("vertex", name)!
    const skip =
      name === "opacity"
        ? (i: number) => !Number.isFinite(refOpacity[i])
        : undefined
    const d = diffAbs(a, b, skip)
    const ok = d.max <= tol
    if (!ok) allOk = false
    info(
      `vertex.${name}`,
      `max=${e(d.max)} mean=${e(d.mean)}${ok ? "" : `  ✗ 超出容差 ${e(tol)}`}`,
    )
  }
  check(
    "vertex 数值列（位置/颜色/logit-opacity/log 尺度）在容差内",
    allOk,
    clamped > 0
      ? `opacity 有 ${clamped} 个 +Inf 被我们 clamp（预期偏离）`
      : "无 clamp 偏离",
  )

  // rot_*：与 unproject 阶段同理，比 Σ 而不是分量。对拍对象是
  // 「同一份 metric.npz 经两条序列化路径（我们 vs plyfile）」，
  // 抽样即可覆盖（完整覆盖已在 unproject 阶段做过）。
  const rotCols = ["rot_0", "rot_1", "rot_2", "rot_3"].map(
    (name) => ref.column("vertex", name)!,
  )
  const ourRot = ["rot_0", "rot_1", "rot_2", "rot_3"].map(
    (name) => ours.column("vertex", name)!,
  )
  const ourScale = ["scale_0", "scale_1", "scale_2"].map(
    (name) => ours.column("vertex", name)!,
  )
  const refScale = ["scale_0", "scale_1", "scale_2"].map(
    (name) => ref.column("vertex", name)!,
  )
  const toGauss = (rot: Float64Array[], scale: Float64Array[]): Gaussians3D => {
    const q = new Float32Array(n * 4)
    const s = new Float32Array(n * 3)
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < 4; k++) q[i * 4 + k] = rot[k][i]
      for (let k = 0; k < 3; k++) s[i * 3 + k] = Math.exp(scale[k][i])
    }
    return {
      meanVectors: new Float32Array(0),
      quaternions: q,
      singularValues: s,
      colors: new Float32Array(0),
      opacities: new Float32Array(0),
    }
  }
  const ga = toGauss(ourRot, ourScale)
  const gb = toGauss(rotCols, refScale)
  const covA = new Float64Array(9)
  const covB = new Float64Array(9)
  let rotMax = 0
  for (let i = 0; i < n; i += 97) {
    covarianceOf(ga.quaternions, ga.singularValues, i, covA)
    covarianceOf(gb.quaternions, gb.singularValues, i, covB)
    for (let k = 0; k < 9; k++) {
      const d = Math.abs(covA[k] - covB[k])
      if (d > rotMax) rotMax = d
    }
  }
  check(
    "rot_*/scale_* 经 Σ 重建后一致（抽样 1/97）",
    rotMax <= 1e-7,
    `max=${e(rotMax)}`,
  )

  // ── 补充 element（full 模式）──
  const extra = ["extrinsic", "frame", "disparity", "color_space", "version"]
  let extraOk = true
  for (const name of extra) {
    const a = oursFull.column(name, name)
    const b = ref.column(name, name)
    if (!a || !b) {
      extraOk = false
      info(`element ${name}`, "缺列")
      continue
    }
    const tol = name === "disparity" ? 1e-6 : 0
    const d = diffAbs(a, b)
    const ok = d.max <= tol
    if (!ok) extraOk = false
    info(
      `element ${name}`,
      `max=${e(d.max)} [${Array.from(a)
        .slice(0, 4)
        .map((v) => v.toFixed(6))
        .join(", ")}]`,
    )
  }
  check(
    "补充 element（extrinsic/frame/disparity/color_space/version）",
    extraOk,
  )

  // intrinsic / image_size：**这里有一处 fixture 自身的口径问题**
  //
  // 官方 `predict.py` 传 `save_ply(gaussians, f_px, (height, width))`，
  // 而 `save_ply` 内部 `image_height, image_width = image_shape` —— 即
  // `image_shape` 的约定是 **(height, width)**。`01_smoke_test.py` 生成
  // reference.ply 时传的是 `(orig_w, orig_h)`，于是主点与 image_size 被交换：
  //   reference: image_size=[2268,3024] intrinsic cx=1134 cy=1512
  //   官方约定: image_size=[3024,2268] intrinsic cx=1512 cy=1134
  // 我们的 `serializePly` 以 `[width, height]` 为约定（与 SceneMetaData 一致），
  // 输出的是**官方约定**那一组。下面对两种解释都接受，但把事实打印出来。
  const intrinsicA = diffAbs(
    oursFull.column("intrinsic", "intrinsic")!,
    ref.column("intrinsic", "intrinsic")!,
  ).max
  const swappedIntrinsic = Float64Array.from(
    oursFull.column("intrinsic", "intrinsic")!,
  )
  ;[swappedIntrinsic[2], swappedIntrinsic[5]] = [
    swappedIntrinsic[5],
    swappedIntrinsic[2],
  ]
  const intrinsicB = diffAbs(
    swappedIntrinsic,
    ref.column("intrinsic", "intrinsic")!,
  ).max
  check(
    "element intrinsic（主点口径）",
    Math.min(intrinsicA, intrinsicB) <= 1e-3,
    intrinsicA <= 1e-3
      ? `与 reference 一致，cx=${oursFull.column("intrinsic", "intrinsic")![2]} cy=${oursFull.column("intrinsic", "intrinsic")![6]}`
      : `WARN: reference.ply 的主点是交换的（fixture 生成时 image_shape 传成 (W,H)）；交换后 max=${e(intrinsicB)}`,
  )

  const ourSize = oursFull.column("image_size", "image_size")!
  const refSize = ref.column("image_size", "image_size")!
  const sizeA = diffAbs(ourSize, refSize).max
  const swappedSize = Float64Array.from(ourSize)
  ;[swappedSize[0], swappedSize[1]] = [swappedSize[1], swappedSize[0]]
  const sizeB = diffAbs(swappedSize, refSize).max
  check(
    "element image_size（宽高口径）",
    Math.min(sizeA, sizeB) === 0,
    sizeA === 0
      ? `[${ourSize[0]}, ${ourSize[1]}]（与 reference 一致）`
      : `WARN: reference=[${refSize[0]}, ${refSize[1]}] 是 (H,W)，我们写 [${ourSize[0]}, ${ourSize[1]}]=(W,H)`,
  )
}

// --------------------------------------------------------------------------- #
// 阶段 7：整图预处理（JPEG 解码器差异）
// --------------------------------------------------------------------------- #

/**
 * ⚠ 本阶段是唯一个**不由算法决定精确度**的环节：`loadImage` 走 sharp/libvips，
 * torch 侧走 PIL/libturbojpeg，两者在 4:2:0 色度上采样与 YCbCr→RGB 上本就
 * 不一致（本用例：luma 均值 0.3256 vs 0.3384，平均 |ΔY|=0.013，强色度边更大；
 * 通道均值 R 0.210/0.250、G 0.367/0.369、B 0.416/0.412）。
 *
 * 这与我们的代码无关，也**不是核心算法**，故这里只做宽松的一致性检查
 * （相关性 + PSNR 下限）——真正的目的是防「整体跑飞」（行列互换、归一化倍率
 * 错、通道顺序错），那类错误会让 r 直接跌到 0.5 以下。精确误差当信息打印。
 *
 * 采样正确性由 `resize.npz` 阶段单独锁定（合成图，无解码差异）。
 */
async function stageImage(): Promise<void> {
  const inp = npz("input.npz")
  const w = inp.orig_width.data[0]
  const h = inp.orig_height.data[0]
  const fPx = inp.f_px.data[0]

  const t0 = Date.now()
  const loaded = await loadImage(DEFAULT_IMAGE)
  info(
    "loadImage",
    `${loaded.image.width}x${loaded.image.height} ch=${loaded.image.channels} f_px=${loaded.fPx.toFixed(4)} f_35mm=${loaded.focal35mm} 来自EXIF=${loaded.focalFromExif}`,
  )
  check(
    "loadImage 尺寸与 f_px",
    loaded.image.width === w &&
      loaded.image.height === h &&
      Math.abs(loaded.fPx - fPx) <= 1e-3,
    `${loaded.image.width}x${loaded.image.height} vs ${w}x${h}`,
  )

  const chw = preprocessImage(loaded.image, INTERNAL_RESOLUTION)
  info("preprocessImage", `${Date.now() - t0}ms`)

  const d = diffAbs(chw, inp.image.data)
  const p = psnr(chw, inp.image.data, 1)
  const corr = correlation(chw, inp.image.data, 31)
  info(
    "与 torch 预处理结果的差异（信息性）",
    `max=${e(d.max)} mean=${e(d.mean)} PSNR=${p.toFixed(2)}dB r=${corr.toFixed(6)}`,
  )
  // 宽松阈值：差异来自 JPEG 解码器与色彩管理，不是算法错误。
  // 实测 r≈0.988、PSNR≈26.6dB；阈值只用来挡「跑飞」。
  check(
    "整图预处理与 torch 一致（宽松：仅防跑飞）",
    corr >= 0.98 && p >= 25,
    `r=${corr.toFixed(6)}（阈值 0.98） PSNR=${p.toFixed(2)}dB（阈值 25dB）`,
  )
}

function maxAbs(a: ArrayLike<number>): number {
  let m = 0
  for (let i = 0; i < a.length; i++) {
    const v = Math.abs(a[i])
    if (v > m) m = v
  }
  return m
}

function meanOf(a: ArrayLike<number>): number {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i]
  return s / a.length
}

function transpose3(m: ArrayLike<number>): Float64Array {
  return new Float64Array([
    m[0],
    m[3],
    m[6],
    m[1],
    m[4],
    m[7],
    m[2],
    m[5],
    m[8],
  ])
}

// --------------------------------------------------------------------------- #
// 主流程
// --------------------------------------------------------------------------- #

const STAGE_FNS: Record<StageName, () => Promise<void> | void> = {
  resize: stageResize,
  fov: stageFov,
  unproject: stageUnproject,
  unit: stageUnit,
  colorspace: stageColorspace,
  ply: stagePly,
  image: stageImage,
}

async function main(): Promise<void> {
  const { values } = parseArgs({
    options: {
      fixtures: { type: "string" },
      only: { type: "string" },
      "no-image": { type: "boolean", default: false },
      list: { type: "boolean", default: false },
      help: { type: "boolean", short: "h", default: false },
    },
    allowPositionals: false,
  })

  if (values.help || values.list) {
    console.log("用法: npx tsx scripts/sharp-compare-fixtures.ts [选项]\n")
    console.log(
      "  --fixtures <dir>  fixture 目录（默认 py-models/out/fixtures）",
    )
    console.log("  --only <a,b>      只跑指定阶段（逗号分隔）")
    console.log("  --no-image        跳过整图预处理阶段（省去加载原图）")
    console.log("  --list            列出阶段\n")
    console.log("阶段:")
    for (const [name, title] of STAGES)
      console.log(`  ${name.padEnd(11)} ${title}`)
    return
  }

  if (values.fixtures) fixturesDir = resolve(values.fixtures)

  const wanted = new Set<StageName>(
    values.only
      ? (values.only.split(",").map((s) => s.trim()) as StageName[])
      : (STAGES.map(([n]) => n) as StageName[]),
  )
  for (const name of wanted) {
    if (!(name in STAGE_FNS)) {
      throw new Error(
        `未知阶段 "${name}"，可选：${STAGES.map(([n]) => n).join(", ")}`,
      )
    }
  }
  if (values["no-image"]) wanted.delete("image")

  console.log(`fixtures: ${fixturesDir}`)
  console.log(`阶段: ${[...wanted].join(", ")}\n`)

  for (const [name, title] of STAGES) {
    if (!wanted.has(name)) continue
    currentStage = name
    console.log(`── [${name}] ${title}`)
    const t0 = Date.now()
    await STAGE_FNS[name]()
    console.log(`   (${Date.now() - t0}ms)\n`)
  }

  // ── 汇总 ──
  const pass = lines.filter((l) => l.status === "pass").length
  const fail = lines.filter((l) => l.status === "fail")
  const infoCount = lines.filter((l) => l.status === "info").length

  console.log("=".repeat(72))
  console.log("按阶段统计")
  for (const [name] of STAGES) {
    const of = lines.filter((l) => l.stage === name)
    if (of.length === 0) continue
    const f = of.filter((l) => l.status === "fail").length
    const p = of.filter((l) => l.status === "pass").length
    console.log(
      `  ${name.padEnd(11)} ${p} 通过 / ${f} 失败 / ${of.length - p - f} 信息`,
    )
  }
  console.log("=".repeat(72))
  if (fail.length === 0) {
    console.log(`全部通过：${pass} 项断言（另有 ${infoCount} 条信息）`)
    return
  }
  console.log(`失败 ${fail.length} 项：`)
  for (const l of fail)
    console.log(`  [${l.stage}] ${l.label}${l.detail ? `  — ${l.detail}` : ""}`)
  process.exitCode = 1
}

main().catch((err) => {
  console.error(`\n对拍失败: ${err instanceof Error ? err.message : err}`)
  process.exitCode = 1
})
