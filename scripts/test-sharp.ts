/**
 * 单元级自检：验证 `sharp/` 与 `export/` 的数值实现。
 *
 * 这些测试不依赖 ort，也不需要模型文件，纯数学核对。
 * 覆盖三处最容易出 bias 的地方：
 *   1. NDC -> metric 反投影矩阵（对照 predict.py 的构造）
 *   2. 协方差 compose/decompose 往返一致性
 *   3. PLY 颜色编码：linearRGB -> sRGB -> SH 的可逆性与已知值
 *
 * 用法：npx tsx scripts/test-sharp.ts
 */

import {
  encodeColorSpace,
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
  decomposeCovarianceMatrix,
  rgbToSphericalHarmonics,
  rotationMatrixFromQuaternion,
  sphericalHarmonicsToRgb,
} from "../src/spatial-scene/sharp/linalg.ts"
import type { Gaussians3D } from "../src/spatial-scene/sharp/types.ts"
import {
  applyTransform,
  getUnprojectionMatrix,
  identity4,
  mat4Mul,
} from "../src/spatial-scene/sharp/unproject.ts"

let failed = 0
let passed = 0

function check(cond: boolean, msg: string, detail = ""): void {
  if (cond) {
    passed++
    console.log(`  PASS  ${msg}`)
  } else {
    failed++
    console.log(`  FAIL  ${msg}${detail ? `  ${detail}` : ""}`)
  }
}

function near(a: number, b: number, tol = 1e-6): boolean {
  return Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b))
}

function nearArr(
  a: ArrayLike<number>,
  b: ArrayLike<number>,
  tol = 1e-6,
): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) if (!near(a[i], b[i], tol)) return false
  return true
}

// ── 1. FOV ──
console.log("\n[1] FOV / 焦距")
{
  // convert_focallength: 30mm 在 4032x3024 上 = 30 * 5040 / 43.267 ≈ 3494.6
  // 注意用的是**对角线**换算，所以值会比强按宽算的大。
  const f = convertFocallength(4032, 3024, 30)
  const expected = (30 * Math.hypot(4032, 3024)) / Math.hypot(36, 24)
  check(
    near(f, expected, 1e-12),
    `convert_focallength(4032,3024,30mm) = ${f.toFixed(2)} px`,
  )

  // EXIF fallback 链
  check(resolveFocalLength35mm(undefined) === 30.0, "无 EXIF -> 30mm")
  check(resolveFocalLength35mm({}) === 30.0, "空 EXIF -> 30mm")
  // 对照 io.py：只有「35mm 字段」才做 <1 判空；普通 FocalLength 不做（原实现的疏漏，一并复刻）
  check(
    resolveFocalLength35mm({ FocalLength: 0 }) === 0.0,
    "FocalLength=0 原样（原实现无 <1 检查）",
  )
  check(
    resolveFocalLength35mm({ FocalLength: 5 }) === 42.0,
    "FocalLength=5 -> 5*8.4=42 (<10 修正)",
  )
  check(
    resolveFocalLength35mm({ FocalLength: 50 }) === 50.0,
    "FocalLength=50 原样",
  )
  check(
    resolveFocalLength35mm({ FocalLengthIn35mmFilm: 24, FocalLength: 50 }) ===
      24.0,
    "优先 FocalLengthIn35mmFilm",
  )
  check(
    resolveFocalLength35mm({ FocalLengthIn35mmFilm: 0, FocalLength: 35 }) ===
      35.0,
    "35mm 字段为 0 时回退 FocalLength",
  )

  // disparity_factor 分母是 width（不是 1536）
  check(
    near(disparityFactor(641.71, 545), 641.71 / 545),
    "disparity_factor = f_px/width",
  )

  // intrinsics_resized：第 0 行乘 1536/W，第 1 行乘 1536/H
  const k = intrinsicsResized(641.71, 545, 748, 1536)
  const sx = 1536 / 545
  const sy = 1536 / 748
  check(near(k[0], 641.71 * sx), "K[0,0] = f_px * 1536/W")
  check(near(k[2], (545 / 2) * sx), "K[0,2] = (W/2) * 1536/W")
  check(near(k[5], 641.71 * sy), "K[1,1] = f_px * 1536/H")
  check(near(k[6], (748 / 2) * sy), "K[1,2] = (H/2) * 1536/H")
  check(near(k[10], 1), "K[2,2] = 1")
}

// ── 2. 反投影矩阵 ──
console.log("\n[2] NDC -> metric 反投影")
{
  const W = 545
  const H = 748
  const fPx = 641.71
  const k = intrinsicsResized(fPx, W, H, 1536)
  const M = getUnprojectionMatrix(identity4(), k, [1536, 1536])

  // 期望：对角缩放 diag(W/(2*f_px_resized), H/(2*f_px_resized_y), 1)
  // f_px_resized_x = f_px * 1536/W  => scale_x = 1536/(2*f_px*1536/W) = W/(2*f_px)
  check(
    near(M[0], W / (2 * fPx), 1e-5),
    `scaleX = W/(2*f_px) = ${(W / (2 * fPx)).toFixed(6)}`,
  )
  check(
    near(M[5], H / (2 * fPx), 1e-5),
    `scaleY = H/(2*f_px) = ${(H / (2 * fPx)).toFixed(6)}`,
  )
  check(near(M[10], 1), "scaleZ = 1（NDC z 即深度）")
  check(
    near(M[3], 0) && near(M[7], 0) && near(M[11], 0),
    "无平移（主点在图像中心）",
  )

  // 逆矩阵往返
  const I = mat4Mul(M, getUnprojectionMatrix(identity4(), k, [1536, 1536]))
  // 注意：这里 M*M 不是单位阵，改成验证 M 的对角性即可
  check(
    nearArr([M[1], M[2], M[4], M[6], M[8], M[9]], [0, 0, 0, 0, 0, 0]),
    "矩阵为纯对角",
  )
  void I

  // 语义校验：NDC 中心 (0,0,5) 应映射到光轴上 (0,0,5)
  const g: Gaussians3D = {
    meanVectors: new Float32Array([0, 0, 5]),
    singularValues: new Float32Array([0.01, 0.01, 0.01]),
    quaternions: new Float32Array([1, 0, 0, 0]),
    colors: new Float32Array([0.5, 0.5, 0.5]),
    opacities: new Float32Array([1]),
  }
  const out = applyTransform(g, M)
  check(
    near(out.meanVectors[0], 0, 1e-6) && near(out.meanVectors[1], 0, 1e-6),
    "NDC 原点保持在光轴",
  )
  check(near(out.meanVectors[2], 5, 1e-6), "NDC z 不变（z 即深度）")

  // 语义校验：NDC x=1 在 z=5 处 -> x = 5 * scaleX = 5 * W/(2f)
  // 注意 applyTransform 的 z 不变，且 x 是线性缩放（与 z 无关）。
  const g2: Gaussians3D = {
    meanVectors: new Float32Array([1, 1, 5]),
    singularValues: new Float32Array([0.01, 0.01, 0.01]),
    quaternions: new Float32Array([1, 0, 0, 0]),
    colors: new Float32Array([0.5, 0.5, 0.5]),
    opacities: new Float32Array([1]),
  }
  const o2 = applyTransform(g2, M)
  check(near(o2.meanVectors[0], M[0], 1e-5), "NDC x=1 -> scaleX")
  check(near(o2.meanVectors[1], M[5], 1e-5), "NDC y=1 -> scaleY")

  // 物理一致性：NDC 的半宽为 1，对应图像宽度 W；世界空间半宽应为 z·W/(2f_px)。
  // 由于变换是线性的（与 z 无关），只需验证 scaleX 与 scaleY 的**比例**：
  //   scaleY / scaleX == H / W
  // 两者共用同一 f_px，分母各取自己的尺寸。这是对角矩阵必须满足的约束，
  // 能抳住「行/列错配」类错误。
  check(
    near(M[5] / M[0], H / W, 1e-9),
    `scaleY/scaleX == H/W = ${(H / W).toFixed(6)}`,
  )
}

// ── 3. 协方差 compose/decompose 往返 ──
console.log("\n[3] 协方差 compose/decompose 往返")
{
  const cases: {
    q: [number, number, number, number]
    s: [number, number, number]
  }[] = [
    { q: [1, 0, 0, 0], s: [1, 2, 3] },
    { q: [Math.SQRT1_2, Math.SQRT1_2, 0, 0], s: [1, 2, 3] },
    { q: [0.5, 0.5, 0.5, 0.5], s: [0.1, 0.2, 0.3] },
    { q: [1 / 3, 2 / 3, 0.8017837257372732, 0], s: [0.01, 1, 1] },
    { q: [1, 2, 3, 4], s: [3, 2, 1] },
    { q: [0.1, -0.7, 0.3, 0.2], s: [5, 0.5, 0.05] },
  ]

  // 真正需要的不变量：compose(decompose(Σ)) == Σ。
  //
  // 不能直接比较四元数：decompose 走 SVD/特征分解，约定奇异值**降序**，
  // 因此当输入的 s 不是降序时，需要旋转来重排主轴，四元数本来就会变。
  // 比较协方差矩阵则不受此影响（compose 与 s 的顺序无关）。
  for (const [i, c] of cases.entries()) {
    const qn = Math.hypot(...c.q)
    const q = c.q.map((v) => v / qn)
    const R = rotationMatrixFromQuaternion(q[0], q[1], q[2], q[3])
    const cov = composeCovarianceMatrix(R, c.s[0], c.s[1], c.s[2])
    const { quaternion, singularValues } = decomposeCovarianceMatrix(cov)

    const R2 = rotationMatrixFromQuaternion(
      quaternion[0],
      quaternion[1],
      quaternion[2],
      quaternion[3],
    )
    const cov2 = composeCovarianceMatrix(
      R2,
      singularValues[0],
      singularValues[1],
      singularValues[2],
    )

    let maxErr = 0
    for (let k = 0; k < 9; k++)
      maxErr = Math.max(maxErr, Math.abs(cov[k] - cov2[k]))
    check(
      maxErr < 1e-10,
      `case ${i}: compose(decompose(Σ)) == Σ (err=${maxErr.toExponential(2)})`,
    )

    // 奇异值应为输入的降序排列
    const sortedS = [...c.s].sort((a, b) => b - a)
    check(
      sortedS.every((s, k) => near(s, singularValues[k], 1e-4)),
      `case ${i}: 奇异值降序 [${singularValues.map((v) => v.toFixed(4)).join(", ")}]`,
    )

    // 四元数必须归一化（PLY 渲染器直接使用，不归一化会出错）
    const qNorm = Math.hypot(...quaternion)
    check(
      near(qNorm, 1, 1e-9),
      `case ${i}: 四元数已归一化 |q|=${qNorm.toFixed(9)}`,
    )
  }
}

// ── 4. 颜色空间 ──
console.log("\n[4] 颜色空间")
{
  check(
    near(linearRGB2sRGB(new Float32Array([0]))[0], 0),
    "linearRGB 0 -> sRGB 0",
  )
  check(
    near(linearRGB2sRGB(new Float32Array([1]))[0], 1, 1e-6),
    "linearRGB 1 -> sRGB 1",
  )
  // 阈值分支：0.0031308 以下走线性
  check(
    near(linearRGB2sRGB(new Float32Array([0.001]))[0], 0.001 * 12.92),
    "线性分支",
  )
  // 已知值：linear 0.5 -> sRGB 0.735357
  check(
    near(linearRGB2sRGB(new Float32Array([0.5]))[0], 0.7353569, 1e-5),
    "linear 0.5 -> sRGB 0.735357",
  )
  // 往返
  const x = new Float32Array([0.0005, 0.01, 0.2, 0.5, 0.9, 1])
  const rt = sRGB2linearRGB(linearRGB2sRGB(x))
  check(nearArr(rt, x, 1e-5), "linearRGB -> sRGB -> linearRGB 往返")
  check(
    encodeColorSpace("sRGB") === 0 && encodeColorSpace("linearRGB") === 1,
    "color_space 编码 0/1",
  )
}

// ── 5. SH 颜色编码 ──
console.log("\n[5] degree-0 球谐")
{
  const c0 = Math.sqrt(1 / (4 * Math.PI))
  // 0.5 应编码为 0
  check(
    near(rgbToSphericalHarmonics(new Float32Array([0.5]))[0], 0, 1e-7),
    "rgb=0.5 -> SH=0",
  )
  // 往返
  const rgb = new Float32Array([0, 0.25, 0.5, 0.75, 1])
  const rt = sphericalHarmonicsToRgb(rgbToSphericalHarmonics(rgb))
  check(nearArr(rt, rgb, 1e-6), "rgb -> SH -> rgb 往返")
  // 已知系数
  check(near(c0, 0.28209479177387814, 1e-12), `C0 = ${c0}`)
  check(
    near(rgbToSphericalHarmonics(new Float32Array([1]))[0], 0.5 / c0, 1e-6),
    "rgb=1 -> SH = 0.5/C0",
  )
}

console.log()
console.log(`结果: ${passed} 通过, ${failed} 失败`)
if (failed > 0) process.exit(1)
