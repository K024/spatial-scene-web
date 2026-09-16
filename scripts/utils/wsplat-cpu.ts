/**
 * CPU 参考光栅器：`src/spatial-scene/wsplat` 的 WGSL 的**数值基准**。
 *
 * 数学按 **gsplat 的前向公式**逐条实现（`antialiased` 的透明度补偿由 `eps2d` 驱动：
 * `eps2d=0` 时补偿恒等于 1，等价于 `classic`）
 * （WGSL 侧也已改成同一套语义，不再用 playcanvas 的 quad 近似）：
 *   1. 从 (四元数, 尺度) 组 3D 协方差 `Σ3 = R S² Rᵀ`
 *   2. 视图旋转 `W`（world -> camera）作用到 Σ3，再用**像素空间**雅可比
 *      `J = [[f/z, 0, -f·x/z²], [0, f/z, -f·y/z²]]`（真像素焦距 `f = f_px`）
 *   3. `Σ2 = J W Σ3 Wᵀ Jᵀ`，然后 `Σ2' = Σ2 + eps2d·I`（默认 eps2d = 0，见下）
 *   4. `aaFactor = sqrt(max(det Σ2 / det Σ2', MIN_COMPENSATION²))`
 *   5. `α(p) = min(0.99, opacity · aaFactor · exp(-½ · dᵀ Σ2'⁻¹ d))`
 *      其中 `d` 是像素中心相对高斯中心的偏移（单位像素）；低于 `alphaClip` 丢弃
 *   6. back-to-front（远 -> 近）做 `over` 合成，累积
 *      `A`、`C_premul`、`ED = Σ T·α·z`；`D = ED / A`
 *
 * ── 与 WGSL 的一致性（尺度换算，不要改错）──
 * WGSL 在 `focal = 2·f_px` 上建 Σ2（是本模块真像素焦距 Σ2 的 **4 倍**），所以：
 *   - WGSL 注入的 `EPS2D`（= 本模块 `eps2d` × 4）对应本模块的 `eps2d`；
 *   - WGSL 的 quad 像素半轴 `0.5·3.33·sqrt(λ_wgsl)` 对应本模块的 `3.33·σ_cpu`。
 * 本模块的 `eps2d` 选项就是 **gsplat 语义的值（默认 0，对齐 SHARP/ml-sharp 官方渲染）**，
 * 内部按 4 倍换算到 WGSL 尺度。
 * quad 只决定光栅化的范围，核两边都是**真高斯**，所以差异只剩 f16 附件/插值的舍入。
 *
 * ── 剔除顺序与 `quad.ts` 的 `vsSplat` 逐条一致 ──
 *   `原始 opacity ≤ alphaClip` -> `z ≤ eps` -> `minPixelSize` -> 视锥
 *   -> `(AA 补偿后) alpha ≤ alphaClip` -> 绘制。
 * `minPixelSize` / 视锥是 playcanvas 的性能剔除（gsplat 没有），保留但判据与 WGSL 相同。
 *
 * 纯 CPU、无 GPU 依赖；node 脚本专用（不进 `src/`）。
 */

import type { Gaussians3D } from "../../src/spatial-scene/sharp/types.ts"
import {
  computeViewDepths,
  sortSplatsBackToFront,
} from "../../src/spatial-scene/wsplat/sort.ts"

export interface CpuCamera {
  /** world -> camera，列主序 16 元素。 */
  viewMatrix: Float32Array
  fx: number
  fy: number
  cx: number
  cy: number
  width: number
  height: number
}

export interface CpuRenderOptions {
  gaussians: Gaussians3D
  camera: CpuCamera
  /**
   * 2D 协方差对角加项，**gsplat 语义的值**（真像素焦距的 Σ2 上）。
   *
   * 默认 `0` —— 对齐 SHARP / ml-sharp 的官方渲染（`GSplatRenderer` 传 `eps2d=0`）。
   * gsplat 的默认值是 0.3；要复现它就显式传 0.3（此时才会算透明度补偿）。
   * WGSL 侧的 Σ2 建在 `focal = 2·f_px` 上（4 倍），所以那边用的是 `eps2d × 4`；
   * 本模块内部也会按 4 倍换算（见下方 `WGSL_COV_SCALE` 的一致性说明）。
   */
  eps2d?: number
  /** 低于该 α 直接跳过（对齐上游 `1/255`）。 */
  alphaClip?: number
  /** 与 WGSL 的 `minPixelSize` 对齐（默认 2）。 */
  minPixelSize?: number
}

export interface CpuRenderResult {
  /** 直通线性 RGB，`w*h*3`。 */
  rgb: Float32Array
  /** 累积 alpha，`w*h`。 */
  alpha: Float32Array
  /** 真实度量深度 D（米），背景处 0。 */
  depth: Float32Array
  /** 透射率 T = 1 - A。 */
  transmission: Float32Array
  /** 未归一化累积深度 ED。 */
  accumulatedDepth: Float32Array
  /**
   * 「最近点 z」：所有贡献过该像素的高斯中，视图空间 z 最小者。
   * 这是 z-buffer 语义的深度，与 alpha 加权 D 不同，只作对照参考。
   */
  nearestDepth: Float32Array
  visible: Uint8Array
  /** sRGB 预览 RGBA8（目视 / 存 PNG）。 */
  preview: Uint8Array
  /** 参与了合成的高斯数（通过可见性剔除与包围盒检查的实例数）。 */
  drawnSplats: number
  /**
   * 剔除计数（诊断用，与 WGSL 的**同一套判据与顺序**）。
   *
   * 用来与 GPU 的 `WSplatStats` 交叉验证：两边应当逐项接近
   * （残差只来自 quad 插值与 f16 附件的舍入）。
   */
  cull: {
    total: number
    behindCamera: number
    nonPositiveDeterminant: number
    alphaClip: number
    alphaClipAfterAa: number
    minPixelSize: number
    outsideView: number
    drawn: number
  }
}

/** gsplat 语义的 eps2d（真像素焦距尺度）；默认 0 = 对齐 SHARP/ml-sharp 官方渲染。 */
const DEFAULT_EPS2D = 0
const DEFAULT_ALPHA_CLIP = 1 / 255
/** 单颗高斯的逐像素 alpha 上限（gsplat 的 MAX_ALPHA）。 */
const MAX_ALPHA = 0.99
/** 补偿因子下限（gsplat 的 MIN_COMPENSATION）。 */
const MIN_COMPENSATION = 0.005
/** quad 半轴 = GAUSS_SIGMA_RADIUS · σ（gsplat 的 radius 常数）。 */
const GAUSS_SIGMA_RADIUS = 3.33
/** 真高斯指数系数：α ∝ exp(-GAUSS_K2 · A)，A = r² / GAUSS_SIGMA_RADIUS²。 */
const GAUSS_K2 = 0.5 * GAUSS_SIGMA_RADIUS * GAUSS_SIGMA_RADIUS

/**
 * WGSL 的 2D 协方差相对本模块的**尺度倍数**。
 *
 * `gsplatCorner.ts` 的 `focal = viewport_size.x · projMat00 = 2·f_px`
 * （上游 playcanvas 的写法，见该文件头「量纲自洽」一段），所以它的
 * `Σ2 = J Σv Jᵀ` 是本模块 `(f_px / z)` 版本的 **4 倍**。
 * 推论：
 *   - WGSL 注入的 `EPS2D = eps2d × 4` 对应本模块尺度上的 `eps2d`；
 *   - WGSL 的 quad 像素半轴 `0.5·3.33·sqrt(λ_wgsl)` 对应本模块的 `3.33·σ_cpu`。
 * 不换算就会得到「CPU 预测剔除 73.55% vs GPU 实际 46.7%」这种对不上的数。
 */
const WGSL_COV_SCALE = 4

/** 跑 CPU 参考。 */
export function renderSplatsCpu(options: CpuRenderOptions): CpuRenderResult {
  const { gaussians, camera } = options
  const eps2d = options.eps2d ?? DEFAULT_EPS2D
  const alphaClip = options.alphaClip ?? DEFAULT_ALPHA_CLIP

  const count = gaussians.opacities.length
  const { width, height } = camera
  const px = width * height

  // 与 WGSL 一致的 minPixelSize / vmin（`gsplatCorner.ts`）
  const minPixelSize = options.minPixelSize ?? 2
  const vmin = Math.min(1024, Math.min(width, height))
  const cull = {
    total: count,
    behindCamera: 0,
    nonPositiveDeterminant: 0,
    alphaClip: 0,
    alphaClipAfterAa: 0,
    minPixelSize: 0,
    outsideView: 0,
    drawn: 0,
  }

  const colorAcc = new Float32Array(px * 3)
  const alphaAcc = new Float32Array(px)
  const edAcc = new Float32Array(px)
  const nearestAcc = new Float32Array(px)

  // ── back-to-front ──
  const depths = computeViewDepths(
    gaussians.meanVectors,
    camera.viewMatrix,
    count,
  )
  const order = sortSplatsBackToFront(depths, count)

  const mv = camera.viewMatrix
  let drawn = 0

  for (let k = 0; k < count; k++) {
    const i = order[k]

    // 1) 原始 opacity 的 alphaClip（与 `quad.ts` 同序：在剔除相机后方之前）
    if (gaussians.opacities[i] <= alphaClip) {
      cull.alphaClip++
      continue
    }

    const z = depths[i]
    // 2) 相机后方 / 近零
    if (!(z > 1e-4)) {
      cull.behindCamera++
      continue
    }

    // 视图空间中心
    const mx = gaussians.meanVectors[i * 3]
    const my = gaussians.meanVectors[i * 3 + 1]
    const mz = gaussians.meanVectors[i * 3 + 2]
    const vx = mv[0] * mx + mv[4] * my + mv[8] * mz + mv[12]
    const vy = mv[1] * mx + mv[5] * my + mv[9] * mz + mv[13]

    const u = (camera.fx * vx) / z + camera.cx
    const v = (camera.fy * vy) / z + camera.cy

    // ── world -> camera 的 3x3 旋转（列主序）──
    const w00 = mv[0]
    const w01 = mv[4]
    const w02 = mv[8]
    const w10 = mv[1]
    const w11 = mv[5]
    const w12 = mv[9]
    const w20 = mv[2]
    const w21 = mv[6]
    const w22 = mv[10]

    // ── Σ3 = R S² Rᵀ，再旋转到视图空间：Σv = W Σ3 Wᵀ ──
    const cov = covariance3toView(gaussians, i, [
      [w00, w01, w02],
      [w10, w11, w12],
      [w20, w21, w22],
    ])

    // ── 像素空间雅可比（真像素焦距）──
    const j1x = camera.fx / z
    const j1y = camera.fy / z
    const j2x = (-j1x * vx) / z
    const j2y = (-j1y * vy) / z

    // Σ2 = J Σv Jᵀ（2x2）
    const a00 =
      j1x * cov[0][0] * j1x +
      j1x * cov[0][2] * j2x +
      j2x * cov[2][0] * j1x +
      j2x * cov[2][2] * j2x
    const a01 =
      j1x * cov[0][1] * j1y +
      j1x * cov[0][2] * j2y +
      j2x * cov[2][1] * j1y +
      j2x * cov[2][2] * j2y
    const a11 =
      j1y * cov[1][1] * j1y +
      j1y * cov[1][2] * j2y +
      j2y * cov[2][1] * j1y +
      j2y * cov[2][2] * j2y

    const detRaw = a00 * a11 - a01 * a01
    // 3) 非正定（CPU 兜底，WGSL 没有这条；正常数据不触发）
    if (!(detRaw > 0)) {
      cull.nonPositiveDeterminant++
      continue
    }

    // Σ2' = Σ2 + eps2d·I（gsplat 语义：eps2d 加在真像素焦距的 Σ2 上，默认 0）
    const b00 = a00 + eps2d
    const b11 = a11 + eps2d
    const detBlur = b00 * b11 - a01 * a01
    if (!(detBlur > 0)) continue

    // WGSL 尺度的 Σ2'（4 倍）：quad 半轴 / minPixelSize 用
    const d1 = b00 * WGSL_COV_SCALE
    const off2 = a01 * WGSL_COV_SCALE
    const d2 = b11 * WGSL_COV_SCALE
    const mid2 = 0.5 * (d1 + d2)
    const rad2 = Math.hypot((d1 - d2) / 2, off2)
    const lam1 = mid2 + rad2
    const lam2 = Math.max(mid2 - rad2, 0.1)
    // 像素半轴 = min(3.33σ, vmin)（与 WGSL 的 0.5·3.33·sqrt(λ_wgsl) 等价）
    const pixelRadius1 = Math.min(
      0.5 * GAUSS_SIGMA_RADIUS * Math.sqrt(lam1),
      vmin,
    )
    const pixelRadius2 = Math.min(
      0.5 * GAUSS_SIGMA_RADIUS * Math.sqrt(lam2),
      vmin,
    )
    const l1 = 2 * pixelRadius1
    const l2 = 2 * pixelRadius2

    // 4) minPixelSize（与 WGSL 同式：max(l1,l2) < minPixelSize）
    if (Math.max(l1, l2) < minPixelSize) {
      cull.minPixelSize++
      continue
    }

    // 椭圆主轴方向（比例不随尺度变，故直接用 WGSL 尺度）
    const dir = normalize2(off2, lam1 - d1)
    const perpX = dir.y
    const perpY = -dir.x

    // 5) 视锥剔除：quad 的轴对齐包围盒
    const extX = Math.abs(pixelRadius1 * dir.x) + Math.abs(pixelRadius2 * perpX)
    const extY = Math.abs(pixelRadius1 * dir.y) + Math.abs(pixelRadius2 * perpY)
    const x0 = Math.max(0, Math.floor(u - extX))
    const x1 = Math.min(width - 1, Math.ceil(u + extX))
    const y0 = Math.max(0, Math.floor(v - extY))
    const y1 = Math.min(height - 1, Math.ceil(v + extY))
    if (x1 < x0 || y1 < y0) {
      cull.outsideView++
      continue
    }

    // 6) AA 补偿后的 alphaClip（与 WGSL 同序：在视锥之后）
    const aaFactor = Math.sqrt(
      Math.max(detRaw / detBlur, MIN_COMPENSATION * MIN_COMPENSATION),
    )
    const alpha = gaussians.opacities[i] * aaFactor
    if (alpha <= alphaClip) {
      cull.alphaClipAfterAa++
      continue
    }

    drawn++
    cull.drawn++
    const cr = gaussians.colors[i * 3]
    const cg = gaussians.colors[i * 3 + 1]
    const cb = gaussians.colors[i * 3 + 2]

    const invR1Sq = 1 / (pixelRadius1 * pixelRadius1)
    const invR2Sq = 1 / (pixelRadius2 * pixelRadius2)

    for (let y = y0; y <= y1; y++) {
      const dy = y + 0.5 - v
      const rowBase = y * width
      for (let x = x0; x <= x1; x++) {
        const dx = x + 0.5 - u
        // quad 参数 A = r² / 3.33²：du 沿主轴（半轴 pixelRadius1），dv 沿副轴
        const du = dx * dir.x + dy * dir.y
        const dv = dx * perpX + dy * perpY
        const A = du * du * invR1Sq + dv * dv * invR2Sq
        if (A > 1) continue
        // 真高斯 α = opacity·aaFactor·exp(-½ r²)
        let a = alpha * Math.exp(-GAUSS_K2 * A)
        if (a > MAX_ALPHA) a = MAX_ALPHA
        if (a < alphaClip) continue
        const idx = rowBase + x
        const k3 = idx * 3
        colorAcc[k3] = cr * a + colorAcc[k3] * (1 - a)
        colorAcc[k3 + 1] = cg * a + colorAcc[k3 + 1] * (1 - a)
        colorAcc[k3 + 2] = cb * a + colorAcc[k3 + 2] * (1 - a)
        edAcc[idx] = z * a + edAcc[idx] * (1 - a)
        alphaAcc[idx] = a + alphaAcc[idx] * (1 - a)
        if (a > 0 && (nearestAcc[idx] === 0 || z < nearestAcc[idx])) {
          nearestAcc[idx] = z
        }
      }
    }
  }

  // ── resolve ──
  const rgb = new Float32Array(px * 3)
  const outAlpha = new Float32Array(px)
  const depth = new Float32Array(px)
  const transmission = new Float32Array(px)
  const accumulatedDepth = new Float32Array(px)
  const nearestDepth = new Float32Array(px)
  const visible = new Uint8Array(px)
  const preview = new Uint8Array(px * 4)

  for (let i = 0; i < px; i++) {
    // colorAcc 是预乘的（C = Σ c·α·T），需要还原直通
    const A = alphaAcc[i]
    outAlpha[i] = A
    transmission[i] = 1 - A
    accumulatedDepth[i] = edAcc[i]
    nearestDepth[i] = nearestAcc[i]
    const invA = 1 / Math.max(A, 1e-6)
    const r = colorAcc[i * 3] * invA
    const g = colorAcc[i * 3 + 1] * invA
    const b = colorAcc[i * 3 + 2] * invA
    rgb[i * 3] = r
    rgb[i * 3 + 1] = g
    rgb[i * 3 + 2] = b
    const d = edAcc[i] * invA
    depth[i] = A > 0 ? d : 0
    visible[i] = A > 0 ? 1 : 0
    preview[i * 4] = linearToSrgbByte(r)
    preview[i * 4 + 1] = linearToSrgbByte(g)
    preview[i * 4 + 2] = linearToSrgbByte(b)
    preview[i * 4 + 3] = Math.round(Math.min(1, Math.max(0, A)) * 255)
  }

  return {
    rgb,
    alpha: outAlpha,
    depth,
    transmission,
    accumulatedDepth,
    nearestDepth,
    visible,
    preview,
    drawnSplats: drawn,
    cull,
  }
}

/**
 * `Σv = W · R S² Rᵀ · Wᵀ`（3x3）。
 *
 * 先算 R S² Rᵀ（从四元数 + 奇异值），再左右各乘 W / Wᵀ。
 * 全部用扁平数组，避免为 118 万个高斯分配临时对象。
 */
type Mat3Rows = [
  [number, number, number],
  [number, number, number],
  [number, number, number],
]

/** 复用的临时 3x3（单线程，安全）。 */
const tmpCov: Mat3Rows = [
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
]

function covariance3toView(
  gaussians: Gaussians3D,
  i: number,
  w: Mat3Rows,
): Mat3Rows {
  const qw = gaussians.quaternions[i * 4]
  const qx = gaussians.quaternions[i * 4 + 1]
  const qy = gaussians.quaternions[i * 4 + 2]
  const qz = gaussians.quaternions[i * 4 + 3]
  // 归一化（与 WGSL 一致）
  const len = Math.hypot(qw, qx, qy, qz) || 1
  const rw = qw / len
  const rx = qx / len
  const ry = qy / len
  const rz = qz / len

  // R（行主序）
  const r00 = 1 - 2 * (ry * ry + rz * rz)
  const r01 = 2 * (rx * ry - rw * rz)
  const r02 = 2 * (rx * rz + rw * ry)
  const r10 = 2 * (rx * ry + rw * rz)
  const r11 = 1 - 2 * (rx * rx + rz * rz)
  const r12 = 2 * (ry * rz - rw * rx)
  const r20 = 2 * (rx * rz - rw * ry)
  const r21 = 2 * (ry * rz + rw * rx)
  const r22 = 1 - 2 * (rx * rx + ry * ry)

  const sx = gaussians.singularValues[i * 3]
  const sy = gaussians.singularValues[i * 3 + 1]
  const sz = gaussians.singularValues[i * 3 + 2]

  // M = R · diag(s) => Σ3 = M Mᵀ
  const m00 = r00 * sx
  const m01 = r01 * sy
  const m02 = r02 * sz
  const m10 = r10 * sx
  const m11 = r11 * sy
  const m12 = r12 * sz
  const m20 = r20 * sx
  const m21 = r21 * sy
  const m22 = r22 * sz

  const s00 = m00 * m00 + m01 * m01 + m02 * m02
  const s01 = m00 * m10 + m01 * m11 + m02 * m12
  const s02 = m00 * m20 + m01 * m21 + m02 * m22
  const s11 = m10 * m10 + m11 * m11 + m12 * m12
  const s12 = m10 * m20 + m11 * m21 + m12 * m22
  const s22 = m20 * m20 + m21 * m21 + m22 * m22

  // Σv = W Σ3 Wᵀ
  for (let row = 0; row < 3; row++) {
    const wr0 = w[row][0]
    const wr1 = w[row][1]
    const wr2 = w[row][2]
    // (W Σ3) 的第 row 行
    const t0 = wr0 * s00 + wr1 * s01 + wr2 * s02
    const t1 = wr0 * s01 + wr1 * s11 + wr2 * s12
    const t2 = wr0 * s02 + wr1 * s12 + wr2 * s22
    for (let col = 0; col < 3; col++) {
      tmpCov[row][col] = t0 * w[col][0] + t1 * w[col][1] + t2 * w[col][2]
    }
  }
  return tmpCov
}

/**
 * 单位化主轴方向。
 *
 * **退化保护**：各向同性时 `(x, y) = (0, 0)`，JS 里 `|| 1` 只会得到 (0,0) 单位向量，
 * 进而把包围盒缩成一个像素。与 WGSL 侧保持一致，兜底回 (1,0)
 * （正圆时任何单位方向都等价，见 `gsplatCorner.ts` 的同名注释）。
 */
function normalize2(x: number, y: number): { x: number; y: number } {
  const len = Math.hypot(x, y)
  if (len <= 1e-12) return { x: 1, y: 0 }
  return { x: x / len, y: y / len }
}

function linearToSrgbByte(x: number): number {
  const c = x < 0 ? 0 : x > 1 ? 1 : x
  const s = c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055
  return Math.round(Math.min(1, Math.max(0, s)) * 255)
}
