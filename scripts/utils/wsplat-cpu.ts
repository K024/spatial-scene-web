/**
 * CPU 参考光栅器：WGSL 的**数值基准**。
 *
 * 这里只有**数学实现**（可复用）；判定与阈值**不在这里**，全部集中在
 * `scripts/wsplat-golden.ts`（唯一判定入口）——这样只有一个判定点，
 * 又不必把光栅化器复制一份。
 *
 * 严格按 gsplat（`rasterize_mode="antialiased"`）的前向公式，而不是照抄上游
 * playcanvas 的 quad 近似：
 *   1. 从 (四元数, 尺度) 组 3D 协方差 `Σ3 = R S² Rᵀ`
 *   2. 视图旋转 `W`（world -> camera）作用到 Σ3，再用**像素空间**雅可比
 *      `J = [[f/z, 0, -f·x/z²], [0, f/z, -f·y/z²]]`
 *   3. `Σ2 = J W Σ3 Wᵀ Jᵀ`，然后 `Σ2' = Σ2 + eps2d·I`（eps2d = 0.3，单位像素²）
 *   4. `α(p) = opacity · sqrt(det Σ2 / det Σ2') · exp(-½ · dᵀ Σ2'⁻¹ d)`
 *      其中 `d` 是像素中心相对高斯中心的偏移（单位像素）
 *   5. back-to-front（远 -> 近）做 `over` 合成，累积
 *      `A`、`C_premul`、`ED = Σ T·α·z`；`D = ED / A`
 *
 * 与 WGSL 的**有意差异**（已知且可接受）：
 *   - WGSL 走「quad + normExp」：在 ~2.83σ 处把衰减归一化到 0，等价于 gsplat 的
 *     截断核但边缘略软；
 *   - WGSL 的 α 在 `alphaClipForward` 以下被丢弃。
 *   所以两边的差异应集中在高斯的边缘（golden 的 NCC 门就是为这个留的余量）。
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
   * 2D 协方差对角加项，**取上游 WGSL 的值**（默认 0.3）。
   *
   * 注意：WGSL 的 Σ2 建在 `focal = 2·f_px` 上，是本模块 Σ2 的 4 倍，
   * 所以内部会先除以 4 再用（见 `WGSL_COV_SCALE`）。这样 CPU 参考才与
   * GPU 逐条一致 —— 实测该项会让 AA 后的 alphaClip 计数差 682/1.18M。
   */
  eps2d?: number
  /** 低于该 α 直接跳过（对齐上游 `1/255`）。 */
  alphaClip?: number
  /** 椭圆半轴上限（像素），防止个别巨大高斯把 CPU 参考拖死。 */
  maxRadiusPx?: number
  /** 与 WGSL 的 `minPixelSize` 对齐（只用于剔除统计，默认 2）。 */
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
   * 剔除计数（诊断用，与 WGSL 的**同一套判据**）。
   *
   * 判据与顺序都与 `quad.ts` 的 `vsSplat` 逐条对齐（含 λ 的尺度换算），
   * 用来与 GPU 的 `WSplatStats` 交叉验证：两边应当逐项接近（残差只来自
   * 「CPU 用 3σ 包围盒、GPU 用 2.83σ quad」这类边界差异）。
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

const DEFAULT_EPS2D = 0.3
const DEFAULT_ALPHA_CLIP = 1 / 255

/**
 * WGSL 的 2D 协方差相对本模块的**尺度倍数**。
 *
 * `gsplatCorner.ts` 的 `focal = viewport_size.x · projMat00 = 2·f_px`
 * （上游 playcanvas 的写法，见该文件头「量纲自洽」一段），所以它的
 * `Σ2 = J Σv Jᵀ` 是本模块 `(f_px / z)` 版本的 **4 倍**。
 * 推论（实测验证过）：
 *   - WGSL 的 `eps2d = 0.3` 对应本模块尺度上的 `0.075`；
 *   - WGSL 的 `l = 2·min(sqrt(2λ), vmin)` 对应本模块的 `2·min(sqrt(8λ), vmin)`。
 * 不换算就会得到「CPU 预测剔除 73.55% vs GPU 实际 46.7%」这种对不上的数。
 */
const WGSL_COV_SCALE = 4

/** 跑 CPU 参考。 */
export function renderSplatsCpu(options: CpuRenderOptions): CpuRenderResult {
  const { gaussians, camera } = options
  const eps2d = options.eps2d ?? DEFAULT_EPS2D
  const alphaClip = options.alphaClip ?? DEFAULT_ALPHA_CLIP
  const maxRadiusPx = options.maxRadiusPx ?? 512

  const count = gaussians.opacities.length
  const { width, height } = camera
  const px = width * height

  // 与 WGSL 一致的 minPixelSize / vmin（`gsplatCorner.ts`）；仅诊断用
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
    const z = depths[i]
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

    // ── 像素空间雅可比 ──
    const j1x = camera.fx / z
    const j1y = camera.fy / z
    const j2x = (-j1x * vx) / z
    const j2y = (-j1y * vy) / z

    // Σ2 = J Σv Jᵀ（2x2），再 + eps2d·I
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
    if (!(detRaw > 0)) {
      cull.nonPositiveDeterminant++
      continue
    }
    // eps2d 的定义域是 WGSL 的 Σ2（= 本模块的 4 倍），所以要除回去。
    // 证据：upstream playcanvas 在 `focal = viewport_size.x * projMat00`（= 2·f_px）
    // 的协方差上硬编码 + 0.3（vert/gsplatCorner.js:52），而 gsplat 的 0.3 作用在
    // 真像素焦距的协方差上 —— 两者差 (2f/f)² = 4 倍。
    const eps2dLocal = eps2d / WGSL_COV_SCALE
    const b00 = a00 + eps2dLocal
    const b11 = a11 + eps2dLocal
    const detBlur = b00 * b11 - a01 * a01
    if (!(detBlur > 0)) continue

    // ── 剔除：与 WGSL 顶点阶段逐条对齐（顺序也一致，否则计数对不上）──
    //
    // WGSL 的顺序（`quad.ts` 的 vsSplat + `gsplatCorner.ts`）：
    //   原始 opacity <= alphaClip  ->  z <= eps  ->  minPixelSize  ->  视锥
    //   ->  (AA 补偿后) alpha <= alphaClip  ->  绘制
    // 这里的 `alphaClip`（原始 opacity）必须在 minPixelSize **之前**判，
    // 否则同一批高斯在两边会被归到不同的原因里（实测差 ~0.4%）。
    if (gaussians.opacities[i] <= alphaClip) {
      cull.alphaClip++
      continue
    }

    // minPixelSize：按 WGSL 同式（注意 λ 要换算到 WGSL 的尺度，见 WGSL_COV_SCALE）
    {
      const d1 = a00 * WGSL_COV_SCALE + eps2d
      const off2 = a01 * WGSL_COV_SCALE
      const d2 = a11 * WGSL_COV_SCALE + eps2d
      const mid2 = 0.5 * (d1 + d2)
      const rad2 = Math.sqrt(((d1 - d2) / 2) ** 2 + off2 * off2)
      const lam1 = mid2 + rad2
      const lam2 = Math.max(mid2 - rad2, 0.1)
      const L1 = 2 * Math.min(Math.sqrt(2 * lam1), vmin)
      const L2 = 2 * Math.min(Math.sqrt(2 * lam2), vmin)
      if (Math.max(L1, L2) < minPixelSize) {
        cull.minPixelSize++
        continue
      }
    }

    const aaFactor = Math.sqrt(Math.max(detRaw / detBlur, 0))
    const alpha = gaussians.opacities[i] * aaFactor
    if (alpha <= alphaClip) {
      cull.alphaClipAfterAa++
      continue
    }

    // 椭圆范围：3σ 主轴（CPU 自己的核，比 GPU 的 2.83σ quad 略宽，尾部差 ~1% 能量）
    const mid = 0.5 * (b00 + b11)
    const rad = Math.sqrt(max0((b00 - b11) / 2) ** 2 + a01 * a01)
    const lambda1 = mid + rad
    const lambda2 = Math.max(mid - rad, 1e-6)
    const r1 = Math.min(3 * Math.sqrt(lambda1), maxRadiusPx)
    const r2 = Math.min(3 * Math.sqrt(lambda2), maxRadiusPx)

    // 逆协方差（conic）
    const inv = 1 / detBlur
    const c00 = b11 * inv
    const c01 = -a01 * inv
    const c11 = b00 * inv

    // 保守包围盒：主轴方向 + 3σ 的轴向投影
    const diagVec = normalize2(a01, lambda1 - b00)
    const v1x = r1 * diagVec.x
    const v1y = r1 * diagVec.y
    const v2x = -r2 * diagVec.y
    const v2y = r2 * diagVec.x
    const extX = Math.abs(v1x) + Math.abs(v2x)
    const extY = Math.abs(v1y) + Math.abs(v2y)

    const x0 = Math.max(0, Math.floor(u - extX))
    const x1 = Math.min(width - 1, Math.ceil(u + extX))
    const y0 = Math.max(0, Math.floor(v - extY))
    const y1 = Math.min(height - 1, Math.ceil(v + extY))
    if (x1 < x0 || y1 < y0) {
      cull.outsideView++
      continue
    }

    drawn++
    cull.drawn++
    const cr = gaussians.colors[i * 3]
    const cg = gaussians.colors[i * 3 + 1]
    const cb = gaussians.colors[i * 3 + 2]

    for (let y = y0; y <= y1; y++) {
      const dy = y + 0.5 - v
      const rowBase = y * width
      for (let x = x0; x <= x1; x++) {
        const dx = x + 0.5 - u
        const power = c00 * dx * dx + 2 * c01 * dx * dy + c11 * dy * dy
        if (power > 0.5 * 3 * 3 * 2) continue
        const a = alpha * Math.exp(-0.5 * power)
        if (a < alphaClip) continue
        const idx = rowBase + x
        const t = 1 - alphaAcc[idx]
        // 该像素剩余透射率：本 splat 的贡献是 a，按「后画的更近」的 over 规则，
        // 已累积的是更近的层，所以更新为
        //   C = C + (1 - A_existing_partial) ... 见下
        // 直接按 over：dst = src + dst*(1 - src.a)，src = (c*a, a)
        // 但我们是 back-to-front 顺序累加，等价于 dst 是已画的更远层，
        // 故这里写成 C_new = c*a + C_old*(1-a) —— 注意这会让「更远」的层被
        // 之后（更近）的层弱化，方向正确。
        void t
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

function max0(x: number): number {
  return x > 0 ? x : 0
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
