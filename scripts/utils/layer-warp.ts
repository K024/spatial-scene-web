/**
 * Novel-view warp：把 L 层 RGBD 投到另一个相机，按层序 `over` 合成。
 *
 * ── 这件事在测什么 ──
 * 分层表示的**外推代价**：层边界放得不对，novel 视角下就会在轮廓处露空洞（gap）
 * 或把前景纹样拖到背景上（bleed）。所以这个 warp 就是 E3 的被测对象。
 *
 * ── 为什么用「逐像素真实深度」而不是「层平面深度」 ──
 * 层的像素带的是 `wsplat` 渲染出来的**逐像素度量深度** `D = ED/A`（不是一个平面）。
 * 用平面深度去 warp 等于测一个 MPI，而本管线的中间表示是「带深度的层」
 * （`MXISceneBuilder.process:layer:face:color:depth:` 收的就是 depth map，
 * `enableDepthTessellation` 也说明网格是跟着 depth 走的）。
 * 所以这里做的是 **DIBR**：层内用真实深度做 z-buffer，层间用 `over`。
 *
 * ── 为什么层间用 `over` 而不是深度测试 ──
 * LDI 的定义就是「层是深度有序的，novel 视角下按层序 `over` 即可」。
 * 层序在 novel 视角下**可能失效**（表面重排），而那正是我们要测的失败模式：
 * 它表现为 gap / bleed，正好是 E3 的两个指标。若在这里加深度测试，就等于
 * 把失败悄悄修掉，指标反而失去判别力。
 *
 * ── 已知的度量下限（不是 bug，但必须知道） ──
 * 这是 CPU 逐像素前向 splat（默认 1 px 足迹），而 GT 是 GPU 带足迹的高斯光栅化。
 * 所以 warp 会有一套**与分辨率/hit 率相关的舍入空洞**。用 `yaw = pitch = 0`
 * 的自恒等 warp 可以把这套下限量出来，再据此判读各档位之间的差异。
 */

import {
  createWSplatCamera,
  type WSplatCamera,
} from "../../src/spatial-scene/wsplat/camera.ts"
import type { WSplatFrame } from "../../src/spatial-scene/wsplat/types.ts"

/** 把参考相机的 view matrix 拆成 `R`(world->camera) 与 `t`（列主序输入）。 */
function decomposeView(m: ArrayLike<number>): {
  /** 行主序 3x3：`r[row*3+col] = R_cw[row][col]`。 */
  r: Float64Array
  t: Float64Array
} {
  const r = new Float64Array(9)
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) {
      // 列主序：m[col*4+row] = R_cw[row][col]
      r[row * 3 + col] = m[col * 4 + row]
    }
  }
  const t = new Float64Array([m[12], m[13], m[14]])
  return { r, t }
}

/**
 * 造一个「绕自身光心**纯旋转**」的 novel 相机。
 *
 * 纯旋转（不平移）是有意的：它正是 `maxViewAngleDeg` / `maxRelativeDisparity`
 * 描述的工作区间，而且它把「视差位移」和「基线平移」分开 —— 分层要解决的
 * 就是视差位移造成的遮挡变化。
 *
 * 旋转在**相机坐标系**里施加：`A' = A · D`（`A` = cam2world，列是 right/down/forward）。
 * 这样光心不动，只有朝向变。
 */
/**
 * 把「图像位移占幅宽/幅高比」换成纯旋转角度（弧度）。
 *
 * ── 为什么必须这样参数化 ──
 * 绕光心旋转 `θ`，深度 `z` 处的点横向移动 `z·tanθ`，投影像素位移 `f·tanθ` ——
 * **与深度无关**。化成占幅宽比就是
 * ```
 * s = f·tanθ / W = tanθ / (2·tan(fov/2))
 * ```
 * 所以 `s` 直接就是「画面整体平移了百分之几的幅宽」，这才是「外推幅度」的物理量。
 *
 * 反例（实测）：裸角度 `yaw = 20°` 在 `fovX = 60°` 下等于 `s = 0.29` ——
 * 近三成幅宽，画面必然大面积截断；而截断在所有 `L` 上一样，
 * 于是把「L 有没有用」的差异全淹没了。`s = 0.05` 才是同一量级的位移。
 */
export function imageShiftToAngleRad(shift: number, fovRad: number): number {
  return Math.atan(2 * shift * Math.tan(fovRad / 2))
}

/** `imageShiftToAngleRad` 的逆：纯旋转角度（弧度）→ 图像位移占幅宽/幅高比。 */
export function angleRadToImageShift(angleRad: number, fovRad: number): number {
  return Math.tan(angleRad) / (2 * Math.tan(fovRad / 2))
}

/**
 * 纯旋转之后，novel 视角里**参考视角看不到**的那部分占整幅的比例（单轴）。
 *
 * NDC 幅宽是 2，位移占宽 `s` ⇒ NDC 位移 `2s`；novel 的 `u` 对应参考的 `u − 2s`，
 * 落在 `[−1,1]` 外就是未观察到，所以比例就是 `s`（`s ≤ 1` 时）。
 *
 * 这个数字必须说出来：它**与分层无关**，是外推题目的固有代价，
 * 判读各档位差异时要先把它扣掉。
 */
export function newlyExposedFraction(shift: number): number {
  return Math.min(1, Math.abs(shift))
}

export function createNovelCamera(
  reference: WSplatCamera,
  yawDeg: number,
  pitchDeg: number,
): WSplatCamera {
  const { r } = decomposeView(reference.viewMatrix)
  // A[row][col] = R_cw[col][row]（见 decomposeView 的索引推导）
  const a = new Float64Array(9)
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) a[row * 3 + col] = r[col * 3 + row]
  }
  const yaw = (yawDeg * Math.PI) / 180
  const pitch = (pitchDeg * Math.PI) / 180
  const cy = Math.cos(yaw)
  const sy = Math.sin(yaw)
  const cp = Math.cos(pitch)
  const sp = Math.sin(pitch)
  // D = Ry(yaw) · Rx(pitch)，坐标轴 = (right, down, forward)
  const d = new Float64Array([
    cy,
    sy * sp,
    sy * cp,
    0,
    cp,
    -sp,
    -sy,
    cy * sp,
    cy * cp,
  ])
  const ap = new Float64Array(9)
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) {
      let sum = 0
      for (let k = 0; k < 3; k++) sum += a[row * 3 + k] * d[k * 3 + col]
      ap[row * 3 + col] = sum
    }
  }
  return createWSplatCamera({
    intrinsics: {
      focalLengthPx: reference.width / (2 * Math.tan(reference.fovX / 2)),
      width: reference.width,
      height: reference.height,
    },
    position: reference.position,
    rotation: [
      [ap[0], ap[1], ap[2]],
      [ap[3], ap[4], ap[5]],
      [ap[6], ap[7], ap[8]],
    ],
    near: reference.near,
    far: reference.far,
  })
}

/** novel 视角的合成产物（与 `WSplatFrame` 同语义，但只带数值通道）。 */
export interface NovelComposite {
  width: number
  height: number
  /** 直通线性 RGB（`A` 处为 0）。 */
  rgb: Float32Array
  alpha: Float32Array
  /** 真实度量深度（米）= `ED / A`。 */
  depth: Float32Array
  accumulatedDepth: Float32Array
  visible: Uint8Array
}

export interface WarpOptions {
  layers: readonly WSplatFrame[]
  reference: WSplatCamera
  novel: WSplatCamera
  /** `alpha <= 阈值` 的参考像素不参与 warp（默认 0）。 */
  alphaThreshold?: number
  /** 每个参考像素在 novel 视角铺几个像素（1 = 最近点，2 = 2x2）。 */
  footprint?: 1 | 2
  /**
   * 层间合成方式。默认 `"opaqueSurface"`。
   *
   * ── 为什么默认不是按层序 `over` ──
   * `over` 会**把 L 的作用抹掉**：合成深度是各层的 alpha 期望 `Σ T·α·z / A`，
   * 而 L=1 的渲染给出的本来就是同一个期望值 —— 于是「多分层」与「不分层」
   * 在指标上不可区分（实测各 L 的 gap/bleed/colorMAE 完全一样）。
   * 分层的价值恰恰在于**同一像素上可以同时存在多个深度不同的表面**。
   *
   * ── `opaqueSurface`（= L 层网格开深度缓冲的语义）──
   * 由近及远累积，颜色/α 与 `over` **完全一致**（预乘 `over` 可结合，本就在参考视角无损），
   * 但同时记录**α 累积首次跨过 `tauAlpha` 的那一层深度** `zSurface` ——
   * 那才是「网格在这里的表面深度」。
   * L=1 时 `zSurface` 只能取到混合期望深度（错），L 越大越接近真实表面深度。
   * 纯「最近者胜」不行：半透明像素会被丢掉，参考视角直接漏 7%+ 的空洞。
   *
   * `"over"` 保留给「层序一定成立」的情形（如自恒等 warp）。
   */
  composite?: "opaqueSurface" | "over"
  out?: NovelComposite
}

/**
 * 逐层 warp 到 novel 相机，再合成为一张 RGBD。
 *
 * 层内用 z-buffer 取最近（一个参考像素可能和别的参考像素投到同一 novel 像素）；
 * 层间按 `composite` 决定：默认深度测试（取最近表面），可选 `over`。
 */
export function warpLayersToNovel(options: WarpOptions): NovelComposite {
  const { layers, reference, novel } = options
  const threshold = options.alphaThreshold ?? 0
  const footprint = options.footprint ?? 1
  const composite = options.composite ?? "opaqueSurface"
  const width = reference.width
  const height = reference.height
  const count = width * height

  const out: NovelComposite = options.out ?? {
    width,
    height,
    rgb: new Float32Array(count * 3),
    alpha: new Float32Array(count),
    depth: new Float32Array(count),
    accumulatedDepth: new Float32Array(count),
    visible: new Uint8Array(count),
  }
  out.rgb.fill(0)
  out.alpha.fill(0)
  out.depth.fill(0)
  out.accumulatedDepth.fill(0)
  out.visible.fill(0)

  // 参考相机：view -> world
  const ref = decomposeView(reference.viewMatrix)
  // novel 相机：world -> view
  const nov = decomposeView(novel.viewMatrix)
  const novelProj = novel.projectionMatrix
  const fx = width / (2 * Math.tan(reference.fovX / 2))
  const fy = height / (2 * Math.tan(reference.fovY / 2))
  const halfW = width / 2
  const halfH = height / 2

  // 单层临时缓冲（复用）
  const layerZ = new Float32Array(count)
  /** `opaqueSurface`：α 首次跨过阈值的那个深度（= 网格在这里的表面深度）。 */
  const zSurface = new Float32Array(count)
  const layerA = new Float32Array(count)
  const layerColor = new Float32Array(count * 3)

  // 近 → 远（`opaqueSurface` 要前向累积才能定位“变不透明的那一层”）；
  // `over` 则必须远 → 近。
  const forward = composite === "opaqueSurface"
  for (let step = 0; step < layers.length; step++) {
    const k = forward ? step : layers.length - 1 - step
    const frame = layers[k]
    layerZ.fill(Number.POSITIVE_INFINITY)
    layerA.fill(0)
    layerColor.fill(0)

    for (let y = 0; y < height; y++) {
      const ndcY = 1 - (2 * (y + 0.5)) / height
      for (let x = 0; x < width; x++) {
        const source = y * width + x
        const a = frame.alpha[source]
        if (a <= threshold) continue
        const d = frame.depth[source]
        if (!(d > 0)) continue

        const ndcX = (2 * (x + 0.5)) / width - 1
        // 参考视图空间点
        const px = (ndcX * d * width) / (2 * fx)
        const py = (-ndcY * d * height) / (2 * fy)
        const pz = d
        // view -> world（R^T (p - t)）
        const vx = px - ref.t[0]
        const vy = py - ref.t[1]
        const vz = pz - ref.t[2]
        const wx = ref.r[0] * vx + ref.r[3] * vy + ref.r[6] * vz
        const wy = ref.r[1] * vx + ref.r[4] * vy + ref.r[7] * vz
        const wz = ref.r[2] * vx + ref.r[5] * vy + ref.r[8] * vz
        // world -> novel view
        const qx = nov.r[0] * wx + nov.r[1] * wy + nov.r[2] * wz + nov.t[0]
        const qy = nov.r[3] * wx + nov.r[4] * wy + nov.r[5] * wz + nov.t[1]
        const qz = nov.r[6] * wx + nov.r[7] * wy + nov.r[8] * wz + nov.t[2]
        if (!(qz > 1e-6)) continue
        // novel 投影（同一套内参 / near-far）
        const cx = (novelProj[0] * qx) / qz
        const cyNdc = (novelProj[5] * qy) / qz
        const u = (cx + 1) * halfW - 0.5
        const v = (1 - cyNdc) * halfH - 0.5

        const r0 = frame.rgb[source * 3] * a
        const g0 = frame.rgb[source * 3 + 1] * a
        const b0 = frame.rgb[source * 3 + 2] * a

        const x0 = Math.round(u)
        const y0 = Math.round(v)
        const span = footprint === 1 ? 1 : 2
        for (let dy = 0; dy < span; dy++) {
          const ty = y0 + dy
          if (ty < 0 || ty >= height) continue
          for (let dx = 0; dx < span; dx++) {
            const tx = x0 + dx
            if (tx < 0 || tx >= width) continue
            const target = ty * width + tx
            if (qz >= layerZ[target]) continue
            layerZ[target] = qz
            layerA[target] = a
            layerColor[target * 3] = r0
            layerColor[target * 3 + 1] = g0
            layerColor[target * 3 + 2] = b0
          }
        }
      }
    }

    if (!forward) {
      // 层间 `over`（远 -> 近：本层盖在已累积的远端之上）
      for (let i = 0; i < count; i++) {
        const a = layerA[i]
        if (a <= 0) continue
        const w = 1 - a
        out.rgb[i * 3] = layerColor[i * 3] + w * out.rgb[i * 3]
        out.rgb[i * 3 + 1] = layerColor[i * 3 + 1] + w * out.rgb[i * 3 + 1]
        out.rgb[i * 3 + 2] = layerColor[i * 3 + 2] + w * out.rgb[i * 3 + 2]
        out.accumulatedDepth[i] = layerZ[i] * a + w * out.accumulatedDepth[i]
        out.alpha[i] = a + w * out.alpha[i]
      }
    } else {
      // 网格语义：由近及远累积（预乘 `over` 不可交换但可结合，所以颜色/α 与
      // 远->近 完全一致），并在 α 首次跨过阈值时钉下表面深度。
      //
      // ⚠ 前向累积的公式是 `C += T·C_k`（把第 k 层放到**已累积栈的后面**），
      //   不是 `C = C_k + T·C`（那是后向累积）。写反了会把层序整体颠倒。
      for (let i = 0; i < count; i++) {
        const a = layerA[i]
        if (a <= 0) continue
        const accumulated = out.alpha[i]
        const transmittance = 1 - accumulated
        const next = accumulated + transmittance * a
        if (accumulated < threshold && next >= threshold) {
          out.depth[i] = layerZ[i]
        }
        out.rgb[i * 3] += transmittance * layerColor[i * 3]
        out.rgb[i * 3 + 1] += transmittance * layerColor[i * 3 + 1]
        out.rgb[i * 3 + 2] += transmittance * layerColor[i * 3 + 2]
        out.accumulatedDepth[i] += transmittance * layerZ[i] * a
        out.alpha[i] = next
      }
    }
  }

  for (let i = 0; i < count; i++) {
    const a = out.alpha[i]
    if (a <= 0) continue
    out.visible[i] = 1
    // 没跨过不透明阈值的像素（很淡的镂空/发丝）没有「表面深度」，退回 alpha 期望
    if (out.depth[i] <= 0) out.depth[i] = out.accumulatedDepth[i] / a
    // 预乘 -> 直通（两种模式的 `rgb` 都是预乘存储）
    out.rgb[i * 3] /= a
    out.rgb[i * 3 + 1] /= a
    out.rgb[i * 3 + 2] /= a
  }
  return out
}

// ────────────────────────── 为什么这里**没有**指标 ──────────────────────────
// 早期版本在这里实现了 novel 视角的标量指标（gapRate / bleedRate / colorMAE /
// fairCoverageMask / compareNovel）。它们已被**删除**，理由：
//
// 1. **没有判别力**：42 个档位（L × method）的标量误差全落在 0.0753–0.0770，
//    而同期目测图显示 L=1/L=2 时近景主体整个不见了、L≥8 才回来。
// 2. **测的是截断不是分层**：20° 偏转在 60° FOV 下丢掉约 1/3–1/2 画面，
//    这个损失在所有 L 上一样，于是把 L 的差异完全淹没。
// 3. **指标口径本身没有稳定定义**：`gapRate` 要不要筛「公平像素」、深度真值该不该用
//    GT 的 alpha 期望（`ED/A`，它天然偏向 L=1）—— 每一版都给出不同结论。
//    一个需要反复重定义才能用的指标，不适合当判据。
// 4. 上游 ml-sharp **仓库内没有评测实现**，论文也明确弃用 PSNR/SSIM
//    （§C.1：1% 平移就让 PSNR 掉到 11.2），它用的 DISTS/LPIPS 是外观指标且本仓库没有。
//
// 所以外推的判断回到**目测**（`scripts/layering-render-views.ts` 出对比图 +
// `scripts/layering-export-layers.ts` 出逐层 PLY 到编辑器里环绕看）。
// **计算指标只保留一件事**：确认分层没有破坏参考视角 —— 那是 `layering-golden.ts` 的 E2，
// 它有一条「阈值级无损」（A 跨 τ=0.5 翻转的像素数为 0）的硬门。
