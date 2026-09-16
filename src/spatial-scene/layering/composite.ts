/**
 * 线性空间合成 + 「分层混合 ≡ 整场渲染」的对照工具。
 *
 * ── 恒等式 ──
 * `over` 是结合的，且每层渲染各自 `clear`（`WSplatRenderer.drawLayer` 的语义），
 * 所以对全量高斯的任意硬划分 `{层 k}`（层间远→近、层内 back-to-front）：
 * ```
 * 整场渲染 == over_k(层 k 渲染)          // 顺序：远 -> 近
 * ```
 * 逐像素写开（`C` = **直通**线性 RGB，`A` = α，`ED` = 未归一化累积深度）：
 * ```
 * T_k = Π_{j<k} (1 − A_j)                 // 层 k 之前的透射率（j = 更远的层）
 * C_out = Σ_k T_k · C_k · A_k             // 预乘
 * A_out = 1 − Π_k (1 − A_k)
 * ED_out = Σ_k T_k · ED_k
 * D_out = ED_out / A_out                  // 期望深度（米）
 * ```
 * ⚠ 合成必须在**线性**域（本模块的 `rgb` 就是线性 f32），且深度必须在
 * `(ED, A)` 空间合并：`Σ T·ED / Σ T·A`，**不能**直接平均各层的 `D`。
 *
 * ── 为什么值得单独成文件 ──
 * 这是 layering 的**唯一验收口径**（参考视角数值正确），也是 meshing 前的体检：
 * 合成结果与 `LayeredRgbd.direct` 的差（α / 预乘色 / 深度）直接量化了
 * 「分层是否无损」。f16 附件会引入 ~1e-3 量级的舍入差，阈值由调用方定
 * （`scripts/layering-check.ts`）。
 */

import type { WSplatFrame } from "../wsplat/types.ts"

/** 合成后的逐像素产物（预乘 + α + 深度）。 */
export interface CompositedFrame {
  readonly width: number
  readonly height: number
  /** **预乘**线性 RGB，长度 `w*h*3`。 */
  readonly rgb: Float32Array
  /** 累积 α，长度 `w*h`。 */
  readonly alpha: Float32Array
  /** 未归一化累积深度 `Σ T·α·z`，长度 `w*h`。 */
  readonly accumulatedDepth: Float32Array
  /** 期望深度 `ED / A`（米）；`A == 0` 处为 0。 */
  readonly depth: Float32Array
}

/**
 * 把 L 层帧按 `over` 合成成一张线性 RGBAD。
 *
 * `frames[k]` 层索引 0 = 最近（与 `LayeredRgbd` 一致），内部从最远层开始累加。
 */
export function compositeLayerFrames(
  frames: readonly WSplatFrame[],
  out?: CompositedFrame,
): CompositedFrame {
  if (frames.length === 0) throw new Error("frames 为空")
  const { width, height } = frames[0]
  const pixels = width * height
  const target = out ?? allocateComposited(width, height)
  if (target.width !== width || target.height !== height) {
    throw new Error(
      `输出尺寸 ${target.width}x${target.height} 与帧 ${width}x${height} 不一致`,
    )
  }
  target.rgb.fill(0)
  target.accumulatedDepth.fill(0)

  const { rgb, accumulatedDepth, alpha, depth } = target
  const transmission = new Float32Array(pixels).fill(1)

  // `over` 的迭代方向：**远 -> 近**，每步把当前层压到已有结果**上面**。
  //
  // 展开式是 `P = Σ_k T_k·C_k·A_k`（`T_k = Π_{j<k}(1−A_j)`，j = 比 k 更近的层），
  // 迭代形式就是 `P ← (1−A_k)·P + C_k·A_k`：
  // 已累加的**更远**层整体被当前层的 `(1−A_k)` 衰减，而当前层以权重 1 叠在最上层。
  // （写成 `P += T·C_k·A_k`（T 来自已处理的远层）是 `under`，方向恰好反了：
  //  那样近层会被远层衰减，画面上层序静默倒置。）
  for (let k = frames.length - 1; k >= 0; k--) {
    const frame = frames[k]
    if (frame.width !== width || frame.height !== height) {
      throw new Error(`第 ${k} 层帧尺寸与第 0 层不一致`)
    }
    const fa = frame.alpha
    const frgb = frame.rgb
    const fed = frame.accumulatedDepth
    for (let i = 0; i < pixels; i++) {
      const a = clamp01(fa[i])
      const keep = 1 - a
      transmission[i] *= keep
      rgb[i * 3] = keep * rgb[i * 3] + a * frgb[i * 3]
      rgb[i * 3 + 1] = keep * rgb[i * 3 + 1] + a * frgb[i * 3 + 1]
      rgb[i * 3 + 2] = keep * rgb[i * 3 + 2] + a * frgb[i * 3 + 2]
      accumulatedDepth[i] = keep * accumulatedDepth[i] + fed[i]
    }
  }

  for (let i = 0; i < pixels; i++) {
    const a = 1 - transmission[i]
    alpha[i] = a
    depth[i] = a > 0 ? accumulatedDepth[i] / a : 0
  }
  return target
}

/**
 * 单帧 -> 合成格式（**预乘**），用于与 `compositeLayerFrames` 对照。
 *
 * 整场直接渲染的帧是直通 RGB + α，这里做 `C·A` 预乘，其余字段照抄。
 */
export function singleFrameToComposited(
  frame: WSplatFrame,
  out?: CompositedFrame,
): CompositedFrame {
  const { width, height } = frame
  const pixels = width * height
  const target = out ?? allocateComposited(width, height)
  if (target.width !== width || target.height !== height) {
    throw new Error("输出尺寸与帧不一致")
  }
  for (let i = 0; i < pixels; i++) {
    const a = clamp01(frame.alpha[i])
    target.rgb[i * 3] = frame.rgb[i * 3] * a
    target.rgb[i * 3 + 1] = frame.rgb[i * 3 + 1] * a
    target.rgb[i * 3 + 2] = frame.rgb[i * 3 + 2] * a
    target.alpha[i] = a
    target.accumulatedDepth[i] = frame.accumulatedDepth[i]
    target.depth[i] = a > 0 ? frame.accumulatedDepth[i] / a : 0
  }
  return target
}

function allocateComposited(width: number, height: number): CompositedFrame {
  const pixels = width * height
  return {
    width,
    height,
    rgb: new Float32Array(pixels * 3),
    alpha: new Float32Array(pixels),
    accumulatedDepth: new Float32Array(pixels),
    depth: new Float32Array(pixels),
  }
}

/** 两张合成帧的逐像素差（全是「越小越一致」的量）。 */
export interface CompositedDiff {
  /** `|ΔA|` 的最大值。 */
  readonly alphaMaxAbs: number
  /** `|ΔA|` 的平均值。 */
  readonly alphaMae: number
  /** 预乘线性 RGB 的三通道 MAE。 */
  readonly premultipliedMae: number
  /** 预乘线性 RGB 的逐通道最大绝对差。 */
  readonly premultipliedMaxAbs: number
  /** 两帧都「可见」（`A > minAlpha`）的像素里，深度相对误差的中位数。 */
  readonly depthMedianRel: number
  /** 参与深度统计的像素数。 */
  readonly depthSamples: number
}

/**
 * 对照两张合成帧。
 *
 * 深度相对误差只在 `minAlpha` 以上的像素统计（`A -> 0` 时 `ED/A` 数值不可信），
 * 且用**中位数**而不是均值：边缘像素的深度本来就由 α 加权平均定义，个别像素的
 * 大偏差是语义性的（不同层的混合比例略有不同），不该淹没整体一致性。
 */
export function compareComposited(
  a: CompositedFrame,
  b: CompositedFrame,
  options: { minAlpha?: number } = {},
): CompositedDiff {
  if (a.width !== b.width || a.height !== b.height) {
    throw new Error("两张合成帧尺寸不一致")
  }
  const minAlpha = options.minAlpha ?? 0.5
  const pixels = a.width * a.height
  let alphaSum = 0
  let alphaMax = 0
  let colorSum = 0
  let colorMax = 0
  const relErrors = new Float32Array(pixels)
  let relCount = 0
  for (let i = 0; i < pixels; i++) {
    const da = Math.abs(a.alpha[i] - b.alpha[i])
    alphaSum += da
    if (da > alphaMax) alphaMax = da
    for (let c = 0; c < 3; c++) {
      const dc = Math.abs(a.rgb[i * 3 + c] - b.rgb[i * 3 + c])
      colorSum += dc
      if (dc > colorMax) colorMax = dc
    }
    if (a.alpha[i] > minAlpha && b.alpha[i] > minAlpha) {
      const za = a.depth[i]
      const zb = b.depth[i]
      if (za > 0 && zb > 0) {
        relErrors[relCount++] = Math.abs(za - zb) / Math.max(za, zb)
      }
    }
  }
  // TypedArray.sort 默认按数值升序（不带比较器），比 Array#sort 快一个量级。
  const sorted = relErrors.subarray(0, relCount)
  sorted.sort()
  return {
    alphaMaxAbs: alphaMax,
    alphaMae: alphaSum / pixels,
    premultipliedMae: colorSum / (pixels * 3),
    premultipliedMaxAbs: colorMax,
    depthMedianRel: relCount > 0 ? sorted[relCount >> 1] : 0,
    depthSamples: relCount,
  }
}

function clamp01(value: number): number {
  return value < 0 ? 0 : value > 1 ? 1 : value
}
