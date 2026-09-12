/**
 * wsplat 渲染结果的**度量**（渲染脚本与 golden 门共用唯一一份）。
 *
 * 从 `scripts/wsplat-render.ts` 抽出来的原因：要按**区域**报误差，
 * 而且「渲染脚本」与「golden 门」必须用同一套公式，否则两边数字对不上无法复核。
 *
 * ── 为什么必须分区域 ──
 * 全局 MAE 会被"大片背景 + 大片不透明皮肤"稀释。三类典型问题
 * （透明边缘穿帮、发丝闪烁、远景排序错）恰恰都出现在**少数像素**上，
 * 全局指标看不到。所以这里按区域分别报：
 *
 *   | 区域 | 判据 | 想抓的问题 |
 *   |---|---|---|
 *   | `全部可见` | `A > 0` 或参考可见 | 基线 |
 *   | `不透明` | `A > 0.995` | 主体内部（排序错在这里最明显：谁压谁） |
 *   | `半透明边缘` | `0.02 < A < 0.9` | 高斯边缘过曝/发黑（over 方向、AA 补偿） |
 *   | `细节/发丝` | 参考灰度梯度 top 5% | 细结构是否被排序/剔除吃掉 |
 *   | `远景背景` | `T = 1 - A > 0.5` | 远景排序错、雾化过头 |
 *
 * 每区都同时报 **RGB 的 MAE（线性域）** 与 **深度中位相对误差**。
 */

/** 深度统计（GPU vs CPU 参考；两者定义相同）。 */
export interface DepthStats {
  medianRel: number
  meanRel: number
  p95Rel: number
  /** 「最近点 z」（z-buffer 语义）的中位相对误差，仅作对照。 */
  nearestRel: number
  /** D 落在 [zMin, zMax]（自身极值）的占比。 */
  inRange: number
  zMin: number
  zMax: number
  /** 两边 `visible` 判定不一致的像素数（验收项：A→0 处不输出 D）。 */
  visibleMismatch: number
}

/** 单区域的误差汇总。 */
export interface RegionStats {
  name: string
  /** 区域像素数。 */
  count: number
  /** 直通线性 RGB 的平均绝对误差。 */
  mae: number
  /** 深度中位相对误差（两边都可见的像素）。 */
  depthMedianRel: number
}

/** 区域判据（对 GPU 帧与 CPU 参考各取所需字段）。 */
export interface RegionInput {
  /** GPU 直通线性 RGB `w*h*3`。 */
  rgb: Float32Array
  alpha: Float32Array
  depth: Float32Array
  visible: Uint8Array
  /** CPU 参考。 */
  reference: {
    rgb: Float32Array
    depth: Float32Array
    visible: Uint8Array
  }
  width: number
  height: number
}

/**
 * 一帧（一层）的内容摘要。
 *
 * ── 为什么这个比全局 MAE 更接近实际需求 ──
 * 做 `splat -> 多层 RGBAD` 时，调用方关心的是「这一层到底装了什么」：
 *   - `coverage`：有多少像素这层有内容（`A > alphaThreshold`）。为 0 = 这层是空的。
 *   - `depthP1/P50/P99`：这层的深度**分布**。视差分层用的是每像素一个视差，
 *     层内深度跨度越大，视差平移时的误差越大（层的“厚度”= 潜在视差错位）。
 *     调用方据此可以判断自己的切开点是否合理（比如层内 p99/p1 > 2 就该再切）。
 *   - `alphaMass`：Σ A，与层的信息量成正比；用来发现“只剩零星像素”的退化层。
 */
export interface WSplatFrameSummary {
  /** 像素数。 */
  pixels: number
  /** `A > alphaThreshold` 的像素数与其占比。 */
  covered: number
  coverage: number
  /** 覆盖像素上的平均 A。 */
  meanAlpha: number
  /** Σ A（整帧），与层的信息量成正比。 */
  alphaMass: number
  /** 可见像素的深度分位数（米）；无可见像素时为 NaN。 */
  depthP1: number
  depthP50: number
  depthP99: number
  /** 深度最小值 / 最大值（米）。 */
  depthMin: number
  depthMax: number
}

/** 算一帧的内容摘要（纯 CPU，读回之后调）。 */
export function summarizeFrame(
  frame: {
    alpha: Float32Array
    depth: Float32Array
    visible: Uint8Array
  },
  alphaThreshold = 0.5,
): WSplatFrameSummary {
  const pixels = frame.alpha.length
  let covered = 0
  let alphaSum = 0
  let alphaOverThreshold = 0
  const depths: number[] = []
  for (let i = 0; i < pixels; i++) {
    const a = frame.alpha[i]
    alphaSum += a
    if (a > alphaThreshold) {
      covered++
      alphaOverThreshold += a
    }
    if (frame.visible[i] && frame.depth[i] > 0) depths.push(frame.depth[i])
  }
  depths.sort((x, y) => x - y)
  const at = (q: number): number =>
    depths.length === 0
      ? Number.NaN
      : depths[Math.min(depths.length - 1, Math.floor(depths.length * q))]
  return {
    pixels,
    covered,
    coverage: covered / pixels,
    meanAlpha: covered > 0 ? alphaOverThreshold / covered : 0,
    alphaMass: alphaSum,
    depthP1: at(0.01),
    depthP50: at(0.5),
    depthP99: at(0.99),
    depthMin: depths[0] ?? Number.NaN,
    depthMax: depths[depths.length - 1] ?? Number.NaN,
  }
}

/**
 * 把摘要和剔除统计打成人看的行（渲染脚本与调用方都可直接用）。
 */
export function formatFrameSummary(
  label: string,
  summary: WSplatFrameSummary,
  stats?: {
    total: number
    drawn: number
    culledMinPixelSize: number
    culledAlphaClip: number
    culledAlphaClipAfterAa: number
    culledFrustum: number
    culledBehindCamera: number
    culledBounds: number
  },
): string {
  const span = summary.depthP99 / Math.max(summary.depthP1, 1e-9)
  const lines = [
    `[${label}] 覆盖 ${(summary.coverage * 100).toFixed(2)}%  ` +
      `平均A ${summary.meanAlpha.toFixed(4)}  ΣA ${(summary.alphaMass / 1e6).toFixed(2)}M  ` +
      `D p1/p50/p99 = ${summary.depthP1.toFixed(3)}/${summary.depthP50.toFixed(3)}/${summary.depthP99.toFixed(3)} m  ` +
      `层内跨度 p99/p1 = ${span.toFixed(2)}x`,
  ]
  if (stats) {
    lines.push(
      `[${label}] 剔除 minPixelSize ${stats.culledMinPixelSize}  ` +
        `alphaClip ${stats.culledAlphaClip + stats.culledAlphaClipAfterAa}  ` +
        `视锥 ${stats.culledFrustum}  相机后方 ${stats.culledBehindCamera}  ` +
        `越界 ${stats.culledBounds}  → 绘制 ${stats.drawn}/${stats.total}`,
    )
  }
  return lines.join("\n")
}

/** 灰度（`A = 0` 记 0）。 */
export function toGray(
  rgb: Float32Array,
  alpha: Float32Array | Uint8Array,
  count: number,
): Float32Array {
  const out = new Float32Array(count)
  for (let i = 0; i < count; i++) {
    out[i] =
      alpha[i] > 0
        ? 0.299 * rgb[i * 3] + 0.587 * rgb[i * 3 + 1] + 0.114 * rgb[i * 3 + 2]
        : 0
  }
  return out
}

/** 掩码内的归一化互相关（零均值）。 */
export function maskedNcc(
  a: Float32Array,
  b: Float32Array,
  mask: Uint8Array,
): number {
  let n = 0
  let ma = 0
  let mb = 0
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i]) continue
    ma += a[i]
    mb += b[i]
    n++
  }
  if (n === 0) return Number.NaN
  ma /= n
  mb /= n
  let num = 0
  let da = 0
  let db = 0
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i]) continue
    const x = a[i] - ma
    const y = b[i] - mb
    num += x * y
    da += x * x
    db += y * y
  }
  return num / Math.sqrt(Math.max(da * db, 1e-12))
}

/** 掩码内的 RGB 平均绝对误差（浮点域）。 */
export function maskedMae(
  a: Float32Array,
  b: Float32Array,
  mask: Uint8Array,
): number {
  let sum = 0
  let n = 0
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i]) continue
    sum +=
      (Math.abs(a[i * 3] - b[i * 3]) +
        Math.abs(a[i * 3 + 1] - b[i * 3 + 1]) +
        Math.abs(a[i * 3 + 2] - b[i * 3 + 2])) /
      3
    n++
  }
  return n > 0 ? sum / n : Number.NaN
}

/** RGBA8 -> RGB8（`frame.preview` 是 4 通道，对原图比要 3 通道）。 */
export function rgba8ToRgb8(rgba: Uint8Array, count: number): Uint8Array {
  const out = new Uint8Array(count * 3)
  for (let i = 0; i < count; i++) {
    out[i * 3] = rgba[i * 4]
    out[i * 3 + 1] = rgba[i * 4 + 1]
    out[i * 3 + 2] = rgba[i * 4 + 2]
  }
  return out
}

/** 掩码内的 RGB8 平均绝对误差（0-255 标度，用来对原图）。 */
export function maskedMae8(
  a: Uint8Array,
  b: Uint8Array,
  mask: Uint8Array,
): number {
  let sum = 0
  let n = 0
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i]) continue
    sum +=
      (Math.abs(a[i * 3] - b[i * 3]) +
        Math.abs(a[i * 3 + 1] - b[i * 3 + 1]) +
        Math.abs(a[i * 3 + 2] - b[i * 3 + 2])) /
      3
    n++
  }
  return n > 0 ? sum / n : Number.NaN
}

/** 灰度 8bit 版的掩码 NCC（对原图用）。 */
export function maskedNcc8(
  a: Uint8Array,
  b: Uint8Array,
  mask: Uint8Array,
): number {
  const n = mask.length
  const ga = new Float32Array(n)
  const gb = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    ga[i] = 0.299 * a[i * 3] + 0.587 * a[i * 3 + 1] + 0.114 * a[i * 3 + 2]
    gb[i] = 0.299 * b[i * 3] + 0.587 * b[i * 3 + 1] + 0.114 * b[i * 3 + 2]
  }
  return maskedNcc(ga, gb, mask)
}

/** 对齐掩码 = 两边任一可见。 */
export function buildMask(
  frame: { visible: Uint8Array },
  reference: { visible: Uint8Array },
): Uint8Array {
  const mask = new Uint8Array(frame.visible.length)
  for (let i = 0; i < mask.length; i++) {
    mask[i] = frame.visible[i] || reference.visible[i] ? 1 : 0
  }
  return mask
}

/**
 * 深度统计（深度门）。
 *
 * 只在「两边 `visible` 判定一致且都可用」的像素上算相对误差；
 * `visible` 不一致的像素单独计数（深度验收项之一）。
 */
export function depthStats(
  frame: {
    depth: Float32Array
    visible: Uint8Array
  },
  reference: {
    depth: Float32Array
    nearestDepth: Float32Array
    visible: Uint8Array
  },
  count: number,
): DepthStats {
  const rel: number[] = []
  const nearestRel: number[] = []
  const values: number[] = []
  let visibleMismatch = 0
  for (let i = 0; i < count; i++) {
    if (frame.visible[i] !== reference.visible[i]) {
      visibleMismatch++
      continue
    }
    if (!frame.visible[i] || !reference.visible[i]) continue
    const a = frame.depth[i]
    const b = reference.depth[i]
    if (!(b > 0)) continue
    rel.push(Math.abs(a - b) / b)
    if (reference.nearestDepth[i] > 0) {
      nearestRel.push(
        Math.abs(a - reference.nearestDepth[i]) / reference.nearestDepth[i],
      )
    }
    values.push(a)
  }
  rel.sort((x, y) => x - y)
  nearestRel.sort((x, y) => x - y)
  values.sort((x, y) => x - y)
  const zMin = values[0] ?? Number.NaN
  const zMax = values[values.length - 1] ?? Number.NaN
  let inRange = 0
  for (const v of values) if (v >= zMin && v <= zMax) inRange++
  return {
    medianRel: med(rel),
    meanRel: rel.reduce((a, b) => a + b, 0) / Math.max(rel.length, 1),
    p95Rel: rel.length === 0 ? Number.NaN : rel[Math.floor(rel.length * 0.95)],
    nearestRel: med(nearestRel),
    inRange: values.length === 0 ? 0 : inRange / values.length,
    zMin,
    zMax,
    visibleMismatch,
  }
}

/** 按区域分别算 RGB MAE 与深度中位相对误差。 */
export function regionStats(input: RegionInput): RegionStats[] {
  const { width, height, alpha, depth, visible } = input
  const count = width * height
  const shapes: { name: string; mask: Uint8Array }[] = [
    { name: "全部可见", mask: buildMask(input, input.reference) },
    { name: "不透明", mask: threshold(alpha, count, (a) => a > 0.995) },
    {
      name: "半透明边缘",
      mask: threshold(alpha, count, (a) => a > 0.02 && a < 0.9),
    },
    { name: "细节/发丝", mask: detailMask(input) },
    { name: "远景背景", mask: threshold(alpha, count, (a) => 1 - a > 0.5) },
  ]

  // 深度相对误差的分子/分母按区域收集（每区独立排序取中位）
  return shapes.map(({ name, mask }) => {
    const rel: number[] = []
    let n = 0
    for (let i = 0; i < count; i++) {
      if (!mask[i]) continue
      n++
      if (
        visible[i] &&
        input.reference.visible[i] &&
        input.reference.depth[i] > 0
      ) {
        rel.push(
          Math.abs(depth[i] - input.reference.depth[i]) /
            input.reference.depth[i],
        )
      }
    }
    rel.sort((x, y) => x - y)
    return {
      name,
      count: n,
      mae: maskedMae(input.rgb, input.reference.rgb, mask),
      depthMedianRel: med(rel),
    }
  })
}

/**
 * 「细节」掩码：参考灰度**梯度幅值**的前 5%。
 *
 * 用参考（CPU）而不是 GPU 的梯度，是为了让掩码与被测对象无关——
 * 否则「GPU 把细节糊掉了」会同时缩小掩码，把问题掩盖掉。
 */
function detailMask(input: RegionInput): Uint8Array {
  const { reference, width, height } = input
  const count = width * height
  const gray = toGray(reference.rgb, reference.visible, count)
  const mag = new Float32Array(count)
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x
      // 中心差分（Sobel 的简化），跨越不可见像素时取 0 贡献
      const gx =
        (gray[i + 1] - gray[i - 1]) *
        (reference.visible[i + 1] && reference.visible[i - 1] ? 1 : 0)
      const gy =
        (gray[i + width] - gray[i - width]) *
        (reference.visible[i + width] && reference.visible[i - width] ? 1 : 0)
      mag[i] = Math.hypot(gx, gy)
    }
  }
  const sorted = Array.from(mag).sort((a, b) => a - b)
  const cut = sorted[Math.floor(sorted.length * 0.95)] ?? 0
  const mask = new Uint8Array(count)
  for (let i = 0; i < count; i++) {
    mask[i] = mag[i] >= cut && mag[i] > 0 ? 1 : 0
  }
  return mask
}

function threshold(
  values: Float32Array,
  count: number,
  pred: (v: number) => boolean,
): Uint8Array {
  const mask = new Uint8Array(count)
  for (let i = 0; i < count; i++) mask[i] = pred(values[i]) ? 1 : 0
  return mask
}

function med(arr: number[]): number {
  return arr.length === 0 ? Number.NaN : arr[Math.floor(arr.length / 2)]
}

/**
 * 按整数倍 box 平均把 RGB8 降到 `rw x rh`（对原图用）。
 *
 * 与渲染同样是「像素网格对齐」的降采样，不做插值，避免引入额外的模糊
 * 影响 MAE 判定。
 */
export function boxDownsampleRgb8(
  src: Uint8Array,
  channels: number,
  W: number,
  H: number,
  rw: number,
  rh: number,
): Uint8Array {
  const fx = W / rw
  const fy = H / rh
  const dst = new Uint8Array(rw * rh * 3)
  for (let y = 0; y < rh; y++) {
    const y0 = Math.floor(y * fy)
    const y1 = Math.min(H, Math.max(y0 + 1, Math.floor((y + 1) * fy)))
    for (let x = 0; x < rw; x++) {
      const x0 = Math.floor(x * fx)
      const x1 = Math.min(W, Math.max(x0 + 1, Math.floor((x + 1) * fx)))
      let r = 0
      let g = 0
      let b = 0
      let n = 0
      for (let sy = y0; sy < y1; sy++) {
        for (let sx = x0; sx < x1; sx++) {
          const o = (sy * W + sx) * channels
          r += src[o]
          g += src[o + 1]
          b += src[o + 2]
          n++
        }
      }
      const d = (y * rw + x) * 3
      dst[d] = Math.round(r / n)
      dst[d + 1] = Math.round(g / n)
      dst[d + 2] = Math.round(b / n)
    }
  }
  return dst
}
