/**
 * 原图回写（refineImage）—— **只改层颜色，不改几何**。
 *
 * 阶段：**layering 的最后一个后处理**（在 `repair` 之后，交付 `LayeredRGBD` 之前）。
 *
 * ── 为什么在 layering 而不是 mesh ──
 * 回写的产物是**层颜色本身**；mesh 只消费层颜色，不该反向改它。
 * 它同时用到「层投影」与「原图投影」的映射（对应 Apple 的
 * `GetPixelTransformationFromMPIToImage` / `ComputeRefinementWeight`）。
 *
 * ── 数据从哪来（**不需要额外的 splat 渲染趟**）──
 * 层 RGBAD 就是从**原图那台参考相机**渲出来的，所以逐项都能在现成数据里找到：
 *
 * | 回写需要 | 来源 |
 * |---|---|
 * | 层颜色（线性直通） | `WSplatFrame.rgb` |
 * | 层覆盖 / 是否干净表面 | `WSplatFrame.alpha` / `.visible` / `.transmission` |
 * | 逐像素深度 D | `WSplatFrame.depth` |
 * | 层像素 ↔ 原图像素映射 | 参考相机 == 原图相机 ⇒ **恒等 + 分辨率缩放**（见下） |
 * | 前景判定 | 调用方给的层号 / 视差 |
 * | 遮挡（只回写每像素最前的层） | 对层栈的 `alpha` 做一次 over（本模块外，脚本里） |
 *
 * ── 映射为什么是恒等 + 缩放 ──
 * `camera.json` 的 fx/fy/size 与推理输入图（含 `--crop`）**逐位一致**
 *（`scripts/sharp-check-camera.ts` 在校验这件事）。层像素 (x,y) 与它所在的高斯
 * 都由同一台相机投影，所以把层像素对应的 3D 点再用同一台相机投回去仍是 (x,y)；
 * 唯一差别是**分辨率**。`GetPixelTransformationFromMPIToImage` 那套深度重投影
 * 只在「层用另一台虚拟相机 / 平面参数化」时才非平凡。因此这里只需要按分辨率缩放
 * 采样原图（脚本用双线性重采样），不需要相机内参。
 *
 * ── 三条权重因子（相乘，可调）──
 * 1. **在层内是否为最近表面**：`alpha` 越接近 1（透射率越接近 0）权重越高 ——
 *    半透明的像素不是「干净表面」，回写会把原图色贴到悬浮的雾上。
 * 2. **深度是否局部平滑**：`|∇D| / D` 越大权重越低 —— 跨轮廓处回写会糊边。
 * 3. **是否在前景 / 未被遮挡**：由调用方给（层号小、或逐像素遮挡掩码）。
 * 三条相乘后夹到 `[0,1]`。
 *
 * ── 边界（必须记住）──
 * - 回写**只在参考视角成立**：外推时那块颜色会露出「贴图感」。实测 `example.ply`
 *   @768/L=8（参考视角亮度 NCC / sRGB MAE）：不回写 `0.9438 / 0.0253`；最前 1 层
 *   `0.9482 / 0.0219`；最前 4 层 `0.9493 / 0.0186`；**全部 8 层 `0.9924 / 0.0051`**。
 *   ⚠ “全部层回写”几乎就是把原图贴回去，参考视角指标**平凡最好**，不能当判据；
 *   而且 MAE 被**背景大层**主导，对「前景细节是否改善」不敏感。所以默认只回写
 *   **最前景的可见层**，结论必须由外推（warp 回写前/后 vs novel GT）决定。
 * - 它**不改几何**，所以不破坏验收 A 的几何无损；但参考视角颜色会从
 *   「splat 近似」变成「原图」，E2 必须分「几何合成 / 颜色合成」两栏报。
 */

import type { SourceImage } from "../sharp/preprocess.ts"
import type { WSplatFrame } from "../wsplat/types.ts"
import type { LayeredRGBD } from "./types.ts"

/** 回写权重的三个可调因子。 */
export interface RefineWeightOptions {
  /**
   * 覆盖率下限（默认 `0.5`）。`alpha < minAlpha` ⇒ 权重 0；
   * 其余从 `minAlpha` 线性升到 1。这是「最近表面」因子的硬门。
   */
  minAlpha?: number
  /** 深度平滑尺度（**相对**深度）：`|∇D| / D` 超过它则权重快速下降（默认 `0.02`）。 */
  smoothnessScale?: number
  /** 平滑因子的指数（默认 `2`）。 */
  smoothnessExponent?: number
  /** 该层是否是前景层（默认 `true`）。`false` ⇒ 整层权重 0（背景层不回写）。 */
  foreground?: boolean
  /**
   * 逐像素**遮挡**掩码（长度 = 像素数，`1` = 被更近的层不透明覆盖）。
   * 给了就把该像素权重置 0 —— 参考视角下每像素只应回写**最前的不透明层**。
   */
  occluded?: Uint8Array
}

/**
 * 算一层每像素的回写权重 `w ∈ [0,1]`（纯函数，不改输入）。
 *
 * `frame` 只需借 `depth` / `alpha` / `visible` / `width` / `height` 五个字段，
 * 便于单测直接造最小对象。
 */
export function computeRefineWeight(
  frame: Pick<WSplatFrame, "depth" | "alpha" | "visible" | "width" | "height">,
  options: RefineWeightOptions = {},
  out: Float32Array = new Float32Array(frame.alpha.length),
): Float32Array {
  const { depth, alpha, visible, width, height } = frame
  const minAlpha = options.minAlpha ?? 0.5
  const smoothScale = options.smoothnessScale ?? 0.02
  const smoothExp = options.smoothnessExponent ?? 2
  const foreground = options.foreground ?? true
  const occluded = options.occluded
  out.fill(0)
  if (!foreground) return out
  const alphaGain = minAlpha < 1 ? 1 / (1 - minAlpha) : 1

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x
      if (!visible[i] || alpha[i] < minAlpha) continue
      if (occluded?.[i]) continue
      const z = depth[i]
      if (!(z > 0)) continue
      // 中心差分；任一邻域无效则该方向记 0（不借背景的 D=0 造假梯度）
      let gx = 0
      let gy = 0
      if (x > 0 && x + 1 < width && visible[i - 1] && visible[i + 1]) {
        gx = (depth[i + 1] - depth[i - 1]) * 0.5
      }
      if (y > 0 && y + 1 < height && visible[i - width] && visible[i + width]) {
        gy = (depth[i + width] - depth[i - width]) * 0.5
      }
      const relativeGradient = Math.hypot(gx, gy) / z
      const wSurface = Math.min(1, (alpha[i] - minAlpha) * alphaGain)
      const wSmooth = 1 / (1 + (relativeGradient / smoothScale) ** smoothExp)
      out[i] = wSurface * wSmooth
    }
  }
  return out
}

/**
 * 把原图颜色按权重混进层颜色：`c ← (1−w)·c_splat + w·c_image`。
 *
 * 两端都是**直通线性 RGB**（不是 premultiplied），与 `WSplatFrame.rgb` 一致。
 * `w = 0` 时逐位等于层颜色；`w = 1` 时逐位等于原图色（方便当单测）。
 */
export function blendImageWriteback(
  rgb: Float32Array,
  imageRgb: Float32Array,
  weight: Float32Array,
  out: Float32Array = new Float32Array(rgb.length),
): Float32Array {
  const pixels = weight.length
  for (let i = 0; i < pixels; i++) {
    const w = weight[i]
    const o = i * 3
    for (let c = 0; c < 3; c++) {
      out[o + c] = rgb[o + c] * (1 - w) + imageRgb[o + c] * w
    }
  }
  return out
}

/**
 * 层像素 -> 原图像素（浮点坐标，给小采样器用）。
 *
 * 参考相机 == 原图相机 ⇒ 映射是**恒等 + 分辨率缩放**（见文件头）。
 * 返回值已加 `+0.5 / −0.5` 的像素中心约定，方便直接双线性插值。
 */
export function layerPixelToImagePixel(
  x: number,
  y: number,
  layerWidth: number,
  layerHeight: number,
  imageWidth: number,
  imageHeight: number,
): { x: number; y: number } {
  return {
    x: ((x + 0.5) * imageWidth) / layerWidth - 0.5,
    y: ((y + 0.5) * imageHeight) / layerHeight - 0.5,
  }
}

/** 整摞回写的旋钮 = 逐层权重因子 + 回写层范围。 */
export interface RefineLayersOptions extends RefineWeightOptions {
  /**
   * 回写层范围：层号 `< foregroundLayers` 才回写（`0` = 关闭，缺省 = `L` = 全部）。
   *
   * ⚠ 逐像素还有 `occluded` 遮挡门，所以「全部层」的实际效果仍是
   * 「每个像素只在它**最前的不透明层**回写一次」。
   */
  foregroundLayers?: number
}

/** 逐像素「已被更近的不透明层覆盖」的判定阈值（与回写权重的 `minAlpha` 默认值一致）。 */
const OPAQUE_ALPHA = 0.5

/**
 * 原图（sRGB uint8 HWC）双线性重采样到层分辨率 + 线性化，返回直通线性 RGB `[w*h*3]`。
 *
 * 参考相机 == 原图相机 ⇒ 映射是恒等 + 缩放，所以只需要分辨率重采样，不需要内参。
 * 双线性对越界索引做 clamp（与 `sharp/preprocess.ts` 的 `F.interpolate` 语义一致）。
 */
export function resampleImageToLinear(
  src: SourceImage,
  width: number,
  height: number,
): Float32Array {
  const out = new Float32Array(width * height * 3)
  const { data, width: sw, height: sh, channels: ch } = src
  for (let y = 0; y < height; y++) {
    const fy = ((y + 0.5) * sh) / height - 0.5
    const y0 = Math.min(sh - 1, Math.max(0, Math.floor(fy)))
    const y1 = Math.min(sh - 1, y0 + 1)
    const ty = Math.min(1, Math.max(0, fy - y0))
    for (let x = 0; x < width; x++) {
      const fx = ((x + 0.5) * sw) / width - 0.5
      const x0 = Math.min(sw - 1, Math.max(0, Math.floor(fx)))
      const x1 = Math.min(sw - 1, x0 + 1)
      const tx = Math.min(1, Math.max(0, fx - x0))
      const o = (y * width + x) * 3
      for (let c = 0; c < 3; c++) {
        const p00 = data[(y0 * sw + x0) * ch + c]
        const p10 = data[(y0 * sw + x1) * ch + c]
        const p01 = data[(y1 * sw + x0) * ch + c]
        const p11 = data[(y1 * sw + x1) * ch + c]
        const top = p00 + (p10 - p00) * tx
        const bot = p01 + (p11 - p01) * tx
        out[o + c] = srgbToLinear((top + (bot - top) * ty) / 255)
      }
    }
  }
  return out
}

/**
 * 对整摞层做原图回写，返回**新的一摞**（除 `frames[].rgb` 外全部复用）。
 *
 * 流程：原图重采样到层分辨率 -> 算逐层遮挡掩码 -> 逐层算权重并混合颜色。
 * 不改几何（`alpha` / `depth` / `transmission` / `visible` 原样保留）。
 *
 * ⚠ 回写只在参考视角成立：外推时那块颜色会露「贴图感」。是否开启由调用方决定。
 */
export function refineLayers(
  layered: LayeredRGBD,
  image: SourceImage,
  options: RefineLayersOptions = {},
): LayeredRGBD {
  const L = layered.frames.length
  if (L === 0) return layered
  const { width, height } = layered
  const pixels = width * height
  const imageLinear = resampleImageToLinear(image, width, height)

  // 遮挡：每像素只回写**最前的不透明层**（参考视角的可见性）。
  // `occluded[k]` = 「更近的层里已经有 α > 阈值 的像素」的掩码。
  const occluded: Uint8Array[] = Array.from(
    { length: L },
    () => new Uint8Array(pixels),
  )
  const front = new Uint8Array(pixels)
  for (let k = 0; k < L; k++) {
    occluded[k].set(front)
    const a = layered.frames[k].alpha
    for (let i = 0; i < pixels; i++) if (a[i] > OPAQUE_ALPHA) front[i] = 1
  }

  const foregroundLayers = options.foregroundLayers ?? L
  const frames: WSplatFrame[] = layered.frames.map((frame, k) => {
    const weight = computeRefineWeight(frame, {
      minAlpha: options.minAlpha,
      smoothnessScale: options.smoothnessScale,
      smoothnessExponent: options.smoothnessExponent,
      foreground: k < foregroundLayers,
      occluded: occluded[k],
    })
    const rgb = blendImageWriteback(frame.rgb, imageLinear, weight)
    // WSplatFrame 是 readonly：用替换引用造新帧，其余字段原样复用。
    return { ...frame, rgb }
  })

  return { ...layered, frames }
}

function srgbToLinear(x: number): number {
  const c = x < 0 ? 0 : x > 1 ? 1 : x
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
}
