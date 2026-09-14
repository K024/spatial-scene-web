/**
 * E5 · 原图回写（`refineImage`）的实测入口。
 *
 * ── 本工具验证的判断 ──
 * 回写**不需要额外的 wsplat 渲染趟**：层 RGBAD 就是从原图那台参考相机渲出来的，
 * 所以「层像素 ↔ 原图像素」是**恒等 + 分辨率缩放**，权重三因子也全在 RGBAD 里
 * （`alpha` / `depth` 的局部梯度 / 层号 + 遮挡）。本工具只做：
 *   1. 渲 L 层 RGBAD（现成通路）；
 *   2. 把原图重采样到层分辨率、线性化；
 *   3. 算每层权重图 `w`（`computeRefineWeight`）并回写颜色（`blendImageWriteback`）；
 *   4. 比「回写前 / 回写后」的参考视角颜色 vs **原图**（NCC / MAE）+ 出图。
 *
 * ── 关键约束 ──
 * - 回写**只改颜色**，不改几何 ⇒ 不破坏验收 A；但参考视角颜色会变，所以 E2 分两栏。
 * - 回写**只在参考视角成立**：外推会露贴图感，所以默认只回写前景层。
 * - 必须用**相机与被摄图一致**的 fixture：`example.ply`（有 EXIF）。
 *   `pier.ply` 的 fov 是猜的，层↔原图会错位 —— 别用它判 E5。
 *
 * 用法：
 *   npx tsx scripts/layering-refine.ts --ply py-models/out/ply/example.ply --layers 8 --scope-scan
 *   npx tsx scripts/layering-refine.ts --image py-models/clones/ml-depth-pro/data/example.jpg --fg-layers 2
 *   npx tsx scripts/layering-refine.ts --ply ... --crop 4,6,542,363 --image .../teaser.jpg
 */

import { mkdirSync, writeFileSync } from "node:fs"
import { resolve } from "node:path"
import { parseArgs } from "node:util"

import {
  buildLayerPermutation,
  permuteNdcDepths,
} from "../src/spatial-scene/layering/bands.ts"
import { computeDisparityStats } from "../src/spatial-scene/layering/disparity-stats.ts"
import { computeLayerPlacement } from "../src/spatial-scene/layering/placement.ts"
import {
  blendImageWriteback,
  computeRefineWeight,
  resampleImageToLinear,
} from "../src/spatial-scene/layering/refine.ts"
import type { LayerSamplingMethod } from "../src/spatial-scene/layering/types.ts"
import { createWSplatRenderer } from "../src/spatial-scene/wsplat/index.ts"
import {
  computeViewDepths,
  sortSplatsBackToFront,
} from "../src/spatial-scene/wsplat/sort.ts"
import type { WSplatFrame } from "../src/spatial-scene/wsplat/types.ts"
import { DEFAULT_IMAGE, REPO_ROOT } from "./utils/common.ts"
import { createSharpCanvas, loadImage, rgbaToPngBuffer } from "./utils/image.ts"
import { withNodeDevice } from "./utils/webgpu.ts"
import { loadWSplatScene } from "./utils/wsplat-scene.ts"

const CLI = {
  width: { type: "string" },
  layers: { type: "string" },
  method: { type: "string" },
  ply: { type: "string" },
  /** 原图（生成 splat 的那张）。默认 `ml-depth-pro/data/example.jpg`。 */
  image: { type: "string" },
  /** 原图的裁切 `x,y,w,h`（与 `sharp-infer --crop` 一致）。 */
  crop: { type: "string" },
  /** 输出目录（默认 `py-models/out/layering-refine`）。 */
  out: { type: "string" },
  /** 回写层范围（层号 **<** 该值才回写；`0` = 关闭，`L` = 全部）。 */
  "fg-layers": { type: "string" },
  /** 扫一遍层范围 `{0,1,2,4,L}` 并打表（渲染只做一次）。 */
  "scope-scan": { type: "boolean" },
  /** 平滑因子尺度（相对深度）。调大 = 更多像素通过平滑门。 */
  "smooth-scale": { type: "string" },
} as const

interface Layer {
  rgb: Float32Array
  alpha: Float32Array
}

/** 层栈 back-to-front `over` 合成，返回**直通**线性色 + 总 alpha。 */
function compositeStraight(
  layers: readonly Layer[],
  pixels: number,
): { rgb: Float32Array; alpha: Float32Array } {
  const acc = new Float32Array(pixels * 3)
  const alpha = new Float32Array(pixels)
  for (let k = layers.length - 1; k >= 0; k--) {
    const layer = layers[k]
    for (let i = 0; i < pixels; i++) {
      const a = layer.alpha[i]
      const o = i * 3
      acc[o] = layer.rgb[o] * a + acc[o] * (1 - a)
      acc[o + 1] = layer.rgb[o + 1] * a + acc[o + 1] * (1 - a)
      acc[o + 2] = layer.rgb[o + 2] * a + acc[o + 2] * (1 - a)
      alpha[i] = a + alpha[i] * (1 - a)
    }
  }
  const rgb = new Float32Array(pixels * 3)
  for (let i = 0; i < pixels; i++) {
    const a = alpha[i]
    if (a <= 0) continue
    const o = i * 3
    rgb[o] = acc[o] / a
    rgb[o + 1] = acc[o + 1] / a
    rgb[o + 2] = acc[o + 2] / a
  }
  return { rgb, alpha }
}

/** 线性明亮度。 */
function luma(r: number, g: number, b: number): number {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b
}

/** 参考视角颜色指标：对**覆盖区**（`alpha > 0.5`）算亮度 NCC 与 sRGB MAE(/255)。 */
function colorMetrics(
  comp: { rgb: Float32Array; alpha: Float32Array },
  imageLinear: Float32Array,
  pixels: number,
): { ncc: number; mae: number; covered: number } {
  let n = 0
  let sumA = 0
  let sumB = 0
  let sumAA = 0
  let sumBB = 0
  let sumAB = 0
  let sumAbs = 0
  for (let i = 0; i < pixels; i++) {
    if (comp.alpha[i] <= 0.5) continue
    const o = i * 3
    const la = luma(comp.rgb[o], comp.rgb[o + 1], comp.rgb[o + 2])
    const lb = luma(imageLinear[o], imageLinear[o + 1], imageLinear[o + 2])
    sumA += la
    sumB += lb
    sumAA += la * la
    sumBB += lb * lb
    sumAB += la * lb
    for (let c = 0; c < 3; c++) {
      sumAbs += Math.abs(
        linearToSrgb(comp.rgb[o + c]) - linearToSrgb(imageLinear[o + c]),
      )
    }
    n++
  }
  if (n === 0) return { ncc: 0, mae: 0, covered: 0 }
  const cov = (sumAB - (sumA * sumB) / n) / n
  const varA = sumAA / n - (sumA / n) ** 2
  const varB = sumBB / n - (sumB / n) ** 2
  const denom = Math.sqrt(varA * varB)
  return {
    ncc: denom > 0 ? cov / denom : 1,
    mae: sumAbs / (n * 3),
    covered: n,
  }
}

/** 线性色 + alpha -> sRGB RGBA8（alpha=0 处置黑）。 */
function toSrgbRgba(
  rgb: Float32Array,
  alpha: Float32Array,
  pixels: number,
): Uint8Array {
  const out = new Uint8Array(pixels * 4)
  for (let i = 0; i < pixels; i++) {
    const a = alpha[i]
    const o = i * 3
    for (let c = 0; c < 3; c++) {
      out[i * 4 + c] = Math.round(
        clamp01(linearToSrgb(a > 0 ? rgb[o + c] : 0)) * 255,
      )
    }
    out[i * 4 + 3] = Math.round(clamp01(a) * 255)
  }
  return out
}

/** 权重图 -> 灰度 RGBA8（黑 = 0，白 = 1）。 */
function weightRgba(weight: Float32Array): Uint8Array {
  const out = new Uint8Array(weight.length * 4)
  for (let i = 0; i < weight.length; i++) {
    const v = Math.round(clamp01(weight[i]) * 255)
    out[i * 4] = v
    out[i * 4 + 1] = v
    out[i * 4 + 2] = v
    out[i * 4 + 3] = 255
  }
  return out
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v
}
function linearToSrgb(x: number): number {
  const c = clamp01(x)
  return c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055
}
function parseCrop(
  v: string | undefined,
): [number, number, number, number] | undefined {
  if (!v) return undefined
  const parts = v.split(",").map(Number)
  if (parts.length !== 4 || parts.some((p) => !Number.isFinite(p))) {
    throw new Error(`--crop 需要 4 个整数 x,y,w,h，收到 ${v}`)
  }
  return [parts[0], parts[1], parts[2], parts[3]]
}

/** 横排拼图（列标签画在底部）。 */
async function writeMontage(
  panels: readonly { label: string; rgba: Uint8Array }[],
  width: number,
  height: number,
  outPath: string,
): Promise<void> {
  const labelHeight = 18
  const buffers = await Promise.all(
    panels.map((p) => rgbaToPngBuffer(p.rgba, width, height)),
  )
  const canvas = createSharpCanvas(
    width * panels.length,
    height,
    [0, 0, 0, 255],
  )
  const composites: { input: Buffer; left: number; top: number }[] = []
  for (let i = 0; i < buffers.length; i++) {
    composites.push({ input: buffers[i], left: i * width, top: 0 })
    const svg = `<svg width="${width}" height="${labelHeight}" xmlns="http://www.w3.org/2000/svg"><rect width="${width}" height="${labelHeight}" fill="black" fill-opacity="0.65"/><text x="4" y="13" font-family="monospace" font-size="12" fill="#7CFC00">${escapeXml(panels[i].label)}</text></svg>`
    composites.push({
      input: Buffer.from(svg),
      left: i * width,
      top: height - labelHeight,
    })
  }
  await canvas.composite(composites).png().toFile(outPath)
}

function escapeXml(s: string): string {
  return s.replace(/[<>&]/g, (c) =>
    c === "<" ? "&lt;" : c === ">" ? "&gt;" : "&amp;",
  )
}

async function main(): Promise<void> {
  const args = parseArgs({
    args: process.argv.slice(2),
    options: CLI,
    allowPositionals: false,
    allowNegative: true,
    strict: true,
  })
  const width = Number(args.values.width ?? 768)
  const method = (args.values.method ?? "quantile") as LayerSamplingMethod
  const fgCap = args.values["fg-layers"]
  const scopeScan = args.values["scope-scan"] === true
  const smoothScale = Number(args.values["smooth-scale"] ?? 0.02)
  const ls = (args.values.layers ?? "8")
    .split(",")
    .map((s) => Number(s.trim()))
    .filter((v) => Number.isInteger(v) && v > 0)
  const outDir = resolve(
    REPO_ROOT,
    args.values.out ?? "py-models/out/layering-refine",
  )
  mkdirSync(outDir, { recursive: true })

  const imagePath = args.values.image ?? DEFAULT_IMAGE
  const crop = parseCrop(args.values.crop)

  await withNodeDevice(async (device) => {
    const scene = loadWSplatScene({
      width: String(width),
      plyPath: args.values.ply,
    })
    const { height } = scene
    const pixels = width * height
    const count = scene.gaussians.opacities.length
    const renderer = await createWSplatRenderer(device, {
      size: { width, height },
    })
    renderer.setGaussians(scene.gaussians)
    renderer.setCamera(scene.camera)

    const depth = computeViewDepths(
      scene.gaussians.meanVectors,
      scene.camera.viewMatrix,
      count,
    )
    const order = sortSplatsBackToFront(depth, count)
    const sortedNdc = permuteNdcDepths(depth, order, scene.near, scene.far)
    const stats = computeDisparityStats(
      depth,
      { near: scene.near, far: scene.far, binCount: 256 },
      scene.gaussians.opacities,
    )

    // 原图：只做「重采样到层分辨率 + 线性化」。相机与输入图一致 ⇒ 恒等 + 缩放。
    const loaded = await loadImage(imagePath, undefined, crop)
    const imageLinear = resampleImageToLinear(loaded.image, width, height)
    console.log(
      `PLY ${(scene.plyPath ?? "").replace(/^.*[\\/]/, "")}  ${width}x${height}  ` +
        `原图 ${loaded.image.width}x${loaded.image.height}` +
        `${loaded.focalFromExif ? "" : "（无 EXIF，fov 为 30mm 默认值）"}`,
    )

    for (const L of ls) {
      const placement = computeLayerPlacement(stats, {
        L,
        method,
        near: scene.near,
        far: scene.far,
      })
      const perm = buildLayerPermutation(order, sortedNdc, placement.boundaries)
      renderer.setSplatOrder(perm.permutation)
      const frames: WSplatFrame[] = []
      for (let k = 0; k < L; k++) {
        renderer.drawLayer(perm.table[k])
        frames.push(await renderer.readback())
      }
      renderer.renderSplats()

      // 遮挡：每像素只回写**最前的不透明层**（参考视角的可见性）。
      const occluded: Uint8Array[] = Array.from(
        { length: L },
        () => new Uint8Array(pixels),
      )
      const front = new Uint8Array(pixels)
      for (let k = 0; k < L; k++) {
        occluded[k].set(front)
        const a = frames[k].alpha
        for (let i = 0; i < pixels; i++) if (a[i] > 0.5) front[i] = 1
      }

      const layers: Layer[] = frames.map((f) => ({
        rgb: f.rgb,
        alpha: f.alpha,
      }))
      const before = compositeStraight(layers, pixels)
      const mBefore = colorMetrics(before, imageLinear, pixels)

      console.log(`\n  L=${L}  method=${method}  smooth-scale=${smoothScale}`)
      console.log(
        `    回写前 vs 原图：亮度 NCC ${mBefore.ncc.toFixed(4)}  sRGB MAE ${mBefore.mae.toFixed(4)}（覆盖 ${mBefore.covered}px）`,
      )

      const caps = scopeScan
        ? [...new Set([0, 1, 2, 4, L])].sort((a, b) => a - b)
        : [fgCap !== undefined ? Number(fgCap) : L]
      console.log("    回写层范围          回写后 NCC   MAE      提升(MAE)")
      let best = { cap: caps[0], ncc: 0, mae: Infinity }
      for (const cap of caps) {
        const weights = frames.map((f, k) =>
          computeRefineWeight(f, {
            foreground: k < cap,
            smoothnessScale: smoothScale,
            occluded: occluded[k],
          }),
        )
        const refined = frames.map((f, k) => ({
          rgb: blendImageWriteback(f.rgb, imageLinear, weights[k]),
          alpha: f.alpha,
        }))
        const after = compositeStraight(refined, pixels)
        const m = colorMetrics(after, imageLinear, pixels)
        console.log(
          `    < ${String(cap).padStart(2)} 层（最前 ${cap} 层）   ` +
            `${m.ncc.toFixed(4)}    ${m.mae.toFixed(4)}   ` +
            `${((mBefore.mae - m.mae) * 100).toFixed(2)}%`,
        )
        if (m.mae < best.mae) best = { cap, ncc: m.ncc, mae: m.mae }
      }

      // 出图：原图 | 回写前 | 回写后（配置的 cap，默认全部）
      const writeCap =
        fgCap !== undefined ? Number(fgCap) : scopeScan ? best.cap : L
      const weightForOut = frames.map((f, k) =>
        computeRefineWeight(f, {
          foreground: k < writeCap,
          smoothnessScale: smoothScale,
          occluded: occluded[k],
        }),
      )
      const refinedOut = frames.map((f, k) => ({
        rgb: blendImageWriteback(f.rgb, imageLinear, weightForOut[k]),
        alpha: f.alpha,
      }))
      const afterOut = compositeStraight(refinedOut, pixels)
      const imageSrgb = toSrgbRgba(
        imageLinear,
        new Float32Array(pixels).fill(1),
        pixels,
      )
      const panels = [
        { label: "原图", rgba: imageSrgb },
        { label: "回写前", rgba: toSrgbRgba(before.rgb, before.alpha, pixels) },
        {
          label: `回写后 fg<${writeCap}`,
          rgba: toSrgbRgba(afterOut.rgb, afterOut.alpha, pixels),
        },
      ]
      await writeMontage(
        panels,
        width,
        height,
        resolve(outDir, `L${L}_before-after.png`),
      )
      await writeMontage(
        Array.from({ length: L }, (_, k) => ({
          label: `w L${k}`,
          rgba: weightRgba(weightForOut[k]),
        })),
        width,
        height,
        resolve(outDir, `L${L}_weights.png`),
      )
      console.log(
        `    出图 -> ${resolve(outDir, `L${L}_before-after.png`)}（最佳 layer 范围 = < ${best.cap}）`,
      )
    }
    renderer.destroy()
  })
}

main().catch((err) => {
  console.error("\n[错误]", err instanceof Error ? err.message : err)
  if (err instanceof Error && err.stack) console.error(err.stack)
  process.exitCode = 1
})
