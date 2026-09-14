/**
 * E3 的**目测**入口：把 novel 视角的外推结果写成 PNG，用眼睛判。
 *
 * ── 为什么不用标量指标 ──
 * 1. 外推的本质是「看到之前看不到的东西」，而点对点指标（PSNR/SSIM）对**微小空间
 *    错位**极度敏感 —— ml-sharp 论文 §C.1 实测：1% 平移就让 PSNR 掉到 11.2、SSIM 0.375，
 *    接近「和均值图比」的水平，所以论文**明确弃用** PSNR/SSIM，改用在意的 DISTS/LPIPS。
 * 2. 但 DISTS/LPIPS 是图像感知指标，衡量的是**外观**，不是「分层有没有用」；
 *    而且本仓库没有它们的实现（ml-sharp 仓库内**没有评测代码**，只有论文描述）。
 * 3. 实测过：在现有 fixture 上 42 个档位（L × method）的自研标量误差全落在
 *    0.0753–0.0770，完全没有判别力：截断区域在所有档位上同样缺失，而近景主体消失
 *    表现为**内容错位**、只占几个百分点像素，被平均掉。
 * 所以判据回到**目测**，脚本只负责把该看的图摆出来。
 *
 * ── 看什么 ──
 * 每个角度出一张横排拼图（列已在图上标注）：
 *   `GT`（同角度直接渲染整个 splat 场）
 *   → `L=1/2/4/8/16/32`（各档位的分层合成）
 *   → `diff×4`（|合成 − GT| 放大 4 倍；黑 = 一致）
 *
 * ── 外推幅度用「幅宽占比」，不用裸角度 ──
 * `--shift 0.1,0` 表示「画面整体平移 10% 幅宽」对应的纯旋转。
 * 裸角度是错的参数化：`yaw = 20°` 在 `fovX = 60°` 下等于 `s = 0.29`，
 * 近三成幅宽，画面必然大面积截断，而截断在所有 `L` 上一样，
 * 于是把 L 的差异全淹没了（实测结论）。换算见 `layer-warp.ts`。
 *
 * 判读要点（按重要性）：
 * 1. **轮廓处**：前景边缘是否出现「被拉长的糊边」（= 该切层没切，前景深度铺到了背景上）；
 * 2. **新露出区域**（右侧 `新增可见 ≈ s` 那条）：`L=1` 在新露出处应露空洞或错误内容，
 *    L 增大后应被正确填充；**这片区域与分层无关**，判读时要先把它扣掉；
 * 3. **diff 列**：数值大但眼睛看不出的地方，通常是小错位（论文说这类不重要），
 *    要看**结构性的成片差异**，而不是零散亮点。
 *
 * 用法：
 *   npx tsx scripts/layering-render-views.ts                        # 默认 4 档位移 × L=1..32
 *   npx tsx scripts/layering-render-views.ts --shift 0,0;0.1,0;0.2,0 --width 512
 *   npx tsx scripts/layering-render-views.ts --angles 0,0;20,0     # 裸角度（仅对照用）
 *   npx tsx scripts/layering-render-views.ts --ply py-models/out/ply/pier.ply
 *   npx tsx scripts/layering-render-views.ts --composite over       # 对照另一种合成语义
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
import type { LayerSamplingMethod } from "../src/spatial-scene/layering/types.ts"
import { createWSplatRenderer } from "../src/spatial-scene/wsplat/index.ts"
import {
  computeViewDepths,
  sortSplatsBackToFront,
} from "../src/spatial-scene/wsplat/sort.ts"
import type { WSplatFrame } from "../src/spatial-scene/wsplat/types.ts"
import { REPO_ROOT } from "./utils/common.ts"
import {
  createSharpCanvas,
  rgbaToPngBuffer,
  sharpFromRgba,
} from "./utils/image.ts"
import {
  angleRadToImageShift,
  createNovelCamera,
  imageShiftToAngleRad,
  type NovelComposite,
  newlyExposedFraction,
  warpLayersToNovel,
} from "./utils/layer-warp.ts"
import { withNodeDevice } from "./utils/webgpu.ts"
import { loadWSplatScene } from "./utils/wsplat-scene.ts"

const CLI = {
  width: { type: "string" },
  layers: { type: "string" },
  method: { type: "string" },
  shift: { type: "string" },
  angles: { type: "string" },
  ply: { type: "string" },
  composite: { type: "string" },
  out: { type: "string" },
  scale: { type: "string" },
  gain: { type: "string" },
} as const

/** 一次外推：同时带「幅宽占比」与换算出的度数，两边都打印以免看不懂图。 */
interface ViewSpec {
  yawShift: number
  pitchShift: number
  yawDeg: number
  pitchDeg: number
}

/**
 * 默认外推幅度（幅宽/幅高占比）。
 *
 * `0.20` 对比 `0.05`：前者已是「五分之一幅宽」，接近 SHARP 声明的工作区间边缘
 *（论文 §D.6：基线 < 0.5 m）；后者是安全区。两档摆在一起才能看出「L 从哪一档开始有用」。
 * `0.05,0.05` 两者同时偏移，看对角线方向的填充。
 */
const DEFAULT_SHIFTS = "0,0;0.05,0;0.10,0;0.20,0;0.05,0.05"

async function main(): Promise<void> {
  const args = parseArgs({
    args: process.argv.slice(2),
    options: CLI,
    allowPositionals: false,
    allowNegative: true,
    strict: true,
  })
  const width = Number(args.values.width ?? 384)
  const layers = (args.values.layers ?? "1,2,4,8,16,32")
    .split(",")
    .map((s) => Number(s.trim()))
    .filter((n) => Number.isInteger(n) && n > 0)
  const method = (args.values.method ?? "quantile") as LayerSamplingMethod
  const composite = args.values.composite === "over" ? "over" : "opaqueSurface"
  const scale = Math.max(1, Number(args.values.scale ?? 1))
  // diff 的显示增益：1 = 原始差；大了看得见但容易一片白
  const diffGain = Math.max(0.5, Number(args.values.gain ?? 1))
  // 两种参数化二选一：`--shift`（幅宽占比，默认）或 `--angles`（裸角度，只用做对照）
  const rawAngles = args.values.angles ? parseAngles(args.values.angles) : null
  const shifts = rawAngles
    ? null
    : parseShift(args.values.shift ?? DEFAULT_SHIFTS)
  const outDir = resolve(REPO_ROOT, args.values.out ?? "temp/layers-views")
  mkdirSync(outDir, { recursive: true })

  console.log("=".repeat(88))
  console.log("E3 目测：novel 视角外推对比图（不判定，只出图）")
  console.log("=".repeat(88))

  await withNodeDevice(async (device) => {
    const scene = loadWSplatScene({
      width: String(width),
      plyPath: args.values.ply,
    })
    const { height } = scene
    // 幅宽占比 <-> 角度：必须在拿到 fov 之后换。坐标系见 layer-warp.ts。
    const toAngle = (s: number, fov: number): number =>
      (imageShiftToAngleRad(s, fov) * 180) / Math.PI
    const toShift = (deg: number, fov: number): number =>
      angleRadToImageShift((deg * Math.PI) / 180, fov)
    const specs: ViewSpec[] = shifts
      ? shifts.map((s) => ({
          yawShift: s.yaw,
          pitchShift: s.pitch,
          yawDeg: toAngle(s.yaw, scene.camera.fovX),
          pitchDeg: toAngle(s.pitch, scene.camera.fovY),
        }))
      : (rawAngles ?? []).map((a) => ({
          yawShift: toShift(a.yaw, scene.camera.fovX),
          pitchShift: toShift(a.pitch, scene.camera.fovY),
          yawDeg: a.yaw,
          pitchDeg: a.pitch,
        }))

    console.log(
      `PLY ${scene.plyPath.replace(/^.*[\\/]/, "")}  ` +
        `渲染 ${width}x${height}  倍率 ${scale}x  method ${method}  合成 ${composite}\n` +
        `视场 ${((scene.camera.fovX * 180) / Math.PI).toFixed(2)}°(横) / ` +
        `${((scene.camera.fovY * 180) / Math.PI).toFixed(2)}°(竖)  ` +
        `L ∈ {${layers.join(",")}}`,
    )
    console.log(
      `  外推幅度（幅宽占比 -> 角度 -> 新增可见区域，后者与分层无关，判读时先扣掉）:`,
    )
    for (const s of specs) {
      console.log(
        `    yaw ${(s.yawShift * 100).toFixed(1).padStart(5)}% = ${s.yawDeg.toFixed(2).padStart(6)}°` +
          `   pitch ${(s.pitchShift * 100).toFixed(1).padStart(5)}% = ${s.pitchDeg.toFixed(2).padStart(6)}°` +
          `   新增可见 ≈ 横 ${(newlyExposedFraction(s.yawShift) * 100).toFixed(1)}%` +
          ` / 纵 ${(newlyExposedFraction(s.pitchShift) * 100).toFixed(1)}%`,
      )
    }

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

    for (const spec of specs) {
      const camera =
        spec.yawDeg === 0 && spec.pitchDeg === 0
          ? scene.camera
          : createNovelCamera(scene.camera, spec.yawDeg, spec.pitchDeg)
      // GT：同一 novel 相机下直接渲染整个 splat 场
      renderer.setCamera(camera)
      renderer.sort()
      renderer.renderSplats()
      const gt = await renderer.readback()

      const panels: { label: string; rgba: Uint8Array }[] = [
        { label: "GT (splat)", rgba: gt.preview },
      ]
      for (const L of layers) {
        const placement = computeLayerPlacement(stats, {
          L,
          method,
          near: scene.near,
          far: scene.far,
        })
        const perm = buildLayerPermutation(
          order,
          sortedNdc,
          placement.boundaries,
        )
        renderer.setSplatOrder(perm.permutation)
        const frames: WSplatFrame[] = []
        for (let k = 0; k < L; k++) {
          renderer.drawLayer(perm.table[k])
          frames.push(await renderer.readback())
        }
        const cc = warpLayersToNovel({
          layers: frames,
          reference: scene.camera,
          novel: camera,
          composite,
        })
        panels.push({
          label: `L=${L}`,
          rgba: compositeToSrgbRgba(cc, width, height),
        })
        if (L === layers[Math.floor(layers.length / 2)]) {
          panels.push({
            label: `diff x${diffGain}`,
            rgba: diffRgba(cc, gt, width, height, diffGain),
          })
        }
      }

      const tag = `s${spec.yawShift.toFixed(2)}_${spec.pitchShift.toFixed(2)}`
      const montagePath = resolve(outDir, `montage_${tag}.png`)
      await writeMontage(panels, width, height, scale, montagePath)
      // 同时把 GT 与每档单独存一份，方便放大了看
      for (const p of panels) {
        const name = p.label.replace(/[^A-Za-z0-9]+/g, "_")
        writeFileSync(
          resolve(outDir, `${tag}_${name}.png`),
          await rgbaToPngBuffer(p.rgba, width, height),
        )
      }
      console.log(
        `  ${tag}: ${panels.map((p) => p.label).join(" | ")}\n` +
          `      -> ${montagePath}`,
      )
    }
    renderer.destroy()
  })
}

/** novel 合成（线性 f32）-> sRGB RGBA8，供 PNG 使用。 */
function compositeToSrgbRgba(
  c: NovelComposite,
  width: number,
  height: number,
): Uint8Array {
  const out = new Uint8Array(width * height * 4)
  for (let i = 0; i < width * height; i++) {
    const a = c.alpha[i]
    for (let ch = 0; ch < 3; ch++) {
      const linear = a > 0 ? c.rgb[i * 3 + ch] : 0
      out[i * 4 + ch] = Math.round(clamp01(linearToSrgb(linear)) * 255)
    }
    out[i * 4 + 3] = Math.round(clamp01(a) * 255)
  }
  return out
}

/** |合成 − GT| 的灰度图，`gain` 倍放大；黑 = 一致。 */
function diffRgba(
  c: NovelComposite,
  gt: WSplatFrame,
  width: number,
  height: number,
  gain: number,
): Uint8Array {
  const out = new Uint8Array(width * height * 4)
  for (let i = 0; i < width * height; i++) {
    let d = 0
    for (let ch = 0; ch < 3; ch++) {
      const mine = c.alpha[i] > 0 ? linearToSrgb(c.rgb[i * 3 + ch]) : 0
      const theirs = gt.rgb[i * 3 + ch]
      d = Math.max(d, Math.abs(mine - theirs))
    }
    const v = Math.round(clamp01(d * gain) * 255)
    out[i * 4] = v
    out[i * 4 + 1] = v
    out[i * 4 + 2] = v
    out[i * 4 + 3] = 255
  }
  return out
}

async function png(
  rgba: Uint8Array,
  width: number,
  height: number,
): Promise<Buffer> {
  return rgbaToPngBuffer(rgba, width, height)
}

/**
 * 横排拼图 + 每列顶部一条标签。
 *
 * 标签用 SVG 叠上去，而不是让人对着 console 猜列序 —— 目测工具的第一要求是
 * 「打开就能看懂」，否则很容易看错列得出相反结论。
 */
async function writeMontage(
  panels: readonly { label: string; rgba: Uint8Array }[],
  width: number,
  height: number,
  scale: number,
  outPath: string,
): Promise<void> {
  const labelHeight = 18
  const buffers = await Promise.all(
    panels.map(async (p) => await png(p.rgba, width, height)),
  )
  const totalWidth = width * panels.length
  const canvas = createSharpCanvas(totalWidth, height, [0, 0, 0, 255])
  const composites: { input: Buffer; left: number; top: number }[] = []
  for (let i = 0; i < buffers.length; i++) {
    composites.push({ input: buffers[i], left: i * width, top: 0 })
    const svg = `<svg width="${width}" height="${labelHeight}" xmlns="http://www.w3.org/2000/svg">
      <rect width="${width}" height="${labelHeight}" fill="black" fill-opacity="0.65"/>
      <text x="4" y="13" font-family="monospace" font-size="12" fill="#7CFC00">${escapeXml(panels[i].label)}</text>
    </svg>`
    composites.push({
      input: Buffer.from(svg),
      left: i * width,
      top: height - labelHeight,
    })
  }
  await canvas
    .composite(composites)
    .resize({
      width: totalWidth * scale,
      height: height * scale,
      kernel: "nearest",
    })
    .png()
    .toFile(outPath)
}

function escapeXml(s: string): string {
  return s.replace(/[<>&]/g, (c) =>
    c === "<" ? "&lt;" : c === ">" ? "&gt;" : "&amp;",
  )
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v
}

/**
 * 线性 -> sRGB（单值）。
 *
 * 与 `sharp/colorspace.ts` 的分段式**同式**；那边是逐元素数组接口（为推理批量准备），
 * 这里是逐像素导出，所以用一个单值版本，避免为了一个像素分配数组。
 */
function linearToSrgb(x: number): number {
  const c = clamp01(x)
  return c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055
}

function parseAngles(text: string): { yaw: number; pitch: number }[] {
  const list: { yaw: number; pitch: number }[] = []
  for (const chunk of text.split(";")) {
    const part = chunk.trim()
    if (!part) continue
    const [yaw, pitch] = part.split(",").map((s) => Number(s.trim()))
    if (!Number.isFinite(yaw)) continue
    list.push({ yaw, pitch: Number.isFinite(pitch) ? pitch : 0 })
  }
  return list.length > 0 ? list : [{ yaw: 0, pitch: 0 }]
}

/** `--shift` 解析：分号分隔的 `yaw,pitch`，值为**幅宽/幅高占比**（如 `0.1,-0.05`）。 */
function parseShift(text: string): { yaw: number; pitch: number }[] {
  const list: { yaw: number; pitch: number }[] = []
  for (const chunk of text.split(";")) {
    const part = chunk.trim()
    if (!part) continue
    const [yaw, pitch] = part.split(",").map((s) => Number(s.trim()))
    if (!Number.isFinite(yaw)) continue
    list.push({ yaw, pitch: Number.isFinite(pitch) ? pitch : 0 })
  }
  return list.length > 0 ? list : [{ yaw: 0, pitch: 0 }]
}

main().catch((err) => {
  console.error("\n[错误]", err instanceof Error ? err.message : err)
  if (err instanceof Error && err.stack) console.error(err.stack)
  process.exitCode = 1
})
