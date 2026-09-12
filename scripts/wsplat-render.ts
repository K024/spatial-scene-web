/**
 * 渲染脚本：从 PLY + 相机 json 渲染一帧 RGB+A+D，出图给人看。
 *
 * 阶段：**wsplat 渲染器**（`scripts/wsplat-*`：渲染 / 自检 / 质量门）。
 *
 * 链路：PLY -> `Gaussians3D`（linearRGB）-> `createWSplatGpuData` -> GPU 光栅化
 * -> resolve -> 读回 -> 与 CPU 参考的 NCC / MAE / 深度误差 -> 写 PNG。
 *
 * ── 本脚本**不做判定** ──
 * 门限与 PASS/FAIL 全部集中在 `scripts/wsplat-golden.ts`。这里只负责把数字、
 * 分区域误差、深度/视差/透射率图、与原图对比以及并排图打出来。
 * 两边共用 `utils/wsplat-scene.ts`（装配）与 `utils/wsplat-metrics.ts`（度量），
 * 所以打印出的数字与门里用的完全一致。
 *
 * ── 颜色空间（易错，先看这里）──
 * PLY 里的 `f_dc_*` 已被 `save_ply` 转成 **sRGB**（`gaussians.py` 注释直证），
 * 而 `Gaussians3D.colors` 的语义是 **linearRGB**。所以这里必须做
 * `sRGB(srgb) -> linear`，否则渲染结果会偏亮、且与 CPU 参考对不上。
 *
 * 用法：
 *   npx tsx scripts/wsplat-render.ts                      # 默认 example.ply + 原分辨率
 *   npx tsx scripts/wsplat-render.ts --width 768          # 降分辨率快速迭代
 *   npx tsx scripts/wsplat-render.ts --width internal     # SHARP 内部域 1536
 *   npx tsx scripts/wsplat-render.ts --no-sort --no-cpu    # 只出图
 */

import { mkdirSync } from "node:fs"
import { createRequire } from "node:module"
import { basename, resolve } from "node:path"
import { parseArgs } from "node:util"
import { createWSplatRenderer } from "../src/spatial-scene/wsplat/index.ts"
import { DEFAULT_IMAGE, REPO_ROOT } from "./utils/common.ts"
import {
  alphaToGrayPng,
  depthToGrayPng,
  disparityToColorPng,
  estimateDepthRange,
  transmissionToGrayPng,
} from "./utils/depth-viz.ts"
import { loadImage } from "./utils/image.ts"
import { loadSharpDepths, resampleLayer } from "./utils/sharp-depth.ts"
import { withNodeDevice } from "./utils/webgpu.ts"
import { renderSplatsCpu } from "./utils/wsplat-cpu.ts"
import {
  boxDownsampleRgb8,
  buildMask,
  depthStats as computeDepthStats,
  formatFrameSummary,
  toGray as grayFromRgb,
  maskedMae,
  maskedMae8,
  maskedNcc,
  maskedNcc8,
  regionStats,
  rgba8ToRgb8,
  summarizeFrame,
} from "./utils/wsplat-metrics.ts"
import { loadWSplatScene } from "./utils/wsplat-scene.ts"

/** SH degree-0 基函数值（与 ml-sharp 的 `convert_rgb_to_spherical_harmonics` 互逆）。 */
const SH_C0 = 0.28209479177387814

const CLI_OPTIONS = {
  ply: { type: "string" },
  camera: { type: "string" },
  width: { type: "string" },
  out: { type: "string" },
  "no-sort": { type: "boolean" },
  "no-cpu": { type: "boolean" },
  "compare-aa": { type: "boolean" },
  "max-splats": { type: "string" },
  eps2d: { type: "string" },
  image: { type: "string" },
  "no-image": { type: "boolean" },
} as const

async function main(): Promise<void> {
  const args = parseArgs({
    args: process.argv.slice(2),
    options: CLI_OPTIONS,
    allowPositionals: false,
    strict: true,
  })

  const shouldSort = !args.values["no-sort"]
  const runCpu = !args.values["no-cpu"]
  const compareAa = args.values["compare-aa"] === true
  const maxSplats = args.values["max-splats"]
    ? Number.parseInt(args.values["max-splats"], 10)
    : Number.POSITIVE_INFINITY
  const eps2d = Number.parseFloat(args.values.eps2d ?? "0.3")
  const compareImage = args.values["no-image"] !== true

  console.log("=".repeat(72))
  console.log("wsplat 渲染（出图；判定见 wsplat-golden.ts）")
  console.log("=".repeat(72))

  // ── 1-2. 数据 + 相机（与 golden 门共用同一套装配，见 utils/wsplat-scene.ts）──
  const scene = loadWSplatScene({
    plyPath: args.values.ply,
    cameraPath: args.values.camera,
    width: args.values.width ?? "native",
    maxSplats,
  })
  const { plyPath, cameraPath, gaussians, pose, width, height, scale, camera } =
    scene
  const { p01: zP01, p99: zP99 } = scene.depthRange
  console.log(
    `camera : ${cameraPath} (fx=${pose.fx}, ${pose.width}x${pose.height})`,
  )
  console.log(
    `渲染   : ${width}x${height}  fx=${(pose.fx * scale).toFixed(2)}  fovY=${((camera.fovY * 180) / Math.PI).toFixed(2)}°`,
  )
  console.log(
    `深度范围: z p01=${zP01.toFixed(3)} p99=${zP99.toFixed(3)} -> near=${scene.near.toFixed(3)} far=${scene.far.toFixed(3)}`,
  )

  // ── 3. GPU ──
  const { frame, noAaFrame, stats } = await withNodeDevice(async (device) => {
    const render = async (antialias: boolean) => {
      const renderer = await createWSplatRenderer(device, {
        size: { width, height },
        antialias,
      })
      const t0 = performance.now()
      renderer.setGaussians(gaussians)
      const t1 = performance.now()
      renderer.setCamera(camera)
      if (shouldSort) renderer.sort()
      const t2 = performance.now()
      renderer.renderSplats()
      const result = await renderer.readback()
      const t3 = performance.now()
      // 剔除统计（可观测性）：单独一趟 compute，用与顶点阶段相同的 cull 函数
      renderer.countCulls()
      const stats = await renderer.readSplatStats()
      console.log(
        `[耗时${antialias ? "" : " 无AA"}] 上传 ${(t1 - t0).toFixed(1)}ms  排序 ${shouldSort ? (t2 - t1).toFixed(1) : "跳过"}ms  渲染+读回 ${(t3 - t2).toFixed(1)}ms`,
      )
      renderer.destroy()
      return { result, stats }
    }
    const withAa = await render(true)
    const withoutAa = compareAa ? await render(false) : undefined
    return {
      frame: withAa.result,
      stats: withAa.stats,
      noAaFrame: withoutAa?.result,
    }
  })

  // ── 4. 覆盖率 ──
  let covered = 0
  for (let i = 0; i < frame.visible.length; i++) covered += frame.visible[i]
  console.log(
    `[覆盖率] A>0 的像素 ${covered}/${width * height} = ${((covered / (width * height)) * 100).toFixed(2)}%`,
  )
  // 这一层"装了什么"：覆盖 / ΣA / 深度分布。
  // 层内深度跨度 = 视差平移时的潜在错位量，是分层切点的直接判据。
  console.log(formatFrameSummary("本层", summarizeFrame(frame), stats))

  // ── 5. PNG ──
  const outDir = resolve(
    REPO_ROOT,
    args.values.out ?? "py-models/out/wsplat-check",
  )
  mkdirSync(outDir, { recursive: true })
  const stem = basename(plyPath).replace(/\.ply$/i, "")
  const gpuPath = resolve(outDir, `${stem}_wsplat.png`)
  await sharpFromRgba(frame.preview, width, height).png().toFile(gpuPath)
  console.log(`[png] GPU  ${gpuPath}`)

  // ── 6. 深度 / 视差 / 透射率：各自一张独立 PNG（不叠在颜色上）──
  const zRange = estimateDepthRange(frame.depth, frame.visible)
  const depthGray = depthToGrayPng(frame.depth, frame.visible, zRange)
  const dispColor = disparityToColorPng(frame.depth, frame.visible, zRange)
  const transGray = transmissionToGrayPng(frame.transmission)
  const alphaGray = alphaToGrayPng(frame.alpha)
  const rangeTag = `z${zRange.min.toFixed(2)}-${zRange.max.toFixed(2)}m`
  const depthPath = resolve(outDir, `${stem}_depth_${rangeTag}.png`)
  const dispPath = resolve(outDir, `${stem}_disparity_${rangeTag}.png`)
  const transPath = resolve(outDir, `${stem}_transmission.png`)
  const alphaPath = resolve(outDir, `${stem}_alpha.png`)
  await sharpFromRgba(depthGray, width, height).png().toFile(depthPath)
  await sharpFromRgba(dispColor, width, height).png().toFile(dispPath)
  await sharpFromRgba(transGray, width, height).png().toFile(transPath)
  await sharpFromRgba(alphaGray, width, height).png().toFile(alphaPath)
  console.log(
    `[png] 深度（近=白） ${depthPath}   标定 D ∈ [${zRange.min.toFixed(3)}, ${zRange.max.toFixed(3)}] m（可见像素 1%/99% 分位）`,
  )
  console.log(`[png] 视差（逆深度彩色） ${dispPath}`)
  console.log(`[png] 透射率 T ${transPath}`)
  console.log(`[png] 累积 alpha ${alphaPath}`)

  // ── 7. 与 SHARP 模型自己的深度估计做数值对比 ──
  const sharp = loadSharpDepths()
  if (sharp) {
    const sharpWidth = width
    const sharpHeight = height
    const layers = sharp.layers.map((_, l) =>
      resampleLayer(sharp, l, sharpWidth, sharpHeight),
    )
    const front = new Float32Array(width * height)
    for (let i = 0; i < front.length; i++) {
      const d0 = layers[0]?.[i] ?? 0
      const d1 = layers[1]?.[i] ?? 0
      const valid = [d0, d1].filter((d) => d > 0)
      front[i] = valid.length > 0 ? Math.min(...valid) : 0
    }
    const candidates: [string, Float32Array][] = [
      ["layer0", layers[0] ?? new Float32Array(0)],
      ...(layers.length > 1
        ? ([["layer1", layers[1]]] as [string, Float32Array][])
        : []),
      ["frontmost(min)", front],
    ]
    console.log(
      `\n[SHARP 深度] disparity_factor=${sharp.disparityFactor.toFixed(6)}  f_px=${sharp.focalPx.toFixed(2)}  原图 ${sharp.originalSize[0]}x${sharp.originalSize[1]}  网络域 ${sharp.internalSize}²（非等比拉伸）`,
    )
    for (const [name, map] of candidates) {
      const stats = compareDepthMaps(
        frame.depth,
        frame.visible,
        map,
        width * height,
      )
      console.log(
        `  [${name.padEnd(15)}] 中位相对误差 ${(stats.medianRel * 100).toFixed(2)}%  均值 ${(stats.meanRel * 100).toFixed(2)}%  相关系数 ${stats.corr.toFixed(4)}  有效像素 ${stats.count}`,
      )
    }
    // 只为目视：把 SHARP 的深度按**同一个标定**画出来，便于与我们的 D 直接对比
    const sharpVisible = new Uint8Array(width * height)
    for (let i = 0; i < sharpVisible.length; i++) {
      sharpVisible[i] = front[i] > 0 && frame.visible[i] ? 1 : 0
    }
    const sharpDepthPath = resolve(
      outDir,
      `${stem}_sharp-depth_${rangeTag}.png`,
    )
    await sharpFromRgba(
      depthToGrayPng(front, sharpVisible, zRange),
      width,
      height,
    )
      .png()
      .toFile(sharpDepthPath)
    console.log(`[png] SHARP 深度（同一标定，近=白） ${sharpDepthPath}`)
  } else {
    console.log(
      "\n[SHARP 深度] 未找到 fixtures（ndc.npz / input.npz），跳过对比",
    )
  }

  // ── 8. CPU 参考 + 对齐 ──
  if (runCpu) {
    const t0 = performance.now()
    const reference = renderSplatsCpu({
      gaussians,
      camera: {
        viewMatrix: camera.viewMatrix,
        fx: pose.fx * scale,
        fy: pose.fx * scale,
        cx: width / 2,
        cy: height / 2,
        width,
        height,
      },
      eps2d,
    })
    const t1 = performance.now()
    console.log(`[耗时] CPU 参考 ${(t1 - t0).toFixed(0)}ms`)

    const gpuGray = grayFromRgb(frame.rgb, frame.alpha, width * height)
    const cpuGray = grayFromRgb(reference.rgb, reference.alpha, width * height)
    const mask = buildMask(frame, reference)
    let maskCount = 0
    for (let i = 0; i < mask.length; i++) maskCount += mask[i]
    const ncc = maskedNcc(gpuGray, cpuGray, mask)
    const mae = maskedMae(frame.rgb, reference.rgb, mask)
    console.log(`\n[对齐] 掩码像素 ${maskCount}`)
    console.log(`[对齐] 灰度 NCC = ${ncc.toFixed(5)}`)
    console.log(`[对齐] 直通 RGB MAE = ${mae.toFixed(5)}`)

    // ── 分区域误差（透明边缘 / 发丝 / 远景各自的误差）──
    const regions = regionStats({
      rgb: frame.rgb,
      alpha: frame.alpha,
      depth: frame.depth,
      visible: frame.visible,
      reference,
      width,
      height,
    })
    console.log(
      "\n[区域]  区域           像素数       占比      RGB MAE   深度中位相对误差",
    )
    for (const r of regions) {
      console.log(
        `        ${r.name.padEnd(12)} ${String(r.count).padStart(9)}  ` +
          `${((r.count / (width * height)) * 100).toFixed(2).padStart(6)}%  ` +
          `${r.mae.toFixed(5).padStart(10)}  ${(r.depthMedianRel * 100).toFixed(3).padStart(9)}%`,
      )
    }

    // ── 与原图对比（画面级证据；sharp-check-camera.ts 只核对相机参数，不渲染）──
    if (compareImage) {
      const photoPath = resolve(REPO_ROOT, args.values.image ?? DEFAULT_IMAGE)
      const photo = await loadImage(photoPath)
      const photoRgb = boxDownsampleRgb8(
        photo.image.data as Uint8Array,
        photo.image.channels,
        photo.image.width,
        photo.image.height,
        width,
        height,
      )
      // 只在「渲染基本不透明」的像素上比：A 很小时直通 RGB 是被除法放大的残值
      const opaque = new Uint8Array(width * height)
      let opaqueCount = 0
      for (let i = 0; i < opaque.length; i++) {
        if (frame.alpha[i] > 0.5 && mask[i]) {
          opaque[i] = 1
          opaqueCount++
        }
      }
      const previewRgb = rgba8ToRgb8(frame.preview, width * height)
      const photoNcc = maskedNcc8(previewRgb, photoRgb, opaque)
      const photoMae = maskedMae8(previewRgb, photoRgb, opaque)
      let renderMean = 0
      let photoMean = 0
      for (let i = 0; i < opaque.length; i++) {
        if (!opaque[i]) continue
        renderMean +=
          (frame.rgb[i * 3] + frame.rgb[i * 3 + 1] + frame.rgb[i * 3 + 2]) / 3
        photoMean +=
          (srgbToLinear(photoRgb[i * 3] / 255) +
            srgbToLinear(photoRgb[i * 3 + 1] / 255) +
            srgbToLinear(photoRgb[i * 3 + 2] / 255)) /
          3
      }
      renderMean /= Math.max(opaqueCount, 1)
      photoMean /= Math.max(opaqueCount, 1)
      console.log(
        `\n[原图] ${photoPath} (${photo.image.width}x${photo.image.height})`,
      )
      console.log(
        `[原图] 灰度 NCC = ${photoNcc.toFixed(4)}   MAE = ${photoMae.toFixed(2)}/255（A>0.5 的 ${opaqueCount} 像素）`,
      )
      console.log(
        `[原图] 线性域平均亮度：渲染 ${renderMean.toFixed(4)} / 原图 ${photoMean.toFixed(4)} = ${(renderMean / photoMean).toFixed(4)}x`,
      )
    }

    const cpuPath = resolve(outDir, `${stem}_cpu.png`)
    await sharpFromRgba(reference.preview, width, height).png().toFile(cpuPath)
    const sidePath = resolve(outDir, `${stem}_side-by-side.png`)
    await sharpFromRgba(
      sideBySide(frame.preview, reference.preview, width, height),
      width * 2,
      height,
    )
      .png()
      .toFile(sidePath)
    console.log(`[png] CPU  ${cpuPath}`)
    console.log(`[png] 并排 ${sidePath}`)

    // ── 深度对齐（深度通道）──
    const depthStats = computeDepthStats(frame, reference, width * height)
    console.log(
      `\n[深度] 中位相对误差 ${(depthStats.medianRel * 100).toFixed(3)}%  均值 ${(depthStats.meanRel * 100).toFixed(3)}%  p95 ${(depthStats.p95Rel * 100).toFixed(3)}%`,
    )
    console.log(
      `[深度] D ∈ [zMin, zMax] 占比 ${(depthStats.inRange * 100).toFixed(2)}%  (zMin=${depthStats.zMin.toFixed(3)} zMax=${depthStats.zMax.toFixed(3)})`,
    )
    console.log(
      `[深度] 最近点 z（z-buffer 语义）中位相对误差 ${(depthStats.nearestRel * 100).toFixed(3)}%`,
    )
    console.log(
      `[深度] 背景判定不一致像素 ${depthStats.visibleMismatch} / ${width * height}`,
    )
    // ── AA 对照（细小高斯的稳定性）──
    if (noAaFrame) {
      const noAaGray = grayFromRgb(
        noAaFrame.rgb,
        noAaFrame.alpha,
        width * height,
      )
      const noAaNcc = maskedNcc(noAaGray, cpuGray, mask)
      const noAaMae = maskedMae(noAaFrame.rgb, reference.rgb, mask)
      console.log(
        `\n[AA 对照] 有 AA: NCC=${ncc.toFixed(5)} MAE=${mae.toFixed(5)}   无 AA: NCC=${noAaNcc.toFixed(5)} MAE=${noAaMae.toFixed(5)}`,
      )
      const noAaPath = resolve(outDir, `${stem}_no-aa.png`)
      await sharpFromRgba(noAaFrame.preview, width, height)
        .png()
        .toFile(noAaPath)
      console.log(`[png] 无 AA ${noAaPath}`)
      console.log(
        `[AA 判定] 有 AA ${mae < noAaMae ? "更优" : "**不优于**无 AA（需要查）"}：MAE ${mae.toFixed(5)} vs ${noAaMae.toFixed(5)}`,
      )
    }

    // 判定不在这里：门限集中在 `scripts/wsplat-golden.ts`，
    // 本脚本只负责把数字与图片打出来给人看，避免两处阈值漂移。
    console.log(
      "\n（本脚本不做判定；门限与 PASS/FAIL 见 npx tsx scripts/wsplat-golden.ts）",
    )
  }
}

/**
 * 两张深度图的差异统计（都按米，只算两边都可用的像素）。
 *
 * 同时报 Pearson 相关系数：它能区分「形状像但整体有系统偏移」与「形状本身就不对」。
 */
function compareDepthMaps(
  a: Float32Array,
  aVisible: Uint8Array,
  b: Float32Array,
  count: number,
): { medianRel: number; meanRel: number; corr: number; count: number } {
  const rel: number[] = []
  let n = 0
  let sa = 0
  let sb = 0
  let saa = 0
  let sbb = 0
  let sab = 0
  for (let i = 0; i < count; i++) {
    if (!aVisible[i] || !(a[i] > 0) || !(b[i] > 0)) continue
    rel.push(Math.abs(a[i] - b[i]) / b[i])
    n++
    sa += a[i]
    sb += b[i]
    saa += a[i] * a[i]
    sbb += b[i] * b[i]
    sab += a[i] * b[i]
  }
  if (n === 0) {
    return {
      medianRel: Number.NaN,
      meanRel: Number.NaN,
      corr: Number.NaN,
      count: 0,
    }
  }
  rel.sort((x, y) => x - y)
  const cov = sab / n - (sa / n) * (sb / n)
  const va = saa / n - (sa / n) ** 2
  const vb = sbb / n - (sb / n) ** 2
  return {
    medianRel: rel[Math.floor(n / 2)],
    meanRel: rel.reduce((x, y) => x + y, 0) / n,
    corr: cov / Math.sqrt(Math.max(va * vb, 1e-12)),
    count: n,
  }
}

function sideBySide(
  left: Uint8Array,
  right: Uint8Array,
  width: number,
  height: number,
): Uint8Array {
  const out = new Uint8Array(width * 2 * height * 4)
  for (let y = 0; y < height; y++) {
    out.set(left.subarray(y * width * 4, (y + 1) * width * 4), y * width * 8)
    out.set(
      right.subarray(y * width * 4, (y + 1) * width * 4),
      y * width * 8 + width * 4,
    )
  }
  return out
}

function srgbToLinear(x: number): number {
  return x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4
}

const requireCjs = createRequire(import.meta.url)

function sharpFromRgba(
  rgba: Uint8Array,
  width: number,
  height: number,
): { png(): { toFile(p: string): Promise<unknown> } } {
  const sharp = requireCjs("sharp") as (
    input: Buffer,
    opts: { raw: { width: number; height: number; channels: number } },
  ) => { png(): { toFile(p: string): Promise<unknown> } }
  return sharp(Buffer.from(rgba.buffer, rgba.byteOffset, rgba.byteLength), {
    raw: { width, height, channels: 4 },
  })
}

main().catch((err) => {
  console.error("\n[错误]", err instanceof Error ? err.message : err)
  if (err instanceof Error && err.stack) console.error(err.stack)
  process.exitCode = 1
})
