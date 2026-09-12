/**
 * 分辨率扫描（质量门的前置证据）。
 *
 * 阶段：**wsplat 渲染器**（`scripts/wsplat-*`：渲染 / 自检 / 质量门）。
 *
 * ── 为什么要扫 ──
 * `minPixelSize = 2`、quad 半轴上限 `vmin = min(1024, min(W,H))`、`eps2d = 0.3`
 * 全都是**像素量纲**，所以「提高渲染分辨率」不是免费的：
 *   - 降分辨率：高斯投影变小 -> 被 `minPixelSize` 剔掉的比例上升 -> 细节整片消失；
 *   - 升分辨率：剔除比例下降，但同一屏像素数变多，耗时上升。
 * 之前默认 384 的结论（NCC 0.9688）就不能代表原分辨率（0.9986），
 * 这个脚本把整条曲线一次测出来，顺便回答「有多少高斯被分辨率相关的剔除吃掉」。
 *
 * ── 判据 ──
 * 每档报：GPU vs CPU 的 NCC / MAE、深度中位相对误差、以及 GPU 原子计数给出的
 * 剔除构成（`culledMinPixelSize` 是分辨率相关的那一项）。
 * 注意 GPU↔CPU 的一致性**不随分辨率变差**（两边都是同一套像素量纲公式），
 * 分辨率真正影响的是**画质**（相对原图），所以这里也顺带报与原图的 MAE。
 *
 * ⚠ **「与原图」那一栏不能跨分辨率比较**：低分辨率下原图与渲染**都被平滑**了，
 * MAE 反而更小（实测 384 的 5.40 < 原分辨率的 6.82），看着"更好"其实只是都糊了。
 * 唯一有意义的是**原分辨率**那一行。
 *
 * 用法：
 *   npx tsx scripts/wsplat-scan-res.ts                       # 384/768/1536/3024
 *   npx tsx scripts/wsplat-scan-res.ts --widths 768,1536,3024
 *   npx tsx scripts/wsplat-scan-res.ts --no-photo
 */

import { resolve } from "node:path"
import { parseArgs } from "node:util"
import { createWSplatRenderer } from "../src/spatial-scene/wsplat/index.ts"
import { DEFAULT_IMAGE, REPO_ROOT } from "./utils/common.ts"
import { loadImage } from "./utils/image.ts"
import { withNodeDevice } from "./utils/webgpu.ts"
import { renderSplatsCpu } from "./utils/wsplat-cpu.ts"
import {
  boxDownsampleRgb8,
  buildMask,
  depthStats,
  maskedMae,
  maskedMae8,
  maskedNcc,
  maskedNcc8,
  rgba8ToRgb8,
  summarizeFrame,
  toGray,
} from "./utils/wsplat-metrics.ts"
import { loadWSplatScene } from "./utils/wsplat-scene.ts"

const CLI_OPTIONS = {
  widths: { type: "string" },
  eps2d: { type: "string" },
  "no-photo": { type: "boolean" },
  image: { type: "string" },
} as const

interface Row {
  width: number
  height: number
  ncc: number
  mae: number
  depthMedianRel: number
  depthP95: number
  culledMinPixelSize: string
  culledAlphaClip: string
  culledOther: string
  /** 可见像素的深度 p1~p99 与跨度（= 分层时的"层厚"） */
  depthSpan: string
  drawn: number
  photoMae: number
  photoNcc: number
  brightness: number
  ms: {
    upload: number
    sort: number
    draw: number
    readback: number
    cpu: number
  }
}

async function main(): Promise<void> {
  const args = parseArgs({
    args: process.argv.slice(2),
    options: CLI_OPTIONS,
    allowPositionals: false,
    strict: true,
  })
  const eps2d = Number.parseFloat(args.values.eps2d ?? "0.3")
  const widths = (args.values.widths ?? "384,768,1536,3024")
    .split(",")
    .map((s) => Number.parseInt(s.trim(), 10))

  // 用 native 宽度装配一次，拿到 gaussians / pose（之后只改渲染尺寸）
  const base = loadWSplatScene({ width: "native" })
  const { gaussians, pose } = base
  const nativeWidth = pose.width
  const nativeHeight = pose.height

  const photoPath = resolve(REPO_ROOT, args.values.image ?? DEFAULT_IMAGE)
  const photo =
    args.values["no-photo"] === true ? undefined : await loadImage(photoPath)

  console.log("=".repeat(96))
  console.log("分辨率扫描：剔除构成 + GPU/CPU 一致性 + 与原图差异")
  console.log("=".repeat(96))
  console.log(
    `场景 ${base.plyPath}\n     ${gaussians.opacities.length} 高斯  原图 ${nativeWidth}x${nativeHeight}  eps2d=${eps2d}  minPixelSize=2（取自上游）`,
  )

  const rows: Row[] = []
  await withNodeDevice(async (device) => {
    for (const width of widths) {
      const height = Math.round((width * nativeHeight) / nativeWidth)
      const scale = width / nativeWidth
      const scene = loadWSplatScene({
        width,
        plyPath: base.plyPath,
        cameraPath: base.cameraPath,
      })
      const camera = scene.camera
      const renderer = await createWSplatRenderer(device, {
        size: { width, height },
      })
      const t0 = performance.now()
      renderer.setGaussians(gaussians)
      const t1 = performance.now()
      renderer.setCamera(camera)
      renderer.sort()
      const t2 = performance.now()
      renderer.renderSplats()
      const t3 = performance.now()
      const frame = await renderer.readback()
      const t4 = performance.now()
      // 剔除构成取自 GPU 的原子计数（顶点阶段不允许 read_write，见 wgsl/cullStats.ts）
      renderer.countCulls()
      const stats = await renderer.readSplatStats()
      renderer.destroy()

      const c0 = performance.now()
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
      const c1 = performance.now()

      const count = width * height
      const mask = buildMask(frame, reference)
      const ncc = maskedNcc(
        toGray(frame.rgb, frame.alpha, count),
        toGray(reference.rgb, reference.alpha, count),
        mask,
      )
      const mae = maskedMae(frame.rgb, reference.rgb, mask)
      const depth = depthStats(frame, reference, count)

      let photoMae = Number.NaN
      let photoNcc = Number.NaN
      let brightness = Number.NaN
      if (photo) {
        const photoRgb = boxDownsampleRgb8(
          photo.image.data as Uint8Array,
          photo.image.channels,
          photo.image.width,
          photo.image.height,
          width,
          height,
        )
        const opaque = new Uint8Array(count)
        let n = 0
        for (let i = 0; i < count; i++) {
          if (frame.alpha[i] > 0.5 && mask[i]) {
            opaque[i] = 1
            n++
          }
        }
        const previewRgb = rgba8ToRgb8(frame.preview, count)
        photoNcc = maskedNcc8(previewRgb, photoRgb, opaque)
        photoMae = maskedMae8(previewRgb, photoRgb, opaque)
        let rm = 0
        let pm = 0
        for (let i = 0; i < count; i++) {
          if (!opaque[i]) continue
          rm +=
            (frame.rgb[i * 3] + frame.rgb[i * 3 + 1] + frame.rgb[i * 3 + 2]) / 3
          pm +=
            (srgbToLinear(photoRgb[i * 3] / 255) +
              srgbToLinear(photoRgb[i * 3 + 1] / 255) +
              srgbToLinear(photoRgb[i * 3 + 2] / 255)) /
            3
        }
        brightness = rm / Math.max(pm, 1e-9)
      }

      // GPU 计数（权威）；CPU 参考的同名计数在 golden 里做交叉验证
      const culledOther =
        stats.culledBounds + stats.culledBehindCamera + stats.culledFrustum
      const summary = summarizeFrame(frame)
      rows.push({
        width,
        height,
        ncc,
        mae,
        depthMedianRel: depth.medianRel,
        depthP95: depth.p95Rel,
        culledMinPixelSize: `${stats.culledMinPixelSize} (${((stats.culledMinPixelSize / stats.total) * 100).toFixed(2)}%)`,
        culledAlphaClip: `${stats.culledAlphaClip + stats.culledAlphaClipAfterAa} (${(((stats.culledAlphaClip + stats.culledAlphaClipAfterAa) / stats.total) * 100).toFixed(2)}%)`,
        culledOther: `${culledOther} (${((culledOther / stats.total) * 100).toFixed(2)}%)`,
        depthSpan: `${summary.depthP1.toFixed(2)}~${summary.depthP99.toFixed(2)} (${(summary.depthP99 / summary.depthP1).toFixed(2)}x)`,
        drawn: stats.drawn,
        photoMae,
        photoNcc,
        brightness,
        ms: {
          upload: t1 - t0,
          sort: t2 - t1,
          draw: t3 - t2,
          readback: t4 - t3,
          cpu: c1 - c0,
        },
      })
      console.log(
        `  ${String(width).padStart(4)}x${String(height).padEnd(5)} 完成（GPU 渲染+读回 ${(t4 - t0).toFixed(0)}ms，CPU 参考 ${(c1 - c0).toFixed(0)}ms）`,
      )
    }
  })

  const f = (x: number, d = 4): string =>
    Number.isFinite(x) ? x.toFixed(d) : "n/a"
  console.log("\n── GPU vs CPU 参考（同定义，应不随分辨率变差）──")
  console.log(
    "  渲染宽度      高斯绘制        剔除:minPixelSize      剔除:alphaClip      剔除:其他      NCC        MAE      深度中位    深度p95       层内深度跨度(p1~p99)",
  )
  for (const r of rows) {
    console.log(
      `  ${String(r.width).padStart(5)}x${String(r.height).padEnd(5)} ` +
        `${String(r.drawn).padStart(9)}  ${r.culledMinPixelSize.padStart(22)}  ` +
        `${r.culledAlphaClip.padStart(19)}  ${r.culledOther.padStart(16)}  ` +
        `${f(r.ncc, 5)}  ${f(r.mae, 5)}  ${f(r.depthMedianRel * 100, 3)}%  ${f(r.depthP95 * 100, 3)}%  ${r.depthSpan.padStart(18)}`,
    )
  }
  if (photo) {
    console.log("\n── 与原图（真正随分辨率变的是这一栏）──")
    console.log("  渲染宽度      NCC        MAE/255    线性亮度比")
    for (const r of rows) {
      console.log(
        `  ${String(r.width).padStart(5)}x${String(r.height).padEnd(5)}  ` +
          `${f(r.photoNcc)}  ${f(r.photoMae, 2).padStart(8)}  ${f(r.brightness, 4)}`,
      )
    }
  }
  console.log("\n── 耗时观测（非目标）──")
  console.log(
    "  渲染宽度      上传      排序  splat 绘制  resolve+读回   CPU 参考",
  )
  for (const r of rows) {
    console.log(
      `  ${String(r.width).padStart(5)}x${String(r.height).padEnd(5)}  ` +
        `${r.ms.upload.toFixed(1).padStart(7)}ms ${r.ms.sort.toFixed(1).padStart(7)}ms ` +
        `${r.ms.draw.toFixed(1).padStart(9)}ms ${r.ms.readback.toFixed(1).padStart(11)}ms ` +
        `${r.ms.cpu.toFixed(0).padStart(9)}ms`,
    )
  }
  console.log(
    "\n注：`剔除:minPixelSize` 是**分辨率相关**的那一项（同一份数据在不同渲染宽度下被剔掉的高斯数）。",
  )
}

function srgbToLinear(x: number): number {
  return x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4
}

main().catch((err) => {
  console.error("\n[错误]", err instanceof Error ? err.message : err)
  if (err instanceof Error && err.stack) console.error(err.stack)
  process.exitCode = 1
})
