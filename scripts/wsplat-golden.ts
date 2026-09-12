/**
 * wsplat golden 门：质量门 + 画面级证据 + 耗时观测。
 *
 * 阶段：**wsplat 渲染器**（`scripts/wsplat-*`：渲染 / 自检 / 质量门）。
 *
 * 这是**唯一**的判定入口：所有阈值写死在这里，退出码非 0 即「不过门」。
 * `scripts/wsplat-render.ts` 只负责出图给人看，不再承担判定（它会打印同样的数字）。
 *
 * ── 跑什么 ──
 *   A. 排序单元不变量（纯 CPU，秒级）
 *      - 与 `Array#sort` 的独立实现逐位对拍（含大量相等深度）
 *      - 有效排列 / 深度降序 / 相等深度按下标升序（稳定性）/ 确定性
 *      - 退化输入：0/1 个、全相等、NaN、±Inf、负数、已排序、逆序
 *   B. 渲染 + 与 CPU 参考对比（数值门）
 *   C. 排序真的在起作用 + 确定性
 *      - 不排序（PLY 原顺序）必须明显更差 → 证明排序不是摆设
 *      - 同相机渲染两次逐位一致、重排一次与两次逐位一致 → 排除闪烁来源
 *      - 2 个高斯的合成场景：近蓝远红、PLY 顺序为「近在前」
 *        → 排序后必须是蓝（近压远）。这一条同时锁死**排序方向**与 **over 方向**
 *   D. 分区域误差（透明边缘 / 发丝 / 远景）
 *   E. 与原图对比（画面级证据）
 *   F. 耗时观测（非目标，仅记录）
 *
 * ── 原分辨率实测（3024x2268，1,179,648 高斯，仅作回归对照）──
 * | 项 | 实测 | 门限 |
 * |---|---|---|
 * | GPU vs CPU 灰度 NCC | 0.99975 | > 0.98 |
 * | GPU vs CPU RGB MAE（线性） | 0.00160 | < 0.02 |
 * | D 中位相对误差 | 0.063%（p95 0.394%） | < 2% |
 * | 排序 vs 不排序 MAE | 0.00160 vs 0.05767（36x） | 不排序 > 1.5x |
 * | 开 AA vs 关 AA MAE | 0.00160 vs 0.00181 | 开 < 关 |
 * | 与原图 NCC / MAE | 0.9636 / 6.82（/255） | > 0.9 / < 8 |
 * | 线性域平均亮度比 | 0.9577 | [0.9, 1.1] |
 * | 耗时（上传/排序/绘制/resolve+读回） | 62/54/2.9/855 ms | 仅记录 |
 * 残余差异来自核形状（CPU 用精确高斯 + 3σ 截断，GPU 用 2.83σ quad 归一化核），
 * 属于已知且有意的差异。
 *
 * ── 这些门**不**覆盖什么（别把它们当成全部保证）──
 * 1. 「远景 / 半透明背景的排序质量」：这组数据是**满覆盖**的（`A > 0.5` 占 99.99%，
 *    远景区只有 165 px），所以这一条**在本数据上无法验证**；要真验需要带天空/远景的数据。
 * 2. 与 gsplat 的 `ED` / `D` **符号级逐条对照**（目前只到「两边同式 + 数值一致」）。
 * 3. 浏览器侧（用户指示整体延后，至少等 splat -> RGBAD -> mesh 完全打通）。
 * 4. 全分辨率显存：三附件约 360 MB / 层；多层（LDI）需要分块或降低附件精度。
 * 5. `minPixelSize` 是像素量纲 —— 门必须在**原分辨率**下判；低分辨率下的 NCC/MAE
 *    只反映「高斯被剔了多少」，不代表管线质量（见 `wsplat-scan-res.ts`）。
 *
 * 用法：
 *   npx tsx scripts/wsplat-golden.ts                  # 原分辨率（3024x2268），最慢
 *   npx tsx scripts/wsplat-golden.ts --width 768      # 快速迭代
 *   npx tsx scripts/wsplat-golden.ts --no-photo       # 跳过与原图对比（无原图时）
 */

import { createRequire } from "node:module"
import { resolve } from "node:path"
import { parseArgs } from "node:util"
import type { Gaussians3D } from "../src/spatial-scene/sharp/types.ts"
import { createWSplatCamera } from "../src/spatial-scene/wsplat/camera.ts"
import { createWSplatRenderer } from "../src/spatial-scene/wsplat/index.ts"
import {
  computeViewDepths,
  sortSplatsBackToFront,
} from "../src/spatial-scene/wsplat/sort.ts"
import type {
  WSplatFrame,
  WSplatStats,
} from "../src/spatial-scene/wsplat/types.ts"
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
  regionStats,
  rgba8ToRgb8,
  toGray,
} from "./utils/wsplat-metrics.ts"
import { loadWSplatScene } from "./utils/wsplat-scene.ts"

/**
 * 门限。
 *
 * 取值原则：**实测值 × ~1.5 倍余量**，即「回归门」而不是物理极限。
 * 过不了门时先查原因，不要直接放宽——门限的意义就是不让数值悄悄退化。
 * 括号里是原分辨率（3024x2268）下的实测值。
 */
const GATES = {
  /** GPU vs CPU 参考的灰度 NCC（实测 native 0.99860）。 */
  nccVsCpu: 0.98,
  /** GPU vs CPU 参考的直通线性 RGB MAE（实测 native 0.00369）。 */
  maeVsCpu: 0.02,
  /** 深度中位相对误差（硬门就是 2%）。 */
  depthMedianRel: 0.02,
  /** 分区域 RGB MAE（线性域）。 */
  regionMae: {
    全部可见: 0.02,
    不透明: 0.02,
    半透明边缘: 0.06,
    细节发丝: 0.1,
  } as Record<string, number>,
  /** 与原图（sRGB 8bit）的对比。 */
  photoNcc: 0.9,
  photoMae: 8,
  /** 线性域平均亮度比（渲染 / 原图）允许区间。 */
  brightnessRatio: [0.9, 1.1] as [number, number],
} as const

const CLI_OPTIONS = {
  width: { type: "string" },
  eps2d: { type: "string" },
  "no-photo": { type: "boolean" },
  image: { type: "string" },
} as const

/** 一条门的结论。 */
interface Check {
  name: string
  ok: boolean
  detail: string
}

const checks: Check[] = []

function addCheck(name: string, ok: boolean, detail: string): void {
  checks.push({ name, ok, detail })
  console.log(`  ${ok ? "✓" : "✗"} ${name.padEnd(30)} ${detail}`)
}

async function main(): Promise<void> {
  const args = parseArgs({
    args: process.argv.slice(2),
    options: CLI_OPTIONS,
    allowPositionals: false,
    strict: true,
  })
  const eps2d = Number.parseFloat(args.values.eps2d ?? "0.3")

  const scene = loadWSplatScene({
    width: args.values.width ?? "native",
  })
  const { gaussians, camera, width, height, pose, scale } = scene

  console.log("=".repeat(78))
  console.log("wsplat golden 门")
  console.log("=".repeat(78))
  console.log(
    `场景 : ${scene.plyPath}\n` +
      `       ${gaussians.opacities.length} 高斯  ${pose.width}x${pose.height} -> 渲染 ${width}x${height}  fx=${(pose.fx * scale).toFixed(2)}  eps2d=${eps2d}`,
  )

  // ── A. 排序单元不变量 ──
  console.log("\n[A] 排序单元不变量（CPU）")
  sortUnitChecks()

  // ── B/C/D/E：需要 GPU ──
  await withNodeDevice(async (device) => {
    const renderOnce = async (options: {
      antialias?: boolean
      sort?: boolean
      sorts?: number
      stats?: boolean
    }): Promise<{ frame: WSplatFrame; stats?: WSplatStats }> => {
      const renderer = await createWSplatRenderer(device, {
        size: { width, height },
        antialias: options.antialias ?? true,
      })
      const t0 = performance.now()
      renderer.setGaussians(gaussians)
      const t1 = performance.now()
      renderer.setCamera(camera)
      const times: number[] = []
      for (let i = 0; i < (options.sorts ?? 1); i++) {
        const a = performance.now()
        if (options.sort !== false) renderer.sort()
        const b = performance.now()
        times.push(b - a)
      }
      const t2 = performance.now()
      renderer.renderSplats()
      const t3 = performance.now()
      const frame = await renderer.readback()
      const t4 = performance.now()
      console.log(
        `[耗时] 上传 ${(t1 - t0).toFixed(1)}ms  排序 ${times.map((t) => t.toFixed(1)).join("+")}ms  ` +
          `渲染 ${(t3 - t2).toFixed(1)}ms  resolve+读回 ${(t4 - t3).toFixed(1)}ms`,
      )
      // 剔除统计（可选）：与本次渲染用同一组 uniform
      let stats: WSplatStats | undefined
      if (options.stats) {
        const s0 = performance.now()
        renderer.countCulls()
        stats = await renderer.readSplatStats()
        console.log(`[耗时] 剔除统计 ${(performance.now() - s0).toFixed(1)}ms`)
      }
      renderer.destroy()
      return { frame, stats }
    }

    console.log("\n[B] 渲染 + CPU 参考")
    const { frame, stats } = await renderOnce({ stats: true })
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
    console.log(`[耗时] CPU 参考 ${(performance.now() - t0).toFixed(0)}ms`)

    const count = width * height
    const mask = buildMask(frame, reference)
    const ncc = maskedNcc(
      toGray(frame.rgb, frame.alpha, count),
      toGray(reference.rgb, reference.alpha, count),
      mask,
    )
    const mae = maskedMae(frame.rgb, reference.rgb, mask)
    addCheck(
      "GPU vs CPU 灰度 NCC",
      ncc > GATES.nccVsCpu,
      `${ncc.toFixed(5)} > ${GATES.nccVsCpu}`,
    )
    addCheck(
      "GPU vs CPU RGB MAE",
      mae < GATES.maeVsCpu,
      `${mae.toFixed(5)} < ${GATES.maeVsCpu}`,
    )

    const depth = depthStats(frame, reference, count)
    addCheck(
      "D 中位相对误差",
      depth.medianRel < GATES.depthMedianRel,
      `${(depth.medianRel * 100).toFixed(3)}% < ${GATES.depthMedianRel * 100}%  (p95 ${(depth.p95Rel * 100).toFixed(3)}%)`,
    )
    addCheck(
      "D ∈ [zMin, zMax]",
      depth.inRange > 0.9999,
      `${(depth.inRange * 100).toFixed(2)}%  z∈[${depth.zMin.toFixed(3)}, ${depth.zMax.toFixed(3)}]m`,
    )
    addCheck(
      "A→0 处不输出 D",
      depth.visibleMismatch === 0,
      `visible 判定不一致 ${depth.visibleMismatch} / ${count}`,
    )
    let tErr = 0
    for (let i = 0; i < count; i++) {
      tErr = Math.max(
        tErr,
        Math.abs(frame.transmission[i] - (1 - frame.alpha[i])),
      )
    }
    addCheck("T == 1 - A", tErr < 1e-6, `max|Δ| = ${tErr.toExponential(2)}`)

    // ── 剔除统计（可观测性 + 与 CPU 参考交叉验证）──
    if (!stats) throw new Error("缺少剔除统计")
    const statSum =
      stats.drawn +
      stats.culledBounds +
      stats.culledAlphaClip +
      stats.culledAlphaClipAfterAa +
      stats.culledBehindCamera +
      stats.culledMinPixelSize +
      stats.culledFrustum
    console.log(
      `      剔除: minPixelSize ${stats.culledMinPixelSize}  ` +
        `alphaClip ${stats.culledAlphaClip + stats.culledAlphaClipAfterAa}  ` +
        `视锥 ${stats.culledFrustum}  相机后方 ${stats.culledBehindCamera}  ` +
        `越界 ${stats.culledBounds}  -> 绘制 ${stats.drawn}/${stats.total}`,
    )
    addCheck(
      "剔除统计之和 == 高斯总数",
      statSum === stats.total,
      `${statSum} == ${stats.total}`,
    )
    const gpuAlphaClip = stats.culledAlphaClip + stats.culledAlphaClipAfterAa
    const cpuAlphaClip =
      reference.cull.alphaClip + reference.cull.alphaClipAfterAa
    addCheck(
      "GPU/CPU 剔除计数一致（±2%）",
      Math.abs(stats.culledMinPixelSize - reference.cull.minPixelSize) <=
        0.02 * Math.max(stats.culledMinPixelSize, 1) &&
        Math.abs(gpuAlphaClip - cpuAlphaClip) <=
          0.02 * Math.max(gpuAlphaClip, 1),
      `minPixelSize GPU ${stats.culledMinPixelSize} vs CPU ${reference.cull.minPixelSize}；` +
        `alphaClip GPU ${gpuAlphaClip} vs CPU ${cpuAlphaClip}`,
    )

    // AA 对照：开 AA 必须比关 AA 更接近参考（细小高斯的稳定性）
    const { frame: noAa } = await renderOnce({ antialias: false })
    const noAaMae = maskedMae(noAa.rgb, reference.rgb, mask)
    addCheck(
      "开 AA 优于关 AA",
      mae < noAaMae,
      `MAE ${mae.toFixed(5)} < 无 AA ${noAaMae.toFixed(5)}`,
    )

    // ── C. 排序的作用与确定性 ──
    console.log("\n[C] 排序生效 + 确定性")
    const { frame: unsorted } = await renderOnce({ sort: false })
    const unsortedNcc = maskedNcc(
      toGray(unsorted.rgb, unsorted.alpha, count),
      toGray(reference.rgb, reference.alpha, count),
      mask,
    )
    const unsortedMae = maskedMae(unsorted.rgb, reference.rgb, mask)
    addCheck(
      "不排序明显更差（排序非摆设）",
      unsortedMae > mae * 1.5,
      `不排序 MAE ${unsortedMae.toFixed(5)} (NCC ${unsortedNcc.toFixed(5)}) vs 排序 ${mae.toFixed(5)}`,
    )
    const hashUnsorted = frameHash(unsorted)
    void hashUnsorted

    const { frame: again } = await renderOnce({})
    addCheck(
      "同相机渲染两次逐位一致",
      frameHash(frame) === frameHash(again),
      `hash ${frameHash(frame).slice(0, 16)}…`,
    )
    const { frame: twice } = await renderOnce({ sorts: 2 })
    addCheck(
      "重排一次与两次一致（差异只是耗时）",
      frameHash(frame) === frameHash(twice),
      `hash ${frameHash(twice).slice(0, 16)}…`,
    )

    console.log("\n[C2] 方向性：近蓝远红（PLY 顺序 = 近在前）")
    const direction = await overDirectionCheck(device)
    addCheck(
      "排序方向 = 远→近（近压远）",
      direction.sortedNearWins && direction.unsortedFarWins,
      `排序后中心像素 B-R = ${direction.sortedDelta.toFixed(4)}（应为正）；` +
        `不排序 B-R = ${direction.unsortedDelta.toFixed(4)}（应为负）`,
    )

    // ── D. 分区域误差 ──
    console.log("\n[D] 分区域误差（透明边缘 / 发丝 / 远景）")
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
      "      区域           像素数       占比      RGB MAE   深度中位相对误差",
    )
    for (const r of regions) {
      console.log(
        `      ${r.name.padEnd(12)} ${String(r.count).padStart(9)}  ` +
          `${((r.count / count) * 100).toFixed(2).padStart(6)}%  ` +
          `${r.mae.toFixed(5).padStart(10)}  ${(r.depthMedianRel * 100).toFixed(3).padStart(9)}%`,
      )
    }
    for (const r of regions) {
      const key = r.name.replace("/", "")
      const gate = GATES.regionMae[key]
      if (gate === undefined) continue
      addCheck(
        `区域 MAE: ${r.name}`,
        r.mae < gate,
        `${r.mae.toFixed(5)} < ${gate}  (占 ${((r.count / count) * 100).toFixed(2)}% 像素)`,
      )
    }

    // ── E. 与原图对比 ──
    if (args.values["no-photo"] !== true) {
      console.log("\n[E] 与原图对比（画面级证据）")
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
      const opaque = new Uint8Array(count)
      let opaqueCount = 0
      for (let i = 0; i < count; i++) {
        if (frame.alpha[i] > 0.5 && mask[i]) {
          opaque[i] = 1
          opaqueCount++
        }
      }
      const previewRgb = rgba8ToRgb8(frame.preview, count)
      const photoNcc = maskedNcc8(previewRgb, photoRgb, opaque)
      const photoMae = maskedMae8(previewRgb, photoRgb, opaque)
      let renderMean = 0
      let photoMean = 0
      for (let i = 0; i < count; i++) {
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
      const ratio = renderMean / photoMean
      addCheck(
        "原图灰度 NCC",
        photoNcc > GATES.photoNcc,
        `${photoNcc.toFixed(4)} > ${GATES.photoNcc}  (${photo.image.width}x${photo.image.height} -> ${width}x${height})`,
      )
      addCheck(
        "原图 RGB MAE",
        photoMae < GATES.photoMae,
        `${photoMae.toFixed(2)}/255 < ${GATES.photoMae}  (A>0.5 的 ${opaqueCount} 像素)`,
      )
      addCheck(
        "线性域平均亮度比",
        ratio > GATES.brightnessRatio[0] && ratio < GATES.brightnessRatio[1],
        `${ratio.toFixed(4)} x  ∈ [${GATES.brightnessRatio[0]}, ${GATES.brightnessRatio[1]}]`,
      )
    }
  })

  // ── 汇总 ──
  const failed = checks.filter((c) => !c.ok)
  console.log("\n" + "=".repeat(78))
  console.log(
    `共 ${checks.length} 条，通过 ${checks.length - failed.length}，失败 ${failed.length}`,
  )
  if (failed.length > 0) {
    console.log("\n✗ 未通过:")
    for (const f of failed) console.log(`   - ${f.name}: ${f.detail}`)
    process.exitCode = 1
  } else {
    console.log("✓ 全部门限通过")
  }
}

// ────────────────────────────── A. 排序单元不变量 ──────────────────────────────

/** 与模块内一致的正浮点位模式（非正数 -> 0）。 */
const scratchF32 = new Float32Array(1)
const scratchU32 = new Uint32Array(scratchF32.buffer)
function keyOf(v: number): number {
  scratchF32[0] = v > 0 ? v : 0
  return ~scratchU32[0]
}

function sortUnitChecks(): void {
  // 1) 与独立实现（Array#sort，稳定）逐位对拍
  const n = 200_000
  const depths = new Float32Array(n)
  {
    // 有意制造大量相等值与少量极端值
    let seed = 12345
    const rnd = (): number => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff
      return seed / 0x7fffffff
    }
    for (let i = 0; i < n; i++) {
      const r = rnd()
      depths[i] =
        r < 0.3
          ? Math.fround(1 + Math.floor(rnd() * 5))
          : Math.fround(rnd() * 10)
    }
  }
  const expected = Array.from({ length: n }, (_, i) => i).sort((a, b) => {
    const ka = keyOf(depths[a])
    const kb = keyOf(depths[b])
    return ka - kb || a - b
  })
  const actual = sortSplatsBackToFront(depths, n)
  let mismatch = -1
  for (let i = 0; i < n; i++) {
    if (actual[i] !== expected[i]) {
      mismatch = i
      break
    }
  }
  addCheck(
    "与 Array#sort 独立实现对拍",
    mismatch < 0,
    mismatch < 0
      ? `${n} 个键逐位一致（含 ${new Set(depths).size} 个不同键，大量相等值）`
      : `第 ${mismatch} 位不同: ${actual[mismatch]} vs ${expected[mismatch]}`,
  )

  // 2) 随机 1M：有效排列 + 深度降序
  // 全用正深度：非正数的语义单独在「退化输入」里用 key 检查
  const big = new Float32Array(1_000_003)
  for (let i = 0; i < big.length; i++) {
    big[i] = Math.fround(Math.abs(Math.sin(i * 12.9898) * 43758.5453) + 1e-3)
  }
  const order = sortSplatsBackToFront(big, big.length)
  const seen = new Uint8Array(big.length)
  let dup = 0
  for (let i = 0; i < order.length; i++) {
    if (seen[order[i]]) dup++
    seen[order[i]] = 1
  }
  addCheck(
    "输出是有效排列",
    dup === 0 && order.length === big.length,
    `${big.length} 个下标无重复（非 2 的幂长度）`,
  )
  let desc = true
  let firstBad = -1
  for (let i = 0; i + 1 < order.length; i++) {
    if (big[order[i]] < big[order[i + 1]]) {
      desc = false
      firstBad = i
      break
    }
  }
  addCheck(
    "深度降序（back-to-front）",
    desc,
    desc
      ? `z[0]=${big[order[0]].toFixed(4)} ≥ z[last]=${big[order[order.length - 1]].toFixed(4)}`
      : `第 ${firstBad} 位逆序`,
  )

  // 3) 相等深度 -> 下标升序（稳定性 = tie-break 的唯一来源）
  const ties = new Float32Array(1000).fill(2.5)
  const tieOrder = sortSplatsBackToFront(ties, ties.length)
  let stable = true
  for (let i = 0; i < tieOrder.length; i++)
    if (tieOrder[i] !== i) stable = false
  addCheck("相等深度按下标升序（稳定）", stable, "1000 个同深度值 -> 原序")

  // 4) 确定性
  const a1 = sortSplatsBackToFront(big, big.length)
  const a2 = sortSplatsBackToFront(big, big.length)
  let same = a1.length === a2.length
  for (let i = 0; same && i < a1.length; i++) if (a1[i] !== a2[i]) same = false
  addCheck("确定性（两次调用逐位一致）", same, `${big.length} 个下标`)

  // 5) 退化输入
  const degen: [string, Float32Array, boolean][] = [
    ["空数组", new Float32Array(0), true],
    ["单个", new Float32Array([1.5]), true],
    [
      "NaN / ±Inf / 负数 / 0 / 极大极小 混合",
      new Float32Array([
        Number.NaN,
        Number.POSITIVE_INFINITY,
        Number.NEGATIVE_INFINITY,
        -3,
        0,
        1e-38,
        3.4e38,
        -0,
        5,
      ]),
      true,
    ],
    [
      "已排序（升序）",
      Float32Array.from({ length: 500 }, (_, i) => i + 1),
      true,
    ],
    ["逆序", Float32Array.from({ length: 500 }, (_, i) => 500 - i), true],
  ]
  for (const [label, input, _] of degen) {
    const out = sortSplatsBackToFront(input, input.length)
    let ok = out.length === input.length
    const seen2 = new Set<number>()
    for (const v of out) {
      if (seen2.has(v)) ok = false
      seen2.add(v)
    }
    for (let i = 0; i + 1 < out.length; i++) {
      const za = input[out[i]]
      const zb = input[out[i + 1]]
      // NaN 归 0（会被排到最前），用 key 比较才是精确定义
      if (keyOf(za) > keyOf(zb)) ok = false
    }
    addCheck(`退化输入: ${label}`, ok, `${input.length} 个 -> 无重复且键升序`)
  }

  // 6) computeViewDepths 与手算一致
  const means = new Float32Array([1, 2, 3, -4, 5, -6, 7, 8, 9])
  const view = new Float32Array(16)
  for (let i = 0; i < 16; i++) view[i] = Math.fround(Math.sin(i + 1))
  const d = computeViewDepths(means, view, 3)
  let maxErr = 0
  for (let i = 0; i < 3; i++) {
    const manual =
      view[2] * means[i * 3] +
      view[6] * means[i * 3 + 1] +
      view[10] * means[i * 3 + 2] +
      view[14]
    maxErr = Math.max(maxErr, Math.abs(manual - d[i]))
  }
  addCheck(
    "computeViewDepths == 第三行点积",
    maxErr < 1e-6,
    `max|Δ| = ${maxErr.toExponential(2)}`,
  )
}

// ───────────────────── C2. over 方向（合成 2 高斯场景）─────────────────────

/**
 * 近蓝远红两个高斯，**PLY 顺序 = 近在前**（即原顺序就是错的）。
 * 排序后中心像素应偏蓝（近压远），不排序应偏红。
 */
async function overDirectionCheck(device: GPUDevice): Promise<{
  sortedNearWins: boolean
  unsortedFarWins: boolean
  sortedDelta: number
  unsortedDelta: number
}> {
  const size = 64
  const gaussians: Gaussians3D = {
    meanVectors: new Float32Array([0, 0, 1, 0, 0, 3]),
    singularValues: new Float32Array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    quaternions: new Float32Array([1, 0, 0, 0, 1, 0, 0, 0]),
    // 近 = 蓝，远 = 红（linearRGB）
    colors: new Float32Array([0, 0, 1, 1, 0, 0]),
    opacities: new Float32Array([0.99, 0.99]),
  }
  const camera = createWSplatCamera({
    intrinsics: { focalLengthPx: 100, width: size, height: size },
    near: 0.1,
    far: 10,
  })
  const run = async (sort: boolean): Promise<number> => {
    const renderer = await createWSplatRenderer(device, {
      size: { width: size, height: size },
    })
    renderer.setGaussians(gaussians)
    renderer.setCamera(camera)
    if (sort) renderer.sort()
    renderer.renderSplats()
    const frame = await renderer.readback()
    renderer.destroy()
    const center = ((size / 2) * size + size / 2) * 3
    return frame.rgb[center + 2] - frame.rgb[center]
  }
  const sortedDelta = await run(true)
  const unsortedDelta = await run(false)
  return {
    sortedDelta,
    unsortedDelta,
    sortedNearWins: sortedDelta > 0.1,
    unsortedFarWins: unsortedDelta < -0.1,
  }
}

// ────────────────────────────── 小工具 ──────────────────────────────

/** 帧的确定性指纹（逐位；用于「两次渲染是否一致」）。 */
function frameHash(frame: WSplatFrame): string {
  let h = 0x811c9dc5
  const mix = (bytes: Uint8Array | Float32Array): void => {
    const view =
      bytes instanceof Uint8Array
        ? bytes
        : new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength)
    for (let i = 0; i < view.length; i++) {
      h ^= view[i]
      h = Math.imul(h, 0x01000193) >>> 0
    }
  }
  mix(frame.preview)
  mix(frame.rgb)
  mix(frame.alpha)
  mix(frame.depth)
  mix(frame.transmission)
  mix(frame.accumulatedDepth)
  mix(frame.visible)
  return h.toString(16).padStart(8, "0")
}

function srgbToLinear(x: number): number {
  return x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4
}

void createRequire

main().catch((err) => {
  console.error("\n[错误]", err instanceof Error ? err.message : err)
  if (err instanceof Error && err.stack) console.error(err.stack)
  process.exitCode = 1
})
