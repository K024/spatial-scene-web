/**
 * layering 数值自检（**唯一判定入口**：阈值只在本文件里，退出码非 0 = 不过门）。
 *
 * 阶段：`src/spatial-scene/layering/**`（splat 场 -> L 层线性 RGBAD）。
 *
 * ── 跑什么 ──
 *   [A] CPU 不变量（无 GPU）：放置（边界严格递增 / 质量归一 / 米制往返）、
 *       分带（严格排列 / 层序远→近 / 层内视差递减 / 逐位确定性）、
 *       视图（参考矩形 / 焦距换算 / 相机轴反解）。
 *   [B] 参考视角恒等式：`renderLayerStack(..., includeDirect)` 后
 *       `compositeLayerFrames(frames)` 必须等于 `singleFrameToComposited(direct)` ——
 *       α / 预乘色 / 深度三个度量都在门内。这是 layering 的**唯一验收口径**。
 *       同时验逐层剔除统计恒等式（`drawn + 各项剔除 == total`）。
 *   [C] 视觉输出（**仅观测，不判定**）：逐层 PNG + 合成 / 直渲 / 差异图。
 *
 * ── 为什么 [B] 是恒等式而不是「差不多」──
 * `over` 结合 + 层是排序序列的硬划分 ⇒ 数学上逐位相等；实际差异只来自
 * `rgba16float` 累加附件的舍入与逐层 resolve 的写法（每层各自 resolve 一次
 * `D = ED/A` 会丢一点精度）。门限就是为这两项留的余量，不是「允许近似」。
 *
 * 用法:
 *   npx tsx scripts/layering-check.ts
 *   npx tsx scripts/layering-check.ts --layers 10 --view-scale 1.2 --short-side auto
 *   npx tsx scripts/layering-check.ts --no-visual          # 不出图
 *   npx tsx scripts/layering-check.ts --max-splats 200000  # 调试：只用 N 个高斯
 */

import { mkdirSync, writeFileSync } from "node:fs"
import { resolve } from "node:path"
import { parseArgs } from "node:util"

import {
  buildLayerPermutation,
  compareComposited,
  compositeLayerFrames,
  computeLayerPlacement,
  DEFAULT_LAYERS,
  DEFAULT_MAX_RENDER_SIDE,
  DEFAULT_SHORT_SIDE,
  type LayerPlacement,
  ndcDepthFromZ,
  permuteNdcDepths,
  planLayerStack,
  renderLayerStack,
  resolveLayerView,
  singleFrameToComposited,
  zFromNdcDepth,
} from "../src/spatial-scene/layering/index.ts"
import { computeViewDepths } from "../src/spatial-scene/wsplat/sort.ts"
import { numFlag, REPO_ROOT, shortSideFlag } from "./utils/common.ts"
import { rgbaToPngBuffer } from "./utils/image.ts"
import { linearFrameToRgba8, loadWSplatScene } from "./utils/scene.ts"
import { withNodeDevice } from "./utils/webgpu.ts"

/**
 * 门限（实测留 ~2 倍余量的回归门，不是物理极限）。
 *
 * `alpha` 与预乘色都在 `rgba16float` 附件里累加：单次写入的分辨率 ~2^-11 ≈ 4.9e-4，
 * 10 层累加的最坏情况 ~5e-3。
 */
const GATES = {
  alphaMaxAbs: 1e-2,
  premultipliedMae: 2e-3,
  depthMedianRel: 0.02,
} as const

const CLI = {
  layers: { type: "string" },
  "view-scale": { type: "string" },
  "render-scale": { type: "string" },
  "max-side": { type: "string" },
  "short-side": { type: "string" },
  "range-overlap": { type: "string" },
  "min-pixel-size": { type: "string" },
  ply: { type: "string" },
  camera: { type: "string" },
  "max-splats": { type: "string" },
  out: { type: "string" },
  "no-visual": { type: "boolean" },
} as const

const checks: { name: string; ok: boolean; detail: string }[] = []

function addCheck(name: string, ok: boolean, detail: string): void {
  checks.push({ name, ok, detail })
  console.log(`  ${ok ? "✓" : "✗"} ${name.padEnd(30)} ${detail}`)
}

async function main(): Promise<void> {
  const args = parseArgs({
    args: process.argv.slice(2),
    options: CLI,
    allowPositionals: false,
    strict: true,
  })
  const layers = Math.round(
    numFlag("--layers", args.values.layers, DEFAULT_LAYERS),
  )
  const viewScale = numFlag("--view-scale", args.values["view-scale"], 1.2)
  const renderScale = numFlag("--render-scale", args.values["render-scale"], 1)
  // 分辨率默认「短边 auto」：min(SHARP 内部 1536, 原图短边)；--max-side 只作长边硬上限。
  const shortSide = shortSideFlag(
    "--short-side",
    args.values["short-side"],
    DEFAULT_SHORT_SIDE,
  )
  const maxSide = numFlag(
    "--max-side",
    args.values["max-side"],
    DEFAULT_MAX_RENDER_SIDE,
  )
  const rangeOverlap = numFlag(
    "--range-overlap",
    args.values["range-overlap"],
    0,
  )
  const outDir = resolve(REPO_ROOT, args.values.out ?? "temp/layering-check")

  console.log("=".repeat(78))
  console.log("layering 数值自检")
  console.log("=".repeat(78))

  console.log("\n[A] CPU 不变量")
  placementChecks(layers)
  bandsChecks(layers)
  viewChecks()

  const scene = loadWSplatScene({
    ply: args.values.ply,
    camera: args.values.camera,
    maxSplats: args.values["max-splats"]
      ? Math.round(numFlag("--max-splats", args.values["max-splats"], 0))
      : undefined,
  })
  console.log(
    `\n场景: ${scene.plyPath}\n      ${scene.gaussians.opacities.length} 个高斯  ` +
      `参考 ${scene.camera.width}x${scene.camera.height}  fx=${scene.focalLengthPx.toFixed(1)}px  ` +
      `near=${scene.near.toFixed(3)}m far=${scene.far.toFixed(1)}m`,
  )

  await withNodeDevice(async (device) => {
    console.log("\n[B] 参考视角恒等式（分层混合 == 整场渲染）")
    const t0 = Date.now()
    const layered = await renderLayerStack(device, scene.gaussians, {
      camera: scene.camera,
      layers,
      viewScale,
      renderScale,
      shortSide,
      maxRenderSide: maxSide,
      rangeOverlap,
      minPixelSize: args.values["min-pixel-size"]
        ? numFlag("--min-pixel-size", args.values["min-pixel-size"], 2)
        : undefined,
      includeDirect: true,
    })
    console.log(
      `      渲染完成 ${Date.now() - t0}ms  ` +
        `画布 ${layered.width}x${layered.height}  ` +
        `像素倍率 ${layered.view.pixelScale.toFixed(4)}  fx=${layered.view.focalLengthPx.toFixed(1)}px`,
    )
    printPlacement(layered.placement, layered.stats)

    const direct = layered.direct
    if (!direct) throw new Error("includeDirect 没有返回 direct 帧")
    const composite = compositeLayerFrames(layered.frames)
    const directComposited = singleFrameToComposited(direct)
    const diff = compareComposited(composite, directComposited)

    addCheck(
      "α 最大绝对差",
      diff.alphaMaxAbs <= GATES.alphaMaxAbs,
      `max=${diff.alphaMaxAbs.toExponential(2)}  mae=${diff.alphaMae.toExponential(2)}  (门 ${GATES.alphaMaxAbs})`,
    )
    addCheck(
      "预乘色 MAE",
      diff.premultipliedMae <= GATES.premultipliedMae,
      `mae=${diff.premultipliedMae.toExponential(2)}  max=${diff.premultipliedMaxAbs.toExponential(2)}  (门 ${GATES.premultipliedMae})`,
    )
    addCheck(
      "深度中位相对误差",
      diff.depthMedianRel <= GATES.depthMedianRel,
      `${diff.depthMedianRel.toExponential(2)} @ ${diff.depthSamples} 像素  (门 ${GATES.depthMedianRel})`,
    )

    // 逐层剔除统计恒等式（`minPixelSize` 会静默剔高斯，必须报出来）。
    let statsOk = true
    const emptyLayers: number[] = []
    for (let k = 0; k < layered.L; k++) {
      const s = layered.stats[k]
      const sum =
        s.drawn +
        s.culledBounds +
        s.culledAlphaClip +
        s.culledAlphaClipAfterAa +
        s.culledBehindCamera +
        s.culledMinPixelSize +
        s.culledFrustum
      if (sum !== s.total) statsOk = false
      if (s.drawn === 0) emptyLayers.push(k)
    }
    addCheck(
      "逐层剔除恒等式",
      statsOk,
      `${layered.L} 层  drawn+剔除==total${emptyLayers.length > 0 ? `  ⚠ 空层 ${emptyLayers.join(",")}` : ""}`,
    )

    console.log("\n[C] 视觉输出")
    if (args.values["no-visual"] === true) {
      console.log("  --no-visual：跳过")
    } else {
      await writeVisuals(
        outDir,
        layered.frames,
        composite,
        direct,
        directComposited,
      )
    }
  })

  const failed = checks.filter((c) => !c.ok)
  console.log(`\n${"=".repeat(78)}`)
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

// ═══════════════════════════ A. CPU 不变量 ═══════════════════════════

function placementChecks(L: number): void {
  const near = 0.5
  const far = 120
  const n = 300_000
  const depths = new Float32Array(n)
  const weights = new Float32Array(n)
  let seed = 987654321
  const rnd = (): number => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff
    return seed / 0x7fffffff
  }
  // 双峰分布（近处一团、远处一片）+ 少量离群深度，覆盖空箱段。
  for (let i = 0; i < n; i++) {
    const u = rnd()
    const z =
      u < 0.35 ? 2 + rnd() * 3 : u < 0.95 ? 15 + rnd() * 60 : 100 + rnd() * 3000
    depths[i] = z
    weights[i] = rnd()
  }

  const placement = computeLayerPlacement(depths, {
    L,
    near,
    far,
    weights,
    method: "quantile",
  })

  let increasing = true
  for (let i = 1; i < L; i++) {
    if (!(placement.layerDepths[i] > placement.layerDepths[i - 1])) {
      increasing = false
    }
  }
  increasing =
    increasing &&
    placement.boundaries[0] === 0 &&
    placement.boundaries[L] === 1 &&
    placement.layerDepths[0] >= 0 &&
    placement.layerDepths[L - 1] <= 1
  for (let i = 1; i <= L; i++) {
    if (!(placement.boundaries[i] > placement.boundaries[i - 1])) {
      increasing = false
    }
  }
  addCheck(
    "边界严格递增",
    increasing,
    `b[0]=${placement.boundaries[0].toFixed(3)} b[L]=${placement.boundaries[L].toFixed(3)}`,
  )

  let massSum = 0
  let massOk = true
  for (let i = 0; i < L; i++) {
    massSum += placement.layerMass[i]
    if (!(placement.layerMass[i] >= 0)) massOk = false
    if (
      placement.layerMass[i] === 0 &&
      !(placement.boundaries[i + 1] > placement.boundaries[i])
    ) {
      massOk = false
    }
  }
  addCheck(
    "质量归一",
    massOk && Math.abs(massSum - 1) < 1e-6,
    `Σmass=${massSum.toFixed(9)}`,
  )

  let roundTrip = 0
  for (let i = 0; i < L; i++) {
    const z = zFromNdcDepth(placement.layerDepths[i], near, far)
    roundTrip = Math.max(
      roundTrip,
      Math.abs(ndcDepthFromZ(z, near, far) - placement.layerDepths[i]),
    )
  }
  addCheck(
    "视差 <-> 米制往返",
    roundTrip < 1e-6,
    `max|Δn|=${roundTrip.toExponential(2)}`,
  )
}

function bandsChecks(L: number): void {
  const near = 0.5
  const far = 200
  const n = 120_000
  const depths = new Float32Array(n)
  let seed = 24680
  for (let i = 0; i < n; i++) {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff
    depths[i] = 0.4 + (seed / 0x7fffffff) * 250
  }
  const placements: LayerPlacement[] = [
    computeLayerPlacement(depths, { L, near, far, method: "quantile" }),
    computeLayerPlacement(depths, { L, near, far, method: "uniform" }),
  ]

  let allOk = true
  let detail = ""
  for (const placement of placements) {
    const order = Array.from({ length: n }, (_, i) => i).sort(
      (a, b) => depths[b] - depths[a] || a - b,
    )
    const globalOrder = Uint32Array.from(order)
    const sortedNdc = permuteNdcDepths(depths, globalOrder, near, far)
    const perm = buildLayerPermutation(
      globalOrder,
      sortedNdc,
      placement.boundaries,
    )

    // 1) 严格排列
    const seen = new Uint8Array(n)
    let permOk = perm.length === n
    for (let i = 0; i < perm.length; i++) {
      const index = perm.permutation[i]
      if (index >= n || seen[index]) permOk = false
      seen[index] = 1
    }
    // 2) 排列序列的视差递减（= 真 back-to-front；用「高斯 -> 视差」反查表验）
    const ndcBySplat = new Float32Array(n)
    for (let i = 0; i < n; i++) ndcBySplat[globalOrder[i]] = sortedNdc[i]
    let ordered = true
    let previous = Number.POSITIVE_INFINITY
    for (let i = 0; i < perm.length; i++) {
      const value = ndcBySplat[perm.permutation[i]]
      if (value > previous + 1e-9) ordered = false
      previous = value
    }
    // 3) 层表连续、覆盖整段
    let tableOk = true
    let cursor = 0
    for (let k = perm.table.length - 1; k >= 0; k--) {
      const entry = perm.table[k]
      if (entry.base !== cursor) tableOk = false
      cursor += entry.count
    }
    if (cursor !== n) tableOk = false
    // 4) 每层元素落在自己的边界带内（首/末层额外吸收图外的越界值）
    let bandsOk = true
    for (let k = 0; k < L; k++) {
      const { base, count } = perm.table[k]
      const lo = placement.boundaries[k]
      const hi = placement.boundaries[k + 1]
      for (let i = 0; i < count; i++) {
        const value = ndcBySplat[perm.permutation[base + i]]
        const lowerOk = k === 0 ? true : value >= lo - 1e-9
        const upperOk = k === L - 1 ? true : value <= hi + 1e-9
        if (!lowerOk || !upperOk) bandsOk = false
      }
    }
    allOk = allOk && permOk && ordered && tableOk && bandsOk
    detail += `${placement.method}:${permOk && ordered && tableOk && bandsOk ? "ok" : "FAIL"} `
  }

  // 5) 确定性：同一输入两次的结果逐位相同
  const planA = planLayerStack(depths, undefined, { layers: L, near, far })
  const planB = planLayerStack(depths, undefined, { layers: L, near, far })
  let deterministic = planA.permutation.length === planB.permutation.length
  if (deterministic) {
    for (let i = 0; i < planA.permutation.length; i++) {
      if (
        planA.permutation.permutation[i] !== planB.permutation.permutation[i]
      ) {
        deterministic = false
        break
      }
    }
  }
  addCheck("分带（排列/层序/带内）", allOk, detail.trim())
  addCheck("逐位确定性", deterministic, `${n} 个下标两次结果一致`)
}

function viewChecks(): void {
  const reference = { width: 3024, height: 2268, focalLengthPx: 2620.958 }
  // 短边 auto = min(1536, 原图短边)；参考内容短边应正好落在这个像素数上。
  const view = resolveLayerView(reference, { viewScale: 1.2 })
  const rect = view.referenceRect
  const expectedShort = Math.min(
    1536,
    Math.min(reference.width, reference.height),
  )
  const refShort = Math.min(rect.width, rect.height)
  const shortOk = Math.abs(refShort - expectedShort) < 1e-6
  // 参考矩形必须与画布同心、同尺度（焦距关系），且不超出画布。
  const scaleFromRect = rect.width / reference.width
  const sameScale = Math.abs(scaleFromRect - view.pixelScale) < 1e-9
  const focalOk =
    Math.abs(view.focalLengthPx - reference.focalLengthPx * scaleFromRect) <
    1e-6
  const centered =
    Math.abs(rect.x + rect.width / 2 - view.width / 2) < 1e-9 &&
    Math.abs(rect.y + rect.height / 2 - view.height / 2) < 1e-9
  const withinCanvas =
    rect.width <= view.width + 1 && rect.height <= view.height + 1
  addCheck(
    "视图（短边定尺/居中）",
    sameScale && focalOk && centered && withinCanvas && shortOk,
    `${view.width}x${view.height}  参考短边=${refShort.toFixed(1)}px（目标 ${expectedShort}）  pixelScale=${view.pixelScale.toFixed(4)}  fx=${view.focalLengthPx.toFixed(1)}px`,
  )
}

// ═══════════════════════════ 观测输出 ═══════════════════════════

function printPlacement(
  placement: LayerPlacement,
  stats: readonly {
    layerIndex: number
    drawn: number
    total: number
    culledMinPixelSize: number
  }[],
): void {
  console.log(
    "      层  视差带宽      米制深度带            质量     drawn/total  小高斯剔除",
  )
  for (let k = 0; k < placement.L; k++) {
    const s = stats[k]
    console.log(
      `      ${String(k).padStart(2)}  n[${placement.boundaries[k].toFixed(4)}, ${placement.boundaries[k + 1].toFixed(4)})  ` +
        `z[${placement.boundariesZ[k].toFixed(2)}, ${placement.boundariesZ[k + 1].toFixed(2)})m  ` +
        `${(placement.layerMass[k] * 100).toFixed(2).padStart(6)}%  ` +
        `${String(s.drawn).padStart(7)}/${String(s.total).padEnd(7)}  ${s.culledMinPixelSize}`,
    )
  }
}

async function writeVisuals(
  outDir: string,
  frames: readonly {
    width: number
    height: number
    rgb: Float32Array
    alpha: Float32Array
  }[],
  composite: {
    width: number
    height: number
    rgb: Float32Array
    alpha: Float32Array
  },
  direct: {
    width: number
    height: number
    rgb: Float32Array
    alpha: Float32Array
  },
  directComposited: {
    width: number
    height: number
    rgb: Float32Array
    alpha: Float32Array
  },
): Promise<void> {
  mkdirSync(outDir, { recursive: true })
  const pixels = composite.width * composite.height
  for (let k = 0; k < frames.length; k++) {
    const frame = frames[k]
    const rgba = linearFrameToRgba8(frame.rgb, frame.alpha, pixels)
    const png = await rgbaToPngBuffer(rgba, frame.width, frame.height)
    writeFileSync(
      resolve(outDir, `layer_${String(k).padStart(2, "0")}.png`),
      png,
    )
  }
  // 合成（预乘 -> 直通）与直渲的对照图
  const compositeRgba = new Uint8Array(pixels * 4)
  const directRgba = new Uint8Array(pixels * 4)
  const diffRgba = new Uint8Array(pixels * 4)
  const straight = new Float32Array(pixels * 3)
  for (let i = 0; i < pixels; i++) {
    const a = composite.alpha[i]
    const inv = a > 0 ? 1 / a : 0
    straight[i * 3] = composite.rgb[i * 3] * inv
    straight[i * 3 + 1] = composite.rgb[i * 3 + 1] * inv
    straight[i * 3 + 2] = composite.rgb[i * 3 + 2] * inv
  }
  compositeRgba.set(linearFrameToRgba8(straight, composite.alpha, pixels))
  directRgba.set(linearFrameToRgba8(direct.rgb, direct.alpha, pixels))
  for (let i = 0; i < pixels; i++) {
    for (let c = 0; c < 3; c++) {
      const d = Math.abs(
        composite.rgb[i * 3 + c] - directComposited.rgb[i * 3 + c],
      )
      diffRgba[i * 4 + c] = Math.min(255, Math.round(d * 20 * 255))
    }
    diffRgba[i * 4 + 3] = 255
  }
  writeFileSync(
    resolve(outDir, "composite.png"),
    await rgbaToPngBuffer(compositeRgba, composite.width, composite.height),
  )
  writeFileSync(
    resolve(outDir, "direct.png"),
    await rgbaToPngBuffer(directRgba, direct.width, direct.height),
  )
  writeFileSync(
    resolve(outDir, "diff_x20.png"),
    await rgbaToPngBuffer(diffRgba, composite.width, composite.height),
  )
  console.log(
    `      写出到 ${outDir}（layer_*.png / composite.png / direct.png / diff_x20.png）`,
  )
}

main().catch((err) => {
  console.error(`\n自检失败: ${err instanceof Error ? err.stack : err}`)
  process.exitCode = 1
})
