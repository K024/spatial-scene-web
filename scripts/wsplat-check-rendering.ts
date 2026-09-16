/**
 * wsplat 渲染管线**数值自检（单文件版）**。
 *
 * 阶段：**wsplat 渲染器**（`src/spatial-scene/wsplat/**`）。
 *
 * 之前那套测试拆在 `wsplat-golden / render / check-blit / check-corner / scan-res`
 * 五个脚本 + 两个 util 里，判定入口、装配、度量各占一份。本次只保留**一个**脚本：
 * 把「数值上有没有问题」需要的证据全部内联，覆盖不如以前全，但每条都是真的在算数。
 *
 * ── 跑什么 ──
 *   [A] 排序单元不变量（纯 CPU）：与独立实现对拍 / 有效排列 + 深度降序 / 稳定性 /
 *       确定性 / `computeViewDepths` 与手算点积一致。
 *   [B] 解析解：两个纯色 quad（远→近）叠加后 resolve 出的 A / T / ED / D / 直通 RGB
 *       必须等于闭式解；清空不画时必须是背景语义（A=0 T=1 D=0 visible=0）。
 *   [C] 方向性：近蓝远红两个高斯，PLY 原顺序 = 近在前；排序后中心像素必须偏蓝
 *       （近压远），不排序必须偏红。同时锁死排序方向与 over 方向。
 *   [D] 协方差雅可比 y 行符号：合成一个 45° 倾斜的各向异性高斯，CPU 参考的
 *       alpha 足迹二阶矩 `cov_uv` 必须为正；GPU 默认 `cameraYAxis="down"` 与之同号，
 *       反向 `"up"` 必须反号（否则这个测试没有区分能力）。
 *   [E] 真实场景 GPU vs CPU 参考（默认 `temp/test.ply`）：灰度 NCC / 线性 RGB MAE /
 *       深度中位相对误差 / `T == 1-A` / 可见性判定一致 / 剔除统计恒等式与 GPU↔CPU
 *       剔除计数交叉验证 / AA 补偿路径（eps2d=0.3）优于不补偿 / 不排序明显更差 / 两次渲染逐位一致 /
 *       LDI 区间切分后按 `over` 合成与整帧一致。
 *   [F] 视觉对比（**仅观测，不判定**）：用 `sharp-compare-fixtures.ts` 的那套 fixtures
 *       （`py-models/out/fixtures/reference.ply` + `ml-depth-pro/data/example.jpg`）
 *       按 2×2 对照渲染（透明度补偿 AA × 低通 eps2d，含 ml-sharp 官方的 classic+eps0），
 *       打印每组与原图的 NCC / MAE / PSNR / SSIM / 亮度比，并把
 *       原图 | AA(eps0.3) | ml-sharp(eps0) | 差异 写到 `temp/wsplat-check/`。
 *
 * 判定阈值都写在本文件里；退出码非 0 即「不过」。没有真实场景文件时 [E] 自动跳过
 *（`--no-scene` 可强制跳过）；fixture 缺失时 [F] 自动跳过（`--no-visual` 可强制跳过，
 * `--visual-width N` 改渲染宽度，`--out DIR` 改输出目录，默认 `temp/wsplat-check`）。
 *
 * 用法：
 *   npx tsx scripts/wsplat-check-rendering.ts                 # 默认 384 宽 + [F] 768 宽出图
 *   npx tsx scripts/wsplat-check-rendering.ts --width 768
 *   npx tsx scripts/wsplat-check-rendering.ts --no-scene      # 只跑合成测试 + [F]
 *   npx tsx scripts/wsplat-check-rendering.ts --no-visual     # 不出视觉对比图
 */

import { existsSync, mkdirSync, readFileSync } from "node:fs"
import { createRequire } from "node:module"
import { resolve } from "node:path"
import { parseArgs } from "node:util"

import type { Gaussians3D } from "../src/spatial-scene/sharp/types.ts"
import {
  createWSplatCamera,
  type WSplatCamera,
} from "../src/spatial-scene/wsplat/camera.ts"
import {
  createWSplatRenderer,
  type WSplatCameraYAxis,
  type WSplatRenderer,
  type WSplatRendererOptions,
} from "../src/spatial-scene/wsplat/index.ts"
import {
  computeViewDepths,
  sortSplatsBackToFront,
} from "../src/spatial-scene/wsplat/sort.ts"
import type {
  WSplatFrame,
  WSplatStats,
} from "../src/spatial-scene/wsplat/types.ts"
import { DEFAULT_IMAGE, numFlag, REPO_ROOT } from "./utils/common.ts"
import {
  createSharpCanvas,
  loadImage,
  rgbaToPngBuffer,
  sharpFromRgba,
} from "./utils/image.ts"
import { parsePly } from "./utils/ply.ts"
import { withNodeDevice } from "./utils/webgpu.ts"
import { type CpuRenderResult, renderSplatsCpu } from "./utils/wsplat-cpu.ts"

/** SH degree-0 基函数值（与 `save_ply` 的 `convert_rgb_to_spherical_harmonics` 互逆）。 */
const SH_C0 = 0.28209479177387814

/**
 * [F] 视觉对比用的 fixtures，与 `scripts/sharp-compare-fixtures.ts` 同一套。
 *
 * `reference.ply` 是 PyTorch `save_ply` 的参考输出（1179648 高斯）；
 * `example.jpg` 是它的输入原图。⚠ `reference.ply` 的 `image_size` / 主点是
 * **交换过的**（fixture 生成时 `image_shape` 传成了 `(W,H)`，见
 * `sharp-compare-fixtures.ts` 的说明），所以 [F] 用原图的实际宽高覆盖它。
 */
const FIXTURE_PLY = "py-models/out/fixtures/reference.ply"
const FIXTURE_IMAGE = DEFAULT_IMAGE

/** 阈值（实测值留 ~1.5 倍余量的回归门，不是物理极限）。 */
const GATES = {
  /** GPU vs CPU 灰度 NCC。 */
  nccVsCpu: 0.9999,
  /** GPU vs CPU 直通线性 RGB MAE。 */
  maeVsCpu: 0.005,
  /** 深度中位相对误差。 */
  depthMedianRel: 0.02,
  /** GPU↔CPU 剔除计数的相对容差。 */
  cullRel: 0.05,
  /** 解析解容差（f16 附件）。 */
  analyticTol: 5e-3,
} as const

const CLI_OPTIONS = {
  width: { type: "string" },
  eps2d: { type: "string" },
  ply: { type: "string" },
  camera: { type: "string" },
  "max-splats": { type: "string" },
  "no-scene": { type: "boolean" },
  "visual-width": { type: "string" },
  "no-visual": { type: "boolean" },
  out: { type: "string" },
} as const

const checks: { name: string; ok: boolean; detail: string }[] = []

function addCheck(name: string, ok: boolean, detail: string): void {
  checks.push({ name, ok, detail })
  console.log(`  ${ok ? "✓" : "✗"} ${name.padEnd(34)} ${detail}`)
}

async function main(): Promise<void> {
  const args = parseArgs({
    args: process.argv.slice(2),
    options: CLI_OPTIONS,
    allowPositionals: false,
    strict: true,
  })
  const width = Math.max(
    16,
    Math.round(numFlag("--width", args.values.width, 384)),
  )
  const eps2d = numFlag("--eps2d", args.values.eps2d, 0)

  console.log("=".repeat(78))
  console.log("wsplat 渲染管线数值自检（单文件）")
  console.log("=".repeat(78))

  console.log("\n[A] 排序单元不变量（CPU）")
  sortUnitChecks()

  await withNodeDevice(async (device) => {
    console.log("\n[B] 解析解：预乘 over + resolve")
    await analyticBlitCheck(device)

    console.log("\n[C] 方向性：近蓝远红（PLY 顺序 = 近在前）")
    await overDirectionCheck(device)

    console.log("\n[D] 协方差雅可比 y 行符号")
    await cornerSignCheck(device)

    if (args.values["no-scene"] === true) {
      console.log("\n[E] 真实场景：已用 --no-scene 跳过")
    } else {
      await realSceneCheck(device, {
        width,
        eps2d,
        ply: args.values.ply,
        camera: args.values.camera,
        maxSplats: args.values["max-splats"],
      })
    }

    if (args.values["no-visual"] === true) {
      console.log("\n[F] 视觉对比：已用 --no-visual 跳过")
    } else {
      console.log("\n[F] 视觉对比（仅观测，不判定）")
      await visualComparison(device, {
        width: Math.max(
          16,
          Math.round(
            numFlag("--visual-width", args.values["visual-width"], 768),
          ),
        ),
        out: args.values.out,
      })
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

// ═══════════════════════════ A. 排序单元不变量 ═══════════════════════════

/** 与 `sort.ts` 一致的正浮点位模式（非正数 -> 0）。 */
const scratchF32 = new Float32Array(1)
const scratchU32 = new Uint32Array(scratchF32.buffer)
function keyOf(v: number): number {
  scratchF32[0] = v > 0 ? v : 0
  return ~scratchU32[0]
}

function sortUnitChecks(): void {
  // 1) 与独立实现（稳定 Array#sort）逐位对拍，含大量相等深度
  const n = 150_000
  const depths = new Float32Array(n)
  let seed = 12345
  const rnd = (): number => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff
    return seed / 0x7fffffff
  }
  for (let i = 0; i < n; i++) {
    const r = rnd()
    depths[i] =
      r < 0.3 ? Math.fround(1 + Math.floor(rnd() * 5)) : Math.fround(r * 10)
  }
  const expected = Array.from({ length: n }, (_, i) => i).sort((a, b) => {
    const d = keyOf(depths[a]) - keyOf(depths[b])
    return d !== 0 ? d : a - b
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
      ? `${n} 个下标逐位一致（${new Set(depths).size} 个不同键，含大量相等值）`
      : `第 ${mismatch} 位不同: ${actual[mismatch]} vs ${expected[mismatch]}`,
  )

  // 2) 非 2 的幂长度：有效排列 + 深度降序
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
    "有效排列 + 深度降序（back-to-front）",
    dup === 0 && order.length === big.length && desc,
    desc
      ? `${big.length} 个下标无重复，z ${big[order[0]].toFixed(3)} ≥ ${big[order[order.length - 1]].toFixed(3)}`
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
  let same = a1.length === order.length
  for (let i = 0; same && i < a1.length; i++)
    if (a1[i] !== order[i]) same = false
  addCheck("确定性（两次调用逐位一致）", same, `${big.length} 个下标`)

  // 5) computeViewDepths == 第三行点积
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

// ═════════════════════ [B] 解析解：纯色 over + resolve ═════════════════════

const FAR_SOLID = { color: [0.8, 0.2, 0.05] as const, alpha: 0.5, depth: 2 }
const NEAR_SOLID = { color: [0.1, 0.4, 0.9] as const, alpha: 0.5, depth: 1 }

async function analyticBlitCheck(device: GPUDevice): Promise<void> {
  const width = 256
  const height = 256
  const renderer = await createWSplatRenderer(device, {
    size: { width, height },
  })
  try {
    // 远 -> 近（back-to-front）
    renderer.clearSplatBuffers()
    renderer.renderSolidColor(
      [...FAR_SOLID.color, FAR_SOLID.alpha],
      FAR_SOLID.depth,
    )
    renderer.renderSolidColor(
      [...NEAR_SOLID.color, NEAR_SOLID.alpha],
      NEAR_SOLID.depth,
    )
    const frame = await renderer.readback()

    const aFar = FAR_SOLID.alpha
    const aNear = NEAR_SOLID.alpha
    const tFar = 1 - aNear
    const tNear = 1
    const expectedA = 1 - (1 - aFar) * (1 - aNear)
    const expectedT = 1 - expectedA
    const expectedEd =
      FAR_SOLID.depth * aFar * tFar + NEAR_SOLID.depth * aNear * tNear
    const expectedD = expectedEd / expectedA
    const expectedPremul = [0, 1, 2].map(
      (c) =>
        FAR_SOLID.color[c] * aFar * tFar + NEAR_SOLID.color[c] * aNear * tNear,
    )
    const expectedStraight = expectedPremul.map((v) => v / expectedA)

    const center = Math.floor(height / 2) * width + Math.floor(width / 2)
    const rows: [string, number, number][] = [
      ["A (累积 alpha)", expectedA, frame.alpha[center]],
      ["T (透射率)", expectedT, frame.transmission[center]],
      ["ED (未归一化)", expectedEd, frame.accumulatedDepth[center]],
      ["D (期望深度)", expectedD, frame.depth[center]],
      ["straight R", expectedStraight[0], frame.rgb[center * 3]],
      ["straight G", expectedStraight[1], frame.rgb[center * 3 + 1]],
      ["straight B", expectedStraight[2], frame.rgb[center * 3 + 2]],
    ]
    for (const [label, want, got] of rows) {
      const diff = Math.abs(want - got)
      const rel = want !== 0 ? diff / Math.abs(want) : diff
      addCheck(
        `解析解 ${label}`,
        rel < GATES.analyticTol || diff < GATES.analyticTol,
        `want=${want.toFixed(6)} got=${got.toFixed(6)} (rel ${rel.toExponential(2)})`,
      )
    }

    let covered = 0
    for (let i = 0; i < frame.visible.length; i++) covered += frame.visible[i]
    addCheck(
      "纯色叠加覆盖整屏",
      covered === width * height,
      `${covered}/${width * height}`,
    )

    // 背景语义：清空后什么都不画
    renderer.clearSplatBuffers()
    const empty = await renderer.readback()
    const bg = 0
    addCheck(
      "背景语义 A=0 T=1 D=0 visible=0",
      empty.alpha[bg] === 0 &&
        empty.transmission[bg] === 1 &&
        empty.depth[bg] === 0 &&
        empty.visible[bg] === 0,
      `A=${empty.alpha[bg]} T=${empty.transmission[bg]} D=${empty.depth[bg]} visible=${empty.visible[bg]}`,
    )
  } finally {
    renderer.destroy()
  }
}

// ═══════════════════════ [C] over 方向（2 个高斯） ═══════════════════════

async function overDirectionCheck(device: GPUDevice): Promise<void> {
  const size = 64
  const gaussians: Gaussians3D = {
    meanVectors: new Float32Array([0, 0, 1, 0, 0, 3]),
    singularValues: new Float32Array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    quaternions: new Float32Array([1, 0, 0, 0, 1, 0, 0, 0]),
    colors: new Float32Array([0, 0, 1, 1, 0, 0]), // 近 = 蓝，远 = 红
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
    try {
      renderer.setGaussians(gaussians)
      renderer.setCamera(camera)
      if (sort) renderer.sort()
      renderer.renderSplats()
      const frame = await renderer.readback()
      const center = (Math.floor(size / 2) * size + Math.floor(size / 2)) * 3
      return frame.rgb[center + 2] - frame.rgb[center]
    } finally {
      renderer.destroy()
    }
  }
  const sortedDelta = await run(true)
  const unsortedDelta = await run(false)
  addCheck(
    "排序方向 = 远→近（近压远）",
    sortedDelta > 0.1 && unsortedDelta < -0.1,
    `排序 B-R=${sortedDelta.toFixed(4)}（应为正）；不排序 B-R=${unsortedDelta.toFixed(4)}（应为负）`,
  )
}

// ══════════════════ [D] 协方差雅可比 y 行符号 ══════════════════

const CORNER_SIZE = 192
const CORNER_FOCAL = 200

async function cornerSignCheck(device: GPUDevice): Promise<void> {
  // 光轴上的 45° 长条：J 的透视项为 0，椭圆倾斜完全由 3D 协方差旋转决定，符号可解析预测
  const gaussians: Gaussians3D = {
    meanVectors: new Float32Array([0, 0, 2]),
    singularValues: new Float32Array([0.05, 0.015, 0.015]),
    quaternions: new Float32Array([
      Math.cos(Math.PI / 8),
      0,
      0,
      Math.sin(Math.PI / 8),
    ]),
    colors: new Float32Array([1, 1, 1]),
    opacities: new Float32Array([1]),
  }
  const intrinsics = {
    focalLengthPx: CORNER_FOCAL,
    width: CORNER_SIZE,
    height: CORNER_SIZE,
  }
  const camera = createWSplatCamera({ intrinsics, near: 0.5, far: 10 })

  const cpu = renderSplatsCpu({
    gaussians,
    camera: {
      viewMatrix: camera.viewMatrix,
      fx: CORNER_FOCAL,
      fy: CORNER_FOCAL,
      cx: CORNER_SIZE / 2,
      cy: CORNER_SIZE / 2,
      width: CORNER_SIZE,
      height: CORNER_SIZE,
    },
  })
  const cpuMoment = footprintMoment(cpu.alpha, CORNER_SIZE)
  addCheck(
    "CPU 参考 cov_uv 为正（合成设置有效）",
    cpuMoment.cov > 0,
    `cov_uv=${fmt(cpuMoment.cov)}（α 覆盖 ${cpuMoment.count} px）`,
  )

  const results: { axis: WSplatCameraYAxis; cov: number }[] = []
  for (const axis of ["down", "up"] as const) {
    const renderer = await createWSplatRenderer(device, {
      size: { width: CORNER_SIZE, height: CORNER_SIZE },
      cameraYAxis: axis,
    })
    try {
      renderer.setGaussians(gaussians)
      renderer.setCamera(camera)
      renderer.renderSplats()
      const frame = await renderer.readback()
      results.push({ axis, cov: footprintMoment(frame.alpha, CORNER_SIZE).cov })
    } finally {
      renderer.destroy()
    }
  }
  const down = results.find((r) => r.axis === "down")
  const up = results.find((r) => r.axis === "up")
  addCheck(
    '默认 cameraYAxis="down" 与 CPU 同号',
    down !== undefined && Math.sign(down.cov) === Math.sign(cpuMoment.cov),
    `down cov_uv=${fmt(down?.cov)}`,
  )
  addCheck(
    '反向 "up" 与 CPU 反号（测试有区分能力）',
    up !== undefined && Math.sign(up.cov) !== Math.sign(cpuMoment.cov),
    `up cov_uv=${fmt(up?.cov)}`,
  )
}

/** alpha 足迹相对质心的二阶矩：`cov = Σα·du·dv / Σα`。 */
function footprintMoment(
  alpha: Float32Array,
  size: number,
): { cov: number; count: number } {
  let sum = 0
  let su = 0
  let sv = 0
  let count = 0
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const a = alpha[y * size + x]
      if (a <= 0) continue
      sum += a
      su += a * x
      sv += a * y
      count++
    }
  }
  if (sum <= 0) return { cov: Number.NaN, count: 0 }
  const cu = su / sum
  const cv = sv / sum
  let m = 0
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const a = alpha[y * size + x]
      if (a <= 0) continue
      m += a * (x - cu) * (y - cv)
    }
  }
  return { cov: m / sum, count }
}

function fmt(x: number | undefined): string {
  return x === undefined || !Number.isFinite(x) ? String(x) : x.toFixed(4)
}

// ═══════════════════ [E] 真实场景 GPU vs CPU 参考 ═══════════════════

interface SceneOptions {
  width: number
  eps2d?: number
  ply?: string
  camera?: string
  maxSplats?: string
  /** 覆盖 PLY 里的 native 尺寸（fixture 的 `image_size` 可能是交换过的）。 */
  nativeSize?: { width: number; height: number }
}

interface Scene {
  gaussians: Gaussians3D
  camera: WSplatCamera
  width: number
  height: number
  /** 渲染像素焦距（= 原图 fx × scale）。 */
  fx: number
  count: number
  plyPath: string
}

async function realSceneCheck(
  device: GPUDevice,
  options: SceneOptions,
): Promise<void> {
  const scene = loadScene(options)
  if (!scene) {
    console.log("  (未找到真实场景 PLY，跳过 [E])")
    return
  }
  const { gaussians, camera, width, height, fx } = scene
  console.log(
    `  场景 ${scene.plyPath}\n        ${scene.count} 高斯  渲染 ${width}x${height}  fx=${fx.toFixed(2)}  eps2d=${options.eps2d}`,
  )

  const size = { width, height }
  const { frame, stats } = await runGpu(device, gaussians, camera, size, {
    stats: true,
  })
  const cpuCamera = {
    viewMatrix: camera.viewMatrix,
    fx,
    fy: fx,
    cx: width / 2,
    cy: height / 2,
    width,
    height,
  }
  const reference = renderSplatsCpu({
    gaussians,
    camera: cpuCamera,
    eps2d: options.eps2d,
  })

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
    "GPU vs CPU 线性 RGB MAE",
    mae < GATES.maeVsCpu,
    `${mae.toFixed(5)} < ${GATES.maeVsCpu}`,
  )

  const depth = depthStats(frame, reference, count)
  addCheck(
    "深度中位相对误差 D",
    depth.medianRel < GATES.depthMedianRel,
    `${(depth.medianRel * 100).toFixed(3)}% < ${GATES.depthMedianRel * 100}%  (p95 ${(depth.p95Rel * 100).toFixed(3)}%)`,
  )
  addCheck(
    "可见性判定一致（A→0 处不输出 D）",
    depth.visibleMismatch === 0,
    `visible 不一致 ${depth.visibleMismatch} / ${count}`,
  )

  let tErr = 0
  for (let i = 0; i < count; i++) {
    tErr = Math.max(
      tErr,
      Math.abs(frame.transmission[i] - (1 - frame.alpha[i])),
    )
  }
  addCheck("T == 1 - A", tErr < 1e-3, `max|Δ| = ${tErr.toExponential(2)}`)

  // ── 剔除统计 ──
  if (!stats) throw new Error("缺少剔除统计")
  const statSum =
    stats.drawn +
    stats.culledBounds +
    stats.culledAlphaClip +
    stats.culledAlphaClipAfterAa +
    stats.culledBehindCamera +
    stats.culledMinPixelSize +
    stats.culledFrustum
  addCheck(
    "剔除统计之和 == 高斯总数",
    statSum === stats.total,
    `${statSum} == ${stats.total}`,
  )
  console.log(
    `      剔除: 绘制 ${stats.drawn}  minPixelSize ${stats.culledMinPixelSize}  ` +
      `alphaClip ${stats.culledAlphaClip + stats.culledAlphaClipAfterAa}  ` +
      `视锥 ${stats.culledFrustum}  相机后方 ${stats.culledBehindCamera}  越界 ${stats.culledBounds}`,
  )
  const gpuAlphaClip = stats.culledAlphaClip + stats.culledAlphaClipAfterAa
  const cpuAlphaClip =
    reference.cull.alphaClip + reference.cull.alphaClipAfterAa
  const relDiff = (a: number, b: number): number =>
    Math.abs(a - b) / Math.max(a, b, 1)
  addCheck(
    "GPU↔CPU 剔除计数交叉验证（±5%）",
    relDiff(stats.culledMinPixelSize, reference.cull.minPixelSize) <
      GATES.cullRel && relDiff(gpuAlphaClip, cpuAlphaClip) < GATES.cullRel,
    `minPixelSize GPU ${stats.culledMinPixelSize} vs CPU ${reference.cull.minPixelSize}；alphaClip GPU ${gpuAlphaClip} vs CPU ${cpuAlphaClip}`,
  )

  // ── AA（透明度补偿）路径 ──
  // 主对比用的是 SHARP 官方口径（`eps2d=0`，此时补偿恒等于 1）；
  // 这里单独用 gsplat 默认的 `eps2d=0.3` 验证“补偿确实把有低通模糊的结果拉近
  // antialiased 参考”，否则 AA 分支没有任何覆盖。
  const aaReference = renderSplatsCpu({
    gaussians,
    camera: cpuCamera,
    eps2d: 0.3,
  })
  const aaMask = buildMask(frame, aaReference)
  const aaOn = await runGpu(device, gaussians, camera, size, {
    antialias: true,
    eps2d: 0.3,
  })
  const aaOff = await runGpu(device, gaussians, camera, size, {
    antialias: false,
    eps2d: 0.3,
  })
  const aaOnMae = maskedMae(aaOn.frame.rgb, aaReference.rgb, aaMask)
  const aaOffMae = maskedMae(aaOff.frame.rgb, aaReference.rgb, aaMask)
  addCheck(
    "AA 补偿路径（eps2d=0.3）优于不补偿",
    aaOnMae < aaOffMae,
    `补偿 MAE ${aaOnMae.toFixed(5)} < 不补偿 ${aaOffMae.toFixed(5)}`,
  )

  // ── 排序的作用与确定性 ──
  const unsorted = await runGpu(device, gaussians, camera, size, {
    sort: false,
  })
  const unsortedNcc = maskedNcc(
    toGray(unsorted.frame.rgb, unsorted.frame.alpha, count),
    toGray(reference.rgb, reference.alpha, count),
    mask,
  )
  const unsortedMae = maskedMae(unsorted.frame.rgb, reference.rgb, mask)
  addCheck(
    "不排序明显更差（排序非摆设）",
    unsortedMae > mae * 1.5,
    `不排序 MAE ${unsortedMae.toFixed(5)} (NCC ${unsortedNcc.toFixed(5)}) vs 排序 ${mae.toFixed(5)}`,
  )
  const again = await runGpu(device, gaussians, camera, size, {})
  addCheck(
    "同相机渲染两次逐位一致",
    frameHash(frame) === frameHash(again.frame),
    `hash ${frameHash(frame).slice(0, 16)}…`,
  )

  // ── LDI：区间切分后按 over 合成 == 整帧 ──
  const split = Math.floor(scene.count / 2)
  const splitCheck = await ldiSplitCheck(
    device,
    gaussians,
    camera,
    size,
    split,
    frame,
  )
  addCheck("LDI 区间切分 + over 合成 == 整帧", splitCheck.ok, splitCheck.detail)
}

interface GpuRunOptions {
  antialias?: boolean
  /** gsplat 语义的 eps2d（默认 0.3）。 */
  eps2d?: number
  sort?: boolean
  stats?: boolean
}

async function runGpu(
  device: GPUDevice,
  gaussians: Gaussians3D,
  camera: WSplatCamera,
  size: { width: number; height: number },
  options: GpuRunOptions,
): Promise<{ frame: WSplatFrame; stats?: WSplatStats }> {
  const rendererOptions: WSplatRendererOptions = {
    size,
    antialias: options.antialias,
    eps2d: options.eps2d,
  }
  const renderer = await createWSplatRenderer(device, rendererOptions)
  const t0 = performance.now()
  try {
    renderer.setGaussians(gaussians)
    renderer.setCamera(camera)
    if (options.sort !== false) renderer.sort()
    renderer.renderSplats()
    const frame = await renderer.readback()
    const t1 = performance.now()
    console.log(
      `      [耗时]${options.antialias === false ? " 无AA" : ""}${options.sort === false ? " 不排序" : ""} ${(t1 - t0).toFixed(0)}ms`,
    )
    let stats: WSplatStats | undefined
    if (options.stats) {
      renderer.countCulls()
      stats = await renderer.readSplatStats()
    }
    return { frame, stats }
  } finally {
    renderer.destroy()
  }
}

/** 用两个独立渲染的区间帧按 `over` 合成，与整帧逐像素比对。 */
async function ldiSplitCheck(
  device: GPUDevice,
  gaussians: Gaussians3D,
  camera: WSplatCamera,
  size: { width: number; height: number },
  split: number,
  full: WSplatFrame,
): Promise<{ ok: boolean; detail: string }> {
  const renderer = await createWSplatRenderer(device, { size })
  try {
    renderer.setGaussians(gaussians)
    renderer.setCamera(camera)
    renderer.sort()
    const total = renderer.permutationLength
    // 远端一半 [0, split)，近端一半 [split, total)，各自 clear 后渲染
    renderer.renderSplatsRange({ firstInstance: 0, instanceCount: split })
    const far = await renderer.readback()
    renderer.renderSplatsRange({
      firstInstance: split,
      instanceCount: total - split,
    })
    const near = await renderer.readback()

    const count = size.width * size.height
    // 直接比对 blend 累加器（A / ED / 预乘色），避免 resolve 除以 A 放大误差。
    let maxA = 0
    let maxEd = 0
    let maxPremul = 0
    let maxD = 0
    let maxRgb = 0
    for (let i = 0; i < count; i++) {
      const aFar = far.alpha[i]
      const aNear = near.alpha[i]
      const a = aNear + aFar * (1 - aNear)
      if (a <= 1e-6 || full.alpha[i] <= 1e-6) continue
      maxA = Math.max(maxA, Math.abs(a - full.alpha[i]))
      const ed =
        near.accumulatedDepth[i] + far.accumulatedDepth[i] * (1 - aNear)
      maxEd = Math.max(maxEd, Math.abs(ed - full.accumulatedDepth[i]))
      maxD = Math.max(maxD, Math.abs(ed / a - full.depth[i]))
      for (let c = 0; c < 3; c++) {
        const cPremul =
          near.rgb[i * 3 + c] * aNear + far.rgb[i * 3 + c] * aFar * (1 - aNear)
        maxPremul = Math.max(
          maxPremul,
          Math.abs(cPremul - full.rgb[i * 3 + c] * full.alpha[i]),
        )
        maxRgb = Math.max(maxRgb, Math.abs(cPremul / a - full.rgb[i * 3 + c]))
      }
    }
    const ok = maxA < 0.02 && maxEd < 0.05 && maxPremul < 0.03
    return {
      ok,
      detail:
        `max|ΔA|=${maxA.toExponential(2)} max|ΔED|=${maxEd.toExponential(2)} ` +
        `max|Δpremul|=${maxPremul.toExponential(2)} (导出量 max|ΔD|=${maxD.toExponential(2)} max|ΔRGB|=${maxRgb.toExponential(2)})`,
    }
  } finally {
    renderer.destroy()
  }
}

// ────────────────────────── 场景装配 ──────────────────────────

/** 读 PLY + 相机 json 并装配 `Gaussians3D` 与 `WSplatCamera`；文件缺失返回 undefined。 */
function loadScene(options: SceneOptions): Scene | undefined {
  const plyPath = resolve(REPO_ROOT, options.ply ?? "temp/test.ply")
  if (!existsSync(plyPath)) return undefined
  const cameraPath = resolve(
    REPO_ROOT,
    options.camera ?? plyPath.replace(/\.ply$/i, ".camera.json"),
  )

  const buf = readFileSync(plyPath)
  const ply = parsePly(
    new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength),
  )
  const vertex = ply.element("vertex")
  if (!vertex) throw new Error(`${plyPath} 缺少 vertex element`)
  const column = (name: string): Float64Array => {
    const c = vertex.columns.get(name)
    if (!c) throw new Error(`${plyPath} 缺少 vertex.${name}`)
    return c
  }

  const maxSplats = options.maxSplats
    ? Number.parseInt(options.maxSplats, 10)
    : Number.POSITIVE_INFINITY
  const count =
    maxSplats === Number.POSITIVE_INFINITY
      ? vertex.count
      : Math.min(vertex.count, maxSplats)

  const x = column("x")
  const y = column("y")
  const z = column("z")
  const opacity = column("opacity")
  const fdc = [column("f_dc_0"), column("f_dc_1"), column("f_dc_2")]
  const scaleCols = [column("scale_0"), column("scale_1"), column("scale_2")]
  const rot = [
    column("rot_0"),
    column("rot_1"),
    column("rot_2"),
    column("rot_3"),
  ]

  const gaussians: Gaussians3D = {
    meanVectors: new Float32Array(count * 3),
    singularValues: new Float32Array(count * 3),
    quaternions: new Float32Array(count * 4),
    colors: new Float32Array(count * 3),
    opacities: new Float32Array(count),
  }
  for (let i = 0; i < count; i++) {
    gaussians.meanVectors[i * 3] = x[i]
    gaussians.meanVectors[i * 3 + 1] = y[i]
    gaussians.meanVectors[i * 3 + 2] = z[i]
    for (let c = 0; c < 3; c++) {
      gaussians.singularValues[i * 3 + c] = Math.exp(scaleCols[c][i])
      // PLY 的 f_dc 已是 sRGB；Gaussians3D.colors 的语义是 linearRGB
      gaussians.colors[i * 3 + c] = srgbToLinear(
        clamp01(0.5 + SH_C0 * fdc[c][i]),
      )
    }
    for (let c = 0; c < 4; c++) gaussians.quaternions[i * 4 + c] = rot[c][i]
    gaussians.opacities[i] = 1 / (1 + Math.exp(-opacity[i]))
  }

  // ── 相机 ──
  let fx = 0
  let nativeWidth = 0
  let nativeHeight = 0
  let position: [number, number, number] = [0, 0, 0]
  let rotation:
    | [
        [number, number, number],
        [number, number, number],
        [number, number, number],
      ]
    | undefined
  if (existsSync(cameraPath)) {
    const poses = JSON.parse(readFileSync(cameraPath, "utf8")) as {
      fx: number
      width: number
      height: number
      position?: [number, number, number]
      rotation?: [
        [number, number, number],
        [number, number, number],
        [number, number, number],
      ]
    }[]
    const pose = poses[0]
    if (!pose || !Number.isFinite(pose.fx)) {
      throw new Error(`${cameraPath} 缺少 fx（SuperSplat 相机 json 格式）`)
    }
    fx = pose.fx
    nativeWidth = pose.width
    nativeHeight = pose.height
    if (pose.position) position = pose.position
    if (pose.rotation) rotation = pose.rotation
  } else {
    // 回退：PLY 自带的 intrinsic / image_size
    const intrinsic = ply.column("intrinsic", "intrinsic")
    const imageSize = ply.column("image_size", "image_size")
    if (!intrinsic || !imageSize) {
      throw new Error(`既没有 ${cameraPath}，PLY 里也没有 intrinsic/image_size`)
    }
    fx = intrinsic[0]
    nativeWidth = imageSize[0]
    nativeHeight = imageSize[1]
  }

  if (options.nativeSize) {
    nativeWidth = options.nativeSize.width
    nativeHeight = options.nativeSize.height
  }

  const width = options.width
  const height = Math.max(1, Math.round((width * nativeHeight) / nativeWidth))
  const pixelScale = width / nativeWidth

  // near/far 由点云视图 z 的 1%/99% 分位推（与相机朝向无关，这里相机朝 +z）
  const zs = new Float32Array(count)
  for (let i = 0; i < count; i++) zs[i] = gaussians.meanVectors[i * 3 + 2]
  zs.sort()
  const at = (q: number): number =>
    zs[Math.min(count - 1, Math.max(0, Math.floor(count * q)))]
  const near = Math.max(0.01, at(0.01) * 0.5)
  const far = Math.max(near * 4, at(0.99) * 2)

  const camera = createWSplatCamera({
    intrinsics: { focalLengthPx: fx * pixelScale, width, height },
    position,
    rotation,
    near,
    far,
  })

  return {
    gaussians,
    camera,
    width,
    height,
    fx: fx * pixelScale,
    count,
    plyPath,
  }
}

function srgbToLinear(x: number): number {
  return x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x
}

// ────────────────────────── 度量 ──────────────────────────

function toGray(
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

function maskedNcc(a: Float32Array, b: Float32Array, mask: Uint8Array): number {
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

function maskedMae(a: Float32Array, b: Float32Array, mask: Uint8Array): number {
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

/** 对齐掩码 = 两边任一可见。 */
function buildMask(
  frame: { visible: Uint8Array },
  reference: { visible: Uint8Array },
): Uint8Array {
  const mask = new Uint8Array(frame.visible.length)
  for (let i = 0; i < mask.length; i++) {
    mask[i] = frame.visible[i] || reference.visible[i] ? 1 : 0
  }
  return mask
}

function depthStats(
  frame: { depth: Float32Array; visible: Uint8Array },
  reference: CpuRenderResult,
  count: number,
): { medianRel: number; p95Rel: number; visibleMismatch: number } {
  const rel: number[] = []
  let visibleMismatch = 0
  for (let i = 0; i < count; i++) {
    if (frame.visible[i] !== reference.visible[i]) {
      visibleMismatch++
      continue
    }
    if (!frame.visible[i]) continue
    const b = reference.depth[i]
    if (!(b > 0)) continue
    rel.push(Math.abs(frame.depth[i] - b) / b)
  }
  rel.sort((x, y) => x - y)
  return {
    medianRel: rel.length === 0 ? Number.NaN : rel[Math.floor(rel.length / 2)],
    p95Rel: rel.length === 0 ? Number.NaN : rel[Math.floor(rel.length * 0.95)],
    visibleMismatch,
  }
}

/** 帧的确定性指纹（逐位）。 */
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

// ═══════════════════ [F] 视觉对比（仅观测，不判定） ═══════════════════

interface VisualOptions {
  /** 渲染宽度（高度按原图比例推）。 */
  width: number
  /** 输出目录（相对仓库根）。 */
  out?: string
}

/**
 * 用 `sharp-compare-fixtures.ts` 的 fixtures 渲染一帧并出图。
 *
 * ⚠ **只打印、只写 PNG，不做任何判定**：这里比的是「SHARP 重建出的 splat 渲染」
 * 与「原始输入照片」，差异来自重建本身（位姿 / FOV / 高斯拟合），不是管线正确性。
 * 判定仍然只看 [A]~[E]。
 *
 * `reference.ply` 的 `image_size` / 主点是交换过的（fixture 生成时的旧口径），
 * 所以这里用原图的实际宽高覆盖 native 尺寸，主点按图像中心取（本来就是居中的）。
 */
async function visualComparison(
  device: GPUDevice,
  options: VisualOptions,
): Promise<void> {
  const plyPath = resolve(REPO_ROOT, FIXTURE_PLY)
  if (!existsSync(plyPath) || !existsSync(FIXTURE_IMAGE)) {
    console.log(
      `  (未找到 fixture，跳过 [F])\n    ply:   ${plyPath}\n    image: ${FIXTURE_IMAGE}`,
    )
    return
  }

  const photo = await loadImage(FIXTURE_IMAGE)
  const imgW = photo.image.width
  const imgH = photo.image.height
  const scene = loadScene({
    width: options.width,
    ply: FIXTURE_PLY,
    nativeSize: { width: imgW, height: imgH },
  })
  if (!scene) {
    console.log(`  (fixture PLY 无法加载，跳过 [F]): ${plyPath}`)
    return
  }
  const { gaussians, camera, width, height, fx } = scene
  console.log(`      fixture ${FIXTURE_PLY}  ${scene.count} 高斯`)
  console.log(
    `      原图 ${imgW}x${imgH}  渲染 ${width}x${height}  fx=${fx.toFixed(2)}`,
  )
  const count = width * height
  // 原图 -> 渲染分辨率（只看图，lanczos 就够，不做像素网格对齐）
  const photoRgba = await resizeRgba(photo.image, width, height)
  const photoRgb = rgba8ToRgb8(photoRgba, count)

  // ml-sharp 的官方渲染 = gsplat `rasterize_mode="classic"` + `eps2d=0`（见 utils/gsplat.py）。
  // 这里做 2×2 对照：透明度补偿（AA）× 低通模糊（eps2d），看哪一组最接近原图。
  const variants = [
    { label: "AA on  eps 0.3（gsplat 默认）", antialias: true, eps2d: 0.3 },
    { label: "AA off eps 0.3（classic+模糊）", antialias: false, eps2d: 0.3 },
    { label: "AA off eps 0.0（ml-sharp）", antialias: false, eps2d: 0.0 },
    { label: "AA on  eps 0.0", antialias: true, eps2d: 0.0 },
  ]
  interface VariantResult {
    label: string
    frame: WSplatFrame
    rgb: Uint8Array
    ncc: number
    mae: number
    psnr: number
    ssim: number
    brightness: number
    covered: number
  }
  const results: VariantResult[] = []
  for (const v of variants) {
    const { frame } = await runGpu(
      device,
      gaussians,
      camera,
      { width, height },
      { antialias: v.antialias, eps2d: v.eps2d },
    )
    const rgb = rgba8ToRgb8(frame.preview, count)
    const mask = new Uint8Array(count)
    let covered = 0
    for (let i = 0; i < count; i++) {
      if (frame.alpha[i] > 0.5) {
        mask[i] = 1
        covered++
      }
    }
    results.push({
      label: v.label,
      frame,
      rgb,
      ncc: maskedNcc8(rgb, photoRgb, mask),
      mae: maskedMae8(rgb, photoRgb, mask),
      // PSNR / SSIM 是 3DGS 论文的标准指标，全图算（不做掩码）；本场景覆盖 100%，两者一致
      psnr: psnr8(rgb, photoRgb, count),
      ssim: ssimGray8(rgb, photoRgb, width, height),
      brightness: brightnessRatioLinear(frame.rgb, photoRgb, mask, count),
      covered,
    })
  }
  console.log(
    `      覆盖(A>0.5) ${results[0].covered}/${count} = ${((results[0].covered / count) * 100).toFixed(2)}%`,
  )
  console.log("      [观测] vs 原图（仅观测，不判定）：")
  for (const r of results) {
    console.log(
      `        ${r.label.padEnd(26)} NCC ${fmtNum(r.ncc, 4)}  MAE ${fmtNum(r.mae, 2)}/255  ` +
        `PSNR ${fmtNum(r.psnr, 2)} dB  SSIM ${fmtNum(r.ssim, 4)}  亮度 ${fmtNum(r.brightness, 4)}`,
    )
  }
  console.log(
    "      （数字反映重建保真度；PSNR/SSIM 按 3DGS 惯例算全图，差异集中在发丝 / 透明边缘 / 远景）",
  )

  // ── 写图：原图 | AA on eps0.3 | ml-sharp(classic+0) | 差异 ──
  const outDir = resolve(REPO_ROOT, options.out ?? "temp/wsplat-check")
  mkdirSync(outDir, { recursive: true })
  const mlsharp = results[2]
  const photoPng = await rgbaToPngBuffer(photoRgba, width, height)
  const aaPng = await rgbaToPngBuffer(results[0].frame.preview, width, height)
  const mlsharpPng = await rgbaToPngBuffer(mlsharp.frame.preview, width, height)
  const mlsharpDiff = diffRgba8(mlsharp.rgb, photoRgb, count)
  const diffPng = await rgbaToPngBuffer(mlsharpDiff, width, height)
  await sharpFromRgba(photoRgba, width, height)
    .png()
    .toFile(resolve(outDir, "photo.png"))
  await sharpFromRgba(results[0].frame.preview, width, height)
    .png()
    .toFile(resolve(outDir, "render-aa.png"))
  await sharpFromRgba(mlsharp.frame.preview, width, height)
    .png()
    .toFile(resolve(outDir, "render-mlsharp.png"))
  await sharpFromRgba(mlsharpDiff, width, height)
    .png()
    .toFile(resolve(outDir, "diff-mlsharp.png"))
  await sharpFromRgba(depthGrayRgba(mlsharp.frame, count), width, height)
    .png()
    .toFile(resolve(outDir, "depth.png"))
  await sharpFromRgba(alphaGrayRgba(mlsharp.frame, count), width, height)
    .png()
    .toFile(resolve(outDir, "alpha.png"))
  const comparePath = resolve(outDir, "compare.png")
  await createSharpCanvas(width * 4, height)
    .composite([
      { input: photoPng, left: 0, top: 0 },
      { input: aaPng, left: width, top: 0 },
      { input: mlsharpPng, left: width * 2, top: 0 },
      { input: diffPng, left: width * 3, top: 0 },
    ])
    .png()
    .toFile(comparePath)
  console.log(
    `      [png] 原图 | AA(eps0.3) | ml-sharp(eps0) | 差异 = ${comparePath}`,
  )
  console.log(
    "      [png] 单图: photo.png / render-aa.png / render-mlsharp.png / diff-mlsharp.png / depth.png / alpha.png",
  )
}

/** 用 sharp 把 RGB(A) 原图缩放到目标尺寸，输出 RGBA8。 */
interface RawImage {
  data: Uint8Array | Float32Array
  width: number
  height: number
  channels: number
}
interface SharpResize {
  resize(opts: { width: number; height: number; kernel: string }): SharpResize
  ensureAlpha(): SharpResize
  raw(): SharpResize
  toBuffer(opts: { resolveWithObject: true }): Promise<{ data: Buffer }>
}
const requireCjs = createRequire(import.meta.url)

async function resizeRgba(
  img: RawImage,
  width: number,
  height: number,
): Promise<Uint8Array> {
  const sharp = requireCjs("sharp") as (
    input: Buffer,
    opts: { raw: { width: number; height: number; channels: number } },
  ) => SharpResize
  const { data } = await sharp(
    Buffer.from(img.data.buffer, img.data.byteOffset, img.data.byteLength),
    { raw: { width: img.width, height: img.height, channels: img.channels } },
  )
    .resize({ width, height, kernel: "lanczos3" })
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true })
  return new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
}

/** RGBA8 -> RGB8。 */
function rgba8ToRgb8(rgba: Uint8Array, count: number): Uint8Array {
  const out = new Uint8Array(count * 3)
  for (let i = 0; i < count; i++) {
    out[i * 3] = rgba[i * 4]
    out[i * 3 + 1] = rgba[i * 4 + 1]
    out[i * 3 + 2] = rgba[i * 4 + 2]
  }
  return out
}

/** 灰度 8bit 的掩码 NCC。 */
function maskedNcc8(a: Uint8Array, b: Uint8Array, mask: Uint8Array): number {
  const n = mask.length
  const ga = new Float32Array(n)
  const gb = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    ga[i] = 0.299 * a[i * 3] + 0.587 * a[i * 3 + 1] + 0.114 * a[i * 3 + 2]
    gb[i] = 0.299 * b[i * 3] + 0.587 * b[i * 3 + 1] + 0.114 * b[i * 3 + 2]
  }
  return maskedNcc(ga, gb, mask)
}

/** RGB8 平均绝对误差（0-255 标度）。 */
function maskedMae8(a: Uint8Array, b: Uint8Array, mask: Uint8Array): number {
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

/** sRGB 8bit RGB 的 PSNR（0-255 域，全图）。 */
function psnr8(a: Uint8Array, b: Uint8Array, count: number): number {
  let mse = 0
  for (let i = 0; i < count; i++) {
    for (let c = 0; c < 3; c++) {
      const d = a[i * 3 + c] - b[i * 3 + c]
      mse += d * d
    }
  }
  mse /= count * 3
  return mse <= 0
    ? Number.POSITIVE_INFINITY
    : 10 * Math.log10((255 * 255) / mse)
}

/**
 * 灰度 SSIM（标准 11×11 高斯窗，sigma=1.5，C1=(0.01)², C2=(0.03)²）。
 *
 * 3DGS 论文的通用指标；这里自己实现以免引入 sharp/额外依赖（窗口卷积 O(n·121)）。
 * 只统计窗内像素（略去 5 像素边框）—— 与常见实现的边界处理差异在 1e-4 量级。
 */
function ssimGray8(
  a: Uint8Array,
  b: Uint8Array,
  width: number,
  height: number,
): number {
  const n = width * height
  const ga = new Float64Array(n)
  const gb = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    ga[i] =
      (0.299 * a[i * 3] + 0.587 * a[i * 3 + 1] + 0.114 * a[i * 3 + 2]) / 255
    gb[i] =
      (0.299 * b[i * 3] + 0.587 * b[i * 3 + 1] + 0.114 * b[i * 3 + 2]) / 255
  }

  const R = 5
  const sigma = 1.5
  const side = 2 * R + 1
  const kernel = new Float64Array(side * side)
  let ksum = 0
  for (let dy = -R; dy <= R; dy++) {
    for (let dx = -R; dx <= R; dx++) {
      const v = Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))
      kernel[(dy + R) * side + (dx + R)] = v
      ksum += v
    }
  }
  for (let i = 0; i < kernel.length; i++) kernel[i] /= ksum

  const C1 = 0.01 * 0.01
  const C2 = 0.03 * 0.03
  let sum = 0
  let cnt = 0
  for (let y = R; y < height - R; y++) {
    for (let x = R; x < width - R; x++) {
      let muA = 0
      let muB = 0
      let va = 0
      let vb = 0
      let cov = 0
      let ki = 0
      for (let dy = -R; dy <= R; dy++) {
        const row = (y + dy) * width + x
        for (let dx = -R; dx <= R; dx++) {
          const w = kernel[ki++]
          muA += w * ga[row + dx]
          muB += w * gb[row + dx]
        }
      }
      ki = 0
      for (let dy = -R; dy <= R; dy++) {
        const row = (y + dy) * width + x
        for (let dx = -R; dx <= R; dx++) {
          const w = kernel[ki++]
          const da = ga[row + dx] - muA
          const db = gb[row + dx] - muB
          va += w * da * da
          vb += w * db * db
          cov += w * da * db
        }
      }
      sum +=
        ((2 * muA * muB + C1) * (2 * cov + C2)) /
        ((muA * muA + muB * muB + C1) * (va + vb + C2))
      cnt++
    }
  }
  return cnt > 0 ? sum / cnt : Number.NaN
}

/** 线性域平均亮度比（渲染 / 原图，掩码内）。 */
function brightnessRatioLinear(
  render: Float32Array,
  photo: Uint8Array,
  mask: Uint8Array,
  count: number,
): number {
  let rm = 0
  let pm = 0
  let n = 0
  for (let i = 0; i < count; i++) {
    if (!mask[i]) continue
    rm += (render[i * 3] + render[i * 3 + 1] + render[i * 3 + 2]) / 3
    pm +=
      (srgbToLinear(photo[i * 3] / 255) +
        srgbToLinear(photo[i * 3 + 1] / 255) +
        srgbToLinear(photo[i * 3 + 2] / 255)) /
      3
    n++
  }
  return n > 0 ? rm / Math.max(pm, 1e-9) : Number.NaN
}

/** 差异图（每通道 |Δ|×3，便于目视）。 */
function diffRgba8(a: Uint8Array, b: Uint8Array, count: number): Uint8Array {
  const out = new Uint8Array(count * 4)
  for (let i = 0; i < count; i++) {
    for (let c = 0; c < 3; c++) {
      const d = Math.abs(a[i * 3 + c] - b[i * 3 + c]) * 3
      out[i * 4 + c] = d > 255 ? 255 : d
    }
    out[i * 4 + 3] = 255
  }
  return out
}

/** 深度灰度图（可见像素按 p1~p99 归一化，近处亮）。 */
function depthGrayRgba(frame: WSplatFrame, count: number): Uint8Array {
  const depths: number[] = []
  for (let i = 0; i < count; i++) {
    if (frame.visible[i] && frame.depth[i] > 0) depths.push(frame.depth[i])
  }
  depths.sort((a, b) => a - b)
  const at = (q: number): number =>
    depths.length === 0
      ? 0
      : depths[Math.min(depths.length - 1, Math.floor(depths.length * q))]
  const p1 = at(0.01)
  const p99 = at(0.99)
  const span = Math.max(p99 - p1, 1e-9)
  const out = new Uint8Array(count * 4)
  for (let i = 0; i < count; i++) {
    let v = 0
    if (frame.visible[i] && frame.depth[i] > 0) {
      const t = Math.min(1, Math.max(0, (frame.depth[i] - p1) / span))
      v = Math.round(255 * (1 - t))
    }
    out[i * 4] = v
    out[i * 4 + 1] = v
    out[i * 4 + 2] = v
    out[i * 4 + 3] = 255
  }
  return out
}

/** alpha 灰度图。 */
function alphaGrayRgba(frame: WSplatFrame, count: number): Uint8Array {
  const out = new Uint8Array(count * 4)
  for (let i = 0; i < count; i++) {
    const v = Math.round(Math.min(1, Math.max(0, frame.alpha[i])) * 255)
    out[i * 4] = v
    out[i * 4 + 1] = v
    out[i * 4 + 2] = v
    out[i * 4 + 3] = 255
  }
  return out
}

/** sRGB(0-1) -> 线性。 */
function fmtNum(x: number, digits: number): string {
  return Number.isFinite(x) ? x.toFixed(digits) : "n/a"
}

main().catch((err) => {
  console.error("\n[错误]", err instanceof Error ? err.message : err)
  if (err instanceof Error && err.stack) console.error(err.stack)
  process.exitCode = 1
})
