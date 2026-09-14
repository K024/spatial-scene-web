/**
 * splat 渲染链路（node + Dawn）通路自检。
 *
 * 阶段：**wsplat 渲染器**（`scripts/wsplat-*`：渲染 / 自检 / 质量门）。
 *
 * 验证的是输出契约本身（`src/spatial-scene/wsplat/types.ts` 的 `WSplatFrame`），
 * 而不是画质：
 *   1. 两个 `rgba16float` 颜色附件 + 预乘 blend（`src=one / dst=oneMinusSrcAlpha`）可建可跑；
 *   2. 附件 1 的 α 通道参与 blend（否则深度不随前面的层衰减）——这里用两次叠加验证；
 *   3. `resolve` 出来的 A / T / D / ED / 直通 RGB 与解析解逐个对照；
 *   4. 预览 PNG 写到 `py-models/out/wsplat-check/`，供目视。
 *
 * 解析解：先画 α=0.5、z=2 的「远 splat」，再画 α=0.5、z=1 的「近 splat」（back-to-front）：
 *   A      = 1 - (1-0.5)(1-0.5) = 0.75
 *   T      = 1 - A = 0.25
 *   premul = Σ cᵢ·αᵢ·Tᵢ ，其中 Tᵢ = Π_{更近的 j}(1-αⱼ)（gsplat 语义）
 *          = c_far·0.5·(1-0.5) + c_near·0.5·1
 *   ED     = 2·0.5·(1-0.5) + 1·0.5·1 = 1.0
 *   D      = ED / A = 4/3
 *
 * ⚠ 注意衰减方向：**远的**那一层要被**近的** α 衰减（近处半透明物挡掉远处一半光），
 * 不是反过来（第一版脚本就在这里写错了，实现对、脚本错）。
 *
 * 用法：`npx tsx scripts/wsplat-check-blit.ts [--width N] [--height N]`
 */

import { mkdirSync } from "node:fs"
import { createRequire } from "node:module"
import { resolve } from "node:path"
import { parseArgs } from "node:util"

import { createWSplatRenderer } from "../src/spatial-scene/wsplat/index.ts"
import { REPO_ROOT } from "./utils/common.ts"
import { withNodeDevice } from "./utils/webgpu.ts"

const FAR = { color: [0.8, 0.2, 0.05] as const, alpha: 0.5, depth: 2 }
const NEAR = { color: [0.1, 0.4, 0.9] as const, alpha: 0.5, depth: 1 }

const CLI_OPTIONS = {
  width: { type: "string" },
  height: { type: "string" },
  out: { type: "string" },
} as const

async function main(): Promise<void> {
  const args = parseArgs({
    args: process.argv.slice(2),
    options: CLI_OPTIONS,
    allowPositionals: false,
    allowNegative: true,
    strict: true,
  })
  const width = Number.parseInt(args.values.width ?? "256", 10)
  const height = Number.parseInt(args.values.height ?? "256", 10)

  console.log("=".repeat(72))
  console.log("wsplat 通路自检（blit + resolve 解析解）")
  console.log("=".repeat(72))

  const problems = await withNodeDevice((device) =>
    renderCheck(device, width, height, args.values.out),
  )

  console.log()
  if (problems.length > 0) {
    console.error("[FAIL] 通路自检未通过：")
    for (const p of problems) console.error(`  - ${p}`)
    process.exitCode = 1
    return
  }
  console.log("[OK] 通路自检通过")
}

/** 真正的检查逻辑（device 生命周期由 withNodeDevice 管）。 */
async function renderCheck(
  device: GPUDevice,
  width: number,
  height: number,
  outDirArg: string | undefined,
): Promise<string[]> {
  const problems: string[] = []
  const renderer = await createWSplatRenderer(device, {
    size: { width, height },
  })

  // 远 → 近（back-to-front，输出契约规定的 over 方向）
  renderer.clearSplatBuffers()
  renderer.renderSolidColor([...FAR.color, FAR.alpha], FAR.depth)
  renderer.renderSolidColor([...NEAR.color, NEAR.alpha], NEAR.depth)

  const frame = await renderer.readback()

  // ── 解析解 ──
  const aFar = FAR.alpha
  const aNear = NEAR.alpha
  const tFar = 1 - aNear
  const tNear = 1
  const expectedA = 1 - (1 - aFar) * (1 - aNear)
  const expectedT = 1 - expectedA
  const expectedEd = FAR.depth * aFar * tFar + NEAR.depth * aNear * tNear
  const expectedD = expectedEd / expectedA
  const expectedPremul = [0, 1, 2].map(
    (c) => FAR.color[c] * aFar * tFar + NEAR.color[c] * aNear * tNear,
  )
  const expectedStraight = expectedPremul.map((v) => v / expectedA)

  const center = Math.floor(height / 2) * width + Math.floor(width / 2)
  const actual = {
    alpha: frame.alpha[center],
    transmission: frame.transmission[center],
    depth: frame.depth[center],
    accumulatedDepth: frame.accumulatedDepth[center],
    straight: [0, 1, 2].map((c) => frame.rgb[center * 3 + c]),
  }

  console.log("\n逐项对照（画面中心像素）")
  const rows: [string, number, number][] = [
    ["A (累积 alpha)", expectedA, actual.alpha],
    ["T (透射率)", expectedT, actual.transmission],
    ["ED (未归一化)", expectedEd, actual.accumulatedDepth],
    ["D (expected depth)", expectedD, actual.depth],
    ["straight R", expectedStraight[0], actual.straight[0]],
    ["straight G", expectedStraight[1], actual.straight[1]],
    ["straight B", expectedStraight[2], actual.straight[2]],
  ]
  for (const [label, want, got] of rows) {
    const diff = Math.abs(want - got)
    const rel = want !== 0 ? diff / Math.abs(want) : diff
    const ok = rel < 4e-3 || diff < 4e-3
    console.log(
      `  ${ok ? "ok  " : "FAIL"} ${label.padEnd(20)} want=${want.toFixed(6)} got=${got.toFixed(6)} (rel ${rel.toExponential(2)})`,
    )
    if (!ok) problems.push(`${label}: want ${want}, got ${got}`)
  }

  // ── 背景语义：清空后不画任何东西，A/T/D 必须分别是 0/1/0 且 visible=0 ──
  renderer.clearSplatBuffers()
  const empty = await renderer.readback()
  const bg = 0
  const bgOk =
    empty.alpha[bg] === 0 &&
    empty.transmission[bg] === 1 &&
    empty.depth[bg] === 0 &&
    empty.visible[bg] === 0
  console.log(
    `  ${bgOk ? "ok  " : "FAIL"} 背景（清空不画）      A=${empty.alpha[bg]} T=${empty.transmission[bg]} D=${empty.depth[bg]} visible=${empty.visible[bg]}`,
  )
  if (!bgOk) problems.push("清空后背景语义不对（应为 A=0 T=1 D=0 visible=0）")

  // ── 预览 PNG ──
  const outDir = resolve(REPO_ROOT, outDirArg ?? "py-models/out/wsplat-check")
  mkdirSync(outDir, { recursive: true })
  const previewPath = resolve(outDir, "m0-solid.png")
  await sharpFromRgba(frame.preview, width, height).png().toFile(previewPath)
  console.log(`\n[png] 预览 ${previewPath}`)

  // ── 前景像素统计（防止整屏都是背景而中心恰好对） ──
  let covered = 0
  for (let i = 0; i < frame.visible.length; i++) covered += frame.visible[i]
  console.log(`[stat] 可见像素 ${covered}/${width * height}`)
  if (covered !== width * height) {
    problems.push(`可见像素 ${covered} != ${width * height}（整屏应当被覆盖）`)
  }

  renderer.destroy()
  return problems
}

const requireCjs = createRequire(import.meta.url)

/** 用 sharp 把 RGBA8 原始像素编码成 PNG。 */
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
