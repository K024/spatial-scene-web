/**
 * 单元测试：**协方差雅可比第二行符号**（相机 y 向下 vs 向上）。
 *
 * 阶段：**wsplat 渲染器**（`scripts/wsplat-*`：渲染 / 自检 / 质量门）。
 *
 * ── 为什么需要这个测试 ──
 * `gsplatCorner` 里 `J` 的第二行符号只影响 2D 协方差的 **off-diagonal**，
 * 也就是「椭圆往哪边倾斜」。各向同性或轴对齐的高斯完全看不出差别，
 * 真实数据上的全局 NCC 差异只有 0.3%（实测 0.972 vs 0.969），
 * **不足以判定**——所以这里造一个各向异性且 45° 倾斜的合成高斯，
 * 直接测「屏幕足迹的倾斜方向」与 CPU 参考是否一致。
 *
 * 合成设置（世界 = 相机坐标系，镜头在原点朝 +z）：
 *   - 单个高斯放在 `(0, 0, z)`，尺度 `(长, 短, 短)`，四元数 = 绕 z 轴转 45°；
 *   - 投影后 2D 椭圆的 **off-diagonal 必须为正**（在 u 右 / v 下的像素坐标里，
 *     长轴指向右下）。这个符号是可解析预测的，不受渲染近似影响。
 *   - 度量：对 alpha 足迹求二阶矩 `cov_uv = Σα·du·dv / Σα`，看它的符号。
 *
 * 结论（运行后打印）：`cameraYAxis` 必须与相机/NDC 约定匹配，
 * 默认 `"down"`（OpenCV，SHARP 度量空间）应当给出正的 `cov_uv`。
 *
 * 用法：`npx tsx scripts/wsplat-check-corner.ts`
 */

import type { Gaussians3D } from "../src/spatial-scene/sharp/types.ts"
import { createWSplatCamera } from "../src/spatial-scene/wsplat/camera.ts"
import {
  createWSplatRenderer,
  type WSplatCameraYAxis,
} from "../src/spatial-scene/wsplat/index.ts"
import { withNodeDevice } from "./utils/webgpu.ts"
import { renderSplatsCpu } from "./utils/wsplat-cpu.ts"

const SIZE = 192
const FOCAL = 200
const Z = 2
const LONG = 0.05
const SHORT = 0.015
/** 绕 z 轴 45°：`(w, x, y, z)`。 */
const QUAT_45 = [Math.cos(Math.PI / 8), 0, 0, Math.sin(Math.PI / 8)]
const NEAR = 0.5
const FAR = 10

async function main(): Promise<void> {
  console.log("=".repeat(72))
  console.log("wsplat 协方差雅可比 y 行符号单元测试")
  console.log("=".repeat(72))

  const intrinsics = { focalLengthPx: FOCAL, width: SIZE, height: SIZE }
  const nearFar = { near: NEAR, far: FAR }

  // 合成：一个 45° 倾斜的长条高斯，放在光轴上（此时 J 的透视项为 0，
  // 椭圆倾斜完全由 3D 协方差旋转决定，符号可解析预测）
  const gaussians: Gaussians3D = {
    meanVectors: new Float32Array([0, 0, Z]),
    singularValues: new Float32Array([LONG, SHORT, SHORT]),
    quaternions: new Float32Array(QUAT_45),
    colors: new Float32Array([1, 1, 1]),
    opacities: new Float32Array([1]),
  }

  // CPU 参考（真值方向）
  const cpu = renderSplatsCpu({
    gaussians,
    camera: {
      viewMatrix: createWSplatCamera({ intrinsics, ...nearFar }).viewMatrix,
      fx: FOCAL,
      fy: FOCAL,
      cx: SIZE / 2,
      cy: SIZE / 2,
      width: SIZE,
      height: SIZE,
    },
  })
  const cpuMoment = footprintMoment(cpu.alpha, SIZE)
  console.log(
    `CPU 参考      : cov_uv = ${fmt(cpuMoment.cov)}  (α 覆盖 ${cpuMoment.count} px)`,
  )

  const results = await withNodeDevice(async (device) => {
    const out: { axis: WSplatCameraYAxis; cov: number; count: number }[] = []
    for (const axis of ["down", "up"] as const) {
      const renderer = await createWSplatRenderer(device, {
        size: { width: SIZE, height: SIZE },
        cameraYAxis: axis,
      })
      renderer.setGaussians(gaussians)
      renderer.setCamera(createWSplatCamera({ intrinsics, ...nearFar }))
      renderer.renderSplats()
      const frame = await renderer.readback()
      const moment = footprintMoment(frame.alpha, SIZE)
      out.push({ axis, cov: moment.cov, count: moment.count })
      renderer.destroy()
    }
    return out
  })

  const problems: string[] = []
  const cpuSign = Math.sign(cpuMoment.cov)
  for (const r of results) {
    const matches = Math.sign(r.cov) === cpuSign
    console.log(
      `${matches ? "ok  " : "--  "} yAxis=${r.axis.padEnd(5)}: cov_uv = ${fmt(r.cov)}  (α 覆盖 ${r.count} px)  ${matches ? "与 CPU 一致" : "与 CPU 相反（预期如此）"}`,
    )
  }

  // 解析预测：45° 倾斜 + (长,短,短) => 像素空间 off-diagonal > 0
  if (!(cpuMoment.cov > 0)) {
    problems.push(
      `CPU 参考的 cov_uv 不是正数（${cpuMoment.cov}），合成设置有问题`,
    )
  }

  // 默认约定（本项目用 OpenCV y 向下）必须与 CPU 一致
  const down = results.find((r) => r.axis === "down")
  const up = results.find((r) => r.axis === "up")
  if (!down || Math.sign(down.cov) !== cpuSign) {
    problems.push(
      `默认约定 cameraYAxis="down" 与 CPU 参考相反（cov_uv=${down?.cov}）`,
    )
  }
  // 反向约定必须**不**一致，否则说明这个测试没有区分能力（off-diagonal 太小）
  if (up && Math.sign(up.cov) === cpuSign) {
    problems.push(
      'cameraYAxis="up" 也与 CPU 一致 ⇒ 本测试无区分能力（椭圆倾斜太小），需调大倾斜',
    )
  }

  console.log()
  if (problems.length > 0) {
    console.error("[FAIL] 协方差方向检查未通过：")
    for (const p of problems) console.error(`  - ${p}`)
    process.exitCode = 1
    return
  }
  console.log(
    '[OK] 默认约定 cameraYAxis="down" 与 CPU 参考一致，且反向约定被证伪',
  )
}

/** alpha 足迹的二阶矩（相对质心）。`cov = Σα·du·dv / Σα`。 */
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

function fmt(x: number): string {
  return Number.isFinite(x) ? x.toFixed(4) : String(x)
}

main().catch((err) => {
  console.error("\n[错误]", err instanceof Error ? err.message : err)
  if (err instanceof Error && err.stack) console.error(err.stack)
  process.exitCode = 1
})
