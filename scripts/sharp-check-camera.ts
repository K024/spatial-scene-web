/**
 * 相机参数核对：只对参数，不含任何渲染器。
 *
 * 阶段：**SHARP 离线管线**（`scripts/sharp-*`：推理 / 导出 / 自检）。
 *
 * ── 这个脚本现在只回答一个问题 ──
 * 「`.camera.json` 里的 `position` / `rotation` / `fx` / `fy` 与
 * `src/spatial-scene/wsplat/camera.ts` 的相机约定是否自洽、且能把点云放进视锥？」
 *
 * 它**不再**渲染、**不再**与原图比 MAE/NCC。原来那个临时点云 z-buffer
 * （无椭圆核、无排序、黑噪点）已删除，理由：
 *   - 它给出的「NCC vs 原图」只是相机对齐的**必要条件**，不能证明 splat 管线对；
 *   - 真正的对齐证据现在由 wsplat 自己的通路给出 —— 见 `scripts/wsplat-golden.ts`
 *     的 `--compare-image`（GPU 渲染 vs 原图 NCC）与 `scripts/wsplat-render.ts` 的目视产物。
 *
 * ── 校验项（每条都打印数值，失败则退出码 1）──
 *   1. json 形状合法（非空数组 + position/rotation/fx/fy/width/height 齐备）
 *   2. json 记录的尺寸 == 输入图（含 `--crop`）尺寸
 *   3. `rotation` 是正交阵（`max|RᵗR - I|`）
 *   4. `det(R) = +1`（不是反射）
 *   5. `fx == fy > 0`（SHARP 的方形像素假设）
 *   6. `viewMatrix` 的旋转部分 == `rotationᵗ`，且 `view · position == 0`
 *      （把相机中心映射到原点）
 *   7. `fovY` 与 wsplat 相机模块同式
 *   8. 探针点（包围盒 8 角 + 质心）全部在相机前方（`z_view > 0`），质心投影在图内
 *   9. 深度往返：`ndcDepthToViewDepth(viewDepthToNdcDepth(z)) == z`
 *      —— 下游「投影随输出一起导出」契约要求可逆的那个式子（见 `types.ts` 的 `WSplatFrame`）
 *
 * 用法：
 *   npx tsx scripts/sharp-check-camera.ts
 *   npx tsx scripts/sharp-check-camera.ts --ply a.ply --camera a.camera.json --image X.jpg
 *   npx tsx scripts/sharp-check-camera.ts --crop 0,0,545,748   # 图被裁过时对齐尺寸
 *   npx tsx scripts/sharp-check-camera.ts --near 0.5 --far 6   # 覆盖默认平面（默认由点云分位数推）
 */

import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { parseArgs } from "node:util"

import type {
  Mat3Tuple,
  SuperSplatCameraPose,
  Vec3Tuple,
} from "../src/spatial-scene/export/camera.ts"
import { createWSplatCamera } from "../src/spatial-scene/wsplat/camera.ts"
import { DEFAULT_IMAGE, REPO_ROOT } from "./utils/common.ts"
import { loadImage } from "./utils/image.ts"
import { readPlyGaussians } from "./utils/ply.ts"

const CLI_OPTIONS = {
  ply: { type: "string" },
  camera: { type: "string" },
  image: { type: "string" },
  crop: { type: "string" },
  near: { type: "string" },
  far: { type: "string" },
} as const

/** 逐条累积的问题；空数组 = 全部通过。 */
const problems: string[] = []

function check(label: string, ok: boolean, detail: string): void {
  console.log(`  ${ok ? "✓" : "✗"} ${label.padEnd(34)} ${detail}`)
  if (!ok) problems.push(`${label}: ${detail}`)
}

async function main(): Promise<void> {
  const args = parseArgs({
    args: process.argv.slice(2),
    options: CLI_OPTIONS,
    allowPositionals: false,
    strict: true,
  })

  const plyPath = resolve(
    REPO_ROOT,
    args.values.ply ?? "py-models/out/ply/example.ply",
  )
  const cameraPath = resolve(
    REPO_ROOT,
    args.values.camera ?? plyPath.replace(/\.ply$/i, ".camera.json"),
  )
  const imagePath = resolve(REPO_ROOT, args.values.image ?? DEFAULT_IMAGE)
  const crop = parseCrop(args.values.crop)

  console.log("=".repeat(72))
  console.log("相机参数核对（只对参数，不渲染）")
  console.log("=".repeat(72))
  console.log(`ply    : ${plyPath}`)
  console.log(`camera : ${cameraPath}`)
  console.log(`image  : ${imagePath}`)

  // ── 1. json 形状 ──
  console.log("\n[1-2] json 与尺寸")
  const raw = JSON.parse(
    readFileSync(cameraPath, "utf8"),
  ) as SuperSplatCameraPose[]
  check(
    "json 是非空数组",
    Array.isArray(raw) && raw.length > 0,
    `长度 ${Array.isArray(raw) ? raw.length : "非数组"}`,
  )
  if (!Array.isArray(raw) || raw.length === 0) {
    report()
    return
  }
  if (raw.length > 1)
    console.log(`  [warn] json 含 ${raw.length} 条位姿，只核对第 1 条`)
  const pose = raw[0]

  const shapeOk =
    isVec3(pose.position) &&
    isMat3(pose.rotation) &&
    Number.isFinite(pose.fx) &&
    Number.isFinite(pose.fy) &&
    Number.isFinite(pose.width) &&
    Number.isFinite(pose.height)
  check(
    "字段齐备 (pos/rot/fx/fy/W/H)",
    shapeOk,
    shapeOk ? `第 1 条位姿完整` : `字段缺失或非数值`,
  )
  if (!shapeOk) {
    report()
    return
  }

  const loaded = await loadImage(imagePath, undefined, crop)
  const W = loaded.image.width
  const H = loaded.image.height
  check(
    "json 尺寸 == 输入图（含 crop）",
    pose.width === W && pose.height === H,
    `json ${pose.width}x${pose.height} vs image ${W}x${H}`,
  )

  // ── 3-5. 旋转与内参 ──
  console.log("\n[3-5] 旋转与内参")
  check(
    "rotation 正交 (max|RᵗR - I|)",
    orthonormalityError(pose.rotation) < 1e-4,
    orthonormalityError(pose.rotation).toExponential(3),
  )
  const det = det3(pose.rotation)
  check(
    "det(R) = +1（非反射）",
    Math.abs(det - 1) < 1e-4,
    `det = ${det.toFixed(8)}`,
  )
  check(
    "fx = fy > 0（方像素）",
    pose.fx > 0 && Math.abs(pose.fx - pose.fy) < 1e-6,
    `fx = ${pose.fx}  fy = ${pose.fy}`,
  )

  // ── 6-7. 与 wsplat 相机模块的一致性 ──
  console.log("\n[6-7] 与 wsplat 相机模块一致")
  const gauss = readPlyGaussians(plyPath)
  const zStats = viewZStats(gauss, pose)
  const near =
    args.values.near !== undefined
      ? Number.parseFloat(args.values.near)
      : Math.max(0.01, zStats.p01 * 0.5)
  const far =
    args.values.far !== undefined
      ? Number.parseFloat(args.values.far)
      : Math.max(near * 4, zStats.p99 * 2)

  const camera = createWSplatCamera({
    intrinsics: {
      focalLengthPx: pose.fx,
      width: pose.width,
      height: pose.height,
    },
    position: pose.position as Vec3Tuple,
    rotation: pose.rotation,
    near,
    far,
  })

  // view 的行主序 3x3 = 列主序数组里的 (row, col) = view[col*4 + row]
  const viewRot: Vec3Tuple[] = [
    [camera.viewMatrix[0], camera.viewMatrix[4], camera.viewMatrix[8]],
    [camera.viewMatrix[1], camera.viewMatrix[5], camera.viewMatrix[9]],
    [camera.viewMatrix[2], camera.viewMatrix[6], camera.viewMatrix[10]],
  ]
  // 期望：view 的 row i == rotation 的第 i 列
  let rotErr = 0
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      rotErr = Math.max(rotErr, Math.abs(viewRot[i][j] - pose.rotation[j][i]))
    }
  }
  check(
    "viewMatrix 旋转部分 == rotationᵗ",
    rotErr < 1e-6,
    `max|Δ| = ${rotErr.toExponential(3)}`,
  )

  const mapped = applyView(camera.viewMatrix, pose.position)
  const centerErr = Math.hypot(mapped[0], mapped[1], mapped[2])
  check(
    "view · position == 0（中心在原点）",
    centerErr < 1e-4,
    `|view·C| = ${centerErr.toExponential(3)} m`,
  )

  const fovYExpected = 2 * Math.atan(pose.height / (2 * pose.fy))
  check(
    "fovY == 2·atan(H / 2fy)",
    Math.abs(camera.fovY - fovYExpected) < 1e-9,
    `${((camera.fovY * 180) / Math.PI).toFixed(6)}°  (fx 域 fovX ${((camera.fovX * 180) / Math.PI).toFixed(3)}°)`,
  )

  // ── 8. 探针点 ──
  console.log("\n[8] 探针点（包围盒 8 角 + 质心）")
  const bbox = plyBoundingBox(gauss)
  const probes: { label: string; p: Vec3Tuple }[] = []
  for (let k = 0; k < 8; k++) {
    probes.push({
      label: `bbox${k}`,
      p: [
        k & 1 ? bbox.max[0] : bbox.min[0],
        k & 2 ? bbox.max[1] : bbox.min[1],
        k & 4 ? bbox.max[2] : bbox.min[2],
      ],
    })
  }
  probes.push({ label: "centroid", p: bbox.center })

  let allFront = true
  let inFrame = 0
  let centroidNdc: [number, number] = [Number.NaN, Number.NaN]
  for (const probe of probes) {
    const cam = applyView(camera.viewMatrix, probe.p)
    if (!(cam[2] > 0)) allFront = false
    // u = fx·x/z + W/2 → ndc = (u/W - 0.5)·2
    const ndcX =
      cam[2] > 0 ? (pose.fx * cam[0]) / cam[2] / (pose.width / 2) : Number.NaN
    const ndcY =
      cam[2] > 0 ? (pose.fy * cam[1]) / cam[2] / (pose.height / 2) : Number.NaN
    const inside = Math.abs(ndcX) <= 1 && Math.abs(ndcY) <= 1
    if (inside) inFrame++
    if (probe.label === "centroid") centroidNdc = [ndcX, ndcY]
    if (
      probe.label === "centroid" ||
      Math.abs(ndcX) > 1 ||
      Math.abs(ndcY) > 1
    ) {
      console.log(
        `      ${probe.label.padEnd(9)} z_view=${cam[2].toFixed(4).padStart(9)} m  ` +
          `ndc=(${ndcX.toFixed(3).padStart(7)}, ${ndcY.toFixed(3).padStart(7)})  ${inside ? "在图内" : "图外"}`,
      )
    }
  }
  check(
    "所有探针点都在相机前方",
    allFront,
    `z_view > 0（含全部包围盒角点，说明朝向没翻）`,
  )
  check(
    "质心投影落在图内",
    Math.abs(centroidNdc[0]) <= 0.9 && Math.abs(centroidNdc[1]) <= 0.9,
    `ndc = (${centroidNdc[0].toFixed(3)}, ${centroidNdc[1].toFixed(3)})，包围盒角点在图内 ${inFrame}/8`,
  )

  // ── 9. 深度往返 ──
  console.log("\n[9] 深度往返（导出投影的可逆性）")
  let maxRel = 0
  for (const z of [
    zStats.p01,
    bbox.center ? viewZStats(gauss, pose).median : 1,
    zStats.p99,
  ]) {
    const round = camera.ndcDepthToViewDepth(viewDepthToNdcDepth(camera, z))
    maxRel = Math.max(maxRel, Math.abs(round - z) / z)
  }
  check(
    "ndcDepthToViewDepth(viewDepthToNdcDepth(z)) == z",
    maxRel < 1e-4,
    `max 相对误差 ${maxRel.toExponential(3)}（z ∈ [${zStats.p01.toFixed(3)}, ${zStats.p99.toFixed(3)}]）`,
  )

  console.log(
    `\n[范围] 视图空间 z：p01=${zStats.p01.toFixed(3)} 中位=${zStats.median.toFixed(3)} p99=${zStats.p99.toFixed(3)} m  ` +
      `→ near=${near.toFixed(3)} far=${far.toFixed(3)}（可用 --near/--far 覆盖）`,
  )
  console.log(
    `[点云] n=${gauss.count}  包围盒 ${bbox.min.map((n) => n.toFixed(2)).join(",")} .. ${bbox.max.map((n) => n.toFixed(2)).join(",")}`,
  )

  report()
}

/** 打印结论并按需设置退出码。 */
function report(): void {
  console.log()
  if (problems.length > 0) {
    console.log("✗ 发现问题:")
    for (const p of problems) console.log(`   - ${p}`)
    console.log(
      "\n（注意：本脚本只核对参数自洽性。渲染与原图的对齐证据见 " +
        "scripts/wsplat-golden.ts --compare-image / scripts/wsplat-render.ts）",
    )
    process.exitCode = 1
    return
  }
  console.log(
    "✓ 通过：位姿 / 内参 / 投影约定自洽，点云位于视锥内。\n" +
      "  渲染与原图的对齐证据见 scripts/wsplat-golden.ts --compare-image。",
  )
}

// ── 小工具 ──

function isVec3(v: unknown): v is Vec3Tuple {
  return (
    Array.isArray(v) && v.length === 3 && v.every((n) => Number.isFinite(n))
  )
}

function isMat3(m: unknown): m is Mat3Tuple {
  return Array.isArray(m) && m.length === 3 && m.every((r) => isVec3(r))
}

/** `max|RᵗR - I|`。 */
function orthonormalityError(r: Mat3Tuple): number {
  let err = 0
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let dot = 0
      for (let k = 0; k < 3; k++) dot += r[k][i] * r[k][j]
      err = Math.max(err, Math.abs(dot - (i === j ? 1 : 0)))
    }
  }
  return err
}

function det3(r: Mat3Tuple): number {
  return (
    r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1]) -
    r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0]) +
    r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
  )
}

/** 列主序 4x4 作用在点上（w=1）。 */
function applyView(m: Float32Array, p: Vec3Tuple): [number, number, number] {
  return [
    m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
    m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
    m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
    // 这里刻意只返回 3 个分量：相机空间 w 恒为 1（仿射）
  ]
}

/** 视图深度 -> NDC 深度（WebGPU `[0,1]`），与 `camera.ts` 里的反函数配对。 */
function viewDepthToNdcDepth(
  camera: { near: number; far: number },
  viewZ: number,
): number {
  return (camera.far / (camera.far - camera.near)) * (1 - camera.near / viewZ)
}

/** 点云的视图空间 z 分位数 + 包围盒。 */
function viewZStats(
  g: ReturnType<typeof readPlyGaussians>,
  pose: SuperSplatCameraPose,
): { p01: number; median: number; p99: number } {
  const cam = { position: pose.position, rotation: pose.rotation }
  const zs = new Float32Array(g.count)
  for (let i = 0; i < g.count; i++) {
    const zc = cameraZ(cam, [g.x[i], g.y[i], g.z[i]])
    zs[i] = zc
  }
  zs.sort()
  const at = (q: number): number =>
    zs[Math.min(g.count - 1, Math.max(0, Math.floor(g.count * q)))]
  return { p01: at(0.01), median: at(0.5), p99: at(0.99) }
}

/** 视图空间 z（= forward·(p - C)）。 */
function cameraZ(
  cam: { position: Vec3Tuple; rotation: Mat3Tuple },
  p: Vec3Tuple,
): number {
  const r = cam.rotation
  const dx = p[0] - cam.position[0]
  const dy = p[1] - cam.position[1]
  const dz = p[2] - cam.position[2]
  // forward = rotation 第 3 列
  return r[0][2] * dx + r[1][2] * dy + r[2][2] * dz
}

function plyBoundingBox(g: ReturnType<typeof readPlyGaussians>): {
  min: Vec3Tuple
  max: Vec3Tuple
  center: Vec3Tuple
} {
  let mn = [Infinity, Infinity, Infinity]
  let mx = [-Infinity, -Infinity, -Infinity]
  const axes = [g.x, g.y, g.z]
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < g.count; i++) {
      const v = axes[c][i]
      if (v < mn[c]) mn[c] = v
      if (v > mx[c]) mx[c] = v
    }
  }
  mn = [mn[0], mn[1], mn[2]]
  mx = [mx[0], mx[1], mx[2]]
  return {
    min: mn as Vec3Tuple,
    max: mx as Vec3Tuple,
    center: [(mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2],
  }
}

/** 解析 `--crop x,y,w,h`。 */
function parseCrop(
  v: string | undefined,
): [number, number, number, number] | undefined {
  if (v === undefined) return undefined
  const parts = v.split(",").map((s) => Number.parseInt(s.trim(), 10))
  if (parts.length !== 4 || parts.some((n) => !Number.isFinite(n))) {
    throw new Error(`--crop 需要 4 个整数 x,y,w,h，收到: ${v}`)
  }
  return parts as [number, number, number, number]
}

main().catch((err) => {
  console.error("\n[错误]", err instanceof Error ? err.message : err)
  if (err instanceof Error && err.stack) console.error(err.stack)
  process.exitCode = 1
})
