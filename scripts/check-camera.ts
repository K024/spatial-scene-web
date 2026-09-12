/**
 * 相机视角离线校验：验证「SuperSplat 里跳到 .camera.json 给的位姿 == 原图视角」。
 *
 * ── 为什么要这个脚本 ──
 * 相机参数写错（坐标系翻转、fov 用错域、主点偏移）在 SuperSplat 里表现为
 * 「看起来也像那么回事」的轻微错位，肉眼很难判定。这里用**同一套坐标约定**
 * 离线渲染一遍：把 PLY 高斯按 json 里的位姿投影、每个像素保留最近的高斯，
 * 与输入图逐像素比（MAE / 灰度相关 NCC），并且拿几组**故意扰动**的相机做对照——
 * 只有正确位姿明显更优，才能说明参数是对的。
 *
 * ── 相机约定 ──
 * 与 `src/spatial-scene/export/camera.ts` 完全一致：json 里的 position 是相机中心、
 * `rotation` 是 cam2world（第 3 列 = 朝向），全部写在 **PLY 坐标系**里。
 * 投影用原始图像域的 `fx = fy = f_px`、主点在图像中心。
 *
 * ── 注意 ──
 * 这里只做「点云 z-buffer」而不是完整 3DGS 光栅化（ellipsoid splatting）：
 * 目的是校验相机，不是复现 SuperSplat 的画质。渲染结果会比编辑器噪一些，
 * 但结构与色彩分布足以判定对齐与否。
 *
 * 用法：
 *   npx tsx scripts/check-camera.ts                    # 默认 example.ply + 同名 json
 *   npx tsx scripts/check-camera.ts --image X.jpg --crop 0,0,545,748
 *   npx tsx scripts/check-camera.ts --ply a.ply --camera a.camera.json --width 512
 *
 * 产物：PNG 写到 `py-models/out/camera-check/`（参考图 / 渲染图 / 并排对比），
 * 可直接打开目视核对。
 */

import { mkdirSync, readFileSync } from "node:fs"
import { createRequire } from "node:module"
import { basename, resolve } from "node:path"
import { parseArgs } from "node:util"

import type {
  Mat3Tuple,
  SuperSplatCameraPose,
  Vec3Tuple,
} from "../src/spatial-scene/export/camera.ts"
import { fovDeg } from "../src/spatial-scene/export/camera.ts"
import { DEFAULT_IMAGE, REPO_ROOT } from "./utils/common.ts"
import { loadImage } from "./utils/image.ts"
import { type PlyGaussians, readPlyGaussians } from "./utils/ply.ts"

/** SH degree-0 基函数值（与 `sharp/linalg.ts` 里的 C0 一致）。 */
const SH_C0 = 0.28209479177387814

const CLI_OPTIONS = {
  ply: { type: "string" },
  camera: { type: "string" },
  image: { type: "string" },
  crop: { type: "string" },
  out: { type: "string" },
  width: { type: "string" },
  "min-alpha": { type: "string" },
} as const

/** 参与渲染/对比的相机（PLY 坐标系，主点默认在图像中心）。 */
interface RenderCamera {
  position: Vec3Tuple
  /** cam2world，行主序。 */
  rotation: Mat3Tuple
  fx: number
  fy: number
  /** 主点（像素，原图域）。 */
  cx: number
  cy: number
}

/** 一次渲染的结果。 */
interface RenderResult {
  /** RGB8，长度 rw*rh*3；未覆盖像素为 0。 */
  rgb: Uint8Array
  /** 1 = 该像素有高斯落到（用于把未覆盖像素排除出统计）。 */
  mask: Uint8Array
  /** 覆盖率 = mask 的均值。 */
  coverage: number
}

async function main(): Promise<void> {
  const args = parseArgs({
    args: process.argv.slice(2),
    options: CLI_OPTIONS,
    allowPositionals: false,
    strict: true,
  })

  const crop = parseCrop(args.values.crop)
  const targetWidth = Number.parseInt(args.values.width ?? "384", 10)
  const minAlpha = Number.parseFloat(args.values["min-alpha"] ?? "0.5")

  // ── 1. 输入：PLY / 相机 json / 原图 ──
  const plyPath = resolve(
    REPO_ROOT,
    args.values.ply ?? "py-models/out/ply/example.ply",
  )
  const cameraPath = resolve(
    REPO_ROOT,
    args.values.camera ?? plyPath.replace(/\.ply$/i, ".camera.json"),
  )
  const imagePath = resolve(REPO_ROOT, args.values.image ?? DEFAULT_IMAGE)

  console.log("=".repeat(72))
  console.log("相机视角离线校验")
  console.log("=".repeat(72))

  const poses = JSON.parse(
    readFileSync(cameraPath, "utf8"),
  ) as SuperSplatCameraPose[]
  if (!Array.isArray(poses) || poses.length === 0) {
    throw new Error(`${cameraPath} 不是非空数组（SuperSplat 的相机 json 格式）`)
  }
  if (poses.length > 1) {
    console.log(`[warn] json 含 ${poses.length} 条位姿，只校验第 1 条`)
  }
  const pose = poses[0]

  const loaded = await loadImage(imagePath, undefined, crop)
  const { width: W, height: H } = loaded.image
  console.log(`ply    : ${plyPath}`)
  console.log(`camera : ${cameraPath}`)
  console.log(`image  : ${imagePath} (${W}x${H})`)
  console.log()
  console.log(
    `[pose] position = [${pose.position.map((n) => n.toFixed(4)).join(", ")}]  ` +
      `rotation 第 3 列(朝向) = [${pose.rotation[2].map((n) => n.toFixed(4)).join(", ")}]`,
  )
  console.log(
    `[pose] fx=fy=${pose.fx}  ${pose.width}x${pose.height}  ` +
      `fov=${fovDeg(pose.fx, Math.max(pose.width, pose.height)).toFixed(3)}°(较长轴)`,
  )

  // 尺寸不一致 = json 与这张图（或裁切区域）不匹配，继续算下去没有意义
  if (pose.width !== W || pose.height !== H) {
    throw new Error(
      `相机 json 记录的是 ${pose.width}x${pose.height}，而输入图（含 --crop）是 ${W}x${H}；` +
        "请确认 --image/--crop 与生成该 ply/json 时一致",
    )
  }

  // ── 2. 读 PLY ──
  const gauss = readPlyGaussians(plyPath)
  console.log(
    `[ply] n=${gauss.count}  x∈[${min(gauss.x).toFixed(3)}, ${max(gauss.x).toFixed(3)}] ` +
      `y∈[${min(gauss.y).toFixed(3)}, ${max(gauss.y).toFixed(3)}] ` +
      `z∈[${min(gauss.z).toFixed(3)}, ${max(gauss.z).toFixed(3)}]`,
  )

  // ── 3. 渲染分辨率：整数倍降采样，参考图与渲染图共用同一套像素网格 ──
  const factor = Math.max(1, Math.ceil(W / targetWidth))
  const rw = Math.floor(W / factor)
  const rh = Math.floor(H / factor)
  console.log(
    `[render] ${rw}x${rh} (1/${factor} 降采样, minAlpha=${minAlpha})\n`,
  )

  const reference = boxDownsample(loaded.image.data, W, H, rw, rh, factor)

  // ── 4. 导出位姿 + 若干扰动位姿 ──
  const cam: RenderCamera = {
    position: pose.position,
    rotation: pose.rotation,
    fx: pose.fx,
    fy: pose.fy,
    cx: W / 2,
    cy: H / 2,
  }

  const cases: { label: string; cam: RenderCamera }[] = [
    { label: "as-exported", cam },
    // fov 偏大/偏小 15%
    { label: "fov x0.85", cam: scaleFocal(cam, 0.85) },
    { label: "fov x1.15", cam: scaleFocal(cam, 1.15) },
    // 沿相机自身 x/y 轴平移 0.15m（约等于近景视差的量级）
    { label: "shift +x", cam: translateLocal(cam, [0.15, 0, 0]) },
    { label: "shift -y", cam: translateLocal(cam, [0, -0.15, 0]) },
    // 绕相机自身 y 轴偏航 3°
    { label: "yaw +3deg", cam: rotateLocalY(cam, 3) },
    // 主点偏移到 55%
    { label: "pp +5%", cam: { ...cam, cx: W * 0.55, cy: H * 0.55 } },
  ]

  console.log("[对比]  label          MAE(0-255)   NCC(灰度)   覆盖")
  let correct: { ncc: number; mae: number } | null = null
  let bestPerturbedNcc = -Infinity
  const rendered = new Map<string, Uint8Array>()

  for (const c of cases) {
    const out = render(gauss, c.cam, rw, rh, factor, minAlpha)
    const mae = meanAbsError(out.rgb, reference, out.mask)
    const ncc = normalizedCrossCorrelation(out.rgb, reference, out.mask)
    console.log(
      `         ${c.label.padEnd(14)} ${mae.toFixed(3).padStart(10)} ` +
        `${ncc.toFixed(4).padStart(12)}   ${(out.coverage * 100).toFixed(1)}%`,
    )
    if (c.label === "as-exported") {
      correct = { ncc, mae }
      rendered.set(c.label, out.rgb)
    } else {
      bestPerturbedNcc = Math.max(bestPerturbedNcc, ncc)
    }
  }
  const ok = correct as { ncc: number; mae: number }

  // ── 5. 写 PNG（目视核对）──
  const suffix = crop ? `_${crop.join("x")}` : ""
  const outDir = resolve(
    REPO_ROOT,
    args.values.out ?? "py-models/out/camera-check",
  )
  mkdirSync(outDir, { recursive: true })
  const stem = basename(plyPath).replace(/\.ply$/i, "") + suffix

  const writePng = async (
    name: string,
    rgb: Uint8Array,
    w: number,
    h: number,
  ): Promise<string> => {
    const path = resolve(outDir, name)
    await sharpFromRaw(rgb, w, h).png().toFile(path)
    return path
  }

  // 并排（左：参考图 / 右：渲染图），错位方向一眼可见
  const renderRgb = rendered.get("as-exported") as Uint8Array
  const sideBySide = new Uint8Array(rw * 2 * rh * 3)
  for (let y = 0; y < rh; y++) {
    const src = y * rw * 3
    sideBySide.set(reference.subarray(src, src + rw * 3), y * rw * 6)
    sideBySide.set(renderRgb.subarray(src, src + rw * 3), y * rw * 6 + rw * 3)
  }

  const refPath = await writePng(`${stem}_reference.png`, reference, rw, rh)
  const renderPath = await writePng(`${stem}_render.png`, renderRgb, rw, rh)
  const sidePath = await writePng(
    `${stem}_side-by-side.png`,
    sideBySide,
    rw * 2,
    rh,
  )
  console.log()
  console.log(`[png] 参考（左） ${refPath}`)
  console.log(`[png] 渲染         ${renderPath}`)
  console.log(`[png] 并排         ${sidePath}`)

  // ── 6. 断言 ──
  console.log()
  const problems: string[] = []
  if (ok.ncc < 0.5) {
    problems.push(
      `导出位姿的灰度相关只有 ${ok.ncc.toFixed(4)}（< 0.5），渲染与原图不对齐`,
    )
  }
  if (ok.ncc <= bestPerturbedNcc) {
    problems.push(
      `扰动位姿（最好 ${bestPerturbedNcc.toFixed(4)}）不差于导出位姿` +
        `（${ok.ncc.toFixed(4)}），校验无法区分，参数可疑`,
    )
  }
  if (problems.length > 0) {
    console.log("✗ 发现问题:")
    for (const p of problems) console.log(`   - ${p}`)
    process.exit(1)
  }
  console.log(
    `✓ 通过：导出位姿 NCC=${ok.ncc.toFixed(4)} 优于所有扰动（最好 ` +
      `${bestPerturbedNcc.toFixed(4)}），MAE=${ok.mae.toFixed(3)}`,
  )
}

/** 按 factor 做 box 平均降采样。 */
function boxDownsample(
  src: ArrayLike<number>,
  W: number,
  H: number,
  rw: number,
  rh: number,
  factor: number,
): Uint8Array {
  const dst = new Uint8Array(rw * rh * 3)
  for (let y = 0; y < rh; y++) {
    for (let x = 0; x < rw; x++) {
      let r = 0
      let g = 0
      let b = 0
      let n = 0
      for (let dy = 0; dy < factor; dy++) {
        const sy = y * factor + dy
        if (sy >= H) break
        for (let dx = 0; dx < factor; dx++) {
          const sx = x * factor + dx
          if (sx >= W) break
          const o = (sy * W + sx) * 3
          r += src[o]
          g += src[o + 1]
          b += src[o + 2]
          n++
        }
      }
      const d = (y * rw + x) * 3
      dst[d] = Math.round(r / n)
      dst[d + 1] = Math.round(g / n)
      dst[d + 2] = Math.round(b / n)
    }
  }
  return dst
}

/**
 * 点云 z-buffer 渲染。
 *
 * 每个高斯按 `u = fx·X/Z + cx`、`v = fy·Y/Z + cy` 投到一个像素，
 * 该像素保留**深度最小**的那个高斯；`alpha < minAlpha` 的高斯不参与
 * （它们大概率是各层之间的“雾”，会把最近表面挡掉）。
 */
function render(
  g: PlyGaussians,
  cam: RenderCamera,
  rw: number,
  rh: number,
  factor: number,
  minAlpha: number,
): RenderResult {
  const { position: C, rotation: R, fx, fy, cx, cy } = cam
  const rgb = new Uint8Array(rw * rh * 3)
  const mask = new Uint8Array(rw * rh)
  const zbuf = new Float32Array(rw * rh).fill(Infinity)

  for (let i = 0; i < g.count; i++) {
    if (g.alpha[i] < minAlpha) continue

    // X_cam = Rᵗ·(p - C)；Rᵗ 的第 i 行 = R 的第 i 列
    const dx = g.x[i] - C[0]
    const dy = g.y[i] - C[1]
    const dz = g.z[i] - C[2]
    const xc = R[0][0] * dx + R[1][0] * dy + R[2][0] * dz
    const yc = R[0][1] * dx + R[1][1] * dy + R[2][1] * dz
    const zc = R[0][2] * dx + R[1][2] * dy + R[2][2] * dz
    if (zc <= 1e-4) continue

    const u = (fx * xc) / zc + cx
    const v = (fy * yc) / zc + cy
    const px = Math.floor(u / factor)
    const py = Math.floor(v / factor)
    if (px < 0 || px >= rw || py < 0 || py >= rh) continue

    const idx = py * rw + px
    if (zc >= zbuf[idx]) continue
    zbuf[idx] = zc
    mask[idx] = 1
    const d = idx * 3
    // 颜色：SH DC -> sRGB 字节。save_ply 已把 linearRGB 转成 sRGB 写入 f_dc
    for (let c = 0; c < 3; c++) {
      const value = 0.5 + SH_C0 * g.fdc[i * 3 + c]
      rgb[d + c] = Math.max(0, Math.min(255, Math.round(value * 255)))
    }
  }

  let covered = 0
  for (let i = 0; i < mask.length; i++) covered += mask[i]
  return { rgb, mask, coverage: covered / mask.length }
}

/** 掩码内的平均绝对误差（0-255 标度）。 */
function meanAbsError(a: Uint8Array, b: Uint8Array, mask: Uint8Array): number {
  let sum = 0
  let n = 0
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i]) continue
    const o = i * 3
    sum +=
      (Math.abs(a[o] - b[o]) +
        Math.abs(a[o + 1] - b[o + 1]) +
        Math.abs(a[o + 2] - b[o + 2])) /
      3
    n++
  }
  return n > 0 ? sum / n : Number.NaN
}

/**
 * 掩码内的灰度归一化互相关。
 *
 * 零均值 NCC ∈ [-1, 1]，越接近 1 越对齐。它与整体亮度/增益无关，
 * 所以不会被「点云整体比原图暗」干扰。
 */
function normalizedCrossCorrelation(
  a: Uint8Array,
  b: Uint8Array,
  mask: Uint8Array,
): number {
  const va: number[] = []
  const vb: number[] = []
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i]) continue
    const o = i * 3
    va.push(0.299 * a[o] + 0.587 * a[o + 1] + 0.114 * a[o + 2])
    vb.push(0.299 * b[o] + 0.587 * b[o + 1] + 0.114 * b[o + 2])
  }
  const n = va.length
  if (n === 0) return Number.NaN
  let ma = 0
  let mb = 0
  for (let i = 0; i < n; i++) {
    ma += va[i]
    mb += vb[i]
  }
  ma /= n
  mb /= n
  let sab = 0
  let saa = 0
  let sbb = 0
  for (let i = 0; i < n; i++) {
    const da = va[i] - ma
    const db = vb[i] - mb
    sab += da * db
    saa += da * da
    sbb += db * db
  }
  return sab / Math.sqrt(saa * sbb)
}

// ── 相机扰动工具（扰动都定义在相机自身坐标系里，量级更直观）──

/** 焦距乘系数（模拟 FOV 偏差）。 */
function scaleFocal(cam: RenderCamera, k: number): RenderCamera {
  return { ...cam, fx: cam.fx * k, fy: cam.fy * k }
}

/**
 * 沿相机自身 x/y 轴平移。
 *
 * `rotation` 是 cam2world（列为相机轴），所以相机坐标系里的偏移 `o`
 * 映射到 PLY 坐标系就是 `R·o`。
 */
function translateLocal(cam: RenderCamera, offset: Vec3Tuple): RenderCamera {
  const r = cam.rotation
  const d: Vec3Tuple = [
    r[0][0] * offset[0] + r[0][1] * offset[1] + r[0][2] * offset[2],
    r[1][0] * offset[0] + r[1][1] * offset[1] + r[1][2] * offset[2],
    r[2][0] * offset[0] + r[2][1] * offset[1] + r[2][2] * offset[2],
  ]
  return {
    ...cam,
    position: [
      cam.position[0] + d[0],
      cam.position[1] + d[1],
      cam.position[2] + d[2],
    ],
  }
}

/** 绕相机自身 y 轴偏航 `deg` 度：新朝向矩阵 = R·Ry(θ)。 */
function rotateLocalY(cam: RenderCamera, deg: number): RenderCamera {
  const t = (deg * Math.PI) / 180
  const c = Math.cos(t)
  const s = Math.sin(t)
  const r = cam.rotation
  // R·Ry 的第 i 列 = c·Rcol0 - s·Rcol2 (i=0) / Rcol1 (i=1) / s·Rcol0 + c·Rcol2 (i=2)
  const col0: Vec3Tuple = [
    c * r[0][0] - s * r[0][2],
    c * r[1][0] - s * r[1][2],
    c * r[2][0] - s * r[2][2],
  ]
  const col1: Vec3Tuple = [r[0][1], r[1][1], r[2][1]]
  const col2: Vec3Tuple = [
    s * r[0][0] + c * r[0][2],
    s * r[1][0] + c * r[1][2],
    s * r[2][0] + c * r[2][2],
  ]
  const rotation: Mat3Tuple = [
    [col0[0], col1[0], col2[0]],
    [col0[1], col1[1], col2[1]],
    [col0[2], col1[2], col2[2]],
  ]
  return { ...cam, rotation }
}

// ── 小工具 ──

function min(a: Float32Array): number {
  let m = Infinity
  for (let i = 0; i < a.length; i++) if (a[i] < m) m = a[i]
  return m
}

function max(a: Float32Array): number {
  let m = -Infinity
  for (let i = 0; i < a.length; i++) if (a[i] > m) m = a[i]
  return m
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

/** sharp 是 CJS，ESM 下用 createRequire 引入。 */
const requireCjs = createRequire(import.meta.url)

/** 用 sharp 把 RGB8 原始像素编码成 PNG。 */
function sharpFromRaw(
  rgb: Uint8Array,
  width: number,
  height: number,
): { png(): { toFile(p: string): Promise<unknown> } } {
  const sharp = requireCjs("sharp") as (
    input: Buffer,
    opts: { raw: { width: number; height: number; channels: number } },
  ) => { png(): { toFile(p: string): Promise<unknown> } }
  return sharp(Buffer.from(rgb.buffer, rgb.byteOffset, rgb.byteLength), {
    raw: { width, height, channels: 3 },
  })
}

main().catch((err) => {
  console.error("\n[错误]", err instanceof Error ? err.message : err)
  if (err instanceof Error && err.stack) console.error(err.stack)
  process.exit(1)
})
