/**
 * 跑一次 SHARP 推理并导出 PLY（node 侧，供 SuperSplat 查看）。
 *
 * 用法：
 *   npx tsx scripts/infer-ply.ts                    # 默认 DML + fp16 模型
 *   npx tsx scripts/infer-ply.ts --ep webgpu        # 实验性 WebGPU EP
 *   npx tsx scripts/infer-ply.ts --full             # 追加 ml-sharp 元数据 element
 *   npx tsx scripts/infer-ply.ts --crop 0,0,545,748 # 从拼图里裁一张（teaser 图用）
 *   npx tsx scripts/infer-ply.ts --image X.jpg --out out.ply
 *
 * 除 `.ply` 外还会写一个同名 `.camera.json`：SuperSplat 的相机位姿格式
 * （INRIA `cameras.json`）。把两个文件**一起**拖进 https://superspl.at/editor，
 * 相机就会跳到拍摄时的精确视角（见 `src/spatial-scene/export/camera.ts` 的坐标系说明）。
 * 离线核对这个视角下画面是否与输入图对齐：`npx tsx scripts/check-camera.ts`。
 *
 * 环境变量：
 *   SHARP_EP      覆盖执行提供者（逗号分隔的候选链）
 *   SHARP_MODEL   覆盖模型路径
 *   SHARP_IMAGE   覆盖输入图
 *   SHARP_ORT_LOG ort 日志级别
 *
 * 产出写到 `py-models/out/ply/`，可直接拖入 https://superspl.at/editor 查看。
 *
 * ── 实测性能（fp16 模型，1,179,648 个高斯，RTX 级独显）──
 *   DML 前向     ~2.8 s（会话加载完成后；首次运行含编译会明显更久）
 *   CPU 前向     ~73 s（仅作参考，不代表目标平台）
 *   PLY 序列化   ~280 ms，产出 63 MB（14 个 float32 属性）
 */

import { mkdirSync, writeFileSync } from "node:fs"
import { basename, resolve } from "node:path"
import { parseArgs } from "node:util"

import {
  fovDeg,
  poseTarget,
  superSplatCameraJson,
  superSplatCameraPose,
} from "../src/spatial-scene/export/camera.ts"
import { gaussiansToPly } from "../src/spatial-scene/export/ply.ts"
import { runSharp } from "../src/spatial-scene/infer/index.ts"
import type { ExecutionProviderHint } from "../src/spatial-scene/infer/session.ts"
import type { SceneMetaData } from "../src/spatial-scene/sharp/types.ts"
import {
  DEFAULT_IMAGE,
  humanSize,
  MODEL_FP16,
  prepareOrtEnv,
  REPO_ROOT,
  requireFile,
  TEASER_IMAGE,
} from "./utils/common.ts"
import { loadImage } from "./utils/image.ts"

// ── 必须在 import onnxruntime-node 之前（见 utils/common.ts 说明）──
prepareOrtEnv("error")

/** 支持的命令行选项。boolean 型只接受 `--flag`，string 型必须带值 `--key value`。 */
const CLI_OPTIONS = {
  model: { type: "string" },
  image: { type: "string" },
  out: { type: "string" },
  crop: { type: "string" },
  ep: { type: "string" },
  full: { type: "boolean" },
} as const

async function main(): Promise<void> {
  const args = parseArgs({
    args: process.argv.slice(2),
    options: CLI_OPTIONS,
    allowPositionals: false,
    strict: true,
  })

  const modelPath = requireFile(
    args.values.model ?? process.env.SHARP_MODEL ?? MODEL_FP16,
    "模型",
  )
  const imagePath = requireFile(
    args.values.image ?? process.env.SHARP_IMAGE ?? (await defaultImagePath()),
    "输入图",
  )
  /**
   * 可选取图区域 `--crop x,y,w,h`（像素）。
   *
   * 用途：仓库里唯一的现成图 `ml-sharp/data/teaser.jpg` 是 3×2 对比拼图
   * （上排参考照片、下排渲染结果），直接喂给模型得到的不是单场景。
   * 用 `--crop` 从其中裁出一张真实照片再推理。
   */
  const crop = parseCrop(args.values.crop)
  const ep = (args.values.ep ?? "auto") as ExecutionProviderHint
  const full = args.values.full === true

  // 默认输出名带上 crop 信息，避免同一张图的不同裁切互相覆盖
  const stem = basename(imagePath).replace(/\.[^.]+$/, "")
  const cropSuffix = crop ? `_${crop.join("x")}` : ""
  const outPath = resolve(
    REPO_ROOT,
    args.values.out ?? `py-models/out/ply/${stem}${cropSuffix}.ply`,
  )

  console.log("=".repeat(72))
  console.log("SHARP 推理 + PLY 导出")
  console.log("=".repeat(72))
  console.log(`model : ${modelPath}`)
  console.log(`image : ${imagePath}`)
  console.log(
    `ep    : ${ep}${process.env.SHARP_EP ? ` (SHARP_EP=${process.env.SHARP_EP})` : ""}`,
  )
  console.log(`full  : ${full}`)
  console.log()

  // ── 1. 加载图像 + 焦距 ──
  const t0 = Date.now()
  const loaded = await loadImage(imagePath, undefined, crop)
  console.log(
    `[image] ${loaded.image.width}x${loaded.image.height} ch=${loaded.image.channels} ` +
      `focal35mm=${loaded.focal35mm.toFixed(2)}mm (${loaded.focalFromExif ? "EXIF" : "default"}) ` +
      `f_px=${loaded.fPx.toFixed(2)}  ${Date.now() - t0}ms`,
  )

  // ── 2. 推理 ──
  const { createNodeSession } = await import("./utils/platform.node.ts")
  const t1 = Date.now()
  const { session, result } = await runSharp({
    image: loaded.image,
    fPx: loaded.fPx,
    imageWidth: loaded.image.width,
    createSession: createNodeSession,
    model: { modelPath },
    provider: ep,
  })
  const inferMs = Date.now() - t1
  console.log(
    `[infer] ${inferMs}ms  ep=${result.capabilities.activeProvider} ` +
      `available=[${result.capabilities.availableProviders.join(",")}] fp16Input=${result.capabilities.fp16Input}`,
  )
  console.log(
    `        disparity_factor = ${result.disparityFactor.toFixed(8)} (= f_px / width)`,
  )

  // ── 3. 统计（用于判断反投影是否正确）──
  reportGaussians("NDC   ", result.ndc)
  reportGaussians("metric", result.metric)

  const m = result.unprojectionMatrix
  console.log("[unproject] 4x4 (行主序):")
  for (let r = 0; r < 4; r++) {
    console.log(
      `   ${[0, 1, 2, 3].map((c) => m[r * 4 + c].toFixed(6).padStart(12)).join(" ")}`,
    )
  }

  // ── 4. 导出 PLY ──
  const meta: SceneMetaData = {
    focalLengthPx: loaded.fPx,
    resolutionPx: [loaded.image.width, loaded.image.height],
    colorSpace: "linearRGB",
  }
  const t2 = Date.now()
  const ply = gaussiansToPly(result.metric, meta, { full })
  console.log(`[ply] 序列化 ${Date.now() - t2}ms -> ${humanSize(ply.length)}`)

  mkdirSync(resolve(outPath, ".."), { recursive: true })
  writeFileSync(outPath, ply)
  console.log(`[ply] 写入 ${outPath}`)

  // ── 5. 导出相机位姿（SuperSplat 的 json 格式）──
  const pose = superSplatCameraPose({
    name: stem + cropSuffix,
    focalLengthPx: loaded.fPx,
    imageShape: [loaded.image.width, loaded.image.height],
    extrinsics: result.extrinsics,
  })
  const target = poseTarget({
    position: pose.position,
    forward: pose.rotation[2],
  })
  const fmt = (v: readonly number[]): string =>
    v.map((n) => n.toFixed(4)).join(", ")
  console.log(`[camera] position = [${fmt(pose.position)}]  (PLY 坐标系)`)
  console.log(`[camera] target   = [${fmt(target)}]  (position + 10 * 朝向)`)
  console.log(
    `[camera] fov      = ${fovDeg(loaded.fPx, loaded.image.width).toFixed(3)}°(横) / ` +
      `${fovDeg(loaded.fPx, loaded.image.height).toFixed(3)}°(竖)  ` +
      `fx=fy=${loaded.fPx.toFixed(3)} @ ${loaded.image.width}x${loaded.image.height}`,
  )

  // 命名：把 .ply 换成 .camera.json；无扩展名时直接追加（避免覆盖 --out 指定的文件）
  const cameraPath = outPath.toLowerCase().endsWith(".ply")
    ? `${outPath.slice(0, -4)}.camera.json`
    : `${outPath}.camera.json`
  writeFileSync(cameraPath, superSplatCameraJson([pose]))
  console.log(`[camera] 写入 ${cameraPath}`)

  await session.dispose()

  console.log()
  console.log(
    "提示：把该 .ply 与 .camera.json 一起拖入 https://superspl.at/editor。",
  )
  console.log(
    "      只丢 .ply 会得到自动取景的环绕视角；带上 json 才回到拍摄视角。",
  )
  console.log(
    "      想两轴都精确复原，把编辑器视口宽高比调成与原图一致（" +
      `${loaded.image.width}:${loaded.image.height}）——` +
      "SuperSplat 把 fov 作用在视口较长的轴上。",
  )
  console.log(
    "      离线核对（无需浏览器/GPU）：npx tsx scripts/check-camera.ts",
  )
  if (full) {
    console.log(
      "      --full 模式含 ml-sharp 补充 element，SuperSplat 可能忽略它们。",
    )
  }
}

/** 解析 `--crop x,y,w,h`；未给出时返回 undefined。 */
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

/**
 * 选择默认输入图。
 *
 * 优先 `ml-depth-pro/data/example.jpg`（py-models 脚本使用的标准测试图），
 * 不存在时回退到 ml-sharp 自带的 teaser 拼图（需配合 `--crop` 才有意义）。
 */
async function defaultImagePath(): Promise<string> {
  const { existsSync } = await import("node:fs")
  if (existsSync(DEFAULT_IMAGE)) return DEFAULT_IMAGE
  return requireFile(TEASER_IMAGE, "输入图（默认图不存在，请用 --image 指定）")
}

/**
 * 打印一组高斯的统计量，便于快速判断数值是否合理。
 *
 * 分别报告 x / y / z 的最大绝对值：z 在 NDC 里量级远大于 x,y，
 * 合并成一个 maxAbs 会把 x/y 的变化完全掩盖（反投影对 x/y 的缩放就看不出来）。
 */
function reportGaussians(
  label: string,
  g: {
    meanVectors: Float32Array
    singularValues: Float32Array
    opacities: Float32Array
  },
): void {
  const n = g.opacities.length
  let minZ = Infinity,
    maxZ = -Infinity
  let maxX = 0,
    maxY = 0,
    maxZabs = 0
  for (let i = 0; i < n; i++) {
    const x = g.meanVectors[i * 3 + 0]
    const y = g.meanVectors[i * 3 + 1]
    const z = g.meanVectors[i * 3 + 2]
    if (z < minZ) minZ = z
    if (z > maxZ) maxZ = z
    const ax = Math.abs(x),
      ay = Math.abs(y),
      az = Math.abs(z)
    if (ax > maxX) maxX = ax
    if (ay > maxY) maxY = ay
    if (az > maxZabs) maxZabs = az
  }
  let sumA = 0
  let solid = 0
  for (let i = 0; i < n; i++) {
    sumA += g.opacities[i]
    if (g.opacities[i] > 0.8) solid++
  }
  console.log(
    `[${label}] n=${n} z∈[${minZ.toFixed(4)}, ${maxZ.toFixed(4)}] ` +
      `max|x|=${maxX.toFixed(4)} max|y|=${maxY.toFixed(4)} max|z|=${maxZabs.toFixed(4)} ` +
      `meanAlpha=${(sumA / n).toFixed(4)} solid(α>0.8)=${((solid / n) * 100).toFixed(1)}%`,
  )
}

main().catch((err) => {
  console.error("\n[错误]", err instanceof Error ? err.message : err)
  if (err instanceof Error && err.stack) console.error(err.stack)
  process.exit(1)
})
