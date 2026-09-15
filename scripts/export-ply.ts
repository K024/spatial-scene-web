/**
 * 单图 -> 3DGS PLY + SuperSplat 相机 json。
 *
 * 这是整条 node 管线的入口脚本，串起：
 *   `loadImage`（sharp 解码 + EXIF 焦距）
 *     -> `runSharp`（预处理 -> onnxruntime-node 推理 -> NDC 反投影）
 *     -> `serializePly`（3DGS 二进制 PLY）
 *     -> `superSplatCameraJson`（原图视角的相机位姿）
 *
 * 产物默认落在 `public/exports/`（Vite 会把 `public/` 原样挂到站点根，
 * 因此 web 侧可以用 `/exports/sample.ply` 直接取）。
 *
 * 用法:
 *   npx tsx scripts/export-ply.ts
 *   npx tsx scripts/export-ply.ts --image photo.jpg --out public/exports/photo.ply
 *   npx tsx scripts/export-ply.ts --focal 35 --provider dml
 *   npx tsx scripts/export-ply.ts --full              # 追加 ml-sharp 的补充 element
 *   npx tsx scripts/export-ply.ts --limit 200000      # 调试：只写前 N 个高斯
 *
 * 环境变量:
 *   SHARP_EP        覆盖 EP 候选（`platform.node.ts`），例如 `cpu`、`dml,cpu`
 *   SHARP_ORT_LOG   ort 日志级别（默认 error）
 */

import { copyFileSync, mkdirSync, writeFileSync } from "node:fs"
import { basename, dirname, resolve } from "node:path"
import { parseArgs } from "node:util"

import {
  fovDeg,
  superSplatCameraJson,
  superSplatCameraPose,
} from "../src/spatial-scene/export/camera.ts"
import { serializePly } from "../src/spatial-scene/export/ply.ts"
import { runSharp } from "../src/spatial-scene/infer/index.ts"
import type { ExecutionProviderHint } from "../src/spatial-scene/infer/session.ts"
import type { Gaussians3D } from "../src/spatial-scene/sharp/types.ts"
import {
  DEFAULT_IMAGE,
  humanSize,
  MODEL_FP16,
  numFlag,
  REPO_ROOT,
  requireFile,
} from "./utils/common.ts"
import { loadImage } from "./utils/image.ts"
import { createNodeSession } from "./utils/platform.node.ts"

const PROVIDERS: readonly ExecutionProviderHint[] = [
  "auto",
  "dml",
  "cuda",
  "coreml",
  "webgpu",
  "wasm",
  "cpu",
]

/**
 * 默认输出路径（相对仓库根）。
 *
 * 与 `package.json` 的 `npm run sample` 对应；`sample.camera.json` 与
 * `sample.ply` 同目录同名（见 {@link cameraPathFor}）。
 */
const DEFAULT_OUT = resolve(REPO_ROOT, "public", "exports", "sample.ply")

/** `foo.ply` -> `foo.camera.json`（同目录、同主干名）。 */
function cameraPathFor(plyPath: string): string {
  return `${plyPath.replace(/\.ply$/i, "")}.camera.json`
}

/** 取前 `n` 个高斯（调试用；PLY 的 element count 随之变小）。 */
function head(g: Gaussians3D, n: number): Gaussians3D {
  return {
    meanVectors: g.meanVectors.subarray(0, n * 3),
    singularValues: g.singularValues.subarray(0, n * 3),
    quaternions: g.quaternions.subarray(0, n * 4),
    colors: g.colors.subarray(0, n * 3),
    opacities: g.opacities.subarray(0, n),
  }
}

/** 解析 `--crop x,y,w,h`。 */
function parseCrop(
  raw: string | undefined,
): [number, number, number, number] | undefined {
  if (raw === undefined) return undefined
  const parts = raw.split(",").map((s) => Number(s.trim()))
  if (parts.length !== 4 || parts.some((v) => !Number.isFinite(v))) {
    throw new Error(`--crop 需要 "x,y,w,h" 四个数字（收到 "${raw}"）`)
  }
  return [parts[0], parts[1], parts[2], parts[3]]
}

/** 场景统计（用于日志：确认坐标/尺度/深度合理）。 */
function summarize(g: Gaussians3D, n: number): string[] {
  const lo = [Infinity, Infinity, Infinity]
  const hi = [-Infinity, -Infinity, -Infinity]
  let zMin = Infinity
  let zMax = -Infinity
  let pMin = Infinity
  let pMax = -Infinity
  let pSum = 0
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) {
      const v = g.meanVectors[i * 3 + c]
      if (v < lo[c]) lo[c] = v
      if (v > hi[c]) hi[c] = v
    }
    const z = g.meanVectors[i * 3 + 2]
    if (z < zMin) zMin = z
    if (z > zMax) zMax = z
    const p = g.opacities[i]
    if (p < pMin) pMin = p
    if (p > pMax) pMax = p
    pSum += p
  }
  const fmt = (v: number) => v.toFixed(3)
  return [
    `bbox  x[${fmt(lo[0])}, ${fmt(hi[0])}] y[${fmt(lo[1])}, ${fmt(hi[1])}] z[${fmt(lo[2])}, ${fmt(hi[2])}]`,
    `深度 z（= 相机坐标系下的前方距离）范围 [${fmt(zMin)}, ${fmt(zMax)}]`,
    `不透明度 [${pMin.toExponential(2)}, ${pMax.toFixed(4)}] 均值 ${(pSum / n).toFixed(4)}`,
  ]
}

async function main(): Promise<void> {
  const { values } = parseArgs({
    options: {
      image: { type: "string" },
      out: { type: "string" },
      camera: { type: "string" },
      model: { type: "string" },
      provider: { type: "string" },
      focal: { type: "string" },
      crop: { type: "string" },
      limit: { type: "string" },
      full: { type: "boolean", default: false },
      help: { type: "boolean", short: "h", default: false },
    },
    allowPositionals: false,
  })

  if (values.help) {
    console.log(
      [
        "用法: npx tsx scripts/export-ply.ts [选项]",
        "",
        "  --image <path>     输入图（默认 ml-depth-pro/data/example.jpg）",
        "  --out <path>       输出 PLY（默认 public/exports/sample.ply）",
        "  --camera <path>    输出相机 json（默认与 PLY 同名的 .camera.json）",
        "  --model <path>     ONNX 模型（默认 sharp_fp16.onnx）",
        `  --provider <ep>    ${PROVIDERS.join(" | ")}（默认 auto）`,
        "  --focal <mm>       显式指定 35mm 等效焦距（跳过 EXIF）",
        "  --crop x,y,w,h     从原图裁一块再推理",
        "  --limit <n>        调试：只导出前 n 个高斯",
        "  --full             追加 ml-sharp 的补充 element（extrinsic/intrinsic/...）",
      ].join("\n"),
    )
    return
  }

  const imagePath = requireFile(values.image ?? DEFAULT_IMAGE, "输入图像")
  const outPath = resolve(values.out ?? DEFAULT_OUT)
  const cameraPath = resolve(values.camera ?? cameraPathFor(outPath))
  const modelPath = requireFile(values.model ?? MODEL_FP16, "ONNX 模型")
  const provider = (values.provider ?? "auto") as ExecutionProviderHint
  if (!PROVIDERS.includes(provider)) {
    throw new Error(
      `未知 --provider "${provider}"，可选：${PROVIDERS.join(", ")}`,
    )
  }
  const explicitFocal = values.focal
    ? numFlag("--focal", values.focal, 0)
    : undefined
  const crop = parseCrop(values.crop)
  const limit = values.limit
    ? Math.floor(numFlag("--limit", values.limit, 0))
    : undefined

  console.log(`输入: ${imagePath}`)
  console.log(`图像: ${basename(imagePath)}`)
  console.log(`输出: ${outPath}`)
  console.log(`相机: ${cameraPath}`)
  console.log(`模型: ${modelPath}`)
  console.log(
    `EP:   ${provider}${process.env.SHARP_EP ? `（SHARP_EP=${process.env.SHARP_EP}）` : ""}`,
  )
  console.log("")

  // ── 1. 读图（EXIF 自动旋转 + 焦距 fallback 链）──
  const t0 = Date.now()
  const loaded = await loadImage(imagePath, explicitFocal, crop)
  const { width, height } = loaded.image
  console.log(
    `[1/4] 读图 ${width}x${height}  f_px=${loaded.fPx.toFixed(3)}  ` +
      `f_35mm=${loaded.focal35mm}${loaded.focalFromExif ? "（EXIF）" : "（默认/显式）"}  ` +
      `${Date.now() - t0}ms`,
  )

  // ── 2. 推理 ──
  const t1 = Date.now()
  const { result, session } = await runSharp({
    image: loaded.image,
    fPx: loaded.fPx,
    imageWidth: width,
    createSession: createNodeSession,
    model: { modelPath },
    provider,
  })
  console.log(
    `[2/4] 推理完成 EP=${result.capabilities.activeProvider} ` +
      `(${result.capabilities.availableProviders.join("/")}) ` +
      `fp16输入=${result.capabilities.fp16Input}  ${Date.now() - t1}ms`,
  )

  // ── 3. 序列化 ──
  let gaussians = result.metric
  let n = result.metric.opacities.length
  if (limit !== undefined && limit > 0 && limit < n) {
    console.log(`      --limit ${limit}：只导出前 ${limit} 个高斯（调试）`)
    gaussians = head(gaussians, limit)
    n = limit
  }
  for (const line of summarize(gaussians, n)) console.log(`      ${line}`)

  const t2 = Date.now()
  const ply = serializePly({
    gaussians,
    fPx: loaded.fPx,
    imageShape: [width, height],
    full: values.full ?? false,
  })
  mkdirSync(dirname(outPath), { recursive: true })
  writeFileSync(outPath, ply)
  console.log(
    `[3/4] PLY 已写 ${n} 个高斯${values.full ? "（full：含补充 element）" : ""}  ` +
      `${Date.now() - t2}ms`,
  )

  // ── 4. 相机 json（SuperSplat 可直接导入）──
  //
  // SHARP 的度量空间本身就是相机坐标系（extrinsics = I），所以位姿是
  // position=[0,0,0] + rotation=I、朝 +z；fov 用**原始** f_px 与原图尺寸给出。
  const pose = superSplatCameraPose({
    name: basename(imagePath),
    focalLengthPx: loaded.fPx,
    imageShape: [width, height],
    extrinsics: result.extrinsics,
  })
  writeFileSync(cameraPath, superSplatCameraJson([pose]))
  console.log(`[4/4] 相机 json 已写`)

  // 参考照片也复制一份到产物目录：sidecar 里的 `img_name` 指向它，
  // web 侧拿来做「渲染结果 vs 原图」的叠加/闪烁比对。
  const refImagePath = resolve(dirname(outPath), basename(imagePath))
  copyFileSync(imagePath, refImagePath)
  console.log(`      参考照片已复制 ${refImagePath}`)

  console.log("")
  // rotation 的第 3 列即相机朝向（同 `camera.ts: cameraPoseFromExtrinsics`）
  const forward = pose.rotation[2]
  console.log(
    `相机: position=[${pose.position.join(", ")}] forward=[${forward.join(", ")}]`,
  )
  console.log(
    `FOV:  x=${fovDeg(pose.fx, width).toFixed(2)}°  y=${fovDeg(pose.fy, height).toFixed(2)}°  ` +
      `（fx=fy=${pose.fx.toFixed(2)}px，图像 ${width}x${height}）`,
  )
  console.log(
    `产物: ${outPath} (${humanSize(ply.byteLength)})  +  ${cameraPath}`,
  )
  console.log(`      ${refImagePath}`)
  console.log(
    "提示: 拖入 https://superspl.at/editor 后，先导入相机 json 再导入 PLY，即可回到原图视角。",
  )
}

main().catch((err) => {
  console.error(`\n导出失败: ${err instanceof Error ? err.message : err}`)
  process.exitCode = 1
})
