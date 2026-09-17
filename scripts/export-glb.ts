/**
 * GLB 交付产物导出（**整条管线的入口**：图 / PLY -> layered RGBAD -> mesh -> GLB）。
 *
 * ```
 * --image 模式（默认推荐）：
 *   loadImage(EXIF 焦距) -> runSharp(onnxruntime-node) -> 度量空间高斯
 *     -> layering -> meshing -> buildGlb -> 写盘          （全程内存，不落 PLY）
 * --ply 模式（兼容 / 复现既有产物）：
 *   loadWSplatScene(PLY + camera.json) -> layering -> meshing -> buildGlb
 * ```
 *
 * 判定「对不对」用 `scripts/meshing-check.ts`（那边在内存里跑同一套 `buildGlb`
 * 并验往返字节，不写盘）；本脚本只负责出交付产物。
 *
 * 产物是**自描述**的，查看器只需要 GLB 本身：
 * - 每层一个 glTF mesh / node（`extras.layerIndex` / `depthRange`），节点顺序远 -> 近；
 * - 相机参数写在 `cameras[0].perspective`（`yfov` 弧度 / `znear` / `zfar` / `aspectRatio`）
 *   和 `reference_camera` node 上，另有一份在 `scenes[0].extras.camera`（含原图像素焦距、
 *   扩视角 `referenceRect`）。
 *
 * 用法:
 *   npx tsx scripts/export-glb.ts                                  # 默认：从 public/exports/sample.ply 复现
 *   npx tsx scripts/export-glb.ts --image photo.jpg               # 整条推理管线（图 -> GLB）
 *   npx tsx scripts/export-glb.ts --image photo.jpg --out public/exports/photo.glb
 *   npx tsx scripts/export-glb.ts --image photo.jpg --focal 35 --provider dml
 *   npx tsx scripts/export-glb.ts --image teaser.jpg --crop 0,0,1000,750
 *   npx tsx scripts/export-glb.ts --image photo.jpg --limit 200000  # 调试：只用前 N 个高斯
 *   npx tsx scripts/export-glb.ts --image photo.jpg --short-side 1024
 *   npx tsx scripts/export-glb.ts --image pano.jpg --max-side 2048   # 全景图：长边硬上限
 *   npx tsx scripts/export-glb.ts --texture-format png              # 退回未压缩 PNG
 *   npx tsx scripts/export-glb.ts --texture-encoding uastc          # 更高质量 KTX2
 *
 * 分辨率：默认 `--short-side auto` = `min(1536（SHARP 内部分辨率）, 原图短边)` ——
 * 把参考内容的短边渲染到这么多像素上（长边由宽高比决定，可用 `--max-side` 兜底）。
 *
 * 环境变量:
 *   SHARP_EP       覆盖 EP 候选，例如 `cpu`、`dml,cpu`
 *   SHARP_ORT_LOG  ort 日志级别（默认 error）
 */

import { mkdirSync, writeFileSync } from "node:fs"
import { dirname, resolve } from "node:path"
import { parseArgs } from "node:util"
import { runSharp } from "../src/spatial-scene/infer/index.ts"
import type { ExecutionProviderHint } from "../src/spatial-scene/infer/session.ts"
import {
  DEFAULT_MAX_RENDER_SIDE,
  DEFAULT_SHORT_SIDE,
  renderLayerStack,
} from "../src/spatial-scene/layering/index.ts"
import {
  buildGlbAsync,
  buildMeshScene,
  type MeshScene,
} from "../src/spatial-scene/meshing/index.ts"
import type { Gaussians3D } from "../src/spatial-scene/sharp/types.ts"
import {
  DEFAULT_IMAGE,
  humanSize,
  MODEL_FP16,
  numFlag,
  REPO_ROOT,
  requireFile,
  shortSideFlag,
} from "./utils/common.ts"
import { loadImage } from "./utils/image.ts"
import { createNodeSession } from "./utils/platform.node.ts"
import {
  buildSceneFromGaussians,
  type GaussiansScene,
  loadWSplatScene,
} from "./utils/scene.ts"
import { withNodeDevice } from "./utils/webgpu.ts"

const PROVIDERS: readonly ExecutionProviderHint[] = [
  "auto",
  "dml",
  "cuda",
  "coreml",
  "webgpu",
  "wasm",
  "cpu",
]

const CLI = {
  image: { type: "string" },
  model: { type: "string" },
  provider: { type: "string" },
  focal: { type: "string" },
  crop: { type: "string" },
  limit: { type: "string" },
  ply: { type: "string" },
  camera: { type: "string" },
  "max-splats": { type: "string" },
  layers: { type: "string" },
  "view-scale": { type: "string" },
  "short-side": { type: "string" },
  "max-side": { type: "string" },
  "min-cell": { type: "string" },
  "max-cell": { type: "string" },
  "max-error": { type: "string" },
  "snap-px": { type: "string" },
  dilate: { type: "string" },
  "alpha-cutoff": { type: "string" },
  "min-island": { type: "string" },
  "texture-format": { type: "string" },
  "texture-encoding": { type: "string" },
  "texture-quality": { type: "string" },
  pixel: { type: "boolean" },
  "no-double-sided": { type: "boolean" },
  out: { type: "string" },
} as const

async function main(): Promise<void> {
  const args = parseArgs({
    args: process.argv.slice(2),
    options: CLI,
    allowPositionals: false,
    strict: true,
  })
  const layers = Math.round(numFlag("--layers", args.values.layers, 10))
  const viewScale = numFlag("--view-scale", args.values["view-scale"], 1.2)
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
  const outPath = resolve(
    REPO_ROOT,
    args.values.out ?? "public/exports/layered.glb",
  )
  const textureFormat = args.values["texture-format"] ?? "ktx2"
  if (textureFormat !== "ktx2" && textureFormat !== "png") {
    throw new Error(`未知 --texture-format "${textureFormat}"，可选 ktx2 | png`)
  }
  const textureEncoding = args.values["texture-encoding"] ?? "etc1s"
  if (textureEncoding !== "etc1s" && textureEncoding !== "uastc") {
    throw new Error(
      `未知 --texture-encoding "${textureEncoding}"，可选 etc1s | uastc`,
    )
  }
  const textureQuality = args.values["texture-quality"]
    ? numFlag("--texture-quality", args.values["texture-quality"], 180)
    : undefined

  console.log("=".repeat(78))
  console.log("导出 GLB")
  console.log("=".repeat(78))

  // ── 1. 场景来源：图（整条推理管线）或 PLY ──
  const scene = args.values.image
    ? await sceneFromImage(args.values)
    : loadWSplatScene({
        ply: args.values.ply,
        camera: args.values.camera,
        maxSplats: args.values["max-splats"]
          ? Math.round(numFlag("--max-splats", args.values["max-splats"], 0))
          : undefined,
      })
  console.log(
    `\n场景: ${scene.gaussians.opacities.length} 个高斯  ` +
      `参考 ${scene.camera.width}x${scene.camera.height}  ` +
      `near=${scene.near.toFixed(3)}m far=${scene.far.toFixed(1)}m`,
  )

  // ── 2. 分层 -> 网格 -> GLB ──
  let meshScene: MeshScene | undefined
  await withNodeDevice(async (device) => {
    const t0 = Date.now()
    const layered = await renderLayerStack(device, scene.gaussians, {
      camera: scene.camera,
      layers,
      viewScale,
      shortSide,
      maxRenderSide: maxSide,
    })
    const rect = layered.view.referenceRect
    console.log(
      `\n分层渲染 ${Date.now() - t0}ms  ${layered.L} 层  ` +
        `画布 ${layered.width}x${layered.height}  ` +
        `参考短边 ${Math.min(rect.width, rect.height).toFixed(0)}px（${shortSide === "auto" ? "auto" : "显式"}）  ` +
        `pixelScale=${layered.view.pixelScale.toFixed(4)}`,
    )

    const t1 = Date.now()
    meshScene = buildMeshScene(layered, {
      alpha: {
        alphaCutoff: args.values["alpha-cutoff"]
          ? numFlag("--alpha-cutoff", args.values["alpha-cutoff"], 1 / 255)
          : undefined,
        dilatePx: args.values.dilate
          ? numFlag("--dilate", args.values.dilate, 2)
          : undefined,
        minIslandPixels: args.values["min-island"]
          ? numFlag("--min-island", args.values["min-island"], 8)
          : undefined,
      },
      lod: args.values.pixel
        ? false
        : {
            minCellPx: numFlag("--min-cell", args.values["min-cell"], 4),
            maxCellPx: numFlag("--max-cell", args.values["max-cell"], 128),
            maxError: numFlag("--max-error", args.values["max-error"], 0.005),
            snapBoundaryPx: args.values["snap-px"]
              ? numFlag("--snap-px", args.values["snap-px"], 4)
              : undefined,
          },
    })
    console.log(
      `网格化 ${Date.now() - t1}ms  ${meshScene.report.triangleCount} 三角面  ` +
        `${meshScene.report.vertexCount} 顶点  ${meshScene.layers.length} 层`,
    )
  })
  if (!meshScene) throw new Error("网格化没有产出")

  const t2 = Date.now()
  if (textureFormat === "ktx2") {
    console.log(
      `\nKTX2 编码 ${meshScene.layers.length} 层（${textureEncoding}${
        textureEncoding === "etc1s" ? ` q=${textureQuality ?? 180}` : ""
      }）...`,
    )
  }
  const glb = await buildGlbAsync(meshScene, {
    name: "spatial-scene",
    doubleSided: args.values["no-double-sided"] !== true,
    textureFormat,
    textureEncoding,
    textureQuality,
  })
  mkdirSync(dirname(outPath), { recursive: true })
  writeFileSync(outPath, glb)

  // 相机参数核对（查看器只依赖 GLB 本身，所以这里把写进去的东西打出来）。
  const camera = meshScene.camera
  const rect = meshScene.view.referenceRect
  console.log(
    `\n相机  yfov=${((camera.fovY * 180) / Math.PI).toFixed(2)}°  ` +
      `aspect=${(meshScene.view.width / meshScene.view.height).toFixed(4)}  ` +
      `near=${camera.near.toFixed(3)}m  far=${camera.far.toFixed(2)}m`,
  )
  console.log(
    `      原图焦距 ${meshScene.view.referenceFocalLengthPx.toFixed(1)}px  ` +
      `渲染焦距 ${meshScene.view.focalLengthPx.toFixed(1)}px  ` +
      `referenceRect=[${rect.x.toFixed(1)}, ${rect.y.toFixed(1)}, ${rect.width.toFixed(1)}, ${rect.height.toFixed(1)}]`,
  )
  console.log(
    `\n纹理  ${textureFormat}${textureFormat === "ktx2" ? `/${textureEncoding}` : ""}` +
      `${textureFormat === "ktx2" && textureEncoding === "etc1s" ? ` q=${textureQuality ?? 180}` : ""}`,
  )
  console.log(
    `写出 ${outPath}  ${humanSize(glb.byteLength)}  ${Date.now() - t2}ms`,
  )
}

/**
 * `--image` 路径：读图 -> 推理 -> 度量空间高斯 -> 场景。
 *
 * **全程内存**：不导出 PLY、也不重新读回来（`serializePly` 只是「给 SuperSplat
 * 用」的旁路产物，交付链路不需要它）。
 */
async function sceneFromImage(values: {
  image?: string
  model?: string
  provider?: string
  focal?: string
  crop?: string
  limit?: string
}): Promise<GaussiansScene> {
  const imagePath = requireFile(values.image ?? DEFAULT_IMAGE, "输入图像")
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

  console.log(`\n输入图: ${imagePath}`)
  console.log(`模型:   ${modelPath}`)
  console.log(
    `EP:     ${provider}${process.env.SHARP_EP ? `（SHARP_EP=${process.env.SHARP_EP}）` : ""}`,
  )

  const t0 = Date.now()
  const loaded = await loadImage(imagePath, explicitFocal, crop)
  const { width, height } = loaded.image
  console.log(
    `[1/2] 读图 ${width}x${height}  f_px=${loaded.fPx.toFixed(3)}  ` +
      `f_35mm=${loaded.focal35mm}${loaded.focalFromExif ? "（EXIF）" : "（默认/显式）"}  ` +
      `${Date.now() - t0}ms`,
  )

  const t1 = Date.now()
  const { result, session } = await runSharp({
    image: loaded.image,
    fPx: loaded.fPx,
    imageWidth: width,
    createSession: createNodeSession,
    model: { modelPath },
    provider,
  })
  // 推理产物已在内存里；会话可以立刻释放（下面还会吃几百 MB 的分层/网格）
  await session.dispose().catch(() => {})
  console.log(
    `[2/2] 推理 ${result.capabilities.activeProvider} ` +
      `(${result.capabilities.availableProviders.join("/")}) ` +
      `fp16输入=${result.capabilities.fp16Input}  ${Date.now() - t1}ms`,
  )

  let gaussians: Gaussians3D = result.metric
  if (limit !== undefined && limit > 0 && limit < gaussians.opacities.length) {
    console.log(`      --limit ${limit}：只用前 ${limit} 个高斯（调试）`)
    gaussians = head(gaussians, limit)
  }

  return buildSceneFromGaussians({
    gaussians,
    focalLengthPx: loaded.fPx,
    width,
    height,
  })
}

/** 取前 `n` 个高斯（调试用）。 */
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

main().catch((err) => {
  console.error(`\n导出失败: ${err instanceof Error ? err.stack : err}`)
  process.exitCode = 1
})
