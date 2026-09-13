/**
 * **Web UI（backroad）= 交付入口**：上传图片 -> 生成 -> 导出 **GLB（three.js 可查看）**。
 *
 * ── 唯一职责 ──
 * ```
 * 上传图片 -> runSharp 推理 -> Gaussians3D -> WSplatScene -> 分层(+原图回写)
 *          -> MeshScene -> GLB（KHR_draco_mesh_compression + KHR_materials_unlit + 内嵌相机/extras）
 * ```
 * 只跑通这一条。没有内置样例、不做逐层预览 / 诊断表 / metadata 面板 ——
 * 那些不是交付链路。输出（GLB）写在**临时目录**，不落仓库。
 *
 * ── 重跑模型（照 Streamlit 的约定；backroad 同样每次交互自上而下重跑）──
 * Widget 只返回当前值；**昂贵 / 有副作用的工作不挂在 render 路径上**：
 * 1. 只有点「生成 GLB」那一次 rerun 才计算（≈ `if st.button(...)`）。
 * 2. 推理按图片身份、生成按 `(图片, 参数)` **memo 一次且永不 evict**（≈ `@st.cache_data`/`cache_resource`）；
 *    写盘只发生在 memo 内（同一配置只写一次）。下载按钮按路径**懒读**。
 *
 * ⚠ backroad 按钮一次点击触发 **两次** rerun（`set_value` 置 true，随后 `unset_value`）：
 * 生成只挂在 `true` 那次，第二次从 memo 读。
 *
 * ── 固定选择 ──
 * - 渲染宽度 = **原图分辨率**（不为速度假设缩小质量）；层数上限 **8**；
 *   推理 EP **自动**（win `dml->cpu` / linux `cuda->cpu` / mac `coreml->cpu`）。
 *
 * ── 入口 ──
 * ```
 * npm run webui            # http://localhost:3333
 * WEBUI_PORT=4000 npm run webui
 * ```
 */

import { mkdirSync, readFileSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { basename, extname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { run } from "@backroad/backroad"
import type { LayerSamplingMethod } from "../src/spatial-scene/layering/index.ts"
import type { MeshScene } from "../src/spatial-scene/meshing/index.ts"
import type { SourceImage } from "../src/spatial-scene/sharp/preprocess.ts"
import {
  buildGlb,
  DEFAULT_LAYER_OVERLAP,
  inferSceneFromImage,
  type StageProgress,
} from "./pack/generate.ts"
import { prepareOrtEnv } from "./utils/common.ts"
import { createNodeDevice } from "./utils/webgpu.ts"
import { assembleWSplatScene, type WSplatScene } from "./utils/wsplat-scene.ts"

// ── 必须在 onnxruntime-node 被求值前（见 utils/common.ts）──
prepareOrtEnv("error")

/** 出货上界：层数 = mesh 数 = 显存 / 代价。 */
const MAX_LAYERS = 8
/** 直方图箱数（固定，不暴露）。 */
const BIN_COUNT = 256
/** 产物（GLB）目录：系统临时目录，不落仓库。 */
const OUTPUT_DIR = resolve(tmpdir(), "spatial-scene-webui")

const METHODS: readonly LayerSamplingMethod[] = [
  "quantile",
  "uniform",
  "uniformNonEmpty",
  "importance",
  "hybrid",
  "frontWeighted",
  "errorDriven",
]

/** 输入源 = 上传的图片（webui 只处理图片）。 */
interface Source {
  /** memo 键：用上传临时路径 + 大小，换图即失效。 */
  readonly key: string
  readonly path: string
  readonly label: string
}

interface UiConfig {
  readonly layers: number
  readonly method: LayerSamplingMethod
  /** 层间重叠（只影响报告的 `layerRanges`）。 */
  readonly overlap: number
  readonly refine: boolean
  /** 几何补齐（own-gap + hidden 外推）。 */
  readonly complete: boolean
  /** hidden 外推最大距离（px）。 */
  readonly maxDist: number
  /** hidden 是否外推 RGBA（关 = 只补几何、保留原纹理 α）。 */
  readonly extrapRgba: boolean
  /** own-gap 回填（关 = 定位边缘硬点）。 */
  readonly ownGap: boolean
  /** 撕裂清理（视差去噪 + 小面片门）。**缺省关**（库默认开）。 */
  readonly tearCleanup: boolean
  readonly draco: boolean
  /** LOD 出面（受限四叉树）开关与参数。 */
  readonly lod: boolean
  readonly lodMinCell: number
  readonly lodSnap: boolean
}

/** 一次生成的结果引用（GLB 字节在磁盘，下载时懒读）。 */
interface View {
  readonly sourceKey: string
  readonly config: UiConfig
  readonly width: number
  readonly height: number
  readonly near: number
  readonly far: number
  readonly gaussianCount: number
  readonly totalTriangles: number
  readonly refined: boolean
  readonly inferMs: number
  readonly ep: string
  readonly layersMs: number
  readonly meshMs: number
  readonly glbMs: number
  readonly glbName: string
  readonly glbPath: string
  readonly glbBytes: number
}

// ────────────────────────────── device / 串行化 / memo ──────────────────────────────

let devicePromise: Promise<GPUDevice> | undefined
function getDevice(): Promise<GPUDevice> {
  devicePromise ??= createNodeDevice()
  return devicePromise
}

/** 退出前必须释放（Dawn-node 不 destroy 会 SIGSEGV）。 */
async function disposeDevice(): Promise<void> {
  const pending = devicePromise
  devicePromise = undefined
  if (!pending) return
  try {
    ;(await pending).destroy()
  } catch {
    // 退出 / 测试清理路径
  }
}

/** 串行化 GPU 使用（推理走 ort、生成走 Dawn device，避免并发争抢）。 */
let deviceQueue: Promise<unknown> = Promise.resolve()
function enqueue<T>(fn: () => Promise<T>): Promise<T> {
  const next = deviceQueue.then(fn, fn)
  deviceQueue = next.then(
    () => undefined,
    () => undefined,
  )
  return next
}

interface CachedSource {
  readonly scene: WSplatScene
  readonly sourceImage: SourceImage
  readonly inferMs: number
  readonly ep: string
}

// memo（≈ st.cache_*）：只增不删；副作用（写盘）只在 memo 内发生一次。
const sourceMemo = new Map<string, CachedSource>()
const sourceInflight = new Map<string, Promise<CachedSource>>()
const viewMemo = new Map<string, View>()
const viewInflight = new Map<string, Promise<View>>()

function configKey(source: Source, cfg: UiConfig): string {
  return [
    source.key,
    cfg.layers,
    cfg.method,
    cfg.overlap,
    cfg.refine,
    cfg.complete
      ? `complete${cfg.maxDist}${cfg.extrapRgba ? "" : "d"}${cfg.ownGap ? "" : "n"}`
      : "nocomplete",
    cfg.draco,
    cfg.tearCleanup ? "tearClean" : "tearRaw",
    cfg.lod ? `lod${cfg.lodMinCell}${cfg.lodSnap ? "s" : "n"}` : "dense",
  ].join("|")
}

/** 上传图片 -> `WSplatScene`（推理，慢）。**按 source 只算一次**。 */
function getSource(
  source: Source,
  onStage?: StageProgress,
): Promise<CachedSource> {
  const hit = sourceMemo.get(source.key)
  if (hit) return Promise.resolve(hit)
  const running = sourceInflight.get(source.key)
  if (running) return running

  // ⚠ 不在这里 `enqueue()`：调用方（`ensureBuilt`）已串行化，嵌套会自锁。
  const promise = (async (): Promise<CachedSource> => {
    const inferred = await inferSceneFromImage({
      imagePath: source.path,
      onStage,
    })
    return {
      scene: assembleWSplatScene({
        gaussians: inferred.gaussians,
        pose: inferred.pose,
      }),
      sourceImage: inferred.loaded.image,
      inferMs: inferred.inferMs,
      ep: inferred.capabilities.activeProvider,
    }
  })()
    .then((value) => {
      sourceMemo.set(source.key, value)
      return value
    })
    .finally(() => {
      sourceInflight.delete(source.key)
    })
  sourceInflight.set(source.key, promise)
  return promise
}

/**
 * **唯一的副作用入口**：`(source, config)` -> `View`，并落盘 GLB 到临时目录。
 * 只有点「生成 GLB」时才调用。
 */
function ensureBuilt(
  source: Source,
  cfg: UiConfig,
  onStage?: StageProgress,
): Promise<View> {
  const key = configKey(source, cfg)
  const hit = viewMemo.get(key)
  if (hit) return Promise.resolve(hit)
  const running = viewInflight.get(key)
  if (running) return running

  const promise = enqueue(async () => {
    const cached = await getSource(source, onStage)
    const device = await getDevice()
    const result = await buildGlb(device, cached.scene, cached.sourceImage, {
      layers: cfg.layers,
      method: cfg.method,
      binCount: BIN_COUNT,
      overlap: cfg.overlap,
      refine: cfg.refine,
      complete: cfg.complete
        ? {
            ownGap: cfg.ownGap,
            hidden: { maxDistancePx: cfg.maxDist, rgba: cfg.extrapRgba },
          }
        : false,
      draco: cfg.draco,
      mesh: {
        tears: {
          denoise: cfg.tearCleanup,
          minPatchPixels: cfg.tearCleanup ? 16 : 1,
        },
        ...(cfg.lod
          ? {
              lod: {
                minCellPx: cfg.lodMinCell,
                maxError: 0.005,
                snapBoundary: cfg.lodSnap,
              },
            }
          : {}),
      },
      onStage,
    })
    return toView(source, cfg, cached, result)
  })
    .then((view) => {
      viewMemo.set(key, view)
      return view
    })
    .finally(() => {
      viewInflight.delete(key)
    })
  viewInflight.set(key, promise)
  return promise
}

function toView(
  source: Source,
  cfg: UiConfig,
  cached: CachedSource,
  result: {
    meshScene: MeshScene
    glb: Uint8Array
    refined: boolean
    layersMs: number
    meshMs: number
    glbMs: number
  },
): View {
  let totalTriangles = 0
  for (const layer of result.meshScene.layers) {
    totalTriangles += layer.triangleCount
  }

  const glbName = glbFileName(source, cfg)
  const glbPath = saveGlb(glbName, result.glb)
  const scene = cached.scene

  return {
    sourceKey: source.key,
    config: cfg,
    width: scene.width,
    height: scene.height,
    near: scene.near,
    far: scene.far,
    gaussianCount: scene.gaussians.opacities.length,
    totalTriangles,
    refined: result.refined,
    inferMs: cached.inferMs,
    ep: cached.ep,
    layersMs: result.layersMs,
    meshMs: result.meshMs,
    glbMs: result.glbMs,
    glbName,
    glbPath,
    glbBytes: result.glb.length,
  }
}

function glbFileName(source: Source, cfg: UiConfig): string {
  const stem = sanitize(basename(source.path, extname(source.path)))
  const ov = `ov${cfg.overlap}`
  const comp = cfg.complete
    ? `c${cfg.maxDist}${cfg.extrapRgba ? "" : "d"}${cfg.ownGap ? "" : "n"}`
    : "nc"
  const tear = cfg.tearCleanup ? "tc" : "tr"
  return `${stem}_L${cfg.layers}_${cfg.method}_${ov}_${comp}_${tear}${cfg.draco ? "_draco" : ""}.glb`
}

function sanitize(name: string): string {
  return name.replace(/[^a-zA-Z0-9._-]+/g, "_").slice(0, 64) || "scene"
}

/** 写 GLB 到临时目录。**唯一的写盘点**，同一配置只调用一次。 */
function saveGlb(name: string, bytes: Uint8Array): string {
  mkdirSync(OUTPUT_DIR, { recursive: true })
  const path = resolve(OUTPUT_DIR, name)
  writeFileSync(path, bytes)
  return path
}

// ────────────────────────────── UI ──────────────────────────────

type NodeManager = Parameters<Parameters<typeof run>[0]>[0]

async function executor(br: NodeManager): Promise<void> {
  br.title({ label: "Spatial Scene — 图片 -> GLB" })
  br.write({
    body:
      `上传图片 -> 生成多层 mesh -> 导出 GLB（three.js 可查看）。渲染分辨率 = **原图分辨率**；` +
      `层数上限 ${MAX_LAYERS}；推理执行提供者按系统自动选择。`,
  })

  // ── 参数（平铺；widget 只返回当前值，不触发计算）──
  const uploads = br.fileUpload({
    label: "上传图片",
    accept: { "image/*": [".jpg", ".jpeg", ".png", ".webp"] },
  })
  const layers = br.slider({
    label: "层数 L",
    min: 1,
    max: MAX_LAYERS,
    step: 1,
    defaultValue: MAX_LAYERS,
  })
  const method = br.select({
    label: "层放置方法",
    options: METHODS.map((m) => ({ value: m, label: m })),
    defaultValue: "quantile",
  }) as LayerSamplingMethod
  const overlap = br.numberInput({
    label: "层间重叠（只影响 layerRanges）",
    defaultValue: DEFAULT_LAYER_OVERLAP,
    min: 0,
    max: 0.1,
    step: 0.005,
    precision: 3,
  })
  const refine = br.toggle({ label: "原图回写 refine", defaultValue: true })
  const complete = br.toggle({
    label: "几何补齐（own-gap + hidden 外推）",
    defaultValue: true,
  })
  const maxDist = br.numberInput({
    label: "hidden 外推最大距离（px）",
    defaultValue: 16,
    min: 0,
    max: 128,
    step: 2,
    precision: 0,
  })
  const extrapRgba = br.toggle({
    label: "hidden 外推 RGBA（开 = 拉伸边缘；关 = 只补几何）",
    defaultValue: false,
  })
  const ownGap = br.toggle({
    label: "own-gap 回填（关 = 定位边缘硬点）",
    defaultValue: true,
  })
  const tearCleanup = br.toggle({
    label: "撕裂清理（视差去噪 + 小面片门；库默认开）",
    defaultValue: false,
  })
  const draco = br.toggle({ label: "Draco 压缩几何", defaultValue: true })
  const lod = br.toggle({
    label: "LOD 出面（自适应四叉树）",
    defaultValue: true,
  })
  const lodMinCell = br.slider({
    label: "LOD 最小格子（像素，越大面越少）",
    min: 1,
    max: 8,
    step: 1,
    defaultValue: 4,
  })
  const lodSnap = br.toggle({
    label: "LOD 边界贴合（靠纹理 α 掩边）",
    defaultValue: true,
  })

  const source = resolveSource(uploads[0])
  if (!source) {
    br.write({ body: "请先 **上传一张图片**。" })
    return
  }

  const cfg: UiConfig = {
    layers: Math.min(layers, MAX_LAYERS),
    method,
    overlap: Math.max(0, overlap),
    refine,
    complete,
    maxDist: Math.max(0, Math.round(maxDist)),
    extrapRgba,
    ownGap,
    tearCleanup,
    draco,
    lod,
    lodMinCell,
    lodSnap,
  }

  // ── 显式触发（≈ `if st.button(...)`）：只有 `true` 那次 rerun 计算 ──
  const generate = br.button({ label: "生成 GLB" })
  if (generate) {
    // 实时进度：`streamable` 就地重渲染；每次 await 之间会刷到前端。
    const status = br.streamable()
    try {
      await ensureBuilt(source, cfg, makeProgress(status))
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      br.write({ body: `❌ **生成失败**：\n\n\`\`\`\n${msg}\n\`\`\`` })
      return
    }
  }

  // ── 只读 memo 渲染（不触发计算、不写盘）──
  const view = viewMemo.get(configKey(source, cfg))
  if (!view) {
    br.write({
      body:
        `**${source.label}** · L=${cfg.layers} · ${cfg.method} · ` +
        `overlap=${cfg.overlap} · refine=${refine ? "on" : "off"} · ` +
        `complete=${cfg.complete ? `hidden<=${cfg.maxDist}px${cfg.extrapRgba ? "" : " depth-only"}${cfg.ownGap ? "" : " no-own-gap"}` : "off"} · ` +
        `draco=${draco ? "on" : "off"}\n\n点 **「生成 GLB」** 开始。`,
    })
    return
  }

  br.downloadButton({
    label: `下载 GLB（${(view.glbBytes / 1024 / 1024).toFixed(2)} MB）`,
    // 懒读：点击下载时才读盘。
    data: () => readFileSync(view.glbPath),
    filename: view.glbName,
    mime: "model/gltf-binary",
  })

  // 少量 status（不铺诊断表）：分阶段计时。
  br.stats({
    items: [
      { label: "分辨率", value: `${view.width}×${view.height}` },
      { label: "层数", value: view.config.layers },
      {
        label: "层重叠 / 补齐",
        value:
          `${view.config.overlap} / ` +
          (view.config.complete
            ? `hidden<=${view.config.maxDist}px${view.config.extrapRgba ? "" : "・depth-only"}${view.config.ownGap ? "" : "・no-own-gap"}`
            : "off"),
      },
      { label: "三角", value: view.totalTriangles.toLocaleString() },
      {
        label: "GLB",
        value: `${(view.glbBytes / 1024 / 1024).toFixed(2)} MB`,
      },
      {
        label: "推理/分层/mesh/GLB",
        value: `${fmtS(view.inferMs)} / ${fmtS(view.layersMs)} / ${fmtS(view.meshMs)} / ${fmtS(view.glbMs)}`,
      },
    ],
  })
  br.write({
    body:
      `\`${view.glbPath}\`\n\n` +
      `three.js：\`GLTFLoader\` + \`DRACOLoader\`` +
      `${view.config.draco ? "（已 Draco 压缩，需要 decoder）" : "（未压缩，无需 decoder）"}；` +
      "逐层 `renderOrder` 取 `mesh.userData.layerIndex`（back-to-front）。",
  })
}

/** 上传文件 -> Source；没上传返回 undefined。 */
function resolveSource(
  file:
    | { filepath: string; originalFilename?: string | null; size?: number }
    | undefined,
): Source | undefined {
  if (!file?.filepath) return undefined
  const label = file.originalFilename ?? basename(file.filepath)
  return {
    key: `upload:${file.filepath}:${file.size ?? 0}`,
    path: file.filepath,
    label,
  }
}

// ────────────────────────────── main ──────────────────────────────

/**
 * 实时进度：就地更新的 markdown（`streamable`），带已用时间与阶段列表。
 *
 * 每个阶段后 `await setTimeout(0)` 让出**宏任务**，确保这次 patch 在下一个
 * 长同步段（推理 / 打包）之前刷到前端。
 */
function makeProgress(status: {
  update: (body: string) => void
}): StageProgress {
  const t0 = Date.now()
  const lines: string[] = []
  return async (stage, detail) => {
    lines.push(`- ${stage}${detail ? ` · ${detail}` : ""}`)
    status.update(
      `**生成中… ${((Date.now() - t0) / 1000).toFixed(1)}s**\n\n${lines.join("\n")}`,
    )
    await new Promise<void>((resolve) => setTimeout(resolve, 0))
  }
}

function fmtS(ms: number): string {
  return `${(ms / 1000).toFixed(1)}s`
}

async function main(): Promise<void> {
  const port = Number(process.env.WEBUI_PORT ?? 3333)
  const shutdown = async (): Promise<void> => {
    await disposeDevice()
    process.exit(0)
  }
  process.once("SIGINT", () => void shutdown())
  process.once("SIGTERM", () => void shutdown())

  console.log(`[webui] 图片 -> GLB @ http://localhost:${port}`)
  console.log(`[webui] 点「生成 GLB」触发；产物临时目录：${OUTPUT_DIR}`)
  await run(executor, { server: { port }, appearance: { mode: "dark" } })
}

/** 仅当作为入口执行时才起服务（便于测试脚本 import 上面的函数）。 */
const isEntry =
  process.argv[1] !== undefined &&
  resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url))

if (isEntry) {
  main().catch((err) => {
    console.error("\n[错误]", err instanceof Error ? err.message : err)
    if (err instanceof Error && err.stack) console.error(err.stack)
    process.exitCode = 1
  })
}

// 供测试脚本使用（不经 backroad）。
export {
  ensureBuilt as buildView,
  resolveSource,
  type Source,
  type UiConfig,
  type View,
}
