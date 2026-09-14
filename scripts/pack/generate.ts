/**
 * 端到端生成：**图片 -> 推理 -> 分层 -> mesh -> GLB**（node 侧交付链路）。
 *
 * ── 在全链路里的位置 ──
 * ```
 * 上传图片 -> runSharp(onnx) -> Gaussians3D(metric)
 *          -> assembleWSplatScene -> renderLayerStack -> (refineLayers)
 *          -> toMeshingInput -> buildMeshScene -> exportMeshSceneToGlb
 * ```
 * 调用方（`scripts/pack/sample.ts`、后续的 UI 入口）只需要这里的两个函数；
 * 推理与分层/mesh 的细节都在各自模块里。
 *
 * ── 为什么不落 PLY ──
 * 推理产物直接在内存里装配成 `WSplatScene`（`assembleWSplatScene`），
 * 省掉 60MB 级 PLY 的往返；PLY 只用于既有的 golden / 目测脚本。
 */

import { basename } from "node:path"
import { superSplatCameraPose } from "../../src/spatial-scene/export/camera.ts"
import { runSharp } from "../../src/spatial-scene/infer/index.ts"
import type {
  CreateSession,
  ExecutionProviderHint,
  SessionCapabilities,
} from "../../src/spatial-scene/infer/session.ts"
import {
  type LayerCompletionOptions,
  type LayeredRGBD,
  type LayerSamplingMethod,
  refineLayers,
} from "../../src/spatial-scene/layering/index.ts"
import {
  buildMeshScene,
  type MeshingOptions,
  type MeshScene,
  toMeshingInput,
} from "../../src/spatial-scene/meshing/index.ts"
import type { SourceImage } from "../../src/spatial-scene/sharp/preprocess.ts"
import type { Gaussians3D } from "../../src/spatial-scene/sharp/types.ts"
import type { WSplatFrame } from "../../src/spatial-scene/wsplat/types.ts"
import { MODEL_FP16, prepareOrtEnv } from "../utils/common.ts"
import { type LoadedImage, loadImage } from "../utils/image.ts"
import { renderLayerStack } from "../utils/meshing-scene.ts"
import type { CameraPose, WSplatScene } from "../utils/wsplat-scene.ts"
import { exportMeshSceneToGlb } from "./glb.ts"

// ────────────────────────────── 推理 ──────────────────────────────

/**
 * 阶段进度回调：在**每个阶段边界**调用（调用方负责 flush，例如更新 UI）。
 *
 * `detail` 是简短状态（如 `12.3s` / `dml` / `2.1 MB`）。返回 Promise 时会被 `await`，
 * 以便调用方让出事件循环、把进度推给前端。
 */
export type StageProgress = (
  stage: string,
  detail?: string,
) => void | Promise<void>

export interface InferSceneOptions {
  /** 输入图路径（已落盘的任意格式，sharp 可读）。 */
  imagePath: string
  /** 执行提供者偏好（默认 `auto`：Windows -> dml，macOS -> webgpu）。 */
  ep?: ExecutionProviderHint
  /** 模型路径；缺省 fp16。 */
  modelPath?: string
  /**
   * 会话工厂；缺省 `scripts/utils/platform.node.ts` 的实现。
   *
   * 用于计时（会话加载单独量一次）与测试注入。
   */
  createSession?: CreateSession
  /** 阶段进度（可选）。 */
  onStage?: StageProgress
}

/** 一张图 -> 原始高斯 + 位姿（分辨率无关，可在内存里缓存）。 */
export interface InferredScene {
  /** 度量空间高斯（`result.metric`）。 */
  readonly gaussians: Gaussians3D
  /** 相机位姿（PLY 坐标系，供 `assembleWSplatScene` 按任意宽度装配）。 */
  readonly pose: CameraPose
  readonly loaded: LoadedImage
  readonly capabilities: SessionCapabilities
  /** 会话加载（模型读盘 + EP 初始化）耗时（ms）。`inferMs` **含**它。 */
  readonly sessionLoadMs: number
  /** 载入图片 + 会话加载 + 前向 + 反投影的总耗时（ms）。 */
  readonly inferMs: number
  readonly gaussianCount: number
  /** 推理得到的反投影矩阵（4x4 行主序），调试用。 */
  readonly unprojectionMatrix: Float64Array
}

/**
 * 跑一次 SHARP 推理，得到**原始**高斯 + 位姿（不装配场景，不落盘）。
 *
 * 分辨率由调用方在装配时决定（`assembleWSplatScene({ width })`），
 * 所以改渲染宽度不需要重跑推理。
 *
 * ⚠ 必须在 `createNodeSession`（onnxruntime-node）被求值前调用 `prepareOrtEnv()`；
 * 这里在动态 import 之前调用，`platform.node.ts` 内部也再 patch 一次（幂等）。
 */
export async function inferSceneFromImage(
  options: InferSceneOptions,
): Promise<InferredScene> {
  prepareOrtEnv("error")

  await options.onStage?.("载入图片")
  const loaded = await loadImage(options.imagePath)
  const { createNodeSession } = await import("../utils/platform.node.ts")
  const createSession = options.createSession ?? createNodeSession

  // 会话加载（读模型 + 建 EP 会话）单独计时：它是「推理耗时」里最大的一块固定成本，
  // 与分辨率无关，混在 inferMs 里看不出到底是加载慢还是前向慢。
  let sessionLoadMs = 0
  const timedCreateSession: CreateSession = async (createOptions) => {
    const t = Date.now()
    const session = await createSession(createOptions)
    sessionLoadMs = Date.now() - t
    return session
  }

  await options.onStage?.(
    "推理",
    `${loaded.image.width}×${loaded.image.height}`,
  )
  const t0 = Date.now()
  const { session, result } = await runSharp({
    image: loaded.image,
    fPx: loaded.fPx,
    imageWidth: loaded.image.width,
    createSession: timedCreateSession,
    model: { modelPath: options.modelPath ?? MODEL_FP16 },
    provider: options.ep ?? "auto",
  })
  const inferMs = Date.now() - t0
  await session.dispose()
  await options.onStage?.(
    "推理完成",
    `${(inferMs / 1000).toFixed(1)}s · ${result.capabilities.activeProvider}`,
  )

  const pose = superSplatCameraPose({
    name: basename(options.imagePath),
    focalLengthPx: loaded.fPx,
    imageShape: [loaded.image.width, loaded.image.height],
    extrinsics: result.extrinsics,
  })

  return {
    gaussians: result.metric,
    pose,
    loaded,
    capabilities: result.capabilities,
    sessionLoadMs,
    inferMs,
    gaussianCount: result.metric.opacities.length,
    unprojectionMatrix: result.unprojectionMatrix,
  }
}

// ────────────────────────────── 分层 -> mesh -> GLB ──────────────────────────────

export interface BuildGlbOptions {
  /** 层数。默认 `8`。 */
  layers?: number
  /** 层放置方法。默认 `"quantile"`。 */
  method?: LayerSamplingMethod
  /** 直方图箱数。默认 `256`。 */
  binCount?: number
  /** 层间重叠（只影响报告的 `layerRanges`）。默认 `DEFAULT_LAYER_OVERLAP`。 */
  overlap?: number
  /** 原图回写；给了 `sourceImage` 时默认开。 */
  refine?: boolean
  /**
   * 几何补齐（`refineLayers` 的 own-gap 回填 + hidden 有界外推）。
   *
   * 缺省 = 开启（`{}`）；`false` = 关闭（退回纯颜色回写）。
   * 需要 `sourceImage`（与 `refine` 同一条件）。
   */
  complete?: LayerCompletionOptions | false
  /** Draco 压缩几何（默认开）。 */
  draco?: boolean
  /** meshing 选项（含 `lod` 出面）。缺省 = 逐像素出面（百万级）。 */
  mesh?: MeshingOptions
  /** 阶段进度（可选）。 */
  onStage?: StageProgress
}

/**
 * 层间重叠（层深范围交叠）的默认值：视差域绝对量 **`0.02`**。
 *
 * ── 它是什么 / 不是什么 ──
 * 只作用于 `toMeshingInput` 报告的 `layerRanges`（渲染侧按层深范围剔除/裁剪的余量），
 * **不改层分配 / 排列 / 几何**（重复绘制会破坏参考视角无损，详见 `layering/bands.ts`）。
 *
 * ── 为什么默认不是 0 ──
 * 几何补齐把隐藏区深度夹到本层带 `[b_k, b_{k+1}]`，边界上的顶点若用零余量区间做剔除，
 * 会在切分处被削掉一条；给一点余量最省事。
 * 量级参考：`L=8` 的平均层带宽是 `1/8 = 0.125`，`0.02 ≈ 层带宽的 1/6`，
 * 足以吸收边界舍入而不产生明显 over-draw；层数越多可适当调小。
 */
export const DEFAULT_LAYER_OVERLAP = 0.02

/** 一次「场景 -> GLB」的产物。 */
export interface BuildGlbResult {
  readonly meshScene: MeshScene
  readonly layered: LayeredRGBD
  /** 用于预览的层帧（若 refine 生效则是回写后的颜色）。 */
  readonly frames: readonly WSplatFrame[]
  readonly glb: Uint8Array
  readonly refined: boolean
  /** 分层渲染耗时（ms）。 */
  readonly layersMs: number
  /** 网格化耗时（ms）。 */
  readonly meshMs: number
  /** GLB 打包耗时（ms，含纹理 PNG 编码 + Draco）。 */
  readonly glbMs: number
}

/**
 * `WSplatScene (+ 原图) -> MeshScene -> GLB 字节`。
 *
 * 原图缺省则跳过 refine（例如直接拿现成 PLY 的场景）。
 */
export async function buildGlb(
  device: GPUDevice,
  scene: WSplatScene,
  sourceImage: SourceImage | undefined,
  options: BuildGlbOptions = {},
): Promise<BuildGlbResult> {
  const t0 = Date.now()
  await options.onStage?.("分层渲染")
  const base = await renderLayerStack(device, scene, {
    layers: options.layers ?? 8,
    method: options.method ?? "quantile",
    binCount: options.binCount ?? 256,
  })
  const layersMs = Date.now() - t0
  await options.onStage?.("分层完成", `${(layersMs / 1000).toFixed(1)}s`)

  let layered: LayeredRGBD = base
  let refined = false
  if (options.refine !== false && sourceImage) {
    layered = refineLayers(base, sourceImage, { complete: options.complete })
    refined = true
  }

  const t1 = Date.now()
  await options.onStage?.("网格化")
  const input = toMeshingInput(
    layered,
    scene.camera,
    options.overlap ?? DEFAULT_LAYER_OVERLAP,
  )
  const meshScene = buildMeshScene(input, options.mesh ?? {})
  const meshMs = Date.now() - t1
  await options.onStage?.("网格化完成", `${(meshMs / 1000).toFixed(1)}s`)

  const t2 = Date.now()
  await options.onStage?.("打包 GLB")
  const glb = await exportMeshSceneToGlb(meshScene, { draco: options.draco })
  const glbMs = Date.now() - t2
  await options.onStage?.(
    "GLB 完成",
    `${(glbMs / 1000).toFixed(1)}s · ${(glb.length / 1024 / 1024).toFixed(2)} MB`,
  )

  return {
    meshScene,
    layered,
    frames: layered.frames,
    glb,
    refined,
    layersMs,
    meshMs,
    glbMs,
  }
}
