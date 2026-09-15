/**
 * 场景加载状态（全局信号）+ worker 生命周期。
 *
 * 数据流：
 *   `startLoad()` -> worker（下载/解析/转换/排序/打包）
 *                -> `meta` 信号（信息面板 + 相机取景）
 *                -> 顶点缓冲交给渲染器（`viewer.pushSplats`）
 *
 * 相机内参的来源有三档，按优先级：
 *   1. PLY 里的 `intrinsic`/`image_size`（`--full` 模式的产物）
 *   2. 同目录 sidecar `*.camera.json`（`export-ply.ts` 默认产出）
 *   3. 兜底：假定视场角 40°（只影响取景，不影响几何）
 */

// 副作用导入：把 useValue() 装到 Signal.prototype（供 React 组件消费）
import "./signals-hook.ts"
import { computed, signal } from "@preact/signals-react"
import {
  loadScene,
  type SceneLoader,
} from "../spatial-scene/render/loader.browser.ts"
import type { Vec3 } from "../spatial-scene/render/math.ts"
import type {
  LoadProgress,
  PackedSplats,
  SceneMeta,
  SortCamera,
} from "../spatial-scene/render/types.ts"
import { resetCamera } from "./camera.ts"

/** 加载状态机。 */
export type LoadStatus = "idle" | "loading" | "ready" | "error"

/** 默认 PLY 地址（`public/exports/` 由 `npm run sample` 生成）。 */
export const DEFAULT_PLY_URL = "/exports/sample.ply"

/** 场景取景参数：相机在哪、看多远、用多大视场。 */
export interface SceneView {
  /** 轨道枢轴（PLY 系，恒在 +z 轴上：x=y=0）。 */
  pivot: Vec3
  /** 参考距离（= `pivot.z`）。`zoom=1` 时相机恰在世界原点。 */
  referenceDistance: number
  /** 参考相机焦距（原图域 px）。 */
  focalPx: number
  /** 原图尺寸 `[width, height]`。 */
  imageSize: [number, number]
  /** 焦距来源，面板上会显示（用来区分「真内参」和「兜底」）。 */
  focalSource: "ply" | "sidecar" | "fallback"
  /** 参考照片 URL（叠加比对用；缺失时为 null）。 */
  referenceImageUrl: string | null
  /** 场景 z 的范围，用于近远裁剪与信息显示。 */
  depthRange: [number, number]
}

/** 当前加载的 PLY 地址。 */
export const plyUrl = signal(DEFAULT_PLY_URL)

export const status = signal<LoadStatus>("idle")
export const progress = signal<LoadProgress>({ stage: "fetch", ratio: 0 })
export const error = signal<string | null>(null)
export const meta = signal<SceneMeta | null>(null)
export const view = signal<SceneView | null>(null)
/** 上次排序信息（用时 + 用的哪台相机）。 */
export const sortInfo = signal<{
  sortMs: number
  packMs: number
  camera: SortCamera
  ms: number
} | null>(null)
/** 重排序进行中。 */
export const sorting = signal(false)

/** 参考照片地址（派生；叠加比对按钮据此启用/禁用）。 */
export const referenceImageUrl = computed(
  () => view.value?.referenceImageUrl ?? null,
)

/** worker 门面（渲染器与 store 之间的桥）。 */
let loader: SceneLoader | null = null

export function getLoader(): SceneLoader | null {
  return loader
}

/** 加载进度百分比（0..100，取整）。 */
export function progressPercent(p: LoadProgress): number {
  return Math.round(p.ratio * 100)
}

/**
 * 开始加载场景。
 *
 * @param url PLY 地址（默认 {@link DEFAULT_PLY_URL}）。
 * @param onPacked 顶点缓冲就绪回调（渲染器挂载后才会用到，可能早于渲染器）。
 */
export async function startLoad(
  url: string = plyUrl.peek(),
  onPacked?: (packed: PackedSplats) => void,
): Promise<void> {
  if (status.peek() === "loading") return
  status.value = "loading"
  error.value = null
  meta.value = null
  view.value = null
  sortInfo.value = null
  progress.value = { stage: "fetch", ratio: 0 }
  plyUrl.value = url

  try {
    const next = await loadScene(url, (p) => {
      progress.value = p
    })
    loader?.dispose()
    loader = next
    if (!next.result) throw new Error("worker 未返回结果")

    const { meta: sceneMeta, packed } = next.result
    meta.value = sceneMeta
    sortInfo.value = {
      sortMs: packed.sortMs,
      packMs: packed.packMs,
      camera: { eye: [0, 0, 0], forward: [0, 0, 1] },
      ms: packed.sortMs + packed.packMs,
    }
    view.value = await buildSceneView(sceneMeta, url)
    resetCamera()
    onPacked?.(packed)
    status.value = "ready"
  } catch (err) {
    status.value = "error"
    error.value = err instanceof Error ? err.message : String(err)
  }
}

/**
 * 用新相机重排序（不重新下载/解析）。
 *
 * @param camera 排序相机（PLY 系）。
 * @returns 新的顶点缓冲；失败或未加载时返回 null。
 */
export async function resort(
  camera: SortCamera,
  onPacked?: (packed: PackedSplats) => void,
): Promise<PackedSplats | null> {
  if (!loader || sorting.peek()) return null
  sorting.value = true
  try {
    const packed = await loader.resort(camera)
    sortInfo.value = {
      sortMs: packed.sortMs,
      packMs: packed.packMs,
      camera,
      ms: packed.sortMs + packed.packMs,
    }
    onPacked?.(packed)
    return packed
  } finally {
    sorting.value = false
  }
}

/** 释放 worker（页面卸载/换场景时）。 */
export function disposeLoader(): void {
  loader?.dispose()
  loader = null
}

/** 由 PLY 元信息 + sidecar 推导取景参数。 */
async function buildSceneView(
  sceneMeta: SceneMeta,
  plyPath: string,
): Promise<SceneView> {
  const { bounds, aux } = sceneMeta
  // 枢轴放在 +z 轴上：x=y=0，z 取前后包围盒中点
  const pivotZ = (bounds.min[2] + bounds.max[2]) / 2
  const pivot: Vec3 = [0, 0, pivotZ]
  const depthRange: [number, number] = [bounds.min[2], bounds.max[2]]

  // 1) PLY 自带内参（--full 模式）
  if (aux.intrinsic && aux.imageSize) {
    return {
      pivot,
      referenceDistance: Math.abs(pivotZ),
      focalPx: aux.intrinsic[0],
      imageSize: [aux.imageSize[0], aux.imageSize[1]],
      focalSource: "ply",
      referenceImageUrl: null, // full 模式没有 sidecar 图片，交给下面兜底
      depthRange,
    }
  }

  // 2) sidecar json（默认路径）
  const sidecar = await fetchCameraSidecar(plyPath)
  if (sidecar) {
    return {
      pivot,
      referenceDistance: Math.abs(pivotZ),
      focalPx: sidecar.fx,
      imageSize: [sidecar.width, sidecar.height],
      focalSource: "sidecar",
      referenceImageUrl: sidecar.imageUrl,
      depthRange,
    }
  }

  // 3) 兜底：假定竖向 40° 视场，图像尺寸取「未知」
  //    （取 1:1，于是 contain 缩放的基准与原图无关，仅用于看东西）
  const fallbackH = 1024
  const focalFallback = fallbackH / (2 * Math.tan((40 * Math.PI) / 360))
  return {
    pivot,
    referenceDistance: Math.abs(pivotZ),
    focalPx: focalFallback,
    imageSize: [fallbackH, fallbackH],
    focalSource: "fallback",
    referenceImageUrl: null,
    depthRange,
  }
}

interface CameraSidecar {
  fx: number
  fy: number
  width: number
  height: number
  imageUrl: string | null
}

/** 读取同目录的 `*.camera.json`；缺失/格式不符时返回 null（不抛错）。 */
async function fetchCameraSidecar(
  plyPath: string,
): Promise<CameraSidecar | null> {
  const url = sidecarUrlFor(plyPath)
  try {
    const res = await fetch(url)
    if (!res.ok) return null
    const json: unknown = await res.json()
    const first = Array.isArray(json) ? json[0] : json
    if (!first || typeof first !== "object") return null
    const rec = first as Record<string, unknown>
    const fx = Number(rec.fx)
    const width = Number(rec.width)
    const height = Number(rec.height)
    if (
      !Number.isFinite(fx) ||
      !Number.isFinite(width) ||
      !Number.isFinite(height)
    ) {
      return null
    }
    const fy = Number.isFinite(Number(rec.fy)) ? Number(rec.fy) : fx
    // 参考照片与 sidecar 同目录（`export-ply.ts` 会复制一份过去）
    const imageUrl =
      typeof rec.img_name === "string" && rec.img_name.length > 0
        ? new URL(rec.img_name, new URL(url, location.href)).href
        : null
    return { fx, fy, width, height, imageUrl }
  } catch {
    return null
  }
}

/** `/exports/sample.ply` -> `/exports/sample.camera.json`。 */
export function sidecarUrlFor(plyPath: string): string {
  return `${plyPath.replace(/\.ply$/i, "")}.camera.json`
}
