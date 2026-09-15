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
 *   3. 兜底：按**默认 35mm 等效焦距**（EXIF 也拿不到焦距时的那一条链）
 *      反推一组自洽内参，见 `fallbackIntrinsics()`
 */

// 副作用导入：把 useValue() 装到 Signal.prototype（供 React 组件消费）
import "./signals-hook.ts"
import { computed, signal } from "@preact/signals-react"
import {
  loadScene,
  loadSceneFromBytes,
  type SceneLoader,
} from "../spatial-scene/render/loader.browser.ts"
import type { Vec3 } from "../spatial-scene/render/math.ts"
import type {
  LoadProgress,
  PackedSplats,
  SceneMeta,
  SortCamera,
} from "../spatial-scene/render/types.ts"
import {
  convertFocallength,
  resolveFocalLength35mm,
} from "../spatial-scene/sharp/fov.ts"
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

/**
 * 当前数据源。
 *
 * 本地文件与 URL 在加载完后行为几乎一样，差别只有两处：
 * - 本地文件**没有同目录**，不会去试探 sidecar json（浏览器拿不到文件的旁边有什么）；
 * - 重试时得重新读一遍那个 File 对象，而不是重新 fetch 地址。
 */
export type PlySource =
  | { kind: "url"; url: string }
  | { kind: "file"; name: string }

/** 当前数据源（URL 或拖入/选中的本地文件）。 */
export const plySource = signal<PlySource>({
  kind: "url",
  url: DEFAULT_PLY_URL,
})

/** 最近一次读入的本地文件（重试用；File 只是句柄，不占内存）。 */
let lastFile: File | null = null

/** 顶点缓冲就绪回调：`startLoad` 只在首帧传一次，重试/换文件时要沿用。 */
let onPackedHook: ((packed: PackedSplats) => void) | undefined

/** 面板/提示里显示的数据源名字。 */
export const plyLabel = computed(() => {
  const s = plySource.value
  if (s.kind === "file") return s.name
  const cut = s.url.split("?")[0]
  return cut.slice(cut.lastIndexOf("/") + 1) || s.url
})

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
 * 开始加载场景（URL）。
 *
 * @param url PLY 地址（默认用当前数据源里的地址）。
 * @param onPacked 顶点缓冲就绪回调（只需首次传入，之后会被记住）。
 */
export async function startLoad(
  url?: string,
  onPacked?: (packed: PackedSplats) => void,
): Promise<void> {
  const current = plySource.peek()
  const target = url ?? (current.kind === "url" ? current.url : DEFAULT_PLY_URL)
  plySource.value = { kind: "url", url: target }
  lastFile = null
  return runLoad(
    onPacked,
    (onProgress) => loadScene(target, onProgress),
    target,
  )
}

/**
 * 加载本地 PLY（拖拽 / 文件选择）。
 *
 * 不做 sidecar 试探：浏览器无法知道用户本地文件旁边有什么，
 * 所以直接走 `buildSceneView` 的兵底档（PLY 带内参则用 PLY 的）。
 */
export async function startLoadFromFile(
  file: File,
  onPacked?: (packed: PackedSplats) => void,
): Promise<void> {
  if (!/\.ply$/i.test(file.name)) {
    status.value = "error"
    error.value = `只支持 .ply 文件（收到的是「${file.name}」）`
    return
  }
  const buffer = await file.arrayBuffer()
  lastFile = file
  plySource.value = { kind: "file", name: file.name }
  return runLoad(
    onPacked,
    (onProgress) => loadSceneFromBytes(buffer, onProgress),
    null,
  )
}

/** 重新加载当前数据源（错误浮层与面板的「重新加载」共用）。 */
export async function reload(
  onPacked?: (packed: PackedSplats) => void,
): Promise<void> {
  const src = plySource.peek()
  if (src.kind === "file" && lastFile)
    return startLoadFromFile(lastFile, onPacked)
  return startLoad(src.kind === "url" ? src.url : DEFAULT_PLY_URL, onPacked)
}

/**
 * 加载主流程（URL / 文件共用）。
 *
 * @param sidecarBase 推导 sidecar json 的基准路径；`null` = 不试（本地文件）。
 */
async function runLoad(
  onPacked: ((packed: PackedSplats) => void) | undefined,
  load: (onProgress: (p: LoadProgress) => void) => Promise<SceneLoader>,
  sidecarBase: string | null,
): Promise<void> {
  if (onPacked) onPackedHook = onPacked
  // 加载中再触发（重试按钮 / 又拖一个文件）直接忽略：worker 没有取消机制，
  // 硬抢会把两个加载的收尾互相踩掉
  if (status.peek() === "loading") return

  // 换场景时**先**释放旧 worker：它握着 60 MB+ 的 SoA，而且调用方已经
  // 不再需要它（resort 也只对当前场景有意义）。等到新场景加载完再释放
  // 会让内存峰值翻倍。注意必须在 try 之外——它自己的异常不应该把整个
  // 加载判成失败（早先 dispose 的 bug 就是这么暴露出来的）。
  loader?.dispose()
  loader = null

  status.value = "loading"
  error.value = null
  meta.value = null
  view.value = null
  sortInfo.value = null
  progress.value = { stage: "fetch", ratio: 0 }

  try {
    const next = await load((p) => {
      progress.value = p
    })
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
    view.value = await buildSceneView(sceneMeta, sidecarBase)
    resetCamera()
    onPackedHook?.(packed)
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
  // 记住这次用的是哪个 loader：过程中可能被换场景替掉
  const active = loader
  sorting.value = true
  try {
    const packed = await active.resort(camera)
    // 换场景了（loader 已被替换/释放）：这次结果属于旧场景，丢掉。
    // 不判的话晚到的排序结果会覆写新场景的 sortInfo，而且顶点缓冲
    // 也不属于当前渲染器了。
    if (loader !== active) return null
    sortInfo.value = {
      sortMs: packed.sortMs,
      packMs: packed.packMs,
      camera,
      ms: packed.sortMs + packed.packMs,
    }
    onPacked?.(packed)
    return packed
  } catch {
    // 释放中的 worker（换场景 / 卸载）会让这次 RPC 抛
    // "Proxy has been released and is not useable"。这不是用户需要
    // 知道的错误，也不能让它变成 unhandled rejection。
    return null
  } finally {
    sorting.value = false
  }
}

/** 释放 worker（页面卸载/换场景时）。 */
export function disposeLoader(): void {
  loader?.dispose()
  loader = null
}

/** 由 PLY 元信息 + sidecar 推导取景参数。
 *
 * @param sidecarBase 推导 sidecar json 的基准路径；`null` = 跳过这一步
 *   （本地文件没有「同目录」），直接走兵底档。
 */
async function buildSceneView(
  sceneMeta: SceneMeta,
  sidecarBase: string | null,
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

  // 2) sidecar json（URL 默认路径；本地文件跳过）
  const sidecar = sidecarBase ? await fetchCameraSidecar(sidecarBase) : null
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

  // 3) 兜底：没有任何相机信息
  const fb = fallbackIntrinsics()
  return {
    pivot,
    referenceDistance: Math.abs(pivotZ),
    focalPx: fb.focalPx,
    imageSize: fb.imageSize,
    focalSource: "fallback",
    referenceImageUrl: null,
    depthRange,
  }
}

/**
 * 兜底时假定的「图像尺寸」。
 *
 * 取 4:3（与管线默认输入图 3024×2268 同比例，也贴近手机默认出图），
 * 数值 36×27 —— 与 35mm 全画幅同量级，便于读作「传感器 mm」。
 */
const FALLBACK_SENSOR: [number, number] = [36, 27]

/**
 * 无相机信息时的兜底内参。
 *
 * 复用 `sharp/fov.ts` 里那条已有的默认链，而不是另外拍一个角度：
 * - 等效焦距取 `resolveFocalLength35mm(undefined)` = **30 mm**
 *   （与 `load_rgb` 拿不到 EXIF 焦距时的默认值同一个来源，这里不写死 30）；
 * - 换算用同一个 `convertFocallength`。注意它走的是**对角线**（不是按宽），
 *   所以「同样的 30 mm」在不同宽高比下 f_px 并不相同 —— 这正是必须复用
 *   这个函数、而不能自己除一下的原因。
 *
 * 因为口径相同，这套兜底与「默认输入图 3024×2268 + 30 mm」的**三轴视场
 * 逐项相同**（水平 59.96° / 垂直 46.79° / 对角线 71.59°），即没有相机信息时
 * 取景与默认管线完全一致。绝对值无物理含义，真正进入取景的只是
 * `f_px / imageSize` 的比值。
 */
function fallbackIntrinsics(): {
  focalPx: number
  imageSize: [number, number]
} {
  const f35mm = resolveFocalLength35mm(undefined)
  return {
    focalPx: convertFocallength(FALLBACK_SENSOR[0], FALLBACK_SENSOR[1], f35mm),
    imageSize: [...FALLBACK_SENSOR],
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
