/**
 * 渲染端的**全局状态**（signal store）。
 *
 * ── 唯一状态源 ──
 * 整个应用只有这里放状态，组件不持有「业务状态」（只有 three 对象 ref / 动画 ref）。
 * 读取一律走 `$xxx.useValue()`；**`$` 前缀是约定**：signal 是数据作用域而不是 hook，
 * 理由与编译器行为见 `signal-hooks.ts` 头部。非 React 侧（`useFrame` / 事件回调）
 * 一律用 `.peek()` / `.value`，不要在那里 `useValue()`。
 *
 * ── 为什么是 signal 而不是 context/zustand ──
 * 渲染端有**非 React 的写入者**（异步 GLB 载入、r3f 的 `useFrame` 里每帧读可见性）。
 * signal 的好处是：React 侧订阅是精确的（`useSyncExternalStore`），非 React 侧可以直接
 * `.value =` / `.peek()` 写读，不需要 `invalidate` 调度 —— 于是「载入完成」不需要
 * 把 three 对象塞进 `useState`，也就不会在 React Compiler 的 memo 化下出意外。
 *
 * ── 载入流程的所有权 ──
 * 载入是**命令式**的（模块级 async 函数），不是 effect：`loadGlb*` 自己写
 * `$phase` / `$error` / `$progress`，并用一个自增 `loadSeq` 处理竞态
 * （用户在载入途中再丢一个文件时，旧的那次结果直接丢弃并释放）。
 */

import { computed, signal } from "@preact/signals-react"
import {
  disposeScene,
  type LoadProgress,
  loadGlbScene,
  SAMPLE_URL,
} from "../viewer/scene.ts"
import type { LayerState, LoadedScene } from "../viewer/types.ts"
import { disposeSceneWireframe } from "../viewer/wireframe.ts"
import "./signal-hooks.ts"

export type { LayerState, LoadedScene, SceneMeta } from "../viewer/types.ts"
export { SAMPLE_URL }

// ────────────────────────────── 状态 ──────────────────────────────

export type SourceKind = "none" | "url" | "file"

export interface ModelSource {
  readonly kind: SourceKind
  readonly url: string
  readonly name: string
}

export type LoadPhase = "empty" | "loading" | "ready" | "error"

/** 当前 GLB 来源（空态 / URL / 本地文件）。 */
export const $source = signal<ModelSource>({ kind: "none", url: "", name: "" })

/** 载入相位。 */
export const $phase = signal<LoadPhase>("empty")

/** 载入失败原因（`$phase === "error"` 时有值）。 */
export const $error = signal<string | null>(null)

/** 载入进度：`0..1`；`null` = 总长未知（objectURL / gzip）。 */
export const $progress = signal<number | null>(null)

/** 已装配的场景（null = 空态）。 */
export const $scene = signal<LoadedScene | null>(null)

/** 逐层状态（**唯一**的可见性权威，顺序 = 层号升序，背衬在最后）。 */
export const $layers = signal<readonly LayerState[]>([])

/** 面板展开 / 收起。 */
export const $panelOpen = signal(true)

/** 地面网格。 */
export const $showGrid = signal(true)

/** 右下角坐标轴。 */
export const $showGizmo = signal(true)

/**
 * 视差摆动：**固定焦点**，让相机绕**参考机位**做小幅圆周运动，露出层间视差。
 * 驱动在 `viewer/ParallaxSway.tsx`；这里只是开关。不要用 OrbitControls 的
 * `autoRotate`（那是绕焦点公转，语义不同）。
 */
export const $sway = signal(false)

/**
 * 线框（几何调试）视图。
 *
 * **独占**语义：开启时只画线框，标准（纹理）渲染不参与绘制；关闭时原样换回。
 * 实现是原地换 `mesh.material`（几何 / renderOrder / 可见性逻辑全部复用），
 * 见 `viewer/wireframe.ts`。
 */
export const $wireframe = signal(false)

/** 独显（solo）：非 null 时只显示这一层。 */
export const $solo = signal<string | null>(null)

/** 面板里悬停 / 高亮的层 key（用于联动高亮）。 */
export const $hovered = signal<string | null>(null)

/** 「重置到参考视角」的触发计数器（+1 即请求一次，r3f 侧监听）。 */
export const $resetViewToken = signal(0)

/** 拖拽悬停中（覆盖层用）。 */
export const $dragActive = signal(false)

/** 浏览器是否支持 WebGL2（不支持时显示降级提示）。 */
export const $webgl = signal(true)

// ────────────────────────────── 派生 ──────────────────────────────

/** 可见层数（考虑 solo）。 */
export const $visibleCount = computed(() => {
  const solo = $solo.value
  const layers = $layers.value
  if (solo) return layers.filter((layer) => layer.key === solo).length
  return layers.filter((layer) => layer.visible).length
})

/** 三角面总数。 */
export const $totalTriangles = computed(() =>
  $layers.value.reduce((sum, layer) => sum + layer.triangles, 0),
)

/** 顶点总数。 */
export const $totalVertices = computed(() =>
  $layers.value.reduce((sum, layer) => sum + layer.vertices, 0),
)

/** 深度跨度（米），来自元数据；缺失时用逐层范围的并集。 */
export const $depthSpan = computed<readonly [number, number]>(() => {
  const scene = $scene.value
  if (!scene) return [0, 0]
  const depths = scene.meta.layerDepths
  if (depths.length > 0) return [depths[0], depths[depths.length - 1]]
  const layers = $layers.value.filter((layer) => layer.kind === "layer")
  if (layers.length === 0) return [scene.meta.near, scene.meta.far]
  return [layers[0].depth[0], layers[layers.length - 1].depth[1]]
})

// ────────────────────────────── 动作 ──────────────────────────────

/**
 * 释放一个**已挂载过**的场景。
 *
 * 顺序不能倒：先拆线框材质（它会顺手把 `mesh.material` 还回基准材质），
 * 再 `disposeScene` —— 否则线框模式下贴图会被漏掉（见 `disposeSceneWireframe`）。
 */
function releaseScene(scene: LoadedScene): void {
  disposeSceneWireframe(scene)
  disposeScene(scene.root)
}

/** 自增序号：只有最后一次载入的结果会被采纳（竞态保护）。 */
let loadSeq = 0

/** 从 URL 载入（`?glb=`、内置样例）。 */
export async function loadGlbUrl(url: string, name: string): Promise<void> {
  await runLoad({ kind: "url", url, name }, url, name)
}

/** 从本地文件载入（拖拽 / 文件选择）。objectURL 由 store 负责释放。 */
export async function loadGlbFile(file: File): Promise<void> {
  const url = URL.createObjectURL(file)
  const adopted = await runLoad(
    { kind: "file", url, name: file.name },
    url,
    file.name,
  )
  // 没被采纳（竞态落败 / 载入失败）时，store 不会持有这个 URL，自己释放。
  if (!adopted) revokeUrl(url)
}

async function runLoad(
  source: ModelSource,
  url: string,
  name: string,
): Promise<boolean> {
  const seq = ++loadSeq
  const previousSource = $source.peek()
  $source.value = source
  $phase.value = "loading"
  $error.value = null
  $progress.value = null
  $solo.value = null
  $hovered.value = null

  const onProgress: LoadProgress = (ratio) => {
    if (seq === loadSeq) $progress.value = ratio
  }

  try {
    const loaded = await loadGlbScene(url, name, onProgress)
    if (seq !== loadSeq) {
      // 已经被更新的载入取代：释放掉这份，别泄漏。
      disposeScene(loaded.root)
      return false
    }
    const previous = $scene.peek()
    $scene.value = loaded
    $layers.value = loaded.layers
    $progress.value = 1
    $phase.value = "ready"
    if (previous && previous.root !== loaded.root) releaseScene(previous)
    // 换源成功后才释放旧的 objectURL（失败时调用方还会用它重试 / 读取）。
    if (previousSource.kind === "file" && previousSource.url !== url) {
      revokeUrl(previousSource.url)
    }
    return true
  } catch (error) {
    if (seq !== loadSeq) return false
    $phase.value = "error"
    $error.value = error instanceof Error ? error.message : String(error)
    $progress.value = null
    return false
  }
}

/** 清空到空态（释放当前场景）。 */
export function clearScene(): void {
  loadSeq++
  const previous = $scene.peek()
  $scene.value = null
  $layers.value = []
  $phase.value = "empty"
  $error.value = null
  $progress.value = null
  $solo.value = null
  if (previous) releaseScene(previous)
  const source = $source.peek()
  if (source.kind === "file") revokeUrl(source.url)
  $source.value = { kind: "none", url: "", name: "" }
}

function revokeUrl(url: string): void {
  if (!url) return
  try {
    URL.revokeObjectURL(url)
  } catch {
    // 已释放 / 不支持：忽略
  }
}

export function setLayerVisible(key: string, visible: boolean): void {
  $layers.value = $layers.value.map((layer) =>
    layer.key === key ? { ...layer, visible } : layer,
  )
}

export function toggleLayer(key: string): void {
  const current = $layers.peek().find((layer) => layer.key === key)
  if (current) setLayerVisible(key, !current.visible)
}

/** 全显 / 全隐（同时清掉 solo，否则看起来「没反应」）。 */
export function setAllVisible(visible: boolean): void {
  $solo.value = null
  $layers.value = $layers.value.map((layer) => ({ ...layer, visible }))
}

/** 独显切换：再点一次取消。 */
export function toggleSolo(key: string): void {
  $solo.value = $solo.peek() === key ? null : key
}

/** 请求重置到参考视角。 */
export function requestResetView(): void {
  $resetViewToken.value = $resetViewToken.peek() + 1
}

// ────────────────────────────── 启动 ──────────────────────────────

/**
 * 决定首屏载入什么：
 * 1. `?glb=<url>` —— 显式指定（生成端产物 / 任意 GLB）；
 * 2. 否则尝试内置样例 `/models/sample.glb`（Vite 从 `public/` 提供）；
 * 3. 都没有 → 停在空态（拖拽 / 选择文件）。
 */
export function bootstrap(): void {
  const params = new URLSearchParams(window.location.search)
  const explicit = params.get("glb")
  if (explicit) {
    void loadGlbUrl(explicit, basename(explicit))
    return
  }
  void loadSampleIfPresent()
}

async function loadSampleIfPresent(): Promise<void> {
  try {
    const response = await fetch(SAMPLE_URL, { method: "HEAD" })
    const type = response.headers.get("content-type") ?? ""
    if (!response.ok || type.includes("text/html")) return
    await loadGlbUrl(SAMPLE_URL, "sample.glb")
  } catch {
    // 样例不存在 / 离线：安静地停在空态。
  }
}

function basename(url: string): string {
  const clean = url.split("?")[0].split("#")[0]
  const parts = clean.split("/")
  return parts[parts.length - 1] || clean
}
