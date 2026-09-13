/**
 * 渲染端（web）的数据契约 —— **纯类型**，不 import 任何 three 之外的东西。
 *
 * 这里描述的不是「磁盘上的 GLB」，而是**装好之后**的场景：
 * 一个 three 对象树 + 逐层元数据 + 相机/统计元数据。GLB -> 这个形状的解析在
 * `scene.ts`；这份数据在 `store/viewer.ts` 里以 signal 形式成为全局状态。
 *
 * ── 命名 / 分层的唯一权威是 glTF extras ──
 * node 侧导出器（`scripts/pack/glb.ts`）写了三处结构化信息：
 * - `mesh.extras.layerIndex`：`0..L-1` = 层（0 = 最近），`-1` = 背衬平面；
 * - `mesh.extras.depthRange` / `disparityRange` / `triangleCount` / `vertexCount`；
 * - `scene.extras.{near,far,verticalFOV,aspectRatio,layerDepths,layerRanges,stats}`。
 *
 * 渲染端**只认 extras，不解析名字**：glTF 名字（`layer_00` / `backing`）会被
 * three 的 `createUniqueName` 改名，不可靠；`layerIndex` 才是稳定键。
 */

import type { Mesh, Object3D } from "three"

/** 三维三元组（只读，避免下游误改）。 */
export type Triple = readonly [number, number, number]

/** 场景包围盒（glTF 世界系：Y 上、−Z 前、相机在原点）。 */
export interface SceneBounds {
  readonly center: Triple
  readonly min: Triple
  readonly max: Triple
  readonly size: Triple
}

/** 视差统计的**渲染端摘要**（`scene.extras.stats` 的子集）。 */
export interface DisparitySummary {
  readonly minimum: number
  readonly maximum: number
  /** 中位数（由 `stats.quantiles` 在 `p = 0.5` 处插值）。 */
  readonly median: number
  readonly mean: number
  readonly stdDev: number
}

/** `scene.extras` 解析后的相机 / 统计元数据。 */
export interface SceneMeta {
  /** 投影近平面（米，生成期渲染相机）。 */
  readonly near: number
  /** 投影远平面（米）。 */
  readonly far: number
  /** 垂直视场角（**弧度**）。 */
  readonly verticalFOV: number
  /** 参考图宽高比（`width / height`）。 */
  readonly aspectRatio: number
  /** 纹理是否预乘 α（本管线恒为 `false`）。 */
  readonly premultipliedAlpha: boolean
  /** 逐层代表深度（米，索引与层号对齐，0 = 最近）。 */
  readonly layerDepths: readonly number[]
  /** `[L*2]` 逐层视差域范围 `[lo, hi]`。 */
  readonly layerRanges: readonly number[]
  readonly disparity: DisparitySummary | null
}

/** 一「层」在渲染端的可切换单元（层 或 背衬平面）。 */
export interface LayerState {
  /** 稳定键：`layer_00` / `backing`（由 `layerIndex` 派生，不用 glTF 名字）。 */
  readonly key: string
  /** glTF 侧的名字（仅用于展示）。 */
  readonly name: string
  readonly kind: "layer" | "backing"
  /** `0..L-1`；背衬 = `-1`。 */
  readonly layerIndex: number
  /** 展示名（中文）。 */
  readonly label: string
  /** 该层真实深度带 `[lo, hi]`（米）。 */
  readonly depth: readonly [number, number]
  readonly triangles: number
  readonly vertices: number
  /** 是否可见（用户开关）。 */
  readonly visible: boolean
}

/** 一次载入的完整产物（three 对象树 + 元数据）。 */
export interface LoadedScene {
  readonly url: string
  readonly name: string
  /** glTF 场景根（相机在原点朝 −Z；直接 `<primitive object={root} />`）。 */
  readonly root: Object3D
  /** `key -> Mesh`：可见性 / 材质淡入淡出按 key 找对象。 */
  readonly meshes: ReadonlyMap<string, Mesh>
  readonly meta: SceneMeta
  readonly bounds: SceneBounds
  /** GLB 字节数（自取自 parse，顺带给出可靠进度）。 */
  readonly bytes: number
  /** 初始的逐层状态（`visible` 全为 `true`）。 */
  readonly layers: readonly LayerState[]
}
