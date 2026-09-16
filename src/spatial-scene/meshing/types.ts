/**
 * meshing 的**纯数据类型层**（模块的输入 / 输出契约）。
 *
 * 本文件不得 import WebGPU / DOM / node / three（对照 `layering/types.ts` 的定位）。
 *
 * ══════════════════════════ 模块边界 ══════════════════════════
 * ```
 * image -> splat 场 -> layered RGBAD -> [MeshScene] -> GLB -> three.js
 *                                       ^^^^^^^^^^^ 本模块的产物
 * ```
 * - 只做 **LayeredRgbd -> MeshScene**，产物是**与渲染器无关的内存数据**；
 * - `glb.ts` 负责把 `MeshScene` 序列化成 GLB（纯 TS、零依赖、不落盘）；
 * - **不做体积优化**（不量化、不 Draco、不图集合并）—— 先要正确、能被普通光栅器直接吃。
 *
 * ══════════════════════════ 本阶段的简化（刻意的）══════════════════════════
 * - **只剔全透明**：支撑 = `alpha > cutoff`（默认 `1/255`）+ 深度外扩（见 `support.ts`）；
 * - **不做撕裂**：深度断层不单独处理。每层是渲出来的 heightfield（每像素恰好一个深度），
 *   层内断层在参考视角上被 α 加权坍缩；斜视角下会出现「橡皮布」大三角 ——
 *   这是已知取舍，不是遗漏；
 * - **不做基底平面 / 回填 / 裙边**：参考视角之外的表现留给后续版本。
 *
 * ══════════════════════════ 坐标 / UV / 绕序（不可含糊）══════════════════════════
 * 1. `positions` 是**世界坐标**（米），与 splat 同系（OpenCV：相机在原点朝 `+z`，
 *    `y` 向下，深度 = 视图空间 `z`）；
 * 2. `uvs` 是**图像域归一化**坐标，原点在**左上**：`u = (x+0.5)/width`、`v = (y+0.5)/height`；
 * 3. 三角形绕序使**几何法线朝向相机**（相机空间法线 `z < 0`）；
 * 4. 纹理是**直通（非预乘）线性 RGB + α**，`premultipliedAlpha == false`。
 */

import type { LayerView } from "../layering/types.ts"
import type { WSplatCamera } from "../wsplat/camera.ts"
import type { LodOptions } from "./lod.ts"

/** 支撑 / 深度外扩选项（`support.ts`）。 */
export interface SupportOptions {
  /** α 门：`alpha > alphaCutoff` 才算几何。默认 `1/255`（**只剔全透明**）。 */
  readonly alphaCutoff?: number
  /** 深度外扩半径（像素，4-邻域）。默认 2；`0` = 不外扩。 */
  readonly dilatePx?: number
  /** 小岛清理：4-连通支撑分量小于该像素数则整块丢弃。默认 8；`<= 1` 关闭。 */
  readonly minIslandPixels?: number
}

/** 直通线性 RGB + α 的纹理缓冲。**引用**输入帧，不拷贝。 */
export interface RgbaTexture {
  readonly width: number
  readonly height: number
  /** 直通线性 RGB，长度 `width*height*3`。 */
  readonly rgb: Float32Array
  /** α，长度 `width*height`。 */
  readonly alpha: Float32Array
}

/** 单层的诊断计数（供脚本 / 调参；渲染器忽略）。 */
export interface LayerMeshReport {
  /** 真实可见（非全透明）的像素数。 */
  readonly seedPixels: number
  /** 深度外扩进来的像素数。 */
  readonly dilatedPixels: number
  /** 最终支撑像素数（小岛清理后）。 */
  readonly supportPixels: number
  readonly removedIslandPixels: number
  /** 四叉树叶子数（≈ 四边形数）。 */
  readonly leaves: number
  readonly levels: readonly number[]
  /** 输出三角面 vs 视差场的最大误差（抽样实测）。 */
  readonly maxTriangleError: number
  /** 已到最小格但误差仍超限的叶子数（>0 说明该层比 `minCellPx` 更精细才能表达）。 */
  readonly minCellViolations: number
  /** 含贴边外推顶点、无法验差的叶子数。 */
  readonly skippedLeaves: number
  /** 四角有贴边外推的叶子数。 */
  readonly snappedLeaves: number
}

/**
 * 一层（一层 RGBAD）的 relief 网格 —— 本模块的**输出单元**。
 *
 * 顶点按四叉树格子生成；`texture` 引用该层 `WSplatFrame` 的 `rgb/alpha`（不拷贝）。
 */
export interface LayerMesh {
  /** 层号：0 = 最近。 */
  readonly layerIndex: number
  /** 渲染画布尺寸（= 纹理尺寸）。 */
  readonly width: number
  readonly height: number
  readonly vertexCount: number
  readonly triangleCount: number
  /** xyz（米，世界坐标），长度 `vertexCount * 3`。**本模块拥有**。 */
  readonly positions: Float32Array
  /** uv（图像域归一化，左上原点），长度 `vertexCount * 2`。**本模块拥有**。 */
  readonly uvs: Float32Array
  /** 三角形索引，长度 `triangleCount * 3`。**本模块拥有**。 */
  readonly indices: Uint32Array
  /** 该层的纹理（引用输入帧）。 */
  readonly texture: RgbaTexture
  /** 该层真实深度带 `[lo, hi]`（米，来自 `placement.boundariesZ`）。 */
  readonly depthRange: readonly [number, number]
  readonly report: LayerMeshReport
}

/** 整场诊断（渲染器忽略）。 */
export interface MeshReport {
  readonly layers: readonly LayerMeshReport[]
  /** 总三角面数。 */
  readonly triangleCount: number
  readonly vertexCount: number
}

/**
 * mesh 场景 = 多层网格 + 相机 / 视图元数据（渲染器无关的内存对象）。
 *
 * ⚠ 除几何与纹理外，其余字段都**引用输入**（`layered`），不拷贝。
 */
export interface MeshScene {
  /** 逐层网格（索引与 `layerDepths` 对齐，0 = 最近）。 */
  readonly layers: readonly LayerMesh[]
  /** 每层代表深度（**米**），引用 `placement.layerDepthsZ`。 */
  readonly layerDepths: Float32Array
  /** `[L*2]` 每层视差域范围，引用 `layered.ranges`。 */
  readonly layerRanges: Float32Array
  readonly near: number
  readonly far: number
  /** 扩视角视图（`GLB` 里的相机与 `extras` 都用它）。 */
  readonly view: LayerView
  /** 渲染相机（= 扩视角相机；所有层共用）。 */
  readonly camera: WSplatCamera
  /** 层纹理是否预乘 α。本链路层帧是**直通**，恒为 `false`。 */
  readonly premultipliedAlpha: false
  readonly report: MeshReport
}

/** meshing 总选项。缺省即默认。 */
export interface MeshingOptions {
  /** 支撑 / 深度外扩（见 `support.ts`）。 */
  readonly alpha?: SupportOptions
  /**
   * LOD 出面选项。缺省 = 默认选项；`false` = **逐像素出面**（`minCellPx = maxCellPx = 1`，
   * 面数量级上升，仅作对照 / 调试基线）。
   */
  readonly lod?: LodOptions | false
}
