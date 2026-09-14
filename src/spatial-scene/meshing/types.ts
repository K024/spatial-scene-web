/**
 * meshing 的**纯数据类型层**（模块的输入 / 输出契约）。
 *
 * 本文件**不得** import 任何 WebGPU / DOM / node / three 类型
 * （对照 `layering/types.ts`、`wsplat/types.ts` 的定位）：它是 web / node 共用的数据契约层。
 *
 * ══════════════════════════════ 模块边界 ══════════════════════════════
 *
 * ```
 * image (+EXIF focal) -> splat 参数场 -> layered RGBAD -> [MeshScene] -> WebGL2 / three.js
 *                                                        ^^^^^^^^^^^ 本模块的产物
 * ```
 * - 本模块只做 **LayeredRGBD -> MeshScene**，产物是**与渲染器无关的内存数据**。
 * - **导出格式 / 存储 / 元信息序列化不在本模块**（图集打包、ASTC、USDZ、变体、落盘
 *   都归下游 packing / export 模块）。这里**不落盘**，层数据进程内传递。
 * - 交付渲染器是 **WebGL2 / three.js**；`src/**` 内不得出现 WebGPU 依赖。
 *   node 侧 WebGPU / CPU 光栅器只作 `scripts/` 下的**测试基准**，不是交付路径。
 *
 * ══════════════════════════════ 唯一硬要求 ══════════════════════════════
 *
 * **深度断层必须被正确撕裂**（见 `tears.ts`）。其余（外扩 / 小岛 / 洞 / 透明）可简化：
 * - 每层是渲出来的 2D 帧（每像素恰好一个深度）⇒ 几何恒为 heightfield ⇒ 层内**遮挡型断层**
 *   靠**局部**撕裂解决，不存在需要全局图搜索的多深度节点集；
 * - 同像素多深度的**穿插问题**（fence / foliage / 发丝）在渲染期已被 α 加权平均坍缩，
 *   meshing 无法恢复 ⇒ 属于 **layering**，不在这里解。
 *
 * ══════════════════════════════ 透明部分 ══════════════════════════════
 *
 * 有序层栈 + 逐层 mesh 的模型下，透明由光栅器 **back-to-front α 混合**解决，
 * mesh 只负责「几何支撑掩码 + 逐 texel 保留 α」。支撑掩码用**迟滞**而非单阈值
 * （见 `tolerance.ts`）：软边随强核心保留、孤立弱块丢弃。
 *
 * ══════════════════════════════ 内存复用约定 ══════════════════════════════
 *
 * 输出结构只**拥有**真正新生成的数据（`positions / uvs / indices` 与背衬纹理），
 * 其余一律**引用输入**，不拷贝：
 * - `LayerMesh.texture.rgb/.alpha` ⇒ 直接引用该层 `WSplatFrame` 的 `rgb` / `alpha`；
 * - `MeshScene.layerDepths` ⇒ 引用 `LayerPlacement.layerDepthsZ`；
 * - `MeshScene.layerRanges` ⇒ 引用输入的 `ranges`；
 * - `MeshScene.stats` ⇒ 引用输入的 `DisparityStats`。
 *
 * ══════════════════════════════ 坐标 / UV / 绕序约定（不可含糊）══════════════════════════════
 *
 * 1. `positions` 是**度量空间**（米），与 splat 同系：相机在原点朝 `+z`（OpenCV），
 *    `y` 向下，深度 = 视图空间 `z`。
 * 2. `uvs` 是**图像域归一化**坐标，原点在**左上**，`u` 向右、`v` 向下：
 *    `u = (x + 0.5) / width`、`v = (y + 0.5) / height`（取像素中心）。
 *    渲染器若纹理原点在左下（WebGL 默认），需自行翻转 v。
 * 3. 三角形绕序使**几何法线朝向相机**（相机空间法线 `z < 0`），即 OpenCV 相机下的正面。
 *    裙边断壁的绕序相反（法线朝外）；`indices` 里**表面三角形在前、墙三角形在后**
 *    （`triangleCount - wallTriangleCount` 是分界），渲染端若只做单面剔除需注意。
 * 4. 图集 slice 不在本模块：每层自带一张 `texture`，消费者按层渲染即可；
 *    packing 模块若要合图集，自会重写 `uvs` 并建立 triangle->slice 映射。
 *
 * ══════════════════════════════ 参考实现（只借鉴几何/拓扑）══════════════════════════════
 *
 * SOLIDI: *3D Photography using Context-aware Layered Depth Inpainting* (CVPR 2020),
 * `vt-vl-lab/3d-photo-inpainting` @ `de04467`（MIT）：
 * - `tears.ts`        ← `mesh.py:tear_edges`、`remove_redundant_edge`/`judge_dangle`（despeckle）
 * - `tolerance.ts`    ← `mesh.py:generate_init_node`（小岛剔除）
 * - `relief.ts`       ← `mesh.py:create_mesh`（像素网格 + `extrapolation_thickness`）、
 *                        `mesh.py:generate_face`（四邻域出 2 三角、对角方向）
 * - `tolerance.ts`  支撑 / `tears.ts` 撕裂 / `relief.ts` 网格 / `backfill.ts` 回填
 * - `backfill.ts`     ← 无对应（它用修补网络）；改走 Apple `MXIBackLayer` + backing plane 的**
 *                        算法级**路线（`_concatRGBD` / `_downscaleAlphtaWeighted` / `_backLayerBlend`）
 * **不移植** SOLIDI 的 LDI 图机制（`reassign_floating_island`/`group_edges`/
 * `combine_end_node`/`remove_dangling`/…）与 `networks.py`：那些解决的是「每像素多深度 +
 * 需发现边 + 给网络闭合 mask」，在本链路**前提不成立**。
 */

import type { DisparityStats, LayeredRGBD } from "../layering/types.ts"
import type { WSplatCamera } from "../wsplat/camera.ts"
import type { LodOptions } from "./lod.ts"

/**
 * meshing 的输入 = layering 的产出 + 渲染用的相机。
 *
 * `LayeredRGBD` 已带 `L / width / height / near / far / placement / frames / ranges / stats`；
 * 这里只补一个 `camera` —— 反投影顶点与算 FoV / aspect 都要它，而它不在这份契约里。
 */
export interface MeshingInput extends LayeredRGBD {
  readonly camera: WSplatCamera
}

/**
 * α 容差（迟滞支撑掩码）选项 —— 对应上游 `Min/MaxOpacityTolerance`。
 *
 * 为什么是**迟滞**而不是单阈值：层带交界的软边（`0 < α < strong`）若被一刀切掉，
 * 相邻层之间会出现能量缺口 → 接缝变暗 / 露底；而背景里的孤立低 α 噪点若被保留，
 * 会变成幽灵面。迟滞同时解决两者：**弱像素只在与强像素 4-连通时才保留**。
 *
 * @see SOLIDI `mesh.py:generate_init_node`（`min_node_in_cc` 小岛剔除）——
 *   vt-vl-lab/3d-photo-inpainting @ de04467 (MIT)
 */
export interface AlphaToleranceOptions {
  /** 强阈值 `aHi`：`α >= strongAlpha` 视为确定表面。默认 `0.5`。 */
  strongAlpha?: number
  /** 弱阈值 `aLo`：`α >= weakAlpha` 才算候选。默认 `0.05`。`< strongAlpha`。 */
  weakAlpha?: number
  /**
   * 小岛剔除：4-连通支撑分量小于该像素数则整块丢弃。默认 `32`。
   * `<= 1` 关闭（`0`/`1` 时任何非空分量都保留）。
   */
  minIslandPixels?: number
}

/**
 * 撕裂选项 —— 本模块的**硬要求**。
 *
 * 阈值活在**视差域**：`tearEps = clamp(tearScale · 该层带宽度, min, max)`。
 * 用米制深度是错的：`n` 对 `1/z` 仿射，远景会被压缩，米制阈值会漏撕裂。
 *
 * @see SOLIDI `mesh.py:tear_edges`（`depth_threshold` 断边）/
 *   `mesh.py:remove_redundant_edge`（短段 despeckle）——
 *   vt-vl-lab/3d-photo-inpainting @ de04467 (MIT)
 */
export interface TearOptions {
  /** 撕裂阈值相对**该层视差带宽度**的比例。默认 `0.4`。 */
  tearScale?: number
  /** 阈值下限（视差域），防极窄带把噪声当断层。默认 `0.002`。 */
  minTearDisparity?: number
  /** 阈值上限（视差域），防极宽带（远景）阈值大到永不撕裂。默认 `0.06`。 */
  maxTearDisparity?: number
  /**
   * 最小撕裂段长（**边数**）：孤立/极短的撕裂连通分量视为噪声，取消撕裂（重新连上）。
   * 这是 `remove_redundant_edge` 提炼出的 despeckle。默认 `6`；`<= 1` 关闭。
   */
  minTearSegmentLength?: number
  /** 阈值再乘的系数（`>1` = 更少撕裂）。一般由 `layerBias` 推导，不直接手填。 */
  thresholdScale?: number
  /**
   * 撕裂判据前先对**视差场**做 3×3 中值去噪（只在支撑内取样）。默认 `true`。
   *
   * 撕裂判据用的就是 α 混合出来的 `depth`，数值上不可靠；孤立毛刺会产生假撕裂。
   * **顶点位置不受影响**（只用去噪场算阈值比较）。
   */
  denoise?: boolean
  /**
   * 「小面片」门：torn 边从支撑图移除后，分量像素数小于此值的，其边界 torn 边
   * 取消（重新连上）。治“中间小三角被撕掉”。默认 `16`；`<= 1` 关闭。
   */
  minPatchPixels?: number
  /**
   * **按层地位**减少撕裂：层号 `k >= (1 − bottomFraction)·L` 的底部层（背景）
   * 撕裂阈值乘 `bottomScale`。缺省 `0.25` / `2`（可通过 `bottomFraction: 0` 关闭）。
   *
   * 背景层视差形变小、橡皮布不明显，但撕裂留缝会直接露背景 ⇒ 宁可少撕。
   */
  layerBias?: { bottomFraction?: number; bottomScale?: number }
}

/** relief 网格化选项。 */
export interface ReliefOptions {
  /**
   * 对角翻转（`AllowDiagonalFlip`）：在两种四边形剖分里选**对角线更短**的那种，
   * 减少斜视角下的长条三角形。默认 `true`。
   *
   * @see SOLIDI `mesh.py:generate_face` —— vt-vl-lab/3d-photo-inpainting @ de04467 (MIT)
   */
  allowDiagonalFlip?: boolean
  /**
   * 不透明三角形的 α 门（用于 `opaqueTriangleCount`，供 `SeparateOpaqueGeometry` 用）。
   * 三个角 α 都 `>=` 它才算不透明。默认 `0.99`。
   */
  opaqueAlpha?: number
}

/**
 * 裙边 / 断壁（skirt）选项。
 *
 * ── 为什么需要它 ──
 * 撕裂、支撑外缘、图像外框都会在网格上留下**开放边**。斜视角下近面与后层之间会
 * 露出细缝（经典 crack）。沿视线方向把边界顶点向**后**挤出一圈几何（断壁）来遮住它。
 *
 * 宽度活在**视差域**（与撕裂同一理由）：`skirtN = clamp(scale·bandWidth, min, max)`。
 *
 * SOLIDI 对应做法是向外 `extrapolate`（用 inpainting 发明内容）；我们改为向后挤出，
 * 靠真实后层补，不发明颜色。
 *
 * @see SOLIDI `mesh.py:create_mesh`（`extrapolation_thickness`）/
 *   `mesh_tools.py:extrapolate` —— vt-vl-lab/3d-photo-inpainting @ de04467 (MIT)
 */
export interface SkirtOptions {
  /** 裙边视差宽度相对**该层视差带宽度**的比例。默认 `0.5`。 */
  scale?: number
  /** 下限（视差域）。默认 `0.002`。 */
  minDisparity?: number
  /** 上限（视差域）。默认 `0.04`。 */
  maxDisparity?: number
}

/**
 * 回填 / 背衬平面的选项。
 *
 * 算法级：把 L 层 α 加权降采样合成一张粗纹理，贴在最远深度的背衬平面上，
 * 消除多层网格边缘遮不住的透视穿帮（宽空档）。
 *
 * @see Apple `MXIBackLayer`（`_concatRGBD` → `_downscaleAlphtaWeighted` → `_backLayerBlend`）
 *   + `generateBackingPlaneMesh:atDepth:`。**纯算法，不跑网络**。
 */
export interface BackfillOptions {
  /** 是否生成背衬平面。默认 `true`。 */
  enabled?: boolean
  /** α 加权降采样倍率（整数）。默认 `4`。 */
  downscale?: number
  /** 背衬平面深度（米）；缺省用最远层边界 `placement.boundariesZ[L]`。 */
  depth?: number
}

/** `buildLayerRelief` 的**运行时数据**（不是旋钮）：视差、近远平面、裙边宽度。 */
export interface LayerReliefContext {
  /** 逐像素视差；裙边宽度换算用。 */
  readonly disparity?: ArrayLike<number>
  /** 投影近平面（米），裙边换回真实深度用。 */
  readonly near?: number
  /** 投影远平面（米）。 */
  readonly far?: number
  /** 裙边视差宽度；`<= 0` / 缺省 = 不生成裙边。 */
  readonly skirtDisparity?: number
}

/** meshing 总选项。缺省即默认；所有默认值都写在各自 `*.ts` 的实现里，便于单测。 */
export interface MeshingOptions {
  readonly alpha?: AlphaToleranceOptions
  readonly tears?: TearOptions
  readonly relief?: ReliefOptions
  /** 裙边 / 断壁。`false` 关闭；缺省 = 开启（用 `SkirtOptions` 默认）。 */
  readonly skirt?: SkirtOptions | false
  /** 背衬平面 / 回填。 */
  readonly backing?: BackfillOptions
  /**
   * LOD 出面（受限四叉树自适应）。缺省 / `false` = 逐像素出面（现有行为）。
   * 面数从百万级压到 10–100k 量级；语义与约束见 `lod.ts`。
   */
  readonly lod?: LodOptions | false
}

/**
 * 直通（非预乘）线性 RGB + α 的纹理缓冲。
 *
 * ⚠ 层纹理**引用**输入帧的 `rgb` / `alpha`（不拷贝）；背衬纹理是本模块新生成的（已降采样）。
 * `premultipliedAlpha = false` 时渲染端自行预乘，或加载时预处理。
 */
export interface RgbaTexture {
  readonly width: number
  readonly height: number
  /** 直通线性 RGB，长度 `width*height*3`。 */
  readonly rgb: Float32Array
  /** α，长度 `width*height`。 */
  readonly alpha: Float32Array
}

/**
 * 一层（或背衬平面）的 relief 网格 —— 本模块的**输出单元**。
 *
 * 顶点按**掩码像素**生成（一个支撑像素一个顶点），所以四边形天然共享顶点；
 * 撕裂只是**不发射**那个四边形，不改顶点集合。
 *
 * `texture` 是该 mesh 的纹理：普通层引用其 `WSplatFrame`；背衬平面用降采样合成纹理。
 * `layerIndex === -1` 表示这是背衬平面，不是层。
 */
export interface LayerMesh {
  /** 层号（0 = 最近）；`-1` = 背衬平面。 */
  readonly layerIndex: number
  readonly width: number
  readonly height: number
  readonly vertexCount: number
  /** 三角形总数（= 表面三角形 + 裙边断壁三角形）。 */
  readonly triangleCount: number
  /** 裙边断壁三角形数（总排在 `indices` **末尾**，前面是表面三角形）。 */
  readonly wallTriangleCount: number
  /** 三个角 α 都过 `opaqueAlpha` 的三角形数（`SeparateOpaqueGeometry` 用）。 */
  readonly opaqueTriangleCount: number
  /** xyz（米），长度 `vertexCount * 3`。**本模块拥有**。 */
  readonly positions: Float32Array
  /** uv（图像域归一化，左上原点），长度 `vertexCount * 2`。**本模块拥有**。 */
  readonly uvs: Float32Array
  /** 三角形索引，长度 `triangleCount * 3`。**本模块拥有**。 */
  readonly indices: Uint32Array
  /** 该网格的纹理（层：引用输入帧；背衬：降采样合成）。 */
  readonly texture: RgbaTexture
  /** 该层视差带 `[lo, hi]`（普通层来自 `placement.boundaries`；背衬 `[0,1]`）。 */
  readonly disparityRange: readonly [number, number]
  /** 该层真实深度带 `[lo, hi]`（米）；背衬为 `[depth, depth]`。 */
  readonly depthRange: readonly [number, number]
}

/** 单层的诊断计数（供 golden / 调参；渲染器忽略）。 */
export interface LayerMeshReport {
  readonly layerIndex: number
  /** 迟滞支撑掩码的像素数。 */
  readonly supportPixels: number
  /** 被小岛剔除的像素数。 */
  readonly removedIslandPixels: number
  /** 实际生效的撕裂阈值（视差域）。 */
  readonly tearEps: number
  /** 撕裂边数（despeckle 之后）。 */
  readonly tornEdges: number
  /** despeckle 取消掉的边数。 */
  readonly despeckledEdges: number
  /** 发射的四边形数。 */
  readonly quadsEmitted: number
  /** 因缺角或撕裂被跳过的四边形数。 */
  readonly quadsSkipped: number
  /** 网格边界边数。 */
  readonly boundaryEdges: number
  /** 裙边断壁三角形数。 */
  readonly wallTriangles: number
  /** 实际生效的裙边视差宽度。 */
  readonly skirtDisparity: number
  /**
   * LOD 出面：**输出三角面** vs 视差场的最大误差（见 `lod.ts` 的 `LodStats`）。
   * 逐像素出面（`relief`）路径不填。
   */
  readonly maxTriangleError?: number
  /** LOD 出面：已到 `minCell` 但输出面误差仍超 `maxError` 的叶子数（必须报出来）。 */
  readonly minCellViolations?: number
  /** LOD 出面：含贴边外推（snap）顶点、无法用预测面验差的叶子数。 */
  readonly errorSkippedLeaves?: number
}

/** 整场诊断。 */
export interface MeshReport {
  readonly layers: readonly LayerMeshReport[]
}

/**
 * mesh 场景 = 多层网格 + 背衬平面 + 相机 / 统计元数据（语义对齐 `MXIScene`，但仍是内存对象）。
 *
 * ⚠ 不落盘、不打包图集（见文件头「模块边界」）。除 `layers` / `backingPlane` 的几何与
 * 背衬纹理外，其余字段都**引用输入**。
 */
export interface MeshScene {
  /** 逐层网格（索引与 `layerDepths` 对齐，0 = 最近）。 */
  readonly layers: readonly LayerMesh[]
  /** 背衬平面（回填）；`null` = 未生成。`layerIndex === -1`。 */
  readonly backingPlane: LayerMesh | null
  /** 每层代表深度（**米**，引用 `placement.layerDepthsZ`）。 */
  readonly layerDepths: Float32Array
  /** `[L*2]` 每层视差域范围（引用输入的 `ranges`，含 `LayerRangesOverlap`）。 */
  readonly layerRanges: Float32Array
  readonly near: number
  readonly far: number
  /** 垂直视场角（弧度），来自渲染相机。 */
  readonly verticalFOV: number
  readonly aspectRatio: number
  /** 直接喂 `MXIScene.attributes`（引用输入）。 */
  readonly stats: DisparityStats
  /** 纹理是否预乘 α：层帧是**直通**，故恒为 `false`。 */
  readonly premultipliedAlpha: boolean
  /** 诊断（渲染器忽略）。 */
  readonly report: MeshReport
}
