/**
 * layering 的**纯数据类型层**（web / node 共用）。
 *
 * 本文件不得 import WebGPU / node 类型（对照 `wsplat/types.ts` 的定位）；
 * GPU 相关的东西都在 `render.ts`。
 *
 * ── 在全链路里的位置 ──
 * ```
 * image (+EXIF focal) -> splat 参数场 -> [layered RGBAD] -> mesh（交付物）
 * ```
 * 本模块只做中间那一环：**splat 场 -> L 层 RGBAD**。splat 与 RGBAD 都是中间表示，
 * 交付物是 mesh；渲染侧只光栅化多层 mesh。
 *
 * ── 参考视角恒等式（本模块的验收口径）──
 * `over` 是**结合的**，所以「层 = 全局 back-to-front 排序的一段连续区间」时：
 * ```
 * render(全部) == over_k(render(层 k))        // 层序 远 -> 近
 * ```
 * **逐层渲染必须各自 clear**（`drawLayer` 就是这么做的），每层因此持有自己原本的
 * 颜色/α/ED，而不是被更近的层衰减过的值 —— 这正是 LDI 要的语义，也是 mesh 阶段
 * 能一层一个网格的前提。跨层合并深度必须在 `(ED, A)` 空间做：
 * `D_total = Σ T_k·ED_k / Σ T_k·A_k`（见 `composite.ts`），**不能**直接平均各层 D。
 *
 * ── 为什么所有深度都活在「视差域」──
 * `n(z) = far/(far−near) · (1 − near/z)`，即 WebGPU 的 NDC 深度（0 = near，1 = far）。
 * 它是 `1/z` 的**严格仿射**变换，所以在 `n` 上等距 ≡ 在视差上等距。选它的理由：
 * 视差才是视差 / 遮挡的驱动量，且层边界、LOD 误差、撕裂阈值全部在同一域里可比。
 */

import type { WSplatCamera } from "../wsplat/camera.ts"
import type {
  WSplatFrame,
  WSplatResolveOptions,
  WSplatStats,
} from "../wsplat/types.ts"

/** 视差域（= WebGPU NDC 深度）里的一个值。`0` = near，`1` = far。 */
export type NdcDepth = number

/** 层边界放置方法。 */
export type LayerPlacementMethod =
  /** 等质量（= 视差分位，权重默认取 opacities 的「墨量」）。**默认档**。 */
  | "quantile"
  /** 在视差域 `[0,1]` 上几何等距。基线 / 对照用，允许空层。 */
  | "uniform"

/** 一次放置的结果。层索引 `k` 一律 **0 = 最近**。 */
export interface LayerPlacement {
  readonly L: number
  readonly method: LayerPlacementMethod
  /** 投影近平面（米）。 */
  readonly near: number
  /** 投影远平面（米）。 */
  readonly far: number
  /** `[L]` 每层的代表深度（视差域），严格递增。 */
  readonly layerDepths: Float32Array
  /** `[L]` 同上，换成真实度量深度（米），仅供阅读 / 元数据。 */
  readonly layerDepthsZ: Float32Array
  /** `[L+1]` 层边界（视差域）；`[0] == 0`、`[L] == 1`，严格递增。 */
  readonly boundaries: Float32Array
  /** `[L+1]` 同上（米）。 */
  readonly boundariesZ: Float32Array
  /** `[L]` 每层的权重质量占比（**逐样本精确值**，`Σ == 1`）。 */
  readonly layerMass: Float32Array
}

/** 分带结果：每层在 **back-to-front 排序序列**里的连续下标区间（见 `bands.ts`）。 */
export interface LayerBands {
  readonly L: number
  /** 参与分带的高斯总数。 */
  readonly total: number
  /** `[L]` 每层在排序序列里的起始下标。最远层 `first[L-1] == 0`。 */
  readonly first: Int32Array
  /** `[L]` 每层的高斯个数（本实现是硬划分，各层之和 `== total`）。 */
  readonly count: Int32Array
}

/**
 * 层表的一项：层 `k` 占据**排列**里的 `[base, base + count)`。
 *
 * 本实现里层是硬划分（同一个高斯只进一层），所以隐式 `base = Σcount` 也成立；
 * 保留显式的表，是为了让下游（`drawLayer` / mesh）不用假设这件事。
 */
export interface LayerTableEntry {
  readonly base: number
  readonly count: number
}

/**
 * 绘制排列 + 层表。层索引 `k` 一律 **0 = 最近**。
 *
 * 合法性（必须当门来验，不能靠“构造上总对”）：
 * - 层间：`k = L-1`（最远）到 `0`（最近）依次相接 —— 因为 `over` 必须先画远的；
 * - 层内：视差 `n` 递减（back-to-front）；
 * - `permutation` 是全量高斯的**严格排列**（长度 `== 高斯数`、不重不漏）——
 *   `over` 不幂等，同一个高斯进两层会把它的 α 算两次，参考视角无损直接没了。
 *
 * 由全局序切段构造时这三条是**构造保证**的（见 `buildLayerPermutation`）。
 */
export interface LayerPermutation {
  /** 长度 `length`；每项是一个高斯下标（`< 高斯总数`）。 */
  readonly permutation: Uint32Array
  /** 排列长度（本实现恒等于高斯总数）。 */
  readonly length: number
  /** `[L]` 层表（0 = 最近）。 */
  readonly table: readonly LayerTableEntry[]
}

/**
 * 分层渲染用的**视图**（参考视角 -> 扩视角画布）。
 *
 * ── 为什么需要它 ──
 * 单图模型在参考视角边缘外仍有高斯（椭球探出画面、纹理/几何继续外推），
 * 只渲原视角会把它们丢掉，mesh 一到边缘就“切边”。所以渲染画布按 `viewScale`
 * 外扩：**像素焦距不变、画布变大**，于是参考图内容在画布中心保持**逐像素同尺度**，
 * 四周多出来的是原视角外的新内容。
 *
 * ⚠ 这是「扩画布」而不是「缩焦距」：缩焦距会把参考视角的分辨率也一起降掉，
 * 而扩画布保证 `referenceRect` 内的像素与参考视角一一对应（`focalLengthPx` 相同）。
 *
 * meshing 的反投影坐标计算需要全部这些字段：
 * ```
 * 渲染像素 (x, y) -> 相机坐标 ((x+0.5−W/2)/fx·z, (y+0.5−H/2)/fy·z, z)
 * 参考图像素 (u, v) -> 渲染像素 (u + referenceRect.x, v + referenceRect.y)
 * ```
 */
export interface LayerView {
  /** 渲染画布宽（像素）。 */
  readonly width: number
  /** 渲染画布高（像素）。 */
  readonly height: number
  /** 渲染域的像素焦距（与参考域同尺度，见 `pixelScale`）。 */
  readonly focalLengthPx: number
  /** 有效的视角扩倍率（`>= 1`；角度覆盖近似乘它）。 */
  readonly viewScale: number
  /** 有效像素倍率：渲染尺寸 / 参考图像尺寸（含 `maxRenderSide` 的削减）。 */
  readonly pixelScale: number
  /** 参考图像在渲染画布里的矩形（中心内嵌，同尺度）。 */
  readonly referenceRect: LayerViewRect
  /** 参考相机的像素焦距（原图域），便于下游换算。 */
  readonly referenceFocalLengthPx: number
}

/** `LayerView.referenceRect` 的矩形（可为小数，表示连续坐标）。 */
export interface LayerViewRect {
  readonly x: number
  readonly y: number
  readonly width: number
  readonly height: number
}

/** 每层的剔除统计（`WSplatStats` + 层号），用于发现「被静默剔空的层」。 */
export interface LayerSplatStats extends WSplatStats {
  readonly layerIndex: number
}

/** 分层渲染结果：L 层 RGBAD + 元数据，是 **meshing 阶段的输入**。 */
export interface LayeredRgbd {
  readonly L: number
  /** 渲染画布尺寸（= `view.width/height`）。 */
  readonly width: number
  readonly height: number
  /** 投影近 / 远平面（米）。 */
  readonly near: number
  readonly far: number
  /** 扩视角视图（meshing 的坐标换算全在这里）。 */
  readonly view: LayerView
  /** 渲染用的相机（= 扩视角后的参考相机，居中主点）。 */
  readonly camera: WSplatCamera
  readonly placement: LayerPlacement
  /** `[L*2]` 每层视差域深度范围 `[lo, hi]`（含 `rangeOverlap` 余量）。 */
  readonly ranges: Float32Array
  /** `[L]` 层像素，索引与 `placement.layerDepths` 对齐（0 = 最近）。 */
  readonly frames: readonly WSplatFrame[]
  /** `[L]` 逐层剔除统计（`drawLayer` 之后 `countCulls()` 的结果）。 */
  readonly stats: readonly LayerSplatStats[]
  /**
   * 整场直接渲染（**同一个相机、同一批高斯**），仅当 `includeDirect` 时非空。
   *
   * 它是「分层混合 ≡ 整场渲染」这条不变量的对照物：
   * `compositeLayerFrames(frames)` 与 `singleFrameToComposited(direct)` 必须一致。
   */
  readonly direct: WSplatFrame | null
}

/** `renderLayerStack` 的旋钮。 */
export interface LayerRenderOptions {
  /** 参考视角相机（原图视角，`extrinsics = I` 的 SHARP 产物直接用它）。 */
  readonly camera: WSplatCamera
  /** 层数。默认 10；推荐 4~16，硬范围 1~64。 */
  readonly layers?: number
  /** 层边界放置方法。默认 `"quantile"`（按 opacities 加权的等质量分位）。 */
  readonly method?: LayerPlacementMethod
  /** 直方图箱数。默认 256（会被抬到 `>= 2L`）。 */
  readonly binCount?: number
  /** 视角扩倍率（画布倍率，`>= 1`）。默认 1.2。 */
  readonly viewScale?: number
  /** 渲染像素倍率（相对参考图像素）。默认 1。 */
  readonly renderScale?: number
  /**
   * 渲染画布的**短边**目标（像素）或 `"auto"`。默认 `"auto"`。
   *
   * `"auto"` = `min(1536（SHARP 内部分辨率）, 参考图短边)` —— 即「参考内容在
   * 渲染画布里的短边像素数」不超过模型能分辨的分辨率，也不超过原图本身。
   * 短边定尺 → 像素焦距 `fx = 原图 fx · (目标短边 / 原图短边)`；画布 = 参考内容 × `viewScale`。
   */
  readonly shortSide?: ShortSideOption
  /**
   * 渲染最长边**硬上限**（像素）。默认 `0` = 关闭。
   * 短边定尺后长边由宽高比决定，极端全景图才需要它兜底。
   */
  readonly maxRenderSide?: number
  /** `[L*2]` 层范围的视差余量（只写进 `ranges` 元数据，不影响划分）。默认 0。 */
  readonly rangeOverlap?: number
  /** 传给 wsplat resolve 的旋钮（可见 α 阈值等）。 */
  readonly resolve?: WSplatResolveOptions
  /** 以下 4 项直接透传 `createWSplatRenderer`（默认与 wsplat 一致）。 */
  readonly minPixelSize?: number
  readonly alphaClip?: number
  readonly eps2d?: number
  readonly antialias?: boolean
  /** `setGaussians` 的颜色空间。默认 `"linearRGB"`（SHARP 契约）。 */
  readonly colorSpace?: "linearRGB" | "sRGB"
  /** 是否额外渲一帧整场直接渲染（对照物）。默认 `false`。 */
  readonly includeDirect?: boolean
}

/** 分层渲染的默认层数（4~16 是推荐区间）。 */
export const DEFAULT_LAYERS = 10
/** 推荐层数下界。 */
export const MIN_RECOMMENDED_LAYERS = 4
/** 推荐层数上界。 */
export const MAX_RECOMMENDED_LAYERS = 16
/** 层数硬上界（超过它直方图/层表的收益为负）。 */
export const MAX_LAYERS = 64
/** 默认视角扩倍率。 */
export const DEFAULT_VIEW_SCALE = 1.2

/** 渲染短边目标：像素数或 `"auto"`。 */
export type ShortSideOption = number | "auto"

/** 默认短边目标：`auto` = `min(1536（SHARP 内部分辨率）, 参考图短边)`。 */
export const DEFAULT_SHORT_SIDE: ShortSideOption = "auto"

/** 默认渲染最长边硬上限（像素）；`0` = 关闭。 */
export const DEFAULT_MAX_RENDER_SIDE = 0
