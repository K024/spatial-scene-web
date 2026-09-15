/**
 * 渲染层共享类型与常量（**纯数据**，可在 worker 与主线程两侧复用）。
 */

/** 打包后的每高斯浮点数个数。布局见 {@link SPLAT_LAYOUT}。 */
export const SPLAT_STRIDE = 14

/**
 * 交错顶点缓冲（interleaved）内的字段偏移（单位：float）。
 *
 * 顺序按「着色器里读取顺序」排列；紧凑排布为 14 float = 56 字节/高斯。
 * （vec4 属性并不要求 16 字节对齐——那是性能建议而非 WebGL2 的约束——
 * 这里为了省显存不做填充。）
 *
 * | offset | 字段 | 类型 | 语义 |
 * |---|---|---|---|
 * | 0  | center      | vec3  | 度量空间位置（PLY 系：x 右 / y 下 / z 前） |
 * | 3  | scaleLog    | vec3  | `log(奇异值)`，与 PLY `scale_*` 一致 |
 * | 6  | quat        | vec4  | 旋转，**w-first**，已归一化 |
 * | 10 | colorLinear | vec3  | **线性 RGB**（PLY 里是 sRGB 的 SH-DC，已在 worker 里转换） |
 * | 13 | opacityLogit| float | `logit(opacity)`，与 PLY `opacity` 一致 |
 */
export const SPLAT_LAYOUT = {
  center: 0,
  scaleLog: 3,
  quat: 6,
  colorLinear: 10,
  opacityLogit: 13,
} as const

/** PLY 顶点 element 解析出的「结构数组」（未被排序、未被变换的原始量）。 */
export interface SplatSoA {
  count: number
  /** `[N,3]` 度量空间位置。 */
  center: Float32Array
  /** `[N,3]` log 奇异值。 */
  scaleLog: Float32Array
  /** `[N,4]` 归一化四元数，w-first。 */
  quat: Float32Array
  /** `[N,3]` **线性** RGB（已从 SH-DC + sRGB 转换）。 */
  colorLinear: Float32Array
  /** `[N]` logit(opacity)。 */
  opacityLogit: Float32Array
}

/** PLY 里附带的小 element（compact 模式没有；full 模式有）。 */
export interface PlyAuxData {
  /** `intrinsic`（9 个 float，行主序 3x3，原图域）。 */
  intrinsic?: number[]
  /** `extrinsic`（16 个 float，行主序 4x4）。 */
  extrinsic?: number[]
  /** `image_size`（2 个 uint，`[width, height]`）。 */
  imageSize?: [number, number]
  /** `disparity`（2 个 float）。 */
  disparity?: number[]
  /** `color_space`（1 个 uchar：0=sRGB，1=linearRGB）。 */
  colorSpace?: number
  /** `version`（3 个 uchar）。 */
  version?: number[]
}

/** 场景元信息：worker 解析完 PLY 后回传，用于相机取景与信息面板。 */
export interface SceneMeta {
  /** 高斯数量。 */
  count: number
  /** PLY 字节数。 */
  bytes: number
  /** 头信息里的 element 列表（调试/信息展示用）。 */
  elements: { name: string; count: number }[]
  /** 位置包围盒，`[min, max]`，PLY 系。 */
  bounds: { min: [number, number, number]; max: [number, number, number] }
  /** PLY 里的附加 element。 */
  aux: PlyAuxData
}

/** 打包结果：可直接 `bufferData` 上传的交错缓冲 + 排序统计。 */
export interface PackedSplats {
  count: number
  /** `[N * SPLAT_STRIDE]`，布局见 {@link SPLAT_LAYOUT}。 */
  data: Float32Array
  /** 排序耗时（ms）。 */
  sortMs: number
  /** 打包耗时（ms）。 */
  packMs: number
}

/** 排序用的相机（PLY 系）。 */
export interface SortCamera {
  /** 相机位置。 */
  eye: [number, number, number]
  /** 视线方向（单位向量）。 */
  forward: [number, number, number]
}

/**
 * **参考相机**：SHARP 单图推理的原点视角（extrinsics = I）。
 *
 * 度量空间本身就是这台相机的坐标系，所以它在 PLY 系里恒为
 * `eye = (0,0,0)`、`forward = (0,0,1)`。首次排序用的就是它。
 */
export const REFERENCE_CAMERA: SortCamera = {
  eye: [0, 0, 0],
  forward: [0, 0, 1],
}

/** worker -> 主线程的进度阶段。 */
export type LoadStage = "fetch" | "parse" | "convert" | "sort" | "pack" | "done"

/** 各阶段的中文名（面板显示用）。 */
export const STAGE_LABELS: Record<LoadStage, string> = {
  fetch: "下载",
  parse: "解析",
  convert: "颜色转换",
  sort: "按参考相机排序",
  pack: "打包顶点缓冲",
  done: "完成",
}

/** worker -> 主线程的进度事件。 */
export interface LoadProgress {
  stage: LoadStage
  /** 0..1，用于进度条。 */
  ratio: number
  /** 人类可读的补充信息。 */
  detail?: string
}

/** `load()` 的返回：解析元信息 + 已按参考相机排好序的顶点缓冲。 */
export interface LoadResult {
  meta: SceneMeta
  packed: PackedSplats
}

/**
 * worker 暴露给主线程的 RPC 接口（`Comlink.wrap<SplatWorkerApi>`）。
 *
 * 用 Comlink 而不是手写 postMessage 协议：进度回调要跨线程序列化、
 * 顶点缓冲要 transferable 零拷贝，这两件事手写都要自己维护一套
 * message tag + transfer list，RPC 形态把两边都变成普通异步函数调用。
 */
export interface SplatWorkerApi {
  /** 下载 -> 解析 -> 转换 -> 排序 -> 打包；worker 内部保留中间结果供重排序。 */
  load(url: string, onProgress: (p: LoadProgress) => void): Promise<LoadResult>
  /** 用新相机重排序（不重新下载/解析）。 */
  resort(camera: SortCamera): Promise<PackedSplats>
  dispose(): void
}
