/**
 * SHARP 输出契约与核心数据类型。
 *
 * 对照 `py-models/clones/ml-sharp/src/sharp/utils/gaussians.py`：
 *   - `Gaussians3D`  同名同字段（本文件用 camelCase）
 *   - `SceneMetaData` 同名同字段
 *
 * 本模块**不得** import 任何 onnxruntime 包：它是 web/node 共用的纯数据层。
 */

/** 颜色空间标签，对照 `color_space.py` 的 `ColorSpace` Literal。 */
export type ColorSpace = "sRGB" | "linearRGB"

/**
 * 一组 3D 高斯。
 *
 * 张量布局约定与 ml-sharp 一致，均为 **扁平化后的 [N, C]**（而非 [1, N, C]）：
 *   - `meanVectors`    [N, 3]
 *   - `singularValues` [N, 3]   非负
 *   - `quaternions`    [N, 4]   **w-first**：(w, x, y, z)
 *   - `colors`         [N, 3]   linearRGB，值域 ~[0, 1]
 *   - `opacities`      [N]      值域 [0, 1]
 *
 * 对照 `gaussians.py: Gaussians3D`。注意 ml-sharp 在内部保留 batch 维 `[1, N, C]`，
 * 但导出/PLY/渲染路径一律 `flatten(0, 1)`，因此这里直接以扁平 N 为准。
 */
export interface Gaussians3D {
  meanVectors: Float32Array
  singularValues: Float32Array
  quaternions: Float32Array
  colors: Float32Array
  opacities: Float32Array
}

/**
 * 场景元数据。
 *
 * 对照 `gaussians.py: SceneMetaData`。
 *
 * `focalLengthPx` 是**原始图像域**的焦距（不是 1536 内部域的）——见
 * `predict.py: predict_image`，反投影用的是 `intrinsics_resized`（乘过 1536/width），
 * 但 PLY 里写的 `intrinsic` element 用的是原始 `f_px`。
 */
export interface SceneMetaData {
  focalLengthPx: number
  /** (width, height)，原始图像域。 */
  resolutionPx: [number, number]
  colorSpace: ColorSpace
}

/**
 * ONNX 图的 6 个输出名与顺序。
 *
 * 对照 `py-models/scripts/common.py: OUT_NAMES`。
 * 顺序即 `torch.onnx.export(output_names=...)` 的顺序，也是 `sess.run()` 返回值
 * 逐项对应的顺序（node/web 两套 ort 都保证按 outputNames 顺序）。
 */
export const ONNX_OUTPUT_NAMES = [
  "mean_vectors",
  "singular_values",
  "quaternions",
  "colors",
  "opacities",
  "disparity",
] as const

export type OnnxOutputName = (typeof ONNX_OUTPUT_NAMES)[number]

/** ONNX 图的 2 个输入名。对照 `common.py: input_names=["image", "disparity_factor"]`。 */
export const ONNX_INPUT_NAMES = ["image", "disparity_factor"] as const
export type OnnxInputName = (typeof ONNX_INPUT_NAMES)[number]

/**
 * ONNX 图的**原始**输出张量（尚未做 NDC→metric 反投影）。
 *
 * 这是 `shared` 层与平台层之间的传输契约：平台层负责把 ort 的 Tensor
 * 摊平成这里的 `Float32Array`，其余逻辑（反投影、颜色空间、PLY）全部平台无关。
 *
 * 关键：`gaussians` 部分处于 **NDC 空间**。见 `common.py: Wrapper.forward`
 * 直接返回 `self.model(...)` 的输出，而 `unproject_gaussians` 是在
 * `predict.py: predict_image` 里之后才做的。JS 侧必须自己补这一步。
 */
export interface SharpRawOutput {
  /** NDC 空间高斯，[N=1179648, C]。 */
  gaussians: Gaussians3D
  /**
   * 单目视差图，shape [1, 2, 1536, 1536]（2 = `num_monodepth_layers`）。
   *
   * 注意：ONNX wrapper 输出的是 `md.disparity`，即**网络原始视差**，
   * 不是度量深度。度量深度 = `disparityFactor / disparity`（见
   * `predictor.py: monodepth = disparity_factor / monodepth_disparity`）。
   */
  disparity: { data: Float32Array; dims: number[] }
}

/** 内部推理分辨率，被 SPN 编码器结构锁死（见 py-models/README「关键约束 1」）。 */
export const INTERNAL_RESOLUTION = 1536

/**
 * 高斯数量 = 768 × 768 × 2 层。
 *
 * 来自 `params.py: InitializerParams.stride=2` 与 `num_layers=2`：
 * 1536/2 = 768，故 N = 768*768*2 = 1,179,648。
 */
export const NUM_GAUSSIANS = 1179648

/** 网络层数（`num_monodepth_layers`），与 disparity 输出的第 2 维一致。 */
export const NUM_LAYERS = 2
