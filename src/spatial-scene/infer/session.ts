/**
 * 推理会话的**平台无关**抽象。
 *
 * 设计目标（决策 A）：
 *   - 本文件只从 `onnxruntime-common` 引入**类型**，绝不引入 `onnxruntime-node`
 *     或 `onnxruntime-web` 的运行时。这样 web 构建不会被打包进 node 原生模块。
 *   - 平台实现（`platform.node.ts` / `platform.web.ts`）负责：模型加载、
 *     ort 可用性探测、把 ort 的 `Tensor` 摊平成 `Float32Array`。
 *   - 所有数值逻辑（反投影、颜色、PLY）都在 `sharp/` 与 `export/` 中，与本层无关。
 *
 * 术语：这里把「喂给模型的输入」与「模型吐出的输出」都定义成纯数据
 * （`Float32Array` + dims），而非 ort 的 `Tensor` 对象——这是 web/node 共用的
 * 最低公共契约。
 */

/** 张量的纯数据表示（不含 ort 依赖）。 */
export interface RawTensor {
  data: Float32Array
  dims: number[]
}

/**
 * 模型描述符：定位于 ONNX 文件。
 *
 * web 侧给 URL（`modelUrl`）；node 侧给文件路径（`modelPath`）。
 * 两者互斥，由各平台实现自行校验。
 */
export interface ModelRef {
  /** node: ONNX 文件系统路径（外部数据 .data 会被 ort 自动解析）。 */
  modelPath?: string
  /** web: ONNX 文件 URL。 */
  modelUrl?: string
}

/** 执行提供者偏好。各平台自行决定支持哪些值。 */
export type ExecutionProviderHint =
  | "auto" // 平台上按推荐顺序自动选择（见 platform 实现）
  | "dml" // DirectML（node/Windows，本项目本地测试首选）
  | "cuda" // CUDA（node）
  | "coreml" // CoreML（node/macOS；本链路 fp16 模型默认不接管，见 platform.node.ts）
  | "webgpu" // WebGPU（web 与 macOS node 首选）
  | "wasm" // WASM（web 兜底）
  | "cpu" // 纯 CPU（保底，通常极慢）

/** 会话能力，供上层决策（例如 dtype 选择、UI 提示）。 */
export interface SessionCapabilities {
  /** 实际生效的 EP 名（如 "dml" / "webgpu"）。 */
  activeProvider: string
  /** 平台可用的全部 EP 名。 */
  availableProviders: string[]
  /** 是否支持以 fp16 输入调用（影响喂数据的 dtype）。 */
  fp16Input: boolean
}

/** 输入张量的 dtype 声明（由图的 inputMetadata 推导，不可硬编码）。 */
export type InputDType = "float16" | "float32"

/** 单个输入的规格。 */
export interface InputSpec {
  name: string
  dtype: InputDType
  dims: number[]
}

/** 会话句柄：统一的 run 接口。 */
export interface InferenceSessionHandle {
  readonly inputSpecs: InputSpec[]
  readonly outputNames: string[]
  readonly capabilities: SessionCapabilities

  /**
   * 执行一次推理。
   *
   * @param feeds 键为输入名；值必须匹配 `inputSpecs` 声明的 dtype。
   *   fp16 输入以 `Float32Array`（数值语义）传入，由平台实现负责转位模式；
   *   这样调用方无需关心 Float16Array 的可用性问题（见 platform.node 的 monkey patch）。
   */
  run(feeds: Record<string, RawTensor>): Promise<Record<string, RawTensor>>

  /** 释放底层资源。 */
  dispose(): Promise<void>
}

/** 创建会话的选项。 */
export interface CreateSessionOptions {
  model: ModelRef
  provider?: ExecutionProviderHint
  /** 给底层 ort 的额外选项（平台相关，透传）。 */
  extra?: Record<string, unknown>
}

/** 平台会话工厂：由 `platform.*.ts` 实现并导出。 */
export type CreateSession = (
  opts: CreateSessionOptions,
) => Promise<InferenceSessionHandle>

/** 把 `Float32Array` 的数值转成 fp16 位模式（Uint16Array）。 */
export function f32ToF16Bits(src: Float32Array): Uint16Array {
  const out = new Uint16Array(src.length)
  for (let i = 0; i < src.length; i++) out[i] = f32ToF16(src[i])
  return out
}

/** fp16 位模式（Uint16Array）转回数值 Float32Array。 */
export function f16BitsToF32(src: Uint16Array): Float32Array {
  const out = new Float32Array(src.length)
  for (let i = 0; i < src.length; i++) out[i] = f16ToF32(src[i])
  return out
}

/**
 * 单个 float32 -> float16 位模式。
 *
 * 手写位运算而非依赖 `Float16Array`：后者在 Node 24 虽原生存在，但
 * `onnxruntime-node` 的原生 addon 不接受它（见 `platform.node.ts` 的说明）。
 * 手写转换保证两端行为一致，且不依赖宿主对新类型的支持。
 *
 * 采用 round-to-nearest-even，与硬件 fp16 转换语义一致。
 */
export function f32ToF16(value: number): number {
  // 用共享缓冲做位级操作
  F32_SCRATCH[0] = value
  const x = I32_SCRATCH[0]

  const sign = (x >>> 16) & 0x8000
  const exp = (x >>> 23) & 0xff
  let mant = x & 0x7fffff

  // NaN / Inf
  if (exp === 0xff) {
    return (sign | 0x7c00 | (mant !== 0 ? 0x200 : 0)) & 0xffff
  }

  // 重新偏置指数：f32 bias 127 -> f16 bias 15
  const newExp = exp - 127 + 15

  if (newExp >= 0x1f) {
    // 上溢 -> Inf
    return (sign | 0x7c00) & 0xffff
  }
  if (newExp <= 0) {
    // 下溢到 subnormal 或 0
    if (newExp < -10) {
      return sign & 0xffff
    }
    mant |= 0x800000
    const shift = 14 - newExp
    let half = mant >>> shift
    // round-to-nearest-even
    const rem = mant & ((1 << shift) - 1)
    const halfway = 1 << (shift - 1)
    if (rem > halfway || (rem === halfway && (half & 1) === 1)) half++
    return (sign | half) & 0xffff
  }

  // 常规情况
  let half = (newExp << 10) | (mant >>> 13)
  // round-to-nearest-even
  const rem = mant & 0x1fff
  if (rem > 0x1000 || (rem === 0x1000 && (half & 1) === 1)) {
    half++
  }
  return (sign | half) & 0xffff
}

/** 单个 float16 位模式 -> float32。 */
export function f16ToF32(h: number): number {
  const sign = (h & 0x8000) << 16
  const exp = (h >>> 10) & 0x1f
  const mant = h & 0x3ff

  if (exp === 0) {
    if (mant === 0) {
      I32_SCRATCH_OUT[0] = sign
      return F32_SCRATCH_OUT[0]
    }
    // subnormal：正规化
    let m = mant
    let e = -1
    while ((m & 0x400) === 0) {
      m <<= 1
      e--
    }
    m &= 0x3ff
    const exp32 = (127 - 15 + e + 1) << 23
    I32_SCRATCH_OUT[0] = sign | exp32 | (m << 13)
    return F32_SCRATCH_OUT[0]
  }
  if (exp === 0x1f) {
    I32_SCRATCH_OUT[0] = sign | 0x7f800000 | (mant << 13)
    return F32_SCRATCH_OUT[0]
  }
  I32_SCRATCH_OUT[0] = sign | ((exp - 15 + 127) << 23) | (mant << 13)
  return F32_SCRATCH_OUT[0]
}

// 共享 scratch，避免每元素分配（单线程串行使用，安全）
const F32_SCRATCH = new Float32Array(1)
const I32_SCRATCH = new Int32Array(F32_SCRATCH.buffer)
const F32_SCRATCH_OUT = new Float32Array(1)
const I32_SCRATCH_OUT = new Int32Array(F32_SCRATCH_OUT.buffer)
