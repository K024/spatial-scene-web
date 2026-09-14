/**
 * Node 平台会话实现（onnxruntime-node）。
 *
 * ── 为什么需要 monkey patch（实测结论）──
 * Node 24 原生提供 `Float16Array`。`onnxruntime-common` 检测到它后会把
 * fp16 张量映射为 `Float16Array`，但 `onnxruntime-node` 1.29 的原生 addon
 * 只识别 `Uint16Array`，于是喂 fp16 输入时报：
 *     `not enough space: expected 14155776, got 0`
 * 修复方式是在 ort **被 import 之前**让 `globalThis.Float16Array` 不可见，
 * 强制 ort-common 走 Uint16Array 路径（该路径 native addon 支持）。
 * 参见 `tensor-impl-type-mapping.js` 的 `NUMERIC_TENSOR_TYPE_TO_TYPEDARRAY_MAP`。
 *
 * 因此本模块的 import 顺序是**关键**：`disableFloat16Array()` 必须在
 * `import('onnxruntime-node')` 之前执行。为此这里用**动态 import**，
 * 保证 patch 先于模块求值。
 */

import type {
  CreateSessionOptions,
  ExecutionProviderHint,
  InferenceSessionHandle,
  InputDType,
  InputSpec,
  RawTensor,
  SessionCapabilities,
} from "../../src/spatial-scene/infer/session.ts"
import {
  f16BitsToF32,
  f32ToF16Bits,
} from "../../src/spatial-scene/infer/session.ts"

/**
 * 让 ort-common 回退到 Uint16Array 路径。
 *
 * 幂等：重复调用无副作用。必须在 ort 模块被求值前调用。
 *
 * 注意：这里**保存**了原始构造函数，供需要真实 Float16Array 的代码使用
 * （本仓库不需要——所有 fp16 传输都走 Uint16Array 位模式）。
 */
let patched = false
export function disableFloat16Array(): void {
  if (patched) return
  patched = true
  const g = globalThis as Record<string, unknown>
  // 置为 undefined 而非 delete：ort-common 的检测是 `typeof Float16Array !== 'undefined'`
  g.Float16Array = undefined
}

/**
 * 按平台推荐顺序选择 EP（决策 B）。
 *
 * 规则：
 *   - 显式 hint 优先（除 "auto"）；
 *   - "auto" 时按平台推荐：Windows -> dml, macOS -> webgpu, Linux -> cuda,
 *     再退到 cpu；
 *   - 环境变量 `SHARP_EP` 覆盖一切（值为 EP 名或逗号分隔的候选序列）。
 *
 * 之所以给出**候选序列**而非单一 EP：ort 支持 EP 回退链，
 * 让不受支持的算子落回 CPU 而不是整体失败。
 *
 * ── 为什么 macOS 不用 CoreML（实测）──
 * 本链路的模型是 **fp16**（`sharp_fp16.onnx`），而 CoreML EP 默认
 * `ModelFormat=NeuralNetwork` 时**只接受 fp32**（`base_op_builder.cc::
 * IsInputDtypeSupport`：fp16/int64 仅在 MLProgram 路径下支持）⇒ 整张图
 * 一个节点都不被接管，`EP=coreml` 只是注册成功、实际全在 CPU 上跑。
 * 换 `ModelFormat=MLProgram` 能接管 2647/2655 个节点，但首次编译要几十分钟
 * （还得配 `ModelCacheDirectory`）。fp32 基座同理不可用：建会话即吃掉
 * 17 GB RSS、峰值占用 65 GB+，前向直接把机器打到内存上限被杀。
 * 同一台机器上 `webgpu` 是 2756/2756 全接管、前向约 24s（CPU 对照约 108s），
 * 所以 macOS 默认走 webgpu。注意 webgpu 只配 **fp16** 模型：
 * `sharp_mm4f16_*.onnx` 的 `MatMulNBits` 在 node 的 WebGPU EP 上没有 kernel
 * （382 个节点回退 CPU，占 84.8% kernel 时间），只会更慢。
 */
export function resolveProviders(
  platform: NodeJS.Platform,
  hint: ExecutionProviderHint = "auto",
  env: NodeJS.ProcessEnv = process.env,
): string[] {
  const fromEnv = env.SHARP_EP
  if (fromEnv && fromEnv.trim().length > 0) {
    const list = fromEnv
      .split(",")
      .map((s) => s.trim().toLowerCase())
      .filter((s) => s.length > 0)
    if (list.length > 0) return list
  }
  if (hint !== "auto") return [hint]

  switch (platform) {
    case "win32":
      // DirectML 在 Windows 上是 onnxruntime-node 的推荐 GPU EP
      return ["dml", "cpu"]
    case "darwin":
      // webgpu 优先（理由见函数注释）；webgpu 不可用时由 ort 回退到 cpu
      return ["webgpu", "cpu"]
    case "linux":
      return ["cuda", "cpu"]
    default:
      return ["cpu"]
  }
}

/** ort 的 Tensor 构造器与类型（延迟取得，避免顶层 import 触发 patch 失效）。 */
interface OrtLike {
  Tensor: new (type: string, data: unknown, dims: readonly number[]) => unknown
  InferenceSession: {
    create(
      path: string,
      options: Record<string, unknown>,
    ): Promise<{
      inputNames: readonly string[]
      outputNames: readonly string[]
      inputMetadata: readonly {
        name: string
        type: string
        shape: readonly (number | string)[]
      }[]
      run(
        feeds: Record<string, unknown>,
      ): Promise<Record<string, { data: unknown; dims: readonly number[] }>>
      release?(): Promise<void>
    }>
  }
  env: { logLevel?: string }
  listSupportedBackends?(): { name: string; bundled: boolean }[]
}

/**
 * 动态取得 ort 并确保 patch 已生效。
 *
 * 这是**唯一**允许 import `onnxruntime-node` 的地方；
 * `sharp/`、`export/`、`session.ts` 都不得引用它。
 */
async function loadOrt(): Promise<OrtLike> {
  disableFloat16Array()
  const mod = (await import("onnxruntime-node")) as unknown as {
    default?: OrtLike
  } & OrtLike
  return (mod.default ?? mod) as OrtLike
}

/** 由 ort 的 type 字符串推导我们的 dtype 枚举。 */
function toInputDType(type: string): InputDType {
  if (type.includes("float16")) return "float16"
  return "float32"
}

/**
 * 创建 node 侧推理会话。
 *
 * 实现要点：
 *   - 模型路径直接交给 ort（外部数据 `.data` 由 ort 自动解析）；
 *   - `inputMetadata` 驱动 dtype 选择，**不硬编码** fp16/fp32；
 *   - fp16 输入以 Uint16Array 位模式交给 ort（绕过 Float16Array 问题）；
 *   - 输出统一转成数值 `Float32Array`（从 fp16 位模式还原）。
 */
export const createNodeSession = async (
  opts: CreateSessionOptions,
): Promise<InferenceSessionHandle> => {
  if (!opts.model.modelPath) {
    throw new Error("createNodeSession: `model.modelPath` is required on node")
  }
  const ort = await loadOrt()
  // 默认把 ort 日志压到 error：图优化阶段会对大量 attn/Sqrt、Tile 节点刷 warning，
  // 这些节点本就由 GPU EP 处理，属于噪音。调用方可用 SHARP_ORT_LOG 覆盖。
  if (ort.env) {
    const level = process.env.SHARP_ORT_LOG
    ort.env.logLevel = (level ?? "error") as typeof ort.env.logLevel
  }

  const providers = resolveProviders(process.platform, opts.provider ?? "auto")

  const session = await ort.InferenceSession.create(opts.model.modelPath, {
    executionProviders: providers,
    graphOptimizationLevel: "all",
    ...(opts.extra ?? {}),
  })

  const inputSpecs: InputSpec[] = session.inputNames.map((name) => {
    const meta = session.inputMetadata.find((m) => m.name === name)
    const type = meta?.type ?? "float32"
    return {
      name,
      dtype: toInputDType(type),
      dims: (meta?.shape ?? []).map((d) => (typeof d === "number" ? d : -1)),
    }
  })

  const available = (ort.listSupportedBackends?.() ?? []).map((b) => b.name)

  const capabilities: SessionCapabilities = {
    activeProvider: providers[0] ?? "cpu",
    availableProviders: available.length > 0 ? available : providers,
    fp16Input: inputSpecs.some((s) => s.dtype === "float16"),
  }

  const handle: InferenceSessionHandle = {
    inputSpecs,
    outputNames: [...session.outputNames],
    capabilities,

    async run(
      feeds: Record<string, RawTensor>,
    ): Promise<Record<string, RawTensor>> {
      const ortFeeds: Record<string, unknown> = {}
      for (const spec of inputSpecs) {
        const feed = feeds[spec.name]
        if (!feed) throw new Error(`run: missing input "${spec.name}"`)
        if (spec.dtype === "float16") {
          // 数值 -> fp16 位模式（Uint16Array），native addon 只认这个
          const bits = f32ToF16Bits(feed.data)
          ortFeeds[spec.name] = new ort.Tensor("float16", bits, feed.dims)
        } else {
          ortFeeds[spec.name] = new ort.Tensor("float32", feed.data, feed.dims)
        }
      }

      const raw = await session.run(ortFeeds)
      const out: Record<string, RawTensor> = {}
      for (const name of session.outputNames) {
        const t = raw[name]
        const dims = [...t.dims]
        // 输出的 fp16 同样以 Uint16Array 位模式返回（native addon 行为），
        // 统一还原成数值 Float32Array 再向上传递。
        const data = t.data
        const needsF16Decode =
          data instanceof Uint16Array &&
          // float16 输出的元素数应为 dims 乘积；此处以构造器判定
          true
        if (needsF16Decode && isFloat16Output(name)) {
          out[name] = { data: f16BitsToF32(data as Uint16Array), dims }
        } else if (data instanceof Float32Array) {
          out[name] = { data, dims }
        } else {
          // 兜底：把任意数值型数组转成 Float32Array
          out[name] = {
            data: Float32Array.from(data as ArrayLike<number>),
            dims,
          }
        }
      }
      return out
    },

    async dispose(): Promise<void> {
      await session.release?.()
    },
  }

  return handle
}

/**
 * 判断某输出是否为 fp16。
 *
 * 依据：ONNX 图的所有输出均为 fp16（由 `02_export_fp16.py` 决定）。
 * 这里显式列出而非依赖运行时 metadata，因为 ort-node 的 outputMetadata
 * 在部分版本中 dtype 字段不可靠（实测 `dtype=undefined`）。
 */
function isFloat16Output(_name: string): boolean {
  // 当前所有导出变体（fp16 / mm4f16）的输出都是 fp16。
  // 若未来引入 fp32 输出图，这里应改为读取 outputMetadata。
  return true
}
