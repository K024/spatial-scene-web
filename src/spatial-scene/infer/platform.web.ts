/**
 * Web 平台会话实现（onnxruntime-web）。
 *
 * ── 文件隔离（决策 A）──
 * 本模块是**唯一**允许 import `onnxruntime-web` 的地方。`session.ts`、
 * `sharp/*`、`export/*` 都不得引用它，因此浏览器构建不会误打包 node 原生模块，
 * 反之亦然。
 *
 * ── 与 node 实现的差异 ──
 *   1. 模型通过 URL 加载（fetch / HTTP range），不是文件系统路径；
 *   2. EP 是 `webgpu`（首选）与 `wasm`（兜底）。WebGPU EP 在 ort-web 中
 *      仍是实验性特性，故必须保留 wasm 回退；
 *   3. 浏览器没有 Node 的 `Float16Array` 喂数据缺陷——但为了与 node 侧
 *      **共用同一条数据路径**，这里同样以 Uint16Array/fp32 位模式传输，
 *      由 ort-web 内部处理。
 */

import type {
  CreateSessionOptions,
  ExecutionProviderHint,
  InferenceSessionHandle,
  InputDType,
  InputSpec,
  RawTensor,
  SessionCapabilities,
} from "./session.ts"
import { f16BitsToF32, f32ToF16Bits } from "./session.ts"

/** Web 侧可选的 EP 候选序列。 */
export function resolveWebProviders(
  hint: ExecutionProviderHint = "auto",
): string[] {
  if (hint !== "auto") return [hint]
  // WebGPU 优先，wasm 兜底；若浏览器不支持 WebGPU，ort-web 会自动落到 wasm。
  return ["webgpu", "wasm"]
}

/** ort-web 的最小接口（延迟 import）。 */
interface OrtWebLike {
  Tensor: new (type: string, data: unknown, dims: readonly number[]) => unknown
  InferenceSession: {
    create(
      url: string,
      options: Record<string, unknown>,
    ): Promise<{
      inputNames: readonly string[]
      outputNames: readonly string[]
      inputMetadata?: readonly {
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
  env: Record<string, unknown>
}

async function loadOrtWeb(): Promise<OrtWebLike> {
  const mod = (await import("onnxruntime-web")) as unknown as {
    default?: OrtWebLike
  } & OrtWebLike
  return (mod.default ?? mod) as OrtWebLike
}

function toInputDType(type: string | undefined): InputDType {
  return type?.includes("float16") ? "float16" : "float32"
}

/**
 * 创建 web 侧推理会话。
 *
 * 注意：`inputMetadata` 在部分 ort-web 版本里可能缺失，此时按 SHARP 图的
 * 已知契约（image / disparity_factor 均为 fp16）推断 dtype。
 */
export const createWebSession = async (
  opts: CreateSessionOptions,
): Promise<InferenceSessionHandle> => {
  if (!opts.model.modelUrl) {
    throw new Error("createWebSession: `model.modelUrl` is required on web")
  }
  const ort = await loadOrtWeb()
  const providers = resolveWebProviders(opts.provider ?? "auto")

  const session = await ort.InferenceSession.create(opts.model.modelUrl, {
    executionProviders: providers,
    graphOptimizationLevel: "all",
    ...(opts.extra ?? {}),
  })

  const metaByName = new Map(
    (session.inputMetadata ?? []).map((m) => [m.name, m] as const),
  )

  const inputSpecs: InputSpec[] = session.inputNames.map((name) => {
    const meta = metaByName.get(name)
    return {
      name,
      dtype: toInputDType(meta?.type),
      dims: (meta?.shape ?? []).map((d) => (typeof d === "number" ? d : -1)),
    }
  })

  const capabilities: SessionCapabilities = {
    activeProvider: providers[0] ?? "wasm",
    availableProviders: providers,
    fp16Input: inputSpecs.some((s) => s.dtype === "float16"),
  }

  return {
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
          ortFeeds[spec.name] = new ort.Tensor(
            "float16",
            f32ToF16Bits(feed.data),
            feed.dims,
          )
        } else {
          ortFeeds[spec.name] = new ort.Tensor("float32", feed.data, feed.dims)
        }
      }

      const raw = await session.run(ortFeeds)
      const out: Record<string, RawTensor> = {}
      for (const name of session.outputNames) {
        const t = raw[name]
        const dims = [...t.dims]
        const data = t.data
        // 与 node 侧一致：fp16 输出统一还原成数值 Float32Array
        if (data instanceof Uint16Array) {
          out[name] = { data: f16BitsToF32(data), dims }
        } else if (data instanceof Float32Array) {
          out[name] = { data, dims }
        } else {
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
}
