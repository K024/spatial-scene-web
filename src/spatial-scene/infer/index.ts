/**
 * 推理管线装配层（平台无关）。
 *
 * 这里把「预处理 -> 会话 -> 原始输出 -> 反投影 -> 度量空间高斯」串起来。
 * 本文件**不**静态 import 任何 ort 包，也不 import 平台实现，
 * 而是在调用 `runSharp` 时由调用方注入 `createSession`（决策 A + C）。
 *
 * 这样做的收益：
 *   - web 构建只打包 `platform.web.ts`，node 构建只打包 `platform.node.ts`；
 *   - 测试时可注入 mock 会话，不需要真实 ort；
 *   - 数值逻辑与运行时彻底解耦，便于对拍。
 */

import { disparityFactor, intrinsicsResized } from "../sharp/fov.ts"
import { preprocessImage, type SourceImage } from "../sharp/preprocess.ts"
import {
  type Gaussians3D,
  INTERNAL_RESOLUTION,
  NUM_GAUSSIANS,
  ONNX_OUTPUT_NAMES,
  type SharpRawOutput,
} from "../sharp/types.ts"
import {
  applyTransform,
  getUnprojectionMatrix,
  identity4,
} from "../sharp/unproject.ts"
import type {
  CreateSession,
  InferenceSessionHandle,
  RawTensor,
} from "./session.ts"

export interface RunSharpOptions {
  /** 原始图像（HWC）。 */
  image: SourceImage
  /** 原始图像域的像素焦距 `f_px`。 */
  fPx: number
  /** 原始图像宽（用于 `disparity_factor = f_px / width`）。 */
  imageWidth: number
  /** 会话工厂（`createNodeSession` 或 `createWebSession`）。 */
  createSession: CreateSession
  /** 模型引用。 */
  model: { modelPath?: string; modelUrl?: string }
  /** EP hint / 其他。 */
  provider?: import("./session.ts").ExecutionProviderHint
  extra?: Record<string, unknown>
}

/** 完整推理结果（含中间量，便于调试与导出）。 */
export interface RunSharpResult {
  /** NDC 空间高斯（ONNX 原样输出）。 */
  ndc: Gaussians3D
  /** 度量空间高斯（已反投影）。 */
  metric: Gaussians3D
  /** 单目视差 [1, 2, 1536, 1536]。 */
  disparity: { data: Float32Array; dims: number[] }
  /** 反投影矩阵（4x4 行主序）。 */
  unprojectionMatrix: Float64Array
  /** 实际喂入的 disparity_factor。 */
  disparityFactor: number
  /** 会话能力（EP / fp16 支持）。 */
  capabilities: InferenceSessionHandle["capabilities"]
}

/**
 * 把 ONNX 原始输出张量整理成 `SharpRawOutput`。
 *
 * 校验形状与元素数，尽早暴露「导出时的图与预期不符」这类问题
 * （py 侧已有测试，这里只做断言级防护）。
 */
export function parseRawOutput(raw: Record<string, RawTensor>): SharpRawOutput {
  for (const name of ONNX_OUTPUT_NAMES) {
    if (!raw[name]) throw new Error(`parseRawOutput: missing output "${name}"`)
  }

  const expectCount = (name: string, expected: number): void => {
    const n = raw[name].data.length
    if (n !== expected) {
      throw new Error(
        `parseRawOutput: output "${name}" has ${n} elements, expected ${expected}`,
      )
    }
  }

  expectCount("mean_vectors", NUM_GAUSSIANS * 3)
  expectCount("singular_values", NUM_GAUSSIANS * 3)
  expectCount("quaternions", NUM_GAUSSIANS * 4)
  expectCount("colors", NUM_GAUSSIANS * 3)
  expectCount("opacities", NUM_GAUSSIANS)

  const disp = raw["disparity"]
  if (disp.data.length !== 2 * INTERNAL_RESOLUTION * INTERNAL_RESOLUTION) {
    throw new Error(
      `parseRawOutput: disparity has ${disp.data.length} elements, expected ` +
        `${2 * INTERNAL_RESOLUTION * INTERNAL_RESOLUTION}`,
    )
  }

  return {
    gaussians: {
      meanVectors: raw["mean_vectors"].data,
      singularValues: raw["singular_values"].data,
      quaternions: raw["quaternions"].data,
      colors: raw["colors"].data,
      opacities: raw["opacities"].data,
    },
    disparity: { data: disp.data, dims: [...disp.dims] },
  }
}

/**
 * 完整跑一遍 SHARP 推理（单图）。
 *
 * 步骤严格对照 `predict.py: predict_image`：
 *   1. 缩放至 1536×1536、归一化，得 CHW 输入
 *   2. `disparity_factor = f_px / width`
 *   3. 推理，得到 **NDC 空间**高斯
 *   4. `intrinsics_resized`（乘 1536/W 与 1536/H）
 *   5. `unproject_gaussians(extrinsics=I, intrinsics_resized, (1536,1536))`
 *
 * @returns 含 NDC 与 metric 两套高斯的结果。
 */
export async function runSharp(opts: RunSharpOptions): Promise<{
  result: RunSharpResult
  session: InferenceSessionHandle
}> {
  const session = await opts.createSession({
    model: opts.model,
    provider: opts.provider,
    extra: opts.extra,
  })

  try {
    // 1. 预处理
    const imageChw = preprocessImage(opts.image, INTERNAL_RESOLUTION)

    // 2. disparity_factor
    const disp = disparityFactor(opts.fPx, opts.imageWidth)

    // 3. 推理（fp16 以数值语义传入，平台层负责位模式转换）
    const raw = await session.run({
      image: {
        data: imageChw,
        dims: [1, 3, INTERNAL_RESOLUTION, INTERNAL_RESOLUTION],
      },
      disparity_factor: { data: Float32Array.from([disp]), dims: [1] },
    })

    const parsed = parseRawOutput(raw)

    // 4. intrinsics_resized
    const k = intrinsicsResized(
      opts.fPx,
      opts.imageWidth,
      opts.image.height,
      INTERNAL_RESOLUTION,
    )

    // 5. NDC -> metric
    const extrinsics = identity4()
    const unprojectionMatrix = getUnprojectionMatrix(extrinsics, k, [
      INTERNAL_RESOLUTION,
      INTERNAL_RESOLUTION,
    ])
    const metric = applyTransform(parsed.gaussians, unprojectionMatrix)

    return {
      session,
      result: {
        ndc: parsed.gaussians,
        metric,
        disparity: parsed.disparity,
        unprojectionMatrix,
        disparityFactor: disp,
        capabilities: session.capabilities,
      },
    }
  } catch (err) {
    await session.dispose().catch(() => {})
    throw err
  }
}
