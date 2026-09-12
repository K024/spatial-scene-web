/**
 * Node 脚本共享工具。
 *
 * 决策 A：ort 的 `Float16Array` monkey patch 放在这里（`scripts/utils/`），
 * 而不是散落在各脚本里。任何需要 ort 的脚本**必须**先调用
 * `prepareOrtEnv()`，它保证 patch 在 ort 模块求值前生效。
 *
 * 注意：`src/spatial-scene/infer/platform.node.ts` 内部也做了同样的 patch
 * （因为它用动态 import，能自保证顺序）。两处幂等，重复调用无害。
 * 之所以这里再放一份，是为了让「脚本里直接 require ort」的场景也安全。
 */

import { existsSync, statSync } from "node:fs"
import { resolve } from "node:path"

/** 仓库根目录（本文件位于 scripts/utils/ 下，向上两级）。 */
export const REPO_ROOT = resolve(import.meta.dirname, "..", "..")

/** py-models 导出的 ONNX 产物目录。 */
export const ONNX_DIR = resolve(REPO_ROOT, "py-models", "out", "onnx")

/** 默认 fp16 模型（高精度，本地测试目标）。 */
export const MODEL_FP16 = resolve(ONNX_DIR, "sharp_fp16.onnx")

/** 4bit 量化模型（WebGPU 目标，后续用）。 */
export const MODEL_MM4F16 = resolve(ONNX_DIR, "sharp_mm4f16_hqq_bs32.onnx")

/** 默认测试图（ml-sharp 仓库自带的单张示例）。 */
export const DEFAULT_IMAGE = resolve(
  REPO_ROOT,
  "py-models",
  "clones",
  "ml-depth-pro",
  "data",
  "example.jpg",
)

/**
 * 备选图：ml-sharp 的 teaser 拼图。
 *
 * 注意这是一个 3×2 的对比拼图（上排参考照片 / 下排渲染结果），
 * **不是**单场景照片。用作输入时需配合 `--crop` 裁出其中一个面板。
 */
export const TEASER_IMAGE = resolve(
  REPO_ROOT,
  "py-models",
  "clones",
  "ml-sharp",
  "data",
  "teaser.jpg",
)

/**
 * 让 ort-common 回退到 Uint16Array 路径。
 *
 * 必须在 `import('onnxruntime-node')` **之前**调用。实测 Node 24 上
 * `Float16Array` 会让 native addon 报 `not enough space: expected N, got 0`。
 *
 * 幂等。
 */
let patched = false
export function disableFloat16Array(): void {
  if (patched) return
  patched = true
  ;(globalThis as Record<string, unknown>).Float16Array = undefined
}

/**
 * 准备 ort 运行环境：做 patch + 可选日志级别。
 *
 * 调用方应把它放在任何 ort import 之前（脚本顶层第一件事）。
 */
export function prepareOrtEnv(
  logLevel: "error" | "warning" | "info" | "verbose" = "error",
): void {
  disableFloat16Array()
  process.env.ORT_LOG_LEVEL = logLevel
}

/** 断言文件存在，返回其绝对路径。 */
export function requireFile(path: string, label: string): string {
  if (!existsSync(path)) {
    throw new Error(`${label} 不存在: ${path}`)
  }
  return path
}

/** 人类可读的文件大小。 */
export function humanSize(bytes: number): string {
  const units = ["B", "KB", "MB", "GB"]
  let v = bytes
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v.toFixed(i === 0 ? 0 : 2)} ${units[i]}`
}

/** 打印文件大小信息。 */
export function reportFile(path: string, label: string): void {
  const st = statSync(path)
  console.log(`${label}: ${path} (${humanSize(st.size)})`)
}
