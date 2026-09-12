/**
 * 从 SHARP fixtures 里取出「模型自己的深度估计」，用于与 wsplat 渲染出的 D 做数值对比。
 *
 * ── 模型给出的到底是什么 ──
 * ONNX 图的第 6 个输出是 `disparity`，形状 `[2, 1536, 1536]`（2 = `num_monodepth_layers`），
 * 是**网络归一化视差**，不是米。换算关系来自 ml-sharp 自己的代码：
 *
 *   `predict.py:171`        `disparity_factor = f_px / width`        （f_px/width 都是**原图域**）
 *   `initializer.py:122`    `normalized_disparity = disparity_factor / depth`
 *   ⇒                       `depth = disparity_factor / disparity`
 *
 * ── 两个坑（写在这里免得后人再踩）──
 * 1. **网络域是 1536×1536 正方形**：`F.interpolate(size=(1536,1536))` 是**非等比拉伸**，
 *    所以对比时必须按**归一化坐标**映射（`u,v ∈ [0,1]`），不能按像素比例线性缩放。
 * 2. **两层**：`num_monodepth_layers = 2`，第 0 层是可见表面、第 1 层是其后方的支撑面。
 *    本阶段我们的 D 是 alpha 加权深度，通常更接近第 0 层；第 0/1 层与「两者取近」
 *    都会算出来一起报，避免"选错层导致本该对上的对不上"。
 */

import { resolve } from "node:path"
import { REPO_ROOT } from "./common.ts"
import { loadNpz } from "./npz.ts"

/** 网络内部域边长（`INTERNAL_RESOLUTION`）。 */
const INTERNAL = 1536

export interface SharpDepths {
  /** 网络域边长（1536）。 */
  readonly internalSize: number
  /** 每层一张度量深度图，`internalSize²`，单位米。 */
  readonly layers: readonly Float32Array[]
  /** `disparity_factor = f_px / width`（原图域）。 */
  readonly disparityFactor: number
  /** 原图像素焦距。 */
  readonly focalPx: number
  /** 原图尺寸 `[width, height]`。 */
  readonly originalSize: readonly [number, number]
}

/** 读取 fixtures 里的 SHARP 深度估计（缺文件时返回 undefined，便于可选使用）。 */
export function loadSharpDepths(options?: {
  ndcFixture?: string
  inputFixture?: string
}): SharpDepths | undefined {
  const ndcPath =
    options?.ndcFixture ?? resolve(REPO_ROOT, "py-models/out/fixtures/ndc.npz")
  const inputPath =
    options?.inputFixture ??
    resolve(REPO_ROOT, "py-models/out/fixtures/input.npz")

  let ndc: ReturnType<typeof loadNpz>
  let input: ReturnType<typeof loadNpz>
  try {
    ndc = loadNpz(ndcPath)
    input = loadNpz(inputPath)
  } catch {
    return undefined
  }
  const disparity = ndc.disparity
  const factor = input.disparity_factor
  if (!disparity || !factor) return undefined

  const layers = Number.parseInt(String(disparity.shape[0] ?? 2), 10)
  const size = Number.parseInt(String(disparity.shape[1] ?? INTERNAL), 10)
  const out: Float32Array[] = []
  const dFactor = factor.data[0]
  for (let l = 0; l < layers; l++) {
    const map = new Float32Array(size * size)
    const base = l * size * size
    for (let i = 0; i < map.length; i++) {
      const d = disparity.data[base + i]
      // 视差 <= 0 视为无效（远处/背景）；用它当深度会得到负数或爆炸值
      map[i] = d > 1e-6 ? dFactor / d : 0
    }
    out.push(map)
  }
  return {
    internalSize: size,
    layers: out,
    disparityFactor: dFactor,
    focalPx: input.f_px?.data[0] ?? 0,
    originalSize: [
      Number.parseInt(String(input.orig_width?.data[0] ?? 0), 10),
      Number.parseInt(String(input.orig_height?.data[0] ?? 0), 10),
    ],
  }
}

/**
 * 按**归一化坐标**双线性采样（网络域被拉伸到正方形，所以必须这么做）。
 *
 * @param u 归一化横坐标 `[0,1]`（0 = 左）
 * @param v 归一化纵坐标 `[0,1]`（0 = 上）
 */
export function sampleNormalized(
  map: Float32Array,
  size: number,
  u: number,
  v: number,
): number {
  const x = u * size - 0.5
  const y = v * size - 0.5
  const x0 = Math.floor(x)
  const y0 = Math.floor(y)
  const fx = x - x0
  const fy = y - y0
  const at = (ix: number, iy: number): number => {
    const cx = ix < 0 ? 0 : ix >= size ? size - 1 : ix
    const cy = iy < 0 ? 0 : iy >= size ? size - 1 : iy
    return map[cy * size + cx]
  }
  const v00 = at(x0, y0)
  const v10 = at(x0 + 1, y0)
  const v01 = at(x0, y0 + 1)
  const v11 = at(x0 + 1, y0 + 1)
  return (
    v00 * (1 - fx) * (1 - fy) +
    v10 * fx * (1 - fy) +
    v01 * (1 - fx) * fy +
    v11 * fx * fy
  )
}

/**
 * 把某一层重采样到我们的渲染分辨率（近邻边界、双线性内部），返回 `width*height` 的米制深度。
 *
 * 无效值（0）保持 0；调用方用 `visible` 过滤。
 */
export function resampleLayer(
  sharp: SharpDepths,
  layer: number,
  width: number,
  height: number,
): Float32Array {
  const map = sharp.layers[layer]
  if (!map)
    throw new Error(
      `SHARP 只有 ${sharp.layers.length} 层，取不到第 ${layer} 层`,
    )
  const out = new Float32Array(width * height)
  for (let y = 0; y < height; y++) {
    const v = (y + 0.5) / height
    for (let x = 0; x < width; x++) {
      const u = (x + 0.5) / width
      out[y * width + x] = sampleNormalized(map, sharp.internalSize, u, v)
    }
  }
  return out
}
