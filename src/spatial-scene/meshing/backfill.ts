/**
 * 回填 / 背衬平面 —— **纯算法**，不跑任何网络。
 *
 * 阶段：所有层网格化之后。
 *
 * ── 解决什么 ──
 * 多层网格在各层边缘遮不住的地方会露出**宽空档**（`layering-perlayer-quality` 实测单层可达
 * 4.65%）。在所有层之后画一张**最远深度的背衬平面**，贴上一张把 L 层合成后 α 加权降采样的
 * 粗纹理，把透视穿帮堵死。
 *
 * ── 算法（对齐 `MXIBackLayer` 的三个 kernel）──
 * ```
 * _concatRGBD            : 把各层 RGB + Depth 并在一起
 * _downscaleAlphtaWeighted: α 加权降采样（预乘空间做盒平均，保 α 能量）
 * _backLayerBlend        : 混合成一张背衬纹理
 * ```
 * 这里实现为：**back-to-front 合成所有层 → 预乘 → 盒降采样 → 解预乘**。
 * 按 α 预乘再平均等价于「α 加权平均」，且不会让低 α 像素的颜色污染结果。
 *
 * ── 为什么是平面，不是「第 L+1 层」──
 * 它不参与分层合成（不写 `layers`），只在最后画；深度取最远层边界。语义对齐 Apple
 * `generateBackingPlaneMesh:atDepth:` —— 一块铺满视锥的 quad（`layerIndex = -1`）。
 *
 * @see SOLIDI 无对应实现（它用 inpainting 网络补洞）；本模块按 Apple `MXIBackLayer` +
 *   backing plane 的**算法级**路线自研。
 */

import type { WSplatCamera } from "../wsplat/camera.ts"
import type { WSplatFrame } from "../wsplat/types.ts"
import type { BackfillOptions, LayerMesh, RgbaTexture } from "./types.ts"

/** 只借颜色所需的字段。 */
export type BackfillFrame = Pick<
  WSplatFrame,
  "rgb" | "alpha" | "width" | "height"
>

const DEFAULT_DOWNSCALE = 4
const EPS = 1e-6

/**
 * back-to-front 合成所有层（`layers[0]` = 最近），返回直通 RGB + α。
 *
 * `over`（预乘形式）：`C ← c_s·a_s + (1−a_s)·C`，`A ← a_s + (1−a_s)·A`。
 */
export function compositeLayersBackToFront(
  frames: readonly BackfillFrame[],
  width: number,
  height: number,
): RgbaTexture {
  const pixels = width * height
  const premul = new Float32Array(pixels * 3)
  const alpha = new Float32Array(pixels)
  for (let k = frames.length - 1; k >= 0; k--) {
    const frame = frames[k]
    for (let i = 0; i < pixels; i++) {
      const a = frame.alpha[i]
      if (!(a > 0)) continue
      const inv = 1 - a
      const o = i * 3
      premul[o] = frame.rgb[o] * a + premul[o] * inv
      premul[o + 1] = frame.rgb[o + 1] * a + premul[o + 1] * inv
      premul[o + 2] = frame.rgb[o + 2] * a + premul[o + 2] * inv
      alpha[i] = a + alpha[i] * inv
    }
  }
  return unpremultiply({ width, height, rgb: premul, alpha })
}

/**
 * α 加权降采样（整数倍率盒平均）。
 *
 * 预乘空间求和再平均 ⇒ 颜色按 α 加权；α 本身也做盒平均（保能量）。
 * `factor <= 1` 时原样返回。
 */
export function downscaleAlphaWeighted(
  src: RgbaTexture,
  factor: number,
): RgbaTexture {
  const f = Math.max(1, Math.floor(factor))
  if (f <= 1) return src
  const outWidth = Math.max(1, Math.ceil(src.width / f))
  const outHeight = Math.max(1, Math.ceil(src.height / f))
  const premul = new Float32Array(outWidth * outHeight * 3)
  const alpha = new Float32Array(outWidth * outHeight)
  for (let oy = 0; oy < outHeight; oy++) {
    for (let ox = 0; ox < outWidth; ox++) {
      let r = 0
      let g = 0
      let b = 0
      let a = 0
      let count = 0
      const x0 = ox * f
      const y0 = oy * f
      for (let y = y0; y < Math.min(y0 + f, src.height); y++) {
        for (let x = x0; x < Math.min(x0 + f, src.width); x++) {
          const i = y * src.width + x
          const ai = src.alpha[i]
          r += src.rgb[i * 3] * ai
          g += src.rgb[i * 3 + 1] * ai
          b += src.rgb[i * 3 + 2] * ai
          a += ai
          count++
        }
      }
      if (count === 0) continue
      const o = (oy * outWidth + ox) * 3
      premul[o] = r / count
      premul[o + 1] = g / count
      premul[o + 2] = b / count
      alpha[oy * outWidth + ox] = a / count
    }
  }
  return unpremultiply({
    width: outWidth,
    height: outHeight,
    rgb: premul,
    alpha,
  })
}

/** 预乘 → 直通（`c = C / max(A, eps)`）。 */
function unpremultiply(src: RgbaTexture): RgbaTexture {
  const rgb = new Float32Array(src.rgb.length)
  for (let i = 0; i < src.alpha.length; i++) {
    const a = src.alpha[i]
    if (a <= EPS) continue
    const inv = 1 / a
    rgb[i * 3] = src.rgb[i * 3] * inv
    rgb[i * 3 + 1] = src.rgb[i * 3 + 1] * inv
    rgb[i * 3 + 2] = src.rgb[i * 3 + 2] * inv
  }
  return { width: src.width, height: src.height, rgb, alpha: src.alpha }
}

/**
 * 生成背衬网格：铺满视锥的一块 quad，位于 `depth`（米）。
 *
 * 4 个顶点在图像四角（连续坐标 `u ∈ [0, W]`、`v ∈ [0, H]`），UV `[0,1]²`。
 * 顶点用与 relief 完全相同的针孔反投影（`u = x+0.5`，故取 `x = u−0.5`）。
 */
export function buildBackingPlaneMesh(
  camera: WSplatCamera,
  depth: number,
  disparityRange: readonly [number, number],
  depthRange: readonly [number, number],
  texture: RgbaTexture,
): LayerMesh {
  const width = camera.width
  const height = camera.height
  const fx = width / (2 * Math.tan(camera.fovX / 2))
  const fy = height / (2 * Math.tan(camera.fovY / 2))
  const cx = width / 2
  const cy = height / 2
  const view = camera.viewMatrix
  const right: [number, number, number] = [view[0], view[4], view[8]]
  const down: [number, number, number] = [view[1], view[5], view[9]]
  const fwd: [number, number, number] = [view[2], view[6], view[10]]
  const origin = camera.position

  // 四角：x = u − 0.5（因为公式内部用 x+0.5）。
  const corners: Array<[number, number]> = [
    [-0.5, -0.5],
    [width - 0.5, -0.5],
    [-0.5, height - 0.5],
    [width - 0.5, height - 0.5],
  ]
  const positions = new Float32Array(12)
  const uvs = new Float32Array(8)
  for (let c = 0; c < 4; c++) {
    const [x, y] = corners[c]
    const u = x + 0.5
    const v = y + 0.5
    const camX = ((u - cx) / fx) * depth
    const camY = ((v - cy) / fy) * depth
    positions[c * 3] =
      origin[0] + right[0] * camX + down[0] * camY + fwd[0] * depth
    positions[c * 3 + 1] =
      origin[1] + right[1] * camX + down[1] * camY + fwd[1] * depth
    positions[c * 3 + 2] =
      origin[2] + right[2] * camX + down[2] * camY + fwd[2] * depth
    uvs[c * 2] = u / width
    uvs[c * 2 + 1] = v / height
  }
  // 顶点顺序：0=左上, 1=右上, 2=左下, 3=右下。绕序朝向相机（同 relief）。
  const indices = Uint32Array.from([0, 3, 1, 0, 2, 3])
  return {
    layerIndex: -1,
    width,
    height,
    vertexCount: 4,
    triangleCount: 2,
    wallTriangleCount: 0,
    opaqueTriangleCount: 0,
    positions,
    uvs,
    indices,
    texture,
    disparityRange,
    depthRange,
  }
}

/**
 * 建背衬平面：合成 L 层 → α 加权降采样 → 最远深度的 quad。
 *
 * @returns `null` 表示关闭或没有可合成的内容。
 */
export function buildBackingPlane(
  frames: readonly BackfillFrame[],
  camera: WSplatCamera,
  farDepth: number,
  disparityRange: readonly [number, number],
  depthRange: readonly [number, number],
  options: BackfillOptions = {},
): LayerMesh | null {
  if (options.enabled === false) return null
  if (frames.length === 0) return null
  const width = camera.width
  const height = camera.height
  const depth = options.depth ?? farDepth
  const texture = downscaleAlphaWeighted(
    compositeLayersBackToFront(frames, width, height),
    options.downscale ?? DEFAULT_DOWNSCALE,
  )
  return buildBackingPlaneMesh(
    camera,
    depth,
    disparityRange,
    depthRange,
    texture,
  )
}
