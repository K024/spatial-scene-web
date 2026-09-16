/**
 * 参考光栅器：把 `MeshScene` 的**带纹理三角形**用 CPU 画一遍。
 *
 * ── 为什么要它 ──
 * meshing 的结构门（索引在界内 / UV 范围 / 顶点深度）都验不了「UV 是不是贴对了」
 * 「三角形绕序 / 拓扑对不对」。把 mesh 真的画一遍并与 splat 直渲对照，是唯一
 * 能抓住「mesh 整体错位 / UV 翻转 / 索引错」这类错误的检查。
 *
 * ── 口径（与渲染器的差异，读之前先知道）──
 * - 纹理采样用**最近邻**（WGSL/GPU 侧是双线性 + mip），所以高频区域必然有差；
 * - 每层用**全局 z-buffer + over 混合**，远 -> 近画。GPU 侧的 LDI 是「每层独立
 *   渲染再合成」，在参考视角上两者数学等价（`over` 结合），但单层内部的 α 加权
 *   与 z-buffer 的「最近像素获胜」不同，所以半透明边缘会有差异；
 * - 不做背面剔除（与 GLB 的 `doubleSided` 一致）。
 *
 * 结论：它是**粗门**（能抓结构性错误，抓不了 1px 级差异），阈值比 layering 的松一个量级。
 */

import type { MeshScene } from "../../src/spatial-scene/meshing/index.ts"

export interface RasterResult {
  readonly width: number
  readonly height: number
  /** 预乘线性 RGB。 */
  readonly rgb: Float32Array
  /** 累积 α。 */
  readonly alpha: Float32Array
  /** 未归一化累积深度 `Σ T·α·z`。 */
  readonly accumulatedDepth: Float32Array
  /** 期望深度（相机空间 z 的 α 加权），`α == 0` 处为 0。 */
  readonly depth: Float32Array
  readonly trianglesDrawn: number
}

/** CPU 光栅化：远 -> 近逐层画（z-buffer + over 混合）。 */
export function rasterizeMeshScene(scene: MeshScene): RasterResult {
  const { width, height } = scene.view
  const pixels = width * height
  const fx = scene.view.focalLengthPx
  const cx = width / 2
  const cy = height / 2
  const view = scene.camera.viewMatrix

  const rgb = new Float32Array(pixels * 3)
  const alpha = new Float32Array(pixels)
  const depthWeighted = new Float32Array(pixels)
  const zbuf = new Float32Array(pixels).fill(Number.POSITIVE_INFINITY)
  let trianglesDrawn = 0

  // 远 -> 近（层索引 0 = 最近）。
  for (let k = scene.layers.length - 1; k >= 0; k--) {
    const layer = scene.layers[k]
    const { texture, indices, positions, uvs } = layer
    const triCount = layer.triangleCount
    // 顶点投影缓存（每个三角形用 3 个）。
    const sx = new Float64Array(3)
    const sy = new Float64Array(3)
    const sz = new Float64Array(3)
    const su = new Float64Array(3)
    const sv = new Float64Array(3)

    for (let t = 0; t < triCount; t++) {
      let ok = true
      for (let c = 0; c < 3; c++) {
        const v = indices[t * 3 + c]
        const px = positions[v * 3]
        const py = positions[v * 3 + 1]
        const pz = positions[v * 3 + 2]
        const xCam = view[0] * px + view[4] * py + view[8] * pz + view[12]
        const yCam = view[1] * px + view[5] * py + view[9] * pz + view[13]
        const zCam = view[2] * px + view[6] * py + view[10] * pz + view[14]
        if (!(zCam > 1e-6)) {
          ok = false
          break
        }
        sx[c] = (xCam / zCam) * fx + cx
        sy[c] = (yCam / zCam) * fx + cy
        sz[c] = zCam
        su[c] = uvs[v * 2]
        sv[c] = uvs[v * 2 + 1]
      }
      if (!ok) continue

      const area =
        (sx[1] - sx[0]) * (sy[2] - sy[0]) - (sx[2] - sx[0]) * (sy[1] - sy[0])
      if (Math.abs(area) < 1e-9) continue
      const invArea = 1 / area

      const minX = Math.max(0, Math.floor(Math.min(sx[0], sx[1], sx[2])))
      const maxX = Math.min(width - 1, Math.ceil(Math.max(sx[0], sx[1], sx[2])))
      const minY = Math.max(0, Math.floor(Math.min(sy[0], sy[1], sy[2])))
      const maxY = Math.min(
        height - 1,
        Math.ceil(Math.max(sy[0], sy[1], sy[2])),
      )
      if (minX > maxX || minY > maxY) continue
      trianglesDrawn++

      for (let y = minY; y <= maxY; y++) {
        for (let x = minX; x <= maxX; x++) {
          const pxc = x + 0.5
          const pyc = y + 0.5
          const w0 =
            ((sx[1] - pxc) * (sy[2] - pyc) - (sx[2] - pxc) * (sy[1] - pyc)) *
            invArea
          const w1 =
            ((sx[2] - pxc) * (sy[0] - pyc) - (sx[0] - pxc) * (sy[2] - pyc)) *
            invArea
          const w2 = 1 - w0 - w1
          const eps = -1e-6
          if (w0 < eps || w1 < eps || w2 < eps) continue

          const invZ = w0 / sz[0] + w1 / sz[1] + w2 / sz[2]
          if (!(invZ > 0)) continue
          const z = 1 / invZ
          const i = y * width + x
          if (z >= zbuf[i]) continue

          // 透视校正 UV
          const u =
            ((w0 * su[0]) / sz[0] +
              (w1 * su[1]) / sz[1] +
              (w2 * su[2]) / sz[2]) *
            z
          const v =
            ((w0 * sv[0]) / sz[0] +
              (w1 * sv[1]) / sz[1] +
              (w2 * sv[2]) / sz[2]) *
            z
          const tx = Math.min(
            texture.width - 1,
            Math.max(0, Math.round(u * texture.width - 0.5)),
          )
          const ty = Math.min(
            texture.height - 1,
            Math.max(0, Math.round(v * texture.height - 0.5)),
          )
          const ti = ty * texture.width + tx
          const a = clamp01(texture.alpha[ti])
          const keep = 1 - a
          const premult = a
          rgb[i * 3] = keep * rgb[i * 3] + premult * texture.rgb[ti * 3]
          rgb[i * 3 + 1] =
            keep * rgb[i * 3 + 1] + premult * texture.rgb[ti * 3 + 1]
          rgb[i * 3 + 2] =
            keep * rgb[i * 3 + 2] + premult * texture.rgb[ti * 3 + 2]
          depthWeighted[i] = keep * depthWeighted[i] + a * z
          alpha[i] = keep * alpha[i] + a
          zbuf[i] = z
        }
      }
    }
  }

  const depth = new Float32Array(pixels)
  for (let i = 0; i < pixels; i++) {
    depth[i] = alpha[i] > 0 ? depthWeighted[i] / alpha[i] : 0
  }
  return {
    width,
    height,
    rgb,
    alpha,
    accumulatedDepth: depthWeighted,
    depth,
    trianglesDrawn,
  }
}

function clamp01(value: number): number {
  return value < 0 ? 0 : value > 1 ? 1 : value
}
