/**
 * CPU 三角光栅器：meshing 的**测试基准**（不进 `src/`）。
 *
 * 位置：node 侧验证工具。web 端交付渲染器是 **WebGL2 / three.js**，本文件**不是**交付路径，
 * 只是为了在 `scripts/meshing-golden.ts` 里把「层 RGBAD -> mesh -> 参考视角」闭环钉住。
 * （允许改成 WebGPU 光栅器，但那也只是测试基准，不是交付路径。）
 *
 * ── 它验证什么 ──
 * 「逐层 mesh + 该层的 RGBAD 纹理」在**参考视角**光栅化后，应当≈该层的 splat 渲染
 * （`WSplatFrame`）。这是 M1b 的核心门：几何没建错（支撑 / 撕裂 / 反投影 / 绕序），
 * 纹理坐标没接错。
 *
 * ── 约定（与 `meshing/types.ts` 一致，不可含糊）──
 * 1. 顶点 `positions` 是度量空间（米，OpenCV：x 右 / y 下 / z 前）。
 *    投影：`sx = fx·camX/camZ + cx`、`sy = fy·camY/camZ + cy`（y 向下，不翻转）。
 * 2. `uvs` 左上原点；纹理采样用**最近邻**：`tx = round(u·W − 0.5)`。
 *    顶点在像素中心 ⇒ 片元落在同一像素中心时 uv 恰好对回该 texel，所以参考视角下
 *    逐像素应当**近乎精确**相等（差异只来自透视校正与三角形边缘覆盖）。
 * 3. 深度测试：每像素保留**最近**的 `camZ`（heightfield 本不该自遮挡；测试用来暴露错误）。
 * 4. 输出 `rgb` 是**直通**（非预乘）线性值，与 `WSplatFrame.rgb` 同口径，便于直接比。
 *
 * ⚠ 参考视角的「精确」依赖一个前提：mesh 与该层帧用**同一台相机**（本链路成立）。
 */

import type { LayerMesh } from "../../src/spatial-scene/meshing/types.ts"
import type { WSplatCamera } from "../../src/spatial-scene/wsplat/camera.ts"

/** 层纹理 = 该层 RGBAD（直通线性 RGB + α）。 */
export interface MeshTexture {
  readonly width: number
  readonly height: number
  /** 直通线性 RGB，长度 `width*height*3`。 */
  readonly rgb: Float32Array
  /** α，长度 `width*height`。 */
  readonly alpha: Float32Array
}

/** 光栅化结果。 */
export interface MeshRasterResult {
  readonly width: number
  readonly height: number
  /** 直通线性 RGB，长度 `width*height*3`（未覆盖处为 0）。 */
  readonly rgb: Float32Array
  /** α，长度 `width*height`（未覆盖处为 0）。 */
  readonly alpha: Float32Array
  /** 最近表面深度（米）；未覆盖处为 0。 */
  readonly depth: Float32Array
  /** 1 = 该像素被至少一个三角形覆盖。 */
  readonly covered: Uint8Array
  /** 参与光栅化的三角形数（投影有效者）。 */
  readonly trianglesDrawn: number
}

export interface RasterOptions {
  /**
   * 是否做深度测试。默认 `true`。heightfield 不该自遮挡，关掉可暴露穿透错误。
   */
  depthTest?: boolean
}

/**
 * 把一层 mesh 用**该层自己的纹理**在参考视角光栅化。
 *
 * 纯 CPU、确定性；分辨率 = `camera.width × camera.height`（= 层帧尺寸）。
 */
export function rasterizeLayerMesh(
  mesh: LayerMesh,
  texture: MeshTexture,
  camera: WSplatCamera,
  options: RasterOptions = {},
): MeshRasterResult {
  const width = camera.width
  const height = camera.height
  // 纹理可以与相机同尺寸（层纹理），也可以是降采样的（背衬平面）：采样一律用纹理自己的尺寸。
  const texWidth = texture.width
  const texHeight = texture.height
  const pixels = width * height
  const outRgb = new Float32Array(pixels * 3)
  const outAlpha = new Float32Array(pixels)
  const depth = new Float32Array(pixels)
  const covered = new Uint8Array(pixels)
  const depthTest = options.depthTest ?? true

  const fx = width / (2 * Math.tan(camera.fovX / 2))
  const fy = height / (2 * Math.tan(camera.fovY / 2))
  const cx = width / 2
  const cy = height / 2
  const view = camera.viewMatrix
  const vertexCount = mesh.vertexCount

  // 顶点投影：预存屏幕坐标 + 透视校正要用的 invZ 与 u/z、v/z。
  const sx = new Float32Array(vertexCount)
  const sy = new Float32Array(vertexCount)
  const invZ = new Float32Array(vertexCount)
  const uOverZ = new Float32Array(vertexCount)
  const vOverZ = new Float32Array(vertexCount)
  const valid = new Uint8Array(vertexCount)
  for (let v = 0; v < vertexCount; v++) {
    const x = mesh.positions[v * 3]
    const y = mesh.positions[v * 3 + 1]
    const z = mesh.positions[v * 3 + 2]
    const camX = view[0] * x + view[4] * y + view[8] * z + view[12]
    const camY = view[1] * x + view[5] * y + view[9] * z + view[13]
    const camZ = view[2] * x + view[6] * y + view[10] * z + view[14]
    if (!(camZ > 1e-6)) continue
    sx[v] = (fx * camX) / camZ + cx
    sy[v] = (fy * camY) / camZ + cy
    const iz = 1 / camZ
    invZ[v] = iz
    uOverZ[v] = mesh.uvs[v * 2] * iz
    vOverZ[v] = mesh.uvs[v * 2 + 1] * iz
    valid[v] = 1
  }

  depth.fill(Number.POSITIVE_INFINITY)
  let trianglesDrawn = 0

  for (let t = 0; t < mesh.triangleCount; t++) {
    const a = mesh.indices[t * 3]
    const b = mesh.indices[t * 3 + 1]
    const c = mesh.indices[t * 3 + 2]
    if (!valid[a] || !valid[b] || !valid[c]) continue
    trianglesDrawn++

    const ax = sx[a]
    const ay = sy[a]
    const bx = sx[b]
    const by = sy[b]
    const cxs = sx[c]
    const cys = sy[c]

    const minX = Math.max(0, Math.floor(Math.min(ax, bx, cxs)))
    const maxX = Math.min(width - 1, Math.ceil(Math.max(ax, bx, cxs)))
    const minY = Math.max(0, Math.floor(Math.min(ay, by, cys)))
    const maxY = Math.min(height - 1, Math.ceil(Math.max(ay, by, cys)))
    if (minX > maxX || minY > maxY) continue

    // 边函数与面积（屏幕空间）。
    const e0 = (bx - ax) * (cys - ay) - (by - ay) * (cxs - ax)
    if (e0 === 0) continue

    for (let py = minY; py <= maxY; py++) {
      const pcy = py + 0.5
      for (let px = minX; px <= maxX; px++) {
        const pcx = px + 0.5
        // 三个边函数（顶点 a,b,c 对点 p）
        const w0 = (cxs - bx) * (pcy - by) - (cys - by) * (pcx - bx)
        const w1 = (ax - cxs) * (pcy - cys) - (ay - cys) * (pcx - cxs)
        const w2 = (bx - ax) * (pcy - ay) - (by - ay) * (pcx - ax)
        // ⚠ 必须带**亚像素容差**：每个支撑像素的**中心恰好是一个网格顶点**，所以
        // 它总是落在三角形的边上（边函数有一个 ~0）。严格 `>= 0` 会被浮点误差
        // 把其中约 2% 判成“在外”，表现为「used 顶点没被覆盖」的假漏画。
        // `sum = w0+w1+w2 = 2·Area` 与 p 无关，所以放宽容差不会引入除法问题。
        const tol = 1e-3
        const allPos = w0 >= -tol && w1 >= -tol && w2 >= -tol
        const allNeg = w0 <= tol && w1 <= tol && w2 <= tol
        if (!allPos && !allNeg) continue
        const sum = w0 + w1 + w2
        if (sum === 0) continue
        const l0 = w0 / sum
        const l1 = w1 / sum
        const l2 = w2 / sum

        const iz = l0 * invZ[a] + l1 * invZ[b] + l2 * invZ[c]
        if (!(iz > 0)) continue
        const z = 1 / iz
        const i = py * width + px
        if (depthTest && z >= depth[i]) continue

        const u = (l0 * uOverZ[a] + l1 * uOverZ[b] + l2 * uOverZ[c]) / iz
        const vv = (l0 * vOverZ[a] + l1 * vOverZ[b] + l2 * vOverZ[c]) / iz
        const tx = clampInt(Math.round(u * texWidth - 0.5), 0, texWidth - 1)
        const ty = clampInt(Math.round(vv * texHeight - 0.5), 0, texHeight - 1)
        const ti = ty * texWidth + tx
        outRgb[i * 3] = texture.rgb[ti * 3]
        outRgb[i * 3 + 1] = texture.rgb[ti * 3 + 1]
        outRgb[i * 3 + 2] = texture.rgb[ti * 3 + 2]
        outAlpha[i] = texture.alpha[ti]
        depth[i] = z
        covered[i] = 1
      }
    }
  }

  // 未覆盖像素的 depth 归 0（与 WSplatFrame 的「背景 depth=0」同口径）。
  for (let i = 0; i < pixels; i++) if (!covered[i]) depth[i] = 0

  return {
    width,
    height,
    rgb: outRgb,
    alpha: outAlpha,
    depth,
    covered,
    trianglesDrawn,
  }
}

function clampInt(value: number, lo: number, hi: number): number {
  return value < lo ? lo : value > hi ? hi : value
}
