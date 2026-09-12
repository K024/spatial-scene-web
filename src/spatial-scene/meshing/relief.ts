/**
 * relief 网格化：把一层 RGBAD（heightfield）变成顶点 + 表面三角形 + 裙边断壁。
 *
 * 阶段：meshing 的第三步（支撑掩码 + 余量 + 撕裂边表之后）。
 *
 * ── 几何为什么是 heightfield ──
 * 每层都是**渲出来的 2D 帧**：每像素恰好一个 `depth`，所以几何恒为 `z(u,v)`。
 * 这带来一个强简化：**一个支撑像素一个顶点**，四边形天然共享顶点，撕裂只是**不发射**那个
 * 四边形。不需要 LDI 图、浮动岛、悬空边那套「每像素多深度节点」的机制。
 *
 * ── 顶点 = 同内参反投影（+ 余量源映射）──
 * `camera.ts` 已写死：mesh 是 LDI / relief mesh，顶点 = 该像素按**同一透视内参**反投影的
 * `(u, v, z)`。这里用列主序 `viewMatrix` 的轴分解还原相机轴，避免再求一次 4x4 逆：
 * ```
 * right = (view[0], view[4], view[8])   down = (view[1], view[5], view[9])
 * fwd   = (view[2], view[6], view[10])  C    = camera.position
 * world = C + right·camX + down·camY + fwd·z
 * camX  = (u − cx)/fx · z,  camY = (v − cy)/fy · z,  u = x+0.5, v = y+0.5
 * ```
 * `fx = width / (2·tan(fovX/2))`（相机只支持居中主点，`cx=W/2, cy=H/2`）。
 *
 * ⚠ 余量环（见 `margin.ts`）的顶点：**位置**在环像素自己的 `(u,v)`，但 **z 与 uv 取自
 * `context.source[i]`**（最近支撑像素）。这样几何在屏幕平面向外铺开、纹理拉伸边缘，
 * 不会把背景色拉出来。
 *
 * ── 出四边形（含撕裂与对角翻转）──
 * 四边形 `(x,y)-(x+1,y)-(x,y+1)-(x+1,y+1)` 只在**四角都在支撑内**且**四条边都没被撕开**
 * 时发射，出 2 个三角形。两种剖分（对角线 `00–11` 或 `10–01`）按 `allowDiagonalFlip`
 * 选**三维对角线更短**的那种，减少斜视角下的长条三角形。
 *
 * ── 裙边 / 断壁（M2）──
 * 表面发射完之后，找出**网格边界边**（只被 1 个已发射四边形使用）：撕裂边、支撑外缘、
 * 图像外框都会产生它们。对每条边界边，把两个端点各自复制一份、沿视线向**后**挤出
 * （视差 `+skirtDisparity`），再用 2 个三角形缝成一道墙。绕序取该边在**表面三角形**里的
 * 有向形式 `(a→b)` 的反向：`(b, a, sa)`、`(b, sa, sb)` —— 墙法线朝外，斜视角下挡住缝。
 *
 * @see SOLIDI `mesh.py:create_mesh`（像素网格 + `extrapolation_thickness`）、
 *   `mesh.py:generate_face`（四邻域配对出 2 三角、对角方向）、
 *   `mesh.py:tear_edges`（断边）—— vt-vl-lab/3d-photo-inpainting @ de04467 (MIT)。
 *   本实现**不移植**其 LDI 图机制与修补网络：那些解决的是「每像素多深度 + 需发现边 +
 *   给网络闭合 mask」，在本链路（每层是渲出来的 heightfield）前提不成立。
 */

import { zFromNdcDepth } from "../layering/disparity-stats.ts"
import type { WSplatCamera } from "../wsplat/camera.ts"
import type { WSplatFrame } from "../wsplat/types.ts"
import type { TornEdges } from "./tears.ts"
import type {
  LayerMesh,
  LayerReliefContext,
  ReliefOptions,
  SkirtOptions,
} from "./types.ts"

/** 只借几何与纹理要用的字段，便于单测造最小对象。 */
export type ReliefFrame = Pick<
  WSplatFrame,
  "rgb" | "alpha" | "visible" | "depth" | "width" | "height"
>

/** `buildLayerRelief` 的诊断计数。 */
export interface LayerReliefStats {
  readonly vertexCount: number
  readonly triangleCount: number
  readonly wallTriangleCount: number
  readonly opaqueTriangleCount: number
  readonly quadsEmitted: number
  readonly quadsSkipped: number
  /** [M2] 网格边界边数（= 裙边墙的「基线」数）。 */
  readonly boundaryEdges: number
}

/** 把一个像素坐标 + 深度反投影进 `out[v]`（世界坐标，OpenCV 相机约定）。 */
type PlaceVertex = (
  out: Float32Array,
  v: number,
  x: number,
  y: number,
  z: number,
) => void

const DEFAULT_OPAQUE_ALPHA = 0.99

/**
 * 建一层的 relief 网格。
 *
 * @param disparityRange 该层视差带 `[lo, hi]`（只写进元数据，不参与几何）。
 * @param depthRange 该层真实深度带 `[lo, hi]` 米（只写进元数据）。
 * @param context `source`（余量源映射）、`disparity`、`near`/`far`、`skirtDisparity`。
 */
export function buildLayerRelief(
  frame: ReliefFrame,
  camera: WSplatCamera,
  support: Uint8Array,
  torn: TornEdges,
  layerIndex: number,
  disparityRange: readonly [number, number],
  depthRange: readonly [number, number],
  options: ReliefOptions = {},
  context: LayerReliefContext = {},
): { mesh: LayerMesh; stats: LayerReliefStats } {
  const { width, height, alpha, depth } = frame
  if (camera.width !== width || camera.height !== height) {
    throw new Error(
      `relief：相机尺寸 ${camera.width}x${camera.height} 与帧 ${width}x${height} 不一致`,
    )
  }
  const pixels = width * height
  const allowDiagonalFlip = options.allowDiagonalFlip ?? true
  const opaqueAlpha = options.opaqueAlpha ?? DEFAULT_OPAQUE_ALPHA
  const source = context.source

  const fx = width / (2 * Math.tan(camera.fovX / 2))
  const fy = height / (2 * Math.tan(camera.fovY / 2))
  const cx = width / 2
  const cy = height / 2
  const view = camera.viewMatrix
  const right: [number, number, number] = [view[0], view[4], view[8]]
  const down: [number, number, number] = [view[1], view[5], view[9]]
  const fwd: [number, number, number] = [view[2], view[6], view[10]]
  const origin = camera.position

  const sourceOf = (i: number): number => (source ? source[i] : i)
  const place: PlaceVertex = (out, v, x, y, z) => {
    const camX = ((x + 0.5 - cx) / fx) * z
    const camY = ((y + 0.5 - cy) / fy) * z
    out[v * 3] = origin[0] + right[0] * camX + down[0] * camY + fwd[0] * z
    out[v * 3 + 1] = origin[1] + right[1] * camX + down[1] * camY + fwd[1] * z
    out[v * 3 + 2] = origin[2] + right[2] * camX + down[2] * camY + fwd[2] * z
  }

  // ── 顶点：一个掩码像素一个顶点 ──
  const pixelToVertex = new Int32Array(pixels).fill(-1)
  const vertexPixel: number[] = []
  let vertexCount = 0
  for (let i = 0; i < pixels; i++) {
    if (support[i] && depth[sourceOf(i)] > 0) {
      pixelToVertex[i] = vertexCount++
      vertexPixel.push(i)
    }
  }
  // 注：显式用泛型自由形式标注，避免 TS 7 把 `new Float32Array()` 推成 `<ArrayBuffer>`
  // 后无法接 `appendSkirt` 返回的 `<ArrayBufferLike>`。
  let positions: Float32Array = new Float32Array(vertexCount * 3)
  let uvs: Float32Array = new Float32Array(vertexCount * 2)
  for (let v = 0; v < vertexCount; v++) {
    const i = vertexPixel[v]
    const src = sourceOf(i)
    const z = depth[src]
    const x = i % width
    const y = (i - x) / width
    place(positions, v, x, y, z)
    const sx = src % width
    const sy = (src - sx) / width
    uvs[v * 2] = (sx + 0.5) / width
    uvs[v * 2 + 1] = (sy + 0.5) / height
  }

  // ── 四边形：先判定发射，再精确分配索引缓冲 ──
  const quadsX = width - 1
  const emitted = new Uint8Array(Math.max(0, quadsX * (height - 1)))
  let quadsEmitted = 0
  let quadsSkipped = 0
  for (let y = 0; y + 1 < height; y++) {
    const hRow = y * quadsX
    const hRowBelow = (y + 1) * quadsX
    const vRow = y * width
    for (let x = 0; x + 1 < width; x++) {
      const i00 = y * width + x
      const i10 = i00 + 1
      const i01 = i00 + width
      const i11 = i01 + 1
      const ok =
        pixelToVertex[i00] >= 0 &&
        pixelToVertex[i10] >= 0 &&
        pixelToVertex[i01] >= 0 &&
        pixelToVertex[i11] >= 0 &&
        torn.horizontal[hRow + x] === 0 &&
        torn.horizontal[hRowBelow + x] === 0 &&
        torn.vertical[vRow + x] === 0 &&
        torn.vertical[vRow + x + 1] === 0
      if (ok) {
        emitted[y * quadsX + x] = 1
        quadsEmitted++
      } else {
        quadsSkipped++
      }
    }
  }

  const surfaceIndexCount = quadsEmitted * 6
  const surfaceIndices = new Uint32Array(surfaceIndexCount)
  let opaqueTriangleCount = 0
  let cursor = 0
  for (let y = 0; y + 1 < height; y++) {
    for (let x = 0; x + 1 < width; x++) {
      if (emitted[y * quadsX + x] === 0) continue
      const i00 = y * width + x
      const i10 = i00 + 1
      const i01 = i00 + width
      const i11 = i01 + 1
      const v00 = pixelToVertex[i00]
      const v10 = pixelToVertex[i10]
      const v01 = pixelToVertex[i01]
      const v11 = pixelToVertex[i11]

      let useDiagonal10To01 = false
      if (allowDiagonalFlip) {
        useDiagonal10To01 =
          squaredDistance(positions, v10, v01) <
          squaredDistance(positions, v00, v11)
      }

      if (useDiagonal10To01) {
        surfaceIndices[cursor++] = v00
        surfaceIndices[cursor++] = v01
        surfaceIndices[cursor++] = v10
        surfaceIndices[cursor++] = v10
        surfaceIndices[cursor++] = v01
        surfaceIndices[cursor++] = v11
      } else {
        surfaceIndices[cursor++] = v00
        surfaceIndices[cursor++] = v11
        surfaceIndices[cursor++] = v10
        surfaceIndices[cursor++] = v00
        surfaceIndices[cursor++] = v01
        surfaceIndices[cursor++] = v11
      }

      if (
        alpha[i00] >= opaqueAlpha &&
        alpha[i10] >= opaqueAlpha &&
        alpha[i01] >= opaqueAlpha &&
        alpha[i11] >= opaqueAlpha
      ) {
        opaqueTriangleCount += 2
      }
    }
  }

  // ── 裙边 / 断壁（M2）──
  const skirtN = context.skirtDisparity ?? 0
  const disparity = context.disparity
  const near = context.near
  const far = context.far
  let wallTriangleCount = 0
  let boundaryEdges = 0
  let indices: Uint32Array = surfaceIndices

  if (skirtN > 0 && disparity && near !== undefined && far !== undefined) {
    const built = appendSkirt({
      surfaceIndices,
      emitted,
      pixelToVertex,
      vertexPixel,
      positions,
      uvs,
      disparity,
      skirtN,
      near,
      far,
      width,
      height,
      place,
    })
    indices = built.indices
    positions = built.positions
    uvs = built.uvs
    wallTriangleCount = built.wallTriangleCount
    boundaryEdges = built.boundaryEdges
  }

  const finalVertexCount = positions.length / 3
  const mesh: LayerMesh = {
    layerIndex,
    width,
    height,
    vertexCount: finalVertexCount,
    triangleCount: indices.length / 3,
    wallTriangleCount,
    opaqueTriangleCount,
    positions,
    uvs,
    indices,
    // 纹理**引用**输入帧的 rgb/alpha（不拷贝，见 types.ts 的内存复用约定）。
    texture: { width, height, rgb: frame.rgb, alpha: frame.alpha },
    disparityRange,
    depthRange,
  }
  return {
    mesh,
    stats: {
      vertexCount: finalVertexCount,
      triangleCount: mesh.triangleCount,
      wallTriangleCount,
      opaqueTriangleCount,
      quadsEmitted,
      quadsSkipped,
      boundaryEdges,
    },
  }
}

/** `appendSkirt` 的输入。 */
interface SkirtInput {
  readonly surfaceIndices: Uint32Array
  readonly emitted: Uint8Array
  readonly pixelToVertex: Int32Array
  readonly vertexPixel: readonly number[]
  readonly positions: Float32Array
  readonly uvs: Float32Array
  readonly disparity: ArrayLike<number>
  readonly skirtN: number
  readonly near: number
  readonly far: number
  readonly width: number
  readonly height: number
  readonly place: PlaceVertex
}

/**
 * 找出网格边界边并缝出裙边。返回扩容后的 indices / positions / uvs。
 *
 * 边界边 = 只被 **1** 个已发射四边形使用的网格边（0 = 两侧都没面；2 = 内部边）。
 * 有向形式取自使用它的**表面三角形**；墙按反向绕序 `(b, a, sa)`、`(b, sa, sb)` 缝。
 */
function appendSkirt(input: SkirtInput): {
  indices: Uint32Array
  positions: Float32Array
  uvs: Float32Array
  wallTriangleCount: number
  boundaryEdges: number
} {
  const {
    surfaceIndices,
    emitted,
    pixelToVertex,
    vertexPixel,
    disparity,
    skirtN,
    near,
    far,
    width,
    height,
    place,
  } = input
  let positions = input.positions
  let uvs = input.uvs
  let vertexCount = positions.length / 3

  const quadsX = width - 1
  const hCount = height * quadsX
  const edgeCount = hCount + Math.max(0, height - 1) * width
  const hId = (x: number, y: number): number => y * quadsX + x
  const vId = (x: number, y: number): number => hCount + y * width + x

  // 边界边：恰好被 1 个已发射四边形使用。
  const boundary = new Uint8Array(edgeCount)
  let boundaryEdges = 0
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (x + 1 < width) {
        const below = y < height - 1 ? emitted[y * quadsX + x] : 0
        const above = y > 0 ? emitted[(y - 1) * quadsX + x] : 0
        if (below + above === 1) {
          boundary[hId(x, y)] = 1
          boundaryEdges++
        }
      }
      if (y + 1 < height) {
        const right = x < width - 1 ? emitted[y * quadsX + x] : 0
        const left = x > 0 ? emitted[y * quadsX + (x - 1)] : 0
        if (right + left === 1) {
          boundary[vId(x, y)] = 1
          boundaryEdges++
        }
      }
    }
  }

  // 每条边界边取它在**表面三角形**里的有向形式 (a→b)。
  const dirA = new Int32Array(edgeCount).fill(-1)
  const dirB = new Int32Array(edgeCount).fill(-1)
  for (let t = 0; t < surfaceIndices.length / 3; t++) {
    const tri = [
      surfaceIndices[t * 3],
      surfaceIndices[t * 3 + 1],
      surfaceIndices[t * 3 + 2],
    ]
    for (let k = 0; k < 3; k++) {
      const a = tri[k]
      const b = tri[(k + 1) % 3]
      const pa = vertexPixel[a]
      const pb = vertexPixel[b]
      const ax = pa % width
      const ay = (pa - ax) / width
      const bx = pb % width
      const by = (pb - bx) / width
      let id = -1
      if (ay === by && Math.abs(ax - bx) === 1) {
        id = hId(Math.min(ax, bx), ay)
      } else if (ax === bx && Math.abs(ay - by) === 1) {
        id = vId(ax, Math.min(ay, by))
      }
      if (id >= 0 && boundary[id] === 1 && dirA[id] < 0) {
        dirA[id] = a
        dirB[id] = b
      }
    }
  }

  // skirt 顶点：每个边界端点像素一个（懒建），沿视线向后挤。
  const skirtVertex = new Int32Array(width * height).fill(-1)
  const ensureCapacity = (extra: number): void => {
    const needV = vertexCount + extra
    if (positions.length >= needV * 3 && uvs.length >= needV * 2) return
    const grownPos = new Float32Array(Math.max(needV * 3, positions.length * 2))
    grownPos.set(positions)
    positions = grownPos
    const grownUv = new Float32Array(Math.max(needV * 2, uvs.length * 2))
    grownUv.set(uvs)
    uvs = grownUv
  }
  const skirtOf = (pixel: number): number => {
    const existing = skirtVertex[pixel]
    if (existing >= 0) return existing
    ensureCapacity(1)
    const v = vertexCount++
    skirtVertex[pixel] = v
    const x = pixel % width
    const y = (pixel - x) / width
    const n = Math.min(1, disparity[pixel] + skirtN)
    const z = zFromNdcDepth(n, near, far)
    place(positions, v, x, y, z)
    const surfaceV = pixelToVertex[pixel]
    uvs[v * 2] = uvs[surfaceV * 2]
    uvs[v * 2 + 1] = uvs[surfaceV * 2 + 1]
    return v
  }

  const walls: number[] = []
  for (let id = 0; id < edgeCount; id++) {
    if (boundary[id] === 1 && dirA[id] >= 0) {
      const a = dirA[id]
      const b = dirB[id]
      const sa = skirtOf(vertexPixel[a])
      const sb = skirtOf(vertexPixel[b])
      walls.push(b, a, sa, b, sa, sb)
    }
  }

  const indices = new Uint32Array(surfaceIndices.length + walls.length)
  indices.set(surfaceIndices)
  for (let i = 0; i < walls.length; i++) {
    indices[surfaceIndices.length + i] = walls[i]
  }
  return {
    indices,
    positions: positions.slice(0, vertexCount * 3),
    uvs: uvs.slice(0, vertexCount * 2),
    wallTriangleCount: walls.length / 3,
    boundaryEdges,
  }
}

/** 两个顶点之间的**三维**平方距离（对角翻转的判据）。 */
function squaredDistance(
  positions: Float32Array,
  a: number,
  b: number,
): number {
  const dx = positions[a * 3] - positions[b * 3]
  const dy = positions[a * 3 + 1] - positions[b * 3 + 1]
  const dz = positions[a * 3 + 2] - positions[b * 3 + 2]
  return dx * dx + dy * dy + dz * dz
}

/**
 * 解析裙边宽度（视差域）：`clamp(scale·bandWidth, min, max)`。
 *
 * 与撕裂阈值同一域、同一尺度：层带本身就是视差带，所以「该挤多宽」以 band 宽度为单位。
 */
export function resolveSkirtDisparity(
  bandWidth: number,
  options: SkirtOptions = {},
): number {
  const scale = options.scale ?? 0.5
  const min = options.minDisparity ?? 0.002
  const max = options.maxDisparity ?? 0.04
  const lo = Math.min(min, max)
  const hi = Math.max(min, max)
  const value = scale * Math.max(0, bandWidth)
  return value < lo ? lo : value > hi ? hi : value
}
