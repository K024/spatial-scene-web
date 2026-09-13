/**
 * LOD relief 网格化：**受限四叉树**自适应出面（替代逐像素出面）。
 *
 * 阶段：meshing 的出面步骤（`relief.ts` 的自适应替代）。
 *
 * ── 为什么 ──
 * 逐像素出面（每支撑像素一个顶点、每个四角齐全的 2×2 出 2 三角）在 768×576×L 下
 * 轻松到百万面量级，直接渲染 3DGS 都更划算。分层后的每层是**平滑 heightfield**
 * （同像素多深度已被 layering 的 α 加权坍缩），所以大片区域可以用大四边形表示。
 *
 * 本模块把「出面」换成**受限四叉树**（Lindstrom 1996 / ROAM / Pajarola 2002 的
 * raster heightmap 路线），并参考 Facebook *One Shot 3D Photography* §4.4.2 的结论：
 * **深度断层交给边界（我们这里是 support 边界 + `tears.ts`），内部平滑区均匀/误差驱动
 * 粗化即可，不需要更复杂的自适应采样**。
 *
 * ── 硬约束（与 `relief.ts` 完全一致，只是格点变粗）──
 * - **只在 support 内出几何**：叶子必须整块被支撑（min cell 处退化为四角支撑）；
 * - **撕裂保持**：叶子内（含边界）出现 `tears` 边则继续细分；min cell 处直接丢弃，
 *   于是断层处留出约一个 min cell 的缝（和逐像素版留 1px 缝同语义）；
 * - **误差有界**：叶子的 4 角视差做双线性，与内部逐点视差之差 ≤ `maxError`。
 *   视差域是**正确**的度量：`Δscreen = f·t·Δn·(far−near)/(far·near)`，
 *   即 `Δn` 到新视角屏幕位移是全局常数比例，与深度无关。
 *
 * ── 无裂缝（2:1 受限 + 边中点缝合）──
 * 相邻叶子层级差 > 1 会造成 T-junction。构建后做**迭代松弛**：凡有「小 2 级以上的
 * 邻居」的叶子就强制细分，直到稳定；随后每条边按「邻居是否更细」决定是否插入中点，
 * 叶子按其（含中点的）凸多边形**从角点扇形三角化**，共享边的中点顶点被两侧复用 ⇒ 无缝。
 *
 * ── 输出 ──
 * 顶点：四叉树格点（像素坐标处的反投影点）；UV = 该像素在**整层全分辨率纹理**里的
 * 归一化坐标 ⇒ 几何粗化但纹理仍是全分辨率采样（几何/纹理解耦，这是压面数不掉画质的
 * 关键）。索引：表面三角形（无裙边；裙边属视角兜缝，按 meshing 职责边界不在此实现）。
 *
 * ── 边界策略：不剪裁，靠纹理 α 掩边（默认 `snapBoundary`）──
 * 只要 cell 内有支撑就出面，无支撑角点用最近支撑像素的深度、UV 用自己的像素。
 * 所以边界可以粗到 `minCellPx`（甚至 8）而不露底：多出来的那圈几何采样到低 α 纹素，
 * 被逐 texel α 混合自然掩掉，剪影由**全分辨率纹理**决定。粗细由 `minCellPx` 一个旋钮
 * 控制（面数 ∝ 1/minCell²），不做 marching-squares 边界裁剪。
 */

import type { WSplatCamera } from "../wsplat/camera.ts"
import type { TornEdges } from "./tears.ts"

/** LOD 出面选项。 */
export interface LodOptions {
  /**
   * 最小 cell（像素，2 的幂）。默认 `2`。
   * 这是面数的**主要下界旋钮**：support 边界与撕裂只能粗到这一级，
   * 所以轮廓会被量化到 `minCellPx` 像素（Facebook 用 Douglas-Peucker 容差做同一件事）。
   */
  minCellPx?: number
  /** 最大 cell（像素，2 的幂）。默认 `128`。防止整片天空并成一个巨大四边形导致误差累积。 */
  maxCellPx?: number
  /**
   * 视差域最大误差（绝对 NDC 视差）。默认 `0.002`。
   * `0.002` 在「幅宽 50% 平移」下约等于 1px 屏幕误差，与撕裂阈值（band 的 0.4，
   * clamp 到 0.06）相比小一个量级，即几何误差远低于结构性深度台阶。
   */
  maxError?: number
  /**
   * 边界贴合（默认 `true`）。min cell 处，**只要 cell 内有支撑**就出面；没有支撑的
   * 角点取**本 cell 内最近支撑像素的深度**做位置，但 **UV 仍用自己的像素坐标**。
   *
   * 为什么安全：扩出去的那圈几何采样到的是原图的**低 α 纹素**（无支撑 ⇒ `α < aLo`），
   * 分层渲染是逐 texel α 混合，所以那圈几乎不可见；剪影由**全分辨率纹理的 α**决定，
   * 不由几何决定 ⇒ 可以把 `minCellPx` 开大来减面，而不是细扣边界（boundary clipping）。
   *
   * 关掉它则要求四角全支撑，粗 `minCell` 会在剪影处整块丢 cell（漏覆盖 / 露底）。
   */
  snapBoundary?: boolean
}

/** `buildLayerReliefLod` 的逐像素输入（与 `relief.ts` 对齐）。 */
export interface LodFrame {
  readonly width: number
  readonly height: number
  readonly depth: Float32Array
}

/** LOD 出面诊断。 */
export interface LodStats {
  readonly leaves: number
  readonly vertexCount: number
  readonly triangleCount: number
  /** 实际使用的层级直方图：`levels[k]` = cell 边长 `minCell*2^k` 的叶子数。 */
  readonly levelHistogram: readonly number[]
  /** 因 2:1 松弛而额外细分的次数。 */
  readonly balanceSplits: number
}

export interface LodMeshResult {
  readonly positions: Float32Array
  readonly uvs: Float32Array
  readonly indices: Uint32Array
  readonly stats: LodStats
}

interface Node {
  x0: number
  y0: number
  s: number
  children: Node[] | null
  parent: Node | null
}

const DEFAULT_MIN_CELL = 4
const DEFAULT_MAX_CELL = 128
const DEFAULT_MAX_ERROR = 0.005

/** 提点：`positions` / `uvs` / 索引用可增长数组，最后定型。 */
class VertexPool {
  readonly positions: number[] = []
  readonly uvs: number[] = []
  private readonly index = new Map<number, number>()
  private readonly width: number
  private readonly height: number
  private readonly place: (
    x: number,
    y: number,
    z: number,
    out: [number, number, number],
  ) => boolean
  private readonly depth: ArrayLike<number>
  private readonly support: Uint8Array
  /** 是否允许把无支撑角点贴到本 cell 内最近支撑像素（`snapBoundary`）。 */
  private readonly snap: boolean
  /** 贴合搜索半径（= minCell）。 */
  private readonly snapRadius: number

  constructor(
    width: number,
    height: number,
    place: (
      x: number,
      y: number,
      z: number,
      out: [number, number, number],
    ) => boolean,
    depth: ArrayLike<number>,
    support: Uint8Array,
    snap: boolean,
    snapRadius: number,
  ) {
    this.width = width
    this.height = height
    this.place = place
    this.depth = depth
    this.support = support
    this.snap = snap
    this.snapRadius = snapRadius
  }

  /** 取（或建）像素 `(x,y)` 处的顶点；无支撑时按 `snap` 贴最近支撑像素。 */
  get(x: number, y: number): number {
    const key = y * this.width + x
    const existing = this.index.get(key)
    if (existing !== undefined) return existing
    let src = key
    if (!(this.support[key] && this.depth[key] > 0)) {
      if (!this.snap) return -1
      const found = this.nearestSupported(x, y)
      if (found < 0) return -1
      src = found
    }
    const z = this.depth[src]
    if (!(z > 0)) return -1
    const p: [number, number, number] = [0, 0, 0]
    if (!this.place(x, y, z, p)) return -1
    const id = this.positions.length / 3
    this.positions.push(p[0], p[1], p[2])
    // UV 永远取**顶点自己的像素**（不是源像素）：无支撑处纹理 α 本来就低，
    // 扩出来的几何被 alpha 自动掩掉，剪影由全分辨率纹理决定。
    this.uvs.push((x + 0.5) / this.width, (y + 0.5) / this.height)
    this.index.set(key, id)
    return id
  }

  /** 在半径内找最近的支撑像素（Chebyshev，扫描半径环）。 */
  private nearestSupported(x: number, y: number): number {
    for (let r = 1; r <= this.snapRadius; r++) {
      let best = -1
      let bestD = Number.POSITIVE_INFINITY
      for (let dy = -r; dy <= r; dy++) {
        for (let dx = -r; dx <= r; dx++) {
          if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue
          const nx = x + dx
          const ny = y + dy
          if (nx < 0 || ny < 0 || nx >= this.width || ny >= this.height)
            continue
          const k = ny * this.width + nx
          if (!this.support[k] || !(this.depth[k] > 0)) continue
          const d = dx * dx + dy * dy
          if (d < bestD) {
            bestD = d
            best = k
          }
        }
      }
      if (best >= 0) return best
    }
    return -1
  }
}

/**
 * 建一层的 LOD relief 网格。
 *
 * @param support 支撑掩码（1 = 出几何）。
 * @param disparity 逐像素视差（`computeDisparityField` 的结果，粗化误差用）。
 * @param torn 撕裂边表。
 */
export function buildLayerReliefLod(
  frame: LodFrame,
  camera: WSplatCamera,
  support: Uint8Array,
  disparity: ArrayLike<number>,
  torn: TornEdges,
  options: LodOptions = {},
): LodMeshResult {
  const { width, height, depth } = frame
  if (camera.width !== width || camera.height !== height) {
    throw new Error(
      `lod：相机尺寸 ${camera.width}x${camera.height} 与帧 ${width}x${height} 不一致`,
    )
  }
  const minCell = clampPow2(options.minCellPx ?? DEFAULT_MIN_CELL, 1, 1 << 12)
  const maxCell = clampPow2(
    options.maxCellPx ?? DEFAULT_MAX_CELL,
    minCell,
    1 << 12,
  )
  const maxError = Math.max(0, options.maxError ?? DEFAULT_MAX_ERROR)
  const snapBoundary = options.snapBoundary ?? true

  // ── 支撑前缀和（顶点网格）──
  const ps = new Int32Array((width + 1) * (height + 1))
  for (let y = 0; y < height; y++) {
    let row = 0
    for (let x = 0; x < width; x++) {
      row += support[y * width + x]
      ps[(y + 1) * (width + 1) + (x + 1)] = ps[y * (width + 1) + (x + 1)] + row
    }
  }
  const sumSupport = (
    x0: number,
    y0: number,
    x1: number,
    y1: number,
  ): number => {
    const ax = x0
    const ay = y0
    const bx = x1
    const by = y1
    return (
      ps[(by + 1) * (width + 1) + (bx + 1)] -
      ps[ay * (width + 1) + (bx + 1)] -
      ps[(by + 1) * (width + 1) + ax] +
      ps[ay * (width + 1) + ax]
    )
  }

  /** 区域内是否有撕裂边（含边界）。 */
  const hasTear = (x0: number, y0: number, x1: number, y1: number): boolean => {
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x < x1; x++) {
        if (torn.horizontal[y * (width - 1) + x]) return true
      }
    }
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x <= x1; x++) {
        if (torn.vertical[y * width + x]) return true
      }
    }
    return false
  }

  /** 区域是否可整块作为叶子：整块支撑、无撕裂、双线性视差拟合在误差内。 */
  const regionClean = (x0: number, y0: number, s: number): boolean => {
    const x1 = x0 + s
    const y1 = y0 + s
    if (x1 > width - 1 || y1 > height - 1) return false
    if (sumSupport(x0, y0, x1, y1) !== (s + 1) * (s + 1)) return false
    if (hasTear(x0, y0, x1, y1)) return false
    const n00 = disparity[y0 * width + x0]
    const n10 = disparity[y0 * width + x1]
    const n01 = disparity[y1 * width + x0]
    const n11 = disparity[y1 * width + x1]
    const inv = 1 / s
    for (let y = y0; y <= y1; y++) {
      const ty = (y - y0) * inv
      for (let x = x0; x <= x1; x++) {
        const tx = (x - x0) * inv
        const bil =
          n00 * (1 - tx) * (1 - ty) +
          n10 * tx * (1 - ty) +
          n01 * (1 - tx) * ty +
          n11 * tx * ty
        if (Math.abs(disparity[y * width + x] - bil) > maxError) return false
      }
    }
    return true
  }

  /** min cell 的判据：无撕裂 + （贴边界时）cell 内有支撑 / （不贴时）四角支撑。 */
  const minCellClean = (x0: number, y0: number, s: number): boolean => {
    const x1 = x0 + s
    const y1 = y0 + s
    if (x1 > width - 1 || y1 > height - 1) return false
    if (hasTear(x0, y0, x1, y1)) return false
    if (snapBoundary) return sumSupport(x0, y0, x1, y1) > 0
    return (
      (support[y0 * width + x0] &&
        support[y0 * width + x1] &&
        support[y1 * width + x0] &&
        support[y1 * width + x1]) !== 0
    )
  }

  const rootSize = (() => {
    let s = minCell
    while (s < Math.max(width, height)) s *= 2
    return s
  })()

  const build = (
    x0: number,
    y0: number,
    s: number,
    parent: Node | null,
  ): Node | null => {
    if (x0 > width - 1 || y0 > height - 1) return null
    const x1 = Math.min(x0 + s, width - 1)
    const y1 = Math.min(y0 + s, height - 1)
    if (x1 < x0 || y1 < y0) return null
    if (sumSupport(x0, y0, x1, y1) === 0) return null
    if (s <= maxCell && regionClean(x0, y0, s)) {
      return { x0, y0, s, children: null, parent }
    }
    if (s <= minCell) {
      return minCellClean(x0, y0, s)
        ? { x0, y0, s, children: null, parent }
        : null
    }
    const h = s >> 1
    const node: Node = { x0, y0, s, children: [], parent }
    const kids = [
      build(x0, y0, h, node),
      build(x0 + h, y0, h, node),
      build(x0, y0 + h, h, node),
      build(x0 + h, y0 + h, h, node),
    ].filter((n): n is Node => n !== null)
    if (kids.length === 0) return null
    node.children = kids
    return node
  }

  const root = build(0, 0, rootSize, null)
  if (!root) {
    return {
      positions: new Float32Array(),
      uvs: new Float32Array(),
      indices: new Uint32Array(),
      stats: {
        leaves: 0,
        vertexCount: 0,
        triangleCount: 0,
        levelHistogram: [],
        balanceSplits: 0,
      },
    }
  }

  // ── 2:1 受限松弛：有「小 2 级以上的邻居」的叶子强制细分 ──
  const findLeaf = (node: Node | null, x: number, y: number): Node | null => {
    if (!node) return null
    if (!node.children) return node
    const h = node.s >> 1
    const cx = x >= node.x0 + h ? 1 : 0
    const cy = y >= node.y0 + h ? 1 : 0
    for (const c of node.children) {
      if (c.x0 === node.x0 + cx * h && c.y0 === node.y0 + cy * h) {
        return findLeaf(c, x, y)
      }
    }
    return null
  }

  const collectLeaves = (node: Node, out: Node[]): void => {
    if (!node.children) {
      out.push(node)
      return
    }
    for (const c of node.children) collectLeaves(c, out)
  }

  /** 该叶子是否存在「小 2 级以上的邻居」（需要细分）。 */
  const hasThinNeighbor = (leaf: Node): boolean => {
    const { x0, y0, s } = leaf
    const h = s >> 1
    const probes: Array<[number, number]> = [
      [x0 + s, y0 + h],
      [x0 - 1, y0 + h],
      [x0 + h, y0 + s],
      [x0 + h, y0 - 1],
    ]
    for (const [px, py] of probes) {
      if (px < 0 || py < 0 || px >= width || py >= height) continue
      const n = findLeaf(root, px, py)
      if (n && n.s <= s / 4) return true
    }
    return false
  }

  let balanceSplits = 0
  for (let iter = 0; iter < 12; iter++) {
    const leaves: Node[] = []
    collectLeaves(root, leaves)
    const thin = leaves.filter(hasThinNeighbor)
    if (thin.length === 0) break
    for (const leaf of thin) {
      const parent = leaf.parent
      const h = leaf.s >> 1
      const node: Node = {
        x0: leaf.x0,
        y0: leaf.y0,
        s: leaf.s,
        children: [],
        parent,
      }
      const kids = [
        build(leaf.x0, leaf.y0, h, node),
        build(leaf.x0 + h, leaf.y0, h, node),
        build(leaf.x0, leaf.y0 + h, h, node),
        build(leaf.x0 + h, leaf.y0 + h, h, node),
      ].filter((n): n is Node => n !== null)
      node.children = kids.length > 0 ? kids : null
      balanceSplits++
      if (!parent) continue
      parent.children = parent.children!.map((c) => (c === leaf ? node : c))
    }
    if (balanceSplits > 200000) break
  }

  // ── 缝合状态：每条边是否需要中点（邻居更细一级）──
  const leafNeedsMid = (
    x0: number,
    y0: number,
    s: number,
    dir: "R" | "L" | "B" | "T",
  ): boolean => {
    const h = s >> 1
    let px: number
    let py: number
    if (dir === "R") {
      px = x0 + s
      py = y0 + h
    } else if (dir === "L") {
      px = x0 - 1
      py = y0 + h
    } else if (dir === "B") {
      px = x0 + h
      py = y0 + s
    } else {
      px = x0 + h
      py = y0 - 1
    }
    if (px < 0 || py < 0 || px >= width || py >= height) return false
    const n = findLeaf(root, px, py)
    return n !== null && n.s < s
  }

  // ── 出面 ──
  const fx = width / (2 * Math.tan(camera.fovX / 2))
  const fy = height / (2 * Math.tan(camera.fovY / 2))
  const cx = width / 2
  const cy = height / 2
  const view = camera.viewMatrix
  const right: [number, number, number] = [view[0], view[4], view[8]]
  const down: [number, number, number] = [view[1], view[5], view[9]]
  const fwd: [number, number, number] = [view[2], view[6], view[10]]
  const origin = camera.position
  const place = (
    x: number,
    y: number,
    z: number,
    out: [number, number, number],
  ): boolean => {
    if (!(z > 0)) return false
    const camX = ((x + 0.5 - cx) / fx) * z
    const camY = ((y + 0.5 - cy) / fy) * z
    out[0] = origin[0] + right[0] * camX + down[0] * camY + fwd[0] * z
    out[1] = origin[1] + right[1] * camX + down[1] * camY + fwd[1] * z
    out[2] = origin[2] + right[2] * camX + down[2] * camY + fwd[2] * z
    return true
  }
  const pool = new VertexPool(
    width,
    height,
    place,
    depth,
    support,
    snapBoundary,
    minCell,
  )
  const indices: number[] = []
  const leaves: Node[] = []
  collectLeaves(root, leaves)

  const dist2 = (a: number, b: number): number => {
    const ax = pool.positions[a * 3]
    const ay = pool.positions[a * 3 + 1]
    const az = pool.positions[a * 3 + 2]
    const bx = pool.positions[b * 3]
    const by = pool.positions[b * 3 + 1]
    const bz = pool.positions[b * 3 + 2]
    return (ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2
  }

  for (const leaf of leaves) {
    const { x0, y0, s } = leaf
    const h = s >> 1
    const c00 = pool.get(x0, y0)
    const c01 = pool.get(x0, y0 + s)
    const c11 = pool.get(x0 + s, y0 + s)
    const c10 = pool.get(x0 + s, y0)
    if (c00 < 0 || c01 < 0 || c11 < 0 || c10 < 0) continue
    // 多边形（图像坐标 y 向下；绕序与 relief.ts 的表面三角一致）：c00→c01→c11→c10。
    const poly: number[] = [c00]
    if (leafNeedsMid(x0, y0, s, "L")) pushIfValid(poly, x0, y0 + h)
    poly.push(c01)
    if (leafNeedsMid(x0, y0, s, "B")) pushIfValid(poly, x0 + h, y0 + s)
    poly.push(c11)
    if (leafNeedsMid(x0, y0, s, "R")) pushIfValid(poly, x0 + s, y0 + h)
    poly.push(c10)
    if (leafNeedsMid(x0, y0, s, "T")) pushIfValid(poly, x0 + h, y0)
    // 对角翻转：3D 对角线更短者优先（与 relief.ts 同语义）。
    // 无中点时：从 c00 扇出 = 对角线 c00-c11；从 c01 扇出 = 对角线 c01-c10。
    const flip = dist2(c10, c01) < dist2(c00, c11)
    const ordered = flip ? rotateFrom(poly, c01) : poly
    // 扇形三角化（凸多边形，从首顶点扇出）。
    for (let i = 1; i + 1 < ordered.length; i++) {
      indices.push(ordered[0], ordered[i], ordered[i + 1])
    }

    function pushIfValid(target: number[], x: number, y: number): void {
      const v = pool.get(x, y)
      if (v >= 0) target.push(v)
    }
  }

  const positions = Float32Array.from(pool.positions)
  const uvs = Float32Array.from(pool.uvs)
  const levelHistogram: number[] = []
  for (const leaf of leaves) {
    const k = Math.round(Math.log2(leaf.s / minCell))
    levelHistogram[k] = (levelHistogram[k] ?? 0) + 1
  }
  const levelCount = levelHistogram.length
  return {
    positions,
    uvs,
    indices: Uint32Array.from(indices),
    stats: {
      leaves: leaves.length,
      vertexCount: positions.length / 3,
      triangleCount: indices.length / 3,
      levelHistogram: Array.from(
        { length: levelCount },
        (_, i) => levelHistogram[i] ?? 0,
      ),
      balanceSplits,
    },
  }
}

/** 循环左移，使 `start` 成为首元素（保持相对顺序 ⇒ 绕序不变）。 */
function rotateFrom(poly: readonly number[], start: number): number[] {
  const i = poly.indexOf(start)
  if (i < 0) return [...poly]
  return [...poly.slice(i), ...poly.slice(0, i)]
}

function clampPow2(value: number, lo: number, hi: number): number {
  let v = Math.max(1, Math.floor(value))
  // 向上取到最近的 2 的幂（>= lo）
  let p = 1
  while (p < v) p *= 2
  v = Math.max(lo, Math.min(hi, p))
  return v
}
