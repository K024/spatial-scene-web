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
  /**
   * **输出三角面** vs 视差场的最大误差（像素域视差）。
   *
   * 这是对最终几何（含中点缝合、含翻对角）验差的实测值 —— `maxError` 才是门。
   * cell 边长 > 64px 时按步长抽样（见 `polygonFanError` 的 `stride` 注释），
   * 小 cell 是全像素验。**权威判定在 `scripts/meshing-golden.ts` 的 A11**（全像素、独立实现）。
   */
  readonly maxTriangleError: number
  /**
   * 顶点贴边界（无支撑角点 snap 到邻域）而**无法用预测面验差**的叶子数。
   *
   * 这是本模块**最大的诚实性缺口**：snap 圈上的几何顶点位置取自邻域支撑像素、UV 仍用自己的
   * 像素，它本来就不该用视差场判对错。native 实测这个数是 **1 万–22 万片叶子**（随 `minCell`
   * 与是否开 snap 变化），量级和受验叶子相当 —— 所以「误差有界」只能声明在**受验**的那部分上。
   */
  readonly errorSkippedLeaves: number
  /**
   * 已到 `minCell` 但输出面误差仍 > `maxError` 的叶子数 —— **必须报出来**。
   *
   * 到了最小格子就不能再细分，此时「误差有界」不成立；让调用方知道有多少格子超差，
   * 而不是继续把它当作「全局误差有界」。
   *
   * ⚠ **能到的精度由 `minCell` 决定，不是由 `maxError` 决定**（`example.ply` @ native
   * 3024×2268、L=6、`snapBoundary=false` 实测）：
   * ```
   *   minCell=1 e=0.002 → 输出面误差 0.00200、超差 0         （maxError 真的是界）
   *   minCell=2 e=0.002 → 0.03496、超差 55,967   e=0.02 → 超差 22
   *   minCell=4 e=0.002 → 0.05565、超差 58,157   e=0.02 → 超差 324
   *   minCell=8 e=0.002 → 0.06502、超差 32,748   e=0.02 → 超差 590
   * ```
   * 读法：2–16px 的格子里本来就没有「三条边拟合得出」的曲面，所以精度下限跟着 `minCell` 走；
   * 而**超差计数**同时取决于场在该尺度上的粗糙程度（同样是 `minCell=2`，`e=0.002` 超差 5.6 万、
   * `e=0.02` 只剩 22）。⇒ 门要写成 `minCell × maxError` 的**联合声明**。
   */
  readonly minCellViolations: number
  /** 未到 `minCell` 却仍超差的叶子数（正常应为 0 —— 说明验差没驱动细分）。 */
  readonly overErrorLeaves: number
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
  /** 顶点 id -> 所属像素 `(x, y)`（验差要用回像素坐标）。 */
  readonly pixelCoord: number[] = []
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
    this.pixelCoord.push(x, y)
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

  /**
   * cell 的**合法矩形**：`x1 = min(x0+s, width−1)`、`y1 = min(y0+s, height−1)`。
   *
   * 末格必须**裁剪**到这个矩形，而不是整格丢弃。旧实现要求 `x0+s ≤ width−1`，
   * 于是尺寸不是 `minCell` 整数倍的图会把最右/最下一整条带掉光：
   * 实测 8×8 满支撑 + `minCellPx=4` 只出 1 个叶子 / 2 个三角，最远只盖到像素坐标 4，
   * 后 39/64 个像素**没有任何几何**（轮廓处直接露底）。
   */
  const cellRect = (
    x0: number,
    y0: number,
    s: number,
  ): { x0: number; y0: number; x1: number; y1: number } => ({
    x0,
    y0,
    x1: Math.min(x0 + s, width - 1),
    y1: Math.min(y0 + s, height - 1),
  })

  /**
   * 区域是否可整块作为叶子：整块支撑、无撕裂、**输出三角面**的视差误差在门内。
   *
   * ⚠ 误差必须按**实际输出的曲面**（两三角分片线性）验，不能按四角双线性验。
   * 双线性鞍面 `n = 0.5 + a·u·v` 的双线性拟合误差**恒为 0**，而它对应的两三角曲面
   * 在格中心偏差 `a/4`：旧实现因此在鞍面上放行任意大的格子（实测 64×64、
   * `maxError=0.005`、`a=0.1` 时，输出三角面的真实视差误差 0.0258）。
   * 两种对角都验（出面按 3D 对角线择一，两条都在门内就不必知道选了哪条）。
   */
  const regionClean = (
    x0: number,
    y0: number,
    x1: number,
    y1: number,
  ): boolean => {
    if (x1 <= x0 || y1 <= y0) return false
    const rw = x1 - x0 + 1
    const rh = y1 - y0 + 1
    if (sumSupport(x0, y0, x1, y1) !== rw * rh) return false
    if (hasTear(x0, y0, x1, y1)) return false
    const corners = {
      n00: disparity[y0 * width + x0],
      n10: disparity[y0 * width + x1],
      n01: disparity[y1 * width + x0],
      n11: disparity[y1 * width + x1],
    }
    const rect = { x0, y0, x1, y1 }
    return (
      fanError(rect, corners, false, disparity, width, support) <= maxError &&
      fanError(rect, corners, true, disparity, width, support) <= maxError
    )
  }

  /** min cell 的判据：合法矩形 + 无撕裂 + （贴边界时）cell 内有支撑 / （不贴时）四角支撑。 */
  const minCellClean = (
    x0: number,
    y0: number,
    x1: number,
    y1: number,
  ): boolean => {
    if (x1 <= x0 || y1 <= y0) return false
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
    const { x1, y1 } = cellRect(x0, y0, s)
    if (x1 <= x0 || y1 <= y0) return null
    if (sumSupport(x0, y0, x1, y1) === 0) return null
    if (s <= maxCell && regionClean(x0, y0, x1, y1)) {
      return { x0, y0, s, children: null, parent }
    }
    if (s <= minCell) {
      return minCellClean(x0, y0, x1, y1)
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
        maxTriangleError: 0,
        errorSkippedLeaves: 0,
        minCellViolations: 0,
        overErrorLeaves: 0,
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
    const { x1, y1 } = cellRect(x0, y0, s)
    const mx = (x0 + x1) >> 1
    const my = (y0 + y1) >> 1
    const probes: Array<[number, number]> = [
      [x1, my],
      [x0 - 1, my],
      [mx, y1],
      [mx, y0 - 1],
    ]
    for (const [px, py] of probes) {
      if (px < 0 || py < 0 || px >= width || py >= height) continue
      const n = findLeaf(root, px, py)
      if (n && n.s <= s / 4) return true
    }
    return false
  }

  /** 把叶子换成一个（可能为空的）中间节点；返回是否真的细分了。 */
  const splitLeaf = (leaf: Node): boolean => {
    if (leaf.s <= minCell) return false
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
    if (!parent) return false
    parent.children = parent.children!.map((c) => (c === leaf ? node : c))
    return true
  }

  /** 按判据选一批叶子细分，直到不动或到次数上限；返回实际细分次数。 */
  const relax = (
    needsSplit: (leaf: Node) => boolean,
    maxIter: number,
  ): number => {
    let splits = 0
    for (let iter = 0; iter < maxIter; iter++) {
      const leaves: Node[] = []
      collectLeaves(root, leaves)
      const victims = leaves.filter(needsSplit)
      if (victims.length === 0) break
      let moved = false
      for (const leaf of victims) if (splitLeaf(leaf)) moved = true
      if (!moved) break
      splits++
      if (splits > 200000) break
    }
    return splits
  }

  // ① 2:1 受限松弛：有「小 2 级以上的邻居」的叶子强制细分（消 T-junction）。
  const balanceSplits = relax(hasThinNeighbor, 12)

  // ── 缝合状态：每条边是否需要中点（邻居更细一级）──
  function leafNeedsMid(
    x0: number,
    y0: number,
    s: number,
    dir: "R" | "L" | "B" | "T",
  ): boolean {
    const { x1, y1 } = cellRect(x0, y0, s)
    const mx = (x0 + x1) >> 1
    const my = (y0 + y1) >> 1
    let px: number
    let py: number
    if (dir === "R") {
      px = x1
      py = my
    } else if (dir === "L") {
      px = x0 - 1
      py = my
    } else if (dir === "B") {
      px = mx
      py = y1
    } else {
      px = mx
      py = y0 - 1
    }
    if (px < 0 || py < 0 || px >= width || py >= height) return false
    const n = findLeaf(root, px, py)
    return n !== null && n.s < s
  }

  /**
   * 该叶子**实际要输出**的像素多边形（含边中点、已按出面顺序），顶点全部落在支撑内时返回。
   *
   * 返回 `null` = 有顶点需要贴边界外推（snap）⇒ 它不属于「预测面」，不能用来验差。
   * 顺序必须与出面段完全一致：`c00 →(L 中点) c01 →(B 中点) c11 →(R 中点) c10 →(T 中点)`。
   */
  const leafPolygon = (leaf: Node): Array<[number, number, number]> | null => {
    const { x0, y0, s } = leaf
    const { x1, y1 } = cellRect(x0, y0, s)
    const mx = (x0 + x1) >> 1
    const my = (y0 + y1) >> 1
    const midX = x1 - x0 >= 2
    const midY = y1 - y0 >= 2
    const pts: Array<[number, number, number]> = []
    const push = (px: number, py: number): boolean => {
      if (!support[py * width + px] || !(depth[py * width + px] > 0)) {
        return false
      }
      pts.push([px, py, disparity[py * width + px]])
      return true
    }
    const need = (dir: "R" | "L" | "B" | "T"): boolean =>
      leafNeedsMid(x0, y0, s, dir)
    if (!push(x0, y0)) return null
    if (midY && need("L") && !push(x0, my)) return null
    if (!push(x0, y1)) return null
    if (midX && need("B") && !push(mx, y1)) return null
    if (!push(x1, y1)) return null
    if (midY && need("R") && !push(x1, my)) return null
    if (!push(x1, y0)) return null
    if (midX && need("T") && !push(mx, y0)) return null
    return pts
  }

  /** 输出面误差（`null` = 该叶子含贴边外推顶点，不参与验差）。 */
  const leafEmittedError = (leaf: Node): number | null => {
    const pts = leafPolygon(leaf)
    if (!pts) return null
    const { x0, y0, s } = leaf
    const { x1, y1 } = cellRect(x0, y0, s)
    return polygonFanError(pts, disparity, width, { x0, y0, x1, y1 }, support)
  }

  // ② **验差驱动细分**：只按「四角预测面」验差不等于最终三角划分合格 ——
  //    缝合中点会改变扇形三角化，实测仍有少量叶子（s=8/16）输出面误差 0.0050–0.0070，
  //    略超 0.005。这里按**实际输出多边形**再验一遍，超差且还没到 minCell 就继续细分。
  relax((leaf) => {
    if (leaf.s <= minCell) return false
    const err = leafEmittedError(leaf)
    return err !== null && err > maxError
  }, 6)

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
  // ── 验差报告：对**最终**叶子集逐叶统计（超差的非最小叶子已被上一段细分掉）──
  let maxTriangleError = 0
  let errorSkippedLeaves = 0
  let minCellViolations = 0
  let overErrorLeaves = 0
  for (const leaf of leaves) {
    const err = leafEmittedError(leaf)
    if (err === null) {
      errorSkippedLeaves++
      continue
    }
    if (err > maxTriangleError) maxTriangleError = err
    if (err > maxError) {
      if (leaf.s <= minCell) minCellViolations++
      else overErrorLeaves++
    }
  }

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
    const { x1, y1 } = cellRect(x0, y0, s)
    const mx = (x0 + x1) >> 1
    const my = (y0 + y1) >> 1
    // 中点只在边真的有长度时插入（末格被裁剪后可能只剩 1–2 列）。
    const midX = x1 - x0 >= 2
    const midY = y1 - y0 >= 2
    const c00 = pool.get(x0, y0)
    const c01 = pool.get(x0, y1)
    const c11 = pool.get(x1, y1)
    const c10 = pool.get(x1, y0)
    if (c00 < 0 || c01 < 0 || c11 < 0 || c10 < 0) continue
    // 多边形（图像坐标 y 向下；绕序与 relief.ts 的表面三角一致）：c00→c01→c11→c10。
    const poly: number[] = [c00]
    if (midY && leafNeedsMid(x0, y0, s, "L")) pushIfValid(poly, x0, my)
    poly.push(c01)
    if (midX && leafNeedsMid(x0, y0, s, "B")) pushIfValid(poly, mx, y1)
    poly.push(c11)
    if (midY && leafNeedsMid(x0, y0, s, "R")) pushIfValid(poly, x1, my)
    poly.push(c10)
    if (midX && leafNeedsMid(x0, y0, s, "T")) pushIfValid(poly, mx, y0)
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
  if (overErrorLeaves > 0) {
    console.warn(
      `[lod] ${overErrorLeaves} 个非最小叶子输出面误差 > maxError=${maxError}（验差未驱动细分，请查 build 的 regionClean）`,
    )
  }
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
      maxTriangleError,
      errorSkippedLeaves,
      minCellViolations,
      overErrorLeaves,
    },
  }
}

/** 三次重心插值：`pts` 的三角形内返回插值，否则 `null`。 */
function baryInterp(
  ax: number,
  ay: number,
  av: number,
  bx: number,
  by: number,
  bv: number,
  cx: number,
  cy: number,
  cv: number,
  px: number,
  py: number,
): number | null {
  const d = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
  if (Math.abs(d) < 1e-12) return null
  const eps = 1e-6
  const l1 = ((bx - px) * (cy - py) - (cx - px) * (by - py)) / d
  const l2 = ((cx - px) * (ay - py) - (ax - px) * (cy - py)) / d
  const l3 = 1 - l1 - l2
  if (!(l1 >= -eps && l2 >= -eps && l3 >= -eps)) return null
  return l1 * av + l2 * bv + l3 * cv
}

/**
 * **输出曲面**（凸多边形扇形三角化）vs 视差场的最大误差。
 *
 * `pts` = 按出面顺序排列的多边形顶点 `[x, y, n]`（含中点、已按翻对角排好首顶点）。
 * `bounds` = 该叶子的像素矩形；只在矩形内取样。
 *
 * 大 cell（边长 > 64px）按 `stride` 抽样：全像素验差的代价是 `Σ 面积 × 三角数`，
 * 在 3024×2268 上按 6 个三角形全验会翻倍网格化耗时，而超差只会发生在面内中部、
 * 不会只有单个像素。**权威判定用 `scripts/meshing-golden.ts` A11 的全像素实现。**
 */
function polygonFanError(
  pts: readonly (readonly [number, number, number])[],
  disparity: ArrayLike<number>,
  width: number,
  bounds: { x0: number; y0: number; x1: number; y1: number },
  support?: Uint8Array,
): number {
  if (pts.length < 3) return 0
  const span = Math.max(bounds.x1 - bounds.x0, bounds.y1 - bounds.y0)
  const stride = span > 64 ? Math.ceil(span / 64) : 1
  let worst = 0
  for (let y = bounds.y0; y <= bounds.y1; y += stride) {
    for (let x = bounds.x0; x <= bounds.x1; x += stride) {
      const idx = y * width + x
      // 只验**有支撑**的像素：无支撑像素的视差没有意义（0 或别的表面），
      // 拿它当"真值"会把贴边 cell（snapBoundary）判成天文数字误差。
      if (support && !support[idx]) continue
      const actual = disparity[idx]
      let interp: number | null = null
      for (let i = 1; i + 1 < pts.length && interp === null; i++) {
        const a = pts[0]
        const b = pts[i]
        const c = pts[i + 1]
        interp = baryInterp(
          a[0],
          a[1],
          a[2],
          b[0],
          b[1],
          b[2],
          c[0],
          c[1],
          c[2],
          x,
          y,
        )
      }
      if (interp === null) continue
      const err = Math.abs(interp - actual)
      if (err > worst) worst = err
    }
  }
  return worst
}

/**
 * 四角矩形按两种对角之一的扇形三角化误差（`regionClean` 的选择门用）。
 *
 * `flip=false` ⇒ 从 `c00` 扇出（对角线 `c00–c11`）；`flip=true` ⇒ 从 `c01` 扇出
 * （对角线 `c01–c10`）—— 与出面段的 `rotateFrom(poly, c01)` 完全同序。
 */
function fanError(
  rect: { x0: number; y0: number; x1: number; y1: number },
  corners: { n00: number; n10: number; n01: number; n11: number },
  flip: boolean,
  disparity: ArrayLike<number>,
  width: number,
  support: Uint8Array,
): number {
  const a: [number, number, number] = [rect.x0, rect.y0, corners.n00]
  const b: [number, number, number] = [rect.x0, rect.y1, corners.n01]
  const c: [number, number, number] = [rect.x1, rect.y1, corners.n11]
  const d: [number, number, number] = [rect.x1, rect.y0, corners.n10]
  const pts = flip ? [b, c, d, a] : [a, b, c, d]
  return polygonFanError(pts, disparity, width, rect, support)
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
