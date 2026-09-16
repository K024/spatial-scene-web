/**
 * LOD relief 网格化：**受限四叉树**自适应出面。
 *
 * 阶段：meshing 的出面步骤。每层都是渲出来的 heightfield（同像素多深度已被 α 加权
 * 坍缩成一个值），所以几何恒为 `z(u, v)`，网格化就是「在平滑处放大格子、在结构处细分」。
 *
 * ── 为什么必须自适应 ──
 * 逐像素出面在 1024×768 上就轻松到百万面/层，L 层直接失控。本模块把出面换成
 * **受限四叉树**（Lindstrom 1996 / ROAM / Pajarola 2002 的 raster heightmap 路线）：
 * 大片平滑区域用一个四边形表示，误差超限的地方自动细分。语义与
 * Facebook *One Shot 3D Photography* §4.4.2 一致：**深度断层交给支撑边界，
 * 内部平滑区按深度误差粗化**。
 *
 * ── 三条硬约束 ──
 * 1. **只在支撑内出几何**：能当叶子的格子必须整块被支撑（min cell 处退化为「格内有支撑」）；
 * 2. **误差有界**：以**视差域**（NDC，`1/z` 的仿射）度量粗化误差，`maxError` 是门。
 *    视差是正确度量：`Δscreen = f·t·Δn·(far−near)/(far·near)`，`Δn` 到新视角的屏幕位移
 *    是全局常数比例，与深度无关；
 * 3. **无 T-junction 裂缝**：相邻叶子层级差最多 1（2:1 受限松弛），细邻居的边上插入
 *    中点并共享顶点 ⇒ 无缝。不做裁剪式 marching squares —— 剪影由**全分辨率纹理的 α**
 *    决定，几何只需近似覆盖，这是「几何粗化但纹理全分辨率」压面数的关键。
 *
 * ── 边界贴合（`snapBoundaryPx`）──
 * 支撑是渲出来的软边，最外圈 α 很低。min cell 处只要**格内有支撑**就出面；没有支撑的
 * 角点贴到最近的 seed（`SupportField.nearestSeed`，距离不超过 `snapBoundaryPx`）取深度，
 * **UV 仍取自己的像素**（采样到的是低 α 纹素，被逐 texel α 混合自然掩掉）。
 * 关掉它则要求四角全支撑，剪影处会整块丢格子（漏覆盖）。
 *
 * ── 与外部的接口 ──
 * 输入是 `SupportField`（内含**已外扩**的深度场）+ 渲染相机；输出是
 * **世界坐标**（米）顶点 + 归一化 UV（左上原点）+ 三角形索引，纹理仍是该层的全分辨率
 * `rgb/alpha`（几何/纹理解耦）。
 *
 * @see SOLIDI `mesh.py:create_mesh` / `generate_face`（像素网格 + 外推）——
 *   vt-vl-lab/3d-photo-inpainting @ de04467 (MIT)。本实现不移植其 LDI 图机制与修补网络。
 */

import { ndcDepthFromZ } from "../layering/placement.ts"
import type { WSplatCamera } from "../wsplat/camera.ts"
import type { SupportField } from "./support.ts"

/** LOD 出面选项。 */
export interface LodOptions {
  /**
   * 最小 cell（像素，2 的幂）。默认 4。
   * 面数的**主要下界旋钮**：支撑边界最细只能到这一级，所以剪影会被量化到
   * `minCellPx` 像素（由纹理 α 掩边）。
   */
  readonly minCellPx?: number
  /** 最大 cell（像素，2 的幂）。默认 128。防止整片天空并成一个巨四边形。 */
  readonly maxCellPx?: number
  /** 视差域最大误差（绝对 NDC 视差）。默认 0.005。 */
  readonly maxError?: number
  /**
   * 边界贴合半径（像素）。默认 = `minCellPx`；`0` = 关闭贴合。
   * 无支撑的角点贴到半径内最近的 seed 取深度（UV 仍取自己的像素）。
   */
  readonly snapBoundaryPx?: number
}

/** LOD 出面诊断。 */
export interface LodStats {
  readonly leaves: number
  readonly vertexCount: number
  readonly triangleCount: number
  /** 层级直方图：`levels[k]` = 边长 `minCell·2^k` 的叶子数。 */
  readonly levelHistogram: readonly number[]
  /** 因 2:1 松弛额外细分的轮数。 */
  readonly balanceSplits: number
  /** 输出三角面 vs 视差场的最大误差（抽样实测值）。 */
  readonly maxTriangleError: number
  /** 已到 `minCell` 但输出面误差仍 > `maxError` 的叶子数（**必须报出来**）。 */
  readonly minCellViolations: number
  /** 含贴边外推顶点、无法用预测面验差的叶子数。 */
  readonly skippedLeaves: number
  /** 四角里有贴边外推（深度取自最近 seed）的叶子数。 */
  readonly snappedLeaves: number
}

export interface LodMeshResult {
  readonly positions: Float32Array
  readonly uvs: Float32Array
  readonly indices: Uint32Array
  readonly stats: LodStats
}

const DEFAULT_MIN_CELL = 4
const DEFAULT_MAX_CELL = 128
const DEFAULT_MAX_ERROR = 0.005

interface Node {
  x0: number
  y0: number
  s: number
  children: Node[] | null
  parent: Node | null
}

/**
 * 建一层的 LOD relief 网格。
 *
 * @param size 该层的像素尺寸（须与 `support` / 相机一致）。
 * @param camera 渲染相机（`viewMatrix` 决定世界坐标，`near/far` 决定视差域）。
 * @param support `computeSupportField()` 的产物（深度已外扩）。
 */
export function buildLayerLodMesh(
  size: { readonly width: number; readonly height: number },
  camera: WSplatCamera,
  support: SupportField,
  options: LodOptions = {},
): LodMeshResult {
  const { width, height } = size
  if (camera.width !== width || camera.height !== height) {
    throw new Error(
      `lod：相机尺寸 ${camera.width}x${camera.height} 与帧 ${width}x${height} 不一致`,
    )
  }
  if (support.width !== width || support.height !== height) {
    throw new Error("lod：支撑场尺寸与帧不一致")
  }
  const pixels = width * height
  const minCell = clampPow2(options.minCellPx ?? DEFAULT_MIN_CELL, 1, 1 << 12)
  const maxCell = clampPow2(
    options.maxCellPx ?? DEFAULT_MAX_CELL,
    minCell,
    1 << 12,
  )
  const maxError = Math.max(0, options.maxError ?? DEFAULT_MAX_ERROR)
  const snapRadius = Math.max(0, Math.floor(options.snapBoundaryPx ?? minCell))

  // ── 视差场（只在支撑内有意义；深度已外扩，非支撑像素也可供贴合采样）──
  const disparity = new Float32Array(pixels)
  for (let i = 0; i < pixels; i++) {
    const z = support.depth[i]
    disparity[i] = z > 0 ? ndcDepthFromZ(z, camera.near, camera.far) : 0
  }

  // ── 支撑前缀和（O(1) 区域判定）──
  const stride = width + 1
  const ps = new Int32Array(stride * (height + 1))
  for (let y = 0; y < height; y++) {
    let row = 0
    for (let x = 0; x < width; x++) {
      row += support.support[y * width + x]
      ps[(y + 1) * stride + (x + 1)] = ps[y * stride + (x + 1)] + row
    }
  }
  const sumSupport = (x0: number, y0: number, x1: number, y1: number): number =>
    ps[(y1 + 1) * stride + (x1 + 1)] -
    ps[y0 * stride + (x1 + 1)] -
    ps[(y1 + 1) * stride + x0] +
    ps[y0 * stride + x0]

  /** cell 的**合法矩形**：末格裁剪到图像内（而不是整格丢弃）。 */
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
   * 整块可作叶子：整块支撑 + **两种扇出的输出面**视差误差都在门内。
   *
   * 误差必须按**实际输出的曲面**（分片线性三角形）验，不能按四角双线性验：
   * 鞍面 `n = 0.5 + a·u·v` 的双线性拟合误差恒为 0，而两三角曲面在格中心偏差 `a/4`。
   * 出面时按 3D 对角线择一，所以两条都验（保守）。
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
    const corners = {
      n00: disparity[y0 * width + x0],
      n10: disparity[y0 * width + x1],
      n01: disparity[y1 * width + x0],
      n11: disparity[y1 * width + x1],
    }
    const rect = { x0, y0, x1, y1 }
    return (
      fanError(rect, corners, false, disparity, width, support.support) <=
        maxError &&
      fanError(rect, corners, true, disparity, width, support.support) <=
        maxError
    )
  }

  const rootSize = nextPow2(Math.max(width, height))
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
      return { x0, y0, s, children: null, parent }
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
  if (!root) return emptyResult()

  // ── 树查询工具 ──
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

  /** 该叶子四个边中点外是否存在「细 2 级及以上」的邻居（需要继续细分）。 */
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
    if (!parent) return false
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
    parent.children = parent.children!.map((c) => (c === leaf ? node : c))
    return true
  }

  /** 按判据反复细分，直到不动或到轮数上限。返回细分轮数。 */
  const relax = (
    needsSplit: (leaf: Node) => boolean,
    maxIter: number,
  ): number => {
    let rounds = 0
    for (let iter = 0; iter < maxIter; iter++) {
      const leaves: Node[] = []
      collectLeaves(root, leaves)
      const victims = leaves.filter(needsSplit)
      if (victims.length === 0) break
      let moved = false
      for (const leaf of victims) if (splitLeaf(leaf)) moved = true
      if (!moved) break
      rounds++
    }
    return rounds
  }

  /** 某条边上是否需要中点（邻居更细一级）。 */
  const leafNeedsMid = (
    x0: number,
    y0: number,
    s: number,
    dir: "R" | "L" | "B" | "T",
  ): boolean => {
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
   * 该叶子**实际输出**的像素多边形（含边中点），按出面顺序：
   * `c00 →(L) c01 →(B) c11 →(R) c10 →(T)`；任一顶点不在支撑内时返回 `null`
   * （含贴边外推的叶子不能用预测面验差）。
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
      const i = py * width + px
      if (!support.support[i] || !(support.depth[i] > 0)) return false
      pts.push([px, py, disparity[i]])
      return true
    }
    if (!push(x0, y0)) return null
    if (midY && leafNeedsMid(x0, y0, s, "L") && !push(x0, my)) return null
    if (!push(x0, y1)) return null
    if (midX && leafNeedsMid(x0, y0, s, "B") && !push(mx, y1)) return null
    if (!push(x1, y1)) return null
    if (midY && leafNeedsMid(x0, y0, s, "R") && !push(x1, my)) return null
    if (!push(x1, y0)) return null
    if (midX && leafNeedsMid(x0, y0, s, "T") && !push(mx, y0)) return null
    return pts
  }

  const leafEmittedError = (leaf: Node): number | null => {
    const pts = leafPolygon(leaf)
    if (!pts) return null
    const { x0, y0, s } = leaf
    const { x1, y1 } = cellRect(x0, y0, s)
    return polygonFanError(
      pts,
      disparity,
      width,
      { x0, y0, x1, y1 },
      support.support,
    )
  }

  // ① 2:1 受限松弛：消 T-junction。
  let balanceSplits = relax(hasThinNeighbor, 12)
  // ② 按**实际输出多边形**再验差（缝合中点会改变扇形三角化）。
  relax((leaf) => {
    if (leaf.s <= minCell) return false
    const err = leafEmittedError(leaf)
    return err !== null && err > maxError
  }, 6)
  // ③ 再平衡一次：② 的细分可能让某个叶子比邻居细 2 级以上（新 T-junction），
  //    而松弛只会继续细分、不会合并，所以再跑一轮即收敛。
  balanceSplits += relax(hasThinNeighbor, 12)

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

  const vertexOf = new Int32Array(pixels).fill(-1)
  const vx: number[] = []
  const vy: number[] = []
  const vz: number[] = []
  const uvs: number[] = []

  /** 取（或建）像素 `(x,y)` 的顶点；无支撑时按 `snapRadius` 贴最近 seed。 */
  const getVertex = (x: number, y: number): number => {
    const key = y * width + x
    const existing = vertexOf[key]
    if (existing >= 0) return existing
    let src = key
    if (!(support.support[key] === 1 && support.depth[key] > 0)) {
      if (snapRadius === 0) return -1
      const nearest = support.nearestSeed[key]
      if (nearest < 0 || support.distance[key] > snapRadius) return -1
      src = nearest
    }
    const z = support.depth[src]
    if (!(z > 0)) return -1
    const id = vx.length
    const camX = ((x + 0.5 - cx) / fx) * z
    const camY = ((y + 0.5 - cy) / fy) * z
    vx.push(origin[0] + right[0] * camX + down[0] * camY + fwd[0] * z)
    vy.push(origin[1] + right[1] * camX + down[1] * camY + fwd[1] * z)
    vz.push(origin[2] + right[2] * camX + down[2] * camY + fwd[2] * z)
    // UV 永远取**顶点自己的像素**（不是源像素）：外推处纹理 α 本来就低，
    // 多出来的几何被 α 自动掩掉，剪影由全分辨率纹理决定。
    uvs.push((x + 0.5) / width, (y + 0.5) / height)
    vertexOf[key] = id
    return id
  }

  const indices: number[] = []
  const leaves: Node[] = []
  collectLeaves(root, leaves)

  // ── 验差报告：对最终叶子集逐叶统计 ──
  let maxTriangleError = 0
  let minCellViolations = 0
  let skippedLeaves = 0
  let snappedLeaves = 0
  const needsSnap = (x: number, y: number): boolean =>
    !(support.support[y * width + x] === 1 && support.depth[y * width + x] > 0)
  for (const leaf of leaves) {
    const err = leafEmittedError(leaf)
    if (err === null) {
      skippedLeaves++
    } else {
      if (err > maxTriangleError) maxTriangleError = err
      if (err > maxError && leaf.s <= minCell) minCellViolations++
    }
    if (snapRadius > 0) {
      const { x0, y0, s } = leaf
      const { x1, y1 } = cellRect(x0, y0, s)
      if (
        needsSnap(x0, y0) ||
        needsSnap(x0, y1) ||
        needsSnap(x1, y0) ||
        needsSnap(x1, y1)
      ) {
        snappedLeaves++
      }
    }
  }

  const dist2 = (a: number, b: number): number => {
    const dx = vx[a] - vx[b]
    const dy = vy[a] - vy[b]
    const dz = vz[a] - vz[b]
    return dx * dx + dy * dy + dz * dz
  }

  const pushVertex = (target: number[], x: number, y: number): void => {
    const v = getVertex(x, y)
    if (v >= 0) target.push(v)
  }

  for (const leaf of leaves) {
    const { x0, y0, s } = leaf
    const { x1, y1 } = cellRect(x0, y0, s)
    const mx = (x0 + x1) >> 1
    const my = (y0 + y1) >> 1
    const midX = x1 - x0 >= 2
    const midY = y1 - y0 >= 2
    const c00 = getVertex(x0, y0)
    const c01 = getVertex(x0, y1)
    const c11 = getVertex(x1, y1)
    const c10 = getVertex(x1, y0)
    if (c00 < 0 || c01 < 0 || c11 < 0 || c10 < 0) continue

    // 多边形（含边中点，图像坐标 y 向下；绕序与 relief 表面三角一致）：
    // c00 →(L) c01 →(B) c11 →(R) c10 →(T)。
    const poly: number[] = [c00]
    if (midY && leafNeedsMid(x0, y0, s, "L")) pushVertex(poly, x0, my)
    poly.push(c01)
    if (midX && leafNeedsMid(x0, y0, s, "B")) pushVertex(poly, mx, y1)
    poly.push(c11)
    if (midY && leafNeedsMid(x0, y0, s, "R")) pushVertex(poly, x1, my)
    poly.push(c10)
    if (midX && leafNeedsMid(x0, y0, s, "T")) pushVertex(poly, mx, y0)

    // 对角翻转：3D 对角线更短者优先。无中点时从 c00 扇出 = 对角线 c00–c11，
    // 从 c01 扇出 = 对角线 c01–c10。
    const flip = dist2(c10, c01) < dist2(c00, c11)
    const ordered = flip ? rotateFrom(poly, c01) : poly
    for (let i = 1; i + 1 < ordered.length; i++) {
      indices.push(ordered[0], ordered[i], ordered[i + 1])
    }
  }

  const levelHistogram: number[] = []
  for (const leaf of leaves) {
    const k = Math.round(Math.log2(leaf.s / minCell))
    levelHistogram[k] = (levelHistogram[k] ?? 0) + 1
  }

  return {
    positions: Float32Array.from(interleave(vx, vy, vz)),
    uvs: Float32Array.from(uvs),
    indices: Uint32Array.from(indices),
    stats: {
      leaves: leaves.length,
      vertexCount: vx.length,
      triangleCount: indices.length / 3,
      levelHistogram: Array.from(
        { length: levelHistogram.length },
        (_, i) => levelHistogram[i] ?? 0,
      ),
      balanceSplits,
      maxTriangleError,
      minCellViolations,
      skippedLeaves,
      snappedLeaves,
    },
  }
}

/** `[x...] [y...] [z...]` -> 交错 xyz。 */
function interleave(x: number[], y: number[], z: number[]): number[] {
  const out = new Array<number>(x.length * 3)
  for (let i = 0; i < x.length; i++) {
    out[i * 3] = x[i]
    out[i * 3 + 1] = y[i]
    out[i * 3 + 2] = z[i]
  }
  return out
}

function emptyResult(): LodMeshResult {
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
      minCellViolations: 0,
      skippedLeaves: 0,
      snappedLeaves: 0,
    },
  }
}

/** 循环左移，使 `start` 成为首元素（保持相对顺序 ⇒ 绕序不变）。 */
function rotateFrom(poly: readonly number[], start: number): number[] {
  const i = poly.indexOf(start)
  if (i < 0) return [...poly]
  return [...poly.slice(i), ...poly.slice(0, i)]
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
 * 大 cell（边长 > 64px）按 `stride` 抽样：全像素验差的代价是 `Σ 面积 × 三角数`，
 * 而超差只会发生在面内中部、不会只有单个像素。
 */
function polygonFanError(
  pts: readonly (readonly [number, number, number])[],
  disparity: ArrayLike<number>,
  width: number,
  bounds: { x0: number; y0: number; x1: number; y1: number },
  support: Uint8Array,
): number {
  if (pts.length < 3) return 0
  const span = Math.max(bounds.x1 - bounds.x0, bounds.y1 - bounds.y0)
  const step = span > 64 ? Math.ceil(span / 64) : 1
  let worst = 0
  for (let y = bounds.y0; y <= bounds.y1; y += step) {
    for (let x = bounds.x0; x <= bounds.x1; x += step) {
      const idx = y * width + x
      // 只验**有支撑**的像素：无支撑像素的视差没有意义（贴边 cell 会被判成天文数字）。
      if (!support[idx]) continue
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

/** 四角矩形按两种对角之一的扇形三角化误差（`regionClean` 的选择门用）。 */
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

function nextPow2(value: number): number {
  let p = 1
  while (p < value) p *= 2
  return p
}

function clampPow2(value: number, lo: number, hi: number): number {
  const v = Math.max(lo, Math.min(hi, nextPow2(Math.max(1, Math.floor(value)))))
  return v
}
