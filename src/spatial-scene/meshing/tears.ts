/**
 * 深度断层撕裂 —— 本模块的**唯一硬要求**。
 *
 * 阶段：meshing 的第二步（在支撑掩码之后，出三角形之前）。
 *
 * ── 为什么它是硬要求 ──
 * 不撕裂 ⇒ 四边形横跨断层 ⇒ 形成连接近面与远面的「橡皮布」大三角。
 * 斜视角下它把前景色拉到背景，产生最刺眼的 bleed / 鬼影 —— 这是**结构性错误**，
 * 无法用 α 容差 / 层范围 / 外扩掩盖。一旦撕裂，近面与远面成为独立 patch，
 * 露出的就是**真实的后层内容**（层 k+1），语义正确。
 *
 * ── 阈值为什么在视差域 ──
 * `n(z) = far/(far−near)·(1 − near/z)` 对 `1/z` 严格仿射。米制深度在远景被压缩：
 * 同样的「视觉台阶」在远处理论上对应巨大的米差，阈值要么漏撕要么乱撕。
 * 视差才是遮挡 / 外推误差的驱动量，所以：
 * ```
 * tearEps = clamp(tearScale · (boundaries[k+1] − boundaries[k]), min, max)
 * ```
 * 层带本身就是视差带，所以「层内台阶有多陡」天然以 band 宽度为尺度。
 *
 * ── 边表约定（两个方向各一张，互不重叠）──
 * - **横向边** `horizontal[y*(w−1)+x]`：像素 `(x,y)` 与 `(x+1,y)` 之间；
 * - **纵向边** `vertical[y*w+x]`：像素 `(x,y)` 与 `(x,y+1)` 之间（`y < h−1`）。
 * 两个端点都在支撑内且 `|Δn| > tearEps` 才置 1。缺角由出四边形时判（见 `relief.ts`）。
 *
 * ── despeckle（`remove_redundant_edge` 提炼）──
 * 孤立 1–2 px 的深度尖峰会撕出一条假缝。做法：把撕裂边按**共享像素**并成连通分量，
 * 分量边数 `< minTearSegmentLength` 的整段取消（= 重新连上）。
 * ⚠ 不能按「同一行/列的连续段长」过滤：45° 斜向断层是 H/V 交替的阶梯，
 * 每一步 run 都是 1，会被误杀。共享像素的连通分量才认得出阶梯。
 *
 * @see SOLIDI `mesh.py:tear_edges`（`depth_threshold` 断边）——
 *   vt-vl-lab/3d-photo-inpainting @ de04467 (MIT)；
 *   despeckle 提炼自同仓库 `mesh.py:remove_redundant_edge` / `judge_dangle`。
 */

import { ndcDepthFromZ } from "../layering/disparity-stats.ts"
import type { TearOptions } from "./types.ts"

/** 撕裂边表 + 诊断。 */
export interface TornEdges {
  /** `height * (width−1)`，1 = 被撕开。 */
  readonly horizontal: Uint8Array
  /** `(height−1) * width`，1 = 被撕开。 */
  readonly vertical: Uint8Array
  /** 实际生效的阈值（视差域）。 */
  readonly tearEps: number
  /** despeckle 之后的撕裂边数。 */
  readonly tornCount: number
  /** despeckle 取消掉的边数。 */
  readonly despeckledCount: number
}

const DEFAULT_TEAR_SCALE = 0.4
const DEFAULT_MIN_TEAR_DISPARITY = 0.002
const DEFAULT_MAX_TEAR_DISPARITY = 0.06
const DEFAULT_MIN_TEAR_SEGMENT = 6

/**
 * 由深度算视差场（视差域）。`depth<=0` 记 0（无效像素，不会进支撑）。
 *
 * 单独导出是为了让调用方（`index.ts`）只算一次：撕裂与统计共用同一张场。
 */
export function computeDisparityField(
  depth: ArrayLike<number>,
  near: number,
  far: number,
  out: Float32Array = new Float32Array(depth.length),
): Float32Array {
  for (let i = 0; i < depth.length; i++) {
    const z = depth[i]
    out[i] = z > 0 ? ndcDepthFromZ(z, near, far) : 0
  }
  return out
}

/** 解析出实际生效的撕裂阈值（视差域）。 */
export function resolveTearEps(
  bandWidth: number,
  options: TearOptions = {},
): number {
  const scale = options.tearScale ?? DEFAULT_TEAR_SCALE
  const min = options.minTearDisparity ?? DEFAULT_MIN_TEAR_DISPARITY
  const max = options.maxTearDisparity ?? DEFAULT_MAX_TEAR_DISPARITY
  const lo = Math.min(min, max)
  const hi = Math.max(min, max)
  const value = scale * Math.max(0, bandWidth)
  return value < lo ? lo : value > hi ? hi : value
}

/**
 * 算一层的撕裂边表 + despeckle。
 *
 * @param disparity `width*height`，**已换算到视差域**（见 `computeDisparityField`）。
 * @param support `width*height`，1 = 该像素出几何。
 */
export function computeTornEdges(
  disparity: ArrayLike<number>,
  support: Uint8Array,
  width: number,
  height: number,
  bandWidth: number,
  options: TearOptions = {},
): TornEdges {
  if (width < 2 || height < 2) {
    const horizontal = new Uint8Array(Math.max(0, height * (width - 1)))
    const vertical = new Uint8Array(Math.max(0, (height - 1) * width))
    return {
      horizontal,
      vertical,
      tearEps: resolveTearEps(bandWidth, options),
      tornCount: 0,
      despeckledCount: 0,
    }
  }
  const tearEps = resolveTearEps(bandWidth, options)
  const horizontal = new Uint8Array(height * (width - 1))
  const vertical = new Uint8Array((height - 1) * width)

  for (let y = 0; y < height; y++) {
    const row = y * (width - 1)
    const base = y * width
    for (let x = 0; x + 1 < width; x++) {
      const a = base + x
      const b = a + 1
      if (
        support[a] &&
        support[b] &&
        Math.abs(disparity[a] - disparity[b]) > tearEps
      ) {
        horizontal[row + x] = 1
      }
    }
  }
  for (let y = 0; y + 1 < height; y++) {
    const row = y * width
    const below = row + width
    for (let x = 0; x < width; x++) {
      const a = row + x
      const b = below + x
      if (
        support[a] &&
        support[b] &&
        Math.abs(disparity[a] - disparity[b]) > tearEps
      ) {
        vertical[row + x] = 1
      }
    }
  }

  const minSegment = Math.max(
    1,
    Math.floor(options.minTearSegmentLength ?? DEFAULT_MIN_TEAR_SEGMENT),
  )
  const despeckledCount = despeckleTornEdges(
    horizontal,
    vertical,
    width,
    height,
    minSegment,
  )

  let tornCount = 0
  for (let i = 0; i < horizontal.length; i++) tornCount += horizontal[i]
  for (let i = 0; i < vertical.length; i++) tornCount += vertical[i]

  return { horizontal, vertical, tearEps, tornCount, despeckledCount }
}

/**
 * 取消过短的撕裂连通分量。返回被取消的边数。
 *
 * 边 id 布局：`[0, hCount)` 是横向边，`[hCount, hCount+vCount)` 是纵向边。
 *
 * ── 连通判据必须是「端点 Chebyshev 距离 ≤ 1」，不能是「共享像素」──
 * 一条竖直断层的撕裂边是**横向**的：`(3,y)-(4,y)`、`(3,y+1)-(4,y+1)` …
 * 相邻两条**不共享像素**，只有端点 `(3,y)` 与 `(3,y+1)` 相距 1。
 * 若按「共享像素」合并，这条真实断层会被拆成 8 个单边分量，被 despeckle 全部误杀。
 * 45° 斜向断层是 H/V 交替的阶梯，同理。所以：对每条撕裂边的两个端点，取其 3×3
 * 邻域内的像素，把这些像素上的撕裂边 union 到一起。
 *
 * 单个孤立深度尖峰（1 px）产生 4 条绕它的边，它们共享该像素 ⇒ 分量大小 4；
 * 默认阈值 6 会把它清掉，而横跨整幅（h=8）的竖直断层分量大小 8，保留。
 */
function despeckleTornEdges(
  horizontal: Uint8Array,
  vertical: Uint8Array,
  width: number,
  height: number,
  minSegment: number,
): number {
  if (minSegment <= 1) return 0
  const hCount = horizontal.length
  const vCount = vertical.length
  const edgeCount = hCount + vCount
  if (edgeCount === 0) return 0

  // parent[id] = -1 表示该边未被撕裂；否则为并查集父指针。
  const parent = new Int32Array(edgeCount).fill(-1)
  for (let i = 0; i < hCount; i++) if (horizontal[i]) parent[i] = i
  for (let i = 0; i < vCount; i++)
    if (vertical[i]) parent[hCount + i] = hCount + i

  const find = (id: number): number => {
    let root = id
    while (parent[root] !== root) root = parent[root]
    while (parent[id] !== root) {
      const next = parent[id]
      parent[id] = root
      id = next
    }
    return root
  }
  const union = (a: number, b: number): void => {
    const ra = find(a)
    const rb = find(b)
    if (ra !== rb) parent[rb] = ra
  }

  const hId = (x: number, y: number): number => y * (width - 1) + x
  const vId = (x: number, y: number): number => hCount + y * width + x
  const isTorn = (id: number): boolean =>
    id >= 0 && id < edgeCount && parent[id] !== -1

  /** 边 id 的两个端点像素。 */
  const endpoints = (id: number): [number, number, number, number] => {
    if (id < hCount) {
      const y = Math.floor(id / (width - 1))
      const x = id - y * (width - 1)
      return [x, y, x + 1, y]
    }
    const k = id - hCount
    const y = Math.floor(k / width)
    const x = k - y * width
    return [x, y, x, y + 1]
  }

  /** 对给定端点像素，返回它落在 3×3 邻域内像素上的所有撕裂边 id。 */
  const neighborsOf = (px: number, py: number): number[] => {
    const out: number[] = []
    for (let ny = py - 1; ny <= py + 1; ny++) {
      if (ny < 0 || ny >= height) continue
      for (let nx = px - 1; nx <= px + 1; nx++) {
        if (nx < 0 || nx >= width) continue
        if (nx > 0) pushIfTorn(out, hId(nx - 1, ny))
        if (nx < width - 1) pushIfTorn(out, hId(nx, ny))
        if (ny > 0) pushIfTorn(out, vId(nx, ny - 1))
        if (ny < height - 1) pushIfTorn(out, vId(nx, ny))
      }
    }
    return out
  }
  const pushIfTorn = (out: number[], id: number): void => {
    if (isTorn(id)) out.push(id)
  }

  for (let id = 0; id < edgeCount; id++) {
    if (parent[id] === -1) continue
    const [ax, ay, bx, by] = endpoints(id)
    for (const [px, py] of [
      [ax, ay],
      [bx, by],
    ] as const) {
      for (const other of neighborsOf(px, py)) union(id, other)
    }
  }

  // 统计每个根的边数；先**收集**要取消的边，再统一置 -1。
  // （边扫边改 parent 会让后面的 find() 踩到已置 -1 的根，链被压成 0，漏清一条。）
  const sizes = new Map<number, number>()
  for (let id = 0; id < edgeCount; id++) {
    if (parent[id] === -1) continue
    const root = find(id)
    sizes.set(root, (sizes.get(root) ?? 0) + 1)
  }
  const toClear: number[] = []
  for (let id = 0; id < edgeCount; id++) {
    if (parent[id] === -1) continue
    if ((sizes.get(find(id)) ?? 0) < minSegment) toClear.push(id)
  }
  if (toClear.length === 0) return 0
  for (const id of toClear) parent[id] = -1
  const cleared = toClear.length

  for (let i = 0; i < hCount; i++) {
    if (horizontal[i] && parent[i] === -1) horizontal[i] = 0
  }
  for (let i = 0; i < vCount; i++) {
    if (vertical[i] && parent[hCount + i] === -1) vertical[i] = 0
  }
  return cleared
}
