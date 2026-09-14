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
 * ⚠ **但阈值随层数变**：层数一翻倍，band 宽度减半，同一场景的 `tearEps` 就跟着变。
 * 所以「换层数 / 换分层策略」的对照实验必须**先把这个口径锁死**（显式传 `tearEps` 或固定
 * `tearScale` + band 假设），否则测出来的差异分不清是 layering 的进步还是 meshing 换了标准。
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
 * ── 两道湅“中间小面片被撕掉”的保险 ──
 * 1. **视差场去噪**（`denoiseDisparity`，默认开）：撕裂判据用的 `depth` 是 α 加权混合结果、
 *    数值上不可靠，孤立毛刺会产生假撕；先做 3×3 中值（只在支撑内取样）⇒ 真断层不受影响。
 * 2. **小面片门**（`minPatchPixels`，默认 16）：torn 边从支撑图移除后，分量像素数小于阈值的，
 *    其边界 torn 边取消（重新连上）⇒ 治“中间孤立小三角被圈出去”。真断层两侧都是大分量，不命中。
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
  /** 小面片门重新连上的边数。 */
  readonly patchReconnected: number
}

const DEFAULT_TEAR_SCALE = 0.4
const DEFAULT_MIN_TEAR_DISPARITY = 0.002
const DEFAULT_MAX_TEAR_DISPARITY = 0.06
const DEFAULT_MIN_TEAR_SEGMENT = 6
const DEFAULT_MIN_PATCH_PIXELS = 16

/**
 * 3×3 中值（**只在支撑内取样**）：去掉 α 混合深度里的 1–2px 孤立毛刺。
 *
 * 用途：撕裂判据的输入。splat 的 `depth` 是 α 加权混合结果，数值上不可靠，
 * 孤立毛刺会产生假撕裂（把中间小面片撕掉）。中值是边缘保持的，所以真断层
 * （持续多像素）不受影响；**顶点位置仍用原始 `depth`，不受此滤波影响**。
 */
export function denoiseDisparity(
  disparity: ArrayLike<number>,
  support: Uint8Array,
  width: number,
  height: number,
): Float32Array {
  const out = Float32Array.from(disparity)
  const buf = new Float32Array(9)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x
      if (!support[i]) continue
      let n = 0
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy
        if (yy < 0 || yy >= height) continue
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx
          if (xx < 0 || xx >= width) continue
          const j = yy * width + xx
          if (!support[j]) continue
          buf[n++] = disparity[j]
        }
      }
      if (n < 3) continue // 支撑邻居太少：保留原值，不把孤立点抹平
      for (let a = 1; a < n; a++) {
        const v = buf[a]
        let b = a - 1
        while (b >= 0 && buf[b] > v) {
          buf[b + 1] = buf[b]
          b--
        }
        buf[b + 1] = v
      }
      out[i] =
        n % 2 === 1 ? buf[(n - 1) >> 1] : 0.5 * (buf[n / 2 - 1] + buf[n / 2])
    }
  }
  return out
}

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
  const scaled = scale * Math.max(0, bandWidth) * (options.thresholdScale ?? 1)
  return scaled < lo ? lo : scaled > hi ? hi : scaled
}

/**
 * 按**层地位**给撕裂选项加偏置：底部 `bottomFraction` 比例的层（背景）
 * 把撕裂阈值乘 `bottomScale`（>1 = 撕得更少）。
 *
 * 理由：背景层视差形变小，橡皮布不明显；但撕裂会留缝 ⇒ 露背景。
 * 所以底部层宁可少撕（换一点橡皮布）也不漏。默认 `0.25` / `2`；
 * 传 `layerBias: { bottomFraction: 0 }` 可关。
 */
export function resolveLayerTearOptions(
  options: TearOptions = {},
  layerIndex: number,
  layerCount: number,
): TearOptions {
  const bias = options.layerBias ?? {}
  const frac = bias.bottomFraction ?? 0.25
  const scale = bias.bottomScale ?? 2
  if (frac <= 0 || scale === 1 || layerCount <= 0) return options
  const threshold = (1 - frac) * layerCount
  if (layerIndex < threshold - 1e-9) return options
  return { ...options, thresholdScale: (options.thresholdScale ?? 1) * scale }
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
      patchReconnected: 0,
    }
  }
  const tearEps = resolveTearEps(bandWidth, options)
  // 撕裂判据用的视差场：可选中值去噪（顶点位置不用它）
  const field =
    options.denoise === false
      ? disparity
      : denoiseDisparity(disparity, support, width, height)
  const horizontal = new Uint8Array(height * (width - 1))
  const vertical = new Uint8Array((height - 1) * width)

  for (let y = 0; y < height; y++) {
    const row = y * (width - 1)
    const base = y * width
    for (let x = 0; x + 1 < width; x++) {
      const a = base + x
      const b = a + 1
      if (support[a] && support[b] && Math.abs(field[a] - field[b]) > tearEps) {
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
      if (support[a] && support[b] && Math.abs(field[a] - field[b]) > tearEps) {
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

  const minPatch = Math.max(
    1,
    Math.floor(options.minPatchPixels ?? DEFAULT_MIN_PATCH_PIXELS),
  )
  const patchReconnected = cancelSmallPatchTears(
    horizontal,
    vertical,
    support,
    width,
    height,
    minPatch,
  )

  let tornCount = 0
  for (let i = 0; i < horizontal.length; i++) tornCount += horizontal[i]
  for (let i = 0; i < vertical.length; i++) tornCount += vertical[i]

  return {
    horizontal,
    vertical,
    tearEps,
    tornCount,
    despeckledCount,
    patchReconnected,
  }
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

/**
 * 取消「只会圈出一小块支撑」的撕裂（`minPatchPixels` 门）。返回被重新连上的边数。
 *
 * ── 为什么需要它 ──
 * despeckle 按 **torn 边段长度**过滤（孤立毛刺边数少 ⇒ 清掉）。但它治不了
 * 「一条足够长的封闭/半封闭 tear 把**中间一小片面片**圈出去」——边数够了，
 * 却只剥离几个像素的几何，表现为网格中间的小洞。本函数补上这条。
 *
 * ── 判据 ──
 * 把 torn 边从支撑图里**移除**后做 4-连通分量：若某分量像素数 `< minPatchPixels`，
 * 则它被撕裂“孤立”了，把它的**边界 torn 边取消**（重新连上）。迭代到稳定
 * （合并后可能仍小于阈值）。
 *
 * ⚠ 真断层（两侧都是大片支撑）不会命中：两侧分量都 ≥ 阈值。
 */
function cancelSmallPatchTears(
  horizontal: Uint8Array,
  vertical: Uint8Array,
  support: Uint8Array,
  width: number,
  height: number,
  minPatchPixels: number,
): number {
  if (minPatchPixels <= 1 || width < 2 || height < 2) return 0
  const pixels = width * height
  const hStride = width - 1
  const label = new Int32Array(pixels).fill(-1)
  const stack = new Int32Array(pixels)
  const sizeOf: number[] = []
  let canceled = 0

  for (let pass = 0; pass < 8; pass++) {
    label.fill(-1)
    sizeOf.length = 0
    let cid = 0
    for (let seed = 0; seed < pixels; seed++) {
      if (!support[seed] || label[seed] !== -1) continue
      let size = 0
      let top = 0
      stack[top++] = seed
      label[seed] = cid
      while (top > 0) {
        const i = stack[--top]
        size++
        const x = i % width
        const y = (i - x) / width
        if (
          x > 0 &&
          support[i - 1] &&
          label[i - 1] === -1 &&
          !horizontal[y * hStride + x - 1]
        ) {
          label[i - 1] = cid
          stack[top++] = i - 1
        }
        if (
          x + 1 < width &&
          support[i + 1] &&
          label[i + 1] === -1 &&
          !horizontal[y * hStride + x]
        ) {
          label[i + 1] = cid
          stack[top++] = i + 1
        }
        if (
          y > 0 &&
          support[i - width] &&
          label[i - width] === -1 &&
          !vertical[(y - 1) * width + x]
        ) {
          label[i - width] = cid
          stack[top++] = i - width
        }
        if (
          y + 1 < height &&
          support[i + width] &&
          label[i + width] === -1 &&
          !vertical[y * width + x]
        ) {
          label[i + width] = cid
          stack[top++] = i + width
        }
      }
      sizeOf[cid] = size
      cid++
    }

    let passCanceled = 0
    for (let y = 0; y < height; y++) {
      for (let x = 0; x + 1 < width; x++) {
        const e = y * hStride + x
        if (!horizontal[e]) continue
        const a = y * width + x
        const b = a + 1
        if (!support[a] || !support[b]) continue
        if (
          sizeOf[label[a]] < minPatchPixels ||
          sizeOf[label[b]] < minPatchPixels
        ) {
          horizontal[e] = 0
          passCanceled++
        }
      }
    }
    for (let y = 0; y + 1 < height; y++) {
      for (let x = 0; x < width; x++) {
        const e = y * width + x
        if (!vertical[e]) continue
        const a = e
        const b = a + width
        if (!support[a] || !support[b]) continue
        if (
          sizeOf[label[a]] < minPatchPixels ||
          sizeOf[label[b]] < minPatchPixels
        ) {
          vertical[e] = 0
          passCanceled++
        }
      }
    }
    canceled += passCanceled
    if (passCanceled === 0) break
  }
  return canceled
}
