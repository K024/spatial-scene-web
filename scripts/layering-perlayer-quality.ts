/**
 * **逐层质量**判定工具：单层半 α 比例 + 单层**责任缺口**。
 *
 * 阶段：**layering 的逐层验收**（`layering-golden.ts` 之外的第二条判据）。
 *
 * ── 为什么需要它（与验收 A 并列）──
 * 下游**不做任何层混合**：每层各自转 mesh，靠视差去看**底层**露出来的像素。
 * 所以「逐层合成 ≡ 全量渲染」（验收 A）**不能**免除单层责任 ——
 * 栈级完整只说明信息没丢，不说明每一层自己能独立成网格。
 *
 * 合成与单层的差距是巨大的（`pier.ply` @768，`quantile`）：
 * ```
 *              合成        单层最坏（L=8）
 *   半 α       0.000%      13.07%
 *   责任缺口    0           5.13%（最坏层 21.25%）
 * ```
 * 所以判据必须落在**单层**上。
 *
 * ── 两个指标 ──
 * 1. **半 α 比例**（`0.05 <= α < 0.5`，直接读附件）——**硬数字**。
 *    无混合时 τ 是**硬表面提取阈值**，半 α 就是"这个像素到底有没有面"的拓扑不确定，
 *    mesh 会在那里做任意决定。
 * 2. **责任缺口**（`resp - own`）：本层是**前向累计 α 的归属层**（`argmax_k (1−A_{<k})·α_k`）
 *    但本层 `α < 0.5` 的像素。归属用 **α 累积**（参考视角精确值），**不看 wsplat 的 depth**。
 *
 * ⚠ 旧的「可修缝 / 长缝 / 宽空档」**面积分类已完全移除**（那套 tiny-hole 假设被
 * `layering/refine.ts` 的几何补齐取代：own-gap 回填 + hidden 有界外推）。
 * 本工具现在只回答「每层自己缺多少几何」，实际补全在 `refine.ts`。
 *
 * ── 分配策略对照（`--compare`）实测 ──
 * `pier.ply` @768，L=8，τ=0.5：
 * ```
 *   method       最坏层半α%   责任缺口%   合成ΔA
 *   quantile        13.07       5.132     7.37e-4
 *   errorDriven     10.62       5.162     6.06e-4
 * ```
 * ⇒ `errorDriven` 在 L=8 半 α 更低，但缺口不占优；前端偏置 `z_mean − k·σ_z`
 * 无效（见 `computeViewDepthStds`）。
 *
 * ── oracle 的定义与它的局限（务必读完再改）──
 * 责任归属 = `argmax_k (1 − A_{<k})·α_k`（前向累计 α 的边际贡献最大者）。
 * ⚠ **不用 wsplat 的 `depth` 输出**：它是 α 加权混合的期望深度，数值上不可靠；
 * 只有 α 累积能逐像素精确还原参考视角。
 *
 * **要把它变成硬数字，需要硬标签 oracle**：给渲染器加一个 label attachment
 * （逐像素「主导高斯属于哪层」）。在那之前，缺口率当**上界**用。
 *
 * ── 自检（不可删）──
 * `L = 1` 时单层持全部高斯，它的 α 图就是合成 α 图，**缺口必须恰好为 0**。
 * 这一行是本工具唯一的内建正确性证明。
 *
 * 用法：
 *   npx tsx scripts/layering-perlayer-quality.ts --ply py-models/out/ply/pier.ply --layers 1,2,4,8
 *   npx tsx scripts/layering-perlayer-quality.ts --width 1536 --method errorDriven
 *   # E4a 层分配策略对照（method × L × 前端偏置），主打「最坏层半 α / 责任缺口」：
 *   npx tsx scripts/layering-perlayer-quality.ts --compare --layers 1,2,4,8
 *   # E6 几何补齐报告（off / own / own+hidden 对照 + hidden 距离分布）：
 *   npx tsx scripts/layering-perlayer-quality.ts --ply .../example.ply --complete --max-dist 16
 */

import { parseArgs } from "node:util"

import {
  buildLayerPermutation,
  buildLayerPermutationFromAssignment,
  permuteNdcDepths,
} from "../src/spatial-scene/layering/bands.ts"
import {
  computeDisparityStats,
  ndcDepthFromZ,
} from "../src/spatial-scene/layering/disparity-stats.ts"
import {
  compositeAlphaDepth,
  computeLayerOwnership,
  type LayeredRGBD,
  refineLayers,
} from "../src/spatial-scene/layering/index.ts"
import { computeLayerPlacement } from "../src/spatial-scene/layering/placement.ts"
import type { LayerSamplingMethod } from "../src/spatial-scene/layering/types.ts"
import { rotationMatrixFromQuaternion } from "../src/spatial-scene/sharp/linalg.ts"
import type { Gaussians3D } from "../src/spatial-scene/sharp/types.ts"
import { createWSplatRenderer } from "../src/spatial-scene/wsplat/index.ts"
import {
  computeViewDepths,
  sortSplatsBackToFront,
} from "../src/spatial-scene/wsplat/sort.ts"
import type { WSplatFrame } from "../src/spatial-scene/wsplat/types.ts"
import { DEFAULT_IMAGE } from "./utils/common.ts"
import { loadImage } from "./utils/image.ts"
import { renderLayerStack } from "./utils/meshing-scene.ts"
import { withNodeDevice } from "./utils/webgpu.ts"
import { loadWSplatScene } from "./utils/wsplat-scene.ts"

const CLI = {
  width: { type: "string" },
  layers: { type: "string" },
  method: { type: "string" },
  ply: { type: "string" },
  /** oracle 的 α 门（默认 0.5；应该扫 0.5/0.95/0.99 看洞率有多敏感）。 */
  tau: { type: "string" },
  /** 前端偏置系数 `k`：归属深度用 `z_mean − k·σ_z`（默认 0 = 按均值硬分带）。 */
  bias: { type: "string" },
  /** `--compare` 模式下的偏置扫描序列。 */
  biases: { type: "string" },
  /** `--compare` 模式下的 method 序列。 */
  methods: { type: "string" },
  /** E4a 层分配策略对照（打印汇总表，不打印逐层明细）。 */
  compare: { type: "boolean" },
  /** E6 几何补齐报告（报告不判断）：off / own / own+hidden 对照。 */
  complete: { type: "boolean" },
  /** E6 用的原图（缺省 = `DEFAULT_IMAGE`）。 */
  image: { type: "string" },
  /** E6 hidden 外推的最大距离（px，默认 16）。 */
  "max-dist": { type: "string" },
} as const

/** 方形结构元的腐蚀（`dilate=false`）/ 膨胀（`dilate=true`）。 */
function morph(
  a: Float32Array,
  w: number,
  h: number,
  r: number,
  dilate: boolean,
): Float32Array {
  const out = new Float32Array(a.length)
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let best = dilate ? -Infinity : Infinity
      for (let dy = -r; dy <= r; dy++) {
        const yy = y + dy
        if (yy < 0 || yy >= h) continue
        for (let dx = -r; dx <= r; dx++) {
          const xx = x + dx
          if (xx < 0 || xx >= w) continue
          const v = a[yy * w + xx]
          if (dilate ? v > best : v < best) best = v
        }
      }
      out[y * w + x] = best
    }
  }
  return out
}

/**
 * 把有效深度用多趟膨胀填进无效区。
 *
 * 空洞内部 `D = 0`，而**合成空洞判据要算深度梯度**；用原场算会把所有候选都拒掉
 * （踩过的坑：表现为"任何分辨率都 0 个空洞"）。所以必须用「该处应有的表面」算梯度。
 */
function fillDepth(
  depth: Float32Array,
  w: number,
  h: number,
  passes: number,
): Float32Array {
  let z = Float32Array.from(depth)
  for (let p = 0; p < passes; p++) {
    const next = Float32Array.from(z)
    let filled = 0
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x
        if (z[i] > 0) continue
        let s = 0
        let c = 0
        for (const [dx, dy] of [
          [1, 0],
          [-1, 0],
          [0, 1],
          [0, -1],
        ]) {
          const xx = x + dx
          const yy = y + dy
          if (xx < 0 || xx >= w || yy < 0 || yy >= h) continue
          const v = z[yy * w + xx]
          if (v > 0) {
            s += v
            c++
          }
        }
        if (c > 0) {
          next[i] = s / c
          filled++
        }
      }
    }
    z = next
    if (filled === 0) break
  }
  return z
}

/**
 * 合成层洞（边缘安全）：`closing_r(A) − A > 0.2` 且填洞后相对深度梯度小。
 *
 * ⚠ 闭运算 = **先 dilate 再 erode**。反过来写是**开运算**，`closed − A <= 0` 恒成立，
 * 会静默地永远报「0 个空洞」——本项目就在这上面栽过一次。
 */ function compositeHoles(
  alpha: Float32Array,
  zFill: Float32Array,
  w: number,
  h: number,
  r: number,
): number {
  const closed = morph(morph(alpha, w, h, r, true), w, h, r, false)
  let n = 0
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x
      if (closed[i] - alpha[i] <= 0.2) continue
      const zl = zFill[i - 1]
      const zr = zFill[i + 1]
      const zu = zFill[i - w]
      const zd = zFill[i + w]
      if (!(zl > 0 && zr > 0 && zu > 0 && zd > 0)) continue
      const gx = Math.abs(zr - zl) / (zr + zl)
      const gy = Math.abs(zd - zu) / (zd + zu)
      if (Math.max(gx, gy) > 0.05) continue
      n++
    }
  }
  return n
}

/**
 * 每个高斯沿**相机 z 轴**的标准差 `σ_z`（米）。
 *
 * 各向异性高斯的协方差 `Σ = R·diag(s)²·Rᵀ`，所以沿单位方向 `f` 的方差
 * `σ_z² = fᵀ Σ f = Σ_i s_i² (f·col_i(R))²`。`f` 取视图矩阵第三行
 * （与 `computeViewDepths` 同一个 `(viewMatrix[2], [6], [10])`，防两套实现漂移）。
 *
 * 用途（E4a）：前端偏置归属 `z_mean − k·σ_z` 的 σ_z。
 * **实测结论：这条偏置是空操作**（负结果）—— `pier.ply` 的 σ_z 是 `p10 1e-4 /
 * p50 3e-4 / p99 3e-2 / max 7.5e-2 m`，而层带宽 0.06~58 m，`σ_z/带宽 ~ 1e-4~2e-3`，
 * `k ∈ {0,0.5,1}` 最坏层半 α 只动 ±0.02pp。机制：SHARP 高斯是**扁的**（沿视线薄、切向大），
 * σ_z 小但屏幕足迹大 ⇒ 前端偏置要按**切向 / 屏幕足迹**定义，不能按 σ_z。
 */
function computeViewDepthStds(
  gaussians: Gaussians3D,
  viewMatrix: ArrayLike<number>,
  count: number,
): Float32Array {
  const f0 = viewMatrix[2]
  const f1 = viewMatrix[6]
  const f2 = viewMatrix[10]
  const out = new Float32Array(count)
  const q = gaussians.quaternions
  const s = gaussians.singularValues
  for (let i = 0; i < count; i++) {
    const r = rotationMatrixFromQuaternion(
      q[i * 4],
      q[i * 4 + 1],
      q[i * 4 + 2],
      q[i * 4 + 3],
    )
    // 列 = 旋转后的三个主轴；`rotationMatrixFromQuaternion` 是行主序 9 元素
    const c0 = f0 * r[0] + f1 * r[3] + f2 * r[6]
    const c1 = f0 * r[1] + f1 * r[4] + f2 * r[7]
    const c2 = f0 * r[2] + f1 * r[5] + f2 * r[8]
    const s0 = s[i * 3]
    const s1 = s[i * 3 + 1]
    const s2 = s[i * 3 + 2]
    out[i] = Math.sqrt(
      s0 * s0 * c0 * c0 + s1 * s1 * c1 * c1 + s2 * s2 * c2 * c2,
    )
  }
  return out
}

/** 视差域值 `n` 落在哪个层带 `[boundaries[k], boundaries[k+1])` 里（`0` = 最近）。 */
function bandIndexForNdc(
  n: number,
  boundaries: ArrayLike<number>,
  L: number,
): number {
  let k = 0
  while (k < L - 1 && n >= boundaries[k + 1]) k++
  return k
}

/**
 * 到最近「本层有覆盖（`α >= 0.5`）」像素的近似欧氏距离（2-pass chamfer，`1 / √2`）。
 * 用于报告「hidden 缺失像素离本层覆盖面有多远」（调 `--max-dist` 用）。
 */
function distanceToCoverage(
  alpha: Float32Array,
  w: number,
  h: number,
): Float32Array {
  const INF = 1e9
  const out = new Float32Array(alpha.length)
  for (let i = 0; i < alpha.length; i++) out[i] = alpha[i] >= 0.5 ? 0 : INF
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x
      let v = out[i]
      if (x > 0) v = Math.min(v, out[i - 1] + 1)
      if (y > 0) v = Math.min(v, out[i - w] + 1)
      if (x > 0 && y > 0) v = Math.min(v, out[i - w - 1] + Math.SQRT2)
      if (x < w - 1 && y > 0) v = Math.min(v, out[i - w + 1] + Math.SQRT2)
      out[i] = v
    }
  }
  for (let y = h - 1; y >= 0; y--) {
    for (let x = w - 1; x >= 0; x--) {
      const i = y * w + x
      let v = out[i]
      if (x < w - 1) v = Math.min(v, out[i + 1] + 1)
      if (y < h - 1) v = Math.min(v, out[i + w] + 1)
      if (x < w - 1 && y < h - 1) v = Math.min(v, out[i + w + 1] + Math.SQRT2)
      if (x > 0 && y < h - 1) v = Math.min(v, out[i + w - 1] + Math.SQRT2)
      out[i] = v
    }
  }
  return out
}

async function main(): Promise<void> {
  const args = parseArgs({
    args: process.argv.slice(2),
    options: CLI,
    allowPositionals: false,
    strict: true,
  })
  const width = Number(args.values.width ?? 768)
  const method = (args.values.method ?? "quantile") as LayerSamplingMethod
  const bias = Number(args.values.bias ?? 0)
  const compare = args.values.compare === true
  const methods = (args.values.methods ?? "quantile,errorDriven")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean) as LayerSamplingMethod[]
  const biases = (args.values.biases ?? "0,0.5,1")
    .split(",")
    .map((s) => Number(s.trim()))
  const ls = (args.values.layers ?? "1,2,4,8")
    .split(",")
    .map((s) => Number(s.trim()))
    .filter((n) => Number.isInteger(n) && n > 0)
  const taus = (args.values.tau ?? "0.5,0.95,0.99").split(",").map(Number)

  await withNodeDevice(async (device) => {
    const scene = loadWSplatScene({
      width: String(width),
      plyPath: args.values.ply,
    })
    const { height } = scene
    const n = width * height
    const count = scene.gaussians.opacities.length
    const renderer = await createWSplatRenderer(device, {
      size: { width, height },
    })
    renderer.setGaussians(scene.gaussians)
    renderer.setCamera(scene.camera)
    renderer.sort()
    renderer.renderSplats()
    const full = await renderer.readback()

    const depth = computeViewDepths(
      scene.gaussians.meanVectors,
      scene.camera.viewMatrix,
      count,
    )
    const order = sortSplatsBackToFront(depth, count)
    const sortedNdc = permuteNdcDepths(depth, order, scene.near, scene.far)
    const stats = computeDisparityStats(
      depth,
      { near: scene.near, far: scene.far, binCount: 256 },
      scene.gaussians.opacities,
    )
    // E4a 前端偏置要 σ_z；只在真的用到时才算（1.18M 次四元数展开不便宜）。
    const needSigma = bias > 0 || biases.some((b) => b > 0)
    const sigma = needSigma
      ? computeViewDepthStds(scene.gaussians, scene.camera.viewMatrix, count)
      : undefined

    let fullCovered = 0
    let fullSemi = 0
    for (let i = 0; i < n; i++) {
      if (full.alpha[i] > 0.5) fullCovered++
      else if (full.alpha[i] >= 0.05) fullSemi++
    }
    console.log(
      `PLY ${(scene.plyPath ?? "example.ply").replace(/^.*[\\/]/, "")}  ${width}x${height}  ` +
        `near/far ${scene.near.toFixed(3)}/${scene.far.toFixed(3)}m`,
    )
    console.log(
      `  [合成] 覆盖 A>0.5 ${((fullCovered / n) * 100).toFixed(3)}%  ` +
        `半 α ${((fullSemi / n) * 100).toFixed(3)}%  ` +
        `合成空洞(r=2) ${compositeHoles(full.alpha, fillDepth(full.depth, width, height, 64), width, height, 2)}`,
    )

    /** 渲染 L 层（`k` = 前端偏置系数），返回放置结果与逐层 α。 */
    const renderLayers = async (
      L: number,
      m: LayerSamplingMethod,
      k: number,
    ): Promise<{
      placement: ReturnType<typeof computeLayerPlacement>
      layers: Float32Array[]
    }> => {
      const placement = computeLayerPlacement(stats, {
        L,
        method: m,
        near: scene.near,
        far: scene.far,
      })
      let perm: ReturnType<typeof buildLayerPermutation>
      if (k === 0 || !sigma) {
        // 默认路径：硬分带，层 = 全局序的一段连续区间（验收 A 的无损路径）。
        perm = buildLayerPermutation(order, sortedNdc, placement.boundaries)
      } else {
        // 前端偏置：归属用 `z_mean − k·σ_z`，而排序仍用均值深度。
        // 两者不再一致 ⇒ 同一层的高斯在全局序里不连续，必须筛选而不是切段。
        const bandOfPosition = new Int16Array(order.length)
        for (let i = 0; i < order.length; i++) {
          const splat = order[i]
          const zBias = Math.max(1e-4, depth[splat] - k * sigma[splat])
          const nb = ndcDepthFromZ(zBias, scene.near, scene.far)
          bandOfPosition[i] = bandIndexForNdc(nb, placement.boundaries, L)
        }
        perm = buildLayerPermutationFromAssignment(order, bandOfPosition, L)
      }
      renderer.setSplatOrder(perm.permutation)
      const layers: Float32Array[] = []
      for (let kk = 0; kk < L; kk++) {
        renderer.drawLayer(perm.table[kk])
        layers.push((await renderer.readback()).alpha)
      }
      renderer.renderSplats()
      return { placement, layers }
    }

    interface Entry {
      own: number
      semi: number
      solid: number
      resp: number[]
      hole: number[]
    }
    /** 逐层的「半 α / 责任缺口」统计（缺口 = 合成前表面落本层带但本层 α<0.5）。 */
    const layerEntries = (
      layers: readonly Float32Array[],
      boundariesZ: Float32Array,
      L: number,
    ): Entry[] => {
      const per: Entry[] = Array.from({ length: L }, () => ({
        own: 0,
        semi: 0,
        solid: 0,
        resp: taus.map(() => 0),
        hole: taus.map(() => 0),
      }))
      for (let k = 0; k < L; k++) {
        const lo = boundariesZ[k]
        const hi = boundariesZ[k + 1]
        const a = layers[k]
        const e = per[k]
        for (let i = 0; i < n; i++) {
          const av = a[i]
          if (av > 0.5) e.own++
          if (av >= 0.05 && av < 0.5) e.semi++
          if (av >= 0.95) e.solid++
          const fa = full.alpha[i]
          const d = full.depth[i]
          for (let t = 0; t < taus.length; t++) {
            if (!(fa > taus[t] && d >= lo && d < hi)) continue
            // 最后一个带含右端点，避免把 far 平面上的像素漏掉
            if (k === L - 1 && !(d <= hi)) continue
            e.resp[t]++
            if (av < 0.5) e.hole[t]++
          }
        }
      }
      return per
    }
    /** 逐层 α 的 over 合成 vs 全量：偏置会改写参考视角，这是它的代价。 */
    const compositeMetrics = (
      layers: readonly Float32Array[],
      L: number,
    ): { meanAbsA: number; flips: number } => {
      let sum = 0
      let flips = 0
      for (let i = 0; i < n; i++) {
        let t = 1
        for (let k = 0; k < L; k++) t *= 1 - layers[k][i]
        const ca = 1 - t
        sum += Math.abs(ca - full.alpha[i])
        if (ca > 0.5 !== full.alpha[i] > 0.5) flips++
      }
      return { meanAbsA: sum / n, flips }
    }

    if (compare) {
      console.log(
        `\n[E4a] 层分配策略对照（bias = 前端偏置 z_mean − k·σ_z；` +
          `缺口 = 合成前表面落本层带但本层 α<0.5）`,
      )
      console.log(
        "  method         L    k   最坏半α%  缺口%resp   合成ΔA    翻转px",
      )
      for (const m of methods) {
        for (const L of ls) {
          for (const k of biases) {
            const { placement, layers } = await renderLayers(L, m, k)
            const per = layerEntries(layers, placement.boundariesZ, L)
            let worstSemi = 0
            let resp0 = 0
            let hole0 = 0
            for (const e of per) {
              worstSemi = Math.max(worstSemi, e.semi)
              resp0 += e.resp[0]
              hole0 += e.hole[0]
            }
            const cm = compositeMetrics(layers, L)
            console.log(
              `  ${m.padEnd(13)} ${String(L).padStart(2)}  ${k.toFixed(2)}  ` +
                `${((worstSemi / n) * 100).toFixed(2).padStart(9)}%  ` +
                `${((hole0 / Math.max(1, resp0)) * 100).toFixed(3).padStart(7)}%  ` +
                `${cm.meanAbsA.toExponential(3).padStart(9)}  ` +
                `${String(cm.flips).padStart(8)}`,
            )
          }
        }
      }
      return
    }

    // E6 几何补齐（报告，不判断）
    if (args.values.complete === true) {
      const maxDist = Math.max(
        1,
        Math.floor(Number(args.values["max-dist"] ?? 16)),
      )
      const imagePath = args.values.image ?? DEFAULT_IMAGE
      const loaded = await loadImage(imagePath)
      const radii = [2, 4, 8, 16, 32, 64, 128]
      console.log(
        "\n[E6] 几何补齐（报告，不判断；责任缺口 = 本层是 α 归属层但本层 α<0.5）",
      )
      console.log(
        `     image=${imagePath.replace(/^.*[\\/]/, "")}  maxDist=${maxDist}px  ` +
          `own-gap=小洞填补(≤2px,两侧夹住)  ` +
          `hidden 有界外推（有界、带夹紧；报告不判断）`,
      )
      console.log(
        "  L  variant      责任缺口%   own填充  hidden填充  合成ΔA(mean)  翻转  深度越带",
      )
      for (const L of ls) {
        const base: LayeredRGBD = await renderLayerStack(device, scene, {
          layers: L,
          method,
        })
        const bz = base.placement.boundariesZ
        const before = base.frames
        // 参考分配的责任层（α 归属，不看 wsplat 的 depth）
        const refOwner = computeLayerOwnership(before, n).owner
        const measure = (
          frames: readonly WSplatFrame[],
        ): {
          gap: number
          ownFilled: number
          hiddenFilled: number
          meanDA: number
          flips: number
          depthOutOfBand: number
        } => {
          const comp = compositeAlphaDepth(frames, n)
          let respTotal = 0
          let gapPixels = 0
          for (let i = 0; i < n; i++) {
            if (!(comp.alpha[i] > 0.5)) continue
            const k = refOwner[i]
            if (k < 0) continue
            respTotal++
            if (frames[k].alpha[i] < 0.5) gapPixels++
          }
          let ownFilled = 0
          let hiddenFilled = 0
          let depthOutOfBand = 0
          for (let k = 0; k < L; k++) {
            const a0 = before[k].alpha
            const a1 = frames[k].alpha
            const d1 = frames[k].depth
            for (let i = 0; i < n; i++) {
              if (!(a1[i] >= 0.5) || a0[i] >= 0.5) continue
              if (!(full.alpha[i] > 0.5)) continue
              if (refOwner[i] === k) ownFilled++
              else hiddenFilled++
              if (!(d1[i] >= bz[k] - 1e-6 && d1[i] <= bz[k + 1] + 1e-6)) {
                depthOutOfBand++
              }
            }
          }
          let meanDA = 0
          let flips = 0
          for (let i = 0; i < n; i++) {
            meanDA += Math.abs(comp.alpha[i] - full.alpha[i])
            if (comp.alpha[i] > 0.5 !== full.alpha[i] > 0.5) flips++
          }
          return {
            gap: gapPixels / Math.max(1, respTotal),
            ownFilled,
            hiddenFilled,
            meanDA: meanDA / n,
            flips,
            depthOutOfBand,
          }
        }
        const variants: { name: string; complete: false | object }[] = [
          { name: "off", complete: false },
          { name: "own", complete: { hidden: false } },
          {
            name: "own+hidden",
            complete: { hidden: { maxDistancePx: maxDist, rgba: true } },
          },
          {
            name: "own+hid-geo",
            complete: { hidden: { maxDistancePx: maxDist, rgba: false } },
          },
        ]
        for (const v of variants) {
          const frames =
            v.complete === false
              ? before
              : refineLayers(base, loaded.image, {
                  complete: v.complete as never,
                }).frames
          const m = measure(frames)
          console.log(
            `  ${String(L).padStart(2)} ${v.name.padEnd(10)} ` +
              `${(m.gap * 100).toFixed(3).padStart(9)}%  ` +
              `${String(m.ownFilled).padStart(8)} ${String(m.hiddenFilled).padStart(10)}  ` +
              `${m.meanDA.toExponential(3).padStart(12)}  ${String(m.flips).padStart(5)}  ` +
              `${String(m.depthOutOfBand).padStart(7)}`,
          )
        }
        // hidden 缺失像素离本层覆盖的距离分布（调 `--max-dist` 用）。
        // 缺失 = 层 k 是参考 α 归属但本层未覆盖。
        const byRadius = radii.map(() => 0)
        let missAll = 0
        for (let k = 0; k < L; k++) {
          const dt = distanceToCoverage(before[k].alpha, width, height)
          const a = before[k].alpha
          for (let i = 0; i < n; i++) {
            if (refOwner[i] !== k || a[i] >= 0.5) continue
            missAll++
            for (let r = 0; r < radii.length; r++) {
              if (dt[i] <= radii[r]) byRadius[r]++
            }
          }
        }
        console.log(
          `     hidden 缺失按到本层覆盖距离 ≤R 累计：` +
            radii
              .map(
                (r, i) =>
                  `${r}:${((byRadius[i] / Math.max(1, missAll)) * 100).toFixed(1)}%`,
              )
              .join("  "),
        )
      }
      return
    }

    for (const L of ls) {
      const { placement, layers } = await renderLayers(L, method, bias)
      const per = layerEntries(layers, placement.boundariesZ, L)
      const totals = taus.map(() => [0, 0])
      console.log(
        `\n  L=${L}  method=${method}  bias=${bias}   层   本层α>0.5   占全幅   本层半α   本层实α  |` +
          taus.map((t) => `      该负责/洞/洞% (A>${t})`).join(""),
      )
      for (let k = 0; k < L; k++) {
        const e = per[k]
        for (let t = 0; t < taus.length; t++) {
          totals[t][0] += e.resp[t]
          totals[t][1] += e.hole[t]
        }
        console.log(
          `   ${String(k).padStart(3)}  ${String(e.own).padStart(9)}  ` +
            `${((e.own / n) * 100).toFixed(2).padStart(6)}%  ` +
            `${((e.semi / n) * 100).toFixed(2).padStart(7)}%  ` +
            `${((e.solid / n) * 100).toFixed(2).padStart(7)}%  |` +
            taus
              .map(
                (_, t) =>
                  `  ${String(e.resp[t]).padStart(7)}/${String(e.hole[t]).padStart(7)}/` +
                  `${((e.hole[t] / Math.max(1, e.resp[t])) * 100).toFixed(2).padStart(6)}%`,
              )
              .join(""),
        )
      }
      console.log(
        "   合计 " +
          taus
            .map(
              (t, i) =>
                `A>${t}: 负责 ${totals[i][0]} 洞 ${totals[i][1]} ` +
                `(${((totals[i][1] / Math.max(1, totals[i][0])) * 100).toFixed(3)}%)`,
            )
            .join("   "),
      )
      const cm = compositeMetrics(layers, L)
      console.log(
        `   合成 vs 全量：ΔA 均值 ${cm.meanAbsA.toExponential(3)}  跨 0.5 翻转 ${cm.flips} px`,
      )
      if (L === 1) {
        console.log(
          "   [自检] L=1 是单层持全部高斯 ⇒ 责任缺口必须恰好为 0。" +
            (totals.every(([, h]) => h === 0) ? " ✓" : " ✗ 检测器有问题"),
        )
      }
    }
  })
}

main().catch((err) => {
  console.error("\n[错误]", err instanceof Error ? err.message : err)
  if (err instanceof Error && err.stack) console.error(err.stack)
  process.exitCode = 1
})
