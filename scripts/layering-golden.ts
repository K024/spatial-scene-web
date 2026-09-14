/**
 * layering golden 门。
 *
 * 阶段：**splat -> 分层 RGBAD**（`src/spatial-scene/layering/`）。
 *
 * 这是**唯一**的判定入口：所有阈值写死在这里，退出码非 0 即「不过门」。
 * 曲线 / 出图由 `layering-render-views.ts` / `layering-perlayer-quality.ts` 负责，它们**不判定**
 *（避免两处阈值漂移）。
 *
 * ⚠ 验收口径（与下面的段对应）：**A 参考视角阈值级无损**（`A` 跳 τ=0.5 翻转的像素 = 0，
 * 不是逐位相等 —— 残差来自 `rgba16float` **累加附件**的分段舍入，∝ `L × f16_ulp(α)`，是固有代价）；
 * **B 外推只看目测**（无单值指标）；**C 元数据冻结**（`LayeredRGBD` ↔ MXIScene，见 `types.ts`）；
 * **D 逐层质量**（半 α / 可修缝，与 A 并列，看 `layering-perlayer-quality.ts`）。
 *
 * ── 跑什么 ──
 *   A0. 视差域自洽（纯 CPU）
 *       - `ndcDepthFromZ` / `zFromNdcDepth` 往返一致
 *       - 与 `wsplat/camera.ts` 的同一公式逐位一致（**防两套域实现漂移**）
 *   A1. 统计量正确性
 *       - 直方图反解分位数 vs 直接排序分位数（量化精度）
 *       - 一~四阶矩 vs 直接算的中心矩
 *   A2. 放置不变量（7 档 × L ∈ {4,8,16,32} × 5 种分布）
 *       - 层深 / 边界严格单调；`b[0]=0`、`b[L]=1`；质量之和 = 1
 *       - 确定性（两次调用逐位一致）
 *       - 非空层（`uniform` 除外：它是几何基线，数据集中时必然空层）
 *       - `frontWeighted` / `hybrid`：近半区跨度 < `uniform`（= 前密）
 *       - `errorDriven`：目标函数 **严格优于** `quantile`（这是它的存在理由）
 *   A3. 分带一致性
 *       - `overlap = 0`：各层计数之和 == 总数、区间连续、无重叠
 *       - 直方图质量 vs 实际计数（两条独立路径互验）
 *       - `overlap > 0`：每层计数不减少，且相邻层真的交叠
 *   A4. 真实数据（`example.ply`，1.18M 高斯）走一遍同一套不变量
 *   B.  参考视角无损（E2 / 验收 A）：硬分带 + 逐层 `over` 合成 == 全量渲染
 *       - 分带不重不漏；逐层剔除恒等式；`Σ drawn_k == drawn_full`（精确相等）
 *       - `A` / `ED` / `D` / 直通 RGB 四条分别比较（误差下限不同）
 *
 * ── 这些门**不**覆盖什么 ──
 * 1. 视角外推误差（gapRate / bleedRate / colorMAE）—— 见 E3。
 * 2. 哪个档位更好 —— 这里只验「实现是否正确」，不验「选择是否最优」。
 * 3. 密度补偿 / 原图回写（§2.5 / §2.6）—— 尚未实现。
 *
 * 用法：
 *   npx tsx scripts/layering-golden.ts                # 全跑（含 GPU 的 E2），最慢
 *   npx tsx scripts/layering-golden.ts --no-real      # 跳过 example.ply 读取
 *   npx tsx scripts/layering-golden.ts --no-gpu       # 跳过 E2 渲染；A 段全跑（含 A4）
 *   npx tsx scripts/layering-golden.ts --no-gpu --no-real   # 纯 CPU、不需要任何 fixture
 *   npx tsx scripts/layering-golden.ts --width 384    # E2 快速迭代
 *   npx tsx scripts/layering-golden.ts --ply py-models/out/ply/pier.ply
 *
 * `--no-gpu` 只关掉 E2，**A4 仍要读 fixture**（它是纯 CPU，但要 `example.ply`）；
 * 想完全不依赖 fixture 就再加 `--no-real`。
 *
 * `--ply` 换 fixture（默认 `example.ply`）。E2 的无损性与哪张图无关，
 * 但 A4 的分布指纹（视差占用、空层、前密顺序）会变，换图后要重新看一遍。
 */

import { parseArgs } from "node:util"
import {
  buildLayerPermutation,
  buildLayerPermutationFromAssignment,
  computeLayeredBands,
  computeLayerRanges,
  permuteNdcDepths,
  validateLayerPermutation,
} from "../src/spatial-scene/layering/bands.ts"
import {
  computeDisparityStats,
  ndcDepthFromZ,
  quantilesFromBins,
  zFromNdcDepth,
} from "../src/spatial-scene/layering/disparity-stats.ts"
import {
  bandMasses,
  bandMassFromSamples,
  computeLayerPlacement,
  quantizationError,
} from "../src/spatial-scene/layering/placement.ts"
import {
  blendImageWriteback,
  completeLayerGeometry,
  compositeAlphaDepth,
  computeLayerOwnership,
  computeOcclusionMasks,
  computeRefineWeight,
  layerPixelToImagePixel,
} from "../src/spatial-scene/layering/refine.ts"
import type {
  DisparityStats,
  LayerSamplingMethod,
} from "../src/spatial-scene/layering/types.ts"
import {
  viewDepthToNdcDepth,
  type WSplatCamera,
} from "../src/spatial-scene/wsplat/camera.ts"
import { createWSplatRenderer } from "../src/spatial-scene/wsplat/index.ts"
import {
  computeViewDepths,
  sortSplatsBackToFront,
} from "../src/spatial-scene/wsplat/sort.ts"
import type {
  WSplatFrame,
  WSplatStats,
} from "../src/spatial-scene/wsplat/types.ts"
import { numFlag } from "./utils/common.ts"
import { withNodeDevice } from "./utils/webgpu.ts"
import { loadWSplatScene } from "./utils/wsplat-scene.ts"

/**
 * 门限。取值原则：**实测值 × ~1.5 倍余量**（回归门，不是物理极限）。
 * 过不了门先查原因，不要直接放宽。
 */
const GATES = {
  /** 域变换往返 / 与 camera.ts 一致。 */
  domainAbs: 1e-9,
  /** 直方图分位数 vs 排序分位数的最大偏差（`binWidth = 1/256`）。 */
  quantileVsSorted: 0.01,
  /** 矩的最大相对误差。 */
  momentRel: 1e-6,
  /** 质量之和与 1 的偏差。 */
  massSum: 1e-5,
  /** 直方图质量 vs 实际计数 的最大绝对偏差（每层占比）。 */
  massVsCount: 0.03,
  /**
   * 真实数据上：`bandMasses`（直方图近似）vs `bandMassFromSamples`（精确）的最大偏差。
   *
   * 这是**回归门**，量的是直方图分辨率带来的误差。它存在的理由：旧实现把整箱质量
   * 按箱中心归属，而层带可以比一个箱还窄（远景被 `n` 压缩时必然发生），
   * 于是 `pier.ply` L=16 第 14 层**精确 24.17% 被报成 0.00%**。
   * 现已改成按重叠长度摊开，实测最坏 ~0.5%；门取 ~3 倍余量。
   */
  exactMass: 0.015,
  /** `quantile` 每层质量与 `1/L` 的最大偏差（等质量指纹）。 */
  quantileBalance: 0.15,
  /**
   * E2：合成 vs 全量。
   *
   * 取值原则仍是「实测 × ~1.5」，但**下限由实现决定**：
   * - `A` / `ED` 直接来自附件（`ED` 是 rgba32float），应该只差浮点噪声；
   * - 直通 RGB 要过 `resolve` 的**除 α + f16 往返**（`wgsl/resolve.ts`），
   *   所以它的下限是 f16 相对精度（~5e-4），不是 1e-6 量级。
   */
  /**
   * E2：合成 vs 全量。
   *
   * 取值原则仍是「实测 × ~1.5」，但**下限由实现决定**：
   * - `A` 本身直接从 f16 附件读，合成与全量只差**累加分段不同**，均值很小；
   * - `ED` 虽然是 f32，但合成权重 `Π(1−A_j)` 吃的是 f16 量化后的 `A`，
   *   所以误差 ∝ `L × f16_ulp(A)` ≈ `L × 2.4e-4`；实测 L=8..32 均值 1.6~2.0e-3。
   *   这是 **f16 累加附件的固有代价**（`rgba32float` 混合需要可选的
   *   `float32-blendable` feature，不可移植），不是分带出错。
   * 真正有语义的是 `A 不跨 τ=0.5 翻转`（实测 0 个像素）。
   */
  compositeAlphaMean: 1.5e-3,
  compositeEdMean: 3e-3,
  compositeRgb: 2e-3,
  compositeDepth: 1e-3,
} as const

const METHODS: readonly LayerSamplingMethod[] = [
  "uniform",
  "uniformNonEmpty",
  "quantile",
  "importance",
  "hybrid",
  "frontWeighted",
  "errorDriven",
]

const LAYER_COUNTS = [4, 8, 16, 32] as const

const NEAR = 0.2
const FAR = 20

/** 一条门的结论。 */
interface Check {
  name: string
  ok: boolean
  detail: string
}

const checks: Check[] = []

function addCheck(name: string, ok: boolean, detail: string): void {
  checks.push({ name, ok, detail })
  console.log(`  ${ok ? "✓" : "✗"} ${name.padEnd(34)} ${detail}`)
}

/**
 * 跨用例聚合一条不变量：只报「失败数 + 最差用例」，
 * 否则 7 档 × 4 层数 × 5 分布会刷出上百行没人看的输出。
 */
class Invariant {
  private okCount = 0
  private readonly failures: string[] = []
  readonly name: string
  constructor(name: string) {
    this.name = name
  }
  record(ok: boolean, label: string, detail: string): void {
    if (ok) this.okCount++
    else if (this.failures.length < 4) this.failures.push(`${label}: ${detail}`)
    else this.failures.push("")
  }
  report(): void {
    const failed = this.failures.filter((f) => f !== "").length
    const truncated = this.failures.includes("")
    const detail =
      failed === 0
        ? `${this.okCount} 个用例全通过`
        : `${failed} 个用例失败 —— ${this.failures.filter((f) => f !== "").join("; ")}${truncated ? " …" : ""}`
    addCheck(this.name, failed === 0, detail)
  }
}

// ────────────────────────────── 原图回写（E5，纯 CPU） ──────────────────────────────

/**
 * `refine.ts` 的不变量。回写是**纯颜色后处理**，所以能全部在 CPU 上钉死。
 * 这里**不**验「回写是否改善画面」（那是 E5 的实测，要原图 + GPU），只验实现正确。
 */
function refineChecks(): void {
  const w = 8
  const h = 8
  const px = w * h
  const visibleOn = new Uint8Array(px).fill(1)
  const alphaOne = new Float32Array(px).fill(1)
  const flatDepth = new Float32Array(px).fill(1)

  // 1) 门控：不可见 / α 低 / 背景层 / 被遮挡 → 权重 0；否则 ∈ [0,1]。
  {
    const alpha = Float32Array.from(alphaOne)
    alpha[0] = 0.2 // < minAlpha
    const visible = Uint8Array.from(visibleOn)
    visible[1] = 0
    const occluded = new Uint8Array(px)
    occluded[2] = 1
    const wgt = computeRefineWeight(
      { width: w, height: h, depth: flatDepth, alpha, visible },
      { occluded },
    )
    let range = true
    for (let i = 0; i < px; i++) if (wgt[i] < 0 || wgt[i] > 1) range = false
    refRange.record(
      range && wgt[0] === 0 && wgt[1] === 0 && wgt[2] === 0 && wgt[3] === 1,
      "门控 + 范围",
      `w[0]=${wgt[0]} w[1]=${wgt[1]} w[2]=${wgt[2]} w[3]=${wgt[3].toFixed(3)}`,
    )
    const off = computeRefineWeight(
      { width: w, height: h, depth: flatDepth, alpha, visible },
      { foreground: false },
    )
    let allZero = true
    for (let i = 0; i < px; i++) if (off[i] !== 0) allZero = false
    refRange.record(allZero, "背景层 ⇒ 全 0", "foreground=false")
  }

  // 2) 平滑因子：常量深度 w=1；有阶跃的处 w<1（梯度越大越小）。
  {
    const flat = computeRefineWeight({
      width: w,
      height: h,
      depth: flatDepth,
      alpha: alphaOne,
      visible: visibleOn,
    })
    let flatOne = true
    for (let i = 0; i < px; i++)
      if (Math.abs(flat[i] - 1) > 1e-6) flatOne = false
    refSmooth.record(flatOne, "常量深度 ⇒ w=1", `max=${Math.max(...flat)}`)

    const step = new Float32Array(px)
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) step[y * w + x] = x < w / 2 ? 1 : 2
    }
    const sw = computeRefineWeight({
      width: w,
      height: h,
      depth: step,
      alpha: alphaOne,
      visible: visibleOn,
    })
    // 边界两侧（x=w/2-1 与 x=w/2）的中心差分邻域跨阶跃
    const atEdge = Math.max(sw[3], sw[4])
    refSmooth.record(
      atEdge < 0.9 && sw[0] === 1,
      "跨阶跃处 w<1",
      `w[0]=${sw[0].toFixed(3)} max(边界)=${atEdge.toFixed(4)}`,
    )
  }

  // 3) 混合端点：w=0 → 层色逐位不变；w=1 → 原图色逐位相等；w=0.5 → 均值。
  {
    const rgb = Float32Array.from([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    const img = Float32Array.from([0.9, 0.8, 0.7, 0.6, 0.4, 0.2])
    const zero = new Float32Array([0, 0])
    const one = new Float32Array([1, 1])
    const half = new Float32Array([0.5, 0.5])
    const b0 = blendImageWriteback(rgb, img, zero)
    const b1 = blendImageWriteback(rgb, img, one)
    const b5 = blendImageWriteback(rgb, img, half)
    let ok0 = true
    let ok1 = true
    let ok5 = true
    for (let i = 0; i < rgb.length; i++) {
      if (Math.abs(b0[i] - rgb[i]) > 1e-7) ok0 = false
      if (Math.abs(b1[i] - img[i]) > 1e-7) ok1 = false
      if (Math.abs(b5[i] - (rgb[i] + img[i]) * 0.5) > 1e-7) ok5 = false
    }
    refBlend.record(
      ok0 && ok1 && ok5,
      "混合端点",
      `w0=${ok0} w1=${ok1} w.5=${ok5}`,
    )
  }

  // 4) 映射：同分辨率下恒等（参考相机 == 原图相机）。
  {
    let ok = true
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const p = layerPixelToImagePixel(x, y, w, h, w, h)
        if (Math.abs(p.x - x) > 1e-9 || Math.abs(p.y - y) > 1e-9) ok = false
      }
    }
    const scaled = layerPixelToImagePixel(0, 0, 8, 8, 16, 16)
    refMap.record(
      ok && scaled.x === 0.5 && scaled.y === 0.5,
      "同分辨率恒等 / 缩放",
      `scale(0,0)=(${scaled.x},${scaled.y})`,
    )
  }
}

// ────────────────────────────── 几何补齐（refine 扩展，纯 CPU） ──────────────────────────────

/** 造一个最小 `WSplatFrame`（全背景：α=0 / depth=0）。 */
function makeFrame(width: number, height: number): WSplatFrame {
  const px = width * height
  return {
    width,
    height,
    preview: new Uint8Array(px * 4),
    rgb: new Float32Array(px * 3),
    alpha: new Float32Array(px),
    depth: new Float32Array(px),
    transmission: new Float32Array(px).fill(1),
    accumulatedDepth: new Float32Array(px),
    visible: new Uint8Array(px),
  }
}

/** 在 `frame` 的像素 `i` 放一个单表面 `(z, α, rgb)`。 */
function setSurface(
  frame: WSplatFrame,
  i: number,
  z: number,
  a = 1,
  rgb: readonly [number, number, number] = [0.2, 0.4, 0.6],
): void {
  frame.alpha[i] = a
  frame.depth[i] = z
  frame.accumulatedDepth[i] = a * z
  frame.transmission[i] = 1 - a
  frame.visible[i] = 1
  frame.rgb[i * 3] = rgb[0]
  frame.rgb[i * 3 + 1] = rgb[1]
  frame.rgb[i * 3 + 2] = rgb[2]
}

/**
 * `refine` 的**几何补齐**不变量。全是纯 CPU：
 * 合成口径、遮挡掩码、own-gap 回填、hidden 有界外推（含带夹紧与限距）。
 */
function completionChecks(): void {
  const w = 8
  const h = 8
  const px = w * h
  const imageLinear = new Float32Array(px * 3).fill(0.9)
  const placement = {
    near: 1,
    far: 3,
    boundariesZ: new Float32Array([1, 2, 3]),
  }

  // 1) compositeAlphaDepth：远层 α=1/z=2.5，近层 α=0.5/z=1.5。
  //    far: ed=2.5, A=1；near: ed = 0.5*1.5 + 0.5*2.5 = 2.0, A = 1。⇒ D = 2.0。
  {
    const near = makeFrame(w, h)
    const far = makeFrame(w, h)
    for (let i = 0; i < px; i++) {
      setSurface(near, i, 1.5, 0.5)
      setSurface(far, i, 2.5, 1)
    }
    const c = compositeAlphaDepth([near, far], px)
    compOk.record(
      Math.abs(c.alpha[0] - 1) < 1e-6 && Math.abs(c.depth[0] - 2.0) < 1e-5,
      "合成 α / D",
      `A=${c.alpha[0].toFixed(4)} D=${c.depth[0].toFixed(4)}（期望 1 / 2）`,
    )
  }

  // 2) 遮挡掩码：层 0 不透明 ⇒ occluded[0]=0、occluded[1]=1。
  {
    const l0 = makeFrame(w, h)
    const l1 = makeFrame(w, h)
    setSurface(l0, 0, 1.5, 1)
    const occ = computeOcclusionMasks([l0, l1], px)
    occOk.record(
      occ[0][0] === 0 && occ[1][0] === 1 && occ[1][1] === 0,
      "逐层遮挡",
      `occ0[0]=${occ[0][0]} occ1[0]=${occ[1][0]} occ1[1]=${occ[1][1]}`,
    )
  }

  // 3) own-gap 小洞填补（α 归属 + 不外溢）：
  //    a) 洞：层 0 除 (3,3) 外都有支撑（α=1），洞处 α=0.4、层 1 同点 α=0.3。
  //       α 归属：w_0=0.4 > w_1=0.18，A_total=0.58>τ ⇒ 层 0 负责但偏低 ⇒ 应补上。
  //    b) 边缘外点：层 0 只在 x≤2 有支撑，(3,3) 也是最大贡献者，
  //       但外侧无支撑（不被夹住）⇒ **不应**补。
  {
    const hole = 3 * w + 3
    const l0 = makeFrame(w, h)
    const l1 = makeFrame(w, h)
    for (let i = 0; i < px; i++) setSurface(l0, i, 1.5, 1)
    setSurface(l0, hole, 1.5, 0.4)
    setSurface(l1, hole, 2.5, 0.3)
    const out = completeLayerGeometry([l0, l1], placement, imageLinear, w, h, {
      hidden: false,
    })
    const f = out[0]
    const holeOk =
      Math.abs(f.alpha[hole] - 1) < 1e-6 &&
      Math.abs(f.depth[hole] - 1.5) < 1e-5 &&
      Math.abs(f.rgb[hole * 3] - 0.9) < 1e-6

    const m0 = makeFrame(w, h)
    const m1 = makeFrame(w, h)
    for (let y = 0; y < h; y++) {
      for (let x = 0; x <= 2; x++) setSurface(m0, y * w + x, 1.5, 1)
    }
    setSurface(m0, hole, 1.5, 0.4)
    setSurface(m1, hole, 2.5, 0.3)
    const out2 = completeLayerGeometry([m0, m1], placement, imageLinear, w, h, {
      hidden: false,
    })
    const fringeOk = Math.abs(out2[0].alpha[hole] - 0.4) < 1e-6

    ownGapOk.record(
      holeOk && fringeOk,
      "own-gap 小洞填补（α 归属 + 边缘不外溢）",
      `洞 α=${f.alpha[hole]} D=${f.depth[hole]}；边缘 α=${out2[0].alpha[hole]}`,
    )
  }

  // 4) hidden dilate 外推：层 0（更近）盖住中心 2×2（带 0），层 1（更远）只盖住右侧 1×2（带 1，α=0.9）。
  //    层 1 应从自己的支撑向左侧被遮挡区外推：深度夹到本层带 [2,3]，限距 ≤2px；
  //    `rgba:true` 时 α = 源α·(1−d/(R+1))（距离衰减），rgb 本层自有优先（本层 α=0 ⇒ 取源色）。
  {
    const l0 = makeFrame(w, h)
    const l1 = makeFrame(w, h)
    const hiddenBlock: [number, number][] = [
      [3, 3],
      [4, 3],
      [3, 4],
      [4, 4],
    ]
    const supportBlock: [number, number][] = [
      [5, 3],
      [5, 4],
    ]
    for (const [x, y] of hiddenBlock) setSurface(l0, y * w + x, 1.5, 1)
    for (const [x, y] of supportBlock) setSurface(l1, y * w + x, 2.5, 0.9)
    const out = completeLayerGeometry([l0, l1], placement, imageLinear, w, h, {
      ownGap: false,
      hidden: { maxDistancePx: 2, smoothPasses: 1, rgba: true },
    })
    const f = out[1]
    let filled = 0
    let badDepth = 0
    let badAlpha = 0
    let tooFar = 0
    let nearAlpha = -1
    let farAlpha = -1
    const inSupport = (x: number, y: number): boolean =>
      supportBlock.some(([bx, by]) => bx === x && by === y)
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x
        if (!(f.alpha[i] > 1e-6) || inSupport(x, y)) continue
        filled++
        if (!(f.depth[i] >= 2 - 1e-6 && f.depth[i] <= 3 + 1e-6)) badDepth++
        // dilate：α 不超过源 α（0.9），且随距离单调不增
        if (f.alpha[i] > 0.9 + 1e-6) badAlpha++
        // rgb 本层自有优先：本层 α=0 ⇒ 应取源色 (0.2,0.4,0.6)
        if (Math.abs(f.rgb[i * 3] - 0.2) > 1e-6) badAlpha++
        let mdist = Infinity
        for (const [bx, by] of supportBlock) {
          mdist = Math.min(mdist, Math.abs(x - bx) + Math.abs(y - by))
        }
        if (mdist > 2) tooFar++
        if (mdist === 1) nearAlpha = f.alpha[i]
        if (mdist === 2) farAlpha = f.alpha[i]
      }
    }
    const falloffOk = nearAlpha > farAlpha && farAlpha > 0
    hiddenOk.record(
      filled >= 4 &&
        badDepth === 0 &&
        badAlpha === 0 &&
        tooFar === 0 &&
        falloffOk,
      "hidden dilate 外推 + 带夹紧 + 距离衰减 + 本层 rgb 优先",
      `填充 ${filled}，越带 ${badDepth}，越 α/色 ${badAlpha}，超距 ${tooFar}，近/远 α=${nearAlpha.toFixed(2)}/${farAlpha.toFixed(2)}`,
    )
  }

  // 5) hidden 只补几何（`rgba:false`）：同样的外推，但 α/rgb 保留原值，只写 depth。
  {
    const l0 = makeFrame(w, h)
    const l1 = makeFrame(w, h)
    const hiddenBlock: [number, number][] = [
      [4, 3],
      [3, 3],
    ]
    const supportBlock: [number, number][] = [[5, 3]]
    for (const [x, y] of hiddenBlock) setSurface(l0, y * w + x, 1.5, 1)
    for (const [x, y] of supportBlock) setSurface(l1, y * w + x, 2.5, 1)
    const out = completeLayerGeometry([l0, l1], placement, imageLinear, w, h, {
      ownGap: false,
      hidden: { maxDistancePx: 2, smoothPasses: 1, rgba: false },
    })
    const f = out[1]
    const i = 3 * w + 3
    const depthExtended =
      f.depth[i] > 0 && f.depth[i] >= 2 - 1e-6 && f.depth[i] <= 3 + 1e-6
    const appearanceKept =
      Math.abs(f.alpha[i] - l1.alpha[i]) < 1e-6 &&
      Math.abs(f.rgb[i * 3] - l1.rgb[i * 3]) < 1e-6
    hiddenGeoOk.record(
      depthExtended && appearanceKept && f.visible[i] === 1,
      "hidden 只补几何（保留原 α/rgb）",
      `d=${f.depth[i].toFixed(3)} α=${f.alpha[i]}${depthExtended && appearanceKept ? " ✓" : ""}`,
    )
  }

  // 6) α 归属：`owner = argmax_k (1−A_{<k})·α_k`；`frontBefore[k] = A_{<k}`（不看 wsplat depth）。
  {
    const f0 = makeFrame(w, h)
    const f1 = makeFrame(w, h)
    const i = 4 * w + 4
    f0.alpha[i] = 0.4
    f1.alpha[i] = 0.3
    const own = computeLayerOwnership([f0, f1], px)
    const caseA =
      own.owner[i] === 0 &&
      Math.abs(own.frontBefore[1][i] - 0.4) < 1e-6 &&
      Math.abs(own.totalAlpha[i] - 0.58) < 1e-6

    const g0 = makeFrame(w, h)
    const g1 = makeFrame(w, h)
    const j = 5 * w + 5
    g0.alpha[j] = 0.3
    g1.alpha[j] = 0.5
    const own2 = computeLayerOwnership([g0, g1], px)
    const caseB = own2.owner[j] === 1

    ownerOk.record(
      caseA && caseB,
      "α 归属 = argmax 边际贡献",
      `α=(0.4,0.3)→owner ${own.owner[i]}，A<1=${own.frontBefore[1][i].toFixed(2)}，Atotal=${own.totalAlpha[i].toFixed(2)}；α=(0.3,0.5)→owner ${own2.owner[j]}`,
    )
  }
}

// ────────────────────────────── 合成分布 ──────────────────────────────

interface Distribution {
  name: string
  /** 在 `n` 域采样（返回 `[0,1]`）。 */
  sample: (u: number, u2: number) => number
  /** 支撑是否「铺开」——尖峰 / 极小范围上没有「非空层」可言。 */
  spread: boolean
}

const DISTRIBUTIONS: readonly Distribution[] = [
  {
    name: "uniform-n",
    sample: (u) => u,
    spread: true,
  },
  {
    name: "front-biased",
    sample: (u) => Math.sqrt(u),
    spread: true,
  },
  {
    name: "bimodal",
    // 35% 前景（n≈0.15）+ 65% 远景（n≈0.6）
    sample: (u, u2) =>
      u < 0.35 ? 0.15 + (u2 - 0.5) * 0.1 : 0.6 + (u2 - 0.5) * 0.2,
    spread: true,
  },
  {
    name: "background-heavy",
    // 92% 挤在远景，8% 铺开 —— 直方图后段重尾
    sample: (u, u2) => (u < 0.92 ? 0.5 + u2 * 0.45 : u2),
    spread: true,
  },
  {
    name: "spike",
    sample: () => 0.5,
    spread: false,
  },
  {
    name: "tiny-range",
    sample: (_u, u2) => 0.5 + (u2 - 0.5) * 2e-6,
    spread: false,
  },
]

/** 确定性 LCG（同输入必须同输出，门要能复现）。 */
function makeRandom(seed: number): () => number {
  let state = seed >>> 0
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0
    return state / 0x100000000
  }
}

interface SyntheticSample {
  /** 真实度量深度（米）。 */
  z: Float32Array
  /** 对应的视差域值，供排序分位数参考实现使用。 */
  n: Float64Array
}

function makeSynthetic(
  distribution: Distribution,
  count: number,
  seed: number,
): SyntheticSample {
  const random = makeRandom(seed)
  const z = new Float32Array(count)
  const n = new Float64Array(count)
  for (let i = 0; i < count; i++) {
    const value = clamp01(distribution.sample(random(), random()))
    n[i] = value
    z[i] = zFromNdcDepth(value, NEAR, FAR)
  }
  return { z, n }
}

// ────────────────────────────── A0 / A1 ──────────────────────────────

function domainChecks(): void {
  console.log("\n[A0] 视差域自洽")
  let maxRoundTrip = 0
  for (let i = 0; i <= 200; i++) {
    const n = i / 200
    const z = zFromNdcDepth(n, NEAR, FAR)
    maxRoundTrip = Math.max(
      maxRoundTrip,
      Math.abs(ndcDepthFromZ(z, NEAR, FAR) - n),
    )
  }
  addCheck(
    "n -> z -> n 往返一致",
    maxRoundTrip < GATES.domainAbs,
    `max|Δ| = ${maxRoundTrip.toExponential(2)}`,
  )
  addCheck(
    "端点: n(near)=0, n(far)=1",
    ndcDepthFromZ(NEAR, NEAR, FAR) === 0 && ndcDepthFromZ(FAR, NEAR, FAR) === 1,
    `n(${NEAR}) = ${ndcDepthFromZ(NEAR, NEAR, FAR)}, n(${FAR}) = ${ndcDepthFromZ(FAR, NEAR, FAR)}`,
  )
  // 与 wsplat/camera.ts 的同一公式对拍：两处实现不许漂移
  const fakeCamera = { near: NEAR, far: FAR } as unknown as WSplatCamera
  let maxCameraDiff = 0
  for (let i = 1; i <= 200; i++) {
    const z = NEAR + ((FAR - NEAR) * i) / 200
    maxCameraDiff = Math.max(
      maxCameraDiff,
      Math.abs(
        viewDepthToNdcDepth(fakeCamera, z) - ndcDepthFromZ(z, NEAR, FAR),
      ),
    )
  }
  addCheck(
    "与 wsplat/camera.ts 公式一致",
    maxCameraDiff === 0,
    `max|Δ| = ${maxCameraDiff.toExponential(2)}（两处必须逐位相同）`,
  )
}

function statisticsChecks(
  sample: SyntheticSample,
  stats: DisparityStats,
): void {
  const count = sample.n.length
  const sorted = Float64Array.from(sample.n).sort()
  const probs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
  const histogram = quantilesFromBins(
    stats.bins,
    stats.binWidth,
    Float32Array.from(probs),
  )
  let maxQuantileDiff = 0
  for (let i = 0; i < probs.length; i++) {
    maxQuantileDiff = Math.max(
      maxQuantileDiff,
      Math.abs(histogram[i] - sortedQuantile(sorted, probs[i])),
    )
  }
  addCheck(
    "直方图分位数 vs 排序分位数",
    maxQuantileDiff < GATES.quantileVsSorted,
    `max|Δ| = ${maxQuantileDiff.toFixed(5)} < ${GATES.quantileVsSorted}（binWidth=${stats.binWidth.toFixed(5)}）`,
  )

  // 矩：直方图统计（流式原始矩）vs 直接计算的中心矩
  let mean = 0
  for (let i = 0; i < count; i++) mean += sample.n[i]
  mean /= count
  let m2 = 0
  let m3 = 0
  let m4 = 0
  for (let i = 0; i < count; i++) {
    const d = sample.n[i] - mean
    m2 += d * d
    m3 += d * d * d
    m4 += d * d * d * d
  }
  m2 /= count
  m3 /= count
  m4 /= count
  const rel = (a: number, b: number): number =>
    Math.abs(a - b) / Math.max(1e-12, Math.abs(b))
  const skew = m3 / m2 ** 1.5
  const kurt = m4 / m2 ** 2
  const maxMomentRel = Math.max(
    rel(stats.disparityMean, mean),
    rel(stats.disparityVariance, m2),
    rel(stats.disparitySkewness, skew),
    rel(stats.disparityKurtosis, kurt),
  )
  addCheck(
    "一~四阶矩一致",
    maxMomentRel < GATES.momentRel,
    `max 相对误差 = ${maxMomentRel.toExponential(2)}（均值 ${stats.disparityMean.toFixed(5)}，方差 ${stats.disparityVariance.toFixed(5)}，偏度 ${stats.disparitySkewness.toFixed(4)}，峰度 ${stats.disparityKurtosis.toFixed(4)}）`,
  )
  addCheck(
    "minDepth / maxDepth 与视差端点一致",
    Math.abs(stats.minDepth - zFromNdcDepth(stats.minimum, NEAR, FAR)) < 1e-9 &&
      Math.abs(stats.maxDepth - zFromNdcDepth(stats.maximum, NEAR, FAR)) < 1e-9,
    `z ∈ [${stats.minDepth.toFixed(4)}, ${stats.maxDepth.toFixed(4)}] m（n ∈ [${stats.minimum.toFixed(4)}, ${stats.maximum.toFixed(4)}]）`,
  )
}

// ────────────────────────────── A2 / A3 ──────────────────────────────

function placementChecks(
  distribution: Distribution,
  stats: DisparityStats,
  sortedNdc: Float32Array,
): void {
  const label = distribution.name
  for (const L of LAYER_COUNTS) {
    for (const method of METHODS) {
      const caseLabel = `${label}/L=${L}/${method}`
      const placement = computeLayerPlacement(stats, {
        L,
        method,
        near: NEAR,
        far: FAR,
      })
      const { layerDepths, boundaries, layerMass } = placement

      monotonicDepth.record(
        isStrictlyIncreasing(layerDepths),
        caseLabel,
        `depths[0..${L - 1}] = ${formatRange(layerDepths)}`,
      )
      monotonicBounds.record(
        isStrictlyIncreasing(boundaries) &&
          boundaries[0] === 0 &&
          boundaries[L] === 1,
        caseLabel,
        `b[0]=${boundaries[0]}, b[L]=${boundaries[L]}, ${formatRange(boundaries)}`,
      )
      let sum = 0
      for (let i = 0; i < L; i++) sum += layerMass[i]
      massSums.record(
        Math.abs(sum - 1) < GATES.massSum,
        caseLabel,
        `Σ = ${sum.toFixed(9)}`,
      )
      // 确定性：同输入必须逐位同输出
      const again = computeLayerPlacement(stats, {
        L,
        method,
        near: NEAR,
        far: FAR,
      })
      deterministic.record(
        sameF32(layerDepths, again.layerDepths) &&
          sameF32(boundaries, again.boundaries) &&
          sameF32(layerMass, again.layerMass),
        caseLabel,
        "两次调用逐位一致",
      )

      if (method !== "uniform" && distribution.spread) {
        let emptyCount = 0
        for (let i = 0; i < L; i++) if (layerMass[i] <= 0) emptyCount++
        nonEmpty.record(
          emptyCount === 0,
          caseLabel,
          `空层 ${emptyCount} 个（质量 ${formatRange(layerMass)}）`,
        )
      }

      if (method === "quantile" && distribution.spread) {
        let worst = 0
        for (let i = 0; i < L; i++) {
          worst = Math.max(worst, Math.abs(layerMass[i] - 1 / L))
        }
        quantileBalance.record(
          worst < GATES.quantileBalance,
          caseLabel,
          `max|mass - 1/L| = ${worst.toFixed(4)}`,
        )
      }

      // errorDriven 的存在理由：目标函数必须严格更优
      if (method === "errorDriven" && distribution.spread) {
        const challenger = computeLayerPlacement(stats, {
          L,
          method: "quantile",
          near: NEAR,
          far: FAR,
        })
        const mine = quantizationError(stats, layerDepths)
        const baseline = quantizationError(stats, challenger.layerDepths)
        errorDrivenWins.record(
          mine < baseline,
          caseLabel,
          `k-means ${mine.toExponential(3)} < quantile ${baseline.toExponential(3)}（改善 ${((1 - mine / baseline) * 100).toFixed(1)}%）`,
        )
      }

      // ── A3 分带 ──
      const bands = computeLayeredBands(sortedNdc, boundaries)
      let bandSum = 0
      let contiguous = bands.first[L - 1] === 0
      for (let i = 0; i < L; i++) {
        bandSum += bands.count[i]
        if (
          i < L - 1 &&
          bands.first[i] !== bands.first[i + 1] + bands.count[i + 1]
        ) {
          contiguous = false
        }
      }
      bandCoverage.record(
        bandSum === bands.total && contiguous,
        caseLabel,
        `Σcount = ${bandSum}/${bands.total}，连续 = ${contiguous}`,
      )
      if (distribution.spread) {
        let worstMass = 0
        for (let i = 0; i < L; i++) {
          worstMass = Math.max(
            worstMass,
            Math.abs(bands.count[i] / Math.max(1, bands.total) - layerMass[i]),
          )
        }
        massVsCount.record(
          worstMass < GATES.massVsCount,
          caseLabel,
          `max|count/total - layerMass| = ${worstMass.toFixed(4)}`,
        )
      }

      // 分配永远是**划分**（不允许重叠：`over` 不幂等，重复绘制会把 α 算两次），
      // 层间重叠只存在于**报告的 depthRange** 里（给下游 mesh / 剔除留余量）。
      const overlap = 0.05
      const ranges = computeLayerRanges(boundaries, overlap)
      let rangesOk = true
      let widened = 0
      for (let i = 0; i < L; i++) {
        const lo = ranges[i * 2]
        const hi = ranges[i * 2 + 1]
        if (!(lo >= 0 && hi <= 1 && hi > lo)) rangesOk = false
        if (lo < boundaries[i] || hi > boundaries[i + 1]) widened++
      }
      layerRanges.record(
        rangesOk && widened === L,
        caseLabel,
        `合法 = ${rangesOk}，被加宽的层数 = ${widened}/${L}`,
      )

      // 通用排列构造器（E4a 前端偏置用）在**连续归属**下必须与切段版逐位一致。
      // 两条独立路径（filter vs slice）互验：切段版是构造证明，filter 版是通用实现，
      // 一旦哪天切段版被改坏，这里会立刻报出来。
      {
        const identity = new Uint32Array(bands.total)
        for (let i = 0; i < bands.total; i++) identity[i] = i
        const bandOfPosition = new Int16Array(bands.total)
        for (let i = 0; i < bands.total; i++) {
          let k = 0
          while (k < L - 1 && sortedNdc[i] >= boundaries[k + 1]) k++
          bandOfPosition[i] = k
        }
        const sliced = buildLayerPermutation(identity, sortedNdc, boundaries)
        const filtered = buildLayerPermutationFromAssignment(
          identity,
          bandOfPosition,
          L,
        )
        let same = sliced.length === filtered.length
        if (same) {
          for (let i = 0; i < sliced.length && same; i++) {
            if (sliced.permutation[i] !== filtered.permutation[i]) same = false
          }
          for (let k = 0; k < L && same; k++) {
            if (
              sliced.table[k].base !== filtered.table[k].base ||
              sliced.table[k].count !== filtered.table[k].count
            ) {
              same = false
            }
          }
        }
        assignmentPermutation.record(
          same,
          caseLabel,
          `切段/筛选 长度 ${sliced.length}/${filtered.length}`,
        )
      }
    }
  }
}

// aggregated invariants
const monotonicDepth = new Invariant("层深严格递增")
const monotonicBounds = new Invariant("边界严格递增且覆盖 [0,1]")
const massSums = new Invariant("每层质量之和 = 1")
const deterministic = new Invariant("确定性（逐位一致）")
const nonEmpty = new Invariant("无空层（uniform 除外）")
const exactMass = new Invariant("直方图质量 = 精确质量（真实数据）")
const realNonEmpty = new Invariant("真实数据无空层（quantile 族）")
const quantileBalance = new Invariant("quantile 等质量")
const frontDensity = new Invariant(
  "frontWeighted < hybrid < quantile（近半区）",
)
const errorDrivenWins = new Invariant("errorDriven 目标函数更优")
const bandCoverage = new Invariant("分带覆盖全量且连续")
const massVsCount = new Invariant("直方图质量 vs 实际计数")
const layerRanges = new Invariant("层 depthRange 合法且被 overlap 加宽")
const assignmentPermutation = new Invariant(
  "通用排列构造 == 连续区间构造（同一归属）",
)
const refRange = new Invariant("回写权重门控 + 范围 ∈ [0,1]")
const refSmooth = new Invariant("回写平滑因子单调")
const refBlend = new Invariant("回写混合端点（w=0/1/0.5）")
const refMap = new Invariant("层↔原图映射（同分辨率恒等）")
const compOk = new Invariant("几何补齐：合成 α / D")
const occOk = new Invariant("几何补齐：逐层遮挡")
const ownGapOk = new Invariant("几何补齐：own-gap 回填")
const hiddenOk = new Invariant("几何补齐：hidden 有界外推")
const hiddenGeoOk = new Invariant("几何补齐：hidden depth-only")
const ownerOk = new Invariant("几何补齐：α 归属")

/**
 * 前密指纹：`frontWeighted` 的最激进、`quantile` 是等质量基线、`hybrid` 在两者之间。
 *
 * 判据是**近半区边界** `b[half]` 的大小：它越小，说明近半区占掉的视差跨度越窄
 * （= 单位视差上分到的层越多 = 越「前密」）。注意不能拿它跟 `uniform` 的 0.5 比：
 * `frontWeighted` 是在 **target 空间**前密，当前景分布本身就把中位数推远时
 * （如 `front-biased`），`hybrid` 的 `b[half]` 可以超过 0.5 而仍然比 `quantile` 靠前。
 */
function frontDensityChecks(
  distribution: Distribution,
  stats: DisparityStats,
): void {
  if (!distribution.spread) return
  for (const L of LAYER_COUNTS) {
    const half = Math.floor(L / 2)
    const at = (method: LayerSamplingMethod): number =>
      computeLayerPlacement(stats, { L, method, near: NEAR, far: FAR })
        .boundaries[half]
    const weighted = at("frontWeighted")
    const mixed = at("hybrid")
    const baseline = at("quantile")
    frontDensity.record(
      weighted < mixed && mixed < baseline,
      `${distribution.name}/L=${L}`,
      `b[${half}]: frontWeighted ${weighted.toFixed(4)} < hybrid ${mixed.toFixed(4)} < quantile ${baseline.toFixed(4)}`,
    )
  }
}

// ────────────────────────────── A4 真实数据 ──────────────────────────────

function realDataChecks(plyPath?: string): void {
  console.log(
    `\n[A4] 真实数据（${(plyPath ?? "example.ply").replace(/^.*[\\/]/, "")}）`,
  )
  const scene = loadWSplatScene({ plyPath })
  const count = scene.gaussians.opacities.length
  const depth = computeViewDepths(
    scene.gaussians.meanVectors,
    scene.camera.viewMatrix,
    count,
  )
  const stats = computeDisparityStats(
    depth,
    { near: scene.near, far: scene.far, binCount: 256 },
    scene.gaussians.opacities,
  )
  console.log(
    `      高斯 ${count}  渲染 ${scene.width}x${scene.height}  near/far ${scene.near.toFixed(3)}/${scene.far.toFixed(3)} m`,
  )
  console.log(
    `      α 加权视差: 均值 ${stats.disparityMean.toFixed(4)}  方差 ${stats.disparityVariance.toFixed(5)}  ` +
      `偏度 ${stats.disparitySkewness.toFixed(3)}  峰度 ${stats.disparityKurtosis.toFixed(3)}`,
  )
  console.log(
    `      分位: ${Array.from(stats.quantileProbs)
      .map((p, i) => `${p}=${stats.quantiles[i].toFixed(3)}`)
      .join("  ")}`,
  )
  console.log(
    `      深度范围 ${stats.minDepth.toFixed(3)} ~ ${stats.maxDepth.toFixed(3)} m（采样 ${stats.sampleSize}）`,
  )

  // 1.18M 个高斯 vs 原图像素：密度观测（不是门，但分层的一切都建在它上面）
  const nativePixels = scene.pose.width * scene.pose.height
  console.log(
    `      [密度观测] ${(count / nativePixels).toFixed(4)} 高斯/原图像素 ` +
      `(${scene.pose.width}x${scene.pose.height} = ${nativePixels} px)，` +
      `即 1 个高斯覆盖约 ${(nativePixels / count).toFixed(2)} 个像素`,
  )

  // 视差域重排（back-to-front，n 递减）：用与渲染同一条确定性 radix
  const order = sortSplatsBackToFront(depth, count)
  const sortedNdc = permuteNdcDepths(depth, order, scene.near, scene.far)

  let descending = true
  for (let i = 0; i + 1 < count; i++) {
    if (sortedNdc[i] < sortedNdc[i + 1]) descending = false
  }
  addCheck(
    "back-to-front 序列的 n 递减",
    descending,
    `n[0] = ${sortedNdc[0].toFixed(6)} >= n[last] = ${sortedNdc[count - 1].toFixed(6)}`,
  )

  // 用真实分布跑一遍默认档位，并复核不变量
  // NDC 序列（未排序即可，`bandMassFromSamples` 与顺序无关）
  const ndcAll = new Float64Array(count)
  for (let i = 0; i < count; i++) {
    ndcAll[i] = ndcDepthFromZ(depth[i], scene.near, scene.far)
  }
  let worstExactMass = 0
  let worstExactMassCase = ""
  for (const L of LAYER_COUNTS) {
    for (const method of METHODS) {
      const placement = computeLayerPlacement(stats, {
        L,
        method,
        near: scene.near,
        far: scene.far,
      })
      const bands = computeLayeredBands(sortedNdc, placement.boundaries)
      let bandSum = 0
      let emptyBands = 0
      for (let i = 0; i < L; i++) {
        bandSum += bands.count[i]
        if (bands.count[i] === 0) emptyBands++
      }
      // 分位族的层深必须落在**数据支撑**内 —— 落在外面的那部分 n 上没有样本，
      // 那个层必然空（旧实现就是这样在 `pier.ply` 上造出一个空层 + 隔壁 24% 的）。
      let withinSupport = true
      if (method !== "uniform") {
        for (let i = 0; i < L; i++) {
          const v = placement.layerDepths[i]
          if (v < stats.minimum - 1e-6 || v > stats.maximum + 1e-6) {
            withinSupport = false
          }
        }
      }
      const ok =
        isStrictlyIncreasing(placement.layerDepths) &&
        isStrictlyIncreasing(placement.boundaries) &&
        withinSupport &&
        bandSum === count
      realInvariants.record(
        ok,
        `${method}/L=${L}`,
        `Σcount = ${bandSum}，支撑内 = ${withinSupport}，${formatRange(placement.layerDepths)}`,
      )
      if (method !== "uniform") {
        realNonEmpty.record(
          emptyBands === 0,
          `${method}/L=${L}`,
          `空层 ${emptyBands}`,
        )
      }
      const exact = bandMassFromSamples(
        ndcAll,
        scene.gaussians.opacities,
        placement.boundaries,
      )
      let diff = 0
      for (let i = 0; i < L; i++) {
        diff = Math.max(diff, Math.abs(placement.layerMass[i] - exact[i]))
      }
      if (diff > worstExactMass) {
        worstExactMass = diff
        worstExactMassCase = `${method}/L=${L}`
      }
      exactMass.record(
        diff <= GATES.exactMass,
        `${method}/L=${L}`,
        `max|直方图 − 精确| = ${(diff * 100).toFixed(2)}%`,
      )
    }
  }
  console.log(
    `      [质量口径] 直方图 vs 精确最坏偏差 ${(worstExactMass * 100).toFixed(2)}% ` +
      `(${worstExactMassCase})；门限 ${(GATES.exactMass * 100).toFixed(1)}%`,
  )
  // 默认档的每层质量（供人读）
  const readout = computeLayerPlacement(stats, {
    L: 16,
    method: "quantile",
    near: scene.near,
    far: scene.far,
  })
  // 边界约定观测：中点法补边界时，「等质量」并不精确成立。
  // 这不是 bug，而是 `b_k = (s_{k-1}+s_k)/2` 的固有偏差：边界不是等质量切点。
  // 要让 `quantile` 精确等质量，得把边界改成分位切点、层深改成带中点（一次 A/B）。
  let worstBalance = 0
  for (let i = 0; i < readout.L; i++) {
    worstBalance = Math.max(
      worstBalance,
      Math.abs(readout.layerMass[i] - 1 / readout.L),
    )
  }
  console.log(
    `      [边界约定观测] quantile(L=16) 每层质量偏离 1/L 最多 ${(worstBalance * 100).toFixed(2)}% ` +
      `（相对 ${(worstBalance * 16 * 100).toFixed(0)}%）—— 中点边界不是等质量切点`,
  )
  const unused = 1 - (stats.maximum - stats.minimum)
  console.log(
    `      [视差域利用率] 数据只占 [${stats.minimum.toFixed(3)}, ${stats.maximum.toFixed(3)}]，` +
      `空掉 ${(unused * 100).toFixed(1)}% 的 [0,1] —— uniform 会在这一段浪费层`,
  )
  // 自研解到底赢多少：目标函数（加权平方重建误差）的直接对照，不靠肉眼比曲线
  console.log(
    "      [errorDriven vs quantile] 目标函数 Σw(n−s)²（归一化，越小越好）",
  )
  for (const L of LAYER_COUNTS) {
    const baseline = computeLayerPlacement(stats, {
      L,
      method: "quantile",
      near: scene.near,
      far: scene.far,
    })
    const challenger = computeLayerPlacement(stats, {
      L,
      method: "errorDriven",
      near: scene.near,
      far: scene.far,
    })
    const base = quantizationError(stats, baseline.layerDepths)
    const mine = quantizationError(stats, challenger.layerDepths)
    console.log(
      `        L=${String(L).padStart(2)}  quantile ${base.toExponential(3)}  ->  ` +
        `errorDriven ${mine.toExponential(3)}  （降低 ${((1 - mine / base) * 100).toFixed(1)}%）`,
    )
  }
  console.log(
    `      [L=16 quantile] 层深(n)= ${formatRange(readout.layerDepths)}\n` +
      `                     层深(m)= ${Array.from(readout.layerDepthsZ)
        .map((v) => v.toFixed(2))
        .join(" ")}\n` +
      `                     质量   = ${Array.from(readout.layerMass)
        .map((v) => (v * 100).toFixed(1))
        .join(" ")} (%)`,
  )
}

const realInvariants = new Invariant("真实数据不变量")

// ────────────────────── B. 参考视角无损（E2 / 验收 A） ──────────────────────

/** E2 用到的档位：只需边界位置不同（无损性与档位无关，但不同边界会走不同区间）。 */
const E2_METHODS: readonly LayerSamplingMethod[] = [
  "uniform",
  "quantile",
  "errorDriven",
]

/**
 * E2：硬分带 + 逐层渲染的 `over` 合成必须与全量渲染**恒等**。
 *
 * 这是整个分层计划的基石：如果这条不成立，「用 L 个平面代表整场」就没有基准点，
 * E3 的所有外推数字都无从解释。
 *
 * ── 为什么分三组量分别比较 ──
 * 1. `alpha` / `accumulatedDepth`：直接来自附件，不经除法 → 最干净的无损性证据；
 * 2. `depth = ED/A`：多一次除法，看它是否引入偏差；
 * 3. 直通 RGB：经 `resolve` 的**除 α + f16 直通值**往返（见 `wgsl/resolve.ts`），
 *    所以它的误差下限是 f16 相对精度（~5e-4），不是浮点噪声。门限据此而定。
 *
 * ── 逐层剔除统计为什么是硬要求 ──
 * `minPixelSize` 是**像素量纲**且静默：某一层可能什么都没画出来而没人知道。
 * `renderSplatsRange()` 会把区间记下来，所以 `countCulls()` 数的是**本层那一段**。
 * 于是 `Σ drawn_k == drawn_full` 必须**精确相等**（切区间不改变每个高斯的剔除判据），
 * 这是比「看起来有东西」强得多的交叉验证。
 */
async function losslessChecks(width: number): Promise<void> {
  await withNodeDevice(async (device) => {
    console.log(`\n[B] 参考视角无损 E2 / 验收 A（渲染宽 ${width}）`)
    const scene = loadWSplatScene({ width: String(width) })
    const { height } = scene
    const count = scene.gaussians.opacities.length
    const renderer = await createWSplatRenderer(device, {
      size: { width, height },
    })
    renderer.setGaussians(scene.gaussians)
    renderer.setCamera(scene.camera)
    renderer.sort()

    // 全量渲染（参考）
    renderer.renderSplats()
    const full = await renderer.readback()
    const fullDigest = frameDigest(full)
    renderer.countCulls()
    const fullStats = await renderer.readSplatStats()

    // 分层需要的排序（与 `renderer.sort()` 同一条确定性 radix）
    const depth = computeViewDepths(
      scene.gaussians.meanVectors,
      scene.camera.viewMatrix,
      count,
    )
    const order = sortSplatsBackToFront(depth, count)
    const sortedNdc = permuteNdcDepths(depth, order, scene.near, scene.far)
    // 按**高斯下标**索引的视差域值（不是排列），供 `validateLayerPermutation` 用
    const ndcBySplat = new Float32Array(count)
    for (let i = 0; i < count; i++) {
      ndcBySplat[i] = ndcDepthFromZ(depth[i], scene.near, scene.far)
    }
    const stats = computeDisparityStats(
      depth,
      { near: scene.near, far: scene.far, binCount: 256 },
      scene.gaussians.opacities,
    )

    console.log(
      `      高斯 ${count}  全量剔除: 绘制 ${fullStats.drawn}  剔除 ${fullStats.total - fullStats.drawn}`,
    )
    console.log(
      `      [层浪费观测] uniform 把层铺满整个 [0,1]，而数据只占 [${stats.minimum.toFixed(3)}, ${stats.maximum.toFixed(3)}] ` +
        `—— 下面是每档位的空层数（空层在 mesh 阶段是纯浪费）`,
    )

    for (const L of LAYER_COUNTS) {
      for (const method of E2_METHODS) {
        const placement = computeLayerPlacement(stats, {
          L,
          method,
          near: scene.near,
          far: scene.far,
        })
        const caseLabel = `L=${L}/${method}`

        // 1) 造排列 + 层表（排列的所有权归 layering，不再是“全局排序的一段区间”）
        const layerPermutation = buildLayerPermutation(
          order,
          sortedNdc,
          placement.boundaries,
        )

        // 强门：分配是**划分**且从全局序抽取，所以拼接后必须**逐位等于**全局序。
        // 这比“逐层 count 之和 == N”强得多：它同时锁住了顺序、不重不漏、层归属。
        let identical = layerPermutation.length === count
        for (let i = 0; identical && i < count; i++) {
          if (layerPermutation.permutation[i] !== order[i]) identical = false
        }
        permutationIdentity.record(
          identical,
          caseLabel,
          `排列 ${layerPermutation.length} 项 vs 全局序 ${count}；逐位相同 ${identical}`,
        )
        // 弱门（密度补偿引任意下标时才是主要保障）：下标合法 + n 递减
        const validity = validateLayerPermutation(
          layerPermutation.permutation,
          ndcBySplat,
        )
        permutationValidity.record(
          validity.outOfRange.length === 0 && validity.descent.length === 0,
          caseLabel,
          `越界 ${validity.outOfRange.length} 处，逆序 ${validity.descent.length} 处`,
        )
        // 层表必须按**绘制顺序（远 -> 近）**首尾相接铺满排列
        let cursor = 0
        let tableCoverageOk = true
        for (let k = L - 1; k >= 0; k--) {
          const entry = layerPermutation.table[k]
          if (entry.base !== cursor) tableCoverageOk = false
          cursor += entry.count
        }
        tableCoverage.record(
          tableCoverageOk && cursor === layerPermutation.length,
          caseLabel,
          `按远->近铺满 ${cursor}/${layerPermutation.length}`,
        )

        // 2) 换掉整个排列，然后逐层绘制 + 逐层剔除统计
        renderer.setSplatOrder(layerPermutation.permutation)
        const layerFrames: WSplatFrame[] = []
        let drawnSum = 0
        let cullIdentityOk = true
        for (let k = 0; k < L; k++) {
          renderer.drawLayer(layerPermutation.table[k])
          layerFrames.push(await renderer.readback())
          renderer.countCulls()
          const s = await renderer.readSplatStats()
          drawnSum += s.drawn
          if (
            s.total !== layerPermutation.table[k].count ||
            !statsIdentity(s)
          ) {
            cullIdentityOk = false
          }
        }
        cullIdentity.record(
          cullIdentityOk && drawnSum === fullStats.drawn,
          caseLabel,
          `Σdrawn = ${drawnSum} vs 全量 ${fullStats.drawn}；逐层 drawn+剔除==total ${cullIdentityOk}`,
        )
        // 换完排列后整张排列画一遍，必须仍与原始全量**逐位一致**
        // （同时验证 setSplatOrder + permutationLength 的边界处理）
        renderer.renderSplats()
        const resent = await renderer.readback()
        permutationRenderIdentical.record(
          frameDigest(resent) === fullDigest,
          caseLabel,
          `重排后全量指纹 ${frameDigest(resent)} vs 原全量 ${fullDigest}`,
        )

        // 3) over 合成（远 -> 近）并与全量对比
        //    注意像素数 = width*height，**不是**高斯数 —— 别把两者搞混。
        const result = compositeAndCompare(layerFrames, full, width * height)
        alphaMeanLossless.record(
          result.meanAlphaErr < GATES.compositeAlphaMean,
          caseLabel,
          `平均 |ΔA| = ${result.meanAlphaErr.toExponential(3)}  最大 ${result.maxAlphaErr.toExponential(3)}`,
        )
        alphaFlipFree.record(
          result.alphaFlips === 0,
          caseLabel,
          `A 跨 τ=0.5 翻转的像素 ${result.alphaFlips} 个`,
        )
        edMeanLossless.record(
          result.meanEdRelOpaque < GATES.compositeEdMean,
          caseLabel,
          `不透明区 平均相对 |ΔED| = ${result.meanEdRelOpaque.toExponential(3)}  最大 ${result.maxEdRel.toExponential(3)}`,
        )
        rgbLossless.record(
          result.rgbMae < GATES.compositeRgb,
          caseLabel,
          `直通 RGB MAE = ${result.rgbMae.toExponential(3)}（不透明区）`,
        )
        depthLossless.record(
          result.depthMedianRel < GATES.compositeDepth,
          caseLabel,
          `D 中位相对误差 = ${(result.depthMedianRel * 100).toExponential(2)}%`,
        )
        layerEmptiness.record(
          method === "uniform" || result.emptyLayers === 0,
          caseLabel,
          method === "uniform"
            ? `${result.emptyLayers}/${L} 层空（本档位允许，几何基线）`
            : `${result.emptyLayers} 层完全没画出任何像素（A>0 的像素数为 0）`,
        )
      }
    }
  })
}

const permutationIdentity = new Invariant("排列逐位等于全局序（强门）")
const permutationValidity = new Invariant("排列下标合法且 n 递减（弱门）")
const tableCoverage = new Invariant("层表首尾相接铺满排列")
const permutationRenderIdentical = new Invariant("换排列后全量渲染逐位不变")
const cullIdentity = new Invariant("逐层剔除恒等式 + Σdrawn == 全量")
const alphaMeanLossless = new Invariant("合成 A 无损（均值）")
const alphaFlipFree = new Invariant("A 不跨 τ=0.5 翻转")
const edMeanLossless = new Invariant("合成 ED 无损（不透明区均值）")
const rgbLossless = new Invariant("合成直通 RGB 无损")
const depthLossless = new Invariant("合成深度 D 无损")
const layerEmptiness = new Invariant("无「画不出任何像素」的层")

/** 帧的确定性指纹（逐位；用于「换排列后是否还一样」）。 */
function frameDigest(frame: WSplatFrame): string {
  let h = 0x811c9dc5
  const mix = (values: Uint8Array | Float32Array): void => {
    const bytes =
      values instanceof Uint8Array
        ? values
        : new Uint8Array(values.buffer, values.byteOffset, values.byteLength)
    for (let i = 0; i < bytes.length; i++) {
      h ^= bytes[i]
      h = Math.imul(h, 0x01000193) >>> 0
    }
  }
  mix(frame.rgb)
  mix(frame.alpha)
  mix(frame.depth)
  mix(frame.accumulatedDepth)
  return h.toString(16).padStart(8, "0")
}

/** 剔除统计的自检恒等式（与 `wsplat-golden.ts` 同一条）。 */
function statsIdentity(s: WSplatStats): boolean {
  return (
    s.drawn +
      s.culledBounds +
      s.culledAlphaClip +
      s.culledAlphaClipAfterAa +
      s.culledBehindCamera +
      s.culledMinPixelSize +
      s.culledFrustum ===
    s.total
  )
}

interface CompositeResult {
  maxAlphaErr: number
  meanAlphaErr: number
  /** A 跨越 τ=0.5 的像素数（阈值判定翻转）—— 直接对应 mesh 会不会多/少一个面。 */
  alphaFlips: number
  maxEdRel: number
  /** 只在 `A > 0.5` 的像素上算的相对误差均值（避开低信号像素的相对误差爆炸）。 */
  meanEdRelOpaque: number
  rgbMae: number
  depthMedianRel: number
  emptyLayers: number
}

/**
 * 把各层按 `over` 从远到近合成，并与全量渲染逐像素对比。
 *
 * 合成必须在**预乘**空间做：`C_k = rgb_k · A_k`（`rgba` 附件存的是直通值，
 * 见 `wgsl/resolve.ts`）；`over` 为 `C = C_k + (1−A_k)·C`，
 * `A = A_k + (1−A_k)·A`，`ED` 同理。因为各层不重不漏且顺序与全量一致，
 * 结果必须与全量渲染恒等。
 */
function compositeAndCompare(
  layers: readonly WSplatFrame[],
  full: WSplatFrame,
  count: number,
): CompositeResult {
  const L = layers.length
  const alpha = new Float64Array(count)
  const color = new Float64Array(count * 3)
  const ed = new Float64Array(count)

  // 逐层是否画出了任何像素（A > 0）
  let emptyLayers = 0
  for (const frame of layers) {
    let any = false
    for (let i = 0; i < count && !any; i++) if (frame.alpha[i] > 0) any = true
    if (!any) emptyLayers++
  }

  for (let k = L - 1; k >= 0; k--) {
    const frame = layers[k]
    for (let i = 0; i < count; i++) {
      const a = frame.alpha[i]
      const w = 1 - a
      color[i * 3] = frame.rgb[i * 3] * a + w * color[i * 3]
      color[i * 3 + 1] = frame.rgb[i * 3 + 1] * a + w * color[i * 3 + 1]
      color[i * 3 + 2] = frame.rgb[i * 3 + 2] * a + w * color[i * 3 + 2]
      ed[i] = frame.accumulatedDepth[i] + w * ed[i]
      alpha[i] = a + w * alpha[i]
    }
  }

  let maxAlphaErr = 0
  let meanAlphaErr = 0
  let alphaFlips = 0
  let maxEdRel = 0
  let edRelOpaqueSum = 0
  let edRelOpaqueCount = 0
  let rgbSum = 0
  let rgbCount = 0
  const depthRels: number[] = []
  for (let i = 0; i < count; i++) {
    const dAlpha = Math.abs(alpha[i] - full.alpha[i])
    maxAlphaErr = Math.max(maxAlphaErr, dAlpha)
    meanAlphaErr += dAlpha
    // 阈值翻转：合成与全量在 τ=0.5 两侧不一致
    //（直接对应 mesh 会不会多/少一个面，比 max 误差有意义得多）
    if (alpha[i] > 0.5 !== full.alpha[i] > 0.5) alphaFlips++
    const edScale = Math.max(Math.abs(full.accumulatedDepth[i]), 1e-6)
    const edRel = Math.abs(ed[i] - full.accumulatedDepth[i]) / edScale
    maxEdRel = Math.max(maxEdRel, edRel)
    // 相对误差只在有信号的像素上统计：背景处 ED→0，相对误差会爆而无意义
    if (full.alpha[i] > 0.5) {
      edRelOpaqueSum += edRel
      edRelOpaqueCount++
    }
    if (full.visible[i] && full.alpha[i] > 0.5) {
      for (let c = 0; c < 3; c++) {
        rgbSum += Math.abs(color[i * 3 + c] / alpha[i] - full.rgb[i * 3 + c])
      }
      rgbCount += 3
      const myDepth = ed[i] / Math.max(alpha[i], 1e-6)
      const rel =
        Math.abs(myDepth - full.depth[i]) / Math.max(full.depth[i], 1e-6)
      depthRels.push(rel)
    }
  }
  depthRels.sort((a, b) => a - b)
  return {
    maxAlphaErr,
    meanAlphaErr: meanAlphaErr / count,
    alphaFlips,
    maxEdRel,
    meanEdRelOpaque:
      edRelOpaqueCount > 0 ? edRelOpaqueSum / edRelOpaqueCount : 0,
    rgbMae: rgbCount > 0 ? rgbSum / rgbCount : 0,
    depthMedianRel: depthRels.length > 0 ? depthRels[depthRels.length >> 1] : 0,
    emptyLayers,
  }
}

// ────────────────────────────── 小工具 ──────────────────────────────

function isStrictlyIncreasing(values: ArrayLike<number>): boolean {
  for (let i = 1; i < values.length; i++) {
    if (!(values[i] > values[i - 1])) return false
  }
  return true
}

function sameF32(a: Float32Array, b: Float32Array): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false
  return true
}

/** 型 7 分位数（与 `numpy.percentile` 默认一致），作为直方图反解的独立参考。 */
function sortedQuantile(sorted: Float64Array, p: number): number {
  const n = sorted.length
  if (n === 0) return 0
  const h = (n - 1) * p
  const lo = Math.floor(h)
  const hi = Math.min(n - 1, lo + 1)
  return sorted[lo] + (h - lo) * (sorted[hi] - sorted[lo])
}

function formatRange(values: ArrayLike<number>, digits = 4): string {
  const list = Array.from(values, (v) => v.toFixed(digits))
  if (list.length <= 8) return `[${list.join(", ")}]`
  return `[${list.slice(0, 4).join(", ")}, …, ${list.slice(-2).join(", ")}] (${list.length})`
}

function clamp01(value: number): number {
  return value < 0 ? 0 : value > 1 ? 1 : value
}

// ────────────────────────────── main ──────────────────────────────

async function main(): Promise<void> {
  const { values } = parseArgs({
    args: process.argv.slice(2),
    options: {
      real: { type: "boolean", default: true },
      gpu: { type: "boolean", default: true },
      width: { type: "string" },
      ply: { type: "string" },
    },
    allowPositionals: false,
    allowNegative: true,
    strict: true,
  })
  const noReal = !values.real
  const noGpu = !values.gpu
  const width = numFlag("--width", values.width, 768)
  const plyPath = values.ply
  console.log("=".repeat(78))
  console.log("layering golden 门（A 段：统计 / 放置 / 分带）")
  console.log("=".repeat(78))

  domainChecks()

  console.log("\n[A0b] 原图回写 + 几何补齐（refine.ts，纯 CPU）")
  refineChecks()
  completionChecks()

  console.log("\n[A1] 统计量正确性（uniform-n 合成分布，200k 样本）")
  const reference = makeSynthetic(DISTRIBUTIONS[0], 200_000, 0x5eed)
  const referenceStats = computeDisparityStats(reference.z, {
    near: NEAR,
    far: FAR,
    binCount: 256,
  })
  statisticsChecks(reference, referenceStats)

  console.log("\n[A2/A3] 放置不变量与分带（7 档 × L ∈ {4,8,16,32} × 分布）")
  for (const [index, distribution] of DISTRIBUTIONS.entries()) {
    const sample = makeSynthetic(distribution, 200_000, 0x1234 + index * 7919)
    const stats = computeDisparityStats(sample.z, {
      near: NEAR,
      far: FAR,
      binCount: 256,
    })
    const sortedNdc = Float32Array.from(sample.n).sort((a, b) => b - a)
    placementChecks(distribution, stats, sortedNdc)
    frontDensityChecks(distribution, stats)
  }
  monotonicDepth.report()
  monotonicBounds.report()
  massSums.report()
  deterministic.report()
  nonEmpty.report()
  quantileBalance.report()
  frontDensity.report()
  errorDrivenWins.report()
  bandCoverage.report()
  massVsCount.report()
  layerRanges.report()
  assignmentPermutation.report()
  refRange.report()
  refSmooth.report()
  refBlend.report()
  refMap.report()
  compOk.report()
  occOk.report()
  ownGapOk.report()
  hiddenOk.report()
  hiddenGeoOk.report()
  ownerOk.report()

  if (!noReal) {
    realDataChecks(plyPath)
    realInvariants.report()
    realNonEmpty.report()
    exactMass.report()
  }

  if (!noGpu) {
    await losslessChecks(width)
    permutationIdentity.report()
    permutationValidity.report()
    tableCoverage.report()
    permutationRenderIdentical.report()
    cullIdentity.report()
    alphaMeanLossless.report()
    alphaFlipFree.report()
    edMeanLossless.report()
    rgbLossless.report()
    depthLossless.report()
    layerEmptiness.report()
  }

  const failed = checks.filter((c) => !c.ok)
  console.log(`\n${"=".repeat(78)}`)
  console.log(
    `共 ${checks.length} 条，通过 ${checks.length - failed.length}，失败 ${failed.length}`,
  )
  if (failed.length > 0) {
    console.log("\n✗ 未通过:")
    for (const f of failed) console.log(`   - ${f.name}: ${f.detail}`)
    process.exitCode = 1
  } else {
    console.log("✓ 全部门限通过")
  }
}

main().catch((err) => {
  console.error("\n[错误]", err instanceof Error ? err.message : err)
  if (err instanceof Error && err.stack) console.error(err.stack)
  process.exitCode = 1
})
