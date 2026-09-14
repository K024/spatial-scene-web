/**
 * 原图回写（refineImage）+ **逐层几何补齐** —— layering 的最后一个后处理。
 *
 * 阶段：**layering 的收尾**（在色带渲染之后，交付 `LayeredRGBD` 之前）。
 *
 * ── 在全链路里的位置 ──
 * ```
 * splat -> 分层 RGBAD -> [refine + complete] -> mesh
 * ```
 * 本模块做两件事，共用**同一套「逐层可见性（遮挡）」中间量**：
 * ```
 *   ① 颜色回写（refineImage 语义）：本层是参考视角前表面处，把原图色混进层色；
 *   ② 几何补齐（本轮新增）：
 *        own-gap  ：本层该是前表面却有小洞 → 用本层边缘的色/深/α 补上；
 *        hidden   ：本层被更近层**完全**遮挡且缺覆盖 → 有界 dilate 外推
 *                   （贴边、限距、带夹紧；`α = max(own, 源α·衰减)`；仅写 hidden 像素）。
 *     两者都**不发明 α**：色/深/α 取最近支撑像素（与旧 `meshing/margin.ts` 的
 *     `source` 映射同语义，只是搬到 layering 层做）。
 * ```
 *
 * ── 为什么在 layering 而不是 mesh ──
 * 回写与补齐的产物都是**层数据本身**（`WSplatFrame`）；mesh 只消费层数据，不该反向改它。
 * 它同时用到「层投影」与「原图投影」的映射（对应 Apple 的
 * `GetPixelTransformationFromMPIToImage` / `ComputeRefinementWeight`）。
 *
 * ── 数据从哪来（**不需要额外的 splat 渲染趟**）──
 * 层 RGBAD 就是从**原图那台参考相机**渲出来的，所以逐项都能在现成数据里找到：
 *
 * | 需要 | 来源 |
 * |---|---|
 * | 层颜色（线性直通） | `WSplatFrame.rgb` |
 * | 层覆盖 / 干净表面 | `WSplatFrame.alpha` / `.visible` / `.transmission` |
 * | 逐像素深度 D | `WSplatFrame.depth` |
 * | 层像素 ↔ 原图像素映射 | 参考相机 == 原图相机 ⇒ **恒等 + 分辨率缩放** |
 * | 前景判定 | 调用方给的层号 / 视差 |
 * | 遮挡（每像素最前的层） | 对层栈的 `alpha` 做一次 over（见 `computeOcclusionMasks`） |
 * | 合成前表面深度（own-gap 用） | 层栈 `over` 合成 `D = ΣED/ΣA`（见 `compositeAlphaDepth`） |
 * | 本层最近支撑像素（own-gap/hidden 用） | 从本层 `alpha` 做的多源 BFS（见 `nearestSupportSource`） |
 *
 * ── 颜色回写的三条权重因子（相乘，可调）──
 * 1. **在层内是否为最近表面**：`alpha` 越接近 1 权重越高；
 * 2. **深度是否局部平滑**：`|∇D|/D` 越大权重越低（跨轮廓回写会糊边）；
 * 3. **是否在前景 / 未被遮挡**：由调用方给（层号小、或逐像素遮挡掩码）。
 *
 * ── 几何补齐的两个层级（见 `completeLayerGeometry`）──
 * 判据是**几何**（遮挡关系 + 到本层覆盖的距离 + 本层深度带），**不是洞的面积**。
 *
 * ── 为什么是「软间隔」而非更硬的划分或聚类 ──
 * 层与层之间是一个**软过渡区**，而不是硬切面：没观察到的像素先落中间态、再定稿。
 * 本文件把它落到“逐层 RGBD 的几何补齐”上：先观察到再回填（own-gap），
 * 观察不到的有界外推（hidden），而不是去拟合一个 3D 分界面。
 *
 * 实测（`@768`，`L=8`，`quantile`，`maxDist=16`）：责任缺口 `4~5% → 0.3~0.6%`，
 * 合成 `ΔA` 仍在 `1e-3` 量级、`τ=0.5` 翻转 `0`；复现见
 * `layering-perlayer-quality.ts --complete`（报告不判断）。
 *
 * ── 边界（必须记住）──
 * - 颜色回写**只在参考视角成立**：外推时那块颜色会露「贴图感」。默认只回写**最前景的可见层**。
 * - 颜色回写**不改几何**；几何补齐**改**（补 depth/α），且只在 `complete` 开启时发生。
 * - `own-gap` 回填的是**观测到**的前表面小洞（原图色 + 本层边缘深/α），无幻觉；
 *   `hidden` 外推是**外推**（幻觉区），所以有界 + 带夹紧 + 前景不透明门。
 * - 旧的 `α = fillAlpha = 1` 已废弃（会在边缘画出硬色带）；`fillAlpha` 仅作显式覆盖保留。
 */

import type { SourceImage } from "../sharp/preprocess.ts"
import type { WSplatFrame } from "../wsplat/types.ts"
import type { LayeredRGBD, LayerPlacement } from "./types.ts"

/** 回写权重的三个可调因子。 */
export interface RefineWeightOptions {
  /**
   * 覆盖率下限（默认 `0.5`）。`alpha < minAlpha` ⇒ 权重 0；
   * 其余从 `minAlpha` 线性升到 1。这是「最近表面」因子的硬门。
   */
  minAlpha?: number
  /** 深度平滑尺度（**相对**深度）：`|∇D| / D` 超过它则权重快速下降（默认 `0.02`）。 */
  smoothnessScale?: number
  /** 平滑因子的指数（默认 `2`）。 */
  smoothnessExponent?: number
  /** 该层是否是前景层（默认 `true`）。`false` ⇒ 整层权重 0（背景层不回写）。 */
  foreground?: boolean
  /**
   * 逐像素**遮挡**掩码（长度 = 像素数，`1` = 被更近的层不透明覆盖）。
   * 给了就把该像素权重置 0 —— 参考视角下每像素只应回写**最前的不透明层**。
   */
  occluded?: Uint8Array
}

/**
 * 算一层每像素的回写权重 `w ∈ [0,1]`（纯函数，不改输入）。
 *
 * `frame` 只需借 `depth` / `alpha` / `visible` / `width` / `height` 五个字段，
 * 便于单测直接造最小对象。
 */
export function computeRefineWeight(
  frame: Pick<WSplatFrame, "depth" | "alpha" | "visible" | "width" | "height">,
  options: RefineWeightOptions = {},
  out: Float32Array = new Float32Array(frame.alpha.length),
): Float32Array {
  const { depth, alpha, visible, width, height } = frame
  const minAlpha = options.minAlpha ?? 0.5
  const smoothScale = options.smoothnessScale ?? 0.02
  const smoothExp = options.smoothnessExponent ?? 2
  const foreground = options.foreground ?? true
  const occluded = options.occluded
  out.fill(0)
  if (!foreground) return out
  const alphaGain = minAlpha < 1 ? 1 / (1 - minAlpha) : 1

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x
      if (!visible[i] || alpha[i] < minAlpha) continue
      if (occluded?.[i]) continue
      const z = depth[i]
      if (!(z > 0)) continue
      // 中心差分；任一邻域无效则该方向记 0（不借背景的 D=0 造假梯度）
      let gx = 0
      let gy = 0
      if (x > 0 && x + 1 < width && visible[i - 1] && visible[i + 1]) {
        gx = (depth[i + 1] - depth[i - 1]) * 0.5
      }
      if (y > 0 && y + 1 < height && visible[i - width] && visible[i + width]) {
        gy = (depth[i + width] - depth[i - width]) * 0.5
      }
      const relativeGradient = Math.hypot(gx, gy) / z
      const wSurface = Math.min(1, (alpha[i] - minAlpha) * alphaGain)
      const wSmooth = 1 / (1 + (relativeGradient / smoothScale) ** smoothExp)
      out[i] = wSurface * wSmooth
    }
  }
  return out
}

/**
 * 把原图颜色按权重混进层颜色：`c ← (1−w)·c_splat + w·c_image`。
 *
 * ⚠ **这不是管线用的回写**（管线走 {@link applyResidualWriteback}）。保留它是为了
 * 留一个可当单测用的最小混合原语：`w = 1` 时逐位等于原图色。
 * 逐层混原图色**在两层以上同时贡献的像素上是错的** —— 原图色是「多层合成色」，
 * 不是任何一层的真色。实测（近层 α=0.6 红 + 远层 α=1 蓝，合成已等于原图）
 * 旧式按 `w=0.2` 混完，合成偏离原图 0.048；残差式给 0。
 *
 * 两端都是**直通线性 RGB**（不是 premultiplied），与 `WSplatFrame.rgb` 一致。
 * `w = 0` 时逐位等于层颜色；`w = 1` 时逐位等于原图色（方便当单测）。
 */
export function blendImageWriteback(
  rgb: Float32Array,
  imageRgb: Float32Array,
  weight: Float32Array,
  out: Float32Array = new Float32Array(rgb.length),
): Float32Array {
  const pixels = weight.length
  for (let i = 0; i < pixels; i++) {
    const w = weight[i]
    const o = i * 3
    for (let c = 0; c < 3; c++) {
      out[o + c] = rgb[o + c] * (1 - w) + imageRgb[o + c] * w
    }
  }
  return out
}

/** {@link applyResidualWriteback} 的旋钮。 */
export interface ResidualWritebackOptions {
  /**
   * 只在**合成覆盖** `A_total ≥ minCoverage` 的像素回写（默认 `0.9`）。
   *
   * 为什么要这门口：残差回写是「把合成推向原图」。若该像素背后还有没被任何层
   * 表达的背景（`1 − A_total > 0`），把残差全摊到现有层上就等于**把没观测到的背景
   * 烘进层色**，而且分摊式 `Δc ≈ r/A²` 会把残差**放大** `1/A²` 倍（`A=0.9` ⇒ 1.23×，
   * `A=0.5` ⇒ 4×）。所以门不能开到很低。
   *
   * 实测权衡（768、L=6、`example.ply` + 同相机原图、同一套权重，参考视角 vs 原图）：
   * ```
   *   minCoverage  亮度NCC   sRGB MAE  放大 1/A²   （尾部三档完全相同：P99 0.0008、>0.05 占 0.21%）
   *   0.999        0.9684    0.0100    1.00×
   *   0.99         0.9850    0.0046    1.02×
   *   0.90         0.9969    0.0015    1.23×   ← **默认**：两条轴都赢旧式，放大仍温和
   *   0.50         0.9986    0.0012    4.00×   ← 只多 0.0017 NCC，却把放大推到 4 倍，不值
   *   旧式(不看覆盖) 0.9949    0.0041    —      ← 它把原图色涂满全图，聚合指标靠"涂到低 α 像素"刷出来
   * ```
   * ⇒ `0.9` 是拐点：再放宽收益趋平、代价（烘背景 + 放大）继续涨。
   * ⚠ 仍未完成的验收：被烘进的那 ≤10% 未观测背景**会不会在侧视里露馅**（发光/第二主体）
   * —— 参考视角指标看不见它，只能用 `layering-render-views.ts` 或 viewer 侧摆目测。
   */
  minCoverage?: number
  /** 单次回写的颜色上限（默认 `0.25`，防残差过大时整层色被推飞）。 */
  maxDelta?: number
  /** 分母正则项（默认 `1e-4`）。 */
  epsilon?: number
  /** 诊断输出（可选；`refineLayers` 的 `stats` 会转发到这里）。 */
  stats?: RefineStats
}

/**
 * 残差回写的默认覆盖率门。
 *
 * 取值理由与实测表见 {@link ResidualWritebackOptions.minCoverage}：`0.9` 是"两条轴都赢旧式"
 * 的拐点，且 `1/A²` 放大仍只有 1.23×。**脚本的 CLI 默认值也从这里取**（避免两处漂移）。
 */
export const DEFAULT_WRITEBACK_MIN_COVERAGE = 0.9

/** 单次回写的默认颜色上限（防残差过大时整层色被推飞）。 */
export const DEFAULT_WRITEBACK_MAX_DELTA = 0.25

/** 回写 / 补齐的逐像素诊断（供脚本与调参报告引用；渲染器忽略）。 */
export interface RefineStats {
  /** 合成覆盖过门、真正参与回写的像素数。 */
  writebackCoveredPixels: number
  /** 其中被 `maxDelta` 限幅的像素数（限幅后合成**到不了**原图）。 */
  writebackClampedPixels: number
  /** 其中因层色被夹回 `[0,1]` 而没吃满残差的像素数。 */
  writebackRangeClippedPixels: number
  /** 回写后仍残留的合成偏差均值（|Δ|，线性直通域）。 */
  writebackResidualMean: number
}

/**
 * **合成残差回写**：把「合成与原图的差」按各层的边际贡献分回层色。
 *
 * ```text
 * v_k  = (1 − A_{<k})·α_k                # 本层对该像素可见度的边际贡献
 * C    = Σ v_k·c_k                       # 逐层 over 的直通色（与 viewer 的 BLEND 同语义）
 * r    = I − C                           # 参考视角残差（线性直通域）
 * Δc_k = q_k·v_k / (Σ q_j·v_j² + ε) · r
 * ```
 *
 * 性质（`scripts/layering-golden.ts` 的 A0c「残差回写的合成恒等」钉住）：
 * 1. `Σ v_k·Δc_k ≈ r` ⇒ **回写后逐层 over 合成 == 原图**（在 `A_total` 足够处）；
 * 2. `r = 0` ⇒ 逐层不动：已经等于原图的像素**不会被写坏**（旧式做不到）；
 * 3. `q_k = 0`（背景层 / 被遮挡 / 权重低）的层不吃残差，只当贡献者参与分母。
 *
 * ── 分摊与限幅（两趟）──
 * 单趟按 `q·v²` 分摊时，某层可能被推到自己合法范围之外（`[0,1]`）或撞 `maxDelta`，
 * 那一段残差就**丢了**（α=0.6 红 + α=1 蓝：蓝层被要求减红而它本来就是 0 ⇒ 合成差 8.4e-4）。
 * 所以按两趟做：第一趟分摊 + 逐层范围检查，夹住的层退出；第二趟把没吃掉的残差给没夹的层。
 * **限幅仍是尾部的已知来源**（native 实测：约 1.8% 像素撞 `maxDelta`、约 11% 的通道被夹回
 * `[0,1]`）——那些像素的合成到不了原图；要压它就抬 `maxDelta`，代价是层色被推出合法域。
 *
 * @param weights 逐像素置信权重（`computeRefineWeight` 的产物），`q_k`。
 */
export function applyResidualWriteback(
  frames: readonly WSplatFrame[],
  imageLinear: Float32Array,
  weights: readonly Float32Array[],
  options: ResidualWritebackOptions = {},
): WSplatFrame[] {
  const L = frames.length
  if (L === 0) return []
  const pixels = frames[0].alpha.length
  const minCoverage = options.minCoverage ?? DEFAULT_WRITEBACK_MIN_COVERAGE
  const maxDelta = Math.max(0, options.maxDelta ?? DEFAULT_WRITEBACK_MAX_DELTA)
  const eps = options.epsilon ?? 1e-4
  const stats = options.stats
  const { frontBefore, totalAlpha } = computeLayerOwnership(frames, pixels)

  // 逐层残差增量（`Float32Array` 是可选分配：整层 q=0 时不分配）。
  const deltas: (Float32Array | null)[] = new Array(L).fill(null)
  const touched: boolean[] = new Array(L).fill(false)
  const v = new Float32Array(L)
  const q = new Float32Array(L)
  /** 逐像素的「本层还愿意吃残差吗」（复用，避免每像素分配）。 */
  const active = new Uint8Array(L)
  for (let i = 0; i < pixels; i++) {
    const A = totalAlpha[i]
    if (!(A >= minCoverage)) continue
    if (stats) stats.writebackCoveredPixels++
    // 本像素过门 ⇒ 逐层算 v_k / q_k。
    let denom = eps
    for (let k = 0; k < L; k++) {
      const frame = frames[k]
      const ek = (1 - frontBefore[k][i]) * frame.alpha[i]
      const wk = weights[k]?.[i] ?? 0
      v[k] = ek
      q[k] = wk
      denom += wk * ek * ek
    }
    const o = i * 3
    let r0 = imageLinear[o]
    let r1 = imageLinear[o + 1]
    let r2 = imageLinear[o + 2]
    for (let k = 0; k < L; k++) {
      const ek = v[k]
      if (!(ek > 0)) continue
      const rgb = frames[k].rgb
      r0 -= ek * rgb[o]
      r1 -= ek * rgb[o + 1]
      r2 -= ek * rgb[o + 2]
    }
    if (Math.abs(r0) + Math.abs(r1) + Math.abs(r2) <= 1e-7) continue

    // 两趟：第一趟按 q·v² 分摊（每层都做范围检查），被夹住的层退出；
    // 第二趟把没吃掉的残差交给剩下没夹的层。只用一趟时，"某层被推到自己合法范围之外"
    // 会让那段残差直接丢失（实测 8.4e-4 的合成差就来自这里）。
    for (let k = 0; k < L; k++) {
      active[k] = q[k] > 0 && v[k] > 0 ? 1 : 0
    }
    let anyClamped = false
    for (let pass = 0; pass < 2; pass++) {
      let denom = eps
      let survivors = 0
      for (let k = 0; k < L; k++) {
        if (!active[k]) continue
        denom += q[k] * v[k] * v[k]
        survivors++
      }
      if (survivors === 0) break
      for (let k = 0; k < L; k++) {
        if (!active[k]) continue
        const scale = (q[k] * v[k]) / denom
        const delta = deltas[k] ?? new Float32Array(pixels * 3)
        deltas[k] = delta
        touched[k] = true
        const rgb = frames[k].rgb
        let clipped = false
        let got0 = 0
        let got1 = 0
        let got2 = 0
        for (let c = 0; c < 3; c++) {
          const want = clampAbs(
            scale * (c === 0 ? r0 : c === 1 ? r1 : r2),
            maxDelta,
          )
          const base = rgb[o + c] + delta[o + c]
          const next = Math.min(1, Math.max(0, base + want))
          const got = next - base
          if (Math.abs(got - want) > 1e-7) clipped = true
          delta[o + c] += got
          if (c === 0) got0 = got
          else if (c === 1) got1 = got
          else got2 = got
        }
        r0 -= v[k] * got0
        r1 -= v[k] * got1
        r2 -= v[k] * got2
        if (clipped) {
          active[k] = 0
          anyClamped = true
        }
      }
      if (Math.abs(r0) + Math.abs(r1) + Math.abs(r2) <= 1e-7) break
    }
    if (anyClamped && stats) {
      stats.writebackClampedPixels++
      // 两趟之后仍没吃掉的残差（线性直通域，三通道平均）
      stats.writebackResidualMean +=
        (Math.abs(r0) + Math.abs(r1) + Math.abs(r2)) / 3
    }
  }

  if (stats && stats.writebackCoveredPixels > 0) {
    stats.writebackResidualMean /= stats.writebackCoveredPixels
  }
  // 增量已在分摊时按 [0,1] 与 maxDelta 夹过，这里只做写出（不再夹一次）。
  return frames.map((frame, k) => {
    const delta = deltas[k]
    if (!delta || !touched[k]) return frame
    const rgb = Float32Array.from(frame.rgb)
    let clipped = 0
    for (let i = 0; i < rgb.length; i++) {
      const v = rgb[i] + delta[i]
      if (v < 0 || v > 1) clipped++
      rgb[i] = Math.min(1, Math.max(0, v))
    }
    if (stats && clipped > 0) stats.writebackRangeClippedPixels += clipped
    return { ...frame, rgb }
  })
}

function clampAbs(v: number, limit: number): number {
  return v > limit ? limit : v < -limit ? -limit : v
}

/**
 * 层像素 -> 原图像素（浮点坐标，给小采样器用）。
 *
 * 参考相机 == 原图相机 ⇒ 映射是**恒等 + 分辨率缩放**（见文件头）。
 * 返回值已加 `+0.5 / −0.5` 的像素中心约定，方便直接双线性插值。
 *
 * ⚠ **前提必须成立**：层渲染相机与原图相机是同一套内参（像素焦距 + 主点 + 画幅）。
 * 若上游改成「用更大的画布 FOV 渲染分层、最后裁回有效视口」（Apple 头文件里
 * `effectiveFovInRadians` / `effectiveAspectRatio` / `trimmedColorTexture` 提示这种可能），
 * 这里必须换成 **crop/UV 变换**，否则回写、纹理 UV、mesh 反投影、viewer 相机会各用一套口径，
 * 表现为参考视角缩放或半像素错位。**尚未实现**，先别把假设当契约。
 */
export function layerPixelToImagePixel(
  x: number,
  y: number,
  layerWidth: number,
  layerHeight: number,
  imageWidth: number,
  imageHeight: number,
): { x: number; y: number } {
  return {
    x: ((x + 0.5) * imageWidth) / layerWidth - 0.5,
    y: ((y + 0.5) * imageHeight) / layerHeight - 0.5,
  }
}

/**
 * hidden 边界外推的旋钮。
 *
 * 语义：把本层表面从**已有覆盖**向外（屏幕空间）延伸到被更近层遮挡的区域，
 * 产出连续的 `(rgb, depth, α)`。
 *
 * ── 为什么是这个形状（选型结论，别改成「洞的面积」那一套）──
 * 1. **判据用几何**：遮挡关系（更近层的累积 α）+ 到本层覆盖的距离 + 本层深度带夹紧。
 *    不看洞的面积 —— 大面积但**贴边**的遮挡区照样要补（现实里的遮挡都是贴边的），
 *    而远离覆盖的深处空洞不补（那是背衬平面的活）。
 * 2. **有界 + 带夹紧**：`maxDistancePx` 限距、深度夹到本层带 ⇒ 不会穿到遮挡物前面。
 * 3. **只写 hidden 像素**：目标掩码要求`A_{<k} ≥ minFrontAlpha`（默认 0.999，参考视角
 *    **完全**遮挡）⇒ 参考视角逐位不变，外推结果不会泄漏到原图视角。
 * 4. **默认只补几何（depth-only）**：不动 RGB/α，保留原纹理的渐隐；要填色得显式开
 *    {@link HiddenExtendOptions.rgba}（dilate + 距离衰减）。
 */
export interface HiddenExtendOptions {
  /**
   * 从本层覆盖起算的最大外推距离（像素）。默认 `16`。
   *
   * ⚠ 这是**几何上限**（距离），不是面积 —— 大面积但贴边的遮挡区照样补，
   * 远离覆盖的深处空洞不补（交给背衬平面）。
   */
  maxDistancePx?: number
  /**
   * 只在外推区做、且只在填充像素上的平滑趟数（默认 `2`）。
   * 权重按**深度相似度**给，避免跨断层搅拌。`0` = 关闭（退回最近源硬拷贝）。
   */
  smoothPasses?: number
  /**
   * 外推只在「合成前景累积 α ≥ 此值」处进行（缺省 `0.999` = 参考视角**完全**遮挡）。
   * 这是「非 hidden 不受影响」（参考视角恒等）的门；调低会把外推结果露到参考视角。
   */
  minFrontAlpha?: number
  /**
   * 平滑时的深度相似度尺度，相对**本层带宽度**（默认 `0.25`）。
   * 越小越"只和深度接近的邻居平均"。
   */
  depthSimilarity?: number
  /**
   * 是否外推**深度（几何）**。默认 `true`。
   * `false` = 不动本层 depth（但仍可写 RGBA）。
   */
  depth?: boolean
  /**
   * 是否外推 **RGBA**（`rgb` + `α`）。默认 `false`（只补几何）。
   *
   * `false` = 填充像素只写 `depth`，`rgb` / `α` 保留本层原值（原纹理自带渐隐）。
   * `true` = **dilate 方案**：`α = max(本层自有 α, 源 α · (1 − d/(R+1)))`（距离衰减的膨胀），
   * `rgb` **本层自有优先**（α>0 就保留，只抬 α），缺失才取最近支撑色。
   * 作用域仍由 `minFrontAlpha` 锁在“参考视角完全遮挡”区。
   */
  rgba?: boolean
}

/** 逐层几何补齐的旋钮。 */
export interface LayerCompletionOptions {
  /** `own-gap`：本层该是前表面却有小洞 → 用本层边缘的色/深/α 补。默认 `true`。 */
  ownGap?: boolean
  /** `hidden` 边界外推。`false` 关闭；缺省启用（默认参数见 `HiddenExtendOptions`）。 */
  hidden?: HiddenExtendOptions | false
  /**
   * own-gap 小洞的**最大距离**（px，默认 `2`）。
   *
   * 这就是「与 own-gap / hidden 无关的小洞填补」：只看距离，**不看面积**；
   * 距本层支撑超此距离的开放区域不算洞（交给 `hidden` / 背衬）。
   */
  holeRadiusPx?: number
  /**
   * 填充像素写入的 `α`。**缺省 = 取最近支撑像素的 `α`**（不发明 `α=1`）；
   * 传值则强制用该值（旧行为）。
   */
  fillAlpha?: number
  /** 支撑判定用的 α 门（与 meshing 的支撑语义一致）。默认 `0.5`。 */
  supportAlpha?: number
  /**
   * 「洞」两钳口深度一致的**相对**容差（默认 `0.2`）。
   *
   * 只补**同一张表面上的洞**：两侧支撑的深度差超过 `tol·min(z)` 就当成两张不同的面
   * （阶梯、栏杆间隙），不补 —— 否则会把真实的深度断口糊成一张布。
   */
  holeDepthTolerance?: number
}

/** 整摞回写的旋钮 = 逐层权重因子 + 回写层范围 + 几何补齐。 */
export interface RefineLayersOptions extends RefineWeightOptions {
  /**
   * 回写层范围：层号 `< foregroundLayers` 才回写（`0` = 关闭，缺省 = `L` = 全部）。
   *
   * ⚠ 逐像素还有 `occluded` 遮挡门，所以「全部层」的实际效果仍是
   * 「每个像素只在它**最前的不透明层**回写一次」。
   */
  foregroundLayers?: number
  /**
   * 几何补齐（own-gap + hidden 外推）。默认开启；传 `false` 退回**纯颜色回写**。
   *
   * 开启后 `refineLayers` 会改写 `depth` / `alpha` / `transmission` /
   * `accumulatedDepth` / `visible`，不只是 `rgb`。这是与旧版的关键差异。
   */
  complete?: LayerCompletionOptions | false
  /**
   * 颜色回写用**合成残差**（默认 `true`）。
   *
   * `false` = 旧行为（逐层按权重混原图色）—— 两层以上同时贡献的像素上会把已经
   * 正确的合成**写坏**（见 `blendImageWriteback` 的注释）。留这个开关只为做对照实验。
   */
  residual?: boolean
  /** 残差回写的门（覆盖率下限 / 单次增量上限）。 */
  writeback?: ResidualWritebackOptions
  /** 诊断输出对象（可选）：回写覆盖数、限幅数、残留均值。 */
  stats?: RefineStats
}

/** 逐像素「已被更近的不透明层覆盖」的判定阈值（与回写权重的 `minAlpha` 默认值一致）。 */
const OPAQUE_ALPHA = 0.5

/**
 * 逐层遮挡掩码：`occluded[k][i] = 1` 表示像素 `i` 已被**更近**的层（`< k`）不透明覆盖。
 *
 * 这是「参考视角逐层可见性」的显式化，颜色回写与几何补齐共用。
 */
export function computeOcclusionMasks(
  frames: readonly Pick<WSplatFrame, "alpha">[],
  pixels: number,
): Uint8Array[] {
  const L = frames.length
  const occluded: Uint8Array[] = Array.from(
    { length: L },
    () => new Uint8Array(pixels),
  )
  const front = new Uint8Array(pixels)
  for (let k = 0; k < L; k++) {
    occluded[k].set(front)
    const a = frames[k].alpha
    for (let i = 0; i < pixels; i++) if (a[i] > OPAQUE_ALPHA) front[i] = 1
  }
  return occluded
}

/**
 * 层栈 `over` 合成，返回**累积 α** 与**合成深度** `D = ED / A`。
 *
 * 与 `layering-golden.ts` / `meshing/backfill.ts` 的合成口径一致（从远到近）。
 * `D` 是参考视角的 α 加权期望深度，`own-gap` 回填与「责任层」判定都用它。
 */
export function compositeAlphaDepth(
  frames: readonly Pick<WSplatFrame, "alpha" | "accumulatedDepth">[],
  pixels: number,
): { alpha: Float32Array; depth: Float32Array } {
  const alpha = new Float32Array(pixels)
  const ed = new Float32Array(pixels)
  for (let k = frames.length - 1; k >= 0; k--) {
    const a = frames[k].alpha
    const e = frames[k].accumulatedDepth
    for (let i = 0; i < pixels; i++) {
      const av = a[i]
      const w = 1 - av
      ed[i] = e[i] + w * ed[i]
      alpha[i] = av + w * alpha[i]
    }
  }
  const depth = new Float32Array(pixels)
  for (let i = 0; i < pixels; i++) {
    depth[i] = alpha[i] > 1e-6 ? ed[i] / alpha[i] : 0
  }
  return { alpha, depth }
}

/** 逐层深度矩自洽性的体检结果（见 {@link layerMomentResidual}）。 */
export interface MomentResidual {
  /**
   * 最大相对残差 `|ED − A·D| / max(|ED|, |A·D|, eps)`。
   *
   * 逐层像素的 `depth` 是渲染期 α 加权期望深度 `D = ED/A`，所以 `ED == A·D` 是
   * **表示层的不变量**：谁改了 `depth`（或 `α`）谁就必须同步 `accumulatedDepth`，
   * 否则下游 `compositeAlphaDepth` / mesh 用的合成深度就是陈旧值。
   */
  readonly maxRel: number
  /** 相对残差 > 容差的像素数。 */
  readonly violations: number
  /** 残差最大处的层号（`-1` = 无）。 */
  readonly worstLayer: number
}

/**
 * 逐层检查 `ED == A·D`（几何补齐之后必须仍然成立）。
 *
 * 阈值口径：`A` 存在 f16 附件里，`ED` 是 f32，所以不变量本身有 `f16_ulp(A) ≈ 5e-4`
 * 的固有相对误差；默认容差 `2e-3`（4 倍余量）。只统计 `α ≥ alphaFloor` 的像素
 * （α 极小的像素没有几何意义，相对误差会被放大成噪声）。
 */
export function layerMomentResidual(
  frames: readonly Pick<WSplatFrame, "alpha" | "depth" | "accumulatedDepth">[],
  options: { relTolerance?: number; alphaFloor?: number } = {},
): MomentResidual {
  const tol = options.relTolerance ?? 2e-3
  const alphaFloor = options.alphaFloor ?? 0.05
  let maxRel = 0
  let violations = 0
  let worstLayer = -1
  for (let k = 0; k < frames.length; k++) {
    const { alpha, depth, accumulatedDepth } = frames[k]
    for (let i = 0; i < alpha.length; i++) {
      const a = alpha[i]
      if (!(a >= alphaFloor)) continue
      const want = a * depth[i]
      const got = accumulatedDepth[i]
      const denom = Math.max(Math.abs(want), Math.abs(got), 1e-6)
      const rel = Math.abs(got - want) / denom
      if (rel > maxRel) {
        maxRel = rel
        worstLayer = k
      }
      if (rel > tol) violations++
    }
  }
  return { maxRel, violations, worstLayer }
}

/**
 * 前向累计 α（near→far over）→ 逐像素**责任层**与负责比例。
 *
 * 参考视角的合成就是 near→far 的 `over`，所以「这个像素由哪层负责」只能由 α 累积定义：
 * ```
 *   A_{<k}(i) = 1 − Π_{j<k}(1−α_j)          // 被更近层吃掉的可见度
 *   w_k(i)    = (1 − A_{<k}(i)) · α_k(i)     // 本层的边际贡献（负责比例 ∝ w_k）
 *   owner(i)  = argmax_k w_k(i)
 * ```
 * ⚠ **不要用 wsplat 的 `depth` 输出判归属**：它是 α 加权混合的期望深度，数值上不可靠；
 * 只有 α 累积能逐像素精确还原参考视角。
 */
export function computeLayerOwnership(
  frames: readonly Pick<WSplatFrame, "alpha">[],
  pixels: number,
): {
  /** `argmax_k w_k`；全透明时为 `-1`。 */
  owner: Int32Array
  /** `frontBefore[k][i] = A_{<k}(i)`（层 `k` 之前的前向累计 α）。 */
  frontBefore: Float32Array[]
  /** `A_total(i)`。 */
  totalAlpha: Float32Array
} {
  const L = frames.length
  const owner = new Int32Array(pixels).fill(-1)
  const acc = new Float32Array(pixels)
  const best = new Float32Array(pixels)
  const frontBefore: Float32Array[] = new Array(L)
  for (let k = 0; k < L; k++) {
    frontBefore[k] = Float32Array.from(acc)
    const a = frames[k].alpha
    for (let i = 0; i < pixels; i++) {
      const w = (1 - acc[i]) * a[i]
      if (w > best[i]) {
        best[i] = w
        owner[i] = k
      }
      acc[i] += w
    }
  }
  return { owner, frontBefore, totalAlpha: acc }
}

/** 逐层帧的可变副本（只在首次真正修改某层时才拷贝，其余字段原样复用）。 */
interface EditableFrame {
  rgb: Float32Array
  alpha: Float32Array
  depth: Float32Array
  transmission: Float32Array
  accumulatedDepth: Float32Array
  visible: Uint8Array
}

function makeEditable(frame: WSplatFrame): EditableFrame {
  return {
    rgb: Float32Array.from(frame.rgb),
    alpha: Float32Array.from(frame.alpha),
    depth: Float32Array.from(frame.depth),
    transmission: Float32Array.from(frame.transmission),
    accumulatedDepth: Float32Array.from(frame.accumulatedDepth),
    visible: Uint8Array.from(frame.visible),
  }
}

/**
 * 几何补齐：`own-gap` 小洞填补 + `hidden` 边界外推。
 *
 * 共同原则：**不发明 α**。填充像素的 `rgb` / `α` / `depth` 都取**最近支撑像素**
 * （与旧 `meshing/margin.ts` 的 `source` 映射同语义，只是搬到 layering 层做）。
 * 只有显式传入新层 `fillAlpha` 时才会硬写 `α`。
 *
 * ── own-gap（洞，观测）──
 * 条件：合成可见（`A>τ`）、`band(D)==k`、本层 `α<τ`、距本层支撑 ≤ `holeRadiusPx`，
 * 且被支撑**夹在两侧**（洞，不是边缘外点）。
 * 动作：`rgb = 原图`、`depth = 最近支撑深度`（夹到本层带）、`α = 最近支撑 α`、`visible = 1`。
 * 「两侧夹住」这一门是旧版在外缘画出 `α=1` 密集硬点的直接修复。
 *
 * ── hidden（外推，有界）──
 * 条件：`band(D)<k`（被更近层遮挡）、本层 `α<τ`、合成前景 `α ≥ minFrontAlpha`、
 * 且距本层覆盖 ≤ `maxDistancePx`。
 * 动作：多源 BFS，填充像素取「首次到达的源」的色/深，再按深度相似平滑几趟；
 * 深度**夹紧到本层带** ⇒ 不会穿到遮挡物前面。
 * 外观：`hidden.rgba=false`（默认）**只写 depth**（保留原纹理 α 渐隐）；
 * `rgba=true` 走 **dilate**：`α = max(本层自有 α, 源 α·(1−d/(R+1)))`，`rgb` **本层自有优先**。
 *
 * 判据全是**几何**（遮挡关系 / 距离 / 深度带），不涉及洞的面积。
 *
 * @returns 新的帧数组（未改动的层复用原引用）。
 */
export function completeLayerGeometry(
  frames: readonly WSplatFrame[],
  placement: Pick<LayerPlacement, "near" | "far" | "boundariesZ">,
  imageLinear: Float32Array,
  width: number,
  height: number,
  options: LayerCompletionOptions = {},
): WSplatFrame[] {
  const L = frames.length
  const pixels = width * height
  const supportAlpha = options.supportAlpha ?? 0.5
  const fillAlpha = options.fillAlpha
  const doOwn = options.ownGap !== false
  const holeRadius = Math.max(1, Math.floor(options.holeRadiusPx ?? 2))
  const holeDepthTol = Math.max(0, options.holeDepthTolerance ?? 0.2)
  const hidden =
    options.hidden === false
      ? null
      : (options.hidden ?? ({} as HiddenExtendOptions))
  const boundariesZ = placement.boundariesZ

  const out: WSplatFrame[] = frames.slice()
  const editable: (EditableFrame | null)[] = new Array(L).fill(null)
  const ensure = (k: number): EditableFrame => {
    let e = editable[k]
    if (!e) {
      e = makeEditable(frames[k])
      editable[k] = e
      out[k] = { ...frames[k], ...e }
    }
    return e
  }

  // 归属只认 α 累积（不碰 wsplat 的 depth）
  const { owner, frontBefore, totalAlpha } = computeLayerOwnership(
    frames,
    pixels,
  )

  // ── ① own-gap：本层该负责的前表面**小洞** → 用本层边缘的色/深/α 补（不发明 α）──
  // 门：距本层支撑 ≤ `holeRadiusPx`（小洞）+ 被**同表面**支撑夹在两侧（洞，不是边缘外点）。
  // 两条认领路径（缺一不可）：
  //   a) **有主**：合成可见（`A > τ`）且 α 归属是本层（旧路径，补「本层是前表面但 α 偏低」）；
  //   b) **无主**：合成 `A` 本来就低、本层之前没有更近的支撑、**本层之后再没有可观测支撑**
  //      ⇒ 这是「所有层都没画出来」的采样洞（零 α 洞），本层按同表面邻域认领。
  //      b 的第三项是「别把真间隙填死」的关键：栏杆/结构间隙背后**看得见东西**（更远层有
  //      支撑），那种洞不许填；只有背后什么都没有的洞才是渲染丢的采样洞。
  if (doOwn) {
    // `supportedFrom[k][i]` = 层 `k..L−1` 里是否存在 α ≥ τ 的支撑。
    const supportedFrom: Uint8Array[] = new Array(L + 1)
    supportedFrom[L] = new Uint8Array(pixels)
    for (let k = L - 1; k >= 0; k--) {
      const prev = supportedFrom[k + 1]
      const cur = Uint8Array.from(prev)
      const a = frames[k].alpha
      for (let i = 0; i < pixels; i++) if (a[i] >= supportAlpha) cur[i] = 1
      supportedFrom[k] = cur
    }
    for (let k = 0; k < L; k++) {
      const f = frames[k]
      const bandLo = boundariesZ[k]
      const bandHi = boundariesZ[k + 1]
      const ns = nearestSupportSource(
        f.alpha,
        f.visible,
        f.depth,
        supportAlpha,
        width,
        height,
        holeRadius,
      )
      let changed = false
      for (let i = 0; i < pixels; i++) {
        if (f.alpha[i] >= supportAlpha) continue
        const visible = totalAlpha[i] > supportAlpha
        const owned = visible && owner[i] === k
        const unobserved =
          !visible &&
          !(frontBefore[k][i] >= supportAlpha) &&
          supportedFrom[k + 1][i] === 0
        if (!owned && !unobserved) continue
        const s = ns.src[i]
        if (s < 0) continue
        const x = i % width
        const y = (i / width) | 0
        if (
          !isEnclosedSameSurface(
            f.alpha,
            f.depth,
            x,
            y,
            width,
            height,
            supportAlpha,
            holeRadius,
            holeDepthTol,
          )
        ) {
          continue
        }
        const e = ensure(k)
        const o = i * 3
        e.rgb[o] = imageLinear[o]
        e.rgb[o + 1] = imageLinear[o + 1]
        e.rgb[o + 2] = imageLinear[o + 2]
        const a = fillAlpha ?? e.alpha[s]
        writePixelState(e, i, clamp(e.depth[s], bandLo, bandHi), a)
        changed = true
      }
      if (changed) out[k] = { ...frames[k], ...editable[k]! }
    }
  }

  // ── ② hidden 边界外推 ──
  if (hidden && hidden.depth !== false) {
    const maxDist = Math.max(1, Math.floor(hidden.maxDistancePx ?? 16))
    const minFront = hidden.minFrontAlpha ?? 0.999
    const smoothPasses = Math.max(0, Math.floor(hidden.smoothPasses ?? 2))
    const depthSimilarity = hidden.depthSimilarity ?? 0.25
    const writeRgb = hidden.rgba === true
    for (let k = 0; k < L; k++) {
      const e = editable[k] ?? makeEditable(frames[k])
      const bandLo = boundariesZ[k]
      const bandHi = boundariesZ[k + 1]
      const filled = extendHiddenBoundary(e, frontBefore[k], k, boundariesZ, {
        width,
        height,
        maxDist,
        minFront,
        supportAlpha,
        fillAlpha,
        smoothPasses,
        depthSimilarity,
        writeRgb,
        bandLo,
        bandHi,
        L,
      })
      if (filled > 0) {
        editable[k] = e
        out[k] = { ...frames[k], ...e }
      }
    }
  }

  return out
}

interface ExtendParams {
  width: number
  height: number
  maxDist: number
  minFront: number
  supportAlpha: number
  /** `α` 覆盖值；`undefined` = 取最近支撑像素的 `α`。 */
  fillAlpha: number | undefined
  smoothPasses: number
  depthSimilarity: number
  /** 是否写回 `rgb`（`hidden.rgba`）。`false` = 只补几何、保留原纹理。 */
  writeRgb: boolean
  bandLo: number
  bandHi: number
  L: number
}

/**
 * 单层的 hidden 有界外推（就地改 `e`）。返回填充像素数。
 *
 * 算法（dilate + hidden 门）：多源 4-连通 BFS（源 = 本层支撑，目标 = 被更近层**完全**遮挡
 * 且本层缺失的像素），得到逐像素最近源 `src` 与距离 `dist`（≤ `maxDist`）。
 * - **depth**：源深度，夹到本层带 `[bandLo, bandHi]`；再做 `smoothPasses` 趟深度相似平滑。
 * - **α（先扩 α）**：`max(本层自有 α, 源 α · (1 − d/(R+1)))` —— 距离衰减的膨胀，
 *   既要“露出更多原图”，又不能在膨胀边界留硬壳边。
 * - **rgb**：**本层自有优先**（α>0 就保留，只抬 α），缺失才取源色。避开逐通道 max-pool 串色。
 * - 仅在 hidden 像素写入 ⇒ 非 hidden 逐位不变（参考视角恒等）。
 */
function extendHiddenBoundary(
  e: EditableFrame,
  frontAlpha: Float32Array,
  k: number,
  boundariesZ: ArrayLike<number>,
  p: ExtendParams,
): number {
  const { width, height, maxDist, minFront, supportAlpha } = p
  const pixels = width * height

  // 目标掩码：hidden = 被更近层**完全**遮挡 (A_{<k} ≥ minFront) 且本层缺失。
  // 只用 α，不看 depth。
  const target = new Uint8Array(pixels)
  for (let i = 0; i < pixels; i++) {
    if (e.alpha[i] >= supportAlpha) continue
    if (!(frontAlpha[i] >= minFront)) continue
    target[i] = 1
  }

  const dist = new Int32Array(pixels).fill(-1)
  const src = new Int32Array(pixels).fill(-1)
  const filled = new Uint8Array(pixels)
  const queue = new Int32Array(pixels)
  let head = 0
  let tail = 0
  for (let i = 0; i < pixels; i++) {
    if (e.alpha[i] >= supportAlpha && e.visible[i] && e.depth[i] > 0) {
      dist[i] = 0
      src[i] = i
      queue[tail++] = i
    }
  }
  const srcCount = tail
  if (srcCount === 0) return 0

  while (head < tail) {
    const i = queue[head++]
    const d = dist[i]
    if (d >= maxDist) continue
    const x = i % width
    const y = (i / width) | 0
    for (let n = 0; n < 4; n++) {
      const xx = x + (n === 0 ? 1 : n === 1 ? -1 : 0)
      const yy = y + (n === 2 ? 1 : n === 3 ? -1 : 0)
      if (xx < 0 || xx >= width || yy < 0 || yy >= height) continue
      const j = yy * width + xx
      if (dist[j] !== -1 || !target[j]) continue
      dist[j] = d + 1
      src[j] = src[i]
      filled[j] = 1
      queue[tail++] = j
    }
  }

  // 写 depth + （可选）RGB/α。先扩 depth/α，RGB 本层优先。
  let filledCount = 0
  const decayDenom = maxDist + 1
  for (let i = 0; i < pixels; i++) {
    if (!filled[i]) continue
    filledCount++
    const s = src[i]
    const z = clamp(e.depth[s], p.bandLo, p.bandHi)
    let a = e.alpha[i]
    if (p.writeRgb) {
      const ownAlpha = e.alpha[i]
      const dilated = e.alpha[s] * (1 - dist[i] / decayDenom)
      a = p.fillAlpha ?? Math.min(1, Math.max(ownAlpha, dilated))
      if (ownAlpha <= 0) {
        const o = i * 3
        const so = s * 3
        e.rgb[o] = e.rgb[so]
        e.rgb[o + 1] = e.rgb[so + 1]
        e.rgb[o + 2] = e.rgb[so + 2]
      }
    }
    // 统一像素写入口：depth / α / transmission / ED 一次写齐。
    // `ED = α·depth` 是**表示层的不变量**（`depth` 是渲染期 α 加权期望深度 `ED/A`）：
    // depth-only 的 hidden 外推也必须同步 ED，否则合成深度 `ΣED/ΣA` 用的是陈旧值
    // （实测：α=0.2、D=1.2→2.5 时 ED 停在 0.24，合成 D 报 1.2，而真值 2.5）。
    writePixelState(e, i, z, a)
  }
  if (filledCount === 0) return 0

  // 深度平滑：只更新填充像素，取「填充或源」邻居的深度相似加权平均。
  const band = Math.max(1e-6, p.bandHi - p.bandLo)
  const depthScale = Math.max(1e-6, band * p.depthSimilarity)
  for (let pass = 0; pass < p.smoothPasses; pass++) {
    const depthPrev = Float32Array.from(e.depth)
    for (let i = 0; i < pixels; i++) {
      if (!filled[i]) continue
      const x = i % width
      const y = (i / width) | 0
      const di = depthPrev[i]
      let wSum = 0
      let dSum = 0
      for (let n = 0; n < 4; n++) {
        const xx = x + (n === 0 ? 1 : n === 1 ? -1 : 0)
        const yy = y + (n === 2 ? 1 : n === 3 ? -1 : 0)
        if (xx < 0 || xx >= width || yy < 0 || yy >= height) continue
        const j = yy * width + xx
        if (!filled[j] && !(e.alpha[j] >= supportAlpha && e.visible[j]))
          continue
        const w = 1 / (1 + Math.abs(depthPrev[j] - di) / depthScale)
        wSum += w
        dSum += w * depthPrev[j]
      }
      if (wSum <= 0) continue
      const z = clamp(dSum / wSum, p.bandLo, p.bandHi)
      writePixelState(e, i, z, e.alpha[i])
    }
  }
  return filledCount
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v
}

/**
 * 逐像素状态的**唯一写入口**：`depth` / `α` / `transmission` / `accumulatedDepth` 一次写齐。
 *
 * 存在的理由：这三个量不是独立的（`transmission = 1−α`、`ED = α·D`），
 * 分散地写就会出现「改了 D、忘了 ED」这类不一致 —— 而且**只在部分路径上**发生
 * （hidden 的 depth-only 分支），`layering-golden` 的 28 条门当时全绿。
 */
function writePixelState(
  e: EditableFrame,
  i: number,
  depth: number,
  alpha: number,
): void {
  e.depth[i] = depth
  e.alpha[i] = alpha
  e.transmission[i] = 1 - alpha
  e.accumulatedDepth[i] = alpha * depth
  e.visible[i] = 1
}

/**
 * 逐像素最近支撑像素（多源 4-连通 BFS，限半径 `maxDist`）。
 * 用于 own-gap 小洞填补：给出「这个洞可以从哪借色/深/α」。
 * `src[i] = -1` = 半径内无支撑（不算小洞）。
 */
function nearestSupportSource(
  alpha: Float32Array,
  visible: Uint8Array,
  depth: Float32Array,
  supportAlpha: number,
  width: number,
  height: number,
  maxDist: number,
): { src: Int32Array; dist: Int32Array } {
  const pixels = width * height
  const src = new Int32Array(pixels).fill(-1)
  const dist = new Int32Array(pixels).fill(-1)
  const queue = new Int32Array(pixels)
  let head = 0
  let tail = 0
  for (let i = 0; i < pixels; i++) {
    if (alpha[i] >= supportAlpha && visible[i] && depth[i] > 0) {
      src[i] = i
      dist[i] = 0
      queue[tail++] = i
    }
  }
  while (head < tail) {
    const i = queue[head++]
    const d = dist[i]
    if (d >= maxDist) continue
    const x = i % width
    const y = (i / width) | 0
    for (let n = 0; n < 4; n++) {
      const xx = x + (n === 0 ? 1 : n === 1 ? -1 : 0)
      const yy = y + (n === 2 ? 1 : n === 3 ? -1 : 0)
      if (xx < 0 || xx >= width || yy < 0 || yy >= height) continue
      const j = yy * width + xx
      if (dist[j] !== -1) continue
      dist[j] = d + 1
      src[j] = src[i]
      queue[tail++] = j
    }
  }
  return { src, dist }
}

/**
 * 「洞」判据：沿某一轴的两侧 `radius` 内**都有同表面**支撑。
 *
 * 两条：
 * 1. 只在一侧有支撑 = 边缘外点（正是旧 own-gap 在外缘画出 `α=1` 硬点的那种像素），不补；
 * 2. 两侧都有支撑、但深度差超过 `depthTol·min(z)` ⇒ 这是**两个不同表面之间**的真实开口
 *    （栏杆间隙、台阶、层内断层），不补 —— 补了就是把深度断口糊成一张布。
 */
function isEnclosedSameSurface(
  alpha: Float32Array,
  depth: Float32Array,
  x: number,
  y: number,
  width: number,
  height: number,
  supportAlpha: number,
  radius: number,
  depthTol: number,
): boolean {
  const along = (dx: number, dy: number): boolean => {
    let near = -1
    let far = -1
    for (let s = 1; s <= radius; s++) {
      const xx = x + dx * s
      const yy = y + dy * s
      if (xx < 0 || xx >= width || yy < 0 || yy >= height) return false
      const j = yy * width + xx
      if (alpha[j] < supportAlpha || !(depth[j] > 0)) continue
      if (near < 0) near = depth[j]
      else far = depth[j]
      if (far > 0) break
    }
    if (near < 0 || far < 0) return false
    const lo = Math.min(near, far)
    const hi = Math.max(near, far)
    return hi - lo <= depthTol * Math.max(1e-6, lo)
  }
  return (along(-1, 0) && along(1, 0)) || (along(0, -1) && along(0, 1))
}

/**
 * 原图（sRGB uint8 HWC）双线性重采样到层分辨率 + 线性化，返回直通线性 RGB `[w*h*3]`。
 *
 * 参考相机 == 原图相机 ⇒ 映射是恒等 + 缩放，所以只需要分辨率重采样，不需要内参。
 * 双线性对越界索引做 clamp（与 `sharp/preprocess.ts` 的 `F.interpolate` 语义一致）。
 */
export function resampleImageToLinear(
  src: SourceImage,
  width: number,
  height: number,
): Float32Array {
  const out = new Float32Array(width * height * 3)
  const { data, width: sw, height: sh, channels: ch } = src
  for (let y = 0; y < height; y++) {
    const fy = ((y + 0.5) * sh) / height - 0.5
    const y0 = Math.min(sh - 1, Math.max(0, Math.floor(fy)))
    const y1 = Math.min(sh - 1, y0 + 1)
    const ty = Math.min(1, Math.max(0, fy - y0))
    for (let x = 0; x < width; x++) {
      const fx = ((x + 0.5) * sw) / width - 0.5
      const x0 = Math.min(sw - 1, Math.max(0, Math.floor(fx)))
      const x1 = Math.min(sw - 1, x0 + 1)
      const tx = Math.min(1, Math.max(0, fx - x0))
      const o = (y * width + x) * 3
      for (let c = 0; c < 3; c++) {
        const p00 = data[(y0 * sw + x0) * ch + c]
        const p10 = data[(y0 * sw + x1) * ch + c]
        const p01 = data[(y1 * sw + x0) * ch + c]
        const p11 = data[(y1 * sw + x1) * ch + c]
        const top = p00 + (p10 - p00) * tx
        const bot = p01 + (p11 - p01) * tx
        out[o + c] = srgbToLinear((top + (bot - top) * ty) / 255)
      }
    }
  }
  return out
}

/**
 * 对整摞层做「原图回写 + 几何补齐」，返回**新的一摞**。
 *
 * 流程：原图重采样到层分辨率 → （可选）**几何补齐**（own-gap + hidden）
 * → 逐层遮挡掩码 → 逐层权重 → **合成残差回写**。
 *
 * ⚠ **补齐必须排在回写之前**：残差回写的覆盖率门 `A_total ≥ minCoverage` 要用
 * **最终**的 α。先回写再补齐的话，own-gap 会把 α 抬到 1，那些像素从没被回写过，
 * 「覆盖满 ⇒ 合成 == 原图」这条性质就不成立（实测这类像素正是残差尾部的来源）。
 *
 * ⚠ 颜色回写只在参考视角成立；几何补齐改 `depth`/`alpha`，会改变「合成 == 全量」的
 * 恒等（这是有意的：它补的是全量渲染本身就缺的背景）。是否开启由调用方决定。
 */
export function refineLayers(
  layered: LayeredRGBD,
  image: SourceImage,
  options: RefineLayersOptions = {},
): LayeredRGBD {
  const L = layered.frames.length
  if (L === 0) return layered
  const { width, height } = layered
  const pixels = width * height
  const imageLinear = resampleImageToLinear(image, width, height)

  // ① 几何补齐（own-gap + hidden）：先改 α/depth，回写的覆盖率门才看得到最终 α。
  const completed: readonly WSplatFrame[] =
    options.complete === false
      ? layered.frames
      : completeLayerGeometry(
          layered.frames,
          layered.placement,
          imageLinear,
          width,
          height,
          options.complete ?? {},
        )

  // ② 遮挡：每像素只回写**最前的不透明层**（参考视角的可见性）。
  const occluded = computeOcclusionMasks(completed, pixels)

  const foregroundLayers = options.foregroundLayers ?? L
  const weights: Float32Array[] = completed.map((frame, k) => {
    const weight = computeRefineWeight(frame, {
      minAlpha: options.minAlpha,
      smoothnessScale: options.smoothnessScale,
      smoothnessExponent: options.smoothnessExponent,
      foreground: k < foregroundLayers,
      occluded: occluded[k],
    })
    return weight
  })
  // 合成残差回写：`Σ v_k·Δc_k ≈ I − C`，所以**参考视角逐层 over 合成 == 原图**，
  // 且 `r = 0` 的像素逐位不动（旧的逐层混色会把已对的像素写坏）。
  const colorRefined: WSplatFrame[] =
    options.residual === false
      ? completed.map((frame, k) => ({
          ...frame,
          rgb: blendImageWriteback(frame.rgb, imageLinear, weights[k]),
        }))
      : applyResidualWriteback(completed, imageLinear, weights, {
          ...options.writeback,
          stats: options.stats ?? options.writeback?.stats,
        })
  return { ...layered, frames: colorRefined }
}

function srgbToLinear(x: number): number {
  const c = x < 0 ? 0 : x > 1 ? 1 : x
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
}
