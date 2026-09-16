/**
 * wsplat 的纯数据类型。
 *
 * 本文件**不得** import 任何 WebGPU / node 类型：它是 web / node 共用的数据契约层
 * （对照 `sharp/types.ts` 的定位）。GPU 相关的类型都在 `device.ts` / `index.ts`。
 */

/** 渲染尺寸（像素）。 */
export interface WSplatSize {
  width: number
  height: number
}

/**
 * resolve 之后的逐像素产物。
 *
 * 全部按「行主序、每像素若干通道」紧排。
 *
 * ── 这些字段就是与下游（分层 -> mesh）的契约，改之前先想清楚 ──
 * 1. `depth` 是**真实度量深度**（视图空间 z，米），**不是** NDC：下游的层边界 /
 *    near-far / InvZ 全部建在真实深度上（`NDCDepthToRealDepth` / `RealDepthToNDCDepth`）。
 *    NDC 转换留给调用方，因为只有它知道自己要用哪个投影。
 * 2. `transmission` 必须同时暴露：它是分层合成的核心中间量（`1 - A`），
 *    下游 `GetTransmittance` 用的就是它，不额外开 pass。
 * 3. 累加缓冲是**预乘**的（`Σ T·α·c`），所以同时给直通版本（`rgb`）——
 *    预乘便于合成，直通便于显示。
 * 4. 逐像素深度缓冲要能被调用方**统计**（下游 `DisparityStats`：min/max、一~四阶矩、
 *    直方图 bins、分位数），所以深度以 f32 附件读回、绝不做非线性压缩。
 * 5. 渲染用的**投影矩阵必须能随输出一起导出**（`WSplatCamera` 里带着），
 *    否则下游无法把 NDC / InvZ 反推回真实深度。本项目固定为针孔透视，理由见 `camera.ts`。
 *
 * ⚠ 合成多层时要用 `rgba`（线性 f32）+ `alpha` + `accumulatedDepth`，
 * **不要**用 `preview`——它是 8-bit + sRGB，仅供目视。
 * 跨层合并深度必须在 (ED, A) 空间做：`D_total = Σ ED_k / Σ A_k`。
 */
export interface WSplatFrame {
  readonly width: number
  readonly height: number
  /** sRGB 预览，长度 `w*h*4`（RGBA8）。仅用于目视 / 存 PNG，不参与数值验收。 */
  readonly preview: Uint8Array
  /** **直通**线性 RGB，长度 `w*h*3`（f32）。`A <= 阈值` 处为 0。 */
  readonly rgb: Float32Array
  /** 累积 alpha A，长度 `w*h`。 */
  readonly alpha: Float32Array
  /** 真实度量深度 D（米），长度 `w*h`；背景处为 0，请用 `visible` 过滤。 */
  readonly depth: Float32Array
  /** 透射率 T = 1 - A，长度 `w*h`。 */
  readonly transmission: Float32Array
  /** 未归一化的累积深度 ED = Σ T·α·z，长度 `w*h`（gsplat accumulated depth 语义）。 */
  readonly accumulatedDepth: Float32Array
  /** 1 = 该像素 `A > 阈值`，即 `depth` 有效。 */
  readonly visible: Uint8Array
}

/**
 * ⚠ 字段语义的两条硬边界（加新字段前先读这里）。
 *
 * 1. **`ED == A · D` 是层数据的不变量**：`depth` 是渲染期的 α 加权期望深度 `ED / A`
 *    （见 `resolve` 的 `D = ED/A`）。任何后处理只要动了 `depth` 或 `alpha`，就必须同步
 *    `accumulatedDepth`（`layering/refine.ts` 用唯一的 `writePixelState()` 兜住这件事，
 *    并有 `layerMomentResidual()` 体检）。改了一处忘了另一处，下游 `compositeAlphaDepth`、
 *    责任层判定与 mesh 的合成深度就会用**陈旧值** —— 这种错不会让画面立刻崩，只会让
 *    「深度明明改了、合成还说旧值」。
 * 2. **`alpha` 是渲染覆盖，不是「我有多确定」**：`A` 回答的是「这个像素被画了多少」，
 *    与「这个像素的内容有多可信」是两件事（外推出来的像素可以有 `A=1`、置信度极低）。
 *    将来要传"来源 / 置信"就**另开字段**（如 `provenance: observed | splat-derived |
 *    extrapolated | inpainted | fallback` + 独立 confidence），别把语义塞进 `alpha`。
 */

/** resolve 阶段的旋钮。 */
export interface WSplatResolveOptions {
  /**
   * `A <= visibleAlphaThreshold` 的像素视为背景：不输出 D（`visible = 0`）。
   * 默认 0：只有完全没被覆盖（A == 0）的像素才标背景。
   */
  visibleAlphaThreshold?: number
}

/**
 * 一帧的**剔除统计**（GPU 顶点阶段的原子计数，`renderSplats()` 每次自动清零）。
 *
 * ── 为什么需要它 ──
 * `minPixelSize = 2` 是**像素量纲**：实测同一份数据在 384 宽下有 **46.72%** 的高斯
 * 被它剔掉，而原分辨率下是 **0%**。这些剔除是**静默**的（不报错、不警告）。
 * 在 `splat → 多层 RGBAD` 的流程里，这会让某一层"看起来空了一块"而没人知道原因，
 * 所以每个剔除原因都单独计数并可以从 CPU 读回。
 *
 * 恒等式（可当自检用）：`drawn + 各剔除项之和 == total`。
 */
export interface WSplatStats {
  /** 本次上传的高斯总数（= `draw` 的实例数）。 */
  total: number
  /** 通过全部顶点阶段剔除、真正参与光栅化的高斯数。 */
  drawn: number
  /** 排序下标 ≥ `numSplats`（上传数量与 `numSplats` 不一致时会 >0，正常为 0）。 */
  culledBounds: number
  /** **原始 opacity** ≤ `alphaClip`（上游在 AA 补偿之前先剔一次）。 */
  culledAlphaClip: number
  /** AA 补偿后 α ≤ `alphaClip`。 */
  culledAlphaClipAfterAa: number
  /** 视图空间 z ≤ 1e-6（相机后方 / 近零，投影会发散）。 */
  culledBehindCamera: number
  /** 屏幕空间最大直径 `max(l1,l2) < minPixelSize`（**分辨率相关**，最常咬人的一项）。 */
  culledMinPixelSize: number
  /** 视锥 x/y 剔除。 */
  culledFrustum: number
}
