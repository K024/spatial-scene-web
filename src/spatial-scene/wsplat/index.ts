/**
 * wsplat：把 splat（3D 高斯）渲染成**准确的 RGB + A + D**（WebGPU / WGSL）。
 *
 * ── 在全链路里的位置 ──
 * ```
 * image (+EXIF focal) -> splat 参数场 -> layered RGBAD -> mesh（交付物）
 * ```
 * 本模块只做**中间那一环的渲染**；splat 与 RGBAD 都是中间表示，splat 不出生成管线。
 * 分层 / 统计 / 切分由调用方负责：`renderSplats()` 渲染的就是当前上传的那批高斯，
 * 管线里**没有任何全局量**（`numSplats` 只用于边界检查），所以
 * 「渲染子集」与「在全集里渲染这个子集」数学恒等。名字一律 `WSplat*` 前缀
 *（`WSplatRenderer` / `WSplatCamera` / `WSplatFrame` / `WSplatGpuData`），
 * 与将来的场景渲染器区分。
 *
 * ── 本模块不做（非目标）──
 * 分层渲染 / 深度剥离 / 层采样策略、原图精修回写、场景图、相机控制、
 * 与 engine 材质系统耦合、tile-based compute 光栅器、GPU radix、LOD / octree / 流式加载、
 * 编辑操作（选择 / 变换 / palette / color grade）。
 *
 * ── 取向 ──
 * **准确度优先，性能次要**：允许慢；精度上不做 f16 加速取舍；排序正确性优先。
 * 耗时只作为观测值记录在脚本输出里，不是验收目标。
 *
 * ── 逐层渲染（LDI）怎么用：排列的所有权在调用方 ──
 * 渲染器只认一张**排列** `splatOrder: instanceIndex -> splatIndex`，不关心它的语义。
 * 「哪个高斯属于哪一层」是 layering 的事，所以它也只在那里维护一张**层表**
 * `{base, count}[L]` 指向排列里的连续段：
 * ```
 * setGaussians(全部) -> setCamera(camera) -> setSplatOrder(按层分组的排列)
 * for k in 远..近: drawLayer(table[k])   // 内部 clear
 * ```
 * 为什么排列归调用方，而不是「层 = 全局排序的一段区间」：
 * - 层间重叠（`rangeOverlap`）与密度补偿（densify）会让同一个高斯出现在**多个层**，
 *   那时排列长度 > 高斯数，区间方案要么重传要么重排；
 * - 非深度分带的层（语义 / 物体 / 前景区块）在全局深度排序里**根本不连续**，
 *   区间方案表达不了。
 * 代价："各段之并 = 全表且有序" 从自动变成需要证明，所以 layering 侧必须把它当成
 * 一条门来验（排列合法 + 每层内部有序 + 层序还→近）。
 *
 * 因为 `over` 是**结合的**，逐段渲染再按序合成与一次画完恒等；又因为每层各自 clear，
 * 所以**层间顺序无关**（每层的颜色/alpha/ED 都只反映本层，不被更近的层衰减）——
 * 这正是视差渲染要的 LDI 语义：每层持有自己原本的颜色。跨层合并深度必须在
 * (ED, A) 空间做：`D_total = Σ ED_k / Σ A_k`（**不能**直接平均各层的 D）。
 * 层的最大风险是「被静默剔空」（`minPixelSize` 是像素量纲）：用 `countCulls()` /
 * `readSplatStats()` 观测剔除；层内深度跨度（跨度 = 视差平移的潜在错位量，
 * 是「该切几层」的直接判据）可以直接从 `WSplatFrame.depth` 统计。
 *
 * ── 数据契约 ──
 * `types.ts` 是纯数据类型层（**不得** import WebGPU / node），`WSplatFrame` 的字段语义
 * 就是与下游的契约（真实度量深度 / 透射率 / 预乘 / 可导出投影）。
 * `src/` 内不 import node 内置模块；node 专属代码只放 `scripts/`。
 *
 * ── 上游来源与许可（务必随代码保留）──
 * WGSL 数学 fork 自 `playcanvas/engine` **v2.22.2**（MIT，`Copyright (c) 2011-2026
 * PlayCanvas Ltd.`，commit `6e6d0d0690830fcd1d91ffde15e9c87a062e8c43`）。
 * 上游文件**没有**逐文件 license 头，所以每个 fork 文件都自带来源头
 *（`wgsl/chunks/*.ts` 的文件首注释四条：upstream / commit / modifications / 说明），
 * **任何修改都必须登记在 `modifications:` 里**。只 fork WGSL 数学，
 * 不依赖 `playcanvas` 包本身（JS 侧管线自建）。
 * 参考而**不** fork：`playcanvas/supersplat`（MIT）、`nerfstudio-project/gsplat`
 *（Apache-2.0；深度与 antialias 语义的权威来源）。
 * 需要同步上游时：手动 diff 上面那个 commit，它是 pin 住的基线。
 *
 * ── 校验入口 ──
 * | 脚本 | 作用 |
 * |---|---|
 * | `scripts/wsplat-check-rendering.ts` | **唯一判定入口**（阈值只在它里面，退出码非 0 = 不过门）：排序不变量 / 预乘 over 解析解 / over 方向 / 雅可比符号 / 真实场景 GPU↔CPU（NCC、MAE、深度、剔除）/ LDI 合成 |
 * | `scripts/utils/wsplat-cpu.ts` | 参考光栅器：gsplat 前向公式的 CPU 实现（默认对齐 SHARP 的 `classic` + `eps2d=0`；与 WGSL 同尺度、同剔除） |
 * | `scripts/utils/webgpu.ts` | node 侧 Dawn 设备（长期持有引用 + 强制 destroy） |
 *
 * `minPixelSize` 是像素量纲，低分辨率下的结论不能外推；把 `eps2d` 调大
 *（如 gsplat 默认 0.3）会把极小高斯的 quad 撑过阈值，低分辨率下该项可能整项为 0。
 */

import type { Gaussians3D } from "../sharp/types.ts"
import type { WSplatCamera } from "./camera.ts"
import {
  type CreateWSplatGpuDataOptions,
  createWSplatGpuData,
  type WSplatGpuData,
} from "./data.ts"
import {
  createBindGroupLayouts,
  createRenderPipeline,
  createRenderTexture,
  createShaderModule,
  createUniformSlot,
  parseDefs,
  readBufferU32,
  readTexturePixels,
  type UniformSlot,
  withValidationScope,
} from "./device.ts"
import { computeViewDepths, sortSplatsBackToFront } from "./sort.ts"
import type {
  WSplatFrame,
  WSplatResolveOptions,
  WSplatSize,
  WSplatStats,
} from "./types.ts"
import { assembleWgsl, type DefineMap, type WgslUnit } from "./wgsl/assemble.ts"
import { blitWgsl } from "./wgsl/blit.ts"
import { gsplatCenterChunk } from "./wgsl/chunks/gsplatCenter.ts"
import { gsplatCommonChunk } from "./wgsl/chunks/gsplatCommon.ts"
import { gsplatCornerChunk } from "./wgsl/chunks/gsplatCorner.ts"
import { gsplatEvalSHChunk } from "./wgsl/chunks/gsplatEvalSH.ts"
import { gsplatQuatToMat3Chunk } from "./wgsl/chunks/gsplatQuatToMat3.ts"
import { gsplatStructsChunk } from "./wgsl/chunks/gsplatStructs.ts"
import { screenQuadChunk } from "./wgsl/chunks/screenQuad.ts"
import { splatDataChunk } from "./wgsl/chunks/splatData.ts"
import { splatSourceChunk } from "./wgsl/chunks/splatSource.ts"
import { cullStatsWgsl } from "./wgsl/cullStats.ts"
import { presentWgsl } from "./wgsl/present.ts"
import { quadWgsl } from "./wgsl/quad.ts"
import { resolveWgsl } from "./wgsl/resolve.ts"

/**
 * splat pass 的两个累加附件格式。
 *
 * 必须可 blend（`one / one-minus-src-alpha`）：`rgba16float` 是 core 里能 blend 的
 * 最高精度格式；`r32float` / `rgba32float` 需要 `float32-blendable` 扩展，不用。
 */
export const SPLAT_COLOR_FORMAT: GPUTextureFormat = "rgba16float"

/**
 * 剔除统计的槽位数（WGSL 侧为 `array<atomic<u32>, 8>`，实际用前 7 个）。
 *
 * 索引必须与 `wgsl/chunks/splatData.ts` 里的 `SPLAT_STAT_*` 常量一致，
 * 读取顺序见 `readSplatStats()`。
 */
const SPLAT_STATS_SLOTS = 8
export const SPLAT_DEPTH_FORMAT: GPUTextureFormat = "rgba16float"

/**
 * resolve 的三个输出格式，三者之和 4 + 8 + 16 = 28 字节/采样，
 * 必须 ≤ `maxColorAttachmentBytesPerSample`（WebGPU 默认 32，浏览器普遍就取默认值）。
 * 这也是为什么预览用 unorm8、直通颜色用 f16、只有深度用 f32。
 */
export const RESOLVE_PREVIEW_FORMAT: GPUTextureFormat = "rgba8unorm"
export const RESOLVE_RGBA_FORMAT: GPUTextureFormat = "rgba16float"
export const RESOLVE_DEPTH_FORMAT: GPUTextureFormat = "rgba32float"

/** 剔除阈值默认值：**取自上游**，不自创（见 `wgsl/chunks/gsplatCommon.ts` 的数值来源说明）。 */
const DEFAULT_MIN_PIXEL_SIZE = 2
const DEFAULT_ALPHA_CLIP = 1 / 255
/**
 * 默认渲染参数**对齐 SHARP / ml-sharp 的官方渲染口径**
 * （`ml-sharp/src/sharp/utils/gsplat.py` 的 `GSplatRenderer.forward`）：
 * `rasterize_mode="classic"` + `eps2d=0` —— 既不做透明度补偿，也不做低通模糊。
 *
 * 为什么以它为准：本渲染器服务的产物就是 SHARP（见 `sharp/` 与 `export/`）
 * 模型输出的高斯；用与训练/官方渲染不同的 AA 只会引入系统性色偏
 * （实测 `eps2d=0` 比 `eps2d=0.3` 更接近原图：NCC 0.9725 vs 0.9711、
 * MAE 9.74 vs 10.07/255、亮度比 1.0023 vs 0.9983，见 [F] 段）。
 *
 * WGSL 的 Σ2 是 gsplat 的 4 倍，所以注入着色器时 `eps2d × 4`。
 */
const DEFAULT_EPS2D = 0
const DEFAULT_ANTIALIAS = false

/** 所有 `#include` 可用的 chunk。 */
const CHUNKS: readonly WgslUnit[] = [
  splatSourceChunk,
  screenQuadChunk,
  splatDataChunk,
  gsplatStructsChunk,
  gsplatQuatToMat3Chunk,
  gsplatEvalSHChunk,
  gsplatCenterChunk,
  gsplatCornerChunk,
  gsplatCommonChunk,
]

/** splat 管线的编译开关。`GSPLAT_AA` 对应上游同名分支；SHARP 只有 DC，故 `SH_BANDS=0`。 */
const SPLAT_DEFINES: DefineMap = { SH_BANDS: 0 }

/** 相机 y 轴朝向：OpenCV（向下，本项目默认）还是 OpenGL（向上，上游约定）。 */
export type WSplatCameraYAxis = "down" | "up"

export interface WSplatRendererOptions {
  /** 初始渲染尺寸，默认 256×256。 */
  size?: WSplatSize
  /** resolve 旋钮（可见 alpha 阈值等）。 */
  resolve?: WSplatResolveOptions
  /**
   * 画布格式（浏览器传 `navigator.gpu.getPreferredCanvasFormat()`）。
   * 给了就在 init 里一并把 present 管线建好——**故意**放在同一个 validation scope 里，
   * 因为 Dawn-node 反复 push/pop error scope 会崩（见 `device.ts` 的注释）。
   */
  presentFormat?: GPUTextureFormat
  /** 球谐阶数。SHARP 只给 DC，默认 0（共享代码已实现 1..3 度求值，留接口）。 */
  shBands?: 0 | 1 | 2 | 3
  /** 屏幕空间半径剔除阈值（像素）。默认 2，取自上游。 */
  minPixelSize?: number
  /** 低 alpha 剔除阈值。默认 1/255，取自上游。 */
  alphaClip?: number
  /**
   * 相机 y 轴朝向，默认 `"down"`（OpenCV 约定，与 SHARP 的度量空间一致）。
   *
   * 它只影响协方差雅可比的第二行符号（见 `wgsl/chunks/gsplatCorner.ts` 文件头）。
   * 保留 `"up"` 是为了：(1) 与上游 playcanvas 逐行对拍，(2) 跑
   * `scripts/wsplat-check-rendering.ts` 的雅可比符号 A/B 验证（[D] 段，会证明哪一个才对）。
   */
  cameraYAxis?: WSplatCameraYAxis
  /**
   * 是否开启抗锯齿的**透明度补偿**（gsplat 的 `rasterize_mode="antialiased"`）。
   * 默认 `false` —— 对齐 SHARP / ml-sharp 的官方渲染口径（`classic`）。
   *
   * ⚠ 它只控制补偿；低通模糊由 `eps2d` 单独控制，两者相互独立（gsplat 同款）。
   * 且 `eps2d=0` 时补偿恒等于 1，此时开不开没有差别。
   * 打开它 + `eps2d=0.3` 才是 gsplat 的 `antialiased` 口径；对照实验见
   * `scripts/wsplat-check-rendering.ts` 的 [F] 段。
   */
  antialias?: boolean
  /**
   * 低通滤波的 eps2d（**gsplat 语义**：加在真像素焦距的 Σ2 上）。
   * 默认 `0` —— 对齐 SHARP / ml-sharp 的官方渲染（`GSplatRenderer` 传 `eps2d=0`）。
   *
   * WGSL 的 Σ2 建在 `focal = 2·f_px` 上，所以注入着色器的是 `eps2d × 4`。
   * gsplat 的默认值是 **0.3**（与 `rasterize_mode` 无关，无条件加到 2D 协方差）；
   * ml-sharp 显式覆盖成 0，即不做低通模糊。要复现 gsplat 默认就显式传 `0.3`。
   */
  eps2d?: number
}

export interface WSplatRenderer {
  readonly device: GPUDevice
  readonly width: number
  readonly height: number
  /** resolve 预览纹理（`rgba8unorm`）；resize 后对象会变，请重新获取。 */
  readonly previewTexture: GPUTexture
  resize(size: WSplatSize): void
  /**
   * 清空两个累加附件（`loadOp: clear`）。每帧画任何东西之前调用一次。
   *
   * 与 draw 分开是有意的：真实 splat pass 一帧一次 draw，但将来做 depth peeling /
   * 多层时会有多次 draw 叠加到同一批附件上，那时只有第一层需要 clear。
   */
  clearSplatBuffers(): void
  /**
   * 通路验证：用纯色填满整屏，**累加**进输出契约规定的两个附件（`loadOp: load`）。
   *
   * @param rgba 线性 RGB + straight α
   * @param depthMeters 该「splat」的真实度量深度（米），默认 1
   */
  renderSolidColor(rgba: readonly number[], depthMeters?: number): void
  /** 把累加缓冲 resolve 到内部纹理（只编码，不提交）。 */
  encodeResolve(
    encoder: GPUCommandEncoder,
    options?: WSplatResolveOptions,
  ): void
  /** resolve + 提交 + 读回 CPU。 */
  readback(options?: WSplatResolveOptions): Promise<WSplatFrame>
  /**
   * 跑一趟 compute，把逐 splat 的**剔除原因**原子计数（供 `readSplatStats()` 读回）。
   *
   * 必须单独一趟的原因：WebGPU 不允许 vertex 阶段做 `read_write` storage 的
   * `atomicAdd`。这一趟调用的是与顶点阶段**完全相同**的 cull 函数
   * （见 `wgsl/cullStats.ts`），所以判据不会漂移。
   *
   * 每次调用都会先清零，所以统计是"这一次调用"的；不需要统计时不要调（零成本）。
   */
  countCulls(): void
  /**
   * 读回**上一次 `countCulls()`** 的剔除统计。
   *
   * `renderSplats()` 每次都会把计数器清零，所以要在下一次 `renderSplats()`
   * 之前调用。恒等式（可当自检）：`drawn + 各剔除项之和 == total`。
   *
   * 用途：分层（LDI）时判断"某一层是不是被静默剔空了"——`minPixelSize`
   * 是像素量纲，实测 384 宽下会剔掉 46.72% 的高斯（原分辨率 0%）。
   */
  readSplatStats(): Promise<WSplatStats>
  /** 上传高斯（一次性；之后只改 uniform 与排序）。重复调用会替换。 */
  setGaussians(
    gaussians: Gaussians3D,
    options?: CreateWSplatGpuDataOptions,
  ): void
  /** 设置相机（只写 uniform，不重排）。 */
  setCamera(camera: WSplatCamera): void
  /**
   * 按当前相机重算排列（CPU 确定性 radix，back-to-front），写进 order buffer。
   *
   * 相机不动时可以不调：结果一致，只是排列沿用上一次。
   *
   * ⚠ 分层路径**不要**用它：它会用全局 back-to-front 覆盖掉按层分组的排列。
   * 分层应该自己算 `computeViewDepths` + `sortSplatsBackToFront`，再合并成
   * 按层分组的排列，最后一次性 `setSplatOrder()`。
   */
  sort(camera?: WSplatCamera): void
  /**
   * 画一帧 splat，`draw(6, 排列长度)`，vertex pulling。
   *
   * @param options.clear `true`（默认）时同时清空两个累加附件；
   *   多次 draw 叠加（将来的 depth peeling）时传 `false`。
   */
  renderSplats(options?: { clear?: boolean }): void
  /**
   * 只画**排列**里的一段连续区间 `[firstInstance, firstInstance + instanceCount)`。
   *
   * 这是「逐层渲染（LDI）」的最低层原语：`over` 可结合，所以只要排列本身是
   * 合法的绘制顺序（层间还→近、层内 back-to-front），切一段渲染与「在全集里
   * 渲染这段」**数学恒等**。顶点阶段读的 `instance_index` 就是排列里的位置，
   * 于是只需要 draw 的 `firstInstance` / `instanceCount`。
   *
   * 因为每层各自 `clear`，层间顺序无关 —— 这正是 LDI 要的语义：每层持有自己原本的颜色。
   *
   * ⚠ 区间会被记下来，供随后的 `countCulls()` / `readSplatStats()` 统计**同一段**
   *（否则「这一层被静默剔空了」会被全场统计淹没，完全看不出来）。
   *
   * ⚠ 这是**区间**原语：它假设层是排列里的一段。「层是任意下标集合」请用 `drawLayer()`
   * —— 那时候该由调用方把层排成排列里的连续段（layering 的层表就是这样做的）。
   */
  renderSplatsRange(options: {
    firstInstance: number
    instanceCount: number
    clear?: boolean
  }): void
  /**
   * 换掉整个**排列**：`instanceIndex -> splatIndex`（见 `WSplatGpuData.uploadOrder`）。
   *
   * 排列的所有权归调用方（layering）：它才是知道「哪一段属于哪一层」的一方。
   * 渲染器只保证「按你给的顺序画」。
   *
   * 长度可以 `<=` orderCapacity 的任意值，也可以 `> count`（重叠 / 密度补偿）。
   * 换完之后 `renderSplats()` 画的是整张排列表。
   */
  setSplatOrder(order: Uint32Array, length?: number): void
  /** 当前排列长度。 */
  readonly permutationLength: number
  /**
   * 画一层：`table[k] = { base, count }` 指向排列里的一段（layering 的层表）。
   *
   * 就是 `renderSplatsRange({ firstInstance: base, instanceCount: count })`，
   * 但把「层表」这个概念放到 API 上，调用方就不需要自己维护 `base = Σcount`。
   */
  drawLayer(
    layer: { readonly base: number; readonly count: number },
    options?: { clear?: boolean },
  ): void
  /** 把预览纹理送进目标视图（画布）；格式须已在 `options.presentFormat` 里给出。 */
  encodePresent(
    encoder: GPUCommandEncoder,
    target: GPUTextureView,
    format: GPUTextureFormat,
  ): void
  destroy(): void
}

/**
 * 建渲染器。
 *
 * async 是因为建管线要用一个 validation error scope 包起来（把 WGSL 错误的行列信息
 * 直接抛出来），而 error scope 是异步的。
 */
export async function createWSplatRenderer(
  device: GPUDevice,
  options: WSplatRendererOptions = {},
): Promise<WSplatRenderer> {
  const renderer = new Renderer(device, options)
  await renderer.init()
  return renderer
}

interface Targets {
  readonly splatColor: GPUTexture
  readonly splatDepth: GPUTexture
  readonly preview: GPUTexture
  readonly rgba: GPUTexture
  readonly depth: GPUTexture
}

class Renderer implements WSplatRenderer {
  readonly device: GPUDevice
  private widthValue: number
  private heightValue: number
  private readonly resolveDefaults: WSplatResolveOptions
  private readonly presentFormat: GPUTextureFormat | undefined

  private targets!: Targets
  private resolveLayout!: GPUBindGroupLayout
  private splatLayout!: GPUBindGroupLayout
  private blitPipeline!: GPURenderPipeline
  private resolvePipeline!: GPURenderPipeline
  private splatPipeline!: GPURenderPipeline
  private readonly presentPipelines = new Map<
    GPUTextureFormat,
    GPURenderPipeline
  >()
  private readonly presentLayouts = new Map<
    GPUTextureFormat,
    GPUBindGroupLayout
  >()

  private blitUniforms!: UniformSlot
  private resolveUniforms!: UniformSlot
  private splatUniforms!: UniformSlot
  /**
   * 本趟绘制的排列区间。
   *
   * **纯 CPU 记账**，不写进顶点 uniform：
   * - 光栅化那趟靠 `draw(6, count, 0, base)` 切区间，顶点阶段自己就能从
   *   `@builtin(instance_index)` 拿到全局实例号，**不需要**这两个数
   *   （所以 `SplatUniforms` 不必变形）；
   * - 但 `countCulls()` 要它们决定 dispatch 多少个 workgroup，
   *   `readSplatStats().total` 也要它 —— compute 没有 `firstInstance`。
   */
  private instanceOffset = 0
  private instanceCount = 0
  /** 当前排列长度（可 > 高斯数：重叠 / 加密）。 */
  private permutationLengthValue = 0
  private blitBindGroup!: GPUBindGroup
  private resolveBindGroup!: GPUBindGroup
  private splatBindGroup!: GPUBindGroup
  private readonly presentBindGroups = new Map<GPUTextureFormat, GPUBindGroup>()

  private splatStatsBuffer!: GPUBuffer
  private cullStatsPipeline!: GPUComputePipeline
  private cullStatsLayout!: GPUBindGroupLayout
  private cullStatsBindGroup!: GPUBindGroup
  /**
   * 剔除统计那趟的区间 `(firstInstance, instanceCount, 0, 0)`。
   *
   * 单独开一个 uniform，而不是塞进 `SplatUniforms`：光栅化那趟靠
   * `draw(6, count, 0, base)` 切区间，顶点阶段根本不读它；
   * 共用表保持最小 = 分层不需要动顶点 pipeline 的 uniform 布局。
   */
  private cullRangeUniforms!: UniformSlot

  private gpuData: WSplatGpuData | undefined
  private camera: WSplatCamera | undefined
  private readonly minPixelSize: number
  private readonly alphaClip: number
  private readonly shBands: 0 | 1 | 2 | 3
  private readonly cameraYAxis: WSplatCameraYAxis
  private readonly antialias: boolean
  /** gsplat 语义的 eps2d（真像素焦距）；注入 WGSL 时 ×4。 */
  private readonly eps2d: number

  constructor(device: GPUDevice, options: WSplatRendererOptions) {
    this.device = device
    this.widthValue = options.size?.width ?? 256
    this.heightValue = options.size?.height ?? 256
    this.resolveDefaults = {
      visibleAlphaThreshold: options.resolve?.visibleAlphaThreshold ?? 0,
    }
    this.presentFormat = options.presentFormat
    this.minPixelSize = options.minPixelSize ?? DEFAULT_MIN_PIXEL_SIZE
    this.alphaClip = options.alphaClip ?? DEFAULT_ALPHA_CLIP
    this.shBands = options.shBands ?? 0
    this.cameraYAxis = options.cameraYAxis ?? "down"
    this.antialias = options.antialias ?? DEFAULT_ANTIALIAS
    this.eps2d = options.eps2d ?? DEFAULT_EPS2D
  }

  get width(): number {
    return this.widthValue
  }

  get height(): number {
    return this.heightValue
  }

  get previewTexture(): GPUTexture {
    return this.targets.preview
  }

  async init(): Promise<void> {
    await withValidationScope(this.device, "wsplat 初始化", () => {
      this.createPipelines()
      if (this.presentFormat) this.createPresentPipeline(this.presentFormat)
      this.createTargets()
      // 剔除统计：需要 COPY_SRC 才能读回（resize 不复建）
      this.splatStatsBuffer = this.device.createBuffer({
        label: "wsplat:splatStats",
        size: SPLAT_STATS_SLOTS * 4,
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_DST |
          GPUBufferUsage.COPY_SRC,
      })
    })
  }

  resize(size: WSplatSize): void {
    if (size.width === this.widthValue && size.height === this.heightValue)
      return
    this.widthValue = size.width
    this.heightValue = size.height
    this.destroyTargets()
    this.createTargets()
  }

  clearSplatBuffers(): void {
    const encoder = this.device.createCommandEncoder({ label: "wsplat:clear" })
    const pass = encoder.beginRenderPass({
      label: "wsplat:clear",
      colorAttachments: [
        {
          view: this.targets.splatColor.createView(),
          clearValue: [0, 0, 0, 0],
          loadOp: "clear",
          storeOp: "store",
        },
        {
          view: this.targets.splatDepth.createView(),
          clearValue: [0, 0, 0, 0],
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    })
    pass.end()
    this.device.queue.submit([encoder.finish()])
  }

  setGaussians(
    gaussians: Gaussians3D,
    options?: CreateWSplatGpuDataOptions,
  ): void {
    this.gpuData?.destroy()
    const data = createWSplatGpuData(this.device, gaussians, options)
    // 初始排列 = 原顺序（先不排序；`sort()` / `setSplatOrder()` 会覆盖它）
    const identity = new Uint32Array(data.count)
    for (let i = 0; i < data.count; i++) identity[i] = i
    data.uploadOrder(identity)
    this.gpuData = data
    this.setFullRange(data.orderLength)
    if (this.camera) this.writeSplatUniforms()
    this.splatBindGroup = this.device.createBindGroup({
      label: "splat",
      layout: this.splatLayout,
      entries: [
        {
          binding: this.splatUniforms.binding,
          resource: { buffer: this.splatUniforms.buffer },
        },
        { binding: 1, resource: { buffer: data.rgba } },
        { binding: 2, resource: { buffer: data.geometry } },
        { binding: 3, resource: { buffer: data.order } },
      ],
    })
    // 剔除统计的 bind group：同一批资源 + stats buffer（compute 布局，多一个 binding）
    this.cullStatsBindGroup = this.device.createBindGroup({
      label: "cullStats",
      layout: this.cullStatsLayout,
      entries: [
        {
          binding: this.splatUniforms.binding,
          resource: { buffer: this.splatUniforms.buffer },
        },
        { binding: 1, resource: { buffer: data.rgba } },
        { binding: 2, resource: { buffer: data.geometry } },
        { binding: 3, resource: { buffer: data.order } },
        { binding: 4, resource: { buffer: this.splatStatsBuffer } },
        {
          binding: this.cullRangeUniforms.binding,
          resource: { buffer: this.cullRangeUniforms.buffer },
        },
      ],
    })
  }

  setCamera(camera: WSplatCamera): void {
    this.camera = camera
    this.writeSplatUniforms()
  }

  sort(camera?: WSplatCamera): void {
    const target = camera ?? this.camera
    if (!target) throw new Error("sort() 需要先 setCamera()")
    const data = this.gpuData
    if (!data) throw new Error("setGaussians() 之前不能排序")
    const depths = computeViewDepths(data.means, target.viewMatrix, data.count)
    const order = sortSplatsBackToFront(depths, data.count)
    data.uploadOrder(order)
    if (camera) {
      this.camera = camera
      this.writeSplatUniforms()
    }
  }

  get permutationLength(): number {
    return this.permutationLengthValue
  }

  setSplatOrder(order: Uint32Array, length?: number): void {
    const data = this.gpuData
    if (!data) throw new Error("setGaussians() 之前不能 setSplatOrder()")
    data.uploadOrder(order, length)
    this.setFullRange(data.orderLength)
    this.writeSplatUniforms()
  }

  /** 把本趟区间设成「整张排列」。 */
  private setFullRange(length: number): void {
    this.permutationLengthValue = length
    this.instanceOffset = 0
    this.instanceCount = length
  }

  /**
   * 把本趟区间写进**剔除统计自己的** uniform（顶点那趟不读它）。
   *
   * 由 `countCulls()` 自己写，而不是由 `renderSplatsRange()` 写：
   * 区间唯一的消费者就是剔除统计，这样就不用假设调用顺序。
   */
  private writeCullRange(firstInstance: number, instanceCount: number): void {
    this.cullRangeUniforms.view.set({
      offset: firstInstance,
      count: instanceCount,
    })
    this.cullRangeUniforms.upload()
  }

  renderSplats(options: { clear?: boolean } = {}): void {
    this.renderSplatsRange({
      firstInstance: 0,
      instanceCount: this.permutationLengthValue,
      clear: options.clear,
    })
  }

  drawLayer(
    layer: { readonly base: number; readonly count: number },
    options: { clear?: boolean } = {},
  ): void {
    this.renderSplatsRange({
      firstInstance: layer.base,
      instanceCount: layer.count,
      clear: options.clear,
    })
  }

  renderSplatsRange(options: {
    firstInstance: number
    instanceCount: number
    clear?: boolean
  }): void {
    const data = this.gpuData
    if (!data) throw new Error("setGaussians() 之前不能 renderSplats()")
    if (!this.camera) throw new Error("setCamera() 之前不能 renderSplats()")
    const clear = options.clear ?? true
    const firstInstance = Math.max(0, Math.floor(options.firstInstance))
    const instanceCount = Math.max(
      0,
      Math.min(
        Math.floor(options.instanceCount),
        this.permutationLengthValue - firstInstance,
      ),
    )
    if (firstInstance + instanceCount > data.orderCapacity) {
      throw new Error(
        `绘制区间 [${firstInstance}, ${firstInstance + instanceCount}) 超过排列容量 ${data.orderCapacity}`,
      )
    }
    this.instanceOffset = firstInstance
    this.instanceCount = instanceCount
    this.writeSplatUniforms()

    const encoder = this.device.createCommandEncoder({ label: "wsplat:splat" })
    const pass = encoder.beginRenderPass({
      label: "wsplat:splat",
      colorAttachments: [
        {
          view: this.targets.splatColor.createView(),
          clearValue: [0, 0, 0, 0],
          loadOp: clear ? "clear" : "load",
          storeOp: "store",
        },
        {
          view: this.targets.splatDepth.createView(),
          clearValue: [0, 0, 0, 0],
          loadOp: clear ? "clear" : "load",
          storeOp: "store",
        },
      ],
    })
    pass.setPipeline(this.splatPipeline)
    pass.setBindGroup(0, this.splatBindGroup)
    // 每个 splat 6 个顶点（两个三角形），实例数 = 本趟区间长度，
    // 起始实例号 = 排序序列里的偏移（顶点阶段直接拿它当排序下标）。
    pass.draw(6, instanceCount, 0, firstInstance)
    pass.end()
    this.device.queue.submit([encoder.finish()])
  }

  private writeSplatUniforms(): void {
    const camera = this.camera
    if (!camera) return
    const view = this.splatUniforms.view
    view.views.matrix_view.set(camera.viewMatrix)
    view.views.matrix_projection.set(camera.projectionMatrix)
    view.views.viewport_size.set(camera.viewportSize)
    view.set({
      minPixelSize: this.minPixelSize,
      numSplats: this.gpuData?.count ?? 0,
      alphaClipForward: this.alphaClip,
      debugMode: 0,
    })
    this.splatUniforms.upload()
  }

  renderSolidColor(rgba: readonly number[], depthMeters = 1): void {
    this.blitUniforms.view.set({
      color: [rgba[0] ?? 0, rgba[1] ?? 0, rgba[2] ?? 0, rgba[3] ?? 1],
      depth: [depthMeters, 0, 0, 0],
    })
    this.blitUniforms.upload()

    const encoder = this.device.createCommandEncoder({ label: "wsplat:blit" })
    const pass = encoder.beginRenderPass({
      label: "wsplat:blit",
      colorAttachments: [
        {
          view: this.targets.splatColor.createView(),
          loadOp: "load",
          storeOp: "store",
        },
        {
          view: this.targets.splatDepth.createView(),
          loadOp: "load",
          storeOp: "store",
        },
      ],
    })
    pass.setPipeline(this.blitPipeline)
    pass.setBindGroup(0, this.blitBindGroup)
    pass.draw(3)
    pass.end()
    this.device.queue.submit([encoder.finish()])
  }

  encodeResolve(
    encoder: GPUCommandEncoder,
    options?: WSplatResolveOptions,
  ): void {
    const threshold =
      options?.visibleAlphaThreshold ??
      this.resolveDefaults.visibleAlphaThreshold ??
      0
    this.resolveUniforms.view.set({ params: [threshold, 0, 0, 0] })
    this.resolveUniforms.upload()

    const pass = encoder.beginRenderPass({
      label: "wsplat:resolve",
      colorAttachments: [
        {
          view: this.targets.preview.createView(),
          clearValue: [0, 0, 0, 0],
          loadOp: "clear",
          storeOp: "store",
        },
        {
          view: this.targets.rgba.createView(),
          clearValue: [0, 0, 0, 0],
          loadOp: "clear",
          storeOp: "store",
        },
        {
          view: this.targets.depth.createView(),
          clearValue: [0, 0, 0, 0],
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    })
    pass.setPipeline(this.resolvePipeline)
    pass.setBindGroup(0, this.resolveBindGroup)
    pass.draw(3)
    pass.end()
  }

  async readback(options?: WSplatResolveOptions): Promise<WSplatFrame> {
    const encoder = this.device.createCommandEncoder({
      label: "wsplat:resolve",
    })
    this.encodeResolve(encoder, options)
    this.device.queue.submit([encoder.finish()])

    const { width, height } = this
    const preview = await readTexturePixels(this.device, this.targets.preview, {
      width,
      height,
      bytesPerPixel: 4,
    })
    const rgba = await readTexturePixels(this.device, this.targets.rgba, {
      width,
      height,
      bytesPerPixel: 8,
    })
    const depth = await readTexturePixels(this.device, this.targets.depth, {
      width,
      height,
      bytesPerPixel: 16,
    })

    return buildFrame({ width, height, preview, rgba, depth })
  }

  countCulls(): void {
    if (!this.gpuData) throw new Error("setGaussians() 之前不能 countCulls()")
    if (!this.camera) throw new Error("setCamera() 之前不能 countCulls()")
    // 区间的唯一消费者是本趟统计，所以在这里写，不依赖调用顺序
    this.writeCullRange(this.instanceOffset, this.instanceCount)
    // 每次从 0 开始
    this.device.queue.writeBuffer(
      this.splatStatsBuffer,
      0,
      new Uint32Array(SPLAT_STATS_SLOTS),
    )
    const encoder = this.device.createCommandEncoder({
      label: "wsplat:cullStats",
    })
    const pass = encoder.beginComputePass({ label: "wsplat:cullStats" })
    pass.setPipeline(this.cullStatsPipeline)
    pass.setBindGroup(0, this.cullStatsBindGroup)
    pass.dispatchWorkgroups(Math.ceil(this.instanceCount / 64))
    pass.end()
    this.device.queue.submit([encoder.finish()])
  }

  async readSplatStats(): Promise<WSplatStats> {
    const raw = await readBufferU32(
      this.device,
      this.splatStatsBuffer,
      SPLAT_STATS_SLOTS,
      { label: "wsplat:splatStats" },
    )
    return {
      total: this.instanceCount,
      drawn: raw[0],
      culledBounds: raw[1],
      culledAlphaClip: raw[2],
      culledAlphaClipAfterAa: raw[3],
      culledBehindCamera: raw[4],
      culledMinPixelSize: raw[5],
      culledFrustum: raw[6],
    }
  }

  encodePresent(
    encoder: GPUCommandEncoder,
    target: GPUTextureView,
    format: GPUTextureFormat,
  ): void {
    const pipeline = this.presentPipelines.get(format)
    const bindGroup = this.presentBindGroups.get(format)
    if (!pipeline || !bindGroup) {
      throw new Error(
        `present 管线未建立: ${format}（请用 options.presentFormat 预先声明）`,
      )
    }
    const pass = encoder.beginRenderPass({
      label: "wsplat:present",
      colorAttachments: [{ view: target, loadOp: "clear", storeOp: "store" }],
    })
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bindGroup)
    pass.draw(3)
    pass.end()
  }

  destroy(): void {
    this.destroyTargets()
    this.splatStatsBuffer?.destroy()
  }

  // ────────────────────────────── 内部 ──────────────────────────────

  private createPipelines(): void {
    const device = this.device

    // ── blit（纯色通路）──
    {
      const code = assembleWgsl(blitWgsl, CHUNKS)
      const module = createShaderModule(device, code, "blit")
      const defs = parseDefs(code)
      this.blitUniforms = createUniformSlot(device, defs, "uniforms")
      const layouts = createBindGroupLayouts(
        device,
        defs,
        {
          vertex: { entryPoint: "vsBlit" },
          fragment: { entryPoint: "fsBlit" },
        },
        "blit",
      )
      this.blitBindGroup = device.createBindGroup({
        label: "blit",
        layout: layouts[0],
        entries: [
          {
            binding: this.blitUniforms.binding,
            resource: { buffer: this.blitUniforms.buffer },
          },
        ],
      })
      this.blitPipeline = createRenderPipeline(device, {
        label: "blit",
        layout: device.createPipelineLayout({ bindGroupLayouts: layouts }),
        vertex: { module, entryPoint: "vsBlit" },
        fragment: {
          module,
          entryPoint: "fsBlit",
          targets: [
            premultipliedTarget(SPLAT_COLOR_FORMAT),
            premultipliedTarget(SPLAT_DEPTH_FORMAT),
          ],
        },
        primitive: { topology: "triangle-list" },
      })
    }

    // ── resolve ──
    {
      const code = assembleWgsl(resolveWgsl, CHUNKS)
      const module = createShaderModule(device, code, "resolve")
      const defs = parseDefs(code)
      this.resolveUniforms = createUniformSlot(device, defs, "uniforms")
      const layouts = createBindGroupLayouts(
        device,
        defs,
        {
          vertex: { entryPoint: "vsResolve" },
          fragment: { entryPoint: "fsResolve" },
        },
        "resolve",
      )
      this.resolveLayout = layouts[0]
      this.resolvePipeline = createRenderPipeline(device, {
        label: "resolve",
        layout: device.createPipelineLayout({ bindGroupLayouts: layouts }),
        vertex: { module, entryPoint: "vsResolve" },
        fragment: {
          module,
          entryPoint: "fsResolve",
          targets: [
            { format: RESOLVE_PREVIEW_FORMAT },
            { format: RESOLVE_RGBA_FORMAT },
            { format: RESOLVE_DEPTH_FORMAT },
          ],
        },
        primitive: { topology: "triangle-list" },
      })
    }

    // ── splat（instanced quad + vertex pulling）──
    {
      const code = assembleWgsl(quadWgsl, CHUNKS, {
        ...SPLAT_DEFINES,
        SH_BANDS: this.shBands,
        GSPLAT_AA: this.antialias ? 1 : 0,
        GSPLAT_CAMERA_Y_DOWN: this.cameraYAxis === "down" ? 1 : 0,
        GSPLAT_EPS2D: wgslEps2d(this.eps2d),
      })
      const module = createShaderModule(device, code, "splat")
      const defs = parseDefs(code)
      this.splatUniforms = createUniformSlot(device, defs, "uniforms")
      const layouts = createBindGroupLayouts(
        device,
        defs,
        {
          vertex: { entryPoint: "vsSplat" },
          fragment: { entryPoint: "fsSplat" },
        },
        "splat",
      )
      this.splatLayout = layouts[0]
      this.splatPipeline = createRenderPipeline(device, {
        label: "splat",
        layout: device.createPipelineLayout({ bindGroupLayouts: layouts }),
        vertex: { module, entryPoint: "vsSplat" },
        fragment: {
          module,
          entryPoint: "fsSplat",
          targets: [
            premultipliedTarget(SPLAT_COLOR_FORMAT),
            premultipliedTarget(SPLAT_DEPTH_FORMAT),
          ],
        },
        primitive: { topology: "triangle-list" },
      })
    }

    // ── 剔除统计（compute，见 wgsl/cullStats.ts）──
    // 与 splat pass 共用 chunk，但 SPLAT_COUNT_CULLS=1 才会真的原子加。
    {
      const code = assembleWgsl(cullStatsWgsl, CHUNKS, {
        ...SPLAT_DEFINES,
        SPLAT_COUNT_CULLS: 1,
        SH_BANDS: this.shBands,
        GSPLAT_AA: this.antialias ? 1 : 0,
        GSPLAT_CAMERA_Y_DOWN: this.cameraYAxis === "down" ? 1 : 0,
        GSPLAT_EPS2D: wgslEps2d(this.eps2d),
      })
      const module = createShaderModule(device, code, "cullStats")
      const defs = parseDefs(code)
      this.cullRangeUniforms = createUniformSlot(device, defs, "layerRange")
      const layouts = createBindGroupLayouts(
        device,
        defs,
        { compute: { entryPoint: "csCountCulls" } },
        "cullStats",
      )
      this.cullStatsLayout = layouts[0]
      this.cullStatsPipeline = device.createComputePipeline({
        label: "cullStats",
        layout: device.createPipelineLayout({
          bindGroupLayouts: [this.cullStatsLayout],
        }),
        compute: { module, entryPoint: "csCountCulls" },
      })
    }
  }

  private createPresentPipeline(format: GPUTextureFormat): void {
    const device = this.device
    const code = assembleWgsl(presentWgsl, CHUNKS)
    const module = createShaderModule(device, code, `present:${format}`)
    const defs = parseDefs(code)
    const layouts = createBindGroupLayouts(
      device,
      defs,
      {
        vertex: { entryPoint: "vsPresent" },
        fragment: { entryPoint: "fsPresent" },
      },
      "present",
    )
    this.presentPipelines.set(
      format,
      createRenderPipeline(device, {
        label: `present:${format}`,
        layout: device.createPipelineLayout({ bindGroupLayouts: layouts }),
        vertex: { module, entryPoint: "vsPresent" },
        fragment: { module, entryPoint: "fsPresent", targets: [{ format }] },
        primitive: { topology: "triangle-list" },
      }),
    )
    this.presentLayouts.set(format, layouts[0])
  }

  private createTargets(): void {
    const device = this.device
    const { width, height } = this
    this.targets = {
      splatColor: createRenderTexture(device, {
        label: "wsplat:splatColor",
        width,
        height,
        format: SPLAT_COLOR_FORMAT,
      }),
      splatDepth: createRenderTexture(device, {
        label: "wsplat:splatDepth",
        width,
        height,
        format: SPLAT_DEPTH_FORMAT,
      }),
      preview: createRenderTexture(device, {
        label: "wsplat:preview",
        width,
        height,
        format: RESOLVE_PREVIEW_FORMAT,
      }),
      rgba: createRenderTexture(device, {
        label: "wsplat:rgba",
        width,
        height,
        format: RESOLVE_RGBA_FORMAT,
      }),
      depth: createRenderTexture(device, {
        label: "wsplat:depth",
        width,
        height,
        format: RESOLVE_DEPTH_FORMAT,
      }),
    }

    this.resolveBindGroup = device.createBindGroup({
      label: "resolve",
      layout: this.resolveLayout,
      entries: [
        {
          binding: this.resolveUniforms.binding,
          resource: { buffer: this.resolveUniforms.buffer },
        },
        { binding: 1, resource: this.targets.splatColor.createView() },
        { binding: 2, resource: this.targets.splatDepth.createView() },
      ],
    })

    // present 的 bind group 依赖预览纹理，所以只能在纹理建好之后建；
    // resize 之后预览纹理换了，这里会整体重建一遍。
    this.presentBindGroups.clear()
    for (const [format, layout] of this.presentLayouts) {
      this.presentBindGroups.set(
        format,
        device.createBindGroup({
          label: `present:${format}`,
          layout,
          entries: [
            { binding: 0, resource: this.targets.preview.createView() },
          ],
        }),
      )
    }
  }

  private destroyTargets(): void {
    if (!this.targets) return
    for (const texture of Object.values(this.targets)) {
      texture.destroy()
    }
  }
}

/** eps2d 注入 WGSL 的字面量（WGSL 的 Σ2 是 gsplat 的 4 倍）。 */
function wgslEps2d(eps2d: number): string {
  const v = eps2d * 4
  return Number.isInteger(v) ? `${v}.0` : String(v)
}

/** 预乘混合：color 与 alpha 两个通道都用 `src=one, dst=one-minus-src-alpha`。 */
function premultipliedTarget(format: GPUTextureFormat): GPUColorTargetState {
  const blend: GPUBlendComponent = {
    srcFactor: "one",
    dstFactor: "one-minus-src-alpha",
    operation: "add",
  }
  return { format, blend: { color: blend, alpha: blend } }
}

/** 把三个 resolve 附件的原始字节拼成 `WSplatFrame`。 */
function buildFrame(raw: {
  width: number
  height: number
  preview: Uint8Array
  rgba: Uint8Array
  depth: Uint8Array
}): WSplatFrame {
  const { width, height, preview } = raw
  const count = width * height
  // rgba 附件是 f16：手动解码（不依赖 Float16Array —— Safari 26 之前没有）。
  const rgbaView = new DataView(
    raw.rgba.buffer,
    raw.rgba.byteOffset,
    raw.rgba.byteLength,
  )
  const depthF32 = new Float32Array(
    raw.depth.buffer,
    raw.depth.byteOffset,
    count * 4,
  )

  const rgb = new Float32Array(count * 3)
  const alpha = new Float32Array(count)
  const outDepth = new Float32Array(count)
  const transmission = new Float32Array(count)
  const accumulatedDepth = new Float32Array(count)
  const visible = new Uint8Array(count)

  for (let i = 0; i < count; i++) {
    rgb[i * 3] = halfToFloat(rgbaView.getUint16(i * 8, true))
    rgb[i * 3 + 1] = halfToFloat(rgbaView.getUint16(i * 8 + 2, true))
    rgb[i * 3 + 2] = halfToFloat(rgbaView.getUint16(i * 8 + 4, true))
    alpha[i] = halfToFloat(rgbaView.getUint16(i * 8 + 6, true))
    outDepth[i] = depthF32[i * 4]
    transmission[i] = depthF32[i * 4 + 1]
    accumulatedDepth[i] = depthF32[i * 4 + 2]
    visible[i] = depthF32[i * 4 + 3] > 0.5 ? 1 : 0
  }

  return {
    width,
    height,
    preview,
    rgb,
    alpha,
    depth: outDepth,
    transmission,
    accumulatedDepth,
    visible,
  }
}

/** IEEE 754 binary16 -> f32。 */
function halfToFloat(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1
  const exponent = (bits >> 10) & 0x1f
  const mantissa = bits & 0x03ff
  if (exponent === 0) return sign * 2 ** -14 * (mantissa / 1024)
  if (exponent === 0x1f) {
    return mantissa === 0 ? sign * Number.POSITIVE_INFINITY : Number.NaN
  }
  return sign * 2 ** (exponent - 15) * (1 + mantissa / 1024)
}
