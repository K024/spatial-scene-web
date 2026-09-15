/**
 * WebGL2 高斯泼溅渲染器。
 *
 * ── 一次 draw call ──
 * 顶点缓冲在 worker 里已按「远 -> 近」排好，因此渲染时**不需要深度缓冲、
 * 不需要每帧排序**：`drawArraysInstanced(TRIANGLE_STRIP, 0, 4, N)` 一次画完，
 * 混合顺序由同一 draw call 内图元的提交顺序保证（GL 规范要求按序光栅化）。
 * 相机只做重投影、顺序不变——这正是本项目要验证的取舍。
 *
 * ── 管线 ──
 *   splat pass   -> RGBA16F（线性空间、预乘 alpha 混合）
 *   post pass    -> 曝光 + 线性->sRGB，贴到画布
 *   overlay pass -> 参考照片按参考内参投影成「无限远背景板」，叠加/闪烁比对
 *
 * `direct` 管线（面板可切）则只有 splat + overlay 两个 pass：splat 直接混进
 * 画布，sRGB 编码在**顶点阶段**做（见 SPLAT_VS 的 u_encodeSrgb）。
 *
 * ── 为什么中间要一张浮点 RT ──
 * PLY 里是 sRGB 域的 SH 系数，公开渲染器把 sRGB 当线性直接混合（重叠处偏暗）。
 * 这里先转回线性、在线性空间做预乘混合，最后统一编码回 sRGB——与 SHARP
 * 自己的渲染器语义一致。硬件混合发生在写入 RT 时，所以 RT 必须真的是线性的
 * （RGBA16F 每个通道 16 bit 浮点，暗部精度足够）。
 */

import * as twgl from "twgl.js"

import { type CameraState, RenderCamera } from "./camera.ts"
import { hexToRgb } from "./math.ts"
import {
  OVERLAY_FS,
  OVERLAY_VS,
  POST_FS,
  POST_VS,
  SPLAT_FS,
  SPLAT_VS,
} from "./shaders.ts"
import type { PackedSplats } from "./types.ts"
import { SPLAT_LAYOUT, SPLAT_STRIDE } from "./types.ts"

/** 参考图比对模式。 */
export type OverlayMode = "off" | "overlay" | "blink"

/**
 * 渲染管线。
 *
 * - `linear16f`（默认）：splat -> RGBA16F 线性空间预乘混合 -> 后处理编码回
 *   sRGB。与 SHARP 自己的渲染器语义一致（重叠处更接近物理正确）。
 * - `direct`：splat 直接混进画布（RGBA8、sRGB 空间），没有后处理 pass。
 *   省掉一次全屏读写，且混合读写从 16 B/片元降到 8 B/片元——实测本项目
 *   片元数是屏像素的 20 倍（temp/overdraw-bench.ts），所以这一层往往是
 *   真正的瓶颈。代价：sRGB 空间混合（重叠处偏亮/偏平），且 8 bit 累积在
 *   暗部有精度损失。
 */
export type PipelineMode = "linear16f" | "direct"

/** 每帧渲染参数（由 store 提供，渲染器只读）。 */
export interface RenderParams {
  camera: CameraState
  /** 全局尺度倍数。 */
  splatScale: number
  /** 全局不透明度倍数。 */
  opacityScale: number
  /** 曝光（线性空间倍数）。 */
  exposure: number
  /** 背景色（`#rrggbb`）。 */
  background: string
  /**
   * 亚像素抗锯齿：主轴方差的下限半径（像素，0 = 关闭）。
   * 只作用于比它更细的那个轴，不会把大高斯弄糊。
   */
  aaMinPx: number
  /** 单颗高斯半轴的下限/上限（像素）。 */
  minPx: number
  maxPx: number
  /** 画布像素倍率（已含 devicePixelRatio 与用户的分辨率倍数）。 */
  pixelRatio: number
  /** 渲染管线（见 {@link PipelineMode}）。 */
  pipeline: PipelineMode
  overlayMode: OverlayMode
  /** 叠加透明度（overlay 模式）。 */
  overlayOpacity: number
  /** 闪烁相位（0..1），由 store 用时间驱动。 */
  overlayPhase: number
}

/** 每帧统计（**瞬时值**；分布/分位数由 `store/viewer.ts` 在滑动窗口里统计）。 */
export interface FrameStats {
  /**
   * CPU **提交**耗时（ms）：JS 里测 `render()` 自身花了多久。
   *
   * 注意：GL 命令是异步下发的，这个数**不包含** GPU 真正执行的时间，
   * 不能当帧耗时看（早期版本就是拿它当帧耗时展示，属于误导）。
   */
  cpuMs: number
  /**
   * 本帧**新取到**的 GPU 实测耗时（ms）。
   *
   * `null` = 本帧没有新样本（计时查询还积压在 GPU 队列里，或结果被丢弃）。
   * 必须区分「没有新样本」与「有样本」：早期版本用 EMA 平滑后每帧都返回一个值，
   * 同一个样本会被反复计入分布，分位数就失真了。
   */
  gpuMs: number | null
  /** 本帧认为硬件计时是否可用（动态判定，见 `createGpuTimer`）。 */
  gpuTimingAvailable: boolean
  width: number
  height: number
  splats: number
  drawCalls: number
}

export interface SplatRenderer {
  /** 替换场景数据（重新上传顶点缓冲）。 */
  setSplats(packed: PackedSplats): void
  /** 参考图（叠加/闪烁用）。 */
  setReferenceImage(image: HTMLImageElement): void
  /** 渲染一帧。 */
  render(params: RenderParams): FrameStats
  /** 实时相机（每帧 `render` 后更新；供 UI 取当前位姿去重排序）。 */
  readonly camera: RenderCamera
  /** GPU 名称（判断是否踩到软件渲染）。 */
  readonly rendererInfo: string
  /** 缺少 `EXT_color_buffer_float`：线性 16F 管线不可用，只能走 direct。 */
  readonly floatRTUnavailable: boolean
  dispose(): void
}

/** 片元 alpha 阈值 1/255：低于此值写入也不可见，直接 discard 省带宽。 */
const ALPHA_CLIP = 1 / 255

/** 全屏后处理三角形（注意：不是四边形，避免对角线处的重复着色）。 */
const SCREEN_TRIANGLE = new Float32Array([-1, -1, 3, -1, -1, 3])

/** 参考图叠加层的 4 个角（triangle strip 顺序，uv 左上为 (0,0)）。 */
const OVERLAY_UV = new Float32Array([0, 0, 1, 0, 0, 1, 1, 1])

/** 高斯四边形：triangle strip 的四个角。 */
const CORNER_QUAD = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1])

/**
 * 创建渲染器。
 *
 * @param canvas 目标画布（尺寸由 {@link RenderParams.pixelRatio} 驱动）。
 */
export function createSplatRenderer(canvas: HTMLCanvasElement): SplatRenderer {
  const glCtx = canvas.getContext("webgl2", {
    alpha: false,
    antialias: false,
    depth: false,
    stencil: false,
    premultipliedAlpha: false,
    preserveDrawingBuffer: false,
    powerPreference: "high-performance",
  })
  if (!glCtx) throw new Error("无法创建 WebGL2 上下文")
  // 显式标注（而不是靠上面的守卫推断）：下面有函数**声明**（ensureRT），
  // 声明会被提升，tsgo 不会把 `!gl` 的收窄带进它的函数体。
  const gl: WebGL2RenderingContext = glCtx

  // 线性 16F 管线要求 RGBA16F 可被渲染（WebGL2 里这是扩展）。缺了就只提供
  // direct 管线——不再像早期版本那样直接抛错，因为单 pass 管线本来就不需要它。
  const floatRTUnavailable = !gl.getExtension("EXT_color_buffer_float")
  if (floatRTUnavailable) {
    console.warn(
      "缺少 EXT_color_buffer_float：线性 16F 管线不可用，已退回直接写画布的 sRGB 管线",
    )
  }
  const rendererInfo = describeRenderer(gl)

  // ── 程序 ──
  const splatProgram = twgl.createProgramInfo(gl, [SPLAT_VS, SPLAT_FS])
  const postProgram = twgl.createProgramInfo(gl, [POST_VS, POST_FS])
  const overlayProgram = twgl.createProgramInfo(gl, [OVERLAY_VS, OVERLAY_FS])

  // ── 静态几何 ──
  const cornerInfo = twgl.createBufferInfoFromArrays(gl, {
    a_corner: { numComponents: 2, data: CORNER_QUAD },
  })
  const screenInfo = twgl.createBufferInfoFromArrays(gl, {
    a_pos: { numComponents: 2, data: SCREEN_TRIANGLE },
  })
  const overlayInfo = twgl.createBufferInfoFromArrays(gl, {
    a_uv: { numComponents: 2, data: OVERLAY_UV },
  })

  // ── 高斯顶点缓冲（缓冲对象一次创建、反复 bufferData）──
  const splatBuffer = gl.createBuffer()
  if (!splatBuffer) throw new Error("gl.createBuffer 失败")
  const stride = SPLAT_STRIDE * 4
  /** 与 splatProgram 的属性一一对应；divisor=1 表示「每实例推进一次」。 */
  const splatInfo: twgl.BufferInfo = {
    numElements: 0,
    attribs: {
      a_corner: attrib(cornerInfo, "a_corner"),
      a_center: instanced(splatBuffer, 3, SPLAT_LAYOUT.center, stride),
      a_scaleLog: instanced(splatBuffer, 3, SPLAT_LAYOUT.scaleLog, stride),
      a_quat: instanced(splatBuffer, 4, SPLAT_LAYOUT.quat, stride),
      a_color: instanced(splatBuffer, 3, SPLAT_LAYOUT.colorLinear, stride),
      a_opacityLogit: instanced(
        splatBuffer,
        1,
        SPLAT_LAYOUT.opacityLogit,
        stride,
      ),
    },
  }

  // ── 参考图纹理（先用 1x1 白占位，避免未加载时采样到未定义）──
  const refTexture = twgl.createTexture(gl, {
    src: [255, 255, 255, 255],
    minMag: gl.LINEAR,
    wrap: gl.CLAMP_TO_EDGE,
  })

  // ── 离屏 RT（线性空间，惰性创建）──
  //
  // 这里有个 twgl 的坑：`format` 同时兼作 internalFormat 与像素 format
  // （内部 `internalFormat = opt.internalFormat || opt.format`，但
  // `format = opt.format || …`），所以只写 `format: gl.RGBA16F` 会把
  // RGBA16F 当**像素格式**传给 texImage2D → INVALID_ENUM → 附件零尺寸
  // → FBO 不完整（glClear/glDraw 全部 INVALID_FRAMEBUFFER_OPERATION）。
  // 必须把 internalFormat 与 format 分开写。
  const rtAttachments: twgl.AttachmentOptions[] = [
    {
      internalFormat: gl.RGBA16F,
      format: gl.RGBA,
      type: gl.HALF_FLOAT,
      minMag: gl.LINEAR,
      wrap: gl.CLAMP_TO_EDGE,
    },
  ]
  // 惰性创建：一直用 direct 管线就一张都不分配（也避开缺扩展时 twgl 打警告）
  let rt: twgl.FramebufferInfo | null = null
  let rtWidth = 0
  let rtHeight = 0
  function ensureRT(w: number, h: number): twgl.FramebufferInfo {
    if (!rt) {
      rt = twgl.createFramebufferInfo(gl, rtAttachments, w, h)
      rtWidth = w
      rtHeight = h
    } else if (w !== rtWidth || h !== rtHeight) {
      twgl.resizeFramebufferInfo(gl, rt, rtAttachments, w, h)
      rtWidth = w
      rtHeight = h
    }
    return rt
  }

  const gpuTimer = createGpuTimer(gl)

  const cam = new RenderCamera()
  let splatCount = 0

  return {
    rendererInfo,
    floatRTUnavailable,
    camera: cam,

    setSplats(packed) {
      gl.bindBuffer(gl.ARRAY_BUFFER, splatBuffer)
      gl.bufferData(gl.ARRAY_BUFFER, packed.data, gl.STATIC_DRAW)
      gl.bindBuffer(gl.ARRAY_BUFFER, null)
      splatCount = packed.count
    },

    setReferenceImage(image) {
      twgl.setTextureFromElement(gl, refTexture, image, {
        minMag: gl.LINEAR,
        wrap: gl.CLAMP_TO_EDGE,
        // v=0 必须对应图片第一行：叠加层的射线方向就是按行序推的
        flipY: 0,
      })
    },

    render(params) {
      const t0 = performance.now()
      // 先取上上帧就已完成的 GPU 计时结果（绝不同步等当前帧，否则会把 GPU 拖死）。
      // null = 本帧没有新样本，不要计入分布。
      const gpuMs = gpuTimer?.poll() ?? null

      const width = Math.max(
        1,
        Math.round((canvas.clientWidth || 1) * params.pixelRatio),
      )
      const height = Math.max(
        1,
        Math.round((canvas.clientHeight || 1) * params.pixelRatio),
      )
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width
        canvas.height = height
      }
      const direct = params.pipeline === "direct" || floatRTUnavailable

      cam.update(params.camera, width, height)

      // GPU 计时从这里开始（含清屏与全部 pass）
      gpuTimer?.begin()

      // ── 1) splat pass ──
      const bg = hexToRgb(params.background)
      // 线性管线写 RT（清屏色需要线性化，最后统一编码回 sRGB）；
      // direct 管线写画布（画布已是 sRGB 编码的 RGBA8，直接用原色清屏）。
      if (direct) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, null)
      } else {
        gl.bindFramebuffer(gl.FRAMEBUFFER, ensureRT(width, height).framebuffer)
      }
      gl.viewport(0, 0, width, height)
      gl.clearColor(
        direct ? bg[0] : srgbToLinear(bg[0]),
        direct ? bg[1] : srgbToLinear(bg[1]),
        direct ? bg[2] : srgbToLinear(bg[2]),
        1,
      )
      gl.clear(gl.COLOR_BUFFER_BIT)
      gl.disable(gl.DEPTH_TEST)

      let drawCalls = 0
      if (splatCount > 0) {
        gl.enable(gl.BLEND)
        gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA)
        // biome-ignore lint/correctness/useHookAtTopLevel: gl.useProgram 是 WebGL 调用，不是 React Hook（规则按 use[A-Z] 前缀误判）
        gl.useProgram(splatProgram.program)
        twgl.setBuffersAndAttributes(gl, splatProgram, splatInfo)
        twgl.setUniforms(splatProgram, {
          u_view: cam.view,
          u_proj: cam.proj,
          u_viewport: [width, height],
          u_invViewport: [1 / width, 1 / height],
          u_focalPx: cam.focalScaled,
          u_scaleMul: params.splatScale,
          u_opacityMul: params.opacityScale,
          u_near: params.camera.near,
          u_aaMinVar: params.aaMinPx * params.aaMinPx,
          u_minPx: params.minPx,
          u_maxPx: params.maxPx,
          u_alphaClip: ALPHA_CLIP,
          u_exposure: params.exposure,
          // direct：片元输出已是 sRGB，混合直接发生在画布上
          u_encodeSrgb: direct ? 1 : 0,
        })
        // 4 顶点/实例 × N 实例；顺序 = 顶点缓冲顺序 = 远到近
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, splatCount)
        drawCalls++
      }

      // ── 2) post pass：线性 -> sRGB，贴到画布（direct 管线没有这一步）──
      if (!direct && rt) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, null)
        gl.viewport(0, 0, width, height)
        gl.disable(gl.BLEND)
        // biome-ignore lint/correctness/useHookAtTopLevel: gl.useProgram 是 WebGL 调用，不是 React Hook（规则按 use[A-Z] 前缀误判）
        gl.useProgram(postProgram.program)
        twgl.setBuffersAndAttributes(gl, postProgram, screenInfo)
        twgl.setUniforms(postProgram, {
          u_hdr: rt.attachments[0],
          u_exposure: params.exposure,
          u_encodeSrgb: 1,
        })
        twgl.drawBufferInfo(gl, screenInfo, gl.TRIANGLES)
        drawCalls++
      }

      // ── 3) overlay pass：参考图比对 ──
      if (params.overlayMode !== "off" && splatCount > 0) {
        const blink = params.overlayMode === "blink"
        if (!blink) {
          gl.enable(gl.BLEND)
          gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)
        }
        // biome-ignore lint/correctness/useHookAtTopLevel: gl.useProgram 是 WebGL 调用，不是 React Hook（规则按 use[A-Z] 前缀误判）
        gl.useProgram(overlayProgram.program)
        twgl.setBuffersAndAttributes(gl, overlayProgram, overlayInfo)
        twgl.setUniforms(overlayProgram, {
          u_view: cam.view,
          u_proj: cam.proj,
          u_imageSize: [params.camera.imageSize[0], params.camera.imageSize[1]],
          u_refFocalPx: params.camera.focalPx,
          u_ref: refTexture,
          u_opacity: params.overlayOpacity,
          u_mode: blink ? 2 : 1,
          u_phase: params.overlayPhase,
        })
        twgl.drawBufferInfo(gl, overlayInfo, gl.TRIANGLE_STRIP)
        drawCalls++
      }

      // ── 统计 ──
      // 这里只收**瞬时**值：帧间隔/分布/分位数全部由 store 在滑动窗口里统计。
      // 早期版本在这里做 EMA + 固定窗口 fps，两者口径不同又都在面板上出现，
      // 既看不出波动也看不出尾部，已经搬到 store/viewer.ts。
      gpuTimer?.end()
      const now = performance.now()

      return {
        cpuMs: now - t0,
        gpuMs: gpuMs,
        gpuTimingAvailable: gpuTimer?.available ?? false,
        width,
        height,
        splats: splatCount,
        drawCalls,
      }
    },

    dispose() {
      gpuTimer?.dispose()
      gl.deleteBuffer(splatBuffer)
      gl.deleteBuffer(attrib(cornerInfo, "a_corner").buffer)
      gl.deleteBuffer(attrib(screenInfo, "a_pos").buffer)
      gl.deleteBuffer(attrib(overlayInfo, "a_uv").buffer)
      gl.deleteTexture(refTexture)
      if (rt) {
        gl.deleteFramebuffer(rt.framebuffer)
        gl.deleteTexture(rt.attachments[0] as WebGLTexture)
      }
      gl.deleteProgram(splatProgram.program)
      gl.deleteProgram(postProgram.program)
      gl.deleteProgram(overlayProgram.program)
    },
  }
}

/**
 * GPU 硬件计时（`EXT_disjoint_timer_query_webgl2`）。
 *
 * 为什么必须用它：`performance.now()` 量到的只是**命令下发**用了多久。GL 调用
 * 是异步的，`drawArraysInstanced` 提交完就返回，真正跑多少时间在 GPU 队列里。
 * 计时查询的用法是在命令流里插入一对时间戳，GPU 执行到那里才会计数，
 * 因此量到的是**真实执行时间**（含全部 pass）。
 *
 * 三个必须遵守的细节：
 * 1. 结果**必须滞后读取**：先问 `QUERY_RESULT_AVAILABLE`，没就绪就下一帧再看。
 *    直接读 `QUERY_RESULT` 会同步等待 GPU，把并行度彻底碱掉。
 * 2. 要查 `GPU_DISJOINT_EXT`：GPU 被抢占/降频时结果不可信，当帧应当丢弃。
 * 3. 不要试图用 `QUERY_COUNTER_BITS_EXT` 「探测」可用性：WebGL 对
 *    `getQueryParameter` 的时序要求很苛——查询未 `beginQuery` 时报
 *    `not a query object yet`，处在 `begin~end` 之间又报 `currently active`，
 *    两种写法都拿不到稳定结论。改为**直接量**：拿到有效值就认为可用，
 *    连续若干帧拿不到就判定该平台计时器被禁用（见 `available`）。
 *
 * 返回 `null` 表示连扩展都没有，调用方应在 UI 上诚实标注。
 */
function createGpuTimer(gl: WebGL2RenderingContext): GpuTimer | null {
  const ext = gl.getExtension(
    "EXT_disjoint_timer_query_webgl2",
  ) as unknown as DisjointTimerExt | null
  if (!ext) return null

  /** 待命查询对象（复用，避免每帧 createQuery）。 */
  const free: WebGLQuery[] = []
  /** 已结束、等 GPU 写完的查询。 */
  const pending: WebGLQuery[] = []
  let active: WebGLQuery | null = null
  /** 拿到过至少一个有效结果。 */
  let sawResult = false
  /** 已经过去多少帧仍未拿到有效结果（用于「放弃」判定）。 */
  let blindFrames = 0

  return {
    get available() {
      // 拿到过结果就是真的可用；否则先乐观登记，超过阈值就认定被禁用
      return sawResult || blindFrames < TIMER_GIVE_UP_FRAMES
    },

    begin() {
      if (active) return
      const q = free.pop() ?? gl.createQuery()
      if (!q) return
      active = q
      blindFrames++
      gl.beginQuery(ext.TIME_ELAPSED_EXT, q)
    },

    end() {
      const q = active
      if (!q) return
      gl.endQuery(ext.TIME_ELAPSED_EXT)
      active = null
      pending.push(q)
      // 积压过多说明 GPU 严重落后（或查询永不就绪）：丢掉最老的，不无限增长
      while (pending.length > MAX_PENDING_QUERIES) {
        const old = pending.shift()
        if (old) free.push(old)
      }
    },

    /**
     * 取**一个已完成查询**的耗时（ms）。
     *
     * 只有在真的消费掉一个结果时才返回数值；否则返回 `null`。
     * 这一点很关键：每次 `drawArraysInstanced` 后 GPU 耗时不会马上可读，
     * 若把「上次的结果」每帧都返回一次，同一个样本会被重复统计，
     * 分位数就变成了加权平均（而且偏向恰好被重复最多的那个值）。
     */
    poll() {
      const q = pending[0]
      if (!q) return null
      if (!gl.getQueryParameter(q, gl.QUERY_RESULT_AVAILABLE)) return null

      pending.shift()
      const disjoint = Boolean(gl.getParameter(ext.GPU_DISJOINT_EXT))
      // 先取值再归还：结果无效时当帧丢弃，不污染统计
      const ns = disjoint ? 0 : Number(gl.getQueryParameter(q, gl.QUERY_RESULT))
      free.push(q)
      if (disjoint || !Number.isFinite(ns) || ns <= 0) return null

      sawResult = true
      blindFrames = 0
      return ns / 1e6
    },

    dispose() {
      if (active) gl.endQuery(ext.TIME_ELAPSED_EXT)
      for (const q of [...free, ...pending]) gl.deleteQuery(q)
      if (active) gl.deleteQuery(active)
      free.length = 0
      pending.length = 0
      active = null
    },
  }
}

/** `EXT_disjoint_timer_query_webgl2` 的字段（比 lib.dom 的声明更宽松）。 */
interface DisjointTimerExt {
  readonly TIME_ELAPSED_EXT: number
  readonly GPU_DISJOINT_EXT: number
}

/** GPU 计时的内部状态机。 */
interface GpuTimer {
  /** 是否认为硬件计时可用（动态判定：拿到过有效值 / 还没超过放弃阈值）。 */
  readonly available: boolean
  begin(): void
  end(): void
  /** 取一个已完成查询的 GPU 耗时（ms）；本帧没有新样本时返回 `null`。 */
  poll(): number | null
  dispose(): void
}

/** GPU 耗时最多的等待帧数：超过就丢弃，避免查询对象堆积。 */
const MAX_PENDING_QUERIES = 4

/** 约 2 秒（60 fps）都没拿到过有效结果，就认为该平台计时器被禁用。 */
const TIMER_GIVE_UP_FRAMES = 120

/** 构造一个「按实例推进」的属性描述。 */ function instanced(
  buffer: WebGLBuffer,
  numComponents: number,
  offsetFloats: number,
  strideBytes: number,
): twgl.AttribInfo {
  return {
    buffer,
    numComponents,
    // type 省略即默认 gl.FLOAT（twgl 的行为），本项目顶点全是 f4
    stride: strideBytes,
    offset: offsetFloats * 4,
    divisor: 1,
  }
}

/** 取某个属性（twgl 类型里 `attribs` 是可选的，这里收敛一次）。 */
function attrib(info: twgl.BufferInfo, name: string): twgl.AttribInfo {
  const a = info.attribs?.[name]
  if (!a) throw new Error(`twgl: bufferInfo 里缺少属性 ${name}`)
  return a
}

/** sRGB -> 线性（背景色清屏用；与 `render/convert.ts` 同一套阈值分段）。 */
function srgbToLinear(x: number): number {
  return x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4
}

/** 取 GPU 名称（面板显示，便于判断是不是跑在软件渲染上）。 */
function describeRenderer(gl: WebGL2RenderingContext): string {
  const ext = gl.getExtension("WEBGL_debug_renderer_info")
  if (!ext) return String(gl.getParameter(gl.RENDERER))
  const name = gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)
  return typeof name === "string" ? name : "unknown"
}
