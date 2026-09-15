/**
 * 渲染器运行时 + 渲染设置（全局信号）。
 *
 * 这一层是「store 与渲染器之间的唯一接线处」：
 * - 渲染器本身不知道信号的存在，每帧通过 {@link buildParams} 拿一个纯参数对象；
 * - 渲染器不回写信号，只返回 `FrameStats`，由 rAF 循环限频推到 {@link viewStats}。
 * 于是热路径（每帧）不触碰任何响应式机制，UI 与渲染彻底解耦。
 */

// 副作用导入：把 useValue() 装到 Signal.prototype（供 React 组件消费）
import "./signals-hook.ts"
import { signal } from "@preact/signals-react"
import type {
  PipelineMode,
  RenderParams,
  SplatRenderer,
} from "../spatial-scene/render/renderer.browser.ts"
import { createSplatRenderer } from "../spatial-scene/render/renderer.browser.ts"
import type { PackedSplats, SortCamera } from "../spatial-scene/render/types.ts"
import {
  appliedPose,
  desiredPose,
  poseDeviationDeg,
  stepCamera,
} from "./camera.ts"
import { resort, sortInfo, view } from "./scene.ts"

// ── 渲染设置 ──────────────────────────────────────────────

/** 全局尺度倍数（>1 更「绒」但更糊，<1 更锐但会有洞）。 */
export const splatScale = signal(1)
/** 全局不透明度倍数。 */
export const opacityScale = signal(1)
/** 曝光（线性空间倍数）。 */
export const exposure = signal(1)
/** 背景色（sRGB 十六进制）。 */
export const background = signal("#0a0b10")
/** 屏幕空间低通方差（像素²）；0.3 是 playcanvas 同款抗锯齿。 */
/**
 * 亚像素抗锯齿：主轴方差下限半径（像素），0 = 关闭（默认）。
 *
 * 默认关闭是刻意的：本项目高斯密集、屏幕尺寸普遍很小，任何「给所有高斯
 * 加一块模糊」的写法（如 playcanvas 的 +0.3 px² 低通）会直接把细节抹平。
 * 开启时取 0.5 px，且只钳制比它更细的那一个轴。
 */
export const aaMinPx = signal(0)
/**
 * 渲染管线：`linear16f` = 线性 16F RT + 后处理（默认，与 SHARP 语义一致）；
 * `direct` = 直接混进画布（sRGB 空间、单 pass）。
 */
export const pipeline = signal<PipelineMode>("linear16f")
/** 缺少 `EXT_color_buffer_float`：线性管线在本机不可用（面板上要禁用选项）。 */
export const floatRTUnavailable = signal(false)
/** 单颗高斯的半轴下限（像素）：太小的高斯直接丢弃。 */
export const minPx = signal(0.3)
/** 单颗高斯的半轴上限（像素）：防止个别高斯铺满整屏。 */
export const maxPx = signal(512)
/** 分辨率倍数（再乘 `min(devicePixelRatio, 2)`）。 */
export const resolutionScale = signal(1)
/** 参考图比对模式。 */
export const overlayMode = signal<"off" | "overlay" | "blink">("off")
/** 叠加透明度（overlay 模式）。 */
export const overlayOpacity = signal(0.5)

// ── 运行时状态 ────────────────────────────────────────────

/** 一组分布统计（ms）。全部是**实测值**（最近邻次序统计量），不做插值。 */
export interface DistStats {
  /** 样本数。p95/p99 需要足够样本才有意义，面板上要把它显示出来。 */
  count: number
  avg: number
  min: number
  p50: number
  p95: number
  p99: number
  max: number
}

/** 空分布（还没采样、或该项不可用）。 */
const EMPTY_DIST: DistStats = {
  count: 0,
  avg: 0,
  min: 0,
  p50: 0,
  p95: 0,
  p99: 0,
  max: 0,
}

/**
 * 统计快照（限频 2 Hz 更新，避免 60 Hz 触发 React 重渲染）。
 *
 * 性能部分全部改成**滑动窗口 + 分位数**：单帧的 EMA（旧的 gpuMs）看着平稳，
 * 但它把卡顿平均掉了，既看不出波动也不知道最差能差到哪。窗口默认 5 s，
 * 平均看趋势、p95/p99/最大看尾部——后者才是「稳不稳」的真正答案。
 */
export interface ViewStats {
  /** 统计窗口长度（ms）。 */
  windowMs: number
  /** 平均帧率（= 1000 / 平均帧间隔，不是瞬时值的平均）。 */
  fps: number
  /** 帧间隔分布（ms）：真正决定观感的量（含提交+等待+合成）。 */
  frame: DistStats
  /** CPU 命令下发耗时分布（ms），只反映 JS 侧开销，不含 GPU 执行。 */
  cpu: DistStats
  /** GPU 实测耗时分布（ms）；计时不可用时 `count = 0`。 */
  gpu: DistStats
  /** 本环境是否拿得到硬件计时（动态判定：拿到过有效值）。 */
  gpuTimingAvailable: boolean
  /** 窗口内被排除的长停顿次数（> {@link MAX_SANE_FRAME_MS}，多半是切标签页）。 */
  stalls: number
  /** 实际渲染分辨率。 */
  width: number
  height: number
  /** 渲染的高斯数。 */
  splats: number
  drawCalls: number
  /** 当前视角相对**排序基线**的偏角（度）。 */
  sortDeviationDeg: number
  /** 请求姿态（度）—— 输入源直接写的那一层。 */
  requestedYawDeg: number
  requestedPitchDeg: number
  /** 实际姿态（度）—— 缓动后真正作用到相机的那一层。 */
  appliedYawDeg: number
  appliedPitchDeg: number
  /** 实际姿态相对参考视角的偏角（度）。 */
  appliedDeviationDeg: number
  /** 相机位置（PLY 系），用于显示是否离开了参考位置。 */
  eyePly: [number, number, number]
  /** 相机到参考相机位置（原点）的距离。 */
  eyeDistance: number
}

/** 统计窗口选项（面板可切）。 */
export const STATS_WINDOWS = [
  { value: 2000, label: "2 s" },
  { value: 5000, label: "5 s" },
  { value: 10000, label: "10 s" },
]

/** 统计窗口长度（ms）。越长越稳，但反映变化越慢。 */
export const statsWindowMs = signal(5000)

/** 面板刷新间隔（ms）：2 Hz。窗口是滑动的，所以刷新不影响统计口径。 */
const STATS_PUSH_MS = 500

/**
 * 超过这个时长的「帧间隔」不计入统计，只计一次长停顿。
 *
 * 这类值（>1 s）几乎都是 rAF 被暂停（切标签页、断点、系统休眠），
 * 而不是渲染慢——把它算进 avg，整个窗口的平均帧率就废了。
 */
const MAX_SANE_FRAME_MS = 1000

/** 样本上限（10 s 窗口 + 高刷屏也不至于无限增长）。 */
const MAX_SAMPLES = 8192

/**
 * 一个采样点。
 *
 * `frameMs = 0` 是一个哨兵：表示这次是长停顿（见 {@link MAX_SANE_FRAME_MS}），
 * 只计入 `stalls`，不进分布；`gpuMs = null` 表示本帧没取到新的计时结果。
 */
interface Sample {
  t: number
  frameMs: number
  cpuMs: number
  gpuMs: number | null
}

let samples: Sample[] = []
/** 最近一次求分布的时间：分布本身不用每帧算，但样本必须每帧收。 */
let lastDistMs = -Infinity
const EMPTY_DISTS = { frame: EMPTY_DIST, cpu: EMPTY_DIST, gpu: EMPTY_DIST }
/** 分布缓存（含窗口内长停顿计数）。 */
let distCache = { ...EMPTY_DISTS, stalls: 0 }

/**
 * 分位数（最近邻次序统计量）。
 *
 * 不插值是有意的：结果一定是**真实测到过**的某一帧，不会出现
 * 「p95 = 17.3 ms」而实际上从来没有哪一帧是 17.3 ms 的情况。
 * n 很小时（<20）p99 会退化成最大值，面板会把样本数一并显示出来。
 */
function dist(values: number[]): DistStats {
  const n = values.length
  if (n === 0) return EMPTY_DIST
  values.sort((a, b) => a - b)
  const at = (q: number) =>
    values[Math.min(n - 1, Math.max(0, Math.ceil(q * n) - 1))]
  let sum = 0
  for (const v of values) sum += v
  return {
    count: n,
    avg: sum / n,
    min: values[0],
    p50: at(0.5),
    p95: at(0.95),
    p99: at(0.99),
    max: values[n - 1],
  }
}

/** 重算窗口内的三个分布（每 {@link STATS_PUSH_MS} 一次）。 */
function recomputeDistributions(now: number): void {
  const cutoff = now - statsWindowMs.value
  if (samples.length > 0 && samples[0].t < cutoff) {
    samples = samples.filter((s) => s.t >= cutoff)
  }
  const frame: number[] = []
  const cpu: number[] = []
  const gpu: number[] = []
  let stalls = 0
  for (const s of samples) {
    if (s.frameMs === 0) {
      stalls++
      continue
    }
    frame.push(s.frameMs)
    cpu.push(s.cpuMs)
    if (s.gpuMs !== null) gpu.push(s.gpuMs)
  }
  distCache = {
    frame: dist(frame),
    cpu: dist(cpu),
    gpu: dist(gpu),
    stalls,
  }
  lastDistMs = now
}

/** 清空统计（切换窗口长度 / 重新挂载画布时调用）。 */
export function resetStats(): void {
  samples = []
  lastDistMs = -Infinity
  distCache = { ...EMPTY_DISTS, stalls: 0 }
}

export const viewStats = signal<ViewStats>({
  windowMs: 5000,
  fps: 0,
  frame: EMPTY_DIST,
  cpu: EMPTY_DIST,
  gpu: EMPTY_DIST,
  gpuTimingAvailable: false,
  stalls: 0,
  width: 1,
  height: 1,
  splats: 0,
  drawCalls: 0,
  sortDeviationDeg: 0,
  requestedYawDeg: 0,
  requestedPitchDeg: 0,
  appliedYawDeg: 0,
  appliedPitchDeg: 0,
  appliedDeviationDeg: 0,
  eyePly: [0, 0, 0],
  eyeDistance: 0,
})

/** 参考照片是否真的加载成功（sidecar 给了 URL 不代表文件存在）。
 *  未成功时强制关闭叠加，避免用 1x1 占位纹理铺满屏幕。 */
export const referenceImageReady = signal(false)

/** GPU 名称（判断是否踩到软件渲染）。 */
export const gpuName = signal("")
/** WebGL 初始化失败信息。 */
export const rendererError = signal<string | null>(null)

let renderer: SplatRenderer | null = null
let canvasEl: HTMLCanvasElement | null = null
let rafId = 0
/** 上一帧时间戳（用于与帧率无关的缓动）。 */
let lastStepMs = 0
/** 上一帧的起始时间戳（统计帧间隔用；与 `lastStepMs` 分开，各自用途不同）。 */
let lastFrameMs = 0
/** 最近一次的顶点缓冲：即使渲染器还没挂载（或热更新重建过）也不丢数据。 */
let latest: PackedSplats | null = null
/** 点击「暂停」后停止 rAF（省电；也便于观察单帧耗时）。 */
export const paused = signal(false)

/** 面板是否展开（纯 UI 状态，但需要跨组件共享：收起后右上角要出「显示面板」按钮）。 */
export const panelOpen = signal(true)

/** 每帧闪烁相位（1.2 Hz 方波），blink 比对用。 */
const BLINK_PERIOD_MS = 830

/** 挂载画布并启动渲染循环（由 React 组件在 mount 时调用）。 */
export function attachCanvas(canvas: HTMLCanvasElement): void {
  detachCanvas()
  canvasEl = canvas
  try {
    renderer = createSplatRenderer(canvas)
    gpuName.value = renderer.rendererInfo
    floatRTUnavailable.value = renderer.floatRTUnavailable
    // 缺扩展时把管线钉在 direct，否则用户会停在一个永远画不出东西的选项上
    if (renderer.floatRTUnavailable) pipeline.value = "direct"
    rendererError.value = null
  } catch (err) {
    renderer = null
    rendererError.value = err instanceof Error ? err.message : String(err)
    return
  }
  if (latest) renderer.setSplats(latest)
  // 新渲染器：帧间隔序列从头开始，避免把重建间隙当成一次卡顿
  lastFrameMs = 0
  resetStats()
  loop()
}

/** 卸载画布（React unmount / 热更新）。 */
export function detachCanvas(): void {
  if (rafId) cancelAnimationFrame(rafId)
  rafId = 0
  renderer?.dispose()
  renderer = null
  canvasEl = null
}

/** 把新顶点缓冲推给渲染器（未挂载时暂存，挂载时补上）。 */
export function pushSplats(packed: PackedSplats): void {
  latest = packed
  renderer?.setSplats(packed)
}

/** 设置参考图（叠加比对用）。仅在加载成功时才允许开启叠加。 */
export function setReferenceImage(image: HTMLImageElement): void {
  renderer?.setReferenceImage(image)
  referenceImageReady.value = true
}

/** 参考图加载失败。 */
export function clearReferenceImage(): void {
  referenceImageReady.value = false
}

/** 当前相机在 PLY 系下的位姿（交给 worker 排序用）。 */
export function currentSortCamera(): SortCamera | null {
  if (!renderer) return null
  return {
    eye: [...renderer.camera.eyePly] as [number, number, number],
    forward: [...renderer.camera.forwardPly] as [number, number, number],
  }
}

/** 按**当前**视角重排序（演示「一次排序」的代价与收益）。 */
export async function reSortNow(): Promise<void> {
  const cam = currentSortCamera()
  if (!cam) return
  await resort(cam, pushSplats)
}

/** 暂停/恢复渲染循环。 */
export function togglePaused(): void {
  paused.value = !paused.value
  if (!paused.value) loop()
}

/** 组装一帧的渲染参数（null = 场景未就绪）。 */
export function buildParams(): RenderParams | null {
  const v = view.value
  if (!v) return null
  const dpr = Math.min(window.devicePixelRatio || 1, 2)
  // 每帧推进一步缓动（整个应用里只有这里会改实际姿态）
  const now = performance.now()
  const dt = lastStepMs === 0 ? 16.7 : now - lastStepMs
  lastStepMs = now
  const pose = stepCamera(dt, now)
  return {
    camera: {
      pivot: v.pivot,
      yaw: pose.yaw,
      pitch: pose.pitch,
      distance: v.referenceDistance * pose.zoom,
      focalPx: v.focalPx,
      imageSize: v.imageSize,
      near: NEAR,
      far: FAR,
    },
    splatScale: splatScale.value,
    opacityScale: opacityScale.value,
    exposure: exposure.value,
    background: background.value,
    aaMinPx: aaMinPx.value,
    minPx: minPx.value,
    maxPx: maxPx.value,
    pixelRatio: dpr * resolutionScale.value,
    pipeline: pipeline.value,
    overlayMode: referenceImageReady.value ? overlayMode.value : "off",
    overlayOpacity: overlayOpacity.value,
    overlayPhase: (performance.now() % BLINK_PERIOD_MS) / BLINK_PERIOD_MS,
  }
}

/** 近/远裁剪面（场景深度约 1~6 m，取足够宽的余量）。 */
const NEAR = 0.05
const FAR = 1000

/** rAF 循环：渲染 + 限频推统计。 */
function loop(): void {
  if (!renderer || paused.value) {
    rafId = 0
    return
  }
  const params = buildParams()
  if (params) {
    const t0 = performance.now()
    const s = renderer.render(params)
    const now = performance.now()

    // ── 采样（每帧都要收，否则帧间隔序列就是错的）──
    // 第一帧没有前一帧时间戳，不采样
    const frameMs = lastFrameMs > 0 ? t0 - lastFrameMs : 0
    lastFrameMs = t0
    if (frameMs > 0) {
      samples.push({
        t: now,
        // 长停顿用哨兵 0 标记（不计入分布，只计数）
        frameMs: frameMs <= MAX_SANE_FRAME_MS ? frameMs : 0,
        cpuMs: s.cpuMs,
        gpuMs: s.gpuMs,
      })
      if (samples.length > MAX_SAMPLES)
        samples.splice(0, samples.length - MAX_SAMPLES)
    }

    // ── 限频刷新面板快照（分布每次重算：窗口滑动，旧样本随时间失效）──
    if (now - lastDistMs >= STATS_PUSH_MS) {
      recomputeDistributions(now)
      const cam = renderer.camera
      const applied = appliedPose()
      const desired = desiredPose(now)
      const toDeg = 180 / Math.PI
      const frame = distCache.frame
      viewStats.value = {
        windowMs: statsWindowMs.value,
        // 平均帧率由**平均帧间隔**得出，不是瞬时 fps 的平均（两者不等价）
        fps: frame.count > 0 ? 1000 / frame.avg : 0,
        frame,
        cpu: distCache.cpu,
        gpu: distCache.gpu,
        gpuTimingAvailable: s.gpuTimingAvailable,
        stalls: distCache.stalls,
        width: s.width,
        height: s.height,
        splats: s.splats,
        drawCalls: s.drawCalls,
        sortDeviationDeg: angleBetweenDeg(
          cam.forwardPly,
          sortInfo.value?.camera.forward ?? [0, 0, 1],
        ),
        requestedYawDeg: desired.yaw * toDeg,
        requestedPitchDeg: desired.pitch * toDeg,
        appliedYawDeg: applied.yaw * toDeg,
        appliedPitchDeg: applied.pitch * toDeg,
        appliedDeviationDeg: poseDeviationDeg(applied),
        eyePly: [...cam.eyePly] as [number, number, number],
        eyeDistance: cam.distanceFromReference,
      }
    }
  }
  rafId = requestAnimationFrame(loop)
}

/** 两个方向向量之间的夹角（度）。 */
function angleBetweenDeg(
  a: readonly [number, number, number],
  b: readonly [number, number, number],
): number {
  const d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
  return (Math.acos(Math.min(1, Math.max(-1, d))) * 180) / Math.PI
}

// 供调试：把渲染器暴露到 window，便于在控制台里试参数
if (typeof window !== "undefined") {
  ;(window as unknown as Record<string, unknown>).__splat = {
    get renderer() {
      return renderer
    },
    get canvas() {
      return canvasEl
    },
  }
}
