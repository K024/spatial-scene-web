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

/** 每帧统计 + 相机派生量（限频 5 Hz 更新，避免 60 Hz 触发 React 重渲染）。 */
export interface ViewStats {
  fps: number
  /** GPU 实测耗时（ms）；-1 = 本环境拿不到硬件计时。 */
  gpuMs: number
  /** 本帧认为硬件计时是否可用（动态判定：拿到过有效值）。 */
  gpuTimingAvailable: boolean
  /** CPU 提交耗时（ms），只反映命令下发开销。 */
  cpuMs: number
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

export const viewStats = signal<ViewStats>({
  fps: 0,
  gpuMs: -1,
  gpuTimingAvailable: false,
  cpuMs: 0,
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
/** 最近一次的顶点缓冲：即使渲染器还没挂载（或热更新重建过）也不丢数据。 */
let latest: PackedSplats | null = null
let lastStatsPush = 0
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
    rendererError.value = null
  } catch (err) {
    renderer = null
    rendererError.value = err instanceof Error ? err.message : String(err)
    return
  }
  if (latest) renderer.setSplats(latest)
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
    const s = renderer.render(params)
    const now = performance.now()
    if (now - lastStatsPush >= 200) {
      lastStatsPush = now
      const cam = renderer.camera
      const applied = appliedPose()
      const desired = desiredPose(now)
      const toDeg = 180 / Math.PI
      viewStats.value = {
        ...s,
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
