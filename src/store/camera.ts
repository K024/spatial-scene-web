/**
 * 相机模式 + 「请求（requested）/ 实际（applied）」双状态。
 *
 * 直接沿用 direct-splat 分支的视角方案（轨道姿态 + 三种输入源 + 指数缓动），
 * 只是把「每帧应用姿态」这一步接到了 three 相机上（见 `ui/CameraDriver.tsx`）。
 *
 * ── 为什么分两层 ──
 * 三种模式（自由 / 视差 / 自动）产出的都是**同一个东西**：一个轨道姿态
 * `{yaw, pitch, zoom}`。差别只在「谁在写请求值」：
 *
 *   free     : 拖拽/滚轮把增量累积进 `freeYaw/freePitch/freeZoom`（信号）
 *   parallax : 指针位置直接映射成角度（绝对映射，松手回中 = 回参考视角）
 *   auto     : 时间驱动的 Lissajous 轨迹（绝对映射）
 *
 * 实际姿态是一个**每帧向请求值指数缓动**的独立状态。这样带来三件事：
 * 1. 模式切换天然平滑 —— 切换只改「请求值的来源」，实际值自己会滑过去；
 * 2. 自动视差的轨迹自带跟随滞后，看起来比硬贴更自然；
 * 3. 输入抖动被时间常数吸收（拖拽手感用更小的常数，见 `FREE_TAU_FACTOR`）。
 *
 * ── 为什么 applied 不是 signal ──
 * 它每帧都变。做成 signal 会让面板（订阅了它）60 Hz 重渲染。
 * 所以它是普通可变对象，只在 rAF/useFrame 循环里被读写；面板要显示时走
 * `poseSnapshot()`（低频读取）。
 */

import { signal } from "@preact/signals-react"

// 副作用导入：把 useValue() 装到 Signal.prototype（供 React 组件消费）
import "./signals-hook.ts"

/** 相机模式。 */
export type CameraMode = "free" | "parallax" | "auto"

/** 模式列表（面板与快捷键共用一处定义）。 */
export const CAMERA_MODES: { value: CameraMode; label: string }[] = [
  { value: "free", label: "自由" },
  { value: "parallax", label: "视差" },
  { value: "auto", label: "自动" },
]

/** 当前模式。 */
export const cameraMode = signal<CameraMode>("free")

// ── 自由模式：请求姿态（输入源直接写）─────────────────────

/** 累积的偏航角（弧度）。 */
export const freeYaw = signal(0)
/** 累积的俯仰角（弧度）。 */
export const freePitch = signal(0)
/** 到枢轴的距离倍数（相对参考距离）。 */
export const freeZoom = signal(1)
/** 拖拽/滚轮灵敏度倍率。 */
export const inputSensitivity = signal(1)

// ── 视差模式的输入通道 ────────────────────────────────────

/**
 * 归一化指针位置，`[-1, 1]`，`(0, 0)` = 画面中心 = 参考视角。
 *
 * **这是视差的唯一输入通道**，所以接陀螺仪时只要把 `deviceorientation` 的
 * `gamma/beta` 归一化后写进这两个信号即可。
 */
export const pointerX = signal(0)
export const pointerY = signal(0)

/** 视差幅度（度）：指针顶到边缘时的相机转角。 */
export const parallaxAmplitude = signal(3)

// ── 自动视差 ──────────────────────────────────────────────

/** 自动视差幅度（度）。 */
export const autoAmplitude = signal(3)
/** 自动视差周期（秒）：偏航走一个完整正弦的时间。 */
export const autoPeriod = signal(9)

// ── 统一缓动 ──────────────────────────────────────────────

/**
 * 缓动时间常数（ms）：实际值追上请求值的特征时间。
 * 0 = 硬贴（无缓动），越大越「飘」。
 */
export const easingMs = signal(160)

/** 俯仰上限：贴到 ±90° 会让 `lookAt` 的 up 退化，留一点余量。 */
export const PITCH_LIMIT = deg2rad(78)

/** 缩放范围（三种模式共用）。 */
export const ZOOM_MIN = 0.45
export const ZOOM_MAX = 2.2

/**
 * 自由模式的缓动系数（乘在统一时间常数上）。
 * 拖拽必须「跟手」，所以用更小的常数；视差/自动用完整常数以获得柔顺感。
 */
const FREE_TAU_FACTOR = 0.35

/**
 * 俯仰相对偏航的幅度比例。
 *
 * 物理上两轴等角 = 等横向位移，但横向构图里竖直方向更容易「晃」，
 * 所以按视觉习惯压到 0.7；这是审美取值，不是几何约束。
 */
const PITCH_RATIO = 0.7

/** 自动视差的俯仰频率比（与偏航取不同频率，轨迹才是一个漫游的 8 字）。 */
const AUTO_PITCH_FREQ_RATIO = 1.37

/** 轨道姿态。 */
export interface Pose {
  yaw: number
  pitch: number
  zoom: number
}

/** 实际姿态：**故意不是 signal**（每帧变化，见文件头注释）。 */
const applied: Pose = { yaw: 0, pitch: 0, zoom: 1 }

/** 缓动吸附阈值（见 `stepCamera`）。 */
const SNAP_EPS = 1e-6

/** 自动视差的相位起点（进入自动模式或复位时重置，保证运动可预期）。 */
let autoStartMs = 0

/** 复用的请求值对象，避免每帧分配。 */
const desireScratch: Pose = { yaw: 0, pitch: 0, zoom: 1 }

/** 求当前模式下的**请求**姿态（纯函数，无副作用）。 */
export function desiredPose(nowMs: number, out: Pose = desireScratch): Pose {
  const mode = cameraMode.peek()
  if (mode === "parallax") {
    const amp = deg2rad(parallaxAmplitude.peek())
    // 方向：两个轴都是「画面跟着指针走」（与自由模式的拖拽同一个约定）。
    out.yaw = clamp(pointerX.peek(), -1, 1) * -amp
    out.pitch = clamp(pointerY.peek(), -1, 1) * amp * PITCH_RATIO
  } else if (mode === "auto") {
    const amp = deg2rad(autoAmplitude.peek())
    const period = Math.max(1, autoPeriod.peek())
    const t = (nowMs - autoStartMs) / 1000
    const w = (2 * Math.PI) / period
    out.yaw = amp * Math.sin(w * t)
    out.pitch =
      amp * PITCH_RATIO * Math.sin(w * AUTO_PITCH_FREQ_RATIO * t + 1.1)
  } else {
    out.yaw = freeYaw.peek()
    out.pitch = freePitch.peek()
  }
  // 缩放在三种模式下共用：切模式不该把用户拉好的距离丢掉
  out.zoom = freeZoom.peek()
  return out
}

/**
 * 每帧步进：把 `applied` 向请求值指数缓动。
 *
 * 用 `alpha = dt / (dt + τ)`（`1 - e^{-u}` 的 Padé 逼近）而不是精确指数：
 * 它**恒在 [0,1)**，任何 dt（卡顿、标签页回来时可能是秒级）都不过冲。
 * 注意不能写成 `x += (target-x) * dt/τ` —— 那只是泰勒一阶项，`dt > τ` 时会振荡。
 */
export function stepCamera(dtMs: number, nowMs: number): Readonly<Pose> {
  const desire = desiredPose(nowMs)
  const base = Math.max(0, easingMs.peek())
  const tau = base * (cameraMode.peek() === "free" ? FREE_TAU_FACTOR : 1)
  const dt = Math.max(dtMs, 0)
  const alpha = tau <= 0 ? 1 : dt / (dt + tau)

  applied.yaw += (desire.yaw - applied.yaw) * alpha
  applied.pitch += (desire.pitch - applied.pitch) * alpha
  applied.zoom += (desire.zoom - applied.zoom) * alpha

  // 吸附到目标：指数（有理式）缓动是渐近的，靠浮点精度自然停下来要好几秒。
  if (Math.abs(desire.yaw - applied.yaw) < SNAP_EPS) applied.yaw = desire.yaw
  if (Math.abs(desire.pitch - applied.pitch) < SNAP_EPS)
    applied.pitch = desire.pitch
  if (Math.abs(desire.zoom - applied.zoom) < SNAP_EPS)
    applied.zoom = desire.zoom
  return applied
}

/** 只读实际姿态（面板/日志用；不产生响应式更新）。 */
export function appliedPose(): Readonly<Pose> {
  return applied
}

/**
 * 相机是否已经**停稳**（实际姿态 == 请求姿态）。
 *
 * `frameloop="demand"` 的判据：自由模式下停稳就不再请求下一帧；视差/自动
 * 模式的请求值随时间变，永远不会停。`stepCamera` 在差值小于 `SNAP_EPS`
 * 时会直接吸附，所以这里用精确相等判断。
 */
export function cameraSettled(nowMs: number): boolean {
  const desire = desiredPose(nowMs)
  return (
    applied.yaw === desire.yaw &&
    applied.pitch === desire.pitch &&
    applied.zoom === desire.zoom
  )
}

/** 由姿态角求相对参考视角（原点朝 +z）的偏角，度。 */
export function poseDeviationDeg(p: { yaw: number; pitch: number }): number {
  const c = Math.cos(p.pitch) * Math.cos(p.yaw)
  return (Math.acos(clamp(c, -1, 1)) * 180) / Math.PI
}

// ── 输入动作 ──────────────────────────────────────────────

/** 叠加一次轨道旋转（自由模式拖拽用）。 */
export function orbit(deltaYaw: number, deltaPitch: number): void {
  freeYaw.value = normalizeAngle(freeYaw.value + deltaYaw)
  freePitch.value = clamp(
    freePitch.value + deltaPitch,
    -PITCH_LIMIT,
    PITCH_LIMIT,
  )
}

/** 叠加缩放（三种模式共用；按比例而非线性，手感更稳）。 */
export function dolly(factor: number): void {
  freeZoom.value = clamp(freeZoom.value * factor, ZOOM_MIN, ZOOM_MAX)
}

/** 写入视差通道（归一化指针位置）。 */
export function setPointer(nx: number, ny: number): void {
  pointerX.value = clamp(nx, -1, 1)
  pointerY.value = clamp(ny, -1, 1)
}

/** 切换模式。切换本身不碰实际姿态，由缓动自然滑过去。 */
export function setCameraMode(mode: CameraMode): void {
  if (cameraMode.peek() === mode) return
  cameraMode.value = mode
  // 进入自动模式时相位归零：从参考姿态起步，轨迹可预期
  if (mode === "auto") autoStartMs = performance.now()
}

/** 在三种模式间循环（快捷键用）。 */
export function cycleCameraMode(): void {
  const i = CAMERA_MODES.findIndex((m) => m.value === cameraMode.peek())
  setCameraMode(CAMERA_MODES[(i + 1) % CAMERA_MODES.length].value)
}

/** 把请求姿态复位到参考视角（实际姿态会缓动跟过去）。 */
export function resetCamera(): void {
  freeYaw.value = 0
  freePitch.value = 0
  freeZoom.value = 1
  pointerX.value = 0
  pointerY.value = 0
  autoStartMs = performance.now()
}

/** 把角度折到 `(-π, π]`，避免长时间拖拽后数值无限增长。 */
function normalizeAngle(a: number): number {
  const twoPi = Math.PI * 2
  let x = a % twoPi
  if (x > Math.PI) x -= twoPi
  if (x <= -Math.PI) x += twoPi
  return x
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v
}

function deg2rad(deg: number): number {
  return (deg * Math.PI) / 180
}
