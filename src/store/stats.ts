/**
 * 渲染统计（GPU 计时 + draw call）与面板快照。
 *
 * 数据源只有一个：`ui/RenderProbe.tsx` 每帧渲染完把一组实测值推进来
 * （{@link pushRenderSample}）。这里做滑动窗口 + 分位数，并**限频**更新
 * `viewStats` 信号（2 Hz），避免 60 Hz 触发 React 重渲染。
 *
 * ── 为什么要有 GPU 实测 ──
 * `performance.now()` 量到的只是**命令下发**用了多久。GL 调用是异步的，
 * `drawElements` 提交完就返回，真正跑多少时间在 GPU 队列里。计时查询
 * （`EXT_disjoint_timer_query_webgl2`）在命令流里插一对时间戳，GPU 执行到
 * 那里才计数，量到的才是真实执行时间（见 `ui/gpuTimer.ts`）。
 *
 * ── 分位数口径 ──
 * p95/p99 用**最近邻次序统计量**（不插值）：结果一定是真实测到过的某一帧。
 * 帧间隔含提交+等待+合成，会被垂直同步钉在刷新周期，所以「稳不稳」主要看
 * GPU 实测那一行的平均值与 p99 差距。
 */

import "./signals-hook.ts"
import { signal } from "@preact/signals-react"

/** 一组分布统计（ms）。全部是实测值，不做插值。 */
export interface DistStats {
  /** 样本数。p95/p99 需要足够样本才有意义。 */
  count: number
  avg: number
  min: number
  p50: number
  p95: number
  p99: number
  max: number
}

const EMPTY_DIST: DistStats = {
  count: 0,
  avg: 0,
  min: 0,
  p50: 0,
  p95: 0,
  p99: 0,
  max: 0,
}

/** 统计快照（限频 2 Hz 更新）。 */
export interface ViewStats {
  windowMs: number
  /** 平均帧率（= 1000 / 渲染帧平均间隔，不是瞬时值平均）。 */
  fps: number
  frame: DistStats
  cpu: DistStats
  gpu: DistStats
  /** 本环境是否拿得到硬件计时（动态判定：拿到过有效值）。 */
  gpuTimingAvailable: boolean
  /** 窗口内被排除的长停顿次数（> 1 s，多半是切标签页）。 */
  stalls: number
  /** 实际绘制缓冲尺寸（像素，含 DPR）。 */
  width: number
  height: number
  drawCalls: number
  triangles: number
}

/** 统计窗口选项（面板可切）。 */
export const STATS_WINDOWS = [
  { value: 2000, label: "2 s" },
  { value: 5000, label: "5 s" },
  { value: 10000, label: "10 s" },
]

/** 统计窗口长度（ms）。越长越稳，但反映变化越慢。 */
export const statsWindowMs = signal(5000)

/** 面板刷新间隔（ms）：2 Hz。与帧解耦（demand 模式下空闲无帧也要刷新）。 */
export const STATS_FLUSH_MS = 500

/**
 * 超过这个时长的「帧间隔」不计入统计，只计一次长停顿。
 * 这类值几乎都是 rAF 被暂停（切标签页、断点、系统休眠），不是渲染慢。
 */
const MAX_SANE_FRAME_MS = 1000

/** 样本上限（10 s 窗口 + 高刷屏也不至于无限增长）。 */
const MAX_SAMPLES = 8192

interface Sample {
  t: number
  kind: "frame" | "stall"
  /** 与上一个**渲染帧**的间隔（ms）；无有效间隔（首帧）时为 0。 */
  frameMs: number
  cpuMs: number
  gpuMs: number | null
}

let samples: Sample[] = []
/** 最近一帧的非分布信息（draw call / 尺寸 / 计时可用性），空闲时也要能显示。 */
let latest: RenderSample | null = null
let distCache = {
  frame: EMPTY_DIST,
  cpu: EMPTY_DIST,
  gpu: EMPTY_DIST,
  stalls: 0,
}

/** 一帧的实测值（由 `RenderProbe` 推进来）。 */
export interface RenderSample {
  t: number
  frameMs: number
  cpuMs: number
  gpuMs: number | null
  drawCalls: number
  triangles: number
  width: number
  height: number
  gpuTimingAvailable: boolean
}

export const viewStats = signal<ViewStats>({
  windowMs: 5000,
  fps: 0,
  frame: EMPTY_DIST,
  cpu: EMPTY_DIST,
  gpu: EMPTY_DIST,
  gpuTimingAvailable: false,
  stalls: 0,
  width: 0,
  height: 0,
  drawCalls: 0,
  triangles: 0,
})

/**
 * 收一帧样本。
 *
 * ⚠ 只记录，**不碰信号**：画布是 `frameloop="demand"`，空闲时根本没有帧，
 * 若在这里推信号，面板会停在「最后一帧恰好是空场景」那一刻。快照由
 * {@link flushStats} 按固定节奏推（探针里的 2 Hz 定时器）。
 */
export function pushRenderSample(s: RenderSample): void {
  latest = s
  samples.push({
    t: s.t,
    kind: s.frameMs > MAX_SANE_FRAME_MS ? "stall" : "frame",
    frameMs: s.frameMs,
    cpuMs: s.cpuMs,
    gpuMs: s.gpuMs,
  })
  if (samples.length > MAX_SAMPLES) {
    samples.splice(0, samples.length - MAX_SAMPLES)
  }
}

/**
 * 重算滑动窗口并把快照推给面板（由探针的 2 Hz 定时器调用）。
 *
 * 与帧解耦是关键：demand 模式下空闲没有帧，但面板仍需刷新（否则会停在
 * 上一次的陈旧值上）；同时 draw call / 尺寸取**最近一帧**，空闲也显示得出来。
 */
export function flushStats(): void {
  const s = latest
  if (!s) return
  const now = performance.now()
  recomputeDistributions(now)
  viewStats.value = {
    windowMs: statsWindowMs.value,
    fps: distCache.frame.count > 0 ? 1000 / distCache.frame.avg : 0,
    frame: distCache.frame,
    cpu: distCache.cpu,
    gpu: distCache.gpu,
    gpuTimingAvailable: s.gpuTimingAvailable,
    stalls: distCache.stalls,
    width: s.width,
    height: s.height,
    drawCalls: s.drawCalls,
    triangles: s.triangles,
  }
}

/** 清空统计（切换窗口长度 / 重新挂载画布 / 换文件时调用）。 */
export function resetStats(): void {
  samples = []
  latest = null
  distCache = {
    frame: EMPTY_DIST,
    cpu: EMPTY_DIST,
    gpu: EMPTY_DIST,
    stalls: 0,
  }
}

/** 重算窗口内的三个分布。 */
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
    if (s.kind === "stall") {
      stalls++
      continue
    }
    cpu.push(s.cpuMs)
    if (s.frameMs > 0) frame.push(s.frameMs)
    if (s.gpuMs !== null) gpu.push(s.gpuMs)
  }
  distCache = {
    frame: dist(frame),
    cpu: dist(cpu),
    gpu: dist(gpu),
    stalls,
  }
}

/** 分位数（最近邻次序统计量，不插值）。 */
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
