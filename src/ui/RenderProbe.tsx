/**
 * 渲染探针：接管 R3F 的最后一帧渲染，顺便量 GPU 实测时间与 draw call。
 *
 * 为什么用 `useFrame(..., priority = 1)`：R3F 只要发现有 `priority > 0` 的回调，
 * 就**不再自动渲染**，把 `gl.render()` 交给它。这样时间戳查询才能精确包住整个
 * 场景绘制（含透明排序的多趟 draw），量到的是整帧 GPU 时间。
 *
 * 每帧推一条样本进 `store/stats.ts`（那边做滑动窗口 + 分位数 + 2 Hz 限频）。
 *
 * ⚠ 画布是 `frameloop="demand"`：空闲时根本不进这里。所以帧间隔只在上一次
 * 也是「连续帧」（相机在动）时才有意义 —— 否则空闲时长会被算成一次超大帧间隔，
 * 把平均帧率拉爆（对应 direct-splat 里的 idle 跳过）。
 */

import { useFrame, useThree } from "@react-three/fiber"
import { useEffect, useRef } from "react"
import { cameraMode, cameraSettled } from "../store/camera.ts"
import {
  flushStats,
  pushRenderSample,
  resetStats,
  STATS_FLUSH_MS,
} from "../store/index.ts"
import { createGpuTimer, type GpuTimer } from "./gpuTimer.ts"

export function RenderProbe() {
  const { gl, scene, camera, size } = useThree()
  // undefined = 还没初始化；null = 平台没有该扩展
  const timerRef = useRef<GpuTimer | null | undefined>(undefined)
  const lastMs = useRef(0)
  /** 上一次渲染时相机是否在动（连续帧序列的判据）。 */
  const lastAnimated = useRef(false)

  useEffect(() => {
    resetStats()
    // 面板刷新与帧解耦：demand 模式下空闲没有帧，但面板仍要按 2 Hz 刷新。
    const id = setInterval(flushStats, STATS_FLUSH_MS)
    return () => {
      clearInterval(id)
      timerRef.current?.dispose()
      timerRef.current = undefined
    }
  }, [])

  useFrame(() => {
    if (timerRef.current === undefined) {
      timerRef.current = createGpuTimer(
        gl.getContext() as WebGL2RenderingContext,
      )
    }
    const timer = timerRef.current
    const t0 = performance.now()
    // 只有「上一次也是连续帧」时，间隔才是真实帧间隔；空闲后的第一帧不计。
    const frameMs =
      lastAnimated.current && lastMs.current > 0 ? t0 - lastMs.current : 0
    lastMs.current = t0
    lastAnimated.current = cameraMode.peek() !== "free" || !cameraSettled(t0)

    timer?.begin()
    const t1 = performance.now()
    gl.render(scene, camera)
    const cpuMs = performance.now() - t1
    timer?.end()
    const gpuMs = timer?.poll() ?? null

    const dpr = gl.getPixelRatio()
    pushRenderSample({
      t: t0,
      frameMs,
      cpuMs,
      gpuMs,
      drawCalls: gl.info.render.calls,
      triangles: gl.info.render.triangles,
      width: Math.round(size.width * dpr),
      height: Math.round(size.height * dpr),
      gpuTimingAvailable: timer?.available ?? false,
    })
  }, 1)

  return null
}
