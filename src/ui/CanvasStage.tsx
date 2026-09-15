/**
 * 全屏画布 + 交互（拖拽旋转 / 滚轮缩放 / 双击复位）。
 *
 * 这一层只做三件事：
 * 1. 把 canvas 交给渲染器（`attachCanvas`），卸载时 `detachCanvas`；
 * 2. 指针事件 -> 相机信号（`orbit` / `dolly` / `resetCamera`）；
 * 3. 参考照片加载完后塞进渲染器（叠加比对用）。
 *
 * React Compiler 已启用，这里不需要任何手写 memo/useCallback。
 */

import { useEffect, useRef } from "react"

import {
  cameraMode,
  dolly,
  inputSensitivity,
  orbit,
  resetCamera,
  setPointer,
} from "../store/camera.ts"
import { referenceImageUrl, status } from "../store/scene.ts"
import {
  attachCanvas,
  clearReferenceImage,
  detachCanvas,
  setReferenceImage,
} from "../store/viewer.ts"

/**
 * 像素 -> 弧度（基准值，还会乘用户灵敏度倍率）。
 *
 * 0.0022 ≈ 0.126°/px：横向拖 400 px 约转 50°；比 three.js OrbitControls
 * 的 `2π·dx/height` 慢一半多——本项目 fov 大、景深浅，同样角度画面位移更大，
 * 按 OrbitControls 的系数会「一拖就飞」。
 */
const DRAG_SENSITIVITY = 0.0022

/** 滚轮 delta -> 缩放指数（基准值，还会乘用户灵敏度倍率）。 */
const WHEEL_SENSITIVITY = 0.0008

/**
 * 单次滚轮事件的 delta 上限。
 * 触控板会以极高频率连发大 delta（一次滑动上百像素），不夹会「一跳到底」。
 */
const MAX_WHEEL_DELTA = 120

export function CanvasStage() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const refUrl = referenceImageUrl.useValue()
  const loadStatus = status.useValue()
  const sensitivity = inputSensitivity.useValue()
  const mode = cameraMode.useValue()

  // 渲染器生命周期：挂载建、卸载毁（StrictMode 下会跑两次，故必须成对）
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    attachCanvas(canvas)
    return () => detachCanvas()
  }, [])

  // 参考照片：sidecar json 里的 img_name 指向同目录的图片。
  // 加载失败必须显式告诉 store，否则叠加层会拿 1x1 占位纹理铺满屏幕。
  useEffect(() => {
    if (!refUrl) {
      clearReferenceImage()
      return
    }
    let cancelled = false
    const img = new Image()
    img.decoding = "async"
    img.onload = () => {
      if (!cancelled) setReferenceImage(img)
    }
    img.onerror = () => {
      if (!cancelled) clearReferenceImage()
    }
    img.src = refUrl
    return () => {
      cancelled = true
    }
  }, [refUrl])

  // 视差跟踪挂在 **window** 上，而不是画布上。
  //
  // 面板是浮在画布之上的**兄弟层**：若监听画布，鼠标一移进面板就会触发
  // 画布的 pointerleave，把视差复位成 0 —— 表现为「鼠标碰到面板，画面就弹回
  // 参考视角」。监听 window 后，光标在面板上也能继续驱动视差（面板本来就是
  // 同一屏的一部分），只有真正离开窗口才回中。
  //
  // 归一化仍以**画布矩形**为基准：视差方向要对应画面内容，而不是窗口。
  useEffect(() => {
    function onMove(e: PointerEvent) {
      const canvas = canvasRef.current
      if (!canvas) return
      const rect = canvas.getBoundingClientRect()
      setPointer(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        ((e.clientY - rect.top) / rect.height) * 2 - 1,
      )
    }
    function onLeave() {
      setPointer(0, 0)
    }
    window.addEventListener("pointermove", onMove)
    // 只有指针真正离开文档才回中（touch-action: none 下不会误触）
    document.addEventListener("pointerleave", onLeave)
    return () => {
      window.removeEventListener("pointermove", onMove)
      document.removeEventListener("pointerleave", onLeave)
    }
  }, [])

  // 指针拖拽：用 pointer capture，指针移出画布也不会断
  const drag = useRef<{ id: number; x: number; y: number } | null>(null)

  function handlePointerDown(e: React.PointerEvent<HTMLDivElement>) {
    if (e.button !== 0) return
    drag.current = { id: e.pointerId, x: e.clientX, y: e.clientY }
    e.currentTarget.dataset.dragging = "true"
    e.currentTarget.setPointerCapture(e.pointerId)
  }

  function handlePointerMove(e: React.PointerEvent<HTMLDivElement>) {
    const d = drag.current
    if (!d || d.id !== e.pointerId) return
    const dx = e.clientX - d.x
    const dy = e.clientY - d.y
    d.x = e.clientX
    d.y = e.clientY
    // 只有自由模式的拖拽会写姿态；视差/自动模式下姿态由输入源/时间驱动，
    // 写了也不会生效（所以干脆不写，避免模式切换时突然跳一下）。
    // 视差的指针位置另由 window 上的监听器统一写入，不在这里重复。
    if (mode !== "free") return
    // 水平：向右拖 -> 相机绕枢轴向左回，画面内容跟着手指走。
    // 垂直：向下拖 -> **抬高**相机（俯视场景），与 OrbitControls / 主流
    // 3DGS 查看器一致（渲染世界是 y 上，而 PLY 系是 y 下，两者差一个翻转）。
    const k = DRAG_SENSITIVITY * sensitivity
    orbit(-dx * k, dy * k)
  }

  function endDrag(e: React.PointerEvent<HTMLDivElement>) {
    if (drag.current?.id !== e.pointerId) return
    drag.current = null
    delete e.currentTarget.dataset.dragging
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId)
    }
  }

  function handleWheel(e: React.WheelEvent<HTMLDivElement>) {
    const delta = Math.max(
      -MAX_WHEEL_DELTA,
      Math.min(MAX_WHEEL_DELTA, e.deltaY),
    )
    dolly(Math.exp(delta * WHEEL_SENSITIVITY * sensitivity))
  }

  /** 指针离开窗口：视差回中（否则会停在最后一个角度上）。
   *  注意这里**不是**画布的 pointerleave —— 参见上面 window 监听器的注释。 */

  return (
    <div
      className="stage absolute inset-0"
      data-mode={mode}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
      onWheel={handleWheel}
      onDoubleClick={resetCamera}
    >
      <canvas ref={canvasRef} className="block h-full w-full" />
      {loadStatus === "ready" ? null : (
        // 数据没来时压一层极暗的底，避免看到纯黑画布以为「坏了」
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(120%_80%_at_50%_0%,#0b1020_0%,#05060a_60%)]" />
      )}
    </div>
  )
}
