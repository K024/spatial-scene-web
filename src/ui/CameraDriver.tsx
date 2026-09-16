/**
 * 每帧把缓动后的轨道姿态应用到 three 相机上（direct-splat 视角方案接入 R3F）。
 *
 * 姿态来自 `store/camera.ts`：`stepCamera()` 内部按模式（自由/视差/自动）解析
 * 请求值再指数缓动。这里只做「姿态 -> 相机位置/朝向/投影」这一步：
 *
 * ```
 * pivot = (0, 0, -focusZ)              // 烘焙后 +z 指向参考相机，枢轴在 −z
 * dir   = (sinY·cosP, sinP, cosY·cosP) // yaw=0,pitch=0 -> (0,0,1)
 * eye   = pivot + dir · focusZ · zoom  // zoom=1 时 eye = 原点（参考相机位）
 * ```
 *
 * 投影用 **contain 拟合**（不是「固定 fovY」）：GLB 里 `extras.camera.focalLengthPx`
 * 是渲染画布（`extras.width/height`）的像素焦距，按 `min(W/vw, H/vh)` 缩放，
 * 于是任何画布宽高比下都能看到完整的分层画布、且不拉伸 —— 与 direct-splat 的
 * `fitScale` 口径一致。
 */

import { invalidate, useFrame } from "@react-three/fiber"
import { useRef } from "react"
import * as THREE from "three"
import { cameraMode, cameraSettled, stepCamera } from "../store/camera.ts"
import { type GlbMeta, glbMeta } from "../store/index.ts"

const RAD2DEG = 180 / Math.PI

export function CameraDriver() {
  const meta = glbMeta.useValue()
  const lastMs = useRef(0)
  const pivot = useRef(new THREE.Vector3())
  const dir = useRef(new THREE.Vector3())

  useFrame((state) => {
    if (!meta) return
    const now = performance.now()
    const dt = lastMs.current === 0 ? 16.7 : now - lastMs.current
    lastMs.current = now
    const pose = stepCamera(dt, now)

    const focus = Math.max(1e-3, meta.focusZ)
    pivot.current.set(0, 0, -focus)
    const cp = Math.cos(pose.pitch)
    dir.current.set(
      Math.sin(pose.yaw) * cp,
      Math.sin(pose.pitch),
      Math.cos(pose.yaw) * cp,
    )

    const cam = state.camera as THREE.PerspectiveCamera
    cam.position
      .copy(pivot.current)
      .addScaledVector(dir.current, focus * pose.zoom)
    cam.up.set(0, 1, 0)
    cam.lookAt(pivot.current)
    applyProjection(cam, meta, state.size.width, state.size.height)

    // `frameloop="demand"`：缓动未到位（或视差/自动在动）就再要一帧，停下来就不画。
    if (cameraMode.peek() !== "free" || !cameraSettled(now)) invalidate()
  })

  return null
}

/** 内参：contain 拟合 GLB 的分层画布；缺元数据时退回 GLB 相机的 fovY。 */
function applyProjection(
  cam: THREE.PerspectiveCamera,
  meta: GlbMeta,
  width: number,
  height: number,
): void {
  const extras = meta.extras
  const viewW = numberValue(extras.width)
  const viewH = numberValue(extras.height)
  const cameraExtras = extras.camera as { focalLengthPx?: number } | undefined
  const focalView = cameraExtras?.focalLengthPx

  if (viewW && viewH && focalView) {
    const fit = Math.min(width / viewW, height / viewH)
    const focal = focalView * fit
    cam.fov = 2 * Math.atan(height / (2 * focal)) * RAD2DEG
  } else if (meta.camera) {
    cam.fov = meta.camera.fovDeg
  }
  cam.aspect = width / Math.max(1, height)
  if (meta.camera) {
    cam.near = meta.camera.near
    cam.far = meta.camera.far
  }
  cam.updateProjectionMatrix()
}

function numberValue(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) && value > 0
    ? value
    : null
}
