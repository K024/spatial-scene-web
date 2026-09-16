/**
 * 全窗口 three 视图：GLB 分层网格 + 网格地面 + 坐标轴 gizmo + 轨道相机。
 *
 * 本组件是**纯消费者**：`<primitive object={glbScene} />` 直接渲染 `GLTFLoader`
 * 载入的场景；视图开关通过 effect 命令式地作用到各层网格上。
 *
 * 相机沿用 direct-splat 的视角方案（`store/camera.ts`）：`CameraDriver` 每帧把
 * 缓动后的轨道姿态应用到 three 相机上，输入（拖拽/滚轮/指针）在这里接线。
 * 不接 drei 的 OrbitControls —— 那套的阻尼/缩放模型与 direct-splat 的
 * 「枢轴 + 球坐标 + 请求/实际双状态」不是一回事，混用会互相打架。
 *
 * 渲染是 `frameloop="demand"`：空闲时不画。但我们**接管了主渲染**
 * （`RenderProbe` 的 `priority=1`），所以「什么时候要一帧」得自己给：
 * 输入/信号变化时 `invalidate()`，相机缓动未到位时由 `CameraDriver` 续帧。
 */

import { GizmoHelper, GizmoViewport, Grid } from "@react-three/drei"
import { Canvas, invalidate } from "@react-three/fiber"
import { useEffect, useMemo, useRef } from "react"
import * as THREE from "three"
import {
  autoAmplitude,
  autoPeriod,
  background,
  cameraMode,
  dolly,
  doubleSided,
  easingMs,
  explode,
  type GlbMeta,
  glbMeta,
  glbScene,
  inputSensitivity,
  isolateLayer,
  layerOpacity,
  orbit,
  parallaxAmplitude,
  resetCamera,
  setPointer,
  showGizmo,
  showGrid,
  wireframe,
} from "../store/index.ts"
import { CameraDriver } from "./CameraDriver.tsx"
import { RenderProbe } from "./RenderProbe.tsx"

/**
 * 像素 -> 弧度（基准值，还会乘用户灵敏度倍率）。
 * 0.0022 ≈ 0.126°/px：横向拖 400 px 约转 50°（direct-splat 的取值）。
 */
const DRAG_SENSITIVITY = 0.0022
/** 滚轮 delta -> 缩放指数（基准值）。 */
const WHEEL_SENSITIVITY = 0.0008
/** 单次滚轮事件的 delta 上限（触控板会以极高频率连发大 delta）。 */
const MAX_WHEEL_DELTA = 120

export function Viewer3D() {
  const bg = background.useValue()
  const scene = glbScene.useValue()
  const meta = glbMeta.useValue()
  const grid = showGrid.useValue()
  const gizmo = showGizmo.useValue()
  const mode = cameraMode.useValue()

  const wrapperRef = useRef<HTMLDivElement>(null)
  const drag = useRef<{ id: number; x: number; y: number } | null>(null)

  // 视差跟踪挂 **window** 上：面板是浮在画布之上的兄弟层，监听画布会在鼠标
  // 移进面板时误触发 pointerleave、把视差复位。归一化仍以画布矩形为基准。
  useEffect(() => {
    function onMove(e: PointerEvent) {
      const el = wrapperRef.current
      if (!el) return
      const rect = el.getBoundingClientRect()
      setPointer(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        ((e.clientY - rect.top) / rect.height) * 2 - 1,
      )
      invalidate()
    }
    function onLeave() {
      setPointer(0, 0)
      invalidate()
    }
    window.addEventListener("pointermove", onMove)
    document.addEventListener("pointerleave", onLeave)
    return () => {
      window.removeEventListener("pointermove", onMove)
      document.removeEventListener("pointerleave", onLeave)
    }
  }, [])

  function onPointerDown(e: React.PointerEvent<HTMLDivElement>) {
    if (e.button !== 0) return
    drag.current = { id: e.pointerId, x: e.clientX, y: e.clientY }
    e.currentTarget.dataset.dragging = "true"
    e.currentTarget.setPointerCapture(e.pointerId)
  }

  function onPointerMove(e: React.PointerEvent<HTMLDivElement>) {
    const d = drag.current
    if (!d || d.id !== e.pointerId) return
    const dx = e.clientX - d.x
    const dy = e.clientY - d.y
    d.x = e.clientX
    d.y = e.clientY
    // 只有自由模式的拖拽会写姿态；视差/自动模式下姿态由输入源/时间驱动。
    if (cameraMode.peek() !== "free") return
    const k = DRAG_SENSITIVITY * inputSensitivity.peek()
    // 水平：向右拖 -> 相机绕枢轴向左回，画面内容跟着手指走。
    // 垂直：向下拖 -> 抬高相机（俯视场景）。
    orbit(-dx * k, dy * k)
    invalidate()
  }

  function endDrag(e: React.PointerEvent<HTMLDivElement>) {
    if (drag.current?.id !== e.pointerId) return
    drag.current = null
    delete e.currentTarget.dataset.dragging
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId)
    }
  }

  function onWheel(e: React.WheelEvent<HTMLDivElement>) {
    const delta = Math.max(
      -MAX_WHEEL_DELTA,
      Math.min(MAX_WHEEL_DELTA, e.deltaY),
    )
    dolly(Math.exp(delta * WHEEL_SENSITIVITY * inputSensitivity.peek()))
    invalidate()
  }

  return (
    // `.stage`（index.css）负责 touch-action 与「自由模式可拖拽」光标。
    <div
      ref={wrapperRef}
      className="stage absolute inset-0"
      data-mode={mode}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
      onWheel={onWheel}
      onDoubleClick={() => {
        resetCamera()
        invalidate()
      }}
    >
      <Canvas
        flat
        // 空闲不画：由输入/信号 `invalidate()` 与 `CameraDriver` 的缓动续帧驱动。
        frameloop="demand"
        dpr={[1, 2]}
        gl={{ antialias: true, powerPreference: "high-performance" }}
        camera={{ position: [0, 0, 0], fov: 50, near: 0.005, far: 5000 }}
      >
        <color attach="background" args={[bg]} />

        {scene ? <primitive object={scene} dispose={null} /> : null}
        <LayerToggles meta={meta} />
        <CameraParamWatcher />
        <CameraDriver />
        <RenderProbe />

        {grid && meta ? <GroundGrid bounds={meta.bounds} /> : null}

        {gizmo ? (
          // renderPriority=2：我们在 priority=1 接管了主渲染，gizmo 必须在它之后
          // 单独画（drei 的 Hud 在 priority!==1 时不会再画一遍主场景）。
          <GizmoHelper
            alignment="bottom-left"
            margin={[76, 76]}
            renderPriority={2}
          >
            <GizmoViewport
              axisColors={["#f87171", "#4ade80", "#38bdf8"]}
              labelColor="#e5e7eb"
            />
          </GizmoHelper>
        ) : null}
      </Canvas>
    </div>
  )
}

/**
 * 相机参数信号 -> `invalidate()` 的桥。
 *
 * 相机参数不是 React props（面板在 Canvas 之外），信号变化不会经过 R3F 的
 * reconciler，所以 demand 模式下需要显式要一帧。这里把会改变请求姿态的信号
 * 读一遍：任一变化本组件重渲染，无依赖的 effect 就 `invalidate()`。
 */
function CameraParamWatcher() {
  cameraMode.useValue()
  inputSensitivity.useValue()
  parallaxAmplitude.useValue()
  autoAmplitude.useValue()
  autoPeriod.useValue()
  easingMs.useValue()
  useEffect(() => {
    invalidate()
  })
  return null
}

/** 网格地面：放在包围盒底部，避免从场景中间穿过。 */
function GroundGrid({ bounds }: { bounds: THREE.Box3 }) {
  const size = useMemo(() => {
    const s = bounds.getSize(new THREE.Vector3())
    return Math.max(s.x, s.y, s.z, 1e-3)
  }, [bounds])
  const center = useMemo(() => bounds.getCenter(new THREE.Vector3()), [bounds])
  return (
    <Grid
      position={[center.x, bounds.min.y - size * 0.06, center.z]}
      args={[size * 6, size * 6]}
      cellSize={size / 24}
      cellThickness={0.6}
      cellColor="#1e293b"
      sectionSize={size / 4}
      sectionThickness={1}
      sectionColor="#0e7490"
      fadeDistance={size * 3}
      fadeStrength={1}
      infiniteGrid
    />
  )
}

/** 把视图开关作用到各层网格（GLB 的材质是 unlit `MeshBasicMaterial`）。 */
function LayerToggles({ meta }: { meta: GlbMeta | null }) {
  const opacity = layerOpacity.useValue()
  const isWireframe = wireframe.useValue()
  const twoSided = doubleSided.useValue()
  const isolated = isolateLayer.useValue()
  const explodeAmount = explode.useValue()

  useEffect(() => {
    if (!meta) return
    const count = meta.layers.length
    const spread = layerSpread(meta)
    for (const layer of meta.layers) {
      // biome-ignore lint/nursery/useReactCompiler: three 网格是命令式对象，在 effect 里改 visible/position/材质是 R3F 的常规做法
      layer.mesh.visible = isolated < 0 || layer.layerIndex === isolated
      layer.mesh.position
        .copy(layer.basePosition)
        .setZ(
          layer.basePosition.z +
            explodeAmount * (count - 1 - layer.layerIndex) * spread,
        )
      const material = layer.mesh.material
      for (const m of Array.isArray(material) ? material : [material]) {
        const basic = m as THREE.MeshBasicMaterial
        basic.opacity = opacity
        basic.wireframe = isWireframe
        basic.side = twoSided ? THREE.DoubleSide : THREE.FrontSide
        basic.transparent = true
        basic.depthWrite = false
        basic.needsUpdate = true
      }
    }
    // 材质/可见性是命令式改的，R3F 不知道 -> 显式要一帧
    invalidate()
  }, [meta, opacity, isWireframe, twoSided, isolated, explodeAmount])

  return null
}

/** 满强度爆炸时相邻层的间距：取各层深度带跨度 / (L−1)，退化时用包围盒。 */
function layerSpread(meta: GlbMeta): number {
  let lo = Number.POSITIVE_INFINITY
  let hi = Number.NEGATIVE_INFINITY
  for (const layer of meta.layers) {
    if (!layer.depthRange) continue
    lo = Math.min(lo, layer.depthRange[0])
    hi = Math.max(hi, layer.depthRange[1])
  }
  const count = Math.max(1, meta.layers.length - 1)
  if (Number.isFinite(lo) && hi > lo) return (hi - lo) / count
  const size = meta.bounds.getSize(new THREE.Vector3())
  return Math.max(1e-3, size.length() * 0.3) / count
}
