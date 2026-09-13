/**
 * 场景装置：相机口径 + 参考视角 + 轨道控制 + 地面网格 + 坐标轴。
 *
 * ── 相机为什么在原点 ──
 * 导出器把几何 bake 到 glTF 系（Y 上 / −Z 前），**参考相机就是单位变换**。
 * 所以「参考视角」= `camera.position = (0,0,0)` + 看向 −Z —— 与 three 的相机默认朝向
 * 完全一致，不需要任何姿态换算。代价是 OrbitControls 的 pivot 不能放在相机位置，
 * 于是 pivot 取**场景包围盒中心的深度** `(0, 0, centerZ)`：既保持参考视角的正对关系，
 * 又让轨道有正常的旋转半径。
 *
 * ── fov 用 GLB 内的 verticalFOV ──
 * 这样「参考视角」下的透视缩短与原图一致（水平方向按画布宽高比自然扩展）。
 */

import {
  GizmoHelper,
  GizmoViewport,
  Grid,
  OrbitControls,
} from "@react-three/drei"
import { useThree } from "@react-three/fiber"
import { type ComponentRef, useEffect, useRef } from "react"
import { type PerspectiveCamera, Vector3 } from "three"
import {
  $resetViewToken,
  $scene,
  $showGizmo,
  $showGrid,
  $sway,
} from "../store/viewer.ts"
import type { LoadedScene } from "../viewer/types.ts"
import { referencePivot, WORLD_UP } from "./camera-pose.ts"
import { cancelSwayRestore, ParallaxSway } from "./ParallaxSway.tsx"
import {
  cancelReferenceReset,
  ReferenceViewReset,
} from "./ReferenceViewReset.tsx"

type ControlsRef = ComponentRef<typeof OrbitControls>

/** 参考焦点的临时容器（只在换场景 / 重置时用一次）。 */
const pivotScratch = new Vector3()

export function SceneRig() {
  const scene = $scene.useValue()
  const showGrid = $showGrid.useValue()
  const showGizmo = $showGizmo.useValue()
  const sway = $sway.useValue()
  const resetToken = $resetViewToken.useValue()

  const camera = useThree((state) => state.camera)
  const invalidate = useThree((state) => state.invalidate)
  const controls = useRef<ControlsRef | null>(null)

  // 相机口径跟随 GLB 元数据（fov / near / far）。
  // 变更放到模块级函数里：three 的相机是**可变句柄**，但 mutate 发生在组件体外，
  // React Compiler 的「不要 mutate hook 返回值」约束才不会被误触发。
  useEffect(() => {
    if (!scene) return
    applyCameraRig(camera as PerspectiveCamera, scene)
    // 命令式改了投影矩阵，按需渲染下要自己续一帧。
    invalidate()
  }, [scene, camera, invalidate])

  // 换场景：**直接落到参考位姿**。新内容没有「从哪里飞过来」的语义，不值得动画；
  // 而且旧场景的相机位姿对新场景的尺度毫无参考价值。
  // 变更放到模块级函数里：three 的相机是**可变句柄**，但 mutate 发生在组件体外，
  // React Compiler 的「不要 mutate hook 返回值」约束才不会被误触发。
  useEffect(() => {
    if (!scene) return
    cancelSwayRestore()
    applyReferencePose(camera as PerspectiveCamera, controls.current, scene)
    // 命令式改了投影矩阵与位姿，按需渲染下要自己续一帧。
    invalidate()
  }, [scene, camera, invalidate])

  // 「回到参考视角」：这里只把 OrbitControls 的**交点**归位（否则用户平移过之后
  // 下一次轨道会绕着错的点转），相机本身由 `ReferenceViewReset` 带动画飞回去。
  useEffect(() => {
    if (!scene) return
    const orbit = controls.current
    if (orbit) {
      const pivotZ = referencePivot(scene, pivotScratch).z
      orbit.target.set(0, 0, pivotZ)
      orbit.update()
    }
    invalidate()
    // `resetToken` 只是「再来一次」的触发位，值本身不参与计算。
    void resetToken
  }, [scene, resetToken, invalidate])

  const grid = scene ? gridParams(scene) : null

  return (
    <>
      {showGrid && scene && grid ? (
        <Grid
          position={[
            scene.bounds.center[0],
            scene.bounds.min[1],
            scene.bounds.center[2],
          ]}
          args={[1, 1]}
          cellSize={grid.cell}
          cellThickness={0.6}
          cellColor="#33405a"
          sectionSize={grid.cell * 5}
          sectionThickness={1.1}
          sectionColor="#55688f"
          fadeDistance={grid.fade}
          fadeStrength={1.6}
          followCamera={false}
          infiniteGrid
        />
      ) : null}

      <ParallaxSway />
      <ReferenceViewReset />

      <OrbitControls
        ref={controls}
        makeDefault
        // 摆动由 `ParallaxSway` 全权驱动相机；两边同时写 `camera.position` 会互相污染。
        enabled={!sway}
        // 用户一动（拖 / 滚 / 平移）：退出摆动，且把两个进行中的相机动画（摆动回放 /
        // 回参考视角）都掐掉 —— 他一上手就不该再有任何东西跟他抢镜头。
        onStart={() => {
          $sway.value = false
          cancelSwayRestore()
          cancelReferenceReset()
        }}
        enableDamping
        dampingFactor={0.08}
        rotateSpeed={0.75}
        zoomSpeed={0.9}
        panSpeed={0.8}
        minDistance={scene ? Math.max(scene.meta.near * 0.05, 0.005) : 0.05}
        maxDistance={scene ? Math.max(scene.meta.far * 3, 20) : 200}
      />

      {showGizmo ? (
        // margin 必须大于 GizmoViewport 自身的半径（`group scale={40}` ⇒ 轴头约在
        // ±60px），否则转到斜角时轴头 / 标签会被视口边缘剪掉。drei 默认值就是 80。
        <GizmoHelper alignment="bottom-left" margin={[64, 64]}>
          <GizmoViewport
            axisColors={["#f87171", "#4ade80", "#60a5fa"]}
            labelColor="#dbe3f0"
          />
        </GizmoHelper>
      ) : null}
    </>
  )
}

function radiansToDegrees(radians: number): number {
  return (radians * 180) / Math.PI
}

/** 相机直接落到参考视角（换场景用；按钮触发的那条带动画，见 `ReferenceViewReset`）。 */
function applyReferencePose(
  camera: PerspectiveCamera,
  orbit: ControlsRef | null,
  scene: LoadedScene,
): void {
  const pivotZ = referencePivot(scene, pivotScratch).z
  camera.position.set(0, 0, 0)
  camera.up.copy(WORLD_UP)
  camera.lookAt(0, 0, pivotZ)
  if (orbit) {
    orbit.target.set(0, 0, pivotZ)
    orbit.update()
  }
}

/**
 * 把 GLB 的相机口径写到 three 相机上。
 *
 * - `fov` 用 GLB 里的 `verticalFOV`：参考视角下的透视缩短才能和原图一致；
 * - `near` 比生成期更小：允许推进到层内部看断层 / 裙边；
 * - `far` 留出轨道余量，拉远时不会把最远的背衬平面裁掉。
 */
function applyCameraRig(camera: PerspectiveCamera, scene: LoadedScene): void {
  camera.fov = radiansToDegrees(scene.meta.verticalFOV)
  camera.near = Math.max(scene.meta.near * 0.25, 1e-4)
  camera.far = Math.max(scene.meta.far * 8, camera.near * 1e3)
  camera.updateProjectionMatrix()
}

/** 网格步长按场景尺度取「好看」的整数（1 / 2 / 5 × 10^n）。 */
function gridParams(scene: LoadedScene): { cell: number; fade: number } {
  const height = Math.max(scene.bounds.size[1], scene.bounds.size[2], 1e-3)
  const cell = niceStep(height / 6)
  return { cell, fade: cell * 60 }
}

function niceStep(raw: number): number {
  const exponent = Math.floor(Math.log10(Math.max(raw, 1e-6)))
  const base = 10 ** exponent
  const mantissa = raw / base
  const snapped =
    mantissa < 1.5 ? 1 : mantissa < 3.5 ? 2 : mantissa < 7.5 ? 5 : 10
  return snapped * base
}
