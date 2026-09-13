/**
 * 「回到参考视角」的**带动画**实现（不是瞬移）。
 *
 * ── 为什么单独一个组件 ──
 * `SceneRig` 里那条 effect 只负责「换场景时直接落到参考位姿」（新内容没有
 * 「从哪里飞过来」的语义），而**按钮按下**要的是可感知的运动：让人看清
 * 自己是从哪个角度、经过哪些层回到正视角的 —— 这本身就是观察场景结构的一部分。
 *
 * ── 曲线：固定时长 + 缓入缓出 ──
 * 位置 `lerpVectors`、朝向 `slerpQuaternions`，同一条 `easeInOutCubic`。
 * 不用指数阻尼：阻尼永远够不到终点（只能靠阈值吸附），按钮型动作观感会「没结尾」。
 *
 * ── 和 OrbitControls / 摆动的分工 ──
 * - OrbitControls **不禁用**：它每帧从当前相机位置反推球坐标再写回，不产生位移，
 *   所以不会和动画打架；动画结束后仍由它接管。
 * - 摆动会被强制关掉（两种相机驱动互斥），并取消摆动的「回放入场机位」（否则
 *   两边同时往不同方向拉）。用户一上手（`SceneRig` 的 `onStart`）也会取消本动画。
 */

import { useFrame, useThree } from "@react-three/fiber"
import { useEffect } from "react"
import { $resetViewToken, $scene, $sway } from "../store/viewer.ts"
import {
  capturePose,
  easeInOutCubic,
  type Pose,
  referencePose,
  WORLD_UP,
} from "./camera-pose.ts"
import { cancelSwayRestore } from "./ParallaxSway.tsx"

/** 一次回位的固定时长（秒）。够看清运动，又不至于让人等。 */
const FLIGHT_SECONDS = 0.8

interface Flight {
  readonly from: Pose
  readonly to: Pose
  elapsed: number
}

/**
 * 正在进行 / 待进行的回位动画。
 *
 * 放在模块级而不是 `useRef`：`SceneRig` 的 `onStart` 要能在用户一拖拽时
 * 就把它掐掉（`cancelReferenceReset`），跨组件共享一个可变槽位最轻。
 */
let flight: Flight | null = null

/** 取消正在进行的回位动画（用户开始操作时调）。 */
export function cancelReferenceReset(): void {
  flight = null
}

export function ReferenceViewReset() {
  const camera = useThree((state) => state.camera)
  const invalidate = useThree((state) => state.invalidate)

  useEffect(() => {
    // 按钮触发：从**当前**位姿起飞（所以必须现抓一份深拷贝）。
    const disposeReset = $resetViewToken.subscribe(() => {
      const scene = $scene.peek()
      if (!scene) return
      // 摆动与回位都是「全权驱动相机」，互斥；顺带取消摆动那边的回放，
      // 否则它会在摆动关掉的同一帧把相机往另一个方向拉。
      $sway.value = false
      cancelSwayRestore()
      // `up` 可能被 GizmoHelper 的轴向 tween 转过，参考位姿按世界 up 定义。
      camera.up.copy(WORLD_UP)
      flight = {
        from: capturePose(camera),
        to: referencePose(scene),
        elapsed: 0,
      }
      invalidate()
    })
    // 换场景时旧的回位目标没有意义。
    const disposeScene = $scene.subscribe(() => {
      flight = null
    })
    return () => {
      disposeReset()
      disposeScene()
    }
  }, [camera, invalidate])

  useFrame((_, delta) => {
    const current = flight
    if (!current) return
    // 摆动被打开：让位。两边都在「全权驱动相机」，后启动的那个赢。
    if ($sway.peek()) {
      flight = null
      return
    }
    current.elapsed += Math.min(delta, 0.1)
    const t = Math.min(current.elapsed / FLIGHT_SECONDS, 1)
    const eased = easeInOutCubic(t)
    camera.position.lerpVectors(
      current.from.position,
      current.to.position,
      eased,
    )
    camera.quaternion.slerpQuaternions(
      current.from.quaternion,
      current.to.quaternion,
      eased,
    )
    if (t >= 1) {
      flight = null
      return
    }
    // 按需渲染：动画期间自己续帧，到终点停手。
    invalidate()
  })

  return null
}
