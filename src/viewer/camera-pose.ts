/**
 * 相机位姿的共享原语 —— `ParallaxSway` 与 `ReferenceViewReset` 共用。
 *
 * ── 为什么单独一个文件 ──
 * 「参考视角」这件事有两个消费者（摆动从它出发、重置回到它），而它由**两条独立的
 * 约定**共同定义，散在两处迟早会漂：
 * 1. **参考机位 = 世界原点**：导出器已把几何 bake 到 glTF 系（Y 上 / −Z 前），
 *    相机就是单位变换 —— 与 three 相机的默认朝向完全一致，不需要任何姿态换算；
 * 2. **焦点 = 光轴上、场景中心深度处** `(0, 0, bounds.center.z)`：它落在光轴上，
 *    所以参考视角下「画面中心」正好是场景中心，围绕它做视差 / 回位都不会歪。
 *
 * ── 朝向一律用 `Matrix4.lookAt`，不要用 `Object3D.lookAt` ──
 * `Object3D.lookAt` 会看 `this.isCamera` 决定正反方向；拿一个普通 `Object3D` 当临时
 * 容器算出来的是**反的**。`Matrix4.lookAt(eye, target, up)` 就是相机约定，没有歧义。
 */

import { Matrix4, Quaternion, Vector3 } from "three"
import type { LoadedScene } from "./types.ts"

/** glTF 系的上方向（导出器已把 OpenCV 的 y 下 翻成 y 上）。 */
export const WORLD_UP = new Vector3(0, 1, 0)

/** 一个可插值的相机位姿。 */
export interface Pose {
  readonly position: Vector3
  readonly quaternion: Quaternion
}

/** 只读位姿（`three` 的相机对象天然满足）。 */
export interface PoseLike {
  readonly position: Vector3
  readonly quaternion: Quaternion
}

const lookMatrix = new Matrix4()
const pivotScratch = new Vector3()

/** 「参考视角」的焦点：光轴上、场景中心深度处。 */
export function referencePivot(scene: LoadedScene, out: Vector3): Vector3 {
  return out.set(0, 0, scene.bounds.center[2])
}

/** 在 `eye` 处看向 `target` 的朝向。 */
export function lookAtQuaternion(
  eye: Vector3,
  target: Vector3,
  out: Quaternion,
): Quaternion {
  lookMatrix.lookAt(eye, target, WORLD_UP)
  return out.setFromRotationMatrix(lookMatrix)
}

/** 参考位姿：机位在原点、朝向焦点。 */
export function referencePose(scene: LoadedScene): Pose {
  const position = new Vector3(0, 0, 0)
  const quaternion = lookAtQuaternion(
    position,
    referencePivot(scene, pivotScratch),
    new Quaternion(),
  )
  return { position, quaternion }
}

/** 抓一份当前位姿的**深拷贝**（之后慢慢插值，不能再被相机改动）。 */
export function capturePose(camera: PoseLike): Pose {
  return {
    position: camera.position.clone(),
    quaternion: camera.quaternion.clone(),
  }
}

/**
 * 三次缓入缓出。
 *
 * 给「按钮触发的一次性镜头运动」用：固定时长 + 缓入缓出，起止速度都是 0，
 * 观感上比指数阻尼更「有头有尾」（阻尼永远够不到终点，只能靠阈值吸附）。
 * 持续型的跟随（摆动）仍然用指数阻尼 —— 那里要的是「永远追得上一个动目标」。
 */
export function easeInOutCubic(t: number): number {
  if (t <= 0) return 0
  if (t >= 1) return 1
  return t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2
}
