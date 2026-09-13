/**
 * 「视差摆动」：**固定焦点，让相机绕参考机位在像平面内做小回路**。
 *
 * ── 为什么不是 OrbitControls 的 `autoRotate` ──
 * `autoRotate` 是**绕 target（焦点）转**：镜头会在一个以主体为圆心的大球面上扫过，
 * 结果是「绕物体公转」，透视关系大变、像在看展台。
 * 本链路想要的是**头部微动**：焦点不动，相机自己在参考机位附近挪一点点，
 * 近景／远景的相对位移就露出来了 —— 这正是多层 mesh 存在的意义（视差），
 * 也才能一眼看出层间撕裂 / 裙边 / 回填有没有做对。
 *
 * ── 轨迹：在像平面（XY）内绕 Z 向轴走一个小回路 ──
 * 场景是**铺在 XY 平面上**的（深度沿 −Z），所以「视差摆动」必须发生在**像平面内**：
 *
 * ```
 * θ = (t − t₀)·ω              // t₀ = 开启摆动的时刻 ⇒ θ=0 时正好落在参考位姿
 * x = R·sin θ
 * y = R·k·(1 − cos θ)         // k = 0.35：压成椭圆，避免相机升得太高
 * z = 0                       // ★ 关键：绝不能有 Z 分量
 * ```
 * - `z ≡ 0` 是硬要求：相机沿视线方向前后挪是**推拉（dolly）**，只改变整体成像缩放，
 *   近景／远景不产生任何相对位移 —— 那不是视差，反而会暴露出「整幅在放大缩小」。
 * - `x` 走 `sin`、`y` 走 `1−cos`，两者相位差 90°（一个椭圆回路），
 *   视觉上就是「绕 Z 向轴转一圈」，而不是沿直线来回扫。
 * - 回路**穿过参考机位**（θ=0 时 `(0,0,0)`），而且相位从开启那一刻清零，
 *   所以「开启摆动」= 从当前机位阻尼滑回参考位姿、然后顺势起摆 ——
 *   入场是**精确**的参考视角，不是随机相位。
 * - 半径取「到焦点距离」的 4%，于是摆动的**角幅度与场景尺度无关**。
 *
 * ── 焦点固定 ──
 * 每帧 `lookAt` 光轴上的焦点，所以那个深度上的点**永远钉在画面中心**，
 * 只有它前后的层在相对移动。
 *
 * ── 和 OrbitControls / 回位动画的分工 ──
 * 摆动态下 `OrbitControls.enabled = false`（在 `SceneRig` 里切断），本组件全权驱动相机；
 * 否则两边都写 `camera.position` 会互相污染（OrbitControls 每帧从当前位置反推球坐标，
 * 会把摆动的偏移当成用户输入累积下去）。
 * 退出摆动时用阻尼把位置 / 朝向回放到进入前的位姿；但**用户一上手**
 * （拖 / 滚 / 重置视角）就调 `cancelSwayRestore()`，不再跟他抢镜头。
 *
 * ── 本组件不订阅任何 signal ──
 * 全部读 `.peek()`，每帧的活全在 `useFrame` 里，所以它**不参与 React 渲染** ——
 * 摆动开关变化不会引起任何组件重渲染。
 *
 * ── 按需渲染下的续帧义务 ──
 * 相机位姿是**命令式**写的，没有 React 提交，r3f 不会自己 `invalidate()`。
 * 所以「摆动中」与「回放中」都由本组件自己续帧；两者都停下时帧率为 0。
 * 启动的第一帧由 `InvalidateOnChange` 负责（它订阅了 `$sway`）。
 */

import { useFrame, useThree } from "@react-three/fiber"
import { Quaternion, Vector3 } from "three"
import { $scene, $sway } from "../store/viewer.ts"
import {
  capturePose,
  lookAtQuaternion,
  type Pose,
  referencePivot,
} from "./camera-pose.ts"
import type { LoadedScene } from "./types.ts"

/** 一圈摆动秒数。太慢看不出运动，太快会晕。 */
const SWAY_PERIOD_SECONDS = 18
const SWAY_OMEGA = (Math.PI * 2) / SWAY_PERIOD_SECONDS
/** 摆动半径 / 到焦点的距离。 */
const RADIUS_RATIO = 0.04
const RADIUS_MIN = 0.004
const RADIUS_MAX = 0.5
/** 竖直半轴 / 水平半轴：把回路压成扁椭圆，不让相机升太高。 */
const VERTICAL_RATIO = 0.35
/** 位姿阻尼速率（1/s）：进出摆动都是这条曲线。 */
const DAMP_RATE = 3.4

// 每帧用的临时对象（单画布，模块级复用即可，不制造 GC 压力）。
const targetPosition = new Vector3()
const pivot = new Vector3()
const aimQuaternion = new Quaternion()

/**
 * 摆动开始前的机位（退出摆动时回放给用户）。
 *
 * 放在**模块级**而不是 `useRef`：只有一个画布，而 `ReferenceViewReset` 与
 * `SceneRig` 需要能在别的触发源（回到参考视角 / 用户上手）处直接掐掉这次回放。
 */
let restorePose: Pose | null = null
/** 已经处理过的场景：换场景时丢弃旧机位。 */
let trackedScene: LoadedScene | null = null
/** 摆动相位的零点（开启摆动的时刻），保证 θ=0 落在参考位姿。 */
let swayEpoch = 0
/** 上一帧的摆动开关，用来检测「刚刚开启」这个边沿。 */
let wasSwaying = false

/** 取消「回到进入摆动前机位」的回放（用户开始操作 / 重置视角时调）。 */
export function cancelSwayRestore(): void {
  restorePose = null
}

export function ParallaxSway() {
  const camera = useThree((state) => state.camera)
  const invalidate = useThree((state) => state.invalidate)

  useFrame((state, delta) => {
    const loaded = $scene.peek()
    if (!loaded) return
    if (trackedScene !== loaded) {
      trackedScene = loaded
      restorePose = null
    }
    // 夹住 delta：切标签页回来时一步到位等于硬切。
    const step = 1 - Math.exp(-Math.min(delta, 0.1) * DAMP_RATE)
    const swaying = $sway.peek()
    // 开启的边沿：相位清零，于是回路从参考机位出发（入场语义精确）。
    if (swaying && !wasSwaying) swayEpoch = state.clock.elapsedTime
    wasSwaying = swaying

    if (swaying) {
      restorePose ??= capturePose(camera)
      const radius = swayRadius(loaded)
      const theta = (state.clock.elapsedTime - swayEpoch) * SWAY_OMEGA
      targetPosition.set(
        radius * Math.sin(theta),
        radius * VERTICAL_RATIO * (1 - Math.cos(theta)),
        // ★ 恒为 0：相机只在像平面里挪。一旦带 Z 分量就变成推拉（dolly）而非视差。
        0,
      )
      camera.position.lerp(targetPosition, step)
      aimAt(camera, loaded, step)
      invalidate()
      return
    }

    const pose = restorePose
    if (!pose) return
    if (damped(camera, pose, step)) {
      restorePose = null
      return
    }
    invalidate()
  })

  return null
}

/** 把相机朝向阻尼到「看向焦点」。 */
function aimAt(
  camera: { position: Vector3; quaternion: Quaternion },
  scene: LoadedScene,
  step: number,
): void {
  lookAtQuaternion(camera.position, referencePivot(scene, pivot), aimQuaternion)
  camera.quaternion.slerp(aimQuaternion, step)
}

/** 位置 + 朝向一起阻尼；返回是否已经到位（到位时吸附，消掉小数尾巴）。 */
function damped(
  camera: { position: Vector3; quaternion: Quaternion },
  pose: Pose,
  step: number,
): boolean {
  if (
    camera.position.distanceToSquared(pose.position) < 1e-8 &&
    Math.abs(camera.quaternion.dot(pose.quaternion)) > 0.999999
  ) {
    camera.position.copy(pose.position)
    camera.quaternion.copy(pose.quaternion)
    return true
  }
  camera.position.lerp(pose.position, step)
  camera.quaternion.slerp(pose.quaternion, step)
  return false
}

/** 摆动半径：随场景尺度缩放，再夹到肉眼可辨 / 不夸张的区间。 */
function swayRadius(scene: LoadedScene): number {
  const distance = Math.max(Math.abs(scene.bounds.center[2]), 1e-3)
  return Math.min(Math.max(distance * RADIUS_RATIO, RADIUS_MIN), RADIUS_MAX)
}
