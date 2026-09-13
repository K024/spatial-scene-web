/**
 * 按需渲染（`frameloop="demand"`）的**失效源**。
 *
 * ── 为什么要这个东西 ──
 * 本应用 99% 的时间是**静止的**（照片贴在层上，没人动）。`frameloop="always"` 会让
 * GPU 以显示器刷新率空转，笔记本风扇 / 电量白白烧掉；`demand` 下 r3f 只在
 * `invalidate()` 之后渲一帧，静止时帧率为 0。
 *
 * ── 谁负责 invalidate（分工明确，别重复也别漏）──
 * 1. **React 驱动的场景图变更** —— r3f 通常自己管：reconciler 的 `commitUpdate` /
 *    `appendChild` / `removeChild` 会走 `invalidateInstance`，在 `internal.frames === 0`
 *    时 `invalidate()`（见 `@react-three/fiber` 源码）。
 *    ⚠ 但这条**对 drei 辅助元素不可靠**：`<GizmoHelper>` 经 `<Hud>` 注册了
 *    `renderPriority` 的 `useFrame`，主场景改由它自己画；开关提交时 `frames` 又常
 *    非 0，于是不会续帧。所以 `$showGrid` / `$showGizmo` **不在本文件列**，而是由
 *    `SceneRig`（`Grid` / `GizmoHelper` 的宿主）在**提交之后**的 `useEffect` 里
 *    显式 `invalidate()` —— 放这里会在 React 提交**之前**触发，可能画出旧状态。
 * 2. **drei 的交互** —— 也自己管：`OrbitControls` 在 `change` 事件里
 *    `invalidate()` + `performance.regress()`（后者喂给 `<AdaptiveDpr />`）。
 * 3. **命令式变更** —— 就是本组件存在的理由：直接改 `mesh.visible` /
 *    `material.opacity` / 相机位姿，**不会**产生任何 React 提交，r3f 无从知道。
 *    这里的信号覆盖了这类变更的**起点**；之后由各自的帧循环接力续帧
 *    （`ModelRoot` 的淡入淡出、`ParallaxSway` 的摆动 / 回放）。
 *    ⚠ **两个例外**：`$wireframe`（换材质必须与续帧同时发生，`ModelRoot` 自己订阅）
 *    与 `$resetViewToken`（`ReferenceViewReset` 自己订阅 + 驱动动画）。
 * 4. **DPR 回落后的回升** —— 单独一条：`regress()` 只把值写回 store，
 *    `setDpr` 也只改 `viewport.dpr`，**都不 invalidate**。真正的生效点在渲染循环里
 *    （首帧开头比 `gl.getPixelRatio()` 再 `setPixelRatio`）。所以交互结束、
 *    `AdaptiveDpr` 恢复满分辨率时没人续帧，画面会一直糊在低 DPR 上 ——
 *    见下面的 `performance.current` 订阅。
 *
 * ── 为什么用 `.subscribe()` 而不是 `useValue()` ──
 * 这里要的是「变了就续一帧」，不是「变了就重渲染」。用 signal 自带的订阅
 * （返回退订函数）表达得最直：组件**一次都不重渲染**，依赖数组也只剩 `invalidate`。
 * 这正是 store 选 signal 的理由之一 —— 非 React 侧可以直接订阅。
 */

import { useThree } from "@react-three/fiber"
import { useEffect } from "react"
import { $layers, $scene, $solo, $sway } from "../store/viewer.ts"

/** 命令式变更的信号：任一变化 → 续一帧。 */
const RENDER_INVALIDATORS = [$scene, $layers, $solo, $sway] as const

export function InvalidateOnChange() {
  const invalidate = useThree((state) => state.invalidate)
  // `performance` 每次 regress / 回升都换成新对象，所以这个选择器会正常触发重渲染。
  const dprScale = useThree((state) => state.performance.current)

  useEffect(() => {
    const disposers = RENDER_INVALIDATORS.map((signal) =>
      signal.subscribe(() => invalidate()),
    )
    return () => {
      for (const dispose of disposers) dispose()
    }
  }, [invalidate])

  // DPR 变化需要一帧才能落地（见文件头第 4 条）。
  useEffect(() => {
    // `dprScale` 只当「变了」的信号用，值本身不参与计算；显式读一下让依赖关系可读。
    void dprScale
    invalidate()
  }, [invalidate, dprScale])

  return null
}
