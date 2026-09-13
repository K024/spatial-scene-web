/**
 * 全项目**唯一**的 signal 读取方式：`Signal.prototype.useValue()`。
 *
 * ── 为什么不用 `@preact/signals-react-transform`（babel transform 方案）──
 * 本项目启用了 React Compiler（`babel-plugin-react-compiler`）。transform 方案会往每个
 * 组件里注入 `useSignals()` 并重写 `sig.value` 读取，而编译器**看不见**注入的订阅 ——
 * 它会把订阅结果当成普通表达式缓存进 `_c()` 槽位，signal 变了组件不重渲染。
 * 所以这里改成**显式订阅**：`useValue()` 内部直接 `useSyncExternalStore`，
 * React 自己就是订阅方，编译器只需（也只会）保守处理这一次调用。
 *
 * ── signal 的命名：`$` 前缀，且**刻意不是 hook** ──
 * signal 是**数据作用域**，不是 hook：`$layers` / `$phase` / `$panelOpen` …… 用 `$`
 * 前缀把这件事写在名字上。这同时满足 React Compiler 的 hook 识别规则
 * （`babel-plugin-react-compiler` 的 `isHook`）：
 *
 * ```
 * MemberExpression 的属性名匹配 /^use[A-Z0-9]/ 且 其 object 是 /^[A-Z]/ 的标识符
 * ```
 *
 * `$layers.useValue()` 的 object 是 `$layers`（非大写开头）⇒ **不是** hook ⇒
 * 编译器无法证明这次调用是纯的 / 可缓存的，于是**整块组件放弃 memo 化**。
 * 实测 `babel-plugin-react-compiler@1.0.0`：这类组件不产出 `_c()`，`useValue()` 调用
 * 每渲染原样执行 —— 这正是我们要的方向（宁可不优化，也不能缓存订阅读取）。
 * 反例是把 signal 命名成 `Layers`：那样会被认成 hook，编译器会开始 memo 化周围代码，
 * 等于把「signal 是纯数据」这个错误假设固化进产物。
 *
 * ── 调用约定（硬要求）──
 * `useValue()` 内部是 hook，所以**必须在组件函数体顶层无条件调用**
 * （不能放进 `if` / 循环 / 事件回调）。它返回的是一个**快照值**，
 * 不是响应式引用：不要在 `useFrame` / 事件回调里读它，那里应该用 `.peek()`。
 *
 * ── 语义 ──
 * - `getSnapshot = () => this.peek()`：**用 `peek()` 而不是 `.value` getter**。
 *   `.value` 是「带订阅语义的读」：在 tracking 上下文（effect / computed / 渲染期的
 *   `useSignals` 运行时）里读它会给当前计算挂依赖，而 `getSnapshot` 只应该在
 *   **订阅之外**偷看一眼当前值 —— 订阅由 `subscribe` 那一侧负责。`peek()` 正是这个语义。
 * - signal 只在 `.value =` 时换引用，所以同一渲染内多次取快照恒等，
 *   不会触发 `useSyncExternalStore` 的无限循环；
 * - `subscribe` 直接用 signal 自带的同步订阅（返回退订函数），
 *   正好是 `useSyncExternalStore` 要的形状；
 * - ⚠ **两个回调都必须 `useCallback` 稳定住**：`useSyncExternalStore` 比较
 *   `subscribe` / `getSnapshot` 的**引用**，每渲染换新函数的话它会在每次 commit 后
 *   退订再重订（订阅抖动），并且在 React 认为快照变化的边界上多触发一次同步。
 *   依赖只有 `this`（signal 实例）——它在一个组件的生命周期里不变，所以订阅只建一次。
 * - `Computed` 继承自 `Signal`，所以补 `Signal.prototype` 就同时覆盖了 computed；
 *   类型侧再给只读接口 `ReadonlySignal` 补一条声明。
 */

import { Signal } from "@preact/signals-react"
import { useCallback, useSyncExternalStore } from "react"

declare module "@preact/signals-core" {
  interface Signal<T = any> {
    /** 订阅读取（`useSyncExternalStore` 实现，见文件头）。必须在组件顶层调用。 */
    useValue(): T
  }
  interface ReadonlySignal<T = any> {
    /** 订阅读取（`useSyncExternalStore` 实现，见文件头）。必须在组件顶层调用。 */
    useValue(): T
  }
}

// 幂等：模块被多次 import 也只定义一次。
if (!Object.hasOwn(Signal.prototype, "useValue")) {
  Object.defineProperty(Signal.prototype, "useValue", {
    configurable: true,
    writable: true,
    value: function useValue<T>(this: Signal<T>): T {
      // 先落到局部常量：`this` 作为依赖项 linter / 编译器都看不清楚，
      // 而 `signal` 是普通变量，依赖意图一目了然（同一个 slot 换成别的 signal 时要重建订阅）。
      const signal = this
      // biome-ignore lint/correctness/useExhaustiveDependencies: 依赖就是 signal 实例；规则把它当「非响应式变量」误判为多余，但同一个 hook slot 换成别的 signal 时订阅必须重建。
      const subscribe = useCallback(
        (onStoreChange: () => void) => signal.subscribe(onStoreChange),
        [signal],
      )
      // biome-ignore lint/correctness/useExhaustiveDependencies: 同上。
      const getSnapshot = useCallback(() => signal.peek(), [signal])
      return useSyncExternalStore(subscribe, getSnapshot, getSnapshot)
    },
  })
}
