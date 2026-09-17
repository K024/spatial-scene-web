/**
 * 把 `useValue()` 装到 Signal 原型上。
 *
 * React Compiler 已启用，但**没有**启用 signals 的 transform，所以读取信号必须显式
 * 走 `useValue()`。把它装在原型上（而不是每处 `useSignals()`）是为了让「读信号」
 * 的写法与普通取值一致：`const x = someSignal.useValue()`。
 *
 * 注意 `useSyncExternalStore` 的订阅/快照回调都用 `useCallback` 固定，
 * 依赖只有 signal 本身；`peek()` 不建立响应式依赖，只在快照时取值。
 */

import { Signal } from "@preact/signals-react"
import { useCallback, useSyncExternalStore } from "react"

declare module "@preact/signals-core" {
  interface Signal<T = any> {
    useValue(): T
  }
  interface ReadonlySignal<T = any> {
    useValue(): T
  }
}

if (!Object.hasOwn(Signal.prototype, "useValue")) {
  Object.defineProperty(Signal.prototype, "useValue", {
    configurable: true,
    writable: true,
    value: function useValue<T>(this: Signal<T>): T {
      const signal = this
      // biome-ignore lint/correctness/useExhaustiveDependencies: this
      const subscribe = useCallback(
        (onStoreChange: () => void) => signal.subscribe(onStoreChange),
        [signal],
      )
      // biome-ignore lint/correctness/useExhaustiveDependencies: this
      const getSnapshot = useCallback(() => signal.peek(), [signal])
      return useSyncExternalStore(subscribe, getSnapshot, getSnapshot)
    },
  })
}
