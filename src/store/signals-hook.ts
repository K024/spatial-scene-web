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
      const subscribe = useCallback(
        (onStoreChange: () => void) => signal.subscribe(onStoreChange),
        [signal],
      )
      const getSnapshot = useCallback(() => signal.peek(), [signal])
      return useSyncExternalStore(subscribe, getSnapshot, getSnapshot)
    },
  })
}
