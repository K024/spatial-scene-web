/**
 * store 统一出口。
 *
 * `import "./signals-hook.ts"` 是**必要的副作用**：它把 `useValue()` 装到
 * `Signal.prototype` 上，是本项目「全局信号 + React 响应式」的粘合点
 * （React Compiler 已启用，但**没有**启用 signals 的 transform，所以读取信号
 * 必须显式走 `useValue()`）。
 */

import "./signals-hook.ts"

export * from "./camera.ts"
export * from "./glb.ts"
export * from "./stats.ts"
export * from "./viewer.ts"
