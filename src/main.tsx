/**
 * 应用入口。
 *
 * 两件事必须在 `createRoot` **之前**做完，顺序不能换：
 * 1. `./store/signal-hooks.ts` —— 装上 `Signal.prototype.useValue`，任何组件求值前就位；
 * 2. `detectWebgl2()` + `bootstrap()` —— 前者决定挂不挂画布（必须在 r3f 渲染前），
 *    后者发起首屏载入（命令式，见 `store/viewer.ts`），早发一帧就少一帧空态。
 */

import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import "./index.css"
import "./store/signal-hooks.ts"
import App from "./App.tsx"
import { $webgl, bootstrap } from "./store/viewer.ts"
import { detectWebgl2 } from "./viewer/webgl.ts"

$webgl.value = detectWebgl2()
bootstrap()

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
