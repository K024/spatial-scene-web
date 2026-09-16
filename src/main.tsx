import { StrictMode } from "react"
import { createRoot } from "react-dom/client"

import "./index.css"
import App from "./App.tsx"
import { loadGlb } from "./store/index.ts"

const root = document.getElementById("root")
if (!root) throw new Error("找不到 #root")

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
)

// 挂载后再启动：让首帧 UI（含加载浮层）先出来，再下载 GLB。
// 查看器只消费 GLB —— 产出产物用 `npm run export-glb`。
void loadGlb()
