import { StrictMode } from "react"
import { createRoot } from "react-dom/client"

import "./index.css"
import App from "./App.tsx"
import { startLoad } from "./store/scene.ts"
import { pushSplats } from "./store/viewer.ts"

const root = document.getElementById("root")
if (!root) throw new Error("找不到 #root")

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
)

// 挂载后再启动加载：PLY 有几十 MB，让首帧 UI（含进度浮层）先出来
void startLoad(undefined, pushSplats)
