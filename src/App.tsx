/**
 * 应用根：**整页 canvas 是主体**，HUD / 面板 / 浮层都是浮在它上面的 DOM。
 *
 * ── 层级 ──
 * ```
 * z-0   <Viewport/>              r3f canvas（铺满视口）
 * z-20  <Hud/> <FloatingPanel/>  常驻控件（面板可收起）
 * z-30  <Overlays/>              空态 / 载入 / 失败 / WebGL 降级
 * z-40  拖拽提示
 * ```
 * 浮层容器一律 `pointer-events-none` + 交互子元素 `pointer-events-auto`，
 * 保证画布的轨道 / 缩放不被透明区域吃掉。
 *
 * ── 拖拽落在 window 上 ──
 * 拖放目标是整个窗口（含 canvas），所以监听 `window` 的 drag 事件，
 * 用「进入计数」抵消 `dragenter`/`dragleave` 在子元素间反复触发。
 */

import { useEffect } from "react"
import { $dragActive, $webgl, loadGlbFile } from "./store/viewer.ts"
import { FloatingPanel } from "./viewer/ui/FloatingPanel.tsx"
import { Hud } from "./viewer/ui/Hud.tsx"
import { Overlays } from "./viewer/ui/Overlays.tsx"
import { Viewport } from "./viewer/Viewport.tsx"

function App() {
  const supported = $webgl.useValue()
  useWindowDrop()

  return (
    <div className="relative h-dvh w-full overflow-hidden bg-[#070810] text-white antialiased">
      {supported ? <Viewport /> : null}
      <Hud />
      <FloatingPanel />
      <Overlays />
    </div>
  )
}

/** 窗口级拖放：丢一个 `.glb` 进来即载入。 */
function useWindowDrop() {
  useEffect(() => {
    let depth = 0

    const hasFiles = (event: DragEvent): boolean =>
      Array.from(event.dataTransfer?.types ?? []).includes("Files")

    const onDragEnter = (event: DragEvent) => {
      if (!hasFiles(event)) return
      event.preventDefault()
      depth += 1
      $dragActive.value = true
    }
    const onDragOver = (event: DragEvent) => {
      if (!hasFiles(event)) return
      // 不 preventDefault，浏览器会去「打开文件」，drop 也不会触发。
      event.preventDefault()
      if (event.dataTransfer) event.dataTransfer.dropEffect = "copy"
    }
    const onDragLeave = (event: DragEvent) => {
      if (!hasFiles(event)) return
      depth = Math.max(0, depth - 1)
      if (depth === 0) $dragActive.value = false
    }
    const onDrop = (event: DragEvent) => {
      if (!hasFiles(event)) return
      event.preventDefault()
      depth = 0
      $dragActive.value = false
      const file = event.dataTransfer?.files?.[0]
      if (file) void loadGlbFile(file)
    }

    window.addEventListener("dragenter", onDragEnter)
    window.addEventListener("dragover", onDragOver)
    window.addEventListener("dragleave", onDragLeave)
    window.addEventListener("drop", onDrop)
    return () => {
      window.removeEventListener("dragenter", onDragEnter)
      window.removeEventListener("dragover", onDragOver)
      window.removeEventListener("dragleave", onDragLeave)
      window.removeEventListener("drop", onDrop)
    }
  }, [])
}

export default App
