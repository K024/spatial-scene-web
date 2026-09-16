/**
 * 应用外壳：全窗口 three 视图 + 右侧浮动玻璃面板。
 *
 *   ┌───────────────────────────────────────────┬──────────┐
 *   │  全屏 three 画布（GLB 分层网格 + grid + gizmo）│  玻璃面板 │
 *   └───────────────────────────────────────────┴──────────┘
 *
 * 自定义产物只能通过**文件选择 / 拖入**加载（默认产物是固定路径，与 export
 * 脚本对齐）。快捷键：M 切换相机模式 · R 回参考视角 · H 收起面板。
 */

import { AnimatePresence, motion } from "motion/react"
import { useEffect, useRef, useState } from "react"
import {
  cycleCameraMode,
  glbMeta,
  glbSource,
  glbStatus,
  loadGlbFromFile,
  panelOpen,
  resetCamera,
} from "./store/index.ts"
import { LoadOverlay } from "./ui/LoadOverlay.tsx"
import { Panel } from "./ui/Panel.tsx"
import { Viewer3D } from "./ui/Viewer3D.tsx"

export default function App() {
  const open = panelOpen.useValue()
  const status = glbStatus.useValue()
  const meta = glbMeta.useValue()
  const source = glbSource.useValue()

  // `dragenter/leave` 会在子元素之间反复触发，所以用深度计数而不是布尔值。
  const [dropping, setDropping] = useState(false)
  const dragDepth = useRef(0)

  function onDragEnter(e: React.DragEvent) {
    if (!e.dataTransfer?.types.includes("Files")) return
    e.preventDefault()
    dragDepth.current += 1
    setDropping(true)
  }
  function onDragOver(e: React.DragEvent) {
    if (!e.dataTransfer?.types.includes("Files")) return
    // 不 preventDefault 的话浏览器会直接导航到文件（整页跳走）
    e.preventDefault()
    e.dataTransfer.dropEffect = "copy"
  }
  function onDragLeave() {
    dragDepth.current = Math.max(0, dragDepth.current - 1)
    if (dragDepth.current === 0) setDropping(false)
  }
  function onDrop(e: React.DragEvent) {
    e.preventDefault()
    dragDepth.current = 0
    setDropping(false)
    const file = e.dataTransfer?.files?.[0]
    if (file) void loadGlbFromFile(file)
  }

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const target = e.target as HTMLElement | null
      if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return
      if (e.metaKey || e.ctrlKey || e.altKey) return
      switch (e.key.toLowerCase()) {
        case "h":
          panelOpen.value = !panelOpen.peek()
          break
        case "r":
          resetCamera()
          break
        case "m":
          cycleCameraMode()
          break
        default:
          break
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [])

  return (
    <div
      className="relative h-full w-full overflow-hidden bg-[#05060a] text-white"
      onDragEnter={onDragEnter}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      <Viewer3D />

      {/* 左上：场景 chip */}
      <div className="pointer-events-none absolute top-4 left-4 z-20">
        <div className="glass-soft flex items-center gap-2.5 rounded-xl px-3 py-2">
          <span className="h-1.5 w-1.5 rounded-full bg-accent-400 shadow-[0_0_8px_1px_rgba(34,211,238,0.6)]" />
          <span className="text-[12px] font-medium tracking-tight text-white/85">
            spatial-scene · layered GLB
          </span>
          {meta ? (
            <>
              <span className="h-3 w-px bg-white/12" />
              <span className="tnum text-[11.5px] text-white/50">
                {meta.layers.length} 层 ·{" "}
                {meta.triangleCount.toLocaleString("en-US")} 面
              </span>
            </>
          ) : null}
          {status === "ready" ? (
            <>
              <span className="h-3 w-px bg-white/12" />
              <span className="max-w-[220px] truncate text-[11px] text-emerald-300/80">
                {source.kind === "file" ? `本地 ${source.label}` : "默认产物"}
              </span>
            </>
          ) : null}
        </div>
      </div>

      <AnimatePresence>
        {open ? (
          <motion.div
            key="panel"
            initial={{ opacity: 0, x: 18 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 18 }}
            transition={{ duration: 0.24, ease: [0.16, 1, 0.3, 1] }}
            className="pointer-events-none absolute inset-0 z-20"
          >
            <Panel />
          </motion.div>
        ) : (
          <motion.button
            key="reopen"
            type="button"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            transition={{ duration: 0.18 }}
            onClick={() => {
              panelOpen.value = true
            }}
            className="glass focus-ring absolute top-4 right-4 z-20 rounded-xl px-3 py-2 text-[12px] text-white/70 hover:text-white"
          >
            显示面板
          </motion.button>
        )}
      </AnimatePresence>

      <LoadOverlay />

      {/* 拖拽放置提示 */}
      <AnimatePresence>
        {dropping ? (
          <motion.div
            key="drop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.16 }}
            className="pointer-events-none absolute inset-0 z-40 flex items-center justify-center p-6"
          >
            <div className="absolute inset-0 bg-black/50 backdrop-blur-[3px]" />
            <div className="glass relative flex flex-col items-center gap-2 rounded-2xl border-2 border-dashed border-accent-400/45 px-10 py-8 text-center">
              <span className="text-[15px] font-medium text-white/90">
                松开以载入 GLB
              </span>
              <span className="max-w-[300px] text-[11.5px] leading-relaxed text-white/50">
                支持 npm run export-glb 产出的分层 GLB（相机与层元数据内置）。
              </span>
            </div>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </div>
  )
}
