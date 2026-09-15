/**
 * 应用外壳：整页编辑器布局。
 *
 *   ┌────────────────────────────────────────────┬──────────┐
 *   │  全屏 WebGL2 画布（拖拽/滚轮交互）          │  玻璃面板 │
 *   │  左上：场景信息 chip                        │  （浮层） │
 *   │  左下：操作提示                             │          │
 *   └────────────────────────────────────────────┴──────────┘
 *
 * 画布不参与布局计算（`absolute inset-0`），面板是浮在上面的
 * `absolute` 卡片，所以「铺满全屏」与「面板不遮挡取景」同时成立。
 * 快捷键：R 复位视角 / P 暂停渲染 / O 切换参考图比对。
 */

import { AnimatePresence, motion } from "motion/react"
import { useEffect, useRef, useState } from "react"
import { cameraMode, cycleCameraMode, resetCamera } from "./store/camera.ts"
import { meta, startLoadFromFile, status } from "./store/scene.ts"
import { overlayMode, panelOpen, paused, togglePaused } from "./store/viewer.ts"
import { CanvasStage } from "./ui/CanvasStage.tsx"
import { fmtInt } from "./ui/format.ts"
import { InfoPanel } from "./ui/InfoPanel.tsx"
import { LoadOverlay } from "./ui/LoadOverlay.tsx"

/** 比对模式的循环顺序。 */
const OVERLAY_CYCLE = ["off", "overlay", "blink"] as const

export default function App() {
  const open = panelOpen.useValue()
  const loadStatus = status.useValue()
  const sceneMeta = meta.useValue()
  const isPaused = paused.useValue()
  const mode = cameraMode.useValue()

  // ── 拖拽放置 ──
  // `dragenter/leave` 会在子元素之间反复触发，所以用深度计数而不是布尔值，
  // 否则鼠标从画布移到提示卡片上，提示就会闪一下。
  const [dropping, setDropping] = useState(false)
  const [dropType, setDropType] = useState<string | null>(null)
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
    // 拖拽阶段拿不到文件名（.ply 的 MIME 通常为空），只能拿个大概
    const item = e.dataTransfer.items?.[0]
    if (item?.kind === "file" && item.type) setDropType(item.type)
  }

  function onDragLeave() {
    dragDepth.current = Math.max(0, dragDepth.current - 1)
    if (dragDepth.current === 0) {
      setDropping(false)
      setDropType(null)
    }
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault()
    dragDepth.current = 0
    setDropping(false)
    setDropType(null)
    const file = e.dataTransfer?.files?.[0]
    if (file) void startLoadFromFile(file)
  }

  // 全局快捷键（输入框聚焦时不拦截）
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const target = e.target as HTMLElement | null
      if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return
      if (e.metaKey || e.ctrlKey || e.altKey) return
      switch (e.key.toLowerCase()) {
        case "r":
          resetCamera()
          break
        case "p":
          togglePaused()
          break
        case "o": {
          const i = OVERLAY_CYCLE.indexOf(overlayMode.peek())
          overlayMode.value = OVERLAY_CYCLE[(i + 1) % OVERLAY_CYCLE.length]
          break
        }
        case "m":
          cycleCameraMode()
          break
        case "h":
          panelOpen.value = !panelOpen.peek()
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
      <CanvasStage />

      {/* ── 左上：场景 chip ── */}
      <div className="pointer-events-none absolute top-4 left-4 z-20 flex items-center gap-2">
        <div className="glass-soft flex items-center gap-2.5 rounded-xl px-3 py-2">
          <span className="h-1.5 w-1.5 rounded-full bg-accent-400 shadow-[0_0_8px_1px_rgba(34,211,238,0.6)]" />
          <span className="text-[12px] font-medium tracking-tight text-white/85">
            spatial-scene-web
          </span>
          {sceneMeta ? (
            <>
              <span className="h-3 w-px bg-white/12" />
              <span className="tnum text-[11.5px] text-white/50">
                {fmtInt(sceneMeta.count)} splats
              </span>
            </>
          ) : null}
          {loadStatus === "ready" ? (
            <>
              <span className="h-3 w-px bg-white/12" />
              <span className="text-[11px] text-emerald-300/80">
                {mode === "free"
                  ? "单次排序 · 固定顺序"
                  : mode === "parallax"
                    ? "视差模式"
                    : "自动视差"}
              </span>
            </>
          ) : null}
        </div>
      </div>

      {/* ── 左下：操作提示（跟模式走，避免提示一个该模式下无效的操作）── */}
      <div className="pointer-events-none absolute bottom-4 left-4 z-20">
        <div className="glass-soft rounded-xl px-3 py-2 text-[11px] text-white/45">
          {mode === "free" ? (
            <>
              <span className="text-white/70">拖拽</span> 旋转 ·{" "}
              <span className="text-white/70">滚轮</span> 缩放 ·{" "}
              <span className="text-white/70">双击 / R</span> 回到参考视角
            </>
          ) : mode === "parallax" ? (
            <>
              <span className="text-white/70">移动鼠标</span> 产生视差（居中 =
              参考视角） · <span className="text-white/70">滚轮</span> 缩放
            </>
          ) : (
            <>
              <span className="text-white/70">自动视差中</span> ·{" "}
              <span className="text-white/70">滚轮</span> 缩放 · 参数在右侧面板
            </>
          )}
          {" · "}
          <span className="text-white/70">M</span> 切换相机模式 ·{" "}
          <span className="text-white/70">O</span> 对比参考图
        </div>
      </div>

      {/* ── 右上：暂停指示（渲染循环关掉时提示，避免误以为卡死）── */}
      {isPaused ? (
        <div className="pointer-events-none absolute top-4 right-4 z-30">
          <div className="glass-soft rounded-xl px-3 py-2 text-[11px] text-amber-200/90">
            已暂停渲染 · <span className="tnum">P</span> 恢复
          </div>
        </div>
      ) : null}

      {/* ── 右侧浮层面板 ── */}
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
            <InfoPanel />
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

      {/* ── 拖拽放置提示（盖在最上层，不抢指针事件）── */}
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
            <motion.div
              initial={{ scale: 0.98, y: 8 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.99, y: 4 }}
              transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
              className="glass relative flex flex-col items-center gap-2 rounded-2xl border-2 border-dashed border-accent-400/45 px-10 py-8 text-center"
            >
              <span className="text-[15px] font-medium text-white/90">
                松开以载入 PLY
              </span>
              <span className="max-w-[286px] text-[11.5px] leading-relaxed text-white/50">
                支持 3DGS PLY。带内参（--full）就用 PLY 里的；没有相机信息时按
                没有相机信息时按默认 30 mm 等效焦距反推，参考图比对自动关闭。
              </span>
              {dropType ? (
                <span className="tnum text-[11px] text-white/30">
                  {dropType}
                </span>
              ) : null}
            </motion.div>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </div>
  )
}
