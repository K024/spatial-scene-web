/**
 * 加载浮层：GLB 下载 / 解析期间盖一层，避免「黑屏无反馈」。
 *
 * 失败态由面板里展示（这里只负责「在加载」这一件事）。
 */

import { AnimatePresence, motion } from "motion/react"
import { glbSource, glbStatus } from "../store/index.ts"

export function LoadOverlay() {
  const status = glbStatus.useValue()
  const source = glbSource.useValue()
  const visible = status === "loading"

  return (
    <AnimatePresence>
      {visible ? (
        <motion.div
          key="overlay"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2, ease: "easeOut" }}
          className="absolute inset-0 z-30 flex items-center justify-center p-6"
        >
          <div className="absolute inset-0 bg-black/40 backdrop-blur-[2px]" />
          <motion.div
            initial={{ opacity: 0, y: 10, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 8, scale: 0.985 }}
            transition={{ duration: 0.24, ease: [0.16, 1, 0.3, 1] }}
            className="glass relative flex w-[360px] items-center gap-3 rounded-2xl px-5 py-4"
          >
            <span className="h-4 w-4 shrink-0 animate-spin rounded-full border-2 border-white/15 border-t-accent-400" />
            <div className="min-w-0 leading-tight">
              <div className="text-[13px] font-medium text-white/90">
                正在加载 GLB
              </div>
              <div className="tnum mt-0.5 truncate text-[11.5px] text-white/45">
                {source.kind === "file" ? source.label : source.url}
              </div>
            </div>
          </motion.div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  )
}
