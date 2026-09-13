/**
 * 全屏浮层：空态 / 载入 / 失败 / 拖拽提示 / WebGL 降级。
 *
 * 五者互斥，且都以 `$phase` / `$dragActive` / `$webgl` 为唯一依据 ——
 * 浮层自身**不持有状态**，所以永远不会和 store 打架。
 */

import { AnimatePresence, motion } from "motion/react"
import type { ReactNode } from "react"
import {
  $dragActive,
  $error,
  $phase,
  $progress,
  $source,
  $webgl,
} from "../../store/viewer.ts"
import { loadSampleGlb, pickGlbFile } from "./actions.ts"
import { GLASS, TextButton } from "./bits.tsx"
import { IconSparkles, IconUpload } from "./icons.tsx"

export function Overlays() {
  const phase = $phase.useValue()
  const dragActive = $dragActive.useValue()
  const supported = $webgl.useValue()

  let content: ReactNode = null
  let key = "none"
  if (!supported) {
    key = "unsupported"
    content = (
      <Card title="浏览器不支持 WebGL2">
        <p className="text-[12px] text-white/60">
          本渲染端依赖 WebGL2。请改用较新版本的 Chrome / Edge / Firefox。
        </p>
      </Card>
    )
  } else if (phase === "empty" && !dragActive) {
    key = "empty"
    content = <EmptyState />
  } else if (phase === "loading") {
    key = "loading"
    content = <LoadingCard />
  } else if (phase === "error") {
    key = "error"
    content = <ErrorCard />
  }

  return (
    <>
      <AnimatePresence mode="wait">
        {content ? (
          <motion.div
            key={key}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.18 }}
            className="pointer-events-none absolute inset-0 z-30 flex items-center justify-center p-6"
          >
            {content}
          </motion.div>
        ) : null}
      </AnimatePresence>
      <AnimatePresence>
        {dragActive ? <DragHint key="drag" /> : null}
      </AnimatePresence>
    </>
  )
}

function Card({
  title,
  icon,
  children,
}: {
  title: string
  icon?: ReactNode
  children: ReactNode
}) {
  return (
    <div
      className={`pointer-events-auto w-full max-w-sm rounded-3xl px-6 py-5 text-center ${GLASS}`}
    >
      <div className="mb-2 flex items-center justify-center gap-2">
        {icon}
        <h2 className="text-[15px] font-semibold text-white/90">{title}</h2>
      </div>
      {children}
    </div>
  )
}

function EmptyState() {
  return (
    <Card
      title="拖入一个 GLB"
      icon={<IconSparkles className="size-4 text-white/55" />}
    >
      <p className="text-[12px] leading-relaxed text-white/55">
        生成端产出的是
        <strong className="font-semibold text-white/75">多层 mesh 场景</strong>
        （L 层 + 背衬平面）。把{" "}
        <code className="rounded bg-white/10 px-1 py-0.5 font-mono text-[11px]">
          .glb
        </code>{" "}
        丢到窗口任意位置，或：
      </p>
      <div className="mt-3.5 flex items-center justify-center gap-2">
        <TextButton onClick={pickGlbFile}>
          <IconUpload className="mr-1 inline size-3.5 align-[-2px]" />
          打开 GLB
        </TextButton>
        <TextButton onClick={loadSampleGlb}>载入内置样例</TextButton>
      </div>
      <p className="mt-3 font-mono text-[10px] text-white/25">
        ?glb=&lt;url&gt; 也可直接指定
      </p>
    </Card>
  )
}

function LoadingCard() {
  const progress = $progress.useValue()
  const source = $source.useValue()

  return (
    <Card title="载入场景">
      <p className="truncate text-[11.5px] text-white/50">
        {source.name || source.url}
      </p>
      <div className="mt-3.5 h-1 w-full overflow-hidden rounded-full bg-white/12">
        <div
          className={
            progress === null
              ? "h-full w-1/3 animate-pulse rounded-full bg-white/60"
              : "h-full rounded-full bg-white/70 transition-[width] duration-150"
          }
          style={
            progress === null ? undefined : { width: `${progress * 100}%` }
          }
        />
      </div>
      <p className="mt-2 font-mono text-[10.5px] text-white/35">
        {progress === null ? "读取中…" : `${Math.round(progress * 100)}%`}
      </p>
    </Card>
  )
}

function ErrorCard() {
  const error = $error.useValue()
  return (
    <Card title="载入失败">
      <p className="rounded-xl border border-red-400/25 bg-red-500/12 px-3 py-2 text-left text-[11.5px] break-all text-red-200/90">
        {error ?? "未知错误"}
      </p>
      <div className="mt-3.5 flex items-center justify-center gap-2">
        <TextButton onClick={pickGlbFile}>换一个文件</TextButton>
        <TextButton onClick={loadSampleGlb}>载入内置样例</TextButton>
      </div>
    </Card>
  )
}

function DragHint() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.15 }}
      className="pointer-events-none absolute inset-0 z-40 flex items-center justify-center bg-[#05060c]/55 backdrop-blur-sm"
    >
      <div className="rounded-3xl border-2 border-dashed border-white/35 px-12 py-10 text-center">
        <IconUpload className="mx-auto size-7 text-white/70" />
        <p className="mt-3 text-[14px] font-medium text-white/85">
          松手载入 .glb
        </p>
      </div>
    </motion.div>
  )
}
