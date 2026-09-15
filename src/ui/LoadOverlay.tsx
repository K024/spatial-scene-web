/**
 * 加载/错误 浮层。
 *
 * 进度阶段列表由 `STAGE_LABELS` 驱动：当前阶段高亮、之前的阶段标为完成。
 * 进度条与转圈用 FlyonUI 的 `progress` / `loading-spinner`
 * （纯 CSS、主题色可覆盖），其余排版仍是本项目自己的玻璃语言。
 */

import { AnimatePresence, motion } from "motion/react"
import { type LoadStage, STAGE_LABELS } from "../spatial-scene/render/types.ts"
import { error, progress, startLoad, status } from "../store/scene.ts"
import { Button } from "./primitives.tsx"

/** 阶段顺序（用于把「已完成的阶段」标出来）。 */
const STAGES: LoadStage[] = ["fetch", "parse", "convert", "sort", "pack"]

export function LoadOverlay() {
  const loadStatus = status.useValue()
  const p = progress.useValue()
  const err = error.useValue()

  const visible = loadStatus === "loading" || loadStatus === "error"

  return (
    <AnimatePresence>
      {visible ? (
        <motion.div
          key="overlay"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.22, ease: "easeOut" }}
          className="absolute inset-0 z-30 flex items-center justify-center p-6"
        >
          {/* 压暗背景，让玻璃卡片浮起来 */}
          <div className="absolute inset-0 bg-black/45 backdrop-blur-[2px]" />

          <motion.div
            initial={{ opacity: 0, y: 12, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 8, scale: 0.985 }}
            transition={{ duration: 0.26, ease: [0.16, 1, 0.3, 1] }}
            className="glass relative w-[400px] overflow-hidden rounded-2xl px-5 py-4"
          >
            {loadStatus === "error" ? (
              <>
                <div className="flex items-center gap-2.5">
                  <span className="flex h-6 w-6 items-center justify-center rounded-full border border-red-400/30 bg-red-500/15 text-[12px] text-red-200">
                    !
                  </span>
                  <h2 className="text-[13.5px] font-medium text-white/90">
                    场景加载失败
                  </h2>
                </div>
                <p className="mt-2.5 text-[12px] leading-relaxed break-words text-red-100/80">
                  {err ?? "未知错误"}
                </p>
                <p className="mt-2 text-[11.5px] leading-relaxed text-white/40">
                  最常见的原因：还没生成 PLY。先跑一次{" "}
                  <code className="tnum rounded bg-white/8 px-1 py-0.5">
                    npm run sample
                  </code>
                  ，产物会落在{" "}
                  <code className="tnum rounded bg-white/8 px-1 py-0.5">
                    public/exports/sample.ply
                  </code>
                  。
                </p>
                <div className="mt-3.5 flex justify-end">
                  <Button
                    variant="primary"
                    onClick={() => void startLoad()}
                    disabled={false}
                  >
                    重试
                  </Button>
                </div>
              </>
            ) : (
              <>
                <div className="flex items-center gap-2.5">
                  <span className="loading loading-spinner loading-sm text-accent-400" />
                  <h2 className="text-[13.5px] font-medium text-white/90">
                    正在准备场景
                  </h2>
                  <span className="tnum ml-auto text-[12px] text-white/55">
                    {Math.round(p.ratio * 100)}%
                  </span>
                </div>

                {/* 阶段列表 */}
                <ul className="mt-3.5 flex flex-col gap-1.5">
                  {STAGES.map((stage) => {
                    const active = p.stage === stage
                    const done =
                      STAGES.indexOf(stage) < STAGES.indexOf(p.stage) ||
                      p.stage === "done"
                    return (
                      <li
                        key={stage}
                        className="flex items-center gap-2 text-[12px]"
                      >
                        <span
                          className={`h-1.5 w-1.5 rounded-full transition-colors ${
                            done
                              ? "bg-emerald-400"
                              : active
                                ? "bg-accent-400"
                                : "bg-white/15"
                          }`}
                        />
                        <span
                          className={
                            active
                              ? "text-white/90"
                              : done
                                ? "text-white/45"
                                : "text-white/28"
                          }
                        >
                          {STAGE_LABELS[stage]}
                        </span>
                        {active && p.detail ? (
                          <span className="tnum ml-auto text-[11px] text-white/45">
                            {p.detail}
                          </span>
                        ) : null}
                      </li>
                    )
                  })}
                </ul>

                {/* FlyonUI 进度条 */}
                <div className="progress mt-3 h-1.5">
                  <div
                    className="progress-bar bg-accent-500"
                    style={{ width: `${Math.max(3, p.ratio * 100)}%` }}
                  />
                </div>
                {p.stage === "done" && p.detail ? (
                  <p className="mt-2 text-[11px] text-white/40">{p.detail}</p>
                ) : null}
              </>
            )}
          </motion.div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  )
}
