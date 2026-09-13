/**
 * 顶部左侧 HUD（常驻小条）。
 *
 * 只放**最高频**的三个动作 + 状态：面板开关、回参考视角、载入相位。
 * 其余全部收进右侧面板 —— canvas 是主体，HUD 不抢注意力。
 */

import {
  $layers,
  $panelOpen,
  $phase,
  $scene,
  $visibleCount,
  requestResetView,
} from "../../store/viewer.ts"
import { GLASS, IconButton } from "./bits.tsx"
import { IconSliders, IconTarget } from "./icons.tsx"

export function Hud() {
  const open = $panelOpen.useValue()
  const phase = $phase.useValue()
  const scene = $scene.useValue()
  const visible = $visibleCount.useValue()
  const layers = $layers.useValue()

  return (
    <div className="pointer-events-none absolute top-3 left-3 z-20 flex items-center gap-2">
      <div
        className={`pointer-events-auto flex items-center gap-2.5 rounded-2xl py-1.5 pr-1.5 pl-3.5 ${GLASS}`}
      >
        <div className="flex flex-col">
          <span className="text-[12.5px] leading-tight font-semibold tracking-wide text-white/90">
            Spatial $scene
          </span>
          <span className="text-[10.5px] leading-tight text-white/40">
            {statusText(phase, layers.length, visible)}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <IconButton
            label="回到参考视角"
            size="xs"
            disabled={!scene}
            onClick={requestResetView}
          >
            <IconTarget className="size-3.5" />
          </IconButton>
          <IconButton
            label={open ? "收起面板" : "展开面板"}
            size="xs"
            active={open}
            onClick={() => {
              $panelOpen.value = !open
            }}
          >
            <IconSliders className="size-3.5" />
          </IconButton>
        </div>
      </div>
    </div>
  )
}

function statusText(
  phase: string,
  layerCount: number,
  visible: number,
): string {
  if (phase === "loading") return "载入中…"
  if (phase === "error") return "载入失败"
  if (phase === "ready") return `${visible}/${layerCount} 层可见`
  return "等待 GLB"
}
