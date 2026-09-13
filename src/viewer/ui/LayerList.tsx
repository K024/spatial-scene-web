/**
 * 层列表：逐层可见性 / 独显。
 *
 * ── 列表顺序 ──
 * 层号升序（0 = 最近），背衬平面排在**最后**（它在物理上最远，但在 UI 上是「兜底」，
 * 单独看一眼的频次最高，放末尾便于固定位置）。
 *
 * ── 行内两条信息 ──
 * 深度带（米）+ 三角数：前者是调层放置策略时最常看的量，后者是显存 / 带宽的代理指标。
 */

import {
  $hovered,
  $layers,
  $solo,
  setLayerVisible,
  toggleSolo,
} from "../../store/viewer.ts"
import type { LayerState } from "../../viewer/types.ts"
import { IconButton, Switch } from "./bits.tsx"
import { formatCount, formatDepthRange } from "./format.ts"
import { IconSolo } from "./icons.tsx"

export function LayerList() {
  const layers = $layers.useValue()
  const solo = $solo.useValue()
  const hovered = $hovered.useValue()

  if (layers.length === 0) {
    return <p className="px-1 py-1 text-[11.5px] text-white/35">尚未载入场景</p>
  }

  return (
    <ul className="flex flex-col gap-0.5">
      {layers.map((layer) => (
        <LayerRow
          key={layer.key}
          layer={layer}
          solo={solo === layer.key}
          hovered={hovered === layer.key}
        />
      ))}
    </ul>
  )
}

function LayerRow({
  layer,
  solo,
  hovered,
}: {
  layer: LayerState
  solo: boolean
  hovered: boolean
}) {
  const dimmed = solo && !hovered

  return (
    <li
      onMouseEnter={() => {
        $hovered.value = layer.key
      }}
      onMouseLeave={() => {
        if ($hovered.peek() === layer.key) $hovered.value = null
      }}
      className={[
        "group flex items-center gap-2.5 rounded-xl py-1.5 pr-1 pl-2.5 transition-colors",
        hovered ? "bg-white/12" : "hover:bg-white/8",
        dimmed ? "opacity-60" : "",
      ].join(" ")}
    >
      <Switch
        checked={layer.visible}
        onChange={(checked) => setLayerVisible(layer.key, checked)}
        label={`显示 / 隐藏 ${layer.label}`}
        size="xs"
      />

      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-1.5">
          <span className="truncate text-[12.5px] leading-tight text-white/85">
            {layer.label}
          </span>
          {layer.kind === "backing" ? (
            <span className="badge badge-xs badge-soft shrink-0 border-white/10 bg-white/10 text-[9.5px] text-white/60">
              回填
            </span>
          ) : null}
        </div>
        <div className="mt-0.5 flex items-center gap-2 font-mono text-[10.5px] tabular-nums text-white/38">
          <span>{formatDepthRange(layer.depth)}</span>
          <span className="text-white/20">·</span>
          <span>{formatCount(layer.triangles)} tri</span>
        </div>
      </div>

      <IconButton
        label={solo ? "取消独显" : "独显该层"}
        active={solo}
        size="xs"
        className="opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100 data-[active=true]:opacity-100"
        data-active={solo}
        onClick={() => toggleSolo(layer.key)}
      >
        <IconSolo className="size-3.5" />
      </IconButton>
    </li>
  )
}
