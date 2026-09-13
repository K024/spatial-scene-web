/**
 * 右侧可收起的**悬浮毛玻璃面板**。
 *
 * ── 结构 ──
 * ```
 * [header 标题 + 收起]  ← 固定
 * [层列表（唯一滚动区）]  ← flex-1 overflow-y-auto
 * [视图控制 / 场景信息 ]  ← 固定
 * [来源动作 + 错误     ]  ← 固定
 * ```
 * 只有层列表滚动：小屏上「视图 / 信息 / 动作」不该被滚走。
 *
 * ── 收起动画 ──
 * `motion` 做 `x + opacity`（spring）。收起后 `pointer-events: none`，
 * 否则不可见的面板会吃掉画布的拖拽 / 轨道操作。
 */

import { motion } from "motion/react"
import {
  $depthSpan,
  $error,
  $layers,
  $panelOpen,
  $phase,
  $scene,
  $showGizmo,
  $showGrid,
  $solo,
  $source,
  $sway,
  $totalTriangles,
  $totalVertices,
  $visibleCount,
  $wireframe,
  clearScene,
  requestResetView,
  setAllVisible,
} from "../../store/viewer.ts"
import { loadSampleGlb, pickGlbFile } from "./actions.ts"
import {
  Divider,
  Field,
  GLASS,
  IconButton,
  Section,
  TextButton,
} from "./bits.tsx"
import {
  formatBytes,
  formatCount,
  formatDegrees,
  formatDepthRange,
  formatMeters,
} from "./format.ts"
import {
  IconAxis,
  IconChevronRight,
  IconGrid,
  IconInfo,
  IconLayers,
  IconRotate,
  IconSliders,
  IconTarget,
  IconUpload,
  IconWireframe,
} from "./icons.tsx"
import { LayerList } from "./LayerList.tsx"

export function FloatingPanel() {
  const open = $panelOpen.useValue()

  return (
    <motion.aside
      initial={false}
      animate={{ x: open ? 0 : 420, opacity: open ? 1 : 0 }}
      transition={{ type: "spring", stiffness: 300, damping: 32, mass: 0.8 }}
      style={{ pointerEvents: open ? "auto" : "none" }}
      aria-hidden={!open}
      className={`fixed top-3 right-3 bottom-3 z-20 flex w-[min(22rem,calc(100vw-1.5rem))] flex-col overflow-hidden rounded-3xl ${GLASS}`}
    >
      <PanelHeader />
      <div className="panel-scroll min-h-0 flex-1 overflow-y-auto overscroll-contain">
        <Section
          title="层"
          icon={<IconLayers className="size-3.5" />}
          aside={<VisibilityBulkActions />}
        >
          <LayerList />
        </Section>
        <Divider />
        <Section title="视图" icon={<IconSliders className="size-3.5" />}>
          <ViewControls />
        </Section>
        <Divider />
        <Section title="场景" icon={<IconInfo className="size-3.5" />}>
          <SceneInfo />
        </Section>
      </div>
      <Divider />
      <SourceActions />
    </motion.aside>
  )
}

function PanelHeader() {
  const source = $source.useValue()
  const scene = $scene.useValue()
  const phase = $phase.useValue()
  const visible = $visibleCount.useValue()
  const layers = $layers.useValue()

  const subtitle =
    phase === "ready" && scene
      ? `${source.name || scene.name} · ${visible}/${layers.length} 层`
      : phase === "loading"
        ? "载入中…"
        : phase === "error"
          ? "载入失败"
          : "未载入场景"

  return (
    <header className="flex items-start gap-2 px-4 pt-3.5 pb-3">
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-1.5">
          <IconLayers className="size-4 text-white/55" />
          <h1 className="text-[13.5px] font-semibold text-white/90">
            空间场景
          </h1>
        </div>
        <p className="mt-1 truncate text-[11px] text-white/45">{subtitle}</p>
      </div>
      <IconButton
        label="收起面板"
        onClick={() => {
          $panelOpen.value = false
        }}
      >
        <IconChevronRight className="size-4" />
      </IconButton>
    </header>
  )
}

function VisibilityBulkActions() {
  const ready = $phase.useValue() === "ready"
  return (
    <div className="flex items-center gap-1">
      <TextButton onClick={() => setAllVisible(true)} disabled={!ready}>
        全显
      </TextButton>
      <TextButton onClick={() => setAllVisible(false)} disabled={!ready}>
        全隐
      </TextButton>
    </div>
  )
}

function ViewControls() {
  const showGrid = $showGrid.useValue()
  const showGizmo = $showGizmo.useValue()
  const sway = $sway.useValue()
  const solo = $solo.useValue()
  const wireframe = $wireframe.useValue()
  const scene = $scene.useValue()
  const triangles = $totalTriangles.useValue()

  return (
    <>
      <div className="flex flex-wrap items-center gap-1.5">
        <IconButton
          label="地面网格"
          active={showGrid}
          onClick={() => {
            $showGrid.value = !showGrid
          }}
        >
          <IconGrid className="size-4" />
        </IconButton>
        <IconButton
          label="坐标轴"
          active={showGizmo}
          onClick={() => {
            $showGizmo.value = !showGizmo
          }}
        >
          <IconAxis className="size-4" />
        </IconButton>
        <IconButton
          label="视差摆动（固定焦点，绕参考机位小幅移动）"
          active={sway}
          onClick={() => {
            $sway.value = !sway
          }}
        >
          <IconRotate className="size-4" />
        </IconButton>
        <IconButton
          label="回到参考视角"
          disabled={!scene}
          onClick={requestResetView}
        >
          <IconTarget className="size-4" />
        </IconButton>
        <IconButton
          label="线框视图（只看几何，不看图片）"
          active={wireframe}
          disabled={!scene}
          onClick={() => {
            $wireframe.value = !wireframe
          }}
        >
          <IconWireframe className="size-4" />
        </IconButton>
        {solo ? (
          <TextButton
            onClick={() => {
              $solo.value = null
            }}
          >
            取消独显
          </TextButton>
        ) : null}
      </div>
      {wireframe ? <WireframeNotice triangles={triangles} /> : null}
    </>
  )
}

/**
 * 线框模式的两条硬约束，摆在 UI 上而不是只写在注释里。
 *
 * 1. **内存**：three 会为每条几何**懒生成**并永久缓存一条线段索引
 *    （`6 × 三角数` 个 uint32），切回标准渲染也不释放 —— 用户有权在点之前知道要付多少。
 * 2. **可读性**：`gl.LINES` 在 WebGL2 上恒为 1 设备像素，而层是「每像素一顶点」的网格，
 *    参考视角下线距 ≈ 1px。不写清楚会被当成 bug。
 */
function WireframeNotice({ triangles }: { triangles: number }) {
  return (
    <p className="mt-2 text-[10.5px] leading-relaxed text-white/35">
      线框由 three 懒生成的线段索引驱动，本次约{" "}
      <span className="font-mono text-white/55">
        {formatBytes(triangles * 6 * 4)}
      </span>
      （6 × 三角数 × 4B，切回普通模式也不释放）。层是「每像素一顶点」的网格，
      参考视角下线距 ≈ 1px，需放大 6~8 倍才能看清格子。
    </p>
  )
}

function SceneInfo() {
  const scene = $scene.useValue()
  const span = $depthSpan.useValue()
  const triangles = $totalTriangles.useValue()
  const vertices = $totalVertices.useValue()

  if (!scene) return <p className="text-[11.5px] text-white/35">—</p>

  const { meta, bounds } = scene
  const disparity = meta.disparity

  return (
    <dl className="flex flex-col">
      <Field label="垂直视场角" value={formatDegrees(meta.verticalFOV)} mono />
      <Field label="参考宽高比" value={meta.aspectRatio.toFixed(3)} mono />
      <Field
        label="near / far"
        value={`${formatMeters(meta.near)} / ${formatMeters(meta.far)}`}
        mono
      />
      <Field label="深度跨度" value={formatDepthRange(span)} mono />
      <Field
        label="包围盒"
        value={bounds.size.map((value) => formatMeters(value)).join(" × ")}
        mono
      />
      <Field label="三角面" value={formatCount(triangles)} mono />
      <Field label="顶点" value={formatCount(vertices)} mono />
      <Field label="GLB" value={formatBytes(scene.bytes)} mono />
      {disparity ? (
        <Field
          label="视差 min/中位/max"
          value={`${disparity.minimum.toFixed(3)} / ${disparity.median.toFixed(3)} / ${disparity.maximum.toFixed(3)}`}
          mono
        />
      ) : null}
      {disparity ? (
        <Field
          label="视差 μ ± σ"
          value={`${disparity.mean.toFixed(3)} ± ${disparity.stdDev.toFixed(4)}`}
          mono
        />
      ) : null}
      <Field
        label="预乘 α"
        value={meta.premultipliedAlpha ? "是" : "否"}
        mono
      />
    </dl>
  )
}

function SourceActions() {
  const source = $source.useValue()
  const scene = $scene.useValue()
  const phase = $phase.useValue()
  const error = $error.useValue()

  return (
    <div className="flex flex-col gap-1.5 px-4 py-3">
      <div className="flex items-center gap-1.5">
        <TextButton className="flex-1" onClick={pickGlbFile}>
          <IconUpload className="mr-1 inline size-3.5 align-[-2px]" />
          打开 GLB
        </TextButton>
        <TextButton className="flex-1" onClick={loadSampleGlb}>
          载入样例
        </TextButton>
        {scene ? <TextButton onClick={clearScene}>关闭</TextButton> : null}
      </div>
      <p
        className="truncate font-mono text-[10px] text-white/30"
        title={source.url}
      >
        {source.kind === "none" ? "拖拽 .glb 到窗口任意位置" : source.url}
      </p>
      {phase === "error" && error ? (
        <p className="rounded-lg border border-red-400/25 bg-red-500/12 px-2 py-1.5 text-[11px] break-all text-red-200/90">
          {error}
        </p>
      ) : null}
    </div>
  )
}
