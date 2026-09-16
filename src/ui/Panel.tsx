/**
 * 右侧浮动玻璃面板：GLB 查看器主控。
 *
 * 只做「换文件 + 调视图 + 看性能」；分层 / 网格化是 `scripts/export-glb.ts` 的事。
 *
 * 结构：品牌头 → 文件 → 概览 → 显示（图标开关）→ 图层（列表内单独显示）→
 *       相机（direct-splat 视角方案）→ 性能（GPU 计时 + draw call）。
 *
 * 所有 `useValue()` 必须在组件顶部无条件调用完（hooks 规则）。
 */

import * as THREE from "three"
import {
  appliedPose,
  autoAmplitude,
  autoPeriod,
  background,
  CAMERA_MODES,
  cameraMode,
  doubleSided,
  easingMs,
  explode,
  type GlbMeta,
  glbError,
  glbMeta,
  glbSource,
  glbStatus,
  inputSensitivity,
  isolateLayer,
  layerOpacity,
  loadGlb,
  loadGlbFromFile,
  panelOpen,
  parallaxAmplitude,
  poseDeviationDeg,
  resetCamera,
  STATS_WINDOWS,
  setCameraMode,
  showGizmo,
  showGrid,
  statsWindowMs,
  viewStats,
  wireframe,
} from "../store/index.ts"
import { fmtBytes, fmtDeg, fmtInt, fmtMs, fmtNum } from "./format.ts"
import {
  IconAxes,
  IconCube,
  IconEye,
  IconEyeOff,
  IconGrid,
  IconRefresh,
  IconTarget,
  IconWireframe,
} from "./icons.tsx"
import {
  Badge,
  Button,
  FilePickerButton,
  IconBar,
  IconToggle,
  PercentileTable,
  Row,
  Section,
  Segmented,
  Slider,
  Stat,
} from "./primitives.tsx"

const BACKGROUND_PRESETS: { label: string; value: string }[] = [
  { label: "深空", value: "#0a0b10" },
  { label: "中灰", value: "#808080" },
  { label: "纯白", value: "#ffffff" },
  { label: "纯黑", value: "#000000" },
]

/** 性能表格的列。 */
const PERF_COLUMNS = ["平均", "p50", "p95", "p99", "最大"]

export function Panel() {
  const v = {
    source: glbSource.useValue(),
    status: glbStatus.useValue(),
    error: glbError.useValue(),
    meta: glbMeta.useValue(),
    background: background.useValue(),
    grid: showGrid.useValue(),
    gizmo: showGizmo.useValue(),
    opacity: layerOpacity.useValue(),
    wireframe: wireframe.useValue(),
    doubleSided: doubleSided.useValue(),
    isolate: isolateLayer.useValue(),
    explode: explode.useValue(),
    mode: cameraMode.useValue(),
    sensitivity: inputSensitivity.useValue(),
    parallaxAmplitude: parallaxAmplitude.useValue(),
    autoAmplitude: autoAmplitude.useValue(),
    autoPeriod: autoPeriod.useValue(),
    easing: easingMs.useValue(),
    stats: viewStats.useValue(),
    statsWindow: statsWindowMs.useValue(),
  }

  const loading = v.status === "loading"
  const pose = appliedPose()

  return (
    <aside className="glass pointer-events-auto absolute top-4 right-4 bottom-4 z-20 flex w-[344px] flex-col overflow-hidden rounded-2xl">
      <header className="flex items-center justify-between gap-3 px-4 pt-3.5 pb-3">
        <div className="flex items-center gap-2.5">
          <span className="relative flex h-7 w-7 items-center justify-center rounded-lg border border-accent-400/30 bg-gradient-to-br from-accent-400/25 to-cyan-500/5">
            <span className="h-2 w-2 rounded-full bg-accent-400 shadow-[0_0_10px_2px_rgba(34,211,238,0.55)]" />
          </span>
          <div className="leading-tight">
            <div className="text-[13px] font-semibold tracking-tight text-white/90">
              Layered GLB
            </div>
            <div className="text-[10.5px] text-white/38">
              分层网格查看器 · three.js
            </div>
          </div>
        </div>
        <button
          type="button"
          onClick={() => {
            panelOpen.value = false
          }}
          title="收起面板"
          className="focus-ring rounded-md border border-white/10 bg-white/5 px-1.5 py-1 text-[11px] text-white/45 transition-colors hover:bg-white/10 hover:text-white/80"
        >
          ⇥
        </button>
      </header>

      <div className="scroll-slim hairline min-h-0 flex-1 overflow-y-auto">
        {v.error ? (
          <Section title="加载失败">
            <p className="rounded-lg border border-red-400/25 bg-red-500/10 px-3 py-2 text-[11.5px] leading-relaxed break-words text-red-100/90">
              {v.error}
            </p>
          </Section>
        ) : null}

        {/* ── 文件 ── */}
        <Section
          title="文件"
          right={
            <Badge
              tone={
                v.status === "ready"
                  ? "ok"
                  : v.status === "error"
                    ? "warn"
                    : "busy"
              }
            >
              {
                {
                  ready: "已就绪",
                  loading: "加载中",
                  error: "失败",
                  idle: "空闲",
                }[v.status]
              }
            </Badge>
          }
          hint="默认读 npm run export-glb 的产物；自定义产物请用文件选择或直接拖进窗口。相机与层元数据都写在 GLB 里，查看器不依赖任何 sidecar。"
        >
          <Row
            label="当前产物"
            value={
              <span className="tnum">
                {v.source.kind === "file" ? v.source.label : v.source.url}
              </span>
            }
            sub={v.source.kind === "file" ? "本地文件" : "默认路径"}
          />
          <div className="flex flex-wrap items-center gap-2">
            <FilePickerButton
              label="选择 GLB 文件"
              accept=".glb"
              variant="primary"
              disabled={loading}
              onPick={(file) => void loadGlbFromFile(file)}
            />
            <Button
              disabled={loading}
              onClick={() => void loadGlb()}
              title="重新加载默认产物"
            >
              <span className="flex items-center gap-1.5">
                <IconRefresh width={13} height={13} />
                默认产物
              </span>
            </Button>
          </div>
        </Section>

        {v.meta ? (
          <>
            <div className="hairline" />
            {/* ── 概览 ── */}
            <Section title="概览">
              <div className="grid grid-cols-2 gap-2">
                <Stat label="图层" value={String(v.meta.layers.length)} />
                <Stat
                  label="三角面"
                  value={fmtInt(v.meta.triangleCount)}
                  accent
                />
              </div>
              <Row
                label="顶点"
                value={fmtInt(v.meta.vertexCount)}
                sub={`文件 ${fmtBytes(v.meta.bytes)}`}
              />
              <Row
                label="网格尺寸"
                value={fmtVec3(metaSize(v.meta))}
                sub={`中心 ${fmtVec3(metaCenter(v.meta))}`}
              />
              <CameraRows meta={v.meta} />
            </Section>

            <div className="hairline" />
            {/* ── 显示（图标开关，放在图层上方）── */}
            <Section title="显示">
              <IconBar title="叠加">
                <IconToggle
                  label="网格地面"
                  active={v.grid}
                  onClick={() => {
                    showGrid.value = !v.grid
                  }}
                >
                  <IconGrid />
                </IconToggle>
                <IconToggle
                  label="坐标轴"
                  active={v.gizmo}
                  onClick={() => {
                    showGizmo.value = !v.gizmo
                  }}
                >
                  <IconAxes />
                </IconToggle>
                <IconToggle
                  label="线框"
                  active={v.wireframe}
                  onClick={() => {
                    wireframe.value = !v.wireframe
                  }}
                >
                  <IconWireframe />
                </IconToggle>
                <IconToggle
                  label="双面"
                  active={v.doubleSided}
                  onClick={() => {
                    doubleSided.value = !v.doubleSided
                  }}
                >
                  <IconCube />
                </IconToggle>
              </IconBar>
              <Row label="背景色">
                <div className="flex items-center gap-1.5">
                  {BACKGROUND_PRESETS.map((p) => (
                    <button
                      key={p.value}
                      type="button"
                      title={p.label}
                      onClick={() => {
                        background.value = p.value
                      }}
                      style={{ background: p.value }}
                      className={`focus-ring h-6 w-6 rounded-md border transition-transform ${
                        v.background === p.value
                          ? "scale-105 border-accent-400"
                          : "border-white/15 hover:scale-105"
                      }`}
                    />
                  ))}
                  <input
                    type="color"
                    aria-label="自定义背景色"
                    className="swatch h-6 w-6"
                    value={v.background}
                    onChange={(e) => {
                      background.value = e.currentTarget.value
                    }}
                  />
                </div>
              </Row>
              <Slider
                label="整体不透明度"
                value={v.opacity}
                min={0.1}
                max={1}
                step={0.05}
                onChange={(x) => {
                  layerOpacity.value = x
                }}
                format={(x) => `${Math.round(x * 100)}%`}
              />
            </Section>

            <div className="hairline" />
            {/* ── 图层（单独显示在列表内）── */}
            <Section
              title="图层"
              right={<Badge tone="neutral">0 = 最近</Badge>}
              hint="点某行的眼睛只显示该层（再点一次回到全部）；节点顺序与 renderOrder 已按远→近排好。"
            >
              <Slider
                label="爆炸视图"
                value={v.explode}
                min={0}
                max={1}
                step={0.02}
                onChange={(x) => {
                  explode.value = x
                }}
                format={(x) => (x === 0 ? "关闭" : `${Math.round(x * 100)}%`)}
              />
              <ul className="mt-0.5 flex flex-col">
                {v.meta.layers.map((layer) => {
                  const isolated = v.isolate === layer.layerIndex
                  const visible = v.isolate < 0 || isolated
                  return (
                    <li
                      key={layer.layerIndex}
                      className="hairline-x flex items-center gap-2 py-1.5"
                    >
                      <IconToggle
                        compact
                        label={
                          isolated
                            ? `显示全部图层（当前单独显示 #${layer.layerIndex}）`
                            : `单独显示图层 #${layer.layerIndex}`
                        }
                        active={isolated}
                        onClick={() => {
                          isolateLayer.value = isolated ? -1 : layer.layerIndex
                        }}
                      >
                        {isolated ? (
                          <IconEye width={13} height={13} />
                        ) : (
                          <IconEyeOff width={13} height={13} />
                        )}
                      </IconToggle>
                      <span
                        className={`tnum w-7 shrink-0 text-[11.5px] ${
                          visible ? "text-white/45" : "text-white/25"
                        }`}
                      >
                        #{layer.layerIndex}
                      </span>
                      <span
                        className={`tnum flex-1 truncate text-[11.5px] ${
                          visible ? "text-white/60" : "text-white/25"
                        }`}
                      >
                        {layer.depthRange
                          ? `${layer.depthRange[0].toFixed(2)} – ${layer.depthRange[1].toFixed(2)} m`
                          : "—"}
                      </span>
                      <span
                        className={`tnum shrink-0 text-[11.5px] ${
                          visible ? "text-white/70" : "text-white/25"
                        }`}
                      >
                        {fmtInt(layer.triangleCount)}
                      </span>
                    </li>
                  )
                })}
              </ul>
            </Section>

            <div className="hairline" />
            {/* ── 相机（direct-splat 视角方案）── */}
            <Section
              title="相机"
              right={
                <IconToggle
                  compact
                  label="回到参考视角"
                  active={false}
                  onClick={resetCamera}
                >
                  <IconTarget width={13} height={13} />
                </IconToggle>
              }
              hint="轨道姿态 + 三种输入源：自由拖拽累积、视差随指针、自动走 8 字轨迹；实际姿态按时间常数缓动追上请求值。"
            >
              <Row label="相机模式">
                <div className="w-[188px]">
                  <Segmented
                    value={v.mode}
                    onChange={setCameraMode}
                    options={CAMERA_MODES}
                  />
                </div>
              </Row>
              {v.mode === "free" ? (
                <Slider
                  label="拖拽/滚轮灵敏度"
                  value={v.sensitivity}
                  min={0.2}
                  max={2}
                  step={0.05}
                  onChange={(x) => {
                    inputSensitivity.value = x
                  }}
                  format={(x) => `${x.toFixed(2)}×`}
                />
              ) : null}
              {v.mode === "parallax" ? (
                <Slider
                  label="视差幅度"
                  value={v.parallaxAmplitude}
                  min={0.5}
                  max={10}
                  step={0.1}
                  onChange={(x) => {
                    parallaxAmplitude.value = x
                  }}
                  format={(x) => `${x.toFixed(1)}°`}
                />
              ) : null}
              {v.mode === "auto" ? (
                <>
                  <Slider
                    label="自动幅度"
                    value={v.autoAmplitude}
                    min={0.5}
                    max={10}
                    step={0.1}
                    onChange={(x) => {
                      autoAmplitude.value = x
                    }}
                    format={(x) => `${x.toFixed(1)}°`}
                  />
                  <Slider
                    label="自动周期"
                    value={v.autoPeriod}
                    min={2}
                    max={30}
                    step={0.5}
                    onChange={(x) => {
                      autoPeriod.value = x
                    }}
                    format={(x) => `${x.toFixed(1)} s`}
                  />
                </>
              ) : null}
              <Slider
                label="缓动时间常数"
                value={v.easing}
                min={0}
                max={500}
                step={10}
                onChange={(x) => {
                  easingMs.value = x
                }}
                format={(x) => (x === 0 ? "硬贴" : `${x.toFixed(0)} ms`)}
              />
              <Row
                label="姿态（实际）"
                value={`${fmtDeg(rad2deg(pose.yaw))} / ${fmtDeg(rad2deg(pose.pitch))}`}
                sub={`相对参考视角 ${poseDeviationDeg(pose).toFixed(2)}° · 距离 ${pose.zoom.toFixed(2)}×`}
              />
            </Section>

            <div className="hairline" />
            {/* ── 性能（GPU 计时 + draw call）── */}
            <Section
              title="性能"
              right={
                <div className="w-[140px] shrink-0">
                  <Segmented
                    value={v.statsWindow}
                    onChange={(x) => {
                      statsWindowMs.value = x
                    }}
                    options={STATS_WINDOWS}
                  />
                </div>
              }
              hint={
                v.stats.gpuTimingAvailable
                  ? "窗口滑动、面板 2 Hz 刷新。帧间隔含提交+等待+合成，会被垂直同步钉在刷新周期，所以「稳不稳」主要看 GPU 实测那一行的平均值与 p99 差距。GPU 实测由 EXT_disjoint_timer_query_webgl2 量出（命令流首尾插时间戳，结果滞后几帧读）。"
                  : "拿不到真实 GPU 时间（浏览器没有 EXT_disjoint_timer_query_webgl2，或平台禁用了计时器——此时查询恒为 0，加载后约 2 秒会自动判定）。GPU 实测那一行会全是 ——；帧间隔依然真实，它含提交+等待+合成。"
              }
            >
              <div className="grid grid-cols-2 gap-2">
                <Stat
                  label="GPU 耗时（平均）"
                  value={v.stats.gpu.count > 0 ? fmtNum(v.stats.gpu.avg) : "—"}
                  unit={v.stats.gpu.count > 0 ? "ms" : undefined}
                  accent
                />
                <Stat
                  label="平均帧率"
                  value={v.stats.frame.count > 0 ? fmtNum(v.stats.fps, 1) : "—"}
                  unit={v.stats.frame.count > 0 ? "fps" : undefined}
                />
              </div>
              <PercentileTable
                unit="ms"
                columns={PERF_COLUMNS}
                rows={[
                  {
                    label: "帧间隔",
                    values: distRow(v.stats.frame),
                    accent: true,
                  },
                  { label: "CPU 提交", values: distRow(v.stats.cpu) },
                  { label: "GPU 实测", values: distRow(v.stats.gpu) },
                ]}
              />
              <div className="grid grid-cols-2 gap-2">
                <Stat label="Draw calls" value={String(v.stats.drawCalls)} />
                <Stat label="三角面" value={fmtInt(v.stats.triangles)} />
              </div>
              <Row
                label="渲染分辨率"
                value={`${v.stats.width}×${v.stats.height}`}
                sub={`样本 ${v.stats.frame.count} 帧 / ${v.stats.windowMs / 1000} s 窗口${
                  v.stats.stalls > 0
                    ? ` · 已排除 ${v.stats.stalls} 次 >1 s 停顿`
                    : ""
                }`}
              />
              <Row
                label="GPU 计时"
                value={
                  <Badge tone={v.stats.gpuTimingAvailable ? "ok" : "warn"}>
                    {v.stats.gpuTimingAvailable
                      ? "硬件计时可用"
                      : "仅 CPU 提交"}
                  </Badge>
                }
              />
            </Section>
          </>
        ) : null}
      </div>

      <footer className="hairline px-4 py-2.5 text-[10.5px] leading-relaxed text-white/30">
        拖入 .glb 载入 · <span className="tnum">拖拽</span> 旋转 ·{" "}
        <span className="tnum">滚轮</span> 缩放 ·{" "}
        <span className="tnum">双击</span> 参考视角 ·{" "}
        <span className="tnum">M</span> 相机模式 ·{" "}
        <span className="tnum">H</span> 面板
      </footer>
    </aside>
  )
}

/** GLB 自带的相机 / 视图元数据（只读展示）。 */
function CameraRows({ meta }: { meta: GlbMeta }) {
  const camera = meta.camera
  const extras = meta.extras
  const viewScale = extras.viewScale
  const pixelScale = extras.pixelScale
  const refFocal = extras.referenceFocalLengthPx
  return (
    <>
      <Row
        label="相机 fov / near/far"
        value={
          camera
            ? `${camera.fovDeg.toFixed(1)}° · ${camera.near.toFixed(2)}–${camera.far.toFixed(1)} m`
            : "—"
        }
        sub={
          camera
            ? `aspect ${camera.aspect.toFixed(3)} · 视口用 contain 拟合画布，不拉伸`
            : "GLB 未包含相机"
        }
      />
      {typeof viewScale === "number" ? (
        <Row
          label="分层视图"
          value={`viewScale ${viewScale.toFixed(2)}× · pixelScale ${typeof pixelScale === "number" ? pixelScale.toFixed(3) : "—"}`}
          sub={
            typeof refFocal === "number"
              ? `原图焦距 ${refFocal.toFixed(1)}px · 画布 ${String(extras.width ?? "?")}×${String(extras.height ?? "?")}`
              : undefined
          }
        />
      ) : null}
    </>
  )
}

/** 分布 -> 表格一行；不可用时给等长 `null` 行（保持网格对齐）。 */
function distRow(d: {
  count: number
  avg: number
  p50: number
  p95: number
  p99: number
  max: number
}): (string | null)[] {
  if (d.count === 0) return PERF_COLUMNS.map(() => null)
  return [d.avg, d.p50, d.p95, d.p99, d.max].map((x) => fmtMs(x))
}

function metaSize(meta: GlbMeta): THREE.Vector3 {
  return meta.bounds.getSize(new THREE.Vector3())
}

function metaCenter(meta: GlbMeta): THREE.Vector3 {
  return meta.bounds.getCenter(new THREE.Vector3())
}

function fmtVec3(v: THREE.Vector3): string {
  return `${v.x.toFixed(2)}, ${v.y.toFixed(2)}, ${v.z.toFixed(2)}`
}

function rad2deg(rad: number): number {
  return (rad * 180) / Math.PI
}
