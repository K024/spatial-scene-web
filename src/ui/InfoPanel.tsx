/**
 * 右侧浮层面板。
 *
 * 结构：品牌头 -> 场景信息 -> 相机/排序 -> 渲染参数 -> 背景与比对 -> 性能。
 *
 * 注意：**所有 `useValue()` 必须在组件顶部无条件调用完**（hooks 规则），
 * 之后再拿普通值去渲染/分支。下面统一用 `const v = {...}` 收口。
 */

import {
  autoAmplitude,
  autoPeriod,
  CAMERA_MODES,
  cameraMode,
  easingMs,
  freePitch,
  freeYaw,
  freeZoom,
  inputSensitivity,
  parallaxAmplitude,
  resetCamera,
  setCameraMode,
} from "../store/camera.ts"
import {
  meta,
  plyUrl,
  referenceImageUrl,
  sortInfo,
  sorting,
  status,
  view,
} from "../store/scene.ts"
import {
  aaMinPx,
  background,
  exposure,
  gpuName,
  maxPx,
  minPx,
  opacityScale,
  overlayMode,
  overlayOpacity,
  panelOpen,
  referenceImageReady,
  rendererError,
  reSortNow,
  resolutionScale,
  splatScale,
  viewStats,
} from "../store/viewer.ts"
import {
  fmtBytes,
  fmtDeg,
  fmtInt,
  fmtMeters,
  fmtMs,
  fmtNum,
  fmtVec3,
} from "./format.ts"
import {
  Badge,
  Button,
  Row,
  Section,
  Segmented,
  Slider,
  Stat,
  Toggle,
} from "./primitives.tsx"

/** 背景色预设。 */
const BACKGROUND_PRESETS: { label: string; value: string }[] = [
  { label: "深空", value: "#0a0b10" },
  { label: "中灰", value: "#808080" },
  { label: "纯白", value: "#ffffff" },
  { label: "纯黑", value: "#000000" },
]

/** 分辨率倍数选项。 */
const RESOLUTION_OPTIONS = [
  { value: 0.5, label: "0.5×" },
  { value: 0.75, label: "0.75×" },
  { value: 1, label: "1×" },
  { value: 1.5, label: "1.5×" },
]

export function InfoPanel() {
  // ── 所有信号订阅（无条件、顺序固定）──
  const v = {
    loadStatus: status.useValue(),
    sceneMeta: meta.useValue(),
    sceneView: view.useValue(),
    sort: sortInfo.useValue(),
    isSorting: sorting.useValue(),
    stats: viewStats.useValue(),
    gpu: gpuName.useValue(),
    glError: rendererError.useValue(),
    refUrl: referenceImageUrl.useValue(),
    refReady: referenceImageReady.useValue(),
    ply: plyUrl.useValue(),
    camYaw: freeYaw.useValue(),
    camPitch: freePitch.useValue(),
    camZoom: freeZoom.useValue(),
    cameraMode: cameraMode.useValue(),
    parallaxAmplitude: parallaxAmplitude.useValue(),
    autoAmplitude: autoAmplitude.useValue(),
    autoPeriod: autoPeriod.useValue(),
    easingMs: easingMs.useValue(),
    sensitivity: inputSensitivity.useValue(),
    splatScale: splatScale.useValue(),
    opacityScale: opacityScale.useValue(),
    exposure: exposure.useValue(),
    minPx: minPx.useValue(),
    maxPx: maxPx.useValue(),
    aaMinPx: aaMinPx.useValue(),
    resolutionScale: resolutionScale.useValue(),
    background: background.useValue(),
    overlayMode: overlayMode.useValue(),
    overlayOpacity: overlayOpacity.useValue(),
  }

  // 不在这里判 `panelOpen`：展开/收起由 App 的 AnimatePresence 控制，
  // 否则退出动画期间本组件已返回 null，动画会“瞬移”。
  const sortDeviation = v.stats.sortDeviationDeg
  const needsResort = sortDeviation > 2.5
  // 复位按钮的可用性看**实际**姿态（请求已归零但实际还在滑回来时，
  // 按钮不应该是灰的）
  const atReference =
    Math.abs(v.stats.appliedYawDeg) < 0.05 &&
    Math.abs(v.stats.appliedPitchDeg) < 0.05 &&
    Math.abs(v.camZoom - 1) < 1e-4

  return (
    <aside className="glass pointer-events-auto absolute top-4 right-4 bottom-4 z-20 flex w-[336px] flex-col overflow-hidden rounded-2xl">
      {/* ── 品牌头 ── */}
      <header className="flex items-center justify-between gap-3 px-4 pt-3.5 pb-3">
        <div className="flex items-center gap-2.5">
          <span className="relative flex h-7 w-7 items-center justify-center rounded-lg border border-accent-400/30 bg-gradient-to-br from-accent-400/25 to-cyan-500/5">
            <span className="h-2 w-2 rounded-full bg-accent-400 shadow-[0_0_10px_2px_rgba(34,211,238,0.55)]" />
          </span>
          <div className="leading-tight">
            <div className="text-[13px] font-semibold tracking-tight text-white/90">
              Spatial Scene
            </div>
            <div className="text-[10.5px] text-white/38">
              3DGS 查看器 · WebGL2
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

      {/* ── 滚动区 ── */}
      <div className="scroll-slim hairline min-h-0 flex-1 overflow-y-auto">
        {v.glError ? (
          <Section title="WebGL 不可用">
            <p className="rounded-lg border border-red-400/25 bg-red-500/10 px-3 py-2 text-[11.5px] leading-relaxed text-red-100/90">
              {v.glError}
            </p>
          </Section>
        ) : null}

        <Section
          title="场景"
          right={
            <Badge
              tone={
                v.loadStatus === "ready"
                  ? "ok"
                  : v.loadStatus === "error"
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
                }[v.loadStatus]
              }
            </Badge>
          }
        >
          <Row
            label="文件"
            value={<span className="tnum">{basename(v.ply)}</span>}
          />
          <Row
            label="高斯"
            value={v.sceneMeta ? fmtInt(v.sceneMeta.count) : "—"}
            sub={
              v.sceneMeta
                ? `四边形顶点 ${fmtInt(v.sceneMeta.count * 4)}`
                : undefined
            }
          />
          <Row
            label="文件大小"
            value={v.sceneMeta ? fmtBytes(v.sceneMeta.bytes) : "—"}
          />
          <Row
            label="深度范围"
            value={
              v.sceneView
                ? `${fmtNum(v.sceneView.depthRange[0])} – ${fmtNum(v.sceneView.depthRange[1])} m`
                : "—"
            }
            sub="相机坐标系 z（前）方向"
          />
          {v.sort ? (
            <Row
              label="排序 / 打包"
              value={`${fmtMs(v.sort.sortMs)} / ${fmtMs(v.sort.packMs)}`}
              sub={sortBaselineLabel(v.sort.camera.eye)}
            />
          ) : null}
        </Section>

        <div className="hairline" />

        <Section
          title="相机与排序"
          right={
            <Badge tone={needsResort ? "warn" : "ok"}>
              {needsResort ? "建议重排" : "已对齐基线"}
            </Badge>
          }
          hint="顺序只在加载时按参考相机排一次，之后固定不变。偏离基线后遮挡与融合顺序会开始出错——这正是「一次排序」要验证的代价。"
        >
          <Row label="相机模式">
            <div className="w-[168px]">
              <Segmented
                value={v.cameraMode}
                onChange={setCameraMode}
                options={CAMERA_MODES}
              />
            </div>
          </Row>
          {v.cameraMode === "free" ? (
            <Slider
              label="鼠标/滚轮灵敏度"
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
          {v.cameraMode === "parallax" ? (
            <>
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
              <p className="text-[11px] leading-relaxed text-white/35">
                光标移向画面边缘，相机就沿该方向平移（居中 = 参考视角）。
                陀螺仪接入点：把 deviceorientation 归一化后写进同一对 pointerX /
                pointerY 通道即可（TODO）。
              </p>
            </>
          ) : null}
          {v.cameraMode === "auto" ? (
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
              <p className="text-[11px] leading-relaxed text-white/35">
                偏航/俯仰取不同频率（8 字轨迹），因此不会退化成一条直线往复。
              </p>
            </>
          ) : null}
          <Slider
            label="缓动时间常数"
            value={v.easingMs}
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
            value={`${fmtDeg(v.stats.appliedYawDeg)} / ${fmtDeg(v.stats.appliedPitchDeg)}`}
            sub={`请求 ${fmtDeg(v.stats.requestedYawDeg)} / ${fmtDeg(v.stats.requestedPitchDeg)} · 缓动使实际值滞后于请求值`}
          />
          <Row
            label="相对参考视角"
            value={fmtDeg(v.stats.appliedDeviationDeg)}
          />
          <Row
            label="相机位置"
            value={
              <span className="text-[11.5px]">{fmtVec3(v.stats.eyePly)}</span>
            }
            sub={`距原点 ${fmtMeters(v.stats.eyeDistance)}`}
          />
          <Row
            label="视场（水平 × 垂直）"
            value={
              v.sceneView
                ? `${fmtNum(fovDeg(v.sceneView.focalPx, v.sceneView.imageSize[0]), 1)}° × ${fmtNum(fovDeg(v.sceneView.focalPx, v.sceneView.imageSize[1]), 1)}°`
                : "—"
            }
            sub={
              v.sceneView
                ? `f_px ${fmtNum(v.sceneView.focalPx, 1)} · ${v.sceneView.imageSize[0]}×${v.sceneView.imageSize[1]} · ${focalSourceLabel(v.sceneView.focalSource)}`
                : undefined
            }
          />
          <div className="mt-0.5 flex gap-2">
            <Button
              onClick={resetCamera}
              disabled={atReference}
              title="把相机放回参考位姿（原点 + 朝 +z）"
            >
              回到参考视角
            </Button>
            <Button
              variant={needsResort ? "warn" : "primary"}
              onClick={() => void reSortNow()}
              disabled={v.isSorting}
              title="按当前相机重排一遍（会重建顶点缓冲）"
            >
              {v.isSorting ? "排序中…" : "按当前视角重排序"}
            </Button>
          </div>
        </Section>

        <div className="hairline" />

        <Section title="渲染参数">
          <Slider
            label="高斯尺度"
            value={v.splatScale}
            min={0.2}
            max={2}
            onChange={(x) => {
              splatScale.value = x
            }}
            format={(x) => `${x.toFixed(2)}×`}
          />
          <Slider
            label="不透明度"
            value={v.opacityScale}
            min={0.1}
            max={1.5}
            onChange={(x) => {
              opacityScale.value = x
            }}
            format={(x) => `${x.toFixed(2)}×`}
          />
          <Slider
            label="曝光"
            value={v.exposure}
            min={0.1}
            max={3}
            onChange={(x) => {
              exposure.value = x
            }}
            format={(x) => `${x.toFixed(2)}×`}
          />
          <Slider
            label="最小像素尺寸"
            value={v.minPx}
            min={0}
            max={3}
            step={0.05}
            onChange={(x) => {
              minPx.value = x
            }}
            format={(x) => `${x.toFixed(2)} px`}
          />
          <Slider
            label="最大像素尺寸"
            value={v.maxPx}
            min={32}
            max={1024}
            step={8}
            onChange={(x) => {
              maxPx.value = x
            }}
            format={(x) => `${x.toFixed(0)} px`}
          />
          <Toggle
            label="亚像素抗锯齿"
            hint="把主轴方差钳到 ≥ 0.5 px（只影响比这更细的轴）；默认关，开了可减轻小高斯的采样闪烁"
            checked={v.aaMinPx > 0}
            onChange={(on) => {
              aaMinPx.value = on ? 0.5 : 0
            }}
          />
          <Row label="渲染分辨率">
            <div className="w-[168px]">
              <Segmented
                value={v.resolutionScale}
                onChange={(x) => {
                  resolutionScale.value = x
                }}
                options={RESOLUTION_OPTIONS}
              />
            </div>
          </Row>
        </Section>

        <div className="hairline" />

        <Section title="背景与参考对比">
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
          <Row label="参考照片比对">
            <div className="w-[168px]">
              <Segmented
                value={v.refReady ? v.overlayMode : "off"}
                disabled={!v.refReady}
                onChange={(x) => {
                  overlayMode.value = x
                }}
                options={[
                  { value: "off", label: "关闭" },
                  { value: "overlay", label: "叠加" },
                  { value: "blink", label: "闪烁" },
                ]}
              />
            </div>
          </Row>
          {v.overlayMode === "overlay" ? (
            <Slider
              label="叠加透明度"
              value={v.overlayOpacity}
              min={0.05}
              max={1}
              onChange={(x) => {
                overlayOpacity.value = x
              }}
              format={(x) => `${Math.round(x * 100)}%`}
            />
          ) : null}
          <p className="text-[11px] leading-relaxed text-white/35">
            {v.refReady
              ? "叠加层按参考内参投影成「无限远背景板」：相机只旋转时逐像素对齐；一旦平移就会出现视差——这是物理正确的行为，也说明固定排序在该状态下已经失效。"
              : v.refUrl
                ? "参考照片加载失败：sidecar 里写了 img_name，但同目录没找到图片。用 npm run sample 重新导出会把原图一并写到 public/exports/。"
                : "未找到参考照片（需要与 PLY 同目录的同名图片）。用 npm run sample 重新导出即可带上。"}
          </p>
        </Section>

        <div className="hairline" />

        <Section title="性能">
          <div className="grid grid-cols-2 gap-2">
            <Stat
              label="GPU 耗时"
              value={v.stats.gpuMs >= 0 ? fmtNum(v.stats.gpuMs, 2) : "—"}
              unit={v.stats.gpuMs >= 0 ? "ms" : undefined}
              accent
            />
            <Stat label="FPS" value={fmtNum(v.stats.fps, 0)} />
            <Stat label="CPU 提交" value={fmtNum(v.stats.cpuMs, 2)} unit="ms" />
            <Stat label="Draw calls" value={String(v.stats.drawCalls)} />
          </div>
          <Row
            label="渲染分辨率"
            value={`${v.stats.width}×${v.stats.height}`}
          />
          <Row
            label="提交高斯"
            value={fmtInt(v.stats.splats)}
            sub="固定顺序 · 单次实例化绘制"
          />
          <p className="text-[11px] leading-relaxed text-white/35">
            {v.stats.gpuTimingAvailable
              ? "GPU 耗时由 EXT_disjoint_timer_query_webgl2 实测：在命令流首尾插时间戳，涵盖全部 pass；结果滞后几帧读取（同步读会等 GPU，反而拖慢）。"
              : "拿不到真实 GPU 时间（无扩展，或平台把计时器禁用了——此时查询恒为 0，加载后约 2 秒会自动判定）。上面只显示 CPU 提交耗时，它不含 GPU 执行时间。"}
          </p>
        </Section>

        <div className="hairline" />

        <Section title="环境">
          <Row wide label="GPU" value={v.gpu || "—"} />
          <Row
            wide
            label="渲染管线"
            value="WebGL2 · RGBA16F 线性混合 · 每高斯 4 顶点实例化 · 单 draw call"
          />
          <Row
            label="GPU 计时"
            value={
              <Badge tone={v.stats.gpuTimingAvailable ? "ok" : "warn"}>
                {v.stats.gpuTimingAvailable ? "硬件计时可用" : "仅 CPU 提交"}
              </Badge>
            }
          />
        </Section>
      </div>

      <footer className="hairline px-4 py-2.5 text-[10.5px] leading-relaxed text-white/30">
        <span className="tnum">M</span> 切换相机模式 ·{" "}
        <span className="tnum">R</span> 复位 · <span className="tnum">P</span>{" "}
        暂停 · <span className="tnum">O</span> 切换比对 ·{" "}
        <span className="tnum">H</span> 收起面板
      </footer>
    </aside>
  )
}

/** 路径末段。 */
function basename(path: string): string {
  return path.split("/").pop() ?? path
}

/** `2·atan(size / (2f))`，度。 */
function fovDeg(focalPx: number, sizePx: number): number {
  return (2 * Math.atan(sizePx / (2 * focalPx)) * 180) / Math.PI
}

/** 弧度 -> 度。 */
function rad2deg(rad: number): number {
  return (rad * 180) / Math.PI
}

/** 焦距来源的中文标签。 */
function focalSourceLabel(source: "ply" | "sidecar" | "fallback"): string {
  return { ply: "PLY 内参", sidecar: "sidecar json", fallback: "兜底 40°" }[
    source
  ]
}

/** 「排序基线」= 原点则是参考相机。 */
function sortBaselineLabel(eye: readonly [number, number, number]): string {
  const atOrigin = Math.hypot(eye[0], eye[1], eye[2]) < 0.02
  return atOrigin ? "基线：参考相机（原点）" : `基线：相机 ${fmtVec3(eye)}`
}
