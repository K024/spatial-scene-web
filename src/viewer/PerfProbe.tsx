/**
 * 性能探针（`window.__spatialPerf`）—— 用来判定「瓶颈到底在哪」。
 *
 * ── 为什么需要它 ──
 * `frameloop="demand"` 下静止时帧率为 0，光看任务管理器 / DevTools 的 FPS 没有意义。
 * 要测性能必须**强制连续渲染**一段时间，同时读 three 的 `renderer.info`。本组件把这两件
 * 事做成一个 console API，避免每次手改代码。
 *
 * ── 怎么用（浏览器 console）──
 * ```
 * __spatialPerf.dump()          // 整帧渲染统计 + 显存估算（table 友好）
 * __spatialPerf.textures()      // 逐层纹理尺寸 / 估算字节
 * await __spatialPerf.benchmark(180)   // 强制连续渲染 180 帧，返回帧时统计
 * __spatialPerf.dpr(2)          // 压力测试：改 DPR，看填充率随像素数缩放
 * ```
 *
 * ── 一个必须知道的坑：`gl.info` 是**每 `gl.render()` 都重置**的 ──
 * `<GizmoHelper>` 经 drei `Hud` 每帧调两次 `gl.render()`（先主场景再 gizmo），
 * 默认 `info.autoReset=true` 下帧尾只能读到**最后一次**（gizmo）的几十个三角形。
 * 所以本探针关掉 `autoReset`，并在每帧 `useFrame`（priority 0，先于 Hud 的 priority 1）
 * 手动 `reset()`，读到的才是整帧合计。
 *
 * ── 怎么读（判定瓶颈）──
 * 1. `render.triangles` 只有几万 ⇒ **顶点/三角形不是瓶颈**，别再从面数下手。
 * 2. `benchmark` 的 `fps` 低、且 `avgMs` 随**可见层数**近似线性增长 ⇒ 填充率 / overdraw /
 *    纹理带宽受限（本管线 7 层全屏 α 混合、`depthWrite=false`、双面，天然多倍 overdraw）。
 * 3. `avgMs` 随**画布像素数**（改 DPR / 缩放窗口）近似线性 ⇒ 纯填充率受限。
 * 4. `textureBytes` 到几百 MB ⇒ 显存 / 上传成本是大头（原生 3024×2268 × 7 层 ≈ 190MB+）。
 * 5. `avgMs` 低但 `dump().triangles`/`calls` 高 ⇒ draw call / 状态切换受限（分层数=draw 数）。
 * 6. `avgMs ≈ 1000/屏幕刷新率`（如 8.33ms = 120Hz）且 `minMs` 远小于它 ⇒ **撞 vsync 上限、
 *    GPU 有余量**，当前尺寸下性能不是问题。
 *
 * 与 `dump()` 配合的**消融实验**（不用改代码）：
 * - 逐层 solo / 隐藏：看 `avgMs` 对层数的斜率；
 * - 开/关背衬平面（最大的一块远景几何）；
 * - 换 `?glb=` 到 768 样例：把纹理分辨率降 16×，看 `avgMs` 变化；
 * - 浏览器缩放 / 改窗口大小，或 `__spatialPerf.dpr(2)`：看像素数的影响。
 */

import { useFrame, useThree } from "@react-three/fiber"
import { useEffect, useRef } from "react"
import type {
  Camera,
  Material,
  Mesh,
  Object3D,
  Texture,
  WebGLRenderer,
} from "three"
import { $scene } from "../store/viewer.ts"

/** 一次 benchmark 的帧时统计（毫秒）。 */
export interface PerfFrameStats {
  readonly frames: number
  readonly avgMs: number
  readonly medianMs: number
  readonly p95Ms: number
  readonly minMs: number
  readonly maxMs: number
  /** 由 `avgMs` 换算（`1000 / avgMs`），仅作直观参考。 */
  readonly fps: number
  /** 每帧 `gl.render()` 次数（>1 = 多 pass；主场景被重复画时会 >2）。 */
  readonly passes: number
  /** 最后一帧的渲染统计（triangle 数是这一帧实际提交的）。 */
  readonly render: PerfDump["render"]
}

/** `renderer.info` + 画布口径快照。 */
export interface PerfDump {
  readonly render: {
    readonly calls: number
    readonly triangles: number
    readonly points: number
    readonly lines: number
    readonly frame: number
  }
  readonly memory: {
    readonly geometries: number
    readonly textures: number
  }
  readonly programs: number
  readonly pixelRatio: number
  /** 每帧 `gl.render()` 调用次数（= 渲染 pass 数）。>1 说明主场景可能被重复画。 */
  readonly passes: number
  /** 场景里所有 mesh 的三角形数（**单遍**理论值）。与 `render.triangles` 对比即可发现重复。 */
  readonly sceneTriangles: number
  /** drawingBuffer 实际像素（= CSS × DPR）。 */
  readonly drawingBuffer: readonly [number, number]
  readonly canvasCss: readonly [number, number]
  /** 逐层纹理估算（RGBA8 + mip ≈ ×4/3）。 */
  readonly textureBytes: number
}

interface PerfApi {
  dump: () => PerfDump
  textures: () => Array<{
    key: string
    width: number
    height: number
    bytes: number
  }>
  benchmark: (frames?: number) => Promise<PerfFrameStats>
  /** 压力测试用：临时改 DPR（`pixelRatio`），看填充率随像素数的缩放。 */
  dpr: (pixelRatio: number) => void
  /** 逐 `gl.render()` 记账（场景身份 + 该 pass 的 calls/triangles），定位重复渲染。 */
  perPass: () => Promise<PerfPass[]>
}

/** 一次 `gl.render()` 的记账。 */
export interface PerfPass {
  readonly sceneType: string
  readonly sceneName: string
  readonly isMain: boolean
  /** 该场景直接子节点数。 */
  readonly children: number
  /** 该 pass 真正画出的 mesh 数（场景图内）。 */
  readonly meshes: number
  readonly calls: number
  readonly triangles: number
}

/** 逐层纹理尺寸 / 估算字节（RGBA8，含 mip ≈ ×4/3）。 */
function collectTextures(): PerfApi["textures"] extends () => infer R
  ? R
  : never {
  const scene = $scene.peek()
  if (!scene) return []
  const out: Array<{
    key: string
    width: number
    height: number
    bytes: number
  }> = []
  for (const [key, mesh] of scene.meshes) {
    for (const material of materialsOf(mesh)) {
      const map = (material as unknown as { map?: Texture | null }).map
      const image = map?.image as
        | { width?: number; height?: number }
        | undefined
      const width = image?.width ?? 0
      const height = image?.height ?? 0
      if (!width || !height) continue
      const bytes = Math.ceil((width * height * 4 * 4) / 3)
      out.push({ key, width, height, bytes })
    }
  }
  return out
}

function materialsOf(mesh: Mesh): readonly Material[] {
  const material = mesh.material
  return Array.isArray(material) ? material : [material]
}

function sceneTriangles(): number {
  const scene = $scene.peek()
  if (!scene) return 0
  let total = 0
  for (const mesh of scene.meshes.values()) {
    const geometry = mesh.geometry
    const count = geometry.index
      ? geometry.index.count
      : (geometry.attributes.position?.count ?? 0)
    total += count / 3
  }
  return Math.round(total)
}

function dumpInfo(
  gl: WebGLRenderer,
  render: PerfDump["render"],
  passes: number,
): PerfDump {
  const textures = collectTextures()
  return {
    render,
    memory: {
      geometries: gl.info.memory.geometries,
      textures: gl.info.memory.textures,
    },
    programs: gl.info.programs?.length ?? 0,
    pixelRatio: gl.getPixelRatio(),
    passes,
    sceneTriangles: sceneTriangles(),
    drawingBuffer: [gl.domElement.width, gl.domElement.height],
    canvasCss: [gl.domElement.clientWidth, gl.domElement.clientHeight],
    textureBytes: textures.reduce((sum, t) => sum + t.bytes, 0),
  }
}

function summarize(
  times: readonly number[],
  render: PerfDump["render"],
  passes: number,
): PerfFrameStats {
  const sorted = [...times].sort((a, b) => a - b)
  const pick = (q: number): number =>
    sorted.length === 0
      ? 0
      : sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * q))]
  const avg = times.reduce((a, b) => a + b, 0) / Math.max(1, times.length)
  return {
    frames: times.length,
    avgMs: avg,
    medianMs: pick(0.5),
    p95Ms: pick(0.95),
    minMs: sorted[0] ?? 0,
    maxMs: sorted[sorted.length - 1] ?? 0,
    fps: avg > 0 ? 1000 / avg : 0,
    passes,
    render,
  }
}

const EMPTY_RENDER: PerfDump["render"] = {
  calls: 0,
  triangles: 0,
  points: 0,
  lines: 0,
  frame: 0,
}

export function PerfProbe() {
  const gl = useThree((state) => state.gl)
  const invalidate = useThree((state) => state.invalidate)
  const setDpr = useThree((state) => state.setDpr)
  const mainScene = useThree((state) => state.scene)
  const remaining = useRef(0)
  const last = useRef(-1)
  const times = useRef<number[]>([])
  const resolve = useRef<((stats: PerfFrameStats) => void) | null>(null)
  /** 上一帧的整帧合计统计（见文件头「gl.info 每次 render 都重置」）。 */
  const lastRender = useRef<PerfDump["render"]>(EMPTY_RENDER)
  const frameCount = useRef(0)
  /** `gl.info.render.frame` 是**累计** `gl.render()` 次数，差分即每帧 pass 数。 */
  const threeFrameAtCapture = useRef(0)
  const lastPasses = useRef(0)

  // `info` 改为按帧手动累计（否则会被 Hud 的第二次 render 重置掉）。
  useEffect(() => {
    gl.info.autoReset = false
    return () => {
      gl.info.autoReset = true
    }
  }, [gl])

  useFrame(() => {
    // priority 0 ⇒ 先于 `Hud`（priority 1）的渲染；此处 `info` 是**上一帧**的合计。
    const threeFrame = gl.info.render.frame
    lastPasses.current = threeFrame - threeFrameAtCapture.current
    threeFrameAtCapture.current = threeFrame
    lastRender.current = {
      calls: gl.info.render.calls,
      triangles: gl.info.render.triangles,
      points: gl.info.render.points,
      lines: gl.info.render.lines,
      frame: frameCount.current,
    }
    gl.info.reset()
    frameCount.current += 1

    if (remaining.current <= 0) return
    const now = performance.now()
    if (last.current >= 0) times.current.push(now - last.current)
    last.current = now
    remaining.current -= 1
    if (remaining.current > 0) {
      // 在 useFrame 内 invalidate ⇒ r3f 会多排一帧（`frames = 2`），形成连续渲染。
      invalidate()
      return
    }
    const stats = summarize(
      times.current,
      lastRender.current,
      lastPasses.current,
    )
    times.current = []
    last.current = -1
    const done = resolve.current
    resolve.current = null
    done?.(stats)
  })

  useEffect(() => {
    const api: PerfApi = {
      // `dump()` 读**活着的** `gl.info`（= 最后一帧的真实合计），
      // 而不是 `lastRender`（那是「上一帧」的快照，空闲时会差一帧）。
      dump: () =>
        dumpInfo(
          gl,
          {
            calls: gl.info.render.calls,
            triangles: gl.info.render.triangles,
            points: gl.info.render.points,
            lines: gl.info.render.lines,
            frame: frameCount.current,
          },
          Math.max(1, gl.info.render.frame - threeFrameAtCapture.current),
        ),
      textures: () => collectTextures(),
      benchmark: (frames = 180) =>
        new Promise<PerfFrameStats>((res) => {
          times.current = []
          last.current = -1
          remaining.current = Math.max(2, Math.floor(frames))
          resolve.current = res
          invalidate()
        }),
      dpr: (pixelRatio) => setDpr(pixelRatio),
      perPass: () =>
        new Promise<PerfPass[]>((resolve) => {
          const original = gl.render
          const records: PerfPass[] = []
          let settled = false
          const finish = (): void => {
            if (settled) return
            settled = true
            gl.render = original
            resolve(records)
          }
          gl.render = function (
            this: WebGLRenderer,
            scene: Object3D,
            camera: Camera,
          ): void {
            const beforeCalls = gl.info.render.calls
            const beforeTris = gl.info.render.triangles
            original.call(gl, scene, camera)
            let meshes = 0
            scene.traverse((o) => {
              if ((o as Mesh).isMesh) meshes += 1
            })
            records.push({
              sceneType: scene.type,
              sceneName: scene.name || "(unnamed)",
              isMain: scene === mainScene,
              children: scene.children.length,
              meshes,
              calls: gl.info.render.calls - beforeCalls,
              triangles: gl.info.render.triangles - beforeTris,
            })
            if (records.length >= 4) finish()
          }
          window.setTimeout(finish, 1200)
          invalidate()
        }),
    }
    const target = window as unknown as { __spatialPerf?: PerfApi }
    target.__spatialPerf = api
    return () => {
      if (target.__spatialPerf === api) delete target.__spatialPerf
    }
  }, [gl, invalidate, setDpr, mainScene])

  return null
}
