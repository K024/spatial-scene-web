/**
 * meshing golden 门。
 *
 * 阶段：**layered RGBAD -> mesh 场景**（`src/spatial-scene/meshing/`）。
 * 这是本模块的**唯一判定入口**：阈值写死在这里，退出码非 0 即「不过门」。
 * 曲线 / 出图由后续 `meshing-render-views.ts` 负责，它**不判定**。
 *
 * ⚠ 验收口径：
 * - **A3 是硬要求**：深度断层必须被正确撕裂 —— 断言「不存在横跨断层的三角形」，
 *   并且带**反向对照**（把撕裂关掉时确实会出现横跨三角形），否则这条门没有牙。
 * - 其余是**简化项的回归门**（支撑迟滞 / 小岛 / despeckle / 几何合法 / 绕序 / 确定性）。
 * - B 类（同像素多深度的穿插）**不在这里设门**：它属于 layering，meshing 无法恢复。
 *
 * 分两段：
 * - **A 段（纯 CPU，默认跑）**：合成输入上把支撑 / 撕裂 / 几何 / 绕序 / 确定性钉死。
 * - **B 段（GPU，默认跑，`--no-gpu` 跳过）**：真实 PLY 走一遍 splat -> L 层 RGBAD -> mesh，
 *   再用 CPU 三角光栅器（`scripts/utils/meshing-cpu.ts`）在**参考视角**把每层 mesh 光栅化，
 *   与该层 splat 帧比 NCC / MAE / α / 漏覆盖 / 深度 —— M1b 的「画面正确」闭环。
 *
 * 用法：
 *   npx tsx scripts/meshing-golden.ts                          # A + B（example.ply @256）
 *   npx tsx scripts/meshing-golden.ts --no-gpu                 # 只跑 A 段
 *   npx tsx scripts/meshing-golden.ts --ply py-models/out/ply/pier.ply --layers 4 --width 192
 */

import {
  computeDisparityStats,
  ndcDepthFromZ,
} from "../src/spatial-scene/layering/disparity-stats.ts"
import { computeLayerPlacement } from "../src/spatial-scene/layering/placement.ts"
import type {
  DisparityStats,
  LayerSamplingMethod,
} from "../src/spatial-scene/layering/types.ts"
import type { BackfillFrame } from "../src/spatial-scene/meshing/backfill.ts"
import {
  buildBackingPlaneMesh,
  buildLayerRelief,
  buildMeshScene,
  compositeLayersBackToFront,
  downscaleAlphaWeighted,
  expandSupportWithMargin,
  toMeshingInput,
} from "../src/spatial-scene/meshing/index.ts"
import {
  computeDisparityField,
  computeTornEdges,
  resolveTearEps,
} from "../src/spatial-scene/meshing/tears.ts"
import { computeSupportMask } from "../src/spatial-scene/meshing/tolerance.ts"
import type {
  LayerMesh,
  MeshingOptions,
  RgbaTexture,
} from "../src/spatial-scene/meshing/types.ts"
import { createWSplatCamera } from "../src/spatial-scene/wsplat/camera.ts"
import { createWSplatRenderer } from "../src/spatial-scene/wsplat/index.ts"
import type { WSplatFrame } from "../src/spatial-scene/wsplat/types.ts"
import { rasterizeLayerMesh } from "./utils/meshing-cpu.ts"
import { renderLayerStack } from "./utils/meshing-scene.ts"
import { withNodeDevice } from "./utils/webgpu.ts"
import { maskedMae, maskedNcc, toGray } from "./utils/wsplat-metrics.ts"
import type { WSplatScene } from "./utils/wsplat-scene.ts"
import { loadWSplatScene } from "./utils/wsplat-scene.ts"

const NEAR = 0.5
const FAR = 10

// ────────────────────────────── 判定框架 ──────────────────────────────

interface Check {
  name: string
  ok: boolean
  detail: string
}

const checks: Check[] = []

function addCheck(name: string, ok: boolean, detail: string): void {
  checks.push({ name, ok, detail })
  console.log(`  ${ok ? "✓" : "✗"} ${name.padEnd(36)} ${detail}`)
}

// ────────────────────────────── 合成输入 ──────────────────────────────

/** 逐像素几何 + α 的最小对象（支撑与 relief 都只需要这几个字段）。 */
interface PixelFrame {
  width: number
  height: number
  depth: Float32Array
  alpha: Float32Array
  visible: Uint8Array
  /** 直通线性 RGB（`buildLayerRelief` 会把它作为纹理**引用**挂到 LayerMesh 上）。 */
  rgb: Float32Array
}

function makePixels(
  width: number,
  height: number,
  depth: ArrayLike<number>,
  alpha?: ArrayLike<number> | number,
  visible?: ArrayLike<number>,
): PixelFrame {
  const pixels = width * height
  const d = Float32Array.from(depth)
  const a =
    typeof alpha === "number"
      ? new Float32Array(pixels).fill(alpha)
      : alpha
        ? Float32Array.from(alpha)
        : new Float32Array(pixels).fill(1)
  const v = visible ? Uint8Array.from(visible) : new Uint8Array(pixels).fill(1)
  return {
    width,
    height,
    depth: d,
    alpha: a,
    visible: v,
    rgb: new Float32Array(pixels * 3),
  }
}

/** 完整的 `WSplatFrame`（`buildMeshScene` 需要）。 */
function makeFullFrame(pixels: PixelFrame): WSplatFrame {
  const { width, height, depth, alpha, visible } = pixels
  const n = width * height
  const rgb = new Float32Array(n * 3)
  const transmission = new Float32Array(n)
  const accumulatedDepth = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    transmission[i] = 1 - alpha[i]
    accumulatedDepth[i] = alpha[i] * depth[i]
    rgb[i * 3] = 0.5
    rgb[i * 3 + 1] = 0.5
    rgb[i * 3 + 2] = 0.5
  }
  return {
    width,
    height,
    preview: new Uint8Array(n * 4),
    rgb,
    alpha,
    depth,
    transmission,
    accumulatedDepth,
    visible,
  }
}

function makeCamera(
  width: number,
  height: number,
  fx = 8,
): ReturnType<typeof createWSplatCamera> {
  return createWSplatCamera({
    intrinsics: { focalLengthPx: fx, width, height },
    near: NEAR,
    far: FAR,
  })
}

// ────────────────────────────── A0 域与阈值 ──────────────────────────────

function domainChecks(): void {
  const depth = Float32Array.from([1, 2, 3, 5, 10])
  const disp = computeDisparityField(depth, NEAR, FAR)
  let maxDiff = 0
  for (let i = 0; i < depth.length; i++) {
    // `computeDisparityField` 输出是 f32，比较用 fround 把参考值对齐到同一精度。
    const ref = Math.fround(ndcDepthFromZ(depth[i], NEAR, FAR))
    maxDiff = Math.max(maxDiff, Math.abs(disp[i] - ref))
  }
  addCheck(
    "视差场与 ndcDepthFromZ 一致",
    maxDiff === 0,
    `max|Δ| = ${maxDiff.toExponential(2)}（f32）`,
  )

  const mid = resolveTearEps(0.1, { tearScale: 0.4 })
  const floored = resolveTearEps(1e-6, {
    tearScale: 0.4,
    minTearDisparity: 0.002,
  })
  const capped = resolveTearEps(1, { tearScale: 0.4, maxTearDisparity: 0.06 })
  const ok =
    Math.abs(mid - 0.04) < 1e-9 &&
    Math.abs(floored - 0.002) < 1e-9 &&
    Math.abs(capped - 0.06) < 1e-9
  addCheck(
    "tearEps = clamp(scale·band, min, max)",
    ok,
    `mid=${mid} floor=${floored} cap=${capped}`,
  )
}

// ────────────────────────────── A1 迟滞支撑 ──────────────────────────────

function toleranceChecks(): void {
  const w = 8
  const h = 8
  const n = w * h
  const depth = new Float32Array(n).fill(2)
  const visible = new Uint8Array(n).fill(1)
  const alpha = new Float32Array(n)
  // 左半 4 列：强（α=1）；紧邻的第 4 列：软边（α=0.3，应与强连通被保留）；
  // 右下角孤立弱块（α=0.3，不连通 ⇒ 应被丢弃）；α=0.01 的更低像素也应丢弃。
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x
      alpha[i] = x < 4 ? 1 : x === 4 ? 0.3 : 0
    }
  }
  alpha[7 * w + 7] = 0.3 // 孤立弱块
  alpha[7 * w + 6] = 0.3
  alpha[0] = 0.01 // 低于 weakAlpha
  visible[1] = 0 // 无效像素：即使 α 高也排除

  const r = computeSupportMask({ width: w, height: h, depth, alpha, visible })
  const at = (x: number, y: number): number => r.support[y * w + x]
  const softKept = at(4, 0) === 1 && at(4, 7) === 1
  const isolatedDropped = at(7, 7) === 0 && at(6, 7) === 0
  const lowAlphaDropped = at(0, 0) === 0
  const invisibleDropped = at(1, 0) === 0
  addCheck(
    "迟滞：软边随强核心保留、孤立弱块丢弃",
    softKept && isolatedDropped,
    `soft@(4,0)=${at(4, 0)} isolated@(7,7)=${at(7, 7)}`,
  )
  addCheck(
    "无效 / 低 α 像素被排除",
    lowAlphaDropped && invisibleDropped,
    `α=0.01→${at(0, 0)} visible=0→${at(1, 0)}`,
  )

  // 小岛剔除：4×4 强块 + 1px 强点。⚠ 用**全新的** visible，别复用上面把 (1,0) 关掉的那张。
  const visible2 = new Uint8Array(n).fill(1)
  const alpha2 = new Float32Array(n)
  for (let y = 0; y < 4; y++) for (let x = 0; x < 4; x++) alpha2[y * w + x] = 1
  alpha2[7 * w + 7] = 1
  const r2 = computeSupportMask(
    { width: w, height: h, depth, alpha: alpha2, visible: visible2 },
    { minIslandPixels: 4 },
  )
  const r3 = computeSupportMask(
    { width: w, height: h, depth, alpha: alpha2, visible: visible2 },
    { minIslandPixels: 0 },
  )
  addCheck(
    "小岛剔除：1px 岛被删、16px 块保留",
    r2.supportPixels === 16 &&
      r2.removedIslandPixels === 1 &&
      r3.supportPixels === 17,
    `minIsland=4 → ${r2.supportPixels}px（删 ${r2.removedIslandPixels}），关闭 → ${r3.supportPixels}px`,
  )
}

// ────────────────────────────── A3 撕裂正确性（硬要求）──────────────────────────────

/** 合成一道竖直深度断层：`x < split` 近、`x >= split` 远。 */
function stepFrame(
  width: number,
  height: number,
  nearDepth: number,
  farDepth: number,
  split: number,
): PixelFrame {
  const depth = new Float32Array(width * height)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++)
      depth[y * width + x] = x < split ? nearDepth : farDepth
  }
  return makePixels(width, height, depth, 1)
}

/** 由 uv 还原顶点的像素列号。 */
function vertexColumn(mesh: LayerMesh, v: number): number {
  return Math.round(mesh.uvs[v * 2] * mesh.width - 0.5)
}

/** 是否存在横跨 `split` 的三角形（有顶点在两侧）。 */
function trianglesSpanning(mesh: LayerMesh, split: number): number {
  let spans = 0
  for (let t = 0; t < mesh.triangleCount; t++) {
    let minX = Number.POSITIVE_INFINITY
    let maxX = Number.NEGATIVE_INFINITY
    for (let k = 0; k < 3; k++) {
      const x = vertexColumn(mesh, mesh.indices[t * 3 + k])
      if (x < minX) minX = x
      if (x > maxX) maxX = x
    }
    if (minX < split && maxX >= split) spans++
  }
  return spans
}

function tearChecks(): void {
  const w = 8
  const h = 8
  const split = 4
  const frame = stepFrame(w, h, 1.0, 3.0, split)
  const camera = makeCamera(w, h)
  const bandWidth = 0.4
  const support = computeSupportMask(frame).support
  const disparity = computeDisparityField(frame.depth, NEAR, FAR)
  const torn = computeTornEdges(disparity, support, w, h, bandWidth)
  const { mesh } = buildLayerRelief(
    frame,
    camera,
    support,
    torn,
    0,
    [0, bandWidth],
    [1, 3],
  )

  // 断层处的横向边必须被撕开：断开的是 (3,y)-(4,y)。
  let tornAtStep = 0
  for (let y = 0; y < h; y++)
    tornAtStep += torn.horizontal[y * (w - 1) + (split - 1)]
  const noSpan = trianglesSpanning(mesh, split)

  // 反向对照：把撕裂阈值抬到远超台阶差，撕裂应失效、出现横跨三角形。
  const tornOff = computeTornEdges(disparity, support, w, h, bandWidth, {
    tearScale: 1e6,
    maxTearDisparity: 1e6,
    minTearSegmentLength: 1,
  })
  const { mesh: meshNoTear } = buildLayerRelief(
    frame,
    camera,
    support,
    tornOff,
    0,
    [0, bandWidth],
    [1, 3],
  )
  const spansNoTear = trianglesSpanning(meshNoTear, split)

  addCheck(
    "★ 断层被撕开（断开处共 8 条边）",
    tornAtStep === h,
    `在 split=${split} 撕开 ${tornAtStep}/${h} 条横向边`,
  )
  addCheck(
    "★ 无三角形横跨断层",
    noSpan === 0,
    `横跨三角形 = ${noSpan}（共 ${mesh.triangleCount} 个）`,
  )
  addCheck(
    "★ 反向对照：关掉撕裂后确实横跨",
    spansNoTear > 0,
    `关撕裂后横跨 = ${spansNoTear}（证明这条门有牙）`,
  )
}

// ────────────────────────────── A4 despeckle ──────────────────────────────

function despeckleChecks(): void {
  const w = 8
  const h = 8
  const depth = new Float32Array(w * h).fill(2)
  depth[4 * w + 4] = 4.0 // 单个孤立深度尖峰（与 2.0 的视差差 0.13 > tearEps 0.06）
  const frame = makePixels(w, h, depth, 1)
  const support = computeSupportMask(frame).support
  const disparity = computeDisparityField(frame.depth, NEAR, FAR)

  const kept = computeTornEdges(disparity, support, w, h, 0.4, {
    minTearSegmentLength: 6,
  })
  const raw = computeTornEdges(disparity, support, w, h, 0.4, {
    minTearSegmentLength: 1,
  })
  // 尖峰四周 4 条边；30° 台阶的连通分量（绕尖峰）应被 despeckle 整段清掉。
  addCheck(
    "despeckle：孤立尖峰的假缝被取消",
    raw.tornCount === 4 && kept.tornCount === 0 && kept.despeckledCount === 4,
    `原始 ${raw.tornCount} → despeckle 后 ${kept.tornCount}（清 ${kept.despeckledCount}）`,
  )

  // 真实断层（横跨整幅）必须活下来：8 条边 ≥ 6。
  const step = stepFrame(w, h, 1.0, 3.0, 4)
  const stepSupport = computeSupportMask(step).support
  const stepDisp = computeDisparityField(step.depth, NEAR, FAR)
  const stepTorn = computeTornEdges(stepDisp, stepSupport, w, h, 0.4, {
    minTearSegmentLength: 6,
  })
  addCheck(
    "despeckle 不误杀真实断层",
    stepTorn.tornCount === h,
    `整幅断层保留 ${stepTorn.tornCount}/${h} 条边`,
  )
}

// ────────────────────────────── A5 几何不变量 ──────────────────────────────

function geometryChecks(): void {
  const w = 8
  const h = 8
  const frame = makePixels(w, h, new Float32Array(w * h).fill(2), 1)
  const camera = makeCamera(w, h)
  const support = computeSupportMask(frame).support
  const disparity = computeDisparityField(frame.depth, NEAR, FAR)
  const torn = computeTornEdges(disparity, support, w, h, 0.4)
  const { mesh } = buildLayerRelief(
    frame,
    camera,
    support,
    torn,
    0,
    [0, 1],
    [2, 2],
  )

  const quadCount = (w - 1) * (h - 1)
  addCheck(
    "顶点/三角形计数（满支撑、无撕裂）",
    mesh.vertexCount === w * h && mesh.triangleCount === quadCount * 2,
    `顶点 ${mesh.vertexCount}，三角 ${mesh.triangleCount}（期望 ${w * h} / ${quadCount * 2}）`,
  )

  let indicesOk = true
  let windingOk = true
  for (let t = 0; t < mesh.triangleCount; t++) {
    const a = mesh.indices[t * 3]
    const b = mesh.indices[t * 3 + 1]
    const c = mesh.indices[t * 3 + 2]
    if (a >= mesh.vertexCount || b >= mesh.vertexCount || c >= mesh.vertexCount)
      indicesOk = false
    const nz = triangleNormalZ(mesh, a, b, c)
    if (!(nz < 0)) windingOk = false
  }
  addCheck(
    "索引全部落在顶点范围内",
    indicesOk,
    `triangleCount=${mesh.triangleCount}`,
  )
  addCheck("绕序：几何法线朝向相机（nz < 0）", windingOk, "全部三角形")

  // 反投影抽查：像素 (x,y) 的有效 z=2 应给出 ((x+0.5−cx)/fx·z, (y+0.5−cy)/fy·z, z)
  const fx = w / (2 * Math.tan(camera.fovX / 2))
  const fy = h / (2 * Math.tan(camera.fovY / 2))
  let maxPosErr = 0
  for (let v = 0; v < mesh.vertexCount; v++) {
    const x = Math.round(mesh.uvs[v * 2] * w - 0.5)
    const y = Math.round(mesh.uvs[v * 2 + 1] * h - 0.5)
    const ex = ((x + 0.5 - w / 2) / fx) * 2
    const ey = ((y + 0.5 - h / 2) / fy) * 2
    maxPosErr = Math.max(
      maxPosErr,
      Math.abs(mesh.positions[v * 3] - ex),
      Math.abs(mesh.positions[v * 3 + 1] - ey),
      Math.abs(mesh.positions[v * 3 + 2] - 2),
    )
  }
  addCheck(
    "反投影与同内参一致",
    maxPosErr < 1e-5,
    `max|Δ| = ${maxPosErr.toExponential(2)}`,
  )

  const again = buildLayerRelief(
    frame,
    camera,
    support,
    torn,
    0,
    [0, 1],
    [2, 2],
  ).mesh
  addCheck(
    "确定性（两次构建逐位一致）",
    sameF32(mesh.positions, again.positions) &&
      sameF32(mesh.uvs, again.uvs) &&
      sameU32(mesh.indices, again.indices),
    "positions / uvs / indices",
  )
}

function triangleNormalZ(
  mesh: LayerMesh,
  a: number,
  b: number,
  c: number,
): number {
  const ax = mesh.positions[a * 3]
  const ay = mesh.positions[a * 3 + 1]
  const bx = mesh.positions[b * 3] - ax
  const by = mesh.positions[b * 3 + 1] - ay
  const cx = mesh.positions[c * 3] - ax
  const cy = mesh.positions[c * 3 + 1] - ay
  return bx * cy - by * cx
}

// ────────────────────────────── A6 对角翻转 ──────────────────────────────

function diagonalFlipChecks(): void {
  const w = 2
  const h = 2
  const depth = Float32Array.from([1, 3, 3, 5]) // 00=近, 11=远；10/01 等距
  const frame = makePixels(w, h, depth, 1)
  // 2×2 帧只有 4 像素 < 默认小岛阈值 32，必须关掉小岛剔除。
  const support = computeSupportMask(frame, { minIslandPixels: 0 }).support
  const disparity = computeDisparityField(frame.depth, NEAR, FAR)
  const tornOff = computeTornEdges(disparity, support, w, h, 1e6, {
    tearScale: 1e6,
    maxTearDisparity: 1e6,
    minTearSegmentLength: 1,
  })
  const camera = makeCamera(w, h)
  const flip = buildLayerRelief(
    frame,
    camera,
    support,
    tornOff,
    0,
    [0, 1],
    [1, 5],
    {
      allowDiagonalFlip: true,
    },
  ).mesh
  const noFlip = buildLayerRelief(
    frame,
    camera,
    support,
    tornOff,
    0,
    [0, 1],
    [1, 5],
    {
      allowDiagonalFlip: false,
    },
  ).mesh

  // 顶点 = 像素行主序：0=(0,0) 1=(1,0) 2=(0,1) 3=(1,1)。
  // 翻转 → 共边 {1,2}（10–01，短）；不翻转 → 共边 {0,3}（00–11，长）。
  const flipShared = sharedEdge(flip.indices)
  const noFlipShared = sharedEdge(noFlip.indices)
  addCheck(
    "对角翻转：选更短的三维对角线",
    sameSet(flipShared, [1, 2]) && sameSet(noFlipShared, [0, 3]),
    `flip 共边 {${flipShared}}，no-flip 共边 {${noFlipShared}}`,
  )
}

function sharedEdge(indices: Uint32Array): [number, number] {
  const edges = new Map<string, number>()
  for (let t = 0; t < indices.length / 3; t++) {
    const tri = [indices[t * 3], indices[t * 3 + 1], indices[t * 3 + 2]]
    for (let k = 0; k < 3; k++) {
      const a = tri[k]
      const b = tri[(k + 1) % 3]
      const key = a < b ? `${a},${b}` : `${b},${a}`
      edges.set(key, (edges.get(key) ?? 0) + 1)
    }
  }
  for (const [key, count] of edges) {
    if (count === 2) {
      const [a, b] = key.split(",").map(Number)
      return [a, b]
    }
  }
  return [-1, -1]
}

function sameSet(a: readonly number[], b: readonly number[]): boolean {
  if (a.length !== b.length) return false
  const sortedA = [...a].sort((x, y) => x - y)
  const sortedB = [...b].sort((x, y) => x - y)
  return sortedA.every((v, i) => v === sortedB[i])
}

// ────────────────────────────── A7 端到端装配 ──────────────────────────────

function integrationChecks(): void {
  const w = 8
  const h = 8
  const depth = new Float32Array(w * h)
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) depth[y * w + x] = 1 + (x / (w - 1)) * 2 // 1..3
  }
  const frame = makePixels(w, h, depth, 0.8)
  const full = makeFullFrame(frame)
  const camera = makeCamera(w, h)
  const stats: DisparityStats = computeDisparityStats(full.depth, {
    near: NEAR,
    far: FAR,
    binCount: 64,
  })
  const L = 2
  const placement = computeLayerPlacement(stats, {
    L,
    method: "quantile",
    near: NEAR,
    far: FAR,
  })
  const options: MeshingOptions = {}
  const ranges = new Float32Array(L * 2)
  const scene = buildMeshScene(
    {
      L,
      width: w,
      height: h,
      near: NEAR,
      far: FAR,
      placement,
      ranges,
      frames: [full, full],
      stats,
      camera,
    },
    options,
  )

  const layersOk =
    scene.layers.length === L &&
    scene.layers[0].layerIndex === 0 &&
    scene.layers[1].layerIndex === 1
  addCheck("端到端：层数 / 层号正确", layersOk, `layers=${scene.layers.length}`)

  // 内存复用：输出字段应**引用**输入，而不是拷贝。
  const reusesInputs =
    scene.layers.every(
      (layer) =>
        layer.texture.rgb === full.rgb && layer.texture.alpha === full.alpha,
    ) &&
    scene.layerDepths === placement.layerDepthsZ &&
    scene.layerRanges === ranges &&
    scene.stats === stats
  addCheck(
    "输出复用输入引用（层纹理 / 层深 / 层范围 / 统计）",
    reusesInputs,
    "texture.rgb/.alpha、layerDepths、layerRanges、stats 均 === 输入",
  )

  const metaOk =
    Math.abs(scene.verticalFOV - camera.fovY) < 1e-9 &&
    Math.abs(scene.aspectRatio - w / h) < 1e-9 &&
    scene.layerDepths.length === L &&
    scene.premultipliedAlpha === false &&
    scene.report.layers.length === L
  addCheck(
    "元数据：FoV / aspect / 深度 / 直通 α / report",
    metaOk,
    `fovY=${scene.verticalFOV.toFixed(4)} aspect=${scene.aspectRatio}`,
  )

  const reportsSane = scene.report.layers.every(
    (r) =>
      r.supportPixels > 0 &&
      r.tearEps > 0 &&
      r.quadsEmitted + r.quadsSkipped === (w - 1) * (h - 1),
  )
  addCheck("诊断计数自洽（发射+跳过 = 四边形总数）", reportsSane, "")

  const backing = scene.backingPlane
  addCheck(
    "端到端：默认生成背衬平面（LayerMesh, layerIndex=-1, 4v/2t）",
    backing !== null &&
      backing.layerIndex === -1 &&
      backing.vertexCount === 4 &&
      backing.triangleCount === 2 &&
      backing.texture.width > 0,
    backing
      ? `深度 ${backing.depthRange[0].toFixed(3)}m，纹理 ${backing.texture.width}x${backing.texture.height}`
      : "null",
  )
}

// ────────────────────────────── 工具 ──────────────────────────────

function sameF32(a: Float32Array, b: Float32Array): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false
  return true
}

function sameU32(a: Uint32Array, b: Uint32Array): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false
  return true
}

// ────────────────────────────── A8 裙边 / 断壁（M2）──────────────────────────────

function skirtChecks(): void {
  const w = 8
  const h = 8
  const frame = makePixels(w, h, new Float32Array(w * h).fill(2), 1)
  const camera = makeCamera(w, h)
  const support = computeSupportMask(frame, { minIslandPixels: 0 }).support
  const disparity = computeDisparityField(frame.depth, NEAR, FAR)
  const torn = computeTornEdges(disparity, support, w, h, 0.4)
  const { mesh, stats } = buildLayerRelief(
    frame,
    camera,
    support,
    torn,
    0,
    [0, 1],
    [2, 2],
    {},
    { disparity, near: NEAR, far: FAR, skirtDisparity: 0.02 },
  )
  // 满支撑 8×8 的网格边界边 = 外框 = 2(w−1) + 2(h−1)。
  const perimeter = 2 * (w - 1) + 2 * (h - 1)
  const surfaceTris = (w - 1) * (h - 1) * 2
  addCheck(
    "裙边：边界边 = 外框，墙 = 2×边界边",
    stats.boundaryEdges === perimeter &&
      stats.wallTriangleCount === perimeter * 2 &&
      mesh.triangleCount === surfaceTris + perimeter * 2,
    `边界 ${stats.boundaryEdges}（期望 ${perimeter}），墙三角 ${stats.wallTriangleCount}`,
  )
  addCheck(
    "裙边：每个边界像素一个 skirt 顶点",
    mesh.vertexCount === w * h + perimeter,
    `顶点 ${mesh.vertexCount}（期望 ${w * h + perimeter}）`,
  )
  // 墙顶点在后：skirt 顶点的 z 应 > 表面 z=2。墙三角形排在 indices 末尾。
  const wallStart = mesh.triangleCount - mesh.wallTriangleCount
  let wallBehind = true
  const surfaceVerts = new Set<number>()
  for (let t = 0; t < wallStart; t++) {
    surfaceVerts.add(mesh.indices[t * 3])
    surfaceVerts.add(mesh.indices[t * 3 + 1])
    surfaceVerts.add(mesh.indices[t * 3 + 2])
  }
  for (let t = wallStart; t < mesh.triangleCount; t++) {
    for (let k = 0; k < 3; k++) {
      const v = mesh.indices[t * 3 + k]
      if (mesh.positions[v * 3 + 2] < 2 - 1e-6) wallBehind = false
    }
  }
  addCheck(
    "裙边：墙三角形在表面之后（z 向后）",
    wallBehind,
    `墙顶点 z ≥ 2，表面顶点集大小 ${surfaceVerts.size}`,
  )
}

// ────────────────────────────── A9 外缘余量（M2）──────────────────────────────

function marginChecks(): void {
  const w = 9
  const h = 9
  const support = new Uint8Array(w * h)
  const center = 4 * w + 4
  support[center] = 1
  const m1 = expandSupportWithMargin(support, w, h, { radiusPixels: 1 })
  let count1 = 0
  let sourceOk = true
  for (let i = 0; i < w * h; i++) {
    count1 += m1.mask[i]
    if (m1.mask[i] && m1.source[i] !== center) sourceOk = false
  }
  const m2 = expandSupportWithMargin(support, w, h, { radiusPixels: 2 })
  let count2 = 0
  for (let i = 0; i < w * h; i++) count2 += m2.mask[i]
  addCheck(
    "余量：radius=1 变成 5 像素（十字），源全指向中心",
    count1 === 5 && m1.marginPixels === 4 && sourceOk,
    `mask=${count1} margin=${m1.marginPixels} sourceOk=${sourceOk}`,
  )
  addCheck(
    "余量：radius=2 变成 13 像素（菱形）",
    count2 === 13,
    `mask=${count2}`,
  )

  // 余量环的顶点：位置在环像素，但 uv/深度取自源。
  const frame = makePixels(w, h, new Float32Array(w * h).fill(2), 0)
  // 只让中心有 α，其余为 0；支撑就是中心一个像素。
  frame.alpha[center] = 1
  const cam = makeCamera(w, h)
  const disparity = computeDisparityField(frame.depth, NEAR, FAR)
  const torn = computeTornEdges(disparity, m1.mask, w, h, 0.4)
  const { mesh } = buildLayerRelief(
    frame,
    cam,
    m1.mask,
    torn,
    0,
    [0, 1],
    [2, 2],
    {},
    { source: m1.source, disparity, near: NEAR, far: FAR },
  )
  // 5 个掩码像素都有顶点；环顶点的 uv 应等于中心（源）的 uv。
  const centerU = (4 + 0.5) / w
  let uvOk = true
  for (let v = 0; v < mesh.vertexCount; v++) {
    if (Math.abs(mesh.uvs[v * 2] - centerU) > 1e-6) uvOk = false
  }
  addCheck(
    "余量：环顶点 uv 取自源像素",
    mesh.vertexCount === 5 && uvOk,
    `顶点 ${mesh.vertexCount}，uvOk=${uvOk}`,
  )
}

// ────────────────────────────── A10 回填 / 背衬平面（M3）──────────────────────────────

function backfillChecks(): void {
  const w = 4
  const h = 4
  const n = w * h
  // 近层：白，α=0.5；远层：黑，α=1。合成应为灰 0.5、α=1。
  const near: BackfillFrame = {
    width: w,
    height: h,
    rgb: new Float32Array(n * 3).fill(1),
    alpha: new Float32Array(n).fill(0.5),
  }
  const far: BackfillFrame = {
    width: w,
    height: h,
    rgb: new Float32Array(n * 3),
    alpha: new Float32Array(n).fill(1),
  }
  const comp = compositeLayersBackToFront([near, far], w, h)
  const overOk =
    Math.abs(comp.rgb[0] - 0.5) < 1e-6 && Math.abs(comp.alpha[0] - 1) < 1e-6
  addCheck(
    "回填：back-to-front over（近 α=0.5 白 over 远黑 = 灰、α=1）",
    overOk,
    `rgb=${comp.rgb[0].toFixed(3)} α=${comp.alpha[0].toFixed(3)}`,
  )
  const down = downscaleAlphaWeighted(comp, 2)
  addCheck(
    "回填：α 加权降采样（4×4 → 2×2，保 α 能量）",
    down.width === 2 && down.height === 2 && Math.abs(down.alpha[0] - 1) < 1e-6,
    `${down.width}x${down.height}, α=${down.alpha[0].toFixed(3)}`,
  )

  const camera = makeCamera(w, h)
  const depth = 5
  const planeTexture: RgbaTexture = {
    width: 1,
    height: 1,
    rgb: new Float32Array(3),
    alpha: new Float32Array(1),
  }
  const plane = buildBackingPlaneMesh(
    camera,
    depth,
    [0, 1],
    [depth, depth],
    planeTexture,
  )
  let zOk = true
  for (let v = 0; v < plane.vertexCount; v++) {
    if (Math.abs(plane.positions[v * 3 + 2] - depth) > 1e-5) zOk = false
  }
  addCheck(
    "背衬平面：4 顶点 / 2 三角，位于指定深度",
    plane.vertexCount === 4 && plane.triangleCount === 2 && zOk,
    `v=${plane.vertexCount} t=${plane.triangleCount} z=${plane.positions[2].toFixed(3)}`,
  )
}

// ────────────────────────────── C 段工具：新视角合成 ──────────────────────────────

/** 把相机沿 `x` 轴平移 `shiftMeters`（保朝向），制造视差 ⇔ 剪影处露出新区域。 */
function createShiftedCamera(
  scene: WSplatScene,
  shiftMeters: number,
): ReturnType<typeof createWSplatCamera> {
  const fx = scene.width / (2 * Math.tan(scene.camera.fovX / 2))
  return createWSplatCamera({
    intrinsics: { focalLengthPx: fx, width: scene.width, height: scene.height },
    position: [shiftMeters, 0, 0],
    near: scene.near,
    far: scene.far,
  })
}

/**
 * 在给定相机下把整个 mesh 场景合成为一张图（预乘累积）。
 *
 * 顺序：先背衬平面（最远背景）→ 层 `L-1`（最远）到 `0`（最近），逐层 `over`。
 * 每层用自己的 RGBAD 纹理光栅化，所以这就是交付渲染器的等价物。
 */
function compositeMeshAtCamera(
  meshScene: ReturnType<typeof buildMeshScene>,
  frames: readonly WSplatFrame[],
  camera: ReturnType<typeof createWSplatCamera>,
): { rgb: Float32Array; alpha: Float32Array } {
  const width = camera.width
  const height = camera.height
  const n = width * height
  const rgb = new Float32Array(n * 3)
  const alpha = new Float32Array(n)

  const over = (
    srcRgb: Float32Array,
    srcAlpha: Float32Array,
    covered: Uint8Array,
  ): void => {
    for (let i = 0; i < n; i++) {
      if (!covered[i]) continue
      const a = srcAlpha[i]
      if (!(a > 0)) continue
      const inv = 1 - a
      rgb[i * 3] = srcRgb[i * 3] * a + rgb[i * 3] * inv
      rgb[i * 3 + 1] = srcRgb[i * 3 + 1] * a + rgb[i * 3 + 1] * inv
      rgb[i * 3 + 2] = srcRgb[i * 3 + 2] * a + rgb[i * 3 + 2] * inv
      alpha[i] = a + alpha[i] * inv
    }
  }

  if (meshScene.backingPlane) {
    const bp = meshScene.backingPlane
    const r = rasterizeLayerMesh(bp, bp.texture, camera)
    over(r.rgb, r.alpha, r.covered)
  }
  for (let k = meshScene.layers.length - 1; k >= 0; k--) {
    const frame = frames[k]
    const r = rasterizeLayerMesh(
      meshScene.layers[k],
      { width, height, rgb: frame.rgb, alpha: frame.alpha },
      camera,
    )
    over(r.rgb, r.alpha, r.covered)
  }
  return { rgb, alpha }
}

// ────────────────────────────── B 段：真实数据光栅化闭环 ──────────────────────────────

/**
 * 真实 PLY 走完整条链路，再用 CPU 光栅器把**每层 mesh** 在参考视角重现该层 splat 帧。
 *
 * 为什么这能当硬门：mesh 与该层帧用**同一台相机**，且顶点就落在像素中心 —— 所以
 * 参考视角下逐像素应当**近乎精确**（差异只来自透视校正与三角形边缘覆盖）。任何
 * 支撑 / 撕裂 / 反投影 / UV / 绕序的错误都会让 NCC 崩掉，而不是「看着还行」。
 */
async function rasterizationChecks(options: {
  plyPath?: string
  width: number
  layers: number
  method: LayerSamplingMethod
}): Promise<void> {
  await withNodeDevice(async (device) => {
    const scene = loadWSplatScene({
      plyPath: options.plyPath,
      width: String(options.width),
    })
    const stack = await renderLayerStack(device, scene, {
      layers: options.layers,
      method: options.method,
    })
    const input = toMeshingInput(stack, scene.camera)
    // 严格参考视角度量用**表面**（关裙边/背衬）：裙边会往遮挡缝里铺几何，把那里的深度
    // 换成墙的深度（对“隐藏缝隙”是对的，但不是“表面重现”）。
    const meshScene = buildMeshScene(input, {
      skirt: false,
      backing: { enabled: false },
    })
    // 带裙边的场景，只验「墙不往前漏」（参考视角仍≈原帧）。
    const skirtScene = buildMeshScene(input)
    const n = scene.width * scene.height
    const name = scene.plyPath.replace(/^.*[\\/]/, "")
    console.log(
      `      ${name}  ${scene.width}x${scene.height}  L=${options.layers}  ${options.method}  ` +
        `near/far ${scene.near.toFixed(3)}/${scene.far.toFixed(3)}m`,
    )
    console.log(
      "      层   支撑px    三角数      NCC       MAE      αMAE   漏α能量  漏≥aLo   深度p99",
    )

    let worstNcc = 1
    let worstMae = 0
    let worstAlphaMae = 0
    let worstCoverageMiss = 0
    let worstEnergyMiss = 0
    let worstMissAboveLo = 0
    let worstSupportMiss = 0
    let worstUsedMiss = 0
    let worstThinShare = 0
    let worstDepthP99 = 0
    let worstSkirtNcc = 1
    let totalTriangles = 0
    let totalSupport = 0

    for (let k = 0; k < meshScene.layers.length; k++) {
      const mesh = meshScene.layers[k]
      const frame = stack.frames[k]
      const raster = rasterizeLayerMesh(
        mesh,
        {
          width: scene.width,
          height: scene.height,
          rgb: frame.rgb,
          alpha: frame.alpha,
        },
        scene.camera,
      )
      // 带裙边的参考视角：墙不得往前漏（NCC 仍≈原帧）。
      {
        const skirtMesh = skirtScene.layers[k]
        const skirtRaster = rasterizeLayerMesh(
          skirtMesh,
          {
            width: scene.width,
            height: scene.height,
            rgb: frame.rgb,
            alpha: frame.alpha,
          },
          scene.camera,
        )
        const skirtMask = new Uint8Array(n)
        let skirtN = 0
        for (let i = 0; i < n; i++) {
          if (frame.visible[i] && raster.covered[i]) {
            skirtMask[i] = 1
            skirtN++
          }
        }
        if (skirtN > 0) {
          const skirtNcc = maskedNcc(
            toGray(frame.rgb, frame.alpha, n),
            toGray(skirtRaster.rgb, skirtRaster.alpha, n),
            skirtMask,
          )
          worstSkirtNcc = Math.min(worstSkirtNcc, skirtNcc)
        }
      }
      // 顶点 = 支撑像素。把它分成两类，避免把「已知拓扑限制」当成 bug：
      // - **used**：被至少一个三角形引用 ⇒ 它的像素中心**必须**被覆盖（正确性）；
      // - **unused**：支撑域里 ≤1px 的细丝（凑不出 2×2 四边形）⇒ 网格无法表达，
      //   是 M1 已知限制（M2 的外缘余量 / 膨胀再解）。
      const vertexAtPixel = new Int32Array(n).fill(-1)
      for (let v = 0; v < mesh.vertexCount; v++) {
        const x = Math.round(mesh.uvs[v * 2] * scene.width - 0.5)
        const y = Math.round(mesh.uvs[v * 2 + 1] * scene.height - 0.5)
        vertexAtPixel[y * scene.width + x] = v
      }
      const used = new Uint8Array(mesh.vertexCount)
      for (let t = 0; t < mesh.indices.length; t++) used[mesh.indices[t]] = 1
      const mask = new Uint8Array(n)
      let visibleCount = 0
      let coveredVisible = 0
      let visibleAlphaSum = 0
      let uncoveredAlphaSum = 0
      let aboveLoCount = 0
      let aboveLoUncovered = 0
      let supportCount = 0
      let supportUncovered = 0
      let usedCount = 0
      let usedUncovered = 0
      let thinCount = 0
      const alphaLo = 0.05
      for (let i = 0; i < n; i++) {
        const a = frame.alpha[i]
        const vi = vertexAtPixel[i]
        if (vi >= 0) {
          supportCount++
          if (used[vi]) {
            usedCount++
            if (!raster.covered[i]) usedUncovered++
          } else {
            thinCount++
          }
          if (!raster.covered[i]) supportUncovered++
        }
        if (!frame.visible[i]) continue
        visibleCount++
        visibleAlphaSum += a
        if (raster.covered[i]) {
          mask[i] = 1
          coveredVisible++
        } else {
          uncoveredAlphaSum += a
        }
        if (a >= alphaLo) {
          aboveLoCount++
          if (!raster.covered[i]) aboveLoUncovered++
        }
      }
      const grayFrame = toGray(frame.rgb, frame.alpha, n)
      const grayRaster = toGray(raster.rgb, raster.alpha, n)
      const ncc =
        coveredVisible > 0 ? maskedNcc(grayFrame, grayRaster, mask) : 0
      const mae =
        coveredVisible > 0 ? maskedMae(frame.rgb, raster.rgb, mask) : 1
      let alphaSum = 0
      let alphaN = 0
      const depthRel: number[] = []
      for (let i = 0; i < n; i++) {
        if (!mask[i]) continue
        alphaSum += Math.abs(frame.alpha[i] - raster.alpha[i])
        alphaN++
        if (frame.depth[i] > 0 && raster.depth[i] > 0) {
          depthRel.push(
            Math.abs(raster.depth[i] - frame.depth[i]) / frame.depth[i],
          )
        }
      }
      const alphaMae = alphaN > 0 ? alphaSum / alphaN : 0
      // 「漏覆盖」不能用 `visible`（α>0）当分母：splat 渲染会留下大量亚阈值边缘像素，
      // 而支撑掩码刻意用 aLo 过滤它们。有意义的度量是：
      // 1. raw：所有 α>0 像素里没被网格覆盖的比例（含亚阈值，仅参考）；
      // 2. energy：丢掉多少 α 能量 = Σ_uncovered α / Σ_visible α（真实损失）；
      // 3. aboveLo：α≥aLo 的像素里没被覆盖的比例（支撑/连通性的漏网）。
      const rawMiss = visibleCount > 0 ? 1 - coveredVisible / visibleCount : 0
      const energyMiss =
        visibleAlphaSum > 0 ? uncoveredAlphaSum / visibleAlphaSum : 0
      const missAboveLo = aboveLoCount > 0 ? aboveLoUncovered / aboveLoCount : 0
      const supportMiss = supportCount > 0 ? supportUncovered / supportCount : 0
      const usedMiss = usedCount > 0 ? usedUncovered / usedCount : 0
      const thinShare = supportCount > 0 ? thinCount / supportCount : 0
      depthRel.sort((a, b) => a - b)
      const depthP99 =
        depthRel.length > 0
          ? depthRel[
              Math.min(depthRel.length - 1, Math.floor(depthRel.length * 0.99))
            ]
          : 0

      worstNcc = Math.min(worstNcc, ncc)
      worstMae = Math.max(worstMae, mae)
      worstAlphaMae = Math.max(worstAlphaMae, alphaMae)
      worstCoverageMiss = Math.max(worstCoverageMiss, rawMiss)
      worstEnergyMiss = Math.max(worstEnergyMiss, energyMiss)
      worstMissAboveLo = Math.max(worstMissAboveLo, missAboveLo)
      worstSupportMiss = Math.max(worstSupportMiss, supportMiss)
      worstUsedMiss = Math.max(worstUsedMiss, usedMiss)
      worstThinShare = Math.max(worstThinShare, thinShare)
      worstDepthP99 = Math.max(worstDepthP99, depthP99)
      totalTriangles += mesh.triangleCount
      totalSupport += meshScene.report.layers[k].supportPixels
      console.log(
        `      ${String(k).padStart(2)}  ${String(meshScene.report.layers[k].supportPixels).padStart(7)}  ` +
          `${String(mesh.triangleCount).padStart(8)}  ` +
          `${ncc.toFixed(5)}  ${mae.toFixed(5)}  ${alphaMae.toFixed(5)}  ` +
          `${(energyMiss * 100).toFixed(3)}%  ${(missAboveLo * 100).toFixed(3)}%  ` +
          `${depthP99.toExponential(2)}  ` +
          `[支撑 ${supportCount}, 细丝 ${thinCount}, used 漏画 ${usedUncovered}/${usedCount}]`,
      )
    }

    console.log(
      `      [合计] 支撑 ${totalSupport}px，三角 ${totalTriangles}，` +
        `最坏 NCC ${worstNcc.toFixed(5)}，最坏 MAE ${worstMae.toFixed(5)}`,
    )
    addCheck(
      "★ 参考视角：逐层 mesh ≈ 层 splat 帧（NCC）",
      worstNcc >= 0.995,
      `最坏层 NCC = ${worstNcc.toFixed(5)}`,
    )
    addCheck(
      "★ 参考视角：颜色 MAE 小",
      worstMae <= 0.01,
      `最坏层 MAE = ${worstMae.toFixed(5)}`,
    )
    addCheck(
      "参考视角：α 保真",
      worstAlphaMae <= 0.01,
      `最坏层 αMAE = ${worstAlphaMae.toFixed(5)}`,
    )
    addCheck(
      "★ 参考视角：used 顶点全部被覆盖（几何自洽）",
      worstUsedMiss === 0,
      `最坏层 used 漏画 = ${(worstUsedMiss * 100).toFixed(4)}%`,
    )
    addCheck(
      "参考视角：≤1px 细丝支撑占比（M1 已知限制，非 bug）",
      worstThinShare <= 0.05,
      `最坏层细丝支撑 = ${(worstThinShare * 100).toFixed(3)}%（未成四边形的支撑）`,
    )
    addCheck(
      "参考视角：α 能量损失小（丢掉的主要是亚阈值像素）",
      worstEnergyMiss <= 0.05,
      `最坏层漏 α 能量 = ${(worstEnergyMiss * 100).toFixed(3)}%（raw 漏覆盖 ${(worstCoverageMiss * 100).toFixed(2)}%）`,
    )
    addCheck(
      "参考视角：α≥aLo 的像素大部分被覆盖",
      worstMissAboveLo <= 0.06,
      `最坏层 α≥0.05 漏覆盖 = ${(worstMissAboveLo * 100).toFixed(3)}%`,
    )
    addCheck(
      "参考视角：网格深度 ≈ splat 深度",
      worstDepthP99 <= 0.01,
      `最坏层深度 p99 相对误差 = ${worstDepthP99.toExponential(2)}`,
    )
    addCheck(
      "★ 参考视角：加裙边后仍≈原帧（墙不往前漏）",
      worstSkirtNcc >= 0.995,
      `最坏层裙边 NCC = ${worstSkirtNcc.toFixed(5)}`,
    )

    // ── C. 新视角：裙边 / 回填的用武之地 ──
    // 相机沿 x 平移，产生视差：剪影处会露出新区域（disocclusion），正是缝/空档出现的地方。
    const shift = scene.near * 0.15
    const novel = createShiftedCamera(scene, shift)
    const gtRenderer = await createWSplatRenderer(device, {
      size: { width: scene.width, height: scene.height },
    })
    gtRenderer.setGaussians(scene.gaussians)
    gtRenderer.setCamera(novel)
    gtRenderer.sort()
    gtRenderer.renderSplats()
    const gt = await gtRenderer.readback()
    gtRenderer.destroy()

    const configs: Array<{
      name: string
      scene: ReturnType<typeof buildMeshScene>
    }> = [
      {
        name: "无裙边无回填",
        scene: buildMeshScene(input, {
          skirt: false,
          backing: { enabled: false },
        }),
      },
      {
        name: "仅裙边    ",
        scene: buildMeshScene(input, { backing: { enabled: false } }),
      },
      { name: "裙边+回填 ", scene: buildMeshScene(input) },
    ]
    const grayGt = toGray(gt.rgb, gt.alpha, n)
    let gtVisible = 0
    for (let i = 0; i < n; i++) if (gt.visible[i]) gtVisible++
    console.log(
      `      [新视角] 平移 ${shift.toFixed(3)}m（近层屏幕位移 ≈ ${((shift / scene.near) * 100).toFixed(0)}% 尺度）`,
    )
    console.log("      配置            漏覆盖(GT可见)  缝区颜色NCC")
    const gapRates: number[] = []
    for (const cfg of configs) {
      const comp = compositeMeshAtCamera(cfg.scene, stack.frames, novel)
      const mask = new Uint8Array(n)
      let gap = 0
      let covered = 0
      const compRgb = new Float32Array(n * 3)
      for (let i = 0; i < n; i++) {
        const a = comp.alpha[i]
        if (gt.visible[i]) {
          if (a < 0.5) gap++
          else {
            mask[i] = 1
            covered++
          }
        }
        if (a > 1e-6) {
          compRgb[i * 3] = comp.rgb[i * 3] / a
          compRgb[i * 3 + 1] = comp.rgb[i * 3 + 1] / a
          compRgb[i * 3 + 2] = comp.rgb[i * 3 + 2] / a
        }
      }
      const grayComp = toGray(compRgb, comp.alpha, n)
      const ncc = covered > 0 ? maskedNcc(grayGt, grayComp, mask) : 0
      const gapRate = gtVisible > 0 ? gap / gtVisible : 0
      gapRates.push(gapRate)
      console.log(
        `      ${cfg.name}  ${(gapRate * 100).toFixed(3)}%        ${ncc.toFixed(5)}`,
      )
    }
    addCheck(
      "★ 新视角：裙边+回填不劣于无处理（gapRate）",
      gapRates[2] <= gapRates[0] + 1e-9,
      `无处理 ${(gapRates[0] * 100).toFixed(3)}% → 裙边+回填 ${(gapRates[2] * 100).toFixed(3)}%`,
    )
    addCheck(
      "新视角：裙边有助于减少缝隙（不劣于无裙边）",
      gapRates[1] <= gapRates[0] + 1e-9,
      `无裙边 ${(gapRates[0] * 100).toFixed(3)}% → 仅裙边 ${(gapRates[1] * 100).toFixed(3)}%`,
    )
  })
}

// ────────────────────────────── main ──────────────────────────────

async function main(): Promise<void> {
  const argv = process.argv.slice(2)
  const argValue = (name: string): string | undefined => {
    const i = argv.indexOf(name)
    return i >= 0 ? argv[i + 1] : undefined
  }
  const noGpu = argv.includes("--no-gpu")
  const width = Number(argValue("--width") ?? 256)
  const layers = Number(argValue("--layers") ?? 8)
  const method = (argValue("--method") ?? "quantile") as LayerSamplingMethod
  const plyPath = argValue("--ply")

  console.log("=".repeat(88))
  console.log("meshing golden 门（layered RGBAD -> mesh）")
  console.log("=".repeat(88))

  console.log("\n[A0] 视差域与撕裂阈值")
  domainChecks()

  console.log("\n[A1/A2] α 迟滞支撑 + 小岛剔除")
  toleranceChecks()

  console.log("\n[A3] ★ 深度断层撕裂（硬要求）")
  tearChecks()

  console.log("\n[A4] despeckle")
  despeckleChecks()

  console.log("\n[A5] 几何不变量（顶点/索引/绕序/反投影/确定性）")
  geometryChecks()

  console.log("\n[A6] 对角翻转")
  diagonalFlipChecks()

  console.log("\n[A7] 端到端装配")
  integrationChecks()

  console.log("\n[A8] 裙边 / 断壁（M2）")
  skirtChecks()

  console.log("\n[A9] 外缘余量（M2）")
  marginChecks()

  console.log("\n[A10] 回填 / 背衬平面（M3）")
  backfillChecks()

  if (noGpu) {
    console.log("\n[B] 真实数据光栅化闭环 —— 已用 --no-gpu 跳过")
  } else {
    console.log(
      "\n[B] 真实数据光栅化闭环（splat -> 层 RGBAD -> mesh -> 参考视角光栅化）",
    )
    await rasterizationChecks({ plyPath, width, layers, method })
  }

  const failed = checks.filter((c) => !c.ok)
  console.log(`\n${"=".repeat(88)}`)
  console.log(
    `共 ${checks.length} 条，通过 ${checks.length - failed.length}，失败 ${failed.length}`,
  )
  if (failed.length > 0) {
    console.log("\n失败项：")
    for (const f of failed) console.log(`   - ${f.name}: ${f.detail}`)
    process.exitCode = 1
  } else {
    console.log("✓ 全部门限通过")
  }
}

main().catch((err) => {
  console.error("\n[错误]", err instanceof Error ? err.message : err)
  if (err instanceof Error && err.stack) console.error(err.stack)
  process.exitCode = 1
})
