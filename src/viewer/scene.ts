/**
 * GLB -> `LoadedScene`（three 对象树 + 元数据）。
 *
 * ── 在全链路里的位置 ──
 * ```
 * node: image -> splat -> layered RGBAD -> mesh -> scene.glb
 * web:  scene.glb -> [本模块] -> LoadedScene -> r3f 渲染
 * ```
 * web 端**只面对 GLB**，不碰 layering / meshing / WebGPU。这个文件是 web 侧的边界。
 *
 * ── 三个约定（都来自 `scripts/pack/glb.ts`，不可偏离）──
 * 1. **几何已 bake 到 glTF 系**（Y 上 / −Z 前），相机在原点、朝 −Z ——
 *    正好是 three 的相机默认朝向，所以「参考视角」= `camera.position = (0,0,0)`。
 * 2. **材质是 `KHR_materials_unlit` + `alphaMode: BLEND` + `doubleSided`**。
 *    GLTFLoader 会给出 `MeshBasicMaterial`，并且**已经**设了
 *    `transparent = true` / `depthWrite = false`（这正是层栈要的画家算法口径）。
 *    这里只再钉三件事：`toneMapped = false`（照片色不能被色调映射动）、
 *    `renderOrder`（glTF 没有 renderOrder，只能渲染端按 `layerIndex` 补）、
 *    以及 `side`（裙边断壁是背面）。
 * 3. **分层由 `extras.layerIndex` 决定**，不看名字（glTF 名字会被 three 的
 *    `createUniqueName` 改名，不可靠）。
 *
 * ── 为什么自己 fetch 而不是 `loader.loadAsync(url)` ──
 * 自取 `ArrayBuffer` 能拿到两样 `FileLoader` 给不了的东西：**总字节数**（UI 要显示大小）
 * 与**真进度**（`content-length` 缺失时也能退化成不定长）。而且绕开 three 的
 * 全局 FileLoader 缓存 —— 换场景时不会有一份旧 buffer 被缓存按住。
 */

import type { Material, Mesh, Object3D } from "three"
import { Box3, DoubleSide, Vector3 } from "three"
import { DRACOLoader } from "three/examples/jsm/loaders/DRACOLoader.js"
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js"
import type {
  LayerState,
  LoadedScene,
  SceneBounds,
  SceneMeta,
  Triple,
} from "./types.ts"

/**
 * Draco decoder 的**本地**路径（`public/draco/`，由 `npm run sync:assets` 从
 * `three/examples/jsm/libs/draco/gltf/` 拷来）。不走 Google CDN：离线可用、版本锁死。
 */
export const DRACO_DECODER_PATH = "/draco/"

/** 内置样例（`npx tsx scripts/pack/sample.ts` 生成；不存在时安静跳过）。 */
export const SAMPLE_URL = "/models/sample.glb"

let loader: GLTFLoader | null = null

/** 单例 loader（DRACOLoader 每次 new 都会重建 worker，必须复用）。 */
function getLoader(): GLTFLoader {
  if (loader) return loader
  const draco = new DRACOLoader()
    .setDecoderPath(DRACO_DECODER_PATH)
    .setDecoderConfig({ type: "wasm" })
  loader = new GLTFLoader().setDRACOLoader(draco)
  return loader
}

/** 载入进度回调：`ratio` 为 `null` 表示总长未知（无 `content-length`）。 */
export type LoadProgress = (ratio: number | null) => void

/**
 * 载入一个 GLB，装成 `LoadedScene`。
 *
 * 失败时抛（调用方负责写 `Phase` / `ErrorMessage`）；成功时不负责释放**上一个**场景，
 * 那是调用方的事（`disposeScene`）。
 */
export async function loadGlbScene(
  url: string,
  name: string,
  onProgress?: LoadProgress,
): Promise<LoadedScene> {
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`载入失败：HTTP ${response.status} ${response.statusText}`)
  }
  const declared = Number(response.headers.get("content-length") ?? 0)
  const data = await readAll(
    response,
    declared > 0 ? declared : null,
    onProgress,
  )

  const gltf = await new Promise<Awaited<ReturnType<GLTFLoader["loadAsync"]>>>(
    (resolve, reject) => {
      getLoader().parse(data, "", resolve, reject)
    },
  )

  const root = gltf.scene
  const meshes = new Map<string, Mesh>()
  const entries: { mesh: Mesh; layerIndex: number }[] = []

  root.traverse((object: Object3D) => {
    const mesh = object as Mesh
    if (!(mesh as { isMesh?: boolean }).isMesh) return
    const layerIndex = readNumber(mesh.userData.layerIndex)
    if (layerIndex === null) return
    meshes.set(layerKey(layerIndex), mesh)
    entries.push({ mesh, layerIndex })
  })

  if (entries.length === 0) {
    throw new Error(
      "GLB 里没有带 `extras.layerIndex` 的 mesh —— 不是本管线的产物？",
    )
  }

  // 背衬平面最远，先画；层按 layerIndex 降序（远 -> 近）。glTF 没有 renderOrder，
  // 只能在这里补：画家算法 + `depthWrite = false`（GLTFLoader 对 BLEND 已设）才能
  // 让「有序层栈 + 逐层 α 混合」成立。
  let layerCount = 0
  for (const { layerIndex } of entries) {
    if (layerIndex >= layerCount) layerCount = layerIndex + 1
  }
  for (const { mesh, layerIndex } of entries) {
    mesh.renderOrder = layerIndex < 0 ? 0 : layerCount - layerIndex
    applyMaterialPolicy(mesh)
  }

  const bounds = computeBounds(root)
  const meta = readSceneMeta(root, bounds)
  const layers: LayerState[] = entries
    .map(({ mesh, layerIndex }) => readLayerState(mesh, layerIndex))
    .sort((a, b) => a.layerIndex - b.layerIndex)

  return {
    url,
    name,
    root,
    meshes,
    meta,
    bounds,
    bytes: data.byteLength,
    layers,
  }
}

/** 层键：`layer_00` / `backing`（与导出器同名，但**不依赖** glTF 名字）。 */
export function layerKey(layerIndex: number): string {
  return layerIndex < 0
    ? "backing"
    : `layer_${String(layerIndex).padStart(2, "0")}`
}

/** 流式读完响应体，边读边报进度（返回可直接喂 `GLTFLoader.parse` 的 `ArrayBuffer`）。 */
async function readAll(
  response: Response,
  total: number | null,
  onProgress?: LoadProgress,
): Promise<ArrayBuffer> {
  const body = response.body
  if (!body) {
    const buffer = await response.arrayBuffer()
    onProgress?.(1)
    return buffer
  }
  const reader = body.getReader()
  const chunks: Uint8Array[] = []
  let loaded = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    if (!value) continue
    chunks.push(value)
    loaded += value.byteLength
    onProgress?.(total === null ? null : Math.min(loaded / total, 1))
  }
  const out = new Uint8Array(loaded)
  let offset = 0
  for (const chunk of chunks) {
    out.set(chunk, offset)
    offset += chunk.byteLength
  }
  onProgress?.(1)
  return out.buffer
}

/**
 * 材质口径。GLTFLoader 对 `alphaMode: BLEND` 已设 `transparent = true`、
 * `depthWrite = false`，这里只补它管不到的（色调映射 / 双面 / 重编译标记）。
 */
function applyMaterialPolicy(mesh: Mesh): void {
  const materials: Material[] = Array.isArray(mesh.material)
    ? mesh.material
    : [mesh.material]
  for (const material of materials) {
    material.transparent = true
    material.depthWrite = false
    material.depthTest = true
    // 照片纹理是最终颜色：不能再走 renderer 的色调映射（r3f 默认 ACESFilmic）。
    material.toneMapped = false
    // 裙边断壁是背面；导出器写了 doubleSided，这里再钉一次防上游改动。
    material.side = DoubleSide
    material.needsUpdate = true
  }
}

function readLayerState(mesh: Mesh, layerIndex: number): LayerState {
  const data = mesh.userData
  const depth = readPair(data.depthRange)
  const triangles = readNumber(data.triangleCount) ?? 0
  const vertices = readNumber(data.vertexCount) ?? 0
  const backing = layerIndex < 0
  return {
    key: layerKey(layerIndex),
    name: mesh.name || layerKey(layerIndex),
    kind: backing ? "backing" : "layer",
    layerIndex,
    label: backing ? "背衬平面" : `层 ${layerIndex}`,
    depth: depth ?? [0, 0],
    triangles,
    vertices,
    visible: true,
  }
}

/** 解析 `scene.extras`；缺失 / 形状不对时用几何反推的兜底值。 */
function readSceneMeta(root: Object3D, bounds: SceneBounds): SceneMeta {
  const extras = root.userData as Record<string, unknown>
  const near = readNumber(extras.near) ?? Math.max(bounds.size[2] * 0.1, 0.05)
  const far = readNumber(extras.far) ?? Math.max(near * 4, bounds.size[2] * 2)
  const verticalFOV = readNumber(extras.verticalFOV) ?? Math.PI / 3
  const aspectRatio = readNumber(extras.aspectRatio) ?? 4 / 3
  const layerDepths = readNumbers(extras.layerDepths) ?? []
  const layerRanges = readNumbers(extras.layerRanges) ?? []
  const premultipliedAlpha = extras.premultipliedAlpha === true
  return {
    near,
    far,
    verticalFOV,
    aspectRatio,
    premultipliedAlpha,
    layerDepths,
    layerRanges,
    disparity: readDisparity(extras.stats),
  }
}

function readDisparity(raw: unknown): SceneMeta["disparity"] {
  if (!raw || typeof raw !== "object") return null
  const stats = raw as Record<string, unknown>
  const minimum = readNumber(stats.minimum)
  const maximum = readNumber(stats.maximum)
  if (minimum === null || maximum === null) return null
  const mean = readNumber(stats.disparityMean) ?? (minimum + maximum) / 2
  const variance = readNumber(stats.disparityVariance) ?? 0
  return {
    minimum,
    maximum,
    mean,
    stdDev: Math.sqrt(Math.max(variance, 0)),
    median: readQuantile(stats, 0.5) ?? (minimum + maximum) / 2,
  }
}

/** 在 `stats.quantileProbs/quantiles` 上线性插值取分位数。 */
function readQuantile(
  stats: Record<string, unknown>,
  probability: number,
): number | null {
  const probs = readNumbers(stats.quantileProbs)
  const values = readNumbers(stats.quantiles)
  if (
    !probs ||
    !values ||
    probs.length !== values.length ||
    values.length < 2
  ) {
    return null
  }
  if (probability <= probs[0]) return values[0]
  if (probability >= probs[probs.length - 1]) return values[values.length - 1]
  for (let i = 1; i < probs.length; i++) {
    if (probability > probs[i]) continue
    const span = probs[i] - probs[i - 1]
    const t = span === 0 ? 0 : (probability - probs[i - 1]) / span
    return values[i - 1] + (values[i] - values[i - 1]) * t
  }
  return values[values.length - 1]
}

function computeBounds(root: Object3D): SceneBounds {
  const box = new Box3().setFromObject(root)
  if (box.isEmpty()) {
    return {
      center: [0, 0, 0],
      min: [0, 0, 0],
      max: [0, 0, 0],
      size: [0, 0, 0],
    }
  }
  const center = box.getCenter(new Vector3())
  const size = box.getSize(new Vector3())
  return {
    center: triple(center),
    min: triple(box.min),
    max: triple(box.max),
    size: triple(size),
  }
}

function triple(v: Vector3): Triple {
  return [round(v.x), round(v.y), round(v.z)]
}

function round(x: number): number {
  return Math.round(x * 1e4) / 1e4
}

/** 释放一棵 three 子树（几何 / 材质 / 贴图）。换场景时调用，避免显存泄漏。 */
export function disposeScene(root: Object3D): void {
  root.traverse((object: Object3D) => {
    const mesh = object as Mesh
    if (!(mesh as { isMesh?: boolean }).isMesh) return
    mesh.geometry?.dispose()
    const materials: Material[] = Array.isArray(mesh.material)
      ? mesh.material
      : [mesh.material]
    for (const material of materials) {
      const map = (material as { map?: { dispose?: () => void } | null }).map
      map?.dispose?.()
      material.dispose()
    }
  })
}

function readNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null
}

function readPair(value: unknown): readonly [number, number] | null {
  if (!Array.isArray(value) || value.length < 2) return null
  const lo = readNumber(value[0])
  const hi = readNumber(value[1])
  return lo === null || hi === null ? null : [lo, hi]
}

function readNumbers(value: unknown): readonly number[] | null {
  if (!Array.isArray(value)) return null
  const out: number[] = []
  for (const item of value) {
    const n = readNumber(item)
    if (n === null) return null
    out.push(n)
  }
  return out
}
