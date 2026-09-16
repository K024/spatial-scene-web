/**
 * GLB 查看器的数据源（全局信号）。
 *
 * 本 UI **只是 GLB 的消费者**：不在浏览器里跑 splat / 分层 / 网格化（那些在
 * `scripts/export-glb.ts` 里产出 GLB）。所以这里只做三件事：
 *   1. `loadGlb(url)`：`GLTFLoader` 载入，存下 three 的 `scene` 与相机；
 *   2. 从 GLB **自带的元数据**里抽出层表与取景参数（`scenes[0].extras` +
 *      每层 `mesh.extras`），查看器不依赖任何 sidecar；
 *   3. 换文件时释放上一份 GPU 资源。
 *
 * 相机参数来自 GLB 自己：`cameras[0].perspective`（yfov / aspect / znear / zfar）
 * + `reference_camera` node 的位姿。UI 不再从 PLY / camera.json 推任何东西。
 *
 * 数据源：默认是固定产物路径 `/exports/layered.glb`（与 `npm run export-glb`
 * 对齐）；自定义产物只能通过**文件选择 / 拖入**加载（一条 viewer，不给填路径）。
 */

import "./signals-hook.ts"
import { signal } from "@preact/signals-react"
import * as THREE from "three"
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js"

/** 默认产物路径（与 `scripts/export-glb.ts` 的默认 `--out` 一致）。 */
export const DEFAULT_GLB_URL = "/exports/layered.glb"

export type GlbStatus = "idle" | "loading" | "ready" | "error"

/** 一层（一个 glTF mesh / node）。 */
export interface GlbLayer {
  /** three 的网格（同时也是 node，位置为单位阵）。 */
  readonly mesh: THREE.Mesh
  /** 层号，0 = 最近。 */
  readonly layerIndex: number
  /** 该层深度带（米，相机空间 z），来自 `mesh.extras.depthRange`。 */
  readonly depthRange: readonly [number, number] | null
  readonly triangleCount: number
  /** 原始位置（爆炸视图要复位用）。 */
  readonly basePosition: THREE.Vector3
}

/** 参考相机信息（直接来自 GLB）。 */
export interface GlbCameraInfo {
  readonly fovDeg: number
  readonly aspect: number
  readonly near: number
  readonly far: number
  /** 相机世界位置。 */
  readonly position: readonly [number, number, number]
  /** 相机世界旋转（四元数 xyzw）。 */
  readonly quaternion: readonly [number, number, number, number]
}

/** 查看器需要的全部元数据（换场景时整体替换）。 */
export interface GlbMeta {
  /** 数据源显示名（默认路径或文件名）。 */
  readonly label: string
  readonly bytes: number
  readonly layers: readonly GlbLayer[]
  readonly triangleCount: number
  readonly vertexCount: number
  readonly camera: GlbCameraInfo | null
  /** 参考相机到场景焦点的距离（米）；默认取景枢轴 = 相机沿视线前进这么多。 */
  readonly focusZ: number
  /** 场景包围盒（three 世界系）。 */
  readonly bounds: THREE.Box3
  readonly near: number
  readonly far: number
  /** `scenes[0].extras` 原样（自描述元数据，面板里做只读展示）。 */
  readonly extras: Record<string, unknown>
}

/** 当前数据源（默认产物 URL / 本地文件）。 */
export type GlbSource =
  | { readonly kind: "url"; readonly label: string; readonly url: string }
  | { readonly kind: "file"; readonly label: string; readonly url: null }

export const glbSource = signal<GlbSource>({
  kind: "url",
  label: DEFAULT_GLB_URL,
  url: DEFAULT_GLB_URL,
})
export const glbStatus = signal<GlbStatus>("idle")
export const glbError = signal<string | null>(null)
/** 当前 glTF 场景（交给 `<primitive object>` 渲染）。 */
export const glbScene = signal<THREE.Group | null>(null)
export const glbMeta = signal<GlbMeta | null>(null)

/** 载入默认产物（或显式 URL）。换文件会替换当前场景并释放旧的 GPU 资源。 */
export async function loadGlb(url: string = DEFAULT_GLB_URL): Promise<void> {
  if (glbStatus.peek() === "loading") return
  glbStatus.value = "loading"
  glbError.value = null
  try {
    const loader = new GLTFLoader()
    const [gltf, bytes] = await Promise.all([
      loader.loadAsync(url),
      fetchByteLength(url),
    ])
    commit(gltf.scene, gltf.cameras, url, bytes)
    glbSource.value = { kind: "url", label: url, url }
  } catch (err) {
    glbStatus.value = "error"
    glbError.value = err instanceof Error ? err.message : String(err)
  }
}

/** 载入本地 GLB 文件（拖入 / 文件选择）。 */
export async function loadGlbFromFile(file: File): Promise<void> {
  if (glbStatus.peek() === "loading") return
  if (!/\.glb$/i.test(file.name)) {
    glbStatus.value = "error"
    glbError.value = `只支持 .glb 文件（收到「${file.name}」）`
    return
  }
  glbStatus.value = "loading"
  glbError.value = null
  try {
    const buffer = await file.arrayBuffer()
    const gltf = await new GLTFLoader().parseAsync(buffer, "")
    commit(gltf.scene, gltf.cameras, file.name, buffer.byteLength)
    glbSource.value = { kind: "file", label: file.name, url: null }
  } catch (err) {
    glbStatus.value = "error"
    glbError.value = err instanceof Error ? err.message : String(err)
  }
}

/** 把刚载入的场景接上（先换新、再释放旧）。 */
function commit(
  scene: THREE.Group,
  cameras: readonly THREE.Camera[],
  label: string,
  bytes: number,
): void {
  scene.updateMatrixWorld(true)
  const meta = buildMeta(label, cameras, scene, bytes)
  // 先换新、再释放旧：旧场景此刻还挂在 React 树上，等这一轮重渲染把它卸掉。
  const previous = glbScene.peek()
  glbScene.value = scene
  glbMeta.value = meta
  glbStatus.value = "ready"
  if (previous) disposeObject3D(previous)
}

/** 释放当前场景（页面卸载 / 主动清空时用）。 */
export function disposeGlb(): void {
  const scene = glbScene.peek()
  glbScene.value = null
  glbMeta.value = null
  if (scene) disposeObject3D(scene)
}

function buildMeta(
  label: string,
  cameras: readonly THREE.Camera[],
  scene: THREE.Group,
  bytes: number,
): GlbMeta {
  const layers: GlbLayer[] = []
  scene.traverse((object) => {
    const mesh = object as THREE.Mesh
    if (!(mesh as { isMesh?: boolean }).isMesh) return
    const data = mesh.userData as {
      layerIndex?: number
      depthRange?: [number, number]
    }
    const layerIndex = typeof data.layerIndex === "number" ? data.layerIndex : 0
    const indices = mesh.geometry.getIndex()
    layers.push({
      mesh,
      layerIndex,
      depthRange:
        Array.isArray(data.depthRange) && data.depthRange.length === 2
          ? [data.depthRange[0], data.depthRange[1]]
          : null,
      triangleCount: indices ? Math.floor(indices.count / 3) : 0,
      basePosition: mesh.position.clone(),
    })
  })
  // 远 -> 近的绘制顺序写进 renderOrder（层号 0 = 最近 = 最后画）。
  layers.sort((a, b) => a.layerIndex - b.layerIndex)
  const layerCount = layers.length
  for (const layer of layers) {
    layer.mesh.renderOrder = layerCount - 1 - layer.layerIndex
    layer.mesh.frustumCulled = true
  }

  const extras = (scene.userData ?? {}) as Record<string, unknown>
  const near = numberOr(extras.near, 0.01)
  const far = numberOr(extras.far, 100)
  const bounds = new THREE.Box3().setFromObject(scene)

  return {
    label,
    bytes,
    layers,
    triangleCount: layers.reduce((sum, l) => sum + l.triangleCount, 0),
    vertexCount: layers.reduce(
      (sum, l) => sum + (l.mesh.geometry.getAttribute("position")?.count ?? 0),
      0,
    ),
    camera: readCamera(cameras),
    focusZ: focusDistance(extras, bounds),
    bounds,
    near,
    far,
    extras,
  }
}

/** 从 GLB 的相机取内参 + 位姿（节点是单位阵时即「参考视角」）。 */
function readCamera(cameras: readonly THREE.Camera[]): GlbCameraInfo | null {
  const camera = cameras[0] as THREE.PerspectiveCamera | undefined
  if (
    !camera ||
    !(camera as { isPerspectiveCamera?: boolean }).isPerspectiveCamera
  ) {
    return null
  }
  camera.updateMatrixWorld(true)
  const position = new THREE.Vector3()
  const quaternion = new THREE.Quaternion()
  camera.getWorldPosition(position)
  camera.getWorldQuaternion(quaternion)
  return {
    fovDeg: camera.fov,
    aspect: camera.aspect,
    near: camera.near,
    far: camera.far,
    position: [position.x, position.y, position.z],
    quaternion: [quaternion.x, quaternion.y, quaternion.z, quaternion.w],
  }
}

/** 取景焦点距离：优先用 `extras.layerDepths` 的中位数，否则用包围盒尺度。 */
function focusDistance(
  extras: Record<string, unknown>,
  bounds: THREE.Box3,
): number {
  const depths = extras.layerDepths
  if (Array.isArray(depths) && depths.length > 0) {
    const sorted = [...(depths as number[])].sort((a, b) => a - b)
    const median = sorted[sorted.length >> 1]
    if (Number.isFinite(median) && median > 0) return median
  }
  const size = bounds.getSize(new THREE.Vector3())
  return Math.max(0.5, size.length() * 0.5)
}

function numberOr(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback
}

/** 读文件字节数（面板显示用）；拿不到就 0。 */
async function fetchByteLength(url: string): Promise<number> {
  try {
    const res = await fetch(url, { method: "HEAD" })
    const length = Number(res.headers.get("content-length"))
    return Number.isFinite(length) ? length : 0
  } catch {
    return 0
  }
}

/** 释放一棵 three 子树里的几何 / 材质 / 纹理。 */
function disposeObject3D(root: THREE.Object3D): void {
  const materials = new Set<THREE.Material>()
  root.traverse((object) => {
    const mesh = object as THREE.Mesh
    mesh.geometry?.dispose()
    const material = mesh.material
    if (Array.isArray(material)) {
      for (const m of material) materials.add(m)
    } else if (material) {
      materials.add(material)
    }
  })
  const textures = new Set<THREE.Texture>()
  for (const material of materials) {
    for (const value of Object.values(material)) {
      if ((value as THREE.Texture | null)?.isTexture) {
        textures.add(value as THREE.Texture)
      }
    }
    material.dispose()
  }
  for (const texture of textures) texture.dispose()
}
