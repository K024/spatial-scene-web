/**
 * M2 打包器：**`MeshScene` -> GLB（+ Draco）**。
 *
 * ── 在全链路里的位置 ──
 * ```
 * ... -> MeshScene -> [本模块] -> scene.glb -> three.js (GLTFLoader + DRACOLoader)
 * ```
 * 这是 node 侧交付路径的最后一环。`MeshScene` 是内存数据，web 端从不接触它；
 * web 端只面对 GLB。所以「MeshScene -> 可渲染格式」整体留在 node 侧（D15 修订）。
 *
 * ── 每个 mesh 怎么变 ──
 * 每层（+ 背衬平面）一个 glTF mesh：
 * - `POSITION`：**bake** OpenCV(Y 下, +Z 前) -> glTF(Y 上, −Z 前)，即 `(x, −y, −z)`；
 * - `TEXCOORD_0`：**不翻转**（`MeshScene.uvs` 与 glTF 同为左上原点、v 向下）；
 * - 材质：`KHR_materials_unlit` + `alphaMode: BLEND` + `doubleSided`（裙边断壁是背面），
 *   `baseColorTexture` = 该层 `rgb/alpha` 编码出的 sRGB PNG（直通 α，零预乘）。
 * - `extras.layerIndex`：glTF 没有 `renderOrder`，渲染端读它设 back-to-front 顺序。
 *
 * ── 相机与元数据内置（无 sidecar）──
 * 相机四参数 -> glTF 原生 `cameras[0].perspective`（`yfov` 是**弧度**）；
 * 参考位姿 = 单位变换的相机 node（bake 后 glTF 相机看 −Z，与 three 一致）。
 * `layerDepths / layerRanges / stats` -> `scene.extras`；逐层范围 -> `mesh.extras`。
 * ⚠ `stats.bins`（256 个 float）**不写**：渲染端用不到，且 `extras` 不能放 TypedArray。
 *
 * ── 为什么不用 three 的 GLTFExporter ──
 * 它依赖 canvas / DOM，node 下不可靠。这里用 `@gltf-transform` + `draco3d`（都不碰 DOM）。
 */

import { Document, NodeIO } from "@gltf-transform/core"
import {
  KHRDracoMeshCompression,
  KHRMaterialsUnlit,
} from "@gltf-transform/extensions"
import { draco } from "@gltf-transform/functions"
import draco3d from "draco3d"
import type {
  LayerMesh,
  MeshScene,
  RgbaTexture,
} from "../../src/spatial-scene/meshing/index.ts"
import { sharpFromRgba } from "../utils/image.ts"

/** 打包选项。 */
export interface GlbExportOptions {
  /** 是否 Draco 压缩几何。默认 `true`。关掉 = 任何 three.js 都不需要 decoder。 */
  draco?: boolean
  /** Draco 量化位。position 默认 14（场景尺度 ~米级足够），texcoord 默认 14。 */
  quantizePosition?: number
  quantizeTexcoord?: number
}

/**
 * `MeshScene` -> GLB 字节。
 *
 * 纯函数（除 Draco 的 wasm 初始化），不落盘；写文件由调用方决定。
 */
export async function exportMeshSceneToGlb(
  meshScene: MeshScene,
  options: GlbExportOptions = {},
): Promise<Uint8Array> {
  const doc = new Document()
  doc.getRoot().setExtras({ generator: "spatial-scene" })
  const buffer = doc.createBuffer()
  const unlit = doc.createExtension(KHRMaterialsUnlit)

  const scene = doc.createScene("scene")
  doc.getRoot().setDefaultScene(scene)

  // 先背衬（最远），再层 L-1..0（远 -> 近）。渲染顺序仍以 extras.layerIndex 为准，
  // 这里只是给「按节点顺序画」的查看器一个好默认。
  const ordered: LayerMesh[] = []
  if (meshScene.backingPlane) ordered.push(meshScene.backingPlane)
  for (let k = meshScene.layers.length - 1; k >= 0; k--) {
    ordered.push(meshScene.layers[k])
  }

  for (const mesh of ordered) {
    const node = await buildMeshNode(doc, buffer, unlit, mesh)
    scene.addChild(node)
  }

  // ── 相机（参考位姿 = 单位变换）──
  const camera = doc
    .createCamera("reference")
    .setType("perspective")
    .setYFov(meshScene.verticalFOV)
    .setAspectRatio(meshScene.aspectRatio)
    .setZNear(meshScene.near)
    .setZFar(meshScene.far)
  scene.addChild(doc.createNode("reference_camera").setCamera(camera))

  // ── 元数据 —— scene.extras ──
  const stats = meshScene.stats
  scene.setExtras({
    near: meshScene.near,
    far: meshScene.far,
    verticalFOV: meshScene.verticalFOV,
    aspectRatio: meshScene.aspectRatio,
    premultipliedAlpha: meshScene.premultipliedAlpha,
    layerDepths: Array.from(meshScene.layerDepths),
    layerRanges: Array.from(meshScene.layerRanges),
    stats: {
      sampleSize: stats.sampleSize,
      weightSum: stats.weightSum,
      minimum: stats.minimum,
      maximum: stats.maximum,
      minDepth: stats.minDepth,
      maxDepth: stats.maxDepth,
      disparityMean: stats.disparityMean,
      disparityVariance: stats.disparityVariance,
      disparitySkewness: stats.disparitySkewness,
      disparityKurtosis: stats.disparityKurtosis,
      quantileProbs: Array.from(stats.quantileProbs),
      quantiles: Array.from(stats.quantiles),
    },
  })

  // ── Draco（几何压缩；纹理不受影响）──
  if (options.draco !== false) {
    await doc.transform(
      draco({
        method: "edgebreaker",
        quantizePosition: options.quantizePosition ?? 14,
        quantizeTexcoord: options.quantizeTexcoord ?? 14,
        quantizationVolume: "mesh",
      }),
    )
  }

  const io = new NodeIO()
    .registerExtensions([KHRDracoMeshCompression, KHRMaterialsUnlit])
    .registerDependencies({
      "draco3d.encoder": await draco3d.createEncoderModule(),
      "draco3d.decoder": await draco3d.createDecoderModule(),
    })
  return io.writeBinary(doc)
}

/** 单个 `LayerMesh` -> glTF node（含 primitive / material / texture / extras）。 */
async function buildMeshNode(
  doc: Document,
  buffer: ReturnType<Document["createBuffer"]>,
  unlit: KHRMaterialsUnlit,
  mesh: LayerMesh,
): Promise<ReturnType<Document["createNode"]>> {
  const positions = bakePositions(mesh.positions, mesh.vertexCount)
  // `LayerMesh.uvs/indices` 是 `Float32Array`/`Uint32Array`（默认 `ArrayBufferLike`），
  // 而 gltf-transform 的 `setArray` 要求 `ArrayBuffer` 背书的 typed array。运行时同一对象。
  const uvs = mesh.uvs as Float32Array<ArrayBuffer>
  const indexArray = mesh.indices as Uint32Array<ArrayBuffer>

  const position = doc
    .createAccessor(`${name(mesh)}_POSITION`)
    .setType("VEC3")
    .setArray(positions)
    .setBuffer(buffer)
  const uv = doc
    .createAccessor(`${name(mesh)}_TEXCOORD_0`)
    .setType("VEC2")
    .setArray(uvs)
    .setBuffer(buffer)
  const indices = doc
    .createAccessor(`${name(mesh)}_INDEX`)
    .setType("SCALAR")
    .setArray(indexArray)
    .setBuffer(buffer)

  const texture = doc
    .createTexture(`${name(mesh)}_baseColor`)
    .setImage(await encodeTexture(mesh.texture))
    .setMimeType("image/png")

  const material = doc
    .createMaterial(`${name(mesh)}_material`)
    .setBaseColorTexture(texture)
    .setBaseColorFactor([1, 1, 1, 1])
    .setAlphaMode("BLEND")
    .setDoubleSided(true)
  // 照片纹理 = 已经是最终颜色，不要再走 PBR 光照（否则需要法线且在 three 里会变黑）。
  material.setExtension("KHR_materials_unlit", unlit.createUnlit())

  const primitive = doc
    .createPrimitive()
    .setAttribute("POSITION", position)
    .setAttribute("TEXCOORD_0", uv)
    .setIndices(indices)
    .setMaterial(material)

  const gltfMesh = doc
    .createMesh(name(mesh))
    .addPrimitive(primitive)
    .setExtras({
      layerIndex: mesh.layerIndex,
      disparityRange: [mesh.disparityRange[0], mesh.disparityRange[1]],
      depthRange: [mesh.depthRange[0], mesh.depthRange[1]],
      triangleCount: mesh.triangleCount,
      vertexCount: mesh.vertexCount,
    })

  return doc.createNode(name(mesh)).setMesh(gltfMesh)
}

function name(mesh: LayerMesh): string {
  return mesh.layerIndex < 0
    ? "backing"
    : `layer_${String(mesh.layerIndex).padStart(2, "0")}`
}

/** OpenCV -> glTF：`(x, y, z) -> (x, -y, -z)`（绕 X 轴 180°，Y 上 / −Z 前）。 */
function bakePositions(
  positions: Float32Array,
  vertexCount: number,
): Float32Array<ArrayBuffer> {
  const out = new Float32Array(vertexCount * 3)
  for (let i = 0; i < vertexCount; i++) {
    out[i * 3] = positions[i * 3]
    out[i * 3 + 1] = -positions[i * 3 + 1]
    out[i * 3 + 2] = -positions[i * 3 + 2]
  }
  return out
}

/**
 * 层纹理（**直通线性 RGB + α**）-> sRGB RGBA8 PNG。
 *
 * 这正是 glTF `baseColorTexture` 的约定（sRGB RGB + **线性 α**），零额外色彩空间转换；
 * 代价是 8-bit + 高光 clamp（匹配肉眼验收口径）。
 */
async function encodeTexture(texture: RgbaTexture): Promise<Uint8Array> {
  return new Uint8Array(await rgbaToPng(texture))
}

async function rgbaToPng(texture: RgbaTexture): Promise<Buffer> {
  const rgba = linearToSrgbRgba(texture.rgb, texture.alpha)
  return sharpFromRgba(rgba, texture.width, texture.height).png().toBuffer()
}

/** 直通线性 RGB + α -> sRGB RGBA8（α=0 处置黑，避免半透明像素的杂色）。 */
function linearToSrgbRgba(rgb: Float32Array, alpha: Float32Array): Uint8Array {
  const pixels = alpha.length
  const out = new Uint8Array(pixels * 4)
  for (let i = 0; i < pixels; i++) {
    const a = clamp01(alpha[i])
    const o = i * 3
    out[i * 4] = Math.round(linearToSrgb(rgb[o]) * 255)
    out[i * 4 + 1] = Math.round(linearToSrgb(rgb[o + 1]) * 255)
    out[i * 4 + 2] = Math.round(linearToSrgb(rgb[o + 2]) * 255)
    out[i * 4 + 3] = Math.round(a * 255)
  }
  return out
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x
}

/** 与 `wsplat/wgsl/resolve.ts` 的 `linearToSrgb` 同式（阈值分段，非纯 gamma）。 */
function linearToSrgb(x: number): number {
  const c = clamp01(x)
  return c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055
}
