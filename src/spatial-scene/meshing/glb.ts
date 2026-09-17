/**
 * `MeshScene` -> **GLB**（glTF 2.0 binary，纯 TS、零依赖、不落盘）。
 *
 * ── 在全链路里的位置 ──
 * ```
 * ... -> MeshScene -> [本文件] -> scene.glb -> three.js (GLTFLoader)
 * ```
 *
 * ── 每个 mesh 怎么变 ──
 * 每层一个 glTF mesh / node：
 * - `POSITION`：**bake** OpenCV（y 下、+Z 前）-> glTF（y 上、−Z 前），即 `(x, −y, −z)`；
 *   顶点顺序同时**反转绕序**（镜像会翻转手性，让正面在 glTF 的 CCW 约定下仍然朝相机）；
 * - `TEXCOORD_0`：**不翻转**（本模块 UV 与 glTF 同为左上原点、v 向下）；
 * - 材质：`KHR_materials_unlit`（照片纹理已是最终颜色，不要 PBR 光照）+
 *   `alphaMode: "BLEND"`（直通 α，零预乘）+ `doubleSided`（斜视角看到背面而不是露洞）；
 * - `mesh.extras.layerIndex` / `depthRange`：**层序提示**。glTF 没有 `renderOrder`，
 *   渲染端要读它设 back-to-front（node 顺序已按远→近排，是给「按节点顺序画」的默认值）。
 *
 * ── three.js 接入备忘 ──
 * ```
 * // GLTFLoader 已经把 alphaMode=BLEND 翻成 transparent + depthWrite=false，
 * // KHR_materials_unlit 翻成 MeshBasicMaterial。层序默认已对（节点远→近）；
 * // 若要显式控制（或换其它引擎），按 extras 设置：
 * mesh.renderOrder = layerCount - 1 - mesh.userData.layerIndex
 * ```
 *
 * ── 相机与元数据内置（无 sidecar）──
 * 相机四参数（`yfov` 是**弧度**）-> `cameras[0].perspective`；参考位姿 = 单位变换的
 * 相机 node（bake 后 glTF 相机看 −Z，与 three 一致）。扩视角的
 * `referenceRect / layerDepths / layerRanges` 等 -> `scene.extras`。
 *
 * ── 纹理为什么在这里编码 ──
 * `buildGlb` 保留 PNG 同步路径，供纯 CPU / 无 WASM 环境使用；
 * `buildGlbAsync` 默认输出 KTX2/BasisUniversal，浏览器与 Node 共用同一个 WASM
 * 编码器接口。KTX2 走 `KHR_texture_basisu`，仍保持 unlit + BLEND 语义。
 */

import { encodePngRgba8 } from "./png.ts"
import type { LayerMesh, MeshScene, RgbaTexture } from "./types.ts"

/** GLB 导出选项。 */
export interface GlbExportOptions {
  /** 场景名。默认 `"layered"`。 */
  readonly name?: string
  /** 材质是否双面。默认 `true`（斜视角兜底；关掉可省一半 draw）。 */
  readonly doubleSided?: boolean
  /** 纹理是否编码成 sRGB（glTF `baseColorTexture` 约定）。默认 `true`。 */
  readonly srgbTextures?: boolean
  /**
   * 纹理格式。`buildGlb` 只支持 `"png"`；`buildGlbAsync` 默认 `"ktx2"`。
   */
  readonly textureFormat?: "png" | "ktx2"
  /** KTX2 编码模式。默认 `"etc1s"`；UASTC 质量更高、体积更大。 */
  readonly textureEncoding?: "etc1s" | "uastc"
  /** ETC1S 质量，范围 1..255。默认 180。 */
  readonly textureQuality?: number
}

/** glTF 组件类型常量。 */
const FLOAT = 5126
const UNSIGNED_SHORT = 5123
const UNSIGNED_INT = 5125
const ARRAY_BUFFER = 34962
const ELEMENT_ARRAY_BUFFER = 34963
const DEFAULT_KTX2_QUALITY = 180

/**
 * `MeshScene` -> GLB 字节（纯函数）。
 *
 * 空层（无顶点/三角形）**不写进 GLB**（glTF 不允许 byteLength 0 的 bufferView），
 * 它们只出现在 `scene.extras.skippedLayers` 里。
 */
export function buildGlb(
  scene: MeshScene,
  options: GlbExportOptions = {},
): Uint8Array {
  if (options.textureFormat === "ktx2") {
    throw new Error("KTX2 编码是异步的；请使用 buildGlbAsync()")
  }
  const srgb = options.srgbTextures ?? true
  const baked = scene.layers.map((layer) => {
    const texture = encodePngTexture(layer.texture, srgb)
    return {
      ...bakeGeometry(layer, texture.uvScale),
      texture,
    }
  })
  return assembleGlb(scene, baked, options)
}

/**
 * `MeshScene` -> GLB 字节，默认使用 KTX2/BasisUniversal。
 *
 * 编码器通过动态 import 选择当前平台的 `ktx2-encoder` 导出：Node 使用本地 WASM，
 * 浏览器使用同一包内带的浏览器 WASM。纹理编码串行执行，避免十层 RGBA 同时占用
 * 大块 WASM 堆内存。
 */
export async function buildGlbAsync(
  scene: MeshScene,
  options: GlbExportOptions = {},
): Promise<Uint8Array> {
  const srgb = options.srgbTextures ?? true
  const textureFormat = options.textureFormat ?? "ktx2"
  const baked: BakedLayer[] = []
  for (const layer of scene.layers) {
    const texture =
      textureFormat === "ktx2"
        ? await encodeKtx2Texture(layer.texture, srgb, options)
        : encodePngTexture(layer.texture, srgb)
    baked.push({
      ...bakeGeometry(layer, texture.uvScale),
      texture,
    })
  }
  return assembleGlb(scene, baked, options)
}

function assembleGlb(
  scene: MeshScene,
  baked: BakedLayer[],
  options: GlbExportOptions,
): Uint8Array {
  const doubleSided = options.doubleSided ?? true

  // 渲染顺序默认值：远 -> 近（three 的透明排序在等距时按 id 递增）。
  const order = baked
    .map((b, index) => ({ b, index }))
    .filter(({ b }) => b.indices.length > 0)
    .sort((x, y) => y.index - x.index)

  // ── 2. 二进制缓冲（每个 bufferView 4 字节对齐）──
  const binary = new BinaryWriter()
  const layout = order.map(({ b }) => ({
    position: binary.alignTo4().writeFloat32(b.positions),
    uv: binary.alignTo4().writeBytes(b.uvs),
    index: binary.alignTo4().writeBytes(b.indices),
    image: binary.alignTo4().writeBytes(b.texture.bytes),
  }))
  const bin = binary.finish()

  // ── 3. JSON ──
  const gltf = buildGltfJson({
    scene,
    order,
    layout,
    bin,
    doubleSided,
    options,
  })

  // ── 4. 容器 ──
  return packGlb(gltf, bin)
}

interface BakedLayer {
  readonly positions: Float32Array
  readonly uvs: Uint16Array
  readonly indices: Uint16Array | Uint32Array
  readonly indexComponentType: typeof UNSIGNED_SHORT | typeof UNSIGNED_INT
  readonly texture: EncodedTexture
  readonly layer: LayerMesh
}

interface EncodedTexture {
  readonly bytes: Uint8Array
  readonly mimeType: "image/png" | "image/ktx2"
  readonly basisu: boolean
  readonly uvScale: readonly [number, number]
}

/** 位置 bake + 绕序反转 + UV / index 量化。 */
function bakeGeometry(
  layer: LayerMesh,
  uvScale: readonly [number, number],
): Omit<BakedLayer, "texture"> {
  const positions = new Float32Array(layer.positions.length)
  for (let i = 0; i < layer.vertexCount; i++) {
    positions[i * 3] = layer.positions[i * 3]
    positions[i * 3 + 1] = -layer.positions[i * 3 + 1]
    positions[i * 3 + 2] = -layer.positions[i * 3 + 2]
  }
  // 镜像翻转手性：反转每个三角形的绕序，正面才仍朝相机（CCW）。
  const indices = new Uint32Array(layer.indices.length)
  for (let t = 0; t < layer.indices.length; t += 3) {
    indices[t] = layer.indices[t]
    indices[t + 1] = layer.indices[t + 2]
    indices[t + 2] = layer.indices[t + 1]
  }
  const compactIndices =
    layer.vertexCount <= 0xffff ? toUint16(indices) : indices
  const uvs = new Uint16Array(layer.uvs.length)
  for (let i = 0; i < layer.uvs.length; i++) {
    const scale = i % 2 === 0 ? uvScale[0] : uvScale[1]
    uvs[i] = Math.round(clamp01(layer.uvs[i]) * scale * 0xffff)
  }
  return {
    positions,
    uvs,
    indices: compactIndices,
    indexComponentType:
      compactIndices instanceof Uint16Array ? UNSIGNED_SHORT : UNSIGNED_INT,
    layer,
  }
}

function toUint16(values: Uint32Array): Uint16Array {
  const out = new Uint16Array(values.length)
  for (let i = 0; i < values.length; i++) out[i] = values[i]
  return out
}

/**
 * 直通线性 RGB + α -> sRGB RGBA8 PNG。
 *
 * `alpha == 0` 的像素 RGB 置黑：`resolve` 的直通色在 α→0 时是 `premult/α`，
 * 数值上有噪声，而 mip / 双线性会把它混进可见像素。
 */
function encodePngTexture(texture: RgbaTexture, srgb: boolean): EncodedTexture {
  return {
    bytes: encodePngRgba8(
      textureToRgba8(texture, srgb),
      texture.width,
      texture.height,
    ),
    mimeType: "image/png",
    basisu: false,
    uvScale: [1, 1],
  }
}

async function encodeKtx2Texture(
  texture: RgbaTexture,
  srgb: boolean,
  options: GlbExportOptions,
): Promise<EncodedTexture> {
  const { encodeToKTX2 } = await import("ktx2-encoder")
  const quality = Math.max(
    1,
    Math.min(255, Math.round(options.textureQuality ?? DEFAULT_KTX2_QUALITY)),
  )
  const isUASTC = options.textureEncoding === "uastc"
  const width = align4(texture.width)
  const height = align4(texture.height)
  const bytes = await encodeToKTX2(
    textureToRgba8(texture, srgb, width, height),
    {
      enableDebug: false,
      isKTX2File: true,
      isUASTC,
      needSupercompression: true,
      generateMipmap: true,
      isYFlip: false,
      isPerceptual: srgb,
      isSetKTX2SRGBTransferFunc: srgb,
      qualityLevel: quality,
      uastcLDRQualityLevel: 2,
      enableRDO: isUASTC,
      rdoQualityLevel: 1.5,
      imageDecoder: async (buffer) => ({
        width,
        height,
        data: buffer,
      }),
    },
  )
  return {
    bytes,
    mimeType: "image/ktx2",
    basisu: true,
    uvScale: [texture.width / width, texture.height / height],
  }
}

function textureToRgba8(
  texture: RgbaTexture,
  srgb: boolean,
  width = texture.width,
  height = texture.height,
): Uint8Array {
  const pixels = texture.width * texture.height
  const rgba = new Uint8Array(width * height * 4)
  const { rgb, alpha } = texture
  for (let i = 0; i < pixels; i++) {
    const a = clamp01(alpha[i])
    const x = i % texture.width
    const y = Math.floor(i / texture.width)
    const o = (y * width + x) * 4
    if (a <= 0) {
      rgba[o + 3] = 0
      continue
    }
    for (let c = 0; c < 3; c++) {
      const v = clamp01(rgb[i * 3 + c])
      rgba[o + c] = Math.round((srgb ? linearRGB2sRGBValue(v) : v) * 255)
    }
    rgba[o + 3] = Math.round(a * 255)
  }
  return rgba
}

function align4(value: number): number {
  return Math.ceil(value / 4) * 4
}

/** 单值线性 -> sRGB（与 `sharp/colorspace.ts: linearRGB2sRGB` 同式）。 */
function linearRGB2sRGBValue(x: number): number {
  return x <= 0.0031308 ? x * 12.92 : 1.055 * x ** (1 / 2.4) - 0.055
}

interface JsonLayerLayout {
  readonly position: { byteOffset: number; byteLength: number }
  readonly uv: { byteOffset: number; byteLength: number }
  readonly index: { byteOffset: number; byteLength: number }
  readonly image: { byteOffset: number; byteLength: number }
}

function buildGltfJson(args: {
  scene: MeshScene
  order: { b: BakedLayer; index: number }[]
  layout: JsonLayerLayout[]
  bin: Uint8Array
  doubleSided: boolean
  options: GlbExportOptions
}): Record<string, unknown> {
  const { scene, order, layout, bin, doubleSided, options } = args
  const bufferViews: Record<string, unknown>[] = []
  const accessors: Record<string, unknown>[] = []
  const meshes: Record<string, unknown>[] = []
  const nodes: Record<string, unknown>[] = []
  const materials: Record<string, unknown>[] = []
  const textures: Record<string, unknown>[] = []
  const images: Record<string, unknown>[] = []

  const pushBufferView = (
    view: {
      byteOffset: number
      byteLength: number
    },
    target?: number,
  ): number => {
    bufferViews.push({
      buffer: 0,
      byteOffset: view.byteOffset,
      byteLength: view.byteLength,
      ...(target !== undefined ? { target } : {}),
    })
    return bufferViews.length - 1
  }

  order.forEach(({ b, index }, slot) => {
    const l = layout[slot]
    const name = layerName(index)

    const positionView = pushBufferView(l.position, ARRAY_BUFFER)
    const uvView = pushBufferView(l.uv, ARRAY_BUFFER)
    const indexView = pushBufferView(l.index, ELEMENT_ARRAY_BUFFER)
    const imageView = pushBufferView(l.image)

    const positionMin = [Infinity, Infinity, Infinity]
    const positionMax = [-Infinity, -Infinity, -Infinity]
    for (let i = 0; i < b.positions.length; i += 3) {
      for (let c = 0; c < 3; c++) {
        const v = b.positions[i + c]
        if (v < positionMin[c]) positionMin[c] = v
        if (v > positionMax[c]) positionMax[c] = v
      }
    }

    const positionAccessor = accessors.length
    accessors.push({
      bufferView: positionView,
      componentType: FLOAT,
      count: b.layer.vertexCount,
      type: "VEC3",
      min: positionMin,
      max: positionMax,
    })
    const uvAccessor = accessors.length
    accessors.push({
      bufferView: uvView,
      componentType: UNSIGNED_SHORT,
      count: b.layer.vertexCount,
      type: "VEC2",
      normalized: true,
    })
    const indexAccessor = accessors.length
    accessors.push({
      bufferView: indexView,
      componentType: b.indexComponentType,
      count: b.indices.length,
      type: "SCALAR",
    })

    images.push({
      name: `${name}_baseColor`,
      bufferView: imageView,
      mimeType: b.texture.mimeType,
    })
    const textureIndex = textures.length
    textures.push(
      b.texture.basisu
        ? {
            sampler: 0,
            extensions: {
              KHR_texture_basisu: { source: textureIndex },
            },
          }
        : { source: textureIndex, sampler: 0 },
    )

    const materialIndex = materials.length
    materials.push({
      name: `${name}_material`,
      pbrMetallicRoughness: {
        baseColorFactor: [1, 1, 1, 1],
        baseColorTexture: { index: textureIndex },
        metallicFactor: 0,
        roughnessFactor: 1,
      },
      alphaMode: "BLEND",
      doubleSided,
      extensions: { KHR_materials_unlit: {} },
    })

    meshes.push({
      name,
      primitives: [
        {
          attributes: {
            POSITION: positionAccessor,
            TEXCOORD_0: uvAccessor,
          },
          indices: indexAccessor,
          material: materialIndex,
        },
      ],
      extras: {
        layerIndex: index,
        depthRange: [b.layer.depthRange[0], b.layer.depthRange[1]],
        vertexCount: b.layer.vertexCount,
        triangleCount: b.layer.triangleCount,
      },
    })
    nodes.push({ name, mesh: meshes.length - 1 })
  })

  const cameraNode = nodes.length
  nodes.push({ name: "reference_camera", camera: 0 })

  const sceneExtras: Record<string, unknown> = {
    generator: GENERATOR,
    near: scene.near,
    far: scene.far,
    width: scene.view.width,
    height: scene.view.height,
    premultipliedAlpha: false,
    layerDepths: Array.from(scene.layerDepths),
    layerRanges: Array.from(scene.layerRanges),
    camera: {
      focalLengthPx: scene.view.focalLengthPx,
      referenceFocalLengthPx: scene.view.referenceFocalLengthPx,
      viewScale: scene.view.viewScale,
      pixelScale: scene.view.pixelScale,
      referenceRect: [
        scene.view.referenceRect.x,
        scene.view.referenceRect.y,
        scene.view.referenceRect.width,
        scene.view.referenceRect.height,
      ],
      // 参考相机的位姿（glTF 系）。当前链路 `extrinsics = I`，所以是「原点、朝 −Z」；
      // 显式写出来，查看器就不必依赖「节点没有 transform 即单位阵」这条隐含约定。
      position: [0, 0, 0],
      rotation: [0, 0, 0, 1],
    },
    layerOrder: order.map(({ index }) => layerName(index)),
    skippedLayers: scene.layers
      .map((layer, index) => ({ layer, index }))
      .filter(({ layer }) => layer.triangleCount === 0)
      .map(({ index }) => index),
  }

  return {
    asset: {
      version: "2.0",
      generator: GENERATOR,
    },
    extensionsUsed: [
      "KHR_materials_unlit",
      ...(order.some(({ b }) => b.texture.basisu)
        ? ["KHR_texture_basisu"]
        : []),
    ],
    ...(order.some(({ b }) => b.texture.basisu)
      ? { extensionsRequired: ["KHR_texture_basisu"] }
      : {}),
    scene: 0,
    scenes: [
      {
        name: options.name ?? "layered",
        nodes: nodes.map((_, i) => i),
        extras: sceneExtras,
      },
    ],
    nodes,
    meshes,
    materials,
    textures,
    images,
    samplers: [
      {
        magFilter: 9729, // LINEAR
        minFilter: 9987, // LINEAR_MIPMAP_LINEAR
        wrapS: 33071, // CLAMP_TO_EDGE
        wrapT: 33071,
      },
    ],
    accessors,
    bufferViews,
    // 空 BIN（所有层都没几何）时不写 buffers / BIN chunk：glTF 要求
    // `buffer.byteLength >= 1`，而 GLB 的 BIN chunk 也要求非空。
    ...(bin.byteLength > 0
      ? { buffers: [{ byteLength: bin.byteLength }] }
      : {}),
    cameras: [
      {
        name: "reference",
        type: "perspective",
        perspective: finitePerspective(scene),
      },
    ],
  }
}

function finitePerspective(scene: MeshScene): Record<string, number> {
  return {
    yfov: finite(scene.camera.fovY, Math.PI / 3),
    znear: Math.max(1e-4, finite(scene.camera.near, 0.01)),
    zfar: Math.max(1e-3, finite(scene.camera.far, 100)),
    aspectRatio: finite(scene.view.width / scene.view.height, 1),
  }
}

function finite(value: number, fallback: number): number {
  return Number.isFinite(value) && value > 0 ? value : fallback
}

function layerName(index: number): string {
  return `layer_${String(index).padStart(2, "0")}`
}

/** glTF `asset.generator` 标识。 */
const GENERATOR = "spatial-scene/meshing"

/** 顺序写 buffer 的对齐器。 */
class BinaryWriter {
  private readonly chunks: Uint8Array[] = []
  private length = 0

  alignTo4(): this {
    const pad = (4 - (this.length % 4)) % 4
    if (pad > 0) this.chunks.push(new Uint8Array(pad))
    this.length += pad
    return this
  }

  writeBytes(bytes: ArrayBufferView): {
    byteOffset: number
    byteLength: number
  } {
    const byteOffset = this.length
    this.chunks.push(
      new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength),
    )
    this.length += bytes.byteLength
    return { byteOffset, byteLength: bytes.byteLength }
  }

  writeFloat32(values: Float32Array): {
    byteOffset: number
    byteLength: number
  } {
    return this.writeBytes(
      new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
    )
  }

  writeUint32(values: Uint32Array): {
    byteOffset: number
    byteLength: number
  } {
    return this.writeBytes(
      new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
    )
  }

  finish(): Uint8Array {
    const out = new Uint8Array(this.length)
    let offset = 0
    for (const chunk of this.chunks) {
      out.set(chunk, offset)
      offset += chunk.byteLength
    }
    return out
  }
}

/** GLB 容器：header + JSON chunk (+ BIN chunk)，均 4 字节对齐。 */
function packGlb(gltf: Record<string, unknown>, bin: Uint8Array): Uint8Array {
  const json = new TextEncoder().encode(JSON.stringify(gltf))
  const jsonPad = (4 - (json.byteLength % 4)) % 4
  // 空 BIN 不写 chunk（GLB 要求 chunkLength > 0）。
  const binChunk =
    bin.byteLength > 0
      ? 8 + bin.byteLength + ((4 - (bin.byteLength % 4)) % 4)
      : 0
  const total = 12 + 8 + json.byteLength + jsonPad + binChunk
  const out = new Uint8Array(total)
  const view = new DataView(out.buffer)

  view.setUint32(0, 0x46546c67, true) // "glTF"
  view.setUint32(4, 2, true)
  view.setUint32(8, total, true)

  view.setUint32(12, json.byteLength + jsonPad, true)
  view.setUint32(16, 0x4e4f534a, true) // "JSON"
  out.set(json, 20)
  for (let i = 0; i < jsonPad; i++) out[20 + json.byteLength + i] = 0x20
  if (binChunk === 0) return out

  const binPad = (4 - (bin.byteLength % 4)) % 4
  const binHeader = 20 + json.byteLength + jsonPad
  view.setUint32(binHeader, bin.byteLength + binPad, true)
  view.setUint32(binHeader + 4, 0x004e4942, true) // "BIN\0"
  out.set(bin, binHeader + 8)
  return out
}

function clamp01(value: number): number {
  return value < 0 ? 0 : value > 1 ? 1 : value
}
