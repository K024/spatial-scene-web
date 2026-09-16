/**
 * meshing 自检（**唯一判定入口**：阈值只在本文件里，退出码非 0 = 不过门）。
 *
 * 阶段：`src/spatial-scene/meshing/**`（分层 RGBAD -> 渲染器无关的 mesh 场景）。
 *
 * ── 跑什么 ──
 *   [A] 分层渲染（复用 layering）。
 *   [B] 网格化。
 *   [C] 网格结构门：索引在界内 / 顶点有限 / UV 在 `[0,1]` / 逐层非空。
 *   [D] 几何 ↔ 帧一致性：抽样顶点的**相机空间 z** 必须等于该 UV 处帧的深度
 *       （贴边外推的顶点深度来自最近 seed，所以看的是分位数而不是最大值）。
 *   [E] mesh 光栅化 ↔ splat 直渲（粗门）。
 *   [F] GLB 自检：**在内存里** `buildGlb()`，验容器 / JSON / accessor / 往返字节，
 *       **不写盘** —— 出交付产物是 `scripts/export-glb.ts` 的事。
 *   [G] 视觉（仅观测）：把 mesh 顶点投到画布上出 PNG（几何覆盖检查）。
 *
 * 用法:
 *   npx tsx scripts/meshing-check.ts
 *   npx tsx scripts/meshing-check.ts --layers 10 --min-cell 4 --max-error 0.005
 *   npx tsx scripts/meshing-check.ts --pixel          # 逐像素出面（对照基线）
 *   npx tsx scripts/meshing-check.ts --no-visual      # 不出图
 */

import { mkdirSync, writeFileSync } from "node:fs"
import { resolve } from "node:path"
import { parseArgs } from "node:util"

import {
  DEFAULT_MAX_RENDER_SIDE,
  DEFAULT_SHORT_SIDE,
  renderLayerStack,
} from "../src/spatial-scene/layering/index.ts"
import {
  buildGlb,
  buildMeshScene,
  type MeshScene,
} from "../src/spatial-scene/meshing/index.ts"
import { humanSize, numFlag, REPO_ROOT, shortSideFlag } from "./utils/common.ts"
import { decodeImageToRgba, rgbaToPngBuffer } from "./utils/image.ts"
import { type RasterResult, rasterizeMeshScene } from "./utils/raster.ts"
import { linearFrameToRgba8, loadWSplatScene } from "./utils/scene.ts"
import { withNodeDevice } from "./utils/webgpu.ts"

const GATES = {
  /** 几何↔帧深度一致性：中位相对误差。 */
  depthMedianRel: 1e-3,
  /** 几何↔帧深度一致性：95 分位相对误差（贴边外推顶点会放宽）。 */
  depthP95Rel: 0.05,
  /** 光栅化粗门（最近邻采样 + z-buffer，见 `utils/raster.ts`；实测 ~1.4e-3，留 7 倍余量）。 */
  rasterAlphaMae: 0.01,
  rasterPremultipliedMae: 0.01,
} as const

const CLI = {
  layers: { type: "string" },
  "view-scale": { type: "string" },
  "max-side": { type: "string" },
  "short-side": { type: "string" },
  "min-cell": { type: "string" },
  "max-error": { type: "string" },
  "max-cell": { type: "string" },
  "snap-px": { type: "string" },
  dilate: { type: "string" },
  "alpha-cutoff": { type: "string" },
  "min-island": { type: "string" },
  pixel: { type: "boolean" },
  ply: { type: "string" },
  camera: { type: "string" },
  "max-splats": { type: "string" },
  "visual-out": { type: "string" },
  "no-visual": { type: "boolean" },
} as const

const checks: { name: string; ok: boolean; detail: string }[] = []

function addCheck(name: string, ok: boolean, detail: string): void {
  checks.push({ name, ok, detail })
  console.log(`  ${ok ? "✓" : "✗"} ${name.padEnd(30)} ${detail}`)
}

async function main(): Promise<void> {
  const args = parseArgs({
    args: process.argv.slice(2),
    options: CLI,
    allowPositionals: false,
    strict: true,
  })
  const layers = Math.round(numFlag("--layers", args.values.layers, 10))
  const viewScale = numFlag("--view-scale", args.values["view-scale"], 1.2)
  // 分辨率默认「短边 auto」：min(SHARP 内部 1536, 原图短边)；--max-side 只作长边硬上限。
  const shortSide = shortSideFlag(
    "--short-side",
    args.values["short-side"],
    DEFAULT_SHORT_SIDE,
  )
  const maxSide = numFlag(
    "--max-side",
    args.values["max-side"],
    DEFAULT_MAX_RENDER_SIDE,
  )
  // 调试图不进 `public/`（那里只放交付产物）。
  const visualDir = resolve(
    REPO_ROOT,
    args.values["visual-out"] ?? "temp/meshing-check",
  )

  console.log("=".repeat(78))
  console.log("meshing 自检")
  console.log("=".repeat(78))

  const scene = loadWSplatScene({
    ply: args.values.ply,
    camera: args.values.camera,
    maxSplats: args.values["max-splats"]
      ? Math.round(numFlag("--max-splats", args.values["max-splats"], 0))
      : undefined,
  })
  console.log(
    `\n场景: ${scene.plyPath}\n      ${scene.gaussians.opacities.length} 个高斯  ` +
      `参考 ${scene.camera.width}x${scene.camera.height}  near=${scene.near.toFixed(3)}m far=${scene.far.toFixed(1)}m`,
  )

  let meshScene: MeshScene | undefined
  let raster: RasterResult | undefined
  await withNodeDevice(async (device) => {
    console.log("\n[A] 分层渲染")
    const t0 = Date.now()
    const layered = await renderLayerStack(device, scene.gaussians, {
      camera: scene.camera,
      layers,
      viewScale,
      shortSide,
      maxRenderSide: maxSide,
      includeDirect: true,
    })
    console.log(
      `      ${Date.now() - t0}ms  画布 ${layered.width}x${layered.height}  ` +
        `pixelScale=${layered.view.pixelScale.toFixed(4)}  fx=${layered.view.focalLengthPx.toFixed(1)}px`,
    )

    console.log("\n[B] 网格化")
    const t1 = Date.now()
    meshScene = buildMeshScene(layered, {
      alpha: {
        alphaCutoff: args.values["alpha-cutoff"]
          ? numFlag("--alpha-cutoff", args.values["alpha-cutoff"], 1 / 255)
          : undefined,
        dilatePx: args.values.dilate
          ? numFlag("--dilate", args.values.dilate, 2)
          : undefined,
        minIslandPixels: args.values["min-island"]
          ? numFlag("--min-island", args.values["min-island"], 8)
          : undefined,
      },
      lod: args.values.pixel
        ? false
        : {
            minCellPx: numFlag("--min-cell", args.values["min-cell"], 4),
            maxCellPx: numFlag("--max-cell", args.values["max-cell"], 128),
            maxError: numFlag("--max-error", args.values["max-error"], 0.005),
            snapBoundaryPx: args.values["snap-px"]
              ? numFlag("--snap-px", args.values["snap-px"], 4)
              : undefined,
          },
    })
    console.log(
      `      ${Date.now() - t1}ms  ${meshScene.report.triangleCount} 三角面  ` +
        `${meshScene.report.vertexCount} 顶点`,
    )
    printLayers(meshScene)
    console.log(
      "      注：`minCell超差` = 已到最小格但输出面误差仍 > maxError 的叶子数。\n" +
        "          精度上限由 minCellPx 决定而不是 maxError（视差场在 4px 尺度本来就不平滑），\n" +
        "          所以这个数很大是正常的；要更贴近 maxError 就调小 --min-cell（面数 ∝ 1/minCell²）。",
    )

    console.log("\n[C] 网格结构门")
    structuralChecks(meshScene)
    console.log("\n[D] 几何 <-> 帧一致性")
    geometryChecks(meshScene, layered.frames)
    console.log("\n[E] mesh 光栅化 <-> splat 直渲（粗门）")
    raster = rasterChecks(meshScene, layered.direct)
  })

  if (!meshScene) throw new Error("网格化没有产出")

  console.log("\n[F] GLB（内存构建，不写盘）")
  {
    const t2 = Date.now()
    const glb = buildGlb(meshScene, {
      name: "spatial-scene",
      doubleSided: true,
    })
    glbChecks(glb)
    await glbRoundTrip(glb, meshScene)
    console.log(
      `      ${humanSize(glb.byteLength)}  ${Date.now() - t2}ms  ` +
        `（交付产物用 npm run export-glb 生成）`,
    )
  }

  if (args.values["no-visual"] !== true) {
    console.log("\n[G] 视觉（仅观测）")
    await writeCoverage(visualDir, meshScene, raster)
  }

  const failed = checks.filter((c) => !c.ok)
  console.log(`\n${"=".repeat(78)}`)
  console.log(
    `共 ${checks.length} 条，通过 ${checks.length - failed.length}，失败 ${failed.length}`,
  )
  if (failed.length > 0) {
    console.log("\n✗ 未通过:")
    for (const f of failed) console.log(`   - ${f.name}: ${f.detail}`)
    process.exitCode = 1
  } else {
    console.log("✓ 全部门限通过")
  }
}

function printLayers(scene: MeshScene): void {
  console.log(
    "      层   支撑 px   叶子    层级                         三角面     maxErr   minCell超差  贴边",
  )
  for (const layer of scene.layers) {
    const r = layer.report
    console.log(
      `      ${String(layer.layerIndex).padStart(2)}  ${String(r.supportPixels).padStart(8)}  ` +
        `${String(r.leaves).padStart(6)}  ${r.levels.join("/").padEnd(26)}  ` +
        `${String(layer.triangleCount).padStart(8)}  ${r.maxTriangleError.toExponential(1)}  ` +
        `${String(r.minCellViolations).padStart(8)}  ${String(r.snappedLeaves).padStart(6)}`,
    )
  }
}

// ═══════════════════════════ 结构门 ═══════════════════════════

function structuralChecks(scene: MeshScene): void {
  let indexOk = true
  let finiteOk = true
  let uvOk = true
  let total = 0
  const emptyLayers: number[] = []
  for (const layer of scene.layers) {
    if (layer.triangleCount === 0) emptyLayers.push(layer.layerIndex)
    for (let i = 0; i < layer.indices.length; i++) {
      if (layer.indices[i] >= layer.vertexCount) indexOk = false
    }
    for (let i = 0; i < layer.positions.length; i++) {
      if (!Number.isFinite(layer.positions[i])) finiteOk = false
    }
    for (let i = 0; i < layer.uvs.length; i++) {
      const v = layer.uvs[i]
      if (!(v >= 0 && v <= 1)) uvOk = false
    }
    total += layer.triangleCount
  }
  addCheck("索引在界内", indexOk, `${total} 个三角面`)
  addCheck("顶点有限", finiteOk, "positions 全为有限值")
  addCheck("UV 在 [0,1]", uvOk, `${scene.layers.length} 层`)
  addCheck(
    "逐层非空",
    emptyLayers.length === 0,
    emptyLayers.length === 0
      ? `${scene.layers.length} 层都有几何`
      : `空层 ${emptyLayers.join(",")}`,
  )
}

// ═══════════════════════════ 几何 <-> 帧 ═══════════════════════════

function geometryChecks(
  scene: MeshScene,
  frames: readonly {
    width: number
    height: number
    depth: Float32Array
    alpha: Float32Array
  }[],
): void {
  const view = scene.camera.viewMatrix
  const errors: number[] = []
  let maxRel = 0
  let sampled = 0
  for (const layer of scene.layers) {
    const frame = frames[layer.layerIndex]
    const stride = Math.max(1, Math.floor(layer.vertexCount / 20000))
    for (let v = 0; v < layer.vertexCount; v += stride) {
      const u = layer.uvs[v * 2]
      const w = layer.uvs[v * 2 + 1]
      const x = Math.min(
        layer.width - 1,
        Math.max(0, Math.round(u * layer.width - 0.5)),
      )
      const y = Math.min(
        layer.height - 1,
        Math.max(0, Math.round(w * layer.height - 0.5)),
      )
      const reference = frame.depth[y * layer.width + x]
      if (!(reference > 0)) continue
      const px = layer.positions[v * 3]
      const py = layer.positions[v * 3 + 1]
      const pz = layer.positions[v * 3 + 2]
      // 相机空间 z（列主序 viewMatrix 的第三行）
      const zCam = view[2] * px + view[6] * py + view[10] * pz + view[14]
      const rel = Math.abs(zCam - reference) / reference
      errors.push(rel)
      if (rel > maxRel) maxRel = rel
      sampled++
    }
  }
  errors.sort((a, b) => a - b)
  const median = errors.length > 0 ? errors[errors.length >> 1] : 0
  const p95 = errors.length > 0 ? errors[Math.floor(errors.length * 0.95)] : 0
  addCheck(
    "顶点深度 == 帧深度",
    median <= GATES.depthMedianRel && p95 <= GATES.depthP95Rel,
    `median=${median.toExponential(2)}  p95=${p95.toExponential(2)}  max=${maxRel.toExponential(2)}  n=${sampled}`,
  )
}

// ═══════════════════════════ 光栅化粗门 ═══════════════════════════

/**
 * 把 mesh 用 CPU 真的画一遍，与 splat 直渲对照。
 *
 * 只做粗判：抓「整体错位 / UV 翻转 / 索引错 / 纹理贴错」这类结构性错误，
 * 不判 1px 级差异（最近邻采样 + z-buffer vs α 加权混合，见 `utils/raster.ts`）。
 */
function rasterChecks(
  scene: MeshScene,
  direct: { rgb: Float32Array; alpha: Float32Array } | null,
): RasterResult {
  if (!direct) throw new Error("缺少 direct 帧（需要 includeDirect）")
  const t0 = Date.now()
  const raster = rasterizeMeshScene(scene)
  const pixels = raster.width * raster.height
  let alphaSum = 0
  let colorSum = 0
  for (let i = 0; i < pixels; i++) {
    const a = Math.min(1, Math.max(0, direct.alpha[i]))
    alphaSum += Math.abs(raster.alpha[i] - a)
    for (let c = 0; c < 3; c++) {
      colorSum += Math.abs(raster.rgb[i * 3 + c] - direct.rgb[i * 3 + c] * a)
    }
  }
  const alphaMae = alphaSum / pixels
  const premultipliedMae = colorSum / (pixels * 3)
  addCheck(
    "光栅化 α MAE",
    alphaMae <= GATES.rasterAlphaMae,
    `${alphaMae.toExponential(2)}  (门 ${GATES.rasterAlphaMae})  ${raster.trianglesDrawn} 个三角面  ${Date.now() - t0}ms`,
  )
  addCheck(
    "光栅化预乘色 MAE",
    premultipliedMae <= GATES.rasterPremultipliedMae,
    `${premultipliedMae.toExponential(2)}  (门 ${GATES.rasterPremultipliedMae})`,
  )
  return raster
}

// ═══════════════════════════ GLB ═══════════════════════════

function glbChecks(glb: Uint8Array): void {
  const view = new DataView(glb.buffer, glb.byteOffset, glb.byteLength)
  const magic = view.getUint32(0, true) === 0x46546c67
  const version = view.getUint32(4, true) === 2
  const total = view.getUint32(8, true) === glb.byteLength
  const jsonLength = view.getUint32(12, true)
  const jsonType = view.getUint32(16, true) === 0x4e4f534a
  const binHeader = 20 + jsonLength
  const binType = view.getUint32(binHeader + 4, true) === 0x004e4942
  addCheck(
    "GLB 容器",
    magic && version && total && jsonType && binType,
    `magic/version/length/JSON/BIN 全部正确`,
  )

  const json = new TextDecoder().decode(glb.subarray(20, 20 + jsonLength))
  let parsed: {
    accessors?: { count: number }[]
    meshes?: { primitives: { indices: number }[] }[]
    buffers?: { byteLength: number }[]
  } = {}
  let jsonOk = true
  try {
    parsed = JSON.parse(json) as typeof parsed
  } catch {
    jsonOk = false
  }
  addCheck(
    "GLB JSON",
    jsonOk,
    jsonOk
      ? `${jsonLength} 字节  meshes=${parsed.meshes?.length ?? 0}  accessors=${parsed.accessors?.length ?? 0}`
      : "JSON.parse 失败",
  )

  // 每个 primitive 的 INDEX accessor 计数必须是 3 的倍数。
  let accessorOk = true
  const accessors = parsed.accessors ?? []
  for (const mesh of parsed.meshes ?? []) {
    for (const primitive of mesh.primitives) {
      const count = accessors[primitive.indices]?.count ?? -1
      if (!(count > 0 && count % 3 === 0)) accessorOk = false
    }
  }
  addCheck("GLB 索引 accessor", accessorOk, `${accessors.length} 个 accessor`)
}

interface GlbLayout {
  readonly json: Record<string, unknown>
  readonly bin: Uint8Array
}

/** 拆开 GLB 容器（已验过头部）。 */
function parseGlb(glb: Uint8Array): GlbLayout {
  const view = new DataView(glb.buffer, glb.byteOffset, glb.byteLength)
  const jsonLength = view.getUint32(12, true)
  const binHeader = 20 + jsonLength
  const binLength = view.getUint32(binHeader, true)
  return {
    json: JSON.parse(
      new TextDecoder().decode(glb.subarray(20, 20 + jsonLength)),
    ) as Record<string, unknown>,
    bin: glb.subarray(binHeader + 8, binHeader + 8 + binLength),
  }
}

/**
 * GLB 往返：从 BIN 里把 POSITION / TEXCOORD_0 / INDEX 读回来，逐项对照 `MeshScene`。
 *
 * 这是「字节层没错位」的强验证：accessor -> bufferView -> buffer 的偏移、
 * 4 字节对齐、bake/绕序反转，全在这里一次性对齐。另外用 sharp 解一张 PNG，
 * 验自写编码器的字节真的能被通用解码器读回来。
 */
async function glbRoundTrip(glb: Uint8Array, scene: MeshScene): Promise<void> {
  const { json, bin } = parseGlb(glb)
  const accessors = json.accessors as {
    bufferView: number
    count: number
    componentType: number
    type: string
  }[]
  const bufferViews = json.bufferViews as {
    byteOffset: number
    byteLength: number
  }[]
  const meshes = json.meshes as {
    primitives: { attributes: Record<string, number>; indices: number }[]
    extras: { layerIndex: number }
  }[]
  const images = json.images as { bufferView: number; mimeType: string }[]

  const readFloats = (
    accessorIndex: number,
    components: number,
  ): Float32Array => {
    const accessor = accessors[accessorIndex]
    const view = bufferViews[accessor.bufferView]
    const count = accessor.count * components
    const out = new Float32Array(count)
    const dataView = new DataView(bin.buffer, bin.byteOffset + view.byteOffset)
    for (let i = 0; i < count; i++) out[i] = dataView.getFloat32(i * 4, true)
    return out
  }
  const readU32 = (accessorIndex: number): Uint32Array => {
    const accessor = accessors[accessorIndex]
    const view = bufferViews[accessor.bufferView]
    const out = new Uint32Array(accessor.count)
    const dataView = new DataView(bin.buffer, bin.byteOffset + view.byteOffset)
    for (let i = 0; i < accessor.count; i++)
      out[i] = dataView.getUint32(i * 4, true)
    return out
  }

  let positionOk = true
  let indexOk = true
  let uvOk = true
  for (const mesh of meshes) {
    const layer = scene.layers[mesh.extras.layerIndex]
    const primitive = mesh.primitives[0]
    const positions = readFloats(primitive.attributes.POSITION, 3)
    const uvs = readFloats(primitive.attributes.TEXCOORD_0, 2)
    const indices = readU32(primitive.indices)
    if (positions.length !== layer.vertexCount * 3) positionOk = false
    if (uvs.length !== layer.vertexCount * 2) uvOk = false
    if (indices.length !== layer.triangleCount * 3) indexOk = false
    for (let i = 0; i < layer.vertexCount && positionOk; i++) {
      // bake：OpenCV (x,y,z) -> glTF (x,-y,-z)
      if (
        positions[i * 3] !== layer.positions[i * 3] ||
        positions[i * 3 + 1] !== -layer.positions[i * 3 + 1] ||
        positions[i * 3 + 2] !== -layer.positions[i * 3 + 2]
      ) {
        positionOk = false
      }
      if (
        uvs[i * 2] !== layer.uvs[i * 2] ||
        uvs[i * 2 + 1] !== layer.uvs[i * 2 + 1]
      ) {
        uvOk = false
      }
    }
    for (let t = 0; t < layer.indices.length && indexOk; t += 3) {
      // 绕序反转：(i0,i1,i2) -> (i0,i2,i1)
      if (
        indices[t] !== layer.indices[t] ||
        indices[t + 1] !== layer.indices[t + 2] ||
        indices[t + 2] !== layer.indices[t + 1]
      ) {
        indexOk = false
      }
    }
  }
  addCheck("GLB 往返 POSITION", positionOk, "bake (x,-y,-z) 逐位一致")
  addCheck("GLB 往返 TEXCOORD_0", uvOk, "UV 未翻转、逐位一致")
  addCheck("GLB 往返 INDEX", indexOk, "绕序反转后逐位一致")

  // 自写 PNG 编码器：用 sharp 解回来验尺寸 + 抽样像素。
  // ⚠ images 与 meshes 同序（都是远->近），所以 images[0] 属于 meshes[0] 那一层，
  // 不是 `scene.layers[0]`。
  const imageView = bufferViews[images[0].bufferView]
  const png = bin.subarray(
    imageView.byteOffset,
    imageView.byteOffset + imageView.byteLength,
  )
  const decoded = await decodeImageToRgba(png)
  const texture = scene.layers[meshes[0].extras.layerIndex].texture
  let pngOk =
    decoded.width === texture.width && decoded.height === texture.height
  let checked = 0
  for (
    let i = 0;
    i < decoded.width * decoded.height && checked < 500;
    i += 997
  ) {
    const a = clamp01(texture.alpha[i])
    if (Math.abs(decoded.data[i * 4 + 3] - Math.round(a * 255)) > 1) {
      pngOk = false
    }
    if (a > 0) {
      for (let c = 0; c < 3; c++) {
        const srgb = linearToSrgb(clamp01(texture.rgb[i * 3 + c]))
        if (Math.abs(decoded.data[i * 4 + c] - Math.round(srgb * 255)) > 1) {
          pngOk = false
        }
      }
    }
    checked++
  }
  addCheck(
    "自写 PNG 可解码",
    pngOk,
    `${decoded.width}x${decoded.height}  抽样 ${checked} 像素  ${humanSize(png.byteLength)}`,
  )
}

/** 与 `sharp/colorspace.ts: linearRGB2sRGB` 同式。 */
function linearToSrgb(x: number): number {
  return x <= 0.0031308 ? x * 12.92 : 1.055 * x ** (1 / 2.4) - 0.055
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x
}

// ═══════════════════════════ 视觉 ═══════════════════════════

/** 把各层 mesh 顶点按渲染相机投到画布上（几何覆盖检查，不做光栅化）。 */
async function writeCoverage(
  outDir: string,
  scene: MeshScene,
  raster: RasterResult | undefined,
): Promise<void> {
  mkdirSync(outDir, { recursive: true })
  const { width, height } = scene.view
  const pixels = width * height
  const rgba = new Uint8Array(pixels * 4)
  // 顶点按深度着色（近暖远冷）。
  const view = scene.camera.viewMatrix
  const fx = scene.view.focalLengthPx
  const cx = width / 2
  const cy = height / 2
  const near = scene.near
  const far = scene.far
  for (const layer of scene.layers) {
    for (let v = 0; v < layer.vertexCount; v++) {
      const px = layer.positions[v * 3]
      const py = layer.positions[v * 3 + 1]
      const pz = layer.positions[v * 3 + 2]
      const xCam = view[0] * px + view[4] * py + view[8] * pz + view[12]
      const yCam = view[1] * px + view[5] * py + view[9] * pz + view[13]
      const zCam = view[2] * px + view[6] * py + view[10] * pz + view[14]
      if (!(zCam > 1e-6)) continue
      const sx = Math.round((xCam / zCam) * fx + cx)
      const sy = Math.round((yCam / zCam) * fx + cy)
      if (sx < 0 || sy < 0 || sx >= width || sy >= height) continue
      const t = Math.min(
        1,
        Math.max(0, (zCam - near) / Math.max(1e-6, far - near)),
      )
      const i = (sy * width + sx) * 4
      rgba[i] = 255
      rgba[i + 1] = Math.round(255 * (1 - t))
      rgba[i + 2] = Math.round(255 * t)
      rgba[i + 3] = 255
    }
  }
  const png = await rgbaToPngBuffer(rgba, width, height)
  const path = resolve(outDir, "mesh-coverage.png")
  writeFileSync(path, png)
  console.log(`      写出 ${path}（顶点覆盖，颜色 = 深度）`)

  // 纹理化光栅化预览（直通色 = 预乘 / α）。
  if (raster) {
    const straight = new Float32Array(pixels * 3)
    for (let i = 0; i < pixels; i++) {
      const a = raster.alpha[i]
      const inv = a > 0 ? 1 / a : 0
      straight[i * 3] = raster.rgb[i * 3] * inv
      straight[i * 3 + 1] = raster.rgb[i * 3 + 1] * inv
      straight[i * 3 + 2] = raster.rgb[i * 3 + 2] * inv
    }
    const renderRgba = linearFrameToRgba8(straight, raster.alpha, pixels)
    const renderPng = await rgbaToPngBuffer(renderRgba, width, height)
    const renderPath = resolve(outDir, "mesh-render.png")
    writeFileSync(renderPath, renderPng)
    console.log(`      写出 ${renderPath}（CPU 光栅化 mesh）`)
  }
}

main().catch((err) => {
  console.error(`\n自检失败: ${err instanceof Error ? err.stack : err}`)
  process.exitCode = 1
})
