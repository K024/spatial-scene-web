/**
 * meshing 模块的**公共入口**（barrel）：layered RGBAD -> mesh 场景 -> GLB。
 *
 * ── 提供什么 ──
 * | 文件 | 内容 |
 * |---|---|
 * | `types.ts` | 数据契约（`MeshScene` / `LayerMesh` / `MeshingOptions` …） |
 * | `support.ts` | 支撑场：非全透明 + **深度外扩**（撑住半透明外围） |
 * | `lod.ts` | 受限四叉树自适应出面（误差有界 + 2:1 无裂缝 + 边界贴合） |
 * | `mesh.ts` | 总装：`buildMeshScene(layered)` |
 * | `glb.ts` | `buildGlb(scene)`：零依赖 GLB（unlit + BLEND + sRGB 纹理） |
 * | `png.ts` | 极简 PNG 编码器（stored deflate，零依赖） |
 *
 * ── 不提供什么（刻意的）──
 * 撕裂 / 断壁 / 基底平面 / 回填 / 图集 / 量化 / Draco 全部不在本阶段
 *（见 `types.ts` 的模块边界）。要缩体积时先从 `glb.ts: encodeTexture` 换实现。
 *
 * ── 最小用法 ──
 * ```ts
 * const scene = buildMeshScene(layered, { lod: { minCellPx: 4, maxError: 0.005 } })
 * const glb = buildGlb(scene)              // Uint8Array
 * await writeFile("scene.glb", glb)
 * ```
 */

export type { GlbExportOptions } from "./glb.ts"
export { buildGlb } from "./glb.ts"
export type { LodMeshResult, LodOptions, LodStats } from "./lod.ts"
export { buildLayerLodMesh } from "./lod.ts"
export { buildMeshScene } from "./mesh.ts"
export { encodePngRgba8 } from "./png.ts"
export type { SupportField, SupportFrame, SupportStats } from "./support.ts"
export { computeSupportField } from "./support.ts"
export type {
  LayerMesh,
  LayerMeshReport,
  MeshingOptions,
  MeshReport,
  MeshScene,
  RgbaTexture,
  SupportOptions,
} from "./types.ts"
