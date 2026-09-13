/**
 * 线框模式（几何调试视图）的材质管理。
 *
 * ══════════════════════════ 为什么是「换材质」而不是「新建线框对象」══════════════════════════
 *
 * 层栈的可见性 / 独显 / α 淡入淡出全都写在 `mesh.visible` + `material.opacity` 上，
 * `renderOrder` 也是渲染端按 `layerIndex` 补的。若另建一套 `Mesh`/`LineSegments`，
 * 这些逻辑就得维护两遍、还必须保持同步。所以这里**原地换材质**：
 * 几何、对象、renderOrder、可见性逻辑全部复用，切换只是 `mesh.material = ...`。
 * 代价是每个 mesh 要记住一对材质（`LayerMaterials`）。
 *
 * ══════════════════════════ 为什么用 `material.wireframe` ══════════════════════════
 *
 * three 的 `WebGLRenderer.renderBufferDirect` 见到 `material.wireframe === true` 时，
 * 会把 index 换成 `geometries.getWireframeAttribute(geometry)`（懒生成、WeakMap 缓存、
 * 按 `index.version` 失效）并切到 `gl.LINES`，`rangeFactor = 2`（每三角形 3 条边）。
 * 零几何代码、零上传，比 `WireframeGeometry`/`LineSegments` 省一份顶点与对象。
 * `EdgesGeometry` 不适用：它按二面角筛边，而 relief 网格近乎共面，筛不出东西。
 *
 * ══════════════════════════ 两个必须说清的代价 / 限制 ══════════════════════════
 *
 * 1. **`gl.LINES` 的线宽在 WebGL2 / ANGLE 上恒为 1 设备像素**，`wireframeLinewidth` 被忽略。
 *    而本链路的层是「每像素一顶点」的 relief 网格：768 宽那版就有 188 万三角形铺在
 *    约 44 万像素上（≈ 4.3 三角/像素）。**参考视角下线框必然糊成一片**，要放大约
 *    6~8 倍才能读出格子。这不是实现缺陷，是网格密度决定的 —— 别试图靠加粗线救它。
 * 2. **线段索引的内存是 `6 × triangleCount` 个 uint32**（768 宽那版 ≈ 43MB）。
 *    它是 three 懒生成后**永久缓存**的（只要几何还活着），切回标准渲染也不释放。
 *    所以这条路径必须由用户显式开启，UI 上把估算值摆出来（见 `FloatingPanel`）。
 *
 * ══════════════════════════ 深度口径 ══════════════════════════
 *
 * 标准模式是 `transparent + depthWrite=false + renderOrder` 的画家算法（层要 α 混合）。
 * 线框看的是**几何本身**，那套口径会让线互相穿透、完全读不出来，所以线框材质改成
 * `depthWrite = true` + `depthTest = true`：近层的线正确遮住远层。
 * `transparent` 仍留 `true`，是为了让逐层 α 淡入淡出在两种模式下走同一套逻辑
 * （`opacity === 1` 时与不透明无差别）。
 */

import type { Material, Mesh } from "three"
import { Color, FrontSide, MeshBasicMaterial } from "three"
import type { LoadedScene } from "./types.ts"

/** 一条 mesh 的两套材质；`wire` 懒建（用户没开过线框就一直是 `null`）。 */
interface LayerMaterials {
  /** 基准（纹理）材质：`GLTFLoader` 给的 unlit + map，切回标准渲染时原样还回去。 */
  readonly base: Material
  wire: MeshBasicMaterial | null
}

/**
 * `mesh -> 材质对`。
 *
 * 用 `WeakMap` 而不是往 `LoadedScene` 里塞字段：这是**渲染策略**，不是场景数据契约；
 * 而且 `WeakMap` 让「材质随 mesh 一起被回收」这件事自动成立，换场景不用手动清。
 * 用 `Mesh` 当键而不是 `key`：`key` 在换场景后会重复（都是 `layer_00`）。
 */
const MATERIALS = new WeakMap<Mesh, LayerMaterials>()

/** 背衬平面的线框色（中性灰：它不是「某一层」，不该占用层色相）。 */
const BACKING_COLOR = "#8b95a8"

/**
 * 登记场景里所有层的基准材质。
 *
 * ⚠ **必须在任何切换之前调用一次**（`ModelRoot` 在场景挂载的 effect 里做）。
 * 否则若在中途登记，`mesh.material` 可能已经是线框材质，基准材质就被认错了。
 */
export function registerLayerMaterials(scene: LoadedScene): void {
  for (const layer of scene.layers) {
    const mesh = scene.meshes.get(layer.key)
    if (!mesh || MATERIALS.has(mesh)) continue
    const material = firstMaterial(mesh)
    if (material) MATERIALS.set(mesh, { base: material, wire: null })
  }
}

/** 取某条 mesh 的材质对（未登记返回 `null`）。 */
export function layerMaterialsOf(mesh: Mesh): LayerMaterials | null {
  return MATERIALS.get(mesh) ?? null
}

/**
 * 整场景切到线框 / 标准渲染。
 *
 * 独占语义：开启时**所有**层（含背衬平面）都换成线框材质，标准渲染不参与绘制；
 * 关闭时原样换回 `GLTFLoader` 给的材质，贴图一直留在显存里，来回切零上传。
 */
export function setSceneWireframe(scene: LoadedScene, on: boolean): void {
  for (const layer of scene.layers) {
    const mesh = scene.meshes.get(layer.key)
    if (!mesh) continue
    const pair = MATERIALS.get(mesh)
    if (!pair) continue
    if (on) {
      pair.wire ??= createWireMaterial(layer.layerIndex, pair.base)
      if (mesh.material !== pair.wire) mesh.material = pair.wire
    } else if (mesh.material !== pair.base) {
      mesh.material = pair.base
    }
  }
}

/**
 * 释放整个场景的线框材质。
 *
 * ⚠ **释放场景前必须先调它**，否则会漏：线框模式下 `mesh.material` 指向线框材质，
 * `disposeScene` 会把它当基准材质释放，于是**基准材质与贴图永远不会被释放**。
 * 这里先把 `mesh.material` 还回基准，再释放线框材质并清掉引用（下次开启重建即可）。
 *
 * 不必放进 `scene.ts` 的 `disposeScene`：那边只认「three 对象图」，不应该知道
 * 渲染策略；这条依赖方向是 store → wireframe，不是反过来。
 */
export function disposeSceneWireframe(scene: LoadedScene): void {
  for (const layer of scene.layers) {
    const mesh = scene.meshes.get(layer.key)
    if (!mesh) continue
    const pair = MATERIALS.get(mesh)
    if (!pair) continue
    if (mesh.material === pair.wire) mesh.material = pair.base
    pair.wire?.dispose()
    pair.wire = null
  }
}

/**
 * 建一条线框材质。
 *
 * - `wireframe: true` —— 见文件头，交给 three 生成线段索引；
 * - `transparent` 继承基准材质的 α（保证淡入淡出跨模式一致）；
 * - `depthWrite/depthTest = true` —— 线框要能被正确遮挡（见文件头「深度口径」）；
 * - `side = FrontSide` —— 线框走 `gl.LINES`，不受面剔除影响；写它只是为了避开 three 对
 *   「`transparent` + `DoubleSide`」的两次绘制（它会白翻倍 draw，见 `scene.ts` 的材质口径）；
 * - `toneMapped: false` —— 与标准模式同一口径（照片色 / 几何色都不过色调映射）。
 */
function createWireMaterial(
  layerIndex: number,
  base: Material,
): MeshBasicMaterial {
  return new MeshBasicMaterial({
    color: layerColor(layerIndex),
    wireframe: true,
    transparent: base.transparent,
    opacity: base.opacity,
    depthWrite: true,
    depthTest: true,
    side: FrontSide,
    toneMapped: false,
  })
}

/**
 * 逐层线框色。
 *
 * 用**黄金角散列**（137.508°）而不是 `layerIndex * 常数`：相邻层号也拉得开色相，
 * 当线密到糊成一片时，至少还能靠颜色判断「这团是哪一层」。
 * `setHSL` 按 sRGB 给定，three 会转到工作色彩空间，输出端再转回来，颜色不漂。
 */
function layerColor(layerIndex: number): Color {
  if (layerIndex < 0) return new Color(BACKING_COLOR)
  return new Color().setHSL(((layerIndex * 137.508) % 360) / 360, 0.72, 0.62)
}

function firstMaterial(mesh: Mesh): Material | null {
  const material = mesh.material
  if (Array.isArray(material)) return material[0] ?? null
  return material ?? null
}
