/**
 * meshing 模块入口：**LayeredRgbd -> MeshScene**。
 *
 * 每层的流水线（顺序不可换）：
 * ```
 * 1. computeSupportField  支撑 = 非全透明 + 深度外扩（最近 seed 扩散）
 * 2. buildLayerLodMesh    受限四叉树自适应出面（误差有界 + 2:1 无裂缝）
 * ```
 * 全场收尾：只组装元数据（相机 / 视图 / 层深度），不生成基底平面、不做回填。
 *
 * 纯 CPU、确定性：同一输入 + 同一选项两次调用逐位相同。
 */

import type { LayeredRgbd } from "../layering/types.ts"
import { buildLayerLodMesh, type LodOptions } from "./lod.ts"
import { computeSupportField } from "./support.ts"
import type {
  LayerMesh,
  LayerMeshReport,
  MeshingOptions,
  MeshScene,
} from "./types.ts"

/**
 * 把分层 RGBAD 变成 mesh 场景。
 *
 * @param layered `renderLayerStack()` 的产物（0 = 最近的层序）。
 */
export function buildMeshScene(
  layered: LayeredRgbd,
  options: MeshingOptions = {},
): MeshScene {
  const { L, width, height, placement, frames, camera, view } = layered
  if (frames.length !== L) {
    throw new Error(`frames 数量 ${frames.length} != L ${L}`)
  }
  if (camera.width !== width || camera.height !== height) {
    throw new Error(
      `相机尺寸 ${camera.width}x${camera.height} 与场景 ${width}x${height} 不一致`,
    )
  }
  const lod = resolveLodOptions(options.lod)

  const layers: LayerMesh[] = []
  let triangleCount = 0
  let vertexCount = 0
  for (let k = 0; k < L; k++) {
    const frame = frames[k]
    if (frame.width !== width || frame.height !== height) {
      throw new Error(
        `第 ${k} 层帧尺寸 ${frame.width}x${frame.height} 与场景 ${width}x${height} 不一致`,
      )
    }
    const support = computeSupportField(frame, options.alpha)
    const mesh = buildLayerLodMesh({ width, height }, camera, support, lod)
    const report: LayerMeshReport = {
      seedPixels: support.stats.seedPixels,
      dilatedPixels: support.stats.dilatedPixels,
      supportPixels: support.stats.supportPixels,
      removedIslandPixels: support.stats.removedIslandPixels,
      leaves: mesh.stats.leaves,
      levels: mesh.stats.levelHistogram,
      maxTriangleError: mesh.stats.maxTriangleError,
      minCellViolations: mesh.stats.minCellViolations,
      skippedLeaves: mesh.stats.skippedLeaves,
      snappedLeaves: mesh.stats.snappedLeaves,
    }
    layers.push({
      layerIndex: k,
      width,
      height,
      vertexCount: mesh.stats.vertexCount,
      triangleCount: mesh.stats.triangleCount,
      positions: mesh.positions,
      uvs: mesh.uvs,
      indices: mesh.indices,
      // 纹理**引用**输入帧的 rgb/alpha（不拷贝，见 types.ts 的内存约定）。
      texture: { width, height, rgb: frame.rgb, alpha: frame.alpha },
      depthRange: [placement.boundariesZ[k], placement.boundariesZ[k + 1]],
      report,
    })
    triangleCount += mesh.stats.triangleCount
    vertexCount += mesh.stats.vertexCount
  }

  return {
    layers,
    // 以下均**引用输入**，不拷贝。
    layerDepths: placement.layerDepthsZ,
    layerRanges: layered.ranges,
    near: layered.near,
    far: layered.far,
    view,
    camera,
    premultipliedAlpha: false,
    report: { layers: layers.map((l) => l.report), triangleCount, vertexCount },
  }
}

/** `lod: false` -> 逐像素出面（`minCellPx == maxCellPx == 1`）。 */
function resolveLodOptions(lod: MeshingOptions["lod"]): LodOptions {
  if (lod === false) return { minCellPx: 1, maxCellPx: 1 }
  return lod ?? {}
}
