/**
 * layering -> meshing 的**输入装配**（纯函数）。
 *
 * ── 在全链路里的位置 ──
 * ```
 * layered RGBAD（layering） -> [toMeshingInput] -> MeshScene（meshing）
 * ```
 * `LayeredRGBD` 已带 `L / width / height / near / far / placement / frames / ranges / stats`；
 * `MeshingInput` 只比它多一个 `camera`（反投影顶点与算 FoV / aspect 要用），
 * 所以这里只做「补 camera + 可选重算 ranges」。
 *
 * ── `overlap` 只改报告 ──
 * `overlap` 只作用于 `ranges`（供渲染侧按深度范围剔除），**不改层分配 / 排列**；
 * 重复绘制会破坏参考视角无损，详见 `layering/bands.ts` 头部。
 * `overlap = 0` 时逐位复用输入的 `ranges`，保持既有 golden 不漂。
 *
 * ── 为什么在 meshing 而不是 layering ──
 * 产物是 meshing 自己的输入契约（`MeshingInput`）；layering 侧不该反向依赖 meshing 的契约。
 * 依赖方向 `meshing -> layering` 本来就有（`relief.ts` / `tears.ts` 用 layering 的视差域换算）。
 */

import { computeLayerRanges } from "../layering/bands.ts"
import type { LayeredRGBD } from "../layering/types.ts"
import type { WSplatCamera } from "../wsplat/camera.ts"
import type { MeshingInput } from "./types.ts"

/**
 * 把 layering 的分层产物装成 `buildMeshScene` 要的输入契约。
 *
 * `camera` 必须与渲出这些层帧的相机**同一台**（否则反投影顶点对不上像素）。
 */
export function toMeshingInput(
  layered: LayeredRGBD,
  camera: WSplatCamera,
  overlap = 0,
): MeshingInput {
  return {
    ...layered,
    ranges:
      overlap === 0
        ? layered.ranges
        : computeLayerRanges(layered.placement.boundaries, overlap),
    camera,
  }
}
