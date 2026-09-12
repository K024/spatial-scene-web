/**
 * 自有 WGSL：**剔除统计**的 compute pass。
 *
 * ── 为什么要单独一趟 compute，而不是在顶点阶段计数 ──
 * WebGPU 规定 `var<storage, read_write>` **不能出现在 vertex 阶段**
 * （`atomicAdd` 需要 read_write），所以顶点阶段数不了数（实测报错原文：
 * "var with 'storage' address space and 'read_write' access mode cannot be used by
 * vertex pipeline stage"）。
 *
 * 所以这里跑一趟 compute：**逐 splat 调用与顶点阶段完全相同的 cull 函数**
 * （`initSource` / `initCenter` / `initCorner`，都来自同一批 chunk），
 * 把每个剔除原因原子计数。
 *
 * ── 与顶点阶段的一致性 ──
 * - 调用顺序逐条对应 `quad.ts` 的 `vsSplat`（bounds -> alphaClip(原始) ->
 *   center -> corner(含 minPixelSize / 视锥) -> AA 后 alphaClip -> drawn）；
 * - 用的是**同一批 WGSL 函数与同一组 uniform**，所以判据不会漂移；
 * - `countOnce*` 只在定义了 `SPLAT_COUNT_CULLS` 时才真的原子加
 *   （见 `splatData.ts`），因此同一份 chunk 在顶点模块里是零开销。
 *
 * 代价：多一趟 O(n) 的顶点前处理（实测 native 1.18M 高斯约 ms 量级），
 * 且**只有显式调用 `countCulls()` 才跑**——不需要统计时零成本。
 */

export const cullStatsWgsl = {
  name: "cullStats",
  code: `#include "splatData"
#include "splatSource"
#include "gsplatStructs"
#include "gsplatQuatToMat3"
#include "gsplatCenter"
#include "gsplatCorner"
#include "gsplatCommon"

@compute @workgroup_size(64)
fn csCountCulls(@builtin(global_invocation_id) gid: vec3u) {
	let index: u32 = gid.x;
	if (index >= uniforms.numSplats) {
		return;
	}

	// vertexIndex = 0 是刻意的：cornerUV 会是 (-1,-1)，正好满足 countOnce 的去重条件
	var source: SplatSource;
	if (!initSource(&source, index, 0u)) {
		countOnceAtVertex(0u, SPLAT_STAT_BOUNDS);
		return;
	}
	if (source.color.a <= uniforms.alphaClipForward) {
		countOnce(source.cornerUV, SPLAT_STAT_ALPHA_CLIP);
		return;
	}

	var center: SplatCenter;
	if (!initCenter(&source, &center)) {
		// initCenter 内部已计数（SPLAT_STAT_BEHIND_CAMERA）
		return;
	}

	var corner: SplatCorner;
	if (!initCorner(&source, &center, &corner)) {
		// initCorner -> initCornerCov 内部已计数（minPixelSize / 视锥）
		return;
	}

	// 与 vsSplat 同序：AA 补偿之后再判一次 alphaClip
	var color: vec4f = source.color;
#if GSPLAT_AA
	color.a = color.a * corner.aaFactor;
#endif
	if (color.a <= uniforms.alphaClipForward) {
		countOnce(source.cornerUV, SPLAT_STAT_ALPHA_CLIP_AA);
		return;
	}

	countOnce(source.cornerUV, SPLAT_STAT_DRAWN);
}
`,
} as const
