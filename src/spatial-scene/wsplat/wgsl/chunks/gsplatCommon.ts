/**
 * forked from playcanvas/engine v2.22.2 (MIT)
 * upstream: src/scene/shader-lib/wgsl/chunks/gsplat/vert/gsplatCommon.js（clipCorner）
 * commit:   6e6d0d0690830fcd1d91ffde15e9c87a062e8c43
 * modifications:
 *   - 只保留 `clipCorner`；去掉 shadow / pick / prepass / dither / fog / tonemap /
 *     scene-texture 深度写法（那些是 playcanvas 的场景耦合，本模块不要）
 *   - **删掉上游的 `normExp`**：核已改为 gsplat 的真高斯 `exp(-0.5·r²)`
 *     （见 `gsplatCorner.ts` 的 `GAUSS_K2` 与 `quad.ts` 的片元），不再做边缘归一化。
 *     保留的 `clipCorner` 仍用上游基于 `exp(-4A)` 的收缩量：它比真高斯所需**更大**，
 *     即只是多光栅化几条会被片元 `alpha < alphaClip` 丢弃的边，结果不变（保守安全）。
 *   - half -> f32
 *   - 新增 `MAX_ALPHA = 0.99`（gsplat 的逐像素 alpha 上限）
 *
 * `clipCorner`：按 alphaClip 收缩 quad，把 alpha < clip 的区域整块剪掉，
 * 避免为近零 alpha 的像素付带宽（与 gsplat 的 `alpha_clip` 同义）。
 */

export const gsplatCommonChunk = {
  name: "gsplatCommon",
  code: `
/// 逐像素 alpha 上限（gsplat 同款：单颗高斯最多贡献 0.99）。
const MAX_ALPHA: f32 = 0.99;

/// 按 alphaClip 收缩 quad（vertex 阶段，无导数依赖）。
fn clipCorner(corner: ptr<function, SplatCorner>, alpha: f32) {
	let clip: f32 = min(1.0, sqrt(max(0.0, log(alpha / uniforms.alphaClipForward))) * 0.5);
	(*corner).offset = (*corner).offset * clip;
	(*corner).uv = (*corner).uv * clip;
}
`,
} as const
