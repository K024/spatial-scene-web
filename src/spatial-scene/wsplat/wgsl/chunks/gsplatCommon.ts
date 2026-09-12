/**
 * forked from playcanvas/engine v2.22.2 (MIT)
 * upstream: src/scene/shader-lib/wgsl/chunks/gsplat/vert/gsplatCommon.js
 *          + src/scene/shader-lib/wgsl/chunks/gsplat/frag/gsplat.js（normExp 部分）
 * commit:   6e6d0d0690830fcd1d91ffde15e9c87a062e8c43
 * modifications:
 *   - 只保留 `clipCorner` 与 `normExp`；去掉 shadow / pick / prepass / dither /
 *     fog / tonemap / scene-texture 深度写法（那些是 playcanvas 的场景耦合，本模块不要）
 *   - half -> f32
 *
 * 这两段是质量的关键，数值**不得自创**：
 *   - `clipCorner`：按 alphaClip 收缩 quad，把 alpha < clip 的区域整块剪掉，
 *     避免为近零 alpha 的像素付带宽（与 gsplat 的 `alpha_clip` 同义）
 *   - `normExp`：把 exp(-4·A) 归一化成在 A=1 处**恰好为 0**，
 *     所以四边形式（quad）边缘的 alpha 严格归零，不出现方形硬边
 */

export const gsplatCommonChunk = {
  name: "gsplatCommon",
  code: `
const EXP4: f32 = exp(-4.0);
const INV_EXP4: f32 = 1.0 / (1.0 - EXP4);

/// A = dot(uv, uv) ∈ [0, 2]（quad 上是 [0,2]，圆外被 discard）
fn normExp(x: f32) -> f32 {
	return (exp(x * -4.0) - EXP4) * INV_EXP4;
}

/// 按 alphaClip 收缩 quad（vertex 阶段，无导数依赖）。
fn clipCorner(corner: ptr<function, SplatCorner>, alpha: f32) {
	let clip: f32 = min(1.0, sqrt(max(0.0, log(alpha / uniforms.alphaClipForward))) * 0.5);
	(*corner).offset = (*corner).offset * clip;
	(*corner).uv = (*corner).uv * clip;
}
`,
} as const
