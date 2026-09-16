/**
 * 通路验证用的 WGSL：全屏纯色「splat」。
 *
 * 它故意走**和真 splat 完全相同的输出契约**（见 `quad.ts`）：两个 `rgba16float`
 * 颜色附件、同一套预乘 blend `src=one / dst=oneMinusSrcAlpha`，其中
 *   - 附件 0 = (rgb·α, α)
 *   - 附件 1 = (z·α, 0, 0, α)   ← α 必须写在第 4 通道！
 *
 * ⚠ 关于附件 1 的第 4 通道（容易写错，已实测）：
 * WebGPU 的 blend 因子取自**同一个附件自己的** fragment 输出，附件之间不共享 α。
 * 若附件 1 只写 `.r` 而把 `.a` 留 0，则 dst 因子变成 `1 - 0 = 1`，深度不会随
 * 前面图层衰减，累加结果就不是 `Σ T·α·z`。因此必须把 α 一并写进附件 1 的 `.a`。
 */

export const blitWgsl = {
  name: "blit",
  code: `#include "screenQuad"

struct BlitUniforms {
	// 线性 RGB + straight α（未预乘；预乘在片元里做）
	color: vec4f,
	// x = 该「splat」所在视图空间真实深度（米），其余预留
	depth: vec4f,
}

@group(0) @binding(0) var<uniform> uniforms: BlitUniforms;

struct BlitOutput {
	@location(0) color: vec4f,
	@location(1) depth: vec4f,
}

@vertex
fn vsBlit(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4f {
	return screenQuadPosition(vertexIndex);
}

@fragment
fn fsBlit() -> BlitOutput {
	let alpha: f32 = clamp(uniforms.color.a, 0.0, 1.0);
	var result: BlitOutput;
	result.color = vec4f(uniforms.color.rgb * alpha, alpha);
	result.depth = vec4f(uniforms.depth.x * alpha, 0.0, 0.0, alpha);
	return result;
}
`,
} as const
