/**
 * 自有 WGSL：把 RGBAD 累加缓冲 resolve 成下游可直接用的三种产物。
 *
 * 输入（同一次 draw 的两个颜色附件）：
 *   srcColor = (预乘线性 RGB, A)          ← Σ T·α·c  /  Σ T·α
 *   srcDepth = (Σ T·α·z, …, …, A')        ← A' 与 srcColor.a 恒等，仅作交叉校验
 *
 * 输出：
 *   @location(0) preview : rgba8unorm   —— sRGB(直通 RGB) + A，仅用于目视 / PNG
 *   @location(1) rgba    : rgba16float  —— 直通线性 RGB + A（便于下游合成）
 *   @location(2) depth   : rgba32float  —— (D, T, ED, visible)
 *        D = ED/A = gsplat 的 expected depth（真实度量深度，米）
 *        T = 1 - A（透射率，Apple `GetTransmittance` 语义）
 *        ED = 未归一化的 Σ T·α·z（gsplat 的 accumulated depth）
 *        visible = A > 阈值 ? 1 : 0（A→0 处不输出 D，标记背景）
 *
 * 这里**不做**非线性压缩、不做 NDC 转换：深度是线性真实度量深度，
 * NDC / InvZ 由调用方按自己的投影矩阵转换（契约见 `types.ts` 的 `WSplatFrame`）。
 */

export const resolveWgsl = {
  name: "resolve",
  code: `#include "screenQuad"

struct ResolveUniforms {
	// x = alpha 可见阈值：A <= x 视为背景（D 不输出）
	// y/z/w 预留（refine 权重、背景色等）
	params: vec4f,
}

@group(0) @binding(0) var<uniform> uniforms: ResolveUniforms;
@group(0) @binding(1) var srcColor: texture_2d<f32>;
@group(0) @binding(2) var srcDepth: texture_2d<f32>;

struct ResolveOutput {
	@location(0) preview: vec4f,
	@location(1) rgba: vec4f,
	@location(2) depth: vec4f,
}

// 与 sharp/colorspace.ts 的 linearRGB2sRGB 同式（阈值分段，非纯 gamma）。
fn linearToSrgb(x: f32) -> f32 {
	let c: f32 = clamp(x, 0.0, 1.0);
	return select(c * 12.92, 1.055 * pow(c, 1.0 / 2.4) - 0.055, c > 0.0031308);
}

@vertex
fn vsResolve(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4f {
	return screenQuadPosition(vertexIndex);
}

@fragment
fn fsResolve(@builtin(position) fragCoord: vec4f) -> ResolveOutput {
	let coord: vec2i = vec2i(i32(fragCoord.x), i32(fragCoord.y));
	let color: vec4f = textureLoad(srcColor, coord, 0);
	let accum: vec4f = textureLoad(srcDepth, coord, 0);

	let alpha: f32 = clamp(color.a, 0.0, 1.0);
	let invAlpha: f32 = 1.0 / max(alpha, 1e-6);
	let straight: vec3f = max(color.rgb * invAlpha, vec3f(0.0));
	let expectedDepth: f32 = accum.r * invAlpha;
	let visible: f32 = select(0.0, 1.0, alpha > uniforms.params.x);

	var result: ResolveOutput;
	result.preview = vec4f(
		linearToSrgb(straight.r),
		linearToSrgb(straight.g),
		linearToSrgb(straight.b),
		alpha
	);
	result.rgba = vec4f(straight, alpha);
	result.depth = vec4f(expectedDepth * visible, 1.0 - alpha, accum.r, visible);
	return result;
}
`,
} as const
