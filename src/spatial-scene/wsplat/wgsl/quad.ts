/**
 * 自有 WGSL：splat 的 instanced-quad 顶点/片元。
 *
 * 为什么是 quad 路而不是 tile 路：tile 路要 prefix sum + radix + tile-intersect +
 * indirect args，只有性能诉求才需要，与「准确度优先」冲突。
 *
 * 零顶点缓冲：`draw(6, numSplats)`，角落由 `@builtin(vertex_index)` 生成，
 * 高斯数据按 `@builtin(instance_index)` 从 storage buffer 拉（vertex pulling）。
 * SuperSplat / PlayCanvas / Scthe 都是这条路，可逐行对照。
 *
 * 输出契约（两个 `rgba16float` 附件、同一套预乘 blend）：
 *   @location(0) color 附件 = (线性 RGB·α, α)
 *   @location(1) 深度附件   = (z·α, 0, 0, **α**)
 * α 必须同时写进深度附件的第 4 通道，否则该附件的 dst 因子恒为 1、深度不衰减
 * （已实测：blend 因子取自**同一个附件自己的** fragment 输出，附件之间**不共享** α）。
 *
 * 片元阶段不做 shadow/pick/prepass/dither（playcanvas 的场景耦合，本模块不要）。
 *
 * 顶点阶段的每个剔除点都会往 `splatStats` 原子计数一次（每实例一次，
 * 见 `splatData.ts` 的 `countOnce*`）；计数器每帧由 CPU 清零后读回，
 * 用于分层（LDI）时判断「某一层是不是被静默剔空了」。
 */

export const quadWgsl = {
  name: "quad",
  code: `#include "splatData"
#include "splatSource"
#include "gsplatStructs"
#include "gsplatQuatToMat3"
#if SH_BANDS > 0
	#include "gsplatEvalSH"
#endif
#include "gsplatCenter"
#include "gsplatCorner"
#include "gsplatCommon"

// 被剔除的高斯：把顶点丢到裁剪体外（z = 2 > w = 1），退化图元不出像素
const discardVec: vec4f = vec4f(0.0, 0.0, 2.0, 1.0);

struct SplatVarying {
	@builtin(position) position: vec4f,
	// quad 归一化坐标（A = dot(uv,uv)）
	@location(0) gaussianUV: vec2f,
	// (线性 RGB, α)（α 已含 AA 补偿与 clip）
	@location(1) gaussianColor: vec4f,
	// 真实度量深度（米）= 视图空间 z
	@location(2) viewDepth: f32,
}

@vertex
fn vsSplat(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> SplatVarying {
	var output: SplatVarying;
	output.position = discardVec;
	output.gaussianUV = vec2f(0.0);
	output.gaussianColor = vec4f(0.0);
	output.viewDepth = 0.0;

	var source: SplatSource;
	if (!initSource(&source, instanceIndex, vertexIndex)) {
		// 这里 cornerUV 还没写，只能用 vertexIndex 去重
		countOnceAtVertex(vertexIndex, SPLAT_STAT_BOUNDS);
		return output;
	}
	if (source.color.a <= uniforms.alphaClipForward) {
		countOnce(source.cornerUV, SPLAT_STAT_ALPHA_CLIP);
		return output;
	}

	var center: SplatCenter;
	if (!initCenter(&source, &center)) {
		return output;
	}
	var corner: SplatCorner;
	if (!initCorner(&source, &center, &corner)) {
		return output;
	}

	var color: vec4f = source.color;
	#if GSPLAT_AA
		// 透明度补偿：Σ 加 eps2d 之后总能量变大，这里乘 sqrt(det Σ / det(Σ+eps2d·I)) 抵回来
		color.a = color.a * corner.aaFactor;
	#endif
	#if SH_BANDS > 0
		// 视角方向取在**模型**空间（上游同款）：modelView 的旋转部分作用于视图方向
		let modelView3x3: mat3x3f = mat3x3f(center.modelView[0].xyz, center.modelView[1].xyz, center.modelView[2].xyz);
		let dir: vec3f = normalize(center.view * modelView3x3);
		var sh: array<vec3f, SH_COEFFS>;
		// TODO: SH 系数流尚未接入（SHARP 只有 DC），接口先留着
		readSHData(&sh);
		color = vec4f(color.rgb + evalSH(&sh, dir), color.a);
	#endif

	if (color.a <= uniforms.alphaClipForward) {
		countOnce(source.cornerUV, SPLAT_STAT_ALPHA_CLIP_AA);
		return output;
	}
	clipCorner(&corner, color.a);

	// 过了全部早退：真正参与光栅化的高斯（每实例只计一次）
	countOnce(source.cornerUV, SPLAT_STAT_DRAWN);

	// quad 顶点的 clip 位置 = 中心 + 偏移（偏移已含 proj.w）
	output.position = center.proj + vec4f(corner.offset.xyz, 0.0);
	output.gaussianUV = corner.uv;
	output.gaussianColor = color;
	output.viewDepth = center.view.z;
	return output;
}

struct SplatFragment {
	@location(0) color: vec4f,
	@location(1) depth: vec4f,
}

@fragment
fn fsSplat(input: SplatVarying) -> SplatFragment {
	let A: f32 = dot(input.gaussianUV, input.gaussianUV);
	if (A > 1.0) {
		discard;
	}
	var alpha: f32 = exp(-GAUSS_K2 * A) * input.gaussianColor.a;
	// gsplat 语义：单颗高斯的逐像素 alpha 上限
	alpha = min(alpha, MAX_ALPHA);
	if (alpha < uniforms.alphaClipForward) {
		discard;
	}
	var output: SplatFragment;
	let rgb: vec3f = input.gaussianColor.rgb;
	output.color = vec4f(rgb * alpha, alpha);
	output.depth = vec4f(input.viewDepth * alpha, 0.0, 0.0, alpha);
	return output;
}
`,
} as const
