/**
 * forked from playcanvas/engine v2.22.2 (MIT)
 * upstream: src/scene/shader-lib/wgsl/chunks/gsplat/vert/gsplatStructs.js
 * commit:   6e6d0d0690830fcd1d91ffde15e9c87a062e8c43
 * modifications:
 *   - 删掉 GSPLAT_FISHEYE 分支（本阶段只做针孔透视）
 *   - 删掉 2DGS 相关字段
 *   - half -> f32（一律 f32）
 *   - SplatSource 增加 rotation/scale/color：上游从 texture 流里读，
 *     我们用 storage buffer（见 splatData.ts），所以数据随 source 一起传
 */

export const gsplatStructsChunk = {
  name: "gsplatStructs",
  code: `
struct SplatSource {
	// 原始 splat 下标
	index: u32,
	// 排序后的位置（= instance index）
	order: u32,
	cornerUV: vec2f,
	// 世界/模型空间数据
	modelCenter: vec3f,
	rotation: vec4f,
	scale: vec3f,
	// (线性 RGB, opacity)
	color: vec4f,
}

struct SplatCenter {
	// 视图空间中心（相机 x 右 / y 下 / z 前）
	view: vec3f,
	// clip 空间中心（w = 视图空间 z）
	proj: vec4f,
	modelView: mat4x4f,
	projMat00: f32,
}

struct SplatCorner {
	// clip 空间偏移（xy 有效）
	offset: vec3f,
	uv: vec2f,
	// 抗锯齿的透明度补偿：sqrt(det Σ / det(Σ + eps2d·I))
	aaFactor: f32,
}
`,
} as const
