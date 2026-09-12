/**
 * forked from playcanvas/engine v2.22.2 (MIT)
 * upstream: src/scene/shader-lib/wgsl/chunks/gsplat/vert/gsplatCenter.js
 * commit:   6e6d0d0690830fcd1d91ffde15e9c87a062e8c43
 * modifications:
 *   - 删掉 GSPLAT_FISHEYE 与 GSPLAT_CENTER_NOPROJ 分支：本阶段只有针孔透视
 *   - **相机空间 z 为正**（OpenCV 约定，SHARP 的度量空间），因此：
 *       * 背面剔除由上游的 `centerView.z > 0` 改成 `centerView.z <= eps`
 *       * 不再需要 `camera_params` uniform（上游用它区分正交/鱼眼）
 *   - 新增：把 `viewDepth`（真实度量深度，米）随 center 一起带出，
 *     供片元写入深度附件。上游没有这条（它的 depth 走 shadow/prepass）
 *   - 新增：背面剔除处 `countOnce` 原子计数（每帧剔除统计，见 `splatData.ts`）
 */

export const gsplatCenterChunk = {
  name: "gsplatCenter",
  code: `
// 视图空间 z <= eps 一律丢弃：零附近投影会发散，负值在相机后方。
const SPLAT_VIEW_Z_EPS: f32 = 1e-6;

fn initCenter(source: ptr<function, SplatSource>, center: ptr<function, SplatCenter>) -> bool {
	let modelView: mat4x4f = uniforms.matrix_view;
	let centerView: vec4f = modelView * vec4f((*source).modelCenter, 1.0);
	if (centerView.z <= SPLAT_VIEW_Z_EPS) {
		countOnce((*source).cornerUV, SPLAT_STAT_BEHIND_CAMERA);
		return false;
	}
	var centerProj: vec4f = uniforms.matrix_projection * centerView;
	// 上游同款：夹住 clip z，避免近平面附近的 splat 出现 z<0 的退化四边形
	centerProj.z = clamp(centerProj.z, 0.0, abs(centerProj.w));
	(*center).proj = centerProj;
	(*center).projMat00 = uniforms.matrix_projection[0][0];
	(*center).view = centerView.xyz / centerView.w;
	(*center).modelView = modelView;
	return true;
}
`,
} as const
