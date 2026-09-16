/**
 * forked from playcanvas/engine v2.22.2 (MIT)
 * upstream: src/scene/shader-lib/wgsl/chunks/gsplat/vert/gsplatEvalSH.js
 * commit:   6e6d0d0690830fcd1d91ffde15e9c87a062e8c43
 * modifications:
 *   - half -> f32（与 data.ts 的「一律 f32」一致）
 *   - 删除 `#if SH_BANDS > 0` 外层包裹：本 chunk 只在 `SH_BANDS > 0` 时被 include
 *     （`quad.ts` 负责条件 include），保留 `SH_COEFFS` 常量
 *
 * 本阶段 SHARP 只给 DC（SH_BANDS = 0），所以这条路径默认不参与编译；
 * 保留它只是为了把接口留出来（SHARP 只给 DC，用不上 1..3 度）。
 */

export const gsplatEvalSHChunk = {
  name: "gsplatEvalSH",
  code: `
#if SH_BANDS == 1
	const SH_COEFFS: i32 = 3;
#elif SH_BANDS == 2
	const SH_COEFFS: i32 = 8;
#elif SH_BANDS == 3
	const SH_COEFFS: i32 = 15;
#else
	const SH_COEFFS: i32 = 0;
#endif

const SH_C1: f32 = 0.4886025119029199;
#if SH_BANDS > 1
	const SH_C2_0: f32 = 1.0925484305920792;
	const SH_C2_1: f32 = -1.0925484305920792;
	const SH_C2_2: f32 = 0.31539156525252005;
	const SH_C2_3: f32 = -1.0925484305920792;
	const SH_C2_4: f32 = 0.5462742152960396;
#endif
#if SH_BANDS > 2
	const SH_C3_0: f32 = -0.5900435899266435;
	const SH_C3_1: f32 = 2.890611442640554;
	const SH_C3_2: f32 = -0.4570457994644658;
	const SH_C3_3: f32 = 0.3731763325901154;
	const SH_C3_4: f32 = -0.4570457994644658;
	const SH_C3_5: f32 = 1.445305721320277;
	const SH_C3_6: f32 = -0.5900435899266435;
#endif

fn evalSH(sh: ptr<function, array<vec3f, SH_COEFFS>>, dir: vec3f) -> vec3f {
	let d: vec3f = dir;
	var result: vec3f = SH_C1 * (-sh[0] * d.y + sh[1] * d.z - sh[2] * d.x);
	#if SH_BANDS > 1
		let xx: f32 = d.x * d.x;
		let yy: f32 = d.y * d.y;
		let zz: f32 = d.z * d.z;
		let xy: f32 = d.x * d.y;
		let yz: f32 = d.y * d.z;
		let xz: f32 = d.x * d.z;
		result = result + (
			sh[3] * (SH_C2_0 * xy) +
			sh[4] * (SH_C2_1 * yz) +
			sh[5] * (SH_C2_2 * (2.0 * zz - xx - yy)) +
			sh[6] * (SH_C2_3 * xz) +
			sh[7] * (SH_C2_4 * (xx - yy))
		);
	#endif
	#if SH_BANDS > 2
		result = result + (
			sh[8]  * (SH_C3_0 * d.y * (3.0 * xx - yy)) +
			sh[9]  * (SH_C3_1 * xy * d.z) +
			sh[10] * (SH_C3_2 * d.y * (4.0 * zz - xx - yy)) +
			sh[11] * (SH_C3_3 * d.z * (2.0 * zz - 3.0 * xx - 3.0 * yy)) +
			sh[12] * (SH_C3_4 * d.x * (4.0 * zz - xx - yy)) +
			sh[13] * (SH_C3_5 * d.z * (xx - yy)) +
			sh[14] * (SH_C3_6 * d.x * (xx - 3.0 * yy))
		);
	#endif
	return result;
}
`,
} as const
