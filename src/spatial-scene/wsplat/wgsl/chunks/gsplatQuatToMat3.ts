/**
 * forked from playcanvas/engine v2.22.2 (MIT)
 * upstream: src/scene/shader-lib/wgsl/chunks/gsplat/vert/gsplatQuatToMat3.js
 * commit:   6e6d0d0690830fcd1d91ffde15e9c87a062e8c43
 * modifications:
 *   - 输入由 half4 改成 f32（我们一律 f32，见 data.ts 的精度取舍）
 *   - quatMul 暂时保留（上游同文件导出，将来做 SH 旋转时用）
 */

export const gsplatQuatToMat3Chunk = {
  name: "gsplatQuatToMat3",
  code: `
fn quatToMat3(r: vec4f) -> mat3x3f {
	let r2: vec4f = r + r;
	let x: f32 = r2.x * r.w;
	let y: vec4f = r2.y * r;
	let z: vec4f = r2.z * r;
	let w: f32 = r2.w * r.w;
	return mat3x3f(
		vec3f(1.0 - z.z - w, y.z + x, y.w - z.x),
		vec3f(y.z - x, 1.0 - y.y - w, z.w + y.x),
		vec3f(y.w + z.x, z.w - y.x, 1.0 - y.y - z.z)
	);
}
fn quatMul(a: vec4f, b: vec4f) -> vec4f {
	return vec4f(
		a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
		a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
		a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
		a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
	);
}
`,
} as const
