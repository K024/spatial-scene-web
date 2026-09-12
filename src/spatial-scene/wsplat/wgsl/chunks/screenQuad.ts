/**
 * 自有 WGSL helper：全屏三角形（blit / resolve 等后处理 pass 共用）。
 *
 * 不是 fork：上游没有等价物（playcanvas 的后处理走 quadRenderer）。
 * 用「一个大三角形」而不是两个三角形的 quad：没有对角线接缝，顶点更少。
 */

export const screenQuadChunk = {
  name: "screenQuad",
  code: `
// 覆盖整个裁剪空间的三角形，顶点坐标直接当 uv 用：
//   v0 = (-1,-1)  v1 = (3,-1)  v2 = (-1,3)
fn screenQuadPosition(vertexIndex: u32) -> vec4f {
	let x: f32 = select(-1.0, 3.0, vertexIndex == 1u);
	let y: f32 = select(-1.0, 3.0, vertexIndex == 2u);
	return vec4f(x, y, 0.0, 1.0);
}

// 裁剪空间坐标 -> [0,1] 的纹理坐标（y 向下，与 WebGPU 纹理行序一致）。
fn screenQuadUv(clip: vec4f) -> vec2f {
	return vec2f(clip.x * 0.5 + 0.5, 0.5 - clip.y * 0.5);
}
`,
} as const
