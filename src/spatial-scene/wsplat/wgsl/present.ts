/**
 * 自有 WGSL：把内部 resolve 预览纹理原样送进画布（swapchain）。
 *
 * 存在的理由：canvas 的 `getPreferredCanvasFormat()` 可能是 `bgra8unorm`
 * （与内部 `rgba8unorm` 预览不同格式），所以展示需要一次显式的格式转换 pass，
 * 而不是让 resolve 直接往画布写（那样 resolve 的 pipeline 就得为每种画布格式各建一份）。
 */

export const presentWgsl = {
  name: "present",
  code: `#include "screenQuad"

@group(0) @binding(0) var srcTexture: texture_2d<f32>;

@vertex
fn vsPresent(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4f {
	return screenQuadPosition(vertexIndex);
}

@fragment
fn fsPresent(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
	return textureLoad(srcTexture, vec2i(i32(fragCoord.x), i32(fragCoord.y)), 0);
}
`,
} as const
