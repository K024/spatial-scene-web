/**
 * 自有 WGSL（**不是** fork）：`SplatSource` 的装配（quad 角点 + 从 storage 拉数据）。
 *
 * 从 `quad.ts` 提出来单独成 chunk，是因为现在有**两个**入口要用它：
 *   - `quad.ts` 的 `vsSplat`（真正的光栅化）
 *   - `cullStats.ts` 的 `csCountCulls`（剔除统计的 compute pass）
 * 两者必须走**完全相同**的取数与角点逻辑，否则统计与实画会对不上。
 */

export const splatSourceChunk = {
  name: "splatSource",
  code: `
/// 6 个顶点覆盖 [-1,1]² 的 quad：v0..2 = (-1,-1),(1,-1),(-1,1)；v3..5 = (-1,1),(1,-1),(1,1)
fn cornerUv(vertexIndex: u32) -> vec2f {
	let x: f32 = select(-1.0, 1.0, vertexIndex == 1u || vertexIndex == 4u || vertexIndex == 5u);
	let y: f32 = select(-1.0, 1.0, vertexIndex == 2u || vertexIndex == 3u || vertexIndex == 5u);
	return vec2f(x, y);
}

fn initSource(source: ptr<function, SplatSource>, instanceIndex: u32, vertexIndex: u32) -> bool {
	let orderIndex: u32 = instanceIndex;
	if (orderIndex >= uniforms.numSplats) {
		return false;
	}
	let splatIndex: u32 = splatOrder[orderIndex];
	(*source).index = splatIndex;
	(*source).order = orderIndex;
	(*source).cornerUV = cornerUv(vertexIndex);
	(*source).modelCenter = getMean(splatIndex);
	(*source).rotation = getRotation(splatIndex);
	(*source).scale = getScale(splatIndex);
	(*source).color = getRgba(splatIndex);
	return true;
}
`,
} as const
