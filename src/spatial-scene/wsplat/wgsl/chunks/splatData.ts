/**
 * 自有 WGSL（**不是** fork）：splat 数据的 storage buffer 读取。
 *
 * 上游 playcanvas 从 **texture** 读（`textureLoad` + 压缩格式解码，共 56 个 chunk），
 * 我们走 **storage buffer + vertex pulling**（quad 路），
 * 所以这一层必须自有；上游的 `gsplatSource` / `gsplatSplat` / `format*` 全部不 fork。
 *
 * 缓冲分组见 `data.ts`：RGBA 一组、D/几何一组（用户决策）。
 *
 * ── 每帧剔除统计（可观测性，见 `types.ts` 的 `WSplatStats`）──
 * 分层（LDI）场景里最危险的失败模式是「某一层的高斯被静默剔空」：
 * `minPixelSize` 是像素量纲，实测 384 宽下 **73.55%** 的高斯会被剔掉（原分辨率 0%）。
 * 所以顶点阶段把每个剔除原因原子计数到 `splatStats`，由 CPU 每帧清零 + 读回。
 * 计数器语义与顺序严格对应 `vsSplat` 的早退顺序（见 `quad.ts`）。
 */

export const splatDataChunk = {
  name: "splatData",
  code: `
struct SplatUniforms {
	matrix_view: mat4x4f,
	matrix_projection: mat4x4f,
	// (width, height, 1/width, 1/height) —— 与上游同名同义
	viewport_size: vec4f,
	minPixelSize: f32,
	// **数据**里有多少个高斯（storage buffer 的边界）
	numSplats: u32,
	alphaClipForward: f32,
	// 运行期调试开关：0 = 关闭；1 = 只画 alpha 覆盖（overdraw 用）
	debugMode: f32,
}

@group(0) @binding(0) var<uniform> uniforms: SplatUniforms;

// RGBA 组：每 splat (r, g, b, opacity)，线性 RGB
@group(0) @binding(1) var<storage, read> splatRgba: array<vec4f>;

// D/几何组：每 splat 10 个 f32 —— [mx,my,mz, sx,sy,sz, qw,qx,qy,qz]
@group(0) @binding(2) var<storage, read> splatGeometry: array<f32>;

// back-to-front 的 splat 下标
@group(0) @binding(3) var<storage, read> splatOrder: array<u32>;

// 每帧剔除统计（原子计数，8 个槽位；只有 compute 模块才声明与写入）。
//
// vertex 阶段不允许 read_write storage（WebGPU 硬限制），所以整段用
// SPLAT_COUNT_CULLS 开关；统计实际由 wgsl/cullStats.ts 的 compute pass 产出。
#if SPLAT_COUNT_CULLS
@group(0) @binding(4) var<storage, read_write> splatStats: array<atomic<u32>, 8>;
#endif

const SPLAT_STAT_DRAWN: u32 = 0u;
const SPLAT_STAT_BOUNDS: u32 = 1u;
const SPLAT_STAT_ALPHA_CLIP: u32 = 2u;
const SPLAT_STAT_ALPHA_CLIP_AA: u32 = 3u;
const SPLAT_STAT_BEHIND_CAMERA: u32 = 4u;
const SPLAT_STAT_MIN_PIXEL_SIZE: u32 = 5u;
const SPLAT_STAT_FRUSTUM: u32 = 6u;

/// 每个高斯**只计一次**。
///
/// vsSplat 每个实例有 6 个顶点（两个三角形），直接在 helper 里原子加会 ×6。
/// cornerUV == (-1,-1) 恰好只在 vertex_index == 0 成立（见 quad.ts 的 cornerUv），
/// 而 cornerUV 在所有剔除点之前就已写入 SplatSource，所以不必改上游函数签名。
fn countOnce(cornerUV: vec2f, reason: u32) {
#if SPLAT_COUNT_CULLS
	if (cornerUV.x == -1.0 && cornerUV.y == -1.0) {
		atomicAdd(&splatStats[reason], 1u);
	}
#endif
}

fn countOnceAtVertex(vertexIndex: u32, reason: u32) {
#if SPLAT_COUNT_CULLS
	if (vertexIndex == 0u) {
		atomicAdd(&splatStats[reason], 1u);
	}
#endif
}

const GEOMETRY_STRIDE: u32 = 10u;

fn getMean(index: u32) -> vec3f {
	let base: u32 = index * GEOMETRY_STRIDE;
	return vec3f(splatGeometry[base], splatGeometry[base + 1u], splatGeometry[base + 2u]);
}

fn getScale(index: u32) -> vec3f {
	let base: u32 = index * GEOMETRY_STRIDE;
	return vec3f(splatGeometry[base + 3u], splatGeometry[base + 4u], splatGeometry[base + 5u]);
}

/// 四元数（w 在前）。归一化放在这里（而不是 CPU 侧 118 万次 sqrt）。
fn getRotation(index: u32) -> vec4f {
	let base: u32 = index * GEOMETRY_STRIDE;
	let q: vec4f = vec4f(
		splatGeometry[base + 6u],
		splatGeometry[base + 7u],
		splatGeometry[base + 8u],
		splatGeometry[base + 9u]
	);
	let len: f32 = length(q);
	return select(vec4f(1.0, 0.0, 0.0, 0.0), q / len, len > 1e-8);
}

/// (线性 RGB, opacity)
fn getRgba(index: u32) -> vec4f {
	return splatRgba[index];
}
`,
} as const
