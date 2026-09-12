/**
 * forked from playcanvas/engine v2.22.2 (MIT)
 * upstream: src/scene/shader-lib/wgsl/chunks/gsplat/vert/gsplatCorner.js
 * commit:   6e6d0d0690830fcd1d91ffde15e9c87a062e8c43
 * modifications:
 *   - 删掉 GSPLAT_FISHEYE / GSPLAT_2DGS 分支（本阶段只做 3DGS 针孔透视）
 *   - half -> f32（一律 f32；上游为省带宽用 half3x3，我们保留 f32 精度）
 *   - **透视雅可比的 y 行取反**（关键，见下方长注释）
 *   - eps2d = 0.3、透明度补偿 sqrt(det Σ / det(Σ+0.3I))、minPixelSize 剔除、
 *     视锥剔除、`l = 2·min(sqrt(2λ), vmin)` 全部与上游逐字一致（「不要自创数值」）
 *   - **新增各向同性退化保护**（见 diagonalVector 处）：上游 normalize((0,0)) = NaN，
 *     会让「投影后 2×2 协方差恰好各向同性」的高斯整个消失（quad 位置全 NaN）。
 *     真实数据里几乎不触发，但合成/极端数据下会静默丢高斯，故用 `select` 兜底。
 *   - 新增：两处剔除（minPixelSize / 视锥）各加一行 `countOnce` 原子计数
 *     （每帧剔除统计，见 `splatData.ts` 与 `types.ts` 的 `WSplatStats`）
 *
 * ── 为什么 y 行要取反（务必读完再改）──
 * 上游相机是右手系朝 -z、相机 y 向上、NDC y 向上，三者同向，所以雅可比两行同号。
 * 我们是 OpenCV 约定：相机 y **向下**、z 向前，而 NDC y **向上**。于是
 *     ndc_x =  (2f/W)·x/z
 *     ndc_y = -(2f/H)·y/z
 * 因此雅可比第二行相对第一行整体取反：
 *     Jx = [ F/z, 0, -F·x/z² ]
 *     Jy = [ 0, -F/z, +F·y/z² ]   ← 与上游符号相反
 * 若漏掉：2×2 协方差的 off-diagonal 符号会错，**协方差各向同性时看不出问题**，
 * 一旦高斯椭球有倾斜，屏幕上的椭圆就会沿 y 镜像（只有 CPU 对齐才会暴露）。
 *
 * 量纲自洽（不要"顺手化简"这些因子）：
 * `focal = viewport_size.x · projMat00 = W·(2f/W) = 2f`，比 gsplat 的 `f` 大一倍；
 * 后来 `c = proj.w · (1/W, 1/H)` 把「像素 -> NDC」的 2 倍补回来，
 * 最终 quad 半轴折算回像素是 ≈ 2.83σ（与 gsplat 的 3σ 同族）。
 *
 * ── ⚠ 未决项：`eps2d` 的尺度（需要决策，未定前不要改）──
 * 上游在 `focal = 2·f_px` 的协方差上硬编码 `+0.3`，而 gsplat 的 `eps2d = 0.3`
 * 是加在**真像素焦距**的协方差上（两者差 `(2f/f)² = 4` 倍）。
 * 所以**本模块目前的 AA 模糊比 gsplat 弱 4 倍**。影响面很小：只影响亚像素高斯，
 * 而这类高斯绝大多数已被 `minPixelSize` 剔掉（实测可观测差异 = 原分辨率下
 * 682/1,179,648 = 0.058% 的高斯被 AA 后的 alphaClip 剔掉）。
 * 两条路，二选一（选完要同步改 `scripts/utils/wsplat-cpu.ts` 的口径）：
 *   (a) 与上游逐字一致（现状，0.3）——CPU 参考已按这个尺度对齐；
 *   (b) 对齐 gsplat 的语义：改成 `1.2`（= 0.3 × (2f/f)²），并在上面 `modifications`
 *       里登记为「按尺度换算得到的派生值，不是自创参数」。
 */

export const gsplatCornerChunk = {
  name: "gsplatCorner",
  code: `
/// 由四元数（w 在前）与尺度求 3D 协方差的上三角（6 个独立分量存成 covA/covB）。
/// 上游同款：M = S·R，cov = Mᵀ M 的六个独立项。
fn computeCovariance(rotation: vec4f, scale: vec3f, covA_ptr: ptr<function, vec3f>, covB_ptr: ptr<function, vec3f>) {
	let rot: mat3x3f = quatToMat3(rotation);
	let s: vec3f = scale;
	let M: mat3x3f = transpose(mat3x3f(
		s.x * rot[0],
		s.y * rot[1],
		s.z * rot[2]
	));
	*covA_ptr = vec3f(dot(M[0], M[0]), dot(M[0], M[1]), dot(M[0], M[2]));
	*covB_ptr = vec3f(dot(M[1], M[1]), dot(M[1], M[2]), dot(M[2], M[2]));
}

/// 由屏幕空间协方差求 quad 的 clip 空间偏移；返回 false = 剔除该高斯。
fn initCornerCov(source: ptr<function, SplatSource>, center: ptr<function, SplatCenter>, corner: ptr<function, SplatCorner>, covA: vec3f, covB: vec3f) -> bool {
	let Vrk: mat3x3f = mat3x3f(
		vec3f(covA.x, covA.y, covA.z),
		vec3f(covA.y, covB.x, covB.y),
		vec3f(covA.z, covB.y, covB.z)
	);

	let focal: f32 = uniforms.viewport_size.x * (*center).projMat00;
	let v: vec3f = (*center).view;

	// ── 透视雅可比（y 行符号由相机约定决定，理由见文件头）──
	let J1: f32 = focal / v.z;
	let J2: vec2f = -J1 / v.z * v.xy;
	#if GSPLAT_CAMERA_Y_DOWN
		// 相机 y 向下 + NDC y 向上 => 第二行整体取反
		let Jy1: f32 = -J1;
		let Jy2: f32 = -J2.y;
	#else
		// 相机 y 向上（上游 playcanvas 的约定）
		let Jy1: f32 = J1;
		let Jy2: f32 = J2.y;
	#endif
	let J: mat3x3f = mat3x3f(
		vec3f(J1, 0.0, J2.x),
		vec3f(0.0, Jy1, Jy2),
		vec3f(0.0, 0.0, 0.0)
	);

	// W = 视图旋转的转置（世界 -> 视图）。我们的世界坐标有时就是相机坐标（SHARP），
	// 但矩阵照算，保证一般位姿也正确。
	let modelView = (*center).modelView;
	let W: mat3x3f = transpose(mat3x3f(modelView[0].xyz, modelView[1].xyz, modelView[2].xyz));
	let T: mat3x3f = W * J;
	let cov: mat3x3f = transpose(T) * Vrk * T;

	// 抗锯齿：denominator 加 eps2d = 0.3（单位：像素²，对齐 gsplat）
	let detOrig: f32 = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];
	let detBlur: f32 = (cov[0][0] + 0.3) * (cov[1][1] + 0.3) - cov[0][1] * cov[1][0];
	(*corner).aaFactor = sqrt(max(detOrig / detBlur, 0.0));

	// 椭圆主轴（对角加 eps2d）
	let diagonal1: f32 = cov[0][0] + 0.3;
	let offDiagonal: f32 = cov[0][1];
	let diagonal2: f32 = cov[1][1] + 0.3;

	let mid: f32 = 0.5 * (diagonal1 + diagonal2);
	let radius: f32 = length(vec2f((diagonal1 - diagonal2) / 2.0, offDiagonal));
	let lambda1: f32 = mid + radius;
	let lambda2: f32 = max(mid - radius, 0.1);

	// 内核大小相对屏幕分辨率设上限（上游同款）
	let vmin: f32 = min(1024.0, min(uniforms.viewport_size.x, uniforms.viewport_size.y));

	let l1: f32 = 2.0 * min(sqrt(2.0 * lambda1), vmin);
	let l2: f32 = 2.0 * min(sqrt(2.0 * lambda2), vmin);

	// 小于 minPixelSize 的剔除（阈值取自上游）
	if (max(l1, l2) < uniforms.minPixelSize) {
		countOnce((*source).cornerUV, SPLAT_STAT_MIN_PIXEL_SIZE);
		return false;
	}

	let c: vec2f = (*center).proj.ww * uniforms.viewport_size.zw;

	// 视锥 x/y 剔除
	if (any((abs((*center).proj.xy) - vec2f(max(l1, l2)) * c) > (*center).proj.ww)) {
		countOnce((*source).cornerUV, SPLAT_STAT_FRUSTUM);
		return false;
	}

	// 椭圆主轴方向。
	// **退化保护（本仓库新增）**：各向同性时 (offDiagonal, λ1-diagonal1) = (0,0)，
	// normalize 会给出 NaN，于是 quad 位置全变 NaN，该高斯**整个不画**
	// （不报错、不报警，只是消失）。此时椭圆是正圆，任何单位方向都等价，
	// 故兜底成 (1,0)。上游没有这个保护：真实 splat 数据里几乎不出现恰好各向同性，
	// 但合成数据 / 球对称高斯会命中，所以必须修。
	let diagRaw: vec2f = vec2f(offDiagonal, lambda1 - diagonal1);
	let diagLen: f32 = length(diagRaw);
	let diagonalVector: vec2f = select(vec2f(1.0, 0.0), diagRaw / diagLen, diagLen > 1e-12);
	let v1: vec2f = l1 * diagonalVector;
	let v2: vec2f = l2 * vec2f(diagonalVector.y, -diagonalVector.x);

	(*corner).offset = vec3f(((*source).cornerUV.x * v1 + (*source).cornerUV.y * v2) * c, 0.0);
	(*corner).uv = (*source).cornerUV;

	return true;
}

fn initCorner(source: ptr<function, SplatSource>, center: ptr<function, SplatCenter>, corner: ptr<function, SplatCorner>) -> bool {
	var covA: vec3f;
	var covB: vec3f;
	computeCovariance((*source).rotation, (*source).scale, &covA, &covB);
	return initCornerCov(source, center, corner, covA, covB);
}
`,
} as const
