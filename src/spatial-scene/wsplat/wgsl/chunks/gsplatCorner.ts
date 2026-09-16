/**
 * forked from playcanvas/engine v2.22.2 (MIT)
 * upstream: src/scene/shader-lib/wgsl/chunks/gsplat/vert/gsplatCorner.js
 * commit:   6e6d0d0690830fcd1d91ffde15e9c87a062e8c43
 * modifications:
 *   - 删掉 GSPLAT_FISHEYE / GSPLAT_2DGS 分支（本阶段只做 3DGS 针孔透视）
 *   - half -> f32（一律 f32；上游为省带宽用 half3x3，我们保留 f32 精度）
 *   - **透视雅可比的 y 行取反**（关键，见下方长注释）
 *   - **AA / 核改为 gsplat 语义**（不再逐字沿用上游的 0.3 / 2.83σ / normExp）：
 *       * `EPS2D` 由 `WSplatRendererOptions.eps2d` 乘 4 注入。本文件的 Σ2 建在
 *         `focal = 2·f_px` 上（4 倍），而 gsplat 的 `eps2d` 加在真像素焦距的 Σ2 上。
 *         默认 **0**（对齐 SHARP/ml-sharp 官方渲染）；gsplat 默认 0.3 即注入 1.2。
 *         这是**按尺度换算得到的派生值，不是自创参数**；
 *       * 补偿因子加 `MIN_COMPENSATION = 0.005` 下限（gsplat 同款，防止极小高斯补偿趋 0）；
 *       * quad 半轴取 `3.33σ`（gsplat 的 radius 常数：`exp(-0.5·3.33²) ≈ 1/255`
 *         恰好等于 `alphaClip`）。片元改用**真高斯** `exp(-0.5·r²)`（见 `quad.ts`），
 *         所以 quad 边缘正好落在 alphaClip 上，无需上游的 `normExp` 边缘归一化。
 *         推论：上游「quad 半轴 ≈ 2.83σ、边缘 alpha 归零」的尾部近似被去掉，
 *         与 gsplat 的核逐点一致（尾部差从 ~1.8e-2 降到舍入级）。
 *   - minPixelSize 剔除、视锥剔除保留（分辨率相关的性能剔除，非 gsplat 语义）
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
 * ── 量纲自洽（不要"顺手化简"这些因子）──
 * `focal = viewport_size.x · projMat00 = W·(2f/W) = 2f`，比 gsplat 的 `f` 大一倍。
 * 因此这里的 `cov`（以及注入的 `EPS2D`）都在「(2·像素)²」量纲上；quad 半轴
 * `pixelRadius = 0.5·3.33·sqrt(λ)` 折算回像素正好是 `3.33σ`，与 gsplat 的 radius 同义。
 */

export const gsplatCornerChunk = {
  name: "gsplatCorner",
  code: `
/// 低通滤波的对角加项。由 WSplatRendererOptions.eps2d（gsplat 语义，默认 0）乘 4 注入
/// （本文件的 Σ2 建在 2·f_px 上）。eps2d = 0 即关低通（ml-sharp 的 classic 口径）。
const EPS2D: f32 = {GSPLAT_EPS2D};
/// 补偿因子下限（gsplat 的 MIN_COMPENSATION）。
const MIN_COMPENSATION: f32 = 0.005;
/// quad 半轴 = GAUSS_SIGMA_RADIUS·σ（gsplat 的 radius 常数）。
const GAUSS_SIGMA_RADIUS: f32 = 3.33;
/// 真高斯的指数系数：alpha = exp(-GAUSS_K2 · A)，A = r² / GAUSS_SIGMA_RADIUS²。
const GAUSS_K2: f32 = 0.5 * GAUSS_SIGMA_RADIUS * GAUSS_SIGMA_RADIUS;

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

	// 抗锯齿：denominator 加 EPS2D（gsplat 的 eps2d 换算到本文件的 2·f_px 尺度）
	let detOrig: f32 = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];
	let detBlur: f32 = (cov[0][0] + EPS2D) * (cov[1][1] + EPS2D) - cov[0][1] * cov[1][0];
	(*corner).aaFactor = sqrt(max(detOrig / detBlur, MIN_COMPENSATION * MIN_COMPENSATION));

	// 椭圆主轴（对角加 EPS2D）
	let diagonal1: f32 = cov[0][0] + EPS2D;
	let offDiagonal: f32 = cov[0][1];
	let diagonal2: f32 = cov[1][1] + EPS2D;

	let mid: f32 = 0.5 * (diagonal1 + diagonal2);
	let radius: f32 = length(vec2f((diagonal1 - diagonal2) / 2.0, offDiagonal));
	let lambda1: f32 = mid + radius;
	let lambda2: f32 = max(mid - radius, 0.1);

	// 内核大小相对屏幕分辨率设上限（上游同款）
	let vmin: f32 = min(1024.0, min(uniforms.viewport_size.x, uniforms.viewport_size.y));

	// quad **像素**半轴 = min(3.33σ, vmin)。λ 在本文件的 4 倍尺度上，故要乘 0.5 折算。
	let pixelRadius1: f32 = min(0.5 * GAUSS_SIGMA_RADIUS * sqrt(lambda1), vmin);
	let pixelRadius2: f32 = min(0.5 * GAUSS_SIGMA_RADIUS * sqrt(lambda2), vmin);
	// 偏移公式里的 l 用「2·像素」量纲（c = proj.w · (1/W,1/H) 会把它折回像素）
	let l1: f32 = 2.0 * pixelRadius1;
	let l2: f32 = 2.0 * pixelRadius2;

	// 小于 minPixelSize 的剔除（阈值取自上游；l 是「2·像素」量纲，l/2 才是像素半轴）
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
