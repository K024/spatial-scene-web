/**
 * GLSL ES 3.0 着色器。
 *
 * ── 三个 pass ──
 * 1. `SPLAT_VS/FS`：每高斯一个实例化四边形，屏幕空间各向异性高斯。
 * 2. `POST_VS/FS`：把 RGBA16F 的线性结果做曝光 + 线性->sRGB 编码后贴到画布。
 * 3. `OVERLAY_VS/FS`：把参考照片按参考相机内参投影成「无限远背景板」，
 *    用于和渲染结果做叠加/闪烁比对。
 *
 * ── 数学约定（最容易错的地方）──
 * - PLY/度量空间：x 右、y **下**、z 前，参考相机在原点朝 +z。
 * - 渲染世界：y 上、相机朝 -z（标准 OpenGL）。两者差一个
 *   `diag(1,-1,-1)` 旋转，已折进 `u_view`，着色器里不用再管。
 * - 2D 协方差用 EWA：`Σ2 = J · W · Σ3 · Wᵀ · Jᵀ`，其中 J 是
 *   「像素坐标对**相机空间**（z>0 朝前）」的雅可比，W 是 world->camera 旋转。
 *   本文件的写法与 playcanvas engine（`gsplatCorner.glsl`）一致，
 *   但刻意把 J/W 显式写出来，避免行/列主序的隐式陷阱。
 */

/** 顶点着色器：实例化四边形（4 顶点/高斯，`a_corner` 为 ±1）。 */
export const SPLAT_VS = /* glsl */ `#version 300 es
precision highp float;

// 每顶点（divisor=0）：四边形的四个角
layout(location = 0) in vec2 a_corner;
// 每实例（divisor=1）：打包后的高斯属性
layout(location = 1) in vec3 a_center;
layout(location = 2) in vec3 a_scaleLog;
layout(location = 3) in vec4 a_quat;      // (w, x, y, z)
layout(location = 4) in vec3 a_color;     // 线性 RGB
layout(location = 5) in float a_opacityLogit;

uniform mat4 u_view;
uniform mat4 u_proj;
uniform vec2 u_viewport;      // 像素
uniform vec2 u_invViewport;   // 1 / 像素
uniform float u_focalPx;      // 视口像素域焦距（fx = fy）
uniform float u_scaleMul;     // 全局尺度倍数
uniform float u_opacityMul;   // 全局不透明度倍数
uniform float u_near;         // 近裁剪（相机空间 z）
uniform float u_aaMinVar;     // 亚像素抗锯齿：特征值下限（像素²）；0 = 关闭
uniform float u_minPx;        // 半轴短于该值直接丢弃（像素）
uniform float u_maxPx;        // 半轴上限（像素，防单颗铺满屏）
uniform float u_alphaClip;    // 1/255

out vec3 v_color;
out float v_alpha;      // 已乘 u_opacityMul，未含空间衰减
out vec2 v_centerPx;    // gl_FragCoord 系（左下原点）
out vec3 v_conic;       // 2D 协方差的逆，按 (a, b, c) 展开

/** 四元数 (w,x,y,z) -> 旋转矩阵（列主序，与数学写法一致）。 */
mat3 quatToMat3(vec4 q) {
  float w = q.x, x = q.y, y = q.z, z = q.w;
  return mat3(
    1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y),
    2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x),
    2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)
  );
}

/** 剔除：把四个角都推到裁剪体外的同一点（z > w 必被裁掉）。 */
void cullSplat() {
  gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
  v_alpha = 0.0;
}

void main() {
  vec4 viewPos = u_view * vec4(a_center, 1.0);
  float z = -viewPos.z;                     // 相机空间深度（前方为正）

  float alpha = (1.0 / (1.0 + exp(-a_opacityLogit))) * u_opacityMul;
  v_alpha = min(alpha, 0.99);
  v_color = a_color;

  if (z <= u_near || v_alpha <= u_alphaClip) {
    cullSplat();
    return;
  }

  // --- 3D 协方差 Σ = R·S·Sᵀ·Rᵀ ---
  vec3 s = exp(a_scaleLog) * u_scaleMul;
  vec4 q = a_quat * inversesqrt(max(dot(a_quat, a_quat), 1e-20));
  mat3 R = quatToMat3(q);
  mat3 M = R * mat3(s.x, 0.0, 0.0, 0.0, s.y, 0.0, 0.0, 0.0, s.z);
  mat3 sigma3 = M * transpose(M);

  // --- 屏幕空间 2D 协方差 Σ2 = J · W · Σ3 · Wᵀ · Jᵀ ---
  float invz = 1.0 / z;
  float j1 = u_focalPx * invz;
  // J 的行：d(px)/d(cam) = (f/z, 0, -f·x/z²) / (0, f/z, -f·y/z²)
  mat3 J = mat3(
    j1, 0.0, 0.0,
    0.0, j1, 0.0,
    -j1 * viewPos.x * invz, -j1 * viewPos.y * invz, 0.0
  );
  // 相机空间（z 朝前）相对视图空间（z 朝后）只需要翻转 z 轴
  mat3 W = mat3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0) * mat3(u_view);
  mat3 T = J * W;
  mat3 sigma2 = T * sigma3 * transpose(T);

  float A = sigma2[0][0];
  float B = sigma2[0][1];
  float C = sigma2[1][1];

  // --- 2x2 对称阵特征分解 -> 主轴/副轴 ---
  float mid = 0.5 * (A + C);
  float rad = length(vec2(0.5 * (A - C), B));
  float l1 = mid + rad;
  float l2 = max(mid - rad, 0.0);

  // 主轴方向（对应 l1）
  vec2 axis = vec2(B, l1 - A);
  float axisLen = length(axis);
  axis = axisLen > 1e-8 ? axis / axisLen : vec2(1.0, 0.0);
  vec2 perp = vec2(-axis.y, axis.x);

  // 亚像素剔除：用**钳位前**的原始尺寸判定
  if (3.0 * sqrt(l1) < u_minPx) {
    cullSplat();
    return;
  }

  // --- 亚像素抗锯齿（可选）---
  //
  // 做法是把两个特征值从下方钳到 u_aaMinVar，再**重建** A/B/C。
  // 为什么不用「给协方差加一个常量」（playcanvas 的 +0.3 px²）：那个量是
  // 加在所有高斯上的，对屏幕尺寸只有几像素的高斯来说比它自己的方差还大
  // ——本项目 1179648 颗密集小高斯，一开就整幅发糊。
  // 钳位只作用于「比 u_aaMinVar 更细」的那一个轴，大高斯完全不受影响，
  // 且 u_aaMinVar = 0 时重建与不重建数学上恒等（纯粹的 no-op）。
  if (u_aaMinVar > 0.0) {
    float a1 = max(l1, u_aaMinVar);
    float a2 = max(l2, u_aaMinVar);
    float cx = axis.x * axis.x;
    float cz = axis.y * axis.y;
    float cxy = axis.x * axis.y;
    A = a1 * cx + a2 * cz;
    B = (a1 - a2) * cxy;
    C = a1 * cz + a2 * cx;
    l1 = a1;
    l2 = a2;
  }

  float r1 = min(3.0 * sqrt(l1), u_maxPx);
  float r2 = min(3.0 * sqrt(l2), u_maxPx);

  // 低 alpha 的椭球没必要铺满 3σ：收缩到「alpha 恰好等于阈值」的等值线。
  // 四边形边界处 power = -4.5（3σ），故归一化因子 9 = 3²。
  float shrink = min(1.0, sqrt(max(0.0, -2.0 * log(u_alphaClip / v_alpha)) / 9.0));

  vec2 offsetPx = (a_corner.x * r1 * axis + a_corner.y * r2 * perp) * shrink;

  vec4 clip = u_proj * viewPos;
  vec2 ndc = clip.xy / clip.w;
  v_centerPx = (ndc * 0.5 + 0.5) * u_viewport;

  // 屏幕外整体剔除（用外接圆近似）
  float rr = max(r1, r2);
  if (any(greaterThan(abs(v_centerPx - 0.5 * u_viewport) - vec2(rr), 0.5 * u_viewport))) {
    cullSplat();
    return;
  }

  gl_Position = vec4(ndc + offsetPx * 2.0 * u_invViewport, clip.z / clip.w, 1.0);

  // 2D 协方差的逆（conic），供片元算 exp(-0.5·dᵀΣ⁻¹d)。
  // det 夹一个下限：关掉抗锯齿时退化（一条线）的协方差会让 det=0 而产生 inf/NaN。
  float det = max(A * C - B * B, 1e-12);
  v_conic = vec3(C, -B, A) / det;
}
`

/** 片元着色器：用 conic 做高斯衰减，输出**预乘** alpha。 */
export const SPLAT_FS = /* glsl */ `#version 300 es
precision highp float;

in vec3 v_color;
in float v_alpha;
in vec2 v_centerPx;
in vec3 v_conic;

uniform float u_alphaClip;

out vec4 outColor;

void main() {
  vec2 d = gl_FragCoord.xy - v_centerPx;
  float power = -0.5 * (v_conic.x * d.x * d.x + 2.0 * v_conic.y * d.x * d.y + v_conic.z * d.y * d.y);
  if (power > 0.0) {
    discard;
  }
  float a = v_alpha * exp(power);
  if (a < u_alphaClip) {
    discard;
  }
  // 预乘 alpha：混合用 ONE / ONE_MINUS_SRC_ALPHA
  outColor = vec4(v_color * a, a);
}
`

/** 全屏三角形顶点着色器（只用一个 3 顶点属性缓冲）。 */
export const POST_VS = /* glsl */ `#version 300 es
precision highp float;

layout(location = 0) in vec2 a_pos;
out vec2 v_uv;

void main() {
  v_uv = a_pos * 0.5 + 0.5;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}
`

/** 后处理：曝光 + （可选）线性->sRGB。 */
export const POST_FS = /* glsl */ `#version 300 es
precision highp float;

in vec2 v_uv;
uniform sampler2D u_hdr;
uniform float u_exposure;
uniform int u_encodeSrgb;
out vec4 outColor;

vec3 linearToSrgb(vec3 c) {
  vec3 lo = c * 12.92;
  vec3 hi = 1.055 * pow(max(c, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
  return mix(hi, lo, step(c, vec3(0.0031308)));
}

void main() {
  vec3 c = texture(u_hdr, v_uv).rgb * u_exposure;
  outColor = vec4(u_encodeSrgb == 1 ? linearToSrgb(c) : c, 1.0);
}
`

/**
 * 参考图叠加层顶点着色器。
 *
 * 把参考照片当作「无限远背景板」：每个 texel 对应一条由参考相机内参
 * 决定的方向射线，再用**当前**相机投影。因为方向向量取 `w = 0`，
 * 视图矩阵的平移分量自动失效——这正是背景板应有的行为
 * （相机平移不改变无限远处的方向），所以相机小幅移动时叠加依然对齐。
 *
 * 旋转方向（PLY 系 -> 渲染世界系）与 `u_view` 里折进的那次翻转一致。
 */
export const OVERLAY_VS = /* glsl */ `#version 300 es
precision highp float;

layout(location = 0) in vec2 a_uv;   // 0..1，左上为 (0,0)
uniform mat4 u_view;
uniform mat4 u_proj;
uniform vec2 u_imageSize;
uniform float u_refFocalPx;
out vec2 v_uv;

void main() {
  vec2 t = a_uv * 2.0 - 1.0;                      // -1..1
  vec3 dirPly = vec3(t * u_imageSize * 0.5 / u_refFocalPx, 1.0);  // y 向下 = PLY 系
  vec3 dirWorld = vec3(dirPly.x, -dirPly.y, -dirPly.z);
  gl_Position = u_proj * (u_view * vec4(dirWorld, 0.0));
  v_uv = a_uv;
}
`

/** 叠加/闪烁片元着色器。 */
export const OVERLAY_FS = /* glsl */ `#version 300 es
precision highp float;

in vec2 v_uv;
uniform sampler2D u_ref;
uniform float u_opacity;
uniform int u_mode;     // 0 关闭 / 1 叠加 / 2 闪烁
uniform float u_phase;  // 0..1，闪烁相位
out vec4 outColor;

void main() {
  if (u_mode == 2 && u_phase < 0.5) {
    // 闪烁的前半周期直接不画，露出下面的渲染结果
    discard;
  }
  vec4 c = texture(u_ref, v_uv);
  outColor = u_mode == 2 ? vec4(c.rgb, 1.0) : vec4(c.rgb, c.a * u_opacity);
}
`
