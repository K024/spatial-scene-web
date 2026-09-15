/**
 * 渲染用的最小矩阵/向量工具（无第三方依赖，零 GC 压力的热路径）。
 *
 * 约定：
 * - `mat4` 一律 **Float32Array(16)、列主序**（WebGL 原生布局，
 *   可直接喂 `uniformMatrix4fv`，不需要转置）。
 * - `vec3` 用普通 `number` 元组；只在每帧个位数的相机计算里使用，
 *   不追求零分配（每帧几十字节垃圾可以忽略）。
 */

/** 三维向量。 */
export type Vec3 = readonly [number, number, number]

/** 新建一个 4x4 矩阵（列主序，单位阵）。 */
export function mat4(): Float32Array {
  const m = new Float32Array(16)
  m[0] = 1
  m[5] = 1
  m[10] = 1
  m[15] = 1
  return m
}

/** `out = a · b`（数学意义的矩阵乘法，与存储布局无关）。 */
export function mat4Multiply(
  out: Float32Array,
  a: Float32Array,
  b: Float32Array,
): Float32Array {
  const a00 = a[0]
  const a01 = a[1]
  const a02 = a[2]
  const a03 = a[3]
  const a10 = a[4]
  const a11 = a[5]
  const a12 = a[6]
  const a13 = a[7]
  const a20 = a[8]
  const a21 = a[9]
  const a22 = a[10]
  const a23 = a[11]
  const a30 = a[12]
  const a31 = a[13]
  const a32 = a[14]
  const a33 = a[15]
  for (let i = 0; i < 4; i++) {
    const b0 = b[i * 4]
    const b1 = b[i * 4 + 1]
    const b2 = b[i * 4 + 2]
    const b3 = b[i * 4 + 3]
    out[i * 4] = a00 * b0 + a10 * b1 + a20 * b2 + a30 * b3
    out[i * 4 + 1] = a01 * b0 + a11 * b1 + a21 * b2 + a31 * b3
    out[i * 4 + 2] = a02 * b0 + a12 * b1 + a22 * b2 + a32 * b3
    out[i * 4 + 3] = a03 * b0 + a13 * b1 + a23 * b2 + a33 * b3
  }
  return out
}

/**
 * 由**像素焦距**构造对称透视投影矩阵（`focal` 已在视口像素域）。
 *
 * 不直接用 fovY：本项目的相机参数来自参考相机的 `f_px`，而视口宽高比
 * 通常与原图不同。这里按「原图等比缩放塞进视口」（contain）换算出的
 * 等效像素焦距建模，保证渲染视角与参考照片一致且不产生非等比拉伸。
 *
 * `proj[0] = 2f/width`、`proj[5] = 2f/height` —— 与着色器里的
 * `ndc = 2f·(x/z)/size` 完全一致（着色器会复用同一个 `f`）。
 */
export function perspectiveFromFocal(
  out: Float32Array,
  focalPx: number,
  width: number,
  height: number,
  near: number,
  far: number,
): Float32Array {
  out.fill(0)
  out[0] = (2 * focalPx) / width
  out[5] = (2 * focalPx) / height
  out[10] = -(far + near) / (far - near)
  out[11] = -1
  out[14] = (-2 * far * near) / (far - near)
  return out
}

/**
 * 视图矩阵（右手系，相机朝 **-z**，`up` 为上方向）。
 *
 * 与 `gl-matrix.lookAt` 同构；`center` 即轨道枢轴点。
 */
export function lookAt(
  out: Float32Array,
  eye: Vec3,
  center: Vec3,
  up: Vec3,
): Float32Array {
  let z0 = eye[0] - center[0]
  let z1 = eye[1] - center[1]
  let z2 = eye[2] - center[2]
  let len = Math.hypot(z0, z1, z2)
  if (len < 1e-12) {
    // 退化：eye 与 center 重合，直接给单位阵（调用方不该走到这里）
    out.set([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    return out
  }
  z0 /= len
  z1 /= len
  z2 /= len

  // x = normalize(cross(up, z))
  let x0 = up[1] * z2 - up[2] * z1
  let x1 = up[2] * z0 - up[0] * z2
  let x2 = up[0] * z1 - up[1] * z0
  len = Math.hypot(x0, x1, x2)
  if (len < 1e-12) {
    // up 与 z 共线：换一个参考轴
    x0 = 1
    x1 = 0
    x2 = 0
  } else {
    x0 /= len
    x1 /= len
    x2 /= len
  }

  // y = cross(z, x)
  const y0 = z1 * x2 - z2 * x1
  const y1 = z2 * x0 - z0 * x2
  const y2 = z0 * x1 - z1 * x0

  out[0] = x0
  out[1] = y0
  out[2] = z0
  out[3] = 0
  out[4] = x1
  out[5] = y1
  out[6] = z1
  out[7] = 0
  out[8] = x2
  out[9] = y2
  out[10] = z2
  out[11] = 0
  out[12] = -(x0 * eye[0] + x1 * eye[1] + x2 * eye[2])
  out[13] = -(y0 * eye[0] + y1 * eye[1] + y2 * eye[2])
  out[14] = -(z0 * eye[0] + z1 * eye[1] + z2 * eye[2])
  out[15] = 1
  return out
}

/** 用 4x4 矩阵变换一个点（含平移，w=1），返回新元组。 */
export function transformPoint(m: Float32Array, p: Vec3): Vec3 {
  const x = p[0]
  const y = p[1]
  const z = p[2]
  const w = m[3] * x + m[7] * y + m[11] * z + m[15]
  const iw = w === 0 ? 1 : 1 / w
  return [
    (m[0] * x + m[4] * y + m[8] * z + m[12]) * iw,
    (m[1] * x + m[5] * y + m[9] * z + m[13]) * iw,
    (m[2] * x + m[6] * y + m[10] * z + m[14]) * iw,
  ]
}

export function subtract(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

export function cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ]
}

export function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

export function length(a: Vec3): number {
  return Math.hypot(a[0], a[1], a[2])
}

export function normalize(a: Vec3): Vec3 {
  const len = Math.hypot(a[0], a[1], a[2])
  if (len < 1e-12) return [0, 0, -1]
  return [a[0] / len, a[1] / len, a[2] / len]
}

/** 限制到 `[lo, hi]`。 */
export function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v
}

/** 角度 -> 弧度。 */
export function deg2rad(deg: number): number {
  return (deg * Math.PI) / 180
}

/** 弧度 -> 角度。 */
export function rad2deg(rad: number): number {
  return (rad * 180) / Math.PI
}

/** `#rrggbb` -> 线性前 `[0,1]³`；解析失败返回黑色。 */
export function hexToRgb(hex: string): [number, number, number] {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim())
  if (!m) return [0, 0, 0]
  const v = Number.parseInt(m[1], 16)
  return [((v >> 16) & 255) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255]
}
