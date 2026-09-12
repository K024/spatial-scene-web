/**
 * 3D 高斯的线性代数：四元数 <-> 旋转矩阵、协方差组装/分解。
 *
 * 严格对照 `ml-sharp/src/sharp/utils/linalg.py` 与
 * `utils/gaussians.py: compose_covariance_matrices / decompose_covariance_matrices`。
 *
 * ── 四元数约定（关键 bias）──
 *   w-first：(w, x, y, z)。
 *   证据：`linalg.py: quaternions_from_rotation_matrices` 末尾
 *   `quaternions_np[:, [3, 0, 1, 2]]` 把 scipy 的 (x,y,z,w) 重排为 (w,x,y,z)。
 *   而 `rotation_matrices_from_quaternions` 取 `real_part = q[..., 0]`，
 *   与之自洽。
 *
 * ── 协方差关系（关键 bias）──
 *   `compose_covariance_matrices`: Σ = R · diag(s)² · Rᵀ
 *   注意是 `diagonal_matrix.square()`，即 s 是**奇异值（标准差）不是方差**。
 */

const SQRT4PI = Math.sqrt(4 * Math.PI)

/**
 * 四元数 -> 旋转矩阵。
 *
 * 对照 `linalg.py: rotation_matrices_from_quaternions`：
 *   q 归一化；real = q0，v = q1..3
 *   R = v vᵀ + w² I + 2w [v]× + [v]×²
 * 其中 [v]× 是叉积矩阵 `get_cross_product_matrix`。
 *
 * @returns 行主序 9 元素旋转矩阵；若四元数模长过小则返回单位阵（避免 NaN）。
 */
export function rotationMatrixFromQuaternion(
  w: number,
  x: number,
  y: number,
  z: number,
): Float64Array {
  const n = Math.hypot(w, x, y, z)
  if (!(n > 0) || !Number.isFinite(n)) {
    return new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1])
  }
  const rw = w / n
  const rx = x / n
  const ry = y / n
  const rz = z / n

  // [v]× = [[0,-z,y],[z,0,-x],[-y,x,0]]
  // R = v vᵀ + w² I + 2w[v]× + [v]×²
  const w2 = rw * rw
  const out = new Float64Array(9)

  // v vᵀ
  const vv = [
    rx * rx,
    rx * ry,
    rx * rz,
    ry * rx,
    ry * ry,
    ry * rz,
    rz * rx,
    rz * ry,
    rz * rz,
  ]
  // [v]×
  const cx = [0, -rz, ry, rz, 0, -rx, -ry, rx, 0]

  for (let i = 0; i < 9; i++) out[i] = vv[i]
  // + w² I
  out[0] += w2
  out[4] += w2
  out[8] += w2
  // + 2w [v]×
  for (let i = 0; i < 9; i++) out[i] += 2 * rw * cx[i]
  // + [v]×²
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      let acc = 0
      for (let k = 0; k < 3; k++) acc += cx[r * 3 + k] * cx[k * 3 + c]
      out[r * 3 + c] += acc
    }
  }
  return out
}

/** 3x3 行主序矩阵乘法 C = A·B。 */
function mat3Mul(a: Float64Array, b: Float64Array): Float64Array {
  const c = new Float64Array(9)
  for (let r = 0; r < 3; r++) {
    for (let col = 0; col < 3; col++) {
      let acc = 0
      for (let k = 0; k < 3; k++) acc += a[r * 3 + k] * b[k * 3 + col]
      c[r * 3 + col] = acc
    }
  }
  return c
}

/** 3x3 转置。 */
function mat3Transpose(a: Float64Array): Float64Array {
  return new Float64Array([
    a[0],
    a[3],
    a[6],
    a[1],
    a[4],
    a[7],
    a[2],
    a[5],
    a[8],
  ])
}

/**
 * 组装单个高斯的协方差矩阵：Σ = R · diag(s)² · Rᵀ。
 *
 * 对照 `gaussians.py: compose_covariance_matrices`。
 * 注意 `diag(s)²`：先用奇异值平方得到方差，再左乘右乘旋转。
 *
 * @param out 可选输出缓冲（9 元素），避免逐高斯分配。
 */
export function composeCovarianceMatrix(
  r: Float64Array,
  s0: number,
  s1: number,
  s2: number,
  out?: Float64Array,
): Float64Array {
  const dst = out ?? new Float64Array(9)
  // D = R · diag(s²)
  const ds = [s0 * s0, s1 * s1, s2 * s2]
  const rd = new Float64Array(9)
  for (let row = 0; row < 3; row++) {
    rd[row * 3 + 0] = r[row * 3 + 0] * ds[0]
    rd[row * 3 + 1] = r[row * 3 + 1] * ds[1]
    rd[row * 3 + 2] = r[row * 3 + 2] * ds[2]
  }
  // Σ = (R·D) · Rᵀ
  const rt = mat3Transpose(r)
  const m = mat3Mul(rd, rt)
  dst.set(m)
  return dst
}

/**
 * 对称 3x3 矩阵的 Jacobi 特征分解。
 *
 * 用途：`decompose_covariance_matrices` 需要 SVD，但对**对称半正定**矩阵
 * Σ，SVD 等价于特征分解 Σ = V Λ Vᵀ，故 singular values = sqrt(λ)，
 * rotations = V（必要时修正反射以构成真旋转）。
 *
 * 选择 Jacobi 而非解析法：3x3 解析特征分解的数值稳定性在近退化
 * （两个特征值接近）时很差，而这里输入恰是可能的细长高斯。
 * Jacobi 迭代对对称矩阵稳定且实现简单，1.18M 次调用在 JS 下可接受
 * （见 export 脚本的耗时统计）。
 *
 * @param m 输入对称矩阵（行主序 9 元素，只读取上三角）。
 * @returns `{ values: [λ0,λ1,λ2] 降序, vectors: 行主序 9 元素，列为特征向量 }`
 */
export function symmetricEigen3(m: Float64Array): {
  values: [number, number, number]
  vectors: Float64Array
} {
  // 工作矩阵 A（会被就地对角化），V 累积特征向量（初始为单位阵）
  const a = new Float64Array(m)
  const v = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1])

  const MAX_SWEEPS = 32
  for (let sweep = 0; sweep < MAX_SWEEPS; sweep++) {
    // 非对角元素平方和
    const off = a[1] * a[1] + a[2] * a[2] + a[5] * a[5]
    if (off <= 1e-30) break

    // 轮流消去 3 个非对角元：(0,1), (0,2), (1,2)
    const pairs: [number, number][] = [
      [0, 1],
      [0, 2],
      [1, 2],
    ]
    for (const [p, q] of pairs) {
      const apq = a[p * 3 + q]
      if (Math.abs(apq) <= 1e-300) continue
      const app = a[p * 3 + p]
      const aqq = a[q * 3 + q]

      // 经典 Jacobi 旋转量（数值稳定形式）：
      //   θ = (a_qq − a_pp) / (2 a_pq)
      //   t = sign(θ) / (|θ| + sqrt(θ² + 1))   ← 取绝对值较小的那个根
      //   c = 1 / sqrt(t² + 1),  s = t · c
      // 这个形式保证 |t| ≤ 1，因此旋转不会放大非对角元。
      //
      // 不能用 `theta = 0.5*atan2(2 apq, app - aqq)` 配 `A' = Gᵀ A G`：
      // 那组符号约定会让非对角元反而增大（实测 off 从 2.4e-1 涨到 2.2e-1+
      // 且持续不收敛）。此处采用与 `A' = Gᵀ A G`（下列更新式）自洽的 t 符号。
      const theta = (aqq - app) / (2 * apq)
      const t =
        theta >= 0
          ? 1 / (theta + Math.sqrt(theta * theta + 1))
          : -1 / (-theta + Math.sqrt(theta * theta + 1))
      const c = 1 / Math.sqrt(t * t + 1)
      const s = t * c

      // 用 Givens 旋转更新 A：A' = Gᵀ A G。
      //
      // **必须先算出受影响的元素再统一写回**：先做列变换再读行变换的
      // 写法会撞上「写后读」，因为两个变换改的是同一批格子（A[p][q] 等），
      // 导致 A' ≠ GᵀAG。这里把新旧值分开，避免交错。
      //
      // G 在 (p,q) 平面的作用：
      //   列变换 (A·G)：  col_p' = c·col_p − s·col_q ,  col_q' = s·col_p + c·col_q
      //   行变换 (Gᵀ·A)：  row_p' = c·row_p − s·row_q ,  row_q' = s·row_p + c·row_q
      const next = new Float64Array(a)
      for (let k = 0; k < 3; k++) {
        const akp = a[k * 3 + p]
        const akq = a[k * 3 + q]
        next[k * 3 + p] = c * akp - s * akq
        next[k * 3 + q] = s * akp + c * akq
      }
      for (let k = 0; k < 3; k++) {
        const apk = next[p * 3 + k]
        const aqk = next[q * 3 + k]
        next[p * 3 + k] = c * apk - s * aqk
        next[q * 3 + k] = s * apk + c * aqk
      }
      a.set(next)

      // 累积特征向量：V' = V G（右乘，与 A 的列变换同构）。
      // 注意 V 的更新只读旧 V，同样需要临时量。
      const vnext = new Float64Array(v)
      for (let k = 0; k < 3; k++) {
        const vkp = v[k * 3 + p]
        const vkq = v[k * 3 + q]
        vnext[k * 3 + p] = c * vkp - s * vkq
        vnext[k * 3 + q] = s * vkp + c * vkq
      }
      v.set(vnext)
    }
  }

  // 特征值按降序排列（与 torch.linalg.svd 的 singular_values 降序一致）
  const idx = [0, 1, 2].sort((i, j) => a[j * 3 + j] - a[i * 3 + i])
  const values = idx.map((i) => a[i * 3 + i]) as [number, number, number]
  // 重排特征向量列
  const vectors = new Float64Array(9)
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) vectors[r * 3 + c] = v[r * 3 + idx[c]]
  }
  return { values, vectors }
}

/**
 * 由协方差矩阵分解出 (四元数, 奇异值)。
 *
 * 对照 `gaussians.py: decompose_covariance_matrices`：
 *   - 特征分解 Σ = V Λ Vᵀ
 *   - `singular_values = sqrt(Λ)`（注意取 sqrt，与 compose 的平方互逆）
 *   - 若 det(V) < 0（反射），翻转**最后一列**使其成为真旋转
 *     （原实现：`rotations[batch_idx, gaussian_idx, :, -1] *= -1`）
 *   - 再由旋转矩阵转回 w-first 四元数
 *
 * 关于"最后一列"的语义：Jacobi 输出的 `vectors` 列为降序特征向量，
 * 故"最后一列"= 最小特征值对应的特征向量，与 SVD 的 VT 末行语义一致
 * （SVD 里 `rotations` 是 U，此处以 V 对应，翻转末列同样可修正 det 符号）。
 *
 * 退化处理：负特征值（数值噪声导致）clamp 到 0，避免 sqrt(NaN)。
 */
export function decomposeCovarianceMatrix(cov: Float64Array): {
  quaternion: [number, number, number, number]
  singularValues: [number, number, number]
} {
  const { values, vectors } = symmetricEigen3(cov)

  const s0 = Math.sqrt(Math.max(values[0], 0))
  const s1 = Math.sqrt(Math.max(values[1], 0))
  const s2 = Math.sqrt(Math.max(values[2], 0))

  // det(V)；若为负则翻转最后一列
  const det =
    vectors[0] * (vectors[4] * vectors[8] - vectors[5] * vectors[7]) -
    vectors[1] * (vectors[3] * vectors[8] - vectors[5] * vectors[6]) +
    vectors[2] * (vectors[3] * vectors[7] - vectors[4] * vectors[6])

  const r = new Float64Array(vectors)
  if (det < 0) {
    r[2] = -r[2]
    r[5] = -r[5]
    r[8] = -r[8]
  }

  return {
    quaternion: rotationMatrixToQuaternion(r),
    singularValues: [s0, s1, s2],
  }
}

/**
 * 旋转矩阵 -> w-first 四元数 `(w, x, y, z)`。
 *
 * 复刻 `linalg.py: quaternions_from_rotation_matrices` 的语义
 * （scipy `Rotation.as_quat()` 得 (x,y,z,w)，再重排为 (w,x,y,z)）。
 * 这里用标准的 Shepperd 分支法，数值稳定且与 scipy 的结果在同一分支内一致
 * （同一旋转的两种四元数符号等价，这里统一取 w >= 0 的代表）。
 */
export function rotationMatrixToQuaternion(
  m: Float64Array,
): [number, number, number, number] {
  const m00 = m[0],
    m01 = m[1],
    m02 = m[2]
  const m10 = m[3],
    m11 = m[4],
    m12 = m[5]
  const m20 = m[6],
    m21 = m[7],
    m22 = m[8]

  const trace = m00 + m11 + m22
  let w: number, x: number, y: number, z: number

  if (trace > 0) {
    const s = Math.sqrt(trace + 1.0) * 2 // s = 4w
    w = 0.25 * s
    x = (m21 - m12) / s
    y = (m02 - m20) / s
    z = (m10 - m01) / s
  } else if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1.0 + m00 - m11 - m22) * 2 // s = 4x
    w = (m21 - m12) / s
    x = 0.25 * s
    y = (m01 + m10) / s
    z = (m02 + m20) / s
  } else if (m11 > m22) {
    const s = Math.sqrt(1.0 + m11 - m00 - m22) * 2 // s = 4y
    w = (m02 - m20) / s
    x = (m01 + m10) / s
    y = 0.25 * s
    z = (m12 + m21) / s
  } else {
    const s = Math.sqrt(1.0 + m22 - m00 - m11) * 2 // s = 4z
    w = (m10 - m01) / s
    x = (m02 + m20) / s
    y = (m12 + m21) / s
    z = 0.25 * s
  }

  const n = Math.hypot(w, x, y, z)
  if (!(n > 0) || !Number.isFinite(n)) return [1, 0, 0, 0]
  w /= n
  x /= n
  y /= n
  z /= n
  // 统一符号：w >= 0（去重同一旋转的两种表示）
  if (w < 0) {
    w = -w
    x = -x
    y = -y
    z = -z
  }
  return [w, x, y, z]
}

/**
 * RGB -> degree-0 球谐系数。
 *
 * 对照 `gaussians.py: convert_rgb_to_spherical_harmonics`：
 *   `(rgb - 0.5) / sqrt(1/(4π))`
 * `save_ply` 用它把 sRGB 颜色写成 PLY 的 f_dc_* 字段。
 */
export function rgbToSphericalHarmonics(
  rgb: Float32Array,
  out?: Float32Array,
): Float32Array {
  const dst = out ?? new Float32Array(rgb.length)
  const c0 = Math.sqrt(1.0 / (4.0 * Math.PI))
  for (let i = 0; i < rgb.length; i++) dst[i] = (rgb[i] - 0.5) / c0
  return dst
}

/** 反向：degree-0 SH -> RGB。对照 `convert_spherical_harmonics_to_rgb`。保留供读回校验。 */
export function sphericalHarmonicsToRgb(
  sh0: Float32Array,
  out?: Float32Array,
): Float32Array {
  const dst = out ?? new Float32Array(sh0.length)
  const c0 = Math.sqrt(1.0 / (4.0 * Math.PI))
  for (let i = 0; i < sh0.length; i++) dst[i] = sh0[i] * c0 + 0.5
  return dst
}

/** 导出供外部复用的常量（避免各处重复定义）。 */
export { SQRT4PI }
