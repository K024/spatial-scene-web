/**
 * NDC -> 度量空间的反投影。
 *
 * 严格对照 `ml-sharp/src/sharp/utils/gaussians.py`：
 *   - `get_unprojection_matrix`  构造 4x4 反投影矩阵
 *   - `apply_transform`          作用于高斯（均值 + 协方差）
 *   - `unproject_gaussians`      入口
 * 以及 `predict.py: predict_image` 中调用它们时的参数（extrinsics = I）。
 *
 * ── 为什么这是最关键的一步 ──
 * ONNX 图输出的是 **NDC 空间**的高斯（见 `common.py: Wrapper.forward` 直接返回
 * `self.model(...)`，而 `unproject_gaussians` 在其后）。若缺少本步，
 * 导出的 PLY 全部落在 NDC 立方体内（约 [-1,1]），SuperSplat 里会是一团糊。
 */

import {
  composeCovarianceMatrix,
  decomposeCovarianceMatrix,
  rotationMatrixFromQuaternion,
} from "./linalg.ts"
import type { Gaussians3D } from "./types.ts"

/**
 * 4x4 行主序矩阵求逆（一般情形，Gauss-Jordan）。
 *
 * 对照 torch 的 `torch.linalg.inv`。对本项目的输入（下三角型的
 * `ndc_matrix @ intrinsics @ extrinsics`），也可解析求解，但保留通用实现
 * 以免未来 extrinsics 非单位阵时静默出错。
 *
 * @throws 若矩阵奇异。
 */
export function invert4(m: ArrayLike<number>): Float64Array {
  const n = 4
  // 增广矩阵 [m | I]
  const a: number[][] = []
  for (let r = 0; r < n; r++) {
    const row = new Array<number>(2 * n).fill(0)
    for (let c = 0; c < n; c++) row[c] = m[r * n + c]
    row[n + r] = 1
    a.push(row)
  }
  for (let c = 0; c < n; c++) {
    // 选主元
    let piv = c
    let best = Math.abs(a[c][c])
    for (let r = c + 1; r < n; r++) {
      const v = Math.abs(a[r][c])
      if (v > best) {
        best = v
        piv = r
      }
    }
    if (best < 1e-300) throw new Error("invert4: matrix is singular")
    if (piv !== c) {
      const t = a[c]
      a[c] = a[piv]
      a[piv] = t
    }
    // 归一化主元行
    const d = a[c][c]
    for (let k = 0; k < 2 * n; k++) a[c][k] /= d
    // 消去其他行
    for (let r = 0; r < n; r++) {
      if (r === c) continue
      const f = a[r][c]
      if (f === 0) continue
      for (let k = 0; k < 2 * n; k++) a[r][k] -= f * a[c][k]
    }
  }
  const out = new Float64Array(16)
  for (let r = 0; r < n; r++)
    for (let c = 0; c < n; c++) out[r * n + c] = a[r][n + c]
  return out
}

/** 4x4 行主序矩阵乘法 C = A·B。 */
export function mat4Mul(
  a: ArrayLike<number>,
  b: ArrayLike<number>,
): Float64Array {
  const c = new Float64Array(16)
  for (let r = 0; r < 4; r++) {
    for (let col = 0; col < 4; col++) {
      let acc = 0
      for (let k = 0; k < 4; k++) acc += a[r * 4 + k] * b[k * 4 + col]
      c[r * 4 + col] = acc
    }
  }
  return c
}

/** 4x4 单位阵（行主序）。 */
export function identity4(): Float64Array {
  return new Float64Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
}

/**
 * 计算 NDC -> 欧氏空间的反投影矩阵。
 *
 * 逐字复刻 `gaussians.py: get_unprojection_matrix`：
 * ```
 * ndc_matrix = [[2/W, 0, -1, 0],
 *               [0, 2/H, -1, 0],
 *               [0, 0, 1, 0],
 *               [0, 0, 0, 1]]
 * return inv(ndc_matrix @ intrinsics @ extrinsics)
 * ```
 * 其中 `W,H = image_shape`（即 1536×1536 内部域，**不是**原始图像尺寸——
 * 见 `predict.py` 传入的是 `internal_shape`）。
 *
 * @param extrinsics 4x4 行主序；推理路径恒为单位阵。
 * @param intrinsics 4x4 行主序；应为 `intrinsicsResized(...)` 的结果。
 * @param imageShape `[width, height]`，内部域（1536, 1536）。
 */
export function getUnprojectionMatrix(
  extrinsics: ArrayLike<number>,
  intrinsics: ArrayLike<number>,
  imageShape: [number, number],
): Float64Array {
  const [imageWidth, imageHeight] = imageShape
  const ndc = new Float64Array([
    2.0 / imageWidth,
    0,
    -1.0,
    0,
    0,
    2.0 / imageHeight,
    -1.0,
    0,
    0,
    0,
    1.0,
    0,
    0,
    0,
    0,
    1,
  ])
  const m = mat4Mul(ndc, mat4Mul(intrinsics, extrinsics))
  return invert4(m)
}

/**
 * 对高斯应用仿射变换（含协方差传播）。
 *
 * 对照 `gaussians.py: apply_transform`：
 *   - 均值：`mean @ Lᵀ + t`
 *   - 协方差：`Σ' = L Σ Lᵀ`（`L = transform[:3,:3]`）
 *   - 再由 `Σ'` 重新分解出 quaternion 与 singular values
 *   - **颜色与不透明度不变**
 *
 * 注意原实现走的是「compose Σ -> 线性变换 -> decompose」，而非直接旋转四元数。
 * 这是因为 L 一般不是正交阵（这里是对角缩放），必须传播协方差再重新分解。
 * 本实现照做。
 *
 * @param transform 4x4 行主序；实际只使用前 3 行。
 * @returns 新高斯（新分配缓冲）。原地变体见 `applyTransformInPlace`。
 */
export function applyTransform(
  gaussians: Gaussians3D,
  transform: ArrayLike<number>,
): Gaussians3D {
  const out: Gaussians3D = {
    meanVectors: new Float32Array(gaussians.meanVectors.length),
    singularValues: new Float32Array(gaussians.singularValues.length),
    quaternions: new Float32Array(gaussians.quaternions.length),
    colors: gaussians.colors,
    opacities: gaussians.opacities,
  }
  applyTransformInto(gaussians, transform, out)
  return out
}

/**
 * `applyTransform` 的原地写出版本：结果写入 `dst`（各缓冲长度需与输入一致）。
 *
 * 颜色/不透明度按引用共享（与原实现一致：这两个字段不被变换修改）。
 * 若调用方需要独立副本，请自行复制。
 */
export function applyTransformInto(
  gaussians: Gaussians3D,
  transform: ArrayLike<number>,
  dst: Gaussians3D,
): void {
  const l = new Float64Array(9)
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 3; c++) l[r * 3 + c] = transform[r * 4 + c]
  const t = [transform[3], transform[7], transform[11]]

  const n = gaussians.opacities.length
  const cov = new Float64Array(9)

  for (let i = 0; i < n; i++) {
    // 均值：mean @ Lᵀ + t
    //
    // 对照 `gaussians.py: apply_transform` 的
    //     mean_vectors @ transform_linear.T + transform_offset
    // 展开即 `Σ_k mean_k * L[j][k]`（下面三项分别对应 j=0,1,2 的列索引写法），
    // 亦即 `L @ mean`。
    //
    // 注意：这两种写法只在 L 对称时才相同。本模型的 L 是对角阵（主点项在
    // inv(ndc@K) 中被消掉），所以“数值上看起来一样”，但**公式不能混**。
    const mx = gaussians.meanVectors[i * 3 + 0]
    const my = gaussians.meanVectors[i * 3 + 1]
    const mz = gaussians.meanVectors[i * 3 + 2]
    dst.meanVectors[i * 3 + 0] = mx * l[0] + my * l[3] + mz * l[6] + t[0]
    dst.meanVectors[i * 3 + 1] = mx * l[1] + my * l[4] + mz * l[7] + t[1]
    dst.meanVectors[i * 3 + 2] = mx * l[2] + my * l[5] + mz * l[8] + t[2]

    // Σ = R diag(s)² Rᵀ
    const q = gaussians.quaternions
    const r = rotationMatrixFromQuaternion(
      q[i * 4],
      q[i * 4 + 1],
      q[i * 4 + 2],
      q[i * 4 + 3],
    )
    composeCovarianceMatrix(
      r,
      gaussians.singularValues[i * 3 + 0],
      gaussians.singularValues[i * 3 + 1],
      gaussians.singularValues[i * 3 + 2],
      cov,
    )

    // Σ' = L Σ Lᵀ
    const tmp = new Float64Array(9) // L Σ
    for (let row = 0; row < 3; row++) {
      for (let col = 0; col < 3; col++) {
        let acc = 0
        for (let k = 0; k < 3; k++) acc += l[row * 3 + k] * cov[k * 3 + col]
        tmp[row * 3 + col] = acc
      }
    }
    const covT = new Float64Array(9) // (L Σ) Lᵀ
    for (let row = 0; row < 3; row++) {
      for (let col = 0; col < 3; col++) {
        let acc = 0
        for (let k = 0; k < 3; k++) acc += tmp[row * 3 + k] * l[col * 3 + k] // L[col][k]
        covT[row * 3 + col] = acc
      }
    }

    const { quaternion, singularValues } = decomposeCovarianceMatrix(covT)
    dst.quaternions[i * 4 + 0] = quaternion[0]
    dst.quaternions[i * 4 + 1] = quaternion[1]
    dst.quaternions[i * 4 + 2] = quaternion[2]
    dst.quaternions[i * 4 + 3] = quaternion[3]
    dst.singularValues[i * 3 + 0] = singularValues[0]
    dst.singularValues[i * 3 + 1] = singularValues[1]
    dst.singularValues[i * 3 + 2] = singularValues[2]
  }
}

/**
 * NDC -> 度量空间（入口）。
 *
 * 对照 `gaussians.py: unproject_gaussians`，并沿用 `predict.py: predict_image`
 * 的调用约定：`extrinsics = I`，`intrinsics = intrinsics_resized`，
 * `image_shape = (1536, 1536)`。
 *
 * @param gaussians NDC 空间高斯。
 * @param extrinsics 4x4 行主序；推理路径传单位阵。
 * @param intrinsics 4x4 行主序；由 `intrinsicsResized(...)` 得到。
 * @param imageShape 内部域 `[1536, 1536]`。
 */
export function unprojectGaussians(
  gaussians: Gaussians3D,
  extrinsics: ArrayLike<number>,
  intrinsics: ArrayLike<number>,
  imageShape: [number, number],
): Gaussians3D {
  const m = getUnprojectionMatrix(extrinsics, intrinsics, imageShape)
  return applyTransform(gaussians, m)
}
