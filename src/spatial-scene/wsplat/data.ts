/**
 * 高斯数据 -> GPU buffer 打包。
 *
 * ── 分组（用户决策）──
 * 「RGBA 一组、D 单独一组」：
 *   - `splatRgba`     : `array<vec4f>`，每 splat `(r, g, b, opacity)`，**线性** RGB。
 *   - `splatGeometry` : `array<f32>`，每 splat **10 个 f32**：
 *                       `[mx, my, mz, sx, sy, sz, qw, qx, qy, qz]`
 *                       位置（决定深度 D）+ 尺度 + 旋转（w 在前）。
 * 这样分组的原因：将来分层/refine 只动其中一组（按层 RGB 或按层深度）时不用整体重排；
 * 且 `array<vec4f>` 的 stride 恰好是 16，`array<f32>` 无 padding，都不浪费带宽。
 *
 * ── 精度 ──
 * **一律 f32**。深度累加与投影对精度敏感（深度门：中位相对误差 < 2%），
 * 而「准确度优先、性能次要」是本阶段的明确取舍，所以不做 f16 打包。
 * 1,179,648 个高斯 = RGBA 18.9 MB + 几何 47.2 MB ≈ 66 MB，可接受。
 *
 * 本文件 import WebGPU 类型，但**不 import** node 内置模块（`src/` 内约束）。
 */

import type { Gaussians3D } from "../sharp/types.ts"

/** 每个 splat 的几何通道数：位置 3 + 尺度 3 + 四元数 4。 */
export const GEOMETRY_STRIDE = 10

/** 一次上传后只改 uniform / 排序的 GPU 侧高斯资产。 */
export interface WSplatGpuData {
  readonly count: number
  /** `array<vec4f>`：`(r, g, b, opacity)`，线性 RGB。 */
  readonly rgba: GPUBuffer
  /** `array<f32>`：每 splat `GEOMETRY_STRIDE` 个。 */
  readonly geometry: GPUBuffer
  /** `array<u32>`：**排列**（见 `uploadOrder`）。 */
  readonly order: GPUBuffer
  /** order buffer 能装多少个下标（= `count`）。 */
  readonly orderCapacity: number
  /** 当前排列的长度（由 `uploadOrder` 决定，恒等于 `count`）。 */
  orderLength: number
  /** 均值（世界坐标）的 CPU 副本，`[N,3]`；排序算深度用，不需要回落 GPU。 */
  readonly means: Float32Array
  /**
   * 覆盖排列。
   *
   * ── 这里装的一定是「全量下标的一个排列」──
   * 它是一张**下标表**：`instanceIndex -> splatIndex`。
   * - 全量渲染时它是 back-to-front 的排序索引；
   * - 分层时它是**按层分组**的排列（层间远→近、层内 back-to-front），
   *   因为 `over` 可结合，这样的排列仍然是合法的绘制顺序。
   *
   * 两种情况下它都是 `[0, count)` 的**严格排列**（长度 `== count`、不重不漏）：
   * 分层是**硬划分**（`over` 不幂等，同一个高斯不能进两层），
   * 而 `rangeOverlap` / 密度补偿都**不会**复制高斯，所以长度不会超过 `count`。
   * 这条不变量在 `uploadOrder` 里直接断言 —— 一旦破了，
   * 顶点阶段的 `orderIndex < numSplats` 就不再是安全的边界。
   *
   * 「哪一段属于哪一层」是调用方的事（层表 `{base,count}` 在 layering 侧）。
   */
  uploadOrder(order: Uint32Array, length?: number): void
  destroy(): void
}

export interface CreateWSplatGpuDataOptions {
  /**
   * 输入 `colors` 的颜色空间。
   * ml-sharp 的 `Gaussians3D.colors` 是 **linearRGB**（`gaussians.py` 注释直证），
   * 所以默认 `"linearRGB"`；只有在直接喂 PLY 解码结果时才可能传 `"sRGB"`。
   */
  colorSpace?: "linearRGB" | "sRGB"
  /**
   * order buffer 的容量（下标个数）。默认并**应当**是 `count`。
   *
   * 不要留余量：排列是严格排列（见 `uploadOrder`），容量大于 `count` 只会
   * 掩盖「长度超过高斯数」这个应当直接报错的状态。
   */
  orderCapacity?: number
}

/** 把 `sRGB` 分量转成线性（阈值分段，与 `sharp/colorspace.ts` 同式）。 */
function srgbToLinear(x: number): number {
  const c = x <= 0 ? 0 : x >= 1 ? 1 : x
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
}

/**
 * 上传一组高斯。
 *
 * @throws 各字段长度与数量不匹配时抛错（比渲染出空白更容易定位）。
 */
export function createWSplatGpuData(
  device: GPUDevice,
  gaussians: Gaussians3D,
  options: CreateWSplatGpuDataOptions = {},
): WSplatGpuData {
  const count = gaussians.opacities.length
  if (count === 0) throw new Error("高斯数量为 0")
  const { meanVectors, singularValues, quaternions, colors, opacities } =
    gaussians
  expectLength(meanVectors, count * 3, "meanVectors")
  expectLength(singularValues, count * 3, "singularValues")
  expectLength(quaternions, count * 4, "quaternions")
  expectLength(colors, count * 3, "colors")

  const toLinear = options.colorSpace === "sRGB"

  // ── RGBA 组 ──
  const rgba = new Float32Array(count * 4)
  for (let i = 0; i < count; i++) {
    const r = colors[i * 3]
    const g = colors[i * 3 + 1]
    const b = colors[i * 3 + 2]
    rgba[i * 4] = toLinear ? srgbToLinear(r) : r
    rgba[i * 4 + 1] = toLinear ? srgbToLinear(g) : g
    rgba[i * 4 + 2] = toLinear ? srgbToLinear(b) : b
    rgba[i * 4 + 3] = opacities[i]
  }

  // ── D/几何组 ──
  const geometry = new Float32Array(count * GEOMETRY_STRIDE)
  for (let i = 0; i < count; i++) {
    const o = i * GEOMETRY_STRIDE
    geometry[o] = meanVectors[i * 3]
    geometry[o + 1] = meanVectors[i * 3 + 1]
    geometry[o + 2] = meanVectors[i * 3 + 2]
    geometry[o + 3] = singularValues[i * 3]
    geometry[o + 4] = singularValues[i * 3 + 1]
    geometry[o + 5] = singularValues[i * 3 + 2]
    // 四元数原样上传；归一化由 WGSL 侧的 quatToMat3 前的 normalize 负责，
    // 免得在 CPU 上多跑 118 万次 sqrt（且能对非单位输入保持一致行为）。
    geometry[o + 6] = quaternions[i * 4]
    geometry[o + 7] = quaternions[i * 4 + 1]
    geometry[o + 8] = quaternions[i * 4 + 2]
    geometry[o + 9] = quaternions[i * 4 + 3]
  }

  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  const rgbaBuffer = device.createBuffer({
    label: "wsplat:splatRgba",
    size: rgba.byteLength,
    usage,
  })
  device.queue.writeBuffer(rgbaBuffer, 0, rgba)
  const geometryBuffer = device.createBuffer({
    label: "wsplat:splatGeometry",
    size: geometry.byteLength,
    usage,
  })
  device.queue.writeBuffer(geometryBuffer, 0, geometry)

  const orderCapacity = Math.max(
    count,
    Math.floor(options.orderCapacity ?? count),
  )
  const orderBuffer = device.createBuffer({
    label: "wsplat:splatOrder",
    size: orderCapacity * 4,
    usage,
  })

  const data: WSplatGpuData = {
    count,
    rgba: rgbaBuffer,
    geometry: geometryBuffer,
    order: orderBuffer,
    orderCapacity,
    orderLength: 0,
    means: meanVectors,
    uploadOrder(order: Uint32Array, length?: number): void {
      const n = length ?? order.length
      if (n !== count || order.length < count) {
        throw new Error(
          `排列必须是全部 ${count} 个高斯的严格排列（收到 length ${n}，` +
            `数组长度 ${order.length}）—— 分层是硬划分，` +
            "同一个高斯不允许出现在两层（`over` 不幂等）",
        )
      }
      device.queue.writeBuffer(orderBuffer, 0, order, 0, count)
      data.orderLength = count
    },
    destroy(): void {
      rgbaBuffer.destroy()
      geometryBuffer.destroy()
      orderBuffer.destroy()
    },
  }
  return data
}

function expectLength(
  array: Float32Array,
  expected: number,
  name: string,
): void {
  if (array.length !== expected) {
    throw new Error(`${name} 长度 ${array.length} != 期望 ${expected}`)
  }
}
