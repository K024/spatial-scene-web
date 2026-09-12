/**
 * WebGPU 胶水层：shader module、uniform/storage buffer、bind group layout、pipeline、纹理与读回。
 *
 * 设计约束：
 * - **平台无关**：只接受调用方传入的 `GPUDevice`，不碰 `navigator` / node 内置模块，
 *   因此同一份代码在浏览器与 `webgpu@0.6.0`（node/Dawn）两侧都能跑。
 * - **布局不手写**：buffer 的 struct layout / bind group layout **一律**由 webgpu-utils
 *   从 WGSL 源码推导（`makeShaderDataDefinitions` / `makeBindGroupLayoutDescriptors`），
 *   避免 TS 侧再抄一份错位。
 * - 编排（几个 pass、按什么顺序）不在这里，见 `index.ts` / 后续的 `renderer.ts`。
 */

import type {
  PipelineDescriptor,
  ShaderDataDefinitions,
  StructuredView,
} from "webgpu-utils"
import {
  makeBindGroupLayoutDescriptors,
  makeShaderDataDefinitions,
  makeStructuredView,
} from "webgpu-utils"

/**
 * 在**一个** validation error scope 里执行一段建管线的代码，出错就抛（带 label）。
 *
 * ⚠ 重要（踩过的坑）：`webgpu@0.6.0`（Dawn 的 node 绑定）在 Windows + Node 24 上
 * **反复 push/pop error scope** 会随机 SIGSEGV（位置漂移：有时在 getCompilationInfo，
 * 有时在 popErrorScope）。实测「整个 init 只用一个 scope」稳定不崩。因此约定：
 *
 *   - 每个 renderer 生命周期内**只开一个** validation scope；
 *   - 不要嵌套，也不要给每个资源各开一个。
 *
 * 另外**不要**用 `GPUShaderModule.getCompilationInfo()`：它同样会随机崩，而 error scope
 * 返回的 WGSL 解析错误文本（含行列）已经够用。
 */
export async function withValidationScope<T>(
  device: GPUDevice,
  label: string,
  fn: () => T | Promise<T>,
): Promise<T> {
  device.pushErrorScope("validation")
  let value: T
  try {
    value = await fn()
  } catch (err) {
    await device.popErrorScope()
    throw err
  }
  const error = await device.popErrorScope()
  if (error) {
    throw new Error(`${label} 时有 WebGPU 验证错误: ${error.message}`)
  }
  return value
}

/** 创建 shader module（label 便于报错定位）。WGSL 语法错误由外层 error scope 报出。 */
export function createShaderModule(
  device: GPUDevice,
  code: string,
  label: string,
): GPUShaderModule {
  return device.createShaderModule({ label, code })
}

/** 创建 render pipeline（须给显式 layout；错误由外层 error scope 报出）。 */
export function createRenderPipeline(
  device: GPUDevice,
  desc: GPURenderPipelineDescriptor,
): GPURenderPipeline {
  return device.createRenderPipeline(desc)
}

/** shader 里声明的资源定义（由 webgpu-utils 从源码解析）。 */
export type WgslDefs = ShaderDataDefinitions

/** 解析 WGSL 源码里的资源/结构体定义。 */
export function parseDefs(code: string): WgslDefs {
  return makeShaderDataDefinitions(code)
}

/**
 * 一个 uniform buffer 槽位：WGSL 里的 `@group(g) @binding(b) var<uniform> name: S`。
 *
 * `view` 由 webgpu-utils 按 WGSL 的 struct layout 生成，写值时直接
 * `slot.view.set({ ... })` 或 `slot.view.views.field.set(...)`，然后 `slot.upload()`。
 */
export interface UniformSlot {
  readonly name: string
  readonly group: number
  readonly binding: number
  readonly view: StructuredView
  readonly buffer: GPUBuffer
  /** 把 `view` 的内容写进 GPU buffer。 */
  upload(): void
}

/** 建立一个 uniform 槽位（buffer 大小按 struct layout 自动定）。 */
export function createUniformSlot(
  device: GPUDevice,
  defs: WgslDefs,
  varName: string,
): UniformSlot {
  const varDef = defs.uniforms?.[varName]
  if (!varDef) {
    throw new Error(`WGSL 里找不到 uniform 变量 "${varName}"`)
  }
  if (varDef.group !== 0) {
    throw new Error(
      `uniform "${varName}" 必须在 @group(0)（当前 ${varDef.group}）`,
    )
  }
  const view = makeStructuredView(varDef)
  const buffer = device.createBuffer({
    label: `uniform:${varName}`,
    size: Math.max(view.arrayBuffer.byteLength, 16),
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  return {
    name: varName,
    group: varDef.group,
    binding: varDef.binding,
    view,
    buffer,
    upload(): void {
      device.queue.writeBuffer(buffer, 0, view.arrayBuffer)
    },
  }
}

/** 建立一个 storage buffer（只读，内容由调用方 `writeBuffer` 填）。 */
export function createStorageBuffer(
  device: GPUDevice,
  byteLength: number,
  label: string,
): GPUBuffer {
  return device.createBuffer({
    label,
    size: align(byteLength, 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })
}

/** 由 WGSL 定义推导 bind group layout（按 group 号返回，稀疏数组）。
 *
 * `desc` 用来判断哪些 stage 真的引用了这些资源（visibility），因此必须传
 * 与实际 pipeline 一致的 entry point 组合。
 */
export function createBindGroupLayouts(
  device: GPUDevice,
  defs: WgslDefs,
  desc: PipelineDescriptor,
  label: string,
): GPUBindGroupLayout[] {
  const descriptors = makeBindGroupLayoutDescriptors(defs, desc)
  return descriptors.map((d, i) =>
    device.createBindGroupLayout({ ...d, label: `${label}:bgl${i}` }),
  )
}

/** 创建渲染目标纹理（默认带 `RENDER_ATTACHMENT | TEXTURE_BINDING | COPY_SRC`）。 */
export function createRenderTexture(
  device: GPUDevice,
  opts: {
    label: string
    width: number
    height: number
    format: GPUTextureFormat
    usage?: GPUTextureUsageFlags
    sampleCount?: number
  },
): GPUTexture {
  return device.createTexture({
    label: opts.label,
    size: [opts.width, opts.height, 1],
    format: opts.format,
    sampleCount: opts.sampleCount ?? 1,
    usage:
      opts.usage ??
      GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_SRC,
  })
}

/** 向上对齐。 */
export function align(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment
}

/**
 * 把纹理读回 CPU。
 *
 * WebGPU 要求 `bytesPerRow` 是 256 的倍数，所以每行都要 padding；
 * 这里负责 padding 与实际像素数据的剥离，返回**紧排**的 `width*height*bpp` 数组。
 */
/**
 * 读回一个 u32 缓冲区（用于每帧剔除统计这种小数据）。
 *
 * 与 `readTexturePixels` 一样：临时 buffer + `mapAsync`，用完即销毁。
 * 不做 256 字节行对齐（buffer 到 buffer 的拷贝没有这个约束）。
 */
export async function readBufferU32(
  device: GPUDevice,
  source: GPUBuffer,
  count: number,
  opts: { label?: string } = {},
): Promise<Uint32Array> {
  const byteLength = count * 4
  const staging = device.createBuffer({
    label: opts.label ?? "readback-u32",
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  })
  const encoder = device.createCommandEncoder({ label: "readback-u32" })
  encoder.copyBufferToBuffer(source, 0, staging, 0, byteLength)
  device.queue.submit([encoder.finish()])

  await staging.mapAsync(GPUMapMode.READ)
  const view = new Uint32Array(staging.getMappedRange())
  const out = new Uint32Array(view)
  staging.unmap()
  staging.destroy()
  return out
}

export async function readTexturePixels(
  device: GPUDevice,
  texture: GPUTexture,
  opts: { width: number; height: number; bytesPerPixel: number },
): Promise<Uint8Array> {
  const { width, height, bytesPerPixel } = opts
  const bytesPerRow = align(width * bytesPerPixel, 256)
  const buffer = device.createBuffer({
    label: "readback",
    size: bytesPerRow * height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  })
  const encoder = device.createCommandEncoder({ label: "readback" })
  encoder.copyTextureToBuffer(
    { texture },
    { buffer, bytesPerRow, rowsPerImage: height },
    [width, height, 1],
  )
  device.queue.submit([encoder.finish()])

  await buffer.mapAsync(GPUMapMode.READ)
  const src = new Uint8Array(buffer.getMappedRange())
  const dst = new Uint8Array(width * height * bytesPerPixel)
  const rowBytes = width * bytesPerPixel
  for (let y = 0; y < height; y++) {
    dst.set(
      src.subarray(y * bytesPerRow, y * bytesPerRow + rowBytes),
      y * rowBytes,
    )
  }
  buffer.unmap()
  buffer.destroy()
  return dst
}
