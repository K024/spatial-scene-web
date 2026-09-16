/**
 * node 侧 WebGPU（`webgpu` = Dawn 的 node 绑定）共享工具。
 *
 * 关键点：Dawn 的 node 绑定把常量/枚举放在导出的 `globals` 对象里，而
 * `src/spatial-scene/wsplat/**` 引用的是**全局**的 `GPUBufferUsage` 等符号
 * （因为同一份代码要在浏览器里跑）。所以这里在 import 时就把 globals 铺到 globalThis。
 */

import { create, globals } from "webgpu"

// 幂等：铺一次即可。
Object.assign(globalThis, globals)

/**
 * 必须**长期持有** `GPU` / `GPUAdapter` 的引用。
 *
 * ⚠ 踩过的坑：`webgpu@0.6.x`（Dawn-node）里如果 `GPU` 实例变成垃圾被 GC 掉，
 * 而它创建出来的 device 还活着，之后第一次让出事件循环就 SIGSEGV。
 * 之前写成函数局部变量（`const gpu = create([])`），于是「脚本一 GC 就崩」，
 * 崩溃位置还会随代码布局漂移，看起来像 getCompilationInfo / popErrorScope /
 * mapAsync 崩，实际全都是同一个原因。
 */
let gpuRef: GPU | undefined
let adapterRef: GPUAdapter | undefined

/** 请求一个 node 侧 device（失败即抛，带可读原因）。 */
export async function createNodeDevice(): Promise<GPUDevice> {
  const gpu = gpuRef ?? create([])
  gpuRef = gpu
  const adapter = await gpu.requestAdapter()
  if (!adapter) {
    throw new Error("node WebGPU: requestAdapter 返回 null")
  }
  adapterRef = adapter
  const device = await adapter.requestDevice()
  if (!device) {
    throw new Error("node WebGPU: requestDevice 返回 null")
  }
  // ⚠ 绝对不要在这台 Dawn-node 上给 device 挂事件监听器！
  //
  // `webgpu@0.6.x`（Dawn 的 node 绑定，Windows + Node 24）里
  // `device.addEventListener("uncapturederror", ...)` 会破坏它自己的事件分发：
  // 之后**第一次让出事件循环**（setTimeout / await Dawn promise / mapAsync …）
  // 就 SIGSEGV。崩溃位置会随代码改动漂移（有时看着像 getCompilationInfo 崩、
  // 有时像 popErrorScope 崩、有时像 mapAsync 崩），极难查。
  // `device.onuncapturederror = fn` 在这个 build 里根本没接线（不会回调）。
  //
  // 所以：不挂任何 device 事件处理器；validation 错误由 Dawn 自己打到 stderr，
  // 程序化检查统一用 `withValidationScope`（一次 init 一个 scope）。
  return device
}

/**
 * 用 node device 跑一段逻辑，结束后**必须** `device.destroy()`。
 *
 * 为什么强制：`webgpu@0.6.x`（Dawn）在 Windows + Node 24 上，如果进程退出时
 * device 仍然存活，析构阶段会 SIGSEGV（exit 139）——即使脚本逻辑本身完全成功。
 * 显式 destroy 后退出码正常（已实测）。所以所有 wsplat 脚本一律走这个包装，
 * 而不是各写各的 `createNodeDevice()`。
 */
export async function withNodeDevice<T>(
  fn: (device: GPUDevice) => Promise<T>,
): Promise<T> {
  const device = await createNodeDevice()
  try {
    return await fn(device)
  } finally {
    device.destroy()
  }
}
