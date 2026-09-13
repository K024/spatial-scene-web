/**
 * wsplat 场景装配的**唯一**入口：PLY -> `Gaussians3D` -> wsplat 相机。
 *
 * 抽出来的理由：`wsplat-render.ts`（目视）与 `wsplat-golden.ts`（门限）
 * 必须渲染**同一个场景**，否则「图上看着对、门里判不过」无法复核。
 * 之前这些逻辑长在 `wsplat-render.ts` 里，golden 只能复制一份 —— 已收敛到这里。
 *
 * 数值约定（与 ml-sharp / gsplat 对齐；必须与 WGSL 侧一致，改动前先看
 * `src/spatial-scene/wsplat/wgsl/chunks/gsplatCommon.ts` 的数值来源说明）：
 *   - `f_dc_*` 已是 **sRGB**（`save_ply` 时转过），故 `colors` 走
 *     `srgbToLinear(clamp01(0.5 + SH_C0·fdc))` 得到线性 RGB；
 *   - `scale_*` 是 log（奇异值），取 exp；
 *   - `opacity` 是 logit，取 sigmoid；
 *   - `rot_*` w-first，原样传给 shader（`gsplatQuatToMat3` 也是 w-first）。
 */

import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import type { Gaussians3D } from "../../src/spatial-scene/sharp/types.ts"
import {
  createWSplatCamera,
  type WSplatCamera,
} from "../../src/spatial-scene/wsplat/camera.ts"
import { REPO_ROOT } from "./common.ts"
import { readPlyGaussiansFull } from "./ply.ts"

/** SH degree-0 基函数值（与 ml-sharp 的 `convert_rgb_to_spherical_harmonics` 互逆）。 */
export const SH_C0 = 0.28209479177387814

/** 相机 json（SuperSplat 位姿格式）里我们关心的字段。 */
export interface CameraPose {
  fx: number
  fy: number
  width: number
  height: number
  position?: readonly [number, number, number]
  rotation?: readonly [
    readonly [number, number, number],
    readonly [number, number, number],
    readonly [number, number, number],
  ]
}

export interface WSplatSceneOptions {
  plyPath?: string
  cameraPath?: string
  /**
   * 渲染宽度：
   *   - `"native"`（默认）：相机 json 记录的原图宽度，**不重采样**；
   *   - `"internal"`：SHARP 内部域 1536；
   *   - 数字 / 数字字符串：任意宽度（快速迭代）。
   */
  width?: string | number
  /** 只用前 N 个高斯（调试用）。 */
  maxSplats?: number
  /** near/far 覆盖（默认由点云视图 z 的 1%/99% 分位推）。 */
  near?: number
  far?: number
}

export interface WSplatScene {
  plyPath: string
  cameraPath: string
  gaussians: Gaussians3D
  camera: WSplatCamera
  pose: CameraPose
  /** 渲染尺寸。 */
  width: number
  height: number
  /** 渲染宽度 / 原图宽度。 */
  scale: number
  near: number
  far: number
  /** 视图空间 z 分位数（日志/门限用）。 */
  depthRange: { p01: number; median: number; p99: number }
}

/** 读相机 json（SuperSplat `cameras.json` 格式，取第 1 条）。 */
export function readCameraPose(path: string): CameraPose {
  const poses = JSON.parse(readFileSync(path, "utf8")) as CameraPose[]
  if (!Array.isArray(poses) || poses.length === 0) {
    throw new Error(`${path} 不是非空数组（SuperSplat 相机 json 格式）`)
  }
  const pose = poses[0]
  if (!Number.isFinite(pose.fx) || !Number.isFinite(pose.width)) {
    throw new Error(`${path} 缺少 fx / width 字段`)
  }
  return pose
}

/** PLY -> `Gaussians3D`（颜色转线性、尺度取 exp、opacity 取 sigmoid）。 */
export function plyToGaussians3D(
  ply: ReturnType<typeof readPlyGaussiansFull>,
  maxSplats: number = Number.POSITIVE_INFINITY,
): Gaussians3D {
  const count =
    maxSplats === Number.POSITIVE_INFINITY
      ? ply.count
      : Math.min(ply.count, maxSplats)
  const meanVectors = new Float32Array(count * 3)
  const singularValues = new Float32Array(count * 3)
  const quaternions = new Float32Array(count * 4)
  const colors = new Float32Array(count * 3)
  const opacities = new Float32Array(count)
  for (let i = 0; i < count; i++) {
    for (let c = 0; c < 3; c++) {
      meanVectors[i * 3 + c] =
        c === 0 ? ply.x[i] : c === 1 ? ply.y[i] : ply.z[i]
      singularValues[i * 3 + c] = Math.exp(ply.logScale[i * 3 + c])
      // f_dc -> sRGB（0.5 偏置 + SH_C0）-> linearRGB
      colors[i * 3 + c] = srgbToLinear(
        clamp01(0.5 + SH_C0 * ply.fdc[i * 3 + c]),
      )
    }
    quaternions[i * 4] = ply.rotation[i * 4]
    quaternions[i * 4 + 1] = ply.rotation[i * 4 + 1]
    quaternions[i * 4 + 2] = ply.rotation[i * 4 + 2]
    quaternions[i * 4 + 3] = ply.rotation[i * 4 + 3]
    opacities[i] = 1 / (1 + Math.exp(-ply.opacityLogit[i]))
  }
  return { meanVectors, singularValues, quaternions, colors, opacities }
}

/** 装配 `WSplatScene` 所需的、与数据来源无关的字段。 */
export interface AssembleWSplatSceneOptions {
  gaussians: Gaussians3D
  pose: CameraPose
  /** 渲染宽度（语义同 `WSplatSceneOptions.width`）。 */
  width?: string | number
  near?: number
  far?: number
  plyPath?: string
  cameraPath?: string
}

/**
 * 由**已在内存里**的高斯 + 相机位姿装配 `WSplatScene`。
 *
 * `loadWSplatScene`（读 PLY）与「上传图片 -> 推理」都走这里，
 * 保证分辨率 / near-far / 相机内参的推导只有一份实现。
 */
export function assembleWSplatScene(
  options: AssembleWSplatSceneOptions,
): WSplatScene {
  const { gaussians, pose } = options
  const width = resolveTargetWidth(options.width, pose.width)
  const height = Math.round((width * pose.height) / pose.width)
  const scale = width / pose.width

  const depthRange = viewDepthQuantiles(gaussians)
  const near = options.near ?? Math.max(0.01, depthRange.p01 * 0.5)
  const far = options.far ?? Math.max(near * 4, depthRange.p99 * 2)

  const camera = createWSplatCamera({
    intrinsics: {
      focalLengthPx: pose.fx * scale,
      width,
      height,
    },
    position: pose.position ?? [0, 0, 0],
    rotation: pose.rotation,
    near,
    far,
  })

  return {
    plyPath: options.plyPath ?? "",
    cameraPath: options.cameraPath ?? "",
    gaussians,
    camera,
    pose,
    width,
    height,
    scale,
    near,
    far,
    depthRange,
  }
}

/** 读 PLY + 相机 json 的原始产物（尚未按分辨率装配成场景）。 */
export interface WSplatSource {
  readonly plyPath: string
  readonly cameraPath: string
  readonly gaussians: Gaussians3D
  readonly pose: CameraPose
}

/** 只做 I/O：读高斯与位姿，不建相机 / 不定 near-far（分辨率无关，可缓存）。 */
export function loadWSplatSource(
  options: WSplatSceneOptions = {},
): WSplatSource {
  const plyPath = resolve(
    REPO_ROOT,
    options.plyPath ?? "py-models/out/ply/example.ply",
  )
  const cameraPath = resolve(
    REPO_ROOT,
    options.cameraPath ?? plyPath.replace(/\.ply$/i, ".camera.json"),
  )
  const ply = readPlyGaussiansFull(plyPath)
  const gaussians = plyToGaussians3D(ply, options.maxSplats)
  const pose = readCameraPose(cameraPath)
  return { plyPath, cameraPath, gaussians, pose }
}

/** 装配场景（读 PLY + 相机 json，建相机）。 */
export function loadWSplatScene(options: WSplatSceneOptions = {}): WSplatScene {
  const source = loadWSplatSource(options)
  return assembleWSplatScene({
    ...source,
    width: options.width,
    near: options.near,
    far: options.far,
  })
}

/** 把 `--width` 的值解析成像素宽度。 */
export function resolveTargetWidth(
  value: string | number | undefined,
  nativeWidth: number,
): number {
  if (value === undefined || value === "native") return nativeWidth
  if (value === "internal") return 1536
  const n = typeof value === "number" ? value : Number.parseInt(value, 10)
  if (!Number.isFinite(n) || n <= 0) {
    throw new Error(`--width 需要正整数 / native / internal，收到: ${value}`)
  }
  return n
}

/** 视图空间 z 的分位数（near/far 与日志用）；注意这里用的是 PLY 坐标系下的 z。 */
function viewDepthQuantiles(gaussians: Gaussians3D): {
  p01: number
  median: number
  p99: number
} {
  const n = gaussians.opacities.length
  const zs = new Float32Array(n)
  for (let i = 0; i < n; i++) zs[i] = gaussians.meanVectors[i * 3 + 2]
  zs.sort()
  const at = (q: number): number =>
    zs[Math.min(n - 1, Math.max(0, Math.floor(n * q)))]
  return { p01: at(0.01), median: at(0.5), p99: at(0.99) }
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x
}

function srgbToLinear(x: number): number {
  return x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4
}
