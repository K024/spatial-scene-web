/**
 * node 侧场景装配：PLY（+ SuperSplat 相机 json）-> `Gaussians3D` + 参考视角相机。
 *
 * 与 `wsplat-check-rendering.ts` 内联的那份是同一套语义（PLY 的颜色/尺度/不透明度
 * 都是**导出态**：`f_dc` sRGB、`scale` log、`opacity` logit），但这里是**共用**版本：
 * layering / meshing 的脚本都要跑「读 PLY -> 建相机」这一步，抄两份必然漂移。
 *
 * 数值约定（改动前先看 `wsplat/wgsl/chunks/gsplatCommon.ts` 的数值来源说明）：
 *   - `f_dc_*` 已是 sRGB（`save_ply` 时转过）-> `srgbToLinear(0.5 + SH_C0·f_dc)`；
 *   - `scale_*` 是 log（奇异值）-> `exp`；
 *   - `opacity` 是 logit -> `sigmoid`；
 *   - `rot_*` w-first，原样。
 */

import { existsSync, readFileSync } from "node:fs"
import { resolve } from "node:path"

import type { Gaussians3D } from "../../src/spatial-scene/sharp/types.ts"
import {
  createWSplatCamera,
  type WSplatCamera,
} from "../../src/spatial-scene/wsplat/camera.ts"
import { REPO_ROOT } from "./common.ts"
import { parsePly } from "./ply.ts"

/** SH degree-0 基函数值（与 `save_ply` 的 `convert_rgb_to_spherical_harmonics` 互逆）。 */
const SH_C0 = 0.28209479177387814

/** SuperSplat 相机 json 里我们关心的字段。 */
export interface CameraPose {
  readonly fx: number
  readonly fy?: number
  readonly width: number
  readonly height: number
  readonly position?: readonly [number, number, number]
  readonly rotation?: readonly [
    readonly [number, number, number],
    readonly [number, number, number],
    readonly [number, number, number],
  ]
}

export interface LoadSceneOptions {
  /** PLY 路径（默认 `temp/test.ply`，回退 `public/exports/sample.ply`）。 */
  readonly ply?: string
  /** 相机 json 路径（默认与 PLY 同主干的 `.camera.json`）。 */
  readonly camera?: string
  /** 只用前 N 个高斯（调试）。 */
  readonly maxSplats?: number
  /** near 覆盖（默认由视图 z 的 1% 分位推）。 */
  readonly near?: number
  /** far 覆盖（默认由视图 z 的 99% 分位推）。 */
  readonly far?: number
}

export interface LoadedScene {
  readonly plyPath: string
  readonly cameraPath: string
  readonly gaussians: Gaussians3D
  /** **参考视角**相机（原图分辨率 / 原图像素焦距）。 */
  readonly camera: WSplatCamera
  /** 原图像素焦距（参考域）。 */
  readonly focalLengthPx: number
  readonly near: number
  readonly far: number
  /** 视图空间 z 的分位数（日志用）。 */
  readonly depthQuantiles: { p01: number; median: number; p99: number }
}

/**
 * 默认 PLY：优先 `public/exports/sample.ply`（`npm run sample` 的产物），
 * 没有则回退 `temp/test.ply`（快速迭代产物）。
 */
export function defaultPlyPath(): string {
  const sample = resolve(REPO_ROOT, "public", "exports", "sample.ply")
  if (existsSync(sample)) return sample
  return resolve(REPO_ROOT, "temp", "test.ply")
}

/** 读 PLY + 相机 json，装配高斯与参考相机。 */
export function loadWSplatScene(options: LoadSceneOptions = {}): LoadedScene {
  const plyPath = resolve(options.ply ?? defaultPlyPath())
  if (!existsSync(plyPath)) {
    throw new Error(
      `PLY 不存在: ${plyPath}\n` +
        "先用 `npm run sample` 生成，或用 --ply 指定别的 PLY（同目录需有同名 .camera.json）",
    )
  }
  const cameraPath = resolve(
    options.camera ?? plyPath.replace(/\.ply$/i, ".camera.json"),
  )
  if (!existsSync(cameraPath)) {
    throw new Error(`相机 json 不存在: ${cameraPath}`)
  }

  const buf = readFileSync(plyPath)
  const ply = parsePly(
    new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength),
  )
  const vertex = ply.element("vertex")
  if (!vertex) throw new Error(`${plyPath} 缺少 vertex element`)
  const column = (name: string): Float64Array => {
    const c = vertex.columns.get(name)
    if (!c) throw new Error(`${plyPath} 缺少 vertex.${name}`)
    return c
  }

  const maxSplats =
    options.maxSplats && options.maxSplats > 0
      ? Math.floor(options.maxSplats)
      : Number.POSITIVE_INFINITY
  const count = Math.min(vertex.count, maxSplats)

  const x = column("x")
  const y = column("y")
  const z = column("z")
  const opacity = column("opacity")
  const fdc = [column("f_dc_0"), column("f_dc_1"), column("f_dc_2")]
  const scaleCols = [column("scale_0"), column("scale_1"), column("scale_2")]
  const rot = [
    column("rot_0"),
    column("rot_1"),
    column("rot_2"),
    column("rot_3"),
  ]

  const gaussians: Gaussians3D = {
    meanVectors: new Float32Array(count * 3),
    singularValues: new Float32Array(count * 3),
    quaternions: new Float32Array(count * 4),
    colors: new Float32Array(count * 3),
    opacities: new Float32Array(count),
  }
  for (let i = 0; i < count; i++) {
    gaussians.meanVectors[i * 3] = x[i]
    gaussians.meanVectors[i * 3 + 1] = y[i]
    gaussians.meanVectors[i * 3 + 2] = z[i]
    for (let c = 0; c < 3; c++) {
      gaussians.singularValues[i * 3 + c] = Math.exp(scaleCols[c][i])
      gaussians.colors[i * 3 + c] = srgbToLinear(
        clamp01(0.5 + SH_C0 * fdc[c][i]),
      )
    }
    for (let c = 0; c < 4; c++) gaussians.quaternions[i * 4 + c] = rot[c][i]
    gaussians.opacities[i] = 1 / (1 + Math.exp(-opacity[i]))
  }

  // ── 相机 ──
  const poses = JSON.parse(readFileSync(cameraPath, "utf8")) as CameraPose[]
  const pose = Array.isArray(poses)
    ? poses[0]
    : (poses as unknown as CameraPose)
  if (!pose || !Number.isFinite(pose.fx) || !Number.isFinite(pose.width)) {
    throw new Error(
      `${cameraPath} 缺少 fx / width（SuperSplat 相机 json 格式）`,
    )
  }
  const width = pose.width
  const height = pose.height

  // near/far 由点云视图 z 的 1%/99% 分位推（相机朝 +z，与朝向无关）。
  const zs = new Float32Array(count)
  for (let i = 0; i < count; i++) zs[i] = gaussians.meanVectors[i * 3 + 2]
  zs.sort()
  const at = (q: number): number =>
    zs[Math.min(count - 1, Math.max(0, Math.floor(count * q)))]
  const p01 = at(0.01)
  const median = at(0.5)
  const p99 = at(0.99)
  const near = options.near ?? Math.max(0.01, p01 * 0.5)
  const far = options.far ?? Math.max(near * 4, p99 * 2)

  const camera = createWSplatCamera({
    intrinsics: { focalLengthPx: pose.fx, width, height },
    position: pose.position ?? [0, 0, 0],
    rotation: pose.rotation,
    near,
    far,
  })

  return {
    plyPath,
    cameraPath,
    gaussians,
    camera,
    focalLengthPx: pose.fx,
    near,
    far,
    depthQuantiles: { p01, median, p99 },
  }
}

/** 直通线性 RGB + α -> sRGB RGBA8（α=0 置黑），供 PNG 预览。 */
export function linearFrameToRgba8(
  rgb: Float32Array,
  alpha: Float32Array,
  pixels: number,
): Uint8Array {
  const out = new Uint8Array(pixels * 4)
  for (let i = 0; i < pixels; i++) {
    const a = clamp01(alpha[i])
    const o = i * 4
    if (a <= 0) continue
    out[o] = Math.round(linearToSrgb(clamp01(rgb[i * 3])) * 255)
    out[o + 1] = Math.round(linearToSrgb(clamp01(rgb[i * 3 + 1])) * 255)
    out[o + 2] = Math.round(linearToSrgb(clamp01(rgb[i * 3 + 2])) * 255)
    out[o + 3] = Math.round(a * 255)
  }
  return out
}

/** 与 `wsplat/wgsl/resolve.ts` 的 `linearToSrgb` 同式（阈值分段，非纯 gamma）。 */
function linearToSrgb(x: number): number {
  return x <= 0.0031308 ? x * 12.92 : 1.055 * x ** (1 / 2.4) - 0.055
}

function srgbToLinear(x: number): number {
  return x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x
}
