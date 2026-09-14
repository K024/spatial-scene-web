/**
 * 逐层 PLY 导出：把 splat 场按分层切成 L 个 PLY，拿到 SuperSplat / 任意 splat 编辑器里看。
 *
 * ── 为什么需要它 ──
 * 固定视角的对比图（`layering-render-views.ts`）只能看一个方向；分层的合理性
 * （层带是否切在真实结构上、有没有把该连的面切开、近景层是不是空壳）**必须环绕着看**。
 * PLY 是唯一能让你在编辑器里自由转、随时开关单层的产物。
 *
 * ── 导出什么（**每层独立**）──
 * ```
 * <out>/L16_quantile/
 *   camera.json                       # **一份**参考相机（所有层共用），原图域
 *   manifest.json                     # 每层 {count, 视差带, 深度带, 文件名...}
 *   all.ply                           # 原始整场（对照组：开关层时用它对齐）
 *   tinted.ply                        # 全部层放在一个文件里，**按层染色**（看层带）
 *   layer_00.ply ... layer_15.ply     # 每层一个 PLY（原场的子集，颜色未改）
 *   layer_00_rgba.png ...             # 每层的参考视角 RGBA（sRGB）
 *   layer_00_d.png ...                # 每层的参考视角深度图（近白远黑）
 * ```
 * 每层都是**独立**产物（不是 novel view 的合并图）：PLY 用来环绕看几何，
 * RGBA/D 用来逐层对像素。各层相机参数完全相同，所以 `camera.json` **只写一份**。
 * `tinted.ply` 只作诊断：几何/不透明度不变，只把颜色替换成近红→远蓝的色阶。
 *
 * ── 注意 ──
 * 各层是对原高斯的**划分**（不重不漏），所以 `layer_*.ply` 的并集 == `all.ply`。
 * 这一点由 `layering-golden.ts` 的排列强门保证（拼接后逐位等于全局序）。
 *
 * 用法：
 *   npx tsx scripts/layering-export-layers.ts                          # L=16 / quantile
 *   npx tsx scripts/layering-export-layers.ts --layers 4,16 --method errorDriven
 *   npx tsx scripts/layering-export-layers.ts --ply py-models/out/ply/pier.ply
 *
 * 输出目录名带 fixture 名（`pier_L8_quantile/`），所以多张图不会互相覆盖。
 *   npx tsx scripts/layering-export-layers.ts --no-tinted --out temp/layers-ply
 *   npx tsx scripts/layering-export-layers.ts --width 768 --no-images   # 只要 PLY（不渲染图）
 */

import { mkdirSync, writeFileSync } from "node:fs"
import { resolve } from "node:path"
import { parseArgs } from "node:util"
import {
  fovDeg,
  poseTarget,
  superSplatCameraJson,
  superSplatCameraPose,
} from "../src/spatial-scene/export/camera.ts"
import { gaussiansToPly } from "../src/spatial-scene/export/ply.ts"
import {
  buildLayerPermutation,
  permuteNdcDepths,
} from "../src/spatial-scene/layering/bands.ts"
import {
  computeDisparityStats,
  ndcDepthFromZ,
} from "../src/spatial-scene/layering/disparity-stats.ts"
import {
  bandMassFromSamples,
  computeLayerPlacement,
} from "../src/spatial-scene/layering/placement.ts"
import type { LayerSamplingMethod } from "../src/spatial-scene/layering/types.ts"
import type { Gaussians3D } from "../src/spatial-scene/sharp/types.ts"
import { createWSplatCamera } from "../src/spatial-scene/wsplat/camera.ts"
import { createWSplatRenderer } from "../src/spatial-scene/wsplat/index.ts"
import {
  computeViewDepths,
  sortSplatsBackToFront,
} from "../src/spatial-scene/wsplat/sort.ts"
import type { WSplatFrame } from "../src/spatial-scene/wsplat/types.ts"
import { REPO_ROOT } from "./utils/common.ts"
import { rgbaToPngBuffer } from "./utils/image.ts"
import { withNodeDevice } from "./utils/webgpu.ts"
import { loadWSplatScene, readCameraPose } from "./utils/wsplat-scene.ts"

const CLI = {
  layers: { type: "string" },
  method: { type: "string" },
  out: { type: "string" },
  ply: { type: "string" },
  /** `tinted.ply`（全层染色，看层带）；`--no-tinted` 关。 */
  tinted: { type: "boolean", default: true },
  /** 逐层 rgba/d 图；`--no-images` 关（只出 PLY）。 */
  images: { type: "boolean", default: true },
  /** 逐层 rgba/d 图的渲染宽度（PLY 头与 camera.json 用**原图域**，不受影响）。 */
  width: { type: "string" },
  full: { type: "boolean" },
} as const

interface LayerManifest {
  index: number
  count: number
  /** 视差域带 `[lo, hi]`（0 = near，1 = far）。 */
  disparity: [number, number]
  /** 真实度量深度带 `[lo, hi]`（米）—— 按该层高斯的实测视图深度分位。 */
  depth: [number, number]
  /** 该层占原场的质量比例（按 α 加权）—— **精确值**，直接逐样本归属，不经直方图。 */
  mass: number
  file: string
  /** 该层的参考视角 RGBA 图（sRGB）；`null` = 本次未出图。 */
  rgbaImage: string | null
  /** 该层的参考视角深度图（灰阶，近=白 远=黑，透明 = 无覆盖）；`null` = 未出图。 */
  depthImage: string | null
}

/**
 * 把 `viewMatrix`（列主序 world->camera）转成 `camera.ts` 要的行主序 4x4。
 *
 * 不写死单位阵：SHARP 推理路径确实是 `extrinsics = I`（相机在原点朝 +z），
 * 但 PLY 一旦来自带位姿的来源，这里也应仍然对 —— 而「打开 PLY 就能回到拍摄视角」
 * 完全依赖这一点。
 */
function rowMajorExtrinsics(view: ArrayLike<number>): Float64Array {
  const out = new Float64Array(16)
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) out[r * 4 + c] = view[c * 4 + r]
  }
  return out
}

/**
 * 参考相机的 SuperSplat 位姿 json（与 `scripts/sharp-infer.ts` 写出的**同格式**）。
 *
 * **每份导出只写一个**（`camera.json`）：所有层共用同一个参考相机，逐层复制 sidecar
 * 只是重复同一份数据。manifest 里每层指向它即可。
 *（编辑器要「同名自动配对」时，把 `camera.json` 复制/改名成对应 `.ply` 的名字拖一次即可。）
 */
function cameraJsonFor(
  scene: ReturnType<typeof loadWSplatScene>,
  name: string,
): string {
  const pose = superSplatCameraPose({
    name,
    // 用**原图域**的 f_px 与尺寸（对照 `save_ply` 的 intrinsic element）
    focalLengthPx: scene.pose.fx,
    imageShape: [scene.pose.width, scene.pose.height],
    extrinsics: rowMajorExtrinsics(scene.camera.viewMatrix),
  })
  return superSplatCameraJson([pose])
}

/**
 * 闭环校验：把写出的 camera json **读回来重建相机**，与参考相机逐项比。
 *
 * 这是「导出能还原视角」的唯一硬证据 —— 只写不验的话，坐标系 / 手性 / 列主序
 * 任何一个错都只会表现为「编辑器里看着不太对」。
 */
function verifyCameraRoundTrip(
  scene: ReturnType<typeof loadWSplatScene>,
  cameraPath: string,
): boolean {
  const pose = readCameraPose(cameraPath)
  const rebuilt = createWSplatCamera({
    intrinsics: {
      focalLengthPx: pose.fx,
      width: pose.width,
      height: pose.height,
    },
    position: pose.position ?? [0, 0, 0],
    rotation: pose.rotation,
    near: scene.near,
    far: scene.far,
  })
  let maxViewDiff = 0
  for (let i = 0; i < 16; i++) {
    maxViewDiff = Math.max(
      maxViewDiff,
      Math.abs(rebuilt.viewMatrix[i] - scene.camera.viewMatrix[i]),
    )
  }
  const fovDiff = Math.abs(
    fovDeg(pose.fx, pose.width) - fovDeg(scene.pose.fx, scene.pose.width),
  )
  const ok = maxViewDiff < 1e-6 && fovDiff < 1e-4
  const fmt = (v: readonly number[]): string =>
    v.map((n) => n.toFixed(4)).join(", ")
  // `readCameraPose` 把 position/rotation 标成可选且 readonly；
  // `poseTarget` 要的是可变元组，所以显式拷一份（也顺便兜住缺省值）。
  const position: [number, number, number] = [
    pose.position?.[0] ?? 0,
    pose.position?.[1] ?? 0,
    pose.position?.[2] ?? 0,
  ]
  const forward: [number, number, number] = [
    pose.rotation?.[2]?.[0] ?? 0,
    pose.rotation?.[2]?.[1] ?? 0,
    pose.rotation?.[2]?.[2] ?? 1,
  ]
  // 更强的校对：算出来的位姿应该与**加载时读到的源位姿**逐位相同。
  // `rowMajorExtrinsics` 是从 viewMatrix 反推的，而 `scene.pose` 是源文件里的值；
  // 两者相等才能说「sidecar 就是源位姿」，而不是「我的读写在自洽」。
  const source = scene.pose
  const same = (a: readonly number[], b: readonly number[]): boolean =>
    a.length === b.length && a.every((v, i) => v === b[i])
  const matchesSource =
    pose.fx === source.fx &&
    pose.width === source.width &&
    pose.height === source.height &&
    same(position, source.position ?? [0, 0, 0]) &&
    !!source.rotation &&
    source.rotation.every((row, r) => same(row, pose.rotation?.[r] ?? []))
  const target = poseTarget({ position, forward })
  console.log(
    `      [camera] position=[${fmt(position)}]  target=[${fmt(target)}]  ` +
      `fov(横)=${fovDeg(pose.fx, pose.width).toFixed(3)}°  fx=fy=${pose.fx.toFixed(3)} @ ${pose.width}x${pose.height}`,
  )
  console.log(
    `      [camera] 读回重建 vs 参考相机: viewMatrix 最大差 ${maxViewDiff.toExponential(2)}, ` +
      `fov 差 ${fovDiff.toExponential(2)}  -> ${ok ? "✓ 视角可还原" : "✗ 对不上"}`,
  )
  console.log(
    `      [camera] sidecar vs 源位姿（加载时读到的那份）: ${matchesSource ? "✓ 逐位相同" : "✗ 不一致"}`,
  )
  return ok && matchesSource
}

/**
 * 逐层深度图（参考视角）：灰阶，近 = 白、远 = 黑；无覆盖（`visible=0`）= 全透明。
 *
 * 归一化用**该层自己的** `[zLo, zHi]`（与 manifest 里同一对值），这样每层内部的
 * 深度结构都看得清；跨层比较请看 manifest 的深度带。
 */
function depthRgba(frame: WSplatFrame, zLo: number, zHi: number): Uint8Array {
  const pixels = frame.depth.length
  const out = new Uint8Array(pixels * 4)
  const span = zHi > zLo ? zHi - zLo : 1
  for (let i = 0; i < pixels; i++) {
    const z = frame.depth[i]
    if (!frame.visible[i] || !(z > 0)) continue
    const t = Math.min(1, Math.max(0, (z - zLo) / span))
    const v = Math.round((1 - t) * 255)
    out[i * 4] = v
    out[i * 4 + 1] = v
    out[i * 4 + 2] = v
    out[i * 4 + 3] = 255
  }
  return out
}

async function main(): Promise<void> {
  const args = parseArgs({
    args: process.argv.slice(2),
    options: CLI,
    allowPositionals: false,
    allowNegative: true,
    strict: true,
  })
  const layerCounts = (args.values.layers ?? "16")
    .split(",")
    .map((s) => Number(s.trim()))
    .filter((n) => Number.isInteger(n) && n > 0)
  const method = (args.values.method ?? "quantile") as LayerSamplingMethod
  const writeTinted = args.values.tinted
  const writeImages = args.values.images
  const fullMode = args.values.full === true
  const renderWidth = String(args.values.width ?? 1024)
  const outRoot = resolve(REPO_ROOT, args.values.out ?? "temp/layers-ply")

  console.log("=".repeat(88))
  console.log("layering -> 逐层 PLY + RGBA/D（供编辑器 / 看图目测）")
  console.log("=".repeat(88))

  let cameraRoundTripOk = true

  await withNodeDevice(async (device) => {
    // 渲染尺寸只影响逐层 rgba/d 图：PLY 头与 `camera.json` 一律用**原图域**，
    // 所以换 `--width` 不会让编辑器里的视角漂移。
    const scene = loadWSplatScene({
      plyPath: args.values.ply,
      width: renderWidth,
    })
    const { width, height } = scene
    const count = scene.gaussians.opacities.length
    const depth = computeViewDepths(
      scene.gaussians.meanVectors,
      scene.camera.viewMatrix,
      count,
    )
    const order = sortSplatsBackToFront(depth, count)
    const sortedNdc = permuteNdcDepths(depth, order, scene.near, scene.far)
    // manifest 里报的质量用**精确值**：直方图近似在远景被 n 压缩时曾错过一整箱
    // （见 `bandMasses` 的说明），而 manifest 是人读的数字，不该带这层物差。
    const ndcAll = new Float64Array(count)
    for (let i = 0; i < count; i++) {
      ndcAll[i] = ndcDepthFromZ(depth[i], scene.near, scene.far)
    }
    const stats = computeDisparityStats(
      depth,
      { near: scene.near, far: scene.far, binCount: 256 },
      scene.gaussians.opacities,
    )
    // PLY 头部写的 f_px / 图像尺寸用**原始图像域**（对照 save_ply 的 intrinsic）
    const meta = {
      focalLengthPx: scene.pose.fx,
      resolutionPx: [scene.pose.width, scene.pose.height] as [number, number],
      colorSpace: "linearRGB" as const,
    }

    const renderer = await createWSplatRenderer(device, {
      size: { width, height },
    })
    renderer.setGaussians(scene.gaussians)
    renderer.setCamera(scene.camera)

    console.log(
      `高斯 ${count}  method ${method}  渲染 ${width}x${height}  ` +
        `视差域占用 [${stats.minimum.toFixed(3)}, ${stats.maximum.toFixed(3)}]`,
    )
    // `img_name` 用 PLY 的 stem（与 `sharp-infer` 的命名一致）。
    const plyStem = scene.plyPath.replace(/^.*[\\/]/, "").replace(/\.ply$/i, "")

    for (const L of layerCounts) {
      const placement = computeLayerPlacement(stats, {
        L,
        method,
        near: scene.near,
        far: scene.far,
      })
      const perm = buildLayerPermutation(order, sortedNdc, placement.boundaries)
      const exactMass = bandMassFromSamples(
        ndcAll,
        scene.gaussians.opacities,
        placement.boundaries,
      )
      const dir = resolve(outRoot, `${plyStem}_L${L}_${method}`)
      mkdirSync(dir, { recursive: true })

      // 相机：**一份**，所有层共用（逐层复制只是重复同一份数据）
      const cameraPath = resolve(dir, "camera.json")
      writeFileSync(cameraPath, cameraJsonFor(scene, plyStem))
      if (L === layerCounts[0]) {
        cameraRoundTripOk = verifyCameraRoundTrip(scene, cameraPath)
      }

      // 整场（对照组）
      writeFileSync(
        resolve(dir, "all.ply"),
        gaussiansToPly(scene.gaussians, meta, { full: fullMode }),
      )

      // 逐层渲染（只出图时需要）：层表 -> 每层各自 clear 渲一次
      let frames: WSplatFrame[] | undefined
      if (writeImages) {
        renderer.setSplatOrder(perm.permutation)
        frames = []
        for (let k = 0; k < L; k++) {
          renderer.drawLayer(perm.table[k])
          frames.push(await renderer.readback())
        }
      }

      const manifest: LayerManifest[] = []
      // 染色版：一次分配整场大小，按层填色
      const tinted: Gaussians3D | undefined = writeTinted
        ? {
            meanVectors: scene.gaussians.meanVectors,
            singularValues: scene.gaussians.singularValues,
            quaternions: scene.gaussians.quaternions,
            colors: new Float32Array(count * 3),
            opacities: scene.gaussians.opacities,
          }
        : undefined

      console.log(`\n[L=${L}] ${dir}`)
      console.log(
        "      层   高斯数    质量%   视差带              深度带(m)              文件",
      )
      for (let k = 0; k < L; k++) {
        const { base, count: n } = perm.table[k]
        const subset = extractSubset(scene.gaussians, perm.permutation, base, n)
        const tag = String(k).padStart(2, "0")
        const fileName = `layer_${tag}.ply`
        writeFileSync(
          resolve(dir, fileName),
          gaussiansToPly(subset, meta, { full: fullMode }),
        )

        if (tinted) {
          const tint = layerTint(k, L)
          for (let i = 0; i < n; i++) {
            const target = perm.permutation[base + i] * 3
            tinted.colors[target] = tint[0]
            tinted.colors[target + 1] = tint[1]
            tinted.colors[target + 2] = tint[2]
          }
        }

        // 该层高斯的实测视图深度分位（不是层平面深度 —— 看的是实际内容）
        let zLo = Number.POSITIVE_INFINITY
        let zHi = 0
        for (let i = 0; i < n; i++) {
          const z = depth[perm.permutation[base + i]]
          if (z < zLo) zLo = z
          if (z > zHi) zHi = z
        }

        // 逐层独立出图：RGBA（sRGB，直接用渲染器的 preview）+ 深度图
        let rgbaImage: string | null = null
        let depthImage: string | null = null
        if (frames) {
          const frame = frames[k]
          rgbaImage = `layer_${tag}_rgba.png`
          depthImage = `layer_${tag}_d.png`
          writeFileSync(
            resolve(dir, rgbaImage),
            await rgbaToPngBuffer(frame.preview, width, height),
          )
          writeFileSync(
            resolve(dir, depthImage),
            await rgbaToPngBuffer(
              depthRgba(frame, n > 0 ? zLo : 0, n > 0 ? zHi : 1),
              width,
              height,
            ),
          )
        }

        const entry: LayerManifest = {
          index: k,
          count: n,
          disparity: [placement.boundaries[k], placement.boundaries[k + 1]],
          depth: n > 0 ? [zLo, zHi] : [0, 0],
          mass: exactMass[k],
          file: fileName,
          rgbaImage,
          depthImage,
        }
        manifest.push(entry)
        console.log(
          `      ${String(k).padStart(2)}  ${String(n).padStart(8)}  ` +
            `${(entry.mass * 100).toFixed(1).padStart(5)}  ` +
            `[${entry.disparity[0].toFixed(3)}, ${entry.disparity[1].toFixed(3)}]  ` +
            `[${entry.depth[0].toFixed(2).padStart(6)}, ${entry.depth[1].toFixed(2).padStart(6)}]  ` +
            `${fileName}`,
        )
      }

      if (tinted) {
        writeFileSync(
          resolve(dir, "tinted.ply"),
          gaussiansToPly(tinted, meta, { full: fullMode }),
        )
        console.log(
          `      -> tinted.ply（全部层放一起，近=红 远=蓝，仅用于看层带）`,
        )
      }
      writeFileSync(
        resolve(dir, "manifest.json"),
        JSON.stringify(
          {
            L,
            method,
            near: scene.near,
            far: scene.far,
            cameraFile: "camera.json",
            layerDepths: Array.from(placement.layerDepths),
            boundaries: Array.from(placement.boundaries),
            layers: manifest,
          },
          null,
          2,
        ),
      )
      console.log(`      -> manifest.json（相机：camera.json，所有层共用）`)
    }
    renderer.destroy()
  })

  console.log(
    `\n产物（每层独立）：\n` +
      `  1. layer_XX.ply -> 拖进 SuperSplat 环绕看几何；把 camera.json 拖一次即回到参考视角；\n` +
      `  2. layer_XX_rgba.png / layer_XX_d.png -> 逐层的参考视角颜色 / 深度，对照 manifest.json 的深度带；\n` +
      `  3. all.ply 记住原样；tinted.ply 看层带是否沿真实结构成片（而不是碎点）。`,
  )
  if (!cameraRoundTripOk) {
    console.error(
      "\n[✗] 相机 camera.json 读回重建与参考相机不一致 —— 编辑器里无法还原视角",
    )
    process.exitCode = 1
  }
}

/**
 * 按排列区间取子集。
 *
 * 逐层调用（而不是一次全切）是为了让峰值内存只等于**最大那一层**，
 * 而不是整场大小 —— 1.18M 高斯的 14 个通道约 66 MB，全量翻倍没必要。
 */
function extractSubset(
  gaussians: Gaussians3D,
  permutation: Uint32Array,
  base: number,
  n: number,
): Gaussians3D {
  const meanVectors = new Float32Array(n * 3)
  const singularValues = new Float32Array(n * 3)
  const quaternions = new Float32Array(n * 4)
  const colors = new Float32Array(n * 3)
  const opacities = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    const src = permutation[base + i]
    for (let c = 0; c < 3; c++) {
      meanVectors[i * 3 + c] = gaussians.meanVectors[src * 3 + c]
      singularValues[i * 3 + c] = gaussians.singularValues[src * 3 + c]
      colors[i * 3 + c] = gaussians.colors[src * 3 + c]
    }
    for (let c = 0; c < 4; c++) {
      quaternions[i * 4 + c] = gaussians.quaternions[src * 4 + c]
    }
    opacities[i] = gaussians.opacities[src]
  }
  return { meanVectors, singularValues, quaternions, colors, opacities }
}

/**
 * 层色阶（线性 RGB）：近 = 红，远 = 蓝。
 *
 * 用线性空间下的两段插值（红→绿→蓝）而不是 HSV，是为了让相邻层在中段也容易分辨；
 * 反正它只用于诊断，不追求任何色彩正确性。
 */
function layerTint(k: number, L: number): [number, number, number] {
  const t = L <= 1 ? 0 : k / (L - 1)
  if (t < 0.5) {
    const u = t * 2
    return [1 - u, u, 0.05]
  }
  const u = (t - 0.5) * 2
  return [0.05, 1 - u, u]
}

main().catch((err) => {
  console.error("\n[错误]", err instanceof Error ? err.message : err)
  if (err instanceof Error && err.stack) console.error(err.stack)
  process.exitCode = 1
})
