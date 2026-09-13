/**
 * 生成**内置示例 GLB**，供 web 渲染端（`npm run dev`）开箱加载。
 *
 * 非交付链路，纯开发便利：
 * ```
 * 内置测试图 -> inferSceneFromImage -> assembleWSplatScene -> buildGlb -> public/models/sample.glb
 * ```
 *
 * ── 默认值是**推荐出面配置**，但旋钮都保留 ──
 * 默认即当前推荐值（受限四叉树 LOD，`minCellPx=4 / maxError=0.005 / snapBoundary`），
 * 直接跑就是 web 该加载的那一个；想换参数仍可用下面的 flag 覆盖（也会写进日志）。
 * 交互式调参走 `npm run webui`，曲线/回归走 `scripts/meshing-lod.ts`。
 *
 * 产物被 `.gitignore` 忽略（体积大、可重建），所以 web 端在缺样例时要能优雅降级。
 *
 * 用法：
 *   npx tsx scripts/pack/sample.ts                                   # 原始分辨率，L=6，LOD 自动
 *   npx tsx scripts/pack/sample.ts --layers 4 --no-draco
 *   npx tsx scripts/pack/sample.ts --lod-min-cell 8                  # 手动指定最小格子
 *   npx tsx scripts/pack/sample.ts --width 768                       # 降采样（快速调试）
 *   npx tsx scripts/pack/sample.ts --no-lod                          # 逐像素出面（对照）
 *   npx tsx scripts/pack/sample.ts --overlap 0.02 --max-dist 16       # 层范围余量 / 隐藏区外推距离
 *   npx tsx scripts/pack/sample.ts --extrap-rgba                       # hidden 也拉伸边缘 RGBA（缺省只补几何）
 *   npx tsx scripts/pack/sample.ts --no-own-gap                       # 只做 hidden 外推（定位边缘硬点）
 *   npx tsx scripts/pack/sample.ts --no-complete                     # 关闭几何补齐（对照）
 *   npx tsx scripts/pack/sample.ts --tear-cleanup                      # 开启撕裂去噪 + 小面片门（库默认，sample 缺省关）
 */

import { mkdirSync, writeFileSync } from "node:fs"
import { resolve } from "node:path"
import { DEFAULT_IMAGE, prepareOrtEnv, REPO_ROOT } from "../utils/common.ts"
import { createNodeDevice } from "../utils/webgpu.ts"
import { assembleWSplatScene } from "../utils/wsplat-scene.ts"
import {
  buildGlb,
  DEFAULT_LAYER_OVERLAP,
  inferSceneFromImage,
} from "./generate.ts"

/**
 * 内置样例的**推荐 LOD 出面配置**（默认值；与 `lod.ts` 默认一致）。
 *
 * - `minCellPx = "auto"`：取 `max(4, round(longerSide / 192))`。
 *   `minCellPx` 是**像素绝对量**，面数 ∝ 面积/minCell²；分辨率翻倍时若不放大 minCell，
 *   面数会 ×4。按长边定比例就能让「相对粗细」与面预算跨分辨率保持不变
 *   （768 → 4px，3024 → 16px，视觉上是同一档）。
 * - `maxError=0.005`：视差域绝对误差（≈幅宽 50% 平移下 2–3px 屏幕误差），本身与分辨率无关。
 * - `snapBoundary`：含支撑的 cell 一律出面，多出的一圈靠纹理 α 掩掉。
 */
const RECOMMENDED = {
  minCellPx: "auto",
  maxError: 0.005,
  snapBoundary: true,
} as const

/** `minCellPx="auto"` 的推导：长边 / 192，下限 4（2 的幂，向下取）。 */
function autoMinCell(width: number, height: number): number {
  const target = Math.max(4, Math.round(Math.max(width, height) / 192))
  let p = 1
  while (p * 2 <= target) p *= 2
  return p
}

interface Args {
  image: string
  layers: number
  width: number | "native"
  draco: boolean
  out: string
  /** 层间重叠（只影响报告的 `layerRanges`）；默认 `DEFAULT_LAYER_OVERLAP`。 */
  overlap: number
  /** 几何补齐（own-gap + hidden 外推）；`--no-complete` 关。 */
  complete: boolean
  /** own-gap 回填（`--no-own-gap` 关，用于定位边缘硬点来源）。 */
  ownGap: boolean
  /** hidden 外推最大距离（px）。 */
  maxDist: number
  /** hidden 是否外推 RGBA（默认关 = 只补几何、保留原纹理 α；`--extrap-rgba` 开）。 */
  extrapRgba: boolean
  /** LOD 出面开关（`--no-lod` 关；默认开）。 */
  lod: boolean
  /** 最小格子（像素）；`"auto"` = 按分辨率推导（见 `autoMinCell`）。 */
  lodMinCell: number | "auto"
  lodMaxError: number
  lodSnap: boolean
  /**
   * 撕裂清理（视差去噪 + 小面片门）。**缺省关**。
   * 库默认是开（`denoise:true, minPatchPixels:16`），这里刻意只在 sample 关掉，便于对照。
   */
  tearCleanup: boolean
}

function parseArgs(argv: readonly string[]): Args {
  const args: Args = {
    image: DEFAULT_IMAGE,
    layers: 6,
    width: "native",
    draco: true,
    out: resolve(REPO_ROOT, "public", "models", "sample.glb"),
    overlap: DEFAULT_LAYER_OVERLAP,
    complete: true,
    ownGap: true,
    maxDist: 16,
    extrapRgba: false,
    lod: true,
    lodMinCell: RECOMMENDED.minCellPx,
    lodMaxError: RECOMMENDED.maxError,
    lodSnap: RECOMMENDED.snapBoundary,
    tearCleanup: false,
  }
  for (let i = 0; i < argv.length; i++) {
    const flag = argv[i]
    const next = () => argv[++i]
    if (flag === "--image") args.image = resolve(next())
    else if (flag === "--layers") args.layers = Number(next())
    else if (flag === "--width") {
      const v = next()
      args.width = v === "native" ? "native" : Number(v)
    } else if (flag === "--no-draco") args.draco = false
    else if (flag === "--out") args.out = resolve(next())
    else if (flag === "--overlap") args.overlap = Number(next())
    else if (flag === "--no-complete") args.complete = false
    else if (flag === "--no-own-gap") args.ownGap = false
    else if (flag === "--max-dist") args.maxDist = Number(next())
    else if (flag === "--extrap-rgba") args.extrapRgba = true
    else if (flag === "--no-lod") args.lod = false
    else if (flag === "--lod-min-cell") args.lodMinCell = Number(next())
    else if (flag === "--lod-max-error") args.lodMaxError = Number(next())
    else if (flag === "--lod-snap") args.lodSnap = true
    else if (flag === "--no-lod-snap") args.lodSnap = false
    else if (flag === "--tear-cleanup") args.tearCleanup = true
  }
  return args
}

async function main(): Promise<void> {
  prepareOrtEnv("error")
  const args = parseArgs(process.argv.slice(2))

  const inferred = await inferSceneFromImage({
    imagePath: args.image,
    onStage: (stage, detail) => {
      console.log(`[sample] ${stage}${detail ? ` · ${detail}` : ""}`)
    },
  })
  const scene = assembleWSplatScene({
    gaussians: inferred.gaussians,
    pose: inferred.pose,
    width: args.width,
  })

  // `minCellPx` 是像素绝对量 ⇒ 默认按分辨率推导，保持相对粗细/面预算跨分辨率不变。
  const lodMinCell =
    args.lodMinCell === "auto"
      ? autoMinCell(scene.width, scene.height)
      : args.lodMinCell

  const device = await createNodeDevice()
  try {
    const result = await buildGlb(device, scene, inferred.loaded.image, {
      layers: args.layers,
      method: "quantile",
      overlap: args.overlap,
      refine: true,
      complete: args.complete
        ? {
            ownGap: args.ownGap,
            hidden: { maxDistancePx: args.maxDist, rgba: args.extrapRgba },
          }
        : false,
      draco: args.draco,
      mesh: {
        tears: {
          denoise: args.tearCleanup,
          minPatchPixels: args.tearCleanup ? 16 : 1,
        },
        ...(args.lod
          ? {
              lod: {
                minCellPx: lodMinCell,
                maxError: args.lodMaxError,
                snapBoundary: args.lodSnap,
              },
            }
          : {}),
      },
      onStage: (stage, detail) => {
        console.log(`[sample] ${stage}${detail ? ` · ${detail}` : ""}`)
      },
    })
    mkdirSync(resolve(args.out, ".."), { recursive: true })
    writeFileSync(args.out, result.glb)
    const tris = result.meshScene.layers.reduce(
      (s, m) => s + m.triangleCount,
      0,
    )
    console.log(
      `[sample] ${args.out} · ${(result.glb.length / 1024 / 1024).toFixed(2)} MB · ` +
        `${scene.width}x${scene.height} · L=${args.layers} · tri=${tris} · ` +
        `overlap=${args.overlap} · complete=${args.complete ? `hidden<=${args.maxDist}px${args.extrapRgba ? "" : " depth-only"}${args.ownGap ? "" : " no-own-gap"}` : "off"} · ` +
        `tearCleanup=${args.tearCleanup ? "on" : "off"} · ` +
        (args.lod
          ? `LOD minCell=${lodMinCell}${
              args.lodMinCell === "auto" ? "(auto)" : ""
            } err=${args.lodMaxError} snap=${args.lodSnap}`
          : "dense") +
        ` · draco=${args.draco}`,
    )
  } finally {
    device.destroy()
  }
}

main().catch((err) => {
  console.error("[sample] 失败:", err instanceof Error ? err.message : err)
  if (err instanceof Error && err.stack) console.error(err.stack)
  process.exitCode = 1
})
