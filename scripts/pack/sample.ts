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
 * 曲线/回归走 `scripts/meshing-lod.ts`。
 *
 * 注：交互式调参的 `npm run webui`（`scripts/webui.ts`）因 `@backroad/backroad`
 *     依赖问题已**暂时移除**；本脚本的 flag 就是当前的调参入口。需要 UI 时
 *     用 `git log --follow -- scripts/webui.ts` 找回。
 *
 * 产物被 `.gitignore` 忽略（体积大、可重建），所以 web 端在缺样例时要能优雅降级。
 *
 * 用法：
 *   npx tsx scripts/pack/sample.ts                                   # 原始分辨率，L=6，LOD 自动
 *   npx tsx scripts/pack/sample.ts --layers 4 --no-draco
 *   npx tsx scripts/pack/sample.ts --method errorDriven               # 层放置方法（默认 quantile）
 *   npx tsx scripts/pack/sample.ts --no-refine                        # 关掉原图回写（对照）
 *   npx tsx scripts/pack/sample.ts --lod-min-cell 8                  # 手动指定最小格子
 *   npx tsx scripts/pack/sample.ts --width 768                       # 降采样（快速调试）
 *   npx tsx scripts/pack/sample.ts --no-lod                          # 逐像素出面（对照）
 *   npx tsx scripts/pack/sample.ts --overlap 0.02 --max-dist 16       # 层范围余量 / 隐藏区外推距离
 *   npx tsx scripts/pack/sample.ts --extrap-rgba                       # hidden 也拉伸边缘 RGBA（缺省只补几何）
 *   npx tsx scripts/pack/sample.ts --no-own-gap                       # 只做 hidden 外推（定位边缘硬点）
 *   npx tsx scripts/pack/sample.ts --no-complete                     # 关闭几何补齐（对照）
 *   npx tsx scripts/pack/sample.ts --tear-cleanup                      # 开启撕裂去噪 + 小面片门（库默认，sample 缺省关）
 *
 * 跑完打印**计时**（会话加载与推理分开）与**统计**（高斯 / 深度 / 分层 / 逐层网格 / GLB）。
 */

import { mkdirSync, writeFileSync } from "node:fs"
import { resolve } from "node:path"
import { parseArgs } from "node:util"
import type { LayerSamplingMethod } from "../../src/spatial-scene/layering/index.ts"
import {
  DEFAULT_IMAGE,
  numFlag,
  prepareOrtEnv,
  REPO_ROOT,
} from "../utils/common.ts"
import { createNodeDevice } from "../utils/webgpu.ts"
import { assembleWSplatScene, type WSplatScene } from "../utils/wsplat-scene.ts"
import {
  type BuildGlbResult,
  buildGlb,
  DEFAULT_LAYER_OVERLAP,
  type InferredScene,
  inferSceneFromImage,
} from "./generate.ts"

/** 层放置方法（与 `layering/types.ts` 的 union 一致）。 */
const METHODS: readonly LayerSamplingMethod[] = [
  "quantile",
  "uniform",
  "uniformNonEmpty",
  "importance",
  "hybrid",
  "frontWeighted",
  "errorDriven",
]

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

/** `parseCli` 的产物，字段与命令行 flag 一一对应（见文件头的用法清单）。 */
interface Args {
  /**
   * 输入图路径（`sharp` 可读的任意格式）。
   *
   * @default DEFAULT_IMAGE（ml-depth-pro 的 example.jpg）
   */
  image: string
  /**
   * 层数（= mesh 数 = 出面 / 显存代价）。
   *
   * @default 6
   */
  layers: number
  /**
   * 层放置方法；候选见 {@link METHODS}。
   *
   * @default "quantile"
   */
  method: LayerSamplingMethod
  /**
   * 渲染宽度；`"native"` = 原图宽度（不重采样）。
   *
   * @default "native"
   */
  width: number | "native"
  /**
   * Draco 压缩几何（`--no-draco` 关）。
   *
   * @default true
   */
  draco: boolean
  /**
   * 输出 GLB 路径（相对路径按仓库根解释）。
   *
   * @default public/models/sample.glb
   */
  out: string
  /**
   * 原图回写（`--no-refine` 关，退回纯分层 / mesh）。
   *
   * @default true
   */
  refine: boolean
  /**
   * 层间重叠（只影响报告的 `layerRanges`，不改层分配）。
   *
   * @default DEFAULT_LAYER_OVERLAP（0.02）
   */
  overlap: number
  /**
   * 几何补齐（own-gap + hidden 外推）；`--no-complete` 关。
   *
   * @default true
   */
  complete: boolean
  /**
   * own-gap 回填（`--no-own-gap` 关，用于定位边缘硬点来源）。
   *
   * @default true
   */
  ownGap: boolean
  /**
   * hidden 外推最大距离（px）。
   *
   * @default 16
   */
  maxDist: number
  /**
   * hidden 是否外推 RGBA（关 = 只补几何、保留原纹理 α）。
   *
   * @default false
   */
  extrapRgba: boolean
  /**
   * LOD 出面开关（`--no-lod` 关 = 逐像素出面）。
   *
   * @default true
   */
  lod: boolean
  /**
   * 最小格子（像素）；`"auto"` = 按分辨率推导（见 `autoMinCell`）。
   *
   * @default "auto"（= RECOMMENDED.minCellPx）
   */
  lodMinCell: number | "auto"
  /**
   * LOD 视差域绝对误差上限。
   *
   * @default 0.005（= RECOMMENDED.maxError）
   */
  lodMaxError: number
  /**
   * LOD 边界贴合（含支撑的 cell 一律出面，多出的一圈靠纹理 α 掩掉）。
   *
   * @default true（= RECOMMENDED.snapBoundary）
   */
  lodSnap: boolean
  /**
   * 撕裂清理（视差去噪 + 小面片门）。
   *
   * 库默认是开（`denoise:true, minPatchPixels:16`），这里刻意只在 sample 关掉，便于对照。
   *
   * @default false
   */
  tearCleanup: boolean
}

/**
 * 命令行声明。
 *
 * 用 kebab-case 作 option 名（`parseArgs` 没有别名，flag 名就是 key），
 * boolean 一律给 `default` + `allowNegative` ⇒ `--no-xxx` 自动可用、`--lod-snap`
 * 这类「默认就已开」的开关也能显式写出来。
 */
const CLI_OPTIONS = {
  image: { type: "string" },
  layers: { type: "string" },
  method: { type: "string" },
  width: { type: "string" },
  draco: { type: "boolean", default: true },
  out: { type: "string" },
  refine: { type: "boolean", default: true },
  overlap: { type: "string" },
  complete: { type: "boolean", default: true },
  "own-gap": { type: "boolean", default: true },
  "max-dist": { type: "string" },
  "extrap-rgba": { type: "boolean", default: false },
  lod: { type: "boolean", default: true },
  "lod-min-cell": { type: "string", default: "auto" },
  "lod-max-error": { type: "string", default: String(RECOMMENDED.maxError) },
  "lod-snap": { type: "boolean", default: true },
  "tear-cleanup": { type: "boolean", default: false },
} as const

/** 解析命令行；未知 flag / 非法枚举值一律抛（`strict`）。 */
function parseCli(argv: readonly string[]): Args {
  const { values } = parseArgs({
    args: [...argv],
    options: CLI_OPTIONS,
    allowPositionals: false,
    allowNegative: true,
    strict: true,
  })

  const method = values.method ?? "quantile"
  if (!METHODS.includes(method as LayerSamplingMethod)) {
    throw new Error(
      `--method 只支持 ${METHODS.join(" / ")}（收到 "${method}"）`,
    )
  }
  const width = values.width ?? "native"

  return {
    image: resolve(values.image ?? DEFAULT_IMAGE),
    layers: numFlag("--layers", values.layers, 6),
    method: method as LayerSamplingMethod,
    width: width === "native" ? "native" : numFlag("--width", width, 0),
    draco: values.draco,
    out: resolve(
      values.out ?? resolve(REPO_ROOT, "public", "models", "sample.glb"),
    ),
    refine: values.refine,
    overlap: numFlag("--overlap", values.overlap, DEFAULT_LAYER_OVERLAP),
    complete: values.complete,
    ownGap: values["own-gap"],
    maxDist: numFlag("--max-dist", values["max-dist"], 16),
    extrapRgba: values["extrap-rgba"],
    lod: values.lod,
    lodMinCell:
      values["lod-min-cell"] === "auto"
        ? RECOMMENDED.minCellPx
        : numFlag("--lod-min-cell", values["lod-min-cell"], 0),
    lodMaxError: numFlag(
      "--lod-max-error",
      values["lod-max-error"],
      RECOMMENDED.maxError,
    ),
    lodSnap: values["lod-snap"],
    tearCleanup: values["tear-cleanup"],
  }
}

// ────────────────────────────── 计时 / 统计日志 ──────────────────────────────

const secs = (ms: number): string => `${(ms / 1000).toFixed(2)}s`

/**
 * 字符串的**显示宽度**：CJK / 全角字符占 2 列，其余占 1 列。
 *
 * `padEnd` 按码元数补齐，中英混排的表会因此错位（终端里 CJK 是双宽），
 * 所以列宽只能按显示宽度算。
 */
function displayWidth(text: string): number {
  let width = 0
  for (const ch of text) {
    const cp = ch.codePointAt(0) ?? 0
    const wide =
      (cp >= 0x1100 && cp <= 0x115f) || // 韩文字母
      (cp >= 0x2e80 && cp <= 0xa4cf) || // CJK 部首 / 假名 / 汉字 / 注音
      (cp >= 0xac00 && cp <= 0xd7a3) || // 韩文音节
      (cp >= 0xf900 && cp <= 0xfaff) || // CJK 兼容汉字
      (cp >= 0xfe30 && cp <= 0xfe6f) || // CJK 兼容形式
      (cp >= 0xff00 && cp <= 0xff60) || // 全角 ASCII
      (cp >= 0xffe0 && cp <= 0xffe6) // 全角符号
    width += wide ? 2 : 1
  }
  return width
}

/** 等宽表格：按显示宽度对齐，中英混排不漏。 */
function printTable(
  indent: string,
  rows: readonly (readonly string[])[],
): void {
  const widths: number[] = []
  for (const row of rows) {
    row.forEach((cell, i) => {
      widths[i] = Math.max(widths[i] ?? 0, displayWidth(cell))
    })
  }
  for (const row of rows) {
    console.log(
      indent +
        row
          .map(
            (c, i) => c + " ".repeat(Math.max(0, widths[i] - displayWidth(c))),
          )
          .join("  ")
          .trimEnd(),
    )
  }
}

/**
 * 计时表。
 *
 * **会话加载与推理分开**：模型读盘 + EP 初始化的耗时与分辨率无关，
 * 混在一起看不出「换分辨率 / 换 EP 到底影响了哪一块」。
 */
function logTiming(
  rows: readonly (readonly [string, number])[],
  totalMs: number,
): void {
  console.log("\n[sample] 计时")
  printTable("  ", [
    ...rows.map(([label, ms]) => [label, secs(ms)]),
    ["──── 合计（含写盘）", secs(totalMs)],
  ])
}

/** opacities 的均值与「实心」（α > 0.8）占比。一次线性扫描，毫秒级。 */
function alphaStats(opacities: Float32Array): {
  mean: number
  solidRatio: number
} {
  let sum = 0
  let solid = 0
  for (let i = 0; i < opacities.length; i++) {
    const a = opacities[i]
    sum += a
    if (a > 0.8) solid++
  }
  const n = opacities.length
  return { mean: n > 0 ? sum / n : 0, solidRatio: n > 0 ? solid / n : 0 }
}

/** 统计块：输入 / 推理能力 / 高斯 / 深度 / 场景 / 分层参数 / 逐层网格 / GLB。 */
function logStats(input: {
  inferred: InferredScene
  scene: WSplatScene
  result: BuildGlbResult
  args: Args
  lodMinCell: number
  out: string
}): void {
  const { inferred, scene, result, args, lodMinCell, out } = input
  const { loaded, capabilities } = inferred
  const mesh = result.meshScene
  const { placement } = result.layered
  const alpha = alphaStats(inferred.gaussians.opacities)
  const depth = scene.depthRange

  console.log("\n[sample] 统计")
  console.log(
    `  输入    ${loaded.image.width}×${loaded.image.height}  f_px=${loaded.fPx.toFixed(2)}  ` +
      `focal35mm=${loaded.focal35mm.toFixed(2)}mm(${loaded.focalFromExif ? "EXIF" : "默认"})`,
  )
  console.log(
    `  推理    EP=${capabilities.activeProvider}  fp16Input=${capabilities.fp16Input}  ` +
      `available=[${capabilities.availableProviders.join(",")}]`,
  )
  console.log(
    `  高斯    ${inferred.gaussianCount.toLocaleString()}  α均值=${alpha.mean.toFixed(3)}  ` +
      `α>0.8=${(alpha.solidRatio * 100).toFixed(1)}%`,
  )
  console.log(
    `  深度    z∈[${mesh.stats.minDepth.toFixed(2)}, ${mesh.stats.maxDepth.toFixed(2)}]m  ` +
      `视图 z p01/中位/p99=${depth.p01.toFixed(2)}/${depth.median.toFixed(2)}/${depth.p99.toFixed(2)}m  ` +
      `视差均值=${mesh.stats.disparityMean.toFixed(3)}`,
  )
  console.log(
    `  场景    ${scene.width}×${scene.height}  scale=${scene.scale.toFixed(3)}  ` +
      `near=${scene.near.toFixed(3)}  far=${scene.far.toFixed(3)}`,
  )
  console.log(
    `  分层    L=${args.layers}  method=${args.method}  refine=${args.refine ? "on" : "off"}  ` +
      `overlap=${args.overlap}  补齐=${args.complete ? `own-gap=${args.ownGap ? "on" : "off"} hidden<=${args.maxDist}px${args.extrapRgba ? "+rgba" : ""}` : "off"}  ` +
      `撕裂清理=${args.tearCleanup ? "on" : "off"}`,
  )
  console.log(
    `  出面    ${args.lod ? `LOD minCell=${lodMinCell}${args.lodMinCell === "auto" ? "(auto)" : ""} err=${args.lodMaxError} snap=${args.lodSnap}` : "逐像素（dense）"}  ` +
      `draco=${args.draco ? "on" : "off"}`,
  )

  // 逐层：位置质量（layering 的 layerMass / 深度带）+ 实际网格（meshing）。
  const rows: string[][] = [
    [
      "k",
      "mass%",
      "z_rep(m)",
      "z_band(m)",
      "disp_band",
      "tex",
      "vert",
      "tri(surface+wall)",
    ],
  ]
  let triTotal = 0
  let vertTotal = 0
  for (let k = 0; k < mesh.layers.length; k++) {
    const m = mesh.layers[k]
    triTotal += m.triangleCount
    vertTotal += m.vertexCount
    rows.push([
      String(m.layerIndex),
      (placement.layerMass[k] * 100).toFixed(1),
      placement.layerDepthsZ[k].toFixed(2),
      `[${placement.boundariesZ[k].toFixed(2)}, ${placement.boundariesZ[k + 1].toFixed(2)}]`,
      `[${m.disparityRange[0].toFixed(3)}, ${m.disparityRange[1].toFixed(3)}]`,
      `${m.texture.width}×${m.texture.height}`,
      m.vertexCount.toLocaleString(),
      `${(m.triangleCount - m.wallTriangleCount).toLocaleString()}+${m.wallTriangleCount.toLocaleString()}`,
    ])
  }
  console.log("\n  逐层")
  printTable("  ", rows)
  const backing = mesh.backingPlane
  console.log(
    `  网格    层合计 tri=${triTotal.toLocaleString()}  vert=${vertTotal.toLocaleString()}  ` +
      (backing
        ? `背衬 tri=${backing.triangleCount.toLocaleString()} (tex ${backing.texture.width}×${backing.texture.height}, z=${backing.depthRange[0].toFixed(2)}m)`
        : "背衬=off"),
  )
  console.log(
    `  产物    ${out}  ${(result.glb.length / 1024 / 1024).toFixed(2)} MB  ` +
      `refined=${result.refined ? "yes" : "no"}`,
  )
}

async function main(): Promise<void> {
  prepareOrtEnv("error")
  const args = parseCli(process.argv.slice(2))
  const t0 = Date.now()

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
      method: args.method,
      overlap: args.overlap,
      refine: args.refine,
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
    const tWrite = Date.now()
    writeFileSync(args.out, result.glb)
    const totalMs = Date.now() - t0

    logTiming(
      [
        ["会话加载 session", inferred.sessionLoadMs],
        ["前向+预处理/反投影", inferred.inferMs - inferred.sessionLoadMs],
        ["推理 infer（含会话加载）", inferred.inferMs],
        ["分层渲染", result.layersMs],
        ["网格化", result.meshMs],
        ["GLB 打包", result.glbMs],
        ["写盘", Date.now() - tWrite],
      ],
      totalMs,
    )
    logStats({ inferred, scene, result, args, lodMinCell, out: args.out })
  } finally {
    device.destroy()
  }
}

main().catch((err) => {
  console.error("[sample] 失败:", err instanceof Error ? err.message : err)
  if (err instanceof Error && err.stack) console.error(err.stack)
  process.exitCode = 1
})
