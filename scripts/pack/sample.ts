/**
 * 生成一个**示例 GLB**，供 web 渲染端（`npm run dev`）开箱验证。
 *
 * 非交付链路，纯开发便利：
 * ```
 * 内置测试图 -> inferSceneFromImage -> assembleWSplatScene -> buildGlb -> public/models/sample.glb
 * ```
 * 产物被 `.gitignore` 忽略（体积大、可重建），所以 web 端在缺样例时要能优雅降级。
 *
 * 用法：
 *   npx tsx scripts/pack/sample.ts                       # 默认 example.jpg，L=8
 *   npx tsx scripts/pack/sample.ts --layers 4 --width 768 --no-draco
 */

import { mkdirSync, writeFileSync } from "node:fs"
import { resolve } from "node:path"
import { DEFAULT_IMAGE, prepareOrtEnv, REPO_ROOT } from "../utils/common.ts"
import { createNodeDevice } from "../utils/webgpu.ts"
import { assembleWSplatScene } from "../utils/wsplat-scene.ts"
import { buildGlb, inferSceneFromImage } from "./generate.ts"

interface Args {
  image: string
  layers: number
  width: number | "native"
  draco: boolean
  out: string
}

function parseArgs(argv: readonly string[]): Args {
  const args: Args = {
    image: DEFAULT_IMAGE,
    layers: 8,
    width: "native",
    draco: true,
    out: resolve(REPO_ROOT, "public", "models", "sample.glb"),
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

  const device = await createNodeDevice()
  try {
    const result = await buildGlb(device, scene, inferred.loaded.image, {
      layers: args.layers,
      method: "quantile",
      refine: true,
      draco: args.draco,
      onStage: (stage, detail) => {
        console.log(`[sample] ${stage}${detail ? ` · ${detail}` : ""}`)
      },
    })
    mkdirSync(resolve(args.out, ".."), { recursive: true })
    writeFileSync(args.out, result.glb)
    console.log(
      `[sample] ${args.out} · ${(result.glb.length / 1024 / 1024).toFixed(2)} MB · ` +
        `${scene.width}x${scene.height} · L=${args.layers}`,
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
