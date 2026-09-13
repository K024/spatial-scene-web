/**
 * 把 three 自带的 **Draco decoder** 拷进 `public/draco/`。
 *
 * 为什么要拷：`DRACOLoader` 默认去 Google CDN 取 decoder，离线 / 内网环境直接失败，
 * 而且 CDN 版本与 `three` 不保证对齐。`scripts/pack/glb.ts` 默认开 Draco 压缩，
 * 所以渲染端必须有本地 decoder。
 *
 * 为什么做成脚本而不是把文件提交进仓库：这三个文件是 `three` 的副本（约 1MB），
 * 提交等于在仓库里放一份会随 `three` 版本漂移的二进制。脚本在 `predev` / `prebuild`
 * 自动跑，产物 `public/draco/` 进 `.gitignore`。
 *
 * 用法：`npm run sync:assets`（`npm run dev` / `npm run build` 会自动先跑）
 */

import { copyFileSync, existsSync, mkdirSync, statSync } from "node:fs"
import { resolve } from "node:path"
import { REPO_ROOT } from "./utils/common.ts"

const SOURCE_DIR = resolve(
  REPO_ROOT,
  "node_modules",
  "three",
  "examples",
  "jsm",
  "libs",
  "draco",
  "gltf",
)

const TARGET_DIR = resolve(REPO_ROOT, "public", "draco")

/** wasm 路径只需要这两个（`draco_decoder.js` 是纯 JS 兜底，现代浏览器用不到）。 */
const FILES = ["draco_wasm_wrapper.js", "draco_decoder.wasm"]

function main(): void {
  if (!existsSync(SOURCE_DIR)) {
    console.warn(
      `[sync:assets] 找不到 ${SOURCE_DIR}；跳过。\n` +
        "              （Draco 压缩的 GLB 将无法载入；`npm i` 后重跑本脚本）",
    )
    return
  }

  mkdirSync(TARGET_DIR, { recursive: true })

  let copied = 0
  for (const file of FILES) {
    const from = resolve(SOURCE_DIR, file)
    const to = resolve(TARGET_DIR, file)
    if (!existsSync(from)) {
      console.warn(`[sync:assets] 源文件缺失：${from}`)
      continue
    }
    if (existsSync(to) && statSync(to).size === statSync(from).size) continue
    copyFileSync(from, to)
    copied += 1
    console.log(`[sync:assets] ${file} -> public/draco/`)
  }

  if (copied === 0) console.log("[sync:assets] 已是最新")
}

main()
