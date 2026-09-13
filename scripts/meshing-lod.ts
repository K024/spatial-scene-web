/**
 * LOD relief 网格化（`lod.ts`）的验证 + 调参曲线。
 *
 * ── A 段（合成，纯 CPU）──
 * 造一个倾斜平面 + 一条深度台阶（撕裂），断言：
 * - 无 T-junction（每条内部边恰好被 2 个三角共享）；
 * - 没有三角形跨过撕裂；
 * - 所有顶点都在支撑内；
 * - 面数随 minCell / maxError 单调下降。
 *
 * ── B 段（真实层帧，GPU）──
 * 走一遍 splat -> L 层 RGBAD，对每档参数用 CPU 光栅器在**参考视角**把 LOD 网格画回来，
 * 与该层 splat 帧比 NCC / MAE / α≥aLo 覆盖率，并给三角数。
 *
 * 用法：
 *   npx tsx scripts/meshing-lod.ts --synthetic
 *   npx tsx scripts/meshing-lod.ts --ply py-models/out/ply/example.ply --width 768 --layers 8
 */

import { ndcDepthFromZ } from "../src/spatial-scene/layering/disparity-stats.ts"
import {
  buildMeshScene,
  computeDisparityField,
  computeSupportMask,
  computeTornEdges,
  toMeshingInput,
} from "../src/spatial-scene/meshing/index.ts"
import {
  buildLayerReliefLod,
  type LodOptions,
} from "../src/spatial-scene/meshing/lod.ts"
import type { LayerMesh } from "../src/spatial-scene/meshing/types.ts"
import { rasterizeLayerMesh } from "./utils/meshing-cpu.ts"
import { renderLayerStack } from "./utils/meshing-scene.ts"
import { withNodeDevice } from "./utils/webgpu.ts"
import { maskedMae, maskedNcc, toGray } from "./utils/wsplat-metrics.ts"
import { loadWSplatScene } from "./utils/wsplat-scene.ts"

interface Args {
  synthetic: boolean
  ply: string
  width: number
  layers: number
}

function parseArgs(argv: readonly string[]): Args {
  const args: Args = {
    synthetic: false,
    ply: "py-models/out/ply/example.ply",
    width: 768,
    layers: 8,
  }
  for (let i = 0; i < argv.length; i++) {
    const next = () => argv[++i]
    if (argv[i] === "--synthetic") args.synthetic = true
    else if (argv[i] === "--ply") args.ply = next()
    else if (argv[i] === "--width") args.width = Number(next())
    else if (argv[i] === "--layers") args.layers = Number(next())
  }
  return args
}

/** 包一个 `LayerMesh` 形状的壳给光栅器用（LOD 结果不实现完整契约）。 */
function asLayerMesh(
  lod: { positions: Float32Array; uvs: Float32Array; indices: Uint32Array },
  width: number,
  height: number,
  texture: { rgb: Float32Array; alpha: Float32Array },
): LayerMesh {
  return {
    layerIndex: 0,
    width,
    height,
    vertexCount: lod.positions.length / 3,
    triangleCount: lod.indices.length / 3,
    wallTriangleCount: 0,
    opaqueTriangleCount: 0,
    positions: lod.positions,
    uvs: lod.uvs,
    indices: lod.indices as Uint32Array<ArrayBuffer>,
    texture: { width, height, rgb: texture.rgb, alpha: texture.alpha },
    disparityRange: [0, 1],
    depthRange: [0, 0],
  }
}

// ────────────────────────────── A 段：合成 ──────────────────────────────

interface Synthetic {
  width: number
  height: number
  depth: Float32Array
  support: Uint8Array
  disparity: Float32Array
  near: number
  far: number
}

function makeSynthetic(): Synthetic {
  const width = 65
  const height = 65
  const near = 0.5
  const far = 10
  const depth = new Float32Array(width * height)
  const support = new Uint8Array(width * height)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x
      // 左半：平滑斜面；右半：明显更近的平面（台阶在 x=32）。
      const z = x < 32 ? 6 + 0.01 * x + 0.005 * y : 3 + 0.002 * y
      depth[i] = z
      support[i] = 1
    }
  }
  const disparity = new Float32Array(width * height)
  for (let i = 0; i < disparity.length; i++) {
    disparity[i] = ndcDepthFromZ(depth[i], near, far)
  }
  return { width, height, depth, support, disparity, near, far }
}

function validate(
  result: { positions: Float32Array; uvs: Float32Array; indices: Uint32Array },
  syn: Synthetic,
  torn: { horizontal: Uint8Array; vertical: Uint8Array },
): {
  tJunctions: number
  tearCross: number
  unsupported: number
  backface: number
} {
  const { width, height } = syn
  // 顶点 -> 像素
  const vx = (v: number): number => Math.round(result.uvs[v * 2] * width - 0.5)
  const vy = (v: number): number =>
    Math.round(result.uvs[v * 2 + 1] * height - 0.5)
  let unsupported = 0
  for (let v = 0; v < result.positions.length / 3; v++) {
    if (!syn.support[vy(v) * width + vx(v)]) unsupported++
  }

  let tearCross = 0
  for (let t = 0; t < result.indices.length / 3; t++) {
    const tri = [
      result.indices[t * 3],
      result.indices[t * 3 + 1],
      result.indices[t * 3 + 2],
    ]
    for (let k = 0; k < 3; k++) {
      const a = tri[k]
      const b = tri[(k + 1) % 3]
      const ax = vx(a)
      const ay = vy(a)
      const bx = vx(b)
      const by = vy(b)
      if (ay === by) {
        const y = ay
        for (let x = Math.min(ax, bx); x < Math.max(ax, bx); x++) {
          if (torn.horizontal[y * (width - 1) + x]) tearCross++
        }
      } else if (ax === bx) {
        const x = ax
        for (let y = Math.min(ay, by); y < Math.max(ay, by); y++) {
          if (torn.vertical[y * width + x]) tearCross++
        }
      }
    }
  }

  // T-junction：一条“长”轴对齐边（未被细分）的中间格点是顶点 ⇒ 裂缝。
  const vertexAt = new Set<number>()
  for (let v = 0; v < result.positions.length / 3; v++) {
    vertexAt.add(vy(v) * width + vx(v))
  }
  const edgeCount = new Map<string, number>()
  const key = (p: number, q: number): string =>
    p < q ? `${p}:${q}` : `${q}:${p}`
  for (let t = 0; t < result.indices.length / 3; t++) {
    const tri = [
      result.indices[t * 3],
      result.indices[t * 3 + 1],
      result.indices[t * 3 + 2],
    ]
    for (let k = 0; k < 3; k++) {
      const a = tri[k]
      const b = tri[(k + 1) % 3]
      const kk = key(a, b)
      edgeCount.set(kk, (edgeCount.get(kk) ?? 0) + 1)
    }
  }
  let tJunctions = 0
  for (const [kk, count] of edgeCount) {
    if (count !== 1) continue
    const [a, b] = kk.split(":").map(Number)
    const ax = vx(a)
    const ay = vy(a)
    const bx = vx(b)
    const by = vy(b)
    if (ay === by && Math.abs(ax - bx) > 1) {
      const mid = (ax + bx) >> 1
      if (vertexAt.has(ay * width + mid)) tJunctions++
    } else if (ax === bx && Math.abs(ay - by) > 1) {
      const mid = (ay + by) >> 1
      if (vertexAt.has(mid * width + ax)) tJunctions++
    }
  }
  // 绕序：`(b-a)×(c-a)` 的 z 分量（相机朝 +z）应 < 0（法线朝相机）。
  // `lod.ts` 出面现在是单面（无裙边），背面会被剔除 ⇒ 绕序错了会直接消失。
  let backface = 0
  for (let t = 0; t < result.indices.length / 3; t++) {
    const a = result.indices[t * 3]
    const b = result.indices[t * 3 + 1]
    const c = result.indices[t * 3 + 2]
    const nz =
      (result.positions[b * 3] - result.positions[a * 3]) *
        (result.positions[c * 3 + 1] - result.positions[a * 3 + 1]) -
      (result.positions[b * 3 + 1] - result.positions[a * 3 + 1]) *
        (result.positions[c * 3] - result.positions[a * 3])
    if (nz > 0) backface++
  }
  return { tJunctions, tearCross, unsupported, backface }
}

function runSynthetic(): void {
  const syn = makeSynthetic()
  // 在 x=32 处造一条撕裂（横向边：把左右两半分开）
  const tornH = new Uint8Array(syn.height * (syn.width - 1))
  const tornV = new Uint8Array((syn.height - 1) * syn.width)
  for (let y = 0; y < syn.height; y++) tornH[y * (syn.width - 1) + 31] = 1

  const camera = {
    width: syn.width,
    height: syn.height,
    fovX: 1,
    fovY: 1,
    near: syn.near,
    far: syn.far,
    viewMatrix: new Float32Array([
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
    ]),
    position: [0, 0, 0] as const,
  } as never

  console.log("=== A 段：合成（65x65，斜面 + x=32 撕裂）===")
  let prev = Number.POSITIVE_INFINITY
  for (const minCell of [1, 2, 4, 8]) {
    for (const maxError of [0.001, 0.005]) {
      const opts: LodOptions = { minCellPx: minCell, maxError, maxCellPx: 64 }
      const r = buildLayerReliefLod(
        { width: syn.width, height: syn.height, depth: syn.depth },
        camera,
        syn.support,
        syn.disparity,
        {
          horizontal: tornH,
          vertical: tornV,
          tearEps: 0,
          tornCount: 0,
          despeckledCount: 0,
          patchReconnected: 0,
        },
        opts,
      )
      const val = validate(r, syn, { horizontal: tornH, vertical: tornV })
      const ok =
        val.tJunctions === 0 &&
        val.tearCross === 0 &&
        val.unsupported === 0 &&
        val.backface === 0
      console.log(
        `  minCell=${minCell} maxErr=${maxError}  tri=${String(r.stats.triangleCount).padStart(6)} ` +
          `vert=${String(r.stats.vertexCount).padStart(6)} leaves=${String(r.stats.leaves).padStart(5)} ` +
          `levels=[${r.stats.levelHistogram.join(",")}] ` +
          `${ok ? "✓" : "✗"} T-junc=${val.tJunctions} tearCross=${val.tearCross} unsupported=${val.unsupported} backface=${val.backface}`,
      )
      if (!ok) process.exitCode = 1
      prev = r.stats.triangleCount
    }
  }
  void prev
}

// ────────────────────────────── B 段：真实层帧 ──────────────────────────────

async function runReal(args: Args): Promise<void> {
  await withNodeDevice(async (device) => {
    const scene = loadWSplatScene({
      plyPath: args.ply,
      width: String(args.width),
    })
    const stack = await renderLayerStack(device, scene, {
      layers: args.layers,
      method: "quantile",
    })
    const input = toMeshingInput(stack, scene.camera)
    const { width, height, near, far, placement } = input
    const n = width * height

    // 预处理只做一次（支撑/视差/撕裂与 LOD 参数无关）。
    const pre = input.frames.map((frame, k) => {
      const { support } = computeSupportMask(frame)
      const disparity = computeDisparityField(frame.depth, near, far)
      const bandWidth = placement.boundaries[k + 1] - placement.boundaries[k]
      const torn = computeTornEdges(
        disparity,
        support,
        width,
        height,
        bandWidth,
      )
      return { frame, support, disparity, torn }
    })

    console.log(
      "\n=== B 段：" +
        `${args.ply}  ${width}x${height}  L=${args.layers}  ` +
        `near/far ${near.toFixed(3)}/${far.toFixed(3)}m ===`,
    )
    console.log(
      " 配置              三角数    顶点数   NCC(覆盖内)  MAE      α≥aLo漏覆盖  漏α能量",
    )

    /** 与 golden B 同口径的逐层度量：NCC 只在“帧可见 且 网格覆盖”的像素上算。 */
    const measure = (
      label: string,
      meshOf: (k: number) => {
        mesh: LayerMesh
        texture: { rgb: Float32Array; alpha: Float32Array }
      },
    ): void => {
      let totalTri = 0
      let totalVert = 0
      let worstNcc = 1
      let worstMae = 0
      let worstMissLo = 0
      let worstEnergy = 0
      for (let k = 0; k < args.layers; k++) {
        const { frame } = pre[k]
        const { mesh, texture } = meshOf(k)
        totalTri += mesh.triangleCount
        totalVert += mesh.vertexCount
        const raster = rasterizeLayerMesh(
          mesh,
          { width, height, ...texture },
          scene.camera,
        )
        const mask = new Uint8Array(n)
        let visibleCount = 0
        let coveredVisible = 0
        let aLoCount = 0
        let aLoUncovered = 0
        let energy = 0
        let energyTotal = 0
        for (let i = 0; i < n; i++) {
          const a = frame.alpha[i]
          if (a > 0) energyTotal += a
          if (frame.visible[i]) {
            visibleCount++
            if (raster.covered[i]) {
              mask[i] = 1
              coveredVisible++
            }
          }
          if (a >= 0.05) {
            aLoCount++
            if (!raster.covered[i]) aLoUncovered++
          }
          if (a > 0 && !raster.covered[i]) energy += a
        }
        const gF = toGray(frame.rgb, frame.alpha, n)
        const gR = toGray(raster.rgb, raster.alpha, n)
        const ncc = coveredVisible > 0 ? maskedNcc(gF, gR, mask) : 0
        const mae =
          coveredVisible > 0 ? maskedMae(frame.rgb, raster.rgb, mask) : 1
        worstNcc = Math.min(worstNcc, ncc)
        worstMae = Math.max(worstMae, mae)
        worstMissLo = Math.max(
          worstMissLo,
          aLoCount > 0 ? aLoUncovered / aLoCount : 0,
        )
        worstEnergy = Math.max(
          worstEnergy,
          energyTotal > 0 ? energy / energyTotal : 0,
        )
      }
      console.log(
        ` ${label.padEnd(18)} ${String(totalTri).padStart(8)}  ${String(totalVert).padStart(8)}  ` +
          `${worstNcc.toFixed(5)}      ${worstMae.toFixed(5)}  ` +
          `${(worstMissLo * 100).toFixed(3)}%        ${(worstEnergy * 100).toFixed(3)}%`,
      )
    }

    // 密集基线（现有 meshing，裙边/背衬关闭）。
    const denseScene = buildMeshScene(input, {
      skirt: false,
      backing: { enabled: false },
    })
    measure("dense(基线)", (k) => ({
      mesh: denseScene.layers[k],
      texture: { rgb: pre[k].frame.rgb, alpha: pre[k].frame.alpha },
    }))

    for (const minCell of [1, 2, 4, 8]) {
      for (const maxError of [0.002, 0.02]) {
        for (const snap of [false, true]) {
          measure(
            `mc=${minCell} e=${maxError} snap=${snap ? "y" : "n"}`,
            (k) => {
              const { frame, support, disparity, torn } = pre[k]
              const lod = buildLayerReliefLod(
                { width, height, depth: frame.depth },
                scene.camera,
                support,
                disparity,
                torn,
                {
                  minCellPx: minCell,
                  maxError,
                  maxCellPx: 128,
                  snapBoundary: snap,
                },
              )
              return {
                mesh: asLayerMesh(lod, width, height, {
                  rgb: frame.rgb,
                  alpha: frame.alpha,
                }),
                texture: { rgb: frame.rgb, alpha: frame.alpha },
              }
            },
          )
        }
      }
    }
  })
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2))
  if (args.synthetic) {
    runSynthetic()
    return
  }
  runSynthetic()
  await runReal(args)
}

main().catch((err) => {
  console.error("[lod] 失败:", err instanceof Error ? err.stack : err)
  process.exitCode = 1
})
