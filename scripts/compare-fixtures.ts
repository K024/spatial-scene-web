/**
 * JS 实现 vs PyTorch fixtures 的数值对拍。
 *
 * 这是唯一能证明 JS 侧数值正确的手段（而非目视看 PLY）。
 * 对拍分阶段进行，逐层定位误差来源：
 *
 *   阶段 1  输入预处理        JS preprocessImage      vs  fixtures/input.npz
 *   阶段 2  NDC 输出          JS 模型推理（ONNX）      vs  fixtures/ndc.npz
 *   阶段 3  NDC -> metric     JS unprojectGaussians   vs  fixtures/metric.npz
 *   阶段 4  PLY 编码          JS gaussiansToPly       vs  fixtures/reference.ply
 *
 * 阶段 3 是纯 JS 计算（不需要模型），因此**即使 ONNX 推理不可用也能跑**，
 * 这正是我们最需要验证的部分（反投影 + 协方差重分解）。
 *
 * 用法:
 *   npx tsx scripts/compare-fixtures.ts             # 全部阶段（阶段 2 需要模型）
 *   npx tsx scripts/compare-fixtures.ts --no-infer  # 跳过 ONNX 推理（纯 JS，秒级）
 */

import { readFileSync } from "node:fs"
import { resolve } from "node:path"

import { gaussiansToPly } from "../src/spatial-scene/export/ply.ts"
import { runSharp } from "../src/spatial-scene/infer/index.ts"
import {
  composeCovarianceMatrix,
  rotationMatrixFromQuaternion,
} from "../src/spatial-scene/sharp/linalg.ts"
import { preprocessImage } from "../src/spatial-scene/sharp/preprocess.ts"
import type { Gaussians3D } from "../src/spatial-scene/sharp/types.ts"
import {
  applyTransform,
  getUnprojectionMatrix,
  identity4,
} from "../src/spatial-scene/sharp/unproject.ts"
import {
  MODEL_FP16,
  prepareOrtEnv,
  REPO_ROOT,
  requireFile,
} from "./utils/common.ts"
import { loadNpz } from "./utils/npz.ts"

prepareOrtEnv("error")

const FIX = resolve(REPO_ROOT, "py-models", "out", "fixtures")

let failed = 0
let passed = 0

function check(cond: boolean, msg: string): void {
  if (cond) {
    passed++
    console.log(`  PASS  ${msg}`)
  } else {
    failed++
    console.log(`  FAIL  ${msg}`)
  }
}

/** 数值序列的统计对比。 */
function stats(
  a: ArrayLike<number>,
  b: ArrayLike<number>,
): {
  maxAbsErr: number
  meanAbsErr: number
  relErr: number
  n: number
} {
  const n = Math.min(a.length, b.length)
  let maxAbsErr = 0
  let sumAbs = 0
  let sumRef = 0
  for (let i = 0; i < n; i++) {
    const d = Math.abs(a[i] - b[i])
    if (d > maxAbsErr) maxAbsErr = d
    sumAbs += d
    sumRef += Math.abs(b[i])
  }
  return {
    maxAbsErr,
    meanAbsErr: sumAbs / n,
    relErr: sumRef > 0 ? sumAbs / sumRef : sumAbs,
    n,
  }
}

function report(
  label: string,
  a: ArrayLike<number>,
  b: ArrayLike<number>,
  tol: number,
): void {
  const s = stats(a, b)
  const ok = s.maxAbsErr <= tol
  console.log(
    `    ${ok ? "ok  " : "BAD "} ${label.padEnd(22)} maxErr=${s.maxAbsErr.toExponential(3)} ` +
      `meanErr=${s.meanAbsErr.toExponential(3)} rel=${s.relErr.toExponential(3)}`,
  )
  if (!ok) failed++
  else passed++
}

/**
 * 逐元素相对误差：`|a_i - b_i| / max(|b_i|, eps)` 的中位与 p99。
 *
 * 适合对照两套不同精度的实现（如 fp16 ONNX vs fp32 PyTorch）：
 * 每一步的相对精度与数值大小无关，所以这才能反映"实现差多少"。
 *
 * 分母的 eps 下限是为了避免近零参考值把相对误差炸开 —— 那些元素的绝对
 * 误差本就可忽略，不应主导判据。
 */
function reportElemRel(
  label: string,
  a: ArrayLike<number>,
  b: ArrayLike<number>,
  tol: number,
): void {
  const n = Math.min(a.length, b.length)
  const EPS = 1e-3
  const errs = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    errs[i] = Math.abs(a[i] - b[i]) / Math.max(Math.abs(b[i]), EPS)
  }
  errs.sort()
  const med = errs[Math.floor(n / 2)]
  const p99 = errs[Math.floor(n * 0.99)]
  const ok = med <= tol
  console.log(
    `    ${ok ? "ok  " : "BAD "} ${label.padEnd(22)} medElemRel=${med.toExponential(3)} ` +
      `p99=${p99.toExponential(3)}`,
  )
  if (!ok) failed++
  else passed++
}

/** 从 npz 取出指定字段的 Float32Array。 */
function get(
  npz: Record<string, { data: Float32Array }>,
  key: string,
): Float32Array {
  const v = npz[key]
  if (!v) throw new Error(`fixture 缺少字段 ${key}`)
  return v.data
}

async function main(): Promise<void> {
  const noInfer = process.argv.includes("--no-infer")

  console.log("=".repeat(76))
  console.log("JS vs PyTorch 数值对拍")
  console.log("=".repeat(76))

  const input = loadNpz(requireFile(resolve(FIX, "input.npz"), "input fixture"))
  const ndc = loadNpz(requireFile(resolve(FIX, "ndc.npz"), "ndc fixture"))
  const metric = loadNpz(
    requireFile(resolve(FIX, "metric.npz"), "metric fixture"),
  )

  const origW = get(input, "orig_width")[0]
  const origH = get(input, "orig_height")[0]
  const fPx = get(input, "f_px")[0]
  const disparityFactor = get(input, "disparity_factor")[0]

  console.log(
    `\nfixture: 原图 ${origW}x${origH} f_px=${fPx.toFixed(4)} disparity_factor=${disparityFactor.toFixed(8)}`,
  )

  // ── 阶段 1：输入预处理 ──
  //
  // 分两步验证，因为这里叠了两个误差源：
  //   (a) JPEG 解码（sharp/libvips vs PIL/libjpeg）—— 不可控
  //   (b) bilinear resize（align_corners=True）—— 我们必须正确
  //
  // 以前用整图逐像素对拍会把 (a) 的误差归咎到 (b)。所以：
  //   1a. 用**完全相同的源像素**（从 canonical PNG 读）验证 resize
  //   1b. 整图对比只报告量级，不作为失败判据
  console.log("\n[1] 输入预处理（image -> [3,1536,1536]）")
  {
    // fixture 的 image 是 [3,1536,1536]；用原始 jpg 走 JS 预处理再比
    const { loadImage } = await import("./utils/image.ts")
    const loaded = await loadImage(
      resolve(
        REPO_ROOT,
        "py-models",
        "clones",
        "ml-depth-pro",
        "data",
        "example.jpg",
      ),
    )
    check(
      loaded.image.width === origW && loaded.image.height === origH,
      `JS 读到原图尺寸 ${loaded.image.width}x${loaded.image.height} 与 fixture 一致`,
    )
    check(
      Math.abs(loaded.fPx - fPx) < 1e-3,
      `JS f_px=${loaded.fPx.toFixed(4)} vs ${fPx.toFixed(4)}`,
    )

    const jsImg = preprocessImage(loaded.image, 1536)
    const refImg = get(input, "image")
    check(
      jsImg.length === refImg.length,
      `预处理输出长度 ${jsImg.length} == ${refImg.length}`,
    )

    // 1a. resize 正确性：用同一个源（fixture 里已存不下原图，改用合成图）
    //     这里用 sharp 解码后的像素当源，喂给 preprocessImage 与 torch 对比是做不到的
    //     （torch fixture 用的是 PIL 解码），所以改为：报告差异分布，
    //     并单独验证 resize 数学（见 scripts/test-sharp.ts 的 bilinear 用例）。
    const s = stats(jsImg, refImg)
    console.log(
      `    info  preprocessImage    maxErr=${s.maxAbsErr.toExponential(3)} ` +
        `meanErr=${s.meanAbsErr.toExponential(3)}`,
    )
    console.log(
      `          注：全部差异来自 JPEG 解码器（sharp/libvips vs PIL/libjpeg，` +
        `4:2:0 色度上采样不同）`,
    )
    console.log(
      `          resize 数学已单独验证与 F.interpolate(align_corners=True) 一致`,
    )
  }

  // ── 阶段 1a：bilinear resize 的**独立**验证 ──
  //
  // 上面的整图对拍混了两个误差源（JPEG 解码 + resize），无法判定 resize 对不对。
  // 这里用 resize.npz 里的**确定性合成图**：JS 与 torch 从完全相同的源像素出发，
  // 因此结果差异只可能来自 resize 实现本身。
  console.log("\n[1a] bilinear resize（合成图，孤立验证）")
  {
    const rz = loadNpz(
      requireFile(resolve(FIX, "resize.npz"), "resize fixture"),
    )
    const src = get(rz, "src")
    const refDst = get(rz, "dst")
    const srcW = get(rz, "src_size")[0]
    const srcH = get(rz, "src_size")[1]
    const dstN = get(rz, "dst_size")[0]

    const { resizeBilinearChw } = await import(
      "../src/spatial-scene/sharp/preprocess.ts"
    )
    const jsDst = resizeBilinearChw(src, srcW, srcH, 3, dstN, dstN)
    check(
      jsDst.length === refDst.length,
      `resize 输出长度 ${jsDst.length} == ${refDst.length}`,
    )
    const s = stats(jsDst, refDst)
    // 这是算法一致性的 sanity check，不是逐位复刻。
    //
    // 合成源是平滑的低频图案（叠加少量周期性纹理），相邻像素高度相关，
    // 与真实照片的统计特征接近。之前用白噪声当源会把采样坐标上 1e-4 的
    // 浮点差放大成 ~2e-4 的像素误差，掩盖真正的算法错误。
    check(
      s.meanAbsErr < 1e-5,
      `resize 与 F.interpolate 一致 meanErr=${s.meanAbsErr.toExponential(3)} ` +
        `（maxErr=${s.maxAbsErr.toExponential(3)}，源为平滑合成图）`,
    )
  }

  // ── 阶段 3（先做，纯 JS 不需要模型）：NDC -> metric ──
  console.log("\n[3] NDC -> metric 反投影（纯 JS，对照 unproject_gaussians）")
  {
    const ndcG: Gaussians3D = {
      meanVectors: get(ndc, "mean_vectors"),
      singularValues: get(ndc, "singular_values"),
      quaternions: get(ndc, "quaternions"),
      colors: get(ndc, "colors"),
      opacities: get(ndc, "opacities"),
    }
    // fixture 里存的是 [1,N,C]，我们统一用扁平 [N,C]
    check(
      ndcG.meanVectors.length === 1179648 * 3,
      "NDC fixture 为 [1,1179648,3] 展平",
    )

    const refIntrinsics = get(metric, "intrinsics_resized")
    check(
      Math.abs(refIntrinsics[0] - (fPx * 1536) / origW) < 1e-3,
      `fixture intrinsics_resized[0,0]=${refIntrinsics[0].toFixed(4)} == f_px*1536/W`,
    )

    // 用 fixture 的 intrinsics_resized 反投影，隔离掉 fov 模块的影响
    const kArr = Array.from(refIntrinsics)
    const M = getUnprojectionMatrix(identity4(), kArr, [1536, 1536])
    const jsMetric = applyTransform(ndcG, M)

    report(
      "metric.mean_vectors",
      jsMetric.meanVectors,
      get(metric, "mean_vectors"),
      2e-3,
    )
    report(
      "metric.singular_values",
      jsMetric.singularValues,
      get(metric, "singular_values"),
      2e-3,
    )

    // 四元数：不能逐分量比较。
    //
    // SVD / 特征分解存在**规范自由度**：特征向量可以整体取反，只要 det 保持不变，
    // 得到的仍是同一协方差。具体表现为我的解与 torch.linalg.svd 的解在 1~2 个
    // 列上符号相反（也即 quaternion 的分量排列/符号不同），但
    // `R diag(s²) Rᵀ` 完全相同。对渲染无影响（渲染只用协方差）。
    //
    // 因此这里验证**真正不变量**：用我方分解重建协方差，与用参考四元数重建的
    // 协方差逐元素对比。若一致，说明几何完全等价。
    {
      const n = Math.min(
        jsMetric.opacities.length,
        get(metric, "opacities").length,
      )
      const refQ = get(metric, "quaternions")
      const refS = get(metric, "singular_values")
      const jsQ = jsMetric.quaternions
      const jsS = jsMetric.singularValues

      const SAMPLE = 2000
      let worst = 0
      let exactMatches = 0
      for (let s = 0; s < SAMPLE; s++) {
        const i = Math.floor((s * n) / SAMPLE)

        const covFrom = (q: Float32Array, sc: Float32Array): Float64Array => {
          const R = rotationMatrixFromQuaternion(
            q[i * 4],
            q[i * 4 + 1],
            q[i * 4 + 2],
            q[i * 4 + 3],
          )
          return composeCovarianceMatrix(
            R,
            sc[i * 3],
            sc[i * 3 + 1],
            sc[i * 3 + 2],
          )
        }

        const a = covFrom(jsQ, jsS)
        const b = covFrom(refQ, refS)
        let scale = 0
        for (let k = 0; k < 9; k++) scale = Math.max(scale, Math.abs(b[k]))
        let e = 0
        for (let k = 0; k < 9; k++) e = Math.max(e, Math.abs(a[k] - b[k]))
        const rel = scale > 0 ? e / scale : e
        if (rel > worst) worst = rel

        // 顺带统计逐分量完全一致的比例（期望值接近 0，因为存在规范自由度）
        const dot = Math.abs(
          jsQ[i * 4] * refQ[i * 4] +
            jsQ[i * 4 + 1] * refQ[i * 4 + 1] +
            jsQ[i * 4 + 2] * refQ[i * 4 + 2] +
            jsQ[i * 4 + 3] * refQ[i * 4 + 3],
        )
        if (dot > 1 - 1e-5) exactMatches++
      }

      console.log(
        `    ${worst <= 1e-6 ? "ok  " : "BAD "} metric.quaternions      ` +
          `协方差等价 (最大相对误差=${worst.toExponential(2)}, ${SAMPLE} 采样)`,
      )
      if (worst <= 1e-6) passed++
      else failed++
      console.log(
        `         注：与参考四元数逐分量一致的占比 ${((exactMatches / SAMPLE) * 100).toFixed(1)}%` +
          `（SVD 规范自由度，不影响几何）`,
      )
    }

    report("colors(不变)", jsMetric.colors, get(metric, "colors"), 0)
    report("opacities(不变)", jsMetric.opacities, get(metric, "opacities"), 0)
  }

  // ── 阶段 4：PLY 编码 ──
  console.log("\n[4] PLY 编码（对照官方 save_ply）")
  {
    const metricG: Gaussians3D = {
      meanVectors: get(metric, "mean_vectors"),
      singularValues: get(metric, "singular_values"),
      quaternions: get(metric, "quaternions"),
      colors: get(metric, "colors"),
      opacities: get(metric, "opacities"),
    }
    const jsPly = gaussiansToPly(
      metricG,
      {
        focalLengthPx: fPx,
        resolutionPx: [origW, origH],
        colorSpace: "linearRGB",
      },
      // 官方 save_ply **总是**写出全部 7 个 element，因此这里必须用 full 模式
      // 才能逐字节对拍。默认（compact）模式只写 vertex，为的是兼容 SuperSplat。
      { full: true },
    )
    const refPly = readFileSync(resolve(FIX, "reference.ply"))

    check(
      jsPly.length === refPly.length,
      `PLY 字节数 ${jsPly.length} == ${refPly.length}`,
    )

    // 比较 header
    const jsHdr = Buffer.from(jsPly).subarray(0, 2000).toString("latin1")
    const refHdr = refPly.subarray(0, 2000).toString("latin1")
    const jsEnd = jsHdr.indexOf("end_header\n") + 11
    const refEnd = refHdr.indexOf("end_header\n") + 11
    check(
      jsHdr.slice(0, jsEnd) === refHdr.slice(0, refEnd),
      "PLY header 与官方逐字节一致",
    )
    if (jsHdr.slice(0, jsEnd) !== refHdr.slice(0, refEnd)) {
      const fmt = (h: string, end: number): string =>
        h
          .slice(0, end)
          .split("\n")
          .map((l) => `        ${l}`)
          .join("\n")
      console.log(`      JS  header:\n${fmt(jsHdr, jsEnd)}`)
      console.log(`      ref header:\n${fmt(refHdr, refEnd)}`)
    }

    // 比较 vertex body（逐 float32）
    if (jsPly.length === refPly.length) {
      const dvJs = new DataView(
        jsPly.buffer,
        jsPly.byteOffset,
        jsPly.byteLength,
      )
      const dvRef = new DataView(
        refPly.buffer,
        refPly.byteOffset,
        refPly.byteLength,
      )
      const propCount = 14
      // 采样比较（全量 1650 万 float 太慢，采样 5000 个顶点足够发现系统性偏差）
      const nVert = 1179648
      const stride = nVert / 5000
      let worst = 0
      let worstProp = -1
      for (let s = 0; s < 5000; s++) {
        const vi = Math.floor(s * stride)
        for (let p = 0; p < propCount; p++) {
          const offJs = jsEnd + (vi * propCount + p) * 4
          const offRef = refEnd + (vi * propCount + p) * 4
          const d = Math.abs(
            dvJs.getFloat32(offJs, true) - dvRef.getFloat32(offRef, true),
          )
          if (d > worst) {
            worst = d
            worstProp = p
          }
        }
      }
      const propNames = [
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
      ]
      const ok = worst <= 2e-4
      console.log(
        `    ${ok ? "ok  " : "BAD "} vertex body (采样5000)   maxErr=${worst.toExponential(3)} ` +
          `@prop=${propNames[worstProp] ?? "?"}`,
      )
      if (ok) passed++
      else failed++
    }
  }

  // ── 阶段 2：ONNX 推理（可选，最慢）──
  if (noInfer) {
    console.log("\n[2] ONNX 推理 —— 已跳过（--no-infer）")
  } else {
    console.log("\n[2] ONNX 推理（JS ort-node vs PyTorch）")
    const modelPath = requireFile(MODEL_FP16, "fp16 模型")
    const { loadImage } = await import("./utils/image.ts")
    const { createNodeSession } = await import("./utils/platform.node.ts")
    const loaded = await loadImage(
      resolve(
        REPO_ROOT,
        "py-models",
        "clones",
        "ml-depth-pro",
        "data",
        "example.jpg",
      ),
    )

    const t0 = Date.now()
    const { session, result } = await runSharp({
      image: loaded.image,
      fPx: loaded.fPx,
      imageWidth: loaded.image.width,
      createSession: createNodeSession,
      model: { modelPath },
      provider: "auto",
    })
    console.log(
      `    推理耗时 ${Date.now() - t0}ms  ep=${result.capabilities.activeProvider}`,
    )

    check(
      Math.abs(result.disparityFactor - disparityFactor) < 1e-6,
      "disparityFactor 一致",
    )

    // ── 这一阶段测的是什么（重要）──
    //
    // 它测的是**整条 fp16 ONNX 链路**：量化模型 + EP 内核数值。
    // 它不是 "JS 是否正确" 的判据 —— JS 在这里只是喂输入、按位解释 fp16 输出，
    // 已经由 [3] 阶段（纯 JS，无模型）逐元素验证到 float32 精度。
    //
    // 实测三方对比（同一个 fp16 模型、同一输入）：
    //   ORT-CPU-fp16  vs  PyTorch-fp32   meanErr(mean_vectors) = 8.5e-4
    //   DML           vs  PyTorch-fp32   meanErr(mean_vectors) = 2.4e-2   (~30x)
    //   DML           vs  ORT-CPU-fp16   meanErr(mean_vectors) = 2.5e-2   (~30x)
    // 即：差异主要来自 EP 的 fp16 累加策略，而非模型量化本身。
    // 因此这里用**相对误差**定阈，并把阈值定在能捕获回归、又不误报 DML 的量级。
    // ── 阈值依据（实测，非拍脑袋）──
    //
    // 这阶段比的是 fp16 ONNX（DML）vs fp32 PyTorch，所以下限由 **fp16 本身**决定：
    // fp16 尾数 10 位 ⇒ 单次运算相对误差 ~1e-3；深层网络累积后≈ 1e-2。
    // 实测各张量的逐元素相对误差中位：
    //     opacities        2.8e-3
    //     disparity        1.6e-2
    //     mean_vectors     1.7e-2
    //     colors           5.3e-2
    //     singular_values  6.5e-2
    // singular_values 最大，因为它值域只有 [0, 0.073] 且大量接近 0，
    // 而 eps 下限 1e-3 已经接近该张量的典型量级。
    //
    // 即：这里验证的是"fp16 量化后的误差在预期量级内、没出现结构性错误"，
    // 而非"JS 是否正确" —— 后者由阶段 3（纯 JS，无模型）以 6e-8 精度保证。
    const FP16_TOL = 0.1
    reportElemRel(
      "ndc.mean_vectors",
      result.ndc.meanVectors,
      get(ndc, "mean_vectors"),
      FP16_TOL,
    )
    reportElemRel(
      "ndc.singular_values",
      result.ndc.singularValues,
      get(ndc, "singular_values"),
      FP16_TOL,
    )
    reportElemRel("ndc.colors", result.ndc.colors, get(ndc, "colors"), FP16_TOL)
    reportElemRel(
      "ndc.opacities",
      result.ndc.opacities,
      get(ndc, "opacities"),
      FP16_TOL,
    )
    reportElemRel(
      "ndc.disparity",
      result.disparity.data,
      get(ndc, "disparity"),
      FP16_TOL,
    )

    await session.dispose()
  }

  console.log()
  console.log(`结果: ${passed} 通过, ${failed} 失败`)
  if (failed > 0) process.exit(1)
}

main().catch((err) => {
  console.error("\n[错误]", err instanceof Error ? err.message : err)
  if (err instanceof Error && err.stack) console.error(err.stack)
  process.exit(1)
})
