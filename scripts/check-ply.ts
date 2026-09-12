/**
 * PLY 读回校验（node 侧）。
 *
 * 用途：验证导出的 PLY 能被正确解析，并检查数值是否合理。
 * 这是对 `export/ply.ts` 的自检——不依赖外部工具，
 * 避免「文件写坏了但没人发现」。
 *
 * 用法：npx tsx scripts/check-ply.ts [path]
 */

import { readFileSync } from "node:fs"
import { resolve } from "node:path"

import { REPO_ROOT } from "./utils/common.ts"

interface PlyVertex {
  x: number
  y: number
  z: number
  fdc: [number, number, number]
  opacityLogit: number
  scaleLog: [number, number, number]
  rot: [number, number, number, number]
}

/** 极简 PLY 头解析：只支持本项目写出的 binary_little_endian。 */
function parseHeader(buf: Buffer): {
  headerEnd: number
  count: number
  props: string[]
  elements: { name: string; count: number; props: [string, string][] }[]
} {
  const text = buf.subarray(0, Math.min(buf.length, 1 << 16)).toString("latin1")
  const endIdx = text.indexOf("end_header\n")
  if (endIdx < 0) throw new Error("未找到 end_header")
  const headerEnd = endIdx + "end_header\n".length

  const lines = text.slice(0, endIdx).split("\n")
  if (lines[0].trim() !== "ply") throw new Error("不是 PLY 文件")
  if (!lines[1].includes("binary_little_endian")) {
    throw new Error(`不支持的格式: ${lines[1]}`)
  }

  const elements: { name: string; count: number; props: [string, string][] }[] =
    []
  for (const line of lines.slice(2)) {
    const parts = line.trim().split(/\s+/)
    if (parts[0] === "element") {
      elements.push({
        name: parts[1],
        count: Number.parseInt(parts[2], 10),
        props: [],
      })
    } else if (parts[0] === "property") {
      elements[elements.length - 1].props.push([parts[1], parts[2]])
    }
  }
  const vertex = elements.find((e) => e.name === "vertex")
  if (!vertex) throw new Error("缺少 vertex element")
  return {
    headerEnd,
    count: vertex.count,
    props: vertex.props.map((p) => p[1]),
    elements,
  }
}

function main(): void {
  const path = resolve(
    REPO_ROOT,
    process.argv[2] ?? "py-models/out/ply/example.ply",
  )
  const buf = readFileSync(path)
  const { headerEnd, count, props, elements } = parseHeader(buf)

  console.log(`文件: ${path}`)
  console.log(
    `elements: ${elements.map((e) => `${e.name}(${e.count})`).join(", ")}`,
  )
  console.log(`vertex properties (${props.length}): ${props.join(", ")}`)

  const stride = props.length * 4
  const expected = headerEnd + count * stride
  console.log(
    `字节数: 实际 ${buf.length} / 期望 ${expected} ${buf.length === expected ? "✓" : "✗ 不匹配"}`,
  )

  const dv = new DataView(buf.buffer, buf.byteOffset + headerEnd)

  const readVertex = (i: number): PlyVertex => {
    const o = i * stride
    return {
      x: dv.getFloat32(o, true),
      y: dv.getFloat32(o + 4, true),
      z: dv.getFloat32(o + 8, true),
      fdc: [
        dv.getFloat32(o + 12, true),
        dv.getFloat32(o + 16, true),
        dv.getFloat32(o + 20, true),
      ],
      opacityLogit: dv.getFloat32(o + 24, true),
      scaleLog: [
        dv.getFloat32(o + 28, true),
        dv.getFloat32(o + 32, true),
        dv.getFloat32(o + 36, true),
      ],
      rot: [
        dv.getFloat32(o + 40, true),
        dv.getFloat32(o + 44, true),
        dv.getFloat32(o + 48, true),
        dv.getFloat32(o + 52, true),
      ],
    }
  }

  // ── 统计（区分「全部」与「不透明」）──
  let zMin = Infinity
  let zMax = -Infinity
  let wSum = 0
  let cz = 0
  let opaque = 0
  let nanCount = 0

  for (let i = 0; i < count; i++) {
    const v = readVertex(i)
    if (!Number.isFinite(v.x + v.y + v.z + v.opacityLogit)) {
      nanCount++
      continue
    }
    const alpha = 1 / (1 + Math.exp(-v.opacityLogit))
    wSum += alpha
    cz += v.z * alpha
    if (v.z < zMin) zMin = v.z
    if (v.z > zMax) zMax = v.z
    if (alpha > 0.8) opaque++
  }

  console.log(`\n[数值]`)
  console.log(`  NaN/Inf: ${nanCount}`)
  console.log(`  z 范围(全部): [${zMin.toFixed(3)}, ${zMax.toFixed(3)}]`)
  console.log(`  alpha 加权质心 z: ${(cz / wSum).toFixed(3)} m`)
  console.log(`  不透明(α>0.8) 占比: ${((opaque / count) * 100).toFixed(1)}%`)

  console.log(`\n[前 3 个顶点]`)
  for (let i = 0; i < 3; i++) {
    const v = readVertex(i)
    const alpha = 1 / (1 + Math.exp(-v.opacityLogit))
    console.log(
      `  v${i}: xyz=[${v.x.toFixed(4)}, ${v.y.toFixed(4)}, ${v.z.toFixed(4)}] ` +
        `α=${alpha.toFixed(4)} scale=[${v.scaleLog.map((s) => Math.exp(s).toFixed(5)).join(", ")}] ` +
        `|q|=${Math.hypot(...v.rot).toFixed(4)}`,
    )
  }

  // ── 断言 ──
  const problems: string[] = []
  if (buf.length !== expected) problems.push("字节数与 header 声明不符")
  if (nanCount > 0) problems.push(`存在 ${nanCount} 个 NaN/Inf`)
  if (!(cz / wSum > 0)) problems.push("质心 z 非正（相机应朝向 +z）")
  if (opaque === 0) problems.push("没有不透明高斯")
  for (let i = 0; i < 20; i++) {
    const q = readVertex(i)
    const norm = Math.hypot(...q.rot)
    if (Math.abs(norm - 1) > 1e-3)
      problems.push(`顶点 ${i} 的四元数未归一化: |q|=${norm}`)
  }

  console.log()
  if (problems.length > 0) {
    console.log("✗ 发现问题:")
    for (const p of problems) console.log(`   - ${p}`)
    process.exit(1)
  }
  console.log("✓ 全部检查通过")
}

main()
