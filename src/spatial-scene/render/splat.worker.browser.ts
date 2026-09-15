/**
 * 场景加载 worker：下载 -> 解析 -> 颜色转换 -> 排序 -> 打包。
 *
 * 为什么放 worker：63 MB PLY 解析 + 1.18M 基数排序 + 打包总计约 1~3 s，
 * 放主线程会让玻璃面板的动效和进度条直接卡住。主线程只拿到最后的
 * 交错顶点缓冲（transferable，零拷贝）去上传 GPU。
 *
 * worker 会**保留**解析后的结构数组，因此后续「换相机重排序」不需要重新
 * 下载/解析，一次往返（~0.3 s）即可。
 *
 * 通信走 Comlink（见 `types.ts: SplatWorkerApi`）：主线程拿到的是普通
 * 异步方法，进度回调与 transferable 缓冲区由 Comlink 负责搬运。
 */

import * as Comlink from "comlink"

import { shDcToLinearRgb } from "./convert.ts"
import { parsePlyHeader, readAuxAndMeta, readVertexSoA } from "./ply.ts"
import { packSplats, sortByViewDepth } from "./sort.ts"
import type {
  LoadProgress,
  LoadResult,
  PackedSplats,
  SortCamera,
  SplatSoA,
  SplatWorkerApi,
} from "./types.ts"
import { REFERENCE_CAMERA } from "./types.ts"

/** worker 内持有的场景状态（一次加载，多次排序）。 */
let state: SplatSoA | null = null

/** 字节 -> `xx.x MB`。 */
function mb(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

/** 下载并报告进度（缺少 `Content-Length` 时退化为不确定进度）。 */
async function fetchWithProgress(
  url: string,
  onProgress: (p: LoadProgress) => void,
): Promise<Uint8Array> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`下载失败 HTTP ${res.status} ${res.statusText}`)
  const total = Number(res.headers.get("content-length") ?? 0)
  const body = res.body
  if (!body) {
    const buf = new Uint8Array(await res.arrayBuffer())
    onProgress({ stage: "fetch", ratio: 1, detail: mb(buf.byteLength) })
    return buf
  }

  const chunks: Uint8Array[] = []
  let loaded = 0
  let lastReport = 0
  const reader = body.getReader()
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    if (!value) continue
    chunks.push(value)
    loaded += value.byteLength
    const now = performance.now()
    // 限流到 ~15 Hz：进度条够顺，又不至于把主线程刷爆
    if (now - lastReport > 66) {
      lastReport = now
      onProgress({
        stage: "fetch",
        ratio: total > 0 ? Math.min(1, loaded / total) : 0,
        detail: total > 0 ? `${mb(loaded)} / ${mb(total)}` : mb(loaded),
      })
    }
  }

  const bytes = new Uint8Array(loaded)
  let offset = 0
  for (const c of chunks) {
    bytes.set(c, offset)
    offset += c.byteLength
  }
  onProgress({ stage: "fetch", ratio: 1, detail: mb(loaded) })
  return bytes
}

/** 排序 + 打包 + 回传（transferable，零拷贝）。 */
function sortAndPack(
  soa: SplatSoA,
  camera: SortCamera,
  onProgress?: (p: LoadProgress) => void,
): PackedSplats {
  onProgress?.({ stage: "sort", ratio: 0 })
  const sort = sortByViewDepth(soa, camera)
  onProgress?.({ stage: "sort", ratio: 1, detail: `${sort.ms.toFixed(0)} ms` })

  onProgress?.({ stage: "pack", ratio: 0 })
  const packed = packSplats(soa, sort.order, sort.ms)
  onProgress?.({
    stage: "pack",
    ratio: 1,
    detail: `${packed.packMs.toFixed(0)} ms`,
  })
  return packed
}

const api: SplatWorkerApi = {
  async load(
    url: string,
    onProgress: (p: LoadProgress) => void,
  ): Promise<LoadResult> {
    const t0 = performance.now()
    const bytes = await fetchWithProgress(url, onProgress)

    const tParse = performance.now()
    onProgress({ stage: "parse", ratio: 0.2 })
    const header = parsePlyHeader(bytes)
    const vertex = readVertexSoA(bytes, header)
    const meta = readAuxAndMeta(bytes, header, vertex)
    onProgress({
      stage: "parse",
      ratio: 1,
      detail: `${meta.count.toLocaleString("en-US")} 个高斯 · ${(
        (performance.now() - tParse) / 1000
      ).toFixed(2)} s`,
    })

    onProgress({ stage: "convert", ratio: 0.5 })
    shDcToLinearRgb(vertex.shDc, vertex.colorLinear)
    onProgress({ stage: "convert", ratio: 1 })

    const soa: SplatSoA = {
      count: vertex.count,
      center: vertex.center,
      scaleLog: vertex.scaleLog,
      quat: vertex.quat,
      colorLinear: vertex.colorLinear,
      opacityLogit: vertex.opacityLogit,
    }
    state = soa

    const packed = sortAndPack(soa, REFERENCE_CAMERA, onProgress)
    onProgress({
      stage: "done",
      ratio: 1,
      detail: `总计 ${((performance.now() - t0) / 1000).toFixed(2)} s`,
    })

    // 顶点缓冲所有权交给主线程（结构化克隆会复制 66 MB，必须转移）
    return Comlink.transfer({ meta, packed }, [packed.data.buffer])
  },

  async resort(camera: SortCamera): Promise<PackedSplats> {
    if (!state) throw new Error("resort 之前必须先 load")
    const packed = sortAndPack(state, camera)
    return Comlink.transfer(packed, [packed.data.buffer])
  },

  dispose(): void {
    state = null
  },
}

Comlink.expose(api)
