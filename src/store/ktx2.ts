/**
 * KTX2 transcoder 的 Vite 资源接线。
 *
 * `KTX2Loader` 自带的是「目录 + 固定文件名」API，不适合 Vite 产出的 hash 文件名。
 * 这里用 `?url` 让 Vite 追踪 JS/WASM 依赖，再预初始化 loader 的内部 worker；
 * loader 后续调用 `init()` 时会直接复用已完成的 `transcoderPending`。
 */

import type * as THREE from "three"
import basisTranscoderJsUrl from "three/examples/jsm/libs/basis/basis_transcoder.js?url"
import basisTranscoderWasmUrl from "three/examples/jsm/libs/basis/basis_transcoder.wasm?url"
import { KTX2Loader } from "three/examples/jsm/loaders/KTX2Loader.js"

interface KTX2LoaderInternals {
  workerConfig: object
  workerSourceURL: string
  transcoderBinary: ArrayBuffer | null
  transcoderPending: Promise<void> | null
  workerPool: {
    setWorkerCreator(creator: () => Worker): void
  }
}

interface KTX2LoaderStatics {
  BasisWorker: () => void
  EngineFormat: object
  EngineType: object
  TranscoderFormat: object
  BasisFormat: object
}

export function createKTX2Loader(renderer: THREE.WebGLRenderer): KTX2Loader {
  const loader = new KTX2Loader().detectSupport(renderer)
  const internals = loader as unknown as KTX2LoaderInternals
  const statics = KTX2Loader as unknown as KTX2LoaderStatics

  internals.transcoderPending = Promise.all([
    fetch(basisTranscoderJsUrl).then((response) => {
      if (!response.ok) {
        throw new Error(
          `KTX2 transcoder JS 加载失败: ${response.status} ${response.statusText}`,
        )
      }
      return response.text()
    }),
    fetch(basisTranscoderWasmUrl).then((response) => {
      if (!response.ok) {
        throw new Error(
          `KTX2 transcoder WASM 加载失败: ${response.status} ${response.statusText}`,
        )
      }
      return response.arrayBuffer()
    }),
  ]).then(([jsContent, binaryContent]) => {
    const workerFunction = statics.BasisWorker.toString()
    const body = [
      "/* constants */",
      `let _EngineFormat = ${JSON.stringify(statics.EngineFormat)}`,
      `let _EngineType = ${JSON.stringify(statics.EngineType)}`,
      `let _TranscoderFormat = ${JSON.stringify(statics.TranscoderFormat)}`,
      `let _BasisFormat = ${JSON.stringify(statics.BasisFormat)}`,
      "/* basis_transcoder.js */",
      jsContent,
      "/* worker */",
      workerFunction.slice(
        workerFunction.indexOf("{") + 1,
        workerFunction.lastIndexOf("}"),
      ),
    ].join("\n")

    internals.workerSourceURL = URL.createObjectURL(new Blob([body]))
    internals.transcoderBinary = binaryContent
    internals.workerPool.setWorkerCreator(() => {
      const worker = new Worker(internals.workerSourceURL)
      const transcoderBinary = internals.transcoderBinary?.slice(0)
      if (!transcoderBinary) throw new Error("KTX2 transcoder WASM 尚未初始化")
      worker.postMessage(
        {
          type: "init",
          config: internals.workerConfig,
          transcoderBinary,
        },
        [transcoderBinary],
      )
      return worker
    })
  })

  return loader
}
