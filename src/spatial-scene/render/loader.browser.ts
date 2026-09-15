/**
 * 场景加载客户端：主线程侧的 worker 门面。
 *
 * 把 `Comlink.wrap` 出来的 RPC 代理包成一个「可取消、可重排」的对象，
 * 并负责 worker 的生命周期（一个页面只需要一个 worker）。
 */

import * as Comlink from "comlink"

import type {
  LoadProgress,
  LoadResult,
  PackedSplats,
  SortCamera,
  SplatWorkerApi,
} from "./types.ts"

/** 加载会话：加载完 `result` 就绪，之后可以反复 `resort`。 */
export interface SceneLoader {
  /** 加载结果（仅 `load` 成功后可用）。 */
  result: LoadResult | null
  /** 用新相机重排序。 */
  resort(camera: SortCamera): Promise<PackedSplats>
  /** 销毁 worker。 */
  dispose(): void
}

/**
 * 启动加载。
 *
 * @param url PLY 地址。
 * @param onProgress 进度回调（跨线程代理，注意调用频率已在 worker 限流）。
 * @returns 加载完成后的会话对象。
 */
export async function loadScene(
  url: string,
  onProgress: (p: LoadProgress) => void,
): Promise<SceneLoader> {
  const worker = new Worker(
    new URL("./splat.worker.browser.ts", import.meta.url),
    {
      type: "module",
      name: "splat-loader",
    },
  )
  const api = Comlink.wrap<SplatWorkerApi>(worker)

  const result = await api.load(url, Comlink.proxy(onProgress))
  return {
    result,
    resort: (camera) => api.resort(camera),
    dispose: () => {
      api[Comlink.releaseProxy]()
      void Promise.resolve(api.dispose()).catch(() => {})
      worker.terminate()
    },
  }
}
