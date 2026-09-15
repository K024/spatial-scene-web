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
 * 启动加载（URL）。
 *
 * @param url PLY 地址。
 * @param onProgress 进度回调（跨线程代理，注意调用频率已在 worker 限流）。
 * @returns 加载完成后的会话对象。
 */
export async function loadScene(
  url: string,
  onProgress: (p: LoadProgress) => void,
): Promise<SceneLoader> {
  const { api, worker } = spawnWorker()
  return wrap(api, worker, api.load(url, Comlink.proxy(onProgress)))
}

/**
 * 启动加载（本地文件字节）。
 *
 * `bytes` 会被**转移**给 worker（零拷贝）——调用后不要再用它。
 * 63 MB 的 PLY 如果走结构化克隆会被复制一份，白添一次内存峰值。
 */
export async function loadSceneFromBytes(
  bytes: ArrayBuffer,
  onProgress: (p: LoadProgress) => void,
): Promise<SceneLoader> {
  const { api, worker } = spawnWorker()
  return wrap(
    api,
    worker,
    api.loadBytes(Comlink.transfer(bytes, [bytes]), Comlink.proxy(onProgress)),
  )
}

/** 起一个 worker 并拿到 RPC 代理。 */
function spawnWorker(): {
  api: Comlink.Remote<SplatWorkerApi>
  worker: Worker
} {
  const worker = new Worker(
    new URL("./splat.worker.browser.ts", import.meta.url),
    {
      type: "module",
      name: "splat-loader",
    },
  )
  return { api: Comlink.wrap<SplatWorkerApi>(worker), worker }
}

/** 把「加载中的 promise」包成会话对象。 */
async function wrap(
  api: Comlink.Remote<SplatWorkerApi>,
  worker: Worker,
  result: Promise<LoadResult>,
): Promise<SceneLoader> {
  const loaded = await result
  let disposed = false
  return {
    result: loaded,
    resort: (camera) => api.resort(camera),
    dispose: () => {
      // 幂等：释放可能被调两次（换场景 + 页面卸载 / 热更新）
      if (disposed) return
      disposed = true
      // 顺序要紧：
      // 1. 先 `terminate()`。这是**确定性**回收（worker 里的 SoA 有 60 MB+），
      //    而 worker 自己的 `dispose()` RPC 只是把 state 置 null 等 GC，
      //    发了也可能被紧接着的 terminate 提前掉——所以不再调它。
      // 2. 再 `releaseProxy()`。一旦撑销，对该代理的**任何**调用（包括
      //    `api.dispose()`）都会同步抛 `Proxy has been released and is not
      //    useable`。早先的写法把撑销放前面，于是 `api.dispose()` 那一抛
      //    中断了整个 dispose：既没走到 `terminate()`（每个 worker 泄漏一
      //    个线程 + 60 MB），调用方还会收到一个假错误。
      worker.terminate()
      api[Comlink.releaseProxy]()
    },
  }
}
