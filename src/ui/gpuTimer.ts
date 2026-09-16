/**
 * GPU 硬件计时（`EXT_disjoint_timer_query_webgl2`）。
 *
 * 与 direct-splat 分支 `render/renderer.browser.ts` 里的同一套实现，抽出来给
 * R3F 的渲染探针用。
 *
 * 为什么必须用它：`performance.now()` 量到的只是**命令下发**用了多久。GL 调用
 * 是异步的，`drawElements` 提交完就返回，真正跑多少时间在 GPU 队列里。计时
 * 查询的用法是在命令流里插入一对时间戳，GPU 执行到那里才会计数，因此量到的
 * 是**真实执行时间**（含全部 pass）。
 *
 * 三个必须遵守的细节：
 * 1. 结果**必须滞后读取**：先问 `QUERY_RESULT_AVAILABLE`，没就绪就下一帧再看。
 *    直接读 `QUERY_RESULT` 会同步等待 GPU，把并行度彻底碱掉。
 * 2. 要查 `GPU_DISJOINT_EXT`：GPU 被抢占/降频时结果不可信，当帧应当丢弃。
 * 3. 不要试图用 `QUERY_COUNTER_BITS_EXT`「探测」可用性：WebGL 对
 *    `getQueryParameter` 的时序要求很苛。改为**直接量**：拿到有效值就认为可用，
 *    连续若干帧拿不到就判定该平台计时器被禁用。
 */

/** 采样窗口内允许积压的最大查询数（超出丢最老的）。 */
const MAX_PENDING_QUERIES = 8

/** 连续多少帧没拿到有效结果就判定「计时器被平台禁用」。 */
const TIMER_GIVE_UP_FRAMES = 120

/** `EXT_disjoint_timer_query_webgl2` 的字段（比 lib.dom 的声明更宽松）。 */
interface DisjointTimerExt {
  readonly TIME_ELAPSED_EXT: number
  readonly GPU_DISJOINT_EXT: number
}

/** GPU 计时的内部状态机。 */
export interface GpuTimer {
  /** 是否认为硬件计时可用（动态判定）。 */
  readonly available: boolean
  begin(): void
  end(): void
  /** 取一个已完成查询的耗时（ms）；没有可用结果时返回 null。 */
  poll(): number | null
  dispose(): void
}

/** 建计时器；没有扩展时返回 null（调用方应在 UI 上诚实标注）。 */
export function createGpuTimer(gl: WebGL2RenderingContext): GpuTimer | null {
  const ext = gl.getExtension(
    "EXT_disjoint_timer_query_webgl2",
  ) as unknown as DisjointTimerExt | null
  if (!ext) return null

  /** 待命查询对象（复用，避免每帧 createQuery）。 */
  const free: WebGLQuery[] = []
  /** 已结束、等 GPU 写完的查询。 */
  const pending: WebGLQuery[] = []
  let active: WebGLQuery | null = null
  let sawResult = false
  let blindFrames = 0

  return {
    get available() {
      return sawResult || blindFrames < TIMER_GIVE_UP_FRAMES
    },

    begin() {
      if (active) return
      const q = free.pop() ?? gl.createQuery()
      if (!q) return
      active = q
      blindFrames++
      gl.beginQuery(ext.TIME_ELAPSED_EXT, q)
    },

    end() {
      const q = active
      if (!q) return
      gl.endQuery(ext.TIME_ELAPSED_EXT)
      active = null
      pending.push(q)
      // 积压过多说明 GPU 严重落后（或查询永不就绪）：丢掉最老的，不无限增长
      while (pending.length > MAX_PENDING_QUERIES) {
        const old = pending.shift()
        if (old) free.push(old)
      }
    },

    /**
     * 取**一个已完成查询**的耗时（ms）。
     *
     * 只有在真的消费掉一个结果时才返回数值，否则返回 null —— 否则同一个样本
     * 会被每帧重复统计，分位数就失真了。
     */
    poll() {
      const q = pending[0]
      if (!q) return null
      if (!gl.getQueryParameter(q, gl.QUERY_RESULT_AVAILABLE)) return null

      pending.shift()
      const disjoint = Boolean(gl.getParameter(ext.GPU_DISJOINT_EXT))
      const ns = disjoint ? 0 : Number(gl.getQueryParameter(q, gl.QUERY_RESULT))
      free.push(q)
      if (disjoint || !Number.isFinite(ns) || ns <= 0) return null

      sawResult = true
      blindFrames = 0
      return ns / 1e6
    },

    dispose() {
      if (active) gl.endQuery(ext.TIME_ELAPSED_EXT)
      for (const q of [...free, ...pending]) gl.deleteQuery(q)
      if (active) gl.deleteQuery(active)
      free.length = 0
      pending.length = 0
      active = null
    },
  }
}
