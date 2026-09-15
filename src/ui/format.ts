/**
 * 面板显示用的格式化函数。
 *
 * 统一放在这里而不是散落在组件里：面板上几十处数字，
 * 只要有一处口径不同（比如千分位、小数位）就会显得很脏。
 */

/** 整数 + 千分位。 */
export function fmtInt(v: number): string {
  return Math.round(v).toLocaleString("en-US")
}

/** 字节 -> `63.0 MB`。 */
export function fmtBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  const kb = bytes / 1024
  if (kb < 1024) return `${kb.toFixed(1)} KB`
  const mbVal = kb / 1024
  if (mbVal < 1024) return `${mbVal.toFixed(1)} MB`
  return `${(mbVal / 1024).toFixed(2)} GB`
}

/** 毫秒：小于 1 s 显示整数 ms，否则显示秒。 */
export function fmtMs(ms: number): string {
  if (ms < 1000) return `${ms.toFixed(ms < 10 ? 1 : 0)} ms`
  return `${(ms / 1000).toFixed(2)} s`
}

/** 保留 n 位小数的定点。 */
export function fmtNum(v: number, digits = 2): string {
  return Number.isFinite(v) ? v.toFixed(digits) : "—"
}

/** 米（度量空间）；大了自动切到厘米。 */
export function fmtMeters(v: number): string {
  if (!Number.isFinite(v)) return "—"
  if (Math.abs(v) < 0.1) return `${(v * 100).toFixed(1)} cm`
  return `${v.toFixed(2)} m`
}

/** 三维向量 -> `(0.00, 0.00, 0.00)`。 */
export function fmtVec3(
  v: readonly [number, number, number],
  digits = 2,
): string {
  return `(${v.map((x) => x.toFixed(digits)).join(", ")})`
}

/** 角度：`12.3°`。 */
export function fmtDeg(deg: number): string {
  return `${deg.toFixed(1)}°`
}
