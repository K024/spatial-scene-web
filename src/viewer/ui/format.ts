/**
 * 渲染端 UI 的格式化助手（纯函数，无状态）。
 */

/** 米 -> 人类可读（< 1m 用厘米，< 1000m 用米，再往上用千米）。 */
export function formatMeters(meters: number): string {
  if (!Number.isFinite(meters)) return "—"
  const abs = Math.abs(meters)
  if (abs < 1) return `${(meters * 100).toFixed(0)} cm`
  if (abs < 1000) return `${meters.toFixed(2)} m`
  return `${(meters / 1000).toFixed(2)} km`
}

/** 深度带 `[lo, hi]`：跨度极小时退化成单值。 */
export function formatDepthRange(range: readonly [number, number]): string {
  const [lo, hi] = range
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return "—"
  if (Math.abs(hi - lo) < 1e-3) return formatMeters(lo)
  return `${formatMeters(lo)} – ${formatMeters(hi)}`
}

/** 大整数缩写（12.4k / 1.20M）。 */
export function formatCount(count: number): string {
  if (!Number.isFinite(count)) return "—"
  if (count < 1000) return String(count)
  if (count < 1_000_000)
    return `${(count / 1000).toFixed(count < 10_000 ? 1 : 0)}k`
  return `${(count / 1_000_000).toFixed(2)}M`
}

/** 字节缩写。 */
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes)) return "—"
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`
}

/** 弧度 -> 度（一位小数）。 */
export function formatDegrees(radians: number): string {
  if (!Number.isFinite(radians)) return "—"
  return `${((radians * 180) / Math.PI).toFixed(1)}°`
}

/** 视差是 `[0,1]` 的 NDC 量，用三位小数足够。 */
export function formatDisparity(value: number): string {
  if (!Number.isFinite(value)) return "—"
  return value.toFixed(3)
}
