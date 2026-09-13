/**
 * WebGL2 可用性探测。
 *
 * 必须在**挂载 `<Canvas>` 之前**同步跑完：r3f 拿不到 WebGL2 上下文时是在渲染期抛错，
 * 那种错误 React 只能靠 error boundary 兜，体验很差。所以探测放在 `main.tsx` 里、
 * `createRoot` 之前，结果写进 `$webgl`，由 `App` 决定挂不挂画布。
 *
 * 探测完立刻 `loseContext()`：只是问一句「有没有」，不该真的占一个上下文
 * （浏览器对同时存在的 WebGL 上下文数量有硬上限）。
 */

export function detectWebgl2(): boolean {
  try {
    const canvas = document.createElement("canvas")
    const context = canvas.getContext("webgl2")
    if (!context) return false
    context.getExtension("WEBGL_lose_context")?.loseContext()
    return true
  } catch {
    return false
  }
}
