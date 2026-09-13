import babel from "@rolldown/plugin-babel"
import tailwindcss from "@tailwindcss/vite"
import react, { reactCompilerPreset } from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    // Tailwind v4（含 FlyOnUI 插件，见 src/index.css）。
    tailwindcss(),
    react(),
    // React Compiler：只做自动 memo 化。**不用** signals 的 transform 方案 ——
    // signal 的订阅在 `Signal.prototype.useValue` 里手写（src/store/signal-hooks.ts）。
    babel({ presets: [reactCompilerPreset()] }),
  ],
})
