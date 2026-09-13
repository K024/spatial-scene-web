/**
 * 全屏画布 + 场景内容。
 *
 * ── `flat` ──
 * 层纹理是**照片级 unlit**，`baseColorTexture` 已经是 sRGB。r3f 默认给 renderer 上
 * ACESFilmic 色调映射，会把照片色压出「电影感」的偏色。`flat` = `NoToneMapping`，
 * 配合 `outputColorSpace = sRGB`（three 默认），得到**逐像素等于原图**的直出。
 * 这条和 `ModelRoot` 里材质的 `toneMapped = false` 是双保险。
 *
 * ── `frameloop="demand"` ──
 * 静止时完全不渲染（帧率 0）。这是本应用最值钱的一项优化：看照片的时间远多于拖拽。
 * 谁负责续帧、为什么不能只靠 r3f 自己，见 `Invalidate.tsx`。
 *
 * ── `<AdaptiveDpr />` ──
 * r3f 内置的性能回落系统：drei 的 `OrbitControls` 每次 `change` 都会
 * `performance.regress()`，`AdaptiveDpr` 据此在交互期间把 `dpr` 降下来
 * （`performance.current × 初始 dpr`），停下后自动回到满分辨率。
 * 照片纹理对这种降采样很敏感，所以**只在交互中**生效是必要的取舍。
 */

import { AdaptiveDpr } from "@react-three/drei"
import { Canvas } from "@react-three/fiber"
import { InvalidateOnChange } from "./Invalidate.tsx"
import { ModelRoot } from "./ModelRoot.tsx"
import { SceneRig } from "./SceneRig.tsx"

/** 站点背景色（与 `index.css` 的 body 一致，避免首帧闪白）。 */
const BACKGROUND = "#070810"

export function Viewport() {
  return (
    <Canvas
      flat
      frameloop="demand"
      dpr={[1, 2]}
      gl={{
        antialias: true,
        alpha: false,
        powerPreference: "high-performance",
      }}
      camera={{ position: [0, 0, 0], fov: 55, near: 0.05, far: 500 }}
      className="absolute inset-0"
    >
      <color attach="background" args={[BACKGROUND]} />
      <InvalidateOnChange />
      <ModelRoot />
      <SceneRig />
      <AdaptiveDpr />
    </Canvas>
  )
}
