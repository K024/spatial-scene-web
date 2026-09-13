/**
 * 把 `LoadedScene.root` 挂进 r3f，并驱动**逐层可见性**与**线框 / 标准渲染的切换**。
 *
 * ── 为什么可见性是「每帧插值」而不是直接 `visible = false` ──
 * 层栈的观感全靠 α 混合；硬切会让「关掉一层」变成跳变。这里把 `material.opacity`
 * 指数逼近目标（0 / 1），只有真正到位才停帧。目标值每帧从 `$layers.peek()` 现取 ——
 * **非 React 读法**，不给 React 加每帧负担，也就不用把「可见性」再镜像一份到 ref 里。
 *
 * ── 为什么用 `$solo` ──
 * 独显是**视图态**而不是层数据（`LayerState.visible`），所以不写回 `$layers`，
 * 而是在这里做一次覆盖。这样「取消 solo」能立刻恢复用户原来的勾选组合。
 *
 * ── α 的真相源在**基准（纹理）材质**上 ──
 * 线框模式是原地换 `mesh.material`（见 `wireframe.ts`），所以 `mesh.material` 可能是
 * 两条材质中的任意一条。这里统一把 α 写在 `pair.base.opacity` 上，再镜像给线框材质 ——
 * 两条材质**永远同 α**，来回切换不会出现「淡出到一半切模式」的错位。
 *
 * ── 本组件同时是线框开关的订阅者 ──
 * 换材质与续帧必须同时发生，所以没有把 `$wireframe` 放进 `InvalidateOnChange` 的列表，
 * 而是在这里自己订阅（那个文件里有对应说明）。
 *
 * ── 按需渲染下的续帧义务 ──
 * `mesh.visible` / `material.opacity` / `mesh.material` 都是**命令式**写的，不产生
 * React 提交，r3f 的 `invalidateInstance` 兜不到。所以切换时、以及还有层没淡到位时，
 * 都自己 `invalidate()`；全部到位后停手，画布回到 0 帧/秒。
 */

import { useFrame, useThree } from "@react-three/fiber"
import { useEffect } from "react"
import type { Material, Texture } from "three"
import { $layers, $scene, $solo, $wireframe } from "../store/viewer.ts"
import {
  layerMaterialsOf,
  registerLayerMaterials,
  setSceneWireframe,
} from "./wireframe.ts"

/** 可见性逼近速率（1/s）；越大越硬。 */
const FADE_RATE = 18
/** 离目标多近就直接吸附过去（省掉无限逼近的小数尾巴）。 */
const SETTLE_EPSILON = 0.002
/** 低于这个 α 直接不画（省掉一个全透明图层的绘制开销）。 */
const VISIBLE_EPSILON = 0.008

export function ModelRoot() {
  const scene = $scene.useValue()
  const gl = useThree((state) => state.gl)
  const r3fScene = useThree((state) => state.scene)
  const camera = useThree((state) => state.camera)
  const invalidate = useThree((state) => state.invalidate)

  // 线框开关：与续帧绑在一起（换材质必须紧跟一帧才看得见）。
  // 开线框会换到一组**新的着色程序**（`USE_WIREFRAME` 变体），所以顺手 `gl.compile`
  // 一下，把顶点/片元编译从「下一帧的渲染里」挪到这里 —— 否则首次开启会卡一下。
  useEffect(() => {
    return $wireframe.subscribe(() => {
      const loaded = $scene.peek()
      if (!loaded) return
      setSceneWireframe(loaded, $wireframe.peek())
      gl.compile(r3fScene, camera)
      invalidate()
    })
  }, [gl, r3fScene, camera, invalidate])

  useFrame((_, delta) => {
    const loaded = $scene.peek()
    if (!loaded) return
    const states = $layers.peek()
    const solo = $solo.peek()
    // 夹住 delta：切标签页回来时 delta 很大，会让插值一步到位（等于硬切）。
    const step = 1 - Math.exp(-Math.min(delta, 0.1) * FADE_RATE)
    let animating = false

    for (const state of states) {
      const mesh = loaded.meshes.get(state.key)
      if (!mesh) continue
      const pair = layerMaterialsOf(mesh)
      if (!pair) continue
      const target = (solo === null ? state.visible : state.key === solo)
        ? 1
        : 0
      let opacity = pair.base.opacity + (target - pair.base.opacity) * step
      if (Math.abs(target - opacity) <= SETTLE_EPSILON) {
        opacity = target
      } else {
        animating = true
      }
      // α 写在基准材质上（真相源），再镜像给线框材质。
      pair.base.opacity = opacity
      if (pair.wire) pair.wire.opacity = opacity
      mesh.visible = opacity > VISIBLE_EPSILON
    }

    if (animating) invalidate()
  })

  // 场景挂载后的**一次性收尾**（在 effect 里做，此时 `<primitive>` 已经进场景图）：
  // 1. 登记每条 mesh 的基准材质 —— 必须在任何线框切换之前（`wireframe.ts` 的硬要求），
  //    并立刻把当前的线框状态套上去（用户可能开着线框又换了个场景）；
  // 2. 各向异性过滤：层是斜着看的浮雕网格，`1×` 采样在掠射角会糊成一片，
  //    开到硬件上限后层间边界与裙边纹理才立得住（代价只是采样，不增显存）；
  // 3. `gl.compile`：预编译本场景的着色器程序，把编译耗时从「载入后的第一帧」
  //    挪到这个 effect 里，避免首帧卡一下。
  useEffect(() => {
    if (!scene) return
    registerLayerMaterials(scene)
    if ($wireframe.peek()) setSceneWireframe(scene, true)

    const anisotropy = gl.capabilities.getMaxAnisotropy()
    for (const mesh of scene.meshes.values()) {
      const pair = layerMaterialsOf(mesh)
      if (!pair) continue
      const texture = (pair.base as Material & { map?: Texture | null }).map as
        | Texture
        | null
        | undefined
      if (!texture || texture.anisotropy === anisotropy) continue
      texture.anisotropy = anisotropy
      texture.needsUpdate = true
    }
    gl.compile(r3fScene, camera)
  }, [scene, gl, r3fScene, camera])

  // `key` 强制换场景时重建 primitive（r3f 对同一 primitive 的 object 变更处理不直观）。
  return scene ? <primitive key={scene.root.uuid} object={scene.root} /> : null
}
