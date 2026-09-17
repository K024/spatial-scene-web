/**
 * 查看器显示状态（全局信号）。
 *
 * 全是「看一眼」用的开关：背景、网格地面、gizmo、图层隔离、爆炸视图。
 * 渲染热路径（每帧）不读这些信号以外的东西。
 */

import "./signals-hook.ts"
import { signal } from "@preact/signals-react"

/** 清屏色（sRGB 十六进制）。 */
export const background = signal("#0a0b10")

/** 网格地面（drei Grid）。 */
export const showGrid = signal(true)
/** 坐标轴 gizmo（drei GizmoHelper）。 */
export const showGizmo = signal(true)
/** WebGL MSAA。上下文创建时固定，切换会重建 Canvas。默认关闭。 */
export const msaa = signal(false)

/** 全部图层的整体不透明度倍率。 */
export const layerOpacity = signal(1)
/** 线框模式（看 LOD 出面密度）。 */
export const wireframe = signal(false)
/** 双面（斜视角看到背面而不是露洞）。 */
export const doubleSided = signal(true)

/** 只显示某一层（-1 = 全部）。 */
export const isolateLayer = signal(-1)
/** 爆炸视图强度（0 = 关）：把各层沿视线方向拉开，看层间结构。 */
export const explode = signal(0)

/** 右侧浮层面板。 */
export const panelOpen = signal(true)
