/**
 * 扩视角视图：把「参考视角相机」变成「分层渲染用的画布 + 相机」。
 *
 * ── 为什么扩画布而不是缩焦距 ──
 * 单图模型在参考视角画面外仍有高斯（椭球探出边缘、几何/纹理继续外推）。只渲原视角
 * 会把它们裁掉，mesh 一到边缘就切边。扩视角有两种等价做法：
 * 1. **缩焦距**（画布不变）：FOV 变大，但参考视角的像素密度也跟着降，等于全局降采样；
 * 2. **扩画布**（焦距不变，本实现）：参考图内容在画布中心**逐像素同尺度**保留，
 *    四周多出来的是原视角外的新内容。
 * 选 2 的理由：`referenceRect` 是**恒等映射**（同像素焦距），meshing 的坐标换算与
 * 分辨率核算都变成一步平移，不需要再引入第二个重采样尺度。
 *
 * ── 与 meshing 的接口（坐标计算的唯一来源）──
 * ```
 * 渲染像素 (x, y)，深度 z（米，视图空间 z）
 *   -> 相机坐标 ((x + 0.5 − W/2) / fx · z, (y + 0.5 − H/2) / fy · z, z)
 *   -> 世界坐标 = C + right·camX + down·camY + forward·z
 * 参考图像素 (u, v)  ->  渲染像素 = (u + referenceRect.x, v + referenceRect.y)
 * ```
 * `view.focalLengthPx` 是**渲染域**的焦距；`view.referenceFocalLengthPx` 是原图域的。
 * 两者差 `view.pixelScale`，`referenceRect` 就是用这个尺度算出来的。
 */

import { createWSplatCamera, type WSplatCamera } from "../wsplat/camera.ts"
import {
  DEFAULT_MAX_RENDER_SIDE,
  DEFAULT_VIEW_SCALE,
  type LayerView,
} from "./types.ts"

/** 参考视角的最小描述（像素焦距 + 图像尺寸）。 */
export interface ReferenceView {
  readonly width: number
  readonly height: number
  /** **参考域**（原图）的像素焦距。 */
  readonly focalLengthPx: number
}

/** `resolveLayerView` 的旋钮。 */
export interface LayerViewOptions {
  /**
   * 视角 / 画布倍率。默认 1.2。
   * `> 1` 扩视角（参考内容内嵌中心，四周是原视角外的新内容）；`< 1` 裁中心。
   */
  readonly viewScale?: number
  /** 渲染像素倍率（相对参考图像素）。默认 1。 */
  readonly renderScale?: number
  /** 渲染最长边上限（像素）。默认 1024；`<= 0` 关闭上限。 */
  readonly maxRenderSide?: number
}

/**
 * 由参考视角与旋钮算出渲染视图。
 *
 * 纯函数、确定性：同样的输入一定给同样的 `width/height/focalLengthPx`。
 */
export function resolveLayerView(
  reference: ReferenceView,
  options: LayerViewOptions = {},
): LayerView {
  const { width: refW, height: refH, focalLengthPx: refFocal } = reference
  if (!(refW > 0) || !(refH > 0)) {
    throw new Error(`参考图像尺寸必须为正，收到 ${refW}x${refH}`)
  }
  if (!(refFocal > 0) || !Number.isFinite(refFocal)) {
    throw new Error(`参考像素焦距必须为正，收到 ${refFocal}`)
  }
  const viewScale = options.viewScale ?? DEFAULT_VIEW_SCALE
  if (!(viewScale > 0) || !Number.isFinite(viewScale)) {
    throw new Error(`viewScale 必须为正，收到 ${viewScale}`)
  }
  let pixelScale = options.renderScale ?? 1
  if (!(pixelScale > 0) || !Number.isFinite(pixelScale)) {
    throw new Error(`renderScale 必须为正，收到 ${pixelScale}`)
  }
  const maxRenderSide = options.maxRenderSide ?? DEFAULT_MAX_RENDER_SIDE

  if (maxRenderSide > 0) {
    const longest = Math.max(refW, refH) * viewScale * pixelScale
    if (longest > maxRenderSide) pixelScale *= maxRenderSide / longest
  }

  const width = Math.max(1, Math.round(refW * viewScale * pixelScale))
  const height = Math.max(1, Math.round(refH * viewScale * pixelScale))
  const rectWidth = refW * pixelScale
  const rectHeight = refH * pixelScale
  return {
    width,
    height,
    focalLengthPx: refFocal * pixelScale,
    viewScale,
    pixelScale,
    referenceRect: {
      x: (width - rectWidth) / 2,
      y: (height - rectHeight) / 2,
      width: rectWidth,
      height: rectHeight,
    },
    referenceFocalLengthPx: refFocal,
  }
}

/**
 * 由参考相机 + 视图造渲染相机（只换内参与画布，位姿/近远平面照抄）。
 *
 * 相机轴从 `viewMatrix`（world -> camera，列主序）反解：行 `i` 就是相机轴 `i`
 * 在世界系里的方向（`right` / `down` / `forward`），于是不需要再求一次逆矩阵。
 */
export function expandedLayerCamera(
  reference: WSplatCamera,
  view: LayerView,
): WSplatCamera {
  return createWSplatCamera({
    intrinsics: {
      focalLengthPx: view.focalLengthPx,
      width: view.width,
      height: view.height,
    },
    position: reference.position,
    rotation: cameraRotationRows(reference),
    near: reference.near,
    far: reference.far,
  })
}

/** 从 `viewMatrix` 反解 cam2world 旋转（行主序，三行 = right / down / forward）。 */
export function cameraRotationRows(
  camera: WSplatCamera,
): [
  readonly [number, number, number],
  readonly [number, number, number],
  readonly [number, number, number],
] {
  const v = camera.viewMatrix
  return [
    [v[0], v[4], v[8]],
    [v[1], v[5], v[9]],
    [v[2], v[6], v[10]],
  ]
}

/** 参考图像素（连续坐标，左上原点）-> 渲染画布像素。 */
export function referenceToRenderPixel(
  view: LayerView,
  u: number,
  v: number,
): [number, number] {
  return [u + view.referenceRect.x, v + view.referenceRect.y]
}

/** 渲染画布像素 -> 参考图像素（`referenceToRenderPixel` 的逆）。 */
export function renderToReferencePixel(
  view: LayerView,
  x: number,
  y: number,
): [number, number] {
  return [x - view.referenceRect.x, y - view.referenceRect.y]
}

/**
 * 渲染像素 -> **相机坐标**（OpenCV：x 右、y 下、z 前）。
 *
 * 就是 meshing 反投影的一半：乘上 cam2world 再加相机中心即得世界坐标。
 * `x/y` 用像素**中心**语义（`+0.5`），与 `layer/relief` 的顶点约定一致。
 */
export function renderPixelToCamera(
  view: LayerView,
  x: number,
  y: number,
  z: number,
): [number, number, number] {
  const fx = view.focalLengthPx
  const fy = view.focalLengthPx
  return [
    ((x + 0.5 - view.width / 2) / fx) * z,
    ((y + 0.5 - view.height / 2) / fy) * z,
    z,
  ]
}
