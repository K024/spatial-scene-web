/**
 * 轨道相机（围绕枢轴旋转，默认停在**参考相机**位姿）。
 *
 * 为什么用「枢轴 + 球坐标」而不是自由飞行：本项目要测的正是
 * 「参考视角附近的小幅观察」，球坐标天然把运动限制在球面上，
 * 距离枢轴的距离不变，视角不会跑飞。
 *
 * 关键约定：
 * - 输入输出的位置/朝向都在 **PLY 系**（x 右 / y 下 / z 前），
 *   与 `spatial-scene/export/camera.ts` 及排序用的是同一套坐标。
 * - 渲染世界系（y 上、朝 -z）只存在于 `view` 矩阵内部：
 *   `view = lookAt(eye, pivot, +y) · diag(1,-1,-1)`。
 *   那个翻转是一次真旋转（det=+1），所以不会引入镜像。
 * - 投影由**像素焦距**给出（不是 fovY）：`f_eff = focalPx · contain(视口, 原图)`，
 *   于是不同宽高比的视口下，渲染内容的取景范围与原照片一致。
 */

import {
  length,
  lookAt,
  mat4,
  mat4Multiply,
  normalize,
  perspectiveFromFocal,
  subtract,
  type Vec3,
} from "./math.ts"

/** PLY 系 -> 渲染世界系的旋转 `diag(1,-1,-1)`（列主序）。 */
const FLIP = new Float32Array([
  1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1,
])

/** 相机的可调参数（存于 store，渲染器每帧读取）。 */
export interface CameraState {
  /** 轨道枢轴（PLY 系）。取场景中心，保证默认视角能看到全貌。 */
  pivot: Vec3
  /** 水平角（弧度）。0 = 参考视角。 */
  yaw: number
  /** 俯仰角（弧度）。0 = 参考视角。 */
  pitch: number
  /** 相机到枢轴的距离。默认 = `|pivot|`，即相机恰好落在参考相机位置。 */
  distance: number
  /** 参考相机原图域焦距（px）。 */
  focalPx: number
  /** 原图尺寸 `[width, height]`。 */
  imageSize: [number, number]
  near: number
  far: number
}

/** 每帧由 `CameraState` 算出的矩阵/基向量。 */
export class RenderCamera {
  readonly view = mat4()
  readonly proj = mat4()
  private readonly viewNoFlip = mat4()

  /** 相机位置（渲染世界系 / PLY 系）。 */
  eyeWorld: Vec3 = [0, 0, 0]
  eyePly: Vec3 = [0, 0, 0]
  /** 视线方向（单位向量）。 */
  forwardWorld: Vec3 = [0, 0, -1]
  forwardPly: Vec3 = [0, 0, 1]
  /** 视口像素域的实际焦距（含 contain 缩放）。 */
  focalScaled = 1
  viewportWidth = 1
  viewportHeight = 1
  /** 原图在视口里的缩放比（= focalScaled / focalPx）。 */
  fitScale = 1

  /**
   * 更新矩阵。
   *
   * @param s 相机状态（PLY 系）。
   * @param width 视口宽（像素，已含 DPR 与分辨率倍数）。
   * @param height 视口高。
   */
  update(s: CameraState, width: number, height: number): void {
    this.viewportWidth = Math.max(1, width)
    this.viewportHeight = Math.max(1, height)

    // 1) 枢轴换到渲染世界系（FLIP 是自逆的）
    const pivot: Vec3 = [s.pivot[0], -s.pivot[1], -s.pivot[2]]

    // 2) 球坐标 -> 相机位置。yaw=0,pitch=0 时 dir=(0,0,1)，
    //    即 eye = pivot + (0,0,|pivot|) = 原点（参考相机位置）。
    const cp = Math.cos(s.pitch)
    const sp = Math.sin(s.pitch)
    const cy = Math.cos(s.yaw)
    const sy = Math.sin(s.yaw)
    const dir: Vec3 = [sy * cp, sp, cy * cp]
    const eye: Vec3 = [
      pivot[0] + dir[0] * s.distance,
      pivot[1] + dir[1] * s.distance,
      pivot[2] + dir[2] * s.distance,
    ]

    this.eyeWorld = eye
    this.eyePly = [eye[0], -eye[1], -eye[2]]
    this.forwardWorld = normalize(subtract(pivot, eye))
    this.forwardPly = [
      this.forwardWorld[0],
      -this.forwardWorld[1],
      -this.forwardWorld[2],
    ]

    // 3) 视图矩阵：世界系 lookAt 之后再折进 PLY->世界 的翻转
    lookAt(this.viewNoFlip, eye, pivot, [0, 1, 0])
    mat4Multiply(this.view, this.viewNoFlip, FLIP)

    // 4) 投影：contain 缩放，保证原图整幅可见且不拉伸
    const [iw, ih] = s.imageSize
    this.fitScale = Math.min(this.viewportWidth / iw, this.viewportHeight / ih)
    this.focalScaled = s.focalPx * this.fitScale
    perspectiveFromFocal(
      this.proj,
      this.focalScaled,
      this.viewportWidth,
      this.viewportHeight,
      s.near,
      s.far,
    )
  }

  /** 相机到参考相机位置（世界原点）的距离，用于面板显示。 */
  get distanceFromReference(): number {
    return length(this.eyeWorld)
  }
}
