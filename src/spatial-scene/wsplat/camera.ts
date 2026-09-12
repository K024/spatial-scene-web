/**
 * WSplatCamera：把内参（像素焦距）+ 外参变成渲染用的矩阵与深度约定。
 *
 * ── 约定（逐条不可含糊）──
 * 1. **坐标轴**：相机空间 = OpenCV 约定，`x` 右、`y` **下**、`z` **前**（正）。
 *    这正是 SHARP 的度量空间（`sharp/unproject.ts`：`extrinsics = I`），
 *    相机在原点朝 `+z`。因此**深度 = 视图空间 z，单位米，正值**。
 * 2. **投影是针孔透视**（理由见文末），逐轴焦距 `fx`/`fy`，主点 `(cx, cy)`。
 * 3. **clip 深度**：WebGPU 约定 `z_ndc = 0` 在 near、`= 1` 在 far，`w_clip = z_view`。
 *    这是唯一适合「颜色附件累加深度」的形式：`w` 就是真实深度，
 *    resolve 出来的 `D` 无需任何反演。
 * 4. **NDC y 向上，相机 y 向下**：所以投影的 y 行带负号。
 *    这一点会传到协方差雅可比（见 `wgsl/chunks/gsplatCorner.ts` 的 `modifications`），
 *    漏掉它 2D 高斯的椭圆朝向会在有 off-diagonal 时镜像。
 * 5. 矩阵按 **列主序** 写进 uniform（WGSL `mat4x4f` 是列主序），
 *    所以这里构造的 `Float32Array(16)` 可以直接 `set()` 进 structured view。
 *
 * 本文件**不 import WebGPU**：只产出纯数字，便于 node 侧（CPU 参考）与浏览器共用。
 *
 * ── 为什么是透视，不是正交（定了，别改）──
 * 1. **正交会直接破坏 3DGS 的成像模型**：2D 协方差是透视雅可比的产物
 *    （`J = [[f/z, 0, -f x/z²], [0, f/z, -f y/z²]]`），屏幕足迹随 z 变；
 *    正交下雅可比退化为常数缩放，投影足迹不再随深度收缩 —— 渲染结果就不是该 splat
 *    场所表达的图像。
 * 2. **正交对不上原图**：透视缩短（灭点、近大远小）是单图重建的核心线索，
 *    丢掉就直接失分，画面级验收（vs 原图 NCC/MAE）必然不过。
 * 3. **与下游 mesh 光栅化不匹配**：下游是透视网格光栅化
 *    （`MXIProcessParameters { projection, linearDepth }` + `verticalFoV` / `nearDistance` /
 *    `farDistance` + `InvZ`），若中间表示按正交构造，重投影时尺度与视差全错。
 * 4. **mesh 重建不需要正交**：本链路的 mesh 是 LDI / relief mesh，拓扑就是深度图的像素
 *    网格，顶点 = 该像素按**同一透视内参**反投影的 `(u, v, z)`；没有"在正交深度上
 *    重建几何"这一步。真要做度量均匀采样（TSDF 那类），正确做法是先把透视深度
 *    反投影成点云，而不是用正交去渲染 splat。
 * 广视场（RGBDPano / cube 面）**不是**正交的替代：那些面仍是透视（每面 90° FOV）。
 * 所以硬要求：渲染深度始终是**线性真实度量深度**（视图空间 z，米），NDC / InvZ
 * 只在渲染边界转换；投影矩阵作为输出契约的一部分（见 `types.ts` 的 `WSplatFrame`）。
 */

/** 透视相机内参（像素域）。 */
export interface WSplatCameraIntrinsics {
  /** 像素焦距（原图域）。SHARP 假设方形像素，故 fx = fy。 */
  focalLengthPx: number
  width: number
  height: number
  /** 主点 x（默认图像中心 `width / 2`）。 */
  cx?: number
  /** 主点 y（默认图像中心 `height / 2`）。 */
  cy?: number
}

/** 4x4 矩阵（列主序，16 个元素）。 */
export type Mat4 = Float32Array

/** 相机朝向（世界坐标）。 */
export type Vec3 = readonly [number, number, number]

export interface WSplatCameraOptions {
  intrinsics: WSplatCameraIntrinsics
  /** 相机中心（世界坐标）。默认 `[0,0,0]`（SHARP 推理路径）。 */
  position?: Vec3
  /**
   * cam2world 旋转，**行主序**，三列依次是相机轴 `(right, down, forward)`。
   * 默认单位阵（与 `position=[0,0,0]` 一起即是 SHARP 的原图视角）。
   */
  rotation?: readonly [Vec3, Vec3, Vec3]
  /** 近平面（米，> 0）。 */
  near: number
  /** 远平面（米，> near）。 */
  far: number
}

export interface WSplatCamera {
  readonly width: number
  readonly height: number
  /** 水平视场角（弧度）。 */
  readonly fovX: number
  /** 垂直视场角（弧度）。 */
  readonly fovY: number
  readonly near: number
  readonly far: number
  /** world -> camera，列主序。 */
  readonly viewMatrix: Mat4
  /** camera -> clip（WebGPU 深度域），列主序。 */
  readonly projectionMatrix: Mat4
  /** `(width, height, 1/width, 1/height)`，喂给上游 fork 的 `viewport_size`。 */
  readonly viewportSize: Float32Array
  /** 相机中心（世界坐标）。 */
  readonly position: Vec3
  /** 朝向单位向量（= cam2world 第 3 列）。 */
  readonly forward: Vec3
  /**
   * NDC 深度（WebGPU `[0,1]`）-> 真实视图深度（米）。
   *
   * 这是下游契约里「投影随输出一起导出」那条：下游要做的
   * `NDCDepthToRealDepth` 就用这个式子，用的是**渲染时同一个投影**。
   */
  ndcDepthToViewDepth(ndcDepth: number): number
}

const IDENTITY_ROTATION: readonly [Vec3, Vec3, Vec3] = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
]

/** 建相机（纯函数，不碰 GPU）。 */
export function createWSplatCamera(options: WSplatCameraOptions): WSplatCamera {
  const { intrinsics } = options
  const { width, height } = intrinsics
  if (!(width > 0) || !(height > 0)) {
    throw new Error(`相机尺寸必须为正: ${width}x${height}`)
  }
  if (!(options.near > 0) || !(options.far > options.near)) {
    throw new Error(
      `near/far 必须满足 0 < near < far，收到 ${options.near}/${options.far}`,
    )
  }

  const fx = intrinsics.focalLengthPx
  const fy = intrinsics.focalLengthPx
  const cx = intrinsics.cx ?? width / 2
  const cy = intrinsics.cy ?? height / 2
  const position = options.position ?? [0, 0, 0]
  const rotation = options.rotation ?? IDENTITY_ROTATION
  const [right, down, forward] = rotation
  const { near, far } = options

  // ── world -> camera（列主序）──
  // 行 i = 相机轴 i，平移 = -轴·C
  const view = new Float32Array(16)
  view[0] = right[0]
  view[1] = down[0]
  view[2] = forward[0]
  view[3] = 0
  view[4] = right[1]
  view[5] = down[1]
  view[6] = forward[1]
  view[7] = 0
  view[8] = right[2]
  view[9] = down[2]
  view[10] = forward[2]
  view[11] = 0
  view[12] = -(
    right[0] * position[0] +
    right[1] * position[1] +
    right[2] * position[2]
  )
  view[13] = -(
    down[0] * position[0] +
    down[1] * position[1] +
    down[2] * position[2]
  )
  view[14] = -(
    forward[0] * position[0] +
    forward[1] * position[1] +
    forward[2] * position[2]
  )
  view[15] = 1

  // ── camera -> clip（列主序，WebGPU 深度域）──
  // 行主序形式：
  //   [ 2fx/W    0        0             0           ]
  //   [ 0       -2fy/H    0             0           ]
  //   [ 0        0        f/(f-n)       -f·n/(f-n)  ]
  //   [ 0        0        1             0           ]
  // 主点偏移先不在矩阵里做（渲染视口就是图像域，`cx=W/2`），
  // 若将来要支持非居中主点，正确做法是给视图矩阵补一个平移，而不是改投影（保持
  // `w = z_view` 这一条无损）。
  if (cx !== width / 2 || cy !== height / 2) {
    throw new Error(
      "暂不支持非居中主点（请先把主点偏移并入外参/裁剪窗口）；" +
        `收到 cx=${cx}, cy=${cy}，图像 ${width}x${height}`,
    )
  }
  const proj = new Float32Array(16)
  const zRange = far - near
  proj[0] = (2 * fx) / width
  proj[5] = (-2 * fy) / height
  proj[10] = far / zRange
  proj[11] = 1
  proj[14] = (-far * near) / zRange
  proj[15] = 0

  return {
    width,
    height,
    fovX: 2 * Math.atan(width / (2 * fx)),
    fovY: 2 * Math.atan(height / (2 * fy)),
    near,
    far,
    viewMatrix: view,
    projectionMatrix: proj,
    viewportSize: new Float32Array([width, height, 1 / width, 1 / height]),
    position,
    forward,
    ndcDepthToViewDepth(ndcDepth: number): number {
      return (near * far) / (far - ndcDepth * zRange)
    },
  }
}

/** 视图空间 z -> NDC 深度（`ndcDepthToViewDepth` 的逆）。 */
export function viewDepthToNdcDepth(
  camera: WSplatCamera,
  viewZ: number,
): number {
  return (camera.far / (camera.far - camera.near)) * (1 - camera.near / viewZ)
}
