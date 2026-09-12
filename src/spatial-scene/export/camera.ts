/**
 * 相机参数导出（SuperSplat 可导入的 INRIA `cameras.json` 格式）。
 *
 * ── 为什么需要它 ──
 * 把 PLY 拖进 https://superspl.at/editor 后，编辑器会把相机放在场景外自动取景，
 * 看到的是「环绕观察」而不是「拍摄时真正的视角」。而 SHARP 的度量空间**本身就是
 * 相机坐标系**（推理路径里 `extrinsics = I`，见 `sharp/unproject.ts`），所以只要把
 * 这个视角的位姿写成 SuperSplat 能吃的 json，就能一键回到原图视角——这既是查看手段，
 * 也是验证 PLY 是否与输入图对齐的最直接办法。
 *
 * ── 坐标系（逐条列出，极易搞错）──
 * 1. PLY 顶点 = SHARP 度量空间：x 右、y **下**、z 前（相机在原点朝 +z）。
 * 2. SuperSplat 载入 PLY 时会对 splat 整体施加 `Rz(180°)`
 *    （= splat-transform 的 `Transform.PLY`，见其 `src/lib/utils/math.ts`），
 *    因此渲染世界坐标 = `(-x, -y, z)`。
 * 3. SuperSplat 的 json 相机导入器对读到的位姿做**同样**的 `(-x, -y, z)`
 *    （`src/file-handler.ts: loadCameraPoses`）。两者相互抵消 ⇒
 *    **json 里的 position / rotation 必须写在 PLY 坐标系里**，
 *    与 PLY 顶点同一套坐标，而不是渲染世界坐标。
 * 4. 导入器把 `rotation` 的第 3 列当作相机朝向（`target = position + 10 * col2`），
 *    即 `rotation` 是 **cam2world**（对应 INRIA `convert.py` 里写出的 `Rᵗ`）。
 * 5. `fov` 由导入器按 `max(fovX, fovY)` 反推，需要 fx / fy / width / height 齐全；
 *    且 SuperSplat 把 fov 作用在视口**较长**的那根轴上
 *    （`src/camera.ts`: `camera.horizontalFov = width > height`）。
 *    ⇒ 只有「视口宽高比 == 原图宽高比」时两轴都精确；否则较长轴精确、另一轴随视口变化。
 *    注意导入器取的是图像尺寸（本 json 里的 width/height），与视口无关。
 *
 * 结论：对单图推理（extrinsics = I），位姿就是 `position=[0,0,0]`、`rotation=I`，
 * 朝向 +z；再用原始 `f_px` 与原始图像尺寸给出 fov。渲染出的画面应当与输入图逐像素对齐
 * （见 `scripts/check-camera.ts` 的离线渲染校验）。
 */

/** `[x, y, z]`。 */
export type Vec3Tuple = [number, number, number]

/** 3x3 行主序矩阵（3 行）。 */
export type Mat3Tuple = [Vec3Tuple, Vec3Tuple, Vec3Tuple]

/**
 * SuperSplat（INRIA 格式）的一条相机位姿。
 *
 * 键名与 `file-handler.ts: loadCameraPoses` 的读法一一对应：
 * 必填 `position` / `rotation`；`fx`/`fy`/`width`/`height` 齐全时才会算 fov；
 * `id` / `img_name` 只用于排序与命名。
 */
export interface SuperSplatCameraPose {
  id: number
  img_name: string
  /** 图像宽（原图域，用于反推 fov）。 */
  width: number
  /** 图像高（原图域）。 */
  height: number
  /** 相机中心，PLY 坐标系。 */
  position: Vec3Tuple
  /** cam2world 旋转，行主序；第 3 列 = 朝向。 */
  rotation: Mat3Tuple
  /** 水平方向像素焦距。 */
  fx: number
  /** 垂直方向像素焦距（与 fx 相同：SHARP 假设方形像素）。 */
  fy: number
}

/** 相机位姿（统一放在 PLY 坐标系里表达）。 */
export interface CameraPose {
  /** 相机中心 `C = -Rᵗ·t`。 */
  position: Vec3Tuple
  /** cam2world 旋转（行主序），第 3 列 = 朝向。 */
  rotation: Mat3Tuple
  /** 朝向单位向量（= rotation 第 3 列），便于打 target 日志。 */
  forward: Vec3Tuple
}

/**
 * 由 4x4 行主序 extrinsics（**world -> camera**，ml-sharp 语义
 * `X_cam = R·X_ply + t`）求出相机在 PLY 坐标系里的位姿。
 *
 * 这里之所以从 extrinsics 走一遍而不是直接写死 `[0,0,0] + I`：
 * 推理路径目前恒为 `extrinsics = I`，但一旦将来支持多视角/位姿条件，
 * 导出逻辑不用改。
 */
export function cameraPoseFromExtrinsics(
  extrinsics: ArrayLike<number>,
): CameraPose {
  // R = extrinsics[:3,:3]（行主序），t = extrinsics[:3,3]
  const e = (r: number, c: number): number => extrinsics[r * 4 + c]
  const t: Vec3Tuple = [e(0, 3), e(1, 3), e(2, 3)]

  // cam2world = Rᵗ：第 i 行 = R 的第 i 列
  const rotation: Mat3Tuple = [
    [e(0, 0), e(1, 0), e(2, 0)],
    [e(0, 1), e(1, 1), e(2, 1)],
    [e(0, 2), e(1, 2), e(2, 2)],
  ]

  // 相机中心 C = -Rᵗ·t
  const position: Vec3Tuple = [
    -(rotation[0][0] * t[0] + rotation[0][1] * t[1] + rotation[0][2] * t[2]),
    -(rotation[1][0] * t[0] + rotation[1][1] * t[1] + rotation[1][2] * t[2]),
    -(rotation[2][0] * t[0] + rotation[2][1] * t[1] + rotation[2][2] * t[2]),
  ]

  return { position, rotation, forward: rotation[2] }
}

/**
 * 组装一条 SuperSplat 位姿。
 *
 * @param name 位姿名（导入后显示在 pose 列表里，一般用输入图文件名）。
 * @param focalLengthPx **原图域**像素焦距（即 `SceneMetaData.focalLengthPx`）。
 * @param imageShape 原图尺寸 `[width, height]`。
 * @param extrinsics 4x4 行主序 world->camera；缺省为单位阵（SHARP 推理路径）。
 * @param id 排序用 id；单张图给 0。
 */
export function superSplatCameraPose(opts: {
  name: string
  focalLengthPx: number
  imageShape: [number, number]
  extrinsics?: ArrayLike<number>
  id?: number
}): SuperSplatCameraPose {
  const pose = cameraPoseFromExtrinsics(opts.extrinsics ?? IDENTITY4)
  return {
    id: opts.id ?? 0,
    img_name: opts.name,
    width: opts.imageShape[0],
    height: opts.imageShape[1],
    position: pose.position,
    rotation: pose.rotation,
    fx: opts.focalLengthPx,
    fy: opts.focalLengthPx,
  }
}

/** 单位阵（4x4 行主序），避免 import 依赖循环。 */
const IDENTITY4 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

/**
 * 由像素焦距与对应方向的像素尺寸求 FOV（度）。
 *
 * `2·atan(size / (2·f))`。SuperSplat 导入时用的就是这个式子，
 * 这里复制一份用于日志核对（保持两处口径一致）。
 */
export function fovDeg(fPx: number, sizePx: number): number {
  return (2 * Math.atan(sizePx / (2 * fPx)) * 180) / Math.PI
}

/**
 * 序列化为 SuperSplat 可直接导入的 json 文本。
 *
 * 导入器接受**数组**（一个元素 = 一条位姿）；单张图就只有一条，
 * SuperSplat 会把它当作常驻 key（timeline 上任何帧都保持这个位姿），
 * 因此拖入后相机立即跳到原图视角。
 *
 * 手写而不是 `JSON.stringify`：向量/矩阵保持单行，sidecar 文件要能一眼看清。
 */
export function superSplatCameraJson(poses: SuperSplatCameraPose[]): string {
  const entries = poses.map((p) => {
    const rows = p.rotation.map((r) => `[${r.map(num).join(", ")}]`).join(", ")
    return [
      "  {",
      `    "id": ${p.id},`,
      `    "img_name": ${JSON.stringify(p.img_name)},`,
      `    "width": ${p.width},`,
      `    "height": ${p.height},`,
      `    "position": [${p.position.map(num).join(", ")}],`,
      `    "rotation": [${rows}],`,
      `    "fx": ${num(p.fx)},`,
      `    "fy": ${num(p.fy)}`,
      "  }",
    ].join("\n")
  })
  return `[\n${entries.join(",\n")}\n]\n`
}

/** 数值 -> json 字面量（保持全部有效位，便于逐位复现）。 */
function num(v: number): string {
  return Number.isFinite(v) ? String(v) : "0"
}

/**
 * 由位姿与朝向距离求 target（仅用于日志/核对，写进 json 的是 position + rotation）。
 *
 * SuperSplat 导入器内部固定 `target = position + 10 * col2`，这里用同样的
 * 10 单位距离，方便把日志里的 target 与编辑器里看到的对起来。
 */
export function poseTarget(
  pose: { position: Vec3Tuple; forward: Vec3Tuple },
  distance = 10,
): Vec3Tuple {
  return [
    pose.position[0] + pose.forward[0] * distance,
    pose.position[1] + pose.forward[1] * distance,
    pose.position[2] + pose.forward[2] * distance,
  ]
}
