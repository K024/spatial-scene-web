/**
 * 面板用的小图标（纯 inline SVG，16×16，`currentColor` 描边）。
 *
 * 不引图标库：只需要十来个、又要与玻璃语言的线宽一致（1.4px），
 * 内联反而更好控制，也不多一份运行时依赖。
 */

import type { SVGProps } from "react"

type IconProps = SVGProps<SVGSVGElement>

function Base({ children, ...props }: IconProps) {
  return (
    <svg
      viewBox="0 0 16 16"
      width={16}
      height={16}
      fill="none"
      stroke="currentColor"
      strokeWidth={1.4}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      {children}
    </svg>
  )
}

/** 网格地面。 */
export function IconGrid(props: IconProps) {
  return (
    <Base {...props}>
      <path d="M2 6.5 8 3.5l6 3-6 3-6-3Z" />
      <path d="M2 9.5l6 3 6-3" />
    </Base>
  )
}

/** 坐标轴。 */
export function IconAxes(props: IconProps) {
  return (
    <Base {...props}>
      <path d="M8 8V2.5" />
      <path d="M8 8 3 12" />
      <path d="M8 8l5 4" />
      <circle cx="8" cy="8" r="1" />
    </Base>
  )
}

/** 线框。 */
export function IconWireframe(props: IconProps) {
  return (
    <Base {...props}>
      <path d="M8 2 14 5.5v5L8 14 2 10.5v-5L8 2Z" />
      <path d="M8 8 2 5.5M8 8l6-2.5M8 8v6" />
    </Base>
  )
}

/** 双面 / 实体。 */
export function IconCube(props: IconProps) {
  return (
    <Base {...props}>
      <path d="M8 2 14 5.5v5L8 14 2 10.5v-5L8 2Z" />
      <path d="M2 5.5 8 9l6-3.5M8 9v5" />
    </Base>
  )
}

/** 眼睛（图层可见 / 单独显示）。 */
export function IconEye(props: IconProps) {
  return (
    <Base {...props}>
      <path d="M1.5 8S3.8 4 8 4s6.5 4 6.5 4-2.3 4-6.5 4S1.5 8 1.5 8Z" />
      <circle cx="8" cy="8" r="1.7" />
    </Base>
  )
}

/** 眼睛带斜线（未单独显示）。 */
export function IconEyeOff(props: IconProps) {
  return (
    <Base {...props}>
      <path d="M1.5 8S3.8 4 8 4c1.1 0 2.1.3 3 .8" />
      <path d="M14.5 8S12.2 12 8 12c-1.1 0-2.1-.3-3-.8" />
      <path d="M2.8 2.8l10.4 10.4" />
    </Base>
  )
}

/** 目标 / 回参考视角。 */
export function IconTarget(props: IconProps) {
  return (
    <Base {...props}>
      <circle cx="8" cy="8" r="5" />
      <path d="M8 1.8v2M8 12.2v2M1.8 8h2M12.2 8h2" />
      <circle cx="8" cy="8" r="1" />
    </Base>
  )
}

/** 文件（选择本地产物）。 */
export function IconFile(props: IconProps) {
  return (
    <Base {...props}>
      <path d="M9 1.8H4.5a1 1 0 0 0-1 1v10.4a1 1 0 0 0 1 1h7a1 1 0 0 0 1-1V5.3L9 1.8Z" />
      <path d="M9 1.8v3.5h3.5" />
    </Base>
  )
}

/** 刷新 / 重新加载。 */
export function IconRefresh(props: IconProps) {
  return (
    <Base {...props}>
      <path d="M13.2 7A5.3 5.3 0 1 1 11.6 3.4" />
      <path d="M13.5 2v3.2h-3.2" />
    </Base>
  )
}

/** 图层（概览用）。 */
export function IconLayers(props: IconProps) {
  return (
    <Base {...props}>
      <path d="M8 2 14 5 8 8 2 5l6-3Z" />
      <path d="M2 8l6 3 6-3" />
      <path d="M2 11l6 3 6-3" />
    </Base>
  )
}
