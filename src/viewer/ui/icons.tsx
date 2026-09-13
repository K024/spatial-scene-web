/**
 * 内联图标（24×24 线性 SVG）。
 *
 * 不引图标库：这里只用得到十来个，且都是描边风格；`currentColor` + `strokeWidth`
 * 统一在基座上，尺寸由调用方的 `className` 决定（Tailwind 的 `size-4` 等）。
 */

import type { SVGProps } from "react"

type IconProps = SVGProps<SVGSVGElement>

function Icon({ children, ...props }: IconProps) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.7}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      {children}
    </svg>
  )
}

export function IconLayers(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="m12 3 9 5-9 5-9-5 9-5Z" />
      <path d="m3 13 9 5 9-5" />
    </Icon>
  )
}

export function IconEye(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12Z" />
      <circle cx="12" cy="12" r="2.8" />
    </Icon>
  )
}

export function IconEyeOff(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M4 4l16 16" />
      <path d="M9.6 5.9A9.6 9.6 0 0 1 12 5.5c6 0 9.5 6.5 9.5 6.5a17 17 0 0 1-3.3 4" />
      <path d="M6.3 7.6A17 17 0 0 0 2.5 12S6 18.5 12 18.5a9.4 9.4 0 0 0 3.6-.7" />
      <path d="M9.9 9.9a2.9 2.9 0 0 0 4.1 4.1" />
    </Icon>
  )
}

export function IconSolo(props: IconProps) {
  return (
    <Icon {...props}>
      <circle cx="12" cy="12" r="3" />
      <path d="M12 2v3M12 19v3M2 12h3M19 12h3" />
    </Icon>
  )
}

export function IconGrid(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M3 9h18M3 15h18M9 3v18M15 3v18" />
      <rect x="3" y="3" width="18" height="18" rx="2" />
    </Icon>
  )
}

export function IconAxis(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M12 21V9" />
      <path d="M12 9 5 4.5M12 9l7-4.5" />
      <path d="M3 21h18" />
    </Icon>
  )
}

export function IconRotate(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M20 12a8 8 0 1 1-2.6-5.9" />
      <path d="M20 4v5h-5" />
    </Icon>
  )
}

export function IconTarget(props: IconProps) {
  return (
    <Icon {...props}>
      <circle cx="12" cy="12" r="8" />
      <path d="M12 2v3M12 19v3M2 12h3M19 12h3" />
      <circle cx="12" cy="12" r="1.6" fill="currentColor" stroke="none" />
    </Icon>
  )
}

export function IconUpload(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M12 16V4" />
      <path d="m7.5 8.5 4.5-4.5 4.5 4.5" />
      <path d="M4 15v3.5A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5V15" />
    </Icon>
  )
}

export function IconSparkles(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M12 3.5 13.6 8 18 9.6 13.6 11.2 12 15.7 10.4 11.2 6 9.6 10.4 8 12 3.5Z" />
      <path d="M18.5 15.5l.7 2 2 .7-2 .7-.7 2-.7-2-2-.7 2-.7.7-2Z" />
    </Icon>
  )
}

export function IconChevronRight(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="m9 5 7 7-7 7" />
    </Icon>
  )
}

export function IconClose(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M6 6l12 12M18 6 6 18" />
    </Icon>
  )
}

export function IconInfo(props: IconProps) {
  return (
    <Icon {...props}>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 11v5" />
      <circle cx="12" cy="8" r="0.9" fill="currentColor" stroke="none" />
    </Icon>
  )
}

export function IconSliders(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M4 7h10M18 7h2M4 17h4M12 17h8" />
      <circle cx="16" cy="7" r="2" />
      <circle cx="10" cy="17" r="2" />
    </Icon>
  )
}

/** 线框 / 几何调试视图：外三角形 + 中点剖分出的内三角形。 */
export function IconWireframe(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M12 3.5 20.5 19h-17z" />
      <path d="M7.75 11.25h8.5" />
      <path d="M7.75 11.25 12 19" />
      <path d="M16.25 11.25 12 19" />
    </Icon>
  )
}
