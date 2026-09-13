/**
 * 面板 / HUD 的共享小部件。
 *
 * ── 样式口径 ──
 * 毛玻璃面板是**亮底**（`bg-white/10` + `backdrop-blur`），所以文字用白色系；
 * 交互控件一律「FlyOnUI 结构类（`btn` / `switch` / `badge`）+ Tailwind 覆盖色」的写法：
 * FlyOnUI 提供形态与焦点态，`bg-white/10` 这类工具类在 utilities 层覆盖它的配色
 * （Tailwind v4 层序 `components < utilities`），于是组件在当前深色底上不违和。
 */

import type { ButtonHTMLAttributes, ReactNode } from "react"

/**
 * 玻璃卡片基座（面板本身与浮层共用）。
 *
 * ── 为什么要有暗色底，而不是只靠 `backdrop-blur` ──
 * `backdrop-filter` 只模糊、**不保证对比度**：场景亮部（天空 / 白墙）透上来会把白字
 * 吃穿，暗部又会让面板和画布糊成一片。所以压一层**半透明近黑**（不是纯黑，带一点背景
 * 的冷蓝）把对比度下限锁死：
 * - 60% 是实测的平衡点：亮部透上来仍能看出「后面有东西」，白字（85% / 90%）稳妥可读；
 * - 再配 `backdrop-blur-2xl` + `backdrop-saturate-150` 保留毛玻璃质感；
 * - 边框用白色低透明度：在深底上是「玻璃边缘」，不是描边。
 * 结论：正文保持亮色系，不改成「亮底深字」——后者在深色画布上会显得很跳。
 */
export const GLASS =
  "border border-white/12 bg-[#0a0c12]/62 shadow-[0_24px_70px_-24px_rgba(0,0,0,0.9)] backdrop-blur-2xl backdrop-saturate-150"

/** 主标题。 */
export function Section({
  title,
  icon,
  aside,
  children,
}: {
  title: string
  icon?: ReactNode
  aside?: ReactNode
  children: ReactNode
}) {
  return (
    <section className="px-3.5 py-3">
      <header className="mb-2 flex items-center gap-2">
        {icon ? <span className="text-white/45">{icon}</span> : null}
        <h2 className="flex-1 text-[11px] font-semibold tracking-[0.14em] text-white/45 uppercase">
          {title}
        </h2>
        {aside}
      </header>
      {children}
    </section>
  )
}

/** 面板分节之间的分隔线。 */
export function Divider() {
  return <div className="mx-3.5 border-t border-white/10" />
}

/** 图标按钮（FlyOnUI `btn` 形态 + 玻璃配色）。 */
export function IconButton({
  label,
  active,
  size = "sm",
  className,
  children,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  label: string
  active?: boolean
  size?: "xs" | "sm"
}) {
  return (
    <button
      type="button"
      title={label}
      aria-label={label}
      aria-pressed={active}
      className={[
        `btn btn-${size} btn-circle border-white/10 text-white/75 hover:border-white/20`,
        active ? "bg-white/25 text-white" : "bg-white/8 hover:bg-white/16",
        className ?? "",
      ].join(" ")}
      {...props}
    >
      {children}
    </button>
  )
}

/** 文字按钮。 */
export function TextButton({
  className,
  children,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      type="button"
      className={[
        "btn btn-xs border-white/10 bg-white/8 text-white/80 hover:border-white/20 hover:bg-white/16",
        className ?? "",
      ].join(" ")}
      {...props}
    >
      {children}
    </button>
  )
}

/** `label / value` 定义行。 */
export function Field({
  label,
  value,
  mono,
}: {
  label: string
  value: ReactNode
  mono?: boolean
}) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-0.5">
      <dt className="shrink-0 text-[11px] text-white/40">{label}</dt>
      <dd
        className={[
          "truncate text-right text-[11.5px] text-white/80",
          mono ? "font-mono tabular-nums" : "",
        ].join(" ")}
      >
        {value}
      </dd>
    </div>
  )
}

/** FlyOnUI 开关（`switch`）。 */
export function Switch({
  checked,
  onChange,
  label,
  size = "sm",
}: {
  checked: boolean
  onChange: (checked: boolean) => void
  label: string
  size?: "xs" | "sm" | "md"
}) {
  return (
    <input
      type="checkbox"
      role="switch"
      aria-label={label}
      className={`switch switch-${size} switch-primary shrink-0 cursor-pointer`}
      checked={checked}
      onChange={(event) => onChange(event.currentTarget.checked)}
    />
  )
}
