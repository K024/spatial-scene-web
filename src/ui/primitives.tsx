/**
 * 面板 UI 原子组件。
 *
 * 全部是「受控 + 回调」的纯展示组件：**不读信号**，由调用方通过
 * `useValue()` 取值后传进来。这样原子组件可以被任何状态方案复用，
 * 也让 React Compiler 的依赖分析保持在调用方一处。
 */

import { Fragment, type ReactNode } from "react"

/** 面板分组标题 + 内容。 */
export function Section({
  title,
  hint,
  children,
  right,
}: {
  title: string
  hint?: string
  children: ReactNode
  right?: ReactNode
}) {
  return (
    <section className="px-4 py-3.5">
      <header className="mb-2.5 flex items-baseline justify-between gap-2">
        <h2 className="text-[10.5px] font-medium tracking-[0.16em] text-white/45 uppercase">
          {title}
        </h2>
        {right}
      </header>
      <div className="flex flex-col gap-2.5">{children}</div>
      {hint ? (
        <p className="mt-2.5 text-[11px] leading-relaxed text-white/35">
          {hint}
        </p>
      ) : null}
    </section>
  )
}

/**
 * 「标签 + 数值」一行，数值等宽。
 *
 * `wide` 用于长文本（GPU 型号、内参来源这种）：改成上下排布并允许断行，
 * 否则右侧数值的 `shrink-0` 会把面板横向撑开。
 */
export function Row({
  label,
  value,
  sub,
  children,
  wide,
}: {
  label: string
  value?: ReactNode
  sub?: string
  children?: ReactNode
  wide?: boolean
}) {
  if (wide) {
    return (
      <div className="min-w-0">
        <div className="text-[12.5px] leading-tight text-white/70">{label}</div>
        <div className="tnum mt-1 rounded-md border border-white/6 bg-white/3 px-2 py-1.5 text-[11px] leading-snug break-words text-white/80">
          {value}
        </div>
        {sub ? (
          <div className="mt-1 text-[10.5px] leading-tight text-white/30">
            {sub}
          </div>
        ) : null}
      </div>
    )
  }
  return (
    <div className="flex items-center justify-between gap-3">
      <div className="min-w-0">
        <div className="text-[12.5px] leading-tight text-white/70">{label}</div>
        {sub ? (
          <div className="mt-0.5 text-[10.5px] leading-tight text-white/30">
            {sub}
          </div>
        ) : null}
      </div>
      {children ?? (
        <div className="tnum max-w-[62%] shrink-0 text-right text-[12.5px] break-words text-white/90">
          {value}
        </div>
      )}
    </div>
  )
}

/** 滑杆行：标签 + 数值 + range。 */
export function Slider({
  label,
  value,
  min,
  max,
  step = 0.01,
  format,
  onChange,
}: {
  label: string
  value: number
  min: number
  max: number
  step?: number
  format?: (v: number) => string
  onChange: (v: number) => void
}) {
  return (
    <label className="block">
      <div className="flex items-baseline justify-between">
        <span className="text-[12.5px] text-white/70">{label}</span>
        <span className="tnum text-[12px] text-white/90">
          {format ? format(value) : value.toFixed(2)}
        </span>
      </div>
      <input
        type="range"
        className="range mt-1"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.currentTarget.value))}
      />
    </label>
  )
}

/** 开关（胶囊 + 滑块）。 */
export function Toggle({
  label,
  checked,
  hint,
  onChange,
}: {
  label: string
  checked: boolean
  hint?: string
  onChange: (v: boolean) => void
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className="focus-ring flex w-full items-center justify-between gap-3 text-left"
    >
      <span className="min-w-0">
        <span className="block text-[12.5px] leading-tight text-white/70">
          {label}
        </span>
        {hint ? (
          <span className="mt-0.5 block text-[10.5px] leading-tight text-white/30">
            {hint}
          </span>
        ) : null}
      </span>
      <span
        className={`relative h-[18px] w-[32px] shrink-0 rounded-full border transition-colors duration-150 ${
          checked
            ? "border-accent-400/70 bg-accent-500/40"
            : "border-white/12 bg-white/8"
        }`}
      >
        <span
          className={`absolute top-[2px] h-[12px] w-[12px] rounded-full bg-white shadow transition-all duration-150 ${
            checked ? "left-[17px]" : "left-[2px]"
          }`}
        />
      </span>
    </button>
  )
}

/** 分段控件（少量互斥选项）。 */
export function Segmented<T extends string | number>({
  options,
  value,
  disabled,
  onChange,
}: {
  options: { value: T; label: string }[]
  value: T
  disabled?: boolean
  onChange: (v: T) => void
}) {
  return (
    <div
      // shrink-0：按钮是 `flex-1`（flex-basis: 0），一旦被上层压窄，
      // 标签就会被挤到溢出；宁可整个控件不缩，由调用方给足宽度。
      className={`flex shrink-0 gap-0.5 rounded-lg border border-white/10 bg-white/4 p-0.5 ${
        disabled ? "pointer-events-none opacity-40" : ""
      }`}
    >
      {options.map((o) => (
        <button
          key={String(o.value)}
          type="button"
          onClick={() => onChange(o.value)}
          className={`focus-ring flex-1 rounded-[6px] px-2 py-1 text-[11.5px] whitespace-nowrap transition-colors duration-150 ${
            o.value === value
              ? "bg-white/12 text-white shadow-sm"
              : "text-white/55 hover:bg-white/6 hover:text-white/80"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

/** 按钮。 */
export function Button({
  children,
  onClick,
  variant = "ghost",
  disabled,
  title,
}: {
  children: ReactNode
  onClick: () => void
  variant?: "ghost" | "primary" | "warn"
  disabled?: boolean
  title?: string
}) {
  const styles = {
    ghost:
      "border-white/12 bg-white/6 text-white/80 hover:bg-white/10 hover:text-white",
    primary:
      "border-accent-400/40 bg-accent-500/22 text-accent-50 hover:bg-accent-500/32",
    warn: "border-amber-400/40 bg-amber-500/18 text-amber-100 hover:bg-amber-500/28",
  }[variant]
  return (
    <button
      type="button"
      title={title}
      disabled={disabled}
      onClick={onClick}
      className={`focus-ring rounded-lg border px-2.5 py-1.5 text-[12px] font-medium transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-40 ${styles}`}
    >
      {children}
    </button>
  )
}

/** 统计小格：标签在上、数值在下。 */
export function Stat({
  label,
  value,
  unit,
  accent,
}: {
  label: string
  value: string
  unit?: string
  accent?: boolean
}) {
  return (
    <div className="rounded-lg border border-white/6 bg-white/3 px-2.5 py-2">
      <div className="text-[10px] tracking-wider text-white/35 uppercase">
        {label}
      </div>
      <div
        className={`tnum mt-0.5 text-[15px] leading-none ${
          accent ? "text-accent-200" : "text-white/90"
        }`}
      >
        {value}
        {unit ? (
          <span className="ml-0.5 text-[10px] text-white/35">{unit}</span>
        ) : null}
      </div>
    </div>
  )
}

/** 状态点 + 文本。 */
export function Badge({
  tone = "neutral",
  children,
}: {
  tone?: "neutral" | "ok" | "warn" | "busy"
  children: ReactNode
}) {
  const dot = {
    neutral: "bg-white/35",
    ok: "bg-emerald-400",
    warn: "bg-amber-400",
    busy: "bg-accent-400 animate-pulse",
  }[tone]
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-white/5 px-2 py-[3px] text-[10.5px] text-white/60">
      <span className={`h-1.5 w-1.5 rounded-full ${dot}`} />
      {children}
    </span>
  )
}

/**
 * 分位数表格：`行 × { 平均 / p95 / p99 / 最大 }`。
 *
 * 专门为性能面板做的：单列数字看不出「稳不稳」，四列并排才能一眼看出
 * 平均值与尾部的差距（差得越多说明抖动越大）。
 * 数值列用等宽数字 + 右对齐，刷新时不会左右跳。
 */
export function PercentileTable({
  columns,
  rows,
  unit,
}: {
  columns: string[]
  rows: { label: string; values: (string | null)[]; accent?: boolean }[]
  unit: string
}) {
  return (
    <div className="rounded-lg border border-white/[0.07] bg-white/[0.02] px-2.5 py-2">
      <div
        className="grid items-baseline gap-x-2 gap-y-1 text-[11px]"
        style={{ gridTemplateColumns: `auto repeat(${columns.length}, 1fr)` }}
      >
        <span className="text-[10px] tracking-wide text-white/30">{unit}</span>
        {columns.map((c) => (
          <span key={c} className="text-right text-[10px] text-white/30">
            {c}
          </span>
        ))}
        {rows.map((r) => (
          <Fragment key={r.label}>
            <span className="text-white/45">{r.label}</span>
            {r.values.map((v, i) => (
              <span
                key={i}
                className={`text-right tabular-nums ${
                  v === null
                    ? "text-white/20"
                    : r.accent
                      ? "text-accent-300"
                      : "text-white/85"
                }`}
              >
                {v ?? "—"}
              </span>
            ))}
          </Fragment>
        ))}
      </div>
    </div>
  )
}
