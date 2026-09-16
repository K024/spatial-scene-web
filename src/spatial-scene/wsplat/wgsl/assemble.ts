/**
 * WGSL 组装器：`#include` 展开 + `#if` 预处理器 + `{DEFINE}` 文本替换 + 编译缓存。
 *
 * 为什么自己写而不是拿现成的：上游 playcanvas 的 chunk 是 `#include "gsplatCornerVS"`
 * 这种**名字寻址**（不带路径、不带扩展名）的字符串拼接，而 Vite / 浏览器都没有
 * 原生的 WGSL 预处理器。fork 上游 chunk 时要保持逐行可对照，所以这里实现的是
 * 上游那套语义（`#include` 名字 / `#if` 常量表达式 / `#ifdef` / `#elif` / `#else` /
 * `#endif`），而不是发明新语法。
 *
 * 与上游的差异（有意为之，为了让 fork 的文件不必依赖上游的路径/命名）：
 * - `#include` 的解析目标由调用方以 `WgslUnit[]` 传入，不读文件系统（`src/` 内不碰 node 内置模块）。
 * - 不认识的 `#` 指令**报错**而不是静默透传，避免上游新指令被悄悄吞掉。
 *
 * 纯函数、无副作用（只读全局缓存），可在 web / node 两侧运行。
 */

/** 可用作 define 的值。数字会被当作 C 常量参与 `#if` 求值。 */
export type DefineValue = string | number | boolean

/** define 表。`undefined` 等价于「未定义」。 */
export type DefineMap = Record<string, DefineValue | undefined>

/** 一个 WGSL 单元（入口或 chunk）。`name` 即 `#include "name"` 里的名字。 */
export interface WgslUnit {
  readonly name: string
  readonly code: string
}

/** 组装失败（循环 include / 类型错误 / 未知指令）。 */
export class WgslAssembleError extends Error {
  constructor(message: string) {
    super(message)
    this.name = "WgslAssembleError"
  }
}

// ────────────────────────────── 表达式求值 ──────────────────────────────

interface Token {
  kind: "num" | "ident" | "op"
  text: string
  value: number
}

const OPERATORS = [
  // 长的在前，保证最长匹配
  "||",
  "&&",
  "<<",
  ">>",
  "<=",
  ">=",
  "==",
  "!=",
  "|",
  "^",
  "&",
  "<",
  ">",
  "+",
  "-",
  "*",
  "/",
  "%",
  "!",
  "~",
  "(",
  ")",
]

function isIdentStart(ch: string): boolean {
  return /[A-Za-z_]/.test(ch)
}

function isIdentPart(ch: string): boolean {
  return /[A-Za-z0-9_]/.test(ch)
}

function isDigit(ch: string): boolean {
  return ch >= "0" && ch <= "9"
}

function tokenize(expr: string): Token[] {
  const out: Token[] = []
  let i = 0
  while (i < expr.length) {
    const ch = expr[i]
    if (/\s/.test(ch)) {
      i++
      continue
    }
    if (isDigit(ch) || (ch === "." && isDigit(expr[i + 1] ?? ""))) {
      let j = i
      while (j < expr.length && /[0-9a-fA-FxX.]/.test(expr[j])) j++
      // 吃掉整数后缀 u/U 与浮点后缀 f/F/h/H
      while (j < expr.length && /[uUfFhH]/.test(expr[j])) j++
      const text = expr.slice(i, j)
      const value = Number.parseInt(
        text.replace(/[uUfFhH]$/, ""),
        text.startsWith("0x") || text.startsWith("0X") ? 16 : 10,
      )
      out.push({ kind: "num", text, value: Number.isNaN(value) ? 0 : value })
      i = j
      continue
    }
    if (isIdentStart(ch)) {
      let j = i
      while (j < expr.length && isIdentPart(expr[j])) j++
      out.push({ kind: "ident", text: expr.slice(i, j), value: 0 })
      i = j
      continue
    }
    const op = OPERATORS.find((o) => expr.startsWith(o, i))
    if (!op) {
      throw new WgslAssembleError(
        `无法识别的表达式字符: ${JSON.stringify(ch)} @ ${expr}`,
      )
    }
    out.push({ kind: "op", text: op, value: 0 })
    i += op.length
  }
  return out
}

/** 优先级（数字越大越紧）。 */
const PRECEDENCE: Record<string, number> = {
  "||": 1,
  "&&": 2,
  "|": 3,
  "^": 4,
  "&": 5,
  "==": 6,
  "!=": 6,
  "<": 7,
  "<=": 7,
  ">": 7,
  ">=": 7,
  "<<": 8,
  ">>": 8,
  "+": 9,
  "-": 9,
  "*": 10,
  "/": 10,
  "%": 10,
}

/** 供预处理器使用的常量表达式求值；非 0 即真。未定义的标识符按 0 处理（与 C 一致）。 */
function evaluate(expr: string, defines: DefineMap): number {
  const tokens = tokenize(expr)
  let pos = 0

  const peek = (): Token | undefined => tokens[pos]
  const next = (): Token => {
    const t = tokens[pos]
    if (!t) throw new WgslAssembleError(`表达式提前结束: ${expr}`)
    pos++
    return t
  }

  /** 解析 `defined(X)` / `defined X`。 */
  const parseDefined = (): number => {
    const named = next()
    if (named.kind === "op" && named.text === "(") {
      const name = next()
      if (name.kind !== "ident")
        throw new WgslAssembleError(`defined(...) 里必须是标识符: ${expr}`)
      const close = next()
      if (!(close.kind === "op" && close.text === ")")) {
        throw new WgslAssembleError(`defined(...) 缺少右括号: ${expr}`)
      }
      return defines[name.text] === undefined ? 0 : 1
    }
    if (named.kind !== "ident")
      throw new WgslAssembleError(`defined 后面必须是标识符: ${expr}`)
    return defines[named.text] === undefined ? 0 : 1
  }

  const parseUnary = (): number => {
    const t = next()
    if (t.kind === "num") return t.value
    if (t.kind === "ident") {
      if (t.text === "defined") return parseDefined()
      if (t.text === "true") return 1
      if (t.text === "false") return 0
      const v = defines[t.text]
      if (v === undefined) return 0
      if (typeof v === "boolean") return v ? 1 : 0
      if (typeof v === "number") return v
      const n = Number.parseFloat(v)
      return Number.isNaN(n) ? 0 : n
    }
    if (t.text === "(") {
      const v = parseBinary(0)
      const close = next()
      if (!(close.kind === "op" && close.text === ")")) {
        throw new WgslAssembleError(`缺少右括号: ${expr}`)
      }
      return v
    }
    if (t.text === "!" || t.text === "-" || t.text === "+" || t.text === "~") {
      const v = parseUnary()
      if (t.text === "!") return v === 0 ? 1 : 0
      if (t.text === "-") return -v
      if (t.text === "+") return v
      return ~v
    }
    throw new WgslAssembleError(
      `表达式里出现意外记号 ${JSON.stringify(t.text)}: ${expr}`,
    )
  }

  const parseBinary = (minPrec: number): number => {
    let lhs = parseUnary()
    for (;;) {
      const t = peek()
      if (!t || t.kind !== "op") break
      const prec = PRECEDENCE[t.text]
      if (prec === undefined || prec < minPrec) break
      next()
      const rhs = parseBinary(prec + 1)
      lhs = applyBinary(t.text, lhs, rhs)
    }
    return lhs
  }

  const value = parseBinary(0)
  if (pos !== tokens.length) {
    throw new WgslAssembleError(
      `表达式尾部有多余记号（${tokens[pos]?.text}）: ${expr}`,
    )
  }
  return value
}

function applyBinary(op: string, a: number, b: number): number {
  switch (op) {
    case "||":
      return a !== 0 || b !== 0 ? 1 : 0
    case "&&":
      return a !== 0 && b !== 0 ? 1 : 0
    case "|":
      return a | b
    case "^":
      return a ^ b
    case "&":
      return a & b
    case "==":
      return a === b ? 1 : 0
    case "!=":
      return a !== b ? 1 : 0
    case "<":
      return a < b ? 1 : 0
    case "<=":
      return a <= b ? 1 : 0
    case ">":
      return a > b ? 1 : 0
    case ">=":
      return a >= b ? 1 : 0
    case "<<":
      return a << b
    case ">>":
      return a >> b
    case "+":
      return a + b
    case "-":
      return a - b
    case "*":
      return a * b
    case "/":
      return b === 0 ? 0 : Math.trunc(a / b)
    case "%":
      return b === 0 ? 0 : a % b
    default:
      throw new WgslAssembleError(`未知运算符 ${op}`)
  }
}

// ────────────────────────────── 组装 ──────────────────────────────

interface Conditional {
  /** 外层是否激活。 */
  parentActive: boolean
  /** 本层的 `#if` 是否成立。 */
  conditionTrue: boolean
  /** 本层是否已经有一个分支被取用过（用于 `#elif` / `#else`）。 */
  taken: boolean
  /** 当前是否在输出。 */
  active: boolean
}

const INCLUDE_RE = /^\s*#include\s+"([^"]+)"\s*$/
const DEFINE_SUB_RE = /\{([A-Za-z_][A-Za-z0-9_]*)\}/g

/**
 * 把入口单元与其依赖的 chunk 组装成最终 WGSL 源码。
 *
 * @param entry   入口单元（自身也参与 `#include` 解析，可被别的单元 include）。
 * @param chunks  所有可被 include 的 chunk（名字唯一）。
 * @param defines 预处理 define；同时用于 `{NAME}` 文本替换。
 */
export function assembleWgsl(
  entry: WgslUnit,
  chunks: readonly WgslUnit[] = [],
  defines: DefineMap = {},
): string {
  const registry = new Map<string, WgslUnit>()
  for (const c of chunks) {
    if (registry.has(c.name)) {
      throw new WgslAssembleError(`chunk 名字重复: ${c.name}`)
    }
    registry.set(c.name, c)
  }
  registry.set(entry.name, entry)

  const cacheKey = `${entry.name}|${cacheKeyOfDefines(defines)}|${chunks
    .map((c) => c.name)
    .sort()
    .join(",")}`
  const cached = CACHE.get(cacheKey)
  if (cached !== undefined) return cached

  const text = expand(entry, registry, defines, [entry.name])
  CACHE.set(cacheKey, text)
  return text
}

const CACHE = new Map<string, string>()

function cacheKeyOfDefines(defines: DefineMap): string {
  return Object.keys(defines)
    .filter((k) => defines[k] !== undefined)
    .sort()
    .map((k) => `${k}=${String(defines[k])}`)
    .join(";")
}

function expand(
  unit: WgslUnit,
  registry: ReadonlyMap<string, WgslUnit>,
  defines: DefineMap,
  stack: string[],
): string {
  const lines = unit.code.split("\n")
  const out: string[] = []
  const conditionals: Conditional[] = []

  const isActive = (): boolean =>
    conditionals.length === 0 || conditionals[conditionals.length - 1].active

  for (const rawLine of lines) {
    const line = rawLine

    // ── #include ──
    const inc = INCLUDE_RE.exec(line)
    if (inc) {
      if (!isActive()) continue
      const name = inc[1]
      const target = registry.get(name)
      if (!target) {
        throw new WgslAssembleError(`${unit.name}: 找不到 #include "${name}"`)
      }
      if (stack.includes(name)) {
        throw new WgslAssembleError(
          `循环 #include: ${[...stack, name].join(" -> ")}`,
        )
      }
      out.push(expand(target, registry, defines, [...stack, name]))
      continue
    }

    const directive = /^\s*#\s*(\w+)\b(.*)$/.exec(line)
    if (directive) {
      const [, keyword, rest] = directive
      const arg = rest.trim()
      switch (keyword) {
        case "if": {
          const parentActive = isActive()
          const conditionTrue = parentActive
            ? evaluate(arg, defines) !== 0
            : false
          conditionals.push({
            parentActive,
            conditionTrue,
            taken: conditionTrue,
            active: parentActive && conditionTrue,
          })
          continue
        }
        case "ifdef":
        case "ifndef": {
          const parentActive = isActive()
          const defined = defines[arg] !== undefined
          const conditionTrue = parentActive
            ? keyword === "ifdef"
              ? defined
              : !defined
            : false
          conditionals.push({
            parentActive,
            conditionTrue,
            taken: conditionTrue,
            active: parentActive && conditionTrue,
          })
          continue
        }
        case "elif": {
          const top = conditionals[conditionals.length - 1]
          if (!top)
            throw new WgslAssembleError(`${unit.name}: #elif 没有对应的 #if`)
          const conditionTrue =
            top.parentActive && !top.taken && evaluate(arg, defines) !== 0
          top.conditionTrue = conditionTrue
          top.taken = top.taken || conditionTrue
          top.active = top.parentActive && conditionTrue
          continue
        }
        case "else": {
          const top = conditionals[conditionals.length - 1]
          if (!top)
            throw new WgslAssembleError(`${unit.name}: #else 没有对应的 #if`)
          const conditionTrue = top.parentActive && !top.taken
          top.conditionTrue = conditionTrue
          top.taken = true
          top.active = top.parentActive && conditionTrue
          continue
        }
        case "endif": {
          if (conditionals.pop() === undefined) {
            throw new WgslAssembleError(`${unit.name}: #endif 没有对应的 #if`)
          }
          continue
        }
        case "define":
        case "undef":
        case "pragma":
          throw new WgslAssembleError(
            `${unit.name}: 不支持 #${keyword}（define 必须由调用方通过 DefineMap 传入）`,
          )
        default:
          throw new WgslAssembleError(
            `${unit.name}: 未知预处理指令 #${keyword}`,
          )
      }
    }

    if (!isActive()) continue
    out.push(substitute(line, defines))
  }

  if (conditionals.length > 0) {
    throw new WgslAssembleError(
      `${unit.name}: 有 ${conditionals.length} 个 #if 没有 #endif`,
    )
  }
  return out.join("\n")
}

function substitute(line: string, defines: DefineMap): string {
  return line.replace(DEFINE_SUB_RE, (whole, name: string) => {
    const v = defines[name]
    if (v === undefined) return whole
    if (typeof v === "boolean") return v ? "1" : "0"
    return String(v)
  })
}

/** 清空组装缓存（测试用）。 */
export function clearWgslCache(): void {
  CACHE.clear()
}
