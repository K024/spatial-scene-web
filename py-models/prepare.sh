#!/usr/bin/env bash
#
# py-models/prepare.sh — 准备 SHARP / DepthPro 仓库与模型产物。
#
# 用法:
#   ./prepare.sh                 # 仅 clone 两个仓库（默认）
#   ./prepare.sh --sharp-model   # + 下载 ml-sharp 官方权重 (~2.8 GB)
#   ./prepare.sh --onnx          # + 下载预导出的 ONNX 产物 (~1.8 GB)
#   ./prepare.sh --all           # 上述全部
#
# 环境变量:
#   GIT_DEPTH=1         git clone --depth（0 表示完整克隆）
#   ARIA2_JOBS=4        aria2c 单文件并发连接数
#   ARIA2_ARGS=""       追加给 aria2c 的参数
#   HF_REPO             ONNX 产物所在的 HuggingFace 仓库
#
# 说明:
#   - 下载工具优先 aria2c（多连接 + 断点续传），回退 wget -c，再回退 curl -C -。
#   - 不下载时只按 HTTP content-length 比对本地文件字节数判断是否完整
#     （不做哈希校验）。
#   - 脚本不创建 venv、不安装 Python 依赖，只负责“取数据”。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CLONES_DIR="$SCRIPT_DIR/clones"
ONNX_DIR="$SCRIPT_DIR/out/onnx"

ML_SHARP_DIR="$CLONES_DIR/ml-sharp"
ML_DEPTH_PRO_DIR="$CLONES_DIR/ml-depth-pro"
SHARP_DOWNLOADS_DIR="$ML_SHARP_DIR/downloads"

GIT_DEPTH="${GIT_DEPTH:-1}"
ARIA2_JOBS="${ARIA2_JOBS:-4}"
ARIA2_ARGS="${ARIA2_ARGS:-}"
HF_REPO="${HF_REPO:-K024/ml-sharp-onnx}"

ML_SHARP_REPO="https://github.com/apple/ml-sharp.git"
ML_DEPTH_PRO_REPO="https://github.com/apple/ml-depth-pro.git"
SHARP_CKPT_URL="https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
HF_BASE="https://huggingface.co/${HF_REPO}/resolve/main"

DO_SHARP_MODEL=0
DO_ONNX=0

# --------------------------------------------------------------------------- #
# 日志
# --------------------------------------------------------------------------- #
log()  { printf '\033[36m[prepare]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[prepare]\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[31m[prepare]\033[0m %s\n' "$*" >&2; }

# --------------------------------------------------------------------------- #
# 参数解析
# --------------------------------------------------------------------------- #
usage() {
  sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sharp-model) DO_SHARP_MODEL=1 ;;
    --onnx)        DO_ONNX=1 ;;
    --all)         DO_SHARP_MODEL=1; DO_ONNX=1 ;;
    -h|--help)     usage 0 ;;
    *) err "未知参数: $1"; usage 1 ;;
  esac
  shift
done

# --------------------------------------------------------------------------- #
# 依赖探测
# --------------------------------------------------------------------------- #
DOWNLOADER=""
if command -v aria2c >/dev/null 2>&1; then
  DOWNLOADER="aria2c"
elif command -v wget >/dev/null 2>&1; then
  DOWNLOADER="wget"
  warn "未找到 aria2c，回退 wget（单连接）。建议安装 aria2 以启用多连接下载。"
elif command -v curl >/dev/null 2>&1; then
  DOWNLOADER="curl"
  warn "未找到 aria2c/wget，回退 curl（单连接）。"
else
  err "需要 aria2c / wget / curl 之一。"
  exit 1
fi

command -v git >/dev/null 2>&1 || { err "需要 git。"; exit 1; }

# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
# 远端文件字节数（HEAD）。失败时输出空串。
remote_size() {
  local url="$1" size=""
  if command -v curl >/dev/null 2>&1; then
    # 跟随重定向；取最后一个 content-length（LFS 场景首行是重定向响应）
    size="$(curl -sIL "$url" 2>/dev/null \
      | tr -d '\r' \
      | awk 'tolower($1)=="content-length:" {v=$2} END{print v}')"
  elif command -v wget >/dev/null 2>&1; then
    size="$(wget --spider --server-response -O /dev/null "$url" 2>&1 \
      | tr -d '\r' \
      | awk 'tolower($1)=="content-length:" {v=$2} END{print v}')"
  fi
  printf '%s' "$size"
}

# 本地文件字节数（不存在输出 0）
local_size() {
  local path="$1"
  if [[ -f "$path" ]]; then
    # 跨平台：Linux/mac 用 stat -c%s / stat -f%z
    stat -c%s "$path" 2>/dev/null || stat -f%z "$path" 2>/dev/null || echo 0
  else
    echo 0
  fi
}

human() {
  awk -v b="$1" 'BEGIN{
    split("B KB MB GB TB", u, " "); i=1;
    while (b >= 1024 && i < 5) { b /= 1024; i++ }
    printf (i==1 ? "%.0f %s" : "%.2f %s"), b, u[i]
  }'
}

# 下载到指定路径；已完整则跳过。$1=url $2=输出路径 $3=说明
fetch() {
  local url="$1" out="$2" label="${3:-$(basename "$2")}"

  local want
  want="$(remote_size "$url")"
  if [[ -z "$want" ]]; then
    warn "无法获取远端大小，将直接下载: $label"
  fi

  local have
  have="$(local_size "$out")"
  if [[ -n "$want" && "$have" == "$want" ]]; then
    log "已存在且完整，跳过: $label ($(human "$have"))"
    return 0
  fi
  if [[ -n "$want" && "$have" != "0" ]]; then
    log "部分存在 ($(human "$have") / $(human "$want"))，续传: $label"
  else
    log "下载 $label $( [[ -n "$want" ]] && echo "($(human "$want"))" )"
  fi

  mkdir -p "$(dirname "$out")"

  case "$DOWNLOADER" in
    aria2c)
      # --continue: 断点续传；-x/-s: 单文件多连接
      # shellcheck disable=SC2086
      aria2c --continue=true \
             --max-connection-per-server="$ARIA2_JOBS" \
             --split="$ARIA2_JOBS" \
             --min-split-size=8M \
             --file-allocation=none \
             --summary-interval=0 \
             --console-log-level=warn \
             --dir="$(dirname "$out")" \
             --out="$(basename "$out")" \
             $ARIA2_ARGS \
             "$url"
      ;;
    wget)
      wget --continue --show-progress --output-document="$out" "$url"
      ;;
    curl)
      curl --location --continue-at - --output "$out" "$url"
      ;;
  esac

  # 下载后校验字节数
  if [[ -n "$want" ]]; then
    local now
    now="$(local_size "$out")"
    if [[ "$now" != "$want" ]]; then
      err "大小不符: $label 本地 $(human "$now") != 远端 $(human "$want")"
      return 1
    fi
    log "完成 $label ($(human "$now"))"
  else
    log "完成 $label ($(human "$(local_size "$out")"))"
  fi
}

# clone 或更新仓库。$1=repo url $2=目标目录 $3=名称
clone_or_update() {
  local url="$1" dir="$2" name="$3"

  if [[ -d "$dir/.git" ]]; then
    log "已存在，尝试更新: $name"
    if git -C "$dir" pull --ff-only --quiet 2>/dev/null; then
      log "更新完成: $name"
    else
      warn "更新失败（可能是浅克隆或本地改动），保持现状: $name"
    fi
    return 0
  fi

  mkdir -p "$(dirname "$dir")"
  local depth_args=()
  if [[ "$GIT_DEPTH" != "0" ]]; then
    depth_args=(--depth "$GIT_DEPTH")
    log "浅克隆 (--depth $GIT_DEPTH): $name"
  else
    log "完整克隆: $name"
  fi

  git clone "${depth_args[@]}" --quiet "$url" "$dir"
  log "克隆完成: $name"
}

# --------------------------------------------------------------------------- #
# 步骤
# --------------------------------------------------------------------------- #
step_clone() {
  log "=== 1/3 克隆仓库 ==="
  clone_or_update "$ML_SHARP_REPO" "$ML_SHARP_DIR" "ml-sharp"
  clone_or_update "$ML_DEPTH_PRO_REPO" "$ML_DEPTH_PRO_DIR" "ml-depth-pro"
}

step_sharp_model() {
  log "=== 2/3 下载 ml-sharp 权重 ==="
  log "目标: $SHARP_DOWNLOADS_DIR"
  fetch "$SHARP_CKPT_URL" "$SHARP_DOWNLOADS_DIR/sharp_2572gikvuh.pt" "SHARP checkpoint"
}

step_onnx() {
  log "=== 3/3 下载 ONNX 产物 ==="
  log "来源: https://huggingface.co/$HF_REPO"
  log "目标: $ONNX_DIR"

  # 4 个文件：两张图各自的 .onnx（图结构）+ .onnx.data（外部权重）
  local files=(
    "sharp_fp16.onnx"
    "sharp_fp16.onnx.data"
    "sharp_mm4f16_hqq_bs32.onnx"
    "sharp_mm4f16_hqq_bs32.onnx.data"
  )
  for f in "${files[@]}"; do
    fetch "$HF_BASE/$f" "$ONNX_DIR/$f" "$f"
  done
}

# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
log "工作目录: $SCRIPT_DIR"
log "下载工具: $DOWNLOADER"

step_clone
[[ "$DO_SHARP_MODEL" == "1" ]] && step_sharp_model
[[ "$DO_ONNX" == "1" ]] && step_onnx

log "全部完成。"
log "下一步（可选）: python -m venv venv && ./venv/bin/pip install -r requirements.txt && ./venv/bin/pip install --no-deps -e clones/ml-sharp"
