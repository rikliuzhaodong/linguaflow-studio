#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ASR_PYTHON="${FLUENTFRAME_ASR_PYTHON:-$PROJECT_DIR/.venv/bin/python}"

if [[ ! -x "$ASR_PYTHON" ]]; then
  echo "找不到项目 Python 虚拟环境：$ASR_PYTHON" >&2
  echo "请按 README 执行：python3.11 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt" >&2
  exit 1
fi

"$ASR_PYTHON" "$PROJECT_DIR/local-asr/server.py" &
ASR_PID=$!

echo "Qwen3-ASR 服务已启动：http://127.0.0.1:8766"
echo "正在启动 FluentFrame 网页，等待看到 Local 地址…"

cleanup() {
  kill "$ASR_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

"$PROJECT_DIR/node_modules/.bin/vinext" dev --host 127.0.0.1
