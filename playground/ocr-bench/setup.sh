#!/usr/bin/env bash
# Create an isolated venv and install CPU deps for Structure + VL + RapidOCR.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="${OCR_BENCH_VENV:-$ROOT/.venv}"
PYTHON="${PYTHON:-}"

if [[ -z "$PYTHON" ]]; then
  for cand in python3.12 python3.11 python3.10 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
      PYTHON="$(command -v "$cand")"
      break
    fi
  done
fi

if [[ -z "$PYTHON" ]]; then
  echo "ERROR: python3 not found (need 3.9+)." >&2
  exit 1
fi

PY_VER="$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "==> Using $PYTHON ($PY_VER)"
"$PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)' \
  || { echo "ERROR: need Python >= 3.9"; exit 1; }

echo "==> Creating venv at $VENV"
"$PYTHON" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -U pip wheel setuptools

PADDLE_VER="${PADDLE_VER:-3.3.0}"
PADDLE_INDEX="${PADDLE_INDEX:-https://www.paddlepaddle.org.cn/packages/stable/cpu/}"

echo "==> Installing paddlepaddle==$PADDLE_VER (CPU) from $PADDLE_INDEX"
python -m pip install "paddlepaddle==${PADDLE_VER}" -i "$PADDLE_INDEX"

echo "==> Installing playground requirements"
python -m pip install -r "$ROOT/requirements.txt"

echo "==> Smoke import"
python - <<'PY'
import paddle
print("paddle:", paddle.__version__)
import paddleocr
print("paddleocr: OK")
from rapidocr import RapidOCR
print("rapidocr: OK")
PY

cat <<EOF

Setup complete.

Activate:
  source $VENV/bin/activate

Quick smoke (1 image, structure):
  ./run.sh --engine structure --limit 1

Full CPU bench (structure + vl + rapid, separate processes):
  ./run.sh --all

EOF
