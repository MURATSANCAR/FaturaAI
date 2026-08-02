#!/usr/bin/env bash
# Convenience runner for the OCR CPU playground.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="${OCR_BENCH_VENV:-$ROOT/.venv}"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "venv missing — run ./setup.sh first" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export FLAGS_use_mkldnn="${FLAGS_use_mkldnn:-1}"
# Skip paddlex hoster connectivity probe (faster cold start).
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="${PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK:-True}"

ALL=0
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      ALL=1
      shift
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

merge_reports() {
  OCR_BENCH_ROOT="$ROOT" python - <<'PY'
import json, os
from pathlib import Path

out = Path(os.environ["OCR_BENCH_ROOT"]) / "out"
parts = []
for eng in ("structure", "vl", "rapid"):
    p = out / f"report-{eng}.json"
    if p.exists():
        parts.append(json.loads(p.read_text(encoding="utf-8")))
if not parts:
    raise SystemExit("no reports to merge")
summary = [
    {
        "engine": s["engine"],
        "ok": s["ok"],
        "fail": s["fail"],
        "peak_rss_gb": s["peak_rss_gb"],
        "sec": s["sec"],
        "fieldScore": s["fieldScore"],
        "fieldHits": s["fieldHits"],
    }
    for s in parts
]
merged = {"compare": summary}
path = out / "report-compare.json"
path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(merged, ensure_ascii=False, indent=2))
print(f"Wrote {path}")
PY
}

if [[ "$ALL" -eq 1 ]]; then
  # Separate processes so Structure/VL/Rapid never share RSS.
  # Default small set for first all-run unless user passed --limit.
  has_limit=0
  if ((${#ARGS[@]})); then
    for a in "${ARGS[@]}"; do
      if [[ "$a" == "--limit" ]]; then has_limit=1; fi
    done
  fi
  EXTRA=()
  if [[ "$has_limit" -eq 0 ]]; then
    EXTRA=(--limit 5)
  fi
  export OCR_BENCH_ROOT="$ROOT"
  cd "$ROOT"
  for eng in structure vl rapid; do
    echo "======== ENGINE=$eng ========"
    python "$ROOT/bench.py" --engine "$eng" "${EXTRA[@]+"${EXTRA[@]}"}" "${ARGS[@]+"${ARGS[@]}"}"
  done
  merge_reports
else
  exec python "$ROOT/bench.py" "${ARGS[@]+"${ARGS[@]}"}"
fi
