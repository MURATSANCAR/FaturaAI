#!/usr/bin/env bash
# Install/update FaturaAI extract (Docling) + API + web on portal host
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER="${SERVER:-nanobase}"
REMOTE_DIR="${REMOTE_DIR:-/data/nanobaseai/fatura}"

echo "==> Syncing to ${SERVER}:${REMOTE_DIR}"
ssh "${SERVER}" "sudo mkdir -p '${REMOTE_DIR}' && sudo chown \"\$(whoami):\" '${REMOTE_DIR}'"
rsync -az --delete \
  --exclude '.git' \
  --exclude 'node_modules' \
  --exclude 'apps/*/node_modules' \
  --exclude 'apps/*/dist' \
  --exclude 'apps/extract/.venv' \
  --exclude '.DS_Store' \
  "${ROOT}/" "${SERVER}:${REMOTE_DIR}/"

echo "==> Remote build + services"
ssh "${SERVER}" bash -s <<'REMOTE'
set -euo pipefail
DIR=/data/nanobaseai/fatura
cd "$DIR"

if ! command -v pdftotext >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y poppler-utils
fi

# --- Extract venv (reuse Docling from QA venv via --system-site-packages if needed) ---
EXTRACT="$DIR/apps/extract"
cd "$EXTRACT"
if [ ! -x .venv/bin/python ]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv .venv --python 3.12
  else
    python3 -m venv .venv
  fi
fi

# Prefer linking Docling from QA venv to avoid multi-GB reinstall
QA_PY=/data/nanobaseai/NanobaseAI-QA/.venv/bin/python
PADDLE_VER="${PADDLE_VER:-3.3.0}"
PADDLE_INDEX="${PADDLE_INDEX:-https://www.paddlepaddle.org.cn/packages/stable/cpu/}"
if [ -x "$QA_PY" ] && "$QA_PY" -c "import docling" 2>/dev/null; then
  echo "Using QA venv python for Docling runtime"
  # Install thin deps into a venv that can see QA packages via PYTHONPATH
  .venv/bin/pip install -q -U pip
  echo "Installing paddlepaddle==${PADDLE_VER} (CPU) for PaddleOCR-VL"
  .venv/bin/pip install -q "paddlepaddle==${PADDLE_VER}" -i "$PADDLE_INDEX" || \
    .venv/bin/pip install -q "paddlepaddle==${PADDLE_VER}"
  .venv/bin/pip install -q -r requirements.txt
  # Wrapper runner
  cat > "$EXTRACT/run.sh" <<'RUN'
#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
QA_SITE=$(/data/nanobaseai/NanobaseAI-QA/.venv/bin/python -c 'import site; print(":".join(site.getsitepackages()))')
export PYTHONPATH="${DIR}:${QA_SITE}:${PYTHONPATH:-}"
export PHOTO_OCR_THREADS="${PHOTO_OCR_THREADS:-6}"
export OMP_NUM_THREADS="${PHOTO_OCR_THREADS}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
WORKERS="${EXTRACT_WORKERS:-2}"
exec "$DIR/.venv/bin/python" -m uvicorn main:app --host 127.0.0.1 --port "${PORT:-8106}" --workers "$WORKERS"
RUN
  chmod +x "$EXTRACT/run.sh"
  EXEC_START="$EXTRACT/run.sh"
else
  echo "Installing docling into extract venv (slow first time)"
  .venv/bin/pip install -q -U pip
  echo "Installing paddlepaddle==${PADDLE_VER} (CPU) for PaddleOCR-VL"
  .venv/bin/pip install -q "paddlepaddle==${PADDLE_VER}" -i "$PADDLE_INDEX" || \
    .venv/bin/pip install -q "paddlepaddle==${PADDLE_VER}"
  .venv/bin/pip install -q -r requirements.txt docling
  EXEC_START="$EXTRACT/.venv/bin/uvicorn main:app --host 127.0.0.1 --port 8106 --workers ${EXTRACT_WORKERS:-2}"
fi

# systemd extract
sudo tee /etc/systemd/system/fatura-extract.service >/dev/null <<EOF
[Unit]
Description=FaturaAI Extract (Docling pipeline)
After=network.target

[Service]
Type=simple
User=administrator
WorkingDirectory=$EXTRACT
Environment=PORT=8106
Environment=ENABLE_DOCLING=1
Environment=ENABLE_DOCLING_OCR=0
Environment=FORCE_IMAGE_OCR=1
Environment=FAST_PATH_PDF=1
Environment=PHOTO_OCR_ENABLED=1
Environment=PHOTO_OCR_ENGINE=auto
Environment=PHOTO_OCR_THREADS=6
Environment=PHOTO_OCR_CONF_THRESHOLD=0.78
Environment=PHOTO_OCR_EARLY_STRUCT=6
Environment=PHOTO_OCR_TARGET_SIDE=1800
Environment=PHOTO_OCR_MAX_SIDE=2400
Environment=PHOTO_OCR_MIN_SIDE=1200
Environment=PHOTO_OCR_WARMUP=1
Environment=PHOTO_OCR_WARMUP_MEDIUM=0
Environment=PHOTO_OCR_MAX_INFLIGHT=3
Environment=PHOTO_OCR_SERIALIZE=1
Environment=PHOTO_OCR_TIMEOUT_S=90
Environment=VL_OCR_ENABLED=0
Environment=VL_OCR_PIPELINE=v1.6
Environment=VL_OCR_DEVICE=cpu
Environment=VL_OCR_THREADS=4
Environment=VL_OCR_SERIALIZE=1
Environment=VL_OCR_TIMEOUT_S=900
Environment=VL_OCR_WARMUP=0
Environment=VL_OCR_SUBPROCESS=1
Environment=PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
Environment=PDF_RASTER_DPI=200
Environment=PDF_RASTER_MAX_PAGES=3
Environment=IMAGE_OCR_SCALE=2.0
Environment=DOCLING_MAX_INFLIGHT=1
Environment=DOCLING_TIMEOUT_S=120
Environment=EXTRACT_WORKERS=5
Environment=OMP_NUM_THREADS=4
Environment=OPENBLAS_NUM_THREADS=1
Environment=MKL_NUM_THREADS=1
Environment=ALLOWED_ORIGINS=https://portal.nanobase.ai
Environment=PYTHONUNBUFFERED=1
ExecStart=$EXEC_START
Restart=on-failure
RestartSec=5
TimeoutStartSec=600
MemoryMax=64G

[Install]
WantedBy=multi-user.target
EOF

# API + web
cd "$DIR"
npm install
npm run build -w @fatura-ai/api
npm run build -w @fatura-ai/web
npm run test -w @fatura-ai/api || true

sudo cp "$DIR/deploy/fatura-api.service" /etc/systemd/system/fatura-api.service
sudo systemctl daemon-reload
sudo systemctl enable fatura-extract fatura-api
sudo systemctl restart fatura-extract
sleep 3
sudo systemctl restart fatura-api

sudo python3 "$DIR/deploy/fix-nginx-fatura.py" || true
sudo nginx -t && sudo systemctl reload nginx

echo "==> Smoke"
curl -sS http://127.0.0.1:8106/health || true
echo
curl -sS http://127.0.0.1:8105/health || true
echo
for f in \
  samples/MDA2022000002839.pdf \
  samples/earsiv_faturaKVI2026000009854.pdf \
  samples/HAVA_SAVUNMA_SISTEMLERI_SANAYI_GIB2026000000059.pdf \
  samples/babymall-BBE2026000018417.pdf
 do
  [ -f "$DIR/$f" ] || continue
  echo "--- $f ---"
  curl -sS -F "file=@$DIR/$f" http://127.0.0.1:8105/extract | python3 -c "
import sys,json
d=json.load(sys.stdin)
inv=d.get('invoice') or {}
val=d.get('validation') or {}
print(d.get('status'), d.get('method'), d.get('durationMs'), 'ms', 'conf', val.get('confidence'), 'lines', len(inv.get('lines') or []), inv.get('invoiceNumber'), (inv.get('totals') or {}).get('payableAmount'), d.get('warnings'))
"
done
REMOTE

echo "==> Done https://portal.nanobase.ai/fatura/"
