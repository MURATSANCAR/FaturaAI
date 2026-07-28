#!/usr/bin/env bash
# Deploy FaturaAI to portal.nanobase.ai and add hub "FaturaAI" card → /fatura/
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
  --exclude '.DS_Store' \
  "${ROOT}/" "${SERVER}:${REMOTE_DIR}/"

echo "==> Remote install + build + nginx + hub patch"
ssh "${SERVER}" bash -s <<'REMOTE'
set -euo pipefail
DIR=/data/nanobaseai/fatura
cd "$DIR"

if ! command -v pdftotext >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y poppler-utils
fi

npm install
npm run build -w @fatura-ai/api
npm run build -w @fatura-ai/web
npm run test -w @fatura-ai/api

sudo cp "$DIR/deploy/fatura-api.service" /etc/systemd/system/fatura-api.service
sudo systemctl daemon-reload
sudo systemctl enable fatura-api
sudo systemctl restart fatura-api

sudo python3 "$DIR/deploy/fix-nginx-fatura.py"
sudo nginx -t
sudo systemctl reload nginx

INDEX_JS=$(ls /data/nanobaseai-mobile/portal/dist/assets/index-*.js | head -1 || true)
if [ -n "${INDEX_JS:-}" ]; then
  sudo python3 "$DIR/deploy/patch-hub-fatura.py" "$INDEX_JS"
else
  echo "WARNING: portal hub index-*.js not found"
fi

echo "==> Smoke"
sleep 1
curl -sS -o /dev/null -w "api health %{http_code}\n" http://127.0.0.1:8105/health
curl -sS -o /dev/null -w "public fatura %{http_code}\n" https://portal.nanobase.ai/fatura/
curl -sS -o /dev/null -w "public fatura-api %{http_code}\n" https://portal.nanobase.ai/fatura-api/health
TITLE=$(curl -sS https://portal.nanobase.ai/fatura/ | tr '\n' ' ' | sed -n 's/.*<title>\([^<]*\)<\/title>.*/\1/p')
echo "fatura title: ${TITLE}"
curl -sS -F "file=@$DIR/samples/HAVA_SAVUNMA_SISTEMLERI_SANAYI_GIB2026000000059.pdf" \
  http://127.0.0.1:8105/extract | python3 -c "import sys,json; d=json.load(sys.stdin); print('extract', d.get('status'), d.get('durationMs'), 'ms', d.get('invoice',{}).get('invoiceNumber'), d.get('invoice',{}).get('totals',{}).get('payableAmount'))"
REMOTE

echo "==> Done. Open https://portal.nanobase.ai/fatura/"
