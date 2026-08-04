# FaturaAI

Nanobase Portal üzerinde e-Arşiv / e-Fatura okuma (PDF + fotoğraf) — prod ölçekli kuyruk.

- UI: https://portal.nanobase.ai/fatura/
- API: https://portal.nanobase.ai/fatura-api/health
- Metrics: https://portal.nanobase.ai/fatura-api/metrics
- Extract: `127.0.0.1:8106` (Docling workers)

## Prod mimari

```
Browser
  → POST /fatura-api/jobs   (202 + jobId)
  → GET  /fatura-api/jobs/:id  (poll: queued|running|done)
API (:8105)
  ├─ rate limit (IP)
  ├─ job queue (max inflight 6, max queue 500)
  ├─ PDF fast-path: pdf-inspector → pdftotext / UBL (~ms–sn)
  └─ weak PDF / photo → Extract v2
Extract (:8106, uvicorn workers=5 × PHOTO_OCR_THREADS=6)
  ├─ pdf-inspector (text/CID) → pdftotext fallback
  ├─ FAST_PATH_PDF (text-layer güçlüyse OCR/Docling skip)
  ├─ Photo/raster OCR (prod-tuned):
  │    PP-OCRv6 Small → (conf<0.78 / alan fail) → Medium
  │    variants (nodeskew/strong/binary) sadece çok zayıf structure’ta
  │    target≈1800 / max≈2400 / raster DPI=180 / inflight=3
  │    backend: OpenVINO (auto) → ONNX Runtime fallback
  └─ Docling structure (+ optional OCR; VL_OCR_ENABLED=0 on CPU prod)
```

### Photo OCR varsayılanları (CPU prod)

| Env | Default | Not |
|-----|---------|-----|
| `PHOTO_OCR_TARGET_SIDE` | 1800 | uzun kenar hedefi |
| `PHOTO_OCR_MAX_SIDE` | 2400 | üst sınır |
| `PHOTO_OCR_MIN_SIDE` | 1200 | upscale eşiği |
| `PHOTO_OCR_CONF_THRESHOLD` | 0.78 | Medium tetik eşiği |
| `PHOTO_OCR_EARLY_STRUCT` | 6 | early-exit structure skoru |
| `PHOTO_OCR_MAX_INFLIGHT` | 3 | worker başına eşzamanlı OCR |
| `PHOTO_OCR_SERIALIZE` | 1 | OOM-safe; inflight=3 kept after load test |
| `PHOTO_OCR_TIMEOUT_S` | 90 | |
| `PDF_RASTER_DPI` | 180 | taranmış PDF raster |
| `VL_OCR_ENABLED` | 0 | CPU prod’da kapalı |

Tipik taranmış sayfa hedefi: **2.5–7 sn** (eski 8–25+ sn); çoğu sayfa **1–2 inference pass**.

## Desteklenen girdiler

- PDF — e-Arşiv / e-Fatura
- Fotoğraf — JPG/PNG/WEBP (+ HEIC best-effort); mobilde **Foto çek**
- Sync `POST /extract` hâlâ var (script/load test)

Ana yol **RapidOCR PP-OCRv6** (Small→Medium). PaddleOCR-VL CPU prod’da kapalı (`VL_OCR_ENABLED=0`); sadece opt-in escalation. Temiz dijital/UBL PDF’ler fast-path’te kalır.

## Deploy

```bash
./deploy/deploy-to-portal.sh
```

Servisler: `fatura-extract`, `fatura-api`

Sunucuda systemd yenileme:

```bash
sudo cp apps/extract/fatura-extract.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl restart fatura-extract
curl -s http://127.0.0.1:8106/health | jq
```

## OCR CPU playground

PP-StructureV3 / PaddleOCR-VL / RapidOCR karşılaştırması (CPU+RAM):

```bash
cd playground/ocr-bench && ./setup.sh
./run.sh --engine rapid --limit 10
```

Detay: `playground/ocr-bench/README.md`
