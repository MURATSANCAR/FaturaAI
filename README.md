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
  └─ weak PDF / foto → Extract v2
Extract (:8106, uvicorn workers=5 × 8 OCR threads)
  ├─ pdf-inspector (text/CID) → pdftotext fallback
  ├─ FAST_PATH_PDF (text-layer güçlüyse OCR/Docling skip)
  ├─ Photo/raster OCR: PP-OCRv6 Small → (conf<0.90 / alan fail) → Medium
  │    backend: OpenVINO (auto) → ONNX Runtime fallback
  └─ Docling structure (+ optional OCR)
```

## Desteklenen girdiler

- PDF — e-Arşiv / e-Fatura
- Fotoğraf — JPG/PNG/WEBP (+ HEIC best-effort); mobilde **Foto çek**
- Sync `POST /extract` hâlâ var (script/load test)

Zayıf / taranmış / bozuk metin PDF ve fotoğraflarda **PaddleOCR-VL-1.6** (layout-agnostic) kullanılır; RapidOCR PP-OCRv6 yedek yoldur. Temiz dijital/UBL PDF’ler fast-path’te kalır.

## Deploy

```bash
./deploy/deploy-to-portal.sh
```

Servisler: `fatura-extract`, `fatura-api`

## OCR CPU playground

PP-StructureV3 / PaddleOCR-VL / RapidOCR karşılaştırması (CPU+RAM):

```bash
cd playground/ocr-bench && ./setup.sh
./run.sh --all --limit 5
```

Detay: `playground/ocr-bench/README.md`
