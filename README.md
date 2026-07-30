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
  ├─ job queue (max inflight 8, max queue 200)
  ├─ PDF fast-path: pdftotext/UBL (~ms–sn)
  └─ weak PDF / foto → Extract v2 Docling
Extract (:8106, uvicorn workers=4)
  ├─ asyncio.to_thread(Docling) + semaphore
  ├─ FAST_PATH_PDF (text-layer güçlüyse Docling skip)
  └─ image OCR (Tesseract + Docling)
```

## Desteklenen girdiler

- PDF — e-Arşiv / e-Fatura
- Fotoğraf — JPG/PNG/WEBP (+ HEIC best-effort); mobilde **Foto çek**
- Sync `POST /extract` hâlâ var (script/load test)

## Deploy

```bash
./deploy/deploy-to-portal.sh
```

Servisler: `fatura-extract`, `fatura-api`
