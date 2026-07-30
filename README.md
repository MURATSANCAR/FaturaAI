# FaturaAI

Nanobase Portal üzerinde e-Arşiv / e-Fatura okuma (PDF + fotoğraf).

- UI: https://portal.nanobase.ai/fatura/
- API: https://portal.nanobase.ai/fatura-api/health
- Extract: `127.0.0.1:8106` (Docling + pdftotext pipeline)

## Desteklenen girdiler

- **PDF** — e-Arşiv ve e-Fatura (metin katmanı / gömülü UBL)
- **Fotoğraf** — JPG, PNG, WEBP (+ HEIC best-effort); mobilde **Yükle** ve **Foto çek**
- Belge tipi heuristic: `earsiv` | `efatura` | `ubl` | `unknown`

## Pipeline (prod)

```
PDF
 ├─ UBL gömülü mü?
 ├─ pdftotext (hızlı alanlar)
 ├─ Docling structure + tablolar (CPU)
 ├─ (opsiyonel) Docling OCR — ENABLE_DOCLING_OCR=1
 ├─ Pydantic Invoice JSON
 └─ Matematik doğrulama → confidence

Fotoğraf / kamera
 ├─ (HEIC ise) JPEG’e çevir
 ├─ Docling IMAGE + OCR (FORCE_IMAGE_OCR=1, varsayılan açık)
 ├─ Aynı alan çıkarımı + doğrulama
 └─ confidence
```

Node API (`8105`) extract v2 servisine proxy eder; PDF’de servis düşerse legacy parser’a düşer. Fotoğraflar için extract v2 zorunludur.

## Deploy

```bash
./deploy/deploy-to-portal.sh
```

Servisler: `fatura-extract`, `fatura-api`
