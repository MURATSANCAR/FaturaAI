# FaturaAI

Nanobase Portal üzerinde e-Arşiv / e-Fatura PDF okuma.

- UI: https://portal.nanobase.ai/fatura/
- API: https://portal.nanobase.ai/fatura-api/health
- Extract: `127.0.0.1:8106` (Docling + pdftotext pipeline)

## Pipeline (prod)

```
PDF
 ├─ UBL gömülü mü?
 ├─ pdftotext (hızlı alanlar)
 ├─ Docling structure + tablolar (CPU)
 ├─ (opsiyonel) Docling OCR — ENABLE_DOCLING_OCR=1
 ├─ Pydantic Invoice JSON
 └─ Matematik doğrulama → confidence
```

Node API (`8105`) extract v2 servisine proxy eder; servis düşerse legacy parser’a düşer.

## Deploy

```bash
./deploy/deploy-to-portal.sh
```

Servisler: `fatura-extract`, `fatura-api`
