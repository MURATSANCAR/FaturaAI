# FaturaAI

Nanobase Portal üzerinde e-Arşiv / e-Fatura PDF okuma.

- UI: https://portal.nanobase.ai/fatura/
- API: https://portal.nanobase.ai/fatura-api/health

## Nasıl çalışır

1. PDF yüklenir → `pdftotext -layout` (Poppler) metni çıkarır
2. GİB alanları parse edilir (gömülü UBL varsa UBL-TR)
3. Ekranda tüm alanlar + okuma süresi (`durationMs`) gösterilir

## Geliştirme

```bash
# Sunucuda (Node + poppler-utils gerekir)
npm install
npm run dev:api   # :8105
npm run dev:web   # :5173 → /fatura-api proxy
npm test
```

## Deploy

```bash
./deploy/deploy-to-portal.sh
```

Örnek fatura: `samples/HAVA_SAVUNMA_SISTEMLERI_SANAYI_GIB2026000000059.pdf`
