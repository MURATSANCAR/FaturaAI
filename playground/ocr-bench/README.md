# OCR CPU playground

PP-StructureV3 vs PaddleOCR-VL vs RapidOCR PP-OCRv6 (prod baseline) — **CPU + RAM only**.

Prod extract servisini etkilemez; ayrı venv.

## Kurulum (sunucu veya local)

```bash
cd playground/ocr-bench
./setup.sh
```

İsteğe bağlı:

```bash
# Python seç
PYTHON=python3.12 ./setup.sh

# Paddle sürümü
PADDLE_VER=3.3.0 ./setup.sh
```

**Linux prod sunucu:** setup sırasında extract’ı durdurmak iyi olur (`MemoryMax` / OOM):

```bash
sudo systemctl stop fatura-extract   # benchmark bitince start
```

## Hızlı smoke

```bash
source .venv/bin/activate
./run.sh --engine structure --limit 1
./run.sh --engine rapid --limit 1
# VL CPU'da yavaş — önce 1 sayfa
./run.sh --engine vl --limit 1
```

## Karşılaştırma (önerilen)

Motorlar **ayrı process**te (RAM karışmasın):

```bash
./run.sh --all --limit 5
```

Çıktılar:

- `out/structure/*.md` — Structure markdown
- `out/vl/*.md` — VL markdown
- `out/rapid/*.md` — RapidOCR düz metin
- `out/report-{structure,vl,rapid}.json` — latency / RSS / field hits
- `out/report-compare.json` — özet tablo

## Metrikler

| Alan | Anlam |
|------|--------|
| `sec.p50/mean` | Sayfa süresi (warmup hariç) |
| `peak_rss_gb` | Process peak RAM |
| `fieldScore` | ETTN / VKN / fatura no / tarih / tutar / ödenecek / tablo ipucu |
| `fieldHits.tableMarkdown` | Markdown tablo üretimi (Structure/VL avantajı) |

Bu fieldScore tam fatura JSON doğrulaması değil; motor seçimi için hızlı sinyal.

## Örnek set

Varsayılan: repo `samples/` altındaki tüm `.png/.jpg/...`  
Özel klasör:

```bash
./run.sh --engine structure --samples ../../samples/layout-smoke2 --limit 8
```

## CPU ipuçları

```bash
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
# VL sayfa başı 30s–birkaç dk sürebilir; timeout uyarısı:
./run.sh --engine vl --limit 3 --timeout-s 300
```

**Aynı anda tek motor.** Structure + VL birlikte yükleme.

## Karar kuralı

1. `rapid` vs `structure`: structure `fieldScore` / `tableMarkdown` belirgin iyiyse → weak-path adayı  
2. `vl` kalitede kazanıp `sec.p50` ≫ 30–60s ise → CPU prod’a koyma; GPU veya offline batch  
3. Üçü de yakınsa → RapidOCR (mevcut OpenVINO path) kalsın
