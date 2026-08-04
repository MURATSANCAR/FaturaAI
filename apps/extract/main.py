"""FaturaAI extract service — UBL → Docling(+tables) → pdftotext heuristics → validate."""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from tax_id import (
    coerce_tax_id,
    digits_only,
    is_valid_tax_id,
    is_valid_vkn,
    normalize_ocr_digits,
    repair_tax_id,
)

PORT = int(os.getenv("PORT", "8106"))
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "ALLOWED_ORIGINS",
        "https://portal.nanobase.ai,http://localhost:5173",
    ).split(",")
    if o.strip()
]
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
ENABLE_DOCLING = os.getenv("ENABLE_DOCLING", "1") == "1"
ENABLE_DOCLING_OCR = os.getenv("ENABLE_DOCLING_OCR", "0") == "1"
# Phone photos always OCR; PDF OCR stays behind ENABLE_DOCLING_OCR unless forced.
FORCE_IMAGE_OCR = os.getenv("FORCE_IMAGE_OCR", "1") == "1"
# Skip Docling when pdftotext already yields a strong invoice (metadata+totals).
FAST_PATH_PDF = os.getenv("FAST_PATH_PDF", "1") == "1"
FAST_PATH_MIN_CONF = float(os.getenv("FAST_PATH_MIN_CONF", "0.82"))
# Firecrawl pdf-inspector before pdftotext (CID/markdown fallback).
PDF_INSPECTOR_ENABLED = os.getenv("PDF_INSPECTOR_ENABLED", "1") == "1"
# Photo path: PP-OCRv6 Small→Medium (OpenVINO auto / ONNX fallback).
PHOTO_OCR_ENABLED = os.getenv("PHOTO_OCR_ENABLED", "1") == "1"
PHOTO_OCR_MIN_CONF = float(os.getenv("PHOTO_OCR_MIN_CONF", "0.55"))
PHOTO_OCR_WARMUP = os.getenv("PHOTO_OCR_WARMUP", "1") == "1"
# Do not preload Medium on every worker — major RAM saver under parallel load.
PHOTO_OCR_WARMUP_MEDIUM = os.getenv("PHOTO_OCR_WARMUP_MEDIUM", "0") == "1"
# Cap concurrent OCR per uvicorn worker (3; SERIALIZE=1 after race seen in load test).
PHOTO_OCR_MAX_INFLIGHT = max(1, int(os.getenv("PHOTO_OCR_MAX_INFLIGHT", "3")))
PHOTO_OCR_TIMEOUT_S = int(os.getenv("PHOTO_OCR_TIMEOUT_S", "90"))
# PaddleOCR-VL: keep OFF on CPU prod (too slow). Opt-in for escalation experiments.
VL_OCR_ENABLED = os.getenv("VL_OCR_ENABLED", "0") == "1"
VL_OCR_TIMEOUT_S = int(os.getenv("VL_OCR_TIMEOUT_S", "900"))
VL_OCR_WARMUP = os.getenv("VL_OCR_WARMUP", "0") == "1"
PDF_RASTER_DPI = max(72, int(os.getenv("PDF_RASTER_DPI", "180")))
DOCLING_MAX_INFLIGHT = max(1, int(os.getenv("DOCLING_MAX_INFLIGHT", "1")))
DOCLING_TIMEOUT_S = int(os.getenv("DOCLING_TIMEOUT_S", "120"))
IMAGE_OCR_SCALE = float(os.getenv("IMAGE_OCR_SCALE", "2.0"))

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".tif",
    ".tiff",
    ".bmp",
    ".heic",
    ".heif",
}

app = FastAPI(title="FaturaAI Extract", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_docling_converter = None
_docling_sem: asyncio.Semaphore | None = None
_photo_ocr_sem: asyncio.Semaphore | None = None
_metrics = {
    "extract_total": 0,
    "extract_ok": 0,
    "extract_partial": 0,
    "vl_ocr": 0,
    "extract_failed": 0,
    "fast_path": 0,
    "photo_ocr": 0,
    "docling_calls": 0,
    "inflight": 0,
    "photo_ocr_inflight": 0,
}


def get_docling_sem() -> asyncio.Semaphore:
    global _docling_sem
    if _docling_sem is None:
        _docling_sem = asyncio.Semaphore(DOCLING_MAX_INFLIGHT)
    return _docling_sem


def get_photo_ocr_sem() -> asyncio.Semaphore:
    global _photo_ocr_sem
    if _photo_ocr_sem is None:
        _photo_ocr_sem = asyncio.Semaphore(PHOTO_OCR_MAX_INFLIGHT)
    return _photo_ocr_sem


async def run_photo_ocr(
    path: Path, *, prefer_vl: bool = False
) -> tuple[str, dict[str, Any]]:
    """Run photo OCR with per-worker concurrency + timeout (OOM-safe).

    Default ladder: RapidOCR first. VL only when prefer_vl=True (escalation).
    """
    sem = get_photo_ocr_sem()
    await sem.acquire()
    _metrics["photo_ocr_inflight"] += 1
    try:
        if prefer_vl and VL_OCR_ENABLED:
            try:
                from vl_ocr import ocr_document

                text, meta = await asyncio.wait_for(
                    asyncio.to_thread(ocr_document, path),
                    timeout=VL_OCR_TIMEOUT_S,
                )
                _metrics["vl_ocr"] += 1
                return text, meta
            except Exception as exc:  # noqa: BLE001
                meta_err = {"engine": "vl-error", "vlError": str(exc)}
                from photo_ocr import ocr_image

                text, meta = await asyncio.wait_for(
                    asyncio.to_thread(ocr_image, path),
                    timeout=PHOTO_OCR_TIMEOUT_S,
                )
                meta = {**meta, **meta_err, "fallbackFrom": "vl"}
                return text, meta

        from photo_ocr import ocr_image

        return await asyncio.wait_for(
            asyncio.to_thread(ocr_image, path),
            timeout=PHOTO_OCR_TIMEOUT_S,
        )
    finally:
        _metrics["photo_ocr_inflight"] = max(0, _metrics["photo_ocr_inflight"] - 1)
        sem.release()


async def run_pdf_raster_ocr(
    path: Path, tmp: Path, max_pages: int = 2, *, prefer_vl: bool = False
) -> tuple[str, dict[str, Any]]:
    """Rasterize PDF pages + OCR under the same inflight cap as photo OCR."""
    sem = get_photo_ocr_sem()
    await sem.acquire()
    _metrics["photo_ocr_inflight"] += 1
    try:
        timeout = (
            max(VL_OCR_TIMEOUT_S * max_pages, 180)
            if prefer_vl and VL_OCR_ENABLED
            else max(PHOTO_OCR_TIMEOUT_S * max_pages, 180)
        )
        return await asyncio.wait_for(
            asyncio.to_thread(
                pdf_raster_ocr_text, path, tmp, max_pages, prefer_vl
            ),
            timeout=timeout,
        )
    finally:
        _metrics["photo_ocr_inflight"] = max(0, _metrics["photo_ocr_inflight"] - 1)
        sem.release()


async def run_docling(
    path: Path, ocr: bool = False, for_image: bool = False
) -> tuple[str, list[Line]]:
    """Run Docling off the event loop with concurrency + timeout caps."""
    sem = get_docling_sem()
    await sem.acquire()
    _metrics["inflight"] += 1
    _metrics["docling_calls"] += 1
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(docling_convert, path, ocr, for_image),
            timeout=DOCLING_TIMEOUT_S,
        )
    finally:
        _metrics["inflight"] = max(0, _metrics["inflight"] - 1)
        sem.release()


def strong_text_invoice(inv: Invoice, validation: Validation) -> bool:
    if validation.confidence < FAST_PATH_MIN_CONF:
        return False
    if not inv.invoiceNumber or inv.totals.payableAmount is None:
        return False
    if not inv.supplier.name or not inv.customer.name:
        return False
    # Lines preferred; allow strong metadata-only if conf still high after soft line miss
    if inv.lines:
        return True
    return validation.confidence >= 0.9 and bool(inv.issueDate)


class Party(BaseModel):
    name: str | None = None
    taxId: str | None = None
    taxIdScheme: Literal["VKN", "TCKN"] | None = None
    taxOffice: str | None = None
    address: str | None = None
    phone: str | None = None
    email: str | None = None
    website: str | None = None


class Line(BaseModel):
    id: str | None = None
    name: str | None = None
    quantity: float | None = None
    unit: str | None = None
    unitPrice: float | None = None
    discountRate: float | None = None
    discountAmount: float | None = None
    vatRate: float | None = None
    vatAmount: float | None = None
    withholdingNote: str | None = None
    lineTotal: float | None = None


class Totals(BaseModel):
    lineExtensionAmount: float | None = None
    discountTotal: float | None = None
    vatAmount: float | None = None
    withholdingVatAmount: float | None = None
    taxInclusiveAmount: float | None = None
    payableAmount: float | None = None
    currency: str = "TRY"


class Invoice(BaseModel):
    documentType: Literal["earsiv", "efatura", "ubl", "unknown"] = "unknown"
    profileId: str | None = None
    customizationId: str | None = None
    invoiceTypeCode: str | None = None
    invoiceNumber: str | None = None
    uuid: str | None = None
    issueDate: str | None = None
    issueTime: str | None = None
    supplier: Party = Field(default_factory=Party)
    customer: Party = Field(default_factory=Party)
    lines: list[Line] = Field(default_factory=list)
    totals: Totals = Field(default_factory=Totals)
    notes: list[str] = Field(default_factory=list)
    iban: str | None = None
    bankName: str | None = None
    bankBranch: str | None = None


class Validation(BaseModel):
    totalsMatch: bool = False
    confidence: float = 0.0
    checks: list[str] = Field(default_factory=list)


class ExtractResponse(BaseModel):
    status: Literal["ok", "partial", "failed"]
    method: str
    durationMs: int
    warnings: list[str] = Field(default_factory=list)
    invoice: Invoice | None = None
    rawTextPreview: str | None = None
    validation: Validation | None = None
    pipeline: list[str] = Field(default_factory=list)


def parse_tr_money(raw: str | None) -> float | None:
    if not raw:
        return None
    s = re.sub(r"(?i)TL|TRY|₺", "", raw).strip()
    s = s.replace("\u00a0", " ")
    # OCR: "1570L,00" / "12O00,00" — letter as thousand/decimal junk
    s = re.sub(r"(?<=\d)[LlIiOo](?=[.,]\d)", "", s)
    s = re.sub(r"(?<=\d)[Oo](?=\d)", "0", s)
    # OCR: "2,00,00" → 2.000,00
    if re.fullmatch(r"\d,\d{2},\d{2}", s):
        s = f"{s[0]}.000,{s[-2:]}"
    # OCR often uses space as decimal: "359 96" / "133 33"
    if re.fullmatch(r"\d{1,6}\s+\d{2}", s):
        s = s.replace(" ", ",")
    # thousands with space: "2 008,31"
    if re.search(r"\d\s+\d{3}[,.]", s):
        s = s.replace(" ", "")
    s = s.replace(" ", "").strip()
    if not s:
        return None
    # US/OCR thousands: 12,000,00 or 12,000.00 (last group = decimals)
    if re.fullmatch(r"\d{1,3}(,\d{3}){1,3}[.,]\d{2,4}", s):
        last_sep = max(s.rfind(","), s.rfind("."))
        head, frac = s[:last_sep], s[last_sep + 1 :]
        s = head.replace(",", "").replace(".", "") + "." + frac
    # TR classic: 1.457.08 / 12.000,00
    elif re.fullmatch(r"\d{1,3}(\.\d{3})+\.\d{2,4}", s):
        head, _, frac = s.rpartition(".")
        s = head.replace(".", "") + "." + frac
    elif "," in s and "." in s:
        # Decide decimal by last separator
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        # 1234,56 → decimal; 1,234 → ambiguous — if 3 digits after comma treat thousands only when another sep
        parts = s.split(",")
        if len(parts) == 2 and len(parts[1]) == 3 and len(parts[0]) <= 3:
            # Ambiguous 12,000 — prefer thousands when exactly 3 fractional digits and value would be huge as decimal
            # Treat as thousands (TR OCR of 12.000 without decimals)
            s = parts[0] + parts[1]
        else:
            s = s.replace(",", ".")
    elif re.fullmatch(r"\d{1,3}(\.\d{3})+", s):
        s = s.replace(".", "")
    try:
        n = float(s)
        return n if n == n else None
    except ValueError:
        return None


_MONEY_TOKEN = (
    r"(?:\d{1,3}(?:[.,\s]\d{3})+[.,]\d{1,8}"  # 12.000,00 / unit 3.749,16667
    r"|\d{1,3}(?:[.\s]\d{3})*[.,]\d{1,8}"
    r"|\d+[LlIiOo]?[.,]\d{1,8}"
    r"|\d,\d{2},\d{2}"  # OCR junk: 2,00,00 → 2.000,00
    r"|\d{1,6}\s\d{2})"
)


def _line_money_hits(s: str) -> list[re.Match[str]]:
    """Money tokens on a line, skipping leading SKU-like codes (e.g. 5007.1234605 APPLE)."""
    hits = list(re.finditer(rf"({_MONEY_TOKEN})\s*(?:TL|TRY)?", s, re.I))
    while hits:
        m = hits[0]
        after = s[m.end() :]
        # Product / PLU code at line start followed by letters is not an amount
        if m.start() <= 2 and re.match(r"\s*[A-Za-zÇĞİÖŞÜçğıöşü]", after):
            hits = hits[1:]
            continue
        break
    return hits


def normalize_ocr_text(text: str) -> str:
    """Generic OCR label/typo normalization for Turkish e-invoice layouts."""
    if not text:
        return text
    # Strip portal / viewer chrome
    text = re.sub(r"(?im)^\s*(?:PDF|XML)\s*indir\s*", "", text)
    text = re.sub(r"(?i)\bPDF\s*indir\b|\bXML\s*indir\b", " ", text)
    # Money OCR: "2.008, 30" → "2.008,30"
    text = re.sub(r"(\d),(\s+)(\d{2})\b", r"\1,\3", text)
    # OCR: "35.99L00TL" / "4,459;00" → proper decimals
    text = re.sub(r"(\d+[.,]\d{2})[Ll](\d{2})\b", r"\1,\2", text)
    text = re.sub(r"(\d+[.,]\d{3})[Ll](\d{2})\b", r"\1,\2", text)
    text = re.sub(r"(\d+[.,]\d{2,3});(\d{2})\b", r"\1,\2", text)
    text = re.sub(r"℃\s*L\b", "TL", text)
    text = re.sub(r"(\d)℃L\b", r"\1TL", text)
    # OCR: "1.49,83" → "1.499,83" (dropped thousands digit)
    text = re.sub(
        r"\b(\d+)\.(\d{2}),(\d{2})\b",
        lambda m: f"{m.group(1)}.{m.group(2)}{m.group(2)[-1]},{m.group(3)}",
        text,
    )
    # Currency OCR junk: T8Y / 7RY / TRV → TRY
    text = re.sub(r"\bT[8B]Y\b", "TRY", text)
    text = re.sub(r"\b[7T]RY\b", "TRY", text)
    text = re.sub(r"\bTRV\b", "TRY", text)
    # Glued "118,09.TRY" / "1.457,08.TRY"
    text = re.sub(r"(\d)\.TRY\b", r"\1 TRY", text, flags=re.I)
    replacements = (
        (r"\bSAYDN\b", "SAYIN"),
        (r"\bSAVIN\b", "SAYIN"),
        (r"\bSAMIN\b", "SAYIN"),
        (r"\bSAY[İI]N\b", "SAYIN"),
        (r"\bETIN\b", "ETTN"),
        (r"\bETTN\b", "ETTN"),
        # Glued / truncated ETTN label (ETN… / ETTNe…)
        (r"\bETT?Ne?\b(?=\s*[:\-]?[0-9A-Fa-f])", "ETTN"),
        (r"(?i)\bETT?N(?=[0-9A-Fa-fİILOS])", "ETTN"),
        (r"\bFatera\b", "Fatura"),
        (r"\bFatara\b", "Fatura"),
        (r"\bFataca\b", "Fatura"),
        (r"\bPatara\b", "Fatura"),
        (r"\bFatara\s*Na\b", "Fatura No"),
        (r"\bFataca\s*Na\b", "Fatura No"),
        (r"\bPatara\s*Na\b", "Fatura No"),
        (r"\bYarihi\b", "Tarihi"),
        (r"\bTanible\b", "Tarihi"),
        (r"\bTartht\b", "Tarihi"),
        (r"\bTaribi\b", "Tarihi"),
        (r"\bTarihl\b", "Tarihi"),
        (r"\bTarila\b", "Tarihi"),
        (r"\bTariba\b", "Tarihi"),
        (r"\bTarthi\b", "Tarihi"),
        (r"\bPataca\b", "Fatura"),
        (r"\bDatara\b", "Fatura"),
        (r"\b[ÖO]DENECEKTUTAR\b", "ÖDENECEK TUTAR"),
        (r"\b[ÖO]denecek\s*Tutar\b", "ÖDENECEK TUTAR"),
        (r"\b[ÖO]denecek\s*Tuter\b", "ÖDENECEK TUTAR"),
        (r"\benecel\s*Tutar\b", "ÖDENECEK TUTAR"),
        # Photo OCR mangling: Odesecck / Öfenecets / Ofesecck / Ödesectk …
        (r"\b[ÖO][A-Za-zçğıöşü]{5,14}\s+T[ou]tar\b", "ÖDENECEK TUTAR"),
        (r"\bVergies?\s*Dald\s*Teglam\s*Tutar\b", "Vergiler Dahil Toplam Tutar"),
        (r"\bVergiler\s*Dahil\s*Toplam\s*Tutar\b", "Vergiler Dahil Toplam Tutar"),
        (r"\bVerg[il]+[eo]r\s*Dah[il]+\s*Topl[ae]m\s*Tutar\b", "Vergiler Dahil Toplam Tutar"),
        (r"\bVarg[iı]kr\s*Dahil\s*Topla[ae]\s*Tutar\b", "Vergiler Dahil Toplam Tutar"),
        (r"\bVarg[iı]kr\s*Dahil\s*Toplam\s*Tutar\b", "Vergiler Dahil Toplam Tutar"),
        (r"\bWerglker\s*Dahil\s*Teplamn?\s*Tutar\b", "Vergiler Dahil Toplam Tutar"),
        (r"\bVergilker\s*Dahil\s*Teplamn?\s*Tutar\b", "Vergiler Dahil Toplam Tutar"),
        (r"\bBesaplaesnKOV\b", "Hesaplanan KDV"),
        (r"\bHesaplanan\s*K\.?\s*D\.?\s*V\.?\b", "Hesaplanan KDV"),
        (r"\bAlica\b", "Alıcı"),
        (r"\bAlici\b", "Alıcı"),
        (r"\bAlics\b", "Alıcı"),
        (r"\bAL[İI]C[İI]\b", "Alıcı"),
        (r"\bSatici\b", "Satıcı"),
        (r"\bSAT[İI]C[İI]\b", "Satıcı"),
        (r"\beArgiv\b", "e-Arşiv"),
        (r"\be-?Arpiv\b", "e-Arşiv"),
        (r"\be-?Arglv\b", "e-Arşiv"),
        (r"TEARETAS\.?", "TICARET A.S."),
        (r"\bTEARET\b", "TICARET"),
        (r"T[İI]CARET\s*A\.?\s*S\.?", "TICARET A.S."),
        (r"HAGAZACILIK", "MAGAZACILIK"),
        (r"MA[ČĆC]AZACILIK", "MAGAZACILIK"),
        (r"DA[ČĆC]IT[İI]M", "DAGITIM"),
        (r"\bXDV\b", "KDV"),
        (r"\bVIN\s*:", "VKN:"),
        (r"\bVIKN\b", "VKN"),
        (r"\bV[İI]KN\b", "VKN"),
        (r"\bVON\s*/\s*TOCN\b", "VKN/TCKN"),
        (r"\bVON\b", "VKN"),
        (r"\bTOCN\b", "TCKN"),
        (r"Kdv\s*Tutan\b", "Kdv Tutari"),
        (r"KDV\s*Tutan\b", "KDV Tutari"),
        (r"[ÖO]zellestirme\s*No", "Ozellestirme No"),
        (r"[ÖO]zelle[sş]tirme\s*No", "Ozellestirme No"),
        (r"Ozellogtirme\s*No", "Ozellestirme No"),
        (r"\b[ÖO]DENECEK\s*TUTAR\b", "ÖDENECEK TUTAR"),
        (r"Örün\s*/?\s*Hizmet\s+Toplam", "Mal Hizmet Toplam"),
        (r"Mal\s*Hizmet\s*Toplam\s*Tutan", "Mal Hizmet Toplam Tutari"),
        (r"Vergiler\s+Dahil\s+Toplam\s+Tutar:\s*12,\.000", "Vergiler Dahil Toplam Tutar: 12.000"),
        # Time OCR: 1838:00 → 18:38:00
        (r"\b(\d{2})(\d{2}):(\d{2})\b", r"\1:\2:\3"),
    )
    for pat, repl in replacements:
        text = re.sub(pat, repl, text, flags=re.I)
    # Glued label+amount: "ÖDENECEK TUTAR:12,000,00"
    text = re.sub(
        rf"([ÖO]DENECEK\s+TUTAR)\s*:?\s*({_MONEY_TOKEN})",
        r"\1: \2",
        text,
        flags=re.I,
    )
    # TR12 / TR1 2 → TR1.2 (Özelleştirme No)
    text = re.sub(
        r"(Ozellestirme\s+No)\s*:?\s*TR\s*([12])\s*[.,]?\s*([0-9])\b",
        r"\1: TR\2.\3",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"(Ozellestirme\s+No)\s*:?\s*TR([12])([0-9])\b",
        r"\1: TR\2.\3",
        text,
        flags=re.I,
    )
    return text


def labeled_amount(text: str, label: str) -> float | None:
    # Plain layout + OCR (comma/dot/space decimals) + optional TRY
    re_ = re.compile(
        rf"{label}(?:\s*\([^)]*\))?\s*:?\s*(?:\|+\s*)?({_MONEY_TOKEN})\s*(?:TL|TRY|T[L1])?",
        re.I,
    )
    matches = list(re_.finditer(text))
    if matches:
        # Prefer larger plausible invoice totals when multiple hits (avoid stray footnotes)
        amounts = [parse_tr_money(m.group(1)) for m in matches]
        amounts = [a for a in amounts if a is not None and a > 0]
        if amounts:
            lab_u = re.sub(r"[^A-Z0-9]", "", label.upper().replace("Ö", "O").replace("İ", "I"))
            if "ODENECEK" in lab_u or "DAHIL" in lab_u or "TOPLAM" in lab_u:
                return max(amounts)
            return amounts[-1]
    re_nl = re.compile(
        rf"{label}(?:\s*\([^)]*\))?\s*:?\s*\|?[^\n]{{0,40}}?\n+\s*\|?\s*({_MONEY_TOKEN})",
        re.I,
    )
    matches = list(re_nl.finditer(text))
    if matches:
        return parse_tr_money(matches[-1].group(1))
    return None


def normalize_vat_rate(vat_rate: float | None) -> float | None:
    """Map OCR-mangled rates (9620.00, 620.00, 96.2) to GİB standards."""
    if vat_rate is None:
        return None
    standard = (0.0, 1.0, 8.0, 10.0, 18.0, 20.0)
    if vat_rate in standard:
        return vat_rate
    for div in (100.0, 10.0):
        cand = round(vat_rate / div, 2)
        if cand in standard:
            return cand
    digits = re.sub(r"[^\d]", "", f"{vat_rate:.2f}")
    for token, val in (("20", 20.0), ("10", 10.0), ("18", 18.0), ("08", 8.0), ("01", 1.0)):
        if token in digits:
            return val
    return vat_rate if vat_rate <= 40 else None


def _append_ocr_try_row(
    out: list[Line],
    *,
    seq: str,
    name: str,
    qty_raw: str,
    unit_raw: str,
    disc_raw: str,
    vat_raw: str,
    total_raw: str,
) -> None:
    qty = parse_tr_money(qty_raw.replace(",", ".")) if "," in qty_raw or "." in qty_raw else None
    if qty is None:
        try:
            qty = float(qty_raw.replace(",", "."))
        except ValueError:
            qty = None
    # OCR often turns 1,0 into 10
    if qty is not None and qty >= 10 and qty == int(qty) and int(qty) % 10 == 0:
        maybe = qty / 10.0
        if 0 < maybe <= 9:
            qty = maybe
    unit_price = parse_tr_money(unit_raw)
    discount = parse_tr_money(disc_raw)
    vat_rate = normalize_vat_rate(parse_percent(vat_raw))
    if vat_rate is not None and vat_rate >= 100:
        maybe = vat_rate % 100
        if maybe in (1, 8, 10, 18, 20):
            vat_rate = float(maybe)
        elif str(int(vat_rate)).endswith("20"):
            vat_rate = 20.0
    line_total = parse_tr_money(total_raw)
    name_clean = re.sub(r"\s+", " ", name).strip(" -")
    # Skip asorti / colorway OCR junk used as product name
    if name_clean and re.fullmatch(r"(?:Asort[iıil]+\s*(?://\s*)?)+", name_clean, re.I):
        return
    if not name_clean or line_total is None:
        return
    # Dedup identical product+total rows (same barcode line repeated without sıra no)
    for prev in out:
        if prev.lineTotal == line_total and (prev.name or "")[:40] == name_clean[:40]:
            # Keep both when totals match but they are distinct line entries
            # (identical SKUs on GİB invoices) — only skip exact duplicate appends
            # from overlapping regexes on the same span.
            if prev.id == seq:
                return
    out.append(
        Line(
            id=seq,
            name=name_clean[:240],
            quantity=qty,
            unit="Adet",
            unitPrice=unit_price,
            discountAmount=discount,
            vatRate=vat_rate,
            lineTotal=line_total,
        )
    )


def infer_qty_unit_from_amounts(
    total: float,
    amounts: list[float],
    qty_hint: float | None = None,
) -> tuple[float, float | None]:
    """Infer quantity + unit price from line total and candidate amounts (generic)."""
    cands = [a for a in amounts if a is not None and 0.001 < a < total * 0.98]
    best: tuple[float, float, float] | None = None  # score, unit, qty
    for a in cands:
        ratio = total / a
        r = round(ratio)
        if r < 1 or r > 100000:
            continue
        if abs(ratio - r) > 0.025:
            continue
        score = abs(ratio - r)
        # Prefer matching explicit "N Adet" hint
        if qty_hint is not None and abs(r - qty_hint) < 0.01:
            score -= 0.5
        # Deprioritize VAT-like rates mistaken as unit price
        if a in (1.0, 8.0, 10.0, 18.0, 20.0) or (a < 40 and r > 40):
            score += 0.35
        # Prefer high-precision GİB unit prices (many decimals → larger magnitude often)
        if a >= 100:
            score -= 0.05
        cand = (score, a, float(r))
        if best is None or cand[0] < best[0]:
            best = cand
    if best is not None:
        return best[2], best[1]
    if qty_hint is not None and qty_hint >= 1:
        return float(qty_hint), round(total / qty_hint, 5)
    return 1.0, None


def parse_photo_amount_lines(text: str) -> list[Line]:
    """Photo/OCR rows: product name on one line, amounts (± N Adet) on nearby lines."""
    rows = text.splitlines()
    out: list[Line] = []
    skip_name = re.compile(
        r"(?i)(?:^|\b)(?:ARA\s*TOPLAM|TOPLAM|KDV|Mal\s*Hizmet|Vergi|ÖDEN|ODEN|ETT?N|Notlar?|"
        r"VKN|TCKN|Tel|Web|E-?Post|SAYIN|Senaryo|Sipari|Miktar|Birim\s*Fiyat|"
        r"Hesaplanan|Iskonto|Ödenecek)"
    )
    for i, ln in enumerate(rows):
        name = ln.strip()
        if len(name) < 6 or skip_name.search(name):
            continue
        if not re.search(r"[A-Za-zÇĞİÖŞÜçğıöşü]{3,}", name):
            continue
        # Table header soup (column titles glued)
        if re.search(r"(?i)Miktar.*(?:Birim|Fiyat)|KDV\s*Oran|Mal\s*Hizmet\s*Tutar", name):
            continue
        letters = re.sub(r"[^A-Za-zÇĞİÖŞÜçğıöşü]", "", name)
        if len(letters) < 4:
            continue
        window = "\n".join(rows[i : i + 3])
        # Require either Adet qty or a high-precision unit price in the window
        qty_m = re.search(r"(?<!\d)(\d{1,4})\s*Ade[t1l]?\b", window, re.I)
        hi_prec = re.search(r"\d{1,3}(?:\.\d{3})+,\d{3,8}", window)
        if not qty_m and not hi_prec:
            continue
        qty_hint = float(qty_m.group(1)) if qty_m else None
        money_hits = re.findall(rf"({_MONEY_TOKEN})\s*T?L?\b", window, re.I)
        money_hits += re.findall(r"(\d{1,3}(?:\.\d{3})+,\d{2,8})", window)
        amounts: list[float] = []
        for raw in money_hits:
            a = parse_tr_money(raw)
            if a is not None and a > 0:
                amounts.append(a)
        if len(amounts) < 2:
            continue
        total = max(amounts)
        if total < 10:
            continue
        # Line total should dominate (not a vat-only row)
        if total < 50 and qty_hint is None:
            continue
        qty, unit = infer_qty_unit_from_amounts(total, amounts, qty_hint)
        if unit is None:
            continue
        # Name quality: prefer product tokens over pure SKU dumps
        if re.match(r"^[\dA-F.\-/\s]{8,}$", name, re.I) and i > 0:
            prev = rows[i - 1].strip()
            if (
                re.search(r"[A-Za-zÇĞİÖŞÜçğıöşü]{3,}", prev)
                and not skip_name.search(prev)
                and not re.search(r"(?i)Miktar|Birim\s*Fiyat", prev)
            ):
                name = prev
        name_q = len(re.sub(r"[^A-Za-zÇĞİÖŞÜçğıöşü]", "", name))
        vowels = len(re.findall(r"[aeıioöuüAEIİOÖUÜ]", name, re.I))
        if name_q < 5 or vowels < 1:
            continue
        out.append(
            Line(
                id=str(len(out) + 1),
                name=re.sub(r"\s+", " ", name).strip()[:240],
                quantity=qty,
                unit="Adet",
                unitPrice=unit,
                lineTotal=total,
            )
        )
        if len(out) >= 8:
            break
    # Prefer the best name for each identical line total
    by_total: dict[float, Line] = {}
    for ln in out:
        t = float(ln.lineTotal or 0)
        prev = by_total.get(t)
        if prev is None:
            by_total[t] = ln
            continue
        def score(x: Line) -> int:
            n = x.name or ""
            letters = len(re.sub(r"[^A-Za-zÇĞİÖŞÜçğıöşü]", "", n))
            digits = len(re.sub(r"\D", "", n))
            money_pen = 40 if re.search(rf"{_MONEY_TOKEN}", n) else 0
            header_pen = 20 if re.search(r"(?i)Miktar|Birim|Oran|Tutar\s*$", n) else 0
            return letters - digits // 3 - money_pen - header_pen

        if score(ln) > score(prev):
            by_total[t] = ln
    dedup = list(by_total.values())
    for i, ln in enumerate(dedup, start=1):
        ln.id = str(i)
    return dedup


def normalize_company_legal_ocr(name: str) -> str:
    """Generic OCR repairs for Turkish legal-entity titles (not brand-specific)."""
    if not name:
        return name
    fixes = (
        (r"\bANONdM\b", "ANONİM"),
        (r"\bANONIM\b", "ANONİM"),
        (r"\bANON[İI]M\b", "ANONİM"),
        (r"\bS[İI]RKET[İI]\b", "ŞİRKETİ"),
        (r"\bSIRKETI\b", "ŞİRKETİ"),
        (r"\bSIRKET\b", "ŞİRKET"),
        (r"\bTEKNOLOI[tT]?\b", "TEKNOLOJİ"),
        (r"\bTEKNOLO(?![JİIjı])[İI]\b", "TEKNOLOJİ"),
        (r"\bTEKNOLOJI\b", "TEKNOLOJİ"),
        (r"\bTEKNOLOJ[İI]\b", "TEKNOLOJİ"),
        (r"BILISIM", "BİLİŞİM"),
        (r"B[İI]?LISIM", "BİLİŞİM"),
        (r"B[İI]L[İI]S[İI]M", "BİLİŞİM"),
        (r"\bLTD\.?\s*STI\b", "LTD. ŞTİ."),
        (r"\bLTD\.?\s*ŞTI\b", "LTD. ŞTİ."),
        (r"\bA\.?\s*S\.?\b", "A.Ş."),
    )
    out = name
    for pat, repl in fixes:
        out = re.sub(pat, repl, out, flags=re.I)
    return re.sub(r"\s+", " ", out).strip()


def parse_split_try_name_amount_lines(text: str) -> list[Line]:
    """Pair numbered product names with separate unit/discount/%/total TRY rows.

    Common phone-OCR GİB layout: names in one column block, amounts in another
    (or names above, amount rows below) — not on the same line.
    """
    # Amount-only: 2.241,6700 TRY 784,58 TRY %20.00 1.457,08 TRY
    amount_re = re.compile(
        rf"(?m)^(?P<up>{_MONEY_TOKEN})\s*TRY\s+"
        rf"(?P<disc>{_MONEY_TOKEN})\s*TRY\s+"
        rf"%?\s*(?P<vat>\d{{1,4}}(?:[.,]\d+)?)\s+"
        rf"(?P<total>{_MONEY_TOKEN})\s*TRY\b",
        re.I,
    )
    amounts = list(amount_re.finditer(text))
    if not amounts:
        return []

    # Numbered product lines without money on the same line
    name_re = re.compile(
        rf"(?m)^(?P<seq>\d{{1,3}})\s+"
        rf"(?P<name>(?=.*[A-Za-zÇĞİÖŞÜçğıöşü]{{3,}}).{{6,200}})$"
    )
    skip_name = re.compile(
        r"(?i)^(?:NET|TOPLAM|KDV|VERGI|ODENE|ÖDENE|ARA|S[ıi]ra|Mal\s*/?\s*Hizmet|"
        r"Birim\s*Fiyat|Iskonto|ETT?N|SAYIN|Fatura)"
    )
    names: list[tuple[str, str]] = []
    rows = text.splitlines()
    for i, ln in enumerate(rows):
        m = name_re.match(ln.strip())
        if not m:
            continue
        name = m.group("name").strip()
        if skip_name.search(name):
            continue
        if re.search(rf"{_MONEY_TOKEN}\s*TRY", name, re.I):
            continue
        # Append asorti / colorway continuations on following lines
        j = i + 1
        while j < len(rows):
            nxt = rows[j].strip()
            if not nxt:
                j += 1
                continue
            if name_re.match(nxt) or amount_re.match(nxt):
                break
            if re.search(r"(?i)Asort|//", nxt) and len(nxt) < 60:
                name = f"{name} {nxt}".strip()
                j += 1
                continue
            break
        names.append((m.group("seq"), re.sub(r"\s+", " ", name)[:240]))

    if not names:
        # Fallback: SKU-like lines without leading sıra no
        sku_re = re.compile(
            rf"(?m)^(?P<name>[A-Z]{{2,5}}[-_][A-Z0-9\-_/]{{3,}}(?:\s*-\s*\d{{8,14}})?"
            rf"(?:\s*-\s*.{{4,80}})?)$",
            re.I,
        )
        for m in sku_re.finditer(text):
            name = m.group("name").strip()
            if re.search(rf"TRY|{_MONEY_TOKEN}", name, re.I):
                continue
            names.append((str(len(names) + 1), name[:240]))

    if not names:
        return []

    out: list[Line] = []
    n = min(len(names), len(amounts))
    for i in range(n):
        seq, name = names[i]
        am = amounts[i]
        # Skip amount rows that are clearly summary (unit ≈ total and huge discount block)
        if re.search(r"(?i)NET\s*TOPLAM|TOPLAM\s*ISKONTO|ODENECEK|VERGI\s*DAHIL", name):
            continue
        _append_ocr_try_row(
            out,
            seq=seq,
            name=name,
            qty_raw="1",
            unit_raw=am.group("up"),
            disc_raw=am.group("disc"),
            vat_raw=am.group("vat"),
            total_raw=am.group("total"),
        )
    # Extra amount rows without names (same SKU repeats)
    for i in range(n, len(amounts)):
        am = amounts[i]
        # Reuse last product name when totals look like duplicate SKU lines
        name = names[-1][1] if names else f"Kalem {i + 1}"
        _append_ocr_try_row(
            out,
            seq=str(i + 1),
            name=name,
            qty_raw="1",
            unit_raw=am.group("up"),
            disc_raw=am.group("disc"),
            vat_raw=am.group("vat"),
            total_raw=am.group("total"),
        )
    return out


def parse_ocr_line_items(text: str) -> list[Line]:
    """Parse GİB-style flat OCR rows into invoice lines."""
    out: list[Line] = []
    # Split layout first (names ≠ amount rows) — common on phone screenshots
    split_lines = parse_split_try_name_amount_lines(text)
    if split_lines:
        return split_lines
    # qty unitPrice TRY discount TRY %vat lineTotal TRY
    row_re = re.compile(
        rf"^(\d{{1,3}})\s+(.+?)\s+"
        rf"(\d+[.,]\d+|\d+)\s+"
        rf"({_MONEY_TOKEN})\s*TRY\s+"
        rf"({_MONEY_TOKEN})\s*TRY\s+"
        rf"[%\$Ss]?\s*([\d.,]+)\s+"
        rf"({_MONEY_TOKEN})\s*[-.]?\s*TRY\b",
        re.I | re.M,
    )
    for m in row_re.finditer(text):
        seq, name, qty_raw, unit_raw, disc_raw, vat_raw, total_raw = m.groups()
        _append_ocr_try_row(
            out,
            seq=seq,
            name=name,
            qty_raw=qty_raw,
            unit_raw=unit_raw,
            disc_raw=disc_raw,
            vat_raw=vat_raw,
            total_raw=total_raw,
        )
    # Unnumbered GİB rows (OCR dropped sıra no) — same TRY layout
    unnumbered_re = re.compile(
        r"^(?!\d{1,3}\s)(?!NET\b|TOPLAM\b|KDV\b|VERGI\b|ODENE|ÖDENE|ARA\b)"
        r"([A-Z0-9][A-Z0-9ÇĞİÖŞÜa-zçğıöşü\-/ ]{6,}?)\s+"
        r"(\d+[.,]\d+|\d+)\s+"
        r"(" + _MONEY_TOKEN + r")\s*TRY\s+"
        r"(" + _MONEY_TOKEN + r")\s*TRY\s+"
        r"[%$Ss]?\s*([\d.,]+)\s+"
        r"(" + _MONEY_TOKEN + r")\s*[-.]?\s*TRY\b",
        re.I | re.M,
    )
    for m in unnumbered_re.finditer(text):
        name, qty_raw, unit_raw, disc_raw, vat_raw, total_raw = m.groups()
        if re.search(r"Asort[iıil]+\s*//", name, re.I) and len(name) < 40:
            continue
        next_id = str(len(out) + 1)
        _append_ocr_try_row(
            out,
            seq=next_id,
            name=name,
            qty_raw=qty_raw,
            unit_raw=unit_raw,
            disc_raw=disc_raw,
            vat_raw=vat_raw,
            total_raw=total_raw,
        )
    # Loose retail GİB row: description … unit TRY discount TRY %vat [vatAmt TRY] total TRY
    if len(out) < 1:
        loose_re = re.compile(
            r"(?m)^(?![-\s]*(?:NET|TOPLAM|KDV|VERGI|ODENE|ÖDENE|ARA|Mal\s*Hizmet\s*Toplam)\b)"
            r"(?P<name>(?=.*[A-Za-zÇĞİÖŞÜçğıöşü]{3,}).{12,160}?)\s+"
            r"(?P<up>" + _MONEY_TOKEN + r")\s*TRY\s+"
            r"(?P<disc>" + _MONEY_TOKEN + r")\s*TRY\s+"
            r"%\s*(?P<vat>\d{1,3}(?:[.,]\d+)?)\s+"
            r"(?:(" + _MONEY_TOKEN + r")\s*TRY\s+)?"
            r"(?P<total>" + _MONEY_TOKEN + r")\s*TRY\b",
            re.I,
        )
        for i, m in enumerate(loose_re.finditer(text), start=1):
            name = re.sub(r"\s+", " ", m.group("name")).strip(" -")
            if re.search(r"Toplam|Iskonto|Ödenecek|Vergi\s*Dahil", name, re.I):
                continue
            total = parse_tr_money(m.group("total"))
            if total is None or total < 1:
                continue
            vat_rate = normalize_vat_rate(parse_percent(m.group("vat")))
            out.append(
                Line(
                    id=str(i),
                    name=name[:240],
                    quantity=1.0,
                    unit="Adet",
                    unitPrice=parse_tr_money(m.group("up")),
                    discountAmount=parse_tr_money(m.group("disc")),
                    vatRate=vat_rate,
                    lineTotal=total,
                )
            )
    if out:
        # Re-number sequentially when mixed numbered/unnumbered
        for i, ln in enumerate(out, start=1):
            ln.id = str(i)
        return out

    # GİB table OCR: "1 21.560 kg 0,089TL … %18,00 345,39 TL 1.918,84"
    # Tolerates OCR junk between unit price and the final %KDV + amounts.
    gib_qty = re.compile(
        rf"(?m)^(?P<seq>\d{{1,3}})\s+"
        rf"(?P<name>.*?)\s*"
        rf"(?P<qty>\d{{1,3}}(?:[.,]\d+)?|\d+[.,]\d+)\s*"
        rf"(?P<unit>kg|adet|ad|NIU|C62|KGM|MTR|LTR)?\s*[|\]]?\s+"
        rf"(?P<unitPrice>{_MONEY_TOKEN})\s*TL?\s+"
        rf".{{0,80}}?"
        rf"%\s*(?P<vat>\d{{1,2}}(?:[.,]\d+)?)\s+"
        rf"(?P<vatAmt>{_MONEY_TOKEN})\s*TL?\s*[|\]]?\s*"
        rf"(?P<total>{_MONEY_TOKEN})",
        re.I,
    )
    for m in gib_qty.finditer(text):
        name = re.sub(r"\s+", " ", (m.group("name") or "")).strip(" -|]")
        if name and re.match(
            r"^(?:ARA|TOPLAM|KDV|Mal\s*Hizmet|Vergi|S[ıi]ra|No\b)", name, re.I
        ):
            continue
        if not name or len(name) < 2:
            name = f"Kalem {m.group('seq')}"
        # Drop OCR leftovers glued into name (unit-price fragments)
        if re.search(r"kg|TL|%\d", name, re.I) and len(name) > 40:
            name = f"Kalem {m.group('seq')}"
        total = parse_tr_money(m.group("total"))
        if total is None or total <= 0:
            continue
        qty = parse_tr_money(m.group("qty")) or float(m.group("qty").replace(",", "."))
        unit_price = parse_tr_money(m.group("unitPrice"))
        vat_rate = normalize_vat_rate(parse_percent(m.group("vat")))
        # Skip the first %0 discount rate if we grabbed it — prefer last % in row
        vat_candidates = re.findall(r"%\s*(\d{1,2}(?:[.,]\d+)?)", m.group(0))
        if vat_candidates:
            # Prefer standard GİB rates from the row
            for cand in reversed(vat_candidates):
                vr = normalize_vat_rate(parse_percent(cand))
                if vr in (1.0, 8.0, 10.0, 18.0, 20.0):
                    vat_rate = vr
                    break
        out.append(
            Line(
                id=m.group("seq"),
                name=name[:240],
                quantity=qty,
                unit=(m.group("unit") or "Adet"),
                unitPrice=unit_price,
                vatRate=vat_rate,
                vatAmount=parse_tr_money(m.group("vatAmt")),
                lineTotal=total,
            )
        )
    if out:
        return out

    # ------------------------------------------------------------------
    # Wide generic row patterns (layout-tolerant OCR fallback ladder)
    # 1 Klasik GİB  2 Retail/telefon  3 Miktar önce
    # 4 Miktarsız+%KDV  5 İsim+tutar  6 Serbest güvenlik ağı
    # ------------------------------------------------------------------
    _wide_skip_name = re.compile(
        r"(?i)^(?:S[ıi]ra|Mal\s*Hizmet|Birim\s*Fiyat|Miktar|A[çc][ıi]klama|AÇIKLAMA|"
        r"TOPLAM|KDV|ÖDEN|ODEN|Vergi|Not:|YALNIZ|ETTN|Fatura|Seri\s*No|"
        r"BIRIM\s*FIYAT|MIKTAR|TUTAR|İrsaliye|Ozellestirme|Özelleştirme|"
        r"e-?Ar[sş]iv|SAYIN|VKN|TCKN|Tel:|E-?Posta|Fiyat\s*Oran|"
        r"Oranı\s*Tutarı|Hizmet\s*Mal|D[ÜU]ZENLEME|F[İI]L[İI]\s*SEVK|"
        r"Tarih[iı]?|Saat|Senaryo|Tipi|No\s*:|Kredi\s*Kart|Banka\s*Kart|"
        r"Ara\s*Toplam|Genel\s*Toplam|Toplam\s*[İI]skonto|Matrah|"
        r"IBAN|İBAN|TR\d{2}|Banka\s*Hesap|Hesap\s*No|Hesap\s*Ad|"
        r"Ma[gğ]aza|Kasa(?:\s*No)?|Kasiyer|Sistem\s*No|Çekmece|"
        r"[ŞS]ube\s*Kod|[ŞS]ube\s*Ad|Swift|BIC|"
        r"Garanti|Albaraka|Yap[ıi]\s*Kredi|İş\s*Bank|Ziraat|Akbank|"
        r"Vak[ıi]fbank|Halkbank|Denizbank|QNB|Finansbank|TEB|ING)",
    )

    def _wide_append(
        *,
        name: str,
        qty: float | None,
        unit_price: float | None,
        vat_rate: float | None,
        vat_amt: float | None,
        total: float | None,
        unit: str = "Adet",
    ) -> None:
        if total is None or total < 1:
            return
        nm = re.sub(r"\s+", " ", (name or "")).strip(" -|")
        nm = re.sub(r"^\d{1,3}\s+", "", nm).strip()
        if not nm or len(nm) < 2:
            nm = f"Kalem {len(out) + 1}"
        if _wide_skip_name.search(nm) or _is_registry_or_chrome_line(nm):
            return
        if _is_bank_or_iban_line(nm):
            return
        if _is_amount_in_words_name(nm):
            return
        if re.search(r"(?i)Toplam|Iskonto|[İI]skonto|Ödenecek|Matrah|Kredi\s*Kart", nm):
            return
        out.append(
            Line(
                id=str(len(out) + 1),
                name=nm[:240],
                quantity=qty if qty and qty > 0 else 1.0,
                unit=unit or "Adet",
                unitPrice=unit_price,
                vatRate=vat_rate,
                vatAmount=vat_amt,
                lineTotal=total,
            )
        )

    # Pattern 1 — Klasik GİB (spaced or glued):
    # "1 DESC 1 Adet 14.583,33TL %20 2.916,67TL 14.583,33TL"
    # "1,0Adet14.583,33TL %20,002.916,67TL14.583,33TL"
    p1 = re.compile(
        rf"(?m)^(?:(?P<seq>\d{{1,3}})\s+)?"
        rf"(?P<name>.{{0,160}}?)\s*"
        rf"(?P<qty>\d{{1,3}}(?:[.,]\d+)?)\s*(?P<unit>Adet|adet|AD|kg|NIU|C62)?\s*"
        rf"(?P<unitPrice>{_MONEY_TOKEN})\s*TL?\s*"
        rf"%\s*(?P<vat>\d{{1,2}})(?:[.,]\d{{2}})?"
        rf"\s*(?P<vatAmt>{_MONEY_TOKEN})\s*TL?\s*"
        rf"(?P<total>{_MONEY_TOKEN})\s*TL?\s*$",
        re.I,
    )
    for m in p1.finditer(text):
        name = (m.group("name") or "").strip()
        if (
            not name
            or _wide_skip_name.search(name)
            or name.count("|") >= 1
            or re.fullmatch(r"[\d\s.,|AdetadetTL%-]+", name)
        ):
            name = _nearby_product_name(text, m.start()) or name
        if name.count("|") >= 1:
            continue
        _wide_append(
            name=name,
            qty=parse_tr_money(m.group("qty"))
            or float(m.group("qty").replace(",", ".")),
            unit_price=parse_tr_money(m.group("unitPrice")),
            vat_rate=normalize_vat_rate(parse_percent(m.group("vat"))),
            vat_amt=parse_tr_money(m.group("vatAmt")),
            total=parse_tr_money(m.group("total")),
            unit=(m.group("unit") or "Adet"),
        )
    if out:
        return out

    # Pattern 2 — Retail / telefon: name + unitPrice + qty + KDV(no %) + total
    # "APPLE IPHONE 15 128 BLACK 49.166,67 1 20 49.166,67"
    p2 = re.compile(
        rf"(?m)^(?P<name>(?=.*[A-Za-zÇĞİÖŞÜçğıöşü]{{3,}}).{{6,140}}?)\s+"
        rf"(?P<unitPrice>{_MONEY_TOKEN})\s+"
        rf"(?P<qty>\d{{1,4}}(?:[.,]\d+)?)\s+"
        rf"(?P<vat>\d{{1,2}}(?:[.,]\d+)?)\s+"
        rf"(?P<total>{_MONEY_TOKEN})\s*$"
    )
    for m in p2.finditer(text):
        _wide_append(
            name=m.group("name"),
            qty=parse_tr_money(m.group("qty"))
            or float(m.group("qty").replace(",", ".")),
            unit_price=parse_tr_money(m.group("unitPrice")),
            vat_rate=normalize_vat_rate(parse_percent(m.group("vat"))),
            vat_amt=None,
            total=parse_tr_money(m.group("total")),
        )
    if out:
        return out

    # Pattern 2b — amounts-only row; name from previous lines (Gürkan)
    p2b = re.compile(
        rf"(?m)^(?P<unitPrice>{_MONEY_TOKEN})\s+"
        rf"(?P<qty>\d{{1,4}}(?:[.,]\d+)?)\s+"
        rf"(?P<vat>\d{{1,2}})\s+"
        rf"(?P<total>{_MONEY_TOKEN})\s*$"
    )
    for m in p2b.finditer(text):
        _wide_append(
            name=_nearby_product_name(text, m.start()) or "",
            qty=parse_tr_money(m.group("qty"))
            or float(m.group("qty").replace(",", ".")),
            unit_price=parse_tr_money(m.group("unitPrice")),
            vat_rate=normalize_vat_rate(parse_percent(m.group("vat"))),
            vat_amt=None,
            total=parse_tr_money(m.group("total")),
        )
    if out:
        return out

    # Pattern 3 — Miktar önce: qty unit unitPrice … %vat … total
    p3 = re.compile(
        rf"(?m)^(?P<qty>\d{{1,4}}(?:[.,]\d+)?)\s*"
        rf"(?P<unit>Adet|adet|AD|kg|NIU|C62)\s+"
        rf"(?P<unitPrice>{_MONEY_TOKEN})\s*TL?\s+"
        rf"(?:.{{0,40}}?)?"
        rf"%\s*(?P<vat>\d{{1,2}}(?:[.,]\d+)?)\s+"
        rf"(?:(?P<vatAmt>{_MONEY_TOKEN})\s*TL?\s+)?"
        rf"(?P<total>{_MONEY_TOKEN})\s*TL?\s*$",
        re.I,
    )
    for m in p3.finditer(text):
        _wide_append(
            name=_nearby_product_name(text, m.start()) or "",
            qty=parse_tr_money(m.group("qty"))
            or float(m.group("qty").replace(",", ".")),
            unit_price=parse_tr_money(m.group("unitPrice")),
            vat_rate=normalize_vat_rate(parse_percent(m.group("vat"))),
            vat_amt=parse_tr_money(m.group("vatAmt")) if m.group("vatAmt") else None,
            total=parse_tr_money(m.group("total")),
            unit=m.group("unit") or "Adet",
        )
    if out:
        return out

    # Pattern 4 — Miktarsız + %KDV: name + money + %vat + money
    p4 = re.compile(
        rf"(?m)^(?P<name>(?=.*[A-Za-zÇĞİÖŞÜçğıöşü]{{3,}}).{{6,160}}?)\s+"
        rf"(?P<unitPrice>{_MONEY_TOKEN})\s*TL?\s+"
        rf"%\s*(?P<vat>\d{{1,2}}(?:[.,]\d+)?)\s+"
        rf"(?:(?P<vatAmt>{_MONEY_TOKEN})\s*TL?\s+)?"
        rf"(?P<total>{_MONEY_TOKEN})\s*TL?\s*$",
        re.I,
    )
    for m in p4.finditer(text):
        _wide_append(
            name=m.group("name"),
            qty=1.0,
            unit_price=parse_tr_money(m.group("unitPrice")),
            vat_rate=normalize_vat_rate(parse_percent(m.group("vat"))),
            vat_amt=parse_tr_money(m.group("vatAmt")) if m.group("vatAmt") else None,
            total=parse_tr_money(m.group("total")),
        )
    if out:
        return out

    # Pattern 4b — Retail e-ticaret: name %vat total (Media Markt)
    # "5007.1234605 APPLE KALEM & AKSESUARLARI  % 20  3.869,00"
    p4b = re.compile(
        rf"(?m)^(?P<name>(?=.*[A-Za-zÇĞİÖŞÜçğıöşü]{{3,}}).{{8,200}}?)\s+"
        rf"%\s*(?P<vat>\d{{1,2}}(?:[.,]\d+)?)\s+"
        rf"(?P<total>{_MONEY_TOKEN})\s*(?:TL|TRY)?\s*$",
        re.I,
    )
    rows_p4b = text.splitlines()
    for m in p4b.finditer(text):
        name = m.group("name")
        # Append SKU/model continuation on the next non-empty line
        line_idx = text[: m.start()].count("\n")
        for j in range(line_idx + 1, min(line_idx + 4, len(rows_p4b))):
            nxt = rows_p4b[j].strip()
            if not nxt:
                continue
            if re.search(rf"{_MONEY_TOKEN}|%\s*\d|KDV|Toplam|Kart", nxt, re.I):
                break
            if re.search(r"[A-Za-zÇĞİÖŞÜçğıöşü0-9]{3,}", nxt) and len(nxt) < 80:
                name = f"{name} {nxt}".strip()
            break
        _wide_append(
            name=name,
            qty=1.0,
            unit_price=parse_tr_money(m.group("total")),
            vat_rate=normalize_vat_rate(parse_percent(m.group("vat"))),
            vat_amt=None,
            total=parse_tr_money(m.group("total")),
        )
    if out:
        return out

    # Pattern 5 — Sadece isim + tutar (tek tutarlı satır)
    p5 = re.compile(
        rf"(?m)^(?P<name>(?=.*[A-Za-zÇĞİÖŞÜçğıöşü]{{3,}}).{{6,160}}?)\s+"
        rf"(?P<total>{_MONEY_TOKEN})\s*TL?\s*$",
        re.I,
    )
    for m in p5.finditer(text):
        name = m.group("name")
        if name.rstrip().endswith(":"):
            continue
        if re.search(r"(?i)%\d|\bKDV\b|\bAdet\b|\||Net\s*Mal|Değeri|Toplam|Kart", name):
            continue
        total = parse_tr_money(m.group("total"))
        if total is None or total < 20:
            continue
        _wide_append(
            name=name,
            qty=1.0,
            unit_price=total,
            vat_rate=None,
            vat_amt=None,
            total=total,
        )
    if out:
        return out

    # Pattern 5b — İsim / %KDV / tutar ayrı satırlarda (Docling/markdown)
    rows5b = [ln.rstrip() for ln in text.splitlines()]
    for i, raw in enumerate(rows5b):
        s = raw.strip()
        if not s or len(s) < 8:
            continue
        if _wide_skip_name.search(s) or _is_bank_or_iban_line(s):
            continue
        # SKU-like leading codes (5007.1234605) are OK; real amounts are not
        if _line_money_hits(s) or re.search(r"%\s*\d", s):
            continue
        if not re.search(r"[A-Za-zÇĞİÖŞÜçğıöşü]{3,}", s):
            continue
        if re.search(
            r"(?i)Toplam|KDV|Ödenecek|Matrah|Kart|IBAN|Net\s*Mal|Fatura\s*No|ETTN",
            s,
        ):
            continue
        name = s
        vat_rate = None
        total = None
        for j in range(i + 1, min(i + 6, len(rows5b))):
            nxt = rows5b[j].strip()
            if not nxt:
                continue
            vm = re.fullmatch(r"%\s*(\d{1,2}(?:[.,]\d+)?)", nxt)
            if vm:
                vat_rate = normalize_vat_rate(parse_percent(vm.group(1)))
                continue
            amt_m = re.fullmatch(rf"({_MONEY_TOKEN})\s*(?:TL|TRY)?", nxt, re.I)
            if amt_m:
                total = parse_tr_money(amt_m.group(1))
                break
            # Don't glue header values (T601) or new product rows onto a label name
            if s.rstrip().endswith(":") or _wide_skip_name.search(s):
                break
            # Model/SKU continuation before amounts
            if (
                not _line_money_hits(nxt)
                and not re.search(r"%\s*\d|KDV|Toplam|Kart", nxt, re.I)
                and not _wide_skip_name.search(nxt)
                and re.search(r"[A-Za-zÇĞİÖŞÜçğıöşü0-9]{3,}", nxt)
                and len(nxt) < 80
                and len(name) < 160
            ):
                name = f"{name} {nxt}".strip()
                continue
            break
        if total is None or total < 20:
            continue
        if s.rstrip().endswith(":") or _wide_skip_name.search(name):
            continue
        _wide_append(
            name=name,
            qty=1.0,
            unit_price=total,
            vat_rate=vat_rate,
            vat_amt=None,
            total=total,
        )
    if out:
        return out

    # Pattern 6 — Serbest güvenlik ağı: satırda money token, isim solda
    for ln in text.splitlines():
        s = ln.strip()
        if not s or len(s) < 8 or s.count("|") >= 2:
            continue
        if _wide_skip_name.search(s) or re.search(
            r"(?i)Mal\s*Hizmet\s*Toplam|Ödenecek|Hesaplanan\s*KDV|Vergiler\s*Dahil|"
            r"Net\s*Mal|Kredi\s*Kart|Banka\s*Kart|KDV\s*Toplam|KDV\s*Oran",
            s,
        ):
            continue
        if _is_bank_or_iban_line(s):
            continue
        money_hits = _line_money_hits(s)
        if len(money_hits) < 1:
            continue
        total = parse_tr_money(money_hits[-1].group(1))
        if total is None or total < 20:
            continue
        # Keep SKU prefix in name when first raw money was a product code
        name_end = money_hits[0].start()
        raw_first = next(re.finditer(rf"({_MONEY_TOKEN})", s), None)
        if raw_first and raw_first.start() < money_hits[0].start():
            name_end = 0  # include leading SKU in name; amount is later
            # Prefer text before the *amount* money (last hit), stripping %vat noise
            name = s[: money_hits[-1].start()]
            name = re.sub(r"%\s*\d{1,2}(?:[.,]\d+)?\s*$", "", name).strip(" -|")
        else:
            name = s[:name_end].strip(" -|")
        name = re.sub(r"\s+", " ", name)
        if name.endswith(":") or len(re.sub(r"[^A-Za-zÇĞİÖŞÜçğıöşü0-9]", "", name)) < 6:
            continue
        unit_price = (
            parse_tr_money(money_hits[0].group(1)) if len(money_hits) >= 2 else total
        )
        vat_rate = None
        vat_m = re.search(r"%\s*(\d{1,2}(?:[.,]\d+)?)", s)
        if vat_m:
            vat_rate = normalize_vat_rate(parse_percent(vat_m.group(1)))
        qty = 1.0
        qty_m = re.search(r"(\d+[.,]\d+|\d+)\s*(?:Adet|adet|kg)\b", s, re.I)
        if qty_m:
            qty = parse_tr_money(qty_m.group(1)) or float(
                qty_m.group(1).replace(",", ".")
            )
        before = len(out)
        _wide_append(
            name=name,
            qty=qty,
            unit_price=unit_price,
            vat_rate=vat_rate,
            vat_amt=None,
            total=total,
        )
        if len(out) > before and len(out) >= 8:
            break
    if out:
        return out

    # Photo OCR: product name line + amount line (qty via N Ade[t] or total÷unit)
    photo_lines = parse_photo_amount_lines(text)
    if photo_lines:
        return photo_lines

    # Numbered GİB row with trailing money tokens (description + amounts)
    # Also collect description-only rows (product text without prices on same line)
    skip_desc = re.compile(
        r"^(?:ARA|TOPLAM|KDV|Mal\s*Hizmet|Vergi|ÖDEN|ODEN|S[ıi]ra|No\b|ETTN|Notlar|Adet\b)",
        re.I,
    )
    desc_only: dict[str, str] = {}
    for m in re.finditer(rf"(?m)^(?P<seq>\d{{1,3}})\s+(?P<body>.{{6,200}})$", text):
        body = m.group("body").strip()
        if skip_desc.match(body):
            continue
        if re.search(rf"{_MONEY_TOKEN}\s*(?:TL|TRY)?", body, re.I):
            continue
        if re.match(r"^(?:Adet|kg)\b", body, re.I):
            continue
        desc_only[m.group("seq")] = re.sub(r"\s+", " ", body)[:240]

    numbered = re.compile(rf"(?m)^(?P<seq>\d{{1,3}})\s+(?P<body>.{{8,220}})$")
    skip_body = re.compile(
        r"^(?:ARA|TOPLAM|KDV|Mal\s*Hizmet|Vergi|ÖDEN|ODEN|S[ıi]ra|No\b|ETTN|Notlar)",
        re.I,
    )
    for m in numbered.finditer(text):
        body = m.group("body").strip()
        if skip_body.match(body):
            continue
        if re.match(r"^adet\s*[x×X]\b", body, re.I):
            continue
        money_hits = list(re.finditer(rf"({_MONEY_TOKEN})\s*(?:TL|TRY)?", body, re.I))
        if len(money_hits) < 1:
            continue
        total = parse_tr_money(money_hits[-1].group(1))
        if total is None or total < 1:
            continue
        # Ignore tiny footnote amounts misread as lines
        if total < 5 and len(money_hits) == 1:
            continue
        name_end = money_hits[0].start()
        name = body[:name_end].strip(" -|")
        name = re.sub(r"\s+", " ", name)
        if re.match(r"^(?:Adet|kg|NIU|C62)\b", name, re.I):
            name = desc_only.get(m.group("seq") or "", "") or f"Kalem {m.group('seq')}"
        # Drop pure header fragments
        if len(re.sub(r"[^A-Za-zÇĞİÖŞÜçğıöşü0-9]", "", name)) < 3:
            continue
        if re.search(r"Miktar|Birim\s*Fiyat|K\.?D\.?V|Oran|Tutar\s*$", name, re.I):
            continue
        unit_price = parse_tr_money(money_hits[-2].group(1)) if len(money_hits) >= 2 else None
        vat_rate = None
        vat_m = re.search(r"%\s*(\d{1,2}(?:[.,]\d+)?)", body)
        if vat_m:
            vat_rate = normalize_vat_rate(parse_percent(vat_m.group(1)))
        qty = 1.0
        qty_m = re.search(r"(\d+[.,]\d+|\d+)\s*(kg|adet|ad|NIU)\b", body, re.I)
        if qty_m:
            qty = parse_tr_money(qty_m.group(1)) or float(qty_m.group(1).replace(",", "."))
        elif re.match(r"^Adet\b", body, re.I):
            qty = 1.0
        out.append(
            Line(
                id=m.group("seq"),
                name=name[:240] or f"Kalem {m.group('seq')}",
                quantity=qty,
                unit=(qty_m.group(2) if qty_m else "Adet"),
                unitPrice=unit_price,
                vatRate=vat_rate,
                lineTotal=total,
            )
        )
        if len(out) >= 30:
            break
    if out:
        by_id: dict[str, Line] = {}
        for ln in out:
            key = ln.id or str(len(by_id))
            prev = by_id.get(key)
            if not prev:
                by_id[key] = ln
                continue
            if len(ln.name or "") > len(prev.name or "") and not re.match(
                r"^(?:Adet|kg|Kalem)\b", ln.name or "", re.I
            ):
                by_id[key] = ln
        return list(by_id.values())

    if out:
        return out

    # Split GİB OCR: qty rows then separate "Iskonto - %vat vatAmt total" rows
    qty_rows = list(
        re.finditer(
            rf"(?m)^(?P<seq>\d{{1,3}})\s+(?P<qty>\d+[.,]\d+|\d+)\s*"
            rf"(?P<unit>kg|adet|ad|NIU)?\s*[|\]]?\s*"
            rf"(?P<unitPrice>{_MONEY_TOKEN})?\s*TL?",
            text,
            re.I,
        )
    )
    amt_rows = list(
        re.finditer(
            rf"(?i)[İI]skonto\s*-?\s*%?\s*(?P<vat>\d{{1,2}}(?:[.,]\d+)?)\s+"
            rf"(?P<vatAmt>{_MONEY_TOKEN})\s*TL?\s+"
            rf"(?P<total>{_MONEY_TOKEN})",
            text,
        )
    )
    if qty_rows and amt_rows and len(amt_rows) >= max(1, len(qty_rows) - 1):
        for i, qm in enumerate(qty_rows):
            if i >= len(amt_rows):
                break
            am = amt_rows[i]
            total = parse_tr_money(am.group("total"))
            if total is None:
                continue
            out.append(
                Line(
                    id=qm.group("seq"),
                    name=f"Kalem {qm.group('seq')}",
                    quantity=parse_tr_money(qm.group("qty"))
                    or float(qm.group("qty").replace(",", ".")),
                    unit=(qm.group("unit") or "Adet"),
                    unitPrice=parse_tr_money(qm.group("unitPrice"))
                    if qm.group("unitPrice")
                    else None,
                    vatRate=normalize_vat_rate(parse_percent(am.group("vat"))),
                    vatAmount=parse_tr_money(am.group("vatAmt")),
                    lineTotal=total,
                )
            )
        if out:
            return out

    # Unnumbered product row: "DESC 1.453,70 1.463,70"
    bare = re.compile(
        rf"(?m)^(?P<name>[A-ZÇĞİÖŞÜa-zçğıöşü0-9][A-ZÇĞİÖŞÜa-zçğıöşü0-9 /.\-]{{4,80}}?)\s+"
        rf"(?P<a>{_MONEY_TOKEN})\s+(?P<b>{_MONEY_TOKEN})\s*$"
    )
    skip_bare = re.compile(
        r"^(?:ARA|TOPLAM|Mal\s*Hizmet|Vergi|Toplam|Odenecek|ÖDEN|KDV|CARD|No\b)",
        re.I,
    )
    for i, m in enumerate(bare.finditer(text), start=1):
        name = m.group("name").strip()
        if skip_bare.search(name):
            continue
        a = parse_tr_money(m.group("a"))
        b = parse_tr_money(m.group("b"))
        if a is None or b is None:
            continue
        total = max(a, b)
        unit = min(a, b)
        if total < 1:
            continue
        out.append(
            Line(
                id=str(i),
                name=name[:240],
                quantity=1.0,
                unit="Adet",
                unitPrice=unit,
                lineTotal=total,
            )
        )
        if len(out) >= 12:
            break
    if out:
        return out

    # Thermal / POS print: barcode name vat qty unitPrice discount gross net
    thermal_re = re.compile(
        rf"(?m)^(?P<barcode>[A-Z0-9]{{8,14}})\s+(?P<name>.+?)\s+"
        rf"(?P<vat>\d{{1,2}}[.,]\d{{2}})\s+"
        rf"(?P<qty>\d+[.,]\d+)\s+"
        rf"(?P<unit>{_MONEY_TOKEN}|1\.\d{{3}}\.\d{{2,5}})\s+"
        rf"(?P<disc>{_MONEY_TOKEN}|E0O|0+)\s+"
        rf"(?P<gross>{_MONEY_TOKEN}|1\.\d{{3}}\.\d{{2,5}})\s+"
        rf"(?P<net>{_MONEY_TOKEN}|1\.\d{{3}}\.\d{{2,5}}|\d+\.\d{{2,5}})\s*$",
        re.I,
    )
    for i, m in enumerate(thermal_re.finditer(text), start=1):
        disc_raw = m.group("disc")
        if re.fullmatch(r"E0O|0+", disc_raw or "", re.I):
            disc = 0.0
        else:
            disc = parse_tr_money(disc_raw)
        unit = parse_tr_money(m.group("unit"))
        gross = parse_tr_money(m.group("gross"))
        net = parse_tr_money(m.group("net"))
        # Prefer unit as net when discount ~0 and OCR net is nonsense
        if unit is not None and (disc or 0) == 0 and (net is None or net < unit * 0.5 or (gross and net > gross * 2)):
            net = unit
        elif net is not None and unit is not None and net > unit * 1.5 and gross is not None and abs(gross - unit * 1.1) < unit:
            net = unit if (disc or 0) == 0 else round(unit - (disc or 0), 2)
        if net is None and unit is not None:
            net = round(unit - (disc or 0), 2) if disc else unit
        # OCR KDV 16 → likely 10 for low-rate goods when unit ~118
        vat_rate = normalize_vat_rate(parse_percent(m.group("vat")))
        if vat_rate == 16.0:
            vat_rate = 10.0
        name = re.sub(r"\s+", " ", m.group("name")).strip()
        if not name or net is None:
            continue
        out.append(
            Line(
                id=str(i),
                name=name[:240],
                quantity=parse_tr_money(m.group("qty")) or 1.0,
                unit="Adet",
                unitPrice=unit,
                discountAmount=disc,
                vatRate=vat_rate,
                lineTotal=net,
            )
        )
    return out


_KNOWN_INVOICE_TYPES = (
    "TEVKIFATIADE",
    "OZELMATRAH",
    "KONAKLAMA",
    "TEVKIFAT",
    "ISTISNA",
    "KOMISYON",
    "SATIS",
    "IADE",
    "SGK",
    "HKS",
)


def normalize_invoice_type(raw: str | None) -> str | None:
    """Map OCR/glued PDF soup to a short GİB InvoiceTypeCode (SATIS, IADE, …)."""
    if not raw:
        return None
    t = (
        raw.upper()
        .replace("İ", "I")
        .replace("Ş", "S")
        .replace("Ğ", "G")
        .replace("Ü", "U")
        .replace("Ö", "O")
        .replace("Ç", "C")
    )
    t = re.sub(r"[^A-Z0-9_]", "", t)
    # OCR: SATTS / SATI5 → SATIS
    if t in {"SATTS", "SATI5", "SAT1S", "SATIS"}:
        return "SATIS"
    # Glued PDF: "SATISEPOSTABILGIARCELIK…" → SATIS
    for code in _KNOWN_INVOICE_TYPES:
        if t == code or t.startswith(code):
            return code
    if len(t) > 24:
        return None
    return t or None


def normalize_customization_id(raw: str | None) -> str | None:
    """Keep only TR1.2 / TR1.0 style ids; drop glued branch/phone soup."""
    if not raw:
        return None
    s = re.sub(r"\s+", "", raw.upper().replace("İ", "I"))
    m = re.match(r"(TR[12](?:\.\d)?)", s)
    if not m:
        m = re.match(r"TR([12])(\d)", s)
        if m:
            return f"TR{m.group(1)}.{m.group(2)}"
        return None
    got = m.group(1)
    return re.sub(r"^TR([12])(\d)$", r"TR\1.\2", got)


def sanitize_tax_office(raw: str | None) -> str | None:
    """Trim Vergi Dairesi to a short office name; drop glued line-item soup."""
    if not raw:
        return None
    s = re.split(
        r"\s{2,}|Vergi\s*(?:No|Num|Kimlik)|VKN\s*/?\s*TCKN|\bVKN\b|\bTCKN\b|"
        r"\bETTN\b|Malzeme|Fatura\s*No|Tel\s*:|E-?Posta|MERS[İI]S",
        raw,
        maxsplit=1,
        flags=re.I,
    )[0]
    s = re.sub(r"(?i)\s*Vergi\s*No\s*$", "", s).strip(" :.-|,/")
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) < 2 or len(s) > 48:
        return None
    if re.search(r"\d{7,}|Malzeme|Adet|ETTN|SATIS", s, re.I):
        return None
    return s


def parse_percent(raw: str | None) -> float | None:
    if not raw:
        return None
    m = re.search(r"(\d+(?:[.,]\d+)?)", raw.replace(",", "."))
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def nearly_equal(a: float, b: float, eps: float = 0.05) -> bool:
    return abs(a - b) <= eps


def status_from(warnings: list[str], validation: Validation) -> Literal["ok", "partial", "failed"]:
    # Soft: ETTN / alıcı tax / unvan / kuruş — shouldn't alone force weak status
    soft = [
        w
        for w in warnings
        if re.search(
            r"uyuşmuyor|0\.0[12]|kuruş|ETTN bulunamadı|"
            r"Alıcı\s+(?:VKN|TCKN|unvanı)|Alıcı vergi",
            w,
            re.I,
        )
    ]
    hard = [w for w in warnings if w not in soft]
    # Primary weight: fatura no + ödenecek; supplier / lines also material
    critical = [
        w
        for w in hard
        if re.search(
            r"Fatura numarası|Ödenecek tutar|Satıcı|kalemi|Fatura tarihi",
            w,
        )
    ]
    if not hard and validation.confidence >= 0.8:
        return "ok"
    if critical:
        return "partial"
    if validation.confidence < 0.5:
        return "partial"
    return "ok" if validation.confidence >= 0.75 else "partial"


def has_any_invoice_field(inv: Invoice) -> bool:
    """True when at least one useful invoice field was bound (no empty-'failed')."""
    if inv.invoiceNumber or inv.uuid or inv.issueDate:
        return True
    if inv.totals.payableAmount is not None or inv.totals.taxInclusiveAmount is not None:
        return True
    if inv.supplier.taxId or inv.customer.taxId:
        return True
    if inv.lines:
        return True
    if inv.supplier.name or inv.customer.name:
        return True
    return False


def sniff_extension(data: bytes, filename: str) -> str:
    """Return a filesystem extension including the leading dot."""
    name_ext = Path(filename).suffix.lower()
    if data[:4] == b"%PDF":
        return ".pdf"
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[:2] in (b"II", b"MM") and len(data) > 3:
        return ".tiff"
    if data[:4] == b"BM":
        return ".bmp"
    # HEIC/HEIF brand in ftyp box
    if len(data) >= 12 and data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in (b"heic", b"heif", b"mif1", b"msf1", b"heix", b"heim"):
            return ".heic"
    if name_ext in IMAGE_EXTENSIONS or name_ext == ".pdf":
        return name_ext
    return name_ext or ".bin"


def is_image_ext(ext: str) -> bool:
    return ext.lower() in IMAGE_EXTENSIONS


def convert_heic_to_jpeg(src: Path, dest: Path) -> None:
    """Best-effort HEIC → JPEG for Docling (no native HEIC in many builds)."""
    try:
        from pillow_heif import register_heif_opener  # type: ignore

        register_heif_opener()
        from PIL import Image

        with Image.open(src) as im:
            rgb = im.convert("RGB")
            rgb.save(dest, format="JPEG", quality=92)
            return
    except Exception:
        pass
    for cmd in (
        ["heif-convert", str(src), str(dest)],
        ["magick", str(src), str(dest)],
        ["convert", str(src), str(dest)],
    ):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
            if r.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
                return
        except Exception:
            continue
    raise RuntimeError(
        "HEIC fotoğraf dönüştürülemedi. Lütfen JPG/PNG olarak kaydedip tekrar yükleyin."
    )


def pdftotext(path: Path) -> str:
    r = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or "pdftotext failed")
    return r.stdout or ""


def extract_pdf_plain_text(path: Path) -> tuple[str, list[str]]:
    """Collect pdf-inspector + pdftotext (both when useful). Returns (text, tags).

    Text is inspector-first, then pdftotext appended — callers that only parse once
    still see both layouts. Prefer extract_pdf_texts + merge for best fields.
    """
    texts, tags = extract_pdf_texts(path)
    if not texts:
        return "", tags
    # Prefer longer / richer blob for single-parse callers.
    return "\n\n".join(texts), tags


def extract_pdf_texts(path: Path) -> tuple[list[str], list[str]]:
    """Return ([text...], pipeline tags) from pdf-inspector and/or pdftotext."""
    tags: list[str] = []
    texts: list[str] = []

    if PDF_INSPECTOR_ENABLED:
        try:
            from pdf_inspector_text import available, extract_pdf_inspector

            if available():
                text, meta = extract_pdf_inspector(path)
                src = meta.get("source") or "unknown"
                tags.append(f"pdf-inspector:{src}")
                if meta.get("pdfType"):
                    tags.append(f"pdf-inspector-type:{meta['pdfType']}")
                if text and not is_unusable_extract_text(text):
                    texts.append(text)
                elif text:
                    tags.append("pdf-inspector-unusable")
            else:
                tags.append("pdf-inspector-unavailable")
        except Exception as exc:  # noqa: BLE001
            tags.append(f"pdf-inspector-error:{exc}")

    try:
        text = pdftotext(path)
        tags.append("pdftotext")
        if is_unusable_extract_text(text):
            tags.append("pdftotext-unusable")
        else:
            texts.append(text)
    except Exception as exc:  # noqa: BLE001
        tags.append(f"pdftotext-error:{exc}")

    return texts, tags


def is_unusable_extract_text(text: str) -> bool:
    """True when text is empty, Docling image stubs only, or CID/binary soup."""
    if not text or not text.strip():
        return True
    stripped = text.strip()
    without_img = re.sub(r"<!--\s*image\s*-->", "", stripped, flags=re.I).strip()
    without_img = re.sub(r"&lt;|--+|#{1,6}\s*", " ", without_img)
    if len(re.sub(r"\s+", "", without_img)) < 40:
        return True
    controls = sum(1 for c in stripped if ord(c) < 32 and c not in "\n\r\t")
    if controls >= 15 or (len(stripped) > 80 and controls / max(len(stripped), 1) > 0.06):
        return True
    letters = sum(1 for c in stripped if c.isalpha())
    if len(stripped) > 120 and letters < 20:
        return True
    return False


def needs_ocr_escalation(
    inv: Invoice, validation: Validation, warnings: list[str]
) -> bool:
    """Generic field-quality gate — escalate to VL when cheap OCR/parse is weak.

    No supplier/template rules: only critical-field presence, confidence,
    status, and coarse amount consistency.
    """
    if not inv.invoiceNumber:
        return True
    if inv.totals.payableAmount is None:
        return True
    if not (inv.supplier.taxId or inv.customer.taxId):
        return True
    if validation.confidence < 0.85:
        return True
    st = status_from(warnings, validation)
    if st in ("partial", "failed"):
        return True
    # Totals / tax warnings mean RapidOCR may be self-consistent but wrong
    if any(re.search(r"(?i)uyu[sş]muyor|bulunamad[ıi]|eksik", w) for w in warnings):
        return True
    pay = float(inv.totals.payableAmount or 0)
    line_sum = sum(float(l.lineTotal or 0) for l in inv.lines if l.lineTotal is not None)
    if pay >= 100 and not inv.lines:
        return True
    if line_sum >= 50 and pay > 0:
        ratio = pay / line_sum if line_sum else 0
        # payable far below / above line sum → likely wrong total binding
        if ratio < 0.4 or ratio > 2.5:
            return True
    le, vat = inv.totals.lineExtensionAmount, inv.totals.vatAmount
    if le is not None and vat is not None and pay > 0:
        expected = float(le) + float(vat)
        if expected >= 50 and abs(pay - expected) / expected > 0.15:
            return True
    ti = inv.totals.taxInclusiveAmount
    if ti is not None and pay > 0 and float(ti) >= 50:
        if abs(pay - float(ti)) / float(ti) > 0.15 and (inv.totals.withholdingVatAmount or 0) == 0:
            return True
    # junk party names that slipped through
    for nm in (inv.supplier.name, inv.customer.name):
        if nm and re.match(
            r"(?i)^(?:e-Belge|table|image|text|Nihai\s*T|ERP\s*Fatura|Özelleştirme|UBL)\b",
            nm,
        ):
            return True
    return False


def pdf_raster_ocr_text(
    pdf_path: Path,
    tmp: Path,
    max_pages: int = 2,
    prefer_vl: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Render PDF pages; RapidOCR by default, VL only on escalation (prefer_vl)."""
    out_prefix = tmp / ("raster_vl" if prefer_vl else "raster")
    r = subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-r",
            str(PDF_RASTER_DPI),
            "-f",
            "1",
            "-l",
            str(max_pages),
            str(pdf_path),
            str(out_prefix),
        ],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    pages = sorted(tmp.glob(f"{out_prefix.name}*.png"))
    if not pages:
        raise RuntimeError(r.stderr.strip() or "pdftoppm produced no pages")

    chunks: list[str] = []
    meta: dict[str, Any] = {
        "pages": 0,
        "engine": None,
        "elapsedMs": 0,
        "dpi": PDF_RASTER_DPI,
        "preferVl": prefer_vl,
    }

    for page in pages[:max_pages]:
        page_text = ""
        page_meta: dict[str, Any] = {}
        if prefer_vl and VL_OCR_ENABLED:
            try:
                from vl_ocr import ocr_document

                page_text, page_meta = ocr_document(page)
                _metrics["vl_ocr"] += 1
            except Exception as exc:  # noqa: BLE001
                page_meta = {"vlError": str(exc)}
        if not page_text.strip():
            from photo_ocr import ocr_image

            page_text, rapid_meta = ocr_image(page)
            page_meta = {
                **page_meta,
                **rapid_meta,
                "fallbackFrom": "vl" if prefer_vl and VL_OCR_ENABLED else None,
            }
        if page_text.strip():
            chunks.append(page_text.strip())
        meta["pages"] = int(meta["pages"]) + 1
        meta["engine"] = page_meta.get("engine") or meta.get("engine")
        meta["elapsedMs"] = int(meta.get("elapsedMs") or 0) + int(
            page_meta.get("elapsedMs") or 0
        )
        if page_meta.get("vlError"):
            meta["vlError"] = page_meta["vlError"]
    return "\n\n".join(chunks), meta


def extract_embedded_ubl(data: bytes) -> str | None:
    latin = data.decode("latin-1", errors="ignore")
    start = -1
    for marker in ("<?xml", "<Invoice", "<cbc:Invoice"):
        i = latin.find(marker)
        if i >= 0 and (start < 0 or i < start):
            start = i
    if start < 0:
        return None
    slice_ = latin[start : start + 2_000_000]
    m = re.search(r"</(?:\w+:)?Invoice>", slice_, re.I)
    if not m:
        return None
    xml = slice_[: m.end()]
    if not re.search(r"CustomizationID|ProfileID|AccountingSupplierParty", xml, re.I):
        return None
    return xml.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore")


def right_field(text: str, label: str) -> str | None:
    m = re.search(rf"{label}\s*:?\s*([^\n]+)", text, re.I)
    if not m:
        # Label on one line, value on next (common OCR): "Fatura No\n:BBE-..."
        m = re.search(rf"{label}\s*:?\s*\n\s*:?\s*([^\n]+)", text, re.I)
    if not m:
        # Markdown table: | Label: | value |
        m = re.search(
            rf"\|\s*{label}\s*:?\s*\|?\s*([^|\n]+)\|?",
            text,
            re.I,
        )
    if not m:
        return None
    raw = m.group(1).strip()
    raw = re.sub(r"&#124;|&nbsp;|&amp;", " ", raw)
    raw = re.sub(r"[|`\[\]]+", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip(" :.-")
    parts = [p.strip() for p in re.split(r"\s{2,}", raw) if p.strip()]
    return (parts[-1] if parts else raw) or None


def first_match(text: str, pattern: str, flags: int = re.I) -> str | None:
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else None


def first_match_money(text: str, pattern: str) -> float | None:
    m = re.search(pattern, text, re.I | re.M)
    return parse_tr_money(m.group(1)) if m else None


def sum_labeled_amounts(text: str, label: str) -> float | None:
    re_ = re.compile(
        rf"{label}(?:\s*\([^)]*\))?\s*:?\s*(?:\|+\s*)?({_MONEY_TOKEN})\s*(?:TL|TRY)?",
        re.I,
    )
    amounts = [parse_tr_money(m.group(1)) for m in re_.finditer(text)]
    amounts = [a for a in amounts if a is not None and a > 0]
    if not amounts:
        return None
    if len(amounts) == 1:
        return amounts[0]
    return round(sum(amounts), 2)


def extract_vat_amount(text: str) -> float | None:
    """Sum multi-rate KDV rows: KDV (%10) + KDV (%20). Prefer footnotes when present."""
    footnote_re = re.compile(
        rf"Kdv\s*Tutar[ıian]?\s*:?\s*({_MONEY_TOKEN})",
        re.I,
    )
    footnotes = [parse_tr_money(m.group(1)) for m in footnote_re.finditer(text)]
    footnotes = [a for a in footnotes if a is not None and a > 0]
    # Prefer multi-rate footnote sum (keep duplicates that are distinct rates)
    if len(footnotes) >= 2:
        return round(sum(footnotes), 2)
    if footnotes:
        return round(footnotes[0], 2)

    rate_re = re.compile(
        rf"(?:Hesaplanan\s+)?[KX]DV(?!\s*(?:TEVK|Tevkifat|Matrah[ıi]?))"
        rf"(?:\s*\(\s*%?\s*[\d.,]+\s*%?\s*\))\s*:?\s*({_MONEY_TOKEN})",
        re.I,
    )
    rate_amounts = [parse_tr_money(m.group(1)) for m in rate_re.finditer(text)]
    rate_amounts = [a for a in rate_amounts if a is not None and a > 0]
    if rate_amounts:
        return round(sum(rate_amounts), 2)

    return labeled_amount(text, r"Hesaplanan KDV(?!\s*Tevkifat)") or labeled_amount(
        text, r"[KX]DV(?!\s*(?:TEVK|Tevkifat|Matrah))"
    )


def _is_registry_or_chrome_line(ln: str) -> bool:
    """Reject GİB registry / viewer chrome mistaken for party names."""
    return bool(
        re.search(
            r"(?:T[İI]CARET\s*S[İI]C[İI]L|TICARETSICIL|MERS[İI]S\s*NO|MERSISNO|"
            r"e-?Ar[sş]iv\s+Fatura|e-?Belge\b|Detay\s*Ekran|Nolu\s+\w+\s+Fatura|"
            r"^Sayfa\s+\d+|^\d{1,2}:\d{2}\b|KB/s|isteerp\.com|https?://|file://|"
            r"Özelleştirme\s*No|Ozellestirme\s*No|UBL\s*Versiyon|ERP\s*Fatura|"
            r"^Nihai\s*T|^table$|^image$|^text$|^header$|"
            r"\b(?:table|image|header|footer|paragraph_title|figure_title)\b|"
            r"YALNIZ\b|ÜçBin|ElliDokuzBin|onyedi|beşyüz|"
            r"Rar\$EX|AppData\\Local\\Temp)",
            ln,
            re.I,
        )
    )


# IBAN: spaced / compact / loosely-split TR + foreign (DE, …)
# Examples:
#   TR86 0001 0000 0000 0000 0000 00
#   TR860001000000000000000000
#   TR 86 00010 0 0000000000000000
#   TR08 0006 2000 3820 0006 2966 83
#   DE89 3704 0044 0532 0130 00
_IBAN_RE = re.compile(
    r"(?ix)"
    r"(?:^|[^\w])"  # avoid matching inside longer alphanumerics
    r"(?:"
    # Spaced TR: TR86 0001 0000 … (groups of 4, optional trailing 00)
    r"TR\s*\d{2}(?:\s+\d{4}){5}\s*\d{0,4}"
    r"|"
    # Compact TR: TR + 24 digits
    r"TR\d{24}"
    r"|"
    # Split / OCR-noisy TR: TR + check + remaining digits with arbitrary spaces
    r"TR\s*\d{2}(?:\s*\d){20,24}"
    r"|"
    # Foreign IBAN (2-letter country ≠ TR): CC + check + 11–30 alnum/spaces
    r"(?!TR)[A-Z]{2}\s*\d{2}(?:[\s]*[A-Z0-9]){11,30}"
    r")"
)


def _is_bank_or_iban_line(ln: str) -> bool:
    """Reject bank/IBAN/payment rows mistaken for product line items."""
    if not ln:
        return False
    if _IBAN_RE.search(ln):
        return True
    return bool(
        re.search(
            r"(?i)"
            r"\bIBAN\b|\bİBAN\b|"
            r"\bBanka\s*Hesap|\bHesap\s*No\b|\bHesap\s*Ad[ıi]\b|"
            r"\b[ŞS]ube\s*Kod|\b[ŞS]ube\s*Ad|\bSwift\b|\bBIC\b|"
            r"Kredi\s*Kart[ıi]?|Banka\s*Kart[ıi]?|Kart[ıi]/\s*Banka|"
            r"\b(?:Garanti|Albaraka|Yap[ıi]\s*Kredi|İş\s*Bank|Is\s*Bank|"
            r"Ziraat|Akbank|Vak[ıi]fbank|Vakifbank|Halkbank|Denizbank|"
            r"QNB|Finansbank|TEB|ING|Şekerbank|Sekerbank)\b",
            ln,
        )
    )


def _is_amount_in_words_name(name: str | None) -> bool:
    """True when a party/product name is actually tutar-yazısı (YALNIZ … TL)."""
    if not name:
        return False
    return bool(
        re.search(
            r"(?i)^YALNIZ\b|YALNIZCA\b|"
            r"(?:Üç|Uc|Bir|İki|Iki|Dört|Dort|Beş|Bes|Altı|Alti|Yedi|Sekiz|Dokuz|On)"
            r"\w*(?:Bin|Yüz|Yuz|Milyon)\w*TL|"
            r"\b(?:Bin|Yüz|Yuz)[A-Za-zÇĞİÖŞÜçğıöşü]*TL\b|"
            r"ElliDokuz|onyedi|beşyüz|dort\s*y[uü]z",
            name,
        )
    )


def _nearby_product_name(text: str, pos: int, *, lookback: int = 900) -> str | None:
    """Pick a product description line just above an amount row."""
    chunk = text[max(0, pos - lookback) : pos]
    lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
    skip = re.compile(
        r"(?i)^(?:S[ıi]ra|Mal\s*Hizmet|Birim\s*Fiyat|Miktar|A[çc][ıi]klama|AÇIKLAMA|"
        r"TOPLAM|KDV|ÖDEN|ODEN|Vergi|Not:|YALNIZ|ETTN|Fatura|Seri\s*No|"
        r"BIRIM\s*FIYAT|MIKTAR|TUTAR|İrsaliye|Ozellestirme|Özelleştirme|"
        r"e-?Ar[sş]iv|SAYIN|VKN|TCKN|Tel:|E-?Posta|Fiyat\s*Oran|"
        r"Oranı\s*Tutarı|Hizmet\s*Mal|D[ÜU]ZENLEME|F[İI]L[İI]\s*SEVK|"
        r"Tarih[iı]?|Saat|Senaryo|Tipi|No\s*:)",
    )
    scored: list[tuple[int, str]] = []
    for idx, ln in enumerate(lines):
        if skip.search(ln) or _is_registry_or_chrome_line(ln) or _is_bank_or_iban_line(ln):
            continue
        if re.fullmatch(rf"{_MONEY_TOKEN}", ln):
            continue
        if re.search(rf"{_MONEY_TOKEN}", ln) and not re.search(
            r"[A-Za-zÇĞİÖŞÜçğıöşü]{3,}", ln
        ):
            continue
        letters = re.sub(r"[^A-Za-zÇĞİÖŞÜçğıöşü0-9]", "", ln)
        if len(letters) < 4:
            continue
        # Join wrapped product lines (PROFILO … / DVBS2 … / TV)
        parts = [ln]
        # Include previous line when current looks like a wrap continuation
        if idx > 0:
            prev = lines[idx - 1]
            if (
                not skip.search(prev)
                and not _is_registry_or_chrome_line(prev)
                and not re.search(rf"{_MONEY_TOKEN}", prev)
                and re.search(r"[A-Za-zÇĞİÖŞÜçğıöşü0-9]{3,}", prev)
            ):
                parts = [prev, ln]
        for nxt in lines[idx + 1 : idx + 4]:
            if skip.search(nxt) or _is_registry_or_chrome_line(nxt):
                break
            if re.search(rf"{_MONEY_TOKEN}", nxt):
                break
            if len(re.sub(r"[^A-Za-zÇĞİÖŞÜçğıöşü0-9]", "", nxt)) < 2:
                break
            parts.append(nxt)
            if len(" ".join(parts)) > 120:
                break
        name = re.sub(r"\s+", " ", " ".join(parts)).strip(" -|")
        name = re.sub(r"^\d{1,3}\s+", "", name).strip()
        if len(name) < 4 or skip.search(name):
            continue
        # Prefer closer lines with model-like tokens
        dist = len(lines) - idx
        score = 100 - dist
        if re.search(r"[A-Z0-9]{4,}", name):
            score += 20
        if re.search(r"(?i)IPHONE|SAMSUNG|APPLE|PROFILO|LED|TV|PENCIL", name):
            score += 40
        if re.search(r"(?i)Tarih|Saat|Düzenleme|Fatura|Oran|Tutar", name):
            score -= 50
        scored.append((score, name[:240]))
    if not scored:
        return None
    branded = [
        s
        for s in scored
        if re.search(r"(?i)IPHONE|SAMSUNG|APPLE|PROFILO|PENCIL|LED\s*TV", s[1])
    ]
    pool = branded or scored
    pool.sort(key=lambda t: t[0], reverse=True)
    # If winner starts mid-wrap, prepend previous product-ish line when available
    best = pool[0][1]
    return best


def _fix_gib_year_digits(digits: str) -> str:
    """Repair OCR year prefix in GİB serial (e.g. 2826 → 2026)."""
    if len(digits) < 4 or not digits[:4].isdigit():
        return digits
    year = int(digits[:4])
    if 1990 <= year <= 2100:
        return digits[:13] if len(digits) > 13 else digits
    pairs = {
        "8": "0",
        "0": "8",
        "3": "2",
        "2": "3",
        "5": "6",
        "6": "5",
        "1": "7",
        "7": "1",
        "4": "9",
        "9": "4",
    }
    ychars = list(digits[:4])
    for i, ch in enumerate(ychars):
        alt = pairs.get(ch)
        if not alt:
            continue
        trial = ychars.copy()
        trial[i] = alt
        y2 = int("".join(trial))
        if 1990 <= y2 <= 2100:
            out = "".join(trial) + digits[4:]
            return out[:13] if len(out) >= 13 else out
    # 28xx / 29xx → 20xx (common closed-loop OCR)
    if digits.startswith(("28", "29")):
        out = "20" + digits[2:]
        return out[:13] if len(out) >= 13 else out
    return digits[:13] if len(digits) > 13 else digits


def normalize_gib_invoice_number(num: str) -> str:
    """Fix common Latin OCR confusions in GİB fatura no (3-char series + digits)."""
    compact = re.sub(r"[^A-Z0-9]", "", num.upper())
    # Prefer classic 3 + 13 shape when long enough
    if len(compact) >= 16 and compact[3:16].isdigit():
        series, digits = compact[:3], compact[3:16]
    else:
        m = re.fullmatch(r"([A-Z0-9]{2,5})(\d{10,20})", compact)
        if not m:
            return compact
        series, digits = m.group(1), m.group(2)
    # Series OCR: 8↔B (closed bowl). Keep intentional digits (e.g. C0Y).
    series = "".join("B" if ch == "8" else ch for ch in series)
    # VAU→YAU / V↔Y when second letter is A (common Latin OCR)
    if len(series) >= 2 and series[0] == "V" and series[1] == "A":
        series = "Y" + series[1:]
    # First-letter E↔B when second is B (EBx… → BBx…)
    if len(series) >= 2 and series[0] == "E" and series[1] == "B":
        series = "B" + series[1:]
    # Digit tail confusions + year repair
    digits = digits.translate(str.maketrans("OILTS", "01115"))
    digits = _fix_gib_year_digits(digits)
    return series + digits


_GIB_NO = r"[A-Z0-9]{3}\d{13}"
_GIB_NO_LOOSE = r"[A-Z0-9]{2,5}\d{10,20}"


def is_gib_invoice_number(num: str) -> bool:
    n = re.sub(r"[^A-Z0-9]", "", num.upper())
    return bool(re.fullmatch(_GIB_NO, n) or re.fullmatch(_GIB_NO_LOOSE, n) or re.fullmatch(r"\d{12,22}", n))


def gib_invoice_number(text: str, file_name: str = "") -> str | None:
    """GIB-style fatura/belge no (3 alphanumeric series + year/seq; tolerate OCR)."""
    # Viewer chrome: "TEG2024000153285 Nolu … Fatura"
    chrome = re.search(rf"\b({_GIB_NO_LOOSE})\s+Nolu\b", text, re.I)
    if chrome:
        return normalize_gib_invoice_number(chrome.group(1))
    labeled = re.search(
        r"(?:Fatura\s*No|Fatera\s*No|Fatara\s*Na|Fataca\s*Na|Patara\s*Na|"
        r"Invoice\s*No|B[EÉ]?[LİI1]?GE\s*N[O0]|BELGE\s*NO)\s*[:\-.]?\s*"
        r"([A-Za-z0-9]{2,5}[\s\-]*\d{10,20}|\d{12,22})",
        text,
        re.I,
    )
    if labeled:
        compact = re.sub(r"[\s\-]+", "", labeled.group(1).upper())
        if is_gib_invoice_number(compact):
            return normalize_gib_invoice_number(compact) if re.search(r"[A-Z]", compact) else compact
    # Compact search: do not require trailing \b — OCR often glues next token (…3538TA:)
    compact = re.sub(r"[\s|]+", "", text.upper()).replace("-", "")
    m = re.search(rf"(?<![A-Z0-9])({_GIB_NO})(?!\d)", compact)
    if m:
        return normalize_gib_invoice_number(m.group(1))
    m = re.search(rf"(?<![A-Z0-9])({_GIB_NO_LOOSE})(?!\d)", compact)
    if m:
        return normalize_gib_invoice_number(m.group(1))
    m = first_match(file_name, rf"({_GIB_NO_LOOSE})")
    return normalize_gib_invoice_number(m) if m else None


def extract_withholding_vat_amount(text: str) -> float | None:
    summed = sum_labeled_amounts(text, "Hesaplanan KDV Tevkifat")
    if summed is not None:
        return summed
    return labeled_amount(text, "KDV Tevkifat")


def strong_photo_invoice(inv: Invoice, validation: Validation) -> bool:
    """Enough fields from photo OCR to skip heavy Docling."""
    if not inv.invoiceNumber or inv.totals.payableAmount is None:
        return False
    if not inv.lines:
        return False
    # Missing supplier VKN/TCKN → keep Docling/tesseract in play (Teknosa header OCR gaps)
    st = normalize_ocr_digits(inv.supplier.taxId) or digits_only(inv.supplier.taxId)
    if not st:
        return False
    # Phone-looking 5… only blocks when checksum fails (valid 5020… VKNs are OK)
    if len(st) == 10 and st.startswith("5") and not is_valid_vkn(st):
        return False
    # GİB serial year must look real after OCR repair
    ym = re.match(r"^[A-Z]{2,5}(\d{4})", inv.invoiceNumber)
    if ym:
        year = int(ym.group(1))
        if year < 2010 or year > 2100:
            fixed = normalize_gib_invoice_number(inv.invoiceNumber)
            ym2 = re.match(r"^[A-Z]{2,5}(\d{4})", fixed or "")
            year = int(ym2.group(1)) if ym2 else year
        if year < 2010 or year > 2100:
            return False
    # Reject when line totals are clearly not the invoice (bad thermal OCR names)
    line_sum = sum(l.lineTotal or 0.0 for l in inv.lines if l.lineTotal is not None)
    payable = inv.totals.payableAmount or 0.0
    if line_sum > 0 and payable > 0:
        ratio = line_sum / payable
        if ratio < 0.35 or ratio > 2.5:
            return False
        # Names that look like OCR noise (almost no vowels / too short words)
        junk = 0
        useful = 0
        for l in inv.lines:
            n = (l.name or "").strip()
            if re.search(r"Mal\s*Hizmet\s*Toplam|Toplam\s*Tutar|Iskonto", n, re.I):
                junk += 2
            if re.search(r"(?i)\bETT?N\b|[0-9a-f]{6,}-[0-9a-f-]{6,}", n):
                junk += 2
            letters = re.sub(r"[^A-Za-zÇĞİÖŞÜçğıöşü]", "", n)
            digits = re.sub(r"\D", "", n)
            if len(digits) >= 6 and len(letters) <= 2:
                junk += 2
            vowels = len(re.findall(r"[aeıioöuüAEIİOÖUÜ]", letters))
            # Product SKUs (HP 146GB SAS…) are vowel-poor but valid
            if (
                len(letters) >= 10
                and vowels <= 1
                and not re.search(r"(?i)\b(?:HP|GB|SAS|SSD|NVMe|USB|CPU|RAM|LED)\b", n)
            ):
                junk += 1
            if (l.lineTotal or 0) >= 10 and len(letters) >= 4 and vowels >= 1:
                useful += 1
        if junk > useful and junk >= max(1, len(inv.lines) // 2):
            return False
        # Tiny "payable" while line OCR is UUID noise — not a real invoice total
        if payable < 20 and line_sum < 20:
            return False
        # Number + payable + at least one real product line is enough to skip Docling
        if useful >= 1 and payable >= 20:
            return True
    if inv.issueDate and (inv.customer.name or inv.supplier.name):
        return True
    return validation.confidence >= PHOTO_OCR_MIN_CONF


def normalize_ocr_uuid(raw: str) -> str | None:
    """Fix common OCR confusions in ETTN (O→0, I/l→1, S→5, R→F, …).

    Also tolerates a short leading group (6–7 hex instead of 8), common in
    ExBilişim-style OCR: a88b302-4db9-… → 0a88b302-4db9-…
    """
    cleaned = raw.strip().upper().replace("‑", "-")
    cleaned = cleaned.replace("\\_", "").replace("_", "").replace("{", "").replace("}", "")
    cleaned = cleaned.replace("–", "-").replace("—", "-")
    # Non-hex OCR lookalikes (do not map valid hex letters B/C/D/A/E/F)
    safe_trans = str.maketrans(
        {
            "O": "0",
            "İ": "1",
            "I": "1",
            "L": "1",
            "S": "5",
            "G": "6",
            "Q": "0",
            "P": "F",
            "R": "F",
            "T": "7",
            "Z": "2",
            "H": "B",
            "N": "A",
            "U": "0",
            "Y": "7",
            "W": "M",
        }
    )
    loose = cleaned.translate(safe_trans)
    hex_all = re.sub(r"[^0-9A-F]", "", loose)
    # 31/30 hex: pad left (dropped leading nibble/char)
    if len(hex_all) == 31:
        hex_all = "0" + hex_all
    elif len(hex_all) == 30:
        hex_all = "00" + hex_all
    if len(hex_all) == 32:
        cleaned = (
            f"{hex_all[0:8]}-{hex_all[8:12]}-{hex_all[12:16]}-"
            f"{hex_all[16:20]}-{hex_all[20:32]}"
        )
    else:
        parts = cleaned.split("-")
        if len(parts) == 5:
            fixed_parts = []
            expected = [8, 4, 4, 4, 12]
            for part, n in zip(parts, expected):
                p = part.translate(safe_trans)
                p = re.sub(r"[^0-9A-F]", "", p)
                if len(p) > n:
                    p = p[:n]
                elif 0 < len(p) < n and (n - len(p)) <= 2:
                    # Short leading/middle group — left-pad with zeros
                    p = p.zfill(n)
                fixed_parts.append(p)
            cleaned = "-".join(fixed_parts)
            # If still short overall, pad first group from concatenated hex
            hex2 = re.sub(r"[^0-9A-F]", "", cleaned)
            if len(hex2) in (30, 31):
                hex2 = hex2.zfill(32)
            if len(hex2) == 32:
                cleaned = (
                    f"{hex2[0:8]}-{hex2[8:12]}-{hex2[12:16]}-"
                    f"{hex2[16:20]}-{hex2[20:32]}"
                )
        else:
            cleaned = loose
            cleaned = re.sub(r"[^0-9A-F-]", "", cleaned)
    cleaned = cleaned.lower()
    if re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        cleaned,
    ):
        return cleaned
    return None


def format_uuid_hex(raw: str) -> str | None:
    """Normalize dashed or undashed 32-hex ETTN."""
    return normalize_ocr_uuid(raw)


def clean_retail_product_name(name: str) -> str:
    """Light OCR cleanup for Turkish retail product names (generic typo fixes only)."""
    name = re.sub(r"\s+", " ", name).strip(" -*|")
    name = re.sub(r"[%xX×]\s*\d{1,2}\s*$", "", name)
    name = re.sub(r"(?<=[A-Za-zÇĞİÖŞÜçğıöşü])\d{3,}H?\b", "", name)
    name = re.sub(r"\bSUPURGE\w*", "SÜPÜRGE", name, flags=re.I)
    fixes = (
        (r"\bSARJAI\b", "ŞARJLI"),
        (r"\bSARJRI\b", "ŞARJLI"),
        (r"\bSARJLI\b", "ŞARJLI"),
        (r"\bDiK\b", "DİK"),
        (r"\bDIK\b", "DİK"),
        (r"\bCAMASIR\b", "ÇAMAŞIR"),
        (r"\bÇAMASIR\b", "ÇAMAŞIR"),
        (r"\bBILGI\b", "BİLGİ"),
        (r"\bFIS[Iİ]?\b", "FİŞİ"),
        (r"\bHAGAZACILIK\b", "MAGAZACILIK"),
    )
    for pat, repl in fixes:
        name = re.sub(pat, repl, name, flags=re.I)
    return name.strip(" -*|")[:240]


def parse_retail_pos_lines(text: str) -> list[Line]:
    """Parse market bilgi fişi / POS lines."""
    out: list[Line] = []
    # qty adet x/× unit \n name \n %vat *total  (VAT+amount may be split across lines)
    block_re = re.compile(
        rf"(?ms)^(?P<qty>\d+)\s*adet\s*[x×X]\s*(?P<unit>{_MONEY_TOKEN})\s*\n"
        rf"(?P<name>[^\n]{{3,80}})\s*\n"
        rf"%?\s*(?P<vat>\d{{1,2}})\s*(?:\n|\s+)\*?\s*(?P<total>{_MONEY_TOKEN})",
        re.I,
    )
    for i, m in enumerate(block_re.finditer(text), start=1):
        qty = float(m.group("qty"))
        unit = parse_tr_money(m.group("unit"))
        total = parse_tr_money(m.group("total"))
        vat = normalize_vat_rate(float(m.group("vat")))
        name = clean_retail_product_name(m.group("name"))
        if not name or total is None:
            continue
        if re.search(r"^(?:ARA\s*TOPLAM|TOPLAM|TOPKDV|KDV)\b", name, re.I):
            continue
        out.append(
            Line(
                id=str(i),
                name=name,
                quantity=qty,
                unit="Adet",
                unitPrice=unit,
                vatRate=vat,
                lineTotal=total,
            )
        )
    if out:
        return out

    # Single-line / glued: "PRODUCT…%20" then nearby "%20 *3.499,00" or "*3.499,00"
    glued = re.compile(
        rf"(?m)^(?P<name>[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü0-9 /.\-]{{5,70}}?)[%\s]*"
        rf"(?P<vat>\d{{1,2}})?\s*$"
    )
    star_amt = re.compile(rf"%\s*(?P<vat>\d{{1,2}})\s*\*?\s*(?P<total>{_MONEY_TOKEN})")
    skip_glued = re.compile(
        r"^(?:ARA|TOPLAM|TOPKDV|KDV|TARIH|BELGE|ETTN|Mgz|FAT|KASIYER|http|www|"
        r"BANKA|BANKASI|BU\s+BELGE|ONAY|KART|AID|YIGIN|SIRA|Provizyon|"
        r"TERMINAL|ISYERI|CHIP|Fatura|E-?AR[SŞ]|B[İI]LG[İI]|"
        r"Hgz\s*Ad[iı]|Mgz\s*Kodu|Ma[ğg]aza\s*(?:Ad[iı]|Kodu))",
        re.I,
    )
    lines_txt = text.splitlines()
    for i, ln in enumerate(lines_txt):
        gm = glued.match(ln.strip())
        if not gm:
            continue
        name = clean_retail_product_name(gm.group("name"))
        if len(name) < 8 or skip_glued.search(name):
            continue
        # Require product-like: has vowel and not mostly digits/slashes
        letters = re.sub(r"[^A-Za-zÇĞİÖŞÜçğıöşü]", "", name)
        vowels = len(re.findall(r"[aeıioöuüAEIİOÖUÜ]", letters))
        if vowels < 2 or len(letters) < 6:
            continue
        vat = normalize_vat_rate(float(gm.group("vat"))) if gm.group("vat") else None
        total = None
        for look in lines_txt[i : i + 5]:
            sm = star_amt.search(look)
            if not sm:
                continue
            cand = parse_tr_money(sm.group("total"))
            if cand is not None and cand >= 1:
                total = cand
                vat = normalize_vat_rate(float(sm.group("vat"))) or vat
                break
        if total is None:
            continue
        out.append(
            Line(
                id=str(len(out) + 1),
                name=name[:240],
                quantity=1.0,
                unit="Adet",
                unitPrice=total,
                vatRate=vat or 20.0,
                lineTotal=total,
            )
        )
        if len(out) >= 8:
            break
    if out:
        return out

    retail_line_re = re.compile(
        rf"(?m)^(?P<name>[A-ZÇĞİÖŞÜ0-9][A-ZÇĞİÖŞÜa-zçğıöşü0-9 /.\-]{{5,70}}?)"
        rf"(?:[%\s]*[xX]?(?P<vat>\d{{1,2}}))?\s+\*?\s*(?P<total>{_MONEY_TOKEN})\s*$",
        re.I,
    )
    skip = re.compile(
        r"^(?:ARA\s*TOPLA[MH]|TOPLAM|TOPKDV|KDV|KRED[İI]?|BANKA|Provizyon|Tutar|TARIH|"
        r"BELGE|ETTN|Mgz|FAT|Faturan|KREDL|KASIYER|TERMINAL)",
        re.I,
    )
    for i, m in enumerate(retail_line_re.finditer(text), start=1):
        name = m.group("name").strip()
        if skip.search(name):
            continue
        if re.search(
            r"https?://|KASIYER|CHIP|ONAY|TERMINAL|ISYERI|AID:|BANKA|BANKASI|KART",
            name,
            re.I,
        ):
            continue
        total = parse_tr_money(m.group("total"))
        if total is None or total < 1:
            continue
        vat = normalize_vat_rate(float(m.group("vat"))) if m.group("vat") else 20.0
        name = re.sub(r"[%\s]*[xX]?\d{1,2}\s*$", "", name).strip(" -")
        name = clean_retail_product_name(name)
        if len(name) < 6 or re.fullmatch(r"[xX]?\d{1,3}.*", name):
            continue
        if re.search(r"^(?:X?\d{1,2}|KDV|TOP)", name, re.I):
            continue
        out.append(
            Line(
                id=str(i),
                name=name,
                quantity=1.0,
                unit="Adet",
                unitPrice=total,
                vatRate=vat,
                lineTotal=total,
            )
        )
        if len(out) >= 8:
            break
    return out


def extract_ettn_candidate(text: str) -> str | None:
    """Find ETTN even when OCR glues/truncates label (ETN…, ETTNe…) or mangles hex."""
    hexish = r"0-9A-Fa-fİILOSBloşPGQZpgqzRrTtHhNnUuYyWwMm"
    # Labeled; tolerate ETT N / ETT: / ETİN / UUID and short leading group (6–8)
    m = re.search(
        rf"(?i)(?:ETT\s*N|ETT\s*:|ETT[İIıiNnNne\{{]{{0,4}}|UUID)\s*[:\-]?\s*"
        rf"([{hexish}_\\]{{6,12}}[-‑]?[{hexish}_\\]{{3,6}}[-‑]?"
        rf"[{hexish}_\\]{{3,6}}[-‑]?[{hexish}_\\]{{3,6}}[-‑]?[{hexish}_\\]{{10,16}})",
        text,
    )
    if m:
        got = format_uuid_hex(m.group(1).replace("‑", "-"))
        if got:
            return got
    # Glued label+uuid: ETNa89b302-... / ETTa88b302-...
    m = re.search(
        rf"(?i)ETT?Ne?([{hexish}]{{6,10}}[-‑][{hexish}]{{3,5}}[-‑]"
        rf"[{hexish}]{{3,5}}[-‑][{hexish}]{{3,5}}[-‑][{hexish}]{{10,14}})",
        text,
    )
    if m:
        got = format_uuid_hex(m.group(1).replace("‑", "-"))
        if got:
            return got
    # Broken leading group 6–7 (not only perfect 8)
    m = re.search(
        rf"\b([{hexish}]{{6,8}}-[{hexish}]{{4}}-[{hexish}]{{4}}-[{hexish}]{{4}}-[{hexish}]{{12}})\b",
        text,
        re.I,
    )
    if m:
        got = format_uuid_hex(m.group(1))
        if got:
            return got
    m = re.search(
        r"\b([0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12})\b",
        text,
        re.I,
    )
    if m:
        got = format_uuid_hex(m.group(1))
        if got:
            return got
    # OCR-mangled dashed UUID (letters mixed in)
    m = re.search(
        rf"\b([{hexish}]{{6,10}}[-‑][{hexish}]{{3,5}}[-‑][{hexish}]{{3,5}}[-‑]"
        rf"[{hexish}]{{3,5}}[-‑][{hexish}]{{10,14}})\b",
        text,
        re.I,
    )
    if m:
        got = format_uuid_hex(m.group(1).replace("‑", "-"))
        if got:
            return got
    # Compact 32-hex (labeled or bare); also 30–32 with pad
    m = re.search(
        rf"(?i)(?:ETT\s*N|ETT\s*:|ETT?Ne?|UUID)\s*[:\-]?\s*([{hexish}]{{30,40}})\b",
        text,
    )
    if m:
        got = format_uuid_hex(m.group(1))
        if got:
            return got
    m = re.search(rf"(?i)\b([{hexish}]{{30,32}})\b", text)
    if m:
        got = format_uuid_hex(m.group(1))
        if got:
            return got
    return None


def is_probable_non_invoice_text(text: str) -> bool:
    """True when short OCR text lacks any e-invoice markers (logo/ad screenshots)."""
    t = (text or "").strip()
    if len(t) >= 800:
        return False
    return not bool(
        re.search(
            r"(?i)e-?Ar[sş]iv|e-?Fatura|\bETTN\b|\bÖdenecek\b|\bOdenecek\b|"
            r"\bSAYIN\b|\bVKN\b|\bTCKN\b|Fatura\s*No|Mal\s*/?\s*Hizmet|"
            r"\bKDV\b|\bGIB\b|\bGİB\b|Özelleştirme|Senaryo",
            t,
        )
    )


def is_garbage_photo_ocr(inv: Invoice, photo_meta: dict[str, Any] | None = None) -> bool:
    """True when OCR structure is empty and no critical invoice fields bound.

    Used to skip Docling/VL on Vulkan/DeFacto-style garbage photos (~15–25s waste).
    """
    struct = int((photo_meta or {}).get("structureScore") or 0)
    if struct > 2:
        return False
    if inv.invoiceNumber:
        return False
    if inv.totals.payableAmount is not None or inv.totals.taxInclusiveAmount is not None:
        return False
    if inv.supplier.taxId or inv.customer.taxId:
        return False
    if inv.lines:
        return False
    return True


def rebalance_party_tax_ids(inv: Invoice, text: str = "") -> None:
    """Fix supplier↔customer tax-id swaps and clear placeholders/phone false VKNs."""
    from tax_id import is_placeholder_tax_id

    s, c = inv.supplier, inv.customer
    st = normalize_ocr_digits(s.taxId) or digits_only(s.taxId)
    ct = normalize_ocr_digits(c.taxId) or digits_only(c.taxId)

    if st and is_placeholder_tax_id(st):
        s.taxId = None
        s.taxIdScheme = None
        st = ""
    if ct and is_placeholder_tax_id(ct):
        c.taxId = None
        c.taxIdScheme = None
        ct = ""

    # Same id on both sides
    if st and ct and st == ct:
        if len(st) == 11:
            # Person TCKN belongs to customer
            s.taxId = None
            s.taxIdScheme = None
            st = ""
            c.taxId, c.taxIdScheme = ct, "TCKN"
        elif len(st) == 10 and st.startswith("5") and not is_valid_vkn(st):
            # Likely phone fragment, not company VKN
            s.taxId = None
            s.taxIdScheme = None
            st = ""

    st = normalize_ocr_digits(s.taxId) or digits_only(s.taxId)
    ct = normalize_ocr_digits(c.taxId) or digits_only(c.taxId)

    # Supplier has TCKN, customer empty → move to customer, try recover VKN
    if st and len(st) == 11 and not ct:
        c.taxId, c.taxIdScheme = st, "TCKN"
        s.taxId = None
        s.taxIdScheme = None
        st = ""

    st = normalize_ocr_digits(s.taxId) or digits_only(s.taxId)
    ct = normalize_ocr_digits(c.taxId) or digits_only(c.taxId)

    # Supplier VKN missing / phone-like → multi-layer header recovery
    phoneish = bool(st and len(st) == 10 and st.startswith("5") and not is_valid_vkn(st))
    if text and (not st or phoneish):
        recovered = (
            find_role_tax_id(text, "supplier")
            or recover_supplier_vkn_from_header(text)
        )
        if recovered and recovered[1] == "VKN" and recovered[0] != ct:
            s.taxId, s.taxIdScheme = recovered

    # Final placeholder sweep
    for party in (s, c):
        raw = normalize_ocr_digits(party.taxId) or digits_only(party.taxId)
        if raw and is_placeholder_tax_id(raw):
            party.taxId = None
            party.taxIdScheme = None


_TAX_OCR = r"0-9OoОİIiılLSsBbGgZz"
_TAX_LABEL = (
    r"(?:VKN\s*/\s*TCKN|TCKN\s*/\s*VKN|VKN|TCKN|VIN|V\.?\s*N\.?|"
    r"Vergi\s*(?:No|Numaras[ıi]|Kimlik(?:\s*No)?))"
)


def _match_tax_id_token(raw: str) -> tuple[str, str] | None:
    return coerce_tax_id(raw)


def find_tax_id_in_region(text: str) -> tuple[str, str] | None:
    """Wide VKN/TCKN finder for a text region (OCR-tolerant)."""
    if not text:
        return None
    patterns = [
        rf"(?i){_TAX_LABEL}\s*:?[.\s]*([{_TAX_OCR}]{{10,11}})\b",
        rf"(?i){_TAX_LABEL}\s*:?[.\s]*((?:[{_TAX_OCR}]{{3}}\s+){{2}}[{_TAX_OCR}]{{4}})\b",
        rf"(?i){_TAX_LABEL}\s*:?[.\s]*((?:[{_TAX_OCR}]{{3}}\s+){{3}}[{_TAX_OCR}]{{2}})\b",
        # "Marmara Kurumlar Vergi Dairesi 6130636884"
        rf"(?i)Vergi\s*Dairesi\s*[A-ZÇĞİÖŞÜa-zçğıöşü .]{{0,48}}?\s+([{_TAX_OCR}]{{10}})\b",
        rf"(?i)\bV\.?\s*D\.?\s*[.:]?\s*[A-ZÇĞİÖŞÜa-zçğıöşü .]{{0,40}}?\s+([{_TAX_OCR}]{{10}})\b",
        # Spaced classic VKN without label nearby
        rf"(?<!\d)((?:\d{{3}}\s+){{2}}\d{{4}})(?!\d)",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text):
            got = _match_tax_id_token(m.group(1))
            if got:
                return got
    return None


def _ok_supplier_vkn(raw: str | None, *, labeled: bool = False) -> tuple[str, str] | None:
    """Accept company VKN; phone-risk only for unlabeled invalid 5-start ids.

    Checksum-valid VKNs are kept even when they start with 5 (e.g. 5020056347).
    Unlabeled 5-start without a valid GİB checksum is treated as a phone fragment.
    Single-digit OCR repair is only applied for labeled Vergi No/VKN candidates.
    """
    from tax_id import is_placeholder_tax_id, is_valid_vkn, repair_tax_id

    n = normalize_ocr_digits(raw) or digits_only(raw)
    if len(n) != 10 or is_placeholder_tax_id(n):
        return None
    if is_valid_vkn(n):
        return n, "VKN"
    if labeled:
        repaired = repair_tax_id(n, "VKN")
        if repaired:
            return repaired
    # Checksum failed: unlabeled 5… → phone risk (do not OCR-repair into another VKN)
    if n.startswith("5"):
        return None
    return None


def recover_supplier_vkn_from_header(text: str) -> tuple[str, str] | None:
    """Multi-layer supplier VKN search in the SAYIN-before (header) region.

    Layers: labeled VKN/VIN/Vergi No → VD/Vergi Dairesi/Mükellef nearby →
    spaced 836 014 4393 → compact 10-digit → firm-line neighbor →
    last-resort clean 10-digit in first 1200 chars (non-phone).
    """
    if not text:
        return None
    sayin = re.search(r"\bSAYIN\b", text, re.I)
    head = text[: sayin.start()] if sayin else text[:1600]
    if not head.strip():
        head = text[:1600]

    # 1) Explicit tax labels (Vergi No / VKN / VIN) — prefer these; 5-start OK if checksum valid
    for pat in (
        rf"(?i){_TAX_LABEL}\s*:?[.\s]*([{_TAX_OCR}]{{10}})\b",
        rf"(?i){_TAX_LABEL}\s*:?[.\s]*((?:[{_TAX_OCR}]{{3}}\s+){{2}}[{_TAX_OCR}]{{4}})\b",
    ):
        for m in re.finditer(pat, head):
            got = _ok_supplier_vkn(m.group(1), labeled=True)
            if got:
                return got

    # 2) VD / Vergi Dairesi / Mükellef / Büyük Mükellef vicinity
    for pat in (
        rf"(?i)(?:B[üu]y[üu]k\s*)?M[üu]kellef(?:ler)?(?:\s*V\.?\s*D\.?)?"
        rf"[A-ZÇĞİÖŞÜa-zçğıöşü ./]{{0,48}}?"
        rf"([{_TAX_OCR}]{{10}}|(?:[{_TAX_OCR}]{{3}}\s+){{2}}[{_TAX_OCR}]{{4}})\b",
        rf"(?i)(?:V\.?\s*D\.?|Vergi\s*Dai(?:resi|resi|r[ae]s[il])|Mukellef\w*|M[üu]kellef\w*)"
        rf"[A-ZÇĞİÖŞÜa-zçğıöşü ./:]{{0,48}}?"
        rf"([{_TAX_OCR}]{{10}}|(?:[{_TAX_OCR}]{{3}}\s+){{2}}[{_TAX_OCR}]{{4}})\b",
        rf"(?i)\bVD\s*[.:]?\s*([{_TAX_OCR}]{{10}})\b",
    ):
        for m in re.finditer(pat, head):
            got = _ok_supplier_vkn(m.group(1), labeled=True)
            if got:
                return got

    # 3) Spaced classic form anywhere in head
    for m in re.finditer(r"(?<!\d)((?:\d{3}\s+){2}\d{4})(?!\d)", head):
        got = _ok_supplier_vkn(m.group(1), labeled=False)
        if got:
            return got

    # 4) Compact 10-digit near company / trade tokens
    for m in re.finditer(
        rf"(?i)(?:A\.?\s*[SŞ]\.?|LTD|ŞT[İI]|T[İI]CARET|SANAY[İI]|MA[ĞG]AZA)"
        rf"[^\n]{{0,40}}?([{_TAX_OCR}]{{10}})\b",
        head,
    ):
        got = _ok_supplier_vkn(m.group(1), labeled=False)
        if got:
            return got
    for m in re.finditer(
        rf"(?i)([{_TAX_OCR}]{{10}})\s*(?:A\.?\s*[SŞ]\.?|LTD|V\.?\s*D\.?)\b",
        head,
    ):
        got = _ok_supplier_vkn(m.group(1), labeled=False)
        if got:
            return got

    # 5) Last resort: first 1200 chars, first clean non-phone 10-digit
    window = text[:1200]
    for m in re.finditer(rf"(?<!\d)([{_TAX_OCR}]{{10}})(?!\d)", window):
        # Skip ids glued to Tel/TCKN/Fax labels
        start = m.start()
        prefix = window[max(0, start - 24) : start]
        if re.search(r"(?i)(?:Tel|Telefon|Fax|TCKN|M[üu][şs]teri)\s*:?\s*$", prefix):
            continue
        labeled = bool(re.search(r"(?i)(?:VKN|Vergi\s*No|V\.?\s*N\.?)\s*:?\s*$", prefix))
        got = _ok_supplier_vkn(m.group(1), labeled=labeled)
        if got:
            return got

    return None


def find_role_tax_id(text: str, role: str) -> tuple[str, str] | None:
    """role: 'supplier' | 'customer' — prefer Satıcı/Alıcı labeled ids."""
    if role == "supplier":
        labeled = re.search(
            rf"(?i)Sat[ıi]c[ıi](?:\s*Bilgileri)?[^\n]{{0,80}}?{_TAX_LABEL}\s*:?[.\s]*"
            rf"([{_TAX_OCR}]{{10,11}})\b",
            text,
        )
        if labeled:
            got = _match_tax_id_token(labeled.group(1))
            if got:
                return got
        # Section: Satıcı Bilgileri … VKN
        sec = re.search(
            r"(?is)Sat[ıi]c[ıi]\s*Bilgileri(.{0,500}?)(?:Al[ıi]c[ıi]\s*Bilgileri|SAYIN|Mal\s*/?\s*Hizmet|$)",
            text,
        )
        if sec:
            got = find_tax_id_in_region(sec.group(1))
            if got:
                return got
    else:
        labeled = re.search(
            rf"(?i)Al[ıi]c[ıi](?:\s*Bilgileri)?[^\n]{{0,80}}?{_TAX_LABEL}\s*:?[.\s]*"
            rf"([{_TAX_OCR}]{{10,11}})\b",
            text,
        )
        if labeled:
            got = _match_tax_id_token(labeled.group(1))
            if got:
                return got
        labeled = re.search(
            rf"(?i)M[üu][şs]teri\s*(?:VKN|TCKN|Vergi\s*No)\s*:?[.\s]*([{_TAX_OCR}]{{10,11}})\b",
            text,
        )
        if labeled:
            got = _match_tax_id_token(labeled.group(1))
            if got:
                return got
        sec = re.search(
            r"(?is)Al[ıi]c[ıi]\s*Bilgileri(.{0,500}?)(?:Mal\s*/?\s*Hizmet|Ara\s*Toplam|Ödenecek|$)",
            text,
        )
        if sec:
            got = find_tax_id_in_region(sec.group(1))
            if got:
                return got
    return None


def prefer_invoice_issue_date(text: str) -> tuple[str | None, str | None]:
    """Prefer Fatura/Tarih near metadata; ignore voucher expiry dates."""
    scrubbed = re.sub(
        r"Son\s+Kullanma\s+Tarihi\s*:?\s*\d{1,2}\s*[-./,]\s*\d{1,2}\s*[-./,]\s*\d{4}",
        " ",
        text,
        flags=re.I,
    )
    _date_tok = rf"([{_TAX_OCR}]{{1,2}}\s*[-./]\s*[{_TAX_OCR}]{{1,2}}\s*[-./]\s*[{_TAX_OCR}]{{4}})"
    # Explicit Fatura Tarihi first (ISO table cells from Docling/GİB)
    fatura_iso = first_match(
        scrubbed,
        r"Fatura\s*(?:Tarihi|Yarihi|Tanible)\s*:?\s*[|]*\s*"
        r"(\d{4}-\d{2}-\d{2}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?)",
    )
    if fatura_iso:
        d, tm = parse_issue_date(fatura_iso)
        if d:
            return d, tm
    fatura_tr = first_match(
        scrubbed,
        rf"Fatura\s*(?:Tarihi|Yarihi|Tanible)\s*:?\s*[|(]*\s*"
        rf"{_date_tok}(?:\s+\d{{1,2}}:\d{{2}}(?::\d{{2}})?)?",
    ) or right_field(scrubbed, "Fatura Tarihi")
    if fatura_tr:
        d, tm = parse_issue_date(fatura_tr.replace(",", "."))
        if d:
            return d, tm

    # Düzenleme / Belge / Fiili Sevk — common GİB / ERP alternate labels
    for label_pat in (
        rf"D[uü]zenleme\s*Tarihi\s*:?\s*{_date_tok}",
        rf"Belge\s*Tarihi\s*:?\s*{_date_tok}",
        rf"Fiili\s*Sevk\s*Tarihi\s*:?\s*{_date_tok}",
        rf"Sevk\s*Tarihi\s*:?\s*{_date_tok}",
    ):
        alt = first_match(scrubbed, label_pat)
        if alt:
            d, tm = parse_issue_date(alt.replace(",", "."))
            if d:
                return d, tm
    for label in ("Düzenleme Tarihi", "Belge Tarihi", "Fiili Sevk Tarihi", "Sevk Tarihi"):
        alt = right_field(scrubbed, label)
        if alt:
            d, tm = parse_issue_date(alt.replace(",", "."))
            if d:
                return d, tm

    tarih_lbl = re.search(
        rf"TAR[İI]H\s*:?\s*{_date_tok}(?:\s+(\d{{1,2}}:\d{{2}}))?",
        scrubbed,
        re.I,
    )
    if tarih_lbl:
        d, _ = parse_issue_date(tarih_lbl.group(1).replace("/", "."))
        tm = tarih_lbl.group(2)
        if tm and len(tm) == 5:
            tm = tm + ":00"
        if d:
            return d, tm
    retail = re.search(
        rf"(?:^|\n)\s*{_date_tok}\s+(?:Saat\s*:?\s*)?(\d{{1,2}}:\d{{2}})?",
        scrubbed,
        re.I,
    )
    if retail:
        d, _ = parse_issue_date(retail.group(1).replace("/", "."))
        tm = retail.group(2)
        if tm and len(tm) == 5:
            tm = tm + ":00"
        if d:
            return d, tm

    issue_raw = (
        right_field(scrubbed, "Tarih")
        or right_field(scrubbed, "Tarth")
        or first_match(
            scrubbed,
            rf"(?:Fatera\s*No|Fatura\s*No)[^\n]{{0,40}}?\n[^\n]*?"
            rf"(?:Tarih|Tarth)\s*:?\s*{_date_tok}",
        )
        or first_match(
            scrubbed,
            rf"(?:Tarih|Tarth|D[uü]zenleme\s*Tarihi)\s*:?\s*{_date_tok}",
        )
        or first_match(
            scrubbed,
            rf"(?<!\d){_date_tok}(?!\d)",
        )
    )
    if issue_raw:
        issue_raw = issue_raw.replace(",", ".")
    return parse_issue_date(issue_raw)


def extract_payable_from_ocr(text: str) -> float | None:
    """Prefer bank/payment lines and OCR-tolerant 'ödenecek/vergi dahil/toplam' labels."""
    bank = re.search(
        rf"(?:BANKASI|Banka\s*/\s*Kredi\s*Kart[ıi]|KRED[İI]\s*KART[İI]?|"
        rf"Kart\s*(?:ile|ödeme)|POS)"
        rf"[^\n]{{0,48}}?\*?\s*({_MONEY_TOKEN})",
        text,
        re.I,
    )
    if bank:
        amt = parse_tr_money(bank.group(1))
        if amt is not None:
            return amt
    oden = None
    dahil = None
    for label in (
        r"[ÖO]DENECEK\s+TUTAR",
        r"Ödenecek\s+Tutar",
        r"[ÖO][df]?e[sş]?e?ne[cçkhs]{1,4}\s+T[OU]TAR",
    ):
        amt = labeled_amount(text, label)
        if amt is not None and amt >= 1:
            oden = amt
            break
    for label in (
        r"VERG[İIEÉ]\s+DAH[İI]L\s+TOPLAM\s+TUTAR",
        r"Vergiler\s+Dahil\s+Toplam\s+Tutar",
        r"Varg[iı]kr\s+Dahil\s+Toplam\s+Tutar",
    ):
        amt = labeled_amount(text, label)
        if amt is not None and amt >= 1:
            dahil = amt
            break
    # When OCR flips leading digit on ödenecek (36.994 vs 26.994), prefer vergi dahil
    if oden is not None and dahil is not None:
        if abs(oden - dahil) < 0.05:
            return oden
        # Same magnitude, one digit OCR slip on the thousands
        if abs(oden - dahil) >= 1000 and abs(oden - dahil) < 20000:
            return dahil
        return oden
    if oden is not None:
        return oden
    if dahil is not None:
        return dahil
    for label in (
        r"(?<!ARA\s)TOPLAM(?!\s*ISKONTO|\s*[İI]SKONTO)",
        r"Genel\s+Toplam",
    ):
        amt = labeled_amount(text, label)
        if amt is not None and amt >= 1:
            return amt
    # TOPLAM *760,00 / TOPLAM 3.499,00 (star prefix)
    m = re.search(rf"(?<!ARA\s)TOPLAM\s*\*?\s*({_MONEY_TOKEN})", text, re.I)
    if m:
        return parse_tr_money(m.group(1))
    return None


def tesseract_ocr(path: Path) -> str:
    r = subprocess.run(
        [
            "tesseract",
            str(path),
            "stdout",
            "-l",
            "tur+eng",
            "--psm",
            "6",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if r.returncode != 0 and not (r.stdout or "").strip():
        raise RuntimeError(r.stderr.strip() or "tesseract failed")
    return r.stdout or ""


def parse_issue_date(raw: str | None) -> tuple[str | None, str | None]:
    if not raw:
        return None, None
    # OCR: O4.O6.2O24 → 04.06.2024
    raw = raw.translate(
        str.maketrans(
            {
                "O": "0",
                "o": "0",
                "О": "0",
                "İ": "1",
                "I": "1",
                "i": "1",
                "ı": "1",
                "l": "1",
                "L": "1",
                "|": "1",
                "S": "5",
                "s": "5",
                "B": "8",
                "G": "6",
                "Z": "2",
            }
        )
    )
    # ISO from Docling/GİB tables: 2026-07-30 13:14:02
    iso = re.search(
        r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{1,2}):(\d{2})(?::(\d{2}))?)?",
        raw,
    )
    if iso:
        year, month, day = int(iso.group(1)), int(iso.group(2)), int(iso.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31 and 1990 <= year <= 2100:
            date = f"{year:04d}-{month:02d}-{day:02d}"
            time_ = None
            if iso.group(4) is not None and iso.group(5) is not None:
                sec = iso.group(6) or "00"
                time_ = f"{int(iso.group(4)):02d}:{iso.group(5)}:{sec}"
            return date, time_
    m = re.search(
        r"(\d{1,2})\s*[-./]\s*(\d{1,2})\s*[-./]\s*(\d{2,4})(?:\s+(\d{1,2}):(\d{2}))?",
        raw,
    )
    if not m:
        return None, None
    day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if year < 100:
        year += 2000 if year < 70 else 1900
    if not (1 <= month <= 12 and 1 <= day <= 31 and 1990 <= year <= 2100):
        return None, None
    date = f"{year:04d}-{month:02d}-{day:02d}"
    time_ = None
    if m.group(4) and m.group(5):
        time_ = f"{int(m.group(4)):02d}:{m.group(5)}:00"
    return date, time_


def empty_party() -> Party:
    return Party()


def extract_supplier(text: str) -> Party:
    party = empty_party()
    sayin = re.search(r"\bSAYIN\b", text, re.I)
    head = text[: sayin.start()] if sayin else text[:1200]
    lines = [
        ln.strip()
        for ln in head.splitlines()
        if ln.strip()
        and not re.match(r"^e-?Ar[sş]iv\s+Fatura$", ln.strip(), re.I)
        and not re.match(r"^Sayfa\s+\d+", ln.strip(), re.I)
        and not re.match(r"^(?:SUBE|ŞUBE)\s*:", ln.strip(), re.I)
        and not re.match(r"^M[ÜU][ŞS]TER", ln.strip(), re.I)
        and not re.match(r"^(?:PDF|XML)\s*indir", ln.strip(), re.I)
        and not re.match(r"^Page\s+\d+", ln.strip(), re.I)
        and not re.match(r"^\d{1,2}:\d{2}\b", ln.strip())
        and not re.match(r"^https?://", ln.strip(), re.I)
        and not re.search(r"Nolu\s+.*Fatura|Detay\s*Ekran|edoksis", ln.strip(), re.I)
        and not re.search(r"<!--\s*image", ln.strip(), re.I)
        and not _is_registry_or_chrome_line(ln.strip())
    ]
    # Explicit Satıcı Ünvanı: / Satıcı:  (require colon to avoid swallowing label)
    sat_m = re.search(
        r"Sat[ıi]c[ıi]\s*(?:[ÜU]nvan[ıiI]?|Unvan|Ad[ıi])?\s*:\s*"
        r"([A-ZÇĞİÖŞÜa-zçğıöşü0-9].{2,120})",
        head,
        re.I,
    )
    if sat_m:
        cand = re.split(r"\s{2,}|Adres\s*:|Tel(?:efon)?\s*:|Vergi|VKN|TCKN", sat_m.group(1), maxsplit=1)[
            0
        ].strip(" :.-[]{}")
        cand = re.sub(
            r"^(?:şube|sube|[ÜU]nvan[ıiI]?|Unvan|Ad[ıi]|Bilgileri)\s*:?\s*",
            "",
            cand,
            flags=re.I,
        ).strip(" :.-[]{}")
        if (
            len(cand) >= 4
            and not _is_registry_or_chrome_line(cand)
            and not re.match(r"^(?:Bilgileri|[ÜU]nvan)\b", cand, re.I)
        ):
            party.name = cand[:180]
    # Prefer a line that looks like a company title (legal form / retail trade words)
    if not party.name:
        company_idx = next(
            (
                i
                for i, ln in enumerate(lines)
                if re.search(
                    r"(?:LTD|ŞT[İI]|A\.?\s*Ş\.?|SANAY[İI]|SAN\.?\s*T[İI]C|"
                    r"T[İI]C(?:ARET)?\b|ANON[İI]M|MA[ĞGČĆC]AZA|DAGITIM|DA[ĞG]ITIM)",
                    ln,
                    re.I,
                )
                and not re.search(r"^(?:PDF|XML|Page|Adres|Tel|Web|Vergi|VKN|VIN|Kurumsal\s+Ofis)", ln, re.I)
                and not _is_registry_or_chrome_line(ln)
                and not _looks_like_address_party_line(ln)
            ),
            None,
        )
        if company_idx is not None:
            company = re.sub(r"^#+\s*", "", lines[company_idx]).strip()
            # Address + legal form glued: keep the legal-entity segment
            if re.search(r"\b(?:MAH\.|CAD\.|SOK\.|NO:)\b", company, re.I):
                ent = re.search(
                    r"((?:[A-ZÇĞİÖŞÜ]+\s+){0,6}T[İI]CARET\s+A\.?\s*[SŞ]\.?)\s*$",
                    company,
                    re.I,
                ) or re.search(
                    r"((?:[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü0-9 .&-]{0,40})?"
                    r"(?:MA[ĞG]AZACILIK|SANAY[İI]).{0,24}"
                    r"(?:A\.?\s*[SŞ]\.?|LTD\.?\s*ŞT[İI]?\.?))",
                    company,
                    re.I,
                )
                if ent:
                    company = ent.group(1).strip()
            # Merge wrapped legal-form continuation ("VE SAN.A.Ş." / "LTD. ŞTİ.")
            if company_idx + 1 < len(lines):
                nxt = lines[company_idx + 1]
                if (
                    re.search(
                        r"^(?:VE\s+)?(?:SAN\.?|T[İI]C\.?|LTD|ŞT[İI]|A\.?\s*Ş)",
                        nxt,
                        re.I,
                    )
                    and not _is_registry_or_chrome_line(nxt)
                    and not re.search(r"MAH\.|CAD\.|NO:", nxt, re.I)
                ):
                    company = f"{company} {nxt}"
            party.name = company[:180]
        elif lines and not re.match(
            r"^(Tel|Web|E-?Posta|Vergi|TCKN|VKN|Kap[ıi]|Telefon|Adres|#)", lines[0], re.I
        ):
            name = re.sub(r"^#+\s*", "", lines[0]).strip()
            if (
                len(lines) > 1
                and re.search(r"(?:LTD|ŞT[İI]|A\.?\s*Ş\.?|SAN\.|T[İI]C\.|ANON[İI]M)", lines[1], re.I)
                and not re.match(r"^(Tel|Web|E-?Posta|Vergi|TCKN|VKN|ŞUBE)", lines[1], re.I)
            ):
                nxt = re.sub(r"^#+\s*", "", lines[1]).strip()
                name = f"{name} {nxt}"
            if (
                name
                and not _is_registry_or_chrome_line(name)
                and not _looks_like_address_party_line(name)
            ):
                party.name = name[:180]
    if party.name:
        party.name = re.sub(r"^#+\s*", "", party.name).strip()
        party.name = re.sub(
            r"^(?:Sat[ıi]c[ıi]\s*(?:\([^)]*\)|\{[^}]*\}|\[[^\]]*\]|[ÜU]nvan[ıiI]?|Unvan|Ad[ıi])?\s*:?\s*)",
            "",
            party.name,
            flags=re.I,
        )
        party.name = re.sub(r"^(?:[ÜU]nvan[ıiI]?|Unvan)\s*:?\s*", "", party.name, flags=re.I)
        # Docling / markdown headings
        party.name = re.sub(r"^#+\s*", "", party.name).strip()
        party.name = re.sub(r"\bHAGAZACILIK\b", "MAGAZACILIK", party.name, flags=re.I)
        party.name = re.sub(r"\bMA[ČĆC]AZACILIK\b", "MAGAZACILIK", party.name, flags=re.I)
        party.name = re.sub(r"\bDA[ČĆC]IT[İI]N\b", "DAGITIM", party.name, flags=re.I)
        party.name = re.sub(r"\bSANLAS\.?\b", "SAN.A.S.", party.name, flags=re.I)
        party.name = re.sub(r"\s+", " ", party.name).strip(" :.-[]{}")[:180]
        if party.name and _looks_like_address_party_line(party.name):
            party.name = None
    # Address lines before SAYIN (MAH./CAD./NO:)
    addr_parts = [
        ln
        for ln in lines
        if re.search(r"\b(?:MAH\.|CAD\.|SK\.|SOK\.|BULVAR|NO:|BLOK)\b", ln, re.I)
        or (
            re.search(r"/\s*[A-ZÇĞİÖŞÜa-zçğıöşü]{3,}", ln)
            and not re.search(r"VKN|TCKN|Tel|Vergi", ln, re.I)
        )
    ]
    if addr_parts:
        party.address = ", ".join(addr_parts[:3])[:240]
    # Vergi No / Dairesi: 4470211661 / KADIKÖY
    vkn_office = re.search(
        rf"(?i)Vergi\s*No\s*/?\s*Dairesi?\s*:?[.\s]*([{_TAX_OCR}]{{10,11}})\s*/\s*"
        r"([A-ZÇĞİÖŞÜa-zçğıöşü ]{2,40})",
        head,
    )
    if vkn_office:
        coerced = coerce_tax_id(vkn_office.group(1))
        if coerced:
            party.taxId, party.taxIdScheme = coerced
        party.taxOffice = vkn_office.group(2).strip()
    if not party.taxOffice:
        party.taxOffice = (
            first_match(
                head, r"Vergi\s*Dai(?:resi|resi|r[ae]s[il])\s*:?\s*([A-ZÇĞİÖŞÜa-zçğıöşü ]{3,40})"
            )
            or ""
        ).strip() or None
    if party.taxOffice:
        party.taxOffice = sanitize_tax_office(party.taxOffice)
    # Prefer role-labeled Satıcı VKN, then head-only (never SAYIN block)
    if not party.taxId:
        role_tid = find_role_tax_id(text, "supplier")
        head_tid = find_tax_id_in_region(head)
        header_vkn = recover_supplier_vkn_from_header(text)
        # Company suppliers: prefer 10-digit VKN over a TCKN that leaked into head
        companyish = bool(
            party.name
            and re.search(r"(?:A\.?\s*[SŞ]\.?|LTD|ŞT[İI]|T[İI]C(?:ARET)?)", party.name, re.I)
        )
        chosen = role_tid or header_vkn or head_tid
        if companyish and chosen and chosen[1] == "TCKN":
            vkn_only = None
            for cand in (role_tid, header_vkn, head_tid):
                if cand and cand[1] == "VKN":
                    vkn_only = cand
                    break
            if not vkn_only:
                # re-scan head for VKN-sized ids only
                m = re.search(
                    rf"(?i)(?:VKN|Vergi\s*No|V\.?\s*N\.?|Vergi\s*Dairesi\s*[A-ZÇĞİÖŞÜa-zçğıöşü .]{{0,40}})"
                    rf"\s*:?[.\s]*([{_TAX_OCR}]{{10}})\b",
                    head,
                )
                if m:
                    vkn_only = coerce_tax_id(m.group(1))
            chosen = vkn_only or (None if companyish else chosen)
        if chosen:
            party.taxId, party.taxIdScheme = chosen
    party.email = first_match(head, r"(?:E-?Posta|E-?Mall|E-?Mail)\s*:?\s*([^\s]+)")
    party.website = first_match(head, r"Web\s*Sitesi\s*:?\s*([^\s]+)")
    phone_raw = first_match(head, r"(?:Tel|Telefon)\s*:?\s*([0-9\s()\-]{10,})")
    party.phone = re.sub(r"\s+", "", phone_raw) if phone_raw else None
    if party.phone and len(re.sub(r"\D", "", party.phone)) < 10:
        party.phone = None
    # Docling often emits SAYIN first → empty head. Recover seller from any
    # legal-entity line in the document (footer / merkez adres / imprint).
    if not party.name or _looks_like_address_party_line(party.name):
        legal = re.compile(
            r"(?m)^(?:#+\s*)?"
            r"([A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü0-9 .&'\-]{2,90}"
            r"(?:Ticaret|Sanayi|Perakende|Ma[ğg]azac[ıi]l[ıi]k|Marketleri|"
            r"Anonim|Limited|Teknoloji|Bili[sş]im|Lojistik|İn[sş]aat)"
            r".{0,48}"
            r"(?:A\.?\s*[SŞ]\.?|LTD\.?\s*ŞT[İI]?\.?|ANON[İI]M\s+Ş[İI]RKET[İI]?))"
            r"\s*$",
            re.I,
        )
        # Prefer the longest legal-entity match (full unvan, not a mid-line fragment)
        best = ""
        for m in legal.finditer(text):
            cand = re.sub(r"^#+\s*", "", m.group(1))
            cand = re.sub(r"\s+", " ", cand).strip(" :.-#")
            if (
                len(cand) >= 8
                and len(cand) > len(best)
                and not _looks_like_address_party_line(cand)
                and _party_name_quality(cand) >= 12
                and not re.search(
                    r"(?:Çözüm\s+Merkezi|İnternet\s+Sitesi|SAYIN|M[ÜU][ŞS]TER)",
                    cand,
                    re.I,
                )
            ):
                best = cand
        if best:
            party.name = best[:180]
    # Final purge: strip residual viewer/chrome fragments from supplier name
    if party.name and (
        _is_registry_or_chrome_line(party.name)
        or _is_amount_in_words_name(party.name)
        or re.search(r"(?i)\be-?Belge\b|file://|Rar\$EX|AppData\\Local", party.name)
        or re.match(r"(?i)^\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", party.name)
        or re.match(r"(?i)^\d{1,2}:\d{2}\b", party.name)
    ):
        party.name = None
    return party


def extract_customer(text: str) -> Party:
    party = empty_party()
    # POS / fiş: "MÜŞTERİ: …" etiketi
    musteri = re.search(
        r"M[ÜU][ŞS]TER[İIÍ]\s*:\s*([A-ZÇĞİÖŞÜa-zçğıöşü .'\-]{3,80})",
        text,
        re.I,
    )
    if musteri:
        party.name = musteri.group(1).strip(" :.-")
        tckn = first_match(text, r"\bTC(?:KN)?\s*:?\s*(\d{11})\b")
        if tckn:
            party.taxId = tckn
            party.taxIdScheme = "TCKN"
        phone = first_match(text, r"(\+90\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2})")
        if phone:
            party.phone = phone
        return party

    # Explicit Alıcı Ünvanı: / Alıcı: (ERP / e-Fatura) — require colon
    alici = re.search(
        r"Al[ıi]c[ıi]\s*(?:[ÜU]nvan[ıiI]?|Unvan|Ad[ıi])?\s*:\s*"
        r"([A-ZÇĞİÖŞÜa-zçğıöşü].{2,80})",
        text,
        re.I,
    )
    if alici:
        cand = re.split(
            r"\s{2,}|Adres\s*:|Tel(?:efon)?\s*:|Vergi|VKN|TCKN|Özelleştirme|Senaryo|Fatura",
            alici.group(1),
            maxsplit=1,
        )[0].strip(" :.-[]{}")
        cand = re.sub(
            r"^(?:şube|sube|[ÜU]nvan[ıiI]?|Unvan|Ad[ıi]|Bilgileri)\)?\s*:?\s*",
            "",
            cand,
            flags=re.I,
        ).strip(" :.-[]{}()")
        if (
            len(cand) >= 3
            and not re.search(r"Nihai|T[uü]ketici|Bilgileri", cand, re.I)
        ):
            party.name = cand[:120]

    sayin = re.search(r"\bSAYIN\b", text, re.I)
    if not sayin and not party.name:
        # HF / ERP layouts: Alıcı Bilgileri without SAYIN
        role_tid = find_role_tax_id(text, "customer")
        if role_tid:
            party.taxId, party.taxIdScheme = role_tid
        alici_sec = re.search(
            r"(?is)Al[ıi]c[ıi]\s*(?:[ÜU]nvan[ıiI]?|Unvan)\s*:\s*"
            r"([A-ZÇĞİÖŞÜa-zçğıöşü0-9 .&'\-]{3,90})",
            text,
        )
        if alici_sec:
            cand = alici_sec.group(1).strip(" :.-")
            cand = re.split(r"\s{2,}|VKN|TCKN|Adres|Vergi", cand, maxsplit=1)[0].strip()
            if len(cand) >= 3 and not re.match(r"^(?:Bilgileri)\b", cand, re.I):
                party.name = cand[:120]
        if party.name or party.taxId:
            return party
        return party
    if sayin:
        block = text[sayin.start() :]
        lines = []
        for ln in block.splitlines():
            cleaned = re.sub(
                r"\s{2,}(Özelleştirme|Senaryo|Fatura\s+Tipi|Fatura\s+No|Fatura\s+Tarihi|Fatura\s+Saati|Sipari[sş].*|Düzenleme.*).*$",
                "",
                ln,
                flags=re.I,
            ).strip()
            if cleaned:
                lines.append(cleaned)
        name_parts: list[str] = []
        addr_parts: list[str] = []
        # Name may sit on the SAYIN line itself: "SAYIN AHMET YILMAZ"
        if lines:
            first = re.sub(r"^\s*SAYIN\s*:?\s*", "", lines[0], flags=re.I).strip()
            if first and not re.match(r"^(Web|E-?Posta|Tel|Vergi|VKN|TCKN|ETTN)\b", first, re.I):
                if re.search(
                    r"\b(mah\.|Mah\.|Bul\.|Cad\.|Sk\.|Sok\.|No:|daire|sitesi|Apartman|Blok)\b",
                    first,
                    re.I,
                ) or re.search(r"\b\d{5}\b", first):
                    addr_parts.append(first)
                else:
                    name_parts.append(first)
        for ln in lines[1:]:
            if re.match(r"^(Web|E-?Posta|Tel|Vergi|VKN|TCKN|ETTN|S[ıi]ra|Mal|NOTLAR|Not:)", ln, re.I):
                break
            if re.search(
                r"\b(mah\.|Mah\.|Bul\.|Cad\.|Sk\.|Sok\.|No:|daire|sitesi|Apartman|Blok)\b",
                ln,
                re.I,
            ) or re.search(r"\b\d{5}\b", ln) or re.search(
                r"/\s*[A-ZÇĞİÖŞÜa-zçğıöşü]{3,}", ln
            ):
                addr_parts.append(ln)
                continue
            if not name_parts:
                name_parts.append(ln)
            elif not addr_parts and len(ln) < 80:
                name_parts.append(ln)
        if not party.name:
            party.name = " ".join(name_parts).strip() or None
        if party.name:
            party.name = re.split(
                r"\s*[|\[]?\s*(?:Senaryo|Fatura\s+No|Fatura\s+Tipi|Özelleştirme)\b",
                party.name,
                maxsplit=1,
                flags=re.I,
            )[0].strip(" ,|;:-")
            halves = party.name.split()
            mid = len(halves) // 2
            if mid > 0 and " ".join(halves[:mid]) == " ".join(halves[mid:]):
                party.name = " ".join(halves[:mid])
            if not party.name:
                party.name = None
        party.address = ", ".join(addr_parts) or None
        if party.address:
            # Prefer compact ilçe/il form; drop marketing OCR glued on same line
            cityish = re.search(
                r"([A-ZÇĞİÍÖŞÜa-zçğıíöşü]{3,}\s*/\s*[A-ZÇĞİÍÖŞÜa-zçğıíöşü]{3,})",
                party.address,
            )
            if cityish:
                party.address = (
                    cityish.group(1)
                    .replace("Í", "İ")
                    .replace("í", "i")
                    .strip()
                )
        near = block[:1500]
        phone_raw = first_match(near, r"(?:Tel|Telefon)\s*:?\s*([0-9\s()\-]{10,})")
        if phone_raw:
            party.phone = re.sub(r"\D", "", phone_raw) or None
            if party.phone and len(party.phone) < 10:
                party.phone = None
    else:
        near = text[alici.start() : alici.start() + 1500] if alici else text[:1500]

    # Prefer Alıcı / SAYIN-region tax id; never steal from satıcı head
    role_tid = find_role_tax_id(text, "customer")
    region_tid = find_tax_id_in_region(near) if near else None
    chosen = role_tid or region_tid
    if chosen:
        party.taxId, party.taxIdScheme = chosen
    party.taxOffice = (first_match(near, r"Vergi\s*Dairesi\s*:?\s*([^\n]+)") or "").split("  ")[0].strip() or None
    if party.taxOffice:
        party.taxOffice = sanitize_tax_office(party.taxOffice)
    party.email = first_match(near, r"E-?Posta\s*:?\s*([^\s]+)")
    return party


def parse_markdown_tables(md: str) -> list[Line]:
    """Parse Docling markdown pipe-tables into invoice lines."""
    lines: list[Line] = []
    rows: list[list[str]] = []
    for ln in md.splitlines():
        if not ln.strip().startswith("|"):
            if rows:
                lines.extend(_rows_to_lines(rows))
                rows = []
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", c.replace(" ", "")) for c in cells if c):
            continue
        rows.append(cells)
    if rows:
        lines.extend(_rows_to_lines(rows))
    return lines


def _rows_to_lines(rows: list[list[str]]) -> list[Line]:
    if not rows:
        return []
    header = [re.sub(r"\s+", " ", h.lower()).strip() for h in rows[0]]
    # Skip non-item summary tables
    header_join = " ".join(header)
    if "mal hizmet toplam" in header_join or (
        len(rows[0]) <= 2 and any("tutar" in h for h in header)
    ):
        # totals-only mini table — ignore for lines
        return []

    def col_prefer(*names: str, exclude: tuple[str, ...] = ()) -> int | None:
        """Pick best header index: prefer longer/more specific label; skip excluded."""
        best_i: int | None = None
        best_score = -1
        for n in names:
            n_l = n.lower()
            for i, h in enumerate(header):
                if n_l not in h:
                    continue
                if any(ex in h for ex in exclude):
                    continue
                # Prefer exact header equality over looser substring (açıklama vs tanım)
                score = len(n_l) * 3 + len(h)
                if h == n_l:
                    score += 200
                elif h.endswith(n_l) or h.startswith(n_l):
                    score += 40
                if score > best_score:
                    best_score = score
                    best_i = i
        return best_i

    def desc_col() -> int | None:
        """Description: prefer filled Tanım/Mal Hizmet over empty Açıklama."""
        candidates: list[int] = []
        for n in (
            "mal/hizmet tanımı",
            "mal hizmet tanımı",
            "malzeme/hizmet",
            "mal hizmet",
            "mal/hizmet",
            "tanım",
            "cinsi",
            "açıklama",
        ):
            i = col_prefer(n, exclude=("oran", "tutar", "miktar", "kod", "no"))
            if i is not None and i not in candidates:
                candidates.append(i)
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        # Prefer column with more non-empty body cells
        best_i = candidates[0]
        best_fill = -1
        for i in candidates:
            fill = 0
            for row in rows[1:]:
                if i < len(row) and row[i].strip():
                    fill += 1
            # Prefer tanım/mal over açıklama on tie
            bonus = 0
            if "tanım" in header[i] or "hizmet" in header[i] or "cinsi" in header[i]:
                bonus = 10
            if fill + bonus > best_fill:
                best_fill = fill + bonus
                best_i = i
        return best_i

    idx_id = col_prefer("sıra no", "sira no", "sıra", "sira")
    idx_name = desc_col()
    idx_code = col_prefer(
        "malzeme no",
        "ürün kodu",
        "urun kodu",
        "stok kodu",
        "satıcı ürün",
        "kod",
        exclude=("posta", "barkod no"),
    )
    idx_qty = col_prefer("miktar")
    idx_unit_price = col_prefer("birim fiyat", "birim fiyati", "fiyat")
    idx_line = col_prefer(
        "mal hizmet tutarı",
        "mal hizmet tutari",
        "hizmet tutarı",
        "satır tutarı",
        "toplam tutar",
        "tutar",
        exclude=("kdv", "vergi", "iskonto", "matrah", "birim"),
    )
    idx_vat_rate = col_prefer("kdv oranı", "vergi oranı", "kdv oran", exclude=("tutar",))
    idx_vat_amt = col_prefer("kdv tutarı", "kdv tutari", "vergi tutarı")
    if idx_name is None:
        idx_name = col_prefer("açıklama", "tanım", "cinsi")
    if idx_line is None:
        # Last resort: plain tutar still excluding kdv/iskonto
        idx_line = col_prefer("tutarı", "tutari", "tutar", exclude=("kdv", "vergi", "iskonto", "matrah"))
    if idx_id is None:
        idx_id = 0
    if idx_vat_rate is None:
        idx_vat_rate = col_prefer("oran", exclude=("iskonto",))
    # Never treat bare "Birim" (Adet/C62) as unit price
    if idx_unit_price is None:
        idx_unit_price = col_prefer("birim fiyat", "fiyat", exclude=("miktar",))

    out: list[Line] = []
    for row in rows[1:]:
        if not row:
            continue
        raw_id = row[idx_id] if idx_id is not None and idx_id < len(row) else row[0]
        id_s = raw_id.strip()
        # Accept Sıra No (1,2,…) or numeric Malzeme/Stok codes as row ids
        if not re.match(r"^\d{1,14}$", id_s):
            continue
        name = row[idx_name] if idx_name is not None and idx_name < len(row) else None
        code = row[idx_code] if idx_code is not None and idx_code < len(row) else None
        # When first column is product code (not sıra), use it as code if name empty of code
        if idx_id == 0 and code is None and re.match(r"^\d{5,}$", id_s):
            code = id_s
        if code and name and code not in name:
            name = f"{code} {name}"
        elif code and not name:
            name = code
        qty_raw = row[idx_qty] if idx_qty is not None and idx_qty < len(row) else None
        qty = None
        unit = None
        if qty_raw:
            qm = re.match(r"([\d.,]+)\s*([A-Za-zÇĞİÖŞÜçğıöşü]+)?", qty_raw.strip())
            if qm:
                qty = float(qm.group(1).replace(",", "."))
                unit = qm.group(2) or "Adet"
        unit_price = (
            parse_tr_money(row[idx_unit_price])
            if idx_unit_price is not None and idx_unit_price < len(row)
            else None
        )
        line_total = (
            parse_tr_money(row[idx_line]) if idx_line is not None and idx_line < len(row) else None
        )
        vat_rate = (
            parse_percent(row[idx_vat_rate])
            if idx_vat_rate is not None and idx_vat_rate < len(row)
            else None
        )
        vat_amount = (
            parse_tr_money(row[idx_vat_amt])
            if idx_vat_amt is not None and idx_vat_amt < len(row)
            else None
        )
        # Heal swapped KDV Tutarı vs Mal Hizmet Tutarı (generic)
        if (
            unit_price is not None
            and qty
            and qty > 0
            and line_total is not None
            and vat_amount is not None
        ):
            expected = round(unit_price * qty, 2)
            if abs(line_total - expected) > 1.0 and abs(vat_amount - expected) <= 1.0:
                line_total, vat_amount = vat_amount, line_total
            elif abs(line_total - expected) > 1.0 and abs(line_total - expected * (vat_rate or 0) / 100) <= 1.0:
                # line_total looks like VAT of unit*qty
                if vat_amount is None or abs(vat_amount - expected) > 1.0:
                    vat_amount, line_total = line_total, expected
        elif unit_price is not None and qty and qty > 0 and line_total is not None:
            expected = round(unit_price * qty, 2)
            if abs(line_total - expected) > 1.0 and abs(line_total - expected) > expected * 0.4:
                # Prefer recomputed extension when "tutar" column was VAT/iskonto
                if vat_amount is not None and abs(line_total - vat_amount) < 0.05:
                    line_total = expected
                elif expected >= 1 and (line_total / expected) < 0.45:
                    if vat_amount is None:
                        vat_amount = line_total
                    line_total = expected

        out.append(
            Line(
                id=id_s if len(id_s) <= 6 else str(len(out) + 1),
                name=(name or "").strip() or None,
                quantity=qty,
                unit=unit,
                unitPrice=unit_price,
                vatRate=vat_rate,
                vatAmount=vat_amount,
                lineTotal=line_total,
            )
        )
    return out


def parse_text_invoice(text: str, file_name: str = "") -> Invoice:
    text = text.replace("\x0c", "\n")
    # Unescape common HTML entities from Docling markdown tables
    text = (
        text.replace("&#124;", "|")
        .replace("&nbsp;", " ")
        .replace("&amp;", "&")
    )
    text = normalize_ocr_text(text)
    doc_type: Literal["earsiv", "efatura", "ubl", "unknown"] = "unknown"
    profile = right_field(text, "Senaryo") or first_match(text, r"ProfileID\s*:?\s*([A-Z0-9_]+)")
    if profile:
        profile = re.sub(r"[^A-Z0-9_]", "", profile.upper().replace("İ", "I").replace("İ", "I"))
    inv_type = right_field(text, "Fatura Tipi")
    inv_type = normalize_invoice_type(inv_type)
    if profile:
        # Glued soup after Senaryo — keep leading profile token only
        profile = re.split(r"(?:FATURA|SATIS|TR\d)", profile, maxsplit=1)[0] or profile
        profile = re.sub(r"[^A-Z0-9_]", "", profile)[:32] or None
        if profile and len(profile) > 24 and not re.search(
            r"EARSIV|TEMEL|TICARI|IHRACAT|KAMU", profile
        ):
            profile = None

    if profile and "EARSIV" in profile:
        doc_type = "earsiv"
    elif profile and re.search(
        r"TEMEL|TICARI|IHRACAT|YOLCU|KAMU|ENERJI|ILAC|HKS", profile
    ):
        doc_type = "efatura"
    elif re.search(r"e-?Ar[sş]iv(?:\s+Fatura)?|EARSIVFATURA|e-?Ar[sş]iv\s+izni", text, re.I):
        doc_type = "earsiv"
    elif re.search(
        r"e-?Fatura|EFATURA|TICARIFATURA|TEMELFATURA|IHRACATFATURA|KAMUFATURA",
        text,
        re.I,
    ):
        doc_type = "efatura"

    inv_no = gib_invoice_number(text, file_name)
    labeled_no = (
        right_field(text, "Fatura No")
        or right_field(text, "Fatera No")
        or right_field(text, "Fatara Na")
        or right_field(text, "Fataca Na")
        or right_field(text, "Patara Na")
        or right_field(text, "Belge No")
        or right_field(text, "BELGE NO")
        or first_match(
            text,
            r"(?:B[EÉ]?[LİI1]?GE|BELGE)\s*N[O0]\s*[:\-.]?\s*([A-Za-z]{2,5}[\s\-]*\d{10,20}|\d{12,22})",
        )
    )
    if labeled_no:
        cleaned = re.sub(r"[^A-Za-z0-9]", "", labeled_no).upper()
        if is_gib_invoice_number(cleaned):
            inv_no = cleaned
    if inv_no:
        inv_no = re.sub(r"\s+", "", inv_no).upper()
        if re.search(r"[A-Z]", inv_no) and is_gib_invoice_number(inv_no):
            inv_no = normalize_gib_invoice_number(inv_no)
        elif not re.fullmatch(r"\d{12,22}", inv_no):
            inv_no = None

    issue_date, issue_time = prefer_invoice_issue_date(text)
    # Align OCR date year with GİB serial year when off by one digit (2036 vs 2026)
    if inv_no and issue_date:
        ym = re.match(r"[A-Z]{2,5}(\d{4})", inv_no)
        if ym:
            series_year = int(ym.group(1))
            try:
                date_year = int(issue_date[:4])
            except ValueError:
                date_year = 0
            if 1990 <= series_year <= 2100 and date_year != series_year:
                sa, sb = f"{date_year:04d}", f"{series_year:04d}"
                diffs = sum(a != b for a, b in zip(sa, sb))
                if diffs <= 1 and abs(date_year - series_year) <= 100:
                    issue_date = f"{series_year:04d}{issue_date[4:]}"
    for label in (
        "Fatura Saati",
        "Düzenleme Zamanı",
        "Düzenleme Zamans",
        "Düzonlenme Zamani",
        "Duzenleme Zamani",
        "Oluşma Zamanı",
    ):
        raw = right_field(text, label) or first_match(
            text, rf"{label}\s*:?\s*(\d{{1,2}}:\d{{2}}:\d{{2}})"
        )
        if raw and not issue_time:
            tm = re.search(r"(\d{1,2}:\d{2}:\d{2})", raw)
            if tm:
                issue_time = tm.group(1)
    if not issue_time:
        tm = first_match(text, r"(?:Zamani|Zamanı|Saati)\s*:?\s*(\d{1,2}:\d{2}:\d{2})")
        if tm:
            issue_time = tm

    uuid = extract_ettn_candidate(text)
    if not uuid:
        uuid = first_match(
            text,
            r"ETTN\s*:?\s*([0-9a-fA-FOoİIlLSBbPpGg]{8}-[0-9a-fA-FOoİIlLSBbPpGg]{4}-[0-9a-fA-FOoİIlLSBbPpGg]{4}-[0-9a-fA-FOoİIlLSBbPpGg]{4}-[0-9a-fA-FOoİIlLSBbPpGg]{12})",
        )
        if uuid:
            uuid = normalize_ocr_uuid(uuid)

    net = (
        labeled_amount(text, "Mal Hizmet Toplam Tutarı")
        or labeled_amount(text, "NET TOPLAM")
        or labeled_amount(text, "Ara Toplam")
        or labeled_amount(text, "Vergiler Hariç Toplam")
    )
    discount = (
        labeled_amount(text, "Toplam [İI]skonto")
        or labeled_amount(text, "TOPLAM [İI]SKONTO")
        or labeled_amount(text, r"TOPLAM\s+ISKONTO")
        or labeled_amount(text, "İskonto Toplamı")
    )
    matrah = (
        sum_labeled_amounts(text, r"KDV Matrah[ıie]")
        or sum_labeled_amounts(text, r"Kdv Matrah\w*")
    )
    ocr_lines = parse_ocr_line_items(text)
    if ocr_lines:
        ocr_lines = [
            ln
            for ln in ocr_lines
            if ln.name
            and not _is_bank_or_iban_line(ln.name)
            and not re.search(r"(?i)Kredi\s*Kart|Banka\s*Kart|\bIBAN\b|\bTR\d{2}", ln.name or "")
        ]
    if not ocr_lines:
        ocr_lines = parse_retail_pos_lines(text)
    # Docling / GİB HTML→markdown pipe tables (generic; not OCR regex lines)
    md_lines = parse_markdown_tables(text)
    if md_lines:
        if not ocr_lines or (
            _lines_useful(md_lines)
            and (
                not _lines_useful(ocr_lines)
                or sum(1 for l in md_lines if l.name) > sum(1 for l in ocr_lines if l.name)
            )
        ):
            ocr_lines = md_lines
    retail_fiş = bool(
        re.search(r"\badet\s*[x×X]\b|B[İI]LG[İI]\s*F[İI][ŞS]|TOPKDV|BELGE\s*N[O0]|BEBGE\s*N[O0]", text, re.I)
    )
    lines_sum = None
    if ocr_lines:
        totals_present = [l.lineTotal for l in ocr_lines if l.lineTotal is not None]
        if totals_present:
            lines_sum = round(sum(totals_present), 2)
    net_minus_disc = (
        round(net - discount, 2) if net is not None and discount and discount > 0 else None
    )
    # Prefer matrah / net-iskonto when line OCR is incomplete vs labeled totals
    if (
        matrah is not None
        and lines_sum is not None
        and abs(lines_sum - matrah) > 0.5
        and (net_minus_disc is None or abs(matrah - net_minus_disc) <= 1.0)
    ):
        line_ext = matrah
    elif (
        lines_sum is not None
        and not retail_fiş
        and (matrah is None or abs(lines_sum - (matrah or 0)) <= 1.0 or net_minus_disc is None)
    ):
        line_ext = lines_sum
    else:
        line_ext = (
            matrah
            if matrah is not None
            else (net_minus_disc if net_minus_disc is not None else net)
        )

    payable = extract_payable_from_ocr(text)
    ara = labeled_amount(text, r"ARA\s*TOPLAM")
    # Fix OCR 'TOPLAM 13.499,00' when ARA/line sum is 3.499,00
    if payable is not None and ara is not None and payable > ara * 1.5 and nearly_equal(payable - 10000, ara, 1.0):
        payable = ara
    if payable is not None and lines_sum is not None and payable > lines_sum * 1.5:
        # leading digit OCR junk
        s = f"{payable:.2f}".replace(".", "")
        if s.startswith("1") and nearly_equal(payable - 10000, lines_sum, 1.0):
            payable = lines_sum
        elif nearly_equal(ara or -1, lines_sum, 0.05):
            payable = lines_sum
    tax_inclusive = (
        labeled_amount(text, r"VERG[İIEÉ]\s+DAH[İI]L\s+TOPLAM\s+TUTAR")
        or labeled_amount(text, "Vergiler Dahil Toplam Tutar")
        or labeled_amount(text, r"VERGI\s+DAHIL\s+TOPLAM\s+TUTAR")
        or labeled_amount(text, "Genel Toplam")
        or payable
    )
    # Prefer tax-inclusive / ödenecek over bank 0.01 rounding when both exist
    odenecek = labeled_amount(text, r"[ÖO]DENECEK\s+TUTAR")
    vergi_dahil = labeled_amount(text, r"VERG[İIEÉ]\s+DAH[İI]L\s+TOPLAM\s+TUTAR") or labeled_amount(
        text, r"VERGI\s+DAHIL\s+TOPLAM\s+TUTAR"
    )
    if odenecek is not None:
        payable = odenecek
    if vergi_dahil is not None:
        tax_inclusive = vergi_dahil
    if payable is None and tax_inclusive is not None:
        payable = tax_inclusive
    if tax_inclusive is None and payable is not None:
        tax_inclusive = payable
    # Photo OCR often flips the leading digit (36.994 vs 26.994). Prefer the
    # candidate that matches line totals × (1+KDV) when available.
    if lines_sum and lines_sum >= 10:
        expected_cands = [round(lines_sum * (1 + r), 2) for r in (0.20, 0.18, 0.10, 0.08, 0.01, 0.0)]
        pool = [a for a in (payable, tax_inclusive, odenecek, vergi_dahil) if a is not None]
        best = None
        for exp in expected_cands:
            for a in pool:
                if abs(a - exp) < 1.0:
                    best = exp
                    break
            if best is not None:
                break
        if best is None:
            for exp in expected_cands:
                for a in pool:
                    # leading-digit slip (~10k on mid invoices)
                    if 5000 <= abs(a - exp) <= 15000:
                        best = exp
                        break
                if best is not None:
                    break
        if best is not None:
            payable = best
            tax_inclusive = best

    bank_pay = None
    bank_name_pay = None
    bank_m = re.search(
        rf"(?:BANKASI|Banka\s*/\s*Kredi\s*Kart[ıi]|KRED[İI]\s*KART[İI]?)"
        rf"[^\n]{{0,48}}?"
        rf"({_MONEY_TOKEN})\s*TRY",
        text,
        re.I,
    )
    if bank_m:
        bank_pay = parse_tr_money(bank_m.group(1))
    # Alınan ödemeler: "… (TEK) - 2.008,30 TRY" (bank/POS without BANKASI)
    taken_m = re.search(
        rf"(?:Al[ıi]nan\s+[ÖO]demeler|Ödeme\s*[ŞSSş]ekli)\s*:?\s*"
        rf"([\s\S]{{0,240}}?)"
        rf"(?P<bname>[A-ZÇĞİÖŞÜa-zçğıöşü][A-ZÇĞİÖŞÜa-zçğıöşü0-9 .]{{2,40}}?)"
        rf"\s*\((?:TEK|[0-9]+\s*Taksit|[A-ZÇĞİÖŞÜa-zçğıöşü0-9 ]{{1,16}})\)\s*[-–]?\s*"
        rf"(?P<amt>{_MONEY_TOKEN})\s*TRY",
        text,
        re.I,
    )
    if not taken_m:
        taken_m = re.search(
            rf"(?m)^(?P<bname>[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü0-9 .]{{2,40}}?)\s*"
            rf"\((?:TEK|[0-9]+\s*Taksit)\)\s*[-–]?\s*(?P<amt>{_MONEY_TOKEN})\s*TRY\b",
            text,
            re.I,
        )
    if taken_m:
        bank_pay = bank_pay or parse_tr_money(taken_m.group("amt"))
        bank_name_pay = re.sub(r"\s+", " ", taken_m.group("bname")).strip()[:80]
    # Only use bank if ödenecek missing
    if payable is None and bank_pay is not None:
        payable = bank_pay
        tax_inclusive = tax_inclusive or bank_pay
    vat = extract_vat_amount(text)
    multi_rate_kdv = len(re.findall(r"[KX]DV\s*\(\s*%", text, re.I)) >= 2
    if vat is None:
        vat = (
            labeled_amount(text, r"TOPKDV")
            or first_match_money(text, rf"TOPKDV\s*\*\s*({_MONEY_TOKEN})")
            or first_match_money(text, rf"(?m)^KDV\s*\*\s*({_MONEY_TOKEN})")
        )
    # Reject product-code fragments mistaken for KDV (e.g. 1.1502)
    base_chk = payable or tax_inclusive
    if vat is not None and base_chk and vat < 10 and base_chk > 100:
        vat = None
    # Tax-inclusive retail: KDV amount sits between iskonto and vergi dahil without a label
    if (vat is None or vat == 0) and (payable or tax_inclusive):
        base = payable or tax_inclusive or 0
        block = re.search(
            rf"Toplam\s+[İI]skonto[\s\S]{{0,220}}?(?:Vergiler\s+Dahil|[ÖO]DENECEK)",
            text,
            re.I,
        )
        if block and base > 0:
            cands = [
                parse_tr_money(x)
                for x in re.findall(rf"({_MONEY_TOKEN})\s*TRY", block.group(0), re.I)
            ]
            cands = [a for a in cands if a and 0 < a < base * 0.5]
            # Prefer amount ≈ base * rate/(100+rate) for 20%
            target = round(base * 20 / 120, 2)
            if cands:
                best = min(cands, key=lambda a: abs(a - target))
                if abs(best - target) <= max(2.0, base * 0.02):
                    vat = best
        if (vat is None or vat == 0) and base > 0 and re.search(
            r"%\s*20|KDV\s*%?\s*20|\b20\s*KD|\bKD[VİI]?\s*20", text, re.I
        ):
            # Last resort for KDV-dahil totals
            vat = round(base * 20 / 120, 2)
    if vat is None and ocr_lines:
        line_vats = [l.vatAmount for l in ocr_lines if l.vatAmount is not None]
        if line_vats:
            vat = round(sum(line_vats), 2)
        elif lines_sum is not None:
            # Infer from dominant line VAT rate
            rates = [l.vatRate for l in ocr_lines if l.vatRate is not None]
            if rates:
                rate = max(set(rates), key=rates.count)
                vat = round(lines_sum * (rate / 100.0), 2)
    withholding = extract_withholding_vat_amount(text)

    # Retail POS: line totals are usually tax-inclusive
    if retail_fiş and lines_sum is not None:
        # Keep only product lines that match TOPLAM when OCR invented footer junk
        if payable is not None and ocr_lines:
            matching = [
                l
                for l in ocr_lines
                if l.lineTotal is not None and nearly_equal(l.lineTotal, payable, 1.5)
            ]
            if matching:
                # Prefer vowel-rich product names
                matching.sort(
                    key=lambda l: -len(re.findall(r"[aeıioöuüAEIİOÖUÜ]", l.name or ""))
                )
                ocr_lines = matching[:1]
                lines_sum = ocr_lines[0].lineTotal
        if payable is None or (payable is not None and nearly_equal(payable, lines_sum, 1.0)):
            payable = lines_sum
        tax_inclusive = payable
        if vat is not None and payable is not None and payable > vat:
            line_ext = round(payable - vat, 2)
        elif line_ext is None and payable is not None and vat is not None:
            line_ext = round(payable - vat, 2)

    # Reconcile: heal missed multi-rate VAT from lines/totals
    if line_ext is None and tax_inclusive is not None and vat and vat > 0:
        line_ext = round(tax_inclusive - vat, 2)
    if line_ext is not None and tax_inclusive is not None and tax_inclusive >= line_ext:
        implied = round(tax_inclusive - line_ext, 2)
        if nearly_equal(line_ext, tax_inclusive, 0.05) and vat and vat > 0.5:
            # Tax-inclusive totals: matrah = dahil - KDV
            line_ext = round(tax_inclusive - vat, 2)
        elif vat is None:
            vat = implied
        elif abs((line_ext + vat) - tax_inclusive) > 0.05:
            # Don't overwrite labeled multi-rate KDV with bad implied (incomplete lines)
            if not multi_rate_kdv or abs(implied - vat) <= 1.0:
                if implied > 0.5 or vat < 0.5:
                    vat = implied
            elif matrah is not None and abs((matrah + vat) - tax_inclusive) <= 1.0:
                line_ext = matrah
            elif net_minus_disc is not None and abs((net_minus_disc + vat) - tax_inclusive) <= 1.0:
                line_ext = net_minus_disc
    if payable is None and tax_inclusive is not None:
        payable = (
            round(tax_inclusive - withholding, 2) if withholding is not None else tax_inclusive
        )
    if (
        tax_inclusive is not None
        and payable is not None
        and withholding is None
        and tax_inclusive > payable + 0.05
    ):
        withholding = round(tax_inclusive - payable, 2)

    # Soft 0.01 reconcile: prefer taxInclusive as payable when off by 1 kuruş vs bank
    if (
        tax_inclusive is not None
        and payable is not None
        and abs(tax_inclusive - payable) <= 0.02
    ):
        payable = tax_inclusive

    # Screenshot OCR often misses ödenecek label but has line totals / KDV matrah
    if payable is None and lines_sum is not None and lines_sum > 0:
        if vat is not None and not retail_fiş:
            payable = round(lines_sum + vat, 2)
            tax_inclusive = tax_inclusive or payable
            line_ext = line_ext or lines_sum
        else:
            payable = lines_sum
            tax_inclusive = tax_inclusive or lines_sum
    if payable is None and line_ext is not None and vat is not None:
        payable = round(line_ext + vat, 2)
        tax_inclusive = tax_inclusive or payable

    iban = first_match(text, r"I?İ?BAN\s*:\s*(TR[\d\s]+)")
    if iban:
        iban = re.sub(r"\s+", "", iban).upper()
    bank = first_match(text, r"([A-ZÇĞİÖŞÜa-zçğıöşü ]+BANKASI)\s*/\s*I?İ?BAN")

    # Profile from OCR without colon
    if not profile:
        profile = first_match(text, r"Senaryo\s*:?\s*([A-Z0-9_]+)")
        if profile:
            profile = re.sub(r"[^A-Z0-9_]", "", profile.upper())

    supplier = extract_supplier(text)
    # Retail: "… VD 1234567890" or "İLÇE/1234567890" — not "Müşteri VKN" / Tel
    musteri_vkn = first_match(text, r"M[uüu][sşs]teri\s+VKN\s*:?\s*(\d{10,11})")
    vd_vkn = first_match(text, r"(?:^|\n)[^\n]*\bVD\s*(\d{10})\b")
    spaced_vkn = first_match(
        text,
        r"(?:V\.?\s*D\.?|Vergi\s*Dairesi|Mukellef\w*|Mükellef\w*)[^\n]{0,40}?"
        r"(\d{3}\s+\d{3}\s+\d{4})\b",
    )
    if spaced_vkn:
        spaced_vkn = re.sub(r"\s+", "", spaced_vkn)
    district_vkn = first_match(
        text,
        r"(?:^|\n)\s*(?!Tel|Fax|TCKN|Telefon)[A-ZÇĞİÖŞÜa-zçğıöşü]{3,}"
        r"(?:\s*/\s*|\s+)(\d{10})\b",
    )
    # Reject phone-looking 10-digit (starts with 5) unless GİB checksum validates
    def _ok_vkn(v: str | None) -> str | None:
        if not v or len(v) != 10:
            return None
        if v.startswith("5"):
            return v if is_valid_vkn(v) else None
        return v

    supplier_vkn = _ok_vkn(spaced_vkn) or _ok_vkn(vd_vkn) or _ok_vkn(district_vkn)
    if not supplier_vkn:
        recovered = recover_supplier_vkn_from_header(text)
        if recovered and recovered[1] == "VKN":
            supplier_vkn = recovered[0]
    if supplier_vkn and supplier_vkn != (musteri_vkn or "")[:10]:
        supplier.taxId = supplier_vkn
        supplier.taxIdScheme = "VKN"
    elif not supplier.taxId and supplier_vkn:
        supplier.taxId = supplier_vkn
        supplier.taxIdScheme = "VKN"
    # Spaced VKN in supplier head: "836 014 4393" (or labeled 5-start valid VKN)
    st_now = normalize_ocr_digits(supplier.taxId) or digits_only(supplier.taxId)
    need_recover = not st_now or (
        len(st_now) == 10 and st_now.startswith("5") and not is_valid_vkn(st_now)
    )
    if need_recover:
        recovered = recover_supplier_vkn_from_header(text)
        if recovered and recovered[1] == "VKN":
            supplier.taxId, supplier.taxIdScheme = recovered
        else:
            head_vkn = first_match(
                text[:2000],
                r"(?<!Tel\s)(?<!TCKN:\s)(?<!TCKN:\s)(\d{3}\s+\d{3}\s+\d{4})\b",
            )
            if head_vkn:
                hv = re.sub(r"\s+", "", head_vkn)
                if _ok_vkn(hv):
                    supplier.taxId = hv
                    supplier.taxIdScheme = "VKN"

    if supplier.name:
        supplier.name = normalize_company_legal_ocr(supplier.name)
        # OCR: common misspellings of MAGAZACILIK
        supplier.name = re.sub(r"\bHAGAZACILIK\b", "MAGAZACILIK", supplier.name, flags=re.I)
        supplier.name = re.sub(r"\bMA[ČĆ]AZACILIK\b", "MAGAZACILIK", supplier.name, flags=re.I)
        # Strip leading short POS/OCR prefixes ("6A ", "hgz ") — not English "THE "
        supplier.name = re.sub(
            r"^(?:[a-z]{1,3}\d*|\d{1,3}[A-Za-z]{0,2})\s+(?=\S{3,})",
            "",
            supplier.name,
        ).strip()
        supplier.name = re.sub(r"^#+\s*", "", supplier.name).strip()
        # Drop POS store-label tails glued onto company title (generic)
        supplier.name = re.split(
            r"\s+(?:Hgz\s*Ad[iı]|Mgz\s*Kodu|Ma[ğg]aza\s*(?:Ad[iı]|Kodu))\s*:",
            supplier.name,
            maxsplit=1,
            flags=re.I,
        )[0].strip()[:180]
        # Drop address tokens glued onto company title (any issuer)
        if re.search(r"\s+(?:MAH\.|CAD\.|SK\.|SOK\.|BULVAR|NO:)\s+", supplier.name, re.I):
            # Address-then-entity: keep legal-form tail
            ent = re.search(
                r"((?:[A-ZÇĞİÖŞÜ]+\s+){0,6}T[İI]CARET\s+A\.?\s*[SŞ]\.?)\s*$",
                supplier.name,
                re.I,
            ) or re.search(
                r"((?:[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü0-9 .&-]{0,30})?"
                r"(?:MA[ĞG]AZACILIK|SANAY[İI]).{0,24}"
                r"(?:A\.?\s*[SŞ]\.?|LTD\.?\s*ŞT[İI]?\.?))",
                supplier.name,
                re.I,
            )
            before = re.split(
                r"\s+(?:MAH\.|CAD\.|SK\.|SOK\.|BULVAR|NO:)",
                supplier.name,
                maxsplit=1,
                flags=re.I,
            )[0].strip()
            if ent and _party_name_quality(ent.group(1)) >= _party_name_quality(before):
                supplier.name = re.sub(r"\s+", " ", ent.group(1)).strip()[:180]
            elif _party_name_quality(before) >= 8:
                supplier.name = before[:180]
            elif ent:
                supplier.name = re.sub(r"\s+", " ", ent.group(1)).strip()[:180]
            else:
                supplier.name = re.split(
                    r"\s+(?:MAH\.|CAD\.|SK\.|SOK\.|BULVAR|NO:|Kap[ıi]\s*No|PROF\.)",
                    supplier.name,
                    maxsplit=1,
                    flags=re.I,
                )[0].strip()[:180]
        else:
            supplier.name = re.split(
                r"\s+(?:Kap[ıi]\s*No|PROF\.)",
                supplier.name,
                maxsplit=1,
                flags=re.I,
            )[0].strip()[:180]
        # Drop OCR junk prefixes (short codes) when a better legal-form title exists later.
        # Keep "THE … ANONİM ŞİRKETİ" — not POS junk like "6A MAGAZA…".
        if (
            len(supplier.name) < 4
            or "mgzkodu" in supplier.name.lower()
            or re.match(r"^[a-z]{2,4}\d", supplier.name, re.I)
            or (
                re.match(r"^\d*[A-Z]{1,3}\s+", supplier.name)
                and not re.search(
                    r"ANON[İI]M|Ş[İI]RKET|TEKNOLOJ|T[İI]CARET|SANAY|MA[ĞG]AZA",
                    supplier.name,
                    re.I,
                )
            )
        ):
            # Use A.\s*Ş (dot required) so case-insensitive "aş" in "Taşıyıcı" cannot match.
            better = first_match(
                text,
                r"((?:[A-ZÇĞİÖŞÜ0-9][A-ZÇĞİÖŞÜa-zçğıöşü0-9 .&-]{4,60}?)"
                r"(?:MA[ĞG]AZACILIK|T[İI]CARET|SANAY[İI]).{0,40}?"
                r"(?:A\.\s*Ş\.?|LTD\.?\s*ŞT[İI]))",
            )
            if better and not re.search(
                r"(?:Arac[ıi]|Ta[şs][ıi]y[ıi]c[ıi])\s+Firma",
                better,
                re.I,
            ):
                supplier.name = re.sub(r"\s+", " ", better).strip()[:180]
                supplier.name = re.sub(
                    r"^(?:[a-z]{1,3}\d*|\d{1,3}[A-Za-z]{0,2})\s+",
                    "",
                    supplier.name,
                ).strip()[:180]

    customer = extract_customer(text)
    if musteri_vkn:
        customer.taxId = musteri_vkn
        customer.taxIdScheme = "VKN" if len(musteri_vkn) == 10 else "TCKN"
        # Customer name wrongly filled with supplier / OCR junk
        cn = (customer.name or "").upper()
        sn = (supplier.name or "").upper()
        if (
            not customer.name
            or (sn and sn[:12] and sn[:12] in cn)
            or re.search(r"sikoyet|[şs]ikayet|www\.|http", customer.name or "", re.I)
        ):
            # Keep a plausible person name from SAYIN even when TCKN present
            if customer.name and re.search(
                r"[A-ZÇĞİÖŞÜa-zçğıöşü]{2,}\s+[A-ZÇĞİÖŞÜa-zçğıöşü]{2,}",
                customer.name,
            ):
                pass
            else:
                customer.name = "Nihai Tüketici"
    # POS: "MÜŞTERİ: …" when name still missing
    if not customer.name or customer.name == "Nihai Tüketici":
        m = re.search(
            r"SAYIN\s*:?\s*([A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü .'\-]{2,60})",
            text,
            re.I,
        )
        if m:
            cand = re.split(
                r"\s{2,}|Telefon|TC\b|TCKN|E-?Mail|Vergi|Mah\.|Cad\.|Sok\.",
                m.group(1),
                maxsplit=1,
            )[0].strip(" :.-")
            if len(cand) >= 5 and not re.search(r"Nihai|T[uü]ketici", cand, re.I):
                customer.name = cand[:80]
    # Photo OCR often drops the SAYIN label; recover person name near TCKN
    if not customer.name or customer.name == "Nihai Tüketici":
        tckn_m = re.search(r"\bTCKN\s*:?\s*\d{11}\b", text, re.I)
        if tckn_m:
            before = text[max(0, tckn_m.start() - 500) : tckn_m.start()]
            for ln in reversed([x.strip() for x in before.splitlines() if x.strip()][-8:]):
                if re.search(
                    r"(?i)ANON[İI]M|Ş[İI]RKET|LTD|A\.?\s*Ş|VKN|ETTN|e-?Ar[sş]iv|"
                    r"Fatura|Senaryo|Sipari|Mah\.|Cad\.|Sok\.|No:|Vergi",
                    ln,
                ):
                    continue
                words = ln.split()
                if 2 <= len(words) <= 6 and re.match(r"^[A-ZÇĞİÖŞÜ]", ln):
                    letters = re.sub(r"[^A-Za-zÇĞİÖŞÜçğıöşü]", "", ln)
                    if len(letters) >= 6:
                        customer.name = ln[:80]
                        break
    if not customer.name or customer.name == "Nihai Tüketici":
        m = re.search(
            r"M[ÜU][ŞS]TER[İIÍ]\s*:?\s*([A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü ]{2,40})",
            text,
            re.I,
        )
        if m:
            cand = re.split(
                r"\s{2,}|Telefon|TC\b|E-?Mail|Vergi|Mah\.|Cad\.",
                m.group(1),
                maxsplit=1,
            )[0].strip(" :.-")
            if len(cand) >= 5 and not re.search(r"Nihai|T[uü]ketici", cand, re.I):
                customer.name = cand[:80]
    if not customer.name:
        near_card = first_match(text, r"(?m)^([A-ZÇĞİÖŞÜ ]{5,40})\s*\nAID:")
        if near_card and not re.search(r"KART|CHIP|ONAY|AID|BANKA", near_card, re.I):
            customer.name = near_card.strip()[:120]
        else:
            customer.name = "Nihai Tüketici"

    # Strip tutar-yazısı / chrome mistaken for customer name
    if customer.name and (
        _is_amount_in_words_name(customer.name)
        or _is_registry_or_chrome_line(customer.name)
        or re.match(
            r"(?i)^(?:Özelleştirme|Ozellestirme|UBL|ERP\s*Fatura|e-Belge|table|image)\b",
            customer.name,
        )
    ):
        customer.name = None
    from tax_id import is_placeholder_tax_id, is_valid_tax_id

    if customer.taxId and (
        is_placeholder_tax_id(customer.taxId)
        or not is_valid_tax_id(customer.taxId, customer.taxIdScheme)
    ):
        customer.taxId = None
        customer.taxIdScheme = None
    if not customer.name:
        # Prefer unlabeled person name above placeholder TCKN/VKN (Media Markt)
        m = re.search(
            r"(?m)^([A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü'. -]{4,60})\s*\n+"
            r"(?:TCKN\s*/\s*VKN|VKN\s*/\s*TCKN|TCKN|VKN)\s*:?\s*\d{10,11}",
            text,
        )
        if m:
            cand = m.group(1).strip()
            if (
                not _is_amount_in_words_name(cand)
                and not _is_registry_or_chrome_line(cand)
                and not re.search(r"(?i)ANON[İI]M|Ş[İI]RKET|LTD|MA[ĞG]AZA|MARKET", cand)
            ):
                customer.name = cand[:80]
        if not customer.name:
            customer.name = "Nihai Tüketici"

    # Drop OCR junk lines when totals clearly don't match payable — but first
    # try healing swapped VAT↔extension (generic), don't wipe recoverable rows.
    if ocr_lines and payable:
        line_sum = sum(l.lineTotal or 0.0 for l in ocr_lines if l.lineTotal is not None)
        if line_sum > 0 and (line_sum / payable) < 0.45:
            healed = False
            for l in ocr_lines:
                if (
                    l.unitPrice
                    and l.quantity
                    and l.quantity > 0
                    and l.lineTotal is not None
                ):
                    expected = round(l.unitPrice * l.quantity, 2)
                    if expected >= 1 and abs(l.lineTotal - expected) > 1.0:
                        if l.vatAmount is None or abs(l.vatAmount - expected) > 1.0:
                            if (l.lineTotal / expected) < 0.45:
                                l.vatAmount = l.vatAmount or l.lineTotal
                                l.lineTotal = expected
                                healed = True
            if healed:
                line_sum = sum(l.lineTotal or 0.0 for l in ocr_lines if l.lineTotal is not None)
            if line_sum > 0 and (line_sum / payable) < 0.45:
                ocr_lines = []
                lines_sum = None
            else:
                lines_sum = round(line_sum, 2) if line_sum else lines_sum
        elif line_sum > 0 and payable < line_sum * 0.55:
            # Payable is a stray footnote; prefer ödenecek/line reconciliation
            od = labeled_amount(text, r"[ÖO]DENECEK\s+TUTAR")
            if od is not None and od >= line_sum * 0.85:
                payable = od
                tax_inclusive = od
            elif vat is not None:
                payable = round(line_sum + vat, 2) if not retail_fiş else line_sum
                tax_inclusive = payable
            elif retail_fiş:
                payable = line_sum
                tax_inclusive = line_sum

    # Reject junk supplier names (never replace with address / office lines)
    if supplier.name and (
        _party_name_quality(supplier.name) < 6
        or _is_registry_or_chrome_line(supplier.name)
        or _looks_like_address_party_line(supplier.name)
        or re.search(r"<!--\s*image", supplier.name or "", re.I)
    ):
        better = next(
            (
                re.sub(r"^#+\s*", "", ln.strip()).strip()
                for ln in text.splitlines()
                if _party_name_quality(ln.strip()) >= 20
                and not _is_registry_or_chrome_line(ln)
                and not _looks_like_address_party_line(ln.strip())
                and re.search(
                    r"(?:A\.?\s*Ş\.?|LTD\.?\s*ŞT[İI]|ANON[İI]M|T[İI]CARET|MA[ĞG]AZA|Market)",
                    ln,
                    re.I,
                )
                and not re.search(
                    r"(?:Arac[ıi]|Ta[şs][ıi]y[ıi]c[ıi])\s+Firma|^Not\s*:|Çözüm\s+Merkezi|"
                    r"İnternet\s+Sitesi|Vergi\s+Dairesi|ETTN\s*:|YALNIZ\b|\|",
                    ln,
                    re.I,
                )
            ),
            None,
        )
        if better:
            supplier.name = better[:180]
        elif (
            _party_name_quality(supplier.name) < 4
            or _is_registry_or_chrome_line(supplier.name)
            or _looks_like_address_party_line(supplier.name)
            or re.search(r"<!--\s*image", supplier.name or "", re.I)
        ):
            supplier.name = None

    # If supplier phone empty but a Tel exists only under SAYIN, keep on customer;
    # if supplier Tel empty and customer phone set, also mirror to supplier when
    # head had blank Tel: (common retail GİB OCR — store phone under SAYIN).
    if not supplier.phone and customer.phone:
        head_tel = first_match(
            text[: (re.search(r"\bSAYIN\b", text, re.I).start() if re.search(r"\bSAYIN\b", text, re.I) else 800)],
            r"(?:Tel|Telefon)\s*:?\s*([0-9\s()\-]{10,})",
        )
        if not head_tel:
            supplier.phone = customer.phone

    customization = (
        right_field(text, "Ozellestirme No")
        or right_field(text, "Özelleştirme No")
        or first_match(text, r"Ozellestirme\s*No\s*:?\s*(TR[\d.]+)")
        or first_match(text, r"Özelleştirme\s*No\s*:?\s*(TR[\d.]+)")
    )
    customization = normalize_customization_id(customization)

    if supplier.taxOffice:
        supplier.taxOffice = sanitize_tax_office(supplier.taxOffice)
    if customer.taxOffice:
        customer.taxOffice = sanitize_tax_office(customer.taxOffice)

    inv = Invoice(
        documentType=doc_type,
        profileId=profile,
        customizationId=customization,
        invoiceTypeCode=inv_type,
        invoiceNumber=inv_no,
        uuid=uuid,
        issueDate=issue_date,
        issueTime=issue_time,
        supplier=supplier,
        customer=customer,
        lines=ocr_lines,
        totals=Totals(
            lineExtensionAmount=line_ext,
            discountTotal=discount,
            withholdingVatAmount=withholding,
            vatAmount=vat,
            taxInclusiveAmount=tax_inclusive,
            payableAmount=payable,
            currency="TRY",
        ),
        notes=[],
        iban=iban,
        bankName=(bank.strip() if bank else None) or bank_name_pay,
        bankBranch=None,
    )
    rebalance_party_tax_ids(inv, text)
    return inv


def _looks_like_address_party_line(name: str) -> bool:
    """True when a line is an office/address block, not a legal company title."""
    n = name.strip()
    if re.search(r"<!--\s*image", n, re.I):
        return True
    if re.match(
        r"^(?:Kurumsal\s+Ofis|Merkez(?:\s+Ofis)?|Adres|Ofis)\s*:",
        n,
        re.I,
    ):
        return True
    if re.search(
        r"(?:Kurumsal\s+Ofis|Merkez\s*:)\s*.{0,40}(?:Mahallesi|Cadde|Bulvar|Sokak|No:)",
        n,
        re.I,
    ):
        return True
    # Street/quarter without legal form (e.g. "Kumludere cad 1 TOPÇUASIM/...")
    if re.search(
        r"(?i)\b(?:cad(?:de)?|mah(?:alle)?\.?|sok(?:ak)?|bul(?:var)?|no:|blok)\b",
        n,
    ) and not re.search(
        r"(?i)(?:A\.?\s*Ş\.?|LTD\.?\s*ŞT[İI]|ANON[İI]M|T[İI]CARET|MA[ĞG]AZA|Market)",
        n,
    ):
        return True
    # Long line dominated by address tokens without a clean legal-form title
    addr_hits = len(
        re.findall(
            r"\b(?:Mahallesi|Mah\.|Cadde(?:si)?|Cad\.|cad\b|Bulvar[ıi]?|Sokak|Sk\.|No:|Kat:|Blok)\b",
            n,
            re.I,
        )
    )
    if addr_hits >= 2 and not re.search(
        r"(?:ANON[İI]M\s+Ş[İI]RKET|A\.?\s*Ş\.?|LTD\.?\s*ŞT[İI])\s*$",
        n,
        re.I,
    ):
        return True
    return False


def _party_name_quality(name: str | None) -> int:
    if not name:
        return 0
    # Docling markdown headings: "## COMPANY A.Ş."
    n = re.sub(r"^#+\s*", "", name.strip()).strip()
    if len(n) < 3 or re.fullmatch(r"[=_\-—.\s]+", n):
        return 0
    if re.search(
        r"(?:[İI]mza|PDF\s*indir|XML\s*indir|<!--\s*image|Page\s+\d+|adet\s*[x×]|"
        r"Hgz\s*Ad[iı]|Mgz\s*Kodu|Ma[ğg]aza\s*(?:Ad[iı]|Kodu)|"
        r"^Vergi\s*Daires|^file:|^https?://|TICARETSICIL|MERSISNO|e-?Ar[sş]iv\s+Fatura)",
        n,
        re.I,
    ):
        return 0
    if _looks_like_address_party_line(n):
        return 0
    letters = len(re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü]", n))
    if letters < 4:
        return 0
    score = letters
    if re.search(
        r"(?:LTD|ŞT[İI]|A\.?\s*Ş|SANAY|TICARET|ANON[İI]M|Ş[İI]RKET|MA[ĞG]AZA|DAGITIM|TEKNOLOJ)",
        n,
        re.I,
    ):
        score += 20
        # Registry labels containing TICARET are not company titles
        if re.search(r"TICARETSICIL|T[İI]CARET\s*S[İI]C[İI]L", n, re.I):
            score -= 40
    if re.search(r"Senaryo|Fatura\s*Tipi|==", n, re.I):
        score -= 30
    return score


def scrub_invoice_lines(inv: Invoice) -> None:
    """Drop payment/IBAN/bank/meta rows that should never be product lines."""
    if not inv.lines:
        return
    meta_name = re.compile(
        r"(?i)^(?:Ma[gğ]aza|Kasa(?:\s*No)?|Kasiyer|Sistem\s*No|Çekmece|Saat|Tarih|"
        r"Fatura\s*Tipi|Online\s*Sipari[sş]|Club\s*Kart)\s*:?\s*\S{0,20}$"
    )
    inv.lines = [
        ln
        for ln in inv.lines
        if ln.name
        and not _is_bank_or_iban_line(ln.name)
        and not meta_name.search(ln.name.strip())
        and not re.search(
            r"(?i)Kredi\s*Kart|Banka\s*Kart|\bIBAN\b|\bİBAN\b|"
            r"Net\s*Mal\s*De[gğ]eri|Hesap\s*Ad|[ŞS]ube\s*(?:Kod|Ad)|Swift|\bBIC\b",
            ln.name or "",
        )
    ]


def _lines_useful(lines: list[Line] | list[dict] | None) -> bool:
    if not lines:
        return False
    named = 0
    totaled = 0
    for l in lines:
        if isinstance(l, dict):
            name, total = l.get("name"), l.get("lineTotal")
        else:
            name, total = l.name, l.lineTotal
        if name and len(re.sub(r"\s+", "", str(name))) >= 3 and not re.match(
            r"^(?:adet|kalem|ad)\b", str(name), re.I
        ):
            named += 1
        if total is not None and total > 0:
            totaled += 1
    return totaled > 0 and named >= max(1, totaled // 2)


def _adopt_vl_invoice(vl: Invoice, prior: Invoice) -> Invoice:
    """After VL escalate: VL wins; prior only fills gaps. Keep distinct tax IDs from prior."""
    merged = merge_invoice(vl, prior)
    # If VL mirrored the same tax onto both parties, restore a distinct prior supplier tax
    if (
        merged.supplier.taxId
        and merged.customer.taxId
        and merged.supplier.taxId == merged.customer.taxId
        and prior.supplier.taxId
        and prior.supplier.taxId != merged.supplier.taxId
    ):
        from tax_id import is_valid_tax_id

        scheme = prior.supplier.taxIdScheme or (
            "TCKN" if len(digits_only(prior.supplier.taxId) or "") == 11 else "VKN"
        )
        if is_valid_tax_id(prior.supplier.taxId, scheme):
            merged.supplier.taxId = prior.supplier.taxId
            merged.supplier.taxIdScheme = scheme  # type: ignore[assignment]
            if prior.supplier.taxOffice:
                merged.supplier.taxOffice = prior.supplier.taxOffice
    if (
        merged.customer.taxId
        and merged.supplier.taxId
        and merged.customer.taxId == merged.supplier.taxId
        and prior.customer.taxId
        and prior.customer.taxId != merged.supplier.taxId
    ):
        from tax_id import is_valid_tax_id

        scheme = prior.customer.taxIdScheme or (
            "TCKN" if len(digits_only(prior.customer.taxId) or "") == 11 else "VKN"
        )
        if is_valid_tax_id(prior.customer.taxId, scheme):
            merged.customer.taxId = prior.customer.taxId
            merged.customer.taxIdScheme = scheme  # type: ignore[assignment]
    if _lines_useful(vl.lines):
        merged.lines = vl.lines
    return merged


def merge_invoice(base: Invoice, overlay: Invoice) -> Invoice:
    data = base.model_dump()
    over = overlay.model_dump()
    for k, v in over.items():
        if k in {"supplier", "customer", "totals", "lines", "notes"}:
            continue
        if not v:
            continue
        cur = data.get(k)
        # Treat unknown documentType as empty so OCR/tesseract can fill earsiv/efatura
        if k == "documentType" and cur in (None, "", "unknown"):
            data[k] = v
        elif v and not cur:
            data[k] = v
    for side in ("supplier", "customer"):
        for k, v in over[side].items():
            if not v:
                continue
            cur = data[side].get(k)
            if not cur:
                data[side][k] = v
                continue
            if k != "name":
                continue
            # Never replace a real person/company with generic retail fallback
            if re.search(r"Nihai\s+T[uü]ketici", str(v), re.I) and not re.search(
                r"Nihai\s+T[uü]ketici", str(cur), re.I
            ):
                continue
            q_new, q_old = _party_name_quality(str(v)), _party_name_quality(str(cur))
            if q_new > q_old + 3:
                data[side][k] = v
            elif (
                q_new >= max(8, q_old - 5)
                and len(str(v)) < len(str(cur))
                and "Senaryo" not in str(v)
                and "Fatura" not in str(v)
                and not re.search(r"^[#=\-]", str(v))
                and not re.search(r"\b[a-z]{3,}\b", str(v))  # avoid OCR-lowercase junk replacing clean names
            ):
                # Prefer cleaner shorter party name when quality is comparable
                data[side][k] = v
            elif q_old < 8 and q_new >= 8:
                data[side][k] = v
    for k, v in over["totals"].items():
        if v is None:
            continue
        cur = data["totals"].get(k)
        if cur is None:
            data["totals"][k] = v
            continue
        # Prefer overlay when base looks like a stray unit/footnote amount
        # (RapidOCR often binds "23,00" while VL has the real ödenecek).
        if (
            k in ("payableAmount", "taxInclusiveAmount", "lineExtensionAmount", "vatAmount")
            and isinstance(cur, (int, float))
            and isinstance(v, (int, float))
            and float(cur) < float(v) * 0.5
            and float(v) >= 50
        ):
            data["totals"][k] = v
    if over["lines"]:
        def _line_sum(lines: list) -> float:
            return sum((l.get("lineTotal") or 0) for l in lines if isinstance(l, dict))

        pay = data["totals"].get("payableAmount") or over["totals"].get("payableAmount")
        over_sum = _line_sum(over["lines"])
        over_ok = bool(pay and over_sum and abs(over_sum - pay) / pay < 0.45)
        if not data["lines"]:
            if over_ok or not pay:
                data["lines"] = over["lines"]
        else:
            base_sum = _line_sum(data["lines"])
            base_ok = bool(pay and base_sum and abs(base_sum - pay) / pay < 0.45)
            if over_ok and not base_ok:
                data["lines"] = over["lines"]
            elif over_ok and base_ok and len(over["lines"]) >= len(data["lines"]):
                data["lines"] = over["lines"]
            # else keep base (do not import junk OCR lines)
    if over["notes"] and not data["notes"]:
        data["notes"] = over["notes"]
    return Invoice.model_validate(data)


def _sanitize_party_tax_id(party: Party, role: str) -> list[str]:
    """Clear checksum-invalid tax IDs so false positives are not reported as success."""
    warnings: list[str] = []
    raw = normalize_ocr_digits(party.taxId) or digits_only(party.taxId)
    if not raw:
        party.taxId = None
        party.taxIdScheme = None
        return warnings

    scheme = party.taxIdScheme
    if scheme is None:
        if len(raw) == 11:
            scheme = "TCKN"
        elif len(raw) == 10:
            scheme = "VKN"

    if is_valid_tax_id(raw, scheme):
        party.taxId = raw
        party.taxIdScheme = "TCKN" if len(raw) == 11 else "VKN"
        return warnings

    repaired = repair_tax_id(raw, scheme)
    if repaired:
        party.taxId, party.taxIdScheme = repaired
        return warnings

    label = scheme or ("TCKN" if len(raw) == 11 else "VKN" if len(raw) == 10 else "vergi kimlik")
    warnings.append(f"{role} {label} geçersiz (doğrulama başarısız) — yok sayıldı")
    party.taxId = None
    party.taxIdScheme = None
    return warnings


def validate_invoice(inv: Invoice) -> tuple[list[str], Validation]:
    warnings: list[str] = []
    checks: list[str] = []
    score = 1.0

    def need(cond: bool, msg: str, weight: float = 0.12) -> None:
        nonlocal score
        if cond:
            checks.append(f"ok:{msg}")
        else:
            warnings.append(msg)
            checks.append(f"fail:{msg}")
            score -= weight

    # Safety net when binder/merge skipped parse_text_invoice rebalance
    rebalance_party_tax_ids(inv)
    supplier_tax_warnings = _sanitize_party_tax_id(inv.supplier, "Satıcı")
    customer_tax_warnings = _sanitize_party_tax_id(inv.customer, "Alıcı")
    for w in supplier_tax_warnings + customer_tax_warnings:
        warnings.append(w)
        checks.append(f"fail:{w}")
        score -= 0.1

    need(bool(inv.invoiceNumber), "Fatura numarası bulunamadı", 0.16)
    need(bool(inv.uuid), "ETTN bulunamadı", 0.04)
    need(bool(inv.issueDate), "Fatura tarihi bulunamadı", 0.08)
    need(bool(inv.supplier.name), "Satıcı unvanı bulunamadı", 0.1)
    need(bool(inv.customer.name), "Alıcı unvanı bulunamadı", 0.04)
    if not supplier_tax_warnings:
        need(bool(inv.supplier.taxId), "Satıcı VKN/TCKN bulunamadı", 0.1)
    if not customer_tax_warnings:
        need(bool(inv.customer.taxId), "Alıcı VKN/TCKN bulunamadı", 0.04)
    need(inv.totals.payableAmount is not None, "Ödenecek tutar bulunamadı", 0.16)
    need(len(inv.lines) > 0, "Mal/hizmet kalemi bulunamadı", 0.14)

    totals_match = True
    le, vat, ti, pay, wh = (
        inv.totals.lineExtensionAmount,
        inv.totals.vatAmount,
        inv.totals.taxInclusiveAmount,
        inv.totals.payableAmount,
        inv.totals.withholdingVatAmount,
    )
    if le is not None and vat is not None and ti is not None:
        if not nearly_equal(le + vat, ti):
            totals_match = False
            warnings.append("Matrah + KDV, vergiler dahil toplam ile uyuşmuyor")
            score -= 0.1
        else:
            checks.append("ok:matrah+kdv")
    if ti is not None and pay is not None and wh is not None:
        if not nearly_equal(ti - wh, pay):
            totals_match = False
            warnings.append("Ödenecek tutar, tevkifat düşülmüş toplam ile uyuşmuyor")
            score -= 0.08
    elif ti is not None and pay is not None and wh is None:
        if not nearly_equal(ti, pay):
            # often equal when no withholding
            if abs(ti - pay) > 0.05:
                totals_match = False
                score -= 0.05
        else:
            checks.append("ok:payable==taxInclusive")

    # Line sum check
    if inv.lines:
        summed = sum(l.lineTotal or 0 for l in inv.lines if l.lineTotal is not None)
        if le is not None and summed > 0 and not nearly_equal(summed, le, 1.0):
            # soft — discounts may explain
            checks.append(f"info:lineSum={summed:.2f} vs extension={le:.2f}")

    score = max(0.0, min(1.0, score))
    return warnings, Validation(totalsMatch=totals_match and len(warnings) == 0, confidence=round(score, 3), checks=checks)


def get_docling_converter(ocr: bool, for_image: bool = False):
    global _docling_converter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption

    # Cache non-OCR PDF converter (common path)
    if not ocr and not for_image and _docling_converter is not None:
        return _docling_converter

    options = PdfPipelineOptions()
    options.do_ocr = ocr
    # Table structure is expensive; for photos RapidOCR already supplies lines.
    options.do_table_structure = not (for_image and ocr)
    if for_image and ocr:
        # Higher scale improves phone-photo OCR quality
        try:
            options.images_scale = IMAGE_OCR_SCALE
        except Exception:
            pass
        # Prefer RapidOCR (same stack as photo OCR) for image documents; Tesseract last
        ocr_opts = None
        try:
            from docling.datamodel.pipeline_options import RapidOcrOptions

            ocr_opts = RapidOcrOptions()
        except Exception:
            ocr_opts = None
        if ocr_opts is None:
            try:
                from docling.datamodel.pipeline_options import TesseractCliOcrOptions

                ocr_opts = TesseractCliOcrOptions(lang=["tur", "eng"])
            except Exception:
                ocr_opts = None
        if ocr_opts is None:
            try:
                from docling.datamodel.pipeline_options import EasyOcrOptions

                ocr_opts = EasyOcrOptions(lang=["tr", "en"])
            except Exception:
                ocr_opts = None
        if ocr_opts is not None:
            options.ocr_options = ocr_opts

    format_options: dict[Any, Any] = {
        InputFormat.PDF: PdfFormatOption(pipeline_options=options),
    }
    if for_image:
        format_options[InputFormat.IMAGE] = ImageFormatOption(pipeline_options=options)

    converter = DocumentConverter(format_options=format_options)
    if not ocr and not for_image:
        _docling_converter = converter
    return converter


def docling_convert(path: Path, ocr: bool = False, for_image: bool = False) -> tuple[str, list[Line]]:
    converter = get_docling_converter(ocr=ocr, for_image=for_image)
    result = converter.convert(str(path))
    md = result.document.export_to_markdown() or ""
    table_lines = parse_markdown_tables(md)
    return md, table_lines


@app.on_event("startup")
def warmup() -> None:
    if ENABLE_DOCLING:
        try:
            get_docling_converter(ocr=False, for_image=False)
        except Exception as exc:  # noqa: BLE001
            print(f"docling warmup skipped: {exc}")
    if VL_OCR_ENABLED and VL_OCR_WARMUP:
        try:
            from vl_ocr import warmup as vl_warmup

            print(f"VL OCR warmup: {vl_warmup()}")
        except Exception as exc:  # noqa: BLE001
            print(f"VL OCR warmup skipped: {exc}")
    if PHOTO_OCR_ENABLED and PHOTO_OCR_WARMUP:
        try:
            from photo_ocr import warmup_engines

            status = warmup_engines(include_medium=PHOTO_OCR_WARMUP_MEDIUM)
            print(f"photo OCR warmup: {status}")
        except Exception as exc:  # noqa: BLE001
            print(f"photo OCR warmup skipped: {exc}")


@app.get("/health")
def health() -> dict[str, Any]:
    photo_status: dict[str, Any] = {"enabled": PHOTO_OCR_ENABLED}
    if PHOTO_OCR_ENABLED:
        try:
            from photo_ocr import engine_status

            photo_status = engine_status()
        except Exception as exc:  # noqa: BLE001
            photo_status = {"enabled": True, "error": str(exc)}
    vl_status: dict[str, Any] = {"enabled": VL_OCR_ENABLED}
    if VL_OCR_ENABLED:
        try:
            from vl_ocr import engine_status as vl_engine_status

            vl_status = vl_engine_status()
        except Exception as exc:  # noqa: BLE001
            vl_status = {"enabled": True, "error": str(exc)}
    return {
        "ok": True,
        "service": "fatura-ai-extract",
        "docling": ENABLE_DOCLING,
        "doclingOcr": ENABLE_DOCLING_OCR,
        "forceImageOcr": FORCE_IMAGE_OCR,
        "photoOcr": PHOTO_OCR_ENABLED,
        "photoOcrStatus": photo_status,
        "vlOcr": VL_OCR_ENABLED,
        "vlOcrStatus": vl_status,
        "photoOcrMaxInflight": PHOTO_OCR_MAX_INFLIGHT,
        "photoOcrTimeoutS": PHOTO_OCR_TIMEOUT_S,
        "vlOcrTimeoutS": VL_OCR_TIMEOUT_S,
        "fastPathPdf": FAST_PATH_PDF,
        "pdfInspector": PDF_INSPECTOR_ENABLED,
        "pdfRasterDpi": PDF_RASTER_DPI,
        "doclingMaxInflight": DOCLING_MAX_INFLIGHT,
        "doclingTimeoutS": DOCLING_TIMEOUT_S,
        "inflight": _metrics["inflight"],
        "photoOcrInflight": _metrics["photo_ocr_inflight"],
        "metrics": dict(_metrics),
        "imageFormats": sorted(ext.lstrip(".") for ext in IMAGE_EXTENSIONS),
    }


@app.get("/metrics")
def metrics_prom() -> Any:
    from fastapi.responses import PlainTextResponse

    lines = [
        f"fatura_extract_total {_metrics['extract_total']}",
        f"fatura_extract_ok {_metrics['extract_ok']}",
        f"fatura_extract_partial {_metrics['extract_partial']}",
        f"fatura_extract_failed {_metrics['extract_failed']}",
        f"fatura_extract_fast_path {_metrics['fast_path']}",
        f"fatura_extract_docling_calls {_metrics['docling_calls']}",
        f"fatura_extract_inflight {_metrics['inflight']}",
    ]
    return PlainTextResponse("\n".join(lines) + "\n")


@app.post("/extract", response_model=ExtractResponse)
async def extract(
    file: UploadFile = File(...),
    filename: str | None = Form(None),
) -> ExtractResponse:
    started = time.perf_counter()
    pipeline: list[str] = []
    name = filename or file.filename or "invoice.pdf"
    data = await file.read()
    _metrics["extract_total"] += 1
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        _metrics["extract_failed"] += 1
        return ExtractResponse(
            status="failed",
            method="none",
            durationMs=int((time.perf_counter() - started) * 1000),
            warnings=[f"Dosya {MAX_UPLOAD_MB}MB limitini aşıyor"],
            pipeline=pipeline,
        )

    ext = sniff_extension(data, name)
    as_image = is_image_ext(ext)
    if not as_image and ext == ".pdf" and not data.startswith(b"%PDF"):
        # Misnamed / corrupt upload (common portal fail)
        _metrics["extract_failed"] += 1
        return ExtractResponse(
            status="failed",
            method="none",
            durationMs=int((time.perf_counter() - started) * 1000),
            warnings=[
                "Dosya geçerli bir PDF değil (bozuk, şifreli veya yanlış uzantı). "
                "Orijinal e-fatura PDF/XML veya net fotoğraf yükleyin."
            ],
            pipeline=pipeline + ["invalid-pdf-magic"],
        )

    with tempfile.TemporaryDirectory(prefix="fatura-ai-") as tmp:
        path = Path(tmp) / f"invoice{ext if ext != '.bin' else ('.jpg' if as_image else '.pdf')}"
        path.write_bytes(data)

        if as_image and path.suffix.lower() in {".heic", ".heif"}:
            jpeg_path = Path(tmp) / "invoice.jpg"
            try:
                await asyncio.to_thread(convert_heic_to_jpeg, path, jpeg_path)
                path = jpeg_path
                pipeline.append("heic-jpeg")
            except Exception as exc:  # noqa: BLE001
                _metrics["extract_failed"] += 1
                return ExtractResponse(
                    status="failed",
                    method="none",
                    durationMs=int((time.perf_counter() - started) * 1000),
                    warnings=[str(exc)],
                    pipeline=pipeline + [f"heic-error:{exc}"],
                )

        invoice = Invoice()
        text = ""
        md = ""

        if as_image:
            pipeline.append("image-input")
            photo_text = ""
            photo_meta: dict[str, Any] = {}
            if PHOTO_OCR_ENABLED:
                try:
                    photo_text, photo_meta = await run_photo_ocr(path, prefer_vl=False)
                    pipeline.append(
                        f"photo-ocr:{photo_meta.get('engine', '?')}:"
                        f"{photo_meta.get('elapsedMs', 0)}ms:"
                        f"{photo_meta.get('lineCount', 0)}"
                    )
                    _metrics["photo_ocr"] += 1
                    if photo_text.strip():
                        text = photo_text
                        md = photo_text
                        if is_probable_non_invoice_text(photo_text):
                            pipeline.append("non-invoice-reject")
                            _metrics["extract_failed"] += 1
                            return ExtractResponse(
                                status="failed",
                                method="non-invoice",
                                durationMs=int((time.perf_counter() - started) * 1000),
                                warnings=[
                                    "Belge e-fatura/e-arşiv olarak tanınmadı "
                                    "(logo/reklam veya yetersiz metin)."
                                ],
                                invoice=Invoice(documentType="unknown"),
                                rawTextPreview=photo_text[:2500],
                                pipeline=pipeline,
                            )
                        from parse_vl_markdown import invoice_from_ocr_text

                        inv_photo = invoice_from_ocr_text(
                            photo_text, name, engine=str(photo_meta.get("engine") or "")
                        )
                        invoice = merge_invoice(invoice, inv_photo)
                        if not invoice.lines and inv_photo.lines:
                            invoice.lines = inv_photo.lines
                        rebalance_party_tax_ids(invoice, photo_text)
                        warnings_ph, validation_ph = validate_invoice(invoice)
                        warnings, validation = warnings_ph, validation_ph
                        pipeline.append("ocr-field-binder")
                        if is_garbage_photo_ocr(invoice, photo_meta):
                            pipeline.append("garbage-ocr-reject")
                            _metrics["extract_failed"] += 1
                            return ExtractResponse(
                                status="failed",
                                method="garbage-ocr",
                                durationMs=int((time.perf_counter() - started) * 1000),
                                warnings=[
                                    "OCR yapısal fatura sinyali üretmedi "
                                    "(structureScore≤1, kritik alan yok) — "
                                    "Docling/VL atlandı."
                                ],
                                invoice=Invoice(documentType="unknown"),
                                rawTextPreview=photo_text[:2500],
                                pipeline=pipeline,
                            )
                        if strong_photo_invoice(invoice, validation_ph):
                            pipeline.append("photo-ocr-fast-path")

                    if (
                        VL_OCR_ENABLED
                        and needs_ocr_escalation(invoice, validation, warnings)
                    ):
                        pipeline.append("vl-escalate")
                        vl_text, vl_meta = await run_photo_ocr(path, prefer_vl=True)
                        if vl_text.strip():
                            pipeline.append(
                                f"photo-vl:{vl_meta.get('engine', '?')}:"
                                f"{vl_meta.get('elapsedMs', 0)}ms"
                            )
                            from parse_vl_markdown import invoice_from_ocr_text

                            inv_vl = invoice_from_ocr_text(
                                vl_text, name, engine=str(vl_meta.get("engine") or "")
                            )
                            pipeline.append("vl-field-binder")
                            invoice = _adopt_vl_invoice(inv_vl, invoice)
                            text = vl_text
                            md = vl_text
                            warnings, validation = validate_invoice(invoice)
                except Exception as exc:  # noqa: BLE001
                    pipeline.append(f"photo-ocr-error:{exc}")

            # Heavy Docling only if photo OCR missed critical fields
            need_docling = not strong_photo_invoice(
                invoice, validate_invoice(invoice)[1]
            )
            photo_struct = int(photo_meta.get("structureScore") or 0)
            if is_garbage_photo_ocr(invoice, photo_meta):
                need_docling = False
                pipeline.append("skip-docling:garbage-ocr")
            if ENABLE_DOCLING and need_docling:
                try:
                    use_ocr = FORCE_IMAGE_OCR or ENABLE_DOCLING_OCR
                    md2, table_lines = await run_docling(path, ocr=use_ocr, for_image=True)
                    pipeline.append("docling-image-ocr" if use_ocr else "docling-image")
                    if table_lines and _lines_useful(table_lines):
                        if not _lines_useful(invoice.lines):
                            invoice.lines = table_lines
                            pipeline.append(f"docling-tables:{len(table_lines)}")
                        else:
                            # Keep photo lines unless Docling has more complete named rows
                            photo_n = len([l for l in invoice.lines if l.name and l.lineTotal])
                            doc_n = len([l for l in table_lines if l.name and l.lineTotal])
                            if doc_n > photo_n:
                                invoice.lines = table_lines
                                pipeline.append(f"docling-tables:{len(table_lines)}")
                    elif table_lines:
                        pipeline.append(f"docling-tables-skipped:{len(table_lines)}")
                    if md2.strip():
                        inv_md = parse_text_invoice(md2.replace("\t", " "), name)
                        # Avoid Docling markdown party junk (<!-- image -->) clobbering photo OCR
                        if inv_md.supplier and _party_name_quality(inv_md.supplier.name) < 8:
                            inv_md.supplier.name = None
                        if inv_md.customer and _party_name_quality(inv_md.customer.name) < 8:
                            inv_md.customer.name = None
                        # Table-cell soup (pipes / Senaryo glued into name)
                        for party in (inv_md.supplier, inv_md.customer):
                            if party and party.name and (
                                "|" in party.name
                                or re.search(r"Senaryo|Fatura\s*No|Sipari[sş]", party.name, re.I)
                            ):
                                party.name = None
                        # Keep photo date when Docling OCR year is clearly worse
                        if invoice.issueDate and inv_md.issueDate and invoice.invoiceNumber:
                            ym = re.match(r"[A-Z]{2,5}(\d{4})", invoice.invoiceNumber)
                            if ym:
                                sy = int(ym.group(1))
                                try:
                                    py, dy = int(invoice.issueDate[:4]), int(inv_md.issueDate[:4])
                                except ValueError:
                                    py = dy = 0
                                if py == sy and dy != sy:
                                    inv_md.issueDate = None
                        invoice = merge_invoice(invoice, inv_md)
                        if not _lines_useful(invoice.lines) and _lines_useful(inv_md.lines):
                            invoice.lines = inv_md.lines
                        md = (md + "\n\n" + md2).strip() if md else md2
                        text = (text + "\n\n" + md2).strip() if text else md2
                        rebalance_party_tax_ids(invoice, text)
                except Exception as exc:  # noqa: BLE001
                    pipeline.append(f"docling-image-error:{exc}")

            # Skip redundant Tesseract when photo OCR already saw invoice structure
            if (need_docling or not invoice.invoiceNumber) and photo_struct < 6:
                try:
                    tess = await asyncio.to_thread(tesseract_ocr, path)
                    if tess.strip():
                        pipeline.append("tesseract")
                        text = (text + "\n\n" + tess).strip() if text else tess
                        inv_tess = parse_text_invoice(tess, name)
                        invoice = merge_invoice(invoice, inv_tess)
                        if not md.strip():
                            md = tess
                        else:
                            md = md + "\n\n" + tess
                except Exception as exc:  # noqa: BLE001
                    pipeline.append(f"tesseract-error:{exc}")

            if (
                not invoice.invoiceNumber
                and not invoice.totals.payableAmount
                and not invoice.lines
                and not invoice.supplier.name
            ):
                _metrics["extract_failed"] += 1
                return ExtractResponse(
                    status="failed",
                    method="none",
                    durationMs=int((time.perf_counter() - started) * 1000),
                    warnings=["Fotoğraf okunamadı — daha net çekim veya PDF deneyin"],
                    pipeline=pipeline,
                )
        else:
            ubl = extract_embedded_ubl(data)
            if ubl and re.search(r"<(?:\w+:)?Invoice[\s>]", ubl, re.I):
                pipeline.append("ubl")

            texts, text_tags = await asyncio.to_thread(extract_pdf_texts, path)
            pipeline.extend(text_tags)
            text = "\n\n".join(texts)

            invoice = Invoice()
            for blob in texts:
                inv_part = parse_text_invoice(blob, name)
                invoice = merge_invoice(invoice, inv_part)
                if not invoice.lines and inv_part.lines:
                    invoice.lines = inv_part.lines
            if not texts:
                invoice = Invoice()
            warnings_fp, validation_fp = validate_invoice(invoice)

            if FAST_PATH_PDF and strong_text_invoice(invoice, validation_fp):
                pipeline.append("fast-path")
                _metrics["fast_path"] += 1
            elif ENABLE_DOCLING:
                try:
                    md, table_lines = await run_docling(path, ocr=False, for_image=False)
                    pipeline.append("docling-structure")
                    if table_lines:
                        invoice.lines = table_lines
                        pipeline.append(f"docling-tables:{len(table_lines)}")
                    if md.strip():
                        inv_md = parse_text_invoice(md.replace("\t", " "), name)
                        invoice = merge_invoice(invoice, inv_md)
                        if not invoice.lines and inv_md.lines:
                            invoice.lines = inv_md.lines
                except Exception as exc:  # noqa: BLE001
                    pipeline.append(f"docling-error:{exc}")

        warnings, validation = validate_invoice(invoice)

        # Final GİB-year ↔ issueDate alignment (after merges)
        if invoice.invoiceNumber and invoice.issueDate:
            ym = re.match(r"^[A-Z]{2,5}(\d{4})", invoice.invoiceNumber)
            if ym:
                series_year = int(ym.group(1))
                try:
                    date_year = int(invoice.issueDate[:4])
                except ValueError:
                    date_year = 0
                if 1990 <= series_year <= 2100 and date_year != series_year:
                    sa, sb = f"{date_year:04d}", f"{series_year:04d}"
                    diffs = sum(a != b for a, b in zip(sa, sb))
                    if diffs <= 1 and abs(date_year - series_year) <= 100:
                        invoice.issueDate = f"{series_year:04d}{invoice.issueDate[4:]}"

        for line in invoice.lines:
            # Recover qty when OCR missed "N Adet" but unit×qty ≈ line total
            if (
                line.lineTotal is not None
                and line.lineTotal >= 10
                and line.unitPrice
                and line.unitPrice > 1
                and (not line.quantity or line.quantity == 1)
            ):
                ratio = line.lineTotal / line.unitPrice
                r = round(ratio)
                if r >= 2 and abs(ratio - r) <= 0.025:
                    line.quantity = float(r)
                    line.unit = line.unit or "Adet"
            if (
                line.quantity
                and line.lineTotal is not None
                and line.quantity > 0
                and (line.unitPrice is None or line.unitPrice <= max(1.0, line.quantity))
            ):
                if line.discountAmount:
                    line.unitPrice = round(line.lineTotal + line.discountAmount, 2)
                else:
                    line.unitPrice = round(line.lineTotal / line.quantity, 2)

        need_ocr = (
            not as_image
            and "fast-path" not in pipeline
            and ENABLE_DOCLING
            and ENABLE_DOCLING_OCR
            and (
                not invoice.lines
                or validation.confidence < 0.7
                or any("kalemi" in w for w in warnings)
            )
        )
        # Scanned / broken-text: RapidOCR first, escalate to VL only if fields weak.
        force_raster = (
            not as_image
            and "fast-path" not in pipeline
            and (PHOTO_OCR_ENABLED or VL_OCR_ENABLED)
            and (
                is_unusable_extract_text(md or text)
                or (
                    not invoice.invoiceNumber
                    and (
                        invoice.totals.payableAmount is None
                        or validation.confidence < 0.55
                        or status_from(warnings, validation) == "failed"
                    )
                )
                or (
                    bool(invoice.supplier.name)
                    and sum(1 for c in (invoice.supplier.name or "") if ord(c) < 32) >= 2
                )
                or needs_ocr_escalation(invoice, validation, warnings)
            )
        )
        if force_raster:
            try:
                raster_txt, raster_meta = await run_pdf_raster_ocr(
                    path, Path(tmp), 2, prefer_vl=False
                )
                if raster_txt.strip():
                    pipeline.append(
                        f"pdf-raster-ocr:{raster_meta.get('engine', '?')}:"
                        f"{raster_meta.get('elapsedMs', 0)}ms:"
                        f"p{raster_meta.get('pages', 0)}"
                    )
                    _metrics["photo_ocr"] += 1
                    from parse_vl_markdown import invoice_from_ocr_text

                    inv_r = invoice_from_ocr_text(
                        raster_txt, name, engine=str(raster_meta.get("engine") or "")
                    )
                    pipeline.append("ocr-field-binder")
                    scrub_invoice_lines(inv_r)
                    if invoice.supplier and invoice.supplier.name:
                        if sum(1 for c in invoice.supplier.name if ord(c) < 32) >= 2:
                            invoice.supplier.name = None
                    scrub_invoice_lines(invoice)
                    invoice = merge_invoice(invoice, inv_r)
                    scrub_invoice_lines(invoice)
                    if not _lines_useful(invoice.lines) and _lines_useful(inv_r.lines):
                        invoice.lines = inv_r.lines
                    # Prefer binder product rows over header/meta false positives
                    elif _lines_useful(inv_r.lines) and invoice.lines:
                        base_meta = all(
                            re.search(
                                r"(?i)^(?:Ma[gğ]aza|Kasa|Kasiyer|Sistem|Kredi\s*Kart)",
                                (ln.name or ""),
                            )
                            for ln in invoice.lines
                        )
                        if base_meta:
                            invoice.lines = inv_r.lines
                    text = (text + "\n\n" + raster_txt).strip() if text else raster_txt
                    md = (md + "\n\n" + raster_txt).strip() if md else raster_txt
                    warnings, validation = validate_invoice(invoice)

                # VL last resort — only when RapidOCR/parse still fails generic gate
                if (
                    VL_OCR_ENABLED
                    and needs_ocr_escalation(invoice, validation, warnings)
                ):
                    pipeline.append("vl-escalate")
                    vl_txt, vl_meta = await run_pdf_raster_ocr(
                        path, Path(tmp), 2, prefer_vl=True
                    )
                    if vl_txt.strip():
                        pipeline.append(
                            f"pdf-raster-vl:{vl_meta.get('engine', '?')}:"
                            f"{vl_meta.get('elapsedMs', 0)}ms:"
                            f"p{vl_meta.get('pages', 0)}"
                        )
                        from parse_vl_markdown import invoice_from_ocr_text

                        inv_vl = invoice_from_ocr_text(
                            vl_txt, name, engine=str(vl_meta.get("engine") or "")
                        )
                        pipeline.append("vl-field-binder")
                        invoice = _adopt_vl_invoice(inv_vl, invoice)
                        text = (text + "\n\n" + vl_txt).strip() if text else vl_txt
                        md = (md + "\n\n" + vl_txt).strip() if md else vl_txt
                        warnings, validation = validate_invoice(invoice)
            except Exception as exc:  # noqa: BLE001
                pipeline.append(f"pdf-raster-ocr-error:{exc}")

        still_weak = (
            not invoice.invoiceNumber
            or validation.confidence < 0.55
            or is_unusable_extract_text(md or text)
        )
        if need_ocr or (force_raster and still_weak and ENABLE_DOCLING):
            try:
                md2, table_lines2 = await run_docling(path, ocr=True, for_image=False)
                pipeline.append("docling-ocr")
                if table_lines2:
                    invoice.lines = table_lines2
                if md2.strip():
                    invoice = merge_invoice(invoice, parse_text_invoice(md2.replace("\t", " "), name))
                warnings, validation = validate_invoice(invoice)
            except Exception as exc:  # noqa: BLE001
                pipeline.append(f"docling-ocr-error:{exc}")

        scrub_invoice_lines(invoice)
        warnings, validation = validate_invoice(invoice)

        method = "+".join(
            p
            for p in pipeline
            if not p.endswith("-error")
            and not p.startswith("docling-error")
            and not p.startswith("docling-image-error")
        ) or "none"
        if "ubl" in pipeline:
            method = "ubl+" + method if method != "none" else "ubl"

        preview = (md or text)[:2500] if (md or text) else None
        status = status_from(warnings, validation)
        # Empty → failed. Any bound field (no/payable/lines/VKN/ETTN/date) → ≥ partial
        if not has_any_invoice_field(invoice):
            status = "failed"
        elif status == "failed":
            status = "partial"

        if status == "ok":
            _metrics["extract_ok"] += 1
        elif status == "partial":
            _metrics["extract_partial"] += 1
        else:
            _metrics["extract_failed"] += 1

        return ExtractResponse(
            status=status,
            method=method,
            durationMs=int((time.perf_counter() - started) * 1000),
            warnings=warnings,
            invoice=invoice,
            rawTextPreview=preview,
            validation=validation,
            pipeline=pipeline,
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=PORT, reload=False)
