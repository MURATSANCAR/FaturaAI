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
# Photo path: RapidOCR PP-OCRv6 (ONNX) before Docling/Tesseract.
PHOTO_OCR_ENABLED = os.getenv("PHOTO_OCR_ENABLED", "1") == "1"
PHOTO_OCR_MIN_CONF = float(os.getenv("PHOTO_OCR_MIN_CONF", "0.55"))
DOCLING_MAX_INFLIGHT = max(1, int(os.getenv("DOCLING_MAX_INFLIGHT", "1")))
DOCLING_TIMEOUT_S = int(os.getenv("DOCLING_TIMEOUT_S", "120"))
IMAGE_OCR_SCALE = float(os.getenv("IMAGE_OCR_SCALE", "3.0"))

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
_metrics = {
    "extract_total": 0,
    "extract_ok": 0,
    "extract_partial": 0,
    "extract_failed": 0,
    "fast_path": 0,
    "photo_ocr": 0,
    "docling_calls": 0,
    "inflight": 0,
}


def get_docling_sem() -> asyncio.Semaphore:
    global _docling_sem
    if _docling_sem is None:
        _docling_sem = asyncio.Semaphore(DOCLING_MAX_INFLIGHT)
    return _docling_sem


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
    r"(?:\d{1,3}(?:[.,\s]\d{3})+[.,]\d{1,4}"  # 12.000,00 / 12,000.00 / 1.453,7
    r"|\d{1,3}(?:[.\s]\d{3})*[.,]\d{1,4}"
    r"|\d+[LlIiOo]?[.,]\d{1,4}"
    r"|\d{1,6}\s\d{2})"
)


def normalize_ocr_text(text: str) -> str:
    """Generic OCR label/typo normalization for Turkish e-invoice layouts."""
    if not text:
        return text
    # Strip portal / viewer chrome
    text = re.sub(r"(?im)^\s*(?:PDF|XML)\s*indir\s*", "", text)
    text = re.sub(r"(?i)\bPDF\s*indir\b|\bXML\s*indir\b", " ", text)
    replacements = (
        (r"\bSAYDN\b", "SAYIN"),
        (r"\bSAVIN\b", "SAYIN"),
        (r"\bSAY[İI]N\b", "SAYIN"),
        (r"\bETIN\b", "ETTN"),
        (r"\bETTN\b", "ETTN"),
        (r"\bFatera\b", "Fatura"),
        (r"\bFatara\b", "Fatura"),
        (r"\bYarihi\b", "Tarihi"),
        (r"\bTanible\b", "Tarihi"),
        (r"\b[ÖO]DENECEKTUTAR\b", "ÖDENECEK TUTAR"),
        (r"\bOdenecek\s*Tutar\b", "ÖDENECEK TUTAR"),
        (r"\benecel\s*Tutar\b", "ÖDENECEK TUTAR"),
        (r"\bVergies?\s*Dald\s*Teglam\s*Tutar\b", "Vergiler Dahil Toplam Tutar"),
        (r"\bVergiler\s*Dahil\s*Toplam\s*Tutar\b", "Vergiler Dahil Toplam Tutar"),
        (r"\bBesaplaesnKOV\b", "Hesaplanan KDV"),
        (r"\bHesaplanan\s*K\.?\s*D\.?\s*V\.?\b", "Hesaplanan KDV"),
        (r"\bAlica\b", "Alıcı"),
        (r"\bAlici\b", "Alıcı"),
        (r"\bAL[İI]C[İI]\b", "Alıcı"),
        (r"\bSatici\b", "Satıcı"),
        (r"\bSAT[İI]C[İI]\b", "Satıcı"),
        (r"\beArgiv\b", "e-Arşiv"),
        (r"\be-?Arpiv\b", "e-Arşiv"),
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


def parse_ocr_line_items(text: str) -> list[Line]:
    """Parse GİB-style flat OCR rows into invoice lines."""
    out: list[Line] = []
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
        # OCR $620.00 / 620.00 → 20
        if vat_rate is not None and vat_rate >= 100:
            maybe = vat_rate % 100
            if maybe in (1, 8, 10, 18, 20):
                vat_rate = float(maybe)
            elif str(int(vat_rate)).endswith("20"):
                vat_rate = 20.0
        line_total = parse_tr_money(total_raw)
        name_clean = re.sub(r"\s+", " ", name).strip(" -")
        # Drop leading OCR junk words glued before SKU
        name_clean = re.sub(
            r"^(?:Aynasa|Asorti\w*(?:\s*//\s*Asorti\w*)*|Hav[il]u\s*3x40\w*:?\s*)",
            "",
            name_clean,
            flags=re.I,
        )
        name_clean = re.sub(r"^(?:Asorti\w*\s*//\s*)+", "", name_clean, flags=re.I).strip(" -")
        name_clean = re.sub(r"\bIstak\b", "Islak", name_clean, flags=re.I)
        name_clean = re.sub(r"\bHlaslu\b", "Havlu", name_clean, flags=re.I)
        name_clean = re.sub(r"\bHaviu\b", "Havlu", name_clean, flags=re.I)
        name_clean = re.sub(r"3x405\b", "3x40h", name_clean, flags=re.I)
        if not name_clean or line_total is None:
            continue
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
    if out:
        return out

    # GİB table OCR: "1 21.560 kg 0,089TL %0,00 0,00TL … %18,00 345,39 TL 1.918,84"
    gib_qty = re.compile(
        rf"(?m)^(?P<seq>\d{{1,3}})\s+"
        rf"(?P<name>.*?)\s*"
        rf"(?P<qty>\d{{1,3}}(?:[.,]\d+)?|\d+[.,]\d+)\s*"
        rf"(?P<unit>kg|adet|ad|NIU|C62|KGM|MTR|LTR)?\s+"
        rf"(?P<unitPrice>{_MONEY_TOKEN})\s*TL?\s+"
        rf"(?:%?\s*(?P<discRate>[\d.,]+)\s+)?"
        rf"(?:(?P<discAmt>{_MONEY_TOKEN}|L\d+[^\s]*)\s*TL?\s+)?"
        rf".{{0,48}}?"
        rf"%\s*(?P<vat>\d{{1,2}}(?:[.,]\d+)?)\s+"
        rf"(?P<vatAmt>{_MONEY_TOKEN})\s*TL?\s+"
        rf"(?P<total>{_MONEY_TOKEN})",
        re.I,
    )
    for m in gib_qty.finditer(text):
        name = re.sub(r"\s+", " ", (m.group("name") or "")).strip(" -")
        if not name or re.match(
            r"^(?:ARA|TOPLAM|KDV|Mal\s*Hizmet|Vergi|S[ıi]ra|No\b)", name, re.I
        ):
            name = f"Kalem {m.group('seq')}"
        total = parse_tr_money(m.group("total"))
        if total is None or total <= 0:
            continue
        qty = parse_tr_money(m.group("qty")) or float(m.group("qty").replace(",", "."))
        unit_price = parse_tr_money(m.group("unitPrice"))
        vat_rate = normalize_vat_rate(parse_percent(m.group("vat")))
        disc_amt = parse_tr_money(m.group("discAmt")) if m.groupdict().get("discAmt") else None
        out.append(
            Line(
                id=m.group("seq"),
                name=name[:240],
                quantity=qty,
                unit=(m.group("unit") or "Adet"),
                unitPrice=unit_price,
                discountAmount=disc_amt,
                vatRate=vat_rate,
                vatAmount=parse_tr_money(m.group("vatAmt")),
                lineTotal=total,
            )
        )
    if out:
        return out

    # Numbered GİB row with trailing money tokens (description + amounts)
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


def normalize_invoice_type(raw: str | None) -> str | None:
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
    return t or None


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
    # Soft warnings that shouldn't block ok after reconcile
    soft = [
        w
        for w in warnings
        if re.search(r"uyuşmuyor|0\.0[12]|kuruş|ETTN bulunamadı", w, re.I)
    ]
    hard = [w for w in warnings if w not in soft]
    critical = [
        w
        for w in hard
        if re.search(r"Fatura numarası|Ödenecek tutar|Satıcı|Alıcı|kalemi|tarihi", w)
    ]
    if not hard and validation.confidence >= 0.8:
        return "ok"
    if critical:
        return "partial"
    if validation.confidence < 0.5:
        return "partial"
    return "ok" if validation.confidence >= 0.75 else "partial"


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
        rf"Kdv\s*Tutar[ıia]?\s*:?\s*({_MONEY_TOKEN})",
        re.I,
    )
    footnotes = [parse_tr_money(m.group(1)) for m in footnote_re.finditer(text)]
    footnotes = [a for a in footnotes if a is not None and a > 0]
    if footnotes:
        unique = sorted({round(a, 2) for a in footnotes})
        return round(sum(unique), 2)

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
    # Reject when line totals are clearly not the invoice (bad thermal OCR names)
    line_sum = sum(l.lineTotal or 0.0 for l in inv.lines if l.lineTotal is not None)
    payable = inv.totals.payableAmount or 0.0
    if line_sum > 0 and payable > 0:
        ratio = line_sum / payable
        if ratio < 0.35 or ratio > 2.5:
            return False
        # Names that look like OCR noise (almost no vowels / too short words)
        junk = 0
        for l in inv.lines:
            n = (l.name or "").strip()
            letters = re.sub(r"[^A-Za-zÇĞİÖŞÜçğıöşü]", "", n)
            vowels = len(re.findall(r"[aeıioöuüAEIİOÖUÜ]", letters))
            if len(letters) >= 8 and vowels <= 1:
                junk += 1
        if junk >= max(1, len(inv.lines) // 2):
            return False
    if inv.issueDate and (inv.customer.name or inv.supplier.name):
        return True
    return validation.confidence >= PHOTO_OCR_MIN_CONF


def normalize_ocr_uuid(raw: str) -> str | None:
    """Fix common OCR confusions in ETTN (O→0, I/l→1, S→5, B→8)."""
    cleaned = raw.strip().upper()
    # Drop obvious OCR insertions inside hex groups while preserving dashes
    parts = cleaned.split("-")
    if len(parts) == 5:
        fixed_parts = []
        expected = [8, 4, 4, 4, 12]
        for part, n in zip(parts, expected):
            p = (
                part.replace("O", "0")
                .replace("İ", "1")
                .replace("I", "1")
                .replace("L", "1")
                .replace("S", "5")
                .replace("P", "F")  # common OCR: F→P
            )
            p = re.sub(r"[^0-9A-F]", "", p)
            if len(p) > n:
                # Prefer keeping leading chars (common: extra digit inserted)
                p = p[:n]
            fixed_parts.append(p)
        cleaned = "-".join(fixed_parts)
    else:
        cleaned = (
            cleaned.replace("O", "0")
            .replace("İ", "1")
            .replace("I", "1")
            .replace("L", "1")
            .replace("S", "5")
            .replace("P", "F")
        )
    cleaned = cleaned.lower()
    if re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        cleaned,
    ):
        return cleaned
    return None


def gib_invoice_number(text: str, file_name: str = "") -> str | None:
    """GIB-style fatura/belge no (tolerate OCR spaces/hyphens)."""
    labeled = re.search(
        r"(?:Fatura\s*No|Fatera\s*No|Invoice\s*No|B[EÉ]?[LİI1]?GE\s*N[O0]|BELGE\s*NO)\s*[:\-.]?\s*"
        r"([A-Za-z]{2,5}[\s\-]*\d{10,20}|\d{12,22})",
        text,
        re.I,
    )
    if labeled:
        compact = re.sub(r"[\s\-]+", "", labeled.group(1).upper())
        if re.fullmatch(r"[A-Z]{2,5}\d{10,20}", compact) or re.fullmatch(r"\d{12,22}", compact):
            return compact
    compact = re.sub(r"[\s|]+", "", text.upper()).replace("-", "")
    m = re.search(r"\b([A-Z]{2,5}\d{10,16})\b", compact)
    if m:
        return m.group(1)
    m = first_match(file_name, r"([A-Z]{2,5}\d{10,})")
    return m.upper() if m else None


def format_uuid_hex(raw: str) -> str | None:
    """Normalize dashed or undashed 32-hex ETTN."""
    cleaned = re.sub(r"[^0-9A-Fa-f]", "", raw)
    if len(cleaned) == 32:
        dashed = (
            f"{cleaned[0:8]}-{cleaned[8:12]}-{cleaned[12:16]}-"
            f"{cleaned[16:20]}-{cleaned[20:32]}"
        )
        return (normalize_ocr_uuid(dashed) or dashed).lower()
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
        if re.search(r"https?://|KASIYER|CHIP|ONAY|TERMINAL|ISYERI|MASTER|AID:", name, re.I):
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
    """Find ETTN even when OCR glues label+uuid (ETTN9A83...) or omits dashes."""
    hexish = r"0-9A-Fa-fİILOSBloşPGQZpgqz"
    m = re.search(
        rf"ETTN\s*[:\-]?\s*([{hexish}]{{8}}[-‑]?[{hexish}]{{4}}[-‑]?"
        rf"[{hexish}]{{4}}[-‑]?[{hexish}]{{4}}[-‑]?[{hexish}]{{12}})",
        text,
        re.I,
    )
    if m:
        return format_uuid_hex(m.group(1).replace("‑", "-"))
    m = re.search(rf"ETTN\s*[:\-]?\s*([{hexish}]{{32}})", text, re.I)
    if m:
        return format_uuid_hex(m.group(1))
    m = re.search(
        r"\b([0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12})\b",
        text,
        re.I,
    )
    return format_uuid_hex(m.group(1)) if m else None


def prefer_invoice_issue_date(text: str) -> tuple[str | None, str | None]:
    """Prefer Fatura/Tarih near metadata; ignore voucher expiry dates."""
    scrubbed = re.sub(
        r"Son\s+Kullanma\s+Tarihi\s*:?\s*\d{1,2}\s*[-./,]\s*\d{1,2}\s*[-./,]\s*\d{4}",
        " ",
        text,
        flags=re.I,
    )
    tarih_lbl = re.search(
        r"TAR[İI]H\s*:?\s*(\d{1,2}[./]\d{1,2}[./]\d{4})(?:\s+(\d{1,2}:\d{2}))?",
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
        r"(?:^|\n)\s*(\d{1,2}[./]\d{1,2}[./]\d{4})\s+(?:Saat\s*:?\s*)?(\d{1,2}:\d{2})?",
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
        right_field(scrubbed, "Fatura Tarihi")
        or right_field(scrubbed, "Fatura Tarihi")
        or right_field(scrubbed, "Tarih")
        or right_field(scrubbed, "Tarth")
        or first_match(
            scrubbed,
            r"Fatura\s*(?:Tarihi|Yarihi|Tanible)\s*:?\s*[|(]*\s*"
            r"(\d{1,2}\s*[-./]\s*\d{1,2}\s*[-./]\s*\d{4}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?)",
        )
        or first_match(
            scrubbed,
            r"(?:Fatera\s*No|Fatura\s*No)[^\n]{0,40}?\n[^\n]*?"
            r"(?:Tarih|Tarth)\s*:?\s*(\d{1,2}\s*[-./,]\s*\d{1,2}\s*[-./,]\s*\d{4})",
        )
        or first_match(
            scrubbed,
            r"(?:Tarih|Tarth)\s*:?\s*(\d{1,2}\s*[-./,]\s*\d{1,2}\s*[-./,]\s*\d{4})",
        )
        or first_match(
            scrubbed,
            r"(?<!\d)(\d{1,2}/\d{1,2}/\d{4})(?!\d)",
        )
    )
    if issue_raw:
        issue_raw = issue_raw.replace(",", ".")
    return parse_issue_date(issue_raw)


def extract_payable_from_ocr(text: str) -> float | None:
    """Prefer bank/payment lines and OCR-tolerant 'ödenecek/vergi dahil/toplam' labels."""
    bank = re.search(
        rf"(?:Kuveyt|Ziraat|Garanti|Yap[ıi]\s*Kredi|Akbank|Denizbank|Vak[ıi]f|"
        rf"Halkbank|İş\s*Bank|TEB|QNB|Enpara|Banka\s*/\s*Kredi\s*Kart[ıi]|KRED[İI]\s*KART)"
        rf"[^\n]{{0,48}}?\*?\s*({_MONEY_TOKEN})",
        text,
        re.I,
    )
    if bank:
        amt = parse_tr_money(bank.group(1))
        if amt is not None:
            return amt
    for label in (
        r"[ÖO]DENECEK\s+TUTAR",
        r"Ödenecek\s+Tutar",
        r"VERG[İIEÉ]\s+DAH[İI]L\s+TOPLAM\s+TUTAR",
        r"Vergiler\s+Dahil\s+Toplam\s+Tutar",
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
    m = re.search(
        r"(\d{1,2})\s*[-./]\s*(\d{1,2})\s*[-./]\s*(\d{4})(?:\s+(\d{1,2}):(\d{2}))?",
        raw,
    )
    if not m:
        return None, None
    day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
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
    ]
    # Explicit Satıcı: label (GİB / ERP layouts)
    sat_m = re.search(
        r"Sat[ıi]c[ıi]\s*[^:\n]{0,24}:?\s*([A-ZÇĞİÖŞÜa-zçğıöşü0-9].{2,120})",
        head,
        re.I,
    )
    if sat_m:
        cand = re.split(r"\s{2,}|Adres\s*:|Tel(?:efon)?\s*:|Vergi", sat_m.group(1), maxsplit=1)[
            0
        ].strip(" :.-[]{}")
        cand = re.sub(r"^(?:şube|sube)\]?\s*:?\s*", "", cand, flags=re.I).strip(" :.-[]{}")
        if len(cand) >= 4:
            party.name = cand[:180]
    # Prefer a line that looks like a company title (legal form / retail trade words)
    if not party.name:
        company = next(
            (
                ln
                for ln in lines
                if re.search(
                    r"(?:LTD|ŞT[İI]|A\.?\s*Ş\.?|SANAY[İI]|T[İI]CARET|ANON[İI]M|MA[ĞG]AZA)",
                    ln,
                    re.I,
                )
                and not re.search(r"^(?:PDF|XML|Page|Adres|Tel|Web|Vergi)", ln, re.I)
            ),
            None,
        )
        if company:
            party.name = company[:180]
        elif lines and not re.match(
            r"^(Tel|Web|E-?Posta|Vergi|TCKN|VKN|Kap[ıi]|Telefon|Adres)", lines[0], re.I
        ):
            name = lines[0]
            if (
                len(lines) > 1
                and re.search(r"(?:LTD|ŞT[İI]|A\.?\s*Ş\.?|SAN\.|T[İI]C\.|ANON[İI]M)", lines[1], re.I)
                and not re.match(r"^(Tel|Web|E-?Posta|Vergi|TCKN|VKN|ŞUBE)", lines[1], re.I)
            ):
                name = f"{lines[0]} {lines[1]}"
            party.name = name[:180]
    if party.name:
        party.name = re.sub(
            r"^(?:Sat[ıi]c[ıi]\s*(?:\([^)]*\)|\{[^}]*\}|\[[^\]]*\])?\s*:?\s*)",
            "",
            party.name,
            flags=re.I,
        )
        party.name = re.sub(r"\bHAGAZACILIK\b", "MAGAZACILIK", party.name, flags=re.I)
        party.name = re.sub(r"\bMAČAZACILIK\b", "MAGAZACILIK", party.name, flags=re.I)
        party.name = re.sub(r"\s+", " ", party.name).strip(" :.-[]{}")[:180]
    # Vergi No / Dairesi: 4470211661 / KADIKÖY
    vkn_office = re.search(
        r"Vergi\s*No\s*/?\s*Dairesi?\s*:?[.\s]*(\d{10,11})\s*/\s*([A-ZÇĞİÖŞÜa-zçğıöşü ]{2,40})",
        head,
        re.I,
    )
    if vkn_office:
        tid = vkn_office.group(1)
        party.taxId = tid
        party.taxIdScheme = "TCKN" if len(tid) == 11 else "VKN"
        party.taxOffice = vkn_office.group(2).strip()
    if not party.taxOffice:
        party.taxOffice = (
            first_match(
                head, r"Vergi\s*Dai(?:resi|resi|r[ae]s[il])\s*:?\s*([A-ZÇĞİÖŞÜa-zçğıöşü ]{3,40})"
            )
            or ""
        ).strip() or None
    if party.taxOffice:
        party.taxOffice = re.split(r"\s{2,}|Vergi\s*num", party.taxOffice, maxsplit=1)[0].strip()
    tckn = first_match(head, r"TCKN\s*:?\s*(\d{11})")
    vkn = first_match(head, r"(?:VKN|Vergi\s*numar\w*|Vergi\s*No)\s*:?[.\s]*(\d{10})")
    if not vkn:
        vkn = first_match(head, r"\bV\.?\s*N\.?\s*:?[.\s]*(\d{10})\b")
    if not vkn:
        vkn = first_match(head, r"Vergi\s*No\s*/?\s*Dairesi?\s*:?[.\s]*(\d{10})")
    # Prefer VKN for company suppliers when both appear
    if vkn and not party.taxId:
        party.taxId, party.taxIdScheme = vkn, "VKN"
    elif tckn and not party.taxId:
        party.taxId, party.taxIdScheme = tckn, "TCKN"
    party.email = first_match(head, r"(?:E-?Posta|E-?Mall|E-?Mail)\s*:?\s*([^\s]+)")
    party.website = first_match(head, r"Web\s*Sitesi\s*:?\s*([^\s]+)")
    party.phone = (
        re.sub(
            r"\s+",
            "",
            first_match(head, r"(?:Tel|Telefon)\s*:?\s*([0-9\s()\-]+)") or "",
        )
        or None
    )
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

    # Explicit Alıcı: (ERP / e-Fatura)
    alici = re.search(
        r"Al[ıi]c[ıi]\s*[^:\nA-ZÇĞİÖŞÜ]{0,20}:?\s*([A-ZÇĞİÖŞÜa-zçğıöşü].{2,80})",
        text,
        re.I,
    )
    if alici:
        cand = re.split(
            r"\s{2,}|Adres\s*:|Tel(?:efon)?\s*:|Vergi|Özelleştirme|Senaryo|Fatura",
            alici.group(1),
            maxsplit=1,
        )[0].strip(" :.-[]{}")
        if len(cand) >= 3 and not re.search(r"Nihai|T[uü]ketici", cand, re.I):
            party.name = cand[:120]

    sayin = re.search(r"\bSAYIN\b", text, re.I)
    if not sayin and not party.name:
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
        near = block[:1500]
    else:
        near = text[alici.start() : alici.start() + 1500] if alici else text[:1500]

    vkn_tckn = first_match(near, r"VKN\s*/\s*TCKN\s*:?\s*(\d{10,11})")
    tckn = first_match(near, r"TCKN\s*:?\s*(\d{11})")
    vkn = first_match(near, r"VKN\s*:?\s*(\d{10})")
    if not vkn:
        vkn = first_match(near, r"Vergi\s*No\s*/?\s*Dairesi?\s*:?\s*(\d{10,11})")
    if tckn:
        party.taxId, party.taxIdScheme = tckn, "TCKN"
    elif vkn:
        party.taxId = vkn
        party.taxIdScheme = "TCKN" if len(vkn) == 11 else "VKN"
    elif vkn_tckn:
        party.taxId = vkn_tckn
        party.taxIdScheme = "TCKN" if len(vkn_tckn) == 11 else "VKN"
    party.taxOffice = (first_match(near, r"Vergi\s*Dairesi\s*:?\s*([^\n]+)") or "").split("  ")[0].strip() or None
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
    header = [h.lower() for h in rows[0]]
    # Skip non-item summary tables
    header_join = " ".join(header)
    if "mal hizmet toplam" in header_join or (
        len(rows[0]) <= 2 and any("tutar" in h for h in header)
    ):
        # totals-only mini table — ignore for lines
        return []

    def col(*names: str) -> int | None:
        for n in names:
            for i, h in enumerate(header):
                if n in h:
                    return i
        return None

    idx_id = col("sıra", "sira")
    idx_name = col("mal/hizmet tanımı", "mal hizmet tanımı", "mal hizmet", "mal/hizmet", "açıklama", "tanım")
    # Avoid matching "ürün kodu" as description
    idx_code = col("ürün kodu", "urun kodu", "satıcı ürün")
    idx_qty = col("miktar")
    idx_unit_price = col("birim fiyat")
    idx_line = col("mal hizmet tutarı", "hizmet tutarı")
    idx_vat_rate = col("kdv oranı")
    idx_vat_amt = col("kdv tutarı", "kdv tutari")
    if idx_name is None:
        idx_name = col("açıklama", "tanım")
    if idx_line is None:
        idx_line = col("tutarı", "tutar")
    if idx_id is None:
        idx_id = 0
    if idx_vat_rate is None:
        idx_vat_rate = col("oran")
    if idx_unit_price is None:
        idx_unit_price = col("birim")
    if idx_code is None:
        idx_code = col("kod")
    # Prefer description column not equal to totals
    out: list[Line] = []
    for row in rows[1:]:
        if not row:
            continue
        raw_id = row[idx_id] if idx_id is not None and idx_id < len(row) else row[0]
        if not re.match(r"^\d+$", raw_id.strip()):
            continue
        name = row[idx_name] if idx_name is not None and idx_name < len(row) else None
        code = row[idx_code] if idx_code is not None and idx_code < len(row) else None
        if code and name and code not in name:
            name = f"{code} {name}"
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
        # Avoid grabbing wrong "tutar" column that is unit price when ambiguous
        if line_total is not None and unit_price is not None and qty and nearly_equal(line_total, unit_price) and qty > 1:
            # likely swapped — keep as-is if product of qty matches alternate
            pass
        out.append(
            Line(
                id=raw_id.strip(),
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
        or right_field(text, "Belge No")
        or right_field(text, "BELGE NO")
        or first_match(
            text,
            r"(?:B[EÉ]?[LİI1]?GE|BELGE)\s*N[O0]\s*[:\-.]?\s*([A-Za-z]{2,5}[\s\-]*\d{10,20}|\d{12,22})",
        )
    )
    if labeled_no:
        cleaned = re.sub(r"[^A-Za-z0-9]", "", labeled_no).upper()
        if re.fullmatch(r"[A-Z]{2,5}\d{10,20}", cleaned) or re.fullmatch(r"\d{12,22}", cleaned):
            inv_no = cleaned
    if inv_no:
        inv_no = re.sub(r"\s+", "", inv_no).upper()
        if not (
            re.fullmatch(r"[A-Z]{2,5}\d{10,20}", inv_no) or re.fullmatch(r"\d{12,22}", inv_no)
        ):
            inv_no = None

    issue_date, issue_time = prefer_invoice_issue_date(text)
    for label in ("Fatura Saati", "Düzenleme Zamanı", "Düzenleme Zamans", "Oluşma Zamanı"):
        raw = right_field(text, label) or first_match(
            text, rf"{label}\s*:?\s*(\d{{1,2}}:\d{{2}}:\d{{2}})"
        )
        if raw and not issue_time:
            tm = re.search(r"(\d{1,2}:\d{2}:\d{2})", raw)
            if tm:
                issue_time = tm.group(1)

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
    if not ocr_lines:
        ocr_lines = parse_retail_pos_lines(text)
    retail_fiş = bool(
        re.search(r"\badet\s*[x×X]\b|B[İI]LG[İI]\s*F[İI][ŞS]|TOPKDV|BELGE\s*N[O0]|BEBGE\s*N[O0]", text, re.I)
    )
    lines_sum = None
    if ocr_lines:
        totals_present = [l.lineTotal for l in ocr_lines if l.lineTotal is not None]
        if totals_present:
            lines_sum = round(sum(totals_present), 2)
    # Prefer line-item sum (true matrah) over partial OCR matrah footnotes
    line_ext = (
        lines_sum
        if lines_sum is not None and not retail_fiş
        else matrah
        if matrah is not None
        else (round(net - discount, 2) if net is not None and discount and discount > 0 else net)
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

    bank_pay = None
    bank_m = re.search(
        rf"(?:Kuveyt|Ziraat|Garanti|Yap[ıi]\s*Kredi|Akbank|Denizbank|Vak[ıi]f|"
        rf"Halkbank|İş\s*Bank|TEB|QNB|Enpara)[^\n]{{0,48}}?"
        rf"({_MONEY_TOKEN})\s*TRY",
        text,
        re.I,
    )
    if bank_m:
        bank_pay = parse_tr_money(bank_m.group(1))
    # Only use bank if ödenecek missing
    if payable is None and bank_pay is not None:
        payable = bank_pay
        tax_inclusive = tax_inclusive or bank_pay
    vat = extract_vat_amount(text)
    if vat is None:
        vat = (
            labeled_amount(text, r"TOPKDV")
            or first_match_money(text, rf"TOPKDV\s*\*\s*({_MONEY_TOKEN})")
            or first_match_money(text, rf"(?m)^KDV\s*\*\s*({_MONEY_TOKEN})")
        )
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
        if payable is None or (payable is not None and nearly_equal(payable, lines_sum, 1.0)):
            payable = lines_sum
        tax_inclusive = payable
        if vat is not None and payable is not None and payable > vat:
            line_ext = round(payable - vat, 2)
        elif line_ext is None and payable is not None and vat is not None:
            line_ext = round(payable - vat, 2)

    # Reconcile: heal missed multi-rate VAT from lines/totals
    if line_ext is not None and tax_inclusive is not None and tax_inclusive >= line_ext:
        implied = round(tax_inclusive - line_ext, 2)
        if vat is None or abs((line_ext + (vat or 0)) - tax_inclusive) > 0.05:
            vat = implied
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
    # Retail: "… VD 1234567890" or "İLÇE/1234567890" — not "Müşteri VKN"
    musteri_vkn = first_match(text, r"M[uüu][sşs]teri\s+VKN\s*:?\s*(\d{10,11})")
    vd_vkn = first_match(text, r"(?:^|\n)[^\n]*\bVD\s*(\d{10})\b")
    district_vkn = first_match(
        text,
        r"(?:^|\n)\s*[A-ZÇĞİÖŞÜa-zçğıöşü]{3,}(?:\s*/\s*|\s+)(\d{10})\b",
    )
    supplier_vkn = vd_vkn or district_vkn
    if supplier_vkn and supplier_vkn != (musteri_vkn or "")[:10]:
        supplier.taxId = supplier_vkn
        supplier.taxIdScheme = "VKN"
    elif not supplier.taxId and supplier_vkn:
        supplier.taxId = supplier_vkn
        supplier.taxIdScheme = "VKN"

    if supplier.name:
        # OCR: common misspellings of MAGAZACILIK
        supplier.name = re.sub(r"\bHAGAZACILIK\b", "MAGAZACILIK", supplier.name, flags=re.I)
        supplier.name = re.sub(r"\bMA[ČĆ]AZACILIK\b", "MAGAZACILIK", supplier.name, flags=re.I)
        # Drop address tokens glued onto company title (any issuer)
        supplier.name = re.split(
            r"\s+(?:MAH\.|CAD\.|SK\.|SOK\.|BULVAR|NO:|Kap[ıi]\s*No|PROF\.)",
            supplier.name,
            maxsplit=1,
            flags=re.I,
        )[0].strip()[:180]
        # Drop OCR junk prefixes (short codes) when a better MAGAZACILIK/A.Ş title exists later
        if (
            len(supplier.name) < 4
            or "mgzkodu" in supplier.name.lower()
            or re.match(r"^[a-z]{2,4}\d", supplier.name, re.I)
        ):
            better = first_match(
                text,
                r"((?:[A-ZÇĞİÖŞÜ0-9][A-ZÇĞİÖŞÜa-zçğıöşü0-9 .&-]{4,60}?)"
                r"(?:MA[ĞG]AZACILIK|T[İI]CARET|SANAY[İI]).{0,40}?(?:A\.?\s*Ş\.?|LTD\.?\s*ŞT[İI]))",
            )
            if better:
                supplier.name = re.sub(r"\s+", " ", better).strip()[:180]

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
            customer.name = "Nihai Tüketici"
    # POS: "MÜŞTERİ: …" when name still missing
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
        if near_card and not re.search(r"MASTER|KART|CHIP|ONAY", near_card, re.I):
            customer.name = near_card.strip()[:120]
        else:
            customer.name = "Nihai Tüketici"

    # Drop OCR junk lines when totals clearly don't match payable
    if ocr_lines and payable:
        line_sum = sum(l.lineTotal or 0.0 for l in ocr_lines if l.lineTotal is not None)
        if line_sum > 0 and (line_sum / payable) < 0.45:
            ocr_lines = []
            lines_sum = None
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

    # Reject junk supplier names
    if supplier.name and _party_name_quality(supplier.name) < 6:
        better = next(
            (
                ln.strip()
                for ln in text.splitlines()
                if _party_name_quality(ln.strip()) >= 20
            ),
            None,
        )
        if better:
            supplier.name = better[:180]
        elif _party_name_quality(supplier.name) < 4:
            supplier.name = None

    return Invoice(
        documentType=doc_type,
        profileId=profile,
        customizationId=right_field(text, "Özelleştirme No")
        or right_field(text, "Ozellestirme No")
        or first_match(text, r"Özelleştirme\s*No\s*:?\s*(TR[\d.]+)"),
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
        bankName=bank.strip() if bank else None,
        bankBranch=None,
    )


def _party_name_quality(name: str | None) -> int:
    if not name:
        return 0
    n = name.strip()
    if len(n) < 3 or re.fullmatch(r"[=_\-—.\s]+", n):
        return 0
    if re.search(
        r"(?:[İI]mza|PDF\s*indir|XML\s*indir|<!--\s*image|Page\s+\d+|adet\s*[x×]|Higz\s*Adi)",
        n,
        re.I,
    ):
        return 0
    letters = len(re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü]", n))
    if letters < 4:
        return 0
    score = letters
    if re.search(r"(?:LTD|ŞT[İI]|A\.?\s*Ş|SANAY|TICARET|MA[ĞG]AZA)", n, re.I):
        score += 20
    if re.search(r"Senaryo|Fatura\s*Tipi|==|^[#]+", n, re.I):
        score -= 30
    return score


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
        if v is not None and data["totals"].get(k) is None:
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

    need(bool(inv.invoiceNumber), "Fatura numarası bulunamadı")
    need(bool(inv.uuid), "ETTN bulunamadı", 0.08)
    need(bool(inv.issueDate), "Fatura tarihi bulunamadı")
    need(bool(inv.supplier.name), "Satıcı unvanı bulunamadı")
    need(bool(inv.customer.name), "Alıcı unvanı bulunamadı")
    need(inv.totals.payableAmount is not None, "Ödenecek tutar bulunamadı")
    need(len(inv.lines) > 0, "Mal/hizmet kalemi bulunamadı", 0.18)

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
    options.do_table_structure = True
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
    if not ENABLE_DOCLING:
        return
    try:
        get_docling_converter(ocr=False, for_image=False)
    except Exception as exc:  # noqa: BLE001
        print(f"docling warmup skipped: {exc}")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "fatura-ai-extract",
        "docling": ENABLE_DOCLING,
        "doclingOcr": ENABLE_DOCLING_OCR,
        "forceImageOcr": FORCE_IMAGE_OCR,
        "photoOcr": PHOTO_OCR_ENABLED,
        "fastPathPdf": FAST_PATH_PDF,
        "doclingMaxInflight": DOCLING_MAX_INFLIGHT,
        "doclingTimeoutS": DOCLING_TIMEOUT_S,
        "inflight": _metrics["inflight"],
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
            if PHOTO_OCR_ENABLED:
                try:
                    from photo_ocr import ocr_image

                    photo_text, photo_meta = await asyncio.to_thread(ocr_image, path)
                    pipeline.append(
                        f"photo-ocr:{photo_meta.get('engine', '?')}:"
                        f"{photo_meta.get('elapsedMs', 0)}ms:"
                        f"{photo_meta.get('lineCount', 0)}"
                    )
                    _metrics["photo_ocr"] += 1
                    if photo_text.strip():
                        text = photo_text
                        md = photo_text
                        inv_photo = parse_text_invoice(photo_text, name)
                        invoice = merge_invoice(invoice, inv_photo)
                        if not invoice.lines and inv_photo.lines:
                            invoice.lines = inv_photo.lines
                        warnings_ph, validation_ph = validate_invoice(invoice)
                        if strong_photo_invoice(invoice, validation_ph):
                            pipeline.append("photo-ocr-fast-path")
                except Exception as exc:  # noqa: BLE001
                    pipeline.append(f"photo-ocr-error:{exc}")

            # Heavy Docling only if photo OCR missed critical fields
            need_docling = not strong_photo_invoice(
                invoice, validate_invoice(invoice)[1]
            )
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
                        invoice = merge_invoice(invoice, inv_md)
                        if not _lines_useful(invoice.lines) and _lines_useful(inv_md.lines):
                            invoice.lines = inv_md.lines
                        md = (md + "\n\n" + md2).strip() if md else md2
                except Exception as exc:  # noqa: BLE001
                    pipeline.append(f"docling-image-error:{exc}")

            if need_docling or not invoice.invoiceNumber:
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

            try:
                text = await asyncio.to_thread(pdftotext, path)
                pipeline.append("pdftotext")
            except Exception as exc:  # noqa: BLE001
                pipeline.append(f"pdftotext-error:{exc}")

            invoice = parse_text_invoice(text, name) if text.strip() else Invoice()
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

        for line in invoice.lines:
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
        if need_ocr:
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
        if not invoice.invoiceNumber and not invoice.totals.payableAmount and not invoice.lines:
            status = "failed"

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
