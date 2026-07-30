"""FaturaAI extract service — UBL → Docling(+tables) → pdftotext heuristics → validate."""

from __future__ import annotations

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
    s = re.sub(r"(?i)TL|TRY|₺", "", raw).replace(" ", "").strip()
    if not s:
        return None
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    elif re.fullmatch(r"\d{1,3}(\.\d{3})+", s):
        s = s.replace(".", "")
    try:
        n = float(s)
        return n if n == n else None
    except ValueError:
        return None


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
        return None
    raw = m.group(1).strip()
    parts = [p.strip() for p in re.split(r"\s{2,}", raw) if p.strip()]
    return (parts[-1] if parts else raw) or None


def first_match(text: str, pattern: str, flags: int = re.I) -> str | None:
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else None


def labeled_amount(text: str, label: str) -> float | None:
    re_ = re.compile(
        rf"{label}(?:\s*\([^)]*\))?\s*:?\s*([\d.\s]+,\d{{2,}})\s*(?:TL|TRY)?",
        re.I,
    )
    matches = list(re_.finditer(text))
    if not matches:
        return None
    return parse_tr_money(matches[-1].group(1))


def parse_issue_date(raw: str | None) -> tuple[str | None, str | None]:
    if not raw:
        return None, None
    m = re.search(
        r"(\d{1,2})\s*[-./]\s*(\d{1,2})\s*[-./]\s*(\d{4})(?:\s+(\d{1,2}):(\d{2}))?",
        raw,
    )
    if not m:
        return None, None
    date = f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"
    time_ = None
    if m.group(4) and m.group(5):
        time_ = f"{m.group(4).zfill(2)}:{m.group(5)}:00"
    return date, time_


def empty_party() -> Party:
    return Party()


def extract_supplier(text: str) -> Party:
    party = empty_party()
    sayin = re.search(r"\bSAYIN\b", text, re.I)
    head = text[: sayin.start()] if sayin else text[:900]
    lines = [
        ln.strip()
        for ln in head.splitlines()
        if ln.strip()
        and not re.match(r"^e-?Ar[sş]iv\s+Fatura$", ln.strip(), re.I)
        and not re.match(r"^Sayfa\s+\d+", ln.strip(), re.I)
    ]
    if lines and not re.match(r"^(Tel|Web|E-?Posta|Vergi|TCKN|VKN|Kap[ıi])", lines[0], re.I):
        name = lines[0]
        if (
            len(lines) > 1
            and re.search(r"(?:LTD|ŞT[İI]|A\.?\s*Ş\.?|SAN\.|T[İI]C\.|ANON[İI]M)", lines[1], re.I)
            and not re.match(r"^(Tel|Web|E-?Posta|Vergi|TCKN|VKN|ŞUBE)", lines[1], re.I)
        ):
            name = f"{lines[0]} {lines[1]}"
        party.name = name[:180]
    party.taxOffice = (first_match(head, r"Vergi\s*Dairesi\s*:?\s*([^\n]+)") or "").split("  ")[0].strip() or None
    tckn = first_match(head, r"TCKN\s*:?\s*(\d{11})")
    vkn = first_match(head, r"VKN\s*:?\s*(\d{10})")
    if tckn:
        party.taxId, party.taxIdScheme = tckn, "TCKN"
    elif vkn:
        party.taxId, party.taxIdScheme = vkn, "VKN"
    party.email = first_match(head, r"E-?Posta\s*:?\s*([^\s]+)")
    party.website = first_match(head, r"Web\s*Sitesi\s*:?\s*([^\s]+)")
    party.phone = re.sub(r"\s+", "", first_match(head, r"Tel\s*:?\s*([0-9\s()]+)") or "") or None
    return party


def extract_customer(text: str) -> Party:
    party = empty_party()
    sayin = re.search(r"\bSAYIN\b", text, re.I)
    if not sayin:
        return party
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
            r"\b(mah\.|Mah\.|Bul\.|Cad\.|Sk\.|No:|daire|sitesi|Ankara|İstanbul|Etimesgut|MAMAK)\b",
            ln,
            re.I,
        ):
            addr_parts.append(ln)
            continue
        if not name_parts:
            name_parts.append(ln)
        elif not addr_parts and len(ln) < 80:
            name_parts.append(ln)
    party.name = " ".join(name_parts).strip() or None
    if party.name:
        halves = party.name.split()
        mid = len(halves) // 2
        if mid > 0 and " ".join(halves[:mid]) == " ".join(halves[mid:]):
            party.name = " ".join(halves[:mid])
    party.address = ", ".join(addr_parts) or None
    near = block[:1500]
    vkn_tckn = first_match(near, r"VKN\s*/\s*TCKN\s*:?\s*(\d{10,11})")
    tckn = first_match(near, r"TCKN\s*:?\s*(\d{11})")
    vkn = first_match(near, r"VKN\s*:?\s*(\d{10})")
    if tckn:
        party.taxId, party.taxIdScheme = tckn, "TCKN"
    elif vkn:
        party.taxId, party.taxIdScheme = vkn, "VKN"
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
    doc_type: Literal["earsiv", "efatura", "ubl", "unknown"] = "unknown"
    if re.search(r"e-?Ar[sş]iv\s+Fatura|EARSIVFATURA", text, re.I):
        doc_type = "earsiv"
    elif re.search(r"e-?Fatura", text, re.I):
        doc_type = "efatura"

    inv_no = right_field(text, "Fatura No") or first_match(file_name, r"([A-Z]{2,5}\d{10,})")
    if inv_no:
        inv_no = re.sub(r"\s+", "", inv_no).upper()

    issue_raw = right_field(text, "Fatura Tarihi") or first_match(
        text, r"(?:^|\n)[^\n]*?\bTarih\s*:?\s*(\d{1,2}\s*[-./]\s*\d{1,2}\s*[-./]\s*\d{4})"
    )
    issue_date, issue_time = parse_issue_date(issue_raw)
    for label in ("Fatura Saati", "Düzenleme Zamanı", "Oluşma Zamanı"):
        raw = right_field(text, label)
        if raw and not issue_time:
            tm = re.search(r"(\d{1,2}:\d{2}:\d{2})", raw)
            if tm:
                issue_time = tm.group(1)

    uuid = first_match(
        text,
        r"ETTN\s*:?\s*([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
    )
    if uuid:
        uuid = uuid.lower()

    net = labeled_amount(text, "Mal Hizmet Toplam Tutarı") or labeled_amount(text, "NET TOPLAM")
    discount = labeled_amount(text, "Toplam [İI]skonto") or labeled_amount(text, "TOPLAM [İI]SKONTO")
    line_ext = (
        round(net - discount, 2) if net is not None and discount and discount > 0 else net
    )

    iban = first_match(text, r"I?İ?BAN\s*:\s*(TR[\d\s]+)")
    if iban:
        iban = re.sub(r"\s+", "", iban).upper()
    bank = first_match(text, r"([A-ZÇĞİÖŞÜa-zçğıöşü ]+BANKASI)\s*/\s*I?İ?BAN")

    return Invoice(
        documentType=doc_type,
        profileId=right_field(text, "Senaryo"),
        customizationId=right_field(text, "Özelleştirme No"),
        invoiceTypeCode=right_field(text, "Fatura Tipi"),
        invoiceNumber=inv_no,
        uuid=uuid,
        issueDate=issue_date,
        issueTime=issue_time,
        supplier=extract_supplier(text),
        customer=extract_customer(text),
        lines=[],
        totals=Totals(
            lineExtensionAmount=line_ext,
            discountTotal=discount,
            withholdingVatAmount=labeled_amount(text, "Hesaplanan KDV Tevkifat"),
            vatAmount=labeled_amount(text, r"Hesaplanan KDV(?!\s*Tevkifat)")
            or labeled_amount(text, "KDV"),
            taxInclusiveAmount=labeled_amount(text, "Vergiler Dahil Toplam Tutar")
            or labeled_amount(text, r"VERG[İI] DAH[İI]L TOPLAM TUTAR"),
            payableAmount=labeled_amount(text, "Ödenecek Tutar")
            or labeled_amount(text, "ÖDENECEK TUTAR"),
            currency="TRY",
        ),
        notes=[],
        iban=iban,
        bankName=bank.strip() if bank else None,
        bankBranch=None,
    )


def merge_invoice(base: Invoice, overlay: Invoice) -> Invoice:
    data = base.model_dump()
    over = overlay.model_dump()
    for k, v in over.items():
        if k in {"supplier", "customer", "totals", "lines", "notes"}:
            continue
        if v and not data.get(k):
            data[k] = v
    for side in ("supplier", "customer"):
        for k, v in over[side].items():
            if v and not data[side].get(k):
                data[side][k] = v
    for k, v in over["totals"].items():
        if v is not None and data["totals"].get(k) is None:
            data["totals"][k] = v
    if over["lines"] and (not data["lines"] or len(over["lines"]) >= len(data["lines"])):
        data["lines"] = over["lines"]
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
        try:
            from docling.datamodel.pipeline_options import EasyOcrOptions

            options.ocr_options = EasyOcrOptions(lang=["tr", "en"])
        except Exception:
            pass

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


def status_from(warnings: list[str], validation: Validation) -> Literal["ok", "partial", "failed"]:
    critical = [w for w in warnings if re.search(r"Fatura numarası|Ödenecek tutar|Satıcı|Alıcı|kalemi", w)]
    if not warnings and validation.confidence >= 0.85:
        return "ok"
    if critical:
        return "partial"
    if validation.confidence < 0.5:
        return "partial"
    return "ok" if validation.confidence >= 0.75 else "partial"


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
        "imageFormats": sorted(ext.lstrip(".") for ext in IMAGE_EXTENSIONS),
    }


@app.post("/extract", response_model=ExtractResponse)
async def extract(
    file: UploadFile = File(...),
    filename: str | None = Form(None),
) -> ExtractResponse:
    started = time.perf_counter()
    pipeline: list[str] = []
    name = filename or file.filename or "invoice.pdf"
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
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
                convert_heic_to_jpeg(path, jpeg_path)
                path = jpeg_path
                pipeline.append("heic-jpeg")
            except Exception as exc:  # noqa: BLE001
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
            if not ENABLE_DOCLING:
                return ExtractResponse(
                    status="failed",
                    method="none",
                    durationMs=int((time.perf_counter() - started) * 1000),
                    warnings=["Fotoğraf okuma için Docling gerekli (ENABLE_DOCLING=1)"],
                    pipeline=pipeline,
                )
            try:
                use_ocr = FORCE_IMAGE_OCR or ENABLE_DOCLING_OCR
                md, table_lines = docling_convert(path, ocr=use_ocr, for_image=True)
                pipeline.append("docling-image-ocr" if use_ocr else "docling-image")
                if table_lines:
                    invoice.lines = table_lines
                    pipeline.append(f"docling-tables:{len(table_lines)}")
                if md.strip():
                    inv_md = parse_text_invoice(md.replace("\t", " "), name)
                    invoice = merge_invoice(invoice, inv_md)
                    if not invoice.lines and inv_md.lines:
                        invoice.lines = inv_md.lines
            except Exception as exc:  # noqa: BLE001
                pipeline.append(f"docling-image-error:{exc}")
                return ExtractResponse(
                    status="failed",
                    method="none",
                    durationMs=int((time.perf_counter() - started) * 1000),
                    warnings=[f"Fotoğraf okunamadı: {exc}"],
                    pipeline=pipeline,
                )
        else:
            # 1) UBL
            ubl = extract_embedded_ubl(data)
            if ubl and re.search(r"<(?:\w+:)?Invoice[\s>]", ubl, re.I):
                pipeline.append("ubl")
                # Minimal UBL handling — defer to text path if complex; still mark method
                # For now continue with text sources which also see UUID etc.

            # 2) pdftotext heuristics
            try:
                text = pdftotext(path)
                pipeline.append("pdftotext")
            except Exception as exc:  # noqa: BLE001
                pipeline.append(f"pdftotext-error:{exc}")

            invoice = parse_text_invoice(text, name) if text.strip() else Invoice()

            # 3) Docling structure/tables
            if ENABLE_DOCLING:
                try:
                    md, table_lines = docling_convert(path, ocr=False, for_image=False)
                    pipeline.append("docling-structure")
                    if table_lines:
                        invoice.lines = table_lines
                        pipeline.append(f"docling-tables:{len(table_lines)}")
                    # Merge metadata from docling markdown too
                    if md.strip():
                        inv_md = parse_text_invoice(md.replace("\t", " "), name)
                        invoice = merge_invoice(invoice, inv_md)
                        if not invoice.lines and inv_md.lines:
                            invoice.lines = inv_md.lines
                except Exception as exc:  # noqa: BLE001
                    pipeline.append(f"docling-error:{exc}")

        warnings, validation = validate_invoice(invoice)

        # Sanitize nonsense unit prices from mis-mapped tables
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

        # 4) OCR fallback if weak PDF result (images already OCR'd above)
        need_ocr = (
            not as_image
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
                md2, table_lines2 = docling_convert(path, ocr=True, for_image=False)
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
            if not p.endswith("-error") and not p.startswith("docling-error") and not p.startswith("docling-image-error")
        ) or "none"
        if "ubl" in pipeline:
            method = "ubl+" + method if method != "none" else "ubl"

        preview = (md or text)[:2500] if (md or text) else None
        status = status_from(warnings, validation)
        if not invoice.invoiceNumber and not invoice.totals.payableAmount and not invoice.lines:
            status = "failed"

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
