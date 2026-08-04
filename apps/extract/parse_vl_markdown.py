"""Generic VL markdown → invoice field binder (layout-agnostic).

Does not use supplier templates. Normalizes PaddleOCR-VL markdown/HTML,
applies synonym + fuzzy label matching and light spatial/section heuristics,
then merges with the classic text parser on the cleaned text.
"""

from __future__ import annotations

import html
import re
from typing import Callable

# Imported lazily-friendly from main to avoid circular imports at module load
# Callers pass parse helpers or we import inside functions.


_LAYOUT_NOISE = re.compile(
    r"(?im)^(?:number|header_image|header|table|image|figure_title|"
    r"paragraph_title|footer|text|vision_footnote|RGB)\s*$"
)
_IMG_LINE = re.compile(r"(?im)^(?:imgs?/[\w./-]+\.(?:jpg|jpeg|png|webp)|RGB)\s*$")
_PATH_LINE = re.compile(r"(?im)^(?:/tmp/|/data/|[A-Za-z]:\\).+\.(?:png|jpg|jpeg|pdf)\s*$")


# Synonym groups — Turkish e-belge variations (generic, not supplier-specific)
PAYABLE_LABELS = [
    r"Ödenecek\s*Tutar",
    r"Odenecek\s*Tutar",
    r"Ödenecek",
    r"Odenecek",
    r"Genel\s*Toplam",
    r"Vergiler\s*Dahil\s*Toplam\s*Tutar",
    r"Vergiler\s*Dahil\s*Toplam",
    r"Toplam\s*Tutar",
    r"Fatura\s*Tutari",
    r"Fatura\s*Tutarı",
    r"FATURA\s*TUTARI",
    r"Net\s*Ödenecek",
]
LINE_EXT_LABELS = [
    r"Mal\s*Hizmet\s*Toplam\s*Tutarı",
    r"Mal\s*Hizmet\s*Toplam\s*Tutari",
    r"Mal\s*/?\s*Hizmet\s*Toplam",
    r"Mal\s*Hizmet\s*Toplam",
    r"Net\s*Toplam\s*Tutar",
    r"Net\s*Toplam",
    r"Ara\s*Toplam",
    r"Toplam\s*Brüt\s*Tutar",
    r"KDV\s*Matrah[ıi]?",
    r"Matrah",
]
VAT_LABELS = [
    r"Hesaplanan\s*KDV(?:\s*\(%?\d+(?:[.,]\d+)?%?\))?",
    r"KDV\s*Tutarı",
    r"KDV\s*Tutari",
    r"Toplam\s*KDV",
    r"KDV\s*Toplam",
]
DISCOUNT_LABELS = [
    r"Toplam\s*[İI]skonto",
    r"İskonto\s*Toplam",
    r"Iskonto\s*Toplam",
    r"Toplam\s*İskonto",
]
WITHHOLD_LABELS = [
    r"Hesaplanan\s*KDV\s*Tevkifat",
    r"KDV\s*Tevkifat",
    r"Tevkifat\s*Tutarı",
]

INVOICE_NO_LABELS = [
    r"Fatura\s*No",
    r"Fatura\s*Numaras[ıi]",
    r"Belge\s*No",
    r"Invoice\s*No",
    r"ERP\s*Fatura\s*No",
]
DATE_LABELS = [
    r"Fatura\s*Tarihi",
    r"Düzenleme\s*Tarihi",
    r"Belge\s*Tarihi",
    r"Tarih",
]
ETTN_LABELS = [r"ETTN", r"Ettn", r"UUID"]
VKN_LABELS = [
    r"VKN",
    r"Vergi\s*No",
    r"Vergi\s*Numaras[ıi]",
    r"Vergi\s*Kimlik\s*No",
]
TCKN_LABELS = [r"TCKN", r"TC\s*Kimlik\s*No", r"T\.?C\.?\s*Kimlik"]
VKN_TCKN_LABELS = [r"VKN\s*/\s*TCKN", r"VKN\s*/\s*TC"]


def normalize_vl_markdown(raw: str) -> str:
    """Strip VL layout tags / image stubs and flatten HTML tables to text."""
    if not raw:
        return ""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = html.unescape(text)

    # HTML tables → rows of cell text
    def _table_repl(m: re.Match[str]) -> str:
        table = m.group(0)
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table, flags=re.I | re.S)
        lines: list[str] = []
        for row in rows:
            cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=re.I | re.S)
            cleaned = []
            for c in cells:
                c = re.sub(r"<[^>]+>", " ", c)
                c = re.sub(r"\s+", " ", c).strip()
                if c:
                    cleaned.append(c)
            if cleaned:
                # key/value two-cell rows → "Key: Value"
                if len(cleaned) == 2 and not re.search(r"\d{1,3}[.,]\d{2}", cleaned[0]):
                    lines.append(f"{cleaned[0]}: {cleaned[1]}")
                else:
                    lines.append(" | ".join(cleaned))
        return "\n".join(lines)

    text = re.sub(r"<table\b[\s\S]*?</table>", _table_repl, text, flags=re.I)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)

    out_lines: list[str] = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if _LAYOUT_NOISE.match(s) or _IMG_LINE.match(s) or _PATH_LINE.match(s):
            continue
        if re.match(r"(?i)^imgs?/", s):
            continue
        # Drop duplicated single-token layout leftovers glued into lines
        s = re.sub(r"(?i)\b(?:paragraph_title|vision_footnote|header_image|figure_title)\b", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        if s:
            out_lines.append(s)

    # De-dupe consecutive identical lines (VL often repeats blocks)
    deduped: list[str] = []
    for ln in out_lines:
        if deduped and deduped[-1] == ln:
            continue
        deduped.append(ln)
    return "\n".join(deduped)


def _label_alt(labels: list[str]) -> str:
    return "(?:" + "|".join(labels) + ")"


def _find_amounts_near_label(text: str, labels: list[str], window: int = 80) -> list[float]:
    from main import parse_tr_money

    alt = _label_alt(labels)
    amounts: list[float] = []
    for m in re.finditer(rf"(?is){alt}\s*:?\s*([^\n]{{0,{window}}})", text):
        chunk = m.group(1)
        # Prefer last money-like token in the window
        for tok in re.findall(
            r"(?<![\d.,])(\d{1,3}(?:[.\s]\d{3})*[.,]\d{2}|\d+[.,]\d{2}|\d{4,})(?![\d])",
            chunk,
        ):
            val = parse_tr_money(tok)
            if val is not None and val >= 0:
                amounts.append(val)
    return amounts


def _best_amount(text: str, labels: list[str]) -> float | None:
    vals = _find_amounts_near_label(text, labels)
    if not vals:
        return None
    # Prefer the largest plausible total for payable/genel toplam style labels
    return max(vals)


def _first_labeled_value(text: str, labels: list[str], value_re: str) -> str | None:
    alt = _label_alt(labels)
    m = re.search(rf"(?is){alt}\s*:?\s*[|]?\s*({value_re})", text)
    return m.group(1).strip() if m else None


def _section_split(text: str) -> tuple[str, str, str]:
    """Rough sections: head (supplier), mid (customer after SAYIN), tail (totals)."""
    sayin = re.search(r"(?im)\bSAYIN\b", text)
    if sayin:
        head = text[: sayin.start()]
        rest = text[sayin.start() :]
    else:
        # Without SAYIN: first third supplier-ish, rest customer+totals
        cut = max(len(text) // 3, 1)
        head, rest = text[:cut], text[cut:]

    # Totals usually in last part after line table keywords
    totals_m = re.search(
        r"(?im)(?:Mal\s*/?\s*Hizmet\s*Toplam|Net\s*Toplam|Ara\s*Toplam|"
        r"Hesaplanan\s*KDV|Ödenecek|Odenecek|Genel\s*Toplam|Vergiler\s*Dahil)",
        rest,
    )
    if totals_m:
        mid = rest[: totals_m.start()]
        tail = rest[totals_m.start() :]
    else:
        mid = rest
        # last 35% as totals fallback
        tcut = int(len(rest) * 0.65)
        mid, tail = rest[:tcut], rest[tcut:]
        if not tail.strip():
            tail = rest
    return head, mid, tail


def _party_name_from_block(block: str) -> str | None:
    junk = re.compile(
        r"(?i)^(?:Tel|Fax|Web|E-?Posta|Vergi|VKN|TCKN|Adres|SAYIN|e-Ar[sş]iv|"
        r"Fatura|ETTN|Senaryo|Özelleştirme|Page\s*\d|Merkez|Sicil|Mersis|"
        r"paragraph_title|table|image|text|e-Belge|V\.?\s*D\.?|ERP\s*Fatura|"
        r"WEB|Nihai\s*T[uü]ketici)\b"
    )
    companyish = re.compile(
        r"(?i)(?:LTD|A\.?\s*Ş|AŞ|SAN\.|T[İI]C\.|Ş[İI]RKET|ANON[İI]M|"
        r"ELEKT|ZÜCC|İTH|İHR|MARKET|TEKNOLOJ|İLET[İI]Ş[İI]M|DAY\.|MAM[UÜ]LL)"
    )
    candidates: list[str] = []
    for ln in block.splitlines()[:16]:
        s = ln.strip(" :|-")
        if len(s) < 5 or junk.match(s):
            continue
        if "|" in s and not companyish.search(s):
            continue
        if re.fullmatch(r"[\d\s./:+-]+", s):
            continue
        if re.search(r"https?://|@|\.com\b", s, re.I):
            continue
        letters = re.sub(r"[^A-Za-zÇĞİÖŞÜçğıöşü]", "", s)
        if len(letters) < 4:
            continue
        candidates.append(s)
    if not candidates:
        return None
    for c in candidates:
        if companyish.search(c):
            return c[:160]
    return candidates[0][:160]


def _person_name_from_block(block: str) -> str | None:
    junk = re.compile(
        r"(?i)^(?:Tel|Fax|Web|E-?Posta|Vergi|VKN|TCKN|Adres|SAYIN|Kap[ıi]|"
        r"Mah\.|Cad\.|Sok\.|No:|Türkiye|Web\s*Sitesi|paragraph_title|"
        r"Ürün\s*Kodu|Miktar|Birim|KDV|table|text|image|ERP\s*Fatura|"
        r"Nihai\s*T[uü]ketici|e-Belge)\b"
    )
    for ln in block.splitlines()[:15]:
        s = re.sub(r"(?i)\b(?:paragraph_title|text|image|table)\b", " ", ln)
        s = re.sub(r"\s+", " ", s).strip(" :|-")
        if len(s) < 5 or junk.match(s):
            continue
        if re.search(r"\d{6,}", s):
            continue
        if re.search(r"(?i)LTD|A\.?\s*Ş|SAN\.|T[İI]C\.", s):
            continue
        words = s.split()
        if 2 <= len(words) <= 5 and all(re.match(r"(?i)^[A-ZÇĞİÖŞÜa-zçğıöşü'.-]+$", w) for w in words):
            return s[:120]
        if 1 <= len(words) <= 4 and re.match(r"(?i)^[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü'. -]+$", s):
            letters = re.sub(r"[^A-Za-zÇĞİÖŞÜçğıöşü]", "", s)
            if len(letters) >= 6:
                return s[:120]
    return None


def _extract_tax_ids(block: str) -> tuple[str | None, str | None, str | None]:
    """Return (taxId, scheme, taxOffice) from a text block."""
    from tax_id import is_valid_tax_id

    vkn_tckn = _first_labeled_value(block, VKN_TCKN_LABELS, r"\d{10,11}")
    tckn = _first_labeled_value(block, TCKN_LABELS, r"\d{11}")
    vkn = _first_labeled_value(block, VKN_LABELS, r"\d{10}")
    office = _first_labeled_value(block, [r"Vergi\s*Dairesi", r"V\.?\s*D\.?"], r"[^\n|:]+")

    tax_id = None
    scheme = None
    if tckn and is_valid_tax_id(tckn, "TCKN"):
        tax_id, scheme = tckn, "TCKN"
    elif vkn_tckn:
        tid = re.sub(r"\D", "", vkn_tckn)
        if len(tid) == 11 and is_valid_tax_id(tid, "TCKN"):
            tax_id, scheme = tid, "TCKN"
        elif len(tid) == 10 and is_valid_tax_id(tid, "VKN"):
            tax_id, scheme = tid, "VKN"
    elif vkn and is_valid_tax_id(vkn, "VKN"):
        tax_id, scheme = vkn, "VKN"

    if office:
        office = re.split(r"\s{2,}|Tel\b|VKN\b|TCKN\b", office, maxsplit=1)[0].strip(" :")[:80] or None
    return tax_id, scheme, office


def _parse_line_rows(text: str) -> list:
    from main import Line, parse_tr_money

    lines: list = []
    # Pipe / HTML-flattened rows with qty + money
    for ln in text.splitlines():
        if not re.search(r"\d+[.,]\d{2}", ln):
            continue
        if re.search(
            r"(?i)Mal\s*Hizmet\s*Toplam|Ödenecek|Odenecek|Hesaplanan\s*KDV|"
            r"Vergiler\s*Dahil|Genel\s*Toplam|Ara\s*Toplam|Net\s*Toplam|"
            r"Birim\s*Fiyat|Miktar\s*\|\s*Birim",
            ln,
        ):
            # skip header / totals rows
            if re.search(r"(?i)^(?:Sıra|Mal\s*Hizmet|Ürün\s*Kodu|Açıklama)\b", ln):
                continue
            if re.search(
                r"(?i)Mal\s*Hizmet\s*Toplam|Ödenecek|Hesaplanan\s*KDV|Vergiler\s*Dahil|Genel\s*Toplam",
                ln,
            ):
                continue
        parts = [p.strip() for p in re.split(r"\s*\|\s*", ln) if p.strip()]
        money_vals = []
        for p in parts:
            v = parse_tr_money(p)
            if v is not None:
                money_vals.append(v)
        if not money_vals:
            continue
        # name = first non-numeric looking cell
        name = None
        for p in parts:
            if parse_tr_money(p) is not None:
                continue
            if re.fullmatch(r"\d{1,4}", p):
                continue
            if len(re.sub(r"[^A-Za-zÇĞİÖŞÜçğıöşü]", "", p)) >= 3:
                name = p
                break
        if not name:
            continue
        if re.search(r"(?i)^KDV\s*\(|Hesaplanan\s*KDV|İskonto|Iskonto|Toplam\b", name):
            continue
        line_total = money_vals[-1]
        unit_price = money_vals[-2] if len(money_vals) >= 2 else None
        qty = None
        for p in parts:
            if re.fullmatch(r"\d{1,4}(?:[.,]\d+)?", p):
                try:
                    qty = float(p.replace(",", "."))
                    if qty > 0:
                        break
                except ValueError:
                    pass
        lines.append(
            Line(
                name=name[:200],
                quantity=qty,
                unitPrice=unit_price,
                lineTotal=line_total,
            )
        )
        if len(lines) >= 40:
            break
    # De-dupe exact duplicates (VL often repeats table blocks)
    uniq: list = []
    seen: set[tuple] = set()
    for l in lines:
        key = (l.name, l.quantity, l.unitPrice, l.lineTotal)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(l)
    return uniq


def _reconcile_totals(inv) -> None:
    from main import nearly_equal

    t = inv.totals
    le, vat, pay, ti = t.lineExtensionAmount, t.vatAmount, t.payableAmount, t.taxInclusiveAmount
    wh = t.withholdingVatAmount
    # Fill taxInclusive from payable when missing
    if ti is None and pay is not None and wh is None:
        t.taxInclusiveAmount = pay
        ti = pay
    # If payable missing but taxInclusive present
    if pay is None and ti is not None and (wh is None or wh == 0):
        t.payableAmount = ti
        pay = ti
    # Prefer payable ≈ le+vat when payable looks like a unit price (too small)
    if le is not None and vat is not None:
        expected = round(le + vat, 2)
        if pay is None or (pay < expected * 0.5 and expected >= 50):
            t.payableAmount = expected
            if t.taxInclusiveAmount is None:
                t.taxInclusiveAmount = expected
        elif pay is not None and expected >= 50 and abs(pay - expected) / max(expected, 1) < 0.02:
            pass
        elif ti is None:
            t.taxInclusiveAmount = expected


def parse_vl_markdown(text: str, file_name: str = ""):
    """Parse PaddleOCR-VL markdown into Invoice without supplier templates."""
    from main import Invoice, merge_invoice, parse_text_invoice
    from tax_id import is_valid_tax_id

    cleaned = normalize_vl_markdown(text)
    # Baseline classic parser on cleaned text (not raw VL noise)
    base = parse_text_invoice(cleaned, file_name)
    inv = Invoice()
    inv = merge_invoice(inv, base)

    head, mid, tail = _section_split(cleaned)
    totals_zone = tail if tail.strip() else cleaned

    # --- Identity ---
    inv_no = _first_labeled_value(
        cleaned,
        INVOICE_NO_LABELS,
        r"[A-Z]{2,5}\d{10,16}|[A-Z]{2,5}\s*\d{10,16}",
    )
    if inv_no:
        inv.invoiceNumber = re.sub(r"\s+", "", inv_no).upper()
    # Reject Mersis-looking 14+ digit "invoice" if a better GİB serial exists elsewhere
    if inv.invoiceNumber and re.fullmatch(r"\d{12,}", inv.invoiceNumber):
        gib = re.search(r"\b([A-Z]{2,5}\d{13,16})\b", cleaned)
        if gib:
            inv.invoiceNumber = gib.group(1)

    ettn = _first_labeled_value(
        cleaned,
        ETTN_LABELS,
        r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}",
    )
    if ettn:
        inv.uuid = ettn.lower()

    # --- Totals via synonyms ---
    payable = _best_amount(totals_zone, PAYABLE_LABELS) or _best_amount(cleaned, PAYABLE_LABELS)
    line_ext = _best_amount(totals_zone, LINE_EXT_LABELS) or _best_amount(cleaned, LINE_EXT_LABELS)
    vat = _best_amount(totals_zone, VAT_LABELS) or _best_amount(cleaned, VAT_LABELS)
    discount = _best_amount(totals_zone, DISCOUNT_LABELS)
    withhold = _best_amount(totals_zone, WITHHOLD_LABELS)

    if payable is not None:
        inv.totals.payableAmount = payable
    if line_ext is not None:
        inv.totals.lineExtensionAmount = line_ext
    if vat is not None:
        inv.totals.vatAmount = vat
    if discount is not None:
        inv.totals.discountTotal = discount
    if withhold is not None:
        inv.totals.withholdingVatAmount = withhold

    # "Vergiler Dahil" synonym often equals payable when no withhold
    tax_incl = _best_amount(
        totals_zone,
        [r"Vergiler\s*Dahil\s*Toplam\s*Tutar", r"Vergiler\s*Dahil\s*Toplam"],
    )
    if tax_incl is not None:
        inv.totals.taxInclusiveAmount = tax_incl
        if inv.totals.payableAmount is None:
            inv.totals.payableAmount = tax_incl

    _reconcile_totals(inv)

    # --- Parties ---
    s_tax, s_scheme, s_office = _extract_tax_ids(head)
    c_tax, c_scheme, c_office = _extract_tax_ids(mid if mid.strip() else cleaned)

    # Fallback: any TCKN/VKN in full text assigned by SAYIN side
    if not s_tax:
        s_tax, s_scheme, s_office = _extract_tax_ids(cleaned[: max(len(cleaned) // 2, 1)])
    if not c_tax:
        # customer tax often after SAYIN
        c_tax, c_scheme, c_office = _extract_tax_ids(mid or cleaned)

    s_name = _party_name_from_block(head) or inv.supplier.name
    # Don't keep path/noise names
    if s_name and (_PATH_LINE.match(s_name) or s_name.lower() in {"table", "image", "text"}):
        s_name = _party_name_from_block(head)

    c_name = _person_name_from_block(mid) or inv.customer.name
    if c_name and (
        _PATH_LINE.match(c_name)
        or c_name.lower() in {"table", "image", "text"}
        or "Ürün Kodu" in c_name
        or "<td>" in (c_name or "")
        or re.match(r"(?i)^(?:ERP\s*Fatura|e-Belge|Nihai\s*T)", c_name or "")
    ):
        c_name = _person_name_from_block(mid)

    if s_name and (
        _PATH_LINE.match(s_name)
        or s_name.startswith("/")
        or re.match(r"(?i)^(?:e-Belge|V\.?\s*D\.?|ERP\s*Fatura|WEB)\b", s_name)
        or "|" in s_name
    ):
        s_name = _party_name_from_block(head)

    if s_name and not (_PATH_LINE.match(s_name) or s_name.startswith("/")):
        inv.supplier.name = s_name
    if s_tax:
        inv.supplier.taxId = s_tax
        inv.supplier.taxIdScheme = s_scheme  # type: ignore[assignment]
    if s_office:
        inv.supplier.taxOffice = s_office

    if c_name and not (_PATH_LINE.match(c_name) or c_name.startswith("/")):
        inv.customer.name = c_name
    if c_tax:
        inv.customer.taxId = c_tax
        inv.customer.taxIdScheme = c_scheme  # type: ignore[assignment]
    elif not inv.customer.taxId:
        # Unlabeled 11-digit near person name (Gürkan-style)
        near = mid[:1200] if mid else cleaned
        for m in re.finditer(r"\b(\d{11})\b", near):
            tid = m.group(1)
            if is_valid_tax_id(tid, "TCKN") and tid != (inv.supplier.taxId or ""):
                inv.customer.taxId = tid
                inv.customer.taxIdScheme = "TCKN"
                break
        if not inv.customer.taxId:
            for m in re.finditer(r"\b(\d{11})\b", cleaned):
                tid = m.group(1)
                if is_valid_tax_id(tid, "TCKN") and tid != (inv.supplier.taxId or ""):
                    inv.customer.taxId = tid
                    inv.customer.taxIdScheme = "TCKN"
                    break
    if c_office:
        inv.customer.taxOffice = c_office

    # If supplier tax equals customer TCKN and a VKN exists in head, prefer VKN for supplier
    if (
        inv.supplier.taxId
        and inv.customer.taxId
        and inv.supplier.taxId == inv.customer.taxId
        and inv.customer.taxIdScheme == "TCKN"
    ):
        head_vkn = _first_labeled_value(head, VKN_LABELS, r"\d{10}")
        if head_vkn and is_valid_tax_id(head_vkn, "VKN"):
            inv.supplier.taxId = head_vkn
            inv.supplier.taxIdScheme = "VKN"

    # --- Lines ---
    vl_lines = _parse_line_rows(cleaned)
    if vl_lines and (not inv.lines or len(vl_lines) >= len(inv.lines)):
        inv.lines = vl_lines

    # Document type hints
    if re.search(r"(?i)e-?Ar[sş]iv", cleaned):
        inv.documentType = "earsiv"
    elif re.search(r"(?i)e-?Fatura", cleaned):
        inv.documentType = "efatura"

    return inv


def invoice_from_ocr_text(text: str, file_name: str = "", engine: str | None = None):
    """OCR/raster text → generic binder (VL markdown or RapidOCR plain text)."""
    # Engine hint kept for pipeline tagging; binder is layout-agnostic.
    _ = engine
    return parse_vl_markdown(text, file_name)
