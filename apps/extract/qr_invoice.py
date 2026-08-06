"""GİB e-Arşiv / e-Fatura QR code reader.

Every GİB e-Arşiv/e-Fatura page carries a QR whose payload is a (often slightly
malformed) JSON with authoritative fields: seller/buyer tax id, invoice no,
date, ETTN, matrah, KDV, and the tax-inclusive / payable totals. Decoding it
gives 100%-accurate ground truth for the key fields, bypassing OCR/parse guesses.

Tolerant on purpose: some ERPs emit invalid JSON (stray quotes/commas, trailing
spaces, multi-rate `kdvmatrah(20.00)` keys), so we regex out key:value pairs
rather than json.loads.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

QR_ENABLED = os.getenv("QR_ENABLED", "1") == "1"
QR_DPI = max(150, int(os.getenv("QR_DPI", "300")))

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


def _decode_symbols(gray: Any) -> list[str]:
    """Return decoded QR text payloads from a grayscale image (pyzbar)."""
    try:
        import cv2
        from pyzbar.pyzbar import ZBarSymbol, decode
    except Exception:
        return []

    out: list[str] = []
    attempts = [gray]
    try:
        attempts.append(cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC))
    except Exception:
        pass
    for im in attempts:
        try:
            res = decode(im, symbols=[ZBarSymbol.QRCODE])
        except Exception:
            res = []
        for r in res:
            try:
                out.append(r.data.decode("utf-8", "replace"))
            except Exception:
                continue
        if out:
            break
    return out


def _payloads_from_path(path: Path, max_pages: int = 2) -> list[str]:
    import cv2

    suffix = path.suffix.lower()
    payloads: list[str] = []
    if suffix in _IMAGE_EXTS:
        img = cv2.imread(str(path))
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            payloads.extend(_decode_symbols(gray))
        return payloads

    # PDF: render first pages to PNG and scan each.
    with tempfile.TemporaryDirectory(prefix="qr-") as tmp:
        prefix = Path(tmp) / "qr"
        subprocess.run(
            ["pdftoppm", "-png", "-r", str(QR_DPI), "-f", "1", "-l", str(max_pages),
             str(path), str(prefix)],
            capture_output=True, timeout=60, check=False,
        )
        for png in sorted(Path(tmp).glob("qr*.png")):
            img = cv2.imread(str(png))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            found = _decode_symbols(gray)
            if found:
                payloads.extend(found)
                break
    return payloads


def _num(raw: str | None) -> float | None:
    if raw is None:
        return None
    s = raw.strip().strip('"').strip()
    if not s:
        return None
    # GİB QR amounts use '.' decimal (e.g. "2477.85") but tolerate TR "1.234,56".
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    s = re.sub(r"[^0-9.\-]", "", s)
    try:
        return round(float(s), 2)
    except ValueError:
        return None


def _field(payload: str, key: str) -> str | None:
    # tolerant: "key" : "value"  or  "key": value  (value may have trailing junk)
    m = re.search(rf'"{re.escape(key)}"\s*:\s*"?([^",}}]+)', payload, re.I)
    if not m:
        return None
    return m.group(1).strip().strip('"').strip()


def _sum_multi(payload: str, base_key: str) -> float | None:
    """Sum multi-rate keys like hesaplanankdv(20.00) / hesaplanankdv(10)."""
    total = 0.0
    hit = False
    for m in re.finditer(rf'"{re.escape(base_key)}\([^)]*\)"\s*:\s*"?([0-9.,]+)', payload, re.I):
        v = _num(m.group(1))
        if v is not None:
            total += v
            hit = True
    return round(total, 2) if hit else None


def parse_qr_payload(payload: str) -> dict[str, Any]:
    """Map a GİB QR JSON-ish payload to normalized invoice fields."""
    out: dict[str, Any] = {}
    stckn = _field(payload, "vkntckn")
    actckn = _field(payload, "avkntckn")
    if stckn:
        out["supplierTaxId"] = re.sub(r"\D", "", stckn)
    if actckn:
        out["customerTaxId"] = re.sub(r"\D", "", actckn)
    no = _field(payload, "no")
    if no:
        out["invoiceNumber"] = no
    ettn = _field(payload, "ettn")
    if ettn and re.search(r"[0-9A-Fa-f]{8}-", ettn):
        out["uuid"] = ettn.lower()
    tarih = _field(payload, "tarih")
    if tarih:
        m = re.search(r"(\d{4})[-./](\d{2})[-./](\d{2})", tarih)
        if m:
            out["issueDate"] = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        else:
            m2 = re.search(r"(\d{2})[-./](\d{2})[-./](\d{4})", tarih)
            if m2:
                out["issueDate"] = f"{m2.group(3)}-{m2.group(2)}-{m2.group(1)}"
    senaryo = (_field(payload, "senaryo") or "").upper()
    if "EARSIV" in senaryo or "ARŞIV" in senaryo or "ARSIV" in senaryo:
        out["documentType"] = "earsiv"
    elif senaryo:
        out["documentType"] = "efatura"

    le = _num(_field(payload, "malhizmettoplam"))
    if le is not None:
        out["lineExtensionAmount"] = le
    vat = _num(_field(payload, "hesaplanankdv")) or _sum_multi(payload, "hesaplanankdv")
    if vat is not None:
        out["vatAmount"] = vat
    ti = _num(_field(payload, "vergidahil"))
    if ti is not None:
        out["taxInclusiveAmount"] = ti
    pay = _num(_field(payload, "odenecek"))
    if pay is not None:
        out["payableAmount"] = pay
    return out


def read_invoice_qr(path: Path, max_pages: int = 2) -> dict[str, Any] | None:
    """Decode the GİB QR and return normalized fields, or None if absent."""
    if not QR_ENABLED:
        return None
    try:
        payloads = _payloads_from_path(Path(path), max_pages=max_pages)
    except Exception:
        return None
    for p in payloads:
        # Only trust GİB invoice QRs (have vkntckn + no), skip URL/other QRs.
        if not re.search(r'"vkntckn"', p, re.I) or not re.search(r'"no"', p, re.I):
            continue
        fields = parse_qr_payload(p)
        if fields.get("invoiceNumber") or fields.get("supplierTaxId"):
            fields["_raw"] = p[:600]
            return fields
    return None
