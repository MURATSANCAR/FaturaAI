"""Turkish TCKN / VKN checksum helpers."""

from __future__ import annotations

import re

# GİB e-Arşiv anonymous / final-consumer placeholder (checksum fails by design).
PLACEHOLDER_TCKN = "11111111111"

# OCR lookalikes inside numeric tax-id / date tokens
_OCR_DIGIT_TRANS = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "О": "0",  # Cyrillic
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


def digits_only(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\D", "", value)


def normalize_ocr_digits(value: str | None) -> str:
    """Map common OCR letter→digit confusions, then keep digits only."""
    if not value:
        return ""
    return digits_only(value.translate(_OCR_DIGIT_TRANS))


def is_placeholder_tckn(value: str | None) -> bool:
    return digits_only(value) == PLACEHOLDER_TCKN


def is_placeholder_tax_id(value: str | None) -> bool:
    """Reject GİB anonymous / all-same-digit / zero tax ids (not real parties)."""
    n = digits_only(value) or normalize_ocr_digits(value)
    if not n:
        return False
    if n in {PLACEHOLDER_TCKN, "0000000000", "00000000000"}:
        return True
    # 111…, 222…, 000… etc.
    if len(n) in (10, 11) and len(set(n)) == 1:
        return True
    return False


def is_valid_tckn(value: str | None) -> bool:
    n = normalize_ocr_digits(value) or digits_only(value)
    # Placeholder is known, but not a real identity — treat as invalid for binding.
    if is_placeholder_tax_id(n):
        return False
    if len(n) != 11 or not n.isdigit() or n[0] == "0":
        return False
    d = [int(c) for c in n]
    odd = d[0] + d[2] + d[4] + d[6] + d[8]
    even = d[1] + d[3] + d[5] + d[7]
    d10 = (odd * 7 - even) % 10
    d11 = sum(d[:10]) % 10
    return d[9] == d10 and d[10] == d11


def is_valid_vkn(value: str | None) -> bool:
    n = normalize_ocr_digits(value) or digits_only(value)
    if is_placeholder_tax_id(n):
        return False
    if len(n) != 10 or not n.isdigit():
        return False
    total = 0
    for i in range(9):
        tmp = (int(n[i]) + (9 - i)) % 10
        powered = (tmp * (2 ** (9 - i))) % 9
        if tmp != 0 and powered == 0:
            powered = 9
        total += powered
    check = (10 - (total % 10)) % 10
    return check == int(n[9])


def is_valid_tax_id(value: str | None, scheme: str | None = None) -> bool:
    n = normalize_ocr_digits(value) or digits_only(value)
    if not n or is_placeholder_tax_id(n):
        return False
    scheme_u = (scheme or "").upper()
    if scheme_u == "TCKN" or (not scheme_u and len(n) == 11):
        return is_valid_tckn(n)
    if scheme_u == "VKN" or (not scheme_u and len(n) == 10):
        return is_valid_vkn(n)
    if len(n) == 11:
        return is_valid_tckn(n)
    if len(n) == 10:
        return is_valid_vkn(n)
    return False


# Single-digit OCR confusions commonly seen on thermal / phone photos.
_OCR_DIGIT_ALTS: dict[str, tuple[str, ...]] = {
    "0": ("8",),
    "8": ("0",),
    "3": ("9",),
    "9": ("3", "4"),
    "1": ("7",),
    "7": ("1", "2"),
    "5": ("6",),
    "6": ("5",),
    "4": ("9",),
    "2": ("7",),
}


def repair_tax_id(value: str | None, scheme: str | None = None) -> tuple[str, str] | None:
    """If checksum fails, try single-digit OCR repairs; return unique valid candidate.

    Does not rewrite an already-valid id (checksum alone cannot recover a wrong
    but checksum-valid OCR read).
    """
    n = normalize_ocr_digits(value) or digits_only(value)
    if not n or is_placeholder_tax_id(n):
        return None
    scheme_u = (scheme or "").upper()
    if not scheme_u:
        scheme_u = "TCKN" if len(n) == 11 else "VKN" if len(n) == 10 else ""
    if scheme_u not in {"VKN", "TCKN"}:
        return None
    if is_valid_tax_id(n, scheme_u):
        return n, scheme_u

    found: list[str] = []
    for i, ch in enumerate(n):
        for alt in _OCR_DIGIT_ALTS.get(ch, ()):
            cand = n[:i] + alt + n[i + 1 :]
            if is_placeholder_tax_id(cand):
                continue
            if is_valid_tax_id(cand, scheme_u) and cand not in found:
                found.append(cand)
    if len(found) == 1:
        return found[0], scheme_u
    return None


def coerce_tax_id(value: str | None) -> tuple[str, str] | None:
    """Normalize OCR tax id → (digits, scheme) if length looks like VKN/TCKN.

    Prefer checksum-valid values; auto-repair single-digit OCR mistakes when the
    raw candidate fails GİB checksum and exactly one repair validates.
    """
    n = normalize_ocr_digits(value)
    if not n or is_placeholder_tax_id(n):
        return None
    if len(n) == 11:
        scheme = "TCKN"
    elif len(n) == 10:
        scheme = "VKN"
    else:
        return None
    if is_valid_tax_id(n, scheme):
        return n, scheme
    repaired = repair_tax_id(n, scheme)
    if repaired:
        return repaired
    # Length-only fallback (caller / sanitize may still drop on checksum)
    return n, scheme
