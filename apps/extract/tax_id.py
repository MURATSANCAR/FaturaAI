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


def coerce_tax_id(value: str | None) -> tuple[str, str] | None:
    """Normalize OCR tax id → (digits, scheme) if length looks like VKN/TCKN."""
    n = normalize_ocr_digits(value)
    if not n or is_placeholder_tax_id(n):
        return None
    if len(n) == 11:
        return n, "TCKN"
    if len(n) == 10:
        return n, "VKN"
    return None
