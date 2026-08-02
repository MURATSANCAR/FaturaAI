"""pdf-inspector (Firecrawl) — dijital PDF sınıflandırma + metin/markdown çıkarma.

pdftotext alternatifi / öncesi: CID fontlu GİB PDF'lerde extract_text boş kalabiliyor;
o durumda markdown flatten edilir. OCR yapmaz — scanned için caller OCR'a düşer.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


PDF_INSPECTOR_ENABLED = os.getenv("PDF_INSPECTOR_ENABLED", "1") == "1"
PDF_INSPECTOR_MIN_CONF = float(os.getenv("PDF_INSPECTOR_MIN_CONF", "0.7"))


def _flatten_markdown(md: str) -> str:
    text = md.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?m)^\s*\|?\s*[-:]+(\s*\|\s*[-:]+)*\s*\|?\s*$", "", text)
    text = text.replace("|", " ")
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"`+", "", text)
    text = re.sub(r"(?m)^#{1,6}\s*", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def available() -> bool:
    if not PDF_INSPECTOR_ENABLED:
        return False
    try:
        import pdf_inspector  # noqa: F401

        return True
    except Exception:
        return False


def extract_pdf_inspector(path: Path | str) -> tuple[str, dict[str, Any]]:
    """Return (text, meta). Empty text when unusable / scanned / disabled."""
    meta: dict[str, Any] = {
        "engine": "pdf-inspector",
        "enabled": PDF_INSPECTOR_ENABLED,
        "available": False,
    }
    if not PDF_INSPECTOR_ENABLED:
        return "", meta

    try:
        import pdf_inspector
    except Exception as exc:  # noqa: BLE001
        meta["error"] = f"import:{exc}"
        return "", meta

    meta["available"] = True
    path_str = str(path)
    try:
        result = pdf_inspector.process_pdf(path_str)
    except Exception as exc:  # noqa: BLE001
        meta["error"] = f"process:{exc}"
        return "", meta

    pdf_type = getattr(result, "pdf_type", None) or ""
    confidence = float(getattr(result, "confidence", 0.0) or 0.0)
    pages_needing_ocr = list(getattr(result, "pages_needing_ocr", None) or [])
    has_encoding_issues = bool(getattr(result, "has_encoding_issues", False))
    processing_ms = getattr(result, "processing_time_ms", None)

    meta.update(
        {
            "pdfType": pdf_type,
            "confidence": confidence,
            "pagesNeedingOcr": pages_needing_ocr,
            "hasEncodingIssues": has_encoding_issues,
            "processingTimeMs": processing_ms,
            "pageCount": getattr(result, "page_count", None),
        }
    )

    if pdf_type in ("scanned", "image_based"):
        meta["source"] = "skipped-non-text"
        return "", meta

    if confidence < PDF_INSPECTOR_MIN_CONF and pages_needing_ocr:
        meta["source"] = "skipped-low-conf"
        return "", meta

    plain = ""
    try:
        plain = pdf_inspector.extract_text(path_str) or ""
    except Exception as exc:  # noqa: BLE001
        meta["extractTextError"] = str(exc)

    md = getattr(result, "markdown", None) or ""
    if len(plain.strip()) >= 40:
        meta["source"] = "extract_text"
        meta["charCount"] = len(plain)
        return plain, meta

    if len(md.strip()) >= 40:
        flat = _flatten_markdown(md)
        meta["source"] = "markdown"
        meta["charCount"] = len(flat)
        meta["markdownCharCount"] = len(md)
        return flat, meta

    meta["source"] = "empty"
    return "", meta
