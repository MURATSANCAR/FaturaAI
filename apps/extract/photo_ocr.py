"""High-quality photo OCR for Turkish invoices / receipts.

Stack (open-source, CPU):
  OpenCV preprocess → RapidOCR PP-OCRv5 Latin (Turkish-capable) + PP-OCRv6 fallback
  → optional Tesseract tur+eng merge for stubborn thermal prints
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PHOTO_OCR_ENABLED = os.getenv("PHOTO_OCR_ENABLED", "1") == "1"
PHOTO_OCR_MIN_SIDE = int(os.getenv("PHOTO_OCR_MIN_SIDE", "2000"))
PHOTO_OCR_TARGET_SIDE = int(os.getenv("PHOTO_OCR_TARGET_SIDE", "3200"))
PHOTO_OCR_MAX_SIDE = int(os.getenv("PHOTO_OCR_MAX_SIDE", "4200"))
PHOTO_OCR_CLAHE = os.getenv("PHOTO_OCR_CLAHE", "1") == "1"
PHOTO_OCR_DUAL = os.getenv("PHOTO_OCR_DUAL", "1") == "1"
PHOTO_OCR_TESSERACT = os.getenv("PHOTO_OCR_TESSERACT", "1") == "1"

_engine_latin = None
_engine_v6 = None
_engine_lock = threading.Lock()


def _make_latin_engine():
    from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR

    return RapidOCR(
        params={
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Det.lang_type": LangDet.CH,
            "Det.model_type": ModelType.MOBILE,
            "Det.ocr_version": OCRVersion.PPOCRV5,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.lang_type": LangRec.LATIN,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Cls.engine_type": EngineType.ONNXRUNTIME,
        }
    )


def _make_v6_engine():
    from rapidocr import RapidOCR

    return RapidOCR()


def get_engines() -> tuple[Any, Any | None]:
    global _engine_latin, _engine_v6
    with _engine_lock:
        if _engine_latin is None:
            _engine_latin = _make_latin_engine()
        if PHOTO_OCR_DUAL and _engine_v6 is None:
            try:
                _engine_v6 = _make_v6_engine()
            except Exception:
                _engine_v6 = None
        return _engine_latin, _engine_v6


def _load_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"image decode failed: {path}")
    try:
        from PIL import Image, ImageOps

        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            rgb = np.array(im.convert("RGB"))
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        pass
    return img


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Light deskew using minAreaRect on ink pixels."""
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thr > 0))
    if len(coords) < 500:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.3 or abs(angle) > 15:
        return gray
    h, w = gray.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        gray, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def preprocess_bgr(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side < PHOTO_OCR_MIN_SIDE:
        scale = PHOTO_OCR_TARGET_SIDE / long_side
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    elif long_side > PHOTO_OCR_MAX_SIDE:
        scale = PHOTO_OCR_MAX_SIDE / long_side
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    if PHOTO_OCR_CLAHE:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(l)
        img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # Mild denoise + unsharp for thermal / phone photos
    img = cv2.bilateralFilter(img, 5, 40, 40)
    blur = cv2.GaussianBlur(img, (0, 0), 1.0)
    img = cv2.addWeighted(img, 1.25, blur, -0.25, 0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = _deskew(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _group_lines(boxes: np.ndarray, txts: list[str], scores: list[float]) -> str:
    items: list[tuple[float, float, float, str, float]] = []
    for box, txt, score in zip(boxes, txts, scores, strict=False):
        arr = np.asarray(box, dtype=np.float32).reshape(-1, 2)
        y0, x0 = float(arr[:, 1].min()), float(arr[:, 0].min())
        y1 = float(arr[:, 1].max())
        items.append((y0, x0, y1 - y0, str(txt).strip(), float(score)))
    items = [it for it in items if it[3]]
    if not items:
        return ""
    items.sort(key=lambda t: (t[0], t[1]))
    heights = [it[2] for it in items if it[2] > 0]
    median_h = float(np.median(heights)) if heights else 20.0
    thresh = max(12.0, median_h * 0.55)

    lines: list[list[tuple[float, str]]] = []
    current: list[tuple[float, str]] = []
    current_y: float | None = None
    for y0, x0, _h, txt, _score in items:
        if current_y is None or abs(y0 - current_y) <= thresh:
            current.append((x0, txt))
            current_y = y0 if current_y is None else (current_y * 0.7 + y0 * 0.3)
        else:
            lines.append(current)
            current = [(x0, txt)]
            current_y = y0
    if current:
        lines.append(current)

    out_lines: list[str] = []
    for line in lines:
        line.sort(key=lambda t: t[0])
        out_lines.append(" ".join(t for _, t in line))
    return "\n".join(out_lines)


def _result_to_text(result: Any) -> tuple[str, float, int]:
    txts = list(getattr(result, "txts", None) or [])
    scores = list(getattr(result, "scores", None) or [1.0] * len(txts))
    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(txts) > 0:
        text = _group_lines(np.asarray(boxes), txts, scores)
    else:
        text = "\n".join(t for t in txts if t)
    mean = float(np.mean(scores)) if scores else 0.0
    return text, mean, len(txts)


def _turkish_richness(text: str) -> int:
    return sum(1 for ch in text if ch in "çğıöşüÇĞİÖŞÜâîûÂÎÛ")


def _tesseract_tur(path: Path) -> str:
    import subprocess
    import tempfile

    img = preprocess_bgr(_load_bgr(path))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        cv2.imwrite(tmp.name, img)
        tmp_path = tmp.name
    try:
        r = subprocess.run(
            ["tesseract", tmp_path, "stdout", "-l", "tur+eng", "--psm", "6"],
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        return r.stdout or ""
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def ocr_image(path: Path) -> tuple[str, dict[str, Any]]:
    """Return (text, meta)."""
    if not PHOTO_OCR_ENABLED:
        return "", {"engine": "disabled", "lineCount": 0, "elapsedMs": 0}

    started = time.perf_counter()
    img = preprocess_bgr(_load_bgr(path))
    latin, v6 = get_engines()

    candidates: list[tuple[str, str, float, int]] = []
    res_l = latin(img)
    text_l, mean_l, n_l = _result_to_text(res_l)
    candidates.append(("latin-ppocrv5", text_l, mean_l, n_l))

    if v6 is not None:
        try:
            res_v = v6(img)
            text_v, mean_v, n_v = _result_to_text(res_v)
            candidates.append(("ppocrv6", text_v, mean_v, n_v))
        except Exception:
            pass

    if PHOTO_OCR_TESSERACT:
        try:
            tess = _tesseract_tur(path)
            if tess.strip():
                candidates.append(("tesseract-tur", tess, 0.7, tess.count("\n") + 1))
        except Exception:
            pass

    # Rank: turkish richness, then mean score, then line count
    best = max(
        candidates,
        key=lambda c: (_turkish_richness(c[1]) * 2 + c[2] * 100 + min(c[3], 120) * 0.05),
    )
    # Prefer primary text for parsing (alts only in meta — avoids duplicate lines)
    text = best[1]
    # If primary misses Turkish product names but another engine has richer TR chars,
    # merge missing high-value lines (name-like) from the richest alternate.
    richest_alt = max(candidates, key=lambda c: _turkish_richness(c[1]))
    if richest_alt[0] != best[0] and _turkish_richness(richest_alt[1]) > _turkish_richness(best[1]) + 2:
        text = richest_alt[1]
        best = richest_alt

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return text, {
        "engine": best[0],
        "engines": [c[0] for c in candidates],
        "lineCount": best[3],
        "elapsedMs": elapsed_ms,
        "preprocessedShape": list(img.shape[:2]),
        "meanScore": best[2],
        "turkishChars": _turkish_richness(best[1]),
        "altLineCounts": {c[0]: c[3] for c in candidates},
    }
