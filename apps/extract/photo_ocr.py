"""High-quality / fast photo OCR for invoices.

Stack (open-source, CPU):
  OpenCV preprocess (upscale + CLAHE) → RapidOCR PP-OCRv6 (ONNX Runtime)

RapidOCR ships PP-OCRv6 det/rec ONNX models — strong on Latin/Turkish docs and
much faster than Docling+Tesseract on phone photos.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PHOTO_OCR_ENABLED = os.getenv("PHOTO_OCR_ENABLED", "1") == "1"
PHOTO_OCR_MIN_SIDE = int(os.getenv("PHOTO_OCR_MIN_SIDE", "1800"))
PHOTO_OCR_TARGET_SIDE = int(os.getenv("PHOTO_OCR_TARGET_SIDE", "2600"))
PHOTO_OCR_MAX_SIDE = int(os.getenv("PHOTO_OCR_MAX_SIDE", "3600"))
PHOTO_OCR_CLAHE = os.getenv("PHOTO_OCR_CLAHE", "1") == "1"

_engine = None
_engine_lock = threading.Lock()


def get_engine() -> Any:
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is not None:
            return _engine
        from rapidocr import RapidOCR

        _engine = RapidOCR()
        return _engine


def _load_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"image decode failed: {path}")
    # EXIF orientation via PIL when available
    try:
        from PIL import Image, ImageOps

        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            rgb = np.array(im.convert("RGB"))
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        pass
    return img


def preprocess_bgr(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side < PHOTO_OCR_MIN_SIDE:
        scale = PHOTO_OCR_TARGET_SIDE / long_side
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC,
        )
    elif long_side > PHOTO_OCR_MAX_SIDE:
        scale = PHOTO_OCR_MAX_SIDE / long_side
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    if PHOTO_OCR_CLAHE:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return img


def _box_key(box: np.ndarray) -> tuple[float, float]:
    # box: 4x2
    ys = box[:, 1]
    xs = box[:, 0]
    return float(ys.min()), float(xs.min())


def _group_lines(boxes: np.ndarray, txts: list[str], scores: list[float]) -> str:
    """Cluster OCR boxes into reading-order text lines."""
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
    thresh = max(12.0, median_h * 0.6)

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


def ocr_image(path: Path) -> tuple[str, dict[str, Any]]:
    """Return (text, meta). meta includes engine, elapsed, lineCount."""
    if not PHOTO_OCR_ENABLED:
        return "", {"engine": "disabled", "lineCount": 0, "elapsedMs": 0}

    import time

    started = time.perf_counter()
    img = preprocess_bgr(_load_bgr(path))
    engine = get_engine()
    result = engine(img)
    txts = list(getattr(result, "txts", None) or [])
    scores = list(getattr(result, "scores", None) or [1.0] * len(txts))
    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(txts) > 0:
        text = _group_lines(np.asarray(boxes), txts, scores)
    else:
        text = "\n".join(t for t in txts if t)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return text, {
        "engine": "rapidocr-ppocrv6",
        "lineCount": len(txts),
        "elapsedMs": elapsed_ms,
        "preprocessedShape": list(img.shape[:2]),
        "meanScore": float(np.mean(scores)) if scores else 0.0,
    }
