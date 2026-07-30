"""High-quality photo OCR for Turkish invoices / receipts.

Stack (open-source, CPU):
  OpenCV preprocess → RapidOCR Latin PP-OCRv5 (primary) + optional PP-OCRv6
  → Tesseract tur+eng only as last-resort if RapidOCR structure is weak

Speed-first strategy (generic, not brand-specific):
  - One good pass + early exit when GİB structure is already readable
  - Extra engines / strong / binary / Tesseract only when structure is weak
  - Modest upscale (default 2000px) instead of always 2800–4200
"""

from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PHOTO_OCR_ENABLED = os.getenv("PHOTO_OCR_ENABLED", "1") == "1"
PHOTO_OCR_MIN_SIDE = int(os.getenv("PHOTO_OCR_MIN_SIDE", "1400"))
PHOTO_OCR_TARGET_SIDE = int(os.getenv("PHOTO_OCR_TARGET_SIDE", "2200"))
PHOTO_OCR_MAX_SIDE = int(os.getenv("PHOTO_OCR_MAX_SIDE", "2800"))
PHOTO_OCR_CLAHE = os.getenv("PHOTO_OCR_CLAHE", "1") == "1"
# Dual engine only as fallback when latin pass is weak (set 1 to always run both).
PHOTO_OCR_DUAL = os.getenv("PHOTO_OCR_DUAL", "0") == "1"
PHOTO_OCR_TESSERACT = os.getenv("PHOTO_OCR_TESSERACT", "1") == "1"
PHOTO_OCR_COLUMNS = os.getenv("PHOTO_OCR_COLUMNS", "1") == "1"
PHOTO_OCR_EARLY_STRUCT = int(os.getenv("PHOTO_OCR_EARLY_STRUCT", "8"))

_engine_latin = None
_engine_v6 = None
_engine_lock = threading.Lock()

_STRUCT_RE = re.compile(
    r"(?:"
    r"\bETTN\b|\bETIN\b|\bTOPLAM\b|\bTOPKDV\b|\bARA\s*TOPLAM\b|"
    r"\bBELGE\s*N[O0]\b|\bBE[LİI]?GE\s*N[O0]\b|"
    r"\bFATURA\s*N[O0]\b|\bFATERA\s*N[O0]\b|\bFATARA\s*N[AO0]\b|"
    r"\bVKN\b|\bKDV\b|\badet\s*[x×X]\b|\b\d+\s*Ade[t1l]?\b|"
    r"\bTAR[İI]H\b|\bMERS[İI]S\b|\b[ÖO]DENECEK\b|"
    r"\bSAY[İI]N\b|\bSAYDN\b|\bSAVIN\b|"
    r"\bSENARYO\b|\bAL[İI]C[İI]\b|\bSAT[İI]C[İI]\b|"
    r"\be-?Ar[sş]iv\b|\bB[İI]LG[İI]\s*F[İIİI]?[ŞS]\b|"
    r"\d{1,3}(?:[.\s]\d{3})*[.,]\d{2}"
    r")",
    re.I,
)

_GIB_HINT_RE = re.compile(
    r"(?:"
    r"\bETTN\b|"
    r"\b(?:FATURA|FATERA|FATARA|FATACA|PATARA)\s*N[AO0]\b|"
    r"\b[A-Z]{2,5}\d{10,16}\b|"
    r"\bVKN\b|"
    r"\b[ÖO]DEN|\bTOPLAM\b|"
    r"\d{1,3}(?:[.,\s]\d{3})+[.,]\d{2}"
    r")",
    re.I,
)


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


def get_engines(*, need_v6: bool = False) -> tuple[Any, Any | None]:
    global _engine_latin, _engine_v6
    with _engine_lock:
        if _engine_latin is None:
            _engine_latin = _make_latin_engine()
        if (need_v6 or PHOTO_OCR_DUAL) and _engine_v6 is None:
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
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thr > 0))
    if len(coords) < 500:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.3 or abs(angle) > 12:
        return gray
    h, w = gray.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        gray, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def _resize_long_side(img: np.ndarray, target: int) -> np.ndarray:
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side == target:
        return img
    scale = target / long_side
    # LANCZOS4 upscales UI screenshots sharper than CUBIC
    if scale > 1:
        interp = getattr(cv2, "INTER_LANCZOS4", cv2.INTER_CUBIC)
    else:
        interp = cv2.INTER_AREA
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=interp)


def preprocess_bgr(
    img: np.ndarray,
    *,
    strong: bool = False,
    deskew: bool = True,
    binary: bool = False,
    target_side: int | None = None,
) -> np.ndarray:
    h, w = img.shape[:2]
    long_side = max(h, w)
    if target_side is None:
        if long_side < 700:
            target_side = min(PHOTO_OCR_MAX_SIDE, 2600 if strong else 2400)
        elif strong:
            target_side = min(PHOTO_OCR_MAX_SIDE, max(PHOTO_OCR_TARGET_SIDE, 2200))
        else:
            target_side = PHOTO_OCR_TARGET_SIDE

    if long_side < PHOTO_OCR_MIN_SIDE or long_side < target_side * 0.9:
        img = _resize_long_side(img, target_side)
    elif long_side > PHOTO_OCR_MAX_SIDE:
        img = _resize_long_side(img, PHOTO_OCR_MAX_SIDE)

    if PHOTO_OCR_CLAHE and not binary:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clip = 2.5 if strong else 2.0
        l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(l)
        img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    if strong and not binary:
        img = cv2.bilateralFilter(img, 5, 35, 35)
        blur = cv2.GaussianBlur(img, (0, 0), 0.9)
        img = cv2.addWeighted(img, 1.35, blur, -0.35, 0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if deskew:
        gray = _deskew(gray)

    if binary:
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9
        )
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _items_from_boxes(
    boxes: np.ndarray, txts: list[str], scores: list[float]
) -> list[tuple[float, float, float, float, str, float]]:
    """(y0, x0, y1, x1, text, score)"""
    items: list[tuple[float, float, float, float, str, float]] = []
    for box, txt, score in zip(boxes, txts, scores, strict=False):
        arr = np.asarray(box, dtype=np.float32).reshape(-1, 2)
        y0, x0 = float(arr[:, 1].min()), float(arr[:, 0].min())
        y1, x1 = float(arr[:, 1].max()), float(arr[:, 0].max())
        t = str(txt).strip()
        if t:
            items.append((y0, x0, y1, x1, t, float(score)))
    return items


def _cluster_lines(
    items: list[tuple[float, float, float, float, str, float]], thresh: float
) -> list[list[tuple[float, float, float, float, str, float]]]:
    if not items:
        return []
    items = sorted(items, key=lambda t: (t[0], t[1]))
    lines: list[list[tuple[float, float, float, float, str, float]]] = []
    current: list[tuple[float, float, float, float, str, float]] = []
    current_y: float | None = None
    for it in items:
        y0 = it[0]
        if current_y is None or abs(y0 - current_y) <= thresh:
            current.append(it)
            current_y = y0 if current_y is None else (current_y * 0.7 + y0 * 0.3)
        else:
            lines.append(current)
            current = [it]
            current_y = y0
    if current:
        lines.append(current)
    return lines


def _group_lines(boxes: np.ndarray, txts: list[str], scores: list[float]) -> str:
    items = _items_from_boxes(boxes, txts, scores)
    if not items:
        return ""
    heights = [it[2] - it[0] for it in items if it[2] > it[0]]
    median_h = float(np.median(heights)) if heights else 20.0
    thresh = max(12.0, median_h * 0.55)

    # Column-aware only when a clear empty vertical gutter exists (GİB header
    # left/right). Full-width tables must NOT be split or rows break apart.
    if PHOTO_OCR_COLUMNS and len(items) >= 12:
        xs = np.array([(it[1] + it[3]) / 2 for it in items], dtype=np.float32)
        x_min, x_max = float(xs.min()), float(xs.max())
        span = x_max - x_min
        if span > 200:
            mid = (x_min + x_max) / 2
            gutter = span * 0.08
            in_gutter = int(np.sum(np.abs(xs - mid) < gutter))
            left = [it for it in items if (it[1] + it[3]) / 2 <= mid - gutter]
            right = [it for it in items if (it[1] + it[3]) / 2 >= mid + gutter]
            if (
                len(left) >= 5
                and len(right) >= 5
                and in_gutter <= max(2, len(items) * 0.05)
            ):
                left_mean = float(np.mean([(it[1] + it[3]) / 2 for it in left]))
                right_mean = float(np.mean([(it[1] + it[3]) / 2 for it in right]))
                if right_mean - left_mean > span * 0.28:
                    left_txt = _lines_to_text(_cluster_lines(left, thresh))
                    right_txt = _lines_to_text(_cluster_lines(right, thresh))
                    return f"{left_txt}\n\n{right_txt}".strip()

    return _lines_to_text(_cluster_lines(items, thresh))


def _lines_to_text(
    lines: list[list[tuple[float, float, float, float, str, float]]],
) -> str:
    out_lines: list[str] = []
    for line in lines:
        line = sorted(line, key=lambda t: t[1])
        out_lines.append(" ".join(t[4] for t in line))
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


def structure_score(text: str) -> int:
    return len(_STRUCT_RE.findall(text or ""))


def _good_enough(text: str, min_struct: int | None = None) -> bool:
    """Stop multi-pass OCR when text already looks like a usable e-invoice."""
    threshold = PHOTO_OCR_EARLY_STRUCT if min_struct is None else min_struct
    if structure_score(text) < threshold:
        return False
    return bool(_GIB_HINT_RE.search(text or ""))


def _rank_key(c: tuple[str, str, float, int]) -> tuple:
    eng, text, mean, n = c
    struct = structure_score(text)
    engine_boost = {
        "latin-ppocrv5": 8.0,
        "latin-ppocrv5-strong": 6.0,
        "latin-ppocrv5-nodeskew": 7.0,
        "latin-ppocrv5-binary": 5.0,
        "ppocrv6": 5.0,
        "ppocrv6-strong": 4.0,
        "ppocrv6-binary": 3.0,
        "ppocrv6-nodeskew": 6.0,
        "tesseract-tur": -25.0,
    }.get(eng, 0.0)
    return (
        struct + engine_boost,
        mean * 40.0,
        _turkish_richness(text) * 0.15,
        min(n, 150) * 0.02,
        len(text),
    )


def _tesseract_tur(path: Path) -> str:
    import subprocess
    import tempfile

    img = preprocess_bgr(_load_bgr(path), strong=False, deskew=False)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        cv2.imwrite(tmp.name, img)
        tmp_path = tmp.name
    try:
        r = subprocess.run(
            ["tesseract", tmp_path, "stdout", "-l", "tur+eng", "--psm", "6"],
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
        return r.stdout or ""
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _run_engine(engine: Any, img: np.ndarray, label: str, candidates: list) -> None:
    try:
        text, mean, n = _result_to_text(engine(img))
        if text.strip():
            candidates.append((label, text, mean, n))
    except Exception:
        pass


def _ettn_hex_count(text: str) -> int:
    m = re.search(r"(?i)ETT?Ne?\s*[:\-]?\s*([^\n]{8,90})", text or "")
    if not m:
        return 0
    return len(re.sub(r"[^0-9A-Fa-f]", "", m.group(1)))


def ocr_image(path: Path) -> tuple[str, dict[str, Any]]:
    """Return (text, meta). Fast-first: early-exit after a strong single pass."""
    if not PHOTO_OCR_ENABLED:
        return "", {"engine": "disabled", "lineCount": 0, "elapsedMs": 0}

    started = time.perf_counter()
    raw = _load_bgr(path)
    h0, w0 = raw.shape[:2]
    long0 = max(h0, w0)
    tiny = long0 < 700
    # Tall phone screenshots / chat exports: deskew often smears fine fonts
    screenshotish = tiny or (h0 > w0 * 1.35 and long0 < 1400)
    latin, _ = get_engines(need_v6=PHOTO_OCR_DUAL or screenshotish)
    v6 = None
    candidates: list[tuple[str, str, float, int]] = []

    # Screenshots (chat/UI): PP-OCRv6 reads dense GİB layouts better.
    target = 2400 if screenshotish else None
    img = preprocess_bgr(
        raw, strong=False, deskew=not screenshotish, target_side=target
    )
    early = False
    if screenshotish:
        _, v6 = get_engines(need_v6=True)
        if v6 is not None:
            _run_engine(v6, img, "ppocrv6", candidates)
        best = max(candidates, key=_rank_key) if candidates else ("", "", 0.0, 0)
        if candidates and _good_enough(best[1]):
            early = True
        else:
            _run_engine(latin, img, "latin-ppocrv5", candidates)
            best = max(candidates, key=_rank_key) if candidates else best
            if candidates and _good_enough(best[1]):
                early = True
        # ETTN often needs a sharper pass on chat screenshots
        if early and _ettn_hex_count(best[1]) < 32:
            prev_best = best
            img_hi = preprocess_bgr(
                raw, strong=True, deskew=False, target_side=min(PHOTO_OCR_MAX_SIDE, 2800)
            )
            if v6 is not None:
                _run_engine(v6, img_hi, "ppocrv6-ettn", candidates)
            _run_engine(latin, img_hi, "latin-ppocrv5-ettn", candidates)
            cand_best = max(candidates, key=_rank_key)
            # Don't drop a readable fatura serial just to chase a broken ETTN line
            has_serial = lambda t: bool(re.search(r"\b[A-Z]{2,5}\d{10,16}\b", t or "", re.I))
            if _ettn_hex_count(cand_best[1]) > _ettn_hex_count(prev_best[1]) or (
                has_serial(cand_best[1]) or not has_serial(prev_best[1])
            ):
                if has_serial(cand_best[1]) or not has_serial(prev_best[1]):
                    best = cand_best
                elif _ettn_hex_count(cand_best[1]) >= 32:
                    best = cand_best
            if has_serial(prev_best[1]) and not has_serial(best[1]):
                best = prev_best
    else:
        _run_engine(latin, img, "latin-ppocrv5", candidates)
        best = max(candidates, key=_rank_key) if candidates else ("", "", 0.0, 0)
        if candidates and _good_enough(best[1]) and not PHOTO_OCR_DUAL:
            early = True

    if not early:
        # Extra engine / preprocess only when the first pass is weak
        if not screenshotish and (
            PHOTO_OCR_DUAL or not _good_enough(best[1], PHOTO_OCR_EARLY_STRUCT)
        ):
            _, v6 = get_engines(need_v6=True)
            if v6 is not None:
                _run_engine(v6, img, "ppocrv6", candidates)
                best = max(candidates, key=_rank_key)

        if candidates and _good_enough(best[1]):
            early = True
        else:
            if v6 is None:
                _, v6 = get_engines(need_v6=True)
            if screenshotish or structure_score(best[1]) < PHOTO_OCR_EARLY_STRUCT + 2:
                img_nd = preprocess_bgr(raw, strong=False, deskew=False)
                _run_engine(latin, img_nd, "latin-ppocrv5-nodeskew", candidates)
                if v6 is not None and structure_score(max(candidates, key=_rank_key)[1]) < 10:
                    _run_engine(v6, img_nd, "ppocrv6-nodeskew", candidates)
                best = max(candidates, key=_rank_key)

            if structure_score(best[1]) < (10 if tiny else 8):
                img2 = preprocess_bgr(raw, strong=True, deskew=not screenshotish)
                _run_engine(latin, img2, "latin-ppocrv5-strong", candidates)
                if v6 is not None:
                    _run_engine(v6, img2, "ppocrv6-strong", candidates)
                best = max(candidates, key=_rank_key)

            if tiny or structure_score(best[1]) < 8:
                img_bin = preprocess_bgr(raw, strong=True, deskew=False, binary=True)
                _run_engine(latin, img_bin, "latin-ppocrv5-binary", candidates)
                if v6 is not None:
                    _run_engine(v6, img_bin, "ppocrv6-binary", candidates)

    if not candidates:
        return "", {
            "engine": "none",
            "lineCount": 0,
            "elapsedMs": int((time.perf_counter() - started) * 1000),
        }

    best = max(candidates, key=_rank_key)
    # Tesseract only when RapidOCR is clearly empty of invoice structure
    if PHOTO_OCR_TESSERACT and structure_score(best[1]) < 4:
        try:
            tess = _tesseract_tur(path)
            if tess.strip():
                candidates.append(("tesseract-tur", tess, 0.55, tess.count("\n") + 1))
                best = max(candidates, key=_rank_key)
        except Exception:
            pass

    text = best[1]
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return text, {
        "engine": best[0],
        "engines": [c[0] for c in candidates],
        "lineCount": best[3],
        "elapsedMs": elapsed_ms,
        "preprocessedShape": list(img.shape[:2]),
        "meanScore": best[2],
        "turkishChars": _turkish_richness(best[1]),
        "structureScore": structure_score(best[1]),
        "altStructure": {c[0]: structure_score(c[1]) for c in candidates},
        "tinyInput": tiny,
        "earlyExit": early,
        "screenshotish": screenshotish,
    }
