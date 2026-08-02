"""High-quality photo OCR for Turkish invoices / receipts.

Prod stack (CPU):
  OpenCV preprocess
  → PP-OCRv6 Small (OpenVINO, else ONNX Runtime)
  → if confidence < threshold or field validation fails
  → PP-OCRv6 Medium
  → Tesseract tur+eng only as last-resort if structure is still weak
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
PHOTO_OCR_TESSERACT = os.getenv("PHOTO_OCR_TESSERACT", "1") == "1"
PHOTO_OCR_COLUMNS = os.getenv("PHOTO_OCR_COLUMNS", "1") == "1"
PHOTO_OCR_EARLY_STRUCT = int(os.getenv("PHOTO_OCR_EARLY_STRUCT", "8"))
# Prefer OpenVINO; fall back to ONNX when init/infer fails (auto).
PHOTO_OCR_ENGINE = os.getenv("PHOTO_OCR_ENGINE", "auto").strip().lower()
PHOTO_OCR_THREADS = max(1, int(os.getenv("PHOTO_OCR_THREADS", "8")))
PHOTO_OCR_CONF_THRESHOLD = float(os.getenv("PHOTO_OCR_CONF_THRESHOLD", "0.90"))
# Force Medium on every page (debug / quality A/B). Default: Small → Medium ladder.
PHOTO_OCR_FORCE_MEDIUM = os.getenv("PHOTO_OCR_FORCE_MEDIUM", "0") == "1"

_engine_small = None
_engine_medium = None
_engine_backend: str | None = None
_engine_lock = threading.Lock()
_engine_errors: list[str] = []

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

_AMOUNT_RE = re.compile(r"\d{1,3}(?:[.\s]\d{3})*[.,]\d{2}")
_SERIAL_RE = re.compile(r"\b[A-Z]{2,5}\d{10,16}\b", re.I)


def _model_type(name: str):
    from rapidocr import ModelType

    key = name.strip().lower()
    if key == "medium":
        return ModelType.MEDIUM
    if key == "tiny":
        return ModelType.TINY
    return ModelType.SMALL


def _resolve_engine_order() -> list[Any]:
    from rapidocr import EngineType

    pref = PHOTO_OCR_ENGINE
    if pref in ("openvino", "ov"):
        return [EngineType.OPENVINO, EngineType.ONNXRUNTIME]
    if pref in ("onnx", "onnxruntime", "ort"):
        return [EngineType.ONNXRUNTIME]
    # auto: OpenVINO first, ONNX fallback
    return [EngineType.OPENVINO, EngineType.ONNXRUNTIME]


def _engine_params(engine_type: Any, model_type: Any) -> dict[str, Any]:
    from rapidocr import OCRVersion

    threads = PHOTO_OCR_THREADS
    return {
        "Det.engine_type": engine_type,
        "Det.lang_type": "tr",
        "Det.model_type": model_type,
        "Det.ocr_version": OCRVersion.PPOCRV6,
        "Rec.engine_type": engine_type,
        "Rec.lang_type": "tr",
        "Rec.model_type": model_type,
        "Rec.ocr_version": OCRVersion.PPOCRV6,
        "Cls.engine_type": engine_type,
        "EngineConfig.onnxruntime.intra_op_num_threads": threads,
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        "EngineConfig.openvino.inference_num_threads": threads,
        "EngineConfig.openvino.performance_hint": "LATENCY",
        "Global.log_level": "error",
    }


def _make_engine(model_size: str, engine_type: Any | None = None) -> tuple[Any, str]:
    from rapidocr import RapidOCR

    model_type = _model_type(model_size)
    order = [engine_type] if engine_type is not None else _resolve_engine_order()
    last_err: Exception | None = None
    for eng in order:
        if eng is None:
            continue
        try:
            ocr = RapidOCR(params=_engine_params(eng, model_type))
            # Smoke infer so OpenVINO reshape / load failures surface at init.
            blank = np.full((64, 64, 3), 255, dtype=np.uint8)
            ocr(blank)
            return ocr, eng.value
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            msg = f"{eng.value}/{model_size}: {type(exc).__name__}: {exc}"
            if msg not in _engine_errors:
                _engine_errors.append(msg)
            continue
    raise RuntimeError(
        f"PP-OCRv6 {model_size} engine init failed"
        + (f": {last_err}" if last_err else "")
    )


def get_engines(*, need_medium: bool = False) -> tuple[Any, Any | None, str]:
    """Return (small, medium|None, backend_name)."""
    global _engine_small, _engine_medium, _engine_backend
    with _engine_lock:
        if _engine_small is None:
            _engine_small, _engine_backend = _make_engine("small")
        if need_medium and _engine_medium is None:
            try:
                # Prefer same backend as Small when possible.
                from rapidocr import EngineType

                preferred = None
                if _engine_backend == "openvino":
                    preferred = EngineType.OPENVINO
                elif _engine_backend == "onnxruntime":
                    preferred = EngineType.ONNXRUNTIME
                _engine_medium, backend_m = _make_engine("medium", preferred)
                if _engine_backend is None:
                    _engine_backend = backend_m
            except Exception as exc:  # noqa: BLE001
                msg = f"medium: {type(exc).__name__}: {exc}"
                if msg not in _engine_errors:
                    _engine_errors.append(msg)
                _engine_medium = None
        return _engine_small, _engine_medium, _engine_backend or "unknown"


def warmup_engines(*, include_medium: bool = True) -> dict[str, Any]:
    """Eager-load Small (+ Medium) so first request is not a cold download."""
    small, medium, backend = get_engines(need_medium=include_medium)
    return {
        "backend": backend,
        "small": small is not None,
        "medium": medium is not None,
        "threads": PHOTO_OCR_THREADS,
        "confThreshold": PHOTO_OCR_CONF_THRESHOLD,
        "errors": list(_engine_errors[-5:]),
    }


def engine_status() -> dict[str, Any]:
    return {
        "enabled": PHOTO_OCR_ENABLED,
        "enginePref": PHOTO_OCR_ENGINE,
        "backend": _engine_backend,
        "smallLoaded": _engine_small is not None,
        "mediumLoaded": _engine_medium is not None,
        "threads": PHOTO_OCR_THREADS,
        "confThreshold": PHOTO_OCR_CONF_THRESHOLD,
        "forceMedium": PHOTO_OCR_FORCE_MEDIUM,
        "errors": list(_engine_errors[-5:]),
    }


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
    threshold = PHOTO_OCR_EARLY_STRUCT if min_struct is None else min_struct
    if structure_score(text) < threshold:
        return False
    return bool(_GIB_HINT_RE.search(text or ""))


def _has_gib_serial(text: str) -> bool:
    return bool(_SERIAL_RE.search(text or ""))


def _ettn_hex_count(text: str) -> int:
    m = re.search(r"(?i)ETT?Ne?\s*[:\-]?\s*([^\n]{8,90})", text or "")
    if not m:
        return 0
    return len(re.sub(r"[^0-9A-Fa-f]", "", m.group(1)))


def field_validation_ok(text: str) -> bool:
    """Lightweight invoice field gate used for Small → Medium fallback."""
    if not text or not text.strip():
        return False
    if not _good_enough(text):
        return False
    has_serial = _has_gib_serial(text)
    has_amount = bool(_AMOUNT_RE.search(text))
    has_ettn = _ettn_hex_count(text) >= 32
    # Accept strong structure with amount, or GİB identity (serial/ETTN).
    return has_amount and (has_serial or has_ettn or structure_score(text) >= 10)


def needs_medium_fallback(text: str, mean_score: float) -> bool:
    if PHOTO_OCR_FORCE_MEDIUM:
        return True
    if mean_score < PHOTO_OCR_CONF_THRESHOLD:
        return True
    if not field_validation_ok(text):
        return True
    return False


def _rank_key(c: tuple[str, str, float, int]) -> tuple:
    eng, text, mean, n = c
    struct = structure_score(text)
    engine_boost = {
        "ppocrv6-medium": 10.0,
        "ppocrv6-medium-strong": 8.0,
        "ppocrv6-medium-nodeskew": 9.0,
        "ppocrv6-medium-binary": 7.0,
        "ppocrv6-small": 6.0,
        "ppocrv6-small-strong": 4.0,
        "ppocrv6-small-nodeskew": 5.0,
        "ppocrv6-small-binary": 3.0,
        "gib-right-panel": 2.0,
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


def ocr_image(path: Path) -> tuple[str, dict[str, Any]]:
    """Return (text, meta). Small primary → Medium on low conf / field fail."""
    if not PHOTO_OCR_ENABLED:
        return "", {"engine": "disabled", "lineCount": 0, "elapsedMs": 0}

    started = time.perf_counter()
    raw = _load_bgr(path)
    h0, w0 = raw.shape[:2]
    long0 = max(h0, w0)
    tiny = long0 < 700
    screenshotish = tiny or (h0 > w0 * 1.35 and long0 < 1400)

    force_medium = PHOTO_OCR_FORCE_MEDIUM
    small, medium, backend = get_engines(need_medium=force_medium)
    candidates: list[tuple[str, str, float, int]] = []

    target = 2400 if screenshotish else None
    img = preprocess_bgr(
        raw, strong=False, deskew=not screenshotish, target_side=target
    )

    used_medium = False
    early = False
    fallback_reason: str | None = None

    if force_medium:
        if medium is None:
            _, medium, backend = get_engines(need_medium=True)
        if medium is not None:
            _run_engine(medium, img, "ppocrv6-medium", candidates)
            used_medium = True
        else:
            _run_engine(small, img, "ppocrv6-small", candidates)
        best = max(candidates, key=_rank_key) if candidates else ("", "", 0.0, 0)
        early = bool(candidates and _good_enough(best[1]))
        fallback_reason = "force-medium"
    else:
        _run_engine(small, img, "ppocrv6-small", candidates)
        best = max(candidates, key=_rank_key) if candidates else ("", "", 0.0, 0)
        if candidates and not needs_medium_fallback(best[1], best[2]):
            early = True
        else:
            mean_score = best[2] if candidates else 0.0
            if not candidates or not best[1].strip():
                fallback_reason = "empty-small"
            elif mean_score < PHOTO_OCR_CONF_THRESHOLD:
                fallback_reason = "low-confidence"
            else:
                fallback_reason = "field-validation"
            _, medium, backend = get_engines(need_medium=True)
            if medium is not None:
                _run_engine(medium, img, "ppocrv6-medium", candidates)
                used_medium = True
                best = max(candidates, key=_rank_key)
                if candidates and _good_enough(best[1]):
                    early = True

    if not early:
        # Extra preprocess only when Small (+ Medium) still weak
        if medium is None:
            _, medium, backend = get_engines(need_medium=True)
        primary = medium if used_medium and medium is not None else small
        primary_label = "ppocrv6-medium" if primary is medium and medium else "ppocrv6-small"

        if screenshotish or structure_score(best[1]) < PHOTO_OCR_EARLY_STRUCT + 2:
            img_nd = preprocess_bgr(raw, strong=False, deskew=False)
            _run_engine(primary, img_nd, f"{primary_label}-nodeskew", candidates)
            if (
                medium is not None
                and primary is not medium
                and structure_score(max(candidates, key=_rank_key)[1]) < 10
            ):
                _run_engine(medium, img_nd, "ppocrv6-medium-nodeskew", candidates)
                used_medium = True
            best = max(candidates, key=_rank_key)

        if structure_score(best[1]) < (10 if tiny else 8):
            img2 = preprocess_bgr(raw, strong=True, deskew=not screenshotish)
            _run_engine(primary, img2, f"{primary_label}-strong", candidates)
            if medium is not None and primary is not medium:
                _run_engine(medium, img2, "ppocrv6-medium-strong", candidates)
                used_medium = True
            best = max(candidates, key=_rank_key)

        if tiny or structure_score(best[1]) < 8:
            img_bin = preprocess_bgr(raw, strong=True, deskew=False, binary=True)
            _run_engine(primary, img_bin, f"{primary_label}-binary", candidates)
            if medium is not None and primary is not medium:
                _run_engine(medium, img_bin, "ppocrv6-medium-binary", candidates)
                used_medium = True

    if not candidates:
        return "", {
            "engine": "none",
            "backend": backend,
            "lineCount": 0,
            "elapsedMs": int((time.perf_counter() - started) * 1000),
            "usedMedium": used_medium,
            "fallbackReason": fallback_reason,
        }

    best = max(candidates, key=_rank_key)

    # GİB screenshot: right-panel metadata crop
    panel_note = ""
    if screenshotish and (
        not _has_gib_serial(best[1]) or _ettn_hex_count(best[1]) < 32
    ):
        if medium is None:
            _, medium, backend = get_engines(need_medium=True)
        h, w = raw.shape[:2]
        panel = raw[0 : int(h * 0.55), int(w * 0.40) : w]
        panel_img = preprocess_bgr(
            panel, strong=True, deskew=False, target_side=min(PHOTO_OCR_MAX_SIDE, 2000)
        )
        panel_cands: list[tuple[str, str, float, int]] = []
        eng = medium or small
        _run_engine(eng, panel_img, "gib-right-panel", panel_cands)
        if panel_cands and panel_cands[0][1].strip():
            merged = f"{best[1]}\n\n{panel_cands[0][1]}".strip()
            best = (f"{best[0]}+panel", merged, best[2], best[3] + panel_cands[0][3])
            panel_note = "panel"
            candidates.append(best)
            if eng is medium:
                used_medium = True

    if PHOTO_OCR_TESSERACT and structure_score(best[1]) < 4:
        try:
            tess = _tesseract_tur(path)
            if tess.strip():
                candidates.append(("tesseract-tur", tess, 0.55, tess.count("\n") + 1))
                best = max(candidates, key=_rank_key)
        except Exception:
            pass

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return best[1], {
        "engine": best[0],
        "backend": backend,
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
        "panelBoost": panel_note or None,
        "usedMedium": used_medium,
        "fallbackReason": fallback_reason,
        "confThreshold": PHOTO_OCR_CONF_THRESHOLD,
        "fieldValidation": field_validation_ok(best[1]),
        "threads": PHOTO_OCR_THREADS,
    }
