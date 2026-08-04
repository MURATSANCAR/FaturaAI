"""PaddleOCR-VL-1.6 document parsing (layout-agnostic).

Used for scanned / weak-text / photo invoices. Produces markdown/text that
feeds the existing field parser — no supplier templates.
"""

from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path
from typing import Any

VL_OCR_ENABLED = os.getenv("VL_OCR_ENABLED", "0") == "1"
VL_OCR_PIPELINE = os.getenv("VL_OCR_PIPELINE", "v1.6").strip()
VL_OCR_DEVICE = os.getenv("VL_OCR_DEVICE", "cpu").strip()
VL_OCR_THREADS = max(1, int(os.getenv("VL_OCR_THREADS", "4")))
VL_OCR_SERIALIZE = os.getenv("VL_OCR_SERIALIZE", "1") == "1"
VL_OCR_WARMUP = os.getenv("VL_OCR_WARMUP", "0") == "1"
# CPU VL can take minutes/page on first load; allow long timeout.
VL_OCR_TIMEOUT_HINT_S = int(os.getenv("VL_OCR_TIMEOUT_S", "300"))

_lock = threading.Lock()
_pipe: Any | None = None
_pipe_error: str | None = None
_infer_lock = threading.Lock()


def _result_to_text(result: Any) -> str:
    parts: list[str] = []

    def walk(obj: Any, depth: int = 0) -> None:
        if obj is None or depth > 8:
            return
        if isinstance(obj, str):
            if obj.strip():
                parts.append(obj)
            return
        if isinstance(obj, (int, float, bool)):
            return
        if isinstance(obj, dict):
            for k in ("markdown", "md", "text", "rec_text", "html", "content"):
                if k in obj and isinstance(obj[k], str):
                    parts.append(obj[k])
            for v in obj.values():
                walk(v, depth + 1)
            return
        if isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v, depth + 1)
            return
        for attr in ("markdown", "text", "json"):
            if hasattr(obj, attr):
                try:
                    val = getattr(obj, attr)
                    if callable(val):
                        val = val()
                    walk(val, depth + 1)
                except Exception:  # noqa: BLE001
                    pass
        if hasattr(obj, "__dict__"):
            walk(vars(obj), depth + 1)

    walk(result)
    text = "\n".join(parts)
    # Prefer longer unique chunks if walk duplicated
    if len(text) > 50:
        return text
    return str(result) if result is not None else ""


def _init_pipe() -> Any:
    global _pipe, _pipe_error
    if _pipe is not None:
        return _pipe
    if _pipe_error:
        raise RuntimeError(_pipe_error)
    with _lock:
        if _pipe is not None:
            return _pipe
        if _pipe_error:
            raise RuntimeError(_pipe_error)
        try:
            from paddleocr import PaddleOCRVL

            kwargs: dict[str, Any] = {
                "device": VL_OCR_DEVICE,
            }
            # Prefer 1.6 pipeline when API supports it
            for extra in (
                {"pipeline_version": VL_OCR_PIPELINE},
                {"pipeline_version": VL_OCR_PIPELINE, "cpu_threads": VL_OCR_THREADS},
                {
                    "device": VL_OCR_DEVICE,
                    "cpu_threads": VL_OCR_THREADS,
                    "use_doc_orientation_classify": False,
                    "use_doc_unwarping": False,
                    "pipeline_version": VL_OCR_PIPELINE,
                },
            ):
                try:
                    _pipe = PaddleOCRVL(**{**kwargs, **extra})
                    break
                except TypeError:
                    continue
            if _pipe is None:
                _pipe = PaddleOCRVL(device=VL_OCR_DEVICE)
        except Exception as exc:  # noqa: BLE001
            _pipe_error = f"PaddleOCRVL init failed: {exc}"
            raise RuntimeError(_pipe_error) from exc
    return _pipe


def engine_status() -> dict[str, Any]:
    return {
        "enabled": VL_OCR_ENABLED,
        "pipeline": VL_OCR_PIPELINE,
        "device": VL_OCR_DEVICE,
        "threads": VL_OCR_THREADS,
        "loaded": _pipe is not None,
        "error": _pipe_error,
        "timeoutHintS": VL_OCR_TIMEOUT_HINT_S,
    }


def warmup() -> dict[str, Any]:
    if not VL_OCR_ENABLED:
        return {"skipped": True, "reason": "disabled"}
    t0 = time.perf_counter()
    _init_pipe()
    return {
        "ok": True,
        "loadMs": int((time.perf_counter() - t0) * 1000),
        **engine_status(),
    }


def ocr_document(path: Path | str) -> tuple[str, dict[str, Any]]:
    """Run PaddleOCR-VL on an image or PDF path; return markdown/text + meta."""
    if not VL_OCR_ENABLED:
        raise RuntimeError("VL OCR disabled (VL_OCR_ENABLED=0)")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    t0 = time.perf_counter()
    pipe = _init_pipe()

    def _predict() -> Any:
        return pipe.predict(input=str(path))

    if VL_OCR_SERIALIZE:
        with _infer_lock:
            raw = _predict()
    else:
        raw = _predict()

    text = _result_to_text(raw)
    # Strip obvious HTML image stubs noise from VL/doc parsers
    text = re.sub(r"<!--\s*image\s*-->", "\n", text, flags=re.I)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    elapsed = int((time.perf_counter() - t0) * 1000)
    meta: dict[str, Any] = {
        "engine": f"paddleocr-vl-{VL_OCR_PIPELINE}",
        "elapsedMs": elapsed,
        "lineCount": len([ln for ln in text.splitlines() if ln.strip()]),
        "charCount": len(text),
        "device": VL_OCR_DEVICE,
    }
    if not text.strip():
        raise RuntimeError("PaddleOCR-VL returned empty text")
    return text, meta
