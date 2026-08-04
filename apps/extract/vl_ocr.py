"""PaddleOCR-VL-1.6 document parsing (layout-agnostic).

Runs inference in a spawned subprocess so uvicorn workers are not killed by
Paddle/native crashes. Produces markdown/text for the existing field parser.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
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
VL_OCR_TIMEOUT_HINT_S = int(os.getenv("VL_OCR_TIMEOUT_S", "300"))
VL_OCR_SUBPROCESS = os.getenv("VL_OCR_SUBPROCESS", "1") == "1"

_infer_lock = threading.Lock()
_last_error: str | None = None


def engine_status() -> dict[str, Any]:
    return {
        "enabled": VL_OCR_ENABLED,
        "pipeline": VL_OCR_PIPELINE,
        "device": VL_OCR_DEVICE,
        "threads": VL_OCR_THREADS,
        "subprocess": VL_OCR_SUBPROCESS,
        "error": _last_error,
        "timeoutHintS": VL_OCR_TIMEOUT_HINT_S,
    }


def warmup() -> dict[str, Any]:
    if not VL_OCR_ENABLED:
        return {"skipped": True, "reason": "disabled"}
    # Warmup via tiny blank image would still download/load model — skip by default.
    return {"skipped": True, "reason": "lazy-load-on-first-request", **engine_status()}


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
    return text if len(text) > 50 else (str(result) if result is not None else "")


def _predict_inprocess(path: Path) -> tuple[str, int]:
    from paddleocr import PaddleOCRVL

    t0 = time.perf_counter()
    try:
        pipe = PaddleOCRVL(
            pipeline_version=VL_OCR_PIPELINE,
            device=VL_OCR_DEVICE,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
    except TypeError:
        pipe = PaddleOCRVL(pipeline_version=VL_OCR_PIPELINE, device=VL_OCR_DEVICE)
    raw = pipe.predict(input=str(path))
    text = _result_to_text(raw)
    elapsed = int((time.perf_counter() - t0) * 1000)
    return text, elapsed


def _predict_subprocess(path: Path) -> tuple[str, int]:
    """Spawn a fresh interpreter so paddle crashes do not take down uvicorn."""
    global _last_error
    worker = Path(__file__).resolve().parent / "_vl_worker.py"
    out_json = Path(tempfile.mkstemp(prefix="vlocr_", suffix=".json")[1])
    env = os.environ.copy()
    env["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    env["VL_OCR_PIPELINE"] = VL_OCR_PIPELINE
    env["VL_OCR_DEVICE"] = VL_OCR_DEVICE
    env["OMP_NUM_THREADS"] = str(VL_OCR_THREADS)
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    cmd = [
        sys.executable,
        str(worker),
        "--input",
        str(path),
        "--output",
        str(out_json),
        "--pipeline",
        VL_OCR_PIPELINE,
        "--device",
        VL_OCR_DEVICE,
    ]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=VL_OCR_TIMEOUT_HINT_S,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        _last_error = f"VL subprocess timeout after {VL_OCR_TIMEOUT_HINT_S}s"
        raise RuntimeError(_last_error) from exc

    wall = int((time.perf_counter() - t0) * 1000)
    if proc.returncode != 0 or not out_json.exists():
        err = (proc.stderr or proc.stdout or "").strip()[-2000:]
        _last_error = f"VL subprocess failed rc={proc.returncode}: {err}"
        raise RuntimeError(_last_error)

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    try:
        out_json.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass
    if not payload.get("ok"):
        _last_error = str(payload.get("error") or "VL worker error")
        raise RuntimeError(_last_error)
    text = str(payload.get("text") or "")
    elapsed = int(payload.get("elapsedMs") or wall)
    return text, elapsed


def ocr_document(path: Path | str) -> tuple[str, dict[str, Any]]:
    """Run PaddleOCR-VL on an image (or PDF if supported); return text + meta."""
    global _last_error
    if not VL_OCR_ENABLED:
        raise RuntimeError("VL OCR disabled (VL_OCR_ENABLED=0)")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    def _run() -> tuple[str, int]:
        if VL_OCR_SUBPROCESS:
            return _predict_subprocess(path)
        return _predict_inprocess(path)

    if VL_OCR_SERIALIZE:
        with _infer_lock:
            text, elapsed = _run()
    else:
        text, elapsed = _run()

    text = re.sub(r"<!--\s*image\s*-->", "\n", text, flags=re.I)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    meta: dict[str, Any] = {
        "engine": f"paddleocr-vl-{VL_OCR_PIPELINE}",
        "elapsedMs": elapsed,
        "lineCount": len([ln for ln in text.splitlines() if ln.strip()]),
        "charCount": len(text),
        "device": VL_OCR_DEVICE,
        "subprocess": VL_OCR_SUBPROCESS,
    }
    if not text.strip():
        _last_error = "PaddleOCR-VL returned empty text"
        raise RuntimeError(_last_error)
    _last_error = None
    return text, meta
