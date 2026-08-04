#!/usr/bin/env python3
"""Subprocess worker for PaddleOCR-VL — keep uvicorn workers alive."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


def result_to_text(result: Any) -> str:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--pipeline", default=os.getenv("VL_OCR_PIPELINE", "v1.6"))
    ap.add_argument("--device", default=os.getenv("VL_OCR_DEVICE", "cpu"))
    args = ap.parse_args()

    out = Path(args.output)
    try:
        from paddleocr import PaddleOCRVL

        t0 = time.perf_counter()
        try:
            pipe = PaddleOCRVL(
                pipeline_version=args.pipeline,
                device=args.device,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
            )
        except TypeError:
            pipe = PaddleOCRVL(pipeline_version=args.pipeline, device=args.device)
        raw = pipe.predict(input=args.input)
        text = result_to_text(raw)
        text = re.sub(r"<!--\s*image\s*-->", "\n", text, flags=re.I)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        elapsed = int((time.perf_counter() - t0) * 1000)
        out.write_text(
            json.dumps(
                {"ok": True, "text": text, "elapsedMs": elapsed, "chars": len(text)},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        out.write_text(
            json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False),
            encoding="utf-8",
        )
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
