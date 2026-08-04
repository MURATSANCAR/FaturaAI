#!/usr/bin/env python3
"""Batch Fehmi retest with PaddleOCR-VL-1.6 (model loaded once)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Ensure extract imports
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("VL_OCR_ENABLED", "1")
os.environ.setdefault("VL_OCR_PIPELINE", "v1.6")
os.environ.setdefault("VL_OCR_DEVICE", "cpu")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("VL_OCR_SUBPROCESS", "0")  # in-process, load once

from vl_ocr import _predict_inprocess, _result_to_text  # noqa: E402
import vl_ocr as vl_mod  # noqa: E402

# Force in-process for batch
vl_mod.VL_OCR_ENABLED = True
vl_mod.VL_OCR_SUBPROCESS = False

from main import (  # noqa: E402
    Invoice,
    merge_invoice,
    parse_text_invoice,
    status_from,
    validate_invoice,
)


def rasterize(pdf: Path, out_prefix: Path, dpi: int = 200) -> list[Path]:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for old in out_prefix.parent.glob(out_prefix.name + "*.png"):
        old.unlink(missing_ok=True)
    subprocess.run(
        ["pdftoppm", "-png", "-r", str(dpi), "-f", "1", "-l", "2", str(pdf), str(out_prefix)],
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    return sorted(out_prefix.parent.glob(out_prefix.name + "*.png"))


def main() -> int:
    base = Path("/tmp/fehmi_test")
    mapping = {
        line.split("\t", 1)[0]: line.split("\t", 1)[1].strip()
        for line in base.joinpath("map.txt").read_text().splitlines()
        if "\t" in line
    }

    print("Loading PaddleOCR-VL-1.6 once…", flush=True)
    t_load = time.perf_counter()
    # warm by first predict on tiny path later; construct pipe now
    from paddleocr import PaddleOCRVL

    try:
        pipe = PaddleOCRVL(
            pipeline_version="v1.6",
            device="cpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
    except TypeError:
        pipe = PaddleOCRVL(pipeline_version="v1.6", device="cpu")
    load_ms = int((time.perf_counter() - t_load) * 1000)
    print(f"model_load_ms={load_ms}", flush=True)

    rows = []
    for pdf in sorted(base.glob("fehmi_*.pdf")):
        original = mapping.get(pdf.name, pdf.name)
        print("=" * 72, flush=True)
        print(pdf.name, "=>", original, flush=True)
        t0 = time.perf_counter()
        pages = rasterize(pdf, Path(f"/tmp/fehmi_batch/{pdf.stem}"))
        raster_ms = int((time.perf_counter() - t0) * 1000)
        texts = []
        vl_ms_total = 0
        for page in pages:
            t1 = time.perf_counter()
            raw = pipe.predict(input=str(page))
            text = _result_to_text(raw)
            took = int((time.perf_counter() - t1) * 1000)
            vl_ms_total += took
            print(f"  page={page.name} vl_ms={took} chars={len(text)}", flush=True)
            if text.strip():
                texts.append(text.strip())
        combined = "\n\n".join(texts)
        inv = parse_text_invoice(combined, original)
        warnings, validation = validate_invoice(inv)
        status = status_from(warnings, validation)
        wall = int((time.perf_counter() - t0) * 1000)
        row = {
            "file": pdf.name,
            "original": original,
            "status": status,
            "conf": validation.confidence,
            "wallMs": wall,
            "rasterMs": raster_ms,
            "vlMs": vl_ms_total,
            "modelLoadMsOnce": load_ms,
            "pages": len(pages),
            "invoiceNumber": inv.invoiceNumber,
            "issueDate": inv.issueDate,
            "uuid": inv.uuid,
            "payable": inv.totals.payableAmount,
            "lines": len(inv.lines),
            "supplier": {
                "name": inv.supplier.name,
                "taxId": inv.supplier.taxId,
                "scheme": inv.supplier.taxIdScheme,
            },
            "customer": {
                "name": inv.customer.name,
                "taxId": inv.customer.taxId,
                "scheme": inv.customer.taxIdScheme,
            },
            "warnings": warnings[:12],
            "preview": combined[:500],
            "engine": "paddleocr-vl-v1.6",
        }
        print(json.dumps({k: row[k] for k in row if k != "preview"}, ensure_ascii=False, indent=2), flush=True)
        rows.append(row)

    out = Path("/tmp/fehmi_vl_results.json")
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("WROTE", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
