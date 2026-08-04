#!/usr/bin/env python3
"""Fehmi VL dump + binder retest.

Usage:
  # Full VL (slow): writes .md then parses with binder
  python fehmi_vl_batch.py

  # Binder only (fast) when .md files already exist
  python fehmi_vl_batch.py --reparse-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("VL_OCR_ENABLED", "1")
os.environ.setdefault("VL_OCR_PIPELINE", "v1.6")
os.environ.setdefault("VL_OCR_DEVICE", "cpu")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("VL_OCR_SUBPROCESS", "0")

from parse_vl_markdown import parse_vl_markdown  # noqa: E402
from main import status_from, validate_invoice  # noqa: E402
from vl_ocr import _result_to_text  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reparse-only", action="store_true")
    args = ap.parse_args()

    base = Path("/tmp/fehmi_test")
    out_dir = Path("/tmp/fehmi_batch")
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        line.split("\t", 1)[0]: line.split("\t", 1)[1].strip()
        for line in base.joinpath("map.txt").read_text().splitlines()
        if "\t" in line
    }

    pipe = None
    load_ms = 0
    if not args.reparse_only:
        print("Loading PaddleOCR-VL-1.6 once…", flush=True)
        t_load = time.perf_counter()
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
        stem = pdf.stem
        md_path = out_dir / f"{stem}.md"
        png = out_dir / f"{stem}-1.png"
        print("=" * 72, flush=True)
        print(pdf.name, "=>", original, flush=True)

        vl_ms = 0
        if args.reparse_only:
            if not md_path.exists():
                print(f"MISSING {md_path} — skip", flush=True)
                continue
            combined = md_path.read_text(encoding="utf-8", errors="replace")
        else:
            if not png.exists():
                print(f"MISSING png {png}", flush=True)
                continue
            assert pipe is not None
            t1 = time.perf_counter()
            raw = pipe.predict(input=str(png))
            combined = _result_to_text(raw)
            vl_ms = int((time.perf_counter() - t1) * 1000)
            md_path.write_text(combined, encoding="utf-8")
            print(f"  wrote {md_path} vl_ms={vl_ms} chars={len(combined)}", flush=True)

        t0 = time.perf_counter()
        inv = parse_vl_markdown(combined, original)
        warnings, validation = validate_invoice(inv)
        status = status_from(warnings, validation)
        parse_ms = int((time.perf_counter() - t0) * 1000)
        row = {
            "file": pdf.name,
            "original": original,
            "status": status,
            "conf": validation.confidence,
            "vlMs": vl_ms,
            "parseMs": parse_ms,
            "modelLoadMsOnce": load_ms,
            "invoiceNumber": inv.invoiceNumber,
            "issueDate": inv.issueDate,
            "uuid": inv.uuid,
            "payable": inv.totals.payableAmount,
            "lineExtension": inv.totals.lineExtensionAmount,
            "vat": inv.totals.vatAmount,
            "taxInclusive": inv.totals.taxInclusiveAmount,
            "lines": [
                {
                    "name": l.name,
                    "qty": l.quantity,
                    "unitPrice": l.unitPrice,
                    "lineTotal": l.lineTotal,
                }
                for l in inv.lines
            ],
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
            "engine": "paddleocr-vl-v1.6+binder",
            "preview": combined[:600],
        }
        print(
            json.dumps({k: row[k] for k in row if k != "preview"}, ensure_ascii=False, indent=2),
            flush=True,
        )
        rows.append(row)

    out = Path("/tmp/fehmi_vl_binder_results.json")
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("WROTE", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
