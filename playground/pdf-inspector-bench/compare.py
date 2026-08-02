#!/usr/bin/env python3
"""Compare pdf-inspector vs golden pdftotext samples + extract parser fields."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXTRACT = ROOT / "apps" / "extract"
SAMPLES = ROOT / "samples"
sys.path.insert(0, str(EXTRACT))

from pdf_inspector_text import extract_pdf_inspector  # noqa: E402


def load_parse():
    import main as extract_main  # noqa: WPS433

    return extract_main.parse_text_invoice, extract_main.validate_invoice


def summarize(invoice) -> dict:
    return {
        "invoiceNumber": invoice.invoiceNumber,
        "issueDate": invoice.issueDate,
        "supplier": invoice.supplier.name,
        "customer": invoice.customer.name,
        "payable": invoice.totals.payableAmount,
        "vat": invoice.totals.vatAmount,
        "lines": len(invoice.lines or []),
        "line0": (invoice.lines[0].name if invoice.lines else None),
    }


def golden_for(pdf: Path) -> Path | None:
    mapping = [
        (lambda n: "HAVA_SAVUNMA" in n, "hava-savunma.pdftotext.txt"),
        (lambda n: "KVI" in n, "kvi.pdftotext.txt"),
        (lambda n: "MDA" in n, "mda.pdftotext.txt"),
        (lambda n: "BBE2026000018085" in n, "babymall-1.pdftotext.txt"),
        (lambda n: "BBE2026000018417" in n, "babymall-3.pdftotext.txt"),
        (lambda n: "exbilisim" in n.lower() or "YAU" in n, "exbilisim-yau.pdftotext.txt"),
    ]
    direct = SAMPLES / f"{pdf.stem}.pdftotext.txt"
    if direct.exists():
        return direct
    for pred, name in mapping:
        if pred(pdf.name):
            p = SAMPLES / name
            if p.exists():
                return p
    return None


def main() -> int:
    parse_text_invoice, validate_invoice = load_parse()
    pdfs = sorted(SAMPLES.glob("*.pdf"))
    if not pdfs:
        print("no sample PDFs in", SAMPLES)
        return 1

    rows = []
    for pdf in pdfs:
        t0 = time.perf_counter()
        text, meta = extract_pdf_inspector(pdf)
        ms = (time.perf_counter() - t0) * 1000

        invoice = parse_text_invoice(text, pdf.name) if text.strip() else None
        if invoice is not None:
            warnings, _validation = validate_invoice(invoice)
        else:
            warnings = ["empty text"]

        golden_path = golden_for(pdf)
        golden_summary = None
        if golden_path:
            gtext = golden_path.read_text(encoding="utf8", errors="replace")
            ginv = parse_text_invoice(gtext, pdf.name)
            gw, _ = validate_invoice(ginv)
            golden_summary = {
                "from": golden_path.name,
                **summarize(ginv),
                "warnings": len(gw),
            }

        row = {
            "file": pdf.name,
            "ms": round(ms, 1),
            "meta": {
                "pdfType": meta.get("pdfType"),
                "source": meta.get("source"),
                "confidence": meta.get("confidence"),
                "charCount": meta.get("charCount") or len(text),
                "hasEncodingIssues": meta.get("hasEncodingIssues"),
            },
            "inspector": summarize(invoice) if invoice else None,
            "inspectorWarnings": warnings,
            "golden": golden_summary,
        }
        rows.append(row)

        print(f"\n=== {pdf.name} ===")
        print(
            f"  inspector: {ms:.0f}ms type={meta.get('pdfType')} src={meta.get('source')} "
            f"chars={meta.get('charCount') or len(text)}"
        )
        if invoice:
            s = summarize(invoice)
            print(
                f"  parse: no={s['invoiceNumber']} date={s['issueDate']} "
                f"payable={s['payable']} lines={s['lines']} "
                f"sup={(s['supplier'] or '')[:40]!r}"
            )
            print(f"  warnings({len(warnings)}): {warnings[:5]}")
        else:
            print("  parse: FAILED (no text)")
        if golden_summary and invoice:
            g = golden_summary
            match_no = invoice.invoiceNumber == g["invoiceNumber"]
            match_pay = invoice.totals.payableAmount == g["payable"]
            print(
                f"  golden({g['from']}): no={g['invoiceNumber']} payable={g['payable']} "
                f"match_no={match_no} match_payable={match_pay}"
            )

    out = Path(__file__).resolve().parent / "last-run.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
