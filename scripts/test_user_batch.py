#!/usr/bin/env python3
"""Timed extract test for the user-uploaded invoice set."""
from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path

IN = Path("/tmp/fatura-user-batch")
OUT = Path("/tmp/fatura-user-batch-out")
URL = "http://127.0.0.1:8105/extract"
TIMEOUT = 180

# Exact files the user attached in this turn
FILES = [
    "evpark-usaktan-earsiv-fatura-ve-telefon-bilgisi-talebim-cevapsiz-1-e35e86d4-8abd-49df-905e-4cfc55c744d8.png",
    "images__2_-08ec207a-8459-4970-b1e7-d3ec82e3016f.png",
    "images__3_-58b7879a-b13f-427a-918b-d2f5578d920e.png",
    "a101-servis-raporuna-ragmen-para-iadesi-yapmiyor-2-9b531e94-4b84-4b81-92f7-ab7eb8a905f4.png",
    "images-8585ab4f-db15-4ac4-b562-bb88ecc46d3b.png",
    "images__1_-1be60773-4a65-47c3-91ce-56ab4ed6e9e8.png",
    "images-e99fe162-a9a7-470c-b65a-e7a90899bc26.png",
    "images__1_-13b9fbcd-b660-4a4a-bc7a-00994ce22526.png",
    "teknosa_2-cb71dfe3-f733-4360-923b-8e2ac87e6053.png",
    "WhatsApp_Image_2026-07-30_at_22.10.28_2-57b37502-7674-4c9f-b196-60de0f8ad369.png",
    "Teknosa-50c0cb18-58bf-4e0c-95ea-da46ec7f2fa1.png",
    "WhatsApp_Image_2026-07-30_at_21.12.30-eccb4237-f121-4b3e-9a34-33bffc6b27a1.png",
    "WhatsApp_Image_2026-07-30_at_21.12.30_2-a71f9d1d-72db-4ed6-bfcd-d949d58c3b8f.png",
    "WhatsApp_Image_2026-07-31_at_02.23.03-a7e33a66-1900-4d9e-ad1b-71f4769dbae3.png",
    "Telefondan_c_ekilen_fatura_go_rsel-a3a618b6-2f16-4475-8f95-ec17efa9b75b.png",
    "WhatsApp_Image_2026-07-31_at_16.08.39-b9b1b216-3094-4832-b960-e9b85cecd67d.png",
    "WhatsApp_Image_2026-07-30_at_22.10.28-a1b12344-b987-4e7d-a68c-dda410aba0a2.png",
    "NanobaseAI-b5535e6d-967b-4ce1-86ff-a5c1036df388.png",
]

LABEL = {
    FILES[0]: "Evpark fiş",
    FILES[1]: "Vulkan/junk?",
    FILES[2]: "Afeks",
    FILES[3]: "A101 fiş",
    FILES[4]: "DeFacto",
    FILES[5]: "Mango",
    FILES[6]: "Elif Gıda",
    FILES[7]: "Orka",
    FILES[8]: "Teknosa web (sikayetvar)",
    FILES[9]: "BabyMall WA",
    FILES[10]: "Teknosa kağıt",
    FILES[11]: "BabyMall laptop",
    FILES[12]: "BabyMall laptop2",
    FILES[13]: "ExBilişim",
    FILES[14]: "BabyMall telefon",
    FILES[15]: "Kamp reklamı (fatura değil)",
    FILES[16]: "BabyMall WA2",
    FILES[17]: "Nanobase logo (fatura değil)",
}


def multipart(path: Path) -> tuple[bytes, str]:
    boundary = "----userbatch7"
    data = path.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def issues(inv: dict, warnings: list) -> list[str]:
    out = []
    s = inv.get("supplier") or {}
    c = inv.get("customer") or {}
    t = inv.get("totals") or {}
    lines = inv.get("lines") or []
    if not inv.get("invoiceNumber"):
        out.append("no_invoiceNo")
    if not inv.get("issueDate"):
        out.append("no_date")
    if not inv.get("uuid"):
        out.append("no_ettn")
    if not s.get("name"):
        out.append("no_supplier_name")
    if not s.get("taxId"):
        out.append("no_supplier_tax")
    if not c.get("name"):
        out.append("no_customer_name")
    # placeholder customer tax is OK as filtered
    if c.get("taxId") in ("11111111111", "11111111110", "1111111111"):
        out.append("customer_placeholder_kept")
    if t.get("payableAmount") is None and t.get("taxInclusiveAmount") is None:
        out.append("no_payable")
    if not lines:
        out.append("no_lines")
    for ln in lines[:3]:
        nm = ln.get("name") or ""
        if re.search(r"(?i)IBAN|Kredi\s*Kart|Garanti\s*Bank|TR\d{2}", nm):
            out.append("line_looks_payment")
    for w in warnings:
        if "geçersiz" in w.lower():
            out.append("tax_checksum_fail")
            break
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in FILES:
        path = IN / name
        label = LABEL.get(name, name)
        row = {"file": name, "label": label, "exists": path.exists()}
        if not path.exists():
            row["error"] = "missing"
            rows.append(row)
            print("MISSING", label)
            continue
        body, ctype = multipart(path)
        req = urllib.request.Request(URL, data=body, headers={"Content-Type": ctype}, method="POST")
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                raw = resp.read()
            ms = int((time.perf_counter() - t0) * 1000)
            d = json.loads(raw)
            inv = d.get("invoice") or {}
            s = inv.get("supplier") or {}
            c = inv.get("customer") or {}
            t = inv.get("totals") or {}
            lines = inv.get("lines") or []
            warns = d.get("warnings") or []
            row.update(
                {
                    "ok_http": True,
                    "ms": ms,
                    "apiMs": d.get("durationMs"),
                    "status": d.get("status"),
                    "method": (d.get("method") or "")[:60],
                    "invoiceNumber": inv.get("invoiceNumber"),
                    "issueDate": inv.get("issueDate"),
                    "uuid": inv.get("uuid"),
                    "supplierName": (s.get("name") or "")[:50] or None,
                    "supplierTax": s.get("taxId"),
                    "customerName": (c.get("name") or "")[:40] or None,
                    "customerTax": c.get("taxId"),
                    "payable": t.get("payableAmount") or t.get("taxInclusiveAmount"),
                    "lineCount": len(lines),
                    "line0": (lines[0].get("name") if lines else None),
                    "warnings": warns[:8],
                    "issues": issues(inv, warns),
                }
            )
            (OUT / f"{path.stem}.json").write_text(
                json.dumps(d, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as e:
            ms = int((time.perf_counter() - t0) * 1000)
            row.update({"ok_http": False, "ms": ms, "error": str(e)[:240], "status": "error"})
        rows.append(row)
        print(
            f"{row.get('ms'):>6}ms {str(row.get('status')):8} "
            f"L={row.get('lineCount')} issues={row.get('issues')} | {label}",
            flush=True,
        )

    ms_list = [r["ms"] for r in rows if r.get("ms") is not None]
    summary = {
        "n": len(rows),
        "latencyMs": {
            "min": min(ms_list) if ms_list else None,
            "p50": sorted(ms_list)[len(ms_list) // 2] if ms_list else None,
            "max": max(ms_list) if ms_list else None,
            "mean": round(sum(ms_list) / len(ms_list), 1) if ms_list else None,
            "total": sum(ms_list) if ms_list else None,
        },
        "byStatus": {},
        "rows": rows,
    }
    for r in rows:
        st = r.get("status") or "?"
        summary["byStatus"][st] = summary["byStatus"].get(st, 0) + 1
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== SUMMARY ===")
    print(json.dumps({k: summary[k] for k in ("n", "latencyMs", "byStatus")}, indent=2))


if __name__ == "__main__":
    main()
