#!/usr/bin/env python3
"""Bench fatura extract on ~100 e-Arşiv-style samples; timing + error report."""
from __future__ import annotations

import json
import re
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

IN_DIR = Path("/tmp/fatura-bench100/in")
OUT_DIR = Path("/tmp/fatura-bench100/out")
URL = "http://127.0.0.1:8105/extract"
N = 100
WORKERS = 2
TIMEOUT = 180

EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


def pick_files(n: int) -> list[Path]:
    locals_ = sorted(IN_DIR.glob("local_*"))
    hfs = sorted(IN_DIR.glob("hf_*"))
    picked = [p for p in locals_ if p.suffix.lower() in EXTS]
    for p in hfs:
        if len(picked) >= n:
            break
        if p.suffix.lower() in EXTS:
            picked.append(p)
    return picked[:n]


def multipart(path: Path) -> tuple[bytes, str]:
    boundary = "----benchboundary7MA4YWxk"
    data = path.read_bytes()
    ctype = {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }.get(path.suffix.lower(), "application/octet-stream")
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        f"Content-Type: {ctype}\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def score_invoice(inv: dict) -> dict:
    s = inv.get("supplier") or {}
    c = inv.get("customer") or {}
    t = inv.get("totals") or {}
    lines = inv.get("lines") or []
    missing = []
    if not inv.get("invoiceNumber"):
        missing.append("invoiceNumber")
    if not inv.get("issueDate"):
        missing.append("issueDate")
    if not s.get("name"):
        missing.append("supplier.name")
    if not s.get("taxId"):
        missing.append("supplier.taxId")
    if not c.get("name"):
        missing.append("customer.name")
    if t.get("payableAmount") is None and t.get("taxInclusiveAmount") is None:
        missing.append("totals.payable")
    if not lines:
        missing.append("lines")
    issues = []
    for ln in lines:
        name = ln.get("name") or ""
        if re.search(
            r"(?i)\bIBAN\b|\bTR\d{2}\b|Kredi\s*Kart|Banka\s*Kart|Garanti\s*Bank",
            name,
        ):
            issues.append("line_looks_like_payment:" + name[:60])
        if re.search(r"(?i)e-?Belge|<!--\s*image|Ma[gğ]aza\s*:", name):
            issues.append("line_looks_like_chrome:" + name[:60])
    for side, party in (("supplier", s), ("customer", c)):
        nm = party.get("name") or ""
        if re.search(r"(?i)e-?Belge|YALNIZ|<!--|table|UBL", nm):
            issues.append(f"{side}_chrome_name")
        if party.get("taxId") in ("11111111111", "0000000000", "1111111111"):
            issues.append(f"{side}_placeholder_taxId")
    return {
        "missing": missing,
        "issues": issues,
        "lineCount": len(lines),
        "payable": t.get("payableAmount") or t.get("taxInclusiveAmount"),
        "supplierName": (s.get("name") or "")[:80] or None,
        "customerName": (c.get("name") or "")[:80] or None,
        "invoiceNumber": inv.get("invoiceNumber"),
    }


def run_one(path: Path) -> dict:
    body, content_type = multipart(path)
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": content_type}, method="POST"
    )
    t0 = time.perf_counter()
    row = {
        "file": path.name,
        "bytes": path.stat().st_size,
        "source": "local" if path.name.startswith("local_") else "hf_synth",
    }
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            raw = resp.read()
        ms = int((time.perf_counter() - t0) * 1000)
        d = json.loads(raw)
        inv = d.get("invoice") or {}
        sc = score_invoice(inv)
        row.update(
            {
                "ok": True,
                "httpMs": ms,
                "apiDurationMs": d.get("durationMs"),
                "status": d.get("status"),
                "method": d.get("method"),
                "warnings": (d.get("warnings") or [])[:12],
                "confidence": (d.get("validation") or {}).get("confidence"),
                **sc,
            }
        )
        (OUT_DIR / f"{path.name}.json").write_text(
            json.dumps({"file": path.name, "ms": ms, "response": d}, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as e:
        ms = int((time.perf_counter() - t0) * 1000)
        row.update(
            {"ok": False, "httpMs": ms, "error": str(e)[:300], "status": "error"}
        )
    return row


def pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    i = min(len(s) - 1, max(0, int(round((p / 100) * (len(s) - 1)))))
    return float(s[i])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = pick_files(N)
    print(f"BENCH start n={len(files)} workers={WORKERS} url={URL}", flush=True)
    rows: list[dict] = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(run_one, p): p for p in files}
        done = 0
        for fut in as_completed(futs):
            row = fut.result()
            rows.append(row)
            done += 1
            print(
                f"[{done}/{len(files)}] {row.get('status')} {row.get('httpMs')}ms "
                f"L={row.get('lineCount')} miss={row.get('missing')} {row['file'][:60]}",
                flush=True,
            )
    wall = time.perf_counter() - t0
    rows.sort(key=lambda r: r["file"])
    ms_list = [r["httpMs"] for r in rows if r.get("httpMs") is not None]
    by_status: dict[str, int] = {}
    for r in rows:
        by_status[r.get("status") or "?"] = by_status.get(r.get("status") or "?", 0) + 1
    miss_counts: dict[str, int] = {}
    issue_counts: dict[str, int] = {}
    warn_counts: dict[str, int] = {}
    for r in rows:
        for m in r.get("missing") or []:
            miss_counts[m] = miss_counts.get(m, 0) + 1
        for i in r.get("issues") or []:
            key = i.split(":")[0]
            issue_counts[key] = issue_counts.get(key, 0) + 1
        for w in r.get("warnings") or []:
            warn_counts[w] = warn_counts.get(w, 0) + 1
    summary = {
        "n": len(rows),
        "wallSec": round(wall, 1),
        "workers": WORKERS,
        "sources": {
            "local": sum(1 for r in rows if r.get("source") == "local"),
            "hf_synth": sum(1 for r in rows if r.get("source") == "hf_synth"),
        },
        "status": by_status,
        "latencyMs": {
            "min": min(ms_list) if ms_list else None,
            "p50": pct(ms_list, 50),
            "p90": pct(ms_list, 90),
            "p95": pct(ms_list, 95),
            "max": max(ms_list) if ms_list else None,
            "mean": round(statistics.mean(ms_list), 1) if ms_list else None,
        },
        "fieldMissingCounts": dict(sorted(miss_counts.items(), key=lambda x: -x[1])),
        "issueCounts": dict(sorted(issue_counts.items(), key=lambda x: -x[1])),
        "topWarnings": dict(sorted(warn_counts.items(), key=lambda x: -x[1])[:15]),
        "completeCritical": sum(
            1
            for r in rows
            if r.get("ok")
            and not set(r.get("missing") or [])
            & {
                "invoiceNumber",
                "totals.payable",
                "supplier.name",
                "lines",
            }
        ),
        "slowest": sorted(
            [
                {
                    "file": r["file"],
                    "ms": r.get("httpMs"),
                    "status": r.get("status"),
                    "method": r.get("method"),
                    "missing": r.get("missing"),
                }
                for r in rows
            ],
            key=lambda x: -(x.get("ms") or 0),
        )[:15],
        "problemCases": [
            {
                "file": r["file"],
                "status": r.get("status"),
                "ms": r.get("httpMs"),
                "missing": r.get("missing"),
                "issues": r.get("issues"),
                "warnings": r.get("warnings"),
                "supplierName": r.get("supplierName"),
                "lineCount": r.get("lineCount"),
                "error": r.get("error"),
            }
            for r in rows
            if (not r.get("ok"))
            or r.get("status") in ("failed", "error")
            or r.get("issues")
            or set(r.get("missing") or [])
            & {"lines", "totals.payable", "supplier.name"}
        ][:40],
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
