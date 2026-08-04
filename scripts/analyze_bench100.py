#!/usr/bin/env python3
import json
from collections import Counter
from pathlib import Path

s = json.loads(Path("/tmp/fatura-bench100/out/summary.json").read_text())
rows = [
    json.loads(l)
    for l in Path("/tmp/fatura-bench100/out/results.jsonl").read_text().splitlines()
    if l.strip()
]

print("=== KEY ===")
print(
    json.dumps(
        {
            k: s[k]
            for k in [
                "n",
                "wallSec",
                "workers",
                "sources",
                "status",
                "latencyMs",
                "completeCritical",
                "fieldMissingCounts",
                "issueCounts",
            ]
        },
        ensure_ascii=False,
        indent=2,
    )
)
print("\n=== TOP WARNINGS ===")
print(json.dumps(s.get("topWarnings"), ensure_ascii=False, indent=2))

print("\n=== BY SOURCE ===")
for src in ("local", "hf_synth"):
    sub = [r for r in rows if r.get("source") == src]
    st = Counter(r.get("status") for r in sub)
    ms = sorted(r["httpMs"] for r in sub if r.get("httpMs") is not None)

    def p(q: float):
        if not ms:
            return None
        return ms[min(len(ms) - 1, int(round(q / 100 * (len(ms) - 1))))]

    miss_lines = sum(1 for r in sub if "lines" in (r.get("missing") or []))
    print(
        src,
        "n",
        len(sub),
        "status",
        dict(st),
        "p50",
        p(50),
        "p90",
        p(90),
        "max",
        max(ms) if ms else None,
        "no_lines",
        miss_lines,
    )

print("\n=== SLOWEST ===")
for r in s["slowest"][:12]:
    method = (r.get("method") or "")[:42]
    print(
        f"{r['ms']:>6}ms  {str(r['status']):8}  {method:42}  {r['file'][:55]}"
    )

print("\n=== FAILED/ERROR ===")
for r in rows:
    if r.get("status") in ("failed", "error") or not r.get("ok"):
        print(r.get("status"), r.get("httpMs"), r.get("missing"), r.get("error"), r["file"][:70])

print("\n=== MISSING lines OR payable ===")
c = 0
for r in rows:
    miss = set(r.get("missing") or [])
    if miss & {"lines", "totals.payable"}:
        c += 1
        if c <= 20:
            print(r.get("status"), r.get("httpMs"), sorted(miss), r["file"][:65])
print("total", c)

print("\n=== ISSUES ===")
for r in rows:
    if r.get("issues"):
        print(r["file"][:55], r["issues"][:3])

# method buckets
print("\n=== METHOD PREFIX ===")
mc = Counter()
for r in rows:
    m = r.get("method") or "?"
    if "ocr" in m or "raster" in m or "photo" in m:
        mc["ocr_path"] += 1
    elif m == "nanobase-ai":
        mc["api_fast"] += 1
    else:
        mc["other"] += 1
print(dict(mc))
