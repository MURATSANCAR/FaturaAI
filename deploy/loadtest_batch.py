#!/usr/bin/env python3
"""FaturaAI batch load test against local API (:8105).

Builds/uses a corpus of N PDFs (hardlinks of 5 sources), submits jobs,
polls to completion, writes JSON + Markdown reports.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

API = os.environ.get("FATURA_API", "http://127.0.0.1:8105")
BOUNDARY = "----faturaLoadBoundary"


def http_json(method: str, url: str, data: bytes | None = None, headers: dict | None = None, timeout: float = 120):
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
        if not body:
            return resp.status, {}
        return resp.status, json.loads(body.decode())


def post_job(path: Path, timeout: float = 120) -> dict:
    raw = path.read_bytes()
    name = path.name
    body = (
        f"--{BOUNDARY}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{name}"\r\n'
        f"Content-Type: application/pdf\r\n\r\n"
    ).encode() + raw + f"\r\n--{BOUNDARY}--\r\n".encode()
    status, data = http_json(
        "POST",
        f"{API}/jobs?t={int(time.time()*1000)}",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"},
        timeout=timeout,
    )
    data["_http"] = status
    return data


def get_job(job_id: str, timeout: float = 60) -> dict:
    _, data = http_json("GET", f"{API}/jobs/{job_id}?t={int(time.time()*1000)}", timeout=timeout)
    return data


def _read_cpu() -> tuple[int, int]:
    """Return (idle, total) jiffies from /proc/stat aggregate cpu line."""
    with open("/proc/stat") as f:
        parts = f.readline().split()
    vals = [int(x) for x in parts[1:]]
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)  # idle + iowait
    total = sum(vals)
    return idle, total


def _read_mem() -> tuple[float, float]:
    """Return (used_gb, total_gb) from /proc/meminfo (MemTotal-MemAvailable)."""
    total_kb = avail_kb = 0
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                total_kb = int(line.split()[1])
            elif line.startswith("MemAvailable:"):
                avail_kb = int(line.split()[1])
    used_gb = (total_kb - avail_kb) / 1024 / 1024
    return used_gb, total_kb / 1024 / 1024


class ResourceSampler(threading.Thread):
    """Sample system-wide CPU% and RAM used every `interval` seconds."""

    def __init__(self, interval: float = 2.0):
        super().__init__(daemon=True)
        self.interval = interval
        # Do not name this `_stop` — that shadows threading.Thread._stop() and
        # breaks join(). Use a distinct attribute.
        self._stop_evt = threading.Event()
        self.cpu_samples: list[float] = []
        self.ram_used: list[float] = []
        self.ram_total_gb = 0.0
        self.cores = os.cpu_count() or 1

    def run(self) -> None:
        prev_idle, prev_total = _read_cpu()
        while not self._stop_evt.wait(self.interval):
            idle, total = _read_cpu()
            d_total = total - prev_total
            d_idle = idle - prev_idle
            prev_idle, prev_total = idle, total
            if d_total > 0:
                self.cpu_samples.append(100.0 * (d_total - d_idle) / d_total)
            used, tot = _read_mem()
            self.ram_used.append(used)
            self.ram_total_gb = tot

    def stop(self) -> None:
        self._stop_evt.set()

    def summary(self) -> dict:
        cpu = self.cpu_samples
        ram = self.ram_used
        return {
            "cores": self.cores,
            "samples": len(cpu),
            "cpuPctAvg": round(statistics.mean(cpu), 1) if cpu else None,
            "cpuPctPeak": round(max(cpu), 1) if cpu else None,
            "coresBusyAvg": round(statistics.mean(cpu) / 100 * self.cores, 1) if cpu else None,
            "coresBusyPeak": round(max(cpu) / 100 * self.cores, 1) if cpu else None,
            "ramTotalGb": round(self.ram_total_gb, 1),
            "ramUsedAvgGb": round(statistics.mean(ram), 1) if ram else None,
            "ramUsedPeakGb": round(max(ram), 1) if ram else None,
        }


def ensure_corpus(src_dir: Path, corpus_dir: Path, total: int) -> list[Path]:
    sources = sorted(src_dir.glob("*.pdf"))
    if len(sources) < 1:
        raise SystemExit(f"No PDFs in {src_dir}")
    corpus_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(corpus_dir.glob("inv-*.pdf"))
    if len(existing) >= total:
        return existing[:total]
    # Prefer hardlinks to save disk
    print(f"Building corpus {len(existing)}→{total} hardlinks in {corpus_dir} …", flush=True)
    for i in range(len(existing), total):
        src = sources[i % len(sources)]
        dest = corpus_dir / f"inv-{i:05d}-{src.stem}.pdf"
        if dest.exists():
            continue
        try:
            os.link(src, dest)
        except OSError:
            dest.write_bytes(src.read_bytes())
        if (i + 1) % 1000 == 0:
            print(f"  … {i+1}/{total}", flush=True)
    files = sorted(corpus_dir.glob("inv-*.pdf"))[:total]
    print(f"Corpus ready: {len(files)} files", flush=True)
    return files


def percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    if f == c:
        return ys[f]
    return ys[f] + (ys[c] - ys[f]) * (k - f)


def run_batch(
    files: list[Path],
    *,
    submit_workers: int,
    poll_workers: int,
    create_interval_ms: int,
    label: str,
    reports_dir: Path,
) -> dict:
    n = len(files)
    print(f"\n=== LOAD TEST {label}: {n} uploads ===", flush=True)
    t0 = time.time()
    sampler = ResourceSampler(interval=2.0)
    sampler.start()

    # health snapshot
    try:
        _, health0 = http_json("GET", f"{API}/health", timeout=10)
    except Exception as e:
        health0 = {"error": str(e)}

    job_ids: list[str | None] = [None] * n
    submit_ms: list[float] = [0.0] * n
    submit_errors = 0
    create_lock = threading.Lock()
    last_create = [0.0]

    def submit_one(idx: int, path: Path) -> None:
        nonlocal submit_errors
        # pace creates a bit to avoid stampeding the queue
        with create_lock:
            wait = max(0.0, (last_create[0] + create_interval_ms / 1000.0) - time.time())
            if wait:
                time.sleep(wait)
            last_create[0] = time.time()
        attempts = 0
        while True:
            attempts += 1
            t_s = time.time()
            try:
                data = post_job(path)
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")
                try:
                    data = json.loads(body)
                except Exception:
                    data = {"warnings": [body], "_http": e.code}
                data["_http"] = e.code
            except Exception as e:
                data = {"warnings": [str(e)], "_http": 0}

            http = data.get("_http") or 0
            jid = data.get("jobId")
            if jid:
                job_ids[idx] = jid
                submit_ms[idx] = (time.time() - t_s) * 1000
                return
            if http in (429, 503) and attempts < 200:
                time.sleep(min(30.0, 1.0 + attempts * 0.15))
                continue
            submit_errors += 1
            submit_ms[idx] = (time.time() - t_s) * 1000
            print(f"  submit fail #{idx} {path.name}: {data.get('warnings')}", flush=True)
            return

    with ThreadPoolExecutor(max_workers=submit_workers) as ex:
        futs = [ex.submit(submit_one, i, p) for i, p in enumerate(files)]
        done = 0
        for _ in as_completed(futs):
            done += 1
            if done % max(1, n // 20) == 0 or done == n:
                print(f"  submitted {done}/{n} (errors={submit_errors})", flush=True)

    t_submit_done = time.time()
    pending = {i: jid for i, jid in enumerate(job_ids) if jid}
    print(f"  submit phase {t_submit_done - t0:.1f}s — tracking {len(pending)} jobs", flush=True)

    results: dict[int, dict] = {}
    durations_ms: list[float] = []
    ok = partial = failed = missing = 0
    lock = threading.Lock()

    def poll_one(idx: int, jid: str) -> None:
        nonlocal ok, partial, failed, missing
        deadline = time.time() + 60 * 60  # 60 min per job max
        while time.time() < deadline:
            try:
                body = get_job(jid)
            except Exception:
                time.sleep(1.5)
                continue
            st = body.get("status")
            if st in ("done", "failed"):
                res = body.get("result") or {}
                estatus = res.get("status") or st
                dur = res.get("durationMs")
                with lock:
                    results[idx] = {
                        "jobId": jid,
                        "file": files[idx].name,
                        "jobStatus": st,
                        "extractStatus": estatus,
                        "durationMs": dur,
                        "method": res.get("method"),
                        "pipeline": res.get("pipeline"),
                        "warnings": (res.get("warnings") or [])[:5],
                        "error": body.get("error"),
                    }
                    if isinstance(dur, (int, float)):
                        durations_ms.append(float(dur))
                    if st == "failed" or estatus == "failed":
                        failed += 1
                    elif estatus == "partial":
                        partial += 1
                    else:
                        ok += 1
                return
            time.sleep(1.2)
        with lock:
            missing += 1
            results[idx] = {"jobId": jid, "file": files[idx].name, "jobStatus": "timeout"}

    with ThreadPoolExecutor(max_workers=poll_workers) as ex:
        futs = [ex.submit(poll_one, i, jid) for i, jid in pending.items()]
        done = 0
        for _ in as_completed(futs):
            done += 1
            if done % max(1, len(pending) // 20) == 0 or done == len(pending):
                elapsed = time.time() - t0
                print(
                    f"  completed {done}/{len(pending)} "
                    f"ok={ok} partial={partial} failed={failed} missing={missing} "
                    f"elapsed={elapsed:.0f}s",
                    flush=True,
                )

    t1 = time.time()
    sampler.stop()
    sampler.join(timeout=5)
    resources = sampler.summary()
    wall = t1 - t0
    submit_wall = t_submit_done - t0
    processed = ok + partial + failed
    throughput = processed / wall if wall > 0 else 0

    try:
        _, health1 = http_json("GET", f"{API}/health", timeout=10)
    except Exception as e:
        health1 = {"error": str(e)}

    report = {
        "label": label,
        "n": n,
        "api": API,
        "startedAt": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
        "finishedAt": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t1)),
        "wallSeconds": round(wall, 2),
        "submitWallSeconds": round(submit_wall, 2),
        "submitWorkers": submit_workers,
        "pollWorkers": poll_workers,
        "createIntervalMs": create_interval_ms,
        "counts": {
            "submitted": len(pending),
            "submitErrors": submit_errors,
            "ok": ok,
            "partial": partial,
            "failed": failed,
            "timeout": missing,
        },
        "throughputJobsPerSec": round(throughput, 3),
        "throughputJobsPerMin": round(throughput * 60, 1),
        "durationMs": {
            "count": len(durations_ms),
            "min": min(durations_ms) if durations_ms else None,
            "max": max(durations_ms) if durations_ms else None,
            "avg": round(statistics.mean(durations_ms), 1) if durations_ms else None,
            "p50": round(percentile(durations_ms, 50) or 0, 1) if durations_ms else None,
            "p95": round(percentile(durations_ms, 95) or 0, 1) if durations_ms else None,
            "p99": round(percentile(durations_ms, 99) or 0, 1) if durations_ms else None,
        },
        "submitMs": {
            "avg": round(statistics.mean(submit_ms), 1) if submit_ms else None,
            "p95": round(percentile(submit_ms, 95) or 0, 1) if submit_ms else None,
        },
        "healthBefore": health0,
        "healthAfter": health1,
        "resources": resources,
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    json_path = reports_dir / f"report-{label}-{stamp}.json"
    md_path = reports_dir / f"report-{label}-{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    md = f"""# FaturaAI Load Test — {label}

| Metric | Value |
|--------|-------|
| Uploads | {n} |
| Wall time | **{wall:.1f}s** ({wall/60:.1f} min) |
| Submit phase | {submit_wall:.1f}s |
| Throughput | **{throughput:.2f} job/s** ({throughput*60:.0f}/min) |
| OK / Partial / Failed / Timeout | {ok} / {partial} / {failed} / {missing} |
| Submit errors | {submit_errors} |
| Extract duration avg | {report['durationMs']['avg']} ms |
| Extract p50 / p95 / p99 | {report['durationMs']['p50']} / {report['durationMs']['p95']} / {report['durationMs']['p99']} ms |
| Extract min / max | {report['durationMs']['min']} / {report['durationMs']['max']} ms |
| Submit workers | {submit_workers} |
| Create interval | {create_interval_ms} ms |
| **CPU** (avg / peak) | **{resources['cpuPctAvg']}% / {resources['cpuPctPeak']}%** of {resources['cores']} cores ({resources['coresBusyAvg']} / {resources['coresBusyPeak']} cores busy) |
| **RAM used** (avg / peak) | **{resources['ramUsedAvgGb']} / {resources['ramUsedPeakGb']} GB** of {resources['ramTotalGb']} GB |

Started: `{report['startedAt']}` · Finished: `{report['finishedAt']}`
"""
    md_path.write_text(md)
    print(md, flush=True)
    print(f"Wrote {json_path}\nWrote {md_path}", flush=True)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/data/nanobaseai/fatura-loadtest/src")
    ap.add_argument("--corpus", default="/data/nanobaseai/fatura-loadtest/corpus-10k")
    ap.add_argument("--reports", default="/data/nanobaseai/fatura-loadtest/reports")
    ap.add_argument("--sizes", default="1000,5000,10000")
    ap.add_argument("--corpus-total", type=int, default=10000)
    ap.add_argument("--submit-workers", type=int, default=8)
    ap.add_argument("--poll-workers", type=int, default=32)
    ap.add_argument("--create-interval-ms", type=int, default=50)
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()

    files_all = ensure_corpus(Path(args.src), Path(args.corpus), args.corpus_total)
    if args.prepare_only:
        return 0

    reports_dir = Path(args.reports)
    summary = []
    for size_s in args.sizes.split(","):
        size = int(size_s.strip())
        if size > len(files_all):
            raise SystemExit(f"Need {size} files, corpus has {len(files_all)}")
        # Drain queue briefly between batches
        for _ in range(30):
            try:
                _, h = http_json("GET", f"{API}/health", timeout=10)
                q = (h.get("queue") or {})
                if (q.get("queued") or 0) == 0 and (q.get("inflight") or 0) == 0:
                    break
            except Exception:
                pass
            time.sleep(2)
        rep = run_batch(
            files_all[:size],
            submit_workers=args.submit_workers,
            poll_workers=args.poll_workers,
            create_interval_ms=args.create_interval_ms,
            label=f"{size}",
            reports_dir=reports_dir,
        )
        summary.append(rep)

    # Combined summary markdown
    lines = ["# FaturaAI Load Test Summary\n", "| Batch | N | Wall (s) | Wall (min) | job/s | job/min | OK | Partial | Failed | p50 ms | p95 ms | p99 ms | CPU avg/peak | cores busy | RAM avg/peak GB |",
             "|-------|---|----------|------------|-------|---------|----|---------|--------|--------|--------|--------|-------------|------------|-----------------|"]
    for r in summary:
        d = r["durationMs"]
        c = r["counts"]
        rs = r.get("resources") or {}
        lines.append(
            f"| {r['label']} | {r['n']} | {r['wallSeconds']} | {r['wallSeconds']/60:.1f} | "
            f"{r['throughputJobsPerSec']} | {r['throughputJobsPerMin']} | "
            f"{c['ok']} | {c['partial']} | {c['failed']} | "
            f"{d['p50']} | {d['p95']} | {d['p99']} | "
            f"{rs.get('cpuPctAvg')}%/{rs.get('cpuPctPeak')}% | "
            f"{rs.get('coresBusyAvg')}/{rs.get('coresBusyPeak')} | "
            f"{rs.get('ramUsedAvgGb')}/{rs.get('ramUsedPeakGb')} |"
        )
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(args.reports) / f"SUMMARY-{stamp}.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines), flush=True)
    print(f"\nSummary: {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
