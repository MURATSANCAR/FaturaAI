#!/usr/bin/env python3
"""CPU playground: PP-StructureV3 vs PaddleOCR-VL vs RapidOCR PP-OCRv6.

One engine per process (recommended). Example:
  python bench.py --engine structure --limit 5
  python bench.py --engine vl --limit 5
  python bench.py --engine rapid --limit 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

# Keep CPU thread sprawl under control (match prod extract knobs).
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("FLAGS_use_mkldnn", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLES = REPO_ROOT / "samples"
DEFAULT_OUT = Path(__file__).resolve().parent / "out"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# Turkish e-Arşiv / e-Fatura heuristics (field hit, not full parse).
_ETTN_RE = re.compile(
    r"\bETTN\b[:\s]*([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\b"
    r"|\b([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\b",
    re.I,
)
_VKN_RE = re.compile(r"\bVKN\b[:\s]*(\d{10,11})\b|\b(\d{10,11})\b")
_INVOICE_NO_RE = re.compile(
    r"(?:FATURA|FATERA|FATARA|BELGE)\s*N[O0]\s*[:\s]*([A-Z]{2,5}\d{10,16})"
    r"|\b([A-Z]{2,5}\d{10,16})\b",
    re.I,
)
_DATE_RE = re.compile(r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\b")
_AMOUNT_RE = re.compile(r"\d{1,3}(?:[.\s]\d{3})*[.,]\d{2}")
_PAYABLE_HINT_RE = re.compile(
    r"(?:[ÖO]DENECEK|ODENECEK|GENEL\s*TOPLAM|TOPLAM\s*TUTAR|TAX\s*INCLUSIVE)",
    re.I,
)
_TABLE_HINT_RE = re.compile(r"^\s*\|.+\|\s*$", re.M)
_STRUCT_HINT_RE = re.compile(
    r"\b(?:ETTN|VKN|KDV|TOPLAM|[ÖO]DENECEK|FATURA|SAY[İI]N|AL[İI]C[İI]|SAT[İI]C[İI])\b",
    re.I,
)


@dataclass
class FieldHits:
    ettn: bool = False
    vkn: bool = False
    invoiceNo: bool = False
    date: bool = False
    amount: bool = False
    payableHint: bool = False
    tableMarkdown: bool = False
    structHints: int = 0

    def score(self) -> float:
        flags = [
            self.ettn,
            self.vkn,
            self.invoiceNo,
            self.date,
            self.amount,
            self.payableHint,
            self.tableMarkdown,
        ]
        return round(sum(1 for x in flags if x) / len(flags), 3)


@dataclass
class Row:
    engine: str
    file: str
    ok: bool
    sec: float
    rss_gb: float
    peak_rss_gb: float
    chars: int
    fieldScore: float
    fields: dict[str, Any] = field(default_factory=dict)
    err: str | None = None
    out_md: str | None = None


def _rss_gb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e9
    except Exception:
        return 0.0


def _collect_images(root: Path, limit: int | None) -> list[Path]:
    files = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if limit is not None:
        files = files[: max(0, limit)]
    return files


def _analyze_text(text: str) -> FieldHits:
    hits = FieldHits()
    if not text:
        return hits
    hits.ettn = bool(_ETTN_RE.search(text))
    # Prefer labeled VKN; fall back to 10–11 digit near VKN keyword only.
    if re.search(r"\bVKN\b[:\s]*\d{10,11}\b", text, re.I):
        hits.vkn = True
    elif re.search(r"\bVKN\b", text, re.I) and _VKN_RE.search(text):
        hits.vkn = True
    hits.invoiceNo = bool(_INVOICE_NO_RE.search(text))
    hits.date = bool(_DATE_RE.search(text))
    hits.amount = bool(_AMOUNT_RE.search(text))
    hits.payableHint = bool(_PAYABLE_HINT_RE.search(text))
    hits.tableMarkdown = bool(_TABLE_HINT_RE.search(text))
    hits.structHints = len(_STRUCT_HINT_RE.findall(text))
    return hits


def _result_to_text(result: Any) -> str:
    """Best-effort flatten of paddleocr / rapidocr outputs to plain text."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    parts: list[str] = []

    def walk(obj: Any, depth: int = 0) -> None:
        if depth > 6:
            return
        if obj is None:
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
        # paddle result objects
        for attr in ("markdown", "text", "json", "str"):
            if hasattr(obj, attr):
                try:
                    val = getattr(obj, attr)
                    if callable(val) and attr == "str":
                        continue
                    if callable(val):
                        val = val()
                    walk(val, depth + 1)
                except Exception:
                    pass
        if hasattr(obj, "__dict__"):
            walk(vars(obj), depth + 1)

    walk(result)
    return "\n".join(parts)


def _save_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8")


def _init_structure(device: str, cpu_threads: int) -> Callable[[str], Any]:
    from paddleocr import PPStructureV3

    pipe = PPStructureV3(
        device=device,
        cpu_threads=cpu_threads,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )

    def predict(path: str) -> Any:
        return pipe.predict(input=path)

    return predict


def _init_vl(device: str, cpu_threads: int) -> Callable[[str], Any]:
    from paddleocr import PaddleOCRVL

    # Official API accepts device=cpu; VL is slow on CPU — expect minutes/page.
    try:
        pipe = PaddleOCRVL(
            device=device,
            cpu_threads=cpu_threads,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
    except TypeError:
        pipe = PaddleOCRVL(device=device)

    def predict(path: str) -> Any:
        return pipe.predict(input=path)

    return predict


def _preprocess_for_rapid(path: str) -> Any:
    """Light prod-like preprocess so RapidOCR sees a usable invoice page."""
    import cv2
    import numpy as np

    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"image decode failed: {path}")
    try:
        from PIL import Image, ImageOps

        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            img = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:
        pass

    h, w = img.shape[:2]
    long_side = max(h, w)
    # Prod-tuned defaults (photo_ocr.py): target 1800, max 2400, min 1200
    target = int(__import__("os").environ.get("PHOTO_OCR_TARGET_SIDE", "1800"))
    max_side = int(__import__("os").environ.get("PHOTO_OCR_MAX_SIDE", "2400"))
    min_side = int(__import__("os").environ.get("PHOTO_OCR_MIN_SIDE", "1200"))
    if long_side < min_side or long_side > max_side:
        scale = target / long_side
        interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=interp)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _init_rapid(threads: int) -> Callable[[str], Any]:
    from rapidocr import EngineType, ModelType, OCRVersion, RapidOCR
    import numpy as np

    # Mirror apps/extract/photo_ocr.py Small ladder (OpenVINO → ONNX).
    order = [EngineType.OPENVINO, EngineType.ONNXRUNTIME]
    last_err: Exception | None = None
    engine = None
    backend = None
    for eng in order:
        try:
            params = {
                "Det.engine_type": eng,
                "Det.lang_type": "tr",
                "Det.model_type": ModelType.SMALL,
                "Det.ocr_version": OCRVersion.PPOCRV6,
                "Rec.engine_type": eng,
                "Rec.lang_type": "tr",
                "Rec.model_type": ModelType.SMALL,
                "Rec.ocr_version": OCRVersion.PPOCRV6,
                "Cls.engine_type": eng,
                "EngineConfig.onnxruntime.intra_op_num_threads": threads,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
                "EngineConfig.openvino.inference_num_threads": threads,
                "EngineConfig.openvino.performance_hint": "LATENCY",
                "Global.log_level": "error",
            }
            engine = RapidOCR(params=params)
            engine(np.zeros((64, 64, 3), dtype=np.uint8))
            backend = getattr(eng, "name", str(eng))
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            engine = None
    if engine is None:
        raise RuntimeError(f"RapidOCR init failed: {last_err}")
    print(f"RapidOCR backend={backend}", flush=True)

    def predict(path: str) -> Any:
        img = _preprocess_for_rapid(path)
        result = engine(img)
        if hasattr(result, "txts") and result.txts is not None:
            return "\n".join(str(t) for t in result.txts if t)
        if hasattr(result, "to_markdown"):
            try:
                return result.to_markdown()
            except Exception:
                pass
        texts: list[str] = []
        payload = result[0] if isinstance(result, (list, tuple)) and result else result
        if payload is None:
            return ""
        try:
            for item in payload:
                if isinstance(item, dict):
                    t = item.get("text") or item.get("txt") or ""
                    if t:
                        texts.append(str(t))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    texts.append(str(item[1]))
        except TypeError:
            return str(result)
        return "\n".join(texts)

    return predict


def _init_engine(name: str, device: str, threads: int) -> Callable[[str], Any]:
    if name == "structure":
        print("Loading PPStructureV3 …", flush=True)
        return _init_structure(device, threads)
    if name == "vl":
        print("Loading PaddleOCRVL … (CPU can take a while)", flush=True)
        return _init_vl(device, threads)
    if name == "rapid":
        print("Loading RapidOCR PP-OCRv6 Small …", flush=True)
        return _init_rapid(threads)
    raise ValueError(f"unknown engine: {name}")


def _extract_and_save(
    engine: str,
    predict: Callable[[str], Any],
    image: Path,
    out_dir: Path,
) -> tuple[str, str | None]:
    raw = predict(str(image))
    text = _result_to_text(raw)
    md_path = out_dir / engine / f"{image.stem}.md"
    # Prefer native save_to_markdown when available.
    saved = False
    try:
        items = raw if isinstance(raw, (list, tuple)) else [raw]
        for res in items:
            if hasattr(res, "save_to_markdown"):
                target = out_dir / engine
                target.mkdir(parents=True, exist_ok=True)
                res.save_to_markdown(save_path=str(target))
                saved = True
            if hasattr(res, "save_to_json"):
                target = out_dir / engine
                target.mkdir(parents=True, exist_ok=True)
                res.save_to_json(save_path=str(target))
    except Exception:
        pass
    if not saved or not md_path.exists():
        # paddle may write differently; always keep a flat .md we control
        _save_markdown(md_path, text)
    else:
        # ensure our analyzer has text even if save path differs
        if not text.strip():
            try:
                # pick newest md under engine dir for this stem
                cands = sorted(
                    (out_dir / engine).rglob(f"*{image.stem}*.md"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if cands:
                    text = cands[0].read_text(encoding="utf-8", errors="ignore")
                    md_path = cands[0]
            except Exception:
                pass
        if text.strip():
            _save_markdown(md_path, text)
    return text, str(md_path)


def run_bench(args: argparse.Namespace) -> int:
    samples = Path(args.samples).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    images = _collect_images(samples, args.limit)
    if not images:
        print(f"No images under {samples}", file=sys.stderr)
        return 2

    print(
        f"engine={args.engine} device={args.device} threads={args.threads} "
        f"images={len(images)} samples={samples}",
        flush=True,
    )

    t_load0 = time.perf_counter()
    predict = _init_engine(args.engine, args.device, args.threads)
    print(f"load_sec={time.perf_counter() - t_load0:.1f} rss_gb={_rss_gb():.2f}", flush=True)

    rows: list[Row] = []
    peak = _rss_gb()

    for i, image in enumerate(images):
        # Optional warmup skip for timing stats (still recorded).
        print(f"[{i + 1}/{len(images)}] {image.relative_to(samples)}", flush=True)
        m0 = _rss_gb()
        t0 = time.perf_counter()
        ok = True
        err = None
        text = ""
        md_path = None
        try:
            text, md_path = _extract_and_save(args.engine, predict, image, out_dir)
        except Exception as e:  # noqa: BLE001
            ok = False
            err = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        sec = time.perf_counter() - t0
        m1 = _rss_gb()
        peak = max(peak, m0, m1)
        hits = _analyze_text(text)
        row = Row(
            engine=args.engine,
            file=str(image.relative_to(samples)),
            ok=ok,
            sec=round(sec, 3),
            rss_gb=round(m1, 3),
            peak_rss_gb=round(peak, 3),
            chars=len(text),
            fieldScore=hits.score(),
            fields=asdict(hits),
            err=err,
            out_md=md_path,
        )
        rows.append(row)
        print(
            f"  ok={ok} sec={row.sec} rss={row.rss_gb}GB "
            f"fieldScore={row.fieldScore} chars={row.chars}",
            flush=True,
        )
        if args.timeout_s and sec > args.timeout_s:
            print(f"  WARN: exceeded --timeout-s {args.timeout_s}", flush=True)

    timed = [r for r in rows[ int(args.skip_warmup) : ] if r.ok]
    summary = {
        "engine": args.engine,
        "device": args.device,
        "threads": args.threads,
        "samples": str(samples),
        "count": len(rows),
        "ok": sum(1 for r in rows if r.ok),
        "fail": sum(1 for r in rows if not r.ok),
        "peak_rss_gb": round(peak, 3),
        "sec": {
            "p50": round(statistics.median([r.sec for r in timed]), 3) if timed else None,
            "mean": round(statistics.mean([r.sec for r in timed]), 3) if timed else None,
            "p95": round(
                statistics.quantiles([r.sec for r in timed], n=20)[18], 3
            )
            if len(timed) >= 20
            else (max(r.sec for r in timed) if timed else None),
            "sum": round(sum(r.sec for r in rows), 3),
        },
        "fieldScore": {
            "mean": round(statistics.mean([r.fieldScore for r in rows if r.ok]), 3)
            if any(r.ok for r in rows)
            else None,
        },
        "fieldHits": {
            k: sum(1 for r in rows if r.ok and r.fields.get(k))
            for k in (
                "ettn",
                "vkn",
                "invoiceNo",
                "date",
                "amount",
                "payableHint",
                "tableMarkdown",
            )
        },
        "rows": [asdict(r) for r in rows],
    }

    report_path = out_dir / f"report-{args.engine}.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "rows"}, indent=2, ensure_ascii=False))
    print(f"Wrote {report_path}", flush=True)
    return 0 if summary["fail"] == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FaturaAI OCR CPU playground")
    p.add_argument(
        "--engine",
        choices=("structure", "vl", "rapid"),
        required=True,
        help="structure=PPStructureV3, vl=PaddleOCR-VL, rapid=prod RapidOCR PP-OCRv6",
    )
    p.add_argument("--samples", default=str(DEFAULT_SAMPLES), help="Image root")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="Output directory")
    p.add_argument("--limit", type=int, default=None, help="Max images")
    p.add_argument("--device", default="cpu", help="cpu | gpu:0 (GPU optional)")
    p.add_argument("--threads", type=int, default=int(os.getenv("OCR_BENCH_THREADS", "4")))
    p.add_argument("--skip-warmup", type=int, default=1, help="Exclude first N from p50/mean")
    p.add_argument("--timeout-s", type=float, default=300.0, help="Warn if page slower than this")
    return p


def main() -> int:
    args = build_parser().parse_args()
    return run_bench(args)


if __name__ == "__main__":
    raise SystemExit(main())
