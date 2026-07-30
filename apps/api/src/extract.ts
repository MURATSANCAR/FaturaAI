import { extractEmbeddedUblXml, extractPdfText } from "./lib/pdf.js";
import { parseGibPdfText, validateInvoice } from "./lib/parse-pdf-text.js";
import { parseUblInvoice } from "./lib/parse-ubl.js";
import { noteFastPath, noteV2 } from "./lib/metrics.js";
import type { ExtractResult, ParsedInvoice } from "./types.js";

const EXTRACT_V2_URL = process.env.EXTRACT_V2_URL ?? "http://127.0.0.1:8106";
const EXTRACT_V2_ENABLED = process.env.EXTRACT_V2_ENABLED !== "0";
/** Prefer fast local PDF parse before Docling (prod default on). */
const PDF_FAST_PATH = process.env.PDF_FAST_PATH !== "0";

async function extractViaV2(
  buffer: Buffer,
  fileName: string,
): Promise<ExtractResult | null> {
  if (!EXTRACT_V2_ENABLED) return null;
  try {
    const form = new FormData();
    form.append("file", new Blob([new Uint8Array(buffer)]), fileName);
    form.append("filename", fileName);
    const res = await fetch(`${EXTRACT_V2_URL.replace(/\/$/, "")}/extract`, {
      method: "POST",
      body: form,
      signal: AbortSignal.timeout(180_000),
    });
    if (!res.ok) return null;
    const data = (await res.json()) as ExtractResult & {
      validation?: ExtractResult["validation"];
      pipeline?: string[];
    };
    if (!data || typeof data.status !== "string") return null;
    noteV2();
    return {
      status: data.status,
      method: data.method || "docling",
      durationMs: data.durationMs ?? 0,
      warnings: data.warnings ?? [],
      invoice: data.invoice,
      rawTextPreview: data.rawTextPreview ?? null,
      validation: data.validation ?? null,
      pipeline: data.pipeline ?? [],
    };
  } catch {
    return null;
  }
}

function legacyResult(
  invoice: ParsedInvoice | null,
  method: ExtractResult["method"],
  durationMs: number,
  warnings: string[],
  preview: string | null,
): ExtractResult {
  const critical = warnings.filter((w) =>
    /Fatura numarası|Ödenecek tutar|Satıcı|Alıcı|kalemi/.test(w),
  );
  const status =
    !invoice
      ? "failed"
      : critical.length > 0
        ? "partial"
        : warnings.length > 2
          ? "partial"
          : "ok";
  return {
    status,
    method,
    durationMs,
    warnings,
    invoice,
    rawTextPreview: preview,
    validation: null,
    pipeline: ["legacy", method],
  };
}

function isImageFileName(fileName: string): boolean {
  return /\.(jpe?g|png|webp|heic|heif|tif|tiff|bmp)$/i.test(fileName);
}

function sniffImageExt(buffer: Buffer): string | null {
  if (buffer.length >= 3 && buffer[0] === 0xff && buffer[1] === 0xd8 && buffer[2] === 0xff) {
    return ".jpg";
  }
  if (
    buffer.length >= 8 &&
    buffer[0] === 0x89 &&
    buffer[1] === 0x50 &&
    buffer[2] === 0x4e &&
    buffer[3] === 0x47
  ) {
    return ".png";
  }
  if (
    buffer.length >= 12 &&
    buffer.subarray(0, 4).toString("ascii") === "RIFF" &&
    buffer.subarray(8, 12).toString("ascii") === "WEBP"
  ) {
    return ".webp";
  }
  if (buffer.length >= 12 && buffer.subarray(4, 8).toString("ascii") === "ftyp") {
    const brand = buffer.subarray(8, 12).toString("ascii");
    if (["heic", "heif", "mif1", "msf1", "heix", "heim"].includes(brand)) {
      return ".heic";
    }
  }
  return null;
}

function normalizeUploadName(buffer: Buffer, fileName: string): string {
  if (/\.pdf$/i.test(fileName) || isImageFileName(fileName)) return fileName;
  if (buffer.subarray(0, 4).toString("latin1") === "%PDF") {
    return fileName.includes(".") ? fileName : "invoice.pdf";
  }
  const imgExt = sniffImageExt(buffer);
  if (imgExt) {
    const base = fileName.replace(/\.[^.]+$/, "") || "invoice";
    return `${base}${imgExt}`;
  }
  return fileName;
}

function isStrongResult(result: ExtractResult): boolean {
  const inv = result.invoice;
  if (!inv || result.status === "failed") return false;
  const critical = (result.warnings ?? []).filter((w) =>
    /Fatura numarası|Ödenecek tutar|Satıcı|Alıcı|kalemi/.test(w),
  );
  if (critical.length > 0) return false;
  return Boolean(
    inv.invoiceNumber &&
      inv.totals.payableAmount != null &&
      inv.supplier.name &&
      inv.customer.name &&
      (inv.lines?.length ?? 0) > 0,
  );
}

async function extractLegacyPdf(
  buffer: Buffer,
  name: string,
  started: number,
): Promise<ExtractResult> {
  const warnings: string[] = [];
  const embedded = extractEmbeddedUblXml(buffer);
  if (embedded) {
    const invoice = parseUblInvoice(embedded);
    if (invoice) {
      warnings.push(...validateInvoice(invoice));
      return legacyResult(
        invoice,
        "ubl",
        Math.round(performance.now() - started),
        warnings,
        embedded.slice(0, 1500),
      );
    }
  }

  const text = await extractPdfText(buffer);
  if (!text.trim()) {
    return legacyResult(
      null,
      "pdf-text",
      Math.round(performance.now() - started),
      ["PDF metni çıkarılamadı (boş çıktı)"],
      null,
    );
  }

  const invoice = parseGibPdfText(text, name);
  warnings.push(...validateInvoice(invoice));
  return legacyResult(
    invoice,
    "pdf-text",
    Math.round(performance.now() - started),
    warnings,
    text.slice(0, 2500),
  );
}

export async function extractInvoice(
  buffer: Buffer,
  fileName: string,
): Promise<ExtractResult> {
  const started = performance.now();
  const name = normalizeUploadName(buffer, fileName);
  const asImage = isImageFileName(name) || Boolean(sniffImageExt(buffer));

  // PDF fast-path: local pdftotext/UBL before expensive Docling
  if (!asImage && PDF_FAST_PATH) {
    try {
      const fast = await extractLegacyPdf(buffer, name, started);
      if (isStrongResult(fast)) {
        noteFastPath();
        return {
          ...fast,
          method: fast.method === "ubl" ? "ubl-fast" : "pdf-text-fast",
          pipeline: [...(fast.pipeline ?? []), "fast-path"],
          durationMs: Math.round(performance.now() - started),
        };
      }
      // Weak PDF → enrich with Docling
      const v2 = await extractViaV2(buffer, name);
      if (v2 && v2.status !== "failed") {
        return {
          ...v2,
          durationMs: Math.round(performance.now() - started),
        };
      }
      // Keep best available
      if (fast.status !== "failed") {
        noteFastPath();
        return {
          ...fast,
          durationMs: Math.round(performance.now() - started),
        };
      }
      if (v2) {
        return {
          ...v2,
          durationMs: Math.round(performance.now() - started),
        };
      }
      return fast;
    } catch (err) {
      const v2 = await extractViaV2(buffer, name);
      if (v2) {
        return {
          ...v2,
          durationMs: Math.round(performance.now() - started),
        };
      }
      return legacyResult(
        null,
        "pdf-text",
        Math.round(performance.now() - started),
        [err instanceof Error ? err.message : String(err)],
        null,
      );
    }
  }

  // Images / fast-path disabled → Docling first
  const v2 = await extractViaV2(buffer, name);
  if (v2 && v2.status !== "failed") {
    return {
      ...v2,
      durationMs: Math.round(performance.now() - started),
    };
  }
  if (v2 && asImage) {
    return {
      ...v2,
      durationMs: Math.round(performance.now() - started),
    };
  }
  if (asImage) {
    return legacyResult(
      null,
      "docling",
      Math.round(performance.now() - started),
      [
        v2?.warnings?.[0] ??
          "Fotoğraf okuma servisi yanıt vermedi. PDF deneyin veya tekrar deneyin.",
      ],
      null,
    );
  }

  try {
    return await extractLegacyPdf(buffer, name, started);
  } catch (err) {
    return legacyResult(
      null,
      "pdf-text",
      Math.round(performance.now() - started),
      [err instanceof Error ? err.message : String(err)],
      null,
    );
  }
}
