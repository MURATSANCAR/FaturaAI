import { extractEmbeddedUblXml, extractPdfText } from "./lib/pdf.js";
import { parseGibPdfText, validateInvoice } from "./lib/parse-pdf-text.js";
import { parseUblInvoice } from "./lib/parse-ubl.js";
import type { ExtractResult, ParsedInvoice } from "./types.js";

const EXTRACT_V2_URL = process.env.EXTRACT_V2_URL ?? "http://127.0.0.1:8106";
const EXTRACT_V2_ENABLED = process.env.EXTRACT_V2_ENABLED !== "0";

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

export async function extractInvoice(
  buffer: Buffer,
  fileName: string,
): Promise<ExtractResult> {
  const started = performance.now();

  const v2 = await extractViaV2(buffer, fileName);
  if (v2 && v2.status !== "failed") {
    return {
      ...v2,
      durationMs: Math.round(performance.now() - started),
    };
  }

  const warnings: string[] = [];
  try {
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

    const invoice = parseGibPdfText(text, fileName);
    warnings.push(...validateInvoice(invoice));
    return legacyResult(
      invoice,
      "pdf-text",
      Math.round(performance.now() - started),
      warnings,
      text.slice(0, 2500),
    );
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
