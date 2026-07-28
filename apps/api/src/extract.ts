import { extractEmbeddedUblXml, extractPdfText } from "./lib/pdf.js";
import { parseGibPdfText, validateInvoice } from "./lib/parse-pdf-text.js";
import { parseUblInvoice } from "./lib/parse-ubl.js";
import type { ExtractResult } from "./types.js";

export async function extractInvoice(
  buffer: Buffer,
  fileName: string,
): Promise<ExtractResult> {
  const started = performance.now();
  const warnings: string[] = [];

  try {
    const embedded = extractEmbeddedUblXml(buffer);
    if (embedded) {
      const invoice = parseUblInvoice(embedded);
      if (invoice) {
        const v = validateInvoice(invoice);
        warnings.push(...v);
        const durationMs = Math.round(performance.now() - started);
        return {
          status: v.length === 0 ? "ok" : warnings.some((w) => /bulunamadı/.test(w)) ? "partial" : "ok",
          method: "ubl",
          durationMs,
          warnings,
          invoice,
          rawTextPreview: embedded.slice(0, 1500),
        };
      }
    }

    const text = await extractPdfText(buffer);
    if (!text.trim()) {
      return {
        status: "failed",
        method: "pdf-text",
        durationMs: Math.round(performance.now() - started),
        warnings: ["PDF metni çıkarılamadı (boş çıktı)"],
        invoice: null,
        rawTextPreview: null,
      };
    }

    const invoice = parseGibPdfText(text, fileName);
    const v = validateInvoice(invoice);
    warnings.push(...v);
    const critical = v.filter((w) =>
      /Fatura numarası|Ödenecek tutar|Satıcı|Alıcı/.test(w),
    );
    const status =
      critical.length > 0 ? "partial" : v.length > 2 ? "partial" : "ok";

    return {
      status,
      method: "pdf-text",
      durationMs: Math.round(performance.now() - started),
      warnings,
      invoice,
      rawTextPreview: text.slice(0, 2500),
    };
  } catch (err) {
    return {
      status: "failed",
      method: "pdf-text",
      durationMs: Math.round(performance.now() - started),
      warnings: [err instanceof Error ? err.message : String(err)],
      invoice: null,
      rawTextPreview: null,
    };
  }
}
