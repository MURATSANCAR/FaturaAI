import { spawn } from "node:child_process";
import { mkdtemp, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const PDF_INSPECTOR_ENABLED = process.env.PDF_INSPECTOR_ENABLED !== "0";

const extractDir = join(
  dirname(fileURLToPath(import.meta.url)),
  "../../../extract",
);

export type PdfTextSource = "pdf-inspector" | "pdftotext";

export interface PdfTextResult {
  text: string;
  source: PdfTextSource;
  meta?: Record<string, unknown>;
}

export async function extractPdfText(buffer: Buffer): Promise<string> {
  const { text } = await extractPdfTextDetailed(buffer);
  return text;
}

export async function extractPdfTextDetailed(
  buffer: Buffer,
): Promise<PdfTextResult> {
  const dir = await mkdtemp(join(tmpdir(), "fatura-ai-"));
  const pdfPath = join(dir, "invoice.pdf");
  try {
    await writeFile(pdfPath, buffer);

    let inspectorText = "";
    let popplerText = "";

    // Run both extractors concurrently — they are independent subprocesses and
    // sequencing them just doubled fast-path latency on every PDF.
    const [inspectedRes, popplerRes] = await Promise.allSettled([
      PDF_INSPECTOR_ENABLED ? runPdfInspector(pdfPath) : Promise.resolve(null),
      runPdftotext(pdfPath),
    ]);

    if (
      inspectedRes.status === "fulfilled" &&
      inspectedRes.value &&
      inspectedRes.value.text.trim().length >= 40
    ) {
      inspectorText = inspectedRes.value.text;
    }
    if (popplerRes.status === "fulfilled") {
      popplerText = popplerRes.value;
    }

    if (inspectorText && popplerText.trim().length >= 40) {
      return {
        text: `${inspectorText}\n\n${popplerText}`,
        source: "pdf-inspector",
        meta: { also: "pdftotext", merged: true },
      };
    }
    if (inspectorText) {
      return { text: inspectorText, source: "pdf-inspector" };
    }
    if (popplerText.trim()) {
      return { text: popplerText, source: "pdftotext" };
    }
    return { text: "", source: "pdftotext" };
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

function runPdfInspector(pdfPath: string): Promise<PdfTextResult> {
  const script = `
import json, sys
from pathlib import Path
sys.path.insert(0, ${JSON.stringify(extractDir)})
from pdf_inspector_text import extract_pdf_inspector
text, meta = extract_pdf_inspector(Path(sys.argv[1]))
print(json.dumps({"text": text, "meta": meta}, ensure_ascii=False))
`;
  return new Promise((resolve, reject) => {
    const child = spawn("python3", ["-c", script, pdfPath], {
      stdio: ["ignore", "pipe", "pipe"],
      env: { ...process.env, PYTHONPATH: extractDir },
    });
    const chunks: Buffer[] = [];
    const errChunks: Buffer[] = [];
    child.stdout.on("data", (c) => chunks.push(c));
    child.stderr.on("data", (c) => errChunks.push(c));
    child.on("error", reject);
    child.on("close", (code) => {
      if (code !== 0) {
        reject(
          new Error(
            Buffer.concat(errChunks).toString("utf8") || "pdf-inspector failed",
          ),
        );
        return;
      }
      try {
        const raw = Buffer.concat(chunks).toString("utf8");
        const data = JSON.parse(raw) as {
          text?: string;
          meta?: Record<string, unknown>;
        };
        resolve({
          text: data.text ?? "",
          source: "pdf-inspector",
          meta: data.meta,
        });
      } catch (err) {
        reject(err);
      }
    });
  });
}

function runPdftotext(pdfPath: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const child = spawn("pdftotext", ["-layout", pdfPath, "-"], {
      stdio: ["ignore", "pipe", "pipe"],
    });
    const chunks: Buffer[] = [];
    const errChunks: Buffer[] = [];
    child.stdout.on("data", (c) => chunks.push(c));
    child.stderr.on("data", (c) => errChunks.push(c));
    child.on("error", (err) => {
      reject(
        new Error(
          "NanobaseAI PDF okuma bileşeni hazır değil. Lütfen daha sonra tekrar deneyin.",
        ),
      );
    });
    child.on("close", (code) => {
      if (code !== 0) {
        reject(
          new Error(
            "NanobaseAI PDF metin çıkarma başarısız oldu. Dosyayı kontrol edip tekrar deneyin.",
          ),
        );
        return;
      }
      resolve(Buffer.concat(chunks).toString("utf8"));
    });
  });
}

/** PDF binary içinde gömülü UBL Invoice XML arar. */
export function extractEmbeddedUblXml(buffer: Buffer): string | null {
  const asLatin = buffer.toString("latin1");
  const markers = ["<?xml", "<Invoice", "<cbc:Invoice", "<efatura:Invoice"];
  let start = -1;
  for (const m of markers) {
    const i = asLatin.indexOf(m);
    if (i >= 0 && (start < 0 || i < start)) start = i;
  }
  if (start < 0) return null;

  const slice = asLatin.slice(start, start + 2_000_000);
  const end =
    slice.search(/<\/(?:\w+:)?Invoice>/i) >= 0
      ? slice.search(/<\/(?:\w+:)?Invoice>/i) +
        slice.match(/<\/(?:\w+:)?Invoice>/i)![0].length
      : -1;
  if (end < 0) return null;
  const xml = slice.slice(0, end);
  if (!/CustomizationID|ProfileID|AccountingSupplierParty/i.test(xml)) return null;
  return Buffer.from(xml, "latin1").toString("utf8");
}
