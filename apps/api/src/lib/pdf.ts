import { spawn } from "node:child_process";
import { mkdtemp, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

export async function extractPdfText(buffer: Buffer): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), "fatura-ai-"));
  const pdfPath = join(dir, "invoice.pdf");
  try {
    await writeFile(pdfPath, buffer);
    return await runPdftotext(pdfPath);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
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
          `pdftotext bulunamadı. poppler-utils kurulu olmalı: ${err.message}`,
        ),
      );
    });
    child.on("close", (code) => {
      if (code !== 0) {
        reject(
          new Error(
            `pdftotext failed (${code}): ${Buffer.concat(errChunks).toString("utf8")}`,
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
