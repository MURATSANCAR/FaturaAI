import { Hono } from "hono";
import { cors } from "hono/cors";
import { serve } from "@hono/node-server";
import { extractInvoice } from "./extract.js";

const app = new Hono();

const allowedOrigins = (
  process.env.ALLOWED_ORIGINS ??
  "https://portal.nanobase.ai,http://localhost:5173,http://127.0.0.1:5173"
).split(",");

app.use(
  "*",
  cors({
    origin: (origin) => {
      if (!origin) return allowedOrigins[0];
      return allowedOrigins.includes(origin) ? origin : allowedOrigins[0];
    },
  }),
);

const ALLOWED_EXT =
  /\.(pdf|jpe?g|png|webp|heic|heif|tif|tiff|bmp)$/i;
const ALLOWED_MIME = new Set([
  "application/pdf",
  "image/jpeg",
  "image/png",
  "image/webp",
  "image/heic",
  "image/heif",
  "image/tiff",
  "image/bmp",
  "image/x-adobe-dng",
]);

function isAllowedUpload(file: File): boolean {
  if (ALLOWED_EXT.test(file.name)) return true;
  if (file.type && ALLOWED_MIME.has(file.type)) return true;
  // Mobile camera often sends empty type or generic image/*
  if (file.type === "image/*" || file.type.startsWith("image/")) return true;
  return false;
}

app.get("/health", (c) => c.json({ ok: true, service: "fatura-ai" }));

app.post("/extract", async (c) => {
  const body = await c.req.parseBody();
  const file = body.file;
  if (!file || !(file instanceof File)) {
    return c.json({ status: "failed", warnings: ["file alanı gerekli"], invoice: null }, 400);
  }
  if (!isAllowedUpload(file)) {
    return c.json(
      {
        status: "failed",
        warnings: ["PDF veya fatura fotoğrafı (JPG/PNG/WEBP/HEIC) yükleyin"],
        invoice: null,
      },
      400,
    );
  }
  const maxMb = Number(process.env.MAX_UPLOAD_MB ?? 20);
  if (file.size > maxMb * 1024 * 1024) {
    return c.json(
      { status: "failed", warnings: [`Dosya ${maxMb}MB limitini aşıyor`], invoice: null },
      400,
    );
  }

  const buffer = Buffer.from(await file.arrayBuffer());
  const result = await extractInvoice(buffer, file.name);
  return c.json(result);
});

const port = Number(process.env.PORT ?? 8105);
console.log(`FaturaAI API listening on :${port}`);
serve({ fetch: app.fetch, port, hostname: "127.0.0.1" });
