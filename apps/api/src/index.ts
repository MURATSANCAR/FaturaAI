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

app.get("/health", (c) => c.json({ ok: true, service: "fatura-ai" }));

app.post("/extract", async (c) => {
  const body = await c.req.parseBody();
  const file = body.file;
  if (!file || !(file instanceof File)) {
    return c.json({ status: "failed", warnings: ["file alanı gerekli"], invoice: null }, 400);
  }
  if (!/\.pdf$/i.test(file.name) && file.type !== "application/pdf") {
    return c.json({ status: "failed", warnings: ["Sadece PDF kabul edilir"], invoice: null }, 400);
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
