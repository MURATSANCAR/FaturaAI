import { Hono } from "hono";
import { cors } from "hono/cors";
import { serve } from "@hono/node-server";
import { extractInvoice } from "./extract.js";
import { enqueueJob, getJob, queueStats } from "./jobs.js";
import { takeRateToken } from "./lib/rate-limit.js";

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
  if (file.type === "image/*" || file.type.startsWith("image/")) return true;
  return false;
}

function clientKey(c: { req: { header: (name: string) => string | undefined } }): string {
  const xf = c.req.header("x-forwarded-for");
  if (xf) return xf.split(",")[0]?.trim() || "unknown";
  return c.req.header("x-real-ip") || "local";
}

function allowWithoutRateLimit(key: string): boolean {
  return key === "local" || key === "127.0.0.1" || key === "::1";
}

async function readUpload(c: {
  req: { parseBody: () => Promise<Record<string, unknown>> };
}): Promise<{ file: File } | { error: string; status: number }> {
  const body = await c.req.parseBody();
  const file = body.file;
  if (!file || !(file instanceof File)) {
    return { error: "file alanı gerekli", status: 400 };
  }
  if (!isAllowedUpload(file)) {
    return { error: "PDF veya fatura fotoğrafı (JPG/PNG/WEBP/HEIC) yükleyin", status: 400 };
  }
  const maxMb = Number(process.env.MAX_UPLOAD_MB ?? 20);
  if (file.size > maxMb * 1024 * 1024) {
    return { error: `Dosya ${maxMb}MB limitini aşıyor`, status: 400 };
  }
  return { file };
}

app.get("/health", (c) =>
  c.json({
    ok: true,
    service: "fatura-ai",
    queue: queueStats(),
    pdfFastPath: process.env.PDF_FAST_PATH !== "0",
  }),
);

app.get("/metrics", (c) => {
  const q = queueStats();
  const lines = [
    `# HELP fatura_queue_queued Jobs waiting`,
    `# TYPE fatura_queue_queued gauge`,
    `fatura_queue_queued ${q.queued}`,
    `# HELP fatura_queue_inflight Jobs running`,
    `# TYPE fatura_queue_inflight gauge`,
    `fatura_queue_inflight ${q.inflight}`,
    `# HELP fatura_jobs_processed_total Completed jobs`,
    `# TYPE fatura_jobs_processed_total counter`,
    `fatura_jobs_processed_total ${q.processed}`,
    `# HELP fatura_jobs_failed_total Failed jobs`,
    `# TYPE fatura_jobs_failed_total counter`,
    `fatura_jobs_failed_total ${q.failed}`,
    `# HELP fatura_fast_path_hits_total PDF fast-path successes`,
    `# TYPE fatura_fast_path_hits_total counter`,
    `fatura_fast_path_hits_total ${q.fastPathHits}`,
    `# HELP fatura_v2_hits_total Docling extract calls`,
    `# TYPE fatura_v2_hits_total counter`,
    `fatura_v2_hits_total ${q.v2Hits}`,
  ];
  return c.text(lines.join("\n") + "\n", 200, {
    "content-type": "text/plain; version=0.0.4",
  });
});

/** Async job create — preferred by UI */
app.post("/jobs", async (c) => {
  const key = clientKey(c);
  if (!allowWithoutRateLimit(key)) {
    const rl = takeRateToken(key);
    if (!rl.ok) {
      c.header("Retry-After", String(rl.retryAfterSec));
      return c.json(
        { status: "failed", warnings: ["Çok fazla istek. Lütfen biraz sonra tekrar deneyin."], jobId: null },
        429,
      );
    }
  }
  const uploaded = await readUpload(c);
  if ("error" in uploaded) {
    return c.json(
      { status: "failed", warnings: [uploaded.error], jobId: null },
      uploaded.status as 400,
    );
  }
  try {
    const buffer = Buffer.from(await uploaded.file.arrayBuffer());
    const job = await enqueueJob(buffer, uploaded.file.name || "invoice.pdf");
    return c.json(
      {
        jobId: job.id,
        status: job.status,
        queuePosition: job.queuePosition,
        fileName: job.fileName,
      },
      202,
    );
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    const code = (err as Error & { code?: string }).code;
    return c.json(
      { status: "failed", warnings: [msg], jobId: null },
      code === "QUEUE_FULL" ? 503 : 500,
    );
  }
});

app.get("/jobs/:id", (c) => {
  const job = getJob(c.req.param("id"));
  if (!job) return c.json({ status: "failed", warnings: ["Job bulunamadı"] }, 404);
  return c.json(job);
});

/** Sync extract — kept for scripts/load tests; still rate-limited */
app.post("/extract", async (c) => {
  const key = clientKey(c);
  if (!allowWithoutRateLimit(key)) {
    const rl = takeRateToken(key);
    if (!rl.ok) {
      c.header("Retry-After", String(rl.retryAfterSec));
      return c.json(
        { status: "failed", warnings: ["Çok fazla istek. Lütfen biraz sonra tekrar deneyin."], invoice: null },
        429,
      );
    }
  }
  const uploaded = await readUpload(c);
  if ("error" in uploaded) {
    return c.json(
      { status: "failed", warnings: [uploaded.error], invoice: null },
      uploaded.status as 400,
    );
  }
  const buffer = Buffer.from(await uploaded.file.arrayBuffer());
  const result = await extractInvoice(buffer, uploaded.file.name || "invoice.pdf");
  return c.json(result);
});

const port = Number(process.env.PORT ?? 8105);
console.log(`FaturaAI API listening on :${port}`);
serve({ fetch: app.fetch, port, hostname: "127.0.0.1" });
