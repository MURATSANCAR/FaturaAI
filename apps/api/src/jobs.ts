import { randomUUID } from "node:crypto";
import { mkdir, writeFile, unlink, rm } from "node:fs/promises";
import { join } from "node:path";
import { extractInvoice } from "./extract.js";
import type { ExtractResult } from "./types.js";
import { pathHitStats } from "./lib/metrics.js";
import { sanitizePublicMessage, toPublicExtractResult } from "./lib/public-facing.js";

export type JobStatus = "queued" | "running" | "done" | "failed";

export type JobRecord = {
  id: string;
  status: JobStatus;
  createdAt: number;
  startedAt: number | null;
  finishedAt: number | null;
  fileName: string;
  queuePosition: number | null;
  result: ExtractResult | null;
  error: string | null;
};

type InternalJob = JobRecord & {
  buffer: Buffer;
};

const MAX_INFLIGHT = Number(process.env.JOB_MAX_INFLIGHT ?? 5);
const MAX_QUEUE = Number(process.env.JOB_MAX_QUEUE ?? 2000);
const JOB_TTL_MS = Number(process.env.JOB_TTL_MS ?? 30 * 60_000);
const JOB_DIR = process.env.JOB_DIR ?? "/tmp/fatura-jobs";

const jobs = new Map<string, InternalJob>();
const queue: string[] = [];
let inflight = 0;
let processed = 0;
let failed = 0;

async function ensureJobDir(): Promise<void> {
  await mkdir(JOB_DIR, { recursive: true });
}

function publicJob(job: InternalJob): JobRecord {
  const pos = job.status === "queued" ? Math.max(1, queue.indexOf(job.id) + 1) : null;
  return {
    id: job.id,
    status: job.status,
    createdAt: job.createdAt,
    startedAt: job.startedAt,
    finishedAt: job.finishedAt,
    fileName: job.fileName,
    queuePosition: pos,
    result: job.result ? toPublicExtractResult(job.result) : null,
    error: job.error ? sanitizePublicMessage(job.error) : null,
  };
}

function purgeExpired(): void {
  const now = Date.now();
  for (const [id, job] of jobs) {
    if (job.finishedAt && now - job.finishedAt > JOB_TTL_MS) {
      jobs.delete(id);
      void unlink(join(JOB_DIR, `${id}.bin`)).catch(() => undefined);
    }
  }
}

async function pump(): Promise<void> {
  while (inflight < MAX_INFLIGHT && queue.length > 0) {
    const id = queue.shift();
    if (!id) break;
    const job = jobs.get(id);
    if (!job || job.status !== "queued") continue;
    inflight += 1;
    job.status = "running";
    job.startedAt = Date.now();
    job.queuePosition = null;
    void runJob(job).finally(() => {
      inflight -= 1;
      void pump();
    });
  }
}

async function runJob(job: InternalJob): Promise<void> {
  try {
    const result = await extractInvoice(job.buffer, job.fileName);
    job.result = result;
    job.status = result.status === "failed" ? "failed" : "done";
    if (job.status === "failed") failed += 1;
    else processed += 1;
    if (result.status === "failed") {
      job.error = sanitizePublicMessage(result.warnings?.[0] ?? "Okuma başarısız");
    }
  } catch (err) {
    job.status = "failed";
    job.error = sanitizePublicMessage(err instanceof Error ? err.message : String(err));
    failed += 1;
  } finally {
    job.finishedAt = Date.now();
    // drop buffer to free RAM
    job.buffer = Buffer.alloc(0);
    void unlink(join(JOB_DIR, `${job.id}.bin`)).catch(() => undefined);
    purgeExpired();
  }
}

export async function enqueueJob(buffer: Buffer, fileName: string): Promise<JobRecord> {
  purgeExpired();
  if (queue.length + inflight >= MAX_QUEUE) {
    const err = new Error(`Kuyruk dolu (max ${MAX_QUEUE}). Lütfen biraz sonra tekrar deneyin.`);
    (err as Error & { code?: string }).code = "QUEUE_FULL";
    throw err;
  }
  await ensureJobDir();
  const id = randomUUID();
  const job: InternalJob = {
    id,
    status: "queued",
    createdAt: Date.now(),
    startedAt: null,
    finishedAt: null,
    fileName,
    queuePosition: queue.length + 1,
    result: null,
    error: null,
    buffer,
  };
  jobs.set(id, job);
  queue.push(id);
  await writeFile(join(JOB_DIR, `${id}.bin`), buffer).catch(() => undefined);
  void pump();
  return publicJob(job);
}

export function getJob(id: string): JobRecord | null {
  const job = jobs.get(id);
  if (!job) return null;
  return publicJob(job);
}

export function queueStats() {
  const hits = pathHitStats();
  return {
    queued: queue.length,
    inflight,
    maxInflight: MAX_INFLIGHT,
    maxQueue: MAX_QUEUE,
    jobsTracked: jobs.size,
    processed,
    failed,
    fastPathHits: hits.fastPathHits,
    v2Hits: hits.v2Hits,
  };
}

export async function resetJobStoreForTests(): Promise<void> {
  queue.length = 0;
  jobs.clear();
  inflight = 0;
  await rm(JOB_DIR, { recursive: true, force: true }).catch(() => undefined);
}
