type Bucket = { tokens: number; updatedAt: number };

const buckets = new Map<string, Bucket>();

/** Defaults support bulk portal uploads; override via env in production if needed. */
const RATE_LIMIT_PER_MIN = Number(process.env.RATE_LIMIT_PER_MIN ?? 600);
const RATE_LIMIT_BURST = Number(process.env.RATE_LIMIT_BURST ?? 80);

export function takeRateToken(key: string): { ok: boolean; retryAfterSec: number } {
  const now = Date.now();
  const ratePerMs = RATE_LIMIT_PER_MIN / 60_000;
  let b = buckets.get(key);
  if (!b) {
    b = { tokens: RATE_LIMIT_BURST, updatedAt: now };
    buckets.set(key, b);
  }
  const elapsed = now - b.updatedAt;
  b.tokens = Math.min(RATE_LIMIT_BURST, b.tokens + elapsed * ratePerMs);
  b.updatedAt = now;
  if (b.tokens < 1) {
    const retryAfterSec = Math.ceil((1 - b.tokens) / ratePerMs / 1000);
    return { ok: false, retryAfterSec: Math.max(1, retryAfterSec) };
  }
  b.tokens -= 1;
  return { ok: true, retryAfterSec: 0 };
}

const cleanup = setInterval(() => {
  const cutoff = Date.now() - 10 * 60_000;
  for (const [k, b] of buckets) {
    if (b.updatedAt < cutoff) buckets.delete(k);
  }
}, 60_000);
if (typeof cleanup.unref === "function") cleanup.unref();
