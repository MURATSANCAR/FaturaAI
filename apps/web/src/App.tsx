import { useCallback, useRef, useState, type ReactNode } from "react";
import type { ExtractResult, InvoiceLine } from "./types";
import { formatDate, formatMoney, formatSeconds } from "./types";
import { sanitizePublicMessage } from "./lib/public-facing";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "/fatura-api";
/** How many jobs may be submitted & polling at once (feeds server workers). */
const MAX_ACTIVE_JOBS = 12;
/** Space POSTs; production was ~120/min — stay safely under even before API redeploy. */
const CREATE_MIN_INTERVAL_MS = 600;
const CREATE_MAX_ATTEMPTS = 60;
const POLL_INTERVAL_MS = 1200;
const UI_FLUSH_MS = 500;
/** Per-job wait after create (server may queue behind others). */
const JOB_WAIT_MS = 15 * 60_000;
/** Above this, skip full invoice cards — list + summary only. */
const DETAIL_CARD_LIMIT = 30;
const LIST_ROW_LIMIT = 80;

type UploadItemStatus = "queued" | "uploading" | "reading" | "done" | "failed";

type UploadItem = {
  id: string;
  fileName: string;
  status: UploadItemStatus;
  progress: string | null;
  result: ExtractResult | null;
  error: string | null;
};

type PendingUpload = { id: string; file: File };

type ActiveJob = {
  id: string;
  jobId: string;
  startedAt: number;
};

function sleep(ms: number) {
  return new Promise<void>((r) => setTimeout(r, ms));
}

function slimResult(result: ExtractResult): ExtractResult {
  if (!result.rawTextPreview) return result;
  return { ...result, rawTextPreview: null };
}

function Field({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="min-w-0">
      <div className="label">{label}</div>
      <div className="value mt-1 break-words [overflow-wrap:anywhere]">{value ?? "—"}</div>
    </div>
  );
}

function docTypeLabel(t: string): string {
  if (t === "earsiv") return "e-Arşiv Fatura";
  if (t === "efatura") return "e-Fatura";
  if (t === "ubl") return "Elektronik Fatura";
  return t;
}

function statusChip(status: ExtractResult["status"]) {
  if (status === "ok") {
    return { className: "bg-emerald-100 text-emerald-800", label: "Tam okundu" };
  }
  if (status === "partial") {
    return { className: "bg-amber-100 text-amber-900", label: "Kısmi okuma" };
  }
  return { className: "bg-red-100 text-red-800", label: "Başarısız" };
}

function LineCard({
  line,
  currency,
}: {
  line: InvoiceLine;
  currency: string;
}) {
  return (
    <article className="rounded-2xl border border-white/70 bg-white/60 p-3.5 sm:p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <p className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">
            #{line.id ?? "—"}
          </p>
          <p className="mt-1 text-sm font-semibold leading-snug text-slate-900 [overflow-wrap:anywhere]">
            {line.name ?? "—"}
          </p>
          {line.withholdingNote && (
            <p className="mt-1 text-xs text-slate-500">{line.withholdingNote}</p>
          )}
        </div>
        <p className="shrink-0 text-sm font-bold text-violet-700">
          {formatMoney(line.lineTotal, currency)}
        </p>
      </div>
      <div className="mt-3 grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
        <div>
          <div className="label">Miktar</div>
          <div className="mt-0.5 font-semibold text-slate-800">
            {line.quantity ?? "—"} {line.unit ?? ""}
          </div>
        </div>
        <div>
          <div className="label">Birim fiyat</div>
          <div className="mt-0.5 font-semibold text-slate-800">
            {formatMoney(line.unitPrice, currency)}
          </div>
        </div>
        <div>
          <div className="label">KDV</div>
          <div className="mt-0.5 font-semibold text-slate-800">
            {line.vatRate != null ? `%${line.vatRate}` : "—"}
          </div>
        </div>
        <div>
          <div className="label">KDV tutarı</div>
          <div className="mt-0.5 font-semibold text-slate-800">
            {formatMoney(line.vatAmount, currency)}
          </div>
        </div>
      </div>
    </article>
  );
}

function InvoiceResult({
  item,
  showRaw,
  onToggleRaw,
}: {
  item: UploadItem;
  showRaw: boolean;
  onToggleRaw: () => void;
}) {
  const result = item.result;
  if (!result) return null;
  const inv = result.invoice;
  const currency = inv?.totals.currency ?? "TRY";
  const chip = statusChip(result.status);

  return (
    <div className="space-y-3 animate-[fadeIn_0.35s_ease] sm:space-y-4">
      <div className="glass flex flex-wrap items-center justify-between gap-3 p-3.5 sm:p-4">
        <div className="min-w-0 flex-1">
          <p className="text-sm font-semibold text-slate-900 [overflow-wrap:anywhere]">
            {item.fileName}
          </p>
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <span className={`rounded-full px-3 py-1 text-xs font-semibold ${chip.className}`}>
              {chip.label}
            </span>
            {result.validation && (
              <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-700">
                Güven %{Math.round(result.validation.confidence * 100)}
              </span>
            )}
          </div>
        </div>
        <div className="text-right">
          <div className="label">Okuma süresi</div>
          <div className="font-display text-xl font-bold text-violet-700 sm:text-2xl">
            {formatSeconds(result.durationMs)}
          </div>
        </div>
      </div>

      {result.warnings.length > 0 && (
        <div className="glass border-amber-200/70 bg-amber-50/60 p-4 text-sm text-amber-950">
          <p className="font-semibold">Uyarılar</p>
          <ul className="mt-2 list-disc space-y-1 pl-5">
            {result.warnings.map((w) => (
              <li key={w}>{sanitizePublicMessage(w)}</li>
            ))}
          </ul>
        </div>
      )}

      {inv && (
        <>
          <section className="glass p-4 sm:p-5">
            <h2 className="font-display text-base font-semibold text-slate-900 sm:text-lg">
              Belge
            </h2>
            <div className="mt-3 grid grid-cols-1 gap-3 sm:mt-4 sm:grid-cols-2 sm:gap-4 lg:grid-cols-3">
              <Field label="Tip" value={docTypeLabel(inv.documentType)} />
              <Field label="Fatura No" value={inv.invoiceNumber} />
              <Field label="ETTN" value={inv.uuid} />
              <Field label="Tarih" value={formatDate(inv.issueDate, inv.issueTime)} />
              <Field label="Senaryo" value={inv.profileId} />
              <Field label="Fatura Tipi" value={inv.invoiceTypeCode} />
              <Field label="Özelleştirme" value={inv.customizationId} />
            </div>
          </section>

          <div className="grid gap-3 sm:gap-4 lg:grid-cols-2">
            <section className="glass p-4 sm:p-5">
              <h2 className="font-display text-base font-semibold text-slate-900 sm:text-lg">
                Satıcı
              </h2>
              <div className="mt-3 grid gap-3 sm:mt-4 sm:gap-4">
                <Field label="Unvan" value={inv.supplier.name} />
                <Field
                  label={inv.supplier.taxIdScheme ?? "Vergi No"}
                  value={inv.supplier.taxId}
                />
                <Field label="Vergi Dairesi" value={inv.supplier.taxOffice} />
                <Field label="Adres" value={inv.supplier.address} />
                <Field label="Telefon" value={inv.supplier.phone} />
                <Field label="E-posta" value={inv.supplier.email} />
                {inv.supplier.website && (
                  <Field label="Web" value={inv.supplier.website} />
                )}
              </div>
            </section>
            <section className="glass p-4 sm:p-5">
              <h2 className="font-display text-base font-semibold text-slate-900 sm:text-lg">
                Alıcı
              </h2>
              <div className="mt-3 grid gap-3 sm:mt-4 sm:gap-4">
                <Field label="Unvan" value={inv.customer.name} />
                <Field
                  label={inv.customer.taxIdScheme ?? "Vergi No"}
                  value={inv.customer.taxId}
                />
                <Field label="Vergi Dairesi" value={inv.customer.taxOffice} />
                <Field label="Adres" value={inv.customer.address} />
              </div>
            </section>
          </div>

          <section className="glass p-4 sm:overflow-hidden sm:p-5">
            <h2 className="font-display text-base font-semibold text-slate-900 sm:text-lg">
              Mal / Hizmet kalemleri
            </h2>

            <div className="mt-3 space-y-2.5 md:hidden">
              {inv.lines.length === 0 ? (
                <p className="py-4 text-center text-sm text-slate-500">Kalem bulunamadı</p>
              ) : (
                inv.lines.map((line, idx) => (
                  <LineCard
                    key={`${line.id ?? "l"}-${idx}`}
                    line={line}
                    currency={currency}
                  />
                ))
              )}
            </div>

            <div className="-mx-1 mt-4 hidden overflow-x-auto md:block">
              <table className="min-w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-slate-200/80 text-[11px] uppercase tracking-wide text-slate-500">
                    <th className="px-2 py-2 font-semibold">#</th>
                    <th className="px-2 py-2 font-semibold">Açıklama</th>
                    <th className="px-2 py-2 font-semibold">Miktar</th>
                    <th className="px-2 py-2 font-semibold">Birim fiyat</th>
                    <th className="px-2 py-2 font-semibold">KDV</th>
                    <th className="px-2 py-2 font-semibold">KDV tutarı</th>
                    <th className="px-2 py-2 font-semibold">Tutar</th>
                  </tr>
                </thead>
                <tbody>
                  {inv.lines.map((line, idx) => (
                    <tr
                      key={`${line.id ?? "l"}-${idx}`}
                      className="border-b border-slate-100/80"
                    >
                      <td className="px-2 py-3 text-slate-500">{line.id}</td>
                      <td className="px-2 py-3">
                        <div className="font-medium text-slate-900">{line.name}</div>
                        {line.withholdingNote && (
                          <div className="mt-1 text-xs text-slate-500">
                            {line.withholdingNote}
                          </div>
                        )}
                      </td>
                      <td className="whitespace-nowrap px-2 py-3">
                        {line.quantity} {line.unit}
                      </td>
                      <td className="whitespace-nowrap px-2 py-3">
                        {formatMoney(line.unitPrice, currency)}
                      </td>
                      <td className="whitespace-nowrap px-2 py-3">
                        {line.vatRate != null ? `%${line.vatRate}` : "—"}
                      </td>
                      <td className="whitespace-nowrap px-2 py-3">
                        {formatMoney(line.vatAmount, currency)}
                      </td>
                      <td className="whitespace-nowrap px-2 py-3 font-semibold">
                        {formatMoney(line.lineTotal, currency)}
                      </td>
                    </tr>
                  ))}
                  {inv.lines.length === 0 && (
                    <tr>
                      <td colSpan={7} className="px-2 py-6 text-center text-slate-500">
                        Kalem bulunamadı
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </section>

          <section className="glass p-4 sm:p-5">
            <h2 className="font-display text-base font-semibold text-slate-900 sm:text-lg">
              Toplamlar
            </h2>
            <div className="mt-3 grid grid-cols-2 gap-3 sm:mt-4 sm:grid-cols-2 lg:grid-cols-3">
              <Field
                label="Mal/Hizmet toplam"
                value={formatMoney(inv.totals.lineExtensionAmount, currency)}
              />
              <Field
                label="İskonto"
                value={formatMoney(inv.totals.discountTotal, currency)}
              />
              <Field label="KDV" value={formatMoney(inv.totals.vatAmount, currency)} />
              <Field
                label="KDV Tevkifat"
                value={formatMoney(inv.totals.withholdingVatAmount, currency)}
              />
              <Field
                label="Vergiler dahil"
                value={formatMoney(inv.totals.taxInclusiveAmount, currency)}
              />
              <Field
                label="Ödenecek tutar"
                value={
                  <span className="text-base font-bold text-violet-700">
                    {formatMoney(inv.totals.payableAmount, currency)}
                  </span>
                }
              />
            </div>
          </section>

          {(inv.notes.length > 0 || inv.iban) && (
            <section className="glass p-4 sm:p-5">
              <h2 className="font-display text-base font-semibold text-slate-900 sm:text-lg">
                Notlar / Ödeme
              </h2>
              <div className="mt-3 grid gap-3 sm:mt-4 sm:grid-cols-2 sm:gap-4">
                <Field label="IBAN" value={inv.iban} />
                <Field label="Banka" value={inv.bankName} />
                <Field label="Şube" value={inv.bankBranch} />
              </div>
              {inv.notes.length > 0 && (
                <ul className="mt-4 space-y-1.5 text-sm text-slate-700">
                  {inv.notes.map((n) => (
                    <li
                      key={n}
                      className="rounded-xl bg-white/50 px-3 py-2 [overflow-wrap:anywhere]"
                    >
                      {n}
                    </li>
                  ))}
                </ul>
              )}
            </section>
          )}

          {result.rawTextPreview && (
            <section className="glass p-4 sm:p-5">
              <button
                type="button"
                className="btn-secondary w-full text-xs sm:w-auto"
                onClick={onToggleRaw}
              >
                {showRaw ? "Kaynak metni gizle" : "Kaynak metni göster"}
              </button>
              {showRaw && (
                <pre className="mt-4 max-h-64 overflow-auto rounded-xl bg-slate-900/90 p-3 text-[11px] leading-relaxed text-slate-100 sm:max-h-80 sm:p-4 sm:text-xs">
                  {result.rawTextPreview}
                </pre>
              )}
            </section>
          )}
        </>
      )}
    </div>
  );
}

const ACCEPT_UPLOAD =
  "application/pdf,.pdf,image/jpeg,image/png,image/webp,image/heic,image/heif,.jpg,.jpeg,.png,.webp,.heic,.heif";
const ACCEPT_CAMERA = "image/*";

let nextItemId = 0;
function createItemId() {
  nextItemId += 1;
  return `inv-${Date.now()}-${nextItemId}`;
}

export default function App() {
  const uploadInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);
  const pendingRef = useRef<PendingUpload[]>([]);
  const activeJobsRef = useRef<Map<string, ActiveJob>>(new Map());
  const itemsRef = useRef<Map<string, UploadItem>>(new Map());
  const itemOrderRef = useRef<string[]>([]);
  const creatingRef = useRef(false);
  const pollingRef = useRef(false);
  const stoppedRef = useRef(false);
  const lastCreateAtRef = useRef(0);
  const createLockRef = useRef(Promise.resolve());
  const uiFlushTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [dragging, setDragging] = useState(false);
  const [items, setItems] = useState<UploadItem[]>([]);
  const [showRawIds, setShowRawIds] = useState<Record<string, boolean>>({});

  const doneCount = items.filter((item) => item.status === "done").length;
  const failedCount = items.filter((item) => item.status === "failed").length;
  const activeCount = items.length - doneCount - failedCount;
  const loading = activeCount > 0;
  const bulkMode = items.length > DETAIL_CARD_LIMIT;
  const batchProgress =
    loading && items.length > 0
      ? `${doneCount + failedCount}/${items.length} tamam · ${activeCount} işlemde (sunucuda en fazla ${MAX_ACTIVE_JOBS})`
      : null;

  const flushItemsToState = useCallback(() => {
    setItems(itemOrderRef.current.map((id) => itemsRef.current.get(id)!).filter(Boolean));
  }, []);

  const scheduleFlush = useCallback(() => {
    if (uiFlushTimerRef.current) return;
    uiFlushTimerRef.current = setTimeout(() => {
      uiFlushTimerRef.current = null;
      flushItemsToState();
    }, UI_FLUSH_MS);
  }, [flushItemsToState]);

  const patchItem = useCallback(
    (id: string, patch: Partial<UploadItem>) => {
      const prev = itemsRef.current.get(id);
      if (!prev) return;
      itemsRef.current.set(id, { ...prev, ...patch });
      scheduleFlush();
    },
    [scheduleFlush],
  );

  const postJob = useCallback(async (file: File): Promise<Response> => {
    const form = new FormData();
    form.append("file", file);
    // Bust any intermediate HTTP cache; each upload must create a fresh job.
    form.append("clientRequestId", `${Date.now()}-${Math.random().toString(36).slice(2)}`);
    const prev = createLockRef.current;
    let release!: () => void;
    createLockRef.current = new Promise<void>((r) => {
      release = r;
    });
    await prev;
    try {
      const wait = Math.max(0, lastCreateAtRef.current + CREATE_MIN_INTERVAL_MS - Date.now());
      if (wait > 0) await sleep(wait);
      lastCreateAtRef.current = Date.now();
      return await fetch(`${API_BASE}/jobs?t=${Date.now()}`, {
        method: "POST",
        body: form,
        cache: "no-store",
      });
    } finally {
      release();
    }
  }, []);

  const pumpCreates = useCallback(async () => {
    if (creatingRef.current || stoppedRef.current) return;
    creatingRef.current = true;
    try {
      while (
        !stoppedRef.current &&
        pendingRef.current.length > 0 &&
        activeJobsRef.current.size < MAX_ACTIVE_JOBS
      ) {
        const next = pendingRef.current.shift();
        if (!next) break;
        patchItem(next.id, {
          status: "uploading",
          progress: "Kuyruğa alınıyor…",
          error: null,
        });

        let create: Response | null = null;
        let created: {
          jobId?: string | null;
          queuePosition?: number | null;
          warnings?: string[];
        } = {};
        let ok = false;

        for (let attempt = 1; attempt <= CREATE_MAX_ATTEMPTS; attempt++) {
          if (stoppedRef.current) return;
          create = await postJob(next.file);
          created = (await create.json()) as typeof created;
          if (create.status === 429 || create.status === 503) {
            const retryAfter = Number(create.headers.get("Retry-After") || "3");
            const waitMs =
              Math.min(60_000, Math.max(2, retryAfter) * 1000) +
              attempt * 200 +
              Math.random() * 400;
            patchItem(next.id, {
              status: "queued",
              progress: `Bekleniyor ${Math.ceil(waitMs / 1000)} sn (${attempt}/${CREATE_MAX_ATTEMPTS})…`,
            });
            await sleep(waitMs);
            continue;
          }
          ok = Boolean(create.ok && created.jobId);
          break;
        }

        if (!ok || !created.jobId) {
          patchItem(next.id, {
            status: "failed",
            progress: null,
            error: sanitizePublicMessage(created.warnings?.[0] ?? `HTTP ${create?.status ?? "?"}`),
          });
          continue;
        }

        activeJobsRef.current.set(next.id, {
          id: next.id,
          jobId: created.jobId,
          startedAt: Date.now(),
        });
        patchItem(next.id, {
          status:
            created.queuePosition && created.queuePosition > 1 ? "queued" : "reading",
          progress:
            created.queuePosition && created.queuePosition > 1
              ? `Sırada (#${created.queuePosition})…`
              : "Okunuyor…",
        });
      }
    } finally {
      creatingRef.current = false;
    }
  }, [patchItem, postJob]);

  const pumpPolls = useCallback(async () => {
    if (pollingRef.current || stoppedRef.current) return;
    pollingRef.current = true;
    try {
      await pumpCreates();
      while (
        !stoppedRef.current &&
        (activeJobsRef.current.size > 0 || pendingRef.current.length > 0)
      ) {
        if (activeJobsRef.current.size === 0) {
          await pumpCreates();
          if (activeJobsRef.current.size === 0) break;
        }
        const entries = [...activeJobsRef.current.values()];
        await Promise.all(
          entries.map(async (job) => {
            try {
              if (Date.now() - job.startedAt > JOB_WAIT_MS) {
                activeJobsRef.current.delete(job.id);
                patchItem(job.id, {
                  status: "failed",
                  progress: null,
                  error: "Okuma zaman aşımına uğradı",
                });
                return;
              }
              const res = await fetch(`${API_BASE}/jobs/${job.jobId}?t=${Date.now()}`, {
                cache: "no-store",
              });
              const body = (await res.json()) as {
                status: string;
                queuePosition?: number | null;
                result?: ExtractResult | null;
                error?: string | null;
                warnings?: string[];
              };
              if (!res.ok) {
                activeJobsRef.current.delete(job.id);
                patchItem(job.id, {
                  status: "failed",
                  progress: null,
                  error: sanitizePublicMessage(
                    body.warnings?.[0] ?? body.error ?? `HTTP ${res.status}`,
                  ),
                });
                return;
              }
              if (body.status === "queued") {
                patchItem(job.id, {
                  status: "queued",
                  progress:
                    body.queuePosition && body.queuePosition > 1
                      ? `Sırada (#${body.queuePosition})…`
                      : "Sırada…",
                });
                return;
              }
              if (body.status === "running") {
                patchItem(job.id, { status: "reading", progress: "Okunuyor…" });
                return;
              }
              if (body.status === "done" || body.status === "failed") {
                activeJobsRef.current.delete(job.id);
                if (!body.result) {
                  patchItem(job.id, {
                    status: "failed",
                    progress: null,
                    error: sanitizePublicMessage(body.error ?? "Okuma sonucu alınamadı"),
                  });
                  return;
                }
                if (body.status === "failed" && !body.result.invoice) {
                  patchItem(job.id, {
                    status: "failed",
                    progress: null,
                    error: sanitizePublicMessage(
                      body.error ?? body.result.warnings?.[0] ?? "Okuma başarısız",
                    ),
                  });
                  return;
                }
                const keepHeavy = itemOrderRef.current.length <= DETAIL_CARD_LIMIT;
                patchItem(job.id, {
                  status: "done",
                  progress: null,
                  result: keepHeavy ? body.result : slimResult(body.result),
                  error: null,
                });
              }
            } catch (e) {
              activeJobsRef.current.delete(job.id);
              patchItem(job.id, {
                status: "failed",
                progress: null,
                error: sanitizePublicMessage(e instanceof Error ? e.message : String(e)),
              });
            }
          }),
        );
        await pumpCreates();
        if (activeJobsRef.current.size === 0 && pendingRef.current.length === 0) break;
        await sleep(POLL_INTERVAL_MS);
      }
    } finally {
      pollingRef.current = false;
      if (
        !stoppedRef.current &&
        (activeJobsRef.current.size > 0 || pendingRef.current.length > 0)
      ) {
        void pumpPolls();
      } else {
        flushItemsToState();
      }
    }
  }, [flushItemsToState, patchItem, pumpCreates]);

  const startPipeline = useCallback(() => {
    stoppedRef.current = false;
    void pumpPolls();
  }, [pumpPolls]);

  const onFiles = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0) return;
      const batch = Array.from(files);
      // Same filename re-upload: drop prior cards so UI looks like a first upload.
      const incomingNames = new Set(batch.map((f) => f.name));
      for (const [id, item] of [...itemsRef.current.entries()]) {
        if (!incomingNames.has(item.fileName)) continue;
        itemsRef.current.delete(id);
        itemOrderRef.current = itemOrderRef.current.filter((x) => x !== id);
        pendingRef.current = pendingRef.current.filter((p) => p.id !== id);
        activeJobsRef.current.delete(id);
      }
      for (const file of batch) {
        const id = createItemId();
        const item: UploadItem = {
          id,
          fileName: file.name,
          status: "queued",
          progress: "Bekliyor…",
          result: null,
          error: null,
        };
        itemsRef.current.set(id, item);
        itemOrderRef.current.push(id);
        pendingRef.current.push({ id, file });
      }
      flushItemsToState();
      startPipeline();
    },
    [flushItemsToState, startPipeline],
  );

  const clearAll = () => {
    stoppedRef.current = true;
    pendingRef.current = [];
    activeJobsRef.current.clear();
    itemsRef.current.clear();
    itemOrderRef.current = [];
    creatingRef.current = false;
    pollingRef.current = false;
    if (uiFlushTimerRef.current) {
      clearTimeout(uiFlushTimerRef.current);
      uiFlushTimerRef.current = null;
    }
    setItems([]);
    setShowRawIds({});
  };

  const listRows = (() => {
    if (items.length <= LIST_ROW_LIMIT) return items;
    const active = items.filter(
      (i) => i.status === "queued" || i.status === "uploading" || i.status === "reading",
    );
    const failed = items.filter((i) => i.status === "failed");
    const done = items.filter((i) => i.status === "done");
    return [...active.slice(0, 40), ...failed.slice(0, 30), ...done.slice(-20)];
  })();
  const hiddenListCount = Math.max(0, items.length - listRows.length);

  return (
    <div className="mx-auto min-h-dvh max-w-6xl px-3 py-5 pb-[max(1.5rem,env(safe-area-inset-bottom))] pt-[max(1.25rem,env(safe-area-inset-top))] sm:px-6 sm:py-8 lg:px-8">
      <header className="mb-5 flex flex-col gap-3 sm:mb-8 sm:flex-row sm:flex-wrap sm:items-end sm:justify-between sm:gap-4">
        <div className="min-w-0">
          <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-violet-600 sm:text-xs">
            Nanobase Portal
          </p>
          <h1 className="mt-1 font-display text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl md:text-5xl">
            FaturaAI
          </h1>
          <p className="mt-2 max-w-xl text-sm leading-relaxed text-slate-600">
            Tek veya toplu e-Arşiv / e-Fatura PDF / fotoğraf yükleyin; her faturanın adı ve okuma
            süresi üstte listelenir.
          </p>
        </div>
        <a
          href="https://portal.nanobase.ai/"
          className="btn-secondary w-full justify-center text-xs sm:w-auto"
        >
          Portal ana sayfa
        </a>
      </header>

      <section
        className={`glass relative overflow-hidden p-4 transition sm:p-6 ${
          dragging ? "ring-2 ring-violet-400" : ""
        }`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          onFiles(e.dataTransfer.files);
        }}
      >
        <div className="flex flex-col items-center gap-4 py-5 text-center sm:py-10">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-violet-600/10 text-violet-700 sm:h-14 sm:w-14">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" aria-hidden>
              <path
                d="M12 16V4m0 0 4 4m-4-4-4 4M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </div>
          <div className="px-1">
            <p className="font-display text-base font-semibold text-slate-900 sm:text-lg">
              <span className="sm:hidden">Tek veya toplu fatura seçin</span>
              <span className="hidden sm:inline">
                PDF / fotoğraf sürükleyip bırakın — tek veya toplu
              </span>
            </p>
            <p className="mt-1 text-sm text-slate-500">
              e-Arşiv · e-Fatura · PDF, JPG, PNG · max 20 MB · birden fazla dosya
            </p>
          </div>
          <div className="flex w-full max-w-sm flex-col gap-2 sm:max-w-none sm:w-auto sm:flex-row sm:flex-wrap sm:items-center sm:justify-center sm:gap-3">
            <button
              type="button"
              className="btn-primary w-full sm:w-auto"
              onClick={() => uploadInputRef.current?.click()}
            >
              Toplu / tek yükle
            </button>
            <button
              type="button"
              className="btn-secondary w-full sm:hidden"
              onClick={() => cameraInputRef.current?.click()}
            >
              Foto çek
            </button>
            {items.length > 0 && (
              <button
                type="button"
                className="btn-secondary w-full sm:w-auto"
                onClick={clearAll}
              >
                {loading ? "Durdur ve temizle" : "Temizle"}
              </button>
            )}
          </div>
          {batchProgress && (
            <p className="text-sm font-medium text-violet-700">{batchProgress}</p>
          )}
          <input
            ref={uploadInputRef}
            type="file"
            accept={ACCEPT_UPLOAD}
            multiple
            className="hidden"
            onChange={(e) => {
              onFiles(e.target.files);
              e.target.value = "";
            }}
          />
          <input
            ref={cameraInputRef}
            type="file"
            accept={ACCEPT_CAMERA}
            capture="environment"
            className="hidden"
            onChange={(e) => {
              onFiles(e.target.files);
              e.target.value = "";
            }}
          />
        </div>

        {items.length > 0 && (
          <div className="border-t border-slate-200/70 pt-4">
            <div className="mb-2 flex items-center justify-between gap-2 px-1">
              <p className="text-left text-xs font-semibold uppercase tracking-wide text-slate-500">
                Yüklenen faturalar ({items.length})
              </p>
              {(doneCount > 0 || failedCount > 0) && (
                <p className="text-xs text-slate-500">
                  {doneCount > 0 && `${doneCount} okundu`}
                  {doneCount > 0 && failedCount > 0 && " · "}
                  {failedCount > 0 && `${failedCount} hata`}
                </p>
              )}
            </div>
            {bulkMode && (
              <p className="mb-2 px-1 text-left text-xs text-slate-500">
                Toplu mod: tarayıcıyı yormamak için liste kısaltıldı; detay kartları gizlendi.
              </p>
            )}
            <ul className="max-h-[28rem] space-y-2 overflow-y-auto">
              {listRows.map((item) => (
                <li
                  key={item.id}
                  className="flex items-start justify-between gap-3 rounded-2xl border border-white/70 bg-white/60 px-3.5 py-3 text-left"
                >
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-semibold text-slate-900 [overflow-wrap:anywhere]">
                      {item.fileName}
                    </p>
                    {item.status !== "done" && item.status !== "failed" && (
                      <p className="mt-1 text-xs text-violet-600">
                        {item.progress ?? "Bekliyor…"}
                      </p>
                    )}
                    {item.error && (
                      <p className="mt-1 text-xs text-red-700 [overflow-wrap:anywhere]">
                        {item.error}
                      </p>
                    )}
                  </div>
                  <div className="shrink-0 text-right">
                    <div className="label">Okuma süresi</div>
                    <div className="font-display text-base font-bold text-violet-700 sm:text-lg">
                      {item.result
                        ? formatSeconds(item.result.durationMs)
                        : item.status === "failed"
                          ? "—"
                          : "…"}
                    </div>
                  </div>
                </li>
              ))}
            </ul>
            {hiddenListCount > 0 && (
              <p className="mt-2 px-1 text-xs text-slate-500">
                +{hiddenListCount} dosya listede gizlendi (özet yukarıda)
              </p>
            )}
          </div>
        )}
      </section>

      {!bulkMode && items.some((item) => item.result || item.error) && (
        <div className="mt-5 space-y-8 sm:mt-6">
          {items.map((item) => {
            if (item.error && !item.result) {
              return (
                <div
                  key={item.id}
                  className="glass border-red-200/80 bg-red-50/70 p-4 text-sm text-red-800"
                >
                  <p className="font-semibold [overflow-wrap:anywhere]">{item.fileName}</p>
                  <p className="mt-1">{item.error}</p>
                </div>
              );
            }
            if (!item.result) return null;
            return (
              <InvoiceResult
                key={item.id}
                item={item}
                showRaw={!!showRawIds[item.id]}
                onToggleRaw={() =>
                  setShowRawIds((prev) => ({ ...prev, [item.id]: !prev[item.id] }))
                }
              />
            );
          })}
        </div>
      )}

      {bulkMode && !loading && failedCount > 0 && (
        <div className="glass mt-5 border-red-200/80 bg-red-50/70 p-4 text-sm text-red-800">
          <p className="font-semibold">{failedCount} dosya başarısız</p>
          <ul className="mt-2 max-h-48 list-disc space-y-1 overflow-y-auto pl-5">
            {items
              .filter((i) => i.status === "failed")
              .slice(0, 50)
              .map((i) => (
                <li key={i.id}>
                  {i.fileName}: {i.error}
                </li>
              ))}
          </ul>
        </div>
      )}

      <footer className="mt-8 pb-2 text-center text-xs text-slate-500 sm:mt-10 sm:pb-6">
        FaturaAI · Nanobase
      </footer>

      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(6px); }
          to { opacity: 1; transform: none; }
        }
      `}</style>
    </div>
  );
}
