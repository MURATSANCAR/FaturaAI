import { useCallback, useRef, useState, type ReactNode } from "react";
import type { ExtractResult, InvoiceLine } from "./types";
import { formatDate, formatMoney, formatSeconds } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "/fatura-api";
/** Max parallel create+poll cycles — avoids browser ERR_INSUFFICIENT_RESOURCES and API 429. */
const UPLOAD_CONCURRENCY = 3;
/** Pace POSTs so we stay under API token-bucket (~600/min with burst). */
const CREATE_MIN_INTERVAL_MS = 200;
const CREATE_MAX_ATTEMPTS = 40;

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

function sleep(ms: number) {
  return new Promise<void>((r) => setTimeout(r, ms));
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
              <li key={w}>{w}</li>
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
  const uploadQueueRef = useRef<PendingUpload[]>([]);
  const activeUploadsRef = useRef(0);
  const lastCreateAtRef = useRef(0);
  const createLockRef = useRef(Promise.resolve());
  const [dragging, setDragging] = useState(false);
  const [items, setItems] = useState<UploadItem[]>([]);
  const [showRawIds, setShowRawIds] = useState<Record<string, boolean>>({});

  const loading = items.some(
    (item) =>
      item.status === "queued" ||
      item.status === "uploading" ||
      item.status === "reading",
  );
  const doneCount = items.filter((item) => item.status === "done").length;
  const failedCount = items.filter((item) => item.status === "failed").length;
  const activeCount = items.length - doneCount - failedCount;
  const batchProgress =
    loading && items.length > 0
      ? activeCount > 0
        ? `${doneCount + failedCount}/${items.length} tamam · ${activeCount} işlemde (en fazla ${UPLOAD_CONCURRENCY} paralel)…`
        : "Okunuyor…"
      : null;

  const patchItem = useCallback((id: string, patch: Partial<UploadItem>) => {
    setItems((prev) => prev.map((item) => (item.id === id ? { ...item, ...patch } : item)));
  }, []);

  const uploadOne = useCallback(
    async (id: string, file: File) => {
      patchItem(id, { status: "uploading", progress: "Kuyruğa alınıyor…", error: null });
      try {
        let create: Response | null = null;
        let created: {
          jobId?: string | null;
          status?: string;
          queuePosition?: number | null;
          warnings?: string[];
        } = {};

        for (let attempt = 1; attempt <= CREATE_MAX_ATTEMPTS; attempt++) {
          const form = new FormData();
          form.append("file", file);

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
            create = await fetch(`${API_BASE}/jobs`, { method: "POST", body: form });
          } finally {
            release();
          }
          created = (await create.json()) as typeof created;

          if (create.status === 429 || create.status === 503) {
            const retryAfter = Number(create.headers.get("Retry-After") || "3");
            const waitMs =
              Math.min(45_000, Math.max(2, retryAfter) * 1000) + attempt * 250 + Math.random() * 500;
            patchItem(id, {
              status: "queued",
              progress: `Rate limit — ${Math.ceil(waitMs / 1000)} sn sonra tekrar (${attempt}/${CREATE_MAX_ATTEMPTS})…`,
            });
            await sleep(waitMs);
            continue;
          }
          break;
        }

        if (!create || !create.ok || !created.jobId) {
          throw new Error(created.warnings?.[0] ?? `HTTP ${create?.status ?? "?"}`);
        }

        const jobId = created.jobId;
        if (created.queuePosition && created.queuePosition > 1) {
          patchItem(id, {
            status: "queued",
            progress: `Sırada (#${created.queuePosition})…`,
          });
        } else {
          patchItem(id, { status: "reading", progress: "Okunuyor…" });
        }

        const started = Date.now();
        while (Date.now() - started < 180_000) {
          await sleep(700);
          const res = await fetch(`${API_BASE}/jobs/${jobId}`);
          const job = (await res.json()) as {
            status: string;
            queuePosition?: number | null;
            result?: ExtractResult | null;
            error?: string | null;
            warnings?: string[];
          };
          if (!res.ok) {
            throw new Error(job.warnings?.[0] ?? job.error ?? `HTTP ${res.status}`);
          }
          if (job.status === "queued") {
            patchItem(id, {
              status: "queued",
              progress:
                job.queuePosition && job.queuePosition > 1
                  ? `Sırada (#${job.queuePosition})…`
                  : "Sırada…",
            });
            continue;
          }
          if (job.status === "running") {
            patchItem(id, { status: "reading", progress: "Okunuyor…" });
            continue;
          }
          if (job.status === "done" || job.status === "failed") {
            if (!job.result) {
              throw new Error(job.error ?? "Okuma sonucu alınamadı");
            }
            if (job.status === "failed" && !job.result.invoice) {
              throw new Error(job.error ?? job.result.warnings?.[0] ?? "Okuma başarısız");
            }
            patchItem(id, {
              status: "done",
              progress: null,
              result: job.result,
              error: null,
            });
            return;
          }
        }
        throw new Error("Okuma zaman aşımına uğradı");
      } catch (e) {
        patchItem(id, {
          status: "failed",
          progress: null,
          error: e instanceof Error ? e.message : String(e),
        });
      }
    },
    [patchItem],
  );

  const pumpUploads = useCallback(() => {
    while (activeUploadsRef.current < UPLOAD_CONCURRENCY && uploadQueueRef.current.length > 0) {
      const next = uploadQueueRef.current.shift();
      if (!next) break;
      activeUploadsRef.current += 1;
      void uploadOne(next.id, next.file).finally(() => {
        activeUploadsRef.current -= 1;
        pumpUploads();
      });
    }
  }, [uploadOne]);

  const onFiles = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0) return;
      const batch = Array.from(files).map((file) => {
        const id = createItemId();
        return {
          file,
          item: {
            id,
            fileName: file.name,
            status: "queued" as const,
            progress: "Bekliyor…",
            result: null,
            error: null,
          } satisfies UploadItem,
        };
      });

      setItems((prev) => [...prev, ...batch.map((b) => b.item)]);
      for (const { file, item } of batch) {
        uploadQueueRef.current.push({ id: item.id, file });
      }
      pumpUploads();
    },
    [pumpUploads],
  );

  const clearAll = () => {
    uploadQueueRef.current = [];
    setItems([]);
    setShowRawIds({});
  };

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
                disabled={loading}
                onClick={clearAll}
              >
                Temizle
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
            <ul className="space-y-2">
              {items.map((item) => (
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
          </div>
        )}
      </section>

      {items.some((item) => item.result || item.error) && (
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
