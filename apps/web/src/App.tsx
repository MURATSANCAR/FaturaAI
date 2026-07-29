import { useCallback, useRef, useState, type ReactNode } from "react";
import type { ExtractResult, InvoiceLine } from "./types";
import { formatDate, formatMoney, formatSeconds } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "/fatura-api";

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

export default function App() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [result, setResult] = useState<ExtractResult | null>(null);
  const [showRaw, setShowRaw] = useState(false);

  const upload = useCallback(async (file: File) => {
    setError(null);
    setResult(null);
    setFileName(file.name);
    setLoading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API_BASE}/extract`, { method: "POST", body: form });
      const data = (await res.json()) as ExtractResult;
      if (!res.ok && !data.invoice) {
        throw new Error(data.warnings?.[0] ?? `HTTP ${res.status}`);
      }
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  const onFiles = (files: FileList | null) => {
    const file = files?.[0];
    if (!file) return;
    void upload(file);
  };

  const inv = result?.invoice;
  const currency = inv?.totals.currency ?? "TRY";

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
            e-Arşiv / e-Fatura PDF yükleyin; tüm alanlar ve okuma süresi anında görünsün.
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
              <span className="sm:hidden">PDF faturanızı seçin</span>
              <span className="hidden sm:inline">PDF faturanızı sürükleyip bırakın</span>
            </p>
            <p className="mt-1 text-sm text-slate-500">veya dosya seçin · max 20 MB</p>
          </div>
          <div className="flex w-full max-w-sm flex-col gap-2 sm:max-w-none sm:w-auto sm:flex-row sm:flex-wrap sm:items-center sm:justify-center sm:gap-3">
            <button
              type="button"
              className="btn-primary w-full sm:w-auto"
              disabled={loading}
              onClick={() => inputRef.current?.click()}
            >
              {loading ? "Okunuyor…" : "Fatura seç"}
            </button>
            {(result || error) && (
              <button
                type="button"
                className="btn-secondary w-full sm:w-auto"
                onClick={() => {
                  setResult(null);
                  setError(null);
                  setFileName(null);
                  setShowRaw(false);
                }}
              >
                Temizle
              </button>
            )}
          </div>
          <input
            ref={inputRef}
            type="file"
            accept="application/pdf,.pdf"
            className="hidden"
            onChange={(e) => onFiles(e.target.files)}
          />
          {fileName && (
            <p className="max-w-full px-2 text-xs text-slate-500 [overflow-wrap:anywhere]">
              Dosya: <span className="font-medium text-slate-700">{fileName}</span>
            </p>
          )}
        </div>
      </section>

      {error && (
        <div className="glass mt-4 border-red-200/80 bg-red-50/70 p-4 text-sm text-red-800">
          {error}
        </div>
      )}

      {result && (
        <div className="mt-5 space-y-3 animate-[fadeIn_0.35s_ease] sm:mt-6 sm:space-y-4">
          <div className="glass flex flex-wrap items-center justify-between gap-3 p-3.5 sm:p-4">
            <div className="flex flex-wrap items-center gap-2">
              <span
                className={`rounded-full px-3 py-1 text-xs font-semibold ${
                  result.status === "ok"
                    ? "bg-emerald-100 text-emerald-800"
                    : result.status === "partial"
                      ? "bg-amber-100 text-amber-900"
                      : "bg-red-100 text-red-800"
                }`}
              >
                {result.status === "ok"
                  ? "Tam okundu"
                  : result.status === "partial"
                    ? "Kısmi okuma"
                    : "Başarısız"}
              </span>
              {result.validation && (
                <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-700">
                  Güven %{Math.round(result.validation.confidence * 100)}
                </span>
              )}
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

                {/* Mobile: cards */}
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

                {/* Desktop: table */}
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
                    onClick={() => setShowRaw((v) => !v)}
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
