import { useCallback, useRef, useState, type ReactNode } from "react";
import type { ExtractResult } from "./types";
import { formatDate, formatMoney, formatSeconds } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "/fatura-api";

function Field({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="min-w-0">
      <div className="label">{label}</div>
      <div className="value mt-1 break-words">{value ?? "—"}</div>
    </div>
  );
}

function docTypeLabel(t: string): string {
  if (t === "earsiv") return "e-Arşiv Fatura";
  if (t === "efatura") return "e-Fatura";
  if (t === "ubl") return "UBL-TR";
  return t;
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
    <div className="mx-auto min-h-screen max-w-6xl px-4 py-8 sm:px-6 lg:px-8">
      <header className="mb-8 flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.14em] text-violet-600">
            Nanobase Portal
          </p>
          <h1 className="mt-1 font-display text-4xl font-bold tracking-tight text-slate-900 sm:text-5xl">
            FaturaAI
          </h1>
          <p className="mt-2 max-w-xl text-sm text-slate-600">
            e-Arşiv / e-Fatura PDF yükleyin; tüm alanlar ve okuma süresi anında görünsün.
          </p>
        </div>
        <a
          href="https://portal.nanobase.ai/"
          className="btn-secondary text-xs"
        >
          Portal ana sayfa
        </a>
      </header>

      <section
        className={`glass relative overflow-hidden p-6 transition ${
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
        <div className="flex flex-col items-center gap-4 py-6 text-center sm:py-10">
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-violet-600/10 text-violet-700">
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
          <div>
            <p className="font-display text-lg font-semibold text-slate-900">
              PDF faturanızı sürükleyip bırakın
            </p>
            <p className="mt-1 text-sm text-slate-500">veya dosya seçin · max 20 MB</p>
          </div>
          <div className="flex flex-wrap items-center justify-center gap-3">
            <button
              type="button"
              className="btn-primary"
              disabled={loading}
              onClick={() => inputRef.current?.click()}
            >
              {loading ? "Okunuyor…" : "Fatura seç"}
            </button>
            {(result || error) && (
              <button
                type="button"
                className="btn-secondary"
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
            <p className="text-xs text-slate-500">
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
        <div className="mt-6 space-y-4 animate-[fadeIn_0.35s_ease]">
          <div className="glass flex flex-wrap items-center justify-between gap-3 p-4">
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
              <span className="rounded-full bg-violet-100 px-3 py-1 text-xs font-semibold text-violet-800">
                {result.method === "ubl" ? "UBL-TR" : "pdftotext"}
              </span>
            </div>
            <div className="text-right">
              <div className="label">Okuma süresi</div>
              <div className="font-display text-2xl font-bold text-violet-700">
                {formatSeconds(result.durationMs)}
              </div>
              <div className="text-xs text-slate-500">{result.durationMs} ms</div>
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
              <section className="glass p-5">
                <h2 className="font-display text-lg font-semibold text-slate-900">Belge</h2>
                <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                  <Field label="Tip" value={docTypeLabel(inv.documentType)} />
                  <Field label="Fatura No" value={inv.invoiceNumber} />
                  <Field label="ETTN" value={inv.uuid} />
                  <Field label="Tarih" value={formatDate(inv.issueDate, inv.issueTime)} />
                  <Field label="Senaryo" value={inv.profileId} />
                  <Field label="Fatura Tipi" value={inv.invoiceTypeCode} />
                  <Field label="Özelleştirme" value={inv.customizationId} />
                </div>
              </section>

              <div className="grid gap-4 lg:grid-cols-2">
                <section className="glass p-5">
                  <h2 className="font-display text-lg font-semibold text-slate-900">Satıcı</h2>
                  <div className="mt-4 grid gap-4">
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
                <section className="glass p-5">
                  <h2 className="font-display text-lg font-semibold text-slate-900">Alıcı</h2>
                  <div className="mt-4 grid gap-4">
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

              <section className="glass overflow-hidden p-5">
                <h2 className="font-display text-lg font-semibold text-slate-900">
                  Mal / Hizmet kalemleri
                </h2>
                <div className="mt-4 overflow-x-auto">
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
                      {inv.lines.map((line) => (
                        <tr key={line.id ?? line.name ?? Math.random()} className="border-b border-slate-100/80">
                          <td className="px-2 py-3 text-slate-500">{line.id}</td>
                          <td className="px-2 py-3">
                            <div className="font-medium text-slate-900">{line.name}</div>
                            {line.withholdingNote && (
                              <div className="mt-1 text-xs text-slate-500">{line.withholdingNote}</div>
                            )}
                          </td>
                          <td className="px-2 py-3 whitespace-nowrap">
                            {line.quantity} {line.unit}
                          </td>
                          <td className="px-2 py-3 whitespace-nowrap">
                            {formatMoney(line.unitPrice, currency)}
                          </td>
                          <td className="px-2 py-3 whitespace-nowrap">
                            {line.vatRate != null ? `%${line.vatRate}` : "—"}
                          </td>
                          <td className="px-2 py-3 whitespace-nowrap">
                            {formatMoney(line.vatAmount, currency)}
                          </td>
                          <td className="px-2 py-3 whitespace-nowrap font-semibold">
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

              <section className="glass p-5">
                <h2 className="font-display text-lg font-semibold text-slate-900">Toplamlar</h2>
                <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
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
                      <span className="text-base text-violet-700">
                        {formatMoney(inv.totals.payableAmount, currency)}
                      </span>
                    }
                  />
                </div>
              </section>

              {(inv.notes.length > 0 || inv.iban) && (
                <section className="glass p-5">
                  <h2 className="font-display text-lg font-semibold text-slate-900">
                    Notlar / Ödeme
                  </h2>
                  <div className="mt-4 grid gap-4 sm:grid-cols-2">
                    <Field label="IBAN" value={inv.iban} />
                    <Field label="Banka" value={inv.bankName} />
                    <Field label="Şube" value={inv.bankBranch} />
                  </div>
                  {inv.notes.length > 0 && (
                    <ul className="mt-4 space-y-1 text-sm text-slate-700">
                      {inv.notes.map((n) => (
                        <li key={n} className="rounded-xl bg-white/50 px-3 py-2">
                          {n}
                        </li>
                      ))}
                    </ul>
                  )}
                </section>
              )}

              {result.rawTextPreview && (
                <section className="glass p-5">
                  <button
                    type="button"
                    className="btn-secondary text-xs"
                    onClick={() => setShowRaw((v) => !v)}
                  >
                    {showRaw ? "Kaynak metni gizle" : "Kaynak metni göster"}
                  </button>
                  {showRaw && (
                    <pre className="mt-4 max-h-80 overflow-auto rounded-xl bg-slate-900/90 p-4 text-xs leading-relaxed text-slate-100">
                      {result.rawTextPreview}
                    </pre>
                  )}
                </section>
              )}
            </>
          )}
        </div>
      )}

      <footer className="mt-10 pb-6 text-center text-xs text-slate-500">
        FaturaAI · pdftotext (Poppler) · Nanobase
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
