export type InvoiceParty = {
  name: string | null;
  taxId: string | null;
  taxIdScheme: "VKN" | "TCKN" | null;
  taxOffice: string | null;
  address: string | null;
  phone: string | null;
  email: string | null;
  website: string | null;
};

export type InvoiceLine = {
  id: string | null;
  name: string | null;
  quantity: number | null;
  unit: string | null;
  unitPrice: number | null;
  discountRate: number | null;
  discountAmount: number | null;
  vatRate: number | null;
  vatAmount: number | null;
  withholdingNote: string | null;
  lineTotal: number | null;
};

export type ParsedInvoice = {
  documentType: string;
  profileId: string | null;
  customizationId: string | null;
  invoiceTypeCode: string | null;
  invoiceNumber: string | null;
  uuid: string | null;
  issueDate: string | null;
  issueTime: string | null;
  supplier: InvoiceParty;
  customer: InvoiceParty;
  lines: InvoiceLine[];
  totals: {
    lineExtensionAmount: number | null;
    discountTotal: number | null;
    vatAmount: number | null;
    withholdingVatAmount: number | null;
    taxInclusiveAmount: number | null;
    payableAmount: number | null;
    currency: string;
  };
  notes: string[];
  iban: string | null;
  bankName: string | null;
  bankBranch: string | null;
};

export type ExtractResult = {
  status: "ok" | "partial" | "failed";
  method: "ubl" | "pdf-text";
  durationMs: number;
  warnings: string[];
  invoice: ParsedInvoice | null;
  rawTextPreview: string | null;
};

export function formatMoney(n: number | null | undefined, currency = "TRY"): string {
  if (n == null || Number.isNaN(n)) return "—";
  return new Intl.NumberFormat("tr-TR", {
    style: "currency",
    currency,
    minimumFractionDigits: 2,
  }).format(n);
}

export function formatDate(iso: string | null, time: string | null): string {
  if (!iso) return "—";
  const [y, m, d] = iso.split("-");
  const date = `${d}.${m}.${y}`;
  return time ? `${date} ${time.slice(0, 5)}` : date;
}

export function formatSeconds(ms: number): string {
  return `${(ms / 1000).toFixed(2)} sn`;
}
