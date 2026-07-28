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

export type InvoiceTotals = {
  lineExtensionAmount: number | null;
  discountTotal: number | null;
  vatAmount: number | null;
  withholdingVatAmount: number | null;
  taxInclusiveAmount: number | null;
  payableAmount: number | null;
  currency: string;
};

export type ParsedInvoice = {
  documentType: "earsiv" | "efatura" | "ubl" | "unknown";
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
  totals: InvoiceTotals;
  notes: string[];
  iban: string | null;
  bankName: string | null;
  bankBranch: string | null;
};

export type ExtractStatus = "ok" | "partial" | "failed";

export type ExtractResult = {
  status: ExtractStatus;
  method: "ubl" | "pdf-text";
  durationMs: number;
  warnings: string[];
  invoice: ParsedInvoice | null;
  rawTextPreview: string | null;
};
