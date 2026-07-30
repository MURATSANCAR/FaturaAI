import type { ParsedInvoice } from "../types.js";
import { reconcileTotals } from "./parse-pdf-text.js";

function tag(name: string): string {
  return `(?:\\w+:)?${name}`;
}

function decode(value: string): string {
  return value
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .trim();
}

function first(xml: string, name: string, scope?: string): string | null {
  const src = scope ?? xml;
  const m = src.match(new RegExp(`<${tag(name)}(?:\\s[^>]*)?>([^<]*)</${tag(name)}>`, "i"));
  return m ? decode(m[1]) : null;
}

function block(xml: string, name: string): string | null {
  const openRe = new RegExp(`<${tag(name)}(?:\\s[^>]*)?>`, "i");
  const open = openRe.exec(xml);
  if (!open || open.index == null) return null;
  let depth = 1;
  let cursor = open.index + open[0].length;
  const openAny = new RegExp(`<${tag(name)}(?:\\s[^>]*)?>`, "gi");
  const closeAny = new RegExp(`</${tag(name)}>`, "gi");
  while (depth > 0 && cursor < xml.length) {
    openAny.lastIndex = cursor;
    closeAny.lastIndex = cursor;
    const nOpen = openAny.exec(xml);
    const nClose = closeAny.exec(xml);
    if (!nClose) return null;
    if (nOpen && nOpen.index < nClose.index) {
      depth++;
      cursor = nOpen.index + nOpen[0].length;
    } else {
      depth--;
      if (depth === 0) return xml.slice(open.index, nClose.index + nClose[0].length);
      cursor = nClose.index + nClose[0].length;
    }
  }
  return null;
}

function money(xml: string, name: string, scope?: string): number | null {
  const src = scope ?? xml;
  const m = src.match(new RegExp(`<${tag(name)}[^>]*>([^<]+)</${tag(name)}>`, "i"));
  if (!m) return null;
  const n = Number.parseFloat(decode(m[1]));
  return Number.isFinite(n) ? n : null;
}

function sumMoneyAll(xml: string, blockName: string, amountName: string): number | null {
  const re = new RegExp(`<${tag(blockName)}[\\s>]`, "gi");
  let sum = 0;
  let found = false;
  for (const m of xml.matchAll(re)) {
    if (m.index == null) continue;
    const b = block(xml.slice(m.index), blockName);
    if (!b) continue;
    const amt = money(b, amountName);
    if (amt != null) {
      sum += amt;
      found = true;
    }
  }
  return found ? Number(sum.toFixed(2)) : null;
}

function partyTaxId(partyXml: string): { taxId: string | null; scheme: "VKN" | "TCKN" | null } {
  const m = partyXml.match(
    new RegExp(
      `<${tag("ID")}[^>]*schemeID=["'](VKN|TCKN|VKN_TCKN)["'][^>]*>([^<]+)</${tag("ID")}>`,
      "i",
    ),
  );
  if (!m) return { taxId: null, scheme: null };
  const scheme = m[1].toUpperCase().includes("TCKN") ? "TCKN" : "VKN";
  return { taxId: decode(m[2]).replace(/\D/g, ""), scheme };
}

function partyName(partyXml: string): string | null {
  const legal = block(partyXml, "PartyLegalEntity");
  const reg = legal ? first(legal, "RegistrationName") : null;
  if (reg) return reg.slice(0, 160);
  const pn = block(partyXml, "PartyName");
  return (pn ? first(pn, "Name") : null)?.slice(0, 160) ?? null;
}

function partyTaxOffice(partyXml: string): string | null {
  const pts = block(partyXml, "PartyTaxScheme");
  if (!pts) return null;
  const scheme = block(pts, "TaxScheme");
  return (scheme ? first(scheme, "Name") : first(pts, "Name"))?.slice(0, 120) ?? null;
}

function asciiUpper(s: string): string {
  return s
    .replace(/İ/g, "I")
    .replace(/ı/g, "I")
    .replace(/Ş/g, "S")
    .replace(/ş/g, "S")
    .replace(/Ğ/g, "G")
    .replace(/ğ/g, "G")
    .replace(/Ü/g, "U")
    .replace(/ü/g, "U")
    .replace(/Ö/g, "O")
    .replace(/ö/g, "O")
    .replace(/Ç/g, "C")
    .replace(/ç/g, "C")
    .toUpperCase()
    .replace(/[^A-Z0-9_]/g, "");
}

function documentKind(profileId: string | null): ParsedInvoice["documentType"] {
  const p = (profileId ?? "").toUpperCase();
  if (p.includes("EARSIV")) return "earsiv";
  if (
    /TEMEL|TICARI|IHRACAT|YOLCU|KAMU|ENERJI|ILAC|HKS/.test(p)
  ) {
    return "efatura";
  }
  return profileId ? "ubl" : "unknown";
}

export function isUblInvoiceXml(text: string): boolean {
  const t = text.trim();
  return /<(?:\w+:)?Invoice[\s>]/i.test(t) && /(?:CustomizationID|ProfileID|AccountingSupplierParty)/i.test(t);
}

export function parseUblInvoice(xml: string): ParsedInvoice | null {
  if (!isUblInvoiceXml(xml)) return null;
  const profileIdRaw = first(xml, "ProfileID");
  const profileId = profileIdRaw ? asciiUpper(profileIdRaw) : null;
  const kind = documentKind(profileId);

  const currency = first(xml, "DocumentCurrencyCode") ?? "TRY";
  const supplierBlock = block(xml, "AccountingSupplierParty");
  const supplierParty = supplierBlock ? block(supplierBlock, "Party") ?? supplierBlock : "";
  const customerBlock = block(xml, "AccountingCustomerParty");
  const customerParty = customerBlock ? block(customerBlock, "Party") ?? customerBlock : "";
  const sTax = partyTaxId(supplierParty);
  const cTax = partyTaxId(customerParty);
  const monetary = block(xml, "LegalMonetaryTotal");

  const lines: ParsedInvoice["lines"] = [];
  const lineRe = new RegExp(`<${tag("InvoiceLine")}[\\s>]`, "gi");
  for (const m of xml.matchAll(lineRe)) {
    if (m.index == null) continue;
    const b = block(xml.slice(m.index), "InvoiceLine");
    if (!b) continue;
    const item = block(b, "Item");
    const qtyRaw = first(b, "InvoicedQuantity");
    const price = block(b, "Price");
    lines.push({
      id: first(b, "ID"),
      name: item ? first(item, "Name") : first(b, "Name"),
      quantity: qtyRaw ? Number.parseFloat(qtyRaw.replace(",", ".")) : null,
      unit: null,
      unitPrice: price ? money(price, "PriceAmount") : money(b, "PriceAmount") ?? money(b, "Price"),
      discountRate: null,
      discountAmount: null,
      vatRate: null,
      vatAmount: money(b, "TaxAmount"),
      withholdingNote: null,
      lineTotal: money(b, "LineExtensionAmount"),
    });
    if (lines.length >= 100) break;
  }

  const notes: string[] = [];
  for (const m of xml.matchAll(new RegExp(`<${tag("Note")}[^>]*>([^<]*)</${tag("Note")}>`, "gi"))) {
    const n = decode(m[1]);
    if (n) notes.push(n.slice(0, 500));
  }

  // TaxTotal may appear multiple times (VAT + other); prefer first document-level TaxAmount
  // WithholdingTaxTotal is separate in UBL-TR
  const vatAmount =
    sumMoneyAll(xml, "TaxTotal", "TaxAmount") ??
    (() => {
      const tax = block(xml, "TaxTotal");
      return tax ? money(tax, "TaxAmount") : null;
    })();
  // If multiple TaxTotals include withholding wrongly, withholding is usually WithholdingTaxTotal
  const withholdingVatAmount = sumMoneyAll(xml, "WithholdingTaxTotal", "TaxAmount");

  // Invoice ID is the first cbc:ID under Invoice — avoid line IDs by scoping head
  const head = xml.slice(0, 20_000);
  const invoiceNumber = first(head, "ID");

  const invoiceTypeRaw = first(xml, "InvoiceTypeCode");
  const invoiceTypeCode = invoiceTypeRaw ? asciiUpper(invoiceTypeRaw) : null;

  const invoice: ParsedInvoice = {
    documentType: kind,
    profileId,
    customizationId: first(xml, "CustomizationID"),
    invoiceTypeCode,
    invoiceNumber,
    uuid: first(xml, "UUID")?.toLowerCase() ?? null,
    issueDate: first(xml, "IssueDate"),
    issueTime: first(xml, "IssueTime"),
    supplier: {
      name: partyName(supplierParty),
      taxId: sTax.taxId,
      taxIdScheme: sTax.scheme,
      taxOffice: partyTaxOffice(supplierParty),
      address: first(supplierParty, "StreetName"),
      phone: first(supplierParty, "Telephone"),
      email: first(supplierParty, "ElectronicMail"),
      website: first(supplierParty, "WebsiteURI"),
    },
    customer: {
      name: partyName(customerParty),
      taxId: cTax.taxId,
      taxIdScheme: cTax.scheme,
      taxOffice: partyTaxOffice(customerParty),
      address: first(customerParty, "StreetName"),
      phone: first(customerParty, "Telephone"),
      email: first(customerParty, "ElectronicMail"),
      website: null,
    },
    lines,
    totals: {
      lineExtensionAmount: monetary
        ? money(monetary, "LineExtensionAmount") ?? money(monetary, "TaxExclusiveAmount")
        : null,
      discountTotal: monetary ? money(monetary, "AllowanceTotalAmount") : null,
      vatAmount,
      withholdingVatAmount,
      taxInclusiveAmount: monetary ? money(monetary, "TaxInclusiveAmount") : null,
      payableAmount: monetary ? money(monetary, "PayableAmount") : null,
      currency,
    },
    notes: notes.slice(0, 12),
    iban: null,
    bankName: null,
    bankBranch: null,
  };

  // Caution: summing ALL TaxTotal can double-count if line TaxTotals exist in XML.
  // Prefer document-level: if lineExtension + first TaxTotal works, keep; reconcile heals.
  if (invoice.totals.vatAmount != null && invoice.totals.lineExtensionAmount != null) {
    const ti = invoice.totals.taxInclusiveAmount;
    if (
      ti != null &&
      Math.abs(invoice.totals.lineExtensionAmount + invoice.totals.vatAmount - ti) > 1
    ) {
      // Likely double-counted TaxTotals — fall back to first TaxTotal only
      const tax = block(xml, "TaxTotal");
      invoice.totals.vatAmount = tax ? money(tax, "TaxAmount") : invoice.totals.vatAmount;
    }
  }

  reconcileTotals(invoice);
  return invoice;
}
