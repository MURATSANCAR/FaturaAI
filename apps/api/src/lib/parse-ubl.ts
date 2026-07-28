import type { ParsedInvoice } from "../types.js";

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

export function isUblInvoiceXml(text: string): boolean {
  const t = text.trim();
  return /<(?:\w+:)?Invoice[\s>]/i.test(t) && /(?:CustomizationID|ProfileID|AccountingSupplierParty)/i.test(t);
}

export function parseUblInvoice(xml: string): ParsedInvoice | null {
  if (!isUblInvoiceXml(xml)) return null;
  const profileId = first(xml, "ProfileID");
  const kind = (profileId ?? "").toUpperCase().includes("EARSIV")
    ? "earsiv"
    : /TICARI|TEMEL|IHRACAT|KAMU/i.test(profileId ?? "")
      ? "efatura"
      : "ubl";

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
    lines.push({
      id: first(b, "ID"),
      name: item ? first(item, "Name") : first(b, "Name"),
      quantity: qtyRaw ? Number.parseFloat(qtyRaw.replace(",", ".")) : null,
      unit: null,
      unitPrice: money(b, "PriceAmount") ?? money(b, "Price"),
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

  return {
    documentType: kind,
    profileId,
    customizationId: first(xml, "CustomizationID"),
    invoiceTypeCode: first(xml, "InvoiceTypeCode"),
    invoiceNumber: first(xml.slice(0, 12000), "ID"),
    uuid: first(xml, "UUID")?.toLowerCase() ?? null,
    issueDate: first(xml, "IssueDate"),
    issueTime: first(xml, "IssueTime"),
    supplier: {
      name: partyName(supplierParty),
      taxId: sTax.taxId,
      taxIdScheme: sTax.scheme,
      taxOffice: first(supplierParty, "TaxScheme") ? first(block(supplierParty, "PartyTaxScheme") ?? "", "Name") : null,
      address: first(supplierParty, "StreetName"),
      phone: first(supplierParty, "Telephone"),
      email: first(supplierParty, "ElectronicMail"),
      website: first(supplierParty, "WebsiteURI"),
    },
    customer: {
      name: partyName(customerParty),
      taxId: cTax.taxId,
      taxIdScheme: cTax.scheme,
      taxOffice: null,
      address: first(customerParty, "StreetName"),
      phone: first(customerParty, "Telephone"),
      email: first(customerParty, "ElectronicMail"),
      website: null,
    },
    lines,
    totals: {
      lineExtensionAmount: monetary ? money(monetary, "LineExtensionAmount") : null,
      discountTotal: monetary ? money(monetary, "AllowanceTotalAmount") : null,
      vatAmount: (() => {
        const tax = block(xml, "TaxTotal");
        return tax ? money(tax, "TaxAmount") : null;
      })(),
      withholdingVatAmount: null,
      taxInclusiveAmount: monetary ? money(monetary, "TaxInclusiveAmount") : null,
      payableAmount: monetary ? money(monetary, "PayableAmount") : null,
      currency,
    },
    notes: notes.slice(0, 12),
    iban: null,
    bankName: null,
    bankBranch: null,
  };
}
