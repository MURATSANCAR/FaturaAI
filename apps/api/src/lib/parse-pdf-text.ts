import { parsePercent, parseTrMoney } from "./money.js";
import type { InvoiceLine, InvoiceParty, ParsedInvoice } from "../types.js";

function rightField(text: string, label: string): string | null {
  // "Fatura No: X" or Babymall-style "Fatura No             X" (colon optional)
  const re = new RegExp(`${label}\\s*:?\\s*([^\\n]+)`, "i");
  const m = text.match(re);
  if (!m) return null;
  const raw = m[1].trim();
  const parts = raw.split(/\s{2,}/).map((p) => p.trim()).filter(Boolean);
  return (parts[parts.length - 1] ?? raw).trim() || null;
}

function firstMatch(text: string, re: RegExp): string | null {
  const m = text.match(re);
  return m?.[1]?.trim() || null;
}

function emptyParty(): InvoiceParty {
  return {
    name: null,
    taxId: null,
    taxIdScheme: null,
    taxOffice: null,
    address: null,
    phone: null,
    email: null,
    website: null,
  };
}

function parseIssueDateTime(raw: string | null): { date: string | null; time: string | null } {
  if (!raw) return { date: null, time: null };
  const m = raw.match(
    /(\d{1,2})\s*[-./]\s*(\d{1,2})\s*[-./]\s*(\d{4})(?:\s+(\d{1,2}):(\d{2}))?/,
  );
  if (!m) return { date: null, time: null };
  const date = `${m[3]}-${m[2].padStart(2, "0")}-${m[1].padStart(2, "0")}`;
  const time = m[4] && m[5] ? `${m[4].padStart(2, "0")}:${m[5]}:00` : null;
  return { date, time };
}

function extractSupplier(text: string): InvoiceParty {
  const party = emptyParty();
  const sayinIdx = text.search(/\bSAYIN\b/i);
  const head = sayinIdx >= 0 ? text.slice(0, sayinIdx) : text.slice(0, 800);

  const lines = head
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean)
    .filter((l) => !/^e-?Ar[sş]iv\s+Fatura$/i.test(l))
    .filter((l) => !/^Sayfa\s+\d+/i.test(l));

  if (lines[0] && !/^(Tel|Web|E-?Posta|Vergi|TCKN|VKN|Kap[ıi])/i.test(lines[0])) {
    let name = lines[0];
    if (
      lines[1] &&
      /(?:LTD|ŞT[İI]|A\.?\s*Ş\.?|SAN\.|T[İI]C\.|ANON[İI]M)/i.test(lines[1]) &&
      !/^(Tel|Web|E-?Posta|Vergi|TCKN|VKN|ŞUBE)/i.test(lines[1])
    ) {
      name = `${lines[0]} ${lines[1]}`;
    }
    party.name = name.slice(0, 180);
  }

  const addrParts: string[] = [];
  for (const line of lines.slice(1)) {
    if (/^(Tel|Web|E-Posta|Vergi|TCKN|VKN)\b/i.test(line)) break;
    if (/Kap[ıi]\s*No/i.test(line) || /Türkiye|mah\.|Cad\.|Bul\./i.test(line) || /\/\s*\w+/.test(line)) {
      addrParts.push(line.replace(/^Kap[ıi]\s*No:\s*/i, "Kapı No: "));
    }
  }
  party.address = addrParts.join(", ").replace(/\s+/g, " ").trim() || null;

  party.phone = firstMatch(head, /Tel\s*:\s*([0-9\s()]+?)(?:\s+Fax|$|\n)/i)?.replace(/\s+/g, "") || null;
  party.email = firstMatch(head, /E-?Posta\s*:\s*([^\s]+)/i);
  party.website = firstMatch(head, /Web\s*Sitesi\s*:\s*([^\s]+)/i);
  if (party.website && /Özelleştirme/i.test(party.website)) party.website = null;

  party.taxOffice = firstMatch(head, /Vergi\s*Dairesi\s*:\s*([^\n]+)/i)
    ?.split(/\s{2,}/)[0]
    ?.trim() || null;

  const tckn = firstMatch(head, /TCKN\s*:\s*(\d{11})/i);
  const vkn = firstMatch(head, /VKN\s*:\s*(\d{10})/i);
  if (tckn) {
    party.taxId = tckn;
    party.taxIdScheme = "TCKN";
  } else if (vkn) {
    party.taxId = vkn;
    party.taxIdScheme = "VKN";
  }

  return party;
}

function looksLikeAddressLine(line: string): boolean {
  return (
    /^(Konut|Kap[ıi]|Ye[sş]il|\/\s*Türkiye)/i.test(line) ||
    /\b(mah\.|Mah\.|Bul\.|Cad\.|Sk\.|No:|daire|sitesi)\b/i.test(line) ||
    /\b(Ankara|İstanbul|Istanbul|İzmir|Izmir|Karabük|Etimesgut|Çorum|Corum)\b/i.test(line)
  );
}

function extractCustomer(text: string): InvoiceParty {
  const party = emptyParty();
  const sayinIdx = text.search(/\bSAYIN\b/i);
  if (sayinIdx < 0) return party;

  const block = text.slice(sayinIdx);
  const lines = block
    .split("\n")
    .map((l) => {
      // Strip right-column metadata (Özelleştirme / Senaryo / Fatura …)
      return l
        .replace(
          /\s{2,}(Özelleştirme|Senaryo|Fatura\s+Tipi|Fatura\s+No|Fatura\s+Tarihi|Fatura\s+Saati|Sipari[sş]\s+No|Sipari[sş]\s+Tarihi).*$/i,
          "",
        )
        .trim();
    })
    .filter(Boolean);

  const nameParts: string[] = [];
  const addrParts: string[] = [];
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i];
    if (/^(Web|E-Posta|Tel|Vergi|VKN|TCKN|ETTN|S[ıi]ra|Mal\s+Hizmet|NOTLAR|Not:)/i.test(line)) {
      break;
    }
    if (looksLikeAddressLine(line)) {
      addrParts.push(line);
      continue;
    }
    if (nameParts.length === 0) {
      nameParts.push(line);
    } else if (!addrParts.length && line.length < 80 && !/\d{5}/.test(line)) {
      // rare multi-line company name
      nameParts.push(line);
    }
  }
  party.name = nameParts.join(" ").replace(/\s+/g, " ").trim() || null;
  // "SEVECEN MARKET SEVECEN MARKET" → tek kez
  if (party.name) {
    const halves = party.name.split(/\s+/);
    const mid = Math.floor(halves.length / 2);
    if (mid > 0) {
      const a = halves.slice(0, mid).join(" ");
      const b = halves.slice(mid).join(" ");
      if (a === b) party.name = a;
    }
  }
  party.address = addrParts.join(", ").replace(/\s+/g, " ").trim() || null;

  // Customer tax info sits near SAYIN block (may share line with right column)
  const near = block.slice(0, 1200);
  party.taxOffice =
    firstMatch(near, /Vergi\s*Dairesi\s*:\s*([^\n]+)/i)
      ?.split(/\s{2,}/)[0]
      ?.trim() || null;
  const vkn = firstMatch(near, /VKN\s*:\s*(\d{10})/i);
  const tckn = firstMatch(near, /TCKN\s*:\s*(\d{11})/i);
  const vknTckn = firstMatch(near, /VKN\s*\/\s*TCKN\s*:?\s*(\d{10,11})/i);
  if (tckn) {
    party.taxId = tckn;
    party.taxIdScheme = "TCKN";
  } else if (vkn) {
    party.taxId = vkn;
    party.taxIdScheme = "VKN";
  } else if (vknTckn) {
    party.taxId = vknTckn;
    party.taxIdScheme = vknTckn.length === 11 ? "TCKN" : "VKN";
  }

  party.email = firstMatch(near, /E-?Posta\s*:\s*([^\s]+)/i);
  if (party.email && /Özelleştirme|Senaryo/i.test(party.email)) party.email = null;

  return party;
}

/** Row with explicit unit (GİB classic): "1 Nakliye 1 Adet 16.000 TL %0 … %20 3.200 TL" */
const LINE_WITH_UNIT =
  /^\s*(\d+)\s+(.+?)\s+(\d+(?:[.,]\d+)?)\s+(Adet|C62|KGM|MTR|LTR|Saat|Gün|Ay|Yıl|NIU)\s+([\d.\s]+(?:,\d{2})?)\s*TL?\s+%?([\d.,]+)\s+([\d.\s]+(?:,\d{2})?)\s*TL?\s+.+?%([\d.,]+)\s+([\d.\s]+(?:,\d{2})?)\s*TL?/i;

/**
 * EDM / e-ticaret layout (no Adet): 
 * "1  Katlanabilir …  1  4.396,99 TL  %20,00  879,40 TL  4.396,99 TL"
 */
const LINE_EDM =
  /^\s*(\d+)\s+(.*?)\s+(\d+(?:[.,]\d+)?)\s+([\d.\s]+,\d{2}|\d+)\s*TL\s+%([\d.,]+)\s+([\d.\s]+,\d{2})\s*TL(?:\s+([\d.\s]+,\d{2})\s*TL)?\s*$/i;

/**
 * Moda Jant / ürün kodlu layout — fiyatlar üst satırda olabilir:
 * "1  468621  205/55 R17 …  4 Adet  %18,00  808,47"
 */
const LINE_PRODUCT_CODE =
  /^\s*(\d+)\s+(\d{4,})\s+(.+?)\s+(\d+(?:[.,]\d+)?)\s+(Adet|C62|KGM|MTR|LTR|NIU)\s+%([\d.,]+)\s+([\d.\s]+,\d{2})\s*$/i;

/**
 * Babymall / TRY layout:
 * "1    1,0   108,3300 TRY   33,33 TRY   %20.00   74,99 TRY"
 */
const LINE_TRY =
  /^\s*(\d+)\s+(\d+,\d+)\s+([\d.]+,\d+)\s*TRY\s+([\d.]+,\d+)\s*TRY\s+%([\d.]+)\s+([\d.]+,\d+)\s*TRY\s*$/i;

const MONEY_PAIR_ROW = /^\s*([\d.\s]+,\d{2})\s+([\d.\s]+,\d{2})\s*$/;

function isLineContinuation(line: string): boolean {
  const t = line.trim();
  if (!t) return false;
  if (/^\d+\s/.test(t)) return false;
  if (/^(S[ıi]ra|Mal\s*\/?\s*Hizmet|No\b|NOTLAR|Not:|ETTN|Ödenecek|ÖDENECEK|Vergiler|Hesaplanan|Toplam|NET TOPLAM|Ta[sş][ıi]yan)/i.test(t)) {
    return false;
  }
  if (/^[%\d]/.test(t) && /TL|TRY/.test(t)) return false;
  if (MONEY_PAIR_ROW.test(t)) return false;
  if (LINE_TRY.test(t)) return false;
  return t.length > 1 && t.length < 160;
}

function extractLines(text: string): InvoiceLine[] {
  const rawLines = text.replace(/\u000c/g, "\n").split("\n");
  const lines: InvoiceLine[] = [];

  for (let i = 0; i < rawLines.length; i++) {
    const row = rawLines[i];
    const withUnit = row.match(LINE_WITH_UNIT);
    if (withUnit) {
      const withholding = rawLines
        .slice(i, i + 3)
        .join("\n")
        .match(/KDV\s*TEVK[İI]FAT[^\n]*\(([^)]+)\)\s*=\s*([\d.\s]+(?:,\d{2})?)/i);
      lines.push({
        id: withUnit[1],
        name: withUnit[2].replace(/\s+/g, " ").trim(),
        quantity: Number.parseFloat(withUnit[3].replace(",", ".")),
        unit: withUnit[4],
        unitPrice: parseTrMoney(withUnit[5]),
        discountRate: parsePercent(withUnit[6]),
        discountAmount: parseTrMoney(withUnit[7]),
        vatRate: parsePercent(withUnit[8]),
        vatAmount: parseTrMoney(withUnit[9]),
        withholdingNote: withholding
          ? `KDV Tevkifat (${withholding[1]}) = ${withholding[2].trim()} TL`
          : null,
        lineTotal: null,
      });
      continue;
    }

    const coded = row.match(LINE_PRODUCT_CODE);
    if (coded) {
      let unitPrice: number | null = null;
      let lineTotal: number | null = null;
      if (i > 0) {
        const pair = rawLines[i - 1].match(MONEY_PAIR_ROW);
        if (pair) {
          unitPrice = parseTrMoney(pair[1]);
          lineTotal = parseTrMoney(pair[2]);
        }
      }
      const qty = Number.parseFloat(coded[4].replace(",", "."));
      if (unitPrice == null && lineTotal != null && qty > 0) {
        unitPrice = Number((lineTotal / qty).toFixed(2));
      }
      lines.push({
        id: coded[1],
        name: `${coded[2]} ${coded[3]}`.replace(/\s+/g, " ").trim(),
        quantity: qty,
        unit: coded[5],
        unitPrice,
        discountRate: null,
        discountAmount: null,
        vatRate: parsePercent(coded[6]),
        vatAmount: parseTrMoney(coded[7]),
        withholdingNote: null,
        lineTotal,
      });
      continue;
    }

    const tryLine = row.match(LINE_TRY);
    if (tryLine) {
      const nameParts: string[] = [];
      if (i > 0 && isLineContinuation(rawLines[i - 1])) {
        nameParts.push(rawLines[i - 1].trim());
      }
      if (i + 1 < rawLines.length && isLineContinuation(rawLines[i + 1])) {
        nameParts.push(rawLines[i + 1].trim());
      }
      lines.push({
        id: tryLine[1],
        name: nameParts.join(" ").replace(/\s+/g, " ").trim() || null,
        quantity: Number.parseFloat(tryLine[2].replace(",", ".")),
        unit: "Adet",
        unitPrice: parseTrMoney(tryLine[3]),
        discountRate: null,
        discountAmount: parseTrMoney(tryLine[4]),
        vatRate: parsePercent(tryLine[5]),
        vatAmount: null,
        withholdingNote: null,
        lineTotal: parseTrMoney(tryLine[6]),
      });
      continue;
    }

    const edm = row.match(LINE_EDM);
    if (!edm) continue;

    const nameParts: string[] = [];
    if (edm[2].trim()) nameParts.push(edm[2].replace(/\s+/g, " ").trim());
    if (i > 0 && isLineContinuation(rawLines[i - 1])) {
      nameParts.unshift(rawLines[i - 1].trim());
    }
    if (i + 1 < rawLines.length && isLineContinuation(rawLines[i + 1])) {
      const nextIsNewLine =
        i + 2 < rawLines.length &&
        (LINE_EDM.test(rawLines[i + 2]) ||
          LINE_WITH_UNIT.test(rawLines[i + 2]) ||
          LINE_PRODUCT_CODE.test(rawLines[i + 2]));
      if (!nextIsNewLine) {
        nameParts.push(rawLines[i + 1].trim());
      }
    }

    const unitPrice = parseTrMoney(edm[4]);
    const lineTotal = edm[7] ? parseTrMoney(edm[7]) : unitPrice;
    lines.push({
      id: edm[1],
      name: nameParts.join(" ").replace(/\s+/g, " ").trim() || null,
      quantity: Number.parseFloat(edm[3].replace(",", ".")),
      unit: "Adet",
      unitPrice,
      discountRate: null,
      discountAmount: null,
      vatRate: parsePercent(edm[5]),
      vatAmount: parseTrMoney(edm[6]),
      withholdingNote: null,
      lineTotal,
    });
  }

  for (const line of lines) {
    if (line.lineTotal == null && line.unitPrice != null && line.quantity != null) {
      line.lineTotal = Number((line.unitPrice * line.quantity).toFixed(2));
    }
  }

  return lines;
}

function labeledAmount(text: string, label: string): number | null {
  // Optional (%18) / colon / TL|TRY: "ÖDENECEK TUTAR   359,96 TRY"
  const re = new RegExp(
    `${label}(?:\\s*\\([^)]*\\))?\\s*:?\\s*([\\d.\\s]+,\\d{2,})\\s*(?:TL|TRY)?`,
    "gi",
  );
  const matches = [...text.matchAll(re)];
  if (matches.length === 0) return null;
  return parseTrMoney(matches[matches.length - 1][1]);
}

export function parseGibPdfText(text: string, fileName = ""): ParsedInvoice {
  const normalized = text.replace(/\u000c/g, "\n");

  let documentType: ParsedInvoice["documentType"] = "unknown";
  if (/e-?Ar[sş]iv\s+Fatura/i.test(normalized) || /EARSIVFATURA/i.test(normalized)) {
    documentType = "earsiv";
  } else if (
    /e-?Fatura/i.test(normalized) ||
    /EFATURA|TICARIFATURA|TEMELFATURA|IHRACATFATURA|KAMUFATURA/i.test(normalized)
  ) {
    documentType = "efatura";
  }

  const customizationId = rightField(normalized, "Özelleştirme No");
  const profileId = rightField(normalized, "Senaryo");
  const invoiceTypeCode = rightField(normalized, "Fatura Tipi");
  let invoiceNumber =
    rightField(normalized, "Fatura No") ||
    firstMatch(fileName, /([A-Z]{2,5}\d{10,})/i)?.toUpperCase() ||
    null;
  if (invoiceNumber) invoiceNumber = invoiceNumber.replace(/\s+/g, "").toUpperCase();

  const issueRaw =
    rightField(normalized, "Fatura Tarihi") ||
    firstMatch(
      normalized,
      /(?:^|\n)[^\n]*?\bTarih\s*:\s*(\d{1,2}\s*[-./]\s*\d{1,2}\s*[-./]\s*\d{4})/i,
    );
  const { date: issueDate, time: issueTimeFromDate } = parseIssueDateTime(issueRaw);
  const saati = rightField(normalized, "Fatura Saati");
  const olusma = rightField(normalized, "Oluşma Zamanı");
  const duzenlemeZamani = rightField(normalized, "Düzenleme Zamanı");
  const issueTime =
    issueTimeFromDate ??
    (saati && /^\d{1,2}:\d{2}(:\d{2})?$/.test(saati.trim())
      ? saati.trim().length === 5
        ? `${saati.trim()}:00`
        : saati.trim()
      : null) ??
    (olusma?.match(/(\d{1,2}:\d{2}:\d{2})/)?.[1] ?? null) ??
    (duzenlemeZamani?.match(/(\d{1,2}:\d{2}:\d{2})/)?.[1] ?? null);

  const uuid = firstMatch(
    normalized,
    /ETTN\s*:\s*([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/i,
  )?.toLowerCase() || null;

  const notesBlock = normalized.split(/NOTLAR\s*:/i)[1] ?? "";
  const notesFromBlock = notesBlock
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l && !/^#?\s*$/.test(l) && !/^e-Ar[sş]iv izni/i.test(l) && !/^Bu Fatura/i.test(l))
    .slice(0, 12);

  const notesFromNot = [...normalized.matchAll(/^\s*Not:\s*(.+)$/gim)]
    .map((m) => m[1].trim())
    .filter((n) => n && !/^#\s*$/.test(n) && n.length > 1)
    .slice(0, 12);

  const notesFromStar = [...normalized.matchAll(/^\s*\*\s*(.+)$/gim)]
    .map((m) => m[1].trim())
    .filter((n) => n.length > 1)
    .slice(0, 12);

  const notes =
    notesFromBlock.length > 0
      ? notesFromBlock
      : notesFromNot.length > 0
        ? notesFromNot
        : notesFromStar;

  const iban =
    firstMatch(normalized, /İ?BAN\s*:\s*(TR[\d\s]+)/i)?.replace(/\s+/g, "").toUpperCase() ||
    null;
  const bankFromIbanLine = firstMatch(
    normalized,
    /([A-ZÇĞİÖŞÜa-zçğıöşü ]+BANKASI)\s*\/\s*I?İ?BAN/i,
  );
  const bankName =
    bankFromIbanLine?.replace(/\s+/g, " ").trim() ??
    notes.find((n) => /BANKASI|BANKA/i.test(n) && !/İBAN|IBAN/i.test(n)) ??
    null;
  const bankBranch =
    notes.find((n) => /ŞUBE|SUBESI|ŞUBESİ/i.test(n)) ?? null;

  const netToplam =
    labeledAmount(normalized, "Mal Hizmet Toplam Tutarı") ??
    labeledAmount(normalized, "NET TOPLAM");
  const discountTotal =
    labeledAmount(normalized, "Toplam İskonto") ??
    labeledAmount(normalized, "TOPLAM [İI]SKONTO");
  // NET TOPLAM iskonto öncesi olabiliyor; matrah = net - iskonto
  const lineExtensionAmount =
    netToplam != null && discountTotal != null && discountTotal > 0
      ? Number((netToplam - discountTotal).toFixed(2))
      : netToplam;

  return {
    documentType,
    profileId,
    customizationId,
    invoiceTypeCode,
    invoiceNumber,
    uuid,
    issueDate,
    issueTime,
    supplier: extractSupplier(normalized),
    customer: extractCustomer(normalized),
    lines: extractLines(normalized),
    totals: {
      lineExtensionAmount,
      discountTotal,
      withholdingVatAmount: labeledAmount(normalized, "Hesaplanan KDV Tevkifat"),
      vatAmount:
        labeledAmount(normalized, "Hesaplanan KDV(?!\\s*Tevkifat)") ??
        labeledAmount(normalized, "KDV"),
      taxInclusiveAmount:
        labeledAmount(normalized, "Vergiler Dahil Toplam Tutar") ??
        labeledAmount(normalized, "VERG[İI] DAH[İI]L TOPLAM TUTAR"),
      payableAmount:
        labeledAmount(normalized, "Ödenecek Tutar") ??
        labeledAmount(normalized, "ÖDENECEK TUTAR"),
      currency: "TRY",
    },
    notes,
    iban,
    bankName,
    bankBranch,
  };
}

export function validateInvoice(invoice: ParsedInvoice): string[] {
  const warnings: string[] = [];
  if (!invoice.invoiceNumber) warnings.push("Fatura numarası bulunamadı");
  if (!invoice.uuid) warnings.push("ETTN bulunamadı");
  if (!invoice.issueDate) warnings.push("Fatura tarihi bulunamadı");
  if (!invoice.supplier.name) warnings.push("Satıcı unvanı bulunamadı");
  if (!invoice.customer.name) warnings.push("Alıcı unvanı bulunamadı");
  if (!invoice.totals.payableAmount) warnings.push("Ödenecek tutar bulunamadı");
  if (invoice.lines.length === 0) warnings.push("Mal/hizmet kalemi bulunamadı");

  const { lineExtensionAmount, vatAmount, taxInclusiveAmount, payableAmount, withholdingVatAmount } =
    invoice.totals;
  if (
    lineExtensionAmount != null &&
    vatAmount != null &&
    taxInclusiveAmount != null &&
    Math.abs(lineExtensionAmount + vatAmount - taxInclusiveAmount) > 0.05
  ) {
    warnings.push("Matrah + KDV, vergiler dahil toplam ile uyuşmuyor");
  }
  if (
    taxInclusiveAmount != null &&
    payableAmount != null &&
    withholdingVatAmount != null &&
    Math.abs(taxInclusiveAmount - withholdingVatAmount - payableAmount) > 0.05
  ) {
    warnings.push("Ödenecek tutar, tevkifat düşülmüş toplam ile uyuşmuyor");
  }

  return warnings;
}
