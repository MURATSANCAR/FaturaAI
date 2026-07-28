import { parsePercent, parseTrMoney } from "./money.js";
import type { InvoiceLine, InvoiceParty, ParsedInvoice } from "../types.js";

function rightField(text: string, label: string): string | null {
  const re = new RegExp(
    `${label}\\s*:\\s*([^\\n]+)`,
    "i",
  );
  const m = text.match(re);
  if (!m) return null;
  // Layout often puts value after lots of spaces on same visual row —
  // take last non-empty token group after label on that match.
  const raw = m[1].trim();
  // Prefer value that appears after large whitespace gap (right column)
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
    .filter((l) => !/^e-?Ar[sş]iv\s+Fatura$/i.test(l));

  if (lines[0] && !/^(Tel|Web|E-Posta|Vergi|TCKN|VKN|Kap[ıi])/i.test(lines[0])) {
    party.name = lines[0].slice(0, 160);
  }

  const addrParts: string[] = [];
  for (const line of lines.slice(1)) {
    if (/^(Tel|Web|E-Posta|Vergi|TCKN|VKN)\b/i.test(line)) break;
    if (/Kap[ıi]\s*No/i.test(line) || /Türkiye|mah\.|Cad\.|Bul\./i.test(line) || /\/\s*\w+/.test(line)) {
      addrParts.push(line.replace(/^Kap[ıi]\s*No:\s*/i, "Kapı No: "));
    }
  }
  party.address = addrParts.join(", ").replace(/\s+/g, " ").trim() || null;

  party.phone = firstMatch(head, /Tel\s*:\s*([0-9\s]+?)(?:\s+Fax|$)/i)?.replace(/\s+/g, "") || null;
  party.email = firstMatch(head, /E-Posta\s*:\s*([^\s]+)/i);
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
  party.address = addrParts.join(", ").replace(/\s+/g, " ").trim() || null;

  // Customer tax info sits near SAYIN block (may share line with right column)
  const near = block.slice(0, 1200);
  party.taxOffice =
    firstMatch(near, /Vergi\s*Dairesi\s*:\s*([^\n]+)/i)
      ?.split(/\s{2,}/)[0]
      ?.trim() || null;
  const vkn = firstMatch(near, /VKN\s*:\s*(\d{10})/i);
  const tckn = firstMatch(near, /TCKN\s*:\s*(\d{11})/i);
  if (vkn) {
    party.taxId = vkn;
    party.taxIdScheme = "VKN";
  } else if (tckn) {
    party.taxId = tckn;
    party.taxIdScheme = "TCKN";
  }

  party.email = firstMatch(near, /E-Posta\s*:\s*([^\s]+)/i);
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

function isLineContinuation(line: string): boolean {
  const t = line.trim();
  if (!t) return false;
  if (/^\d+\s/.test(t)) return false;
  if (/^(S[ıi]ra|Mal\s+Hizmet|No\b|NOTLAR|Not:|ETTN|Ödenecek|Vergiler|Hesaplanan|Toplam|Mal Hizmet Toplam)/i.test(t)) {
    return false;
  }
  if (/^[%\d]/.test(t) && /TL/.test(t)) return false;
  return t.length > 1 && t.length < 120;
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

    const edm = row.match(LINE_EDM);
    if (!edm) continue;

    const nameParts: string[] = [];
    if (edm[2].trim()) nameParts.push(edm[2].replace(/\s+/g, " ").trim());
    // Wrap: en fazla 1 satır yukarı / 1 satır aşağı (EDM çok satırlı ürün adları)
    if (i > 0 && isLineContinuation(rawLines[i - 1])) {
      nameParts.unshift(rawLines[i - 1].trim());
    }
    if (i + 1 < rawLines.length && isLineContinuation(rawLines[i + 1])) {
      // Bir sonraki satır yeni kalem numarasıysa, aradaki wrap o kaleme aittir
      const nextIsNewLine =
        i + 2 < rawLines.length &&
        (LINE_EDM.test(rawLines[i + 2]) || LINE_WITH_UNIT.test(rawLines[i + 2]));
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
  // Allow optional (%20) style suffix between label and amount
  const re = new RegExp(
    `${label}(?:\\s*\\([^)]*\\))?\\s+([\\d.\\s]+,\\d{2})\\s*TL?`,
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
  } else if (/e-?Fatura/i.test(normalized)) {
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

  const issueRaw = rightField(normalized, "Fatura Tarihi");
  const { date: issueDate, time: issueTimeFromDate } = parseIssueDateTime(issueRaw);
  const saati = rightField(normalized, "Fatura Saati");
  const issueTime =
    issueTimeFromDate ??
    (saati && /^\d{1,2}:\d{2}(:\d{2})?$/.test(saati.trim())
      ? saati.trim().length === 5
        ? `${saati.trim()}:00`
        : saati.trim()
      : null);

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

  const notes = notesFromBlock.length > 0 ? notesFromBlock : notesFromNot;

  const iban =
    firstMatch(normalized, /İ?BAN\s*:\s*(TR[\d\s]+)/i)?.replace(/\s+/g, "").toUpperCase() ||
    null;
  const bankName =
    notes.find((n) => /BANKASI|BANKA/i.test(n) && !/İBAN|IBAN/i.test(n)) ?? null;
  const bankBranch =
    notes.find((n) => /ŞUBE|SUBESI|ŞUBESİ/i.test(n)) ?? null;

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
      lineExtensionAmount: labeledAmount(normalized, "Mal Hizmet Toplam Tutarı"),
      discountTotal: labeledAmount(normalized, "Toplam İskonto"),
      // Tevkifat satırını yakalamamak için önce tam eşleşme, sonra genel KDV
      withholdingVatAmount: labeledAmount(normalized, "Hesaplanan KDV Tevkifat"),
      vatAmount: labeledAmount(normalized, "Hesaplanan KDV(?!\\s*Tevkifat)"),
      taxInclusiveAmount: labeledAmount(normalized, "Vergiler Dahil Toplam Tutar"),
      payableAmount: labeledAmount(normalized, "Ödenecek Tutar"),
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
