import { parsePercent, parseTrMoney } from "./money.js";
import { digitsOnly, isValidTaxId } from "./tax-id.js";
import type { InvoiceLine, InvoiceParty, ParsedInvoice } from "../types.js";

function rightField(text: string, label: string): string | null {
  // "Fatura No: X" or "Fatura No             X" (colon optional)
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
  // ISO (Docling/GİB tables): 2026-07-30 13:14:02
  const iso = raw.match(
    /(?<!\d)(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{1,2}):(\d{2})(?::(\d{2}))?)?/,
  );
  if (iso) {
    const year = Number(iso[1]);
    const month = Number(iso[2]);
    const day = Number(iso[3]);
    if (month >= 1 && month <= 12 && day >= 1 && day <= 31 && year >= 1990 && year <= 2100) {
      const date = `${iso[1]}-${iso[2]}-${iso[3]}`;
      const time =
        iso[4] != null && iso[5] != null
          ? `${iso[4].padStart(2, "0")}:${iso[5]}:${(iso[6] ?? "00").padStart(2, "0")}`
          : null;
      return { date, time };
    }
  }
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
    .map((l) => l.replace(/^#+\s*/, "").trim())
    .filter(Boolean)
    .filter((l) => !/^e-?Ar[sş]iv\s+Fatura$/i.test(l))
    .filter((l) => !/^Sayfa\s+\d+/i.test(l));

  if (
    lines[0] &&
    !/^(Tel|Web|E-?Posta|Vergi|TCKN|VKN|Kap[ıi]|Kurumsal\s+Ofis)/i.test(lines[0])
  ) {
    let name = lines[0];
    if (
      lines[1] &&
      /(?:LTD|ŞT[İI]|A\.?\s*Ş\.?|SAN\.|T[İI]C\.|ANON[İI]M|VE\s+SAN)/i.test(lines[1]) &&
      !/^(Tel|Web|E-?Posta|Vergi|TCKN|VKN|ŞUBE|Kurumsal\s+Ofis)/i.test(lines[1]) &&
      // continuation legal-form line, not a full second company
      (lines[1].length < 40 || /^(VE\s+)?(?:SAN|T[İI]C|LTD|TA[ŞS])/i.test(lines[1]))
    ) {
      name = `${lines[0]} ${lines[1]}`;
    }
    party.name = name.slice(0, 180);
  }

  const addrParts: string[] = [];
  for (const line of lines.slice(1)) {
    if (/^(Tel|Web|E-Posta|Vergi|TCKN|VKN)\b/i.test(line)) break;
    if (
      /^Kurumsal\s+Ofis/i.test(line) ||
      /Kap[ıi]\s*No/i.test(line) ||
      /Türkiye|mah\.|Mahallesi|Cad\.|Bul\./i.test(line) ||
      /\/\s*\w+/.test(line)
    ) {
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
    /^(Konut|Kap[ıi]|\/\s*Türkiye)/i.test(line) ||
    /\b(mah\.|Mah\.|Bul\.|Cad\.|Sk\.|Sok\.|No:|daire|sitesi|Apartman|Blok)\b/i.test(line) ||
    /\b\d{5}\b/.test(line) ||
    /\/\s*[A-ZÇĞİÖŞÜa-zçğıöşü]{3,}/.test(line)
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
  // Duplicate name on same line: "FOO FOO" → "FOO"
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
  /^\s*(\d+)\s+(.+?)\s+(\d+(?:[.,]\d+)?)\s+(Adet|C62|KGM|MTR|LTR|Saat|Gün|Ay|Yıl|NIU)\s+([\d.\s]+(?:,\d{2,})?)\s*TL?\s+%?([\d.,]+)\s+([\d.\s]+(?:,\d{2})?)\s*TL?\s+.+?%([\d.,]+)\s+([\d.\s]+(?:,\d{2})?)\s*TL?/i;

/**
 * Qty glued to unit + multi-decimal unit price (Exbilisim / marketplace PDFs):
 * "1     6Adet 3.749,16667TL  %0,00  0,00TL  %20,00  4.499,00TL  22.495,00TL"
 * Product name usually wraps on surrounding lines.
 */
const LINE_GLUED_UNIT =
  /^\s*(\d+)\s+(.*?)\s*(\d+(?:[.,]\d+)?)\s*(Adet|C62|KGM|MTR|LTR|Saat|Gün|Ay|Yıl|NIU)\s+([\d.\s]+,\d{2,})\s*TL\s+%([\d.,]+)\s+([\d.\s]+,\d{2})\s*TL\s+%([\d.,]+)\s+([\d.\s]+,\d{2})\s*TL(?:\s+([\d.\s]+,\d{2})\s*TL)?/i;

/**
 * EDM / e-ticaret layout (no Adet): 
 * "1  Katlanabilir …  1  4.396,99 TL  %20,00  879,40 TL  4.396,99 TL"
 */
const LINE_EDM =
  /^\s*(\d+)\s+(.*?)\s+(\d+(?:[.,]\d+)?)\s+([\d.\s]+,\d{2}|\d+)\s*TL\s+%([\d.,]+)\s+([\d.\s]+,\d{2})\s*TL(?:\s+([\d.\s]+,\d{2})\s*TL)?\s*$/i;

/**
 * Product-code layout — prices may sit on the previous row:
 * "1  468621  DESCRIPTION  4 Adet  %18,00  808,47"
 */
const LINE_PRODUCT_CODE =
  /^\s*(\d+)\s+(\d{4,})\s+(.+?)\s+(\d+(?:[.,]\d+)?)\s+(Adet|C62|KGM|MTR|LTR|NIU)\s+%([\d.,]+)\s+([\d.\s]+,\d{2})\s*$/i;

/**
 * TRY currency line layout:
 * "1    1,0   108,3300 TRY   33,33 TRY   %20.00   74,99 TRY"
 */
const LINE_TRY =
  /^\s*(\d+)\s+(\d+,\d+)\s+([\d.]+,\d+)\s*TRY\s+([\d.]+,\d+)\s*TRY\s+%([\d.]+)\s+([\d.]+,\d+)\s*TRY\s*$/i;

const MONEY_PAIR_ROW = /^\s*([\d.\s]+,\d{2})\s+([\d.\s]+,\d{2})\s*$/;

function isLineContinuation(line: string): boolean {
  const t = line.trim();
  if (!t) return false;
  if (/^\d+\s/.test(t)) return false;
  if (
    /^(S[ıi]ra|Mal\s*\/?\s*Hizmet|No\b|NOTLAR|Not:|ETTN|Ödenecek|ÖDENECEK|Vergiler|Hesaplanan|Toplam|NET TOPLAM|Ta[sş][ıi]yan|Mal\s*Hizmet\s*Toplam)/i.test(
      t,
    )
  ) {
    return false;
  }
  if (/^[%\d]/.test(t) && /TL|TRY/.test(t)) return false;
  if (MONEY_PAIR_ROW.test(t)) return false;
  if (LINE_TRY.test(t)) return false;
  if (LINE_GLUED_UNIT.test(t) || LINE_WITH_UNIT.test(t) || LINE_EDM.test(t) || LINE_PRODUCT_CODE.test(t)) {
    return false;
  }
  return t.length > 1 && t.length < 160;
}

function isNewLineItemRow(line: string): boolean {
  return (
    LINE_GLUED_UNIT.test(line) ||
    LINE_WITH_UNIT.test(line) ||
    LINE_EDM.test(line) ||
    LINE_PRODUCT_CODE.test(line) ||
    LINE_TRY.test(line)
  );
}

function collectWrappedLineName(
  rawLines: string[],
  index: number,
  inlineName: string,
): string | null {
  const parts: string[] = [];
  if (inlineName.trim()) parts.push(inlineName.replace(/\s+/g, " ").trim());

  for (let j = index - 1; j >= Math.max(0, index - 4); j--) {
    if (!isLineContinuation(rawLines[j])) break;
    parts.unshift(rawLines[j].trim());
  }
  for (let j = index + 1; j < Math.min(rawLines.length, index + 5); j++) {
    if (isNewLineItemRow(rawLines[j])) break;
    if (/Mal\s*Hizmet\s*Toplam|Ödenecek|Hesaplanan\s+KDV|Vergiler\s+Dahil/i.test(rawLines[j])) {
      break;
    }
    if (!isLineContinuation(rawLines[j])) break;
    // Continuation immediately before the next line-item belongs to that next item
    if (j + 1 < rawLines.length && isNewLineItemRow(rawLines[j + 1])) break;
    parts.push(rawLines[j].trim());
  }
  const name = parts.join(" ").replace(/\s+/g, " ").trim();
  return name || null;
}

function extractLines(text: string): InvoiceLine[] {
  const rawLines = text.replace(/\u000c/g, "\n").split("\n");
  const lines: InvoiceLine[] = [];

  for (let i = 0; i < rawLines.length; i++) {
    const row = rawLines[i];

    const glued = row.match(LINE_GLUED_UNIT);
    if (glued) {
      lines.push({
        id: glued[1],
        name: collectWrappedLineName(rawLines, i, glued[2] ?? ""),
        quantity: Number.parseFloat(glued[3].replace(",", ".")),
        unit: glued[4],
        unitPrice: parseTrMoney(glued[5]),
        discountRate: parsePercent(glued[6]),
        discountAmount: parseTrMoney(glued[7]),
        vatRate: parsePercent(glued[8]),
        vatAmount: parseTrMoney(glued[9]),
        withholdingNote: null,
        lineTotal: glued[10] ? parseTrMoney(glued[10]) : null,
      });
      continue;
    }

    const withUnit = row.match(LINE_WITH_UNIT);
    if (withUnit) {
      const withholding = rawLines
        .slice(i, i + 3)
        .join("\n")
        .match(/KDV\s*TEVK[İI]FAT[^\n]*\(([^)]+)\)\s*=\s*([\d.\s]+(?:,\d{2})?)/i);
      lines.push({
        id: withUnit[1],
        name: collectWrappedLineName(rawLines, i, withUnit[2]),
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
      lines.push({
        id: tryLine[1],
        name: collectWrappedLineName(rawLines, i, ""),
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

    const unitPrice = parseTrMoney(edm[4]);
    const lineTotal = edm[7] ? parseTrMoney(edm[7]) : unitPrice;
    lines.push({
      id: edm[1],
      name: collectWrappedLineName(rawLines, i, edm[2] ?? ""),
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

/** Sum repeating amount rows (multi-rate KDV / tevkifat etc.). */
function sumLabeledAmounts(text: string, label: string): number | null {
  const re = new RegExp(
    `${label}(?:\\s*\\([^)]*\\))?\\s*:?\\s*([\\d.\\s]+,\\d{2,})\\s*(?:TL|TRY)?`,
    "gi",
  );
  const amounts = [...text.matchAll(re)]
    .map((m) => parseTrMoney(m[1]))
    .filter((n): n is number => n != null && n > 0);
  if (amounts.length === 0) return null;
  if (amounts.length === 1) return amounts[0];
  return Number(amounts.reduce((a, b) => a + b, 0).toFixed(2));
}

/**
 * Multi-rate VAT: "KDV (%10.00) 23,62" + "KDV (%20.00) 291,42".
 * Also footnotes "Kdv Tutarı:23,62" when rate lines missing.
 * Does not include KDV Matrahı / Tevkifat.
 */
function extractVatAmount(text: string): number | null {
  const rateLineRe =
    /(?:Hesaplanan\s+)?KDV(?!\s*(?:TEVK|Tevkifat|Matrah[ıi]?))(?:\s*\(\s*%?\s*[\d.,]+\s*%?\s*\))\s*:?\s*([\d.\s]+,\d{2,})/gi;
  const rateAmounts = [...text.matchAll(rateLineRe)]
    .map((m) => parseTrMoney(m[1]))
    .filter((n): n is number => n != null && n > 0);
  if (rateAmounts.length > 0) {
    return Number(rateAmounts.reduce((a, b) => a + b, 0).toFixed(2));
  }

  // Footnotes: "%10 Kdv Matrahı:... Kdv Tutarı:23,62 TRY"
  const footnoteRe = /Kdv\s*Tutar[ıi]\s*:?\s*([\d.\s]+,\d{2,})/gi;
  const footnotes = [...text.matchAll(footnoteRe)]
    .map((m) => parseTrMoney(m[1]))
    .filter((n): n is number => n != null && n > 0);
  if (footnotes.length > 0) {
    // Dedupe identical pairs that repeat (totals + note)
    const unique = [...new Set(footnotes.map((n) => n.toFixed(2)))].map(Number);
    return Number(unique.reduce((a, b) => a + b, 0).toFixed(2));
  }

  return (
    labeledAmount(text, "Hesaplanan KDV(?!\\s*Tevkifat)") ??
    labeledAmount(text, "KDV(?!\\s*(?:TEVK|Tevkifat|Matrah))")
  );
}

/** Multi-rate withholding: sum all "Hesaplanan KDV Tevkifat (...)" rows. */
function extractWithholdingVatAmount(text: string): number | null {
  const summed = sumLabeledAmounts(text, "Hesaplanan KDV Tevkifat");
  if (summed != null) return summed;
  // "KDV TEVKİFAT" / tevkifat tutarı variants
  const alt =
    sumLabeledAmounts(text, "KDV Tevkifat") ??
    labeledAmount(text, "Tevkifat Tutarı") ??
    labeledAmount(text, "Hesaplanan Tevkifat");
  return alt;
}

const EARSIV_PROFILES = ["EARSIVFATURA", "EARSIV"];
const EFATURA_PROFILES = [
  "TEMELFATURA",
  "TICARIFATURA",
  "IHRACAT",
  "IHRACATFATURA",
  "YOLCUBERABER",
  "YOLCUBERABERFATURA",
  "KAMU",
  "KAMUFATURA",
  "ENERJI",
  "ILAC_TIBBICIHAZ",
  "HKS",
];

function asciiUpper(s: string): string {
  return s
    .replace(/İ/g, "I")
    .replace(/İ/g, "I")
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

function normalizeProfileId(raw: string | null | undefined): string | null {
  if (!raw) return null;
  const p = asciiUpper(raw.trim());
  return p || null;
}

function normalizeInvoiceTypeCode(raw: string | null | undefined): string | null {
  if (!raw) return null;
  const t = asciiUpper(raw.trim());
  // common UI typos / spaced forms
  if (t === "SATIS" || t === "SATIŞ") return "SATIS";
  return t || null;
}

function detectDocumentType(
  text: string,
  profileId: string | null,
): ParsedInvoice["documentType"] {
  const p = profileId ?? "";
  if (EARSIV_PROFILES.some((x) => p.includes(x)) || /EARSIV/i.test(p)) return "earsiv";
  if (EFATURA_PROFILES.some((x) => p.includes(x))) return "efatura";
  if (/e-?Ar[sş]iv(?:\s+Fatura)?|EARSIVFATURA|e-?Ar[sş]iv\s+izni/i.test(text)) return "earsiv";
  if (
    /e-?Fatura\b|EFATURA|TEMELFATURA|TICARIFATURA|IHRACAT|KAMUFATURA|YOLCUBERABER/i.test(
      text,
    )
  ) {
    return "efatura";
  }
  return "unknown";
}

function firstAmount(text: string, labels: string[]): number | null {
  for (const label of labels) {
    const v = labeledAmount(text, label);
    if (v != null) return v;
  }
  return null;
}

function extractTotals(text: string): ParsedInvoice["totals"] {
  const netToplam = firstAmount(text, [
    "Mal Hizmet Toplam Tutarı",
    "Mal\\s*/?\\s*Hizmet Toplam Tutarı",
    "NET TOPLAM",
    "Ara Toplam",
    "Vergiler Hariç Toplam",
    "Vergiler Hariç Tutar",
  ]);
  // Explicit matrah rows (sometimes after discount) — sum multi-rate matrah footnotes
  const matrahLabeled =
    sumLabeledAmounts(text, "KDV Matrah[ıi]") ??
    labeledAmount(text, "TaxExclusiveAmount");

  const discountTotal = firstAmount(text, [
    "Toplam İskonto",
    "TOPLAM [İI]SKONTO",
    "Toplam Iskonto",
    "İskonto Toplamı",
    "AllowanceTotalAmount",
  ]);

  let lineExtensionAmount: number | null = null;
  if (matrahLabeled != null) {
    lineExtensionAmount = matrahLabeled;
  } else if (netToplam != null && discountTotal != null && discountTotal > 0) {
    lineExtensionAmount = Number((netToplam - discountTotal).toFixed(2));
  } else {
    lineExtensionAmount = netToplam;
  }

  const taxInclusiveAmount = firstAmount(text, [
    "Vergiler Dahil Toplam Tutar",
    "VERG[İI] DAH[İI]L TOPLAM TUTAR",
    "Vergiler Dahil Toplam",
    "TaxInclusiveAmount",
  ]);
  const payableAmount = firstAmount(text, [
    "Ödenecek Tutar",
    "ÖDENECEK TUTAR",
    "Odenecek Tutar",
    "PayableAmount",
  ]);

  return {
    lineExtensionAmount,
    discountTotal,
    withholdingVatAmount: extractWithholdingVatAmount(text),
    vatAmount: extractVatAmount(text),
    taxInclusiveAmount,
    payableAmount,
    currency: "TRY",
  };
}

/**
 * Auto-heal common GIB total inconsistencies (multi-rate KDV miss, missing vat, etc.).
 */
export function reconcileTotals(invoice: ParsedInvoice): void {
  const t = invoice.totals;
  const near = (a: number, b: number, eps = 0.05) => Math.abs(a - b) <= eps;

  // If matrah + vat != ti, prefer implied VAT from ti - matrah (covers missed rate rows)
  if (
    t.lineExtensionAmount != null &&
    t.taxInclusiveAmount != null &&
    t.taxInclusiveAmount >= t.lineExtensionAmount
  ) {
    const impliedVat = Number(
      (t.taxInclusiveAmount - t.lineExtensionAmount).toFixed(2),
    );
    if (t.vatAmount == null) {
      t.vatAmount = impliedVat;
    } else if (!near(t.lineExtensionAmount + t.vatAmount, t.taxInclusiveAmount)) {
      // Under/over-count VAT rates → trust arithmetic from solid totals
      t.vatAmount = impliedVat;
    }
  }

  // ISTISNA / 0% : vat may be 0
  if (
    t.lineExtensionAmount != null &&
    t.taxInclusiveAmount != null &&
    near(t.lineExtensionAmount, t.taxInclusiveAmount) &&
    (t.vatAmount == null || t.vatAmount === 0)
  ) {
    t.vatAmount = 0;
  }

  // Payable defaults to taxInclusive when no withholding
  if (t.payableAmount == null && t.taxInclusiveAmount != null) {
    if (t.withholdingVatAmount != null) {
      t.payableAmount = Number(
        (t.taxInclusiveAmount - t.withholdingVatAmount).toFixed(2),
      );
    } else {
      t.payableAmount = t.taxInclusiveAmount;
    }
  }

  // If payable + withholding ≈ taxInclusive but withholding null, leave as-is
  if (
    t.taxInclusiveAmount != null &&
    t.payableAmount != null &&
    t.withholdingVatAmount == null &&
    t.taxInclusiveAmount > t.payableAmount + 0.05
  ) {
    t.withholdingVatAmount = Number(
      (t.taxInclusiveAmount - t.payableAmount).toFixed(2),
    );
  }

  // Fill matrah from payable - vat when missing
  if (
    t.lineExtensionAmount == null &&
    t.payableAmount != null &&
    t.vatAmount != null &&
    t.withholdingVatAmount == null
  ) {
    t.lineExtensionAmount = Number((t.payableAmount - t.vatAmount).toFixed(2));
  }
  if (
    t.lineExtensionAmount == null &&
    t.taxInclusiveAmount != null &&
    t.vatAmount != null
  ) {
    t.lineExtensionAmount = Number(
      (t.taxInclusiveAmount - t.vatAmount).toFixed(2),
    );
  }
}

export function parseGibPdfText(text: string, fileName = ""): ParsedInvoice {
  const normalized = text.replace(/\u000c/g, "\n");

  const customizationId = rightField(normalized, "Özelleştirme No");
  const profileIdRaw =
    rightField(normalized, "Senaryo") ||
    firstMatch(normalized, /ProfileID\s*:?\s*([A-Z0-9_]+)/i);
  const profileId = profileIdRaw ? normalizeProfileId(profileIdRaw) : null;
  const invoiceTypeRaw =
    rightField(normalized, "Fatura Tipi") ||
    firstMatch(normalized, /InvoiceTypeCode\s*:?\s*([A-ZÇĞİÖŞÜ0-9_]+)/i);
  const invoiceTypeCode = invoiceTypeRaw
    ? normalizeInvoiceTypeCode(invoiceTypeRaw)
    : null;
  const documentType = detectDocumentType(normalized, profileId);

  let invoiceNumber =
    rightField(normalized, "Fatura No") ||
    firstMatch(fileName, /([A-Z]{2,5}\d{10,})/i)?.toUpperCase() ||
    null;
  if (invoiceNumber) {
    invoiceNumber = invoiceNumber.replace(/\s+/g, "").toUpperCase();
    if (!/^[A-Z]{2,5}\d{10,16}$/.test(invoiceNumber)) invoiceNumber = null;
  }

  const issueRaw =
    rightField(normalized, "Fatura Tarihi") ||
    rightField(normalized, "Düzenleme Tarihi") ||
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
    olusma?.match(/(\d{1,2}:\d{2}:\d{2})/)?.[1] ??
    duzenlemeZamani?.match(/(\d{1,2}:\d{2}:\d{2})/)?.[1] ??
    null;

  const uuid = firstMatch(
    normalized,
    /ETTN\s*:?\s*([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/i,
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

  const invoice: ParsedInvoice = {
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
    totals: extractTotals(normalized),
    notes,
    iban,
    bankName,
    bankBranch,
  };
  reconcileTotals(invoice);
  return invoice;
}

function sanitizePartyTaxId(party: InvoiceParty, role: string): string[] {
  const warnings: string[] = [];
  const raw = digitsOnly(party.taxId);
  if (!raw) {
    party.taxId = null;
    party.taxIdScheme = null;
    return warnings;
  }

  let scheme = party.taxIdScheme;
  if (!scheme) {
    if (raw.length === 11) scheme = "TCKN";
    else if (raw.length === 10) scheme = "VKN";
  }

  if (isValidTaxId(raw, scheme)) {
    party.taxId = raw;
    party.taxIdScheme = raw.length === 11 ? "TCKN" : "VKN";
    return warnings;
  }

  const label =
    scheme ?? (raw.length === 11 ? "TCKN" : raw.length === 10 ? "VKN" : "vergi kimlik");
  warnings.push(`${role} ${label} geçersiz (doğrulama başarısız) — yok sayıldı`);
  party.taxId = null;
  party.taxIdScheme = null;
  return warnings;
}

export function validateInvoice(invoice: ParsedInvoice): string[] {
  // Reconcile again in case caller mutated totals
  reconcileTotals(invoice);
  const warnings: string[] = [];
  const supplierTaxWarnings = sanitizePartyTaxId(invoice.supplier, "Satıcı");
  const customerTaxWarnings = sanitizePartyTaxId(invoice.customer, "Alıcı");
  warnings.push(...supplierTaxWarnings, ...customerTaxWarnings);
  if (!invoice.invoiceNumber) warnings.push("Fatura numarası bulunamadı");
  if (!invoice.uuid) warnings.push("ETTN bulunamadı");
  if (!invoice.issueDate) warnings.push("Fatura tarihi bulunamadı");
  if (!invoice.supplier.name) warnings.push("Satıcı unvanı bulunamadı");
  if (!invoice.customer.name) warnings.push("Alıcı unvanı bulunamadı");
  if (!supplierTaxWarnings.length && !invoice.supplier.taxId) {
    warnings.push("Satıcı VKN/TCKN bulunamadı");
  }
  if (!customerTaxWarnings.length && !invoice.customer.taxId) {
    warnings.push("Alıcı VKN/TCKN bulunamadı");
  }
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
