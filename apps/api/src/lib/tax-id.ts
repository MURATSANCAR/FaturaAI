/** Turkish TCKN / VKN checksum helpers. */

/** GİB e-Arşiv anonymous / final-consumer placeholder (checksum fails by design). */
export const PLACEHOLDER_TCKN = "11111111111";

const OCR_DIGIT_MAP: Record<string, string> = {
  O: "0",
  o: "0",
  О: "0",
  İ: "1",
  I: "1",
  i: "1",
  ı: "1",
  l: "1",
  L: "1",
  "|": "1",
  S: "5",
  s: "5",
  B: "8",
  G: "6",
  Z: "2",
};

export function digitsOnly(value: string | null | undefined): string {
  if (!value) return "";
  return value.replace(/\D/g, "");
}

export function normalizeOcrDigits(value: string | null | undefined): string {
  if (!value) return "";
  const mapped = [...value].map((ch) => OCR_DIGIT_MAP[ch] ?? ch).join("");
  return digitsOnly(mapped);
}

export function isPlaceholderTckn(value: string | null | undefined): boolean {
  return digitsOnly(value) === PLACEHOLDER_TCKN;
}

export function isPlaceholderTaxId(value: string | null | undefined): boolean {
  const n = digitsOnly(value) || normalizeOcrDigits(value);
  if (!n) return false;
  if (n === PLACEHOLDER_TCKN || n === "0000000000" || n === "00000000000") return true;
  if ((n.length === 10 || n.length === 11) && new Set(n).size === 1) return true;
  return false;
}

export function isValidTckn(value: string | null | undefined): boolean {
  const n = normalizeOcrDigits(value) || digitsOnly(value);
  if (isPlaceholderTaxId(n)) return false;
  if (n.length !== 11 || !/^\d+$/.test(n) || n[0] === "0") return false;
  const d = n.split("").map((c) => Number(c));
  const odd = d[0] + d[2] + d[4] + d[6] + d[8];
  const even = d[1] + d[3] + d[5] + d[7];
  const d10 = (((odd * 7 - even) % 10) + 10) % 10;
  const d11 = d.slice(0, 10).reduce((a, b) => a + b, 0) % 10;
  return d[9] === d10 && d[10] === d11;
}

export function isValidVkn(value: string | null | undefined): boolean {
  const n = normalizeOcrDigits(value) || digitsOnly(value);
  if (isPlaceholderTaxId(n)) return false;
  if (n.length !== 10 || !/^\d+$/.test(n)) return false;
  let total = 0;
  for (let i = 0; i < 9; i++) {
    const tmp = (Number(n[i]) + (9 - i)) % 10;
    let powered = (tmp * 2 ** (9 - i)) % 9;
    if (tmp !== 0 && powered === 0) powered = 9;
    total += powered;
  }
  const check = (10 - (total % 10)) % 10;
  return check === Number(n[9]);
}

export function isValidTaxId(
  value: string | null | undefined,
  scheme?: "VKN" | "TCKN" | null,
): boolean {
  const n = normalizeOcrDigits(value) || digitsOnly(value);
  if (!n || isPlaceholderTaxId(n)) return false;
  if (scheme === "TCKN" || (!scheme && n.length === 11)) return isValidTckn(n);
  if (scheme === "VKN" || (!scheme && n.length === 10)) return isValidVkn(n);
  if (n.length === 11) return isValidTckn(n);
  if (n.length === 10) return isValidVkn(n);
  return false;
}

export function coerceTaxId(
  value: string | null | undefined,
): { taxId: string; scheme: "VKN" | "TCKN" } | null {
  const n = normalizeOcrDigits(value);
  if (!n || isPlaceholderTaxId(n)) return null;
  if (n.length === 11) return { taxId: n, scheme: "TCKN" };
  if (n.length === 10) return { taxId: n, scheme: "VKN" };
  return null;
}
