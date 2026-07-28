/** Turkish invoice money: 16.000,00 / 16.000 TL / 19200.00 */
export function parseTrMoney(raw: string | null | undefined): number | null {
  if (!raw) return null;
  let s = raw.trim().replace(/\s/g, "").replace(/TL|TRY|₺/gi, "");
  if (!s) return null;

  if (s.includes(",") && s.includes(".")) {
    // 16.000,00 → 16000.00
    s = s.replace(/\./g, "").replace(",", ".");
  } else if (s.includes(",")) {
    // 16,00 → 16.00
    s = s.replace(",", ".");
  } else if (/^\d{1,3}(\.\d{3})+$/.test(s)) {
    // 16.000 or 1.234.567 → thousands separators
    s = s.replace(/\./g, "");
  }
  // else: plain 16000 or XML-style 16000.00 — leave as-is

  const n = Number.parseFloat(s);
  return Number.isFinite(n) ? n : null;
}

export function parsePercent(raw: string | null | undefined): number | null {
  if (!raw) return null;
  const m = raw.replace(",", ".").match(/(\d+(?:\.\d+)?)/);
  if (!m) return null;
  const n = Number.parseFloat(m[1]);
  return Number.isFinite(n) ? n : null;
}

export function nearlyEqual(a: number, b: number, eps = 0.02): boolean {
  return Math.abs(a - b) <= eps;
}
