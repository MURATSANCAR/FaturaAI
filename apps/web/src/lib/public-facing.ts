/** Hide internal engine names from portal UI copy. */
const TECH_RE =
  /\b(?:docling|pdftotext|poppler|tesseract|photo-?ocr|rapidocr|paddle|pp-?ocr|onnx|uvicorn|fast-?path|pdf-text|ubl-fast|extract(?:\s*v2)?|legacy|pipeline|heic-jpeg|image-input)\b/i;

const STACK_RE = /\b(?:Error:|Exception|Traceback|ECONNREFUSED|fetch failed|status code)\b/i;

const GENERIC =
  "NanobaseAI okuma sırasında bir sorun oluştu. Lütfen dosyayı kontrol edip tekrar deneyin.";

export function sanitizePublicMessage(msg: string | null | undefined): string {
  if (!msg) return GENERIC;
  const text = String(msg).trim();
  if (!text) return GENERIC;
  if (TECH_RE.test(text) || STACK_RE.test(text)) return GENERIC;
  return text;
}
