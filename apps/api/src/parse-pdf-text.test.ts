import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { parseGibPdfText, validateInvoice } from "./lib/parse-pdf-text.js";
import { nearlyEqual } from "./lib/money.js";

const root = join(dirname(fileURLToPath(import.meta.url)), "../../..");
const sampleText = readFileSync(
  join(root, "samples/hava-savunma.pdftotext.txt"),
  "utf8",
);

describe("parseGibPdfText — HAVA SAVUNMA sample", () => {
  const invoice = parseGibPdfText(
    sampleText,
    "HAVA_SAVUNMA_SISTEMLERI_SANAYI_GIB2026000000059.pdf",
  );

  it("reads document meta", () => {
    assert.equal(invoice.documentType, "earsiv");
    assert.equal(invoice.invoiceNumber, "GIB2026000000059");
    assert.equal(invoice.uuid, "4176f1ad-f933-44f8-9657-8a42b3d31354");
    assert.equal(invoice.issueDate, "2026-07-22");
    assert.equal(invoice.issueTime, "11:59:00");
    assert.equal(invoice.profileId, "EARSIVFATURA");
    assert.equal(invoice.invoiceTypeCode, "TEVKIFAT");
    assert.equal(invoice.customizationId, "TR1.2");
  });

  it("reads supplier (Bahattin)", () => {
    assert.match(invoice.supplier.name ?? "", /Bahattin/i);
    assert.equal(invoice.supplier.taxId, "11440998130");
    assert.equal(invoice.supplier.taxIdScheme, "TCKN");
    assert.match(invoice.supplier.taxOffice ?? "", /ALACA/i);
    assert.equal(invoice.supplier.email, "bahattinyldrm0619@gmail.com");
  });

  it("reads customer (Hava Savunma)", () => {
    assert.match(invoice.customer.name ?? "", /HAVA SAVUNMA/i);
    assert.equal(invoice.customer.taxId, "4590660068");
    assert.equal(invoice.customer.taxIdScheme, "VKN");
    assert.match(invoice.customer.taxOffice ?? "", /DOĞANBEY|DOGANBEY/i);
  });

  it("reads line item", () => {
    assert.equal(invoice.lines.length, 1);
    assert.match(invoice.lines[0].name ?? "", /Nakliye/i);
    assert.equal(invoice.lines[0].quantity, 1);
    assert.equal(invoice.lines[0].unit, "Adet");
    assert.ok(nearlyEqual(invoice.lines[0].unitPrice ?? 0, 16000));
    assert.ok(nearlyEqual(invoice.lines[0].vatRate ?? 0, 20));
    assert.ok(nearlyEqual(invoice.lines[0].vatAmount ?? 0, 3200));
  });

  it("reads totals and IBAN", () => {
    assert.ok(nearlyEqual(invoice.totals.lineExtensionAmount ?? 0, 16000));
    assert.ok(nearlyEqual(invoice.totals.vatAmount ?? 0, 3200));
    assert.ok(nearlyEqual(invoice.totals.withholdingVatAmount ?? 0, 640));
    assert.ok(nearlyEqual(invoice.totals.taxInclusiveAmount ?? 0, 19200));
    assert.ok(nearlyEqual(invoice.totals.payableAmount ?? 0, 18560));
    assert.equal(invoice.iban, "TR190001000080601901655001");
  });

  it("validates without critical warnings", () => {
    const warnings = validateInvoice(invoice);
    assert.deepEqual(warnings, []);
  });
});

describe("parseGibPdfText — KVI EDM sample", () => {
  const kviText = readFileSync(join(root, "samples/kvi.pdftotext.txt"), "utf8");
  const invoice = parseGibPdfText(kviText, "earsiv_faturaKVI2026000009854.pdf");

  it("reads meta and parties", () => {
    assert.equal(invoice.invoiceNumber, "KVI2026000009854");
    assert.equal(invoice.issueDate, "2026-06-22");
    assert.equal(invoice.issueTime, "11:36:57");
    assert.match(invoice.supplier.name ?? "", /KAMPVE/i);
    assert.equal(invoice.customer.name, "MURAT SANCAR");
    assert.equal(invoice.customer.taxId, "11111111111");
  });

  it("reads both line items including wrapped names", () => {
    assert.equal(invoice.lines.length, 2);
    assert.match(invoice.lines[0].name ?? "", /Mangal/i);
    assert.match(invoice.lines[0].name ?? "", /M1\.2|Alev/i);
    assert.doesNotMatch(invoice.lines[0].name ?? "", /KINDLE/);
    assert.ok(nearlyEqual(invoice.lines[0].unitPrice ?? 0, 4396.99));
    assert.ok(nearlyEqual(invoice.lines[0].vatAmount ?? 0, 879.4));
    assert.match(invoice.lines[1].name ?? "", /KINDLE|Çıra|Cira/i);
    assert.doesNotMatch(invoice.lines[1].name ?? "", /Mangal/);
    assert.ok(nearlyEqual(invoice.lines[1].unitPrice ?? 0, 0));
  });

  it("reads totals without line warning", () => {
    assert.ok(nearlyEqual(invoice.totals.payableAmount ?? 0, 5276.39));
    assert.deepEqual(validateInvoice(invoice), []);
  });
});

describe("parseGibPdfText — MDA Moda Jant sample", () => {
  const mdaText = readFileSync(join(root, "samples/mda.pdftotext.txt"), "utf8");
  const invoice = parseGibPdfText(mdaText, "MDA2022000002839.pdf");

  it("reads meta and parties", () => {
    assert.equal(invoice.invoiceNumber, "MDA2022000002839");
    assert.equal(invoice.issueDate, "2022-12-25");
    assert.equal(invoice.issueTime, "14:37:45");
    assert.match(invoice.supplier.name ?? "", /MODA JANT/i);
    assert.match(invoice.supplier.name ?? "", /LTD/i);
    assert.equal(invoice.customer.name, "SEVECEN MARKET");
    assert.equal(invoice.customer.taxId, "47263025470");
  });

  it("reads tire line with product code", () => {
    assert.equal(invoice.lines.length, 1);
    assert.match(invoice.lines[0].name ?? "", /205\/55/);
    assert.equal(invoice.lines[0].quantity, 4);
    assert.ok(nearlyEqual(invoice.lines[0].unitPrice ?? 0, 1122.88));
    assert.ok(nearlyEqual(invoice.lines[0].lineTotal ?? 0, 4491.53));
    assert.ok(nearlyEqual(invoice.lines[0].vatRate ?? 0, 18));
    assert.ok(nearlyEqual(invoice.lines[0].vatAmount ?? 0, 808.47));
  });

  it("reads totals and IBAN", () => {
    assert.ok(nearlyEqual(invoice.totals.payableAmount ?? 0, 5300));
    assert.ok(nearlyEqual(invoice.totals.vatAmount ?? 0, 808.47));
    assert.equal(invoice.iban, "TR690006701000000044285758");
    assert.match(invoice.bankName ?? "", /YAPI KRED/i);
    assert.deepEqual(validateInvoice(invoice), []);
  });
});

describe("parseGibPdfText — Babymall TRY samples", () => {
  it("reads 1.pdf (4 wet wipes)", () => {
    const text = readFileSync(join(root, "samples/babymall-1.pdftotext.txt"), "utf8");
    const inv = parseGibPdfText(text, "1.pdf");
    assert.equal(inv.invoiceNumber, "BBE2026000018417");
    assert.equal(inv.issueDate, "2026-07-22");
    assert.equal(inv.issueTime, "19:10:13");
    assert.match(inv.supplier.name ?? "", /ÖZELCAN|BABYMALL/i);
    assert.equal(inv.customer.name, "MURAT SANCAR");
    assert.equal(inv.lines.length, 4);
    assert.ok(nearlyEqual(inv.lines[0].lineTotal ?? 0, 74.99));
    assert.ok(nearlyEqual(inv.totals.payableAmount ?? 0, 359.96));
    assert.ok(nearlyEqual(inv.totals.discountTotal ?? 0, 133.33));
    assert.deepEqual(validateInvoice(inv), []);
  });

  it("reads 2.pdf (2 multimalt)", () => {
    const text = readFileSync(join(root, "samples/babymall-2.pdftotext.txt"), "utf8");
    const inv = parseGibPdfText(text, "2.pdf");
    assert.equal(inv.invoiceNumber, "BBE2026000018085");
    assert.equal(inv.issueDate, "2026-07-19");
    assert.equal(inv.lines.length, 2);
    assert.match(inv.lines[0].name ?? "", /Multimalt/i);
    assert.ok(nearlyEqual(inv.totals.payableAmount ?? 0, 59.9));
    assert.deepEqual(validateInvoice(inv), []);
  });

  it("reads 3.pdf multi-rate KDV (%10 + %20)", () => {
    const text = readFileSync(join(root, "samples/babymall-3.pdftotext.txt"), "utf8");
    const inv = parseGibPdfText(text, "3.pdf");
    assert.equal(inv.invoiceNumber, "BBE2026000016055");
    assert.equal(inv.lines.length, 3);
    assert.ok(nearlyEqual(inv.totals.discountTotal ?? 0, 784.58));
    assert.ok(nearlyEqual(inv.totals.lineExtensionAmount ?? 0, 1693.27));
    // Must sum both KDV rows: 23.62 + 291.42 = 315.04
    assert.ok(nearlyEqual(inv.totals.vatAmount ?? 0, 315.04));
    assert.ok(nearlyEqual(inv.totals.taxInclusiveAmount ?? 0, 2008.31));
    assert.ok(nearlyEqual(inv.totals.payableAmount ?? 0, 2008.31));
    assert.deepEqual(validateInvoice(inv), []);
  });
});
