import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  isPlaceholderTckn,
  isValidTckn,
  isValidVkn,
  PLACEHOLDER_TCKN,
} from "./lib/tax-id.js";

describe("tax-id checksums", () => {
  it("accepts known valid TCKN/VKN samples", () => {
    assert.equal(isValidTckn("11440998130"), true);
    assert.equal(isValidTckn("47263025470"), true);
    assert.equal(isValidVkn("4590660068"), true);
    assert.equal(isValidVkn("8430903426"), true);
  });

  it("accepts GİB placeholder TCKN", () => {
    assert.equal(isPlaceholderTckn(PLACEHOLDER_TCKN), true);
    assert.equal(isValidTckn(PLACEHOLDER_TCKN), true);
  });

  it("rejects false-positive style TCKN values", () => {
    assert.equal(isValidTckn("12345678901"), false);
    assert.equal(isValidTckn("99999999999"), false);
  });
});
