let fastPathHits = 0;
let v2Hits = 0;

export function noteFastPath(): void {
  fastPathHits += 1;
}

export function noteV2(): void {
  v2Hits += 1;
}

export function pathHitStats() {
  return { fastPathHits, v2Hits };
}
