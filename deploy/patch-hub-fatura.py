#!/usr/bin/env python3
"""Insert FaturaAI card into portal hub bundle."""
from __future__ import annotations

import re
import sys
from pathlib import Path


def main() -> None:
    p = Path(sys.argv[1] if len(sys.argv) > 1 else "")
    if not p.exists():
        raise SystemExit(f"missing hub bundle: {p}")
    t = p.read_text()

    if 'href:"/fatura/"' in t:
        print("portal hub already has FaturaAI card")
        return

    legal = re.search(
        r'\{kind:"external",href:"/legal/",buildModule:"contracts"[^}]*\}',
        t,
    )
    if not legal:
        print("WARNING: Legal card pattern not found; skip hub patch")
        for m in re.finditer(
            r'\{kind:"external",href:"/[^"]+",buildModule:"[^"]+"[^}]{0,120}\}',
            t,
        ):
            print("candidate", m.group(0)[:180])
        raise SystemExit(1)

    card = legal.group(0)
    new_card = card.replace('href:"/legal/"', 'href:"/fatura/"').replace(
        'buildModule:"contracts"', 'buildModule:"fatura"'
    )
    t = t.replace(card, card + "," + new_card, 1)

    fatura_card_re = re.compile(
        r'\{kind:"external",href:"/fatura/",buildModule:"fatura",icon:([^,]+),'
        r'titleKey:"([^"]+)",subtitleKey:"([^"]+)",ctaKey:"([^"]+)",stepsKey:"([^"]+)"([^}]*)\}'
    )
    m = fatura_card_re.search(t)
    if m:
        old = m.group(0)
        new = (
            '{kind:"external",href:"/fatura/",buildModule:"fatura",icon:'
            + m.group(1)
            + ',titleKey:"home.fatura.title",subtitleKey:"home.fatura.subtitle",'
            + 'ctaKey:"home.fatura.cta",stepsKey:"home.fatura.steps"'
            + m.group(6)
            + "}"
        )
        t = t.replace(old, new, 1)

    marker = '"home.contracts.title"'
    idx = 0
    inserts = 0
    fatura_keys = (
        '"home.fatura.title":"FaturaAI",'
        '"home.fatura.subtitle":"e-Arşiv / e-Fatura PDF okuma",'
        '"home.fatura.cta":"Aç",'
        '"home.fatura.steps":"PDF yükle · alanları gör",'
    )
    while True:
        i = t.find(marker, idx)
        if i < 0:
            break
        window = t[max(0, i - 160) : i]
        if "home.fatura.title" in window:
            idx = i + len(marker)
            continue
        t = t[:i] + fatura_keys + t[i:]
        inserts += 1
        idx = i + len(fatura_keys) + len(marker)

    p.write_text(t)
    print(f"patched hub: FaturaAI card + {inserts} locale blocks in {p.name}")


if __name__ == "__main__":
    main()
