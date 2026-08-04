#!/usr/bin/env python3
"""Fix portal.nanobase.ai nginx for FaturaAI paths."""
from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

conf = Path("/etc/nginx/sites-enabled/portal.nanobase.ai")
avail = Path("/etc/nginx/sites-available/portal.nanobase.ai")
bak = Path(f"/tmp/portal.nanobase.ai.bak.fatura.{datetime.now():%Y%m%d%H%M%S}")
shutil.copy2(conf, bak)
print("backup", bak)

text = conf.read_text()
server_pos = text.find("\nserver {")
if server_pos < 0:
    raise SystemExit("no server block")

pre, post = text[:server_pos], text[server_pos:]

# Drop orphan location blocks before server
pre = re.sub(r"\n\s*location\s+=\s+/fatura\s*\{.*?\n\s*\}\n", "\n", pre, flags=re.S)
pre = re.sub(r"\n\s*location\s+/fatura/\s*\{.*?\n\s*\}\n", "\n", pre, flags=re.S)
pre = re.sub(r"\n\s*location\s+/fatura-api/\s*\{.*?\n\s*\}\n", "\n", pre, flags=re.S)

# Remove existing fatura/legal_api upstreams then re-add clean ones
pre = re.sub(r"\nupstream fatura_api \{.*?\n\}\n", "\n", pre, flags=re.S)
pre = re.sub(r"\nupstream legal_api \{.*?\n\}\n", "\n", pre, flags=re.S)

# If a dangling fragment remains (server/keepalive without upstream), strip orphan lines
# that look like leftover from corrupted legal_api
pre = re.sub(
    r"\n\s*# SpecAI Legal backend[^\n]*\n\s*server 127\.0\.0\.1:8098;\n\s*keepalive 16;\n\}\n",
    "\n",
    pre,
)

ups = """
upstream fatura_api {
    server 127.0.0.1:8105;
    keepalive 16;
}

upstream legal_api {
    # SpecAI Legal backend — not 8089 (Superset)
    server 127.0.0.1:8098;
    keepalive 16;
}

"""
m = re.search(r"\nupstream\s+\w+", pre)
if not m:
    raise SystemExit("no upstream left to anchor")
pre = pre[: m.start()] + "\n" + ups + pre[m.start() :]

# Remove fatura locations inside server
post = re.sub(
    r"\n\s*# FaturaAI[^\n]*\n(?:\s*location[^\n]*\{.*?\n\s*\}\n)+",
    "\n",
    post,
    flags=re.S,
)
post = re.sub(r"\n\s*location /fatura-api/ \{.*?\n\s*\}\n", "\n", post, flags=re.S)
post = re.sub(r"\n\s*location = /fatura \{.*?\n\s*\}\n", "\n", post, flags=re.S)
post = re.sub(r"\n\s*location /fatura/ \{.*?\n\s*\}\n", "\n", post, flags=re.S)

locations = """
    # FaturaAI — https://portal.nanobase.ai/fatura/
    location /fatura-api/ {
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Connection "";
        proxy_read_timeout 180s;
        proxy_send_timeout 180s;
        client_max_body_size 25M;
        proxy_no_cache 1;
        proxy_cache_bypass 1;
        add_header Cache-Control "no-store, no-cache, must-revalidate, max-age=0" always;
        add_header Pragma "no-cache" always;
        proxy_pass http://fatura_api/;
    }

    location = /fatura {
        return 301 /fatura/;
    }

    location /fatura/ {
        alias /data/nanobaseai/fatura/apps/web/dist/;
        index index.html;
        add_header Cache-Control "no-cache" always;
    }

"""

if "    location /legal-api/" in post:
    post = post.replace("    location /legal-api/", locations + "    location /legal-api/", 1)
elif "    location /QA/" in post:
    post = post.replace("    location /QA/", locations + "    location /QA/", 1)
else:
    post = post.replace("    index index.html;\n", "    index index.html;\n" + locations, 1)

text = pre + post
if "location " in text[: text.find("\nserver {")]:
    raise SystemExit("location still before server")
if text.count("location /fatura/") != 1:
    raise SystemExit(f"unexpected /fatura/ count: {text.count('location /fatura/')}")
if text.count("upstream fatura_api") != 1:
    raise SystemExit("fatura_api upstream count wrong")
if text.count("upstream legal_api") != 1:
    raise SystemExit("legal_api upstream count wrong")

conf.write_text(text)
avail.write_text(text)
print("wrote clean config")
print("\n".join(text.splitlines()[:45]))
