"""按 metadata.csv 的 pinned URL 礼貌下载全部语料 PDF。

官方礼仪（违反会被主办方处理）：每篇只抓一次、本地缓存、禁止并发爬虫。
- 断点续传：已存在的 ref_id 直接跳过；
- 限速：请求间 sleep（默认 5s），100+ 篇约 15-20 分钟，挂后台跑即可；
- manifest：记录 url / sha256 / 大小 / 状态，写入 data/processed/manifest.json，
  这是全部实验的数据快照凭证。
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"
CORPUS = RAW / "corpus"
PROCESSED = REPO / "data" / "processed"
MANIFEST = PROCESSED / "manifest.json"
SLEEP_SECONDS = 5.0
TIMEOUT = 90
UA = "wattbot-nlp/0.1 (course project; one-time pinned-version fetch)"

# 语料 URL 全部是外网（arXiv/出版社/机构站），默认走本机 Clash 代理；置 CORPUS_PROXY="" 可禁用
PROXY_URL = os.environ.get("CORPUS_PROXY", "http://127.0.0.1:7897")
PROXIES = {"http": PROXY_URL, "https": PROXY_URL} if PROXY_URL else None


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"snapshot_date": None, "documents": {}}


def main() -> None:
    meta = RAW / "metadata.csv"
    if not meta.exists():
        sys.exit("[x] 缺少 data/raw/metadata.csv，先跑 scripts/download_competition_data.py")

    CORPUS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    docs = manifest["documents"]

    with open(meta, newline="", encoding="utf-8-sig") as f:  # 官方 CSV 带 BOM
        rows = list(csv.DictReader(f))
    print(f"[..] 语料共 {len(rows)} 篇，已缓存 {sum(1 for r in rows if (CORPUS / (r['id'] + '.pdf')).exists())} 篇")

    for row in tqdm(rows, desc="corpus"):
        ref_id, url = row["id"], row["url"]
        out = CORPUS / f"{ref_id}.pdf"
        if out.exists() and docs.get(ref_id, {}).get("status") == "ok":
            continue  # 断点续传：每篇只抓一次

        entry = {"url": url, "fetched_at": datetime.now(timezone.utc).isoformat()}
        try:
            resp = requests.get(url, headers={"User-Agent": UA}, proxies=PROXIES, timeout=TIMEOUT)
            resp.raise_for_status()
            if not resp.content[:5].startswith(b"%PDF"):
                # 有些出版社 URL 会跳 HTML 落地页，记录下来人工处理
                entry["status"] = "not_pdf"
                entry["bytes"] = len(resp.content)
            else:
                out.write_bytes(resp.content)
                entry.update(status="ok", sha256=sha256_of(out), bytes=len(resp.content))
        except Exception as exc:  # noqa: BLE001  单篇失败不中断全库
            entry["status"] = "error"
            entry["error"] = repr(exc)[:300]
        docs[ref_id] = entry

        manifest["snapshot_date"] = manifest["snapshot_date"] or datetime.now(timezone.utc).date().isoformat()
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(SLEEP_SECONDS)

    bad = [k for k, v in docs.items() if v.get("status") != "ok"]
    print(f"[done] ok={len(docs) - len(bad)}  bad={len(bad)}")
    if bad:
        print("[!] 需人工复核（重新下载或找替代来源）：")
        for k in bad:
            print(f"    {k}: {docs[k].get('status')} {docs[k].get('error', '')[:120]}")


if __name__ == "__main__":
    main()
