"""构建索引：PDF -> 逐页文本 -> chunks.jsonl（按 config 的分块参数）。

用法：python scripts/build_index.py --config configs/baseline.yaml
产出：data/processed/pages/*.json、data/processed/chunks.jsonl（chunk 数与参数写进头部注释行）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.ingest.chunk import chunk_pages_dir
from src.ingest.parse import parse_documents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load((REPO / args.config).read_text(encoding="utf-8"))
    raw = REPO / cfg["data"]["raw_dir"]
    processed = REPO / cfg["data"]["processed_dir"]

    parse_documents(raw / "metadata.csv", raw / "corpus", processed / "pages")
    chunks = chunk_pages_dir(
        processed / "pages",
        chunk_words=cfg["corpus"]["chunk_size_tokens"],
        overlap_words=cfg["corpus"]["chunk_overlap_tokens"],
    )
    out = processed / "chunks.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")
    refs = {c.ref_id for c in chunks}
    print(f"chunks: {len(chunks)} from {len(refs)} docs -> {out}")


if __name__ == "__main__":
    main()
