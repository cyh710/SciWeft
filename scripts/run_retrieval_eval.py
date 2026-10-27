"""检索评价入口（W1）：dev 集上跑配置指定的检索器，输出文档级 Recall@K / MRR。

用法：
  python scripts/run_retrieval_eval.py --config configs/baseline.yaml --method bm25
  python scripts/run_retrieval_eval.py --config configs/baseline.yaml --method tfidf
结果追加写入 experiments/retrieval_log.csv（config + 方法 + 指标 + chunk 数）。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.eval.retrieval_metrics import parse_gold_ref_ids, summarize
from src.schemas import Chunk
from src.sparse import bm25, tfidf
from src.dense import dense, hybrid


def load_chunks(path: Path) -> list[Chunk]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            chunks.append(Chunk(**json.loads(line)))
    return chunks


def get_retriever(method: str, cfg: dict):
    if method == "tfidf":
        return tfidf, []
    if method == "bm25":
        return bm25, []
    cache_dir = REPO / "data" / "cache"
    if method == "dense":
        return dense, [cache_dir]
    if method == "hybrid":
        return hybrid, [cache_dir]
    raise ValueError(method)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--method", choices=["tfidf", "bm25", "dense", "hybrid"], default="bm25")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "all"])
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    cfg = yaml.safe_load((REPO / args.config).read_text(encoding="utf-8"))
    chunks = load_chunks(REPO / cfg["data"]["processed_dir"] / "chunks.jsonl")
    retriever, build_args = get_retriever(args.method, cfg)
    retriever.build(chunks, *build_args)

    train = pd.read_csv(REPO / "data" / "raw" / "train_QA.csv", encoding="utf-8-sig")
    split = pd.read_csv(REPO / "data" / "processed" / "split.csv")
    if args.split != "all":
        ids = split[split["split"] == args.split]["id"]
        train = train[train["id"].isin(ids)]

    per_q = []
    for row in train.itertuples(index=False):
        gold = parse_gold_ref_ids(row.ref_id)
        if not gold:  # is_NA 题没有 gold 引用，检索评价中跳过
            continue
        per_q.append({"id": row.id, "gold": gold, "docs": retriever.retrieve(row.question, top_k=args.top_k)})

    metrics = summarize(per_q)
    run = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": args.config, "method": args.method, "split": args.split,
        "n_questions": len(per_q), "n_chunks": len(chunks), **metrics,
    }
    out = REPO / "experiments" / "retrieval_log.csv"
    new = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(run))
        if new:
            writer.writeheader()
        writer.writerow(run)
    print(json.dumps(run, ensure_ascii=False, indent=2))
    print(f"-> appended to {out}")


if __name__ == "__main__":
    main()
