"""加权 RRF 融合扫描（W2）：不同 sparse/dense 权重下的文档级检索指标。

用法：HF_ENDPOINT=https://hf-mirror.com python scripts/sweep_fusion.py
结果追加 experiments/retrieval_log.csv（method=hybrid_w{ws}）。
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.dense import dense
from src.eval.retrieval_metrics import (
    dedup_hits_to_docs, parse_gold_ref_ids, recall_at_k, reciprocal_rank,
)
from src.schemas import Chunk, Hit
from src.sparse import bm25


def fused_hits(query: str, w_s: float, w_d: float, k: int = 60, top_k: int = 50) -> list[Hit]:
    fused: dict[str, list] = {}
    for hits, w in ((bm25.retrieve(query, top_k=top_k), w_s),
                    (dense.retrieve(query, top_k=top_k), w_d)):
        for h in hits:
            e = fused.setdefault(h.chunk_id, [h.ref_id, 0.0])
            e[1] += w / (k + h.rank)
    ranked = sorted(fused.items(), key=lambda kv: -kv[1][1])[:top_k]
    return [Hit(chunk_id=c, ref_id=v[0], score=v[1], rank=i + 1) for i, (c, v) in enumerate(ranked)]


def main() -> None:
    chunks = [Chunk(**json.loads(l)) for l in open(REPO / "data/processed/chunks.jsonl", encoding="utf-8")]
    bm25.build(chunks)
    dense.build(chunks, REPO / "data" / "cache")
    sol = pd.read_csv(REPO / "data/raw/train_QA.csv", encoding="utf-8-sig")
    split = pd.read_csv(REPO / "data/processed/split.csv")
    qs = sol[sol["id"].isin(split[split["split"] == "dev"]["id"])]
    gold = {r.id: parse_gold_ref_ids(r.ref_id) for r in qs.itertuples(index=False)}
    gold = {k: v for k, v in gold.items() if v}

    out = REPO / "experiments" / "retrieval_log.csv"
    for w_s, w_d in [(1.0, 0.0), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4), (0.5, 0.5)]:
        r = {1: [], 5: [], 10: []}
        mrr = []
        for qid, g in gold.items():
            docs = dedup_hits_to_docs(fused_hits(qs[qs["id"] == qid].iloc[0]["question"], w_s, w_d))
            for k in r:
                r[k].append(recall_at_k(docs, g, k))
            mrr.append(reciprocal_rank(docs, g))
        run = {"timestamp": datetime.now().isoformat(timespec="seconds"),
               "config": "scripts/sweep_fusion.py", "method": f"hybrid_w{w_s:.1f}",
               "split": "dev", "n_questions": len(gold), "n_chunks": len(chunks),
               **{f"recall@{k}": round(sum(v) / len(v), 4) for k, v in r.items()},
               "mrr": round(sum(mrr) / len(mrr), 4)}
        new = not out.exists()
        with open(out, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(run))
            if new:
                w.writeheader()
            w.writerow(run)
        print(json.dumps(run, ensure_ascii=False))


if __name__ == "__main__":
    main()
