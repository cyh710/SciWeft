"""归因诊断：把端到端错误拆成「检索未召回」与「召回了但生成没用好」。

用法：python scripts/diagnose.py --config configs/baseline.yaml --split dev --run <实验目录名>
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.eval.retrieval_metrics import parse_gold_ref_ids
from src.schemas import Chunk
from src.sparse import bm25, tfidf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--run", required=True, help="experiments/ 下的实验目录名")
    ap.add_argument("--evidence-k", type=int, default=0, help="0 = 用 config 的 top_k_after")
    args = ap.parse_args()

    cfg = yaml.safe_load((REPO / args.config).read_text(encoding="utf-8"))
    run_dir = REPO / "experiments" / args.run
    pred = pd.read_csv(run_dir / "predictions.csv", encoding="utf-8-sig")
    sol = pd.read_csv(run_dir / "solution.csv", encoding="utf-8-sig")

    chunks = []
    with open(REPO / cfg["data"]["processed_dir"] / "chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            chunks.append(Chunk(**json.loads(line)))
    retriever = {"tfidf": tfidf, "bm25": bm25}[cfg["retrieval"]["method"]]
    retriever.build(chunks)
    n_ev = args.evidence_k or cfg["rerank"]["top_k_after"]

    gold_map = {r.id: parse_gold_ref_ids(r.ref_id) for r in sol.itertuples(index=False)}
    pred_map = {r.id: (parse_gold_ref_ids(r.ref_id) if r.ref_id != "is_blank" else []) for r in pred.itertuples(index=False)}
    q_map = {r.id: r.question for r in sol.itertuples(index=False)}

    rows = []
    for qid, gold in gold_map.items():
        if not gold:
            continue
        hits = retriever.retrieve(q_map[qid], top_k=cfg["retrieval"]["top_k"])[:n_ev]
        ev_refs = []
        for h in hits:
            if h.ref_id not in ev_refs:
                ev_refs.append(h.ref_id)
        gold_in_ev = set(gold) & set(ev_refs)
        cited = set(pred_map.get(qid, []))
        rows.append({
            "id": qid, "n_gold": len(gold), "gold_in_evidence": len(gold_in_ev),
            "retrieval_hit": len(gold_in_ev) > 0,
            "gold_in_top1": ev_refs[0] in gold if ev_refs else False,
            "citation_hit": len(cited & set(gold)) > 0,
            "cited_n": len(cited), "gold_n": len(gold),
            "pred_is_blank": pred_map.get(qid) == [],
        })

    df = pd.DataFrame(rows)
    n = len(df)
    miss = df[~df["retrieval_hit"]]
    hit = df[df["retrieval_hit"]]
    print(f"== {args.run} / {args.split} / evidence_k={n_ev} / 有效题 {n} ==")
    print(f"检索未召回 gold (gold 不在证据集): {len(miss)}/{n} = {len(miss)/n:.1%}")
    print(f"  其中 gold 在 top1 文档: {int(miss['gold_in_top1'].sum())} (这说明文档在但没进证据窗口)")
    print(f"已召回 gold: {len(hit)}/{n} = {len(hit)/n:.1%}")
    if len(hit):
        print(f"  其中引用命中 gold: {int(hit['citation_hit'].sum())}/{len(hit)} = {hit['citation_hit'].mean():.1%}")
        print(f"  漏引用(gold在证据但没引): {int((~hit['citation_hit']).sum())} 题")
        over = hit[(hit["cited_n"] > hit["gold_n"])]
        print(f"  多引用(引得比gold多): {len(over)} 题")
    print(f"预测为 is_blank 的题: {int(df['pred_is_blank'].sum())}（其中 gold 非空 {int((df['pred_is_blank'] & (df['gold_n']>0)).sum())} = 过度拒答）")
    print()
    print("按 gold 篇数分层的召回率:")
    for k, g in df.groupby("n_gold"):
        print(f"  gold={k}篇: {len(g)}题, 召回率 {g['retrieval_hit'].mean():.1%}, 引用命中率 {g['citation_hit'].mean():.1%}")
    df.to_csv(run_dir / "diagnosis.csv", index=False)
    print(f"-> {run_dir / 'diagnosis.csv'}")


if __name__ == "__main__":
    main()
