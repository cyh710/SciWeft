"""检索层指标：文档级 Recall@K 与 MRR（W1）。

口径：
- 先把检索出的 chunk 按 ref_id 去重得到文档排名，K 指文档数不是 chunk 数；
- 一题多个 gold ref_id 时，R@K 的分母是 gold 文档数；
- MRR 取首个命中的 gold 文档的倒数排名。
"""
from __future__ import annotations

import ast
import re

import pandas as pd

_LIST_RE = re.compile(r"^\[.*\]$")


def parse_gold_ref_ids(raw) -> list[str]:
    """train_QA 的 ref_id 可能是单个 id，也可能是字符串化列表。"""
    if pd.isna(raw) or str(raw).strip() in {"", "is_blank"}:
        return []
    text = str(raw).strip()
    if _LIST_RE.match(text):
        try:
            return [str(x).strip() for x in ast.literal_eval(text)]
        except (ValueError, SyntaxError):
            pass
    return [t.strip() for t in re.split(r"[;,]", text) if t.strip()]


def dedup_hits_to_docs(hits) -> list[str]:
    """chunk 命中列表 -> 按首次出现排序的去重文档列表。"""
    seen: list[str] = []
    for h in hits:
        if h.ref_id not in seen:
            seen.append(h.ref_id)
    return seen


def recall_at_k(doc_rank: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return 0.0
    top = set(doc_rank[:k])
    return len(top & set(gold)) / len(set(gold))


def reciprocal_rank(doc_rank: list[str], gold: list[str]) -> float:
    gold_set = set(gold)
    for rank, ref in enumerate(doc_rank, start=1):
        if ref in gold_set:
            return 1.0 / rank
    return 0.0


def summarize(per_question: list[dict]) -> dict:
    """per_question: [{id, gold, docs}] -> 汇总指标。"""
    ks = (1, 5, 10)
    rows = []
    for q in per_question:
        docs = dedup_hits_to_docs(q["docs"])
        rows.append({
            **{f"recall@{k}": recall_at_k(docs, q["gold"], k) for k in ks},
            "mrr": reciprocal_rank(docs, q["gold"]),
        })
    df = pd.DataFrame(rows)
    return df.mean().round(4).to_dict()
