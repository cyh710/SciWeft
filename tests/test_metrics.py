"""检索指标单测：gold 解析、去重、R@K、MRR。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.retrieval_metrics import (
    dedup_hits_to_docs, parse_gold_ref_ids, recall_at_k, reciprocal_rank,
)


class _H:
    def __init__(self, ref_id):
        self.ref_id = ref_id


def test_parse_gold():
    assert parse_gold_ref_ids("gao2025") == ["gao2025"]
    assert parse_gold_ref_ids("['patterson2021', 'jegham2025']") == ["patterson2021", "jegham2025"]
    assert parse_gold_ref_ids("is_blank") == []
    assert parse_gold_ref_ids(float("nan")) == []


def test_dedup_keeps_first_seen_order():
    docs = dedup_hits_to_docs([_H("a"), _H("b"), _H("a"), _H("c")])
    assert docs == ["a", "b", "c"]


def test_recall_multi_gold():
    rank = ["x", "a", "y", "b"]
    assert recall_at_k(rank, ["a", "b"], 4) == 1.0
    assert recall_at_k(rank, ["a", "b"], 2) == 0.5
    assert recall_at_k(rank, [], 4) == 0.0


def test_mrr():
    assert reciprocal_rank(["x", "a"], ["a"]) == 0.5
    assert reciprocal_rank(["a"], ["a"]) == 1.0
    assert reciprocal_rank(["x", "y"], ["a"]) == 0.0
