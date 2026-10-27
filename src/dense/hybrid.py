"""RRF Hybrid（M4）：BM25 与 Dense 的倒数排名融合。

异构分数不可直接相加，只在排名上融合（Reciprocal Rank Fusion）：
    RRF(chunk) = Σ_ranker 1 / (rrf_k + rank_ranker(chunk))
两路各取 top_k 候选，融合后按 chunk 输出标准 Hit 列表。
"""
from __future__ import annotations

from src.schemas import Chunk, Hit
from src.sparse import bm25
from src.dense import dense


def build(chunks: list[Chunk], cache_dir=None) -> None:
    bm25.build(chunks)
    dense.build(chunks, cache_dir)


def retrieve(query: str, top_k: int = 50, rrf_k: int = 60,
             w_sparse: float = 0.5, w_dense: float = 0.5) -> list[Hit]:
    """加权 RRF。0.5/0.5 = 标准 RRF（M4 定义）；W2 扫描结论：
    BM25-only R@1/MRR 最强，加 Dense 只换深度召回（R@10 +3.1pp @0.5/0.5），
    0.7/0.3 是深度收益/精度损失的最优折中。"""
    sparse_hits = bm25.retrieve(query, top_k=top_k)
    dense_hits = dense.retrieve(query, top_k=top_k)
    fused: dict[str, dict] = {}
    for hits, w in ((sparse_hits, w_sparse), (dense_hits, w_dense)):
        for h in hits:
            entry = fused.setdefault(h.chunk_id, {"ref_id": h.ref_id, "score": 0.0, "snippet": h.snippet})
            entry["score"] += w / (rrf_k + h.rank)
    ranked = sorted(fused.items(), key=lambda kv: -kv[1]["score"])[:top_k]
    return [
        Hit(chunk_id=cid, ref_id=v["ref_id"], score=v["score"], rank=rank + 1, snippet=v["snippet"])
        for rank, (cid, v) in enumerate(ranked)
    ]
