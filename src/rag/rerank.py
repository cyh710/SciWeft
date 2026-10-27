"""Cross-Encoder 重排（M5）：对召回候选联合编码精排。

- 输入检索层的 top_k 候选（chunk 全文），输出重排后的前 top_k_after；
- 模型 lazy 加载；批量推理；CPU 可跑（MiniLM-L-6 约几十 ms/对）；
- 分数替换为重排分，rank 重排后重编——下游证据选择无感知。
"""
from __future__ import annotations

from src.schemas import Chunk, Hit

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # 可换 BAAI/bge-reranker-base（更强，CPU 慢 3-5 倍）
_models: dict[str, object] = {}


def _get_model(name: str | None = None):
    key = name or MODEL_NAME
    if key not in _models:
        from sentence_transformers import CrossEncoder
        _models[key] = CrossEncoder(key)
    return _models[key]


def rerank(query: str, hits: list[Hit], chunks_by_id: dict[str, Chunk],
           top_k_after: int = 8, max_chars: int = 4000,
           model_name: str | None = None) -> list[Hit]:
    """对 hits 重排并返回前 top_k_after。文本截断到 max_chars 控制延迟。"""
    if not hits:
        return hits
    model = _get_model(model_name)
    pairs = [(query, chunks_by_id[h.chunk_id].text[:max_chars] if h.chunk_id in chunks_by_id else h.snippet)
             for h in hits]
    scores = model.predict(pairs, batch_size=32, show_progress_bar=False)
    rescored = [
        Hit(chunk_id=h.chunk_id, ref_id=h.ref_id, score=float(s), rank=0, snippet=h.snippet)
        for h, s in zip(hits, scores)
    ]
    rescored.sort(key=lambda x: -x.score)
    for i, h in enumerate(rescored):
        h.rank = i + 1
    return rescored[:top_k_after]
