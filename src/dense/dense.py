"""Dense 检索器（M3）：BGE-small + NumPy 余弦相似度（小语料无需 FAISS）。

- chunk 向量一次性编码并缓存到 data/cache/（键 = 模型名 + chunk 数）；
- bge v1.5 对短查询建议加检索指令前缀；
- 向量全部 L2 归一化，点积即余弦。
"""
from __future__ import annotations

import re

import numpy as np

from src.schemas import Chunk, Hit

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_model = None
_corpus: np.ndarray | None = None
_chunks: list[Chunk] | None = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _tokenize(text: str) -> str:  # 保持与 sparse 相同的口径仅用于缓存键，编码用原文
    return text


def embed_corpus(chunks: list[Chunk], cache_dir) -> np.ndarray:
    """编码全部 chunk 并按 (模型, 数量) 缓存。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"emb_{MODEL_NAME.replace('/', '_')}_{len(chunks)}.npy"
    if cache.exists():
        return np.load(cache)
    model = _get_model()
    emb = model.encode([c.text for c in chunks], batch_size=64, normalize_embeddings=True,
                       show_progress_bar=True, convert_to_numpy=True)
    np.save(cache, emb)
    return emb


def build(chunks: list[Chunk], cache_dir=None) -> None:
    global _corpus, _chunks
    _chunks = chunks
    _corpus = embed_corpus(chunks, cache_dir)


def retrieve(query: str, top_k: int = 50) -> list[Hit]:
    assert _corpus is not None, "先调用 build()"
    model = _get_model()
    q = model.encode([QUERY_INSTRUCTION + query], normalize_embeddings=True,
                     convert_to_numpy=True)[0]
    scores = _corpus @ q
    order = np.argsort(-scores)[:top_k]
    return [
        Hit(chunk_id=_chunks[i].chunk_id, ref_id=_chunks[i].ref_id,
            score=float(scores[i]), rank=rank + 1, snippet=_chunks[i].text[:200])
        for rank, i in enumerate(order) if scores[i] > 0
    ]
