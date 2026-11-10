"""Dense 检索器（M3）：BGE-small + NumPy 余弦相似度（小语料无需 FAISS）。

- chunk 向量一次性编码并缓存到 data/cache/（键 = 模型名 + chunk_id 列表哈希，
  换语料或换分块参数自动失效，不会误命中同规模的旧向量）；
- bge v1.5 对短查询建议加检索指令前缀；
- 向量全部 L2 归一化，点积即余弦。
"""
from __future__ import annotations

import hashlib

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


def cache_key(chunks: list[Chunk]) -> str:
    """向量缓存键：模型名 + chunk_id 序列哈希。块数相同但内容不同也不会碰撞。"""
    payload = "|".join(c.chunk_id for c in chunks) + "|" + MODEL_NAME
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def embed_corpus(chunks: list[Chunk], cache_dir) -> np.ndarray:
    """编码全部 chunk 并按 (模型, chunk_id 序列) 缓存。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"emb_{cache_key(chunks)}.npy"
    if cache.exists():
        return np.load(cache)
    model = _get_model()
    emb = model.encode([c.text for c in chunks], batch_size=64, normalize_embeddings=True,
                       show_progress_bar=True, convert_to_numpy=True)
    np.save(cache, emb)
    return emb


def build(chunks: list[Chunk], cache_dir=None) -> None:
    global _corpus, _chunks
    if cache_dir is None:
        raise RuntimeError("dense.build 需要显式传入 cache_dir（脚本里是 data/cache），否则向量无处缓存")
    _chunks = chunks
    _corpus = embed_corpus(chunks, cache_dir)


def retrieve(query: str, top_k: int = 50) -> list[Hit]:
    if _corpus is None:
        raise RuntimeError("dense 检索器未初始化，先调用 build()（python -O 下 assert 会剥离，故显式抛错）")
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
