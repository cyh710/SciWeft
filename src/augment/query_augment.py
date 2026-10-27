"""Query Augmentation（M6，UniRAG 三策略的论文启发简化重实现）。

- paraphrase : 等义改写，保持信息需求不变
- expansion  : 补充同义词/专业术语/相关概念
- abstraction: 提取核心实体与关系，去掉冗余修饰

红线：不宣称复现 UniRAG 联合训练框架；这是策略级可控对比。
生成查询全部走 LLM 磁盘缓存（同题同策略只花钱一次）。
检索融合：对「原查询 + 各变体」的命中列表做 RRF（跨变体 rank 融合）。
"""
from __future__ import annotations

import json
import re

from src.rag.llm import chat
from src.schemas import Hit

STRATEGY_PROMPTS = {
    "paraphrase": "Rewrite the question with different wording but the SAME information need. Do not add, remove or change any entity, number, unit or year. Output ONLY the rewritten question, nothing else.",
    "expansion": "Expand the question with synonyms, technical terms and closely related concepts that relevant documents might use. Keep every original entity, number, unit and year intact. Output ONLY one expanded query on a single line, nothing else.",
    "abstraction": "Reduce the question to its core entities and the exact information need (what quantity/fact is asked, about what subject, under what conditions). Drop politeness and irrelevant qualifiers. Output ONLY the abstracted query, nothing else.",
}

_VARIANT_CAP = 3  # 每策略最多生成几条（防模型输出多行）


def generate_variant(question: str, strategy: str) -> str:
    """生成一条策略变体查询（LLM 缓存兜底）。"""
    raw = chat([{"role": "user", "content": f"{STRATEGY_PROMPTS[strategy]}\n\nQuestion: {question}"}],
               max_tokens=4000, temperature=0)
    lines = [l.strip().lstrip("-•0123456789. ").strip() for l in raw.strip().splitlines() if l.strip()]
    return lines[0] if lines else question


def augment_queries(question: str, strategies: list[str]) -> list[dict]:
    """返回 [{text, strategy}]，恒含 original。"""
    out = [{"text": question, "strategy": "original"}]
    for s in strategies:
        try:
            text = generate_variant(question, s)
            if text and text != question:
                out.append({"text": text, "strategy": s})
        except Exception:  # noqa: BLE001  变体失败不致命：原查询仍在
            continue
    return out[:1 + _VARIANT_CAP]


# ---- M7：规则路由（计划 §4.8 的轻量自适应） ----

_NUM_RE = re.compile(r"\d|how many|how much|what percent|what percentage|what fraction", re.I)
_TF_RE = re.compile(r"\btrue or false\b|\btrue/false\b|^is it true\b", re.I)
_DEF_RE = re.compile(r"what is|what are|which .* (type|kind|term|concept|method|process)|refer to", re.I)
_LONG_WORDS = 24


def adaptive_strategy(question: str) -> str:
    """按问题特征选增强策略（开题方案 §4.8 的规则表）。

    优先级：判断题保原样 > 数字/单位题 expansion > 长题 abstraction > 定义题 paraphrase > 默认原样。
    """
    if _TF_RE.search(question):
        return "original"       # 判断题改写易漂移，保原查询
    if _NUM_RE.search(question):
        return "expansion"      # 精确数字/单位：保实体 + 召回导向扩展
    if len(question.split()) > _LONG_WORDS:
        return "abstraction"    # 复杂长问题：去冗余
    if _DEF_RE.search(question):
        return "paraphrase"     # 定义/概念题：等义改写
    return "original"


def augment_queries_adaptive(question: str) -> list[dict]:
    strategy = adaptive_strategy(question)
    if strategy == "original":
        return [{"text": question, "strategy": "original"}]
    return augment_queries(question, [strategy])


def fuse_variant_hits(hit_lists: list[list[Hit]], top_k: int = 50, rrf_k: int = 60) -> list[Hit]:
    """跨查询变体的 RRF 融合。"""
    fused: dict[str, dict] = {}
    for hits in hit_lists:
        for h in hits:
            e = fused.setdefault(h.chunk_id, {"ref_id": h.ref_id, "score": 0.0, "snippet": h.snippet})
            e["score"] += 1.0 / (rrf_k + h.rank)
    ranked = sorted(fused.items(), key=lambda kv: -kv[1]["score"])[:top_k]
    return [Hit(chunk_id=c, ref_id=v["ref_id"], score=v["score"], rank=i + 1, snippet=v["snippet"])
            for i, (c, v) in enumerate(ranked)]
