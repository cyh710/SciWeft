"""证据选择：从检索命中里组装给 LLM 的证据窗口（W1 收尾，按诊断结果设计）。

诊断（experiments/baseline-bm25_dev_1026_0018）发现三个缺陷：
1. top-N chunk 常被同一篇文档占满 -> 跨文档题召不全（gold=2篇 时召回率仅 40%）；
2. 文档对但块不对 -> BM25 按词重叠排序，含答案的块可能排在更后；
3. 因此模型大量"证据不足"过度拒答（17/49）。

对策（三种策略，可配置对比）：
- top_chunks : 直接取 top-N chunk（基线）
- doc_diverse: 取前 D 篇不同文档，每篇取其最佳 1 块（文档级去重）
- doc_grouped: 取前 D 篇文档，每篇取其最佳 C 块（文档去重 + 深度覆盖）
三者都可叠加 keep_neighbor：附带所选块前后相邻块，缓解跨块断裂。
"""
from __future__ import annotations

from src.schemas import Chunk, Hit


def _chunk_seq(chunk_id: str) -> int:
    try:
        return int(chunk_id.rsplit(":", 1)[1])
    except (IndexError, ValueError):
        return 0


def _window(text: str, from_end: bool, n_words: int) -> str:
    words = text.split()
    return " ".join(words[-n_words:] if from_end else words[:n_words])


def select_evidence(hits: list[Hit], chunks_by_id: dict[str, Chunk], strategy: str = "doc_grouped",
                    n_evidence: int = 6, n_docs: int = 4, per_doc: int = 2,
                    keep_neighbors: bool = True, neighbor_words: int = 0,
                    max_total_words: int | None = None) -> tuple[list[Hit], dict[str, str]]:
    """返回 (送给 LLM 的 hit 列表, chunk_id -> 文本 映射)。

    keep_neighbors=True 时附带相邻块内容：
    - neighbor_words=0 -> 拼接相邻块的完整内容（准确率最高，证据体积 3 倍）；
    - neighbor_words>0 -> 只取相邻块首尾各 N 词的窗口（体积约为整块的 1/2）。
    实测（dev 代理指标"答案值出现在证据中"）：无邻接 79.6% / 窗口200词 79.6% /
    整块 83.7%。本项目的瓶颈是准确率不是 token 成本，故默认取整块。

    max_total_words（默认 None = 不设预算，行为与历史一致）：证据总词数上限，
    按 rank 顺序累加；下一块装不下时截断到剩余额度（加 "[...truncated...]" 标记）
    并停止，其后命中整体丢弃。返回的 hit 列表与文本映射同步裁剪——被预算丢掉
    的块不在窗口内，引用契约（引用仅限证据窗口）由 generate 层据 hit 列表保证。
    非 None 但 <=0 视为配置笔误，直接 ValueError（静默产出空证据会让全题拒答）。
    """
    if max_total_words is not None and max_total_words <= 0:
        raise ValueError(f"max_total_words 必须为正整数或 None，得到 {max_total_words!r}")
    if not hits:
        return [], {}

    if strategy == "top_chunks":
        chosen = hits[:n_evidence]
    else:
        best_per_doc: dict[str, Hit] = {}
        for h in hits:
            if h.ref_id not in best_per_doc:
                best_per_doc[h.ref_id] = h
        order = sorted(best_per_doc.values(), key=lambda h: h.rank)
        picked: list[Hit] = []
        if strategy == "doc_diverse":
            for h in order[:n_docs]:
                picked.append(h)
        else:  # doc_grouped
            by_doc: dict[str, list[Hit]] = {}
            for h in hits:
                by_doc.setdefault(h.ref_id, []).append(h)
            for h in order[:n_docs]:
                picked.extend(by_doc[h.ref_id][:per_doc])
        picked = picked[:n_evidence]
        chosen = picked

    texts: dict[str, str] = {}
    kept: list[Hit] = []
    used = 0
    for h in chosen:
        ch = chunks_by_id.get(h.chunk_id)
        if ch is None:
            full = h.snippet
        else:
            parts = [ch.text]
            if keep_neighbors:
                seq = _chunk_seq(ch.chunk_id)
                prev = chunks_by_id.get(f"{ch.ref_id}:{seq - 1}")
                nxt = chunks_by_id.get(f"{ch.ref_id}:{seq + 1}")
                for nb, from_end, label in ((prev, True, "[...preceding context...] "),
                                            (nxt, False, "[...following context...] ")):
                    if nb is None:
                        continue
                    if neighbor_words > 0:
                        parts.insert(0, label + _window(nb.text, from_end, neighbor_words)) if from_end \
                            else parts.append(label + _window(nb.text, from_end, neighbor_words))
                    else:
                        parts.insert(0, label + nb.text) if from_end else parts.append(label + nb.text)
            full = "\n".join(parts)
        if max_total_words is not None:
            words = full.split()
            remaining = max_total_words - used
            if remaining <= 0:
                break
            if len(words) > remaining:
                full = " ".join(words[:remaining]) + " [...truncated...]"
        texts[h.chunk_id] = full
        kept.append(h)
        if max_total_words is not None:
            used += len(full.split())
    return kept, texts
