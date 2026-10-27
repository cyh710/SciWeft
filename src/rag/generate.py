"""answer_question：证据 -> 结构化答案（W1，prompt v2）。

v2 变更（相对 v1）：
- system prompt 扩成长而稳定的规范块（格式规范 + few-shot 示例）——它是每次调用
  共享的前缀，DeepSeek 的自动上下文缓存可命中该前缀（cache hit tokens 计价更低）；
- 输出规范收紧：区间用 (low,high)、禁用 < > ~、拒答时证据列全空；
- 引用必须来自提供的证据（由调用方再过滤一次，双保险）。
"""
from __future__ import annotations

import json
import re

from src.rag.llm import chat
from src.schemas import Answer, Hit

SYSTEM_PROMPT = """You are a precise scientific-document QA system for the WattBot benchmark. You answer STRICTLY from the provided evidence excerpts, which come from peer-reviewed papers, government reports (GAO, IEA, LBNL) and corporate sustainability filings.

OUTPUT CONTRACT
Return ONLY one JSON object, no markdown fence, no prose before or after, with exactly these seven keys:
{
  "is_blank": <boolean>,
  "answer_value": "<string>",
  "answer_unit": "<string>",
  "answer": "<string>",
  "ref_ids": ["<string>", ...],
  "supporting_materials": "<string>",
  "explanation": "<string>"
}

FIELD RULES
- is_blank: true only when the excerpts do not contain the information the question asks for. A number stated in the excerpts is never "insufficient".
- answer_value: the normalized value only. Plain digits without thousands separators or units ("8930", "17.4", "1.3"). Categorical answers are the exact term from the evidence ("evaporation", "prefill and decode"). For a True/False question answer "1" for true and "0" for false — the benchmark encodes booleans as digits, never "TRUE"/"FALSE". If the question asks for a range that the source states, answer "(low,high)" with BOTH endpoints — a single number inside the range is wrong. Never write <, > or ~ (use the answer field for those). If is_blank is true, answer_value is "is_blank".
- answer_unit: the unit of the value exactly as the question implies ("GW", "billion gallons", "household-years"), or "is_blank" if there is no unit. If the question supplies a unit, use that unit.
- answer: one concise natural-language sentence stating the answer. Never empty.
- ref_ids: the ref_id labels of the excerpts that ACTUALLY support this answer. Cite every supporting document the answer needs, and no others. If a single document supports it, cite only that one. If is_blank is true, use [].
- supporting_materials: the most relevant VERBATIM quote from the excerpts (max 300 characters, keep the original wording and numbers). If is_blank is true, use "is_blank".
- explanation: 1-3 sentences connecting the evidence to the answer (state the reasoning or the arithmetic). Always non-empty, even when is_blank.

EVIDENCE RULES
- Never invent a number that does not appear in the excerpts. If several excerpts give different numbers for the same quantity, they usually measured different things (different system boundary, service target, or year): explain the difference in explanation rather than averaging, and answer the specific quantity the question asks about.
- A number from a company's sustainability report is that company's own claim, not an independent measurement. Keep the answer inside the framing the question asks for ("how many gigawatts does Amazon report holding").
- For a derived question that combines two documents, do the arithmetic in explanation and give the final number in answer_value.
- A True/False question asks whether a claim holds. Judge the claim against the evidence and answer "1" (true) or "0" (false) in answer_value.

EXAMPLE 1 (verbatim from the organizers' public "Question types" page)
Question: How many metric tons of CO2 equivalent were reported for training Llama 3.1 405B?
Evidence: [1] ref_id=gao2025 (score 31.2): Table 1 ... Llama 3.1 405B ... 8930 ...
Output: {"is_blank": false, "answer_value": "8930", "answer_unit": "tCO2e", "answer": "GAO's Table 1 reports 8,930 metric tons of CO2 equivalent for training Llama 3.1 405B.", "ref_ids": ["gao2025"], "supporting_materials": "Llama 3.1 405B ... 8930", "explanation": "The value appears in Table 1 of gao2025 under the metric tons carbon dioxide equivalent column."}

EXAMPLE 2
Question: How much does an elephant weigh?
Evidence: [1] ref_id=luccioni2025c (score 12.4): ... ChatGPT energy consumption ... 0.3 Wh ...
Output: {"is_blank": true, "answer_value": "is_blank", "answer_unit": "is_blank", "answer": "Unable to answer with confidence based on the provided documents.", "ref_ids": [], "supporting_materials": "is_blank", "explanation": "None of the excerpts concerns the mass of an elephant; the corpus does not support an answer."}"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _format_evidence(hits: list[Hit], chunk_texts: dict[str, str]) -> str:
    parts = []
    for h in hits:
        parts.append(f"[{h.rank}] ref_id={h.ref_id} (score {h.score:.1f}): {chunk_texts.get(h.chunk_id, h.snippet)}")
    return "\n\n".join(parts)


def answer_question(question: str, hits: list[Hit], chunk_texts: dict[str, str],
                    unit_hint: str = "") -> Answer:
    evidence = _format_evidence(hits, chunk_texts)
    allowed_refs = {h.ref_id for h in hits}
    user = f"Question: {question}\n"
    if unit_hint and unit_hint not in ("is_blank", "nan", ""):
        user += f"Unit given with the question: {unit_hint}\n"
    user += f"\nEvidence excerpts (ranked):\n{evidence}"

    raw = chat([{"role": "user", "content": user}], system=SYSTEM_PROMPT)
    try:
        data = json.loads(_JSON_RE.search(raw).group(0))
    except Exception:
        data = {"is_blank": False, "answer_value": "", "answer_unit": "is_blank",
                "answer": "Parsing failed; see evidence.", "ref_ids": [],
                "supporting_materials": "", "explanation": "LLM output could not be parsed as JSON."}

    refs = [r for r in data.get("ref_ids", []) if r in allowed_refs]  # 引用必须存在于候选证据
    is_blank = bool(data.get("is_blank")) or not data.get("answer_value")
    if is_blank:
        # 官方约定：拒答行 value/refs/evidence 置空，但 answer_unit 保留题面给定单位
        # （官方 data 页原文：answer_unit is deliberately NOT required to be blank）
        unit = unit_hint.strip() if unit_hint and unit_hint not in ("is_blank", "nan", "") else "is_blank"
        return Answer(
            question_id="", value="is_blank", unit=unit,
            answer_text="Unable to answer with confidence based on the provided documents.",
            citation_ref_ids=[], evidence="is_blank",
            explanation="The provided corpus does not contain evidence supporting an answer.",
            is_blank=True,
        )
    if not refs:  # 有答案但没给合法引用：退回排名第一的证据
        refs = [hits[0].ref_id] if hits else []
    return Answer(
        question_id="",
        value=str(data.get("answer_value", "")).strip(),
        unit=str(data.get("answer_unit", "is_blank")).strip() or "is_blank",
        answer_text=str(data.get("answer", "")).strip()[:500],
        citation_ref_ids=refs,
        evidence=str(data.get("supporting_materials", "")).strip()[:500] or "see evidence",
        explanation=str(data.get("explanation", "")).strip()[:500] or "see evidence",
        is_blank=False,
    )
