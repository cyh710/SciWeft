"""跨模块数据结构契约（单人版）。

对应《项目开发计划》§3.3 最低接口契约。字段只增不改名；改动前先想清楚
第 4 周消融要重跑所有历史配置——签名不稳定等于坑自己。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    """parse_documents 的输出单元。来源 metadata.csv 一行 + 解析结果。"""

    ref_id: str                 # metadata.csv 的 id，最终引用就写它，保持原样
    url: str                    # pinned URL（含 arXiv 版本号）
    title: str = ""
    year: str = ""
    page_count: int = 0
    parse_ok: bool = True       # 失败的另记日志，不中断全库
    parse_error: str = ""


@dataclass
class Chunk:
    """chunk_documents 的输出单元。必须能回溯到文档与页码。"""

    chunk_id: str               # 唯一且持久：{ref_id}:{seq}
    ref_id: str
    text: str
    page_start: int = 0
    page_end: int = 0


@dataclass
class Hit:
    """retrieve 的输出单元，按分数降序。"""

    chunk_id: str
    ref_id: str
    score: float
    rank: int
    snippet: str = ""           # 供 QA 与错误分析用


@dataclass
class AugmentedQuery:
    """augment_query 的输出单元。绝不改变原问题的实体/数值。"""

    text: str
    strategy: str               # original | paraphrase | expansion | abstraction


@dataclass
class Answer:
    """answer_question 的输出单元，直接映射提交 CSV 的列。"""

    question_id: str
    value: str                  # 标准化数值/类别/True-False；不可回答 = is_blank
    unit: str = "is_blank"      # test_Q.csv 已给定单位，与官方保持一致
    answer_text: str = ""       # 自然语言答案（不计分）
    citation_ref_ids: list[str] = field(default_factory=list)
    evidence: str = ""          # supporting_materials：原文引用
    explanation: str = "see evidence"   # 每行必填，空值过不了提交校验
    is_blank: bool = False      # True 时 value/ref_id/ref_url/evidence 全部 is_blank
