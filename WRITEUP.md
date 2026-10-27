# SciWeft — WattBot 2026 writeup

> 提交卡（发布到公开仓库后回填 URL；当前为本地仓库，无远端）

```nix
code_url: https://github.com/cyh710/SciWeft/tree/v1.0-submission
writeup_url: https://github.com/cyh710/SciWeft/blob/v1.0-submission/WRITEUP.md
generation_model: deepseek-flash (DeepSeek v4.1-flash 档, API)
embedding_model: BAAI/bge-small-en-v1.5     # 仅用于 Hybrid 消融实验；最终提交的检索是纯 BM25
reranker_model: cross-encoder/ms-marco-MiniLM-L-6-v2
other_models: none
leaderboard_score: 0.867 (public, submission 57688803)
external_data: none                          # 仅官方 data 页语料，按 pinned URL 抓取
hardware: CPU only (laptop) + external APIs
cost_estimate: ~¥40 API spend (DeepSeek)
wall_clock: ~25 min for the full test set (317 q, incl. query expansion)
```

## Approach

**检索**：语料 122 篇 PDF（pinned URL 快照 + sha256 manifest）→ PyMuPDF 逐页文本 → 固定 400 词窗口 / 40 词 overlap 分块（2868 chunks，chunk 携带 ref_id 与页码）→ BM25（rank-bm25）取 Top-50 → Cross-Encoder（ms-marco-MiniLM-L-6-v2）重排取 Top-16 → 取前 8 块 + 相邻块全文作为证据窗口。

**查询增强**（UniRAG 三策略的论文启发简化重实现，未复现其联合训练）：每题先用生成模型做一次 Expansion（补充同义/术语表达，保持实体不变），原查询与扩展查询分别检索后做 RRF（k=60）融合，再进重排。消融显示 expansion +1.1–2.3pp，paraphrase/abstraction 持平。

**生成**：deepseek-flash（temperature=0，磁盘缓存全量命中，5 次重试 + 429 退避）。单一长 system prompt（输出契约 + 字段规则 + 2 个 few-shot 示例）约束 JSON 输出：`answer_value` 规范化（布尔答 1/0、区间交两端点、单位对齐题面给定值）、引用仅限证据窗口内 ref_id、拒答时按官方约定清空 value/refs/evidence 但**保留题面给定单位**。

## What worked / What didn't

**有效（245 题全量消融，本地 Score.py）**：
- 证据组装与排序器必须配套：Cross-Encoder 重排 + 直接取 Top 块 = +3.1pp；同样的重排配"文档去重"证据策略 = **−5.7pp**（排序被打乱）。
- BM25 是足够强的骨干：Dense（BGE-small）单独弱 28pp；RRF 混合把 R@10 从 0.918 推到 0.949 但牺牲 R@1，端到端反而降分。混合池 + 重排与 BM25 + 重排完全打平（0.733 vs 0.735）——重排器抹平了检索器差异。
- 查询 Expansion +2.3pp；Cross-Encoder 重排 +1.6pp。
- **最大的单项提升来自读数据而不是读文档**：gold 把布尔题编码为 `0/1`（50/245 题），我们按官方页面示例答 TRUE/FALSE 白丢 47 题。prompt 两句话修复后 0.735 → 0.865（+13pp）。同类：拒答行的 `answer_unit` 官方明确保留题面给定值。
- 评分器比文档宽容（区间括号两种都认、带内单值算对、单位自动归一）——所有"格式后处理"实验零增益。

**无效/负结果**：
- M7 规则路由（按题型选增强策略）0.727 < 一律 expansion 0.735：50 题 dev 上的噪声边缘优势在全量上不成立。
- Prompt 缓存只省 ~7%（证据文本每题独占，不可缓存）；省钱的正确姿势是控制证据体积，而证据体积换准确率是正收益（+4pp 命中 ≈ 每轮多 ¥9）。
- 引用手术（修剪/扩展引用集）无干净收益：引用错误以"检索窗口根本没含 gold 文档"（16/44）为主，生成端无解。

## Error analysis（剩余 33 错题的分布）

ref-only 30 / value+ref 21 / value-only 11 / NA 1。引用错误中完全错文档（窗口未覆盖 gold）16 题——指向检索上限；数值错误集中在派生计算与单位对齐。题型错误率：Quote 12% / Table 5% / Figure 18% / Math 36% / CrossPaper 32% / Reconcile 27%。

## Reproduce

```bash
python scripts/download_competition_data.py && python scripts/download_corpus.py
python scripts/build_index.py --config configs/baseline.yaml
python scripts/run_eval.py --config configs/aug_expansion_v3.yaml --split all   # 本地 0.865
python scripts/submit.py --config configs/aug_expansion_v3.yaml --out experiments/submissions/w5_naunit.csv
python Score.py data/raw/train_QA.csv experiments/<run>/predictions.csv
pytest -q   # 17 项单测
```

配置即实验：`configs/` 下 17 个 yaml 覆盖 M1–M7 全部方法开关；`experiments/retrieval_log.csv` 记录全部检索实验；manifest 记录语料 sha256 与修复（agl2025 走 Wayback 等）。

## 课程项目说明

方法矩阵 M1–M7 全部落地并消融，负结果（M7、Hybrid、后处理）与正结果同等呈现。全部实验数据见 `docs/EXPERIMENTS.md`。
