# SciWeft

科学文档检索增强问答（RAG）管线，为 [WattBot 2026](https://www.kaggle.com/competitions/WattBot2026) 竞赛构建。对给定问题从 122 篇科学文献中检索证据，输出规范化数值答案、文档级引用与逐字证据引用；语料无法回答时按官方约定拒答。

最终提交：公榜 0.867（submission 57688803）。全部实验数字可由本仓库配置一键重生成，全部实验数据见 [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)，竞赛披露文档见 [`WRITEUP.md`](WRITEUP.md)。

## 管线

```
问题 ──► Expansion 查询增强（原查询 + 扩展查询）
              │
              ▼
        BM25 Top-50 ──► RRF 融合（k=60）
              │
              ▼
   Cross-Encoder 重排 Top-16（cross-encoder/ms-marco-MiniLM-L-6-v2）
              │
              ▼
   证据窗口：Top-8 块 + 相邻块全文（携带 ref_id + 页码）
              │
              ▼
   LLM 生成（deepseek-flash，temperature=0，磁盘缓存，JSON 输出契约）
              │
              ▼
   answer_value · 引用 · 逐字证据 · explanation ──► 官方 Score.py
```

- **分块**：PyMuPDF 逐页文本，400 词窗口 / 40 词重叠，2868 chunks，全部携带 `ref_id` 与页码
- **检索**：BM25（rank-bm25）；Dense（BGE-small）与加权 RRF Hybrid 作为消融对照实现
- **查询增强**：Expansion（补充同义/术语表达，保持实体不变），原查询与扩展查询分别检索后 RRF 融合；Paraphrase / Abstraction / 规则路由作为消融对照实现
- **输出契约**：布尔答 `1/0`、区间交 `(low,high)`、单位对齐题面给定值、引用仅限证据窗口内文档、拒答行清空 value/refs/evidence 且保留题面给定 `answer_unit`

## 实验结果（245 训练题全量，官方 Score.py 评分）

| 配置 | value | ref F1 | OVERALL |
|---|---|---|---|
| BM25 + Cross-Encoder + Expansion（最终） | 0.690 | 0.841 | 0.865 |
| w/o Expansion | 0.661 | 0.833 | 0.712 |
| w/o Cross-Encoder | 0.661 | 0.865 | 0.719 |
| w/o BM25（全 Dense） | 0.624 | 0.807 | 0.680 |
| Hybrid(RRF 0.7/0.3) + Cross-Encoder + Expansion | 0.686 | 0.845 | 0.733 |
| Expansion 换规则路由 | 0.678 | 0.847 | 0.727 |

注：0.865 为本地全量分数，最终提交公榜 0.867；检索层指标（R@1/5/10、MRR）见 `experiments/retrieval_log.csv`。

## 复现

```bash
# 0) Python 3.13；pip install -r requirements.txt
# 1) 比赛数据（本仓库不含，官方规则 §7 禁止再分发；需在比赛页接受规则）：
printf 'KAGGLE_API_TOKEN=KGAT_你的令牌\nLLM_API_KEY=你的DeepSeek密钥\nLLM_BASE_URL=https://api.deepseek.com\nLLM_MODEL=deepseek-flash\nLLM_KIND=openai\n' > .env
pip install kaggle
python scripts/download_competition_data.py   # 6 个官方文件（含官方 Score.py）→ data/raw/
python scripts/download_corpus.py             # 122 篇语料 PDF，限速 5s/篇 + 断点续传
# 2) 索引与实验
python scripts/build_index.py --config configs/baseline.yaml
python scripts/run_eval.py --config configs/aug_expansion_v3.yaml --split all   # → 0.865
# 3) test 预测与官方评分
python scripts/submit.py --config configs/aug_expansion_v3.yaml --out experiments/submissions/final.csv
python Score.py data/raw/train_QA.csv experiments/<run>/predictions.csv
pytest -q
```

torch 系脚本带 `HF_HUB_OFFLINE=1`（模型已在本地缓存时避免联网检查挂起）。

## 仓库结构

```
src/ingest/    PDF 解析、固定分块、证据溯源
src/sparse/    TF-IDF / BM25 检索器
src/dense/     BGE 向量检索 + 加权 RRF Hybrid
src/augment/   三策略查询增强 + 规则路由
src/rag/       证据窗口组装、Cross-Encoder 重排、LLM 客户端（双协议+缓存）、JSON 答案解析
src/eval/      train/dev 分层划分、文档级 R@K/MRR（chunk 先按 ref_id 去重）
scripts/       数据下载、build_index、run_eval、submit、diagnose、bootstrap_ci、sweep_fusion、export_cases
configs/       17 个实验配置，覆盖方法矩阵全部开关
experiments/   retrieval_log.csv（检索实验指标）；预测/gold 产物不入库，从 config 重生成
docs/          EXPERIMENTS.md：实验记录、消融数据与结论
WRITEUP.md     竞赛披露文档
```

## 声明

- 比赛数据（题目、gold、`Score.py`、语料 PDF）归主办方，官方规则 §7 禁止再分发——本仓库不含任何比赛数据，按上方步骤从官方页面获取。
- 生成/嵌入/重排模型在 [`WRITEUP.md`](WRITEUP.md) 提交卡中披露；提交由代码流水线产出。
