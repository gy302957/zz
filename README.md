# 制度知识库 Agentic RAG + RAGAS 评估流水线

基于 LangGraph 的 agentic RAG 问答系统：agent 自主决定何时检索、调用计算工具还是转人工；
配套 RAGAS 四指标评估与 agent 决策指标评估的双轨评估流水线。

> 语料（`test_case/`）与题库均为合成测试数据，不含任何真实企业信息。

## 架构

**Agent（langgraph，4 工具自主调度）**

| 工具 | 作用 |
|---|---|
| `rag_search` | 混合检索：BM25 + Chroma 向量 → RRF 融合 10 候选 → bge-reranker-large 精排 → top3 原文片段（带 [1][2][3] 引用标记） |
| `calculate_annual_leave` | 按工龄/司龄计算年假天数 |
| `calculate_hotel_cap` | 按城市级别/职级计算差旅住宿上限 |
| `human_handoff` | 知识库覆盖不到时转人工，不编造答案 |

Agent 拿到 rag_search 返回的原文片段后自行综合作答，并在关键信息处标注 [1][2][3] 引用编号以标明来源。

**评估（ragas 0.4.3 + mimo 判官）**

- **RAGAS 四指标**：Faithfulness / AnswerRelevancy / ContextRecall / ContextPrecisionWithReference
  - 按指标隔离失败：单个指标评分失败只留空该格，不丢整行（空 retrieved_contexts 的行仍计 answer_relevancy）
- **Agent 决策指标**（`ragas_eval/agent_metrics.py`，纯规则零 API）：
  - 转人工准确率 / 误转率 / rag_search 调用率 / 决策准确率
  - 负样本应转人工、政策题应作答

## 目录结构

```
├── agent.py                # langgraph agent + 批量作答（含 tool_calls 捕获）
├── rag.py                  # RAG 链（混合检索 + retrieved_contexts 捕获）
├── vectorstore.py          # BM25 + Chroma 双路检索、RRF 融合、精排接入
├── reranker.py             # bge-reranker-large CrossEncoder 单例（懒加载）
├── knowledge_base.py       # 语料清洗入库（chunk_id + md5 去重）
├── wash_data.py            # 文档清洗管道（pdf/html/docx/md/txt）
├── config_data.py          # 配置（chunk/top_k/精排/模型端点）
├── ragas_eval/
│   ├── ragas_questions.py  # 题库生成（单跳 synthesizer）
│   ├── ragas_eval.py       # RAGAS 四指标评估
│   └── agent_metrics.py    # agent 决策指标
├── datasets/               # 题库与作答 CSV
├── experiments/            # 评估结果 CSV
├── test_case/              # 合成语料
└── tests/                  # pytest（23 个）
```

## 环境准备

Python 3.12：

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export MIMO_API_KEY=你的key   # 评估与 agent 的 LLM 调用
```

下载模型（embedding + 精排，走 hf-mirror）：

```bash
HF_ENDPOINT=https://hf-mirror.com python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-large-zh-v1.5', local_dir='ragas_eval/models/bge-large-zh-v1.5')"
HF_ENDPOINT=https://hf-mirror.com python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-reranker-large', local_dir='ragas_eval/models/bge-reranker-large')"
```

## 快速开始

```bash
# 1. 语料入库（可重复执行，md5 去重）
python knowledge_base.py

# 2. agent 批量作答（指定输出文件，避免覆盖已有基线）
python -c "from agent import batch_get_answers; batch_get_answers(output_path='datasets/ragas_testset_with_answers-v3.csv')"

# 3. RAGAS 评估（先改 ragas_eval.py 中两处名字：
#    Dataset.load 的 name → 新作答文件名，arun 的 name → 新实验名）
python ragas_eval/ragas_eval.py

# 4. agent 决策指标（支持多份作答文件对比）
python ragas_eval/agent_metrics.py datasets/ragas_testset_with_answers.csv datasets/ragas_testset_with_answers-v2.csv

# 5. 测试
python -m pytest
```

## 评估基线

题库：50 题（24 单跳 / 19 多跳 / 7 负样本），判官 mimo-v2.6-flash。

| 版本 | 变更 | 计分行 | Faithfulness | AnswerRelevancy | ContextRecall | ContextPrecision | 决策准确率 |
|---|---|---|---|---|---|---|---|
| v1 | agent + 混合检索（无精排） | 46 | 0.806 | 0.655 | 0.887 | 0.753 | 1.000 |
| v2 | + bge-reranker 精排、按指标隔离 | 50 | 0.685 | 0.617 | 0.884 | 0.823 | 0.980 |
| v3 | + 答案带引用、rag_search 返回文档 | 50 | 0.865 | 0.644 | 0.941 | 0.856 | 1.000 |

口径说明：

- 判官单次评估**噪声底约 0.03**（同一批作答连评两次的指标差值），小于该幅度的跨版差异不作数
- v1 为行级跳过口径（空 rc 整行丢弃，46 行）；v2 起按指标隔离（50 行全计分，单格可空）
- **v3 引用效果显著**：Faithfulness +0.18（噪声底 6 倍）、ContextRecall +0.06；多跳 Faithfulness 从 0.70 → 0.93
- 精排收益看 **ContextPrecision +0.07**（两次评估一致）；v3 在此基础上继续提升至 0.856
- 决策准确率：v2 唯一误转（跨年报销题）在 v3 修复，三版决策均满分或接近满分
