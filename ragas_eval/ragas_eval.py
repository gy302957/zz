import ast
import os
import asyncio
from pydantic import BaseModel
from ragas import experiment, Dataset
from ragas.metrics.collections import (
    ContextRecall,
    ContextPrecisionWithReference,
    AnswerRelevancy,
    Faithfulness,
)
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from openai import AsyncOpenAI

MM_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
MM_API_KEY = os.getenv("MIMO_API_KEY")

# API 实测约 50 并发是上限，行级并发留足余量，可按需上调
MAX_CONCURRENT_ROWS = 8

LIST_FIELDS = ("retrieved_contexts", "reference_contexts")


def parse_list_fields(row):
    """CSV 后端把列表字段原样存成字符串。

    不解析的话 ContextPrecision 会逐字符调 LLM（一条样本上万次请求），
    Faithfulness/ContextRecall 会拿到被 \n 打碎的上下文。
    """
    for field in LIST_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip().startswith("["):
            row[field] = ast.literal_eval(value)
    return row


async def main():
    client = AsyncOpenAI(
        api_key=MM_API_KEY,
        base_url=MM_BASE_URL,
        max_retries=6,
    )
    ragas_llm = llm_factory(model="mimo-v2.6-flash", client=client, max_tokens=32768)
    emb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models",
                          "bge-large-zh-v1.5")
    ragas_embedding = HuggingFaceEmbeddings(
        model=emb_path,
        device="cuda"
    )
    faithfulness = Faithfulness(llm=ragas_llm)
    answer_relevancy = AnswerRelevancy(llm=ragas_llm,
                                       embeddings=ragas_embedding,
                                       strictness=1)  # type: ignore
    context_recall = ContextRecall(llm=ragas_llm)
    context_precision_with_reference = ContextPrecisionWithReference(llm=ragas_llm)
    print("✅ RAGAS LLM + Embeddings configured")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    dataset = Dataset.load(name="ragas_testset_with_answers", backend="local/csv",
                       root_dir=project_root)
    for row in dataset:
        parse_list_fields(row)

    first_contexts = dataset[0]["retrieved_contexts"]
    print(f"✅ 已加载 {len(dataset)} 条评估样本, "
          f"首条 contexts={len(first_contexts)} 个 (type={type(first_contexts).__name__})")

    print(f'正在运行 RAGAS 评估（并发 {MAX_CONCURRENT_ROWS}），请稍候...')
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_ROWS)

    class ExperimentResult(BaseModel):
        user_input: str
        synthesizer_name: str
        faithfulness: float
        answer_relevancy: float
        context_recall: float
        context_precision_with_reference: float

    @experiment(ExperimentResult)
    async def run_evaluation(row):
        # semaphore 把同时在跑的样本数限制住，避免打满 API 的并发上限
        async with semaphore:
            faith_result = await faithfulness.ascore(
                user_input=row['user_input'],
                response=row['response'],
                retrieved_contexts=row['retrieved_contexts']
            )
            relevancy_result = await answer_relevancy.ascore(
                user_input=row['user_input'],
                response=row['response']
            )
            recall_result = await context_recall.ascore(
                user_input=row['user_input'],
                retrieved_contexts=row['retrieved_contexts'],
                reference=row['reference']
            )
            context_precision_result = await context_precision_with_reference.ascore(
                user_input=row['user_input'],
                retrieved_contexts=row['retrieved_contexts'],
                reference=row['reference']
            )
            return ExperimentResult(
                user_input=row['user_input'],
                synthesizer_name=row['synthesizer_name'],
                faithfulness=faith_result.value,
                answer_relevancy=relevancy_result.value,
                context_recall=recall_result.value,
                context_precision_with_reference=context_precision_result.value
            )

    exp_results = await run_evaluation.arun(dataset, name="ragas_eval_v1")
    print('✅ 评估完成')
    print(exp_results)


if __name__ == "__main__":
    asyncio.run(main())
