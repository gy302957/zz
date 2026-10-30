import logging
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pydantic import SecretStr
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
from ragas.run_config import RunConfig
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas.llms import llm_factory
from openai import OpenAI
logging.basicConfig(level=logging.DEBUG)

MM_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def main():
    raw_key = os.getenv("MIMO_API_KEY")
    api_key = SecretStr(raw_key) if raw_key else None
    client=OpenAI(
        api_key=raw_key,
        base_url=MM_BASE_URL
    )
    llm = ChatOpenAI(
        model="mimo-v2.6-flash",
        api_key=api_key,
        base_url=MM_BASE_URL,
        temperature=0.0,
    )

    generator = TestsetGenerator.from_langchain(
        llm=llm,
        embedding_model=HuggingFaceEmbeddings(
            model_name=os.path.join(os.path.dirname(__file__), "models", "bge-large-zh-v1.5"),
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        ),
    )

    single_hop_synth = SingleHopSpecificQuerySynthesizer(
        llm=llm_factory(model="mimo-v2.6-flash",client=client))

    doc_path = os.path.join(_PROJECT_ROOT, "test_case", "test_case_1.md")
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"找不到语料文件：{doc_path}")
    with open(doc_path, "r", encoding="utf-8") as f:
        content = f.read()
    docs = [Document(page_content=content, metadata={"source": doc_path})]
    print(f"✅ 成功加载了 {len(docs)} 个文档")

    run_config = RunConfig(max_workers=4, timeout=180)
    testset = generator.generate_with_langchain_docs(
        documents=docs,
        testset_size=30,
        query_distribution=[(single_hop_synth, 1.0)],  # type: ignore
        with_debugging_logs=True,
        run_config=run_config,
    )

    output_path = os.path.join(_PROJECT_ROOT, "datasets", "ragas_testset.csv")
    testset.to_pandas().to_csv(output_path, index=False)
    print(f"✅ 已保存到 {output_path}")


if __name__ == "__main__":
    main()