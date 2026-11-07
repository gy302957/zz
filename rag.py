from langchain_huggingface import HuggingFaceEmbeddings
import config_data as config
from vectorstore import Vectorstore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables.history import RunnableWithMessageHistory
from file_history_store import get_file_chat_history
from pydantic import SecretStr
import torch


class ragservice():
    def __init__(self):
        self.vectorstore = Vectorstore(
            embedding=HuggingFaceEmbeddings(
                model_name=config.embedding_model,
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        )
        

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "请根据检索到的结果，清晰简洁地回答问题，参考资料：{context}"),
            ("system", "我还提供对话的历史消息记录供你参考，如下："),
            (MessagesPlaceholder(variable_name="history")),
            ("user", "请回答用户提问：{input}"),
        ])
        self.chat_model = ChatOpenAI(
            model=config.chat_model,
            base_url=config.base_url,
            api_key=SecretStr(os.getenv("MIMO_API_KEY", "")),
        )

        self.retrieved_contexts: list[str] = []
        self.retriever = self.vectorstore.get_hybrid_retriever()
        self.chain = self.get_chain()

    def retrieve_and_capture(self, query: str) -> list[Document]:
        """执行混合检索并将新结果追加到 retrieved_contexts（评估用的检索捕获）"""
        docs = self.retriever(query)
        for content in (d.page_content for d in docs):
            if content not in self.retrieved_contexts:
                self.retrieved_contexts.append(content)
        return docs

    def get_documents(self, query: str) -> str:
        """检索知识库，返回带 [1][2][3] 引用标记的原文片段。

        agent 拿到带编号的片段后自行综合作答，并在关键信息处标注 [1][2] 等来源编号。
        这样做让回答可溯源，也让 faithfulness 评估能核验每条陈述的依据。
        """
        docs = self.retrieve_and_capture(query)
        if not docs:
            return "没有检索到相关资料"
        parts = []
        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get("source", "")
            tag = f"（来源：{src}）" if src else ""
            parts.append(f"[{i}]{tag} {doc.page_content}")
        return "\n\n".join(parts)

    def get_chain(self):
        def format_document(docs: list[Document]):
            if not docs:
                return "没有检索到相关资料"
            else:
                print(f"--- 检索完成，找到 {len(docs)} 个文档 ---")
                format_str = ""
                format_str += "\n\n".join(doc.page_content for doc in docs)
                return format_str

        def format_for_retriever(value: dict):
            output = value["input"]
            return output

        def format_for_template(value):
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]
            return new_value

        chain = (
            {
                "input": RunnablePassthrough(),
                "context": RunnableLambda(format_for_retriever)
                | RunnableLambda(self.retrieve_and_capture)
                | format_document,
            }
            | RunnableLambda(format_for_template)
            | self.prompt_template
            | self.chat_model
            | StrOutputParser()
        )

        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_file_chat_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        return conversation_chain

