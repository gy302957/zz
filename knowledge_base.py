import hashlib
import os
import config_data as config
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import datetime
import torch
from typing import List, Dict, Any


def get_string_md5(input_string: str, encoding="utf-8"):
    """计算字符串的MD5值"""
    return hashlib.md5(input_string.encode(encoding)).hexdigest()


def save_md5(md5_str: str, path=config.md5_path):
    """保存MD5值到数据库或文件中"""
    with open(path, "a", encoding="utf-8") as f:
        f.write(md5_str + "\n")


def check_md5(md5_str: str, path=config.md5_path):
    """检查MD5值是否已经存在, false表示不存在，true表示存在"""
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8"):
            pass
        return False
    else:
        for line in open(path, "r", encoding="utf-8"):
            if line.strip() == md5_str:
                return True
    return False


class knowledgebaseservice(object):
    def __init__(self):
        os.makedirs(config.persist_directory, exist_ok=True)
        self.Chroma = Chroma(
            collection_name=config.collection_name,
            embedding_function=HuggingFaceEmbeddings(
                model_name=config.embedding_model,
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            ),
            persist_directory=config.persist_directory,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len,
        )

    def upload_by_str(self, data: str, file_name: str):
        """检查上传的字符串是否重复，不重复则保存进向量数据库"""
        knowledge_chunk = self.splitter.split_text(data)
        for i in range(len(knowledge_chunk)):
            md5_str = get_string_md5(knowledge_chunk[i])
            if not check_md5(md5_str):
                self.Chroma.add_texts(
                    [knowledge_chunk[i]],
                    metadatas=[
                        {
                            "chunk_id":md5_str,
                            "source": file_name,
                            "create_time": datetime.datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                            "operator": "周杰棍",
                        }
                    ],
                )
                save_md5(md5_str)
            else:
                pass
        return "上传成功！"

    def get_corpus(self) -> List[Dict[str, Any]]:
        """获取知识库中存储的所有文档原文及元数据。

        Returns:
            list[dict]: 每个元素包含 ``text`` 和 ``metadata`` 两个键。
                适用于外部检索器（如 CustomHybridRetriever）构建 BM25 索引。
        """
        all_docs = self.Chroma.get()
        texts = all_docs.get("documents") or []
        metadatas = all_docs.get("metadatas") or [{}] * len(texts)
        return [
            {"text": texts[i], "metadata": metadatas[i] if i < len(metadatas) else {}}
            for i in range(len(texts))
        ]

    def get_corpus_texts(self) -> List[str]:
        """仅获取所有文档原文（不含元数据），供 BM25 构建使用。"""
        all_docs = self.Chroma.get()
        return all_docs.get("documents") or []


if __name__ == "__main__":
    
    from wash_data import DocumentWashPipeline, FileType
    
    # 1. 清洗MD文件
    pipeline = DocumentWashPipeline()
    cleaned_docs = pipeline.run("test_case/test_case_1.md", file_type=FileType.MARKDOWN)
    
    # 2. 拼接文本并上传
    full_text = "\n".join([doc.page_content for doc in cleaned_docs])
    
    # 3. 上传到知识库
    service = knowledgebaseservice()
    result = service.upload_by_str(full_text, "test_case_1.md")
    
    print("✅ 上传成功！")
