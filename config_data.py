md5_path="./md5.txt"
file_md5_path="./file_md5.txt"


#chroma数据库
collection_name="rag"
persist_directory="./chroma.db"


#spliter
separators=["\n\n","\n",",",".","!","?","，","？","！","。"," ",""]
chunk_size=500
chunk_overlap=200

#retriever
top_k=3

embedding_model="ragas_eval/models/bge-large-zh-v1.5"
chat_model="mimo-v2.6-flash"
base_url="https://token-plan-cn.xiaomimimo.com/v1"

chat_history_storage_path="./chat_history"
session_config={
        "configurable":{
            "session_id":"user-001"
        }
    }