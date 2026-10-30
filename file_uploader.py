import streamlit as st
import tempfile
import os
import hashlib  # 用于计算 MD5，替代你原来的 get_string_md5

# 假设你已经将 wash_data.py 放在同级目录或已安装
from wash_data import DocumentWashPipeline, FileType
import config_data as config
from knowledge_base import knowledgebaseservice,save_md5,check_md5

st.title("知识库文件上传服务")

# 上传组件保持不变
uploaded_file = st.file_uploader("选择要上传的文件", type=["txt", "pdf", "docx", "html", "md"], accept_multiple_files=False)

# 初始化状态 (保持不变)
if "service" not in st.session_state:
    st.session_state["service"] = knowledgebaseservice()
if "counter" not in st.session_state:
    st.session_state["counter"] = 0

# 处理文件上传
if uploaded_file is not None:
    try:
        # 1. ================== 核心修改：文件清洗流程 ==================
        
        # 1.1. 创建临时文件 (将内存中的文件“落地”到磁盘)
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # 1.2. 自动推断文件类型 (Mapping Streamlit MIME Type to wash_data FileType)
        def map_file_type(mime_type: str, file_name: str) -> str:
            if mime_type == "application/pdf":
                return FileType.PDF
            elif mime_type == "text/plain":
                return FileType.TXT
            elif mime_type == "text/html":
                return FileType.HTML
            elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                return FileType.DOCX
            elif mime_type in ["text/markdown", "text/x-markdown"]:
                return FileType.MARKDOWN
            else:
                # 如果 MIME type 不准，尝试用文件后缀
                ext = file_name.lower().split(".")[-1]
                if ext == "pdf": return FileType.PDF
                if ext in ["txt", "text"]: return FileType.TXT
                if ext in ["doc", "docx"]: return FileType.DOCX
                if ext in ["htm", "html"]: return FileType.HTML
                if ext in ["md", "markdown"]: return FileType.MARKDOWN
                return FileType.TXT # 默认回退

        detected_type = map_file_type(uploaded_file.type, uploaded_file.name)

        # 1.3. 调用清洗管道
        pipeline = DocumentWashPipeline()
        
        # 关键：传入临时文件路径和检测到的类型
        # 注意：run() 返回的是 List[Document]，我们取第一个（通常一个文件对应一个 Document 或几个片段）
        cleaned_docs = pipeline.run(tmp_file_path, file_type=detected_type)
        
        # 将清洗后的文本拼接起来 (如果切分了，就用换行符连接)
        if cleaned_docs:
            full_text = "\n".join([doc.page_content for doc in cleaned_docs])
        else:
            raise ValueError("清洗后未获得任何文本内容，请检查文件是否为空或格式是否受支持。")

        # 1.4. 清理临时文件
        os.unlink(tmp_file_path)

        # 2. ================== 原有逻辑：MD5 检查与上传 ==================
        # 注意：这里不能再用 decode('utf-8')，因为 full_text 已经是字符串了
        file_md5 = hashlib.md5(full_text.encode('utf-8')).hexdigest()
        
        if check_md5(file_md5, path=config.file_md5_path):
            st.warning("文件内容已存在，无法重复上传")
        else:
            file_name = uploaded_file.name
            # file_type 和 file_size 可以直接从 uploaded_file 获取，或者用清洗后的数据
            file_size = len(full_text) / 1024  # 这里是清洗后的大小

            save_md5(file_md5, path=config.file_md5_path)
            
            st.subheader("文件已上传:", file_name)
            st.write(f"格式: {uploaded_file.type} | 大小: {file_size:.2f}KB")

            # 显示清洗后的前 500 个字符预览
            with st.expander("查看清洗后的内容预览"):
                st.write(full_text[:500] + "...")

            with st.spinner("正在上传文件..."):
                # 假设 upload_by_str 接收 (文本内容, 文件名)
                result = st.session_state["service"].upload_by_str(full_text, file_name)
                st.write(result)
                st.session_state["counter"] += 1
                st.write(f"已上传 {st.session_state['counter']} 个文件")

    except Exception as e:
        # 异常处理
        st.error(f"处理文件时出错: {e}")
        # 如果出错，也要尝试清理临时文件 (虽然 with 语句通常会处理，但为了安全)
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)