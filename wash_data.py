"""
文档清洗管道 — 文件类型检测 / 加载 / 清洗一站式工具。

专为 RAG 流程设计，输出 ``langchain_core.documents.Document`` 列表，
可直接喂给 RecursiveCharacterTextSplitter 等切分器。

依赖
----
- langchain-core, langchain-community
- pypdf, python-docx, beautifulsoup4, markdown, charset-normalizer

用法
----
    from wash_data import DocumentWashPipeline

    pipeline = DocumentWashPipeline()
    docs = pipeline.run("path/to/file.pdf")
    # docs: list[Document] — 清洗完毕，可直接用于切分

    # 批量处理
    from pathlib import Path
    files = Path("data").rglob("*")
    docs = pipeline.run_batch(files)
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Callable, Iterator, List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. 文件类型枚举
# ---------------------------------------------------------------------------

class FileType:
    """支持的文档类型。"""
    PDF = "pdf"
    HTML = "html"
    DOCX = "docx"
    TXT = "txt"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"

    # 扩展名 → 类型映射
    _EXT_MAP: dict[str, str] = {
        ".pdf": PDF,
        ".htm": HTML,
        ".html": HTML,
        ".xhtml": HTML,
        ".docx": DOCX,
        ".txt": TXT,
        ".text": TXT,
        ".md": MARKDOWN,
        ".markdown": MARKDOWN,
        ".mdown": MARKDOWN,
        ".mdx": MARKDOWN,
    }

    @classmethod
    def from_extension(cls, path: str | Path) -> str:
        """根据文件扩展名推断类型。"""
        ext = Path(path).suffix.lower()
        return cls._EXT_MAP.get(ext, cls.UNKNOWN)

    @classmethod
    def from_magic_bytes(cls, path: str | Path) -> str:
        """用魔数（magic bytes）做二次校验，降低误判率。"""
        try:
            with open(path, "rb") as f:
                header = f.read(16)
        except Exception:
            return cls.UNKNOWN

        # PDF: %PDF
        if header.startswith(b"%PDF"):
            return cls.PDF
        # DOCX / ZIP-based formats: PK\x03\x04
        if header.startswith(b"PK\x03\x04"):
            # 进一步嗅探 — 查找 word/document.xml
            try:
                import zipfile
                with zipfile.ZipFile(path) as z:
                    names = z.namelist()
                if any(n.startswith("word/") for n in names):
                    return cls.DOCX
            except Exception:
                pass
            return cls.UNKNOWN
        # HTML: <html, <!DOCTYPE html
        stripped = header.strip().lower()
        if stripped.startswith(b"<html") or stripped.startswith(b"<!doctype html"):
            return cls.HTML
        if stripped.startswith(b"<!DOCTYPE html"):
            return cls.HTML
        # Markdown: 无魔数特征，靠扩展名即可

        return cls.UNKNOWN

    @classmethod
    def detect(cls, path: str | Path) -> str:
        """综合检测：先看扩展名，再用魔数修正。"""
        ext_type = cls.from_extension(path)
        if ext_type != cls.UNKNOWN:
            return ext_type
        magic_type = cls.from_magic_bytes(path)
        if magic_type != cls.UNKNOWN:
            return magic_type
        return cls.UNKNOWN


# ---------------------------------------------------------------------------
# 2. 文档加载器工厂
# ---------------------------------------------------------------------------

class DocumentLoaderFactory:
    """根据文件类型选择合适的 LangChain 加载器。"""

    @staticmethod
    def load(path: str | Path, file_type: str) -> List[Document]:
        """加载文档并返回 LangChain Document 列表。"""
        loader = DocumentLoaderFactory._get_loader(path, file_type)
        if loader is None:
            logger.warning("未找到 %s 类型的加载器: %s", file_type, path)
            return []
        return loader.load()

    @staticmethod
    def lazy_load(path: str | Path, file_type: str) -> Iterator[Document]:
        """惰性加载，适合大文件。"""
        loader = DocumentLoaderFactory._get_loader(path, file_type)
        if loader is None:
            logger.warning("未找到 %s 类型的加载器: %s", file_type, path)
            return iter([])
        return loader.lazy_load()

    @staticmethod
    def _get_loader(path: str | Path, file_type: str):
        """内部工厂方法。"""
        path_str = str(path)

        if file_type == FileType.PDF:
            from langchain_community.document_loaders import PyPDFLoader
            return PyPDFLoader(path_str, extract_images=False)

        elif file_type == FileType.HTML:
            return CustomBSHTMLLoader(path_str)

        elif file_type == FileType.DOCX:
            from langchain_community.document_loaders import Docx2txtLoader
            return Docx2txtLoader(path_str) 

        elif file_type == FileType.TXT:
            from langchain_community.document_loaders import TextLoader
            return TextLoader(path_str, encoding="utf-8", autodetect_encoding=True)

        elif file_type == FileType.MARKDOWN:
            return CustomMarkdownLoader(path_str)

        return None


# ---------------------------------------------------------------------------
# 2.1 自定义加载器（避免重依赖）
# ---------------------------------------------------------------------------

class CustomHTMLLoader:
    """
    轻量 HTML 加载器，使用 BeautifulSoup 提取文本，
    在 inline 元素前后自动插入空格，保证 ``<b>word</b>`` 不粘连。
    """

    def __init__(self, path: str | Path, encoding: str = "utf-8"):
        self.path = path
        self.encoding = encoding

    def load(self) -> List[Document]:
        from bs4 import BeautifulSoup

        with open(self.path, "r", encoding=self.encoding, errors="replace") as f:
            soup = BeautifulSoup(f.read(), "html.parser")

        # 移除脚本和样式
        for tag in soup(["script", "style", "noscript", "meta", "link"]):
            tag.decompose()

        # 带空格感知的文本提取
        parts: list[str] = []
        self._walk(soup, parts)
        text = "".join(parts)

        # 缩紧连续空白
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        metadata = {
            "source": str(self.path),
            "title": title,
        }
        return [Document(page_content=text.strip(), metadata=metadata)]

    def lazy_load(self) -> Iterator[Document]:
        yield from self.load()

    def _walk(self, elem, parts: list[str]) -> None:
        """递归遍历节点，插入空格隔离 inline 元素。"""
        from bs4.element import NavigableString, Tag
        INLINE_TAGS = {
            "a", "b", "i", "em", "strong", "span", "code", "u",
            "small", "sub", "sup", "mark", "q", "cite", "abbr",
            "tt", "kbd", "samp", "var", "del", "ins", "font",
        }
        BLOCK_TAGS = {
            "p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
            "li", "td", "th", "blockquote", "pre", "section",
            "article", "header", "footer", "nav", "aside",
        }

        if isinstance(elem, NavigableString):
            text = str(elem).strip()
            if text:
                parts.append(text)
                parts.append(" ")
            return

        if not isinstance(elem, Tag):
            return

        for child in elem.children:
            if isinstance(child, Tag):
                tag_name = child.name.lower() if child.name else ""

                if tag_name in ("br", "hr"):
                    parts.append("\n")
                elif tag_name == "img":
                    alt = child.get("alt", "")
                    if alt:
                        parts.append(f"[图: {alt}] ")
                elif tag_name == "table":
                    parts.append("\n")
                    self._walk_table(child, parts)
                    parts.append("\n")
                else:
                    if tag_name in BLOCK_TAGS:
                        parts.append("\n")
                        self._walk(child, parts)
                        parts.append("\n")
                    elif tag_name in INLINE_TAGS:
                        self._walk(child, parts)
                    else:
                        self._walk(child, parts)
            else:
                text = str(child).strip()
                if text:
                    parts.append(text)
                    parts.append(" ")

    def _walk_table(self, table, parts: list[str]) -> None:
        """简易表格提取，用 tab 分隔列。"""
        for row in table.find_all("tr"):
            cells = [cell.get_text(strip=True) for cell in row.find_all(["td", "th"])]
            parts.append("\t".join(cells) + "\n")


class CustomBSHTMLLoader(CustomHTMLLoader):
    """别名保持与 BSHTMLLoader 命名风格一致。"""
    pass


class CustomMarkdownLoader:
    """
    轻量 Markdown 加载器。

    使用 ``markdown`` 库转换为 HTML 后交由 BeautifulSoup 提取文本，
    避免对重依赖 ``unstructured`` 的依赖。
    """

    def __init__(self, path: str | Path, encoding: str = "utf-8"):
        self.path = path
        self.encoding = encoding

    def load(self) -> List[Document]:
        import markdown as md_lib
        from bs4 import BeautifulSoup

        with open(self.path, "r", encoding=self.encoding, errors="replace") as f:
            raw = f.read()

        # 保留 frontmatter（YAML 头）
        frontmatter: dict[str, str] = {}
        body = raw
        if raw.startswith("---"):
            end = raw.find("---", 3)
            if end != -1:
                frontmatter["raw_frontmatter"] = raw[3:end].strip()
                body = raw[end + 3 :]

        # markdown → HTML → text
        html = md_lib.markdown(body, extensions=["extra", "codehilite", "toc"])
        soup = BeautifulSoup(html, "html.parser")

        # 移除不需要的标签
        for tag in soup(["script", "style"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        metadata: dict = {
            "source": str(self.path),
            "file_type": "markdown",
        }
        if frontmatter:
            metadata["frontmatter"] = frontmatter["raw_frontmatter"]

        return [Document(page_content=text, metadata=metadata)]

    def lazy_load(self) -> Iterator[Document]:
        yield from self.load()


# ---------------------------------------------------------------------------
# 3. 清洗策略
# ---------------------------------------------------------------------------

class TextCleaner:
    """文本清洗核心 — 包含通用清洗和各类型专用清洗。"""

    # ------------------------------------------------------------------
    # 通用清洗（所有类型共享）
    # ------------------------------------------------------------------

    @staticmethod
    def common(text: str) -> str:
        """所有文档类型都必须经过的通用清洗。"""
        if not text:
            return ""

        # 1. Unicode 标准化（NFKC：兼容性分解 + 组合）
        text = unicodedata.normalize("NFKC", text)

        # 2. 移除零宽度字符
        zero_width = re.compile(r"[​‌‍⁠﻿]")
        text = zero_width.sub("", text)

        # 3. 移除控制字符（保留 \n \r \t）
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # 4. 统一换行符
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 5. 将制表符替换为空格
        text = text.replace("\t", "    ")

        # 6. 连续空白压缩（保留段落间距）
        text = re.sub(r" {3,}", "  ", text)          # 3+ 空格 → 2 空格
        text = re.sub(r"\n{4,}", "\n\n\n", text)     # 4+ 换行 → 3 换行

        # 7. 去除首尾空白
        text = text.strip()

        return text

    # ------------------------------------------------------------------
    # PDF 专用清洗
    # ------------------------------------------------------------------

    @staticmethod
    def pdf(text: str) -> str:
        """PDF 提取文本的专门清洗。"""
        if not text:
            return ""

        # 1. 修复断词连字符（英文 hyphenated line break）
        #    "comput-\ner" → "computer"  但保留真正的连词符
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

        # 2. 去除页眉页脚特征（行首页码 / 行末页码）
        #   常见的 "123\n" 或 "Page 1 of 10" 等模式
        text = re.sub(r"^\d{1,4}\n", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n\d{1,4}$", "", text, flags=re.MULTILINE)
        text = re.sub(r"(?i)^page\s+\d+\s+of\s+\d+\n?", "", text, flags=re.MULTILINE)
        text = re.sub(r"(?i)\npage\s+\d+\s+of\s+\d+$", "", text, flags=re.MULTILINE)

        # 3. 修复 PDF 提取中常见的粘连空格
        #    "thi s  i s" → "this is"
        text = re.sub(r"(?<=\w)\s(?=\w)", " ", text)

        # 4. 合并因换页导致的重复空行
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text

    # ------------------------------------------------------------------
    # HTML 专用清洗
    # ------------------------------------------------------------------

    @staticmethod
    def html(text: str) -> str:
        """HTML 内容清洗。"""
        if not text:
            return ""

        # 1. 去除残留的 HTML 实体（BSHTMLLoader 已处理大部分，但仍可能有残留）
        import html as html_mod
        text = html_mod.unescape(text)

        # 2. 去除导航/脚本残留文本片段
        #   常见特征：多级菜单项、版权信息等
        text = re.sub(r"(?i)copyright\s*©?\s*\d{4}.*?(?:\n|$)", "", text)
        text = re.sub(r"(?i)all rights reserved.*?(?:\n|$)", "", text)

        # 3. 缩短连续空行
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text

    # ------------------------------------------------------------------
    # DOCX 专用清洗
    # ------------------------------------------------------------------

    @staticmethod
    def docx(text: str) -> str:
        """DOCX 提取文本清洗。"""
        if not text:
            return ""

        # 1. 去除分页符和分节符
        text = text.replace("\f", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 2. 修复列表中常见的项目符号残留
        #   Docx2txtLoader 已经处理了大部分，做收尾
        text = re.sub(r"^[•·●○▪➢]\s*", "• ", text, flags=re.MULTILINE)

        return text

    # ------------------------------------------------------------------
    # TXT 专用清洗
    # ------------------------------------------------------------------

    @staticmethod
    def txt(text: str) -> str:
        """纯文本清洗（编码已由 TextLoader 处理）。"""
        if not text:
            return ""

        # 1. 去除 BOM（如果 TextLoader 未处理）
        text = text.lstrip("﻿")

        # 2. 统一段落间距（纯文本常见问题）
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text

    # ------------------------------------------------------------------
    # Markdown 专用清洗
    # ------------------------------------------------------------------

    @staticmethod
    def markdown(text: str) -> str:
        """Markdown 文本清洗。"""
        if not text:
            return ""

        # 1. 去除 HTML 标签残留（md 中嵌入的 inline HTML）
        text = re.sub(r"<[^>]+>", " ", text)

        # 2. 标准化标题格式（确保 # 后有空格）
        text = re.sub(r"^#+(?=[^\s#])", lambda m: m.group() + " ", text, flags=re.MULTILINE)

        # 3. 去除连续分隔线（---, ***, ___），保留一个
        text = re.sub(r"\n[-*_]{3,}\n", "\n\n", text)

        # 4. 表格行对齐（保持 markdown 表格整洁）
        #   简单的空格规范化
        text = re.sub(r"\| +", "| ", text)
        text = re.sub(r" +\|", " |", text)

        # 5. 连续空行收缩
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text


# ---------------------------------------------------------------------------
# 4. 清洗编排器
# ---------------------------------------------------------------------------

class DocumentWashPipeline:
    """
    文档清洗管道。

    :param normalize_whitespace: 是否执行通用空白规范化（默认 True）
    :param min_text_length:      清洗后文本长度小于此值视为空文档
    :param remove_empty_pages:   是否移除空页面/空段落（默认 True）
    :param extra_metadata:       向所有 Document 注入的额外元数据
    """

    def __init__(
        self,
        normalize_whitespace: bool = True,
        min_text_length: int = 10,
        remove_empty_pages: bool = True,
        extra_metadata: Optional[dict] = None,
    ):
        self.normalize_whitespace = normalize_whitespace
        self.min_text_length = min_text_length
        self.remove_empty_pages = remove_empty_pages
        self.extra_metadata = extra_metadata or {}

        # 各类型对应的清洗函数表
        self._cleaners: dict[str, Callable] = {
            FileType.PDF:      TextCleaner.pdf,
            FileType.HTML:     TextCleaner.html,
            FileType.DOCX:     TextCleaner.docx,
            FileType.TXT:      TextCleaner.txt,
            FileType.MARKDOWN: TextCleaner.markdown,
        }

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run(
        self,
        path: str | Path,
        file_type: Optional[str] = None,
        lazy: bool = False,
    ) -> List[Document]:
        """
        对单个文件执行完整清洗管道。

        :param path:      文件路径
        :param file_type: 手动指定文件类型（None 则自动检测）
        :param lazy:      是否使用惰性加载（大文件推荐）
        :return:          清洗完毕的 Document 列表
        """
        path = Path(path)
        if not path.exists():
            logger.error("文件不存在: %s", path)
            return []
        if not path.is_file():
            logger.error("路径不是文件: %s", path)
            return []

        # 1. 类型检测
        ftype = file_type or FileType.detect(path)
        if ftype == FileType.UNKNOWN:
            logger.warning("无法识别文件类型，回退为 TXT: %s", path)
            ftype = FileType.TXT

        logger.info("处理文件 [%s]: %s", ftype, path)

        # 2. 加载
        try:
            if lazy:
                docs = list(DocumentLoaderFactory.lazy_load(path, ftype))
            else:
                docs = DocumentLoaderFactory.load(path, ftype)
        except Exception as e:
            logger.error("加载文件失败 [%s]: %s — %s", ftype, path, e)
            return []

        if not docs:
            return []

        # 3. 清洗
        cleaned = self._clean_documents(docs, ftype)

        logger.info("清洗完成 [%s]: %d → %d 个文档", ftype, len(docs), len(cleaned))
        return cleaned

    def run_batch(
        self,
        paths: List[str | Path],
        file_type: Optional[str] = None,
        lazy: bool = False,
        show_progress: bool = True,
    ) -> List[Document]:
        """
        批量处理多个文件。

        :param paths:         文件路径列表
        :param file_type:     强制所有文件的类型（None 则逐文件自动检测）
        :param lazy:          惰性加载
        :param show_progress: 是否打印进度日志
        :return:              所有文件清洗后的 Document（按文件顺序平铺）
        """
        all_docs: List[Document] = []
        for i, p in enumerate(paths):
            if show_progress:
                logger.info("进度 [%d/%d]: %s", i + 1, len(paths), p)
            docs = self.run(p, file_type=file_type, lazy=lazy)
            all_docs.extend(docs)
        return all_docs

    # ------------------------------------------------------------------
    # 单字符串清洗（直接传入文本）
    # ------------------------------------------------------------------

    def clean_text(self, text: str, file_type: str = FileType.TXT) -> str:
        """
        直接清洗文本字符串（无需加载文件）。

        适用于自定义加载场景 —— 自行读取文件后传入文本，
        利用本管道的清洗能力进行清洗。

        :param text:      原始文本
        :param file_type: 文本类型（影响专用清洗函数的选择）
        :return:          清洗后的文本
        """
        text = TextCleaner.common(text)
        cleaner = self._cleaners.get(file_type)
        if cleaner:
            text = cleaner(text)
        return text

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _clean_documents(self, docs: List[Document], file_type: str) -> List[Document]:
        """对 Document 列表执行清洗并注入元数据。"""
        cleaned: List[Document] = []
        cleaner = self._cleaners.get(file_type)

        for doc in docs:
            # 读取原始内容
            text = doc.page_content or ""

            # 专用清洗
            if cleaner:
                text = cleaner(text)

            # 通用清洗
            if self.normalize_whitespace:
                text = TextCleaner.common(text)

            # 空文档过滤
            if self.remove_empty_pages and len(text.strip()) < self.min_text_length:
                continue

            # 构建清洗后的 Document
            metadata = {
                **doc.metadata,
                **self.extra_metadata,
                "file_type": file_type,
            }
            cleaned.append(Document(page_content=text, metadata=metadata))

        return cleaned


# ---------------------------------------------------------------------------
# 5. 辅助函数 — 便捷入口
# ---------------------------------------------------------------------------

def wash_document(
    path: str | Path,
    *,
    file_type: Optional[str] = None,
    **kwargs,
) -> List[Document]:
    """
    便捷函数：一行代码清洗单个文件。

    ::

        docs = wash_document("report.pdf")
        print(docs[0].page_content[:200])

    更多参数见 :class:`DocumentWashPipeline`。
    """
    pipeline = DocumentWashPipeline(**kwargs)
    return pipeline.run(path, file_type=file_type)


def wash_text(
    text: str,
    file_type: str = FileType.TXT,
    **kwargs,
) -> str:
    """
    便捷函数：一行代码清洗文本字符串。

    ::

        clean = wash_text(raw_html, file_type=FileType.HTML)
    """
    pipeline = DocumentWashPipeline(**kwargs)
    return pipeline.clean_text(text, file_type=file_type)


# ---------------------------------------------------------------------------
# 6. CLI 入口（可直接命令行运行）
# ---------------------------------------------------------------------------

def cli():
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="文档清洗工具 — 加载、清洗、输出文本",
    )
    parser.add_argument("paths", nargs="+", help="要处理的文件路径")
    parser.add_argument(
        "--type", "-t",
        default=None,
        help="强制指定文件类型 (pdf/html/docx/txt/markdown)",
    )
    parser.add_argument(
        "--lazy", "-l",
        action="store_true",
        help="惰性加载模式（大文件推荐）",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="输出文件路径（不指定则打印到终端）",
    )
    parser.add_argument(
        "--format",
        default="text",
        choices=["text", "json"],
        help="输出格式 (json 含 metadata)",
    )

    args = parser.parse_args()

    pipeline = DocumentWashPipeline()
    all_docs: List[Document] = []

    for path in args.paths:
        ftype = args.type if args.type else None
        docs = pipeline.run(path, file_type=ftype, lazy=args.lazy)
        all_docs.extend(docs)
        print(f"  → {path}: {len(docs)} 个文档", file=sys.stderr)

    if not all_docs:
        print("未产生任何有效文档。", file=sys.stderr)
        sys.exit(1)

    output_text = ""
    if args.format == "json":
        import json
        data = [
            {"content": d.page_content, "metadata": d.metadata}
            for d in all_docs
        ]
        output_text = json.dumps(data, ensure_ascii=False, indent=2)
    else:
        output_text = "\n\n---\n\n".join(
            d.page_content for d in all_docs
        )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_text)
        print(f"结果已写入: {args.output}", file=sys.stderr)
    else:
        print(output_text)


if __name__ == "__main__":
    cli()
