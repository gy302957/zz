from vectorstore import chinese_tokenizer, docs_from_get_result


def test_chinese_tokenizer():
    tokens = chinese_tokenizer("北京住宿标准")
    assert isinstance(tokens, list)
    assert len(tokens) > 1


def test_chinese_tokenizer_ascii():
    tokens = chinese_tokenizer("123")
    assert tokens == ["123"]


def test_docs_from_get_result_normal():
    result = {"ids": ["1"], "documents": ["文档A", "文档B"], "metadatas": [{}]}
    assert docs_from_get_result(result) == ["文档A", "文档B"]


def test_docs_from_get_result_empty_dict():
    assert docs_from_get_result({}) == []


def test_docs_from_get_result_none():
    assert docs_from_get_result(None) == []


def test_docs_from_get_result_no_documents_key():
    assert docs_from_get_result({"ids": ["1"]}) == []