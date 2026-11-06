import asyncio
import os
import types

import pytest

from ragas_eval.ragas_eval import _safe_score, parse_list_fields

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TESTSET_NAME = "ragas_testset_with_answers-v1"
TESTSET_PATH = os.path.join(_PROJECT_ROOT, "datasets", TESTSET_NAME + ".csv")


def test_parse_list_fields_bracket_string():
    row = {"retrieved_contexts": "['a', 'b']", "reference_contexts": "['c']"}
    parse_list_fields(row)
    assert row["retrieved_contexts"] == ["a", "b"]
    assert row["reference_contexts"] == ["c"]


def test_parse_list_fields_plain_string_unchanged():
    row = {"retrieved_contexts": "纯文本内容"}
    parse_list_fields(row)
    assert row["retrieved_contexts"] == "纯文本内容"


def test_parse_list_fields_already_list_unchanged():
    row = {"retrieved_contexts": ["a", "b"], "reference_contexts": None}
    parse_list_fields(row)
    assert row["retrieved_contexts"] == ["a", "b"]
    assert row["reference_contexts"] is None


def test_parse_list_fields_missing_key_ok():
    assert parse_list_fields({}) == {}


def test_load_and_parse_real_dataset():
    """CSV 后端加载出的列表字段是字符串，解析后应变成真正的 list。"""
    if not os.path.exists(TESTSET_PATH):
        pytest.skip(f"{TESTSET_NAME}.csv 不存在")

    from ragas import Dataset

    dataset = Dataset.load(name=TESTSET_NAME, backend="local/csv",
                           root_dir=_PROJECT_ROOT)
    for row in dataset:
        parse_list_fields(row)

    assert len(dataset) > 0
    row = dataset[0]
    assert row["user_input"]
    assert isinstance(row["retrieved_contexts"], list)
    assert len(row["retrieved_contexts"]) >= 1
    assert all(isinstance(c, str) for c in row["retrieved_contexts"])
    assert isinstance(row["reference_contexts"], list)


class _StubMetric:
    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error

    async def ascore(self, **kwargs):
        if self.error is not None:
            raise self.error
        return types.SimpleNamespace(value=self.value)


def test_safe_score_returns_value():
    assert asyncio.run(_safe_score(_StubMetric(value=0.5), "stub")) == 0.5


def test_safe_score_failure_returns_none():
    assert asyncio.run(_safe_score(_StubMetric(error=ValueError("missing")), "stub")) is None
