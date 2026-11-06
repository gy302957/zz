import pandas as pd

from ragas_eval.agent_metrics import (
    decision_correct,
    evaluate,
    flagged_rows,
    has_retrieval,
    is_handoff,
)


def test_is_handoff_at_start():
    assert is_handoff("已为您转接人工客服：010-8888-6666")


def test_is_handoff_late_phrase_not_counted():
    assert not is_handoff("答案正文。" * 40 + "已为您转接人工客服")


def test_is_handoff_no_phrase():
    assert not is_handoff("正常回答，可拨打 010-8888-6666 咨询。")


def test_has_retrieval():
    assert has_retrieval("['a', 'b']")
    assert not has_retrieval("[]")
    assert not has_retrieval("nan")
    assert not has_retrieval("")


def test_decision_correct():
    neg_ok = {"synthesizer_name": "negative_sample", "response": "已为您转接人工客服"}
    neg_bad = {"synthesizer_name": "negative_sample", "response": "硬答了一个"}
    pol_ok = {"synthesizer_name": "single_hop_specific_query_synthesizer", "response": "答案"}
    pol_bad = {"synthesizer_name": "single_hop_specific_query_synthesizer", "response": "已为您转接人工客服"}
    assert decision_correct(neg_ok)
    assert decision_correct(pol_ok)
    assert not decision_correct(neg_bad)
    assert not decision_correct(pol_bad)


def test_evaluate():
    df = pd.DataFrame([
        {"synthesizer_name": "negative_sample", "response": "已为您转接人工客服",
         "retrieved_contexts": "[]", "user_input": "q1"},
        {"synthesizer_name": "negative_sample", "response": "硬答了一个",
         "retrieved_contexts": "['x']", "user_input": "q2"},
        {"synthesizer_name": "single_hop_specific_query_synthesizer", "response": "答案",
         "retrieved_contexts": "['x']", "user_input": "q3"},
        {"synthesizer_name": "multi_hop_specific_query_synthesizer", "response": "已为您转接人工客服",
         "retrieved_contexts": "['x']", "user_input": "q4"},
    ])
    m = evaluate(df)
    assert m["转人工准确率"] == 0.5
    assert m["误转率"] == 0.5
    assert m["rag_search调用率"] == 1.0
    assert m["负样本先检索率"] == 0.5
    assert m["决策准确率"] == 0.5
    assert flagged_rows(df) == [(3, "q4")]
