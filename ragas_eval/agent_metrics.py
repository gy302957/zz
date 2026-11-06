"""基于作答 CSV 计算 agent 决策指标：转人工准确率、误转率、检索调用率、决策准确率"""

import ast

import pandas as pd

HANDOFF_PHRASE = "已为您转接"
HANDOFF_PREFIX_CHARS = 100
NEGATIVE_SYNTH = "negative_sample"


def is_handoff(response) -> bool:
    """回答开头 100 字内宣布转人工才算转接，正文末尾附热线不算"""
    text = str(response)
    idx = text.find(HANDOFF_PHRASE)
    return idx != -1 and idx < HANDOFF_PREFIX_CHARS


def has_retrieval(retrieved_contexts) -> bool:
    """retrieved_contexts 非空即调用过 rag_search"""
    text = str(retrieved_contexts).strip()
    if text in ("", "[]", "nan"):
        return False
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return True
    return not isinstance(value, list) or len(value) > 0


def decision_correct(row) -> bool:
    """负样本应转人工，其余题应作答不转人工"""
    handoff = is_handoff(row["response"])
    if row["synthesizer_name"] == NEGATIVE_SYNTH:
        return handoff
    return not handoff


def evaluate(df: pd.DataFrame) -> dict:
    """计算单份作答 CSV 的决策指标"""
    neg = df[df["synthesizer_name"] == NEGATIVE_SYNTH]
    pol = df[df["synthesizer_name"] != NEGATIVE_SYNTH]
    neg_handoff = sum(is_handoff(r) for r in neg["response"])
    pol_handoff = sum(is_handoff(r) for r in pol["response"])
    pol_retr = sum(has_retrieval(r) for r in pol["retrieved_contexts"])
    neg_retr = sum(has_retrieval(r) for r in neg["retrieved_contexts"])
    correct = sum(decision_correct(row) for _, row in df.iterrows())

    def rate(a, b):
        return a / b if b else float("nan")

    return {
        "题数": len(df),
        "负样本": len(neg),
        "转人工准确率": rate(neg_handoff, len(neg)),
        "误转率": rate(pol_handoff, len(pol)),
        "rag_search调用率": rate(pol_retr, len(pol)),
        "负样本先检索率": rate(neg_retr, len(neg)),
        "决策准确率": rate(correct, len(df)),
    }


def flagged_rows(df: pd.DataFrame) -> list:
    """政策题却转人工的行号与问题摘要"""
    out = []
    for i, row in df.iterrows():
        if row["synthesizer_name"] != NEGATIVE_SYNTH and is_handoff(row["response"]):
            out.append((i, str(row["user_input"])[:40]))
    return out


def main(paths: list) -> None:
    frames = [(path, pd.read_csv(path)) for path in paths]
    keys = ["转人工准确率", "误转率", "rag_search调用率", "负样本先检索率", "决策准确率"]
    print(f"{'文件':44s}{'题数':>4s}{'负样本':>6s}" + "".join(f"{k:>14s}" for k in keys))
    for path, df in frames:
        m = evaluate(df)
        name = path.split("/")[-1]
        print(f"{name:44s}{m['题数']:4d}{m['负样本']:6d}"
              + "".join(f"{m[k]:14.3f}" for k in keys))
    for path, df in frames:
        flags = flagged_rows(df)
        if flags:
            print(f"\n{path.split('/')[-1]} 政策题转人工（需人工确认）:")
            for i, q in flags:
                print(f"  #{i} {q}")


if __name__ == "__main__":
    import sys

    main(sys.argv[1:] or ["datasets/ragas_testset_with_answers-v2.csv"])
