import os
import sqlite3
import sys
import time
from typing import Annotated, TypedDict

import pandas as pd
from pydantic import SecretStr
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

import config_data as config
from rag import ragservice


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


os.makedirs(config.chat_history_storage_path, exist_ok=True)
_checkpointer = SqliteSaver(
    sqlite3.connect(
        os.path.join(config.chat_history_storage_path, "agent.sqlite"),
        check_same_thread=False,
    )
)


_service = ragservice()


@tool
def rag_search(query: str, config: RunnableConfig) -> str:
    """检索公司福利、差旅、休假等制度知识库，回答政策类问题。"""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    return _service.chain.invoke(
        {"input": query},
        {"configurable": {"session_id": f"rag-{thread_id}"}},
    )


@tool
def calculate_annual_leave(work_years: float, company_years: float) -> str:
    """根据累计社会工龄(work_years)和本公司司龄(company_years)计算年假天数。"""
    if work_years < 1:
        base = 0
    elif work_years < 10:
        base = 5
    elif work_years < 20:
        base = 10
    else:
        base = 15

    bonus = 0
    if company_years >= 1:
        bonus = min(int(company_years - 1), 5)

    total = min(base + bonus, 20)
    return f"基础年假 {base} 天 + 司龄福利假 {bonus} 天 = 合计 {total} 天（年度上限 20 天）"


@tool
def calculate_hotel_cap(city_level: str, grade: str) -> str:
    """根据城市级别(city_level: A/B/C)和职级(grade: P1-P5/P6-P8/P9及以上)计算差旅住宿每晚上限。"""
    table = {
        ("A", "P1-P5"): "400",
        ("A", "P6-P8"): "600",
        ("A", "P9及以上"): "实报实销",
        ("B", "P1-P5"): "300",
        ("B", "P6-P8"): "450",
        ("B", "P9及以上"): "实报实销",
        ("C", "P1-P5"): "250",
        ("C", "P6-P8"): "350",
        ("C", "P9及以上"): "实报实销",
    }
    value = table.get((city_level.upper(), grade))
    if value is None:
        return f"没有 {city_level}/{grade} 对应的住宿标准，请确认城市级别和职级。"
    return f"{city_level} 类城市、{grade} 职级的差旅住宿上限为 {value} 元/晚。"


@tool
def human_handoff() -> str:
    """知识库检索不到、或问题超出公司制度范围时，转人工客服，不要编造答案。"""
    return "该问题超出我的能力范围，已为您转接人工客服（人力资源部福利热线 010-8888-6666）。"


TOOLS = [rag_search, calculate_annual_leave, calculate_hotel_cap, human_handoff]

SYSTEM_PROMPT = (
    "你是公司员工福利客服助手，可以调用工具来回答问题。\n"
    "规则：\n"
    "1. 用户问的是制度、政策、规则、报销条件等知识性问题，调用 rag_search 检索知识库。\n"
    "2. 用户给出了具体数字、要求直接计算年假天数或住宿上限时，调用对应计算工具 calculate_annual_leave 或 calculate_hotel_cap。\n"
    "3. 知识库检索不到、或问题超出公司制度范围，调用 human_handoff 转人工，不要编造答案。\n"
    "4. 拿到工具结果后判断是否已经能回答；不能就继续调用工具，能就直接回答。\n"
)


def _build_llm():
    return ChatOpenAI(
        model=config.chat_model,
        base_url=config.base_url,
        api_key=SecretStr(os.getenv("MIMO_API_KEY", "")),
        temperature=0,
    ).bind_tools(TOOLS)


def _agent_node(state: AgentState):
    llm = _build_llm()
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    return {"messages": [llm.invoke(messages)]}


_graph = StateGraph(AgentState)
_graph.add_node("agent", _agent_node)
_graph.add_node("tools", ToolNode(TOOLS))
_graph.add_edge(START, "agent")
_graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", "__end__": END})
_graph.add_edge("tools", "agent")
agent_graph = _graph.compile(checkpointer=_checkpointer)


def _thread_config(thread_id: str) -> dict:
    return {"recursion_limit": 10, "configurable": {"thread_id": thread_id}}


def ask(question: str, thread_id: str = "default") -> str:
    result = agent_graph.invoke(
        {"messages": [("user", question)]},
        config=_thread_config(thread_id),
    )
    return result["messages"][-1].content


def run_with_trace(question: str, thread_id: str = "default"):
    cfg = _thread_config(thread_id)
    snapshot = agent_graph.get_state(cfg)
    n_before = len(snapshot.values.get("messages", [])) if snapshot.values else 0
    result = agent_graph.invoke({"messages": [("user", question)]}, config=cfg)
    tool_calls = []
    for msg in result["messages"][n_before:]:
        for tc in getattr(msg, "tool_calls", []) or []:
            tool_calls.append(tc.get("name"))
    return result["messages"][-1].content, tool_calls


def batch_get_answers(input_path: str = "datasets/ragas_testset.csv", output_path: str | None = None) -> None:
    """读取测试集 CSV，逐条由 agent 作答，把结果写入 response 和 retrieved_contexts 列"""
    df = pd.read_csv(input_path)
    if "user_input" not in df.columns:
        raise ValueError(f"{input_path} 中缺少 user_input 列")
    if "response" not in df.columns:
        df["response"] = ""
    if "retrieved_contexts" not in df.columns:
        df["retrieved_contexts"] = "[]"
    if output_path is None:
        output_path = input_path.replace(".csv", "_with_answers.csv")

    print(f"开始批量作答，共 {len(df)} 条问题...")
    run_id = str(int(pd.Timestamp.now().timestamp()))
    for i, row in df.iterrows():
        query = str(row["user_input"])
        try:
            _service.retrieved_contexts = []
            answer = ask(query, thread_id=f"eval-{run_id}-{i}")
            df.at[i, "response"] = answer
            df.at[i, "retrieved_contexts"] = repr(_service.retrieved_contexts)
            print(f"[{i + 1}/{len(df)}] ✅ {query[:30]} -> {str(answer)[:60]}")
        except Exception as e:
            df.at[i, "response"] = f"[ERROR] {e}"
            df.at[i, "retrieved_contexts"] = "[]"
            print(f"[{i + 1}/{len(df)}] ❌ {query[:30]}　失败: {e}")

    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"✅ 完成，结果已保存到 {output_path}")


if __name__ == "__main__":
    if "--batch" in sys.argv:
        batch_get_answers()
        sys.exit()

    thread = f"demo-{int(time.time())}"
    for q in [
        "我工龄8年，在公司干了3年，年假有几天？",
        "那如果我在公司待满5年呢？",
        "那住院医保统筹后的个人自付能报销多少？",
    ]:
        answer, calls = run_with_trace(q, thread_id=thread)
        print(f"\nQ: {q}")
        print(f"工具调用: {calls if calls else '（未调用工具，直接回答）'}")
        print(f"A: {answer}")
