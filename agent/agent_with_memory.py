"""
带记忆功能的LangGraph ReactAgent示例
"""

import os
from typing import Dict, List, Any, Union

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from misc.memory_graph import create_memory_tools
from misc.utils import create_llm, load_environment


def create_memory_agent(
    api_key: str = None, db_path: str = "db_cache/test_db/memory_db.kuzu"
):
    """创建带有记忆功能的ReactAgent

    Args:
        api_key: OpenAI API密钥（可选）
        db_path: 数据库文件路径（不是目录）

    Returns:
        配置好的ReactAgent
    """
    # 设置OpenAI API密钥
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        load_environment()

    # 创建记忆工具
    memory_save_tool, memory_retrieve_tool = create_memory_tools(db_path)

    # 定义工具列表
    tools = [memory_save_tool, memory_retrieve_tool]

    # 创建语言模型
    model = create_llm()

    # 创建ReactAgent
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt="""你是一个有记忆能力的AI助手。你可以保存重要信息到记忆库中，也可以从记忆库中检索相关信息。

你有以下工具可用:
1. save_memory - 保存重要信息到记忆库中，以便将来检索
2. retrieve_memories - 检索与查询相关的记忆

使用记忆的最佳实践:
- 保存用户提供的重要信息，如偏好、需求、背景等
- 保存对话中的关键结论和决策
- 在回答问题前，检索相关记忆以提供更连贯和个性化的回答
- 为重要记忆设置较高的重要性评分(1-10)

请根据需要使用这些工具，并在与用户交互时展示你的记忆能力。请使用中文回复。""",
    )

    return agent


def print_messages(messages: List[Union[BaseMessage, Dict[str, Any]]]) -> None:
    """打印消息列表中的内容。"""
    for msg in messages:
        if isinstance(msg, HumanMessage) or (
            isinstance(msg, dict) and msg.get("role") == "user"
        ):
            print(f"用户: {msg.content if hasattr(msg, 'content') else msg['content']}")
        elif isinstance(msg, AIMessage) or (
            isinstance(msg, dict) and msg.get("role") == "assistant"
        ):
            print(f"助手: {msg.content if hasattr(msg, 'content') else msg['content']}")
        elif isinstance(msg, ToolMessage) or (
            isinstance(msg, dict) and msg.get("role") == "tool"
        ):
            print(f"工具: {msg.content if hasattr(msg, 'content') else msg['content']}")


def main():
    """主函数"""
    # 设置数据库文件路径
    db_path = "/mnt/data/gyzou/expr_workplace/self-agent-memory/db_cache/memory_db.kuzu"

    # 创建Agent
    agent = create_memory_agent(db_path=db_path)

    print("带记忆功能的AI助手已启动，输入'退出'结束对话")
    print(f"使用数据库: {db_path}")
    print("-" * 50)

    # 保存所有对话历史
    messages = []

    while True:
        # 获取用户输入
        user_input = input("用户: ")

        if user_input.lower() in ["退出", "exit", "quit"]:
            print("助手: 再见！")
            break

        # 添加用户消息
        messages.append(HumanMessage(content=user_input))

        # 运行Agent
        response = agent.invoke({"messages": messages})

        # 更新消息历史
        messages = response["messages"]

        # 打印助手回复（最后一条消息）
        print_messages(messages[-1:])
        print("-" * 50)


if __name__ == "__main__":
    main()
