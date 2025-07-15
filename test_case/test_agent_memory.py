#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试记忆工具与Agent的集成
"""

import os
import time
from typing import Dict, List, Any, Union

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from misc.memory_graph import create_memory_tools
from misc.utils import create_llm, load_environment


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


def create_test_agent():
    """创建测试用Agent"""
    # 加载环境变量
    load_environment()

    # 创建临时数据库文件路径
    db_path = "db_cache/test_db/test_agent_memory.kuzu"
    print(f"使用数据库: {db_path}")

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
1. save_memory - 保存重要信息到记忆库中，以便将来检索。用法: save_memory(content="要保存的内容", importance=1-10的重要性评分)
2. retrieve_memories - 检索与查询相关的记忆。用法: retrieve_memories(query="查询内容")

使用记忆的最佳实践:
- 保存用户提供的重要信息，如偏好、需求、背景等
- 保存对话中的关键结论和决策
- 在回答问题前，检索相关记忆以提供更连贯和个性化的回答
- 为重要记忆设置较高的重要性评分(1-10)

请根据需要使用这些工具，并在与用户交互时展示你的记忆能力。请使用中文回复。""",
    )

    return agent


def test_save_memory():
    """测试保存记忆功能"""
    print("\n===== 测试保存记忆功能 =====")

    agent = create_test_agent()

    # 初始对话
    messages = [
        HumanMessage(
            content="你好，我是张三，我是一名软件工程师，我喜欢Python和机器学习。"
        )
    ]

    # 运行Agent
    print("\n>> 初始对话")
    response = agent.invoke({"messages": messages})
    messages = response["messages"]
    print_messages(messages[-2:])  # 打印最后两条消息（包括可能的工具调用）

    # 继续对话，询问用户信息
    messages.append(HumanMessage(content="你还记得我是谁吗？"))

    print("\n>> 测试记忆检索")
    response = agent.invoke({"messages": messages})
    messages = response["messages"]
    print_messages(messages[-2:])

    print("\n保存记忆功能测试完成")


def test_retrieve_memory():
    """测试检索记忆功能"""
    print("\n===== 测试检索记忆功能 =====")

    agent = create_test_agent()

    # 初始对话，提供多个信息点
    messages = [
        HumanMessage(
            content="你好，我叫李四，我是一名数据科学家，我擅长数据分析和可视化，我最近在研究自然语言处理技术。"
        )
    ]

    # 运行Agent
    print("\n>> 初始对话")
    response = agent.invoke({"messages": messages})
    messages = response["messages"]
    print_messages(messages[-2:])

    # 继续对话，询问特定信息
    messages.append(HumanMessage(content="我擅长什么技术领域？"))

    print("\n>> 测试特定信息检索")
    response = agent.invoke({"messages": messages})
    messages = response["messages"]
    print_messages(messages[-2:])

    # 继续对话，询问另一个信息点
    messages.append(HumanMessage(content="我最近在研究什么？"))

    print("\n>> 测试另一个信息点检索")
    response = agent.invoke({"messages": messages})
    messages = response["messages"]
    print_messages(messages[-2:])

    print("\n检索记忆功能测试完成")


def test_memory_persistence():
    """测试记忆持久化功能"""
    print("\n===== 测试记忆持久化功能 =====")

    # 创建临时数据库文件路径
    db_path = "db_cache/test_db/test_persistence.kuzu"
    print(f"使用数据库: {db_path}")

    # 创建第一个Agent实例
    print("\n>> 创建第一个Agent实例")
    memory_save_tool, memory_retrieve_tool = create_memory_tools(db_path)
    tools = [memory_save_tool, memory_retrieve_tool]
    model = create_llm()
    agent1 = create_react_agent(
        model=model,
        tools=tools,
        prompt="你是一个有记忆能力的AI助手。使用save_memory保存信息，使用retrieve_memories检索信息。请使用中文回复。",
    )

    # 使用第一个Agent保存记忆
    print("\n>> 使用第一个Agent保存记忆")
    messages = [HumanMessage(content="请记住：王五喜欢旅游，特别是去海边。")]
    response = agent1.invoke({"messages": messages})
    messages = response["messages"]
    print_messages(messages[-2:])

    # 创建第二个Agent实例，使用相同的数据库
    print("\n>> 创建第二个Agent实例，使用相同的数据库")
    memory_save_tool, memory_retrieve_tool = create_memory_tools(db_path)
    tools = [memory_save_tool, memory_retrieve_tool]
    agent2 = create_react_agent(
        model=model,
        tools=tools,
        prompt="你是一个有记忆能力的AI助手。使用save_memory保存信息，使用retrieve_memories检索信息。请使用中文回复。",
    )

    # 使用第二个Agent检索记忆
    print("\n>> 使用第二个Agent检索记忆")
    messages = [HumanMessage(content="王五喜欢什么？")]
    response = agent2.invoke({"messages": messages})
    messages = response["messages"]
    print_messages(messages[-2:])

    print("\n记忆持久化功能测试完成")


def main():
    """主函数"""
    print("开始测试记忆工具与Agent的集成...")

    # 测试保存记忆功能
    test_save_memory()

    # 测试检索记忆功能
    test_retrieve_memory()

    # 测试记忆持久化功能
    test_memory_persistence()

    print("\n所有测试完成!")


if __name__ == "__main__":
    main()
