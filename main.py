"""
LangGraph ReactAgent示例
"""

from typing import Dict, List, Any, Union

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from utils import create_llm, get_weather, calculate, load_environment


def main():
    """主函数，创建并运行LangGraph ReactAgent。"""
    print("初始化LangGraph ReactAgent...")

    # 加载环境变量
    if not load_environment():
        return

    # 定义工具
    tools = [get_weather, calculate]

    # 创建语言模型
    model = create_llm()

    # 创建ReactAgent
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt="你是一个有用的助手，可以回答问题、查询天气和进行计算。请使用中文回复。",
    )

    # 运行示例查询
    print("\n示例 1: 查询天气")
    response1 = agent.invoke(
        {"messages": [HumanMessage(content="北京今天天气怎么样？")]}
    )
    print_messages(response1["messages"])

    print("\n示例 2: 数学计算")
    response2 = agent.invoke({"messages": [HumanMessage(content="计算23乘以45")]})
    print_messages(response2["messages"])

    print("\n示例 3: 多轮对话")
    messages = [HumanMessage(content="你能做什么？")]
    response3 = agent.invoke({"messages": messages})
    print_messages(response3["messages"])

    # 添加后续问题到对话
    messages = response3["messages"] + [HumanMessage(content="那么上海的天气如何？")]
    response4 = agent.invoke({"messages": messages})
    print_messages(response4["messages"])


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


if __name__ == "__main__":
    main()
