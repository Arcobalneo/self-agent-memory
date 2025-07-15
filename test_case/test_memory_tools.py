#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试记忆图谱工具的功能
"""

import os
import time
from datetime import datetime

from misc.memory_graph import (
    GraphMemoryStore,
    create_memory_tools,
    MemorySaveTool,
    MemoryRetrieveTool,
)


def test_graph_memory_store():
    """测试GraphMemoryStore类的基本功能"""
    print("\n===== 测试 GraphMemoryStore 基本功能 =====")

    # 创建临时数据库文件路径
    db_path = "db_cache/test_db/test_memory_store.kuzu"
    print(f"使用数据库: {db_path}")

    # 创建记忆存储
    memory_store = GraphMemoryStore(db_path=db_path)

    # 测试添加记忆
    print("\n>> 测试添加记忆")
    memory1_id = memory_store.add_memory("用户喜欢蓝色", importance=5)
    print(f"添加记忆1成功，ID: {memory1_id}")

    memory2_id = memory_store.add_memory("用户不喜欢红色", importance=3)
    print(f"添加记忆2成功，ID: {memory2_id}")

    memory3_id = memory_store.add_memory("用户最喜欢的颜色是蓝色", importance=7)
    print(f"添加记忆3成功，ID: {memory3_id}")

    # 测试通过ID获取记忆
    print("\n>> 测试通过ID获取记忆")
    memory = memory_store.get_memory_by_id(memory1_id)
    if memory:
        print(f"找到记忆: {memory['content']} (重要性: {memory['importance']})")
    else:
        print("未找到记忆")

    # 测试检索相关记忆
    print("\n>> 测试检索相关记忆")
    query = "用户喜欢什么颜色?"
    memories = memory_store.retrieve_relevant_memories(query)
    print(f"检索到 {len(memories)} 条相关记忆:")
    for i, memory in enumerate(memories, 1):
        print(f"{i}. {memory['content']} (重要性: {memory['importance']})")

    # 测试更新记忆重要性
    print("\n>> 测试更新记忆重要性")
    success = memory_store.update_memory_importance(memory1_id, 8)
    if success:
        print(f"更新记忆重要性成功")
        memory = memory_store.get_memory_by_id(memory1_id)
        print(f"更新后的记忆: {memory['content']} (重要性: {memory['importance']})")
    else:
        print("更新记忆重要性失败")

    print("\nGraphMemoryStore 测试完成")


def test_retrieve_memory():
    """测试检索记忆功能"""
    print("\n===== 测试记忆工具 =====")

    # 创建临时数据库文件路径
    db_path = "db_cache/test_db/test_memory_tools.kuzu"
    print(f"使用数据库: {db_path}")

    # 创建记忆工具
    save_tool, retrieve_tool = create_memory_tools(db_path)

    # 测试保存记忆工具
    print("\n>> 测试保存记忆工具")
    result1 = save_tool._run("用户是一位NLP研究者", importance=6)
    print(f"保存结果: {result1}")

    result2 = save_tool._run("用户正在研究多智能体系统", importance=7)
    print(f"保存结果: {result2}")

    result3 = save_tool._run("用户需要一个图谱存储的记忆系统", importance=9)
    print(f"保存结果: {result3}")

    # 测试检索记忆工具 - 使用更精确的查询
    print("\n>> 测试检索记忆工具 - 精确查询")
    query1 = "NLP"
    result = retrieve_tool._run(query1)
    print(f"查询 '{query1}' 的结果:\n{result}")

    # 测试检索记忆工具 - 使用部分匹配的查询
    print("\n>> 测试检索记忆工具 - 部分匹配查询")
    query2 = "研究"
    result = retrieve_tool._run(query2)
    print(f"查询 '{query2}' 的结果:\n{result}")

    # 测试检索记忆工具 - 使用空查询（应返回所有记忆）
    print("\n>> 测试检索记忆工具 - 空查询")
    query3 = ""
    result = retrieve_tool._run(query3)
    print(f"空查询的结果:\n{result}")

    print("\n记忆工具测试完成")


def test_memory_relationships():
    """测试记忆之间的关系"""
    print("\n===== 测试记忆关系 =====")

    # 创建临时数据库文件路径
    db_path = f"test_memory_relationships_{int(time.time())}.kuzu"
    print(f"使用数据库: {db_path}")

    # 创建记忆存储
    memory_store = GraphMemoryStore(db_path=db_path)

    # 添加一系列相关记忆
    print("\n>> 添加一系列相关记忆")
    memory1_id = memory_store.add_memory("用户喜欢机器学习", importance=5)
    print(f"添加记忆1成功，ID: {memory1_id}")

    # 短暂暂停，确保时间戳不同
    time.sleep(1)

    memory2_id = memory_store.add_memory("用户正在研究神经网络", importance=6)
    print(f"添加记忆2成功，ID: {memory2_id}")

    time.sleep(1)

    memory3_id = memory_store.add_memory("用户对深度学习很感兴趣", importance=7)
    print(f"添加记忆3成功，ID: {memory3_id}")

    # 测试相似度关系
    print("\n>> 测试相似度关系 (通过检索相关记忆间接测试)")
    query = "机器学习"
    memories = memory_store.retrieve_relevant_memories(query)
    print(f"检索到 {len(memories)} 条相关记忆:")
    for i, memory in enumerate(memories, 1):
        print(f"{i}. {memory['content']} (重要性: {memory['importance']})")

    print("\n记忆关系测试完成")


def main():
    """主函数"""
    print("开始测试记忆图谱功能...")

    # 测试GraphMemoryStore基本功能
    test_graph_memory_store()

    # 测试检索记忆功能
    test_retrieve_memory()

    # 测试记忆关系
    test_memory_relationships()

    print("\n所有测试完成!")


if __name__ == "__main__":
    main()
