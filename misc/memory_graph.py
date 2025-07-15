import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, ClassVar
from pathlib import Path

import kuzu
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool
from langchain_kuzu.graphs.kuzu_graph import KuzuGraph
from pydantic import BaseModel, Field


class MemoryNode(BaseModel):
    """记忆节点模型"""

    id: str
    content: str
    timestamp: str
    type: str = "Memory"
    importance: int = 1


class GraphMemoryStore:
    """基于KuZu图数据库的记忆存储"""

    def __init__(self, db_path: str = "db_cache/test_db/memory_db.kuzu"):
        """初始化图数据库连接

        Args:
            db_path: 数据库文件路径（不是目录）
        """
        # 确保路径是文件路径而不是目录
        db_file_path = db_path

        # 如果路径不包含文件扩展名，添加.kuzu扩展名
        if not db_file_path.endswith(".kuzu"):
            db_file_path += ".kuzu"

        # 确保父目录存在
        parent_dir = os.path.dirname(db_file_path)
        if parent_dir and not os.path.exists(parent_dir):
            try:
                os.makedirs(parent_dir, exist_ok=True)
                print(f"创建目录: {parent_dir}")
            except Exception as e:
                print(f"创建目录时出错: {e}")

        print(f"初始化KuZu数据库: {db_file_path}")

        # 创建KuZu数据库连接
        self.db = kuzu.Database(db_file_path)
        self.conn = kuzu.Connection(self.db)

        # 初始化图结构
        self._init_graph_schema()

        # 创建KuZu图
        self.graph = KuzuGraph(self.db, allow_dangerous_requests=True)

    def _init_graph_schema(self):
        """初始化图数据库模式"""
        # 创建Memory节点
        try:
            # KuZu使用的是不同于SQL的语法
            # 创建节点类型
            print("创建Memory节点表...")
            create_node_query = "CREATE NODE TABLE IF NOT EXISTS Memory(memory_id STRING, content STRING, timestamp STRING, importance INT, PRIMARY KEY(memory_id))"
            self.conn.execute(create_node_query)

            # 创建关系类型
            print("创建RELATED_TO关系表...")
            create_rel1_query = "CREATE REL TABLE IF NOT EXISTS RELATED_TO(FROM Memory TO Memory, similarity FLOAT)"
            self.conn.execute(create_rel1_query)

            print("创建FOLLOWS关系表...")
            create_rel2_query = "CREATE REL TABLE IF NOT EXISTS FOLLOWS(FROM Memory TO Memory, time_diff FLOAT)"
            self.conn.execute(create_rel2_query)

            # 验证表是否创建成功
            print("验证表结构...")
            try:
                # 在KuZu中查询节点表
                tables_query = "MATCH (n:Memory) RETURN COUNT(n) AS count"
                tables_result = self.conn.execute(tables_query)
                count = 0
                for row in tables_result:
                    count = row[0]
                print(f"Memory节点表存在，当前记录数: {count}")
            except Exception as e:
                print(f"验证Memory节点表时出错: {e}")

            print("图数据库模式初始化完成")
        except Exception as e:
            print(f"初始化图数据库模式时出错: {e}")
            raise e  # 重新抛出异常，因为模式初始化是关键步骤

    def add_memory(self, content: str, importance: int = 1) -> str:
        """添加新记忆到图数据库

        Args:
            content: 记忆内容
            importance: 重要性评分 (1-10)

        Returns:
            记忆ID
        """
        timestamp = datetime.now().isoformat()
        memory_id = f"mem_{timestamp.replace(':', '_').replace('.', '_')}"

        try:
            # 插入记忆节点 - 使用KuZu支持的语法
            query = """
            CREATE (m:Memory {memory_id: $id, content: $content, timestamp: $timestamp, importance: $importance})
            """
            print(f"执行创建节点查询: {query}")
            print(
                f"参数: id={memory_id}, content={content}, timestamp={timestamp}, importance={importance}"
            )

            self.conn.execute(
                query,
                {
                    "id": memory_id,
                    "content": content,
                    "timestamp": timestamp,
                    "importance": importance,
                },
            )

            # 验证节点是否创建成功
            verify_query = """
            MATCH (m:Memory)
            WHERE m.memory_id = $id
            RETURN m.memory_id, m.content
            """
            result = self.conn.execute(verify_query, {"id": memory_id})
            found = False
            for row in result:
                found = True
                print(f"验证节点创建成功: {row}")

            if not found:
                print(f"警告: 节点创建后无法验证")

            # 连接到时间上相邻的记忆
            self._connect_to_recent_memories(memory_id, timestamp)

            # 连接到语义相似的记忆
            self._connect_to_similar_memories(memory_id, content)

            return memory_id
        except Exception as e:
            print(f"添加记忆时出错: {e}")
            return ""

    def _connect_to_recent_memories(self, memory_id: str, timestamp: str) -> None:
        """连接到时间上相邻的记忆

        Args:
            memory_id: 记忆ID
            timestamp: 时间戳
        """
        try:
            # 获取最近的记忆
            query = """
            MATCH (m:Memory)
            WHERE m.memory_id <> $id
            RETURN m.memory_id, m.timestamp
            ORDER BY m.timestamp DESC
            LIMIT 5
            """
            result = self.conn.execute(query, {"id": memory_id})

            # 创建时间关系
            current_time = datetime.fromisoformat(timestamp)
            for row in result:
                other_id = row[0]
                other_time_str = row[1]

                try:
                    other_time = datetime.fromisoformat(other_time_str)
                    time_diff = (current_time - other_time).total_seconds()

                    # 创建FOLLOWS关系
                    rel_query = """
                    MATCH (m1:Memory {memory_id: $id1}), (m2:Memory {memory_id: $id2})
                    CREATE (m1)-[r:FOLLOWS {time_diff: $time_diff}]->(m2)
                    """
                    self.conn.execute(
                        rel_query,
                        {"id1": memory_id, "id2": other_id, "time_diff": time_diff},
                    )
                    print(
                        f"创建时间关系: {memory_id} -> {other_id} (时间差: {time_diff}秒)"
                    )
                except Exception as e:
                    print(f"处理时间关系时出错: {e}")
        except Exception as e:
            print(f"连接到最近记忆时出错: {e}")

    def _connect_to_similar_memories(self, memory_id: str, content: str) -> None:
        """连接到语义相似的记忆

        Args:
            memory_id: 记忆ID
            content: 记忆内容
        """
        try:
            # 获取所有其他记忆
            query = """
            MATCH (m:Memory)
            WHERE m.memory_id <> $id
            RETURN m.memory_id, m.content
            """
            result = self.conn.execute(query, {"id": memory_id})

            # 计算相似度并创建关系
            for row in result:
                other_id, other_content = row

                # 简单的字符串匹配相似度计算
                content_lower = content.lower()
                other_content_lower = other_content.lower()

                # 初始化相似度
                similarity = 0.0

                # 检查内容是否相互包含
                if (
                    content_lower in other_content_lower
                    or other_content_lower in content_lower
                ):
                    # 基础相似度
                    similarity = 0.5

                    # 如果是精确匹配，给最高分
                    if content_lower == other_content_lower:
                        similarity = 1.0
                    # 如果一个是另一个的开头或结尾，给较高分
                    elif (
                        content_lower.startswith(other_content_lower)
                        or content_lower.endswith(other_content_lower)
                        or other_content_lower.startswith(content_lower)
                        or other_content_lower.endswith(content_lower)
                    ):
                        similarity = 0.8
                else:
                    # 计算共同词的相似度
                    words1 = set(content_lower.split())
                    words2 = set(other_content_lower.split())

                    if words1 and words2:
                        common_words = words1.intersection(words2)
                        if common_words:
                            # 使用Jaccard相似度
                            similarity = len(common_words) / len(words1.union(words2))

                # 如果相似度超过阈值，创建关系
                if similarity > 0.1:
                    rel_query = """
                    MATCH (m1:Memory {memory_id: $id1}), (m2:Memory {memory_id: $id2})
                    CREATE (m1)-[r:RELATED_TO {similarity: $similarity}]->(m2)
                    """
                    self.conn.execute(
                        rel_query,
                        {"id1": memory_id, "id2": other_id, "similarity": similarity},
                    )
                    print(
                        f"创建相似度关系: {memory_id} -> {other_id} (相似度: {similarity})"
                    )
        except Exception as e:
            print(f"连接到相似记忆时出错: {e}")

    def retrieve_relevant_memories(
        self, query: str, limit: int = 5, similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """检索与查询相关的记忆

        Args:
            query: 查询字符串
            limit: 返回结果数量限制
            similarity_threshold: 相似度阈值，低于此值的记忆将被过滤

        Returns:
            记忆列表，按重要性排序
        """
        try:
            # 首先获取所有记忆，按重要性排序
            cypher_query = """
            MATCH (m:Memory)
            RETURN m.memory_id, m.content, m.timestamp, m.importance
            ORDER BY m.importance DESC
            LIMIT $limit
            """
            result = self.conn.execute(cypher_query, {"limit": limit})

            # 处理查询结果
            memories = []
            for row in result:
                try:
                    memory_id, content, timestamp, importance = row
                    print(f"找到记忆: {row}")
                    memories.append(
                        {
                            "id": memory_id,
                            "content": content,
                            "timestamp": timestamp,
                            "importance": importance,
                        }
                    )
                except Exception as e:
                    print(f"处理记忆行时出错: {e}")

            print(f"检索到 {len(memories)} 条记忆")

            # 如果查询为空，返回所有记忆
            if not query.strip():
                print(f"筛选后剩余 {len(memories)} 条相关记忆")
                return memories

            # 计算每条记忆与查询的相似度
            relevant_memories = []
            for memory in memories:
                # 简单的字符串匹配相似度计算
                content = memory["content"].lower()
                query_lower = query.lower()

                # 检查是否包含查询词
                if query_lower in content:
                    # 计算一个简单的相似度分数
                    similarity = 0.5  # 基础分数
                    # 如果是精确匹配或接近精确匹配，给更高分数
                    if content == query_lower:
                        similarity = 1.0
                    elif content.startswith(query_lower) or content.endswith(
                        query_lower
                    ):
                        similarity = 0.8

                    # 调整相似度分数，考虑记忆的重要性
                    adjusted_similarity = similarity * (1 + memory["importance"] / 10)

                    print(
                        f"记忆 '{memory['content']}' 与查询 '{query}' 的相似度: {adjusted_similarity}"
                    )

                    if adjusted_similarity >= similarity_threshold:
                        memory["similarity"] = adjusted_similarity
                        relevant_memories.append(memory)
                else:
                    print(f"记忆 '{memory['content']}' 与查询 '{query}' 的相似度: 0.0")

            # 按相似度和重要性排序
            relevant_memories.sort(
                key=lambda x: (x.get("similarity", 0), x["importance"]), reverse=True
            )

            print(f"筛选后剩余 {len(relevant_memories)} 条相关记忆")
            return relevant_memories[:limit]
        except Exception as e:
            print(f"检索相关记忆时出错: {e}")
            return []

    def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """通过ID获取记忆

        Args:
            memory_id: 记忆ID

        Returns:
            记忆信息或None
        """
        try:
            query = """
            MATCH (m:Memory)
            WHERE m.memory_id = $id
            RETURN m.memory_id, m.content, m.timestamp, m.importance
            """

            result = self.conn.execute(query, {"id": memory_id})

            for row in result:
                return {
                    "id": row[0],
                    "content": row[1],
                    "timestamp": row[2],
                    "importance": row[3],
                }

            return None
        except Exception as e:
            print(f"通过ID获取记忆时出错: {e}")
            return None

    def update_memory_importance(self, memory_id: str, importance: int) -> bool:
        """更新记忆的重要性

        Args:
            memory_id: 记忆ID
            importance: 新的重要性评分

        Returns:
            更新是否成功
        """
        try:
            query = """
            MATCH (m:Memory)
            WHERE m.memory_id = $id
            SET m.importance = $importance
            """

            self.conn.execute(query, {"id": memory_id, "importance": importance})
            return True
        except Exception as e:
            print(f"更新记忆重要性时出错: {e}")
            return False


class MemorySaveTool(BaseTool):
    """保存记忆到图数据库的工具"""

    name: ClassVar[str] = "save_memory"
    description: ClassVar[str] = (
        "保存重要信息到记忆库中以便将来检索。输入应该是一个包含'content'和可选'importance'的JSON。"
    )
    memory_store: GraphMemoryStore = Field(default_factory=GraphMemoryStore)

    def _run(self, content: str, importance: int = 1) -> str:
        """保存记忆

        Args:
            content: 记忆内容
            importance: 重要性评分 (1-10)

        Returns:
            操作结果消息
        """
        memory_id = self.memory_store.add_memory(content, importance)
        if memory_id:
            return f"记忆已保存，ID: {memory_id}"
        else:
            return "保存记忆失败"


class MemoryRetrieveTool(BaseTool):
    """从图数据库检索记忆的工具"""

    name: ClassVar[str] = "retrieve_memories"
    description: ClassVar[str] = "检索与查询相关的记忆。输入应该是查询字符串。"
    memory_store: GraphMemoryStore = Field(default_factory=GraphMemoryStore)

    def _run(self, query: str, limit: int = 5) -> str:
        """检索相关记忆

        Args:
            query: 查询内容
            limit: 返回结果数量限制

        Returns:
            格式化的记忆列表
        """
        print(f"开始检索记忆，查询: '{query}'")
        memories = self.memory_store.retrieve_relevant_memories(query, limit)

        if not memories:
            return "没有找到相关记忆。"

        result = f"找到以下相关记忆 (共{len(memories)}条):\n\n"
        for i, memory in enumerate(memories, 1):
            try:
                timestamp = datetime.fromisoformat(memory["timestamp"]).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                result += f"{i}. [{timestamp}] (重要性: {memory['importance']})\n   {memory['content']}\n\n"
            except Exception as e:
                print(f"格式化记忆时出错: {e}")
                result += f"{i}. 记忆格式化错误: {str(memory)}\n\n"

        return result


def create_memory_tools(
    db_path: str = "db_cache/test_db/memory_db.kuzu",
) -> Tuple[MemorySaveTool, MemoryRetrieveTool]:
    """创建记忆工具

    Args:
        db_path: 数据库文件路径

    Returns:
        保存和检索记忆的工具元组
    """
    # 共享同一个记忆存储
    memory_store = GraphMemoryStore(db_path=db_path)

    save_tool = MemorySaveTool(memory_store=memory_store)
    retrieve_tool = MemoryRetrieveTool(memory_store=memory_store)

    return save_tool, retrieve_tool


def test_memory_graph():
    """测试记忆图谱功能"""
    print("开始测试记忆图谱功能...")

    # 创建记忆存储
    db_path = "db_cache/test_db/test_memory.kuzu"
    memory_store = GraphMemoryStore(db_path=db_path)

    # 添加记忆
    memory_id = memory_store.add_memory("这是一条测试记忆", importance=5)
    print(f"添加记忆成功，ID: {memory_id}")

    # 检索记忆
    memories = memory_store.retrieve_relevant_memories("测试记忆")
    print(f"检索到 {len(memories)} 条相关记忆")
    for memory in memories:
        print(f"- {memory['content']} (重要性: {memory['importance']})")

    print("测试完成")


if __name__ == "__main__":
    test_memory_graph()
