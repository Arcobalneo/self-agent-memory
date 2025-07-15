import os
import shutil
import re
import jieba
from typing import Dict, List, Optional, Any, Tuple, ClassVar

from langchain_core.tools import BaseTool
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from pydantic import BaseModel, Field


class MemoryNode(BaseModel):
    """记忆节点模型"""

    id: str
    content: str


class BM25MemoryStore:
    """基于BM25算法的记忆存储"""

    def __init__(self, cache_dir: str = "db_cache/bm25_db"):
        """初始化BM25记忆存储

        Args:
            cache_dir: 缓存目录路径
        """
        # 确保缓存目录存在
        if not os.path.exists(cache_dir):
            try:
                os.makedirs(cache_dir, exist_ok=True)
                print(f"创建缓存目录: {cache_dir}")
            except Exception as e:
                print(f"创建缓存目录时出错: {e}")

        self.cache_dir = cache_dir
        self.memory_file = os.path.join(cache_dir, "memories.txt")

        # 存储所有记忆
        self.memories = []

        # BM25检索相关
        self.corpus = []  # 文本内容列表
        self.tokenized_corpus = []  # 分词后的文本列表
        self.bm25 = None  # BM25检索器

        # 加载已有记忆
        self._load_memories()

    def _tokenize_text(self, text):
        """对文本进行中英文分词

        Args:
            text: 待分词文本

        Returns:
            分词结果列表
        """
        # 用正则分割中英文
        pattern = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z0-9\s]+")
        segments = pattern.findall(text)

        words = []
        for seg in segments:
            if re.match(r"[\u4e00-\u9fff]+", seg):
                # 中文部分用jieba
                words += jieba.lcut(seg)
            else:
                # 英文部分按空格分词
                words += [w for w in seg.strip().split() if w]

        return words

    def _load_memories(self):
        """从文件加载记忆"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                for line in lines:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        memory_id, content = parts[:2]
                        self.memories.append({"id": memory_id, "content": content})
                        self.corpus.append(content)

                # 初始化BM25检索器
                if self.corpus:
                    # 对文本进行中英文分词
                    self.tokenized_corpus = [
                        self._tokenize_text(doc) for doc in self.corpus
                    ]
                    self.bm25 = BM25Okapi(self.tokenized_corpus)
                    print(f"已加载 {len(self.memories)} 条记忆")
                else:
                    self._init_empty_retriever()
            except Exception as e:
                print(f"加载记忆时出错: {e}")
                self._init_empty_retriever()
        else:
            self._init_empty_retriever()

    def _init_empty_retriever(self):
        """初始化空的BM25检索器"""
        self.memories = [{"id": "init_memory", "content": "初始化记忆"}]
        self.corpus = ["初始化记忆"]
        self.tokenized_corpus = [self._tokenize_text("初始化记忆")]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print("初始化空记忆检索器")

    def _save_memories(self):
        """保存记忆到文件"""
        try:
            with open(self.memory_file, "w", encoding="utf-8") as f:
                for memory in self.memories:
                    f.write(f"{memory['id']}\t{memory['content']}\n")
            print(f"记忆已保存到 {self.memory_file}")
        except Exception as e:
            print(f"保存记忆时出错: {e}")

    def add_memory(self, content: str) -> str:
        """添加新记忆

        Args:
            content: 记忆内容

        Returns:
            记忆ID
        """
        # 生成简单的随机ID
        import uuid

        memory_id = f"mem_{str(uuid.uuid4())[:8]}"

        memory = {"id": memory_id, "content": content}

        # 添加到记忆列表
        self.memories.append(memory)

        # 添加到语料库
        self.corpus.append(content)

        # 更新BM25检索器
        self.tokenized_corpus = [self._tokenize_text(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # 保存记忆
        self._save_memories()

        return memory_id

    def retrieve_relevant_memories(
        self, query: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """检索与查询相关的记忆

        Args:
            query: 查询字符串
            limit: 返回结果数量限制

        Returns:
            记忆列表，按相关性排序
        """
        if not self.bm25 or not self.corpus:
            print("没有可用的记忆进行检索")
            return []

        # 处理空查询情况
        if not query or query.strip() == "":
            print("查询为空，返回空列表")
            return []

        try:
            # 对查询进行中英文分词
            tokenized_query = self._tokenize_text(query)

            # 使用BM25检索相关文档
            doc_scores = self.bm25.get_scores(tokenized_query)

            # 将文档ID和分数组合，并按分数降序排序
            scored_docs = [
                (i, score)
                for i, score in enumerate(doc_scores)
                if i < len(self.memories)
            ]
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            # 获取前limit个文档
            top_docs = scored_docs[:limit]

            print(f"BM25检索到 {len(self.corpus)} 条记忆")

            # 转换为记忆格式
            memories = []
            for i, (doc_idx, score) in enumerate(top_docs):
                if doc_idx < len(self.memories):  # 确保索引在有效范围内
                    memories.append(
                        {
                            "id": self.memories[doc_idx]["id"],
                            "content": self.memories[doc_idx]["content"],
                            "score": float(score),
                            "rank": i + 1,
                        }
                    )

            print(f"返回 {len(memories)} 条相关记忆")
            return memories
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
        for memory in self.memories:
            if memory["id"] == memory_id:
                return memory
        return None

    def clear_all_memories(self):
        """清除所有记忆（测试用）"""
        self.memories = []
        self.corpus = []

        # 重新初始化检索器
        self._init_empty_retriever()

        # 如果文件存在，删除它
        if os.path.exists(self.memory_file):
            os.remove(self.memory_file)


class MemorySaveTool(BaseTool):
    """保存记忆到BM25存储的工具"""

    name: ClassVar[str] = "save_memory"
    description: ClassVar[str] = "保存信息到记忆库中以便将来检索。输入应该是记忆内容。"
    memory_store: BM25MemoryStore = Field(default_factory=BM25MemoryStore)

    def _run(self, content: str) -> str:
        """保存记忆

        Args:
            content: 记忆内容

        Returns:
            操作结果消息
        """
        memory_id = self.memory_store.add_memory(content)
        if memory_id:
            return f"记忆已保存，ID: {memory_id}"
        else:
            return "保存记忆失败"


class MemoryRetrieveTool(BaseTool):
    """从BM25存储检索记忆的工具"""

    name: ClassVar[str] = "retrieve_memories"
    description: ClassVar[str] = "检索与查询相关的记忆。输入应该是查询字符串。"
    memory_store: BM25MemoryStore = Field(default_factory=BM25MemoryStore)

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
                score_info = (
                    f" (分数: {memory.get('score', 'N/A'):.4f})"
                    if "score" in memory
                    else ""
                )
                result += f"{i}. {memory['content']}{score_info}\n\n"
            except Exception as e:
                print(f"格式化记忆时出错: {e}")
                result += f"{i}. 记忆格式化错误: {str(memory)}\n\n"

        return result


def create_memory_tools(
    cache_dir: str = "db_cache/bm25_db",
) -> Tuple[MemorySaveTool, MemoryRetrieveTool]:
    """创建记忆工具

    Args:
        cache_dir: 缓存目录路径

    Returns:
        保存和检索记忆的工具元组
    """
    # 共享同一个记忆存储
    memory_store = BM25MemoryStore(cache_dir=cache_dir)

    save_tool = MemorySaveTool(memory_store=memory_store)
    retrieve_tool = MemoryRetrieveTool(memory_store=memory_store)

    return save_tool, retrieve_tool


def test_memory_bm25_basic():
    """基本BM25记忆功能测试"""
    print("\n=== 开始基本BM25记忆功能测试 ===")

    # 创建测试目录并确保测试环境干净
    test_dir = "db_cache/test_basic"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    # 创建记忆存储
    memory_store = BM25MemoryStore(cache_dir=test_dir)

    # 添加明确的测试记忆，包含中英文混合
    memory_id1 = memory_store.add_memory("苹果是一种水果，也是一家科技公司 Apple")
    print(f"添加记忆1成功，ID: {memory_id1}")

    memory_id2 = memory_store.add_memory("香蕉banana是黄色的水果")
    print(f"添加记忆2成功，ID: {memory_id2}")

    memory_id3 = memory_store.add_memory("Python是一种流行的编程语言，用于AI开发")
    print(f"添加记忆3成功，ID: {memory_id3}")

    memory_id4 = memory_store.add_memory(
        "人工智能AI正在改变世界，包括NLP和机器学习ML技术"
    )
    print(f"添加记忆4成功，ID: {memory_id4}")

    # 检索水果相关记忆
    print("\n测试'水果'查询:")
    memories = memory_store.retrieve_relevant_memories("水果")
    print(f"检索到 {len(memories)} 条关于'水果'的相关记忆")
    for memory in memories:
        print(f"- {memory['content']}")

    # 检索编程相关记忆
    print("\n测试'编程'查询:")
    memories = memory_store.retrieve_relevant_memories("编程")
    print(f"检索到 {len(memories)} 条关于'编程'的相关记忆")
    for memory in memories:
        print(f"- {memory['content']}")

    # 检索人工智能相关记忆
    print("\n测试'人工智能'查询:")
    memories = memory_store.retrieve_relevant_memories("人工智能")
    print(f"检索到 {len(memories)} 条关于'人工智能'的相关记忆")
    for memory in memories:
        print(f"- {memory['content']}")

    # 测试英文查询
    print("\n测试英文'Apple'查询:")
    memories = memory_store.retrieve_relevant_memories("Apple")
    print(f"检索到 {len(memories)} 条关于'Apple'的相关记忆")
    for memory in memories:
        print(f"- {memory['content']}")

    # 测试英文查询
    print("\n测试英文'AI'查询:")
    memories = memory_store.retrieve_relevant_memories("AI")
    print(f"检索到 {len(memories)} 条关于'AI'的相关记忆")
    for memory in memories:
        print(f"- {memory['content']}")

    # 测试空查询
    print("\n测试空查询:")
    memories = memory_store.retrieve_relevant_memories("")
    print(f"空查询检索到 {len(memories)} 条记忆")

    # 测试通过ID获取记忆
    memory = memory_store.get_memory_by_id(memory_id1)
    print(f"\n通过ID获取记忆: {memory['content']}")

    # 清除所有记忆
    memory_store.clear_all_memories()
    memories = memory_store.retrieve_relevant_memories("水果")
    print(f"\n清除后检索到 {len(memories)} 条记忆")

    print("=== 基本BM25记忆功能测试完成 ===\n")


def run_all_tests():
    """运行所有测试用例"""
    print("\n======== 开始运行BM25记忆测试 ========")

    test_memory_bm25_basic()

    print("\n======== BM25记忆测试完成 ========")


if __name__ == "__main__":
    run_all_tests()
