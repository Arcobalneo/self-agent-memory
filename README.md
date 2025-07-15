# Agent记忆图谱存储系统

这个项目实现了一个基于图数据库的Agent记忆存储和检索系统。它使用KuZu图数据库来存储Agent的记忆，并提供了工具让ReAct Agent可以保存和检索记忆。

## 项目结构

项目的文件夹结构和主要功能如下：

### 核心目录

- **agent/**: 包含Agent实现
  - `agent_with_memory.py`: 带记忆功能的LangGraph ReactAgent实现

- **misc/**: 包含核心功能模块
  - `memory_graph.py`: 记忆图谱实现，包含GraphMemoryStore类和记忆工具
  - `utils.py`: 通用工具函数，如LLM创建、环境变量处理等

- **db_cache/**: 存储KuZu图数据库文件
  - `memory_db.kuzu`: 主数据库文件
  - `test_db/`: 测试数据库目录

- **test_case/**: 包含测试代码
  - `test_memory_tools.py`: 记忆工具的测试代码
  - `test_agent_memory.py`: 记忆Agent的测试代码

### 主要文件

- `main.py`: 项目主入口，提供基本的ReactAgent示例
- `pyproject.toml`: 项目配置文件
- `.python-version`: Python版本配置

## 技术架构

系统主要包含以下组件：

1. **GraphMemoryStore**: 基于KuZu图数据库的记忆存储类
   - 管理图数据库连接和模式初始化
   - 提供记忆的添加、检索和更新功能
   - 建立记忆之间的关系（时间序列和语义相似性）

2. **记忆工具**: 
   - `MemorySaveTool`: 保存记忆的工具
   - `MemoryRetrieveTool`: 检索记忆的工具

3. **Agent实现**: 
   - 使用LangGraph的create_react_agent框架
   - 集成记忆工具，实现带记忆功能的Agent 