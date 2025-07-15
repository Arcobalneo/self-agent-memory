# LangGraph ReactAgent 示例

这个项目展示了如何使用 LangGraph 构建不同类型的 ReactAgent，从基础示例到高级应用。

## 项目结构

- `main.py`: 基础 LangGraph ReactAgent 示例，包含简单的天气查询和计算工具
- `advanced_agent.py`: 高级 LangGraph ReactAgent 示例，使用图形工作流和状态管理
- `human_in_loop_agent.py`: 人机协作 LangGraph ReactAgent 示例，展示如何实现人类参与
- `multi_agent_system.py`: 多代理协作系统示例，包含研究代理和计划代理
- `requirements.txt`: 项目依赖列表

## 安装

1. 克隆此仓库
2. 创建并激活虚拟环境（推荐）
   ```bash
   python -m venv venv
   source venv/bin/activate  # 在 Windows 上使用 venv\Scripts\activate
   ```
3. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

## 使用方法

### 基础 ReactAgent 示例

```bash
python main.py
```

这个示例展示了如何使用 LangGraph 的 `create_react_agent` 创建一个简单的代理，该代理可以查询天气和进行计算。

### 高级 ReactAgent 示例

```bash
python advanced_agent.py
```

这个示例展示了如何使用 LangGraph 的图形工作流和状态管理功能构建更复杂的代理。

### 人机协作 ReactAgent 示例

```bash
python human_in_loop_agent.py
```

这个示例展示了如何在 LangGraph 代理中实现人类参与（human-in-the-loop）功能，允许代理在执行过程中请求人类输入。

### 多代理协作系统

```bash
python multi_agent_system.py
```

这个示例展示了如何构建多代理协作系统，其中包含研究代理和计划代理，它们可以相互协作完成任务。

## 注意事项

- 这些示例需要 OpenAI API 密钥。请在运行前设置环境变量：
  ```bash
  export OPENAI_API_KEY=your_api_key_here
  ```
  或在 Windows 上：
  ```bash
  set OPENAI_API_KEY=your_api_key_here
  ```

- 示例中的工具函数（如 `get_weather` 和 `search_web`）使用模拟数据，在实际应用中应替换为真实 API 调用。

## 进一步学习

- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [LangChain 官方文档](https://python.langchain.com/docs/get_started/introduction)
- [LangGraph GitHub 仓库](https://github.com/langchain-ai/langgraph)
