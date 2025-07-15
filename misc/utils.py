"""
通用工具和函数
"""

import os
import dotenv
from typing import Dict, List, Union

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


# 创建语言模型实例
def create_llm(temperature: float = 0.3) -> ChatOpenAI:
    """
    创建语言模型实例

    Args:
        temperature: 模型温度参数，控制输出的随机性

    Returns:
        ChatOpenAI: 配置好的语言模型实例
    """
    model_name = os.getenv("MODEL_NAME")
    model_key = os.getenv("MODEL_KEY")
    model_base_url = os.getenv("MODEL_BASE_URL")

    return ChatOpenAI(
        model=model_name,
        api_key=model_key,
        base_url=model_base_url,
        temperature=temperature,
    )


# 工具函数
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。"""
    weather_data = {
        "北京": "晴朗，39°C",
        "上海": "多云，40°C",
        "广州": "雨天，41°C",
        "深圳": "阵雨，42°C",
    }

    return f"{city}的天气是：{weather_data.get(city, '未知')}"


@tool
def calculate(expression: str) -> str:
    """计算简单的数学表达式。"""
    try:
        result = eval(expression)
        return f"计算结果是：{result}"
    except Exception as e:
        return f"计算错误：{str(e)}"


@tool
def search_web(query: str) -> str:
    """搜索网络获取信息。"""
    # 模拟网络搜索结果
    search_results = {
        "北京": "北京是中国的首都，人口约2170万，面积16410平方公里，有故宫、长城等著名景点。",
        "上海": "上海是中国最大的城市，人口约2480万，是重要的金融和商业中心，有东方明珠、外滩等地标。",
        "广州": "广州是广东省省会，人口约1500万，是华南地区的经济中心，以美食和商贸闻名。",
        "深圳": "深圳是中国重要的科技创新中心，人口约1760万，毗邻香港，是中国改革开放的窗口。",
    }

    # 检查查询是否包含关键词
    for key, value in search_results.items():
        if key in query:
            return value

    return f"搜索 '{query}' 的结果：没有找到相关信息。"


# 环境变量处理
def load_environment():
    """
    加载环境变量并检查必要的变量是否存在

    Returns:
        bool: 如果所有必要的环境变量都存在则返回True，否则返回False
    """
    # 加载.env文件中的环境变量
    dotenv.load_dotenv()

    # 检查必要的环境变量
    missing_vars = check_environment_variables()

    if missing_vars:
        print(f"错误: 缺少必要的环境变量: {', '.join(missing_vars)}")
        print("请确保在.env文件中设置了这些变量或在环境中直接设置。")
        return False

    return True


# 检查环境变量
def check_environment_variables() -> List[str]:
    """
    检查必要的环境变量是否存在

    Returns:
        List[str]: 缺少的环境变量列表，如果所有变量都存在则返回空列表
    """
    required_vars = ["MODEL_NAME", "MODEL_KEY", "MODEL_BASE_URL"]
    return [var for var in required_vars if not os.getenv(var)]
