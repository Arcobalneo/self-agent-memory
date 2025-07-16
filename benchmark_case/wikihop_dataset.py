import json
import os
from typing import List, Dict, Any, Optional


class WikiHopQADataset:
    """
    加载和处理WikiHop QA数据集的类
    """

    def __init__(self, data_dir: str):
        """
        初始化WikiHopQA数据集

        参数:
            data_dir: 包含train.json, dev.json和test.json的目录路径
        """
        self.data_dir = data_dir
        self.train_data = None
        self.dev_data = None
        self.test_data = None

    def load_data(self, split: str = "all") -> None:
        """
        加载指定分割的数据

        参数:
            split: 要加载的数据分割，可以是"train", "dev", "test"或"all"
        """
        if split == "all" or split == "train":
            train_path = os.path.join(self.data_dir, "train.json")
            if os.path.exists(train_path):
                with open(train_path, "r", encoding="utf-8") as f:
                    self.train_data = json.load(f)
                print(f"已加载训练集: {len(self.train_data)} 条记录")

        if split == "all" or split == "dev":
            dev_path = os.path.join(self.data_dir, "dev.json")
            if os.path.exists(dev_path):
                with open(dev_path, "r", encoding="utf-8") as f:
                    self.dev_data = json.load(f)
                print(f"已加载验证集: {len(self.dev_data)} 条记录")

        if split == "all" or split == "test":
            test_path = os.path.join(self.data_dir, "test.json")
            if os.path.exists(test_path):
                with open(test_path, "r", encoding="utf-8") as f:
                    self.test_data = json.load(f)
                print(f"已加载测试集: {len(self.test_data)} 条记录")

    def get_data(self, split: str) -> List[Dict[str, Any]]:
        """
        获取指定分割的数据

        参数:
            split: 数据分割，可以是"train", "dev"或"test"

        返回:
            分割数据列表(list<dict>)
        """
        if split == "train":
            if self.train_data is None:
                self.load_data("train")
            return self.train_data
        elif split == "dev":
            if self.dev_data is None:
                self.load_data("dev")
            return self.dev_data
        elif split == "test":
            if self.test_data is None:
                self.load_data("test")
            return self.test_data
        else:
            raise ValueError(f"无效的分割名称: {split}")


# 使用示例
if __name__ == "__main__":
    # 创建数据集实例
    dataset = WikiHopQADataset(
        "/mnt/data/gyzou/expr_workplace/self-agent-memory/benchmark_cache/manual_hf_download/2wikihopqa/data"
    )

    # 加载开发集数据
    dataset.load_data("dev")

    # 获取开发集数据
    dev_data = dataset.get_data("dev")
    print(f"开发集样本数量: {len(dev_data)}")

    # 查看第一个样本
    print("\n第一个样本:")
    sample = dev_data[0]
    print(f"问题: {sample['question']}")
    print(f"答案: {sample['answer']}")
    print(f"上下文实体数量: {len(sample['context'])}")
