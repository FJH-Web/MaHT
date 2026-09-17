"""
数据集模块

本模块提供数据集适配功能，用于处理用户上传的CSV数据。
主要功能：
- DataFrame 适配器：统一数据格式
- 数据集剖析：分析数据类型、缺失值等
- 意图解析：解析用户请求的列

设计原则：
- 使用延迟加载避免循环导入
"""

from statagent.datasets.intent import enforce_locked_mapping, resolve_requested_columns

__all__ = ["DataFrameAdapter", "enforce_locked_mapping", "profile_dataframe", "resolve_requested_columns"]


def __getattr__(name: str):
    """延迟加载数据适配器，避免应用管线初始化期间的循环导入。"""
    if name in {"DataFrameAdapter", "profile_dataframe"}:
        from statagent.datasets.profile import DataFrameAdapter, profile_dataframe

        return {"DataFrameAdapter": DataFrameAdapter, "profile_dataframe": profile_dataframe}[name]
    raise AttributeError(name)