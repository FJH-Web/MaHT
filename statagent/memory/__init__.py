"""
统计内存模块

本模块提供统计推断历史的存储功能。
核心类：
- MemoryRecord: 单条内存记录，包含假设状态和推断结果
- StatisticalMemory: 统计内存，管理所有推断历史

设计原则：
- 不保存聊天记录，只保存结构化的推断过程
- 存储接口使用 JSON，可替换为数据库适配器
"""

from statagent.memory.statistical_memory import MemoryRecord, StatisticalMemory

__all__ = ["MemoryRecord", "StatisticalMemory"]