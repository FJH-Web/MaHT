"""
统计方法注册与调度模块

本模块提供统计方法的注册和调度功能。
核心类：
- StatisticalMethodRegistry: 统计方法注册表，管理所有方法卡片
- StatisticalDispatcher: 统计方法调度器，根据方法ID调用对应的函数

设计原则：
- 每个方法通过 Method Card 注册
- 调度器根据方法ID查找并执行预部署的统计算法
"""

from statagent.api.dispatcher import StatisticalDispatcher
from statagent.api.registry import StatisticalMethodRegistry

__all__ = ["StatisticalDispatcher", "StatisticalMethodRegistry"]