"""
主动假设调度器模块

本模块提供主动假设调度功能。
核心类：
- Scheduler: 调度器抽象基类
- ActiveHypothesisScheduler: 基于效用的主动假设调度器实现
- UtilityEstimate: 效用估计，包含信息增益和成本

调度器根据效用函数 U(H_i) = EIG(H_i) - Cost(H_i) 选择下一个待检验的假设。
"""

from statagent.scheduler.active_scheduler import ActiveHypothesisScheduler, Scheduler, UtilityEstimate

__all__ = ["ActiveHypothesisScheduler", "Scheduler", "UtilityEstimate"]