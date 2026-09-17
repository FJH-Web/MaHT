"""
统计方法调度器模块

本模块提供统一的执行边界，包含验证、日志记录和异常包装功能。
核心功能：
- 验证方法卡片与注册表的一致性
- 检查必需输入参数
- 执行统计方法并记录执行时间
- 包装异常并添加执行日志

设计原则：
- Dispatcher 只允许运行注册表中的本地函数
- 执行前必须验证方法卡片的有效性
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from statagent.api.registry import StatisticalMethodRegistry
from statagent.protocol.statistical_object import MethodResult
from statagent.retrieval.method_card import MethodCard

logger = logging.getLogger("agent.statistics")


class StatisticalExecutionError(RuntimeError):
    """统计方法执行错误异常"""
    pass


@dataclass(slots=True)
class StatisticalDispatcher:
    """统计方法调度器：统一执行边界"""
    
    registry: StatisticalMethodRegistry

    def execute(
        self,
        card: MethodCard,
        data: dict[str, Any],
        parameters: dict[str, Any] | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> MethodResult:
        """执行统计方法
        
        参数：
            card: 方法卡片
            data: 输入数据字典
            parameters: 可选的参数字典
            diagnostics: 可选的诊断信息（如缺失值处理）
            
        返回：
            MethodResult: 方法执行结果
        """
        registered = self.registry.card(card.method_id)
        if registered.api != card.api:
            raise StatisticalExecutionError("Method Card API does not match the registry")
        missing = [name for name in card.inputs if name not in data and name not in (parameters or {})]
        if missing:
            raise StatisticalExecutionError(f"Missing required inputs: {missing}")
        start = time.perf_counter()
        logger.info("statistical_method_started", extra={"method_id": card.method_id, "api": card.api})
        try:
            result = self.registry.function(card.api)(data, dict(parameters or {}))
        except Exception as exc:
            logger.exception("statistical_method_failed", extra={"method_id": card.method_id})
            raise StatisticalExecutionError(f"{card.name} failed: {exc}") from exc
        if diagnostics:
            result.diagnostics.update(diagnostics)
        result.reproducibility.update(
            {
                "method_card_version": card.version,
                "elapsed_seconds": round(time.perf_counter() - start, 6),
                "parameters": dict(parameters or {}),
            }
        )
        logger.info("statistical_method_finished", extra={"method_id": card.method_id})
        return result