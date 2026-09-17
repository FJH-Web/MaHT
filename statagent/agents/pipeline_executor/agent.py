"""
流水线执行器模块

实现自动假设检验流水线：
1. 解析用户需求，确定主检验方法
2. 检查方法的假设依赖
3. 执行假设检验（根据假设检验方法的输入要求构建数据）
4. 根据假设检验结果决定是否切换方法
5. 执行主检验并返回结果
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from statagent.api.dispatcher import StatisticalDispatcher
from statagent.protocol.statistical_object import StatisticalObject
from statagent.retrieval.method_card import AssumptionDep, MethodCard


@dataclass
class AssumptionResult:
    """假设检验结果"""

    assumption_type: str
    test_method_id: str | None
    test_method_name: str
    p_value: float | None
    passed: bool
    conclusion: str
    tested_variable: str | None = None
    result_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """流水线执行结果"""

    original_method_id: str | None
    final_method_id: str
    final_method_name: str
    execution_summary: str
    assumption_results: list[AssumptionResult]
    alternative_used: bool = False
    alternative_method_id: str | None = None
    warnings: list[str] = field(default_factory=list)


def _build_assumption_test_data(
    main_message: StatisticalObject,
    main_data: dict[str, Any],
    test_card: MethodCard,
) -> dict[str, Any]:
    """
    根据假设检验方法的输入要求，构建正确的数据

    参数：
        main_message: 主检验的消息对象
        main_data: 主检验的输入数据
        test_card: 假设检验方法的卡片

    返回：
        适合假设检验方法的数据字典
    """
    test_inputs = test_card.inputs
    data_mapping = main_message.method.get("data_mapping", {}) if main_message.method else {}

    if not data_mapping:
        return main_data

    if test_inputs and test_inputs[0] in main_data:
        return main_data

    if "values" in test_inputs:
        for key, mapping in data_mapping.items():
            if mapping.get("kind") == "column":
                col = mapping.get("column")
                if col and col in main_data:
                    return {"values": main_data[col]}

    if "x" in test_inputs:
        for key, mapping in data_mapping.items():
            if mapping.get("kind") == "column":
                col = mapping.get("column")
                if col and col in main_data:
                    return {"x": main_data[col]}

    return main_data


class PipelineExecutor:
    """流水线执行器"""

    def __init__(self, dispatcher: StatisticalDispatcher, registry: Any):
        self.dispatcher = dispatcher
        self.registry = registry

    def check_and_prepare_method(
        self,
        message: StatisticalObject,
        data: dict[str, Any],
        diagnostics: dict[str, Any],
    ) -> tuple[str, str, list[AssumptionResult], bool]:
        """
        检查假设依赖并返回最终方法ID

        参数：
            message: 统计对象（包含方法选择信息）
            data: 已处理的数据字典
            diagnostics: 诊断信息

        返回：
            tuple: (final_method_id, execution_summary, assumption_results, alternative_used)
        """
        if message.method is None:
            return "", "无方法选择", [], False

        method_id = message.method.get("method_id", "")
        card = self.registry.card(method_id)

        assumption_results: list[AssumptionResult] = []
        final_method_id = method_id
        alternative_used = False

        if not card.assumption_deps:
            return method_id, "该方法无假设依赖", [], False

        for dep in card.assumption_deps:
            if dep.test_method_id is None:
                continue

            try:
                test_card = self.registry.card(dep.test_method_id)

                test_data = _build_assumption_test_data(message, data, test_card)

                result = self.dispatcher.execute(
                    test_card,
                    test_data,
                    {},
                    diagnostics,
                )

                p_value = result.p_value

                tested_var = None
                if "values" in test_data:
                    tested_var = "values"
                elif "x" in test_data:
                    tested_var = "x"

                passed = p_value is not None and p_value >= dep.threshold

                assumption_result = AssumptionResult(
                    assumption_type=dep.assumption_type,
                    test_method_id=dep.test_method_id,
                    test_method_name=test_card.name,
                    p_value=p_value,
                    passed=passed,
                    conclusion=f"假设{'满足' if passed else '不满足'} (p={p_value:.4f if p_value else 'N/A'})",
                    tested_variable=tested_var,
                    result_data={
                        "estimates": result.estimates,
                        "diagnostics": result.diagnostics,
                    },
                )
                assumption_results.append(assumption_result)

                if not passed and dep.on_failure == "use_alternative" and dep.alternative_method_id:
                    final_method_id = dep.alternative_method_id
                    alternative_used = True
                    break

            except Exception as e:
                test_name = "未知"
                if dep.test_method_id:
                    try:
                        test_name = self.registry.card(dep.test_method_id).name
                    except Exception:
                        pass
                assumption_results.append(AssumptionResult(
                    assumption_type=dep.assumption_type,
                    test_method_id=dep.test_method_id,
                    test_method_name=test_name,
                    p_value=None,
                    passed=False,
                    conclusion=f"假设检验执行失败: {str(e)}",
                ))

        summary_parts = [f"检验了 {len(assumption_results)} 个假设"]
        if alternative_used:
            new_card = self.registry.card(final_method_id)
            summary_parts.append(f"由于假设不满足，切换到替代方法: {new_card.name}")
        else:
            summary_parts.append("所有假设满足，执行原方法")

        return final_method_id, "; ".join(summary_parts), assumption_results, alternative_used