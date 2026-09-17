"""
解释 Agent 模块

InterpretationAgent 负责解释不可变的数值结果和本地生成的图表。
主要功能：
- 调用 LLM 对统计结果进行结构化解释
- 对每个图表进行语义解读
- 提供统计意义和实际意义的解释

注意：LLM 解释不能改变数值结果，只能提供语义层面的解读。
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from statagent.agents.base import StatisticalAgent
from statagent.llm.deepseek import DeepSeekClient, LLMError
from statagent.protocol.statistical_object import InterpretationBundle, StatisticalObject


@dataclass(slots=True)
class InterpretationAgent(StatisticalAgent):
    """解释 Agent：对统计结果和可视化进行LLM解释"""
    
    client: DeepSeekClient | None = None
    allow_fallback: bool = True

    def process(self, message: StatisticalObject) -> StatisticalObject:
        """对统计结果和可视化进行解释"""
        if message.result is None or message.method is None:
            return message
        figures = message.result.visualizations
        context = message.hypothesis.metadata.get("analysis_context", {})
        original_request = str(context.get("original_request", message.hypothesis.description))
        locked_columns = [str(item) for item in context.get("requested_columns", [])]
        match = message.method.get("match", {})
        payload = {
            "original_request": original_request,
            "locked_columns": locked_columns,
            "final_data_mapping": match.get("data_mapping", {}),
            "mapping_audit": match.get("mapping_audit", ""),
            "hypothesis": message.hypothesis.to_dict(),
            "method_card": message.method,
            "result": message.result.to_dict(),
            "decision": message.decision.value,
            "figures": [
                {"id": f.id, "title": f.title, "kind": f.kind, "caption": f.caption, "semantic_payload": f.semantic_payload}
                for f in figures
            ],
        }
        if self.client and self.client.configured:
            try:
                try:
                    response = self.client.complete_json(self._system_prompt(), json.dumps(payload, ensure_ascii=False), images=[f.path for f in figures])
                except LLMError:
                    response = self.client.complete_json(self._system_prompt(), json.dumps(payload, ensure_ascii=False))
                bundle = self._validate(
                    response,
                    self.client.model,
                    [f.id for f in figures],
                    [str(key) for key in message.result.estimates],
                )
                short_figures = [f for f in figures if len(bundle.figure_interpretations.get(f.id, "").strip()) < 100]
                if short_figures:
                    bundle = self._expand_figure_interpretations(bundle, short_figures, payload, original_request)
                message.interpretation = self._complete_coverage(bundle, message, original_request)
                return message
            except (LLMError, ValueError, TypeError):
                if not self.allow_fallback:
                    raise
        message.interpretation = self._fallback(message)
        return message

    @staticmethod
    def _system_prompt() -> str:
        return (
            "你是统计结果与可视化图形的中文解释模块。只能读取给定的方法卡、不可变数值结果、假设诊断、图形语义清单以及模型支持时附带的图片。"
            "禁止重新计算、修改或编造任何数字。所有解释必须使用专业、准确、适合中文统计学论文的简体中文。返回 JSON，字段为 summary、"
            "statistical_interpretation、practical_interpretation、assumption_assessment、limitations（数组）、result_interpretations（以统计结果字段为键）、"
            "figure_interpretations（以图形 id 为键）和 conclusion。必须覆盖每个主要统计结果和每幅图。如果 result.diagnostics 中记录了缺失值处理信息（如 removed_rows > 0），"
            "必须在 limitations 中说明删除了多少条记录。original_request 是最高优先级约束，只能解释"
            "locked_columns 与 final_data_mapping 中实际变量，禁止解释索引列或未点名变量，更禁止为映射错误寻找理由。每张非平凡图至少约 120 个汉字，"
            "完整说明坐标轴或视觉编码、分布/趋势/离散/异常值/缺失信息、量化证据、与原始研究问题的关系及图形局限。"
        )

    def _expand_figure_interpretations(self, bundle, figures, payload, original_request):
        request = {
            "original_request": original_request,
            "locked_columns": payload.get("locked_columns", []),
            "final_data_mapping": payload.get("final_data_mapping", {}),
            "figures_to_expand": [{"id": f.id, "title": f.title, "kind": f.kind, "caption": f.caption, "semantic_payload": f.semantic_payload, "current_interpretation": bundle.figure_interpretations.get(f.id, "")} for f in figures],
        }
        prompt = "返回只有 figure_interpretations 字段的 JSON。每图至少约 120 个汉字，全面解释图形编码、形态、量化发现、与原始需求的关系和局限；不得编造数值、解释未锁定列或改变映射。"
        try:
            response = self.client.complete_json(prompt, json.dumps(request, ensure_ascii=False), max_tokens=12000)
            expanded = self._as_interpretation_map(
                response.get("figure_interpretations", {}), [f.id for f in figures]
            )
            for figure in figures:
                text = expanded.get(figure.id, "").strip()
                if text:
                    bundle.figure_interpretations[figure.id] = text
        except (AttributeError, LLMError, ValueError, TypeError):
            pass
        return bundle

    @classmethod
    def _complete_coverage(cls, bundle, message, original_request):
        assert message.result is not None
        for key, value in message.result.estimates.items():
            if not bundle.result_interpretations.get(str(key), "").strip():
                bundle.result_interpretations[str(key)] = f"“{key}”的记录值为 {str(value)[:300]}。该数值应严格围绕“{original_request}”解释，并结合单位、效应方向、置信区间、研究设计和统计假设判断。"
        for figure in message.result.visualizations:
            current = bundle.figure_interpretations.get(figure.id, "").strip()
            if len(current) < 100:
                bundle.figure_interpretations[figure.id] = cls._detailed_fallback_figure(figure, original_request, current)
        return bundle

    @staticmethod
    def _detailed_fallback_figure(figure, original_request: str, existing: str = "") -> str:
        semantic = json.dumps(figure.semantic_payload, ensure_ascii=False, default=str)
        semantic = semantic[:900] + ("……" if len(semantic) > 900 else "")
        lead = f"{existing.rstrip('。')}。" if existing else f"该图为“{figure.title}”，属于{figure.kind}图形。"
        return f"{lead}图形用途与原始研究需求“{original_request}”直接对应；应依据图题、坐标轴、分组及颜色等视觉编码识别比较对象。图形说明为：{figure.caption}。生成该图的可核对语义数据为：{semantic}。解读时应观察中心位置、分布形态、离散程度、组间重叠、趋势、异常点及缺失模式中适用的部分，并与正式统计检验的估计量、效应量、置信区间和 P 值相互印证。图形主要提供探索性模式和数据质量线索；它本身不能证明因果关系，也不能替代抽样不确定性、模型假设及研究设计审查。"

    @staticmethod
    def _validate(
        payload: dict,
        model: str,
        figure_ids: list[str] | None = None,
        result_keys: list[str] | None = None,
    ) -> InterpretationBundle:
        limitations = payload.get("limitations", [])
        if isinstance(limitations, str):
            limitations = [limitations]
        return InterpretationBundle(
            summary=str(payload.get("summary", "")), statistical_interpretation=str(payload.get("statistical_interpretation", "")),
            practical_interpretation=str(payload.get("practical_interpretation", "")), assumption_assessment=str(payload.get("assumption_assessment", "")),
            limitations=[str(x) for x in limitations],
            result_interpretations=InterpretationAgent._as_interpretation_map(payload.get("result_interpretations", {}), result_keys or []),
            figure_interpretations=InterpretationAgent._as_interpretation_map(payload.get("figure_interpretations", {}), figure_ids or []),
            conclusion=str(payload.get("conclusion", "")), model=model,
        )

    @staticmethod
    def _as_interpretation_map(value, expected_ids: list[str] | None = None) -> dict[str, str]:
        """兼容模型返回的字典、对象列表或按图顺序排列的文本列表。"""
        expected = list(expected_ids or [])
        if isinstance(value, dict):
            return {str(key): str(text) for key, text in value.items()}
        if not isinstance(value, list):
            return {}
        normalized: dict[str, str] = {}
        for index, item in enumerate(value):
            fallback_id = expected[index] if index < len(expected) else str(index)
            if isinstance(item, str):
                normalized[fallback_id] = item
                continue
            if not isinstance(item, dict):
                continue
            item_id = str(
                item.get("id")
                or item.get("figure_id")
                or item.get("result_id")
                or item.get("key")
                or fallback_id
            )
            text = (
                item.get("interpretation")
                or item.get("text")
                or item.get("explanation")
                or item.get("analysis")
                or item.get("content")
                or ""
            )
            normalized[item_id] = str(text)
        return normalized

    @staticmethod
    def _fallback(message: StatisticalObject) -> InterpretationBundle:
        result = message.result
        assert result is not None
        context = message.hypothesis.metadata.get("analysis_context", {})
        original_request = str(context.get("original_request", message.hypothesis.description))
        p_value = "未报告" if result.p_value is None else f"{result.p_value:.4g}"
        significance = "低于" if result.p_value is not None and result.p_value < 0.05 else "不低于"
        
        diagnostics = result.diagnostics or {}
        missing_info = ""
        if diagnostics.get("removed_rows", 0) > 0:
            original_rows = diagnostics.get("original_rows", 0)
            valid_rows = diagnostics.get("valid_rows", 0)
            removed = diagnostics.get("removed_rows", 0)
            missing_info = f"分析前因缺失值处理删除了 {removed} 条记录（原始 {original_rows} 条，有效 {valid_rows} 条）。"
        
        limitations = result.warnings or ["由于未获得可用的 DeepSeek 响应，本段采用本地确定性中文回退解释。"]
        if missing_info:
            limitations.insert(0, missing_info)
        
        return InterpretationBundle(
            summary=f"所选统计方法得到的 P 值为 {p_value}。", statistical_interpretation=f"P 值{significance}预设的 0.05 阈值；应结合估计量、效应量和不确定性共同解释。",
            practical_interpretation=f"效应量为：{result.effect_size or '未报告'}。不能仅依据 P 值判断实际或临床重要性。", assumption_assessment=str(result.assumption_checks or "本次方法未返回额外的假设诊断。"),
            limitations=limitations,
            result_interpretations={str(k): f"该结果的记录值为 {str(v)[:300]}；应围绕“{original_request}”并结合单位、效应量和不确定性解释。" for k, v in result.estimates.items()},
            figure_interpretations={f.id: InterpretationAgent._detailed_fallback_figure(f, original_request) for f in result.visualizations},
            conclusion="结论应结合预先设定的研究设计、效应量、不确定性、图形证据与统计假设进行审慎解释。", model="本地回退",
        )