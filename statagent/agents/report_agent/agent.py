"""
报告 Agent 模块

ReportAgent 负责生成分析报告。
主要功能：
- 从统计内存中读取推断历史
- 生成 Markdown 格式报告
- 生成 HTML 格式论文报告

注意：报告生成器不会重新计算统计量，只使用内存中已有的结果。
"""

from __future__ import annotations

from dataclasses import dataclass

from statagent.memory.statistical_memory import StatisticalMemory
from statagent.report.composer import PaperReportComposer
from statagent.i18n import decision_zh, localize_value, method_name_zh


@dataclass(slots=True)
class ReportAgent:
    """报告 Agent：生成统计分析报告"""
    
    memory: StatisticalMemory
    composer: PaperReportComposer | None = None

    def generate(self) -> str:
        """生成 Markdown 格式的文本报告"""
        lines = ["# StatAgent 中文统计分析报告", "", "本报告由 Statistical Memory 生成，未重新计算统计量。", ""]
        for record in self.memory.search():
            message = record.message
            lines.extend([f"## {message.hypothesis.id}: {message.hypothesis.description}", ""])
            lines.append(f"- 统计方法：{method_name_zh(message.method)}")
            lines.append(f"- 统计决策：{decision_zh(message.decision.value)}")
            if message.result:
                lines.append(f"- P 值：{message.result.p_value if message.result.p_value is not None else '未提供'}")
                lines.append(f"- 估计结果：{localize_value(message.result.estimates)}")
                lines.append(f"- 效应量：{localize_value(message.result.effect_size) if message.result.effect_size else '未提供'}")
                lines.append(f"- 假设诊断：{localize_value(message.result.assumption_checks) if message.result.assumption_checks else '未提供'}")
            if message.interpretation:
                lines.append(f"- 结果解释：{message.interpretation.summary}")
                for key, value in message.interpretation.result_interpretations.items():
                    lines.append(f"  - {key}：{value}")
            lines.append("")
        return "\n".join(lines)

    def generate_html(self, destination: str | None = None, title: str = "StatAgent 中文统计分析报告") -> str:
        """生成 HTML 格式的论文报告"""
        composer = self.composer or PaperReportComposer()
        html_report = composer.render_html(self.memory.search(), title)
        if destination:
            from pathlib import Path
            target = Path(destination)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(html_report, encoding="utf-8")
        return html_report