"""
报告渲染模块

本模块提供报告生成和渲染功能。
核心类：
- PaperReportComposer: 论文报告合成器，使用 LLM 生成论文各章节
- PaperSections: 论文章节结构

ReportAgent 消费 Statistical Memory 中的推断历史来生成报告。
"""

from statagent.report.composer import PaperReportComposer, PaperSections

__all__ = ["PaperReportComposer", "PaperSections"]