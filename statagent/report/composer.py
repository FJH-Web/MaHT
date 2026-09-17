"""
论文报告合成器模块

本模块使用 LLM 生成论文各章节内容，并提供确定性的 HTML 渲染器。
主要功能：
- 生成摘要
- 生成引言
- 生成方法部分
- 生成结果部分
- 生成讨论部分
- 生成结论和局限性
- 渲染 HTML 报告
"""

from __future__ import annotations

import base64
import html
import json
from dataclasses import dataclass
from pathlib import Path

from statagent.llm.deepseek import DeepSeekClient, LLMError
from statagent.memory.statistical_memory import MemoryRecord, StatisticalMemory
from statagent.i18n import decision_zh, localize_value, method_name_zh


@dataclass(slots=True)
class PaperSections:
    """论文章节结构"""

    title: str
    abstract: str
    introduction: str
    methods: str
    results: str
    discussion: str
    conclusion: str
    limitations: str
    decision_graph: str = ""
    assumption_audit: str = ""


@dataclass(slots=True)
class PaperReportComposer:
    """论文报告合成器"""
    
    client: DeepSeekClient | None = None
    allow_fallback: bool = True

    def compose(self, records: list[MemoryRecord], title: str = "StatAgent 中文统计分析报告") -> PaperSections:
        """合成论文各章节"""
        constraints = []
        for record in records:
            message = record.message
            context = message.hypothesis.metadata.get("analysis_context", {})
            match = (message.method or {}).get("match", {})
            constraints.append({
                "hypothesis_id": message.hypothesis.id,
                "original_request": context.get("original_request", message.hypothesis.description),
                "locked_columns": context.get("requested_columns", []),
                "final_data_mapping": match.get("data_mapping", {}),
                "mapping_audit": match.get("mapping_audit", ""),
            })
        payload = {
            "title": title,
            "research_constraints": constraints,
            "analyses": [
                {
                    **record.message.to_dict(),
                    "中文方法名称": method_name_zh(record.message.method),
                    "中文统计决策": decision_zh(record.message.decision.value),
                    "中文估计结果": localize_value(record.message.result.estimates) if record.message.result else {},
                }
                for record in records
            ],
        }
        if self.client and self.client.configured:
            try:
                data = self.client.complete_json(self._system_prompt(), json.dumps(payload, ensure_ascii=False))
                return PaperSections(title=title, **{field: str(data.get(field, "")) for field in ("abstract", "introduction", "methods", "results", "discussion", "conclusion", "limitations")})
            except LLMError:
                if not self.allow_fallback:
                    raise
        return self._fallback(records, title)

    @staticmethod
    def _system_prompt() -> str:
        return (
            "你是中文统计学论文的综合写作模块。只能使用给定的 Statistical Memory 记录，不得重新计算、"
            "修改或编造任何数值。必须综合所有假设、所有统计结果、所有单图解释和探索性分析，形成一篇完整"
            "而连贯的中文论文报告。返回 JSON 字段 abstract、introduction、methods、results、discussion、"
            "conclusion、limitations，所有字段内容必须为简体中文。abstract 必须概括研究目的、数据、方法、"
            "主要数值结果和总体结论；results 必须逐项报告主要统计量、P 值、效应量和图形发现；discussion "
            "必须跨分析综合一致与矛盾证据；conclusion 必须给出面向全部结果的总体总结，而不是重复单项结论。"
            "使用审慎的中文期刊论文风格，区分探索性与验证性结论，引用图题，不返回 HTML 或 Markdown 代码块。"
            "research_constraints 中的 original_request 是全文最高优先级约束；所有内容只能围绕 locked_columns "
            "与 final_data_mapping 中实际分析的变量展开。不得把索引列或未点名变量写入结论，不得迁就错误映射而改写研究目的。"
        )

    def _fallback(self, records: list[MemoryRecord], title: str) -> PaperSections:
        """使用本地规则生成论文各章节"""
        if not records:
            return PaperSections(
                title=title,
                abstract="本报告暂无统计分析记录。",
                introduction="本报告暂无研究背景介绍。",
                methods="本报告暂无方法描述。",
                results="尚无已完成的统计分析记录。",
                discussion="暂无可供综合讨论的结果。",
                conclusion="请先执行统计分析。",
                limitations="当前记录中未提供其他局限性。",
            )

        analyses = []
        all_methods = []
        all_decisions = []
        all_results = []
        all_limitations = []
        all_variables = set()

        for record in records:
            message = record.message
            context = message.hypothesis.metadata.get("analysis_context", {})
            match = (message.method or {}).get("match", {})

            original_request = context.get("original_request", message.hypothesis.description)
            data_mapping = match.get("data_mapping", {})

            for key, mapping in data_mapping.items():
                if isinstance(mapping, dict) and mapping.get("kind") == "column":
                    col = mapping.get("column")
                    if col:
                        all_variables.add(col)

            interpretation = message.interpretation
            method_name = method_name_zh(message.method)
            decision = decision_zh(message.decision.value) if message.decision else "未知"

            all_methods.append(method_name)
            all_decisions.append(decision)

            result_text = f"假设：{message.hypothesis.description}。方法：{method_name}。决策：{decision}。"
            if message.result:
                estimates = message.result.estimates or {}
                if estimates:
                    est_parts = []
                    for k, v in estimates.items():
                        if v is not None:
                            est_parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
                    if est_parts:
                        result_text += " 统计量：" + "，".join(est_parts) + "。"
            if interpretation and interpretation.result_interpretations:
                for key, value in interpretation.result_interpretations.items():
                    result_text += f" {key}: {value}。"
            all_results.append(result_text)

            if interpretation:
                if interpretation.conclusion:
                    analyses.append(interpretation.conclusion)
                if interpretation.limitations:
                    all_limitations.extend(interpretation.limitations)

            pipeline_info = message.hypothesis.metadata.get("pipeline_assumption_results", [])
            if pipeline_info:
                pipeline_summary = message.hypothesis.metadata.get("pipeline_summary", "")
                all_results.append(f"流水线执行：{pipeline_summary}")
                for pi in pipeline_info:
                    all_results.append(f"  - {pi.get('test_method', '假设检验')}: p={pi.get('p_value')}, {pi.get('conclusion', '')}")

        variables_text = "、".join(sorted(all_variables)) if all_variables else "数据集中的变量"
        methods_text = "、".join(dict.fromkeys(all_methods))

        abstract = self._generate_abstract(records, variables_text, methods_text)
        introduction = self._generate_introduction(records, variables_text)
        methods = self._generate_methods(records, methods_text)
        results = "\n\n".join(all_results) if all_results else "尚无已完成的统计分析记录。"
        discussion = "\n\n".join(analyses) if analyses else "暂无可供综合讨论的结果。"
        conclusion = self._generate_conclusion(records, all_decisions)
        limitations = "\n".join(dict.fromkeys(all_limitations)) if all_limitations else "当前记录中未提供其他局限性。"

        return PaperSections(
            title=title,
            abstract=abstract,
            introduction=introduction,
            methods=methods,
            results=results,
            discussion=discussion,
            conclusion=conclusion,
            limitations=limitations,
            decision_graph=self._generate_decision_graph(records),
            assumption_audit=self._generate_assumption_audit(records),
        )

    def _generate_abstract(self, records: list[MemoryRecord], variables: str, methods: str) -> str:
        """动态生成摘要"""
        if not records:
            return "本报告暂无统计分析记录。"

        main_records = [r for r in records if (getattr(r.message.method, 'code', None) or r.message.method.get('code', '')) not in ["DIST001", "DIST002"]]
        if not main_records:
            main_records = records

        study_purpose = "分析变量间差异或关联关系"
        if main_records:
            first_hypothesis = main_records[0].message.hypothesis.description
            if "差异" in first_hypothesis:
                study_purpose = "比较不同组别之间是否存在显著差异"
            elif "相关" in first_hypothesis:
                study_purpose = "探索变量之间的相关性"
            elif "分布" in first_hypothesis:
                study_purpose = "检验数据分布特征"

        significance_count = sum(1 for r in records if r.message.decision and r.message.decision.value == "reject_null")
        total_count = len(records)

        abstract = f"本研究旨在{study_purpose}。分析基于{variables}，采用{methods}等统计方法进行处理。"
        if total_count > 0:
            abstract += f"共执行{total_count}项统计检验，其中{significance_count}项达到显著水平。"
        if significance_count > 0:
            abstract += "研究结果表明存在统计显著的差异或关联。"
        else:
            abstract += "研究结果表明未发现统计显著的差异或关联。"

        return abstract

    def _generate_introduction(self, records: list[MemoryRecord], variables: str) -> str:
        """动态生成引言"""
        if not records:
            return "本报告暂无研究背景介绍。"

        main_records = [r for r in records if (getattr(r.message.method, 'code', None) or r.message.method.get('code', '')) not in ["DIST001", "DIST002"]]
        if not main_records:
            main_records = records

        study_purpose = "统计分析"
        if main_records:
            first_hypothesis = main_records[0].message.hypothesis.description
            if "差异" in first_hypothesis:
                study_purpose = "比较不同组别之间是否存在显著差异"
            elif "相关" in first_hypothesis:
                study_purpose = "探索变量之间的相关性"
            elif "分布" in first_hypothesis:
                study_purpose = "检验数据分布特征"

        introduction = f"本研究围绕{study_purpose}这一核心问题展开统计分析。"
        introduction += f"数据分析聚焦于{variables}，"
        introduction += "旨在通过假设检验和统计推断探索变量间的潜在关联及其统计学意义。"

        return introduction

    def _generate_methods(self, records: list[MemoryRecord], methods: str) -> str:
        """动态生成方法部分"""
        if not records:
            return "本报告暂无方法描述。"

        methods_section = f"本研究采用以下统计方法进行分析：{methods}。"
        methods_section += "统计方法由分层检索系统根据研究假设自动确定，并通过注册的本地统计 API 执行。"

        has_pipeline = any(
            r.message.hypothesis.metadata.get("pipeline_assumption_results")
            for r in records
        )
        if has_pipeline:
            methods_section += " 在执行参数检验前，系统自动检验了相应的假设条件（如正态性、方差齐性），并在假设不满足时自动切换至非参数检验方法。"

        return methods_section

    def _generate_conclusion(self, records: list[MemoryRecord], decisions: list[str]) -> str:
        """动态生成结论"""
        if not records:
            return "请先执行统计分析。"

        reject_count = decisions.count("拒绝原假设")
        total = len(decisions)

        if total == 0:
            conclusion = "基于统计分析结果，当前数据未能提供足够的统计证据支持研究假设。"
        elif reject_count > 0:
            conclusion = f"基于{total}项统计检验的结果，共有{reject_count}项达到显著水平，表明研究假设在统计意义上获得部分支持。"
        else:
            conclusion = f"基于{total}项统计检验的结果，未发现统计显著的差异或关联，研究假设未能获得足够证据支持。"

        return conclusion

    def _generate_decision_graph(self, records: list[MemoryRecord]) -> str:
        """生成统计决策流程图 (Mermaid)"""
        if not records:
            return ""

        import re

        def safe_text(text: str) -> str:
            text = str(text)
            text = re.sub(r'[*#\[\]{}|\\<>"\'`$]', '', text)
            text = text.replace('\n', ' ').replace('\r', ' ')
            return text[:20]

        graphs = []
        for i, record in enumerate(records):
            message = record.message
            h_text = safe_text(message.hypothesis.description)
            m_text = safe_text(method_name_zh(message.method))
            decision = decision_zh(message.decision.value) if message.decision else "未知"

            result_parts = []
            if message.result:
                if message.result.p_value is not None:
                    result_parts.append(f"p={message.result.p_value:.3f}")
                if message.result.effect_size:
                    es = message.result.effect_size
                    if isinstance(es.get('value'), (int, float)):
                        result_parts.append(f"{es.get('name','d')}={es['value']:.2f}")
            
            result_str = ' '.join(result_parts) if result_parts else decision

            decision_text = "SUPPORT" if "支持" in decision else "REJECT"

            graph = f"""graph TD
    Q{i+1}["Q{i+1}"] --> M{i+1}
    M{i+1}["{m_text}"] --> R{i+1}
    R{i+1}["{result_str}"] --> C{i+1}
    C{i+1}["{decision_text}"]"""
            graphs.append(graph)

        return "\n\n".join(graphs)

    def _generate_assumption_audit(self, records: list[MemoryRecord]) -> str:
        """生成假设审计表格"""
        if not records:
            return ""

        tables = []
        for i, record in enumerate(records):
            message = record.message
            method_name = method_name_zh(message.method)
            assumption_checks = message.result.assumption_checks if message.result else []

            if not assumption_checks:
                continue

            header = f"### {method_name} 假设审计\n\n| 假设类型 | 检验方法 | 数值/统计量 | 结果 |\n|---------|---------|------------|------|"
            rows = []

            for check in assumption_checks:
                name = check.get("name", "未知")
                check_type = check.get("type", "unknown")
                passed = check.get("passed", False)

                if check_type == "normality":
                    p_val = check.get("p_value")
                    stat = check.get("statistic")
                    status = "✅ PASS" if passed else "❌ FAIL"
                    value_text = f"p={p_val:.4f}" if p_val is not None else f"W={stat:.4f}" if stat else "N/A"
                elif check_type == "variance":
                    p_val = check.get("p_value")
                    status = "✅ PASS" if passed else "❌ FAIL"
                    value_text = f"p={p_val:.4f}" if p_val is not None else "N/A"
                elif check_type == "independence":
                    stat = check.get("statistic")
                    status = "✅ PASS" if passed else "⚠️ WARNING"
                    value_text = f"DW={stat:.4f}" if stat else "N/A"
                elif check_type == "missing":
                    missing = check.get("missing", 0)
                    total = check.get("total", 0)
                    ratio = check.get("ratio", 0)
                    status = "✅ PASS" if passed else "⚠️ WARNING"
                    value_text = f"{missing}/{total} ({ratio*100:.1f}%)"
                elif check_type == "outlier":
                    count = check.get("outlier_count", 0)
                    total = check.get("total", 0)
                    ratio = check.get("ratio", 0)
                    status = "✅ PASS" if passed else "⚠️ WARNING"
                    value_text = f"{count}/{total} ({ratio*100:.1f}%)"
                elif check_type == "influential":
                    count = check.get("influential_count", 0)
                    total = check.get("total", 0)
                    ratio = check.get("ratio", 0)
                    status = "✅ PASS" if passed else "⚠️ WARNING"
                    value_text = f"{count}/{total} ({ratio*100:.1f}%)"
                else:
                    status = "✅ PASS" if passed else "❌ FAIL"
                    value_text = "N/A"

                rows.append(f"| {name} | {name} | {value_text} | {status} |")

            if rows:
                tables.append(header + "\n" + "\n".join(rows))

        return "\n\n".join(tables) if tables else "本分析未执行假设检验。"

    def render_html(self, records: list[MemoryRecord], title: str = "StatAgent 中文统计分析报告") -> str:
        sections = self.compose(records, title)
        blocks = [
            ("摘要", sections.abstract),
            ("引言", sections.introduction),
            ("研究方法", sections.methods),
            ("统计结果与图形发现", sections.results),
            ("综合讨论", sections.discussion),
            ("总体总结", sections.conclusion),
            ("局限性与可复现性", sections.limitations),
        ]

        css = """
        <style>
        .meta { color: #666; font-size: 0.9em; margin-bottom: 20px; }
        .assumption-audit table { border-collapse: collapse; width: 100%; margin: 10px 0; }
        .assumption-audit th, .assumption-audit td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        .assumption-audit th { background-color: #f5f5f5; }
        .result-stats { background: #f9f9f9; padding: 10px; border-radius: 5px; margin: 10px 0; }
        .result-stats .stat-label { font-weight: bold; color: #333; }
        .result-stats .stat-value { color: #0066cc; }
        .result-stats .effect-size { color: #008800; }
        .result-stats .ci { color: #880088; }
        </style>
        """

        body = [f"<h1>{html.escape(sections.title)}</h1>", "<p class='meta'>本报告由 Statistical Memory 生成；大模型仅负责中文解释与论文叙事，不重新计算或修改统计数值。</p>", css]

        if sections.decision_graph:
            body.append("<h2>统计决策流程图</h2>")
            body.append(f"<pre class='mermaid'>{html.escape(sections.decision_graph)}</pre>")

        if sections.assumption_audit:
            body.append("<h2>假设审计</h2>")
            body.append(f"<div class='assumption-audit'>{html.escape(sections.assumption_audit).replace(chr(10), '<br>')}</div>")

        body.extend(f"<h2>{html.escape(name)}</h2><p>{html.escape(text).replace(chr(10), '<br>')}</p>" for name, text in blocks)
        for record in records:
            message = record.message
            if not message.result:
                continue
            body.append(f"<h2>统计图形与逐图解释——{html.escape(message.hypothesis.id)}</h2>")
            body.append("<div class='figure-grid'>")
            for figure in message.result.visualizations:
                image = self._data_uri(figure.path)
                figure_text = message.interpretation.figure_interpretations.get(figure.id, figure.caption) if message.interpretation else figure.caption
                body.append(f"<figure class='figure-card'><img src='{image}' alt='{html.escape(figure.title)}'><figcaption><strong>{html.escape(figure.title)}</strong><br>{html.escape(figure_text)}</figcaption></figure>")
            body.append("</div>")
        template_path = Path(__file__).parent / "templates" / "paper.html"
        template = template_path.read_text(encoding="utf-8")
        return template.replace("{{TITLE}}", html.escape(sections.title)).replace("{{BODY}}", "".join(body))

    @staticmethod
    def _data_uri(path: str) -> str:
        file_path = Path(path)
        if not file_path.exists():
            return ""
        mime = "image/svg+xml" if file_path.suffix.lower() == ".svg" else "image/png"
        return f"data:{mime};base64," + base64.b64encode(file_path.read_bytes()).decode("ascii")