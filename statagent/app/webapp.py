"""
Web 应用模块

本模块提供 Streamlit 用户界面，实现从研究需求到中文论文报告的完整流程。
主要功能：
- CSV 数据上传
- DeepSeek API 配置
- 研究需求输入
- 假设图展示
- 候选方法展示
- 统计结果展示
- 可视化图表展示
- HTML 报告下载
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from statagent.app.pipeline import StatAgentConfig, build_default_app
from statagent.datasets import DataFrameAdapter, profile_dataframe, resolve_requested_columns
from statagent.i18n import decision_zh, localize_value, method_name_zh
from statagent.visualization.fonts import configure_chinese_font

configure_chinese_font()


st.set_page_config(page_title="StatAgent", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    .block-container {max-width: 1450px; padding-top: 1.5rem;}
    .hero {padding: 1.3rem 1.6rem; border-radius: 16px; background: linear-gradient(120deg,#102a43,#246b83); color:white; margin-bottom:1.2rem;}
    .hero h1 {margin:0; font-size:2.25rem;} .hero p {margin:.45rem 0 0; opacity:.9;}
    .stage {border-left:4px solid #2f6b8a; padding:.7rem 1rem; background:#f5f8fa; margin:.5rem 0;}
    </style>
    <div class="hero"><h1>StatAgent</h1><p>以统计假设为中心的智能推断平台 · 从研究需求到中文论文报告</p></div>
    """,
    unsafe_allow_html=True,
)

for key, value in {"data": None, "run": None, "session_id": uuid.uuid4().hex[:10]}.items():
    if key not in st.session_state:
        st.session_state[key] = value


with st.sidebar:
    st.header("运行配置")
    api_key = st.text_input("DeepSeek API 密钥", type="password", help="仅保存在当前网页会话中，不写入项目文件。")
    base_url = st.text_input("DeepSeek 接口根地址", value=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    model = st.text_input("大模型名称", value="deepseek-v4-flash")
    alpha = st.number_input("显著性水平 α", min_value=0.001, max_value=0.20, value=0.05, step=0.01, format="%.3f")
    report_title = st.text_input("报告标题", value="StatAgent 统计分析报告")
    uploaded = st.file_uploader("上传 CSV 数据", type=["csv"])
    st.divider()
    st.caption("大模型：需求解析、候选方法裁决、结果与图形解读、中文论文叙事。\n\n本地算法库：统计计算、诊断、作图、HTML 渲染。")

if uploaded is not None:
    try:
        st.session_state.data = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"CSV 读取失败：{exc}")

frame: pd.DataFrame | None = st.session_state.data
if frame is None:
    st.info("请先在左侧上传 CSV 文件。")
    st.stop()

profile = profile_dataframe(frame)
numeric_columns = frame.select_dtypes(include=[np.number]).columns.tolist()

summary_columns = st.columns(4)
summary_columns[0].metric("样本量", len(frame))
summary_columns[1].metric("变量数", len(frame.columns))
summary_columns[2].metric("数值变量", len(numeric_columns))
summary_columns[3].metric("缺失值", int(frame.isna().sum().sum()))

with st.expander("数据检查与探索性图形", expanded=False):
    preview_tab, distribution_tab, correlation_tab, missing_tab = st.tabs(["数据预览", "变量分布", "相关矩阵", "缺失模式"])
    with preview_tab:
        st.dataframe(frame.head(30), use_container_width=True)
        st.json(localize_value(profile), expanded=False)
    with distribution_tab:
        if numeric_columns:
            selected = st.selectbox("变量", numeric_columns, key="explore_distribution")
            figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))
            clean = frame[selected].dropna()
            axes[0].hist(clean, bins="auto", color="#2f6b8a", edgecolor="white")
            axes[0].set(title=f"{selected} 分布", xlabel=selected, ylabel="频数")
            axes[1].boxplot(clean, patch_artist=True)
            axes[1].set(title=f"{selected} 箱线图", ylabel=selected)
            st.pyplot(figure, use_container_width=True)
    with correlation_tab:
        if len(numeric_columns) >= 2:
            correlation = frame[numeric_columns].corr()
            figure, axis = plt.subplots(figsize=(7, 5))
            image = axis.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1)
            axis.set_xticks(range(len(correlation)), correlation.columns, rotation=45, ha="right")
            axis.set_yticks(range(len(correlation)), correlation.columns)
            figure.colorbar(image, ax=axis, label="Pearson r")
            st.pyplot(figure, use_container_width=False)
        else:
            st.info("至少需要两个数值变量。")
    with missing_tab:
        missing = frame.isna().mean().sort_values(ascending=False)
        figure, axis = plt.subplots(figsize=(8, 3.5))
        axis.bar(missing.index.astype(str), missing.values, color="#e07a5f")
        axis.set(title="变量缺失比例", ylabel="缺失比例", ylim=(0, max(1.0, float(missing.max()) * 1.1)))
        axis.tick_params(axis="x", rotation=45)
        st.pyplot(figure, use_container_width=True)

st.subheader("研究需求")
requirement = st.text_area(
    "用自然语言描述研究问题、变量角色、研究设计和希望检验的假设",
    placeholder="例如：比较 treatment 与 control 两组患者术后评分是否存在差异，并报告效应量、置信信息、假设检验和图形解释。",
    height=120,
)
requested_preview = resolve_requested_columns(requirement, profile) if requirement.strip() else []
if requested_preview:
    st.success("已从研究需求中锁定变量：" + "、".join(requested_preview))
    all_columns = [str(c["name"]) for c in profile.get("columns", [])]
    confirmed_columns = st.multiselect(
        "请确认要分析的变量（可手动添加或删除）：",
        options=all_columns,
        default=requested_preview,
        key="confirmed_variables"
    )
    if confirmed_columns:
        st.success(f"已确认分析变量：{'、'.join(confirmed_columns)}")
elif requirement.strip():
    all_columns = [str(c["name"]) for c in profile.get("columns", [])]
    st.info("尚未在研究需求中识别到精确列名。请从下方选择要分析的变量，或直接输入变量名。")
    confirmed_columns = st.multiselect(
        "请选择要分析的变量：",
        options=all_columns,
        key="confirmed_variables"
    )
else:
    confirmed_columns = []

st.markdown(
    """
    <div class="stage"><strong>大模型语义链：</strong> ① 形成统计假设 → ② 在分层检索得到的候选方法卡中裁决 → ③ 映射数据列 → ④ 逐项解读数值与图片 → ⑤ 综合全部证据生成中文论文摘要与总结</div>
    """,
    unsafe_allow_html=True,
)

if st.button("开始假设驱动分析", type="primary", use_container_width=True):
    if not requirement.strip():
        st.error("请先描述研究需求。")
    else:
        if not api_key:
            st.warning("未提供 DeepSeek 密钥：系统将使用本地回退匹配与模板解释，无法发挥完整大模型能力。")
        cleanup_on_startup = st.sidebar.checkbox("启动时清理旧文件", value=True, help="每次启动时自动删除旧的 artifacts 目录")
        use_temp_dir = st.sidebar.checkbox("使用临时目录", value=False, help="将文件保存到系统临时目录（重启后可能被清理）")
        artifact_dir = Path(tempfile.gettempdir()) / "statagent_artifacts" / st.session_state.session_id if use_temp_dir else Path("artifacts") / st.session_state.session_id
        try:
            with st.status("正在执行证据循环…", expanded=True) as status:
                st.write("解析研究需求并建立统计假设关系图")
                config = StatAgentConfig(
                    alpha=alpha,
                    deepseek_api_key=api_key or None,
                    deepseek_base_url=base_url,
                    deepseek_model=model,
                    memory_path=artifact_dir / "statistical_memory.json" if not use_temp_dir else None,
                    artifact_dir=artifact_dir,
                    cleanup_on_startup=cleanup_on_startup,
                    use_temp_dir=False,
                )
                app = build_default_app(config)
                adapter = DataFrameAdapter(frame)
                st.write("检索方法卡，并由 DeepSeek 在候选集中完成统计方法匹配")
                markdown_report = app.run(
                    requirement,
                    adapter.execution_input,
                    context={"dataset_profile": profile, "report_title": report_title, "confirmed_columns": confirmed_columns},
                    exploratory_data=frame,
                )
                st.write("逐项解释统计结果和可视化图形，并综合全部证据生成中文论文报告")
                html_report = app.report_agent.generate_html(
                    str(artifact_dir / "report.html"), title=report_title
                )
                records = app.report_agent.memory.search()
                fallback_roles = [
                    record.message.hypothesis.id
                    for record in records
                    if record.message.hypothesis.metadata.get("llm_parser") == "local-fallback"
                    or (record.message.interpretation and record.message.interpretation.model == "本地回退")
                    or (record.message.method and "本地回退" in record.message.method.get("match", {}).get("rationale", ""))
                ]
                if fallback_roles:
                    st.warning("以下假设的一个或多个大模型阶段使用了回退路径：" + ", ".join(sorted(set(fallback_roles))) + "。请检查密钥、模型名和接口网关配置。")
                st.session_state.run = {
                    "markdown": markdown_report,
                    "html": html_report,
                    "records": records,
                    "graph": app.hypothesis_manager.graph.to_mermaid(),
                }
                status.update(label="分析完成", state="complete", expanded=False)
        except Exception as exc:
            st.exception(exc)

run = st.session_state.run
if run:
    st.divider()
    st.subheader("分析结果")
    graph_tab, method_tab, result_tab, figure_tab, report_tab = st.tabs(["假设关系图", "方法匹配", "统计结果", "探索性与推断图形", "中文论文报告"])

    with graph_tab:
        st.code(run["graph"], language="mermaid")

    with method_tab:
        for record in run["records"]:
            message = record.message
            match = message.method.get("match", {}) if message.method else {}
            with st.container(border=True):
                st.markdown(f"#### {message.hypothesis.id} · {method_name_zh(message.method)}")
                left, right = st.columns(2)
                left.metric("检索排名", message.method.get("rank", "-") if message.method else "-")
                right.metric("方法编号", message.method.get("method_id", "-") if message.method else "-")
                st.write("**选择理由：**", match.get("rationale", "本地检索排序"))
                st.write("**数据映射：**")
                st.json(localize_value(match.get("data_mapping", {})), expanded=True)
                st.write("**锁定变量：**", "、".join(match.get("locked_columns", [])) or "未锁定")
                st.write("**映射审计：**", match.get("mapping_audit", "未提供"))
                st.caption("候选方法编号：" + ", ".join(message.method.get("candidate_method_ids", [])))

    with result_tab:
        for record in run["records"]:
            message = record.message
            with st.expander(f"{message.hypothesis.id} · {method_name_zh(message.method)}", expanded=True):
                result = message.result
                metric_columns = st.columns(2)
                metric_columns[0].metric("统计决策", decision_zh(message.decision.value))
                metric_columns[1].metric("P 值", f"{result.p_value:.4g}" if result and result.p_value is not None else "未提供")
                st.write("**估计结果**")
                st.json(localize_value(result.estimates) if result else {}, expanded=False)
                st.write("效应量", localize_value(result.effect_size) if result and result.effect_size else "未提供")
                st.write("**假设诊断**")
                st.json(localize_value(result.assumption_checks) if result else [], expanded=False)
                if message.interpretation and message.interpretation.result_interpretations:
                    st.write("**大模型逐项统计结果解释**")
                    for key, explanation in message.interpretation.result_interpretations.items():
                        st.info(f"{key}：{explanation}")

    with figure_tab:
        for record in run["records"]:
            message = record.message
            st.markdown(f"### {message.hypothesis.id} · 探索性分析与方法专属图形")
            figures = message.result.visualizations if message.result else []
            for index in range(0, len(figures), 2):
                columns = st.columns(2)
                for column, figure in zip(columns, figures[index:index + 2]):
                    with column:
                        st.image(figure.path, caption=figure.title, use_container_width=True)
                        interpretation = message.interpretation.figure_interpretations.get(figure.id, figure.caption) if message.interpretation else figure.caption
                        st.info(interpretation)
            if message.interpretation:
                st.markdown("#### 大模型对本项分析的综合解释")
                st.write(message.interpretation.statistical_interpretation)
                st.write(message.interpretation.practical_interpretation)
                st.write("**假设条件：**", message.interpretation.assumption_assessment)
                st.write("**局限：**", "；".join(message.interpretation.limitations))

    with report_tab:
        st.download_button("下载中文 HTML 论文报告", run["html"], file_name="statagent_中文统计报告.html", mime="text/html", type="primary")
        components.html(run["html"], height=900, scrolling=True)

st.caption("StatAgent · 大模型负责语义决策、逐图解释与中文论文叙事；已注册的本地统计 API 负责数值计算和可视化。")