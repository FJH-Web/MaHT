"""数据集级探索性分析图谱，所有图形均由本地确定性代码生成。"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from statagent.protocol.statistical_object import VisualizationArtifact
from statagent.visualization.fonts import configure_chinese_font

configure_chinese_font()


@dataclass(slots=True)
class ExploratoryVisualizationEngine:
    output_dir: Path
    dpi: int = 80
    max_numeric: int = 6
    max_categorical: int = 4

    def render(self, frame: pd.DataFrame, requested_columns: list[str] | None = None) -> list[VisualizationArtifact]:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if requested_columns:
            available_cols = [c for c in requested_columns if c in frame.columns]
            if available_cols:
                frame = frame[available_cols]
                self.max_numeric = min(self.max_numeric, len(frame.select_dtypes(include=[np.number]).columns))
                self.max_categorical = min(self.max_categorical, len([c for c in frame.columns if c not in frame.select_dtypes(include=[np.number]).columns]))
        
        numeric = frame.select_dtypes(include=[np.number]).columns.tolist()[: self.max_numeric]
        categorical = [column for column in frame.columns if column not in numeric][: self.max_categorical]
        scatter_vars = numeric[:3] if len(numeric) >= 3 else numeric
        recipes: list[tuple[str, Callable[[], tuple[plt.Figure, str, str, dict[str, Any]]]]] = [
            ("eda_variable_types", lambda: self._variable_types(frame)),
            ("eda_numeric_distribution", lambda: self._numeric_distributions(frame, numeric)),
            ("eda_boxplot", lambda: self._boxplots(frame, numeric)),
            ("eda_violin", lambda: self._violins(frame, numeric)),
            ("eda_qq", lambda: self._qq_grid(frame, numeric)),
            ("eda_ecdf", lambda: self._ecdf(frame, numeric)),
            ("eda_correlation", lambda: self._correlation(frame, numeric)),
            ("eda_scatter_matrix", lambda: self._scatter_matrix(frame, scatter_vars)),
            ("eda_missing_rate", lambda: self._missing_rate(frame)),
            ("eda_missing_matrix", lambda: self._missing_matrix(frame)),
            ("eda_categorical_frequency", lambda: self._categorical_frequency(frame, categorical)),
            ("eda_outlier_rate", lambda: self._outlier_rate(frame, numeric)),
        ]
        artifacts: list[VisualizationArtifact] = []
        for kind, recipe in recipes:
            try:
                figure, title, caption, semantic = recipe()
                artifact_id = f"EDA-{kind}-{uuid.uuid4().hex[:8]}"
                path = self.output_dir / f"{artifact_id}.png"
                figure.tight_layout()
                figure.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
                plt.close(figure)
                artifacts.append(VisualizationArtifact(artifact_id, title, kind, str(path.resolve()), caption, self._json(semantic)))
            except (ValueError, KeyError, IndexError, TypeError, MemoryError, OSError):
                continue
        return artifacts

    @staticmethod
    def _grid(count: int, width: float = 5.0, height: float = 3.6):
        if count < 1:
            raise ValueError("没有可绘制变量")
        columns = min(3, count)
        rows = int(np.ceil(count / columns))
        figure, axes = plt.subplots(rows, columns, figsize=(width * columns, height * rows), squeeze=False)
        return figure, axes.ravel()

    def _variable_types(self, frame: pd.DataFrame):
        counts = {
            "数值型": len(frame.select_dtypes(include=[np.number]).columns),
            "分类型/文本型": len(frame.select_dtypes(exclude=[np.number, "datetime"]).columns),
            "日期时间型": len(frame.select_dtypes(include=["datetime"]).columns),
        }
        figure, axis = plt.subplots(figsize=(7, 4.3))
        axis.bar(counts.keys(), counts.values(), color=["#2f6b8a", "#e07a5f", "#81b29a"])
        axis.set(title="变量类型构成", xlabel="变量类型", ylabel="变量数量")
        axis.grid(axis="y", alpha=.2)
        return figure, "变量类型构成", "展示数据集中不同变量类型的数量，为后续方法选择提供依据。", {"变量类型计数": counts, "样本量": len(frame), "变量总数": len(frame.columns)}

    def _numeric_distributions(self, frame: pd.DataFrame, numeric: list[str]):
        figure, axes = self._grid(len(numeric))
        summaries = {}
        for axis, column in zip(axes, numeric):
            values = frame[column].dropna().astype(float)
            axis.hist(values, bins="auto", density=True, color="#2f6b8a", alpha=.72, edgecolor="white")
            if len(values) > 2 and values.nunique() > 1:
                grid = np.linspace(values.min(), values.max(), 160)
                axis.plot(grid, stats.gaussian_kde(values)(grid), color="#d1495b", linewidth=2, label="核密度")
                axis.legend()
            axis.set(title=f"{column}：分布", xlabel="取值", ylabel="密度")
            summaries[column] = self._summary(values.to_numpy())
        for axis in axes[len(numeric):]: axis.remove()
        return figure, "数值变量分布与核密度", "直方图结合核密度曲线，用于检查偏态、多峰和异常分布。", summaries

    def _boxplots(self, frame: pd.DataFrame, numeric: list[str]):
        figure, axes = self._grid(len(numeric))
        outliers = {}
        for axis, column in zip(axes, numeric):
            values = frame[column].dropna().astype(float).to_numpy()
            axis.boxplot(values, patch_artist=True, boxprops={"facecolor": "#8ecae6"})
            axis.set(title=f"{column}：箱线图", ylabel="取值", xticks=[])
            q1, q3 = np.quantile(values, [.25, .75]); iqr = q3 - q1
            outliers[column] = int(np.sum((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)))
        for axis in axes[len(numeric):]: axis.remove()
        return figure, "数值变量箱线图", "展示中位数、四分位范围与潜在离群点。", {"潜在离群点数量": outliers}

    def _violins(self, frame: pd.DataFrame, numeric: list[str]):
        usable = [column for column in numeric if frame[column].dropna().nunique() > 1]
        if not usable: raise ValueError("无可用连续变量")
        figure, axis = plt.subplots(figsize=(max(8, len(usable) * 1.25), 4.8))
        values = [frame[column].dropna().astype(float).to_numpy() for column in usable]
        parts = axis.violinplot(values, showmeans=True, showmedians=True)
        for body in parts["bodies"]: body.set_facecolor("#81b29a"); body.set_alpha(.7)
        axis.set(title="数值变量提琴图", xlabel="变量", ylabel="取值", xticks=np.arange(1, len(usable) + 1), xticklabels=usable)
        axis.tick_params(axis="x", rotation=30)
        return figure, "数值变量提琴图", "同时呈现分布形状、均值与中位数，便于发现偏态和多峰。", {column: self._summary(value) for column, value in zip(usable, values)}

    def _qq_grid(self, frame: pd.DataFrame, numeric: list[str]):
        figure, axes = self._grid(len(numeric))
        checks = {}
        for axis, column in zip(axes, numeric):
            values = frame[column].dropna().astype(float).to_numpy()
            stats.probplot(values, dist="norm", plot=axis)
            axis.set_title(f"{column}：正态 Q–Q 图")
            axis.set_xlabel("理论分位数")
            axis.set_ylabel("样本分位数")
            if 3 <= len(values) <= 5000:
                statistic, p_value = stats.shapiro(values)
                checks[column] = {"Shapiro-Wilk统计量": float(statistic), "P值": float(p_value)}
        for axis in axes[len(numeric):]: axis.remove()
        return figure, "正态 Q–Q 诊断", "比较样本分位数与理论正态分位数，辅助判断正态性。", checks

    def _ecdf(self, frame: pd.DataFrame, numeric: list[str]):
        figure, axis = plt.subplots(figsize=(8, 4.8))
        ranges = {}
        for column in numeric:
            values = np.sort(frame[column].dropna().astype(float).to_numpy())
            axis.step(values, np.arange(1, len(values) + 1) / len(values), where="post", label=column)
            ranges[column] = [float(values.min()), float(values.max())]
        axis.set(title="经验累积分布函数", xlabel="取值", ylabel="累计概率")
        axis.legend(ncol=2); axis.grid(alpha=.2)
        return figure, "经验累积分布函数", "比较各数值变量在不同取值处的累计概率。", {"取值范围": ranges}

    def _correlation(self, frame: pd.DataFrame, numeric: list[str]):
        if len(numeric) < 2: raise ValueError("数值变量不足")
        correlation = frame[numeric].corr()
        figure, axis = plt.subplots(figsize=(max(6, len(numeric) * .9), max(5, len(numeric) * .75)))
        image = axis.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1)
        axis.set(title="数值变量相关系数热图", xticks=range(len(numeric)), yticks=range(len(numeric)), xticklabels=numeric, yticklabels=numeric)
        axis.tick_params(axis="x", rotation=45)
        for row in range(len(numeric)):
            for column in range(len(numeric)):
                axis.text(column, row, f"{correlation.iloc[row, column]:.2f}", ha="center", va="center", fontsize=8)
        figure.colorbar(image, ax=axis, label="Pearson 相关系数")
        return figure, "相关系数热图", "展示数值变量之间的线性相关方向与强度。", {"相关矩阵": correlation.to_dict()}

    def _scatter_matrix(self, frame: pd.DataFrame, numeric: list[str]):
        if len(numeric) < 2: raise ValueError("数值变量不足")
        clean = frame[numeric].dropna()
        if len(clean) > 1000: clean = clean.sample(1000, random_state=2026)
        axes = pd.plotting.scatter_matrix(clean, figsize=(3 * len(numeric), 3 * len(numeric)), diagonal="hist", alpha=.45, color="#2f6b8a")
        figure = axes[0, 0].figure
        figure.suptitle("数值变量散点矩阵", y=1.01, fontweight="bold")
        return figure, "数值变量散点矩阵", "对角线为单变量分布，非对角线用于发现线性、非线性与聚类关系。", {"变量": numeric, "展示样本数": len(clean)}

    def _missing_rate(self, frame: pd.DataFrame):
        rates = frame.isna().mean().sort_values(ascending=False)
        figure, axis = plt.subplots(figsize=(max(8, len(rates) * .65), 4.5))
        axis.bar(rates.index.astype(str), rates.values, color="#e07a5f")
        axis.set(title="各变量缺失比例", xlabel="变量", ylabel="缺失比例", ylim=(0, max(.05, min(1.0, float(rates.max()) * 1.15))))
        axis.tick_params(axis="x", rotation=45)
        return figure, "变量缺失比例", "量化各变量的缺失程度，为缺失机制判断和处理策略提供依据。", {"缺失比例": rates.to_dict(), "缺失总数": int(frame.isna().sum().sum())}

    def _missing_matrix(self, frame: pd.DataFrame):
        sample = frame.iloc[: min(500, len(frame))]
        figure, axis = plt.subplots(figsize=(max(8, len(frame.columns) * .65), 5))
        image = axis.imshow(sample.isna().to_numpy().T, aspect="auto", interpolation="nearest", cmap="Greys")
        axis.set(title="缺失模式矩阵", xlabel="样本序号", ylabel="变量", yticks=range(len(frame.columns)), yticklabels=frame.columns)
        figure.colorbar(image, ax=axis, label="缺失标记（1=缺失）")
        return figure, "缺失模式矩阵", "展示缺失值是否在变量或样本之间呈现共同模式。", {"展示样本数": len(sample), "变量缺失数": frame.isna().sum().to_dict()}

    def _categorical_frequency(self, frame: pd.DataFrame, categorical: list[str]):
        if not categorical: raise ValueError("没有分类变量")
        figure, axes = self._grid(len(categorical))
        frequencies = {}
        for axis, column in zip(axes, categorical):
            counts = frame[column].fillna("<缺失>").astype(str).value_counts().head(12)
            axis.barh(np.arange(len(counts)), counts.values, color="#81b29a")
            axis.set(title=f"{column}：类别频数", xlabel="频数", yticks=np.arange(len(counts)), yticklabels=counts.index)
            frequencies[column] = counts.to_dict()
        for axis in axes[len(categorical):]: axis.remove()
        return figure, "分类变量频数", "展示主要类别的样本数量与类别不平衡情况。", frequencies

    def _outlier_rate(self, frame: pd.DataFrame, numeric: list[str]):
        rates = {}
        for column in numeric:
            values = frame[column].dropna().astype(float).to_numpy(); q1, q3 = np.quantile(values, [.25, .75]); iqr = q3 - q1
            rates[column] = float(np.mean((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)))
        figure, axis = plt.subplots(figsize=(max(8, len(rates) * .8), 4.3))
        axis.bar(rates.keys(), rates.values(), color="#d1495b")
        axis.set(title="潜在离群值比例", xlabel="变量", ylabel="离群比例")
        axis.tick_params(axis="x", rotation=35)
        return figure, "潜在离群值比例", "采用 1.5×IQR 规则识别潜在离群值，仅作为诊断而非自动删除依据。", {"离群比例": rates}

    @staticmethod
    def _summary(values: np.ndarray) -> dict[str, Any]:
        return {"样本量": len(values), "均值": float(np.mean(values)), "中位数": float(np.median(values)), "标准差": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, "最小值": float(np.min(values)), "最大值": float(np.max(values))}

    @classmethod
    def _json(cls, value: Any) -> Any:
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, np.generic): return value.item()
        if isinstance(value, dict): return {str(key): cls._json(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)): return [cls._json(item) for item in value]
        return value