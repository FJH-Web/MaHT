"""
可视化引擎模块

本模块提供注册驱动的统计可视化功能。
特点：
- 图表在本地生成，确保可复现性
- 支持中文标签和标题
- 每张 Method Card 声明方法专属可视化配方

主要功能：
- 探索性数据分析可视化
- 方法专属结果可视化
- 图表本地保存和语义解释
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from statagent.protocol.statistical_object import StatisticalObject, VisualizationArtifact
from statagent.visualization.fonts import configure_chinese_font

configure_chinese_font()

ZH_TEXT = {
    "Outcome distribution": "结局变量分布",
    "Distribution summary": "分布概况",
    "Category frequencies": "分类变量频数",
    "Group distributions": "两组分布比较",
    "Between-group comparison": "组间比较",
    "Normal Q–Q diagnostic": "正态 Q–Q 诊断",
    "Paired observations": "配对观测变化",
    "Paired-difference distribution": "配对差值分布",
    "Multi-group comparison": "多组比较",
    "Multi-group distributions": "多组分布比较",
    "Contingency table": "列联表热图",
    "Row-wise proportions": "各行类别构成比例",
    "Kaplan–Meier survival estimate": "Kaplan–Meier 生存曲线",
    "Group-specific survival curves": "分组生存曲线",
    "Event and censoring pattern": "事件与删失分布",
    "Coefficient estimates": "模型系数估计",
    "Observed versus fitted": "观测值与拟合值",
    "Residual diagnostics": "残差诊断",
    "PCA scree plot": "主成分碎石图",
    "PCA score map": "主成分得分图",
    "Feature importance": "变量重要性",
    "Prediction diagnostic": "预测诊断",
    "Sample": "样本",
    "Value": "取值",
    "Frequency": "频数",
    "Density": "密度",
    "Category": "类别",
    "Count": "计数",
    "Group": "组别",
    "Theoretical quantiles": "理论分位数",
    "Observed quantiles": "样本分位数",
    "Measurement": "测量时点",
    "After − Before": "干预后 − 干预前",
    "Column category": "列变量类别",
    "Row category": "行变量类别",
    "Proportion": "比例",
    "Time": "时间",
    "Survival probability": "生存概率",
    "Status": "结局状态",
    "Estimate": "估计值",
    "Term": "模型项",
    "Fitted": "拟合值",
    "Observed": "观测值",
    "Principal component": "主成分",
    "Explained variance ratio": "解释方差比例",
    "Importance": "重要性",
    "Feature": "变量",
    "Predicted": "预测值",
}

Recipe = Callable[[dict[str, Any], StatisticalObject], tuple[plt.Figure, dict[str, Any]]]


@dataclass(slots=True)
class VisualizationEngine:
    output_dir: Path = Path("artifacts/figures")
    dpi: int = 160
    _recipes: dict[str, Recipe] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self._recipes = {
            "distribution": self._distribution,
            "boxplot": self._boxplot,
            "frequency_bar": self._frequency_bar,
            "two_group_distribution": self._two_group_distribution,
            "two_group_boxplot": self._two_group_boxplot,
            "qq_plot": self._qq_plot,
            "paired_profile": self._paired_profile,
            "difference_distribution": self._difference_distribution,
            "multi_group_boxplot": self._multi_group_boxplot,
            "multi_group_distribution": self._multi_group_distribution,
            "contingency_heatmap": self._contingency_heatmap,
            "proportion_bar": self._proportion_bar,
            "kaplan_meier": self._kaplan_meier,
            "grouped_kaplan_meier": self._grouped_kaplan_meier,
            "event_rug": self._event_rug,
            "coefficient_forest": self._coefficient_forest,
            "observed_fitted": self._observed_fitted,
            "probability_diagnostic": self._observed_fitted,
            "residual_diagnostic": self._residual_diagnostic,
            "pca_scree": self._pca_scree,
            "pca_scores": self._pca_scores,
            "feature_importance": self._feature_importance,
            "prediction_diagnostic": self._prediction_diagnostic,
        }

    def render(self, message: StatisticalObject, data: dict[str, Any]) -> list[VisualizationArtifact]:
        if not message.method or not message.result:
            return []
        self.output_dir.mkdir(parents=True, exist_ok=True)
        artifacts: list[VisualizationArtifact] = []
        for recipe_id in message.method.get("visualizations", []):
            recipe = self._recipes.get(recipe_id)
            if recipe is None:
                continue
            try:
                figure, semantic = recipe(data, message)
                artifact_id = f"{message.hypothesis.id}-{recipe_id}-{uuid.uuid4().hex[:8]}"
                path = self.output_dir / f"{artifact_id}.png"
                figure.tight_layout()
                figure.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
                plt.close(figure)
                artifact = VisualizationArtifact(
                    id=artifact_id,
                    title=self._zh(semantic.pop("title", recipe_id.replace("_", " ").title())),
                    kind=recipe_id,
                    path=str(path.resolve()),
                    caption=semantic.pop("caption", "该图由已注册的本地统计方法根据原始数据和不可变统计结果生成。"),
                    semantic_payload=self._json(semantic),
                )
                artifacts.append(artifact)
            except (ValueError, KeyError, IndexError, TypeError):
                continue
        message.result.visualizations.extend(artifacts)
        return artifacts

    def render_exploratory(self, frame: Any, requested_columns: list[str] | None = None) -> list[VisualizationArtifact]:
        from statagent.visualization.eda import ExploratoryVisualizationEngine

        return ExploratoryVisualizationEngine(self.output_dir / "exploratory", dpi=self.dpi).render(frame, requested_columns)

    @staticmethod
    def _figure(title: str, xlabel: str = "", ylabel: str = "") -> tuple[plt.Figure, plt.Axes]:
        figure, axis = plt.subplots(figsize=(7.2, 4.4))
        axis.set_title(ZH_TEXT.get(title, title), fontweight="bold")
        axis.set_xlabel(ZH_TEXT.get(xlabel, xlabel))
        axis.set_ylabel(ZH_TEXT.get(ylabel, ylabel))
        axis.grid(alpha=0.18)
        return figure, axis

    def _distribution(self, data: dict[str, Any], message: StatisticalObject):
        values = np.asarray(data["values"], dtype=float)
        figure, axis = self._figure("Outcome distribution", "Value", "Frequency")
        axis.hist(values, bins="auto", color="#2F6B8A", alpha=0.8, edgecolor="white")
        axis.axvline(np.mean(values), color="#D1495B", linestyle="--", label="均值")
        axis.axvline(np.median(values), color="#EDA734", linestyle=":", label="中位数")
        axis.legend()
        return figure, {"title": "Outcome distribution", "n": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)), "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}

    def _boxplot(self, data: dict[str, Any], message: StatisticalObject):
        values = np.asarray(data["values"], dtype=float)
        figure, axis = self._figure("Distribution summary", "Sample", "Value")
        axis.boxplot(values, patch_artist=True, boxprops={"facecolor": "#8ECAE6"})
        return figure, {"title": "Distribution summary", "quartiles": np.quantile(values, [0.25, 0.5, 0.75]).tolist(), "range": [float(values.min()), float(values.max())]}

    def _frequency_bar(self, data: dict[str, Any], message: StatisticalObject):
        counts = message.result.estimates["count"]
        figure, axis = self._figure("Category frequencies", "Category", "Count")
        axis.bar(list(counts), list(counts.values()), color="#2F6B8A")
        axis.tick_params(axis="x", rotation=30)
        return figure, {"title": "Category frequencies", "counts": counts}

    def _two_group_distribution(self, data: dict[str, Any], message: StatisticalObject):
        a, b = np.asarray(data["group_a"], float), np.asarray(data["group_b"], float)
        figure, axis = self._figure("Group distributions", "Value", "Density")
        axis.hist(a, bins="auto", density=True, alpha=0.55, label="A 组", color="#2F6B8A")
        axis.hist(b, bins="auto", density=True, alpha=0.55, label="B 组", color="#E07A5F")
        axis.legend()
        return figure, {"title": "Group distributions", "group_a": self._summary(a), "group_b": self._summary(b)}

    def _two_group_boxplot(self, data: dict[str, Any], message: StatisticalObject):
        groups = [np.asarray(data["group_a"], float), np.asarray(data["group_b"], float)]
        figure, axis = self._figure("Between-group comparison", "Group", "Value")
        axis.boxplot(groups, tick_labels=["A 组", "B 组"], patch_artist=True)
        return figure, {"title": "Between-group comparison", "medians": [float(np.median(group)) for group in groups], "p_value": message.result.p_value}

    def _qq_plot(self, data: dict[str, Any], message: StatisticalObject):
        values = self._residual_like(data)
        figure, axis = self._figure("Normal Q–Q diagnostic", "Theoretical quantiles", "Observed quantiles")
        stats.probplot(values, dist="norm", plot=axis)
        axis.set_title("正态 Q–Q 诊断", fontweight="bold")
        axis.set_xlabel("理论分位数")
        axis.set_ylabel("样本分位数")
        return figure, {"title": "Normal Q–Q diagnostic", "n": len(values), "assumption_checks": message.result.assumption_checks}

    def _paired_profile(self, data: dict[str, Any], message: StatisticalObject):
        before, after = np.asarray(data["before"], float), np.asarray(data["after"], float)
        figure, axis = self._figure("Paired observations", "Measurement", "Value")
        for left, right in zip(before, after, strict=True):
            axis.plot([0, 1], [left, right], color="#6C757D", alpha=0.45)
        axis.plot([0, 1], [before.mean(), after.mean()], color="#D1495B", linewidth=3, marker="o", label="均值")
        axis.set_xticks([0, 1], ["干预前", "干预后"])
        axis.legend()
        return figure, {"title": "Paired observations", "mean_before": float(before.mean()), "mean_after": float(after.mean()), "mean_change": float((after - before).mean())}

    def _difference_distribution(self, data: dict[str, Any], message: StatisticalObject):
        difference = np.asarray(data["after"], float) - np.asarray(data["before"], float)
        figure, axis = self._figure("Paired-difference distribution", "After − Before", "Frequency")
        axis.hist(difference, bins="auto", color="#81B29A", edgecolor="white")
        axis.axvline(0, color="black", linestyle="--")
        return figure, {"title": "Paired-difference distribution", "mean_difference": float(difference.mean()), "median_difference": float(np.median(difference))}

    def _multi_group_boxplot(self, data: dict[str, Any], message: StatisticalObject):
        groups = [np.asarray(group, float) for group in data["groups"]]
        figure, axis = self._figure("Multi-group comparison", "Group", "Value")
        axis.boxplot(groups, tick_labels=[f"第 {i} 组" for i in range(1, len(groups) + 1)], patch_artist=True)
        return figure, {"title": "Multi-group comparison", "group_summaries": [self._summary(group) for group in groups], "p_value": message.result.p_value}

    def _multi_group_distribution(self, data: dict[str, Any], message: StatisticalObject):
        groups = [np.asarray(group, float) for group in data["groups"]]
        figure, axis = self._figure("Multi-group distributions", "Value", "Density")
        for index, group in enumerate(groups, start=1):
            axis.hist(group, bins="auto", density=True, histtype="step", linewidth=2, label=f"第 {index} 组")
        axis.legend()
        return figure, {"title": "Multi-group distributions", "group_summaries": [self._summary(group) for group in groups]}

    def _contingency_heatmap(self, data: dict[str, Any], message: StatisticalObject):
        table = np.asarray(data["table"], float)
        figure, axis = self._figure("Contingency table", "Column category", "Row category")
        image = axis.imshow(table, cmap="Blues")
        figure.colorbar(image, ax=axis, label="计数")
        for row in range(table.shape[0]):
            for column in range(table.shape[1]):
                axis.text(column, row, f"{table[row, column]:g}", ha="center", va="center")
        return figure, {"title": "Contingency table", "table": table.tolist(), "p_value": message.result.p_value}

    def _proportion_bar(self, data: dict[str, Any], message: StatisticalObject):
        table = np.asarray(data["table"], float)
        proportions = table / np.where(table.sum(axis=1, keepdims=True) == 0, 1, table.sum(axis=1, keepdims=True))
        figure, axis = self._figure("Row-wise proportions", "Row category", "Proportion")
        bottom = np.zeros(table.shape[0])
        for column in range(table.shape[1]):
            axis.bar(np.arange(table.shape[0]), proportions[:, column], bottom=bottom, label=f"第 {column + 1} 类")
            bottom += proportions[:, column]
        axis.legend()
        return figure, {"title": "Row-wise proportions", "proportions": proportions.tolist()}

    def _kaplan_meier(self, data: dict[str, Any], message: StatisticalObject):
        curve = message.result.estimates["curve"]
        figure, axis = self._figure("Kaplan–Meier survival estimate", "Time", "Survival probability")
        axis.step([point["time"] for point in curve], [point["survival"] for point in curve], where="post", color="#2F6B8A")
        axis.set_ylim(0, 1.03)
        return figure, {"title": "Kaplan–Meier survival estimate", "curve": curve, "median_survival": message.result.estimates.get("median_survival")}

    def _grouped_kaplan_meier(self, data: dict[str, Any], message: StatisticalObject):
        time, event, group = np.asarray(data["time"], float), np.asarray(data["event"], int), np.asarray(data["group"])
        figure, axis = self._figure("Group-specific survival curves", "Time", "Survival probability")
        curves = {}
        for level in np.unique(group):
            mask = group == level
            curve = self._km(time[mask], event[mask])
            curves[str(level)] = curve
            axis.step([p[0] for p in curve], [p[1] for p in curve], where="post", label=str(level))
        axis.set_ylim(0, 1.03)
        axis.legend()
        return figure, {"title": "Group-specific survival curves", "curves": curves, "logrank_p_value": message.result.p_value}

    def _event_rug(self, data: dict[str, Any], message: StatisticalObject):
        time, event = np.asarray(data["time"], float), np.asarray(data["event"], int)
        figure, axis = self._figure("Event and censoring pattern", "Time", "Status")
        axis.scatter(time[event == 1], np.ones((event == 1).sum()), marker="x", label="事件", color="#D1495B")
        axis.scatter(time[event == 0], np.zeros((event == 0).sum()), marker="|", label="删失", color="#2F6B8A")
        axis.set_yticks([0, 1], ["删失", "事件"])
        axis.legend()
        return figure, {"title": "Event and censoring pattern", "events": int(event.sum()), "censored": int((event == 0).sum()), "time_range": [float(time.min()), float(time.max())]}

    def _coefficient_forest(self, data: dict[str, Any], message: StatisticalObject):
        coefficients = message.result.estimates.get("coefficients", {})
        names, values, lows, highs = [], [], [], []
        for name, detail in coefficients.items():
            value = detail.get("estimate", detail.get("coef", detail.get("hazard_ratio", detail.get("odds_ratio", 0))))
            interval = detail.get("ci", detail.get("ci_odds_ratio", [value, value]))
            names.append(name); values.append(float(value)); lows.append(float(interval[0])); highs.append(float(interval[1]))
        figure, axis = self._figure("Coefficient estimates", "Estimate", "Term")
        positions = np.arange(len(names))
        axis.errorbar(values, positions, xerr=[np.asarray(values) - np.asarray(lows), np.asarray(highs) - np.asarray(values)], fmt="o", color="#2F6B8A", capsize=3)
        axis.axvline(0, color="black", linestyle="--", alpha=0.5)
        axis.set_yticks(positions, names)
        return figure, {"title": "Coefficient estimates", "coefficients": coefficients}

    def _observed_fitted(self, data: dict[str, Any], message: StatisticalObject):
        y = np.asarray(data["y"], float)
        residuals = np.asarray(message.result.residuals, float)
        fitted = y - residuals
        figure, axis = self._figure("Observed versus fitted", "Fitted", "Observed")
        axis.scatter(fitted, y, alpha=0.7, color="#2F6B8A")
        limits = [float(min(fitted.min(), y.min())), float(max(fitted.max(), y.max()))]
        axis.plot(limits, limits, linestyle="--", color="#D1495B")
        return figure, {"title": "Observed versus fitted", "correlation": float(np.corrcoef(fitted, y)[0, 1]), "observed_range": [float(y.min()), float(y.max())], "fitted_range": [float(fitted.min()), float(fitted.max())]}

    def _residual_diagnostic(self, data: dict[str, Any], message: StatisticalObject):
        residuals = np.asarray(message.result.residuals, float)
        figure, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].scatter(np.arange(len(residuals)), residuals, alpha=0.65)
        axes[0].axhline(0, color="black", linestyle="--")
        axes[0].set(title="残差序列", xlabel="观测序号", ylabel="残差")
        stats.probplot(residuals, dist="norm", plot=axes[1]); axes[1].set_title("残差 Q–Q 图"); axes[1].set_xlabel("理论分位数"); axes[1].set_ylabel("样本分位数")
        return figure, {"title": "Residual diagnostics", "residual_mean": float(residuals.mean()), "residual_sd": float(residuals.std(ddof=1)), "largest_absolute_residual": float(np.abs(residuals).max())}

    def _pca_scree(self, data: dict[str, Any], message: StatisticalObject):
        ratios = message.result.estimates["explained_variance_ratio"]
        figure, axis = self._figure("PCA scree plot", "Principal component", "Explained variance ratio")
        axis.plot(np.arange(1, len(ratios) + 1), ratios, marker="o", color="#2F6B8A")
        return figure, {"title": "PCA scree plot", "explained_variance_ratio": ratios, "cumulative": np.cumsum(ratios).tolist()}

    def _pca_scores(self, data: dict[str, Any], message: StatisticalObject):
        scores = np.asarray(message.result.estimates["scores"], float)
        figure, axis = self._figure("PCA score map", "PC1", "PC2")
        y = scores[:, 1] if scores.shape[1] > 1 else np.zeros(len(scores))
        axis.scatter(scores[:, 0], y, alpha=0.7, color="#2F6B8A")
        return figure, {"title": "PCA score map", "score_ranges": [[float(scores[:, i].min()), float(scores[:, i].max())] for i in range(min(2, scores.shape[1]))]}

    def _feature_importance(self, data: dict[str, Any], message: StatisticalObject):
        importance = np.asarray(message.result.estimates["feature_importance"], float)
        order = np.argsort(importance)
        figure, axis = self._figure("Feature importance", "Importance", "Feature")
        axis.barh(order, importance[order], color="#81B29A")
        axis.set_yticks(order, [f"变量 {i + 1}" for i in order])
        return figure, {"title": "Feature importance", "importance": importance.tolist(), "top_feature_index": int(np.argmax(importance))}

    def _prediction_diagnostic(self, data: dict[str, Any], message: StatisticalObject):
        observed, predicted = np.asarray(data["y"]), np.asarray(message.result.estimates["predictions"])
        figure, axis = self._figure("Prediction diagnostic", "Observed", "Predicted")
        axis.scatter(observed, predicted, alpha=0.55, color="#2F6B8A")
        return figure, {"title": "Prediction diagnostic", "agreement": float(np.mean(observed == predicted)) if observed.dtype.kind not in "fc" else float(np.corrcoef(observed, predicted)[0, 1])}

    def _residual_like(self, data: dict[str, Any]) -> np.ndarray:
        if "group_a" in data:
            return np.concatenate([np.asarray(data["group_a"], float), np.asarray(data["group_b"], float)])
        if "before" in data:
            return np.asarray(data["after"], float) - np.asarray(data["before"], float)
        if "groups" in data:
            return np.concatenate([np.asarray(group, float) for group in data["groups"]])
        return np.asarray(next(iter(data.values())), float)

    @staticmethod
    def _summary(values: np.ndarray) -> dict[str, Any]:
        return {"n": len(values), "mean": float(values.mean()), "median": float(np.median(values)), "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0, "min": float(values.min()), "max": float(values.max())}

    @staticmethod
    def _km(time: np.ndarray, event: np.ndarray) -> list[list[float]]:
        survival, curve = 1.0, [[0.0, 1.0]]
        for current in np.unique(time[event == 1]):
            at_risk = np.sum(time >= current); events = np.sum((time == current) & (event == 1))
            survival *= 1 - events / at_risk; curve.append([float(current), float(survival)])
        return curve

    @classmethod
    def _json(cls, value: Any) -> Any:
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, np.generic): return value.item()
        if isinstance(value, dict): return {str(k): cls._json(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)): return [cls._json(v) for v in value]
        return value

    @staticmethod
    def _zh(value: str) -> str:
        return ZH_TEXT.get(value, value)