"""
统计方法目录模块

本模块是假设检验方法卡片目录，包含所有预部署的本地统计 API。
主要功能：
- 构建默认的统计方法注册表
- 关联每个方法 ID 到对应的 Method Card 和执行函数
- 定义每个方法的可视化配方
- 声明每个方法的假设依赖（用于自动流水线）

设计原则：
- 只包含假设检验方法
- 每个方法都有对应的 Method Card，包含描述、输入输出、假设条件
- 新增方法只需注册新的 Card 与预部署函数，无需修改路由器
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from statagent.api.registry import StatisticalMethodRegistry
from statagent.protocol.statistical_object import MethodResult
from statagent.retrieval.method_card import AssumptionDep, MethodCard
from statagent.statistical_library.association_tests import run_correlation_matrix_tests, run_correlation_test, run_randomness_runs_test
from statagent.statistical_library.categorical import run_chi_square, run_cochran_mantel_haenszel, run_fisher_exact
from statagent.statistical_library.distribution_tests import run_anderson_darling, run_binomial_test, run_chi_square_gof, run_energy_normality, run_henze_zirkler, run_kolmogorov_smirnov, run_royston, run_shapiro_wilk
from statagent.statistical_library.nonparametric import run_friedman, run_kruskal_wallis, run_mann_whitney, run_moses_extreme_reactions, run_wald_wolfowitz, run_wilcoxon, run_wilcoxon_one_sample
from statagent.statistical_library.parametric import run_anova_oneway, run_dunnett, run_pairwise_ttests, run_ttest_independent, run_ttest_one_sample, run_ttest_paired, run_tukey_hsd, run_ztest_one_sample, run_ztest_two_sample
from statagent.statistical_library.survival import run_logrank
from statagent.statistical_library.variance_tests import run_bartlett, run_f_variance, run_levene

Function = Callable[[dict[str, Any], dict[str, Any]], MethodResult]


def _dep(
    assumption_type: str,
    test_method_id: str | None = None,
    threshold: float = 0.05,
    on_failure: str = "use_alternative",
    alternative_method_id: str | None = None,
) -> AssumptionDep:
    """创建假设依赖规范"""
    return AssumptionDep(
        assumption_type=assumption_type,
        test_method_id=test_method_id,
        threshold=threshold,
        on_failure=on_failure,
        alternative_method_id=alternative_method_id,
    )


# 方法假设依赖映射
ASSUMPTION_DEPS: dict[str, tuple[AssumptionDep, ...]] = {
    "PAR001": (  # 独立样本 t 检验
        _dep("normality", "DIST001", threshold=0.05, on_failure="use_alternative", alternative_method_id="NONPAR001"),
        _dep("homogeneity_of_variance", "VAR003", threshold=0.05, on_failure="use_alternative", alternative_method_id="PAR001"),
    ),
    "PAR002": (  # 配对样本 t 检验
        _dep("normality", "DIST001", threshold=0.05, on_failure="use_alternative", alternative_method_id="NONPAR002"),
    ),
    "PAR000": (  # 单样本 t 检验
        _dep("normality", "DIST001", threshold=0.05, on_failure="use_alternative", alternative_method_id="NONPAR004"),
    ),
    "PAR003": (  # 单因素方差分析
        _dep("normality", "DIST001", threshold=0.05, on_failure="use_alternative", alternative_method_id="NONPAR003"),
        _dep("homogeneity_of_variance", "VAR003", threshold=0.05, on_failure="use_alternative", alternative_method_id="PAR003"),
    ),
    "PAR006": (  # 成对多重 t 比较
        _dep("normality", "DIST001", threshold=0.05, on_failure="use_alternative", alternative_method_id="NONPAR003"),
        _dep("homogeneity_of_variance", "VAR003", threshold=0.05, on_failure="use_alternative", alternative_method_id="NONPAR003"),
    ),
    "PAR007": (  # Dunnett t 检验
        _dep("normality", "DIST001", threshold=0.05, on_failure="use_alternative", alternative_method_id="NONPAR003"),
        _dep("homogeneity_of_variance", "VAR003", threshold=0.05, on_failure="use_alternative", alternative_method_id="PAR007"),
    ),
    "PAR008": (  # Tukey HSD
        _dep("normality", "DIST001", threshold=0.05, on_failure="use_alternative", alternative_method_id="NONPAR003"),
        _dep("homogeneity_of_variance", "VAR003", threshold=0.05, on_failure="use_alternative", alternative_method_id="NONPAR003"),
    ),
    "CAT001": (  # 卡方独立性检验
        _dep("expected_frequency", None, threshold=0.05, on_failure="warn"),
    ),
    "COR001": (  # 双变量相关性检验
        _dep("normality", "DIST001", threshold=0.05, on_failure="use_alternative", alternative_method_id="COR001"),
    ),
    "COR002": (  # 多变量两两相关性检验
        _dep("normality", "DIST001", threshold=0.05, on_failure="use_alternative", alternative_method_id="COR002"),
    ),
}


VISUALIZATION_RECIPES: dict[str, tuple[str, ...]] = {
    "DIST001": ("distribution", "qq_plot"), "DIST005": ("distribution",),
    "DIST006": ("frequency_bar",), "DIST007": ("frequency_bar",), "DIST008": ("distribution", "qq_plot"),
    "VAR001": ("two_group_distribution", "two_group_boxplot"),
    "VAR002": ("multi_group_boxplot", "multi_group_distribution"), "VAR003": ("multi_group_boxplot", "multi_group_distribution"),
    "PAR000": ("distribution", "qq_plot"), "PAR001": ("two_group_distribution", "two_group_boxplot", "qq_plot"),
    "PAR002": ("paired_profile", "difference_distribution", "qq_plot"), "PAR003": ("multi_group_boxplot", "multi_group_distribution", "qq_plot"),
    "PAR004": ("distribution", "qq_plot"), "PAR005": ("two_group_distribution", "two_group_boxplot"),
    "PAR006": ("multi_group_boxplot", "multi_group_distribution"), "PAR007": ("multi_group_boxplot", "multi_group_distribution"), "PAR008": ("multi_group_boxplot", "multi_group_distribution"),
    "NONPAR001": ("two_group_distribution", "two_group_boxplot"), "NONPAR002": ("paired_profile", "difference_distribution"),
    "NONPAR003": ("multi_group_boxplot", "multi_group_distribution"), "NONPAR004": ("distribution", "boxplot"),
    "NONPAR005": ("two_group_distribution", "two_group_boxplot"), "NONPAR006": ("two_group_distribution",),
    "NONPAR007": ("multi_group_boxplot", "multi_group_distribution"),
    "CAT001": ("contingency_heatmap", "proportion_bar"), "CAT002": ("contingency_heatmap", "proportion_bar"),
    "SURV002": ("grouped_kaplan_meier", "event_rug"), "RUN001": ("distribution",),
}


def _card(method_id: str, name: str, branch: str, description: str, inputs: tuple[str, ...], outputs: tuple[str, ...], api: str, tags: tuple[str, ...], assumptions: tuple[str, ...] = (), example: dict[str, Any] | None = None) -> MethodCard:
    deps = ASSUMPTION_DEPS.get(method_id, ())
    return MethodCard(
        method_id=method_id, name=name, category=(branch, name), description=description,
        inputs=inputs, outputs=outputs, api=api,
        references=("用户提供的假设检验方法汇总表", "SciPy/Statsmodels 官方统计 API 文档"),
        assumptions=assumptions, assumption_deps=deps, example=example or {"data_keys": list(inputs)}, tags=tags,
        visualizations=VISUALIZATION_RECIPES.get(method_id, ()), version="2.0.0",
    )


def method_definitions() -> list[tuple[MethodCard, Function]]:
    """Only methods with an explicit null hypothesis and inferential decision are registered."""
    return [
        (_card("DIST001", "Shapiro–Wilk 正态性检验", "分布假设检验", "检验单个连续变量是否来自正态总体。", ("values",), ("w", "p_value"), "run_shapiro_wilk", ("正态性", "单变量", "分布检验"), ("独立同分布的连续观测", "样本量至少为 3")), run_shapiro_wilk),
        (_card("DIST002", "Energy 多元正态性检验", "分布假设检验", "使用能量距离与参数自助法检验多元正态性。", ("x",), ("energy", "p_value"), "run_energy_normality", ("多元正态性", "Energy", "自助法"), ("独立同分布的多元连续观测", "协方差矩阵非奇异")), run_energy_normality),
        (_card("DIST003", "Henze–Zirkler 多元正态性检验", "分布假设检验", "基于经验特征函数距离检验多元正态性。", ("x",), ("hz", "p_value"), "run_henze_zirkler", ("多元正态性", "Henze-Zirkler", "HZ"), ("独立同分布", "协方差矩阵非奇异")), run_henze_zirkler),
        (_card("DIST004", "Royston 多元正态性检验", "分布假设检验", "组合边际 Shapiro–Wilk 信息并按相关结构自助校准。", ("x",), ("h", "p_value"), "run_royston", ("多元正态性", "Royston", "Shapiro-Wilk"), ("样本量 3–5000", "协方差结构可估计")), run_royston),
        (_card("DIST005", "Kolmogorov–Smirnov 分布检验", "分布假设检验", "检验样本是否来自预先完整指定的连续分布。", ("values",), ("d", "p_value"), "run_kolmogorov_smirnov", ("K-S", "拟合优度", "连续分布"), ("分布参数事先给定", "连续观测")), run_kolmogorov_smirnov),
        (_card("DIST006", "卡方分布拟合优度检验", "分布假设检验", "检验分类频数是否符合预设概率分布。", ("values",), ("chi2", "p_value", "expected"), "run_chi_square_gof", ("卡方", "拟合优度", "分类频数"), ("类别互斥", "期望频数充分")), run_chi_square_gof),
        (_card("DIST007", "二项分布精确检验", "分布假设检验", "检验二分类事件的发生概率是否等于预设概率。", ("values",), ("successes", "p_value"), "run_binomial_test", ("二项检验", "比例", "精确检验"), ("固定试验次数", "各次试验独立且成功概率恒定")), run_binomial_test),
        (_card("DIST008", "Anderson–Darling 正态性检验", "分布假设检验", "Anderson–Darling 检验对分布尾部偏倚更敏感，样本量小时比 K–S 检验更准确。", ("values",), ("a2", "p_value"), "run_anderson_darling", ("Anderson-Darling", "正态性", "A²"), ("独立同分布的连续观测", "样本量至少为 5")), run_anderson_darling),
        (_card("VAR001", "两样本方差 F 检验", "方差齐性检验", "检验两个正态总体的方差是否相等。", ("group_a", "group_b"), ("f", "p_value"), "run_f_variance", ("F检验", "方差比", "两组"), ("两组独立", "两总体正态")), run_f_variance),
        (_card("VAR002", "Bartlett 方差齐性检验", "方差齐性检验", "检验多个正态总体的方差是否相等。", ("groups",), ("chi2", "p_value"), "run_bartlett", ("Bartlett", "方差齐性", "多组"), ("组间独立", "各组近似正态")), run_bartlett),
        (_card("VAR003", "Levene 方差齐性检验", "方差齐性检验", "以中心离差检验多个总体的方差齐性，默认使用中位数中心。", ("groups",), ("w", "p_value"), "run_levene", ("Levene", "Brown-Forsythe", "方差齐性"), ("组间独立",)), run_levene),
        (_card("PAR000", "单样本 t 检验", "参数检验", "检验正态总体均值是否等于预设值。", ("values",), ("t", "p_value"), "run_ttest_one_sample", ("t检验", "单样本", "均值"), ("观测独立", "总体近似正态")), run_ttest_one_sample),
        (_card("PAR001", "独立样本 t 检验", "参数检验", "比较两个独立总体的均值，默认使用 Welch 校正。", ("group_a", "group_b"), ("t", "p_value", "cohens_d"), "run_ttest_independent", ("t检验", "独立样本", "均值"), ("两组独立", "结局近似正态")), run_ttest_independent),
        (_card("PAR002", "配对样本 t 检验", "参数检验", "检验配对差值的总体均值是否为零。", ("before", "after"), ("t", "p_value", "cohens_dz"), "run_ttest_paired", ("t检验", "配对", "重复测量"), ("观测正确配对", "差值近似正态")), run_ttest_paired),
        (_card("PAR003", "单因素方差分析", "参数检验", "检验三个及以上独立总体的均值是否全部相等。", ("groups",), ("f", "p_value", "eta_squared"), "run_anova_oneway", ("ANOVA", "多组均值", "F检验"), ("观测独立", "残差正态", "方差齐性")), run_anova_oneway),
        (_card("PAR004", "单样本 Z 检验", "参数检验", "在总体标准差已知时检验单个总体均值。", ("values",), ("z", "p_value"), "run_ztest_one_sample", ("Z检验", "单样本", "已知方差"), ("总体标准差已知", "观测独立")), run_ztest_one_sample),
        (_card("PAR005", "双样本 Z 检验", "参数检验", "在两总体标准差已知时检验均值差。", ("group_a", "group_b"), ("z", "p_value"), "run_ztest_two_sample", ("Z检验", "双样本", "已知方差"), ("两总体标准差已知", "两组独立")), run_ztest_two_sample),
        (_card("PAR006", "均值的成对多重 t 比较", "参数检验", "对多组均值执行全部成对 t 检验并采用 Holm 校正。", ("groups",), ("comparisons", "adjusted_p_value"), "run_pairwise_ttests", ("多重比较", "Holm", "成对t检验"), ("组间独立", "预先规定多重性校正")), run_pairwise_ttests),
        (_card("PAR007", "Dunnett t 检验", "参数检验", "将多个实验组分别与一个对照组比较并控制家族错误率。", ("groups",), ("comparisons", "adjusted_p_value"), "run_dunnett", ("Dunnett", "对照组", "多重比较"), ("组间独立", "组内正态", "方差齐性")), run_dunnett),
        (_card("PAR008", "Tukey HSD 检验", "参数检验", "对所有组均值进行家族错误率校正后的成对比较。", ("groups",), ("comparisons", "adjusted_p_value"), "run_tukey_hsd", ("Tukey", "HSD", "多重比较"), ("组间独立", "组内正态", "方差齐性")), run_tukey_hsd),
        (_card("NONPAR001", "Mann–Whitney U / Wilcoxon 秩和检验", "非参数检验", "比较两个独立样本的秩分布。", ("group_a", "group_b"), ("u", "p_value"), "run_mann_whitney", ("Wilcoxon秩和", "Mann-Whitney", "两独立样本"), ("观测独立", "变量至少有序")), run_mann_whitney),
        (_card("NONPAR002", "配对 Wilcoxon 符号秩检验", "非参数检验", "比较两个配对样本的位置差异。", ("before", "after"), ("w", "p_value"), "run_wilcoxon", ("Wilcoxon", "配对", "符号秩"), ("正确配对", "差值分布近似对称")), run_wilcoxon),
        (_card("NONPAR003", "Kruskal–Wallis 秩和检验", "非参数检验", "比较多个独立样本的秩分布。", ("groups",), ("h", "p_value"), "run_kruskal_wallis", ("Kruskal-Wallis", "多独立样本", "秩和"), ("观测独立", "变量至少有序")), run_kruskal_wallis),
        (_card("NONPAR004", "单样本 Wilcoxon 符号秩检验", "非参数检验", "检验单样本位置是否等于预设中位位置。", ("values",), ("w", "p_value"), "run_wilcoxon_one_sample", ("Wilcoxon", "单样本", "中位数"), ("差值分布近似对称",)), run_wilcoxon_one_sample),
        (_card("NONPAR005", "Moses 极限反应检验", "非参数检验", "检验实验组的极端反应范围是否大于对照组。", ("group_a", "group_b"), ("range_difference", "p_value"), "run_moses_extreme_reactions", ("Moses", "极限反应", "离散程度"), ("两组独立", "至少为有序变量")), run_moses_extreme_reactions),
        (_card("NONPAR006", "Wald–Wolfowitz 两样本游程检验", "非参数检验", "根据合并排序后的标签游程检验两个样本分布是否相同。", ("group_a", "group_b"), ("runs", "z", "p_value"), "run_wald_wolfowitz", ("Wald-Wolfowitz", "游程", "两样本"), ("两组独立", "连续分布时并列值较少")), run_wald_wolfowitz),
        (_card("NONPAR007", "Friedman 秩和检验", "非参数检验", "比较三个及以上配对或区组样本的位置差异。", ("groups",), ("chi2", "p_value", "kendalls_w"), "run_friedman", ("Friedman", "配对多样本", "区组"), ("各条件观测一一配对", "变量至少有序")), run_friedman),
        (_card("CAT001", "卡方独立性检验", "列联表检验", "检验列联表中的两个分类变量是否独立。", ("table",), ("chi2", "p_value", "cramers_v"), "run_chi_square", ("卡方", "独立性", "列联表"), ("使用独立计数", "期望频数充分")), run_chi_square),
        (_card("CAT002", "Fisher 精确检验", "列联表检验", "对 2×2 列联表执行精确独立性检验。", ("table",), ("odds_ratio", "p_value"), "run_fisher_exact", ("Fisher", "精确检验", "2x2"), ("使用独立计数",)), run_fisher_exact),
        (_card("CAT003", "Cochran–Mantel–Haenszel 检验", "列联表检验", "在分层条件下检验两个二分类变量的条件独立性。", ("row", "column", "strata"), ("chi2", "p_value", "pooled_odds_ratio"), "run_cochran_mantel_haenszel", ("CMH", "分层列联表", "条件独立"), ("层间独立", "共同优势比具有解释意义")), run_cochran_mantel_haenszel),
        (_card("COR001", "双变量相关性检验", "相关性检验", "检验两个变量的 Pearson、Spearman 或 Kendall 相关是否为零。", ("variable_x", "variable_y"), ("correlation", "p_value"), "run_correlation_test", ("相关检验", "Pearson", "Spearman", "Kendall"), ("成对观测",)), run_correlation_test),
        (_card("COR002", "多变量两两相关性检验", "相关性检验", "对多个变量执行两两相关检验并采用 Holm 校正。", ("x",), ("comparisons", "adjusted_p_value"), "run_correlation_matrix_tests", ("相关矩阵", "多重比较", "Holm"), ("观测相互独立", "预先规定相关系数类型")), run_correlation_matrix_tests),
        (_card("RUN001", "随机性游程检验", "随机性检验", "根据观测序列在截断值两侧的游程数检验序列随机性。", ("values",), ("runs", "z", "p_value"), "run_randomness_runs_test", ("runs test", "随机性", "序列"), ("观测顺序有意义", "二分类序列概率稳定")), run_randomness_runs_test),
        (_card("SURV002", "Log-rank 生存曲线检验", "生存假设检验", "比较两个右删失总体的生存函数。", ("time", "event", "group"), ("chi2", "p_value"), "run_logrank", ("Log-rank", "生存曲线", "右删失"), ("删失独立", "比例风险时效能较高")), run_logrank),
    ]


def build_default_registry() -> StatisticalMethodRegistry:
    registry = StatisticalMethodRegistry()
    for card, function in method_definitions():
        registry.register(card, function)
    return registry