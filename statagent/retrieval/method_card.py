"""
方法卡片模块

Method Card 是统计知识树的叶子节点，包含统计方法的完整元数据。

每个 Method Card 包含：
- method_id: 方法唯一标识符
- name: 方法名称
- category: 分类（分支和方法名）
- description: 方法描述
- inputs: 必需的输入字段
- outputs: 输出的统计量
- api: 调用的API名称
- references: 参考文献
- assumptions: 方法假设条件
- assumption_deps: 假设依赖规范（用于自动流水线）
- example: 使用示例
- tags: 搜索标签
- visualizations: 可视化配方
- version: 版本号

新增方法只需注册新的 Card 与预部署函数，无需修改路由器。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


class AssumptionFailureAction(str, Enum):
    """假设检验失败时的动作"""
    USE_ALTERNATIVE = "use_alternative"  # 使用替代方法
    WARN = "warn"                         # 仅警告
    ABORT = "abort"                       # 中止执行


class AssumptionType(str, Enum):
    """假设类型"""
    NORMALITY = "normality"                         # 正态性
    HOMOSCEDASTICITY = "homogeneity_of_variance"    # 方差齐性
    INDEPENDENCE = "independence"                   # 独立性
    EXPECTED_FREQUENCY = "expected_frequency"       # 期望频数
    LINEARITY = "linearity"                         # 线性关系
    CONTINUITY = "continuity"                        # 连续性


@dataclass(frozen=True, slots=True)
class AssumptionDep:
    """假设依赖规范：声明方法对某个假设的依赖及检验方式"""
    
    assumption_type: str              # 假设类型 (AssumptionType 值)
    test_method_id: str | None = None   # 检验方法ID (如 DIST001)
    threshold: float = 0.05          # p值阈值
    on_failure: str = "use_alternative"  # 失败时的动作
    alternative_method_id: str | None = None  # 替代方法ID
    
    def __post_init__(self) -> None:
        if self.on_failure not in ("use_alternative", "warn", "abort"):
            raise ValueError(f"Invalid on_failure action: {self.on_failure}")
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AssumptionDep":
        return cls(**dict(data))


@dataclass(frozen=True, slots=True)
class MethodCard:
    """方法卡片：统计方法的元数据容器"""
    
    method_id: str
    name: str
    category: tuple[str, ...]
    description: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    api: str
    references: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    assumption_deps: tuple[AssumptionDep, ...] = ()
    example: Mapping[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    visualizations: tuple[str, ...] = ()
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        required = (self.method_id, self.name, self.description, self.api)
        if not all(value.strip() for value in required):
            raise ValueError("method_id, name, description and api are required")
        if not self.category:
            raise ValueError("category cannot be empty")

    @property
    def search_text(self) -> str:
        """生成用于检索的文本"""
        return " ".join(
            (
                self.name,
                self.description,
                *self.category,
                *self.tags,
                *self.inputs,
                *self.assumptions,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["assumption_deps"] = [dep.to_dict() for dep in self.assumption_deps]
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MethodCard":
        values = dict(data)
        for key in ("category", "inputs", "outputs", "references", "assumptions", "tags", "visualizations"):
            values[key] = tuple(values.get(key, ()))
        
        assumption_deps = values.pop("assumption_deps", ())
        if assumption_deps:
            values["assumption_deps"] = tuple(AssumptionDep.from_dict(d) for d in assumption_deps)
        else:
            values["assumption_deps"] = ()
        
        return cls(**values)