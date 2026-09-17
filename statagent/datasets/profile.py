"""
数据集剖析模块

本模块提供结构化的数据集剖析功能和LLM提出的数据绑定验证。
主要功能：
- 数据集剖析：分析数据类型、缺失值、唯一值等
- DataFrame适配器：将数据绑定到统计方法

数据剖析结果包含：
- 行数和列数
- 每列的数据类型
- 每列的非缺失值数量
- 每列的缺失值数量
- 每列的唯一值数量
- 数值列的均值、最小值、最大值
- 分类列的类别列表
- 标识符候选识别
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from statagent.app.pipeline import ExecutionInput
from statagent.protocol.statistical_object import StatisticalObject
from statagent.datasets.intent import eligible_profile_columns, enforce_locked_mapping, is_identifier_name, mapping_columns


def profile_dataframe(frame: pd.DataFrame) -> dict[str, Any]:
    """剖析 DataFrame 并返回结构化信息"""
    columns: list[dict[str, Any]] = []
    for name in frame.columns:
        series = frame[name]
        item: dict[str, Any] = {
            "name": str(name),
            "dtype": str(series.dtype),
            "non_missing": int(series.notna().sum()),
            "missing": int(series.isna().sum()),
            "unique": int(series.nunique(dropna=True)),
            "position": len(columns) + 1,
        }
        sequence_identifier = False
        if pd.api.types.is_numeric_dtype(series) and series.notna().all() and len(series):
            values = series.to_numpy()
            sequence_identifier = bool(np.array_equal(values, np.arange(len(series))) or np.array_equal(values, np.arange(1, len(series) + 1)))
        item["identifier_candidate"] = bool(is_identifier_name(str(name)) or sequence_identifier)
        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna().astype(float)
            item.update({"kind": "numeric", "mean": float(clean.mean()) if len(clean) else None, "min": float(clean.min()) if len(clean) else None, "max": float(clean.max()) if len(clean) else None})
        else:
            item.update({"kind": "categorical", "levels": list(map(str, series.dropna().unique()[:20]))})
        columns.append(item)
    return {"rows": len(frame), "columns": columns, "missing_total": int(frame.isna().sum().sum())}


class DataFrameAdapter:
    """DataFrame 适配器：将数据绑定到统计方法"""
    
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self._original_rows = len(frame)
        self._diagnostics: dict[str, Any] = {}
        self._paired_clean_data = None
        self._paired_cols: list[str] = []

    def execution_input(self, message: StatisticalObject) -> ExecutionInput:
        """将 DataFrame 转换为统计方法的执行输入"""
        self._diagnostics = {}
        self._paired_clean_data = None
        self._paired_cols = []
        
        if not message.method:
            raise ValueError("Data binding requires a selected Method Card")
        match = message.method.get("match", {})
        proposed = dict(match.get("data_mapping", {}))
        context = message.hypothesis.metadata.get("analysis_context", {})
        locked = list(context.get("requested_columns", []))
        if locked:
            proposed = enforce_locked_mapping(message.method["inputs"], locked, proposed)
            unauthorized = set(mapping_columns(proposed)) - set(locked)
            if unauthorized:
                raise ValueError(f"数据映射包含用户未指定的变量：{sorted(unauthorized)}")
        
        required_columns = mapping_columns(proposed)
        inputs = message.method.get("inputs", [])
        
        paired_inputs = {"variable_x", "variable_y", "x", "y", "group_a", "group_b", "before", "after"}
        input_roles = set(inputs) & paired_inputs
        
        self._paired_cols = []
        if len(input_roles) >= 2:
            paired_cols = []
            for role in inputs:
                spec = proposed.get(role)
                if spec:
                    col = spec if isinstance(spec, str) else spec.get("column")
                    if col:
                        paired_cols.append(col)
            
            if len(paired_cols) >= 2:
                clean_frame = self.frame[paired_cols].dropna()
                self._diagnostics["original_rows"] = self._original_rows
                self._diagnostics["valid_rows"] = len(clean_frame)
                self._diagnostics["removed_rows"] = self._original_rows - len(clean_frame)
                self._paired_cols = paired_cols
                self._paired_clean_data = clean_frame
        
        bound: dict[str, Any] = {}
        for role in message.method["inputs"]:
            spec = proposed.get(role)
            bound[role] = self._bind(role, spec, bound, required_columns)
        return ExecutionInput(data=bound, parameters=dict(match.get("parameters", {})), diagnostics=self._diagnostics)

    def _bind(self, role: str, spec: Any, bound: dict[str, Any], required_columns: list[str]) -> Any:
        if isinstance(spec, str):
            spec = {"kind": "column", "column": spec}
        if isinstance(spec, dict):
            if role == "x" and spec.get("kind") == "columns":
                spec = {"kind": "matrix", "columns": spec.get("columns", [])}
            if role == "groups" and spec.get("kind") == "columns":
                columns = self._validate_columns(spec.get("columns", []))
                return [self._column(name, required_columns) for name in columns]
            handler = self._handlers().get(str(spec.get("kind", "column")))
            if handler:
                return handler(spec, required_columns)
        return self._fallback(role, bound, required_columns)

    def _handlers(self) -> dict[str, Callable[[dict[str, Any], list[str]], Any]]:
        return {
            "column": lambda spec, cols: self._column(spec["column"], cols),
            "matrix": lambda spec, cols: self._matrix(spec["columns"], cols),
            "columns": lambda spec, cols: [self._column(name, cols) for name in self._validate_columns(spec["columns"])],
            "grouped": lambda spec, cols: self._grouped(spec, cols),
            "contingency": lambda spec, cols: self._contingency(spec, cols),
        }

    def _column(self, name: str, required_columns: list[str]) -> list[Any]:
        self._validate_columns([name])
        if hasattr(self, "_paired_clean_data") and self._paired_clean_data is not None and name in self._paired_cols:
            return self._paired_clean_data[name].tolist()
        return self.frame[name].dropna().tolist()

    def _matrix(self, columns: list[str], required_columns: list[str]) -> list[list[float]]:
        valid_cols = self._validate_columns(columns)
        if required_columns and set(valid_cols).issubset(set(required_columns)):
            clean = self.frame[valid_cols].dropna()
            self._diagnostics["matrix_columns"] = valid_cols
            if "valid_rows" not in self._diagnostics:
                self._diagnostics["original_rows"] = self._original_rows
                self._diagnostics["valid_rows"] = len(clean)
                self._diagnostics["removed_rows"] = self._original_rows - len(clean)
            return clean.to_numpy().tolist()
        return self.frame[valid_cols].dropna().to_numpy().tolist()

    def _grouped(self, spec: dict[str, Any], required_columns: list[str]) -> list[list[Any]]:
        value, group = spec["value_column"], spec["group_column"]
        self._validate_columns([value, group])
        cols = [value, group]
        if required_columns and set(cols).issubset(set(required_columns)):
            clean = self.frame[cols].dropna()
            self._diagnostics["original_rows"] = self._original_rows
            self._diagnostics["valid_rows"] = len(clean)
            self._diagnostics["removed_rows"] = self._original_rows - len(clean)
            return [part[value].tolist() for _, part in clean.groupby(group, sort=True)]
        clean = self.frame[[value, group]].dropna()
        return [part[value].tolist() for _, part in clean.groupby(group, sort=True)]

    def _contingency(self, spec: dict[str, Any], required_columns: list[str]) -> list[list[int]]:
        row, column = spec["row_column"], spec["column_column"]
        self._validate_columns([row, column])
        cols = [row, column]
        if required_columns and set(cols).issubset(set(required_columns)):
            clean = self.frame[cols].dropna()
            self._diagnostics["original_rows"] = self._original_rows
            self._diagnostics["valid_rows"] = len(clean)
            self._diagnostics["removed_rows"] = self._original_rows - len(clean)
            return pd.crosstab(clean[row], clean[column]).to_numpy().tolist()
        return pd.crosstab(self.frame[row], self.frame[column]).to_numpy().tolist()

    def _fallback(self, role: str, bound: dict[str, Any], required_columns: list[str]) -> Any:
        profile = profile_dataframe(self.frame)
        eligible = eligible_profile_columns(profile)
        numeric = [name for name in self.frame.select_dtypes(include=[np.number]).columns.tolist() if name in eligible]
        categorical = [name for name in self.frame.columns if name not in numeric]
        
        def _get_xy_pair():
            if len(numeric) < 2:
                raise ValueError("相关性检验至少需要两个数值变量")
            x_cols = numeric[1:] if "y" in bound else numeric[:-1]
            y_col = numeric[0]
            cols_needed = x_cols + [y_col]
            clean_data = self.frame[cols_needed].dropna()
            if clean_data.empty:
                raise ValueError("所选变量在删除缺失值后无有效观测")
            self._diagnostics["original_rows"] = self._original_rows
            self._diagnostics["valid_rows"] = len(clean_data)
            self._diagnostics["removed_rows"] = self._original_rows - len(clean_data)
            return clean_data[x_cols].to_numpy().tolist(), clean_data[y_col].tolist()
        
        role_builders: dict[str, Callable[[], Any]] = {
            "values": lambda: self._column(numeric[0], required_columns),
            "group_a": lambda: self._column(numeric[0], required_columns),
            "group_b": lambda: self._column(numeric[1], required_columns),
            "before": lambda: self._column(numeric[0], required_columns),
            "after": lambda: self._column(numeric[1], required_columns),
            "groups": lambda: [self._column(name, required_columns) for name in numeric],
            "table": lambda: pd.crosstab(self.frame[categorical[0]], self.frame[categorical[1]]).to_numpy().tolist(),
            "time": lambda: self._column(numeric[0], required_columns),
            "event": lambda: self._column(numeric[1], required_columns),
            "group": lambda: self._column(categorical[0] if categorical else numeric[2], required_columns),
            "x": lambda: _get_xy_pair()[0],
            "y": lambda: _get_xy_pair()[1],
        }
        
        if role in ("x", "y"):
            x_result, y_result = _get_xy_pair()
            if role == "x":
                return x_result
            else:
                return y_result
        
        try:
            return role_builders[role]()
        except (KeyError, IndexError) as exc:
            raise ValueError(f"Cannot bind required input '{role}'. Review the LLM data mapping or dataset columns.") from exc

    def _validate_columns(self, names: list[str]) -> list[str]:
        missing = [name for name in names if name not in self.frame.columns]
        if missing:
            raise ValueError(f"Unknown dataset columns in mapping: {missing}")
        return names