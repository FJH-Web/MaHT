"""
假设图模块：可变有向无环图 G_H

理论映射说明：
- 论文中的假设图 G_H=(V_H, E_D) 对应本模块的 HypothesisGraph 类
- 顶点 V_H 是 HypothesisState 对象
- 有向边 E_D 表示假设之间的前置依赖关系

核心功能：
- 支持假设节点的添加、删除、查询
- 强制无环依赖检查
- 提供拓扑排序和就绪节点查询
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable

from statagent.protocol.statistical_object import Evidence, HypothesisState, HypothesisStatus


@dataclass(slots=True)
class HypothesisGraph:
    """假设图类：管理假设节点及其依赖关系
    
    这是一个有向无环图（DAG），用于管理统计推断过程中的假设及其依赖。
    
    属性：
        _nodes: 假设ID到假设状态的映射字典
    """
    
    _nodes: dict[str, HypothesisState] = field(default_factory=dict)

    def add_node(self, node: HypothesisState) -> None:
        if node.id in self._nodes:
            raise KeyError(f"Hypothesis already exists: {node.id}")
        missing = set(node.dependencies) - self._nodes.keys()
        if missing:
            raise KeyError(f"Unknown dependencies: {sorted(missing)}")
        self._nodes[node.id] = node
        self._assert_acyclic()

    def delete_node(self, hypothesis_id: str, *, cascade: bool = False) -> None:
        dependents = [n.id for n in self._nodes.values() if hypothesis_id in n.dependencies]
        if dependents and not cascade:
            raise ValueError(f"Node has dependents: {dependents}")
        for dependent_id in dependents:
            self.delete_node(dependent_id, cascade=True)
        del self._nodes[hypothesis_id]

    def get(self, hypothesis_id: str) -> HypothesisState:
        return self._nodes[hypothesis_id]

    def nodes(self) -> tuple[HypothesisState, ...]:
        return tuple(self._nodes.values())

    def update_evidence(self, hypothesis_id: str, evidence: Evidence) -> None:
        self.get(hypothesis_id).evidence.append(evidence)

    def update_status(self, hypothesis_id: str, status: HypothesisStatus) -> None:
        self.get(hypothesis_id).status = status

    def topological_order(self) -> list[HypothesisState]:
        indegree = {node_id: len(node.dependencies) for node_id, node in self._nodes.items()}
        children: dict[str, list[str]] = {node_id: [] for node_id in self._nodes}
        for node in self._nodes.values():
            for parent in node.dependencies:
                children[parent].append(node.id)
        queue = deque(sorted(node_id for node_id, degree in indegree.items() if degree == 0))
        ordered: list[HypothesisState] = []
        while queue:
            node_id = queue.popleft()
            ordered.append(self._nodes[node_id])
            for child in sorted(children[node_id]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if len(ordered) != len(self._nodes):
            raise ValueError("Hypothesis graph contains a cycle")
        return ordered

    def ready_nodes(self, terminal: Iterable[HypothesisStatus] | None = None) -> list[HypothesisState]:
        terminal_set = set(terminal or (HypothesisStatus.SUPPORTED, HypothesisStatus.REJECTED))
        return [
            node
            for node in self.topological_order()
            if node.status in {HypothesisStatus.PROPOSED, HypothesisStatus.READY}
            and all(self.get(parent).status in terminal_set for parent in node.dependencies)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {"nodes": [node.to_dict() for node in self.topological_order()]}

    def to_mermaid(self) -> str:
        lines = ["flowchart TD"]
        for node in self.topological_order():
            label = node.description.replace('"', "'")
            lines.append(f'  {self._safe_id(node.id)}["{node.id}: {label}"]')
            for parent in node.dependencies:
                lines.append(f"  {self._safe_id(parent)} --> {self._safe_id(node.id)}")
        return "\n".join(lines)

    @staticmethod
    def _safe_id(value: str) -> str:
        return "H_" + "".join(char if char.isalnum() else "_" for char in value)

    def _assert_acyclic(self) -> None:
        self.topological_order()