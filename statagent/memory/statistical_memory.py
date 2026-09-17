"""
统计内存模块

本模块提供统计推断历史的存储功能。

理论映射说明：
- 统计内存记录 H（假设）-> Method（方法）-> Result（结果）-> inferential summaries（推断总结）
- -> assumptions（假设）-> Decision（决策）的完整链条

设计原则：
- 不保存聊天记录，只保存结构化的推断过程
- 存储接口使用 JSON，可替换为数据库适配器
- 支持版本历史和推断谱系追溯
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from statagent.protocol.statistical_object import StatisticalObject, utc_now


@dataclass(slots=True)
class MemoryRecord:
    """内存记录：保存单条推断历史"""
    
    record_id: str
    message: StatisticalObject
    created_at: str = field(default_factory=utc_now)
    revision: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryRecord":
        values = dict(data)
        values["message"] = StatisticalObject.from_dict(values["message"])
        return cls(**values)


class StatisticalMemory:
    """统计内存：存储推断历史和版本"""
    
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._records: dict[str, list[MemoryRecord]] = {}
        if self.path and self.path.exists():
            self._load()

    def update(self, record: MemoryRecord) -> None:
        """更新或添加记录"""
        history = self._records.setdefault(record.record_id, [])
        record.revision = len(history) + 1
        history.append(deepcopy(record))
        self._persist()

    def recall(self, record_id: str) -> MemoryRecord:
        """获取最新的记录"""
        return self._records[record_id][-1]

    def history(self, record_id: str) -> tuple[MemoryRecord, ...]:
        return tuple(self._records.get(record_id, ()))

    def search(
        self,
        *,
        hypothesis_id: str | None = None,
        method_id: str | None = None,
        decision: str | None = None,
    ) -> list[MemoryRecord]:
        records = [versions[-1] for versions in self._records.values() if versions]
        if hypothesis_id:
            records = [r for r in records if r.message.hypothesis.id == hypothesis_id]
        if method_id:
            records = [
                r for r in records if r.message.result and r.message.result.method_id == method_id
            ]
        if decision:
            records = [r for r in records if r.message.decision.value == decision]
        return records

    def merge(self, other: "StatisticalMemory") -> None:
        for record_id, versions in other._records.items():
            existing = {record.message.timestamp for record in self._records.get(record_id, [])}
            for record in versions:
                if record.message.timestamp not in existing:
                    self._records.setdefault(record_id, []).append(record)
            self._records[record_id].sort(key=lambda item: item.created_at)
            for revision, record in enumerate(self._records[record_id], start=1):
                record.revision = revision
        self._persist()

    def export(self, destination: str | Path) -> Path:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return target

    def to_dict(self) -> dict[str, Any]:
        return {key: [record.to_dict() for record in value] for key, value in self._records.items()}

    def _persist(self) -> None:
        if self.path:
            self.export(self.path)

    def _load(self) -> None:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self._records = {
            key: [MemoryRecord.from_dict(item) for item in versions]
            for key, versions in raw.items()
        }