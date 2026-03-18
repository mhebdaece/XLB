from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class LevelState:
    level: int
    f_0: Any
    f_1: Any
    bc_mask: Any
    missing_mask: Any
    cell_role_mask: Any
    iface_pull_mask: Any
    coverage_mask: Any
    interface_buffers: Dict[str, Any]


@dataclass
class MultilevelState:
    levels: Dict[int, LevelState]

    def __getitem__(self, level: int) -> LevelState:
        return self.levels[level]
