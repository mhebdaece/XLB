from __future__ import annotations

from dataclasses import dataclass

from .hierarchy import StaticRefinementHierarchy


@dataclass
class ParentChildMap:
    """Precomputed parent/child metadata placeholder for static refinement."""

    parent_level: int
    child_level: int
    refinement_ratio: int = 2

    @property
    def children_per_parent(self) -> int:
        return self.refinement_ratio**3


@dataclass
class InterfaceMap:
    """Placeholder for coarse/fine interface masks and directional crossing metadata."""

    coarse_level: int
    fine_level: int


class HierarchyMappings:
    """Container for inter-level mapping metadata used by the refined stepper."""

    def __init__(self, hierarchy: StaticRefinementHierarchy):
        self.parent_child = {}
        self.interfaces = {}
        for level in sorted(hierarchy.levels):
            if level == 0:
                continue
            key = (level - 1, level)
            self.parent_child[key] = ParentChildMap(parent_level=level - 1, child_level=level, refinement_ratio=hierarchy.refinement_ratio)
            self.interfaces[key] = InterfaceMap(coarse_level=level - 1, fine_level=level)
