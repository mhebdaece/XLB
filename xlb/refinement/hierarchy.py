from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple

from xlb import DefaultConfig
from xlb.grid import grid_factory

from .region import RefinementRegion


@dataclass
class LevelGrid:
    """Level-local dense grid plus spatial metadata inside the hierarchy."""

    level: int
    shape: Tuple[int, int, int]
    origin: Tuple[int, int, int]
    parent_level: int | None
    grid: object
    regions: list[RefinementRegion] = field(default_factory=list)


class StaticRefinementHierarchy:
    """Static hierarchy of dense per-level grids aligned at 2:1 refinement."""

    def __init__(self, coarse_shape: Tuple[int, int, int], levels: Dict[int, LevelGrid], refinement_ratio: int = 2):
        if len(coarse_shape) != 3:
            raise ValueError("Static refinement currently supports 3D coarse grids only")
        if refinement_ratio != 2:
            raise ValueError("Only 2:1 static refinement is supported")
        if 0 not in levels:
            raise ValueError("Hierarchy must define a root level 0")
        self.coarse_shape = coarse_shape
        self.levels = dict(sorted(levels.items()))
        self.refinement_ratio = refinement_ratio

    @classmethod
    def from_boxes(
        cls,
        coarse_shape: Tuple[int, int, int],
        boxes: Iterable[RefinementRegion],
        compute_backend=None,
        refinement_ratio: int = 2,
    ) -> "StaticRefinementHierarchy":
        compute_backend = compute_backend or DefaultConfig.default_backend
        root = LevelGrid(
            level=0,
            shape=coarse_shape,
            origin=(0, 0, 0),
            parent_level=None,
            grid=grid_factory(coarse_shape, compute_backend=compute_backend),
            regions=[],
        )
        levels = {0: root}
        per_level_regions: Dict[int, list[RefinementRegion]] = {}
        for region in boxes:
            per_level_regions.setdefault(region.level, []).append(region)

        for level in sorted(per_level_regions):
            regions = per_level_regions[level]
            cls._validate_regions(regions, coarse_shape, refinement_ratio)
            min_start = tuple(min(region.coarse_start[d] for region in regions) for d in range(3))
            max_stop = tuple(max(region.coarse_stop[d] for region in regions) for d in range(3))
            parent_shape = tuple(max_stop[d] - min_start[d] for d in range(3))
            level_shape = tuple(refinement_ratio * extent for extent in parent_shape)
            level_origin = tuple(refinement_ratio * min_start[d] for d in range(3))
            levels[level] = LevelGrid(
                level=level,
                shape=level_shape,
                origin=level_origin,
                parent_level=level - 1,
                grid=grid_factory(level_shape, compute_backend=compute_backend),
                regions=list(regions),
            )

        return cls(coarse_shape=coarse_shape, levels=levels, refinement_ratio=refinement_ratio)

    @staticmethod
    def _validate_regions(regions: Iterable[RefinementRegion], coarse_shape: Tuple[int, int, int], refinement_ratio: int):
        regions = list(regions)
        for region in regions:
            if region.refinement_ratio != refinement_ratio:
                raise ValueError("All regions in the hierarchy must use the same refinement ratio")
            if any(start < 0 for start in region.coarse_start):
                raise ValueError("Refinement regions must be within the parent domain")
            if any(stop > bound for stop, bound in zip(region.coarse_stop, coarse_shape)):
                raise ValueError("Refinement regions must be within the parent domain")

        for i, left in enumerate(regions):
            for right in regions[i + 1 :]:
                overlaps = all(left.coarse_start[d] < right.coarse_stop[d] and right.coarse_start[d] < left.coarse_stop[d] for d in range(3))
                if overlaps:
                    raise ValueError("Overlapping refinement regions on the same level are not supported")

    @property
    def finest_level(self) -> int:
        return max(self.levels)

    def get_level(self, level: int) -> LevelGrid:
        return self.levels[level]
