from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class RefinementRegion:
    """Static 2:1 refinement region described in parent-level index space."""

    level: int
    coarse_start: Tuple[int, int, int]
    coarse_stop: Tuple[int, int, int]
    refinement_ratio: int = 2
    parent_level: int | None = None

    def __post_init__(self):
        if self.level < 1:
            raise ValueError("RefinementRegion.level must be >= 1")
        if self.refinement_ratio != 2:
            raise ValueError("Only 2:1 static refinement is supported")
        if len(self.coarse_start) != 3 or len(self.coarse_stop) != 3:
            raise ValueError("Static refinement currently supports 3D regions only")
        if self.parent_level is not None and self.parent_level != self.level - 1:
            raise ValueError("RefinementRegion.parent_level must equal level - 1 when provided")
        if any(stop <= start for start, stop in zip(self.coarse_start, self.coarse_stop)):
            raise ValueError("RefinementRegion.coarse_stop must be greater than coarse_start in every dimension")

    @property
    def resolved_parent_level(self) -> int:
        return self.parent_level if self.parent_level is not None else self.level - 1

    @property
    def coarse_shape(self) -> Tuple[int, int, int]:
        return tuple(stop - start for start, stop in zip(self.coarse_start, self.coarse_stop))

    @property
    def fine_shape(self) -> Tuple[int, int, int]:
        return tuple(self.refinement_ratio * extent for extent in self.coarse_shape)

    @property
    def fine_origin(self) -> Tuple[int, int, int]:
        return tuple(self.refinement_ratio * start for start in self.coarse_start)
