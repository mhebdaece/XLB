from __future__ import annotations

from xlb.helper.nse_solver import create_nse_fields
from xlb.precision_policy import Precision
from xlb.refinement.state import LevelState, MultilevelState


def create_multilevel_nse_fields(hierarchy, velocity_set=None, compute_backend=None, precision_policy=None):
    levels = {}
    for level_index, level in hierarchy.levels.items():
        _, f_0, f_1, missing_mask, bc_mask = create_nse_fields(
            grid=level.grid,
            velocity_set=velocity_set,
            compute_backend=compute_backend,
            precision_policy=precision_policy,
        )
        cell_role_mask = level.grid.create_field(cardinality=1, dtype=Precision.UINT8)
        iface_pull_mask = level.grid.create_field(cardinality=velocity_set.q, dtype=Precision.BOOL)
        coverage_mask = level.grid.create_field(cardinality=1, dtype=Precision.BOOL)
        interface_buffers = {
            "fine_store": level.grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision),
            "coarse_pull": level.grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision),
            "fine_pull": level.grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision),
        }
        levels[level_index] = LevelState(
            level=level_index,
            f_0=f_0,
            f_1=f_1,
            bc_mask=bc_mask,
            missing_mask=missing_mask,
            cell_role_mask=cell_role_mask,
            iface_pull_mask=iface_pull_mask,
            coverage_mask=coverage_mask,
            interface_buffers=interface_buffers,
        )

    return MultilevelState(levels=levels)
