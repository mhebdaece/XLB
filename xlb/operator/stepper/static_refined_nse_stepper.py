from __future__ import annotations

from typing import Mapping

from xlb.helper.initializers import initialize_eq
from xlb.helper.multilevel_fields import create_multilevel_nse_fields
from xlb.operator.stepper.stepper import Stepper
from xlb.refinement import HierarchyMappings


class StaticRefinedIncompressibleNavierStokesStepper(Stepper):
    """Initial static-refinement stepper scaffold for multilevel NSE integration."""

    def __init__(self, hierarchy, boundary_conditions=None, collision_type="BGK", streaming_scheme="pull"):
        self.hierarchy = hierarchy
        self.collision_type = collision_type
        self.streaming_scheme = streaming_scheme
        self.boundary_conditions_by_level = self._normalize_boundary_conditions(boundary_conditions or {})
        self.mappings = HierarchyMappings(hierarchy)
        super().__init__(hierarchy.get_level(0).grid, self.boundary_conditions_by_level.get(0, []))

    @staticmethod
    def _normalize_boundary_conditions(boundary_conditions):
        if isinstance(boundary_conditions, Mapping):
            return {int(level): list(bcs) for level, bcs in boundary_conditions.items()}
        return {0: list(boundary_conditions)}

    def prepare_fields(self, initializer=None):
        state = create_multilevel_nse_fields(
            hierarchy=self.hierarchy,
            velocity_set=self.velocity_set,
            compute_backend=self.compute_backend,
            precision_policy=self.precision_policy,
        )

        if initializer is None:
            for level_state in state.levels.values():
                initialized = initialize_eq(
                    level_state.f_0,
                    self.hierarchy.get_level(level_state.level).grid,
                    self.velocity_set,
                    self.precision_policy,
                    self.compute_backend,
                )
                level_state.f_0 = initialized
                if self.compute_backend.name == "JAX":
                    level_state.f_1 = initialized.copy()
                else:
                    import warp as wp

                    wp.copy(level_state.f_1, initialized)
        else:
            for level_state in state.levels.values():
                initialized = initializer(
                    self.hierarchy.get_level(level_state.level).grid,
                    self.velocity_set,
                    self.precision_policy,
                    self.compute_backend,
                )
                level_state.f_0 = initialized
                if self.compute_backend.name == "JAX":
                    level_state.f_1 = initialized.copy()
                else:
                    import warp as wp

                    wp.copy(level_state.f_1, initialized)

        return state

    def __call__(self, state, omega_by_level, timestep, callback=None):
        raise NotImplementedError(
            "StaticRefinedIncompressibleNavierStokesStepper currently introduces the multilevel hierarchy/state scaffold only; "
            "coarse/fine coupling operators and recursive advancement are not implemented yet."
        )
