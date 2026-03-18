import pytest
import xlb

from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.operator.stepper import StaticRefinedIncompressibleNavierStokesStepper
from xlb.refinement import RefinementRegion, StaticRefinementHierarchy


@pytest.fixture(autouse=True)
def init_xlb_env():
    velocity_set = xlb.velocity_set.D3Q19(precision_policy=PrecisionPolicy.FP32FP32, compute_backend=ComputeBackend.WARP)
    xlb.init(
        velocity_set=velocity_set,
        default_backend=ComputeBackend.WARP,
        default_precision_policy=PrecisionPolicy.FP32FP32,
    )


def test_static_refined_stepper_prepare_fields_returns_level_state_for_each_grid():
    hierarchy = StaticRefinementHierarchy.from_boxes(
        coarse_shape=(8, 8, 8),
        boxes=[RefinementRegion(level=1, coarse_start=(2, 2, 2), coarse_stop=(4, 4, 4))],
        compute_backend=ComputeBackend.WARP,
    )
    stepper = StaticRefinedIncompressibleNavierStokesStepper(hierarchy=hierarchy)

    state = stepper.prepare_fields()

    assert set(state.levels) == {0, 1}
    root = state[0]
    fine = state[1]
    assert root.f_0.shape == (19, 8, 8, 8)
    assert fine.f_0.shape == (19, 4, 4, 4)
    assert root.cell_role_mask.shape == (1, 8, 8, 8)
    assert fine.iface_pull_mask.shape == (19, 4, 4, 4)
    assert set(root.interface_buffers) == {"fine_store", "coarse_pull", "fine_pull"}


def test_static_refined_stepper_call_is_explicitly_not_implemented_yet():
    hierarchy = StaticRefinementHierarchy.from_boxes(
        coarse_shape=(8, 8, 8),
        boxes=[RefinementRegion(level=1, coarse_start=(2, 2, 2), coarse_stop=(4, 4, 4))],
        compute_backend=ComputeBackend.WARP,
    )
    stepper = StaticRefinedIncompressibleNavierStokesStepper(hierarchy=hierarchy)
    state = stepper.prepare_fields()

    with pytest.raises(NotImplementedError, match="multilevel hierarchy/state scaffold only"):
        stepper(state, omega_by_level={0: 1.0, 1: 1.0}, timestep=0)
