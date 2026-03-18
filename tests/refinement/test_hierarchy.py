import pytest
import xlb

from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.refinement import RefinementRegion, StaticRefinementHierarchy


@pytest.fixture(autouse=True)
def init_xlb_env():
    velocity_set = xlb.velocity_set.D3Q19(precision_policy=PrecisionPolicy.FP32FP32, compute_backend=ComputeBackend.WARP)
    xlb.init(
        velocity_set=velocity_set,
        default_backend=ComputeBackend.WARP,
        default_precision_policy=PrecisionPolicy.FP32FP32,
    )


def test_static_refinement_hierarchy_from_boxes_builds_root_and_child_levels():
    hierarchy = StaticRefinementHierarchy.from_boxes(
        coarse_shape=(16, 12, 10),
        boxes=[RefinementRegion(level=1, coarse_start=(2, 2, 2), coarse_stop=(6, 5, 4))],
        compute_backend=ComputeBackend.WARP,
    )

    assert hierarchy.get_level(0).shape == (16, 12, 10)
    assert hierarchy.get_level(1).shape == (8, 6, 4)
    assert hierarchy.get_level(1).origin == (4, 4, 4)
    assert hierarchy.finest_level == 1


def test_static_refinement_hierarchy_rejects_overlapping_regions_on_same_level():
    with pytest.raises(ValueError, match="Overlapping refinement regions"):
        StaticRefinementHierarchy.from_boxes(
            coarse_shape=(16, 16, 16),
            boxes=[
                RefinementRegion(level=1, coarse_start=(2, 2, 2), coarse_stop=(5, 5, 5)),
                RefinementRegion(level=1, coarse_start=(4, 4, 4), coarse_stop=(7, 7, 7)),
            ],
            compute_backend=ComputeBackend.WARP,
        )
