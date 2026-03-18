import xlb

from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.operator.stepper import StaticRefinedIncompressibleNavierStokesStepper
from xlb.refinement import RefinementRegion, StaticRefinementHierarchy


velocity_set = xlb.velocity_set.D3Q19(
    precision_policy=PrecisionPolicy.FP32FP32,
    compute_backend=ComputeBackend.WARP,
)

xlb.init(
    velocity_set=velocity_set,
    default_backend=ComputeBackend.WARP,
    default_precision_policy=PrecisionPolicy.FP32FP32,
)

hierarchy = StaticRefinementHierarchy.from_boxes(
    coarse_shape=(8, 8, 8),
    boxes=[RefinementRegion(level=1, coarse_start=(2, 2, 2), coarse_stop=(4, 4, 4))],
    compute_backend=ComputeBackend.WARP,
)

stepper = StaticRefinedIncompressibleNavierStokesStepper(hierarchy=hierarchy)
state = stepper.prepare_fields()

print("Static refinement scaffold demo")
print(f"Levels: {sorted(state.levels.keys())}")
for level, level_state in state.levels.items():
    print(
        f"level={level} grid_shape={hierarchy.get_level(level).shape} "
        f"f_shape={level_state.f_0.shape} buffers={sorted(level_state.interface_buffers.keys())}"
    )
