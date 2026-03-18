# Static 2:1 Mesh Refinement Plan for XLB

## Goal

Add **static** multilevel mesh refinement to XLB for LBM simulations, with these hard constraints:

- **No AMR**: the refinement layout is fixed at initialization time.
- **2:1 spatial refinement**: each refined coarse cell is represented by **8 children** in 3D.
- **2:1 temporal subcycling**: each refined level advances with **two fine steps per one coarse step**.
- **Correct coarse/fine coupling**: coarse-to-fine and fine-to-coarse communication must be modeled explicitly, not disguised as ordinary physical boundary conditions.

This plan is intentionally based on the current XLB code structure and on the operator decomposition exposed by Neon `apps/lbmMultiRes` (store / coalescence / explosion / fused finest variants), but it is adapted to XLB's Python + dense-grid architecture rather than copied literally.

---

## 1. What the current XLB codebase already implies

### 1.1 XLB is currently single-level everywhere

The current grid factory creates exactly one dense grid (`WarpGrid` or `JaxGrid`) from a single shape tuple, with no hierarchy metadata, patch ownership, parent/child map, or level-local origin/extent information. That means static refinement cannot be added by a small extension to `Grid`; it needs a new hierarchy object above the existing grids. See `grid_factory`, `Grid`, `WarpGrid`, and `JaxGrid`.【F:xlb/grid/grid.py†L9-L19】【F:xlb/grid/grid.py†L22-L75】【F:xlb/grid/warp_grid.py†L9-L28】【F:xlb/grid/jax_grid.py†L16-L49】

### 1.2 The Navier-Stokes stepper is a monolithic single-level driver

`IncompressibleNavierStokesStepper.prepare_fields()` allocates one `f_0`, one `f_1`, one `bc_mask`, and one `missing_mask`, and then processes all BCs against that single grid. Its runtime path also assumes one level: stream, apply BCs, compute macros, build equilibrium, collide, apply post-collision BCs, and write back. That is the main reason the static-refinement implementation should live in a **new multilevel stepper**, not inside the current class. See the prepare path and both JAX/Warp stepping paths.【F:xlb/operator/stepper/nse_stepper.py†L43-L83】【F:xlb/operator/stepper/nse_stepper.py†L111-L162】【F:xlb/operator/stepper/nse_stepper.py†L163-L197】【F:xlb/operator/stepper/nse_stepper.py†L261-L374】

### 1.3 Streaming is whole-array pull streaming on one level

The current `Stream` operator performs plain same-level pull streaming: JAX uses `jnp.roll`, while Warp computes `pull_index = index - c` with periodic wrapping. Neither path has any concept of coarse/fine interfaces, level jumps, or interface-selective replacement of populations. Therefore, any multiresolution path must either (a) skip interface populations in the regular stream or (b) overwrite those interface populations in explicit coarse/fine coupling operators after the bulk stream. See `Stream.jax_implementation()` and `Stream._construct_warp()`.【F:xlb/operator/stream/stream.py†L18-L41】【F:xlb/operator/stream/stream.py†L43-L89】

### 1.4 Boundary masking is built for physical BCs, not level coupling

`IndicesBoundaryMasker` and `MeshBoundaryMasker` construct `bc_mask` and `missing_mask` around physical boundaries and embedded geometry. Those masks identify where populations are missing after same-level streaming and which BC owns the site. They are not designed to identify covered coarse cells, coarse/fine interfaces, child ownership, or direction-selective inter-level transfers. Static refinement therefore needs **additional refinement-interface metadata**, separate from BC metadata. See the existing maskers and the stepper BC pipeline. 【F:xlb/operator/boundary_masker/indices_boundary_masker.py†L12-L109】【F:xlb/operator/boundary_masker/indices_boundary_masker.py†L110-L209】【F:xlb/operator/boundary_masker/mesh_boundary_masker.py†L11-L57】【F:xlb/operator/stepper/nse_stepper.py†L85-L109】

### 1.5 XLB's public API centers on `grid_factory()`, `prepare_fields()`, and a stepper call

The current examples create one grid, one stepper, call `prepare_fields()`, and then iterate by calling the stepper and swapping buffers. A multilevel design that preserves that mental model will fit XLB best. See the public exports and example setup patterns. 【F:xlb/grid/__init__.py†L1-L5】【F:xlb/operator/stepper/__init__.py†L1-L3】【F:xlb/__init__.py†L8-L31】【F:examples/cfd/flow_past_sphere_3d.py†L1-L111】【F:examples/cfd/flow_past_sphere_3d.py†L145-L159】【F:examples/cfd/windtunnel_3d.py†L1-L95】

### 1.6 The repository roadmap already points toward refinement and Neon interop

The README explicitly lists grid refinement as in-progress work and separately mentions future Neon + Warp scaling. That makes a static refinement implementation aligned with the repo direction, provided it is staged carefully. 【F:README.md†L165-L186】

---

## 2. Recommended scope lock for the first implementation

### 2.1 Functional scope

Implement:

- static refinement only,
- 3D only for the first deliverable,
- ratio `2` in space and time only,
- pull streaming only,
- Warp backend first,
- BGK first, then KBC parity.

Do **not** implement:

- runtime refine/coarsen decisions,
- arbitrary AMR trees,
- mixed refinement ratios,
- JAX parity in the first PR,
- fused kernels in the first PR.

### 2.2 Why 3D-first is the correct cut

Your requested topology is “one cell divided into 8 smaller cells,” which is inherently the 3D octree case. XLB supports both 2D and 3D, but the 3D path is the correct first target because it matches the required child cardinality and avoids designing for two topologies at once. The existing repository already contains substantial 3D examples that can serve as validation targets. 【F:examples/cfd/flow_past_sphere_3d.py†L13-L21】【F:examples/cfd/windtunnel_3d.py†L18-L28】

### 2.3 Why Warp-first is the correct cut

The Warp implementation is the natural first target because:

- the current Warp path already enforces pull streaming, which matches the multiresolution schedule cleanly, 【F:xlb/operator/stepper/nse_stepper.py†L36-L41】
- XLB's mesh masking path is already Warp-centric even when JAX is otherwise used, 【F:xlb/operator/boundary_masker/mesh_boundary_masker.py†L24-L35】
- the current JAX field layout is sharded and will complicate multilevel recursion and cross-level patch alignment on day one. 【F:xlb/grid/jax_grid.py†L20-L49】

---

## 3. Core architectural decision: patch-backed static hierarchy, not arbitrary octree storage

### 3.1 User semantics vs implementation strategy

Externally, the feature should behave like **static cell refinement**: selected coarse cells are refined 2:1 and each coarse cell corresponds to 8 children. Internally, XLB should store refined regions as **static, axis-aligned dense patches** per level, each backed by an ordinary `WarpGrid`/`JaxGrid`-style dense field allocation.

### 3.2 Why patch-backed storage is the right v1 design for XLB

XLB's current field allocation and operator interfaces are all dense-array oriented. Replacing that with an arbitrary sparse octree layout would multiply the implementation surface area across grids, stream/collision operators, masking, examples, tests, and post-processing. A patch-backed hierarchy allows XLB to keep using its dense kernels within each level while adding a multilevel orchestration layer above them. That is much more compatible with how fields are currently created through `create_nse_fields()` and grid-local `create_field()`. 【F:xlb/helper/nse_solver.py†L6-L35】【F:xlb/grid/warp_grid.py†L15-L28】【F:xlb/grid/jax_grid.py†L29-L49】

### 3.3 Supported topology for v1

Support one or more static boxes such that:

- each level is fully aligned to its parent cell lattice,
- each level refines by exactly `2`,
- every fine patch is strictly contained in a parent level domain,
- nesting is allowed, but partial overlap between unrelated patches on the same level is rejected in v1.

---

## 4. New domain model to add

Create a new package:

- `xlb/refinement/region.py`
- `xlb/refinement/hierarchy.py`
- `xlb/refinement/mappings.py`
- `xlb/refinement/state.py`
- optional helpers in `xlb/helper/multilevel_fields.py`

### 4.1 `RefinementRegion`

Represents one static patch defined in parent-level coordinates.

Suggested fields:

- `level: int`
- `parent_level: int`
- `coarse_start: tuple[int, int, int]`
- `coarse_stop: tuple[int, int, int]`
- `refinement_ratio: int = 2`
- derived `fine_shape`
- derived `fine_origin`
- validation helpers for alignment and bounds.

### 4.2 `LevelGrid`

A thin wrapper that owns:

- a normal dense XLB grid for that level,
- the level origin in root-coarse coordinates,
- active-cell extents,
- coverage metadata describing which parent cells are refined away by finer data.

### 4.3 `StaticRefinementHierarchy`

Owns all levels and maps.

Suggested responsibilities:

- build level objects from user-supplied refinement boxes,
- validate nesting and 2:1 alignment,
- compute parent/child indexing maps,
- compute coarse-interface and fine-interface masks,
- expose level-local and global physical coordinate transforms,
- provide composite sampling rules for output.

### 4.4 `ParentChildMap` / interface maps

Precompute and store:

- parent coarse cell → 8 child fine cells,
- fine child → parent coarse cell,
- coarse cells covered by fine patches,
- coarse-interface cells,
- fine-interface cells,
- direction masks indicating which populations cross a coarse/fine interface.

### 4.5 `MultilevelState`

For each level store:

- `f_0[level]`
- `f_1[level]`
- `bc_mask[level]`
- `missing_mask[level]`
- `cell_role_mask[level]`
- `iface_pull_mask[level]`
- explicit inter-level buffers, initially unfused.

---

## 5. Add refinement metadata that is separate from BC metadata

Introduce new refinement-specific masks/arrays for every level.

### 5.1 `cell_role_mask`

Suggested enum roles:

- `BULK`
- `PHYSICAL_BC`
- `COVERED_COARSE`
- `COARSE_INTERFACE`
- `FINE_INTERFACE`
- `INACTIVE`

### 5.2 `iface_pull_mask`

A direction-selective mask of shape similar to `missing_mask` that marks populations that must be supplied from another level instead of same-level streaming.

### 5.3 `coverage_mask`

Marks coarse cells whose solution is superseded by finer cells in the composite physical domain.

### 5.4 Why this separation matters

The current `bc_mask`/`missing_mask` pipeline is intentionally tied to physical BC logic and BC registry IDs. Reusing that structure for refinement would overload the BC abstraction and make the coarse/fine coupling path brittle. Refinement interfaces are **topological coupling operators**, not boundary conditions. That is consistent with the current BC architecture and with the way the existing stepper dispatches BC behavior through BC IDs. 【F:xlb/operator/stepper/nse_stepper.py†L85-L109】【F:xlb/operator/stepper/nse_stepper.py†L224-L294】

---

## 6. New operator family for coarse/fine communication

Add a new operator package:

- `xlb/operator/refinement/store_fine.py`
- `xlb/operator/refinement/store_coarse.py` *(optional in v1; see below)*
- `xlb/operator/refinement/coalescence.py`
- `xlb/operator/refinement/explosion.py`
- later: fused variants such as `fused_stream_explosion.py`, `fused_stream_coalescence.py`, `fused_finest.py`

### 6.1 `store_fine`

Runs after collision on the fine level and prepares data needed for later fine-to-coarse transfer.

Recommended v1 design:

- use explicit refinement buffers,
- avoid writing encoded temporary values back into the main distribution arrays,
- keep the operator readable and unit-testable.

### 6.2 `store_coarse`

Optional for v1. Neon exposes both coarse-initiated and fine-initiated storage paths, but XLB does not need both for the initial correctness-first implementation. Prefer **fine-initiated storage only** at first.

### 6.3 `coalescence`

Fills coarse populations that pull across a coarse/fine interface by reconstructing them from fine-level stored/interface data.

### 6.4 `explosion`

Fills fine interface populations that depend on coarse-level neighbors by distributing coarse information down to the fine side.

### 6.5 Initial buffer strategy

Use explicit arrays such as:

- `fine_store[level]`
- `coarse_pull_buffer[level]`
- `fine_pull_buffer[level]`

That is slower than a fully fused GPU strategy but far easier to verify. Only after correctness is established should XLB migrate to Neon-style in-place/fused communication kernels.

---

## 7. Required stepper refactor before adding the multilevel stepper

Refactor `IncompressibleNavierStokesStepper` into reusable single-level pieces without changing its public behavior.

### 7.1 Why the refactor is necessary

Right now, XLB mixes field preparation, BC handling, streaming, collision, and backend-specific logic inside one operator implementation. A multilevel stepper needs to call these same-level pieces repeatedly for each level and substep. Without extracting them, the multilevel implementation will duplicate too much logic and diverge quickly from the uniform-grid path.

### 7.2 Suggested extraction points

Split the current logic into helpers such as:

- `_prepare_single_level_fields(grid, bcs, initializer)`
- `_stream_bulk_jax(...)`
- `_stream_bulk_warp(...)`
- `_compute_macro_and_collision_jax(...)`
- `_compute_macro_and_collision_warp(...)`
- `_apply_streaming_bcs(...)`
- `_apply_collision_bcs(...)`
- `_swap_or_store_level_buffers(...)`

All of these are mechanically derivable from the current `prepare_fields()`, JAX path, and Warp path. 【F:xlb/operator/stepper/nse_stepper.py†L43-L83】【F:xlb/operator/stepper/nse_stepper.py†L111-L162】【F:xlb/operator/stepper/nse_stepper.py†L163-L197】【F:xlb/operator/stepper/nse_stepper.py†L302-L374】

### 7.3 New stepper class

Add:

- `xlb/operator/stepper/static_refined_nse_stepper.py`

Suggested class name:

- `StaticRefinedIncompressibleNavierStokesStepper`

Responsibilities:

- own a `StaticRefinementHierarchy`,
- prepare per-level fields and metadata,
- orchestrate recursive multilevel advancement,
- dispatch same-level bulk stream/collision to the existing kernels,
- invoke `store`, `coalescence`, and `explosion` at the correct points.

---

## 8. Time integration schedule: the most important algorithmic addition

The multilevel schedule should follow the same conceptual structure as Neon's recursive 2:1 advancement:

```python

def advance(level):
    collide_and_store(level)
    if has_finer(level):
        advance(level + 1)      # fine substep 1
    stream_and_couple(level)

    if has_finer(level):
        collide_and_store(level)
        advance(level + 1)      # fine substep 2
        stream_and_couple(level)
```

### 8.1 Semantics of each phase

`collide_and_store(level)`:

- operate only on active cells of that level,
- compute macroscopic variables,
- compute equilibrium,
- collide,
- prepare any fine-side store buffers needed for later coalescence.

`stream_and_couple(level)`:

- perform same-level **bulk** streaming,
- apply `explosion` where fine cells require coarse information,
- apply `coalescence` where coarse cells require fine information,
- apply physical BCs,
- finalize the next buffer for the level.

### 8.2 Why the schedule cannot be collapsed into the current one-step operator

The present stepper assumes one stream-collide cycle per global time step on one grid. Static refinement requires:

- two fine advances for each coarse advance,
- explicit parent/child communication between substeps,
- level-specific active-cell masks and interface handling.

That is fundamentally outside the contract of the current single-level stepper. 【F:xlb/operator/stepper/nse_stepper.py†L111-L162】【F:xlb/operator/stepper/nse_stepper.py†L302-L374】

---

## 9. Physics and numerics that must be level-aware

### 9.1 Per-level `omega`

The current examples pass one scalar `omega` to a uniform-grid stepper. In a static-refined simulation, each level must have its own `omega[level]` so that the physical viscosity remains consistent under `dt_fine = dt_coarse / 2` and `dx_fine = dx_coarse / 2`. Existing examples and the current stepper interface demonstrate the present single-scalar assumption. 【F:examples/cfd/flow_past_sphere_3d.py†L13-L21】【F:examples/cfd/flow_past_sphere_3d.py†L145-L151】【F:xlb/operator/stepper/nse_stepper.py†L111-L118】

### 9.2 Per-level BC masks

Every level needs its own `bc_mask` and `missing_mask`, with geometry/boundaries rasterized independently on that level grid. The current `prepare_fields()` path already does BC setup per grid; the multilevel design should preserve that one-grid-at-a-time behavior. 【F:xlb/operator/stepper/nse_stepper.py†L43-L83】

### 9.3 Covered coarse cells must not evolve as normal fluid cells

Once a coarse region is refined, those coarse cells remain only as parent/interface support. They must not contribute duplicate fluid evolution in the composite physical solution.

### 9.4 Composite output must select the finest available data

Current examples post-process one grid by directly computing macros from a single field. A multilevel run needs a composite sampling layer that chooses the finest valid level at each physical location before saving images or VTK. Existing post-processing code makes the single-level assumption explicit. 【F:examples/cfd/windtunnel_3d.py†L98-L191】【F:examples/cfd/flow_past_sphere_3d.py†L114-L143】

---

## 10. Detailed file-by-file implementation plan

### 10.1 New files

#### `xlb/refinement/__init__.py`
Export public refinement types.

#### `xlb/refinement/region.py`
Implement `RefinementRegion` and validation helpers.

#### `xlb/refinement/hierarchy.py`
Implement `StaticRefinementHierarchy`, level creation, patch validation, coordinate transforms, and composite sampling helpers.

#### `xlb/refinement/mappings.py`
Implement parent/child maps, interface masks, and directional crossing maps.

#### `xlb/refinement/state.py`
Implement `MultilevelState` container and level-local state objects.

#### `xlb/helper/multilevel_fields.py`
Add helpers to allocate per-level `f_0`, `f_1`, `bc_mask`, `missing_mask`, refinement masks, and explicit interface buffers.

#### `xlb/operator/refinement/store_fine.py`
Correctness-first fine-side accumulation/store operator.

#### `xlb/operator/refinement/coalescence.py`
Fine-to-coarse interface reconstruction operator.

#### `xlb/operator/refinement/explosion.py`
Coarse-to-fine distribution operator.

#### `xlb/operator/stepper/static_refined_nse_stepper.py`
New recursive multilevel stepper.

#### Example files

- `examples/cfd/lid_driven_cavity_3d_static_refine.py`
- `examples/cfd/windtunnel_3d_static_refine.py`

These should mirror current examples but replace the single grid with a hierarchy object and accept per-level relaxation data.

#### Test files

- `tests/refinement/test_hierarchy.py`
- `tests/refinement/test_store_coalescence.py`
- `tests/refinement/test_explosion.py`
- `tests/refinement/test_static_refined_stepper_warp.py`

### 10.2 Existing files to modify

#### `xlb/__init__.py`
Export the refinement package and multilevel stepper, preserving the current public import style. 【F:xlb/__init__.py†L8-L31】

#### `xlb/grid/__init__.py`
Export the multilevel grid wrapper or hierarchy-facing factory if one is introduced. 【F:xlb/grid/__init__.py†L1-L5】

#### `xlb/operator/stepper/stepper.py`
Generalize the base stepper contract so a multilevel `prepare_fields()` can return structured state rather than one flat tuple. 【F:xlb/operator/stepper/stepper.py†L1-L28】

#### `xlb/operator/stepper/__init__.py`
Export the new static refined stepper. 【F:xlb/operator/stepper/__init__.py†L1-L3】

#### `xlb/operator/stepper/nse_stepper.py`
Refactor into reusable single-level building blocks while preserving existing single-grid behavior. 【F:xlb/operator/stepper/nse_stepper.py†L13-L41】【F:xlb/operator/stepper/nse_stepper.py†L43-L83】【F:xlb/operator/stepper/nse_stepper.py†L111-L197】【F:xlb/operator/stepper/nse_stepper.py†L198-L374】

#### `xlb/operator/stream/stream.py`
Keep the current uniform-grid stream intact, but add either:

- an interface-aware masked stream path for multilevel use, or
- no changes to `Stream` itself and instead handle all interface replacement externally in refinement operators.

The second option is recommended for v1. 【F:xlb/operator/stream/stream.py†L18-L41】【F:xlb/operator/stream/stream.py†L43-L89】

#### `xlb/helper/nse_solver.py`
Add multilevel field allocation helpers or route them to `multilevel_fields.py`. 【F:xlb/helper/nse_solver.py†L6-L35】

#### Existing examples and utilities
No breaking changes. Add new multiresolution examples instead of modifying the existing ones.

---

## 11. Recommended implementation phases

### Phase 1 — correctness-first Warp MVP

Deliver:

- 3D only,
- one refined rectangular patch,
- BGK only,
- pull streaming only,
- explicit `store_fine`, `coalescence`, `explosion`,
- recursive two-substep schedule,
- one validation example: lid-driven cavity with a refined central box.

Exit criteria:

- runs stably,
- passes hierarchy and communication unit tests,
- composite output works,
- shows agreement with a uniform fine-grid reference in the refined region.

### Phase 2 — static hierarchy generalization

Deliver:

- multiple nested patches,
- KBC parity,
- wind tunnel / sphere validation example,
- composite VTK and image output,
- performance baseline measurements.

### Phase 3 — performance optimization

Only after Phase 2 correctness:

- collision-fused store,
- stream-fused explosion,
- stream-fused coalescence,
- fused finest kernels,
- reduced temporary storage and fewer kernel launches.

### Phase 4 — JAX parity

Deliver:

- same hierarchy data model,
- functional JAX interface ops,
- Python-level recursion around jitted level-local kernels first,
- only later a flattened loop if worthwhile.

---

## 12. Validation plan

### 12.1 Topology/unit tests

Test:

- patch alignment,
- bounds checking,
- parent/child map correctness,
- coarse coverage masks,
- interface direction masks.

### 12.2 Communication micro-tests

Construct a one-interface synthetic setup and verify:

- `explosion` fills the correct fine-side populations,
- `coalescence` reconstructs the correct coarse-side populations,
- level buffers are written only where intended.

### 12.3 Conservation checks

Run a periodic box with one refinement patch and verify acceptable mass/momentum drift over a long run.

### 12.4 Uniform-fine reference comparison

Compare a static-refined run against a uniform fine-grid reference over the refined physical subregion.

### 12.5 Application validation

Use:

- 3D lid-driven cavity centerlines,
- flow past sphere or wind tunnel over geometry.

These map naturally onto the current example set already present in the repo. 【F:examples/cfd/flow_past_sphere_3d.py†L1-L159】【F:examples/cfd/windtunnel_3d.py†L1-L191】

---

## 13. High-risk pitfalls to avoid

1. **Do not encode coarse/fine coupling in `bc_mask`.**
   That would conflate topology coupling with physical BC ownership and break the current BC dispatch model. 【F:xlb/operator/stepper/nse_stepper.py†L224-L294】

2. **Do not run the current whole-array `Stream` unchanged on interface populations.**
   It only knows same-level pull indices and periodic wrapping. 【F:xlb/operator/stream/stream.py†L18-L41】【F:xlb/operator/stream/stream.py†L43-L70】

3. **Do not keep one global `omega`.**
   Refined levels need level-aware relaxation to preserve the intended physics.

4. **Do not evolve covered coarse cells as normal fluid cells.**
   That will double-count portions of the domain.

5. **Do not start with fused kernels.**
   You want debuggable explicit buffers first.

6. **Do not target Warp and JAX equally in the first implementation.**
   The design surface is too large; establish the data model and schedule first.

---

## 14. Recommended end-state public API

```python
hierarchy = StaticRefinementHierarchy.from_boxes(
    coarse_shape=(Nx, Ny, Nz),
    boxes=[
        RefinementRegion(level=1, coarse_start=(...), coarse_stop=(...)),
        RefinementRegion(level=2, coarse_start=(...), coarse_stop=(...)),
    ],
    refinement_ratio=2,
)

stepper = StaticRefinedIncompressibleNavierStokesStepper(
    hierarchy=hierarchy,
    boundary_conditions=level_bcs,
    collision_type="BGK",
)

state = stepper.prepare_fields()

for coarse_step in range(num_steps):
    state = stepper(state, omega_by_level, coarse_step)
```

This keeps the user experience aligned with today's XLB workflow while introducing explicit multilevel state and a per-level `omega` contract.

---

## 15. Bottom line

The correct XLB implementation path is:

- **static**, not adaptive,
- **multilevel**, not hacked into a single uniform grid,
- **patch-backed**, not arbitrary sparse octree storage in v1,
- **Warp-first**, not dual-backend from day one,
- **correctness-first**, then Neon-style fusion.

If XLB tries to short-circuit the hierarchy model, the recursive subcycling schedule, or the dedicated coarse/fine operators, the result will be fragile and physically wrong at the refinement interface.
