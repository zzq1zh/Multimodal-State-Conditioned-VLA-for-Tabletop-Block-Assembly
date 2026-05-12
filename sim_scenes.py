"""
sim_scenes.py - SO-ARM100 pick-and-place

Grasp a chosen cube from the source bin and place it on a cell of the 3x3 target grid.
Wrist roll stays fixed; the gripper stays as vertical as possible.

Key design (tuned from the XML asset and runtime diagnostics):
  - Low orientation_cost (0.05) so IK prioritizes position over orientation
  - Target rotation from the current ee_site (local Y fixed to [0,0,1] for downward tool axis)
  - Wrist roll joint velocity forced to zero each step
  - Transport and place rely on real contact/friction; no per-step teleport or snap-to-pad in sim
"""

from pathlib import Path
import time
import numpy as np
import mujoco
import mujoco.viewer
import mink

# ------------------------- Paths -------------------------
# Directory of this file
_HERE = Path(__file__).resolve().parent
# MuJoCo model XML path
_XML = _HERE / "gym_so100" / "assets" / "so100_puzzle.xml"

# ------------------------- Cubes (core data) -------------------------
# Per-color cube body names in the XML asset
CUBE_BODIES: dict[str, list[str]] = {
    "yellow": ["cube_yellow_0", "cube_yellow_1"],
    "green":  ["cube_green_0"],
    "purple": ["cube_purple_0", "cube_purple_1", "cube_purple_2"],
    "orange": ["cube_orange_0", "cube_orange_1", "cube_orange_2"],
}
# Flattened list of all cube body names for iteration
ALL_CUBE_NAMES: list[str] = [n for ns in CUBE_BODIES.values() for n in ns]
TOTAL_CUBE_COUNT: int = len(ALL_CUBE_NAMES)
# Distinct color keys
COLOR_NAMES = list(CUBE_BODIES.keys())


def max_picks_per_color_physical() -> dict[str, int]:
    """Number of bodies per color in the scene XML (upper bound on global picks per color)."""
    return {c: len(ns) for c, ns in CUBE_BODIES.items()}


def color_and_index_for_body(body_name: str) -> tuple[str, int]:
    """Map a MuJoCo body name to semantic color and instance index within that color."""
    for color, names in CUBE_BODIES.items():
        if body_name in names:
            return str(color), int(names.index(body_name))
    raise ValueError(f"Unknown cube body {body_name!r}")


def build_task_text_for_body(body_name: str, target_cell: int) -> str:
    """
    Build the task line from the real body name so the color word matches the XML asset
    (avoids drifting from what is visible on screen; color is a plain Python str).
    """
    color, _ = color_and_index_for_body(body_name)
    return f"Pick up the {color} cube and place it in grid cell {int(target_cell)}."


# Inactive cubes for this episode: negative X, below table, away from cam_global / cam_oblique frusta (world frame)
_STASH_CUBES_X0 = -0.62
_STASH_CUBES_Y = 0.20
_STASH_CUBES_Z = -0.12
_STASH_CUBES_DX = 0.024

# ------------------------- Target grid (core data) -------------------------
# grid_base matches so100_puzzle.xml; slightly pulled toward the robot base from desk center (0, 0.6)
_GO = np.array([0.0, 0.5, 0.001])
# Grid half-extent = 1.5 * cell size (3 cells across the 3x3 board)
_GH = 0.06
# Cell edge length (visually aligned with so100_puzzle.xml grid_base)
_CS = 0.04
# 3x3 cell centers in world frame
GRID_CENTERS = np.array([
    [_GO[0] - _GH + _CS * (c + 0.5),  # cell center X
     _GO[1] - _GH + _CS * (r + 0.5),  # cell center Y
     0.012]                          # cell center Z (slightly above table)
    for r in range(3) for c in range(3)
], dtype=np.float64)


def xy_near_grid_cell(
    x: float,
    y: float,
    *,
    tol: float = 0.02,
) -> bool:
    """
    True if cube bottom-center XY lies near any placement cell center (aligned with ``GRID_CENTERS``).

    Used to tell cubes already on the grid from cubes still in the bin or in the air (pillars / task body pick).
    """
    for i in range(9):
        gc = GRID_CENTERS[i]
        if abs(x - float(gc[0])) <= tol and abs(y - float(gc[1])) <= tol:
            return True
    return False


def xy_in_source_bin(
    x: float,
    y: float,
    *,
    margin: float = 0.006,
) -> bool:
    """True if cube center lies in the source-bin scatter region (small margin)."""
    return (
        _SRC_X_RANGE[0] - margin <= x <= _SRC_X_RANGE[1] + margin
        and _SRC_Y_RANGE[0] - margin <= y <= _SRC_Y_RANGE[1] + margin
    )


# ------------------------- Source bin (core data) -------------------------
# Source bin center (XY)
_SRC_CENTER  = np.array([0.15, 0.42])
# Usable X extent inside the bin (interior)
_SRC_X_RANGE = (0.092, 0.208)
# Usable Y extent inside the bin (interior)
_SRC_Y_RANGE = (0.312, 0.528)
# Initial cube Z (just above bin floor)
CUBE_Z    = 0.012
# Cube half-edge 1 cm (full cube is 2 cm)
CUBE_HALF = 0.01
# Minimum safe gap between cube centers
MIN_GAP   = 0.008
# Extra inset from walls after half-thickness clearance (meters), reduces edge hugging
_SRC_PLACEMENT_EDGE_MARGIN = 0.012
# uniform_xy=False 3x3 layout: vertical row spacing = (bin height/2) * this factor; <1 tightens rows but still >= min gap
_SRC_GRID_Y_SPAN_FRAC = 0.78

# ------------------------- Joints (core state) -------------------------
# Arm DoF excluding gripper (5: Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll)
ARM_DOF        = 5
# Wrist roll index in qpos
IDX_WRIST_ROLL = 4
# Gripper joint index (6th actuator)
IDX_JAW        = 5

# Gripper command values (core control)
JAW_OPEN      = 0.45      # Wide open near bin for dense cube alignment
JAW_PRE_GRASP = 0.15      # Pre-grasp above bin (slightly larger than cube)
JAW_CLOSED    = 0.05      # Closed grasp
# On-grid release / lift / HOME use the same aperture: ramp only to here (avoid reopen-then-reclose)
JAW_HOME      = 0.08

# Default home joints (first 6 entries of so100_puzzle.xml keyframe home)
# Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw (=JAW_HOME)
HOME_QPOS = np.array([0.0, -2.0, 2.0, 1.58, 0.0, JAW_HOME], dtype=np.float64)

# Grasp center offset in ee_site local frame so IK targets match the physical pad center
GRASP_LOCAL_OFFSET = np.array([-0.002, -0.041, 0.0])

# Key Z heights for grasp motion (world Z)
PAD_SAFE_Z  = 0.12    # Safe travel height: clears bin rim
PAD_CARRY_Z = 0.09    # Carry height after grasp: slightly lower for compact motion
# Grasp height: pad center Z; slightly below cube center (CUBE_Z=0.012) for deeper wrap (0.014 was shallow)
PAD_GRASP_Z = 0.010
PAD_PLACE_Z = 0.010   # Place height (dynamic offset used in run())
# Hold at pre-grasp before closing to let vibration settle (simulation time)
PRE_CLOSE_PAUSE_SIM_SEC = 1.0

# IK / control integration
IK_DT         = 0.008  # IK time step per solve
CTRL_SUBSTEPS = 8      # Physics substeps per control tick (smooth sim)
# EE orientation weight: low default for alignment; higher in carry to keep tool vertical (decoupled from posture)
EE_ORIENTATION_COST_DEFAULT = 0.05
EE_ORIENTATION_COST_CARRY   = 0.95
# Post-place vertical lift: more steps + smoothstep end easing
PHASE8_LIFT_STEPS = 340


# ========================= Helpers (core logic) =========================

def _body_pos(model, data, name):
    """World position [X, Y, Z] of the named body."""
    return data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)].copy()

def _set_body_pos(model, data, name, pos):
    """Hard-set a body's world position by writing its free-joint qpos."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    ja = model.body_jntadr[bid]
    if ja < 0: return
    qa = model.jnt_qposadr[ja]
    data.qpos[qa:qa+3] = pos


def _set_free_body_pose(
    model,
    data,
    name: str,
    pos: np.ndarray,
    quat_wxyz: np.ndarray | None = None,
) -> None:
    """Free-joint cube: set position + quaternion and zero velocities."""
    if quat_wxyz is None:
        quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    ja = model.body_jntadr[bid]
    if ja < 0:
        return
    qa = model.jnt_qposadr[ja]
    data.qpos[qa : qa + 3] = pos
    data.qpos[qa + 3 : qa + 7] = quat_wxyz
    da = model.jnt_dofadr[ja]
    if da >= 0:
        data.qvel[da : da + 6] = 0.0


def _stash_cube_off_table(model, data, name: str, slot: int) -> None:
    """Move a cube off-screen (not part of the active bin set)."""
    pos = np.array(
        [
            _STASH_CUBES_X0 + slot * _STASH_CUBES_DX,
            _STASH_CUBES_Y,
            _STASH_CUBES_Z,
        ],
        dtype=np.float64,
    )
    _set_free_body_pose(model, data, name, pos)


def active_cube_triplets(num_cubes: int | None) -> list[tuple[str, int, str]]:
    """
    First K cubes in ``ALL_CUBE_NAMES`` order as ``(color, index_within_color, body_name)``.

    ``num_cubes is None`` or ``K >= TOTAL_CUBE_COUNT`` means all 9 cubes.
    """
    k = TOTAL_CUBE_COUNT if num_cubes is None else int(num_cubes)
    if k < 1 or k > TOTAL_CUBE_COUNT:
        raise ValueError(f"num_cubes must be in 1..{TOTAL_CUBE_COUNT}, got {k}")
    out: list[tuple[str, int, str]] = []
    for cn in ALL_CUBE_NAMES[:k]:
        for color, names in CUBE_BODIES.items():
            if cn in names:
                out.append((color, names.index(cn), cn))
                break
    return out


def active_color_counts_for_num_cubes(num_cubes: int | None) -> dict[str, int]:
    """Per-color instance counts among active cubes for ``num_cubes`` (same as ``active_cube_triplets``)."""
    from collections import Counter

    return dict(Counter(c for c, _, _ in active_cube_triplets(num_cubes)))


def active_color_cube_index_pairs(num_cubes: int | None) -> list[tuple[str, int]]:
    """List of ``(color, cube_index)`` for sampling in recording/eval."""
    return [(c, i) for c, i, _ in active_cube_triplets(num_cubes)]


def active_indices_for_color(color: str, num_cubes: int | None) -> list[int]:
    """Cube indices for ``color`` still in the active bin set (``num_cubes is None`` => all bodies of that color)."""
    if num_cubes is None:
        return list(range(len(CUBE_BODIES[color])))
    return sorted(i for c, i, _ in active_cube_triplets(num_cubes) if c == color)


def cube_index_for_repeat_mode(
    color: str, repeat_index: int, mode: str, num_cubes: int | None
) -> int:
    """Same first/cycle semantics as ``record_pick_place_uniform_grid``, respecting ``num_cubes`` truncation."""
    idxs = active_indices_for_color(color, num_cubes)
    if not idxs:
        raise ValueError(f"No active cube for color {color!r} with num_cubes={num_cubes}")
    if mode == "first":
        return idxs[0]
    if mode == "cycle":
        return idxs[repeat_index % len(idxs)]
    raise ValueError(mode)


def _source_cube_center_xy_bounds(*, near_target_center_ratio: float = 0.0):
    """Allowed XY range for cube centers in the bin (includes CUBE_HALF + ``_SRC_PLACEMENT_EDGE_MARGIN``)."""
    m = float(_SRC_PLACEMENT_EDGE_MARGIN)
    xmin = float(_SRC_X_RANGE[0] + CUBE_HALF + m)
    xmax = float(_SRC_X_RANGE[1] - CUBE_HALF - m)
    ymin = float(_SRC_Y_RANGE[0] + CUBE_HALF + m)
    ymax = float(_SRC_Y_RANGE[1] - CUBE_HALF - m)
    r = float(np.clip(near_target_center_ratio, 0.0, 0.95))
    if r > 0.0:
        # Pull layout toward the middle row of the nominal 3x3 bin grid: keep three X columns, tighten Y toward center row.
        y_mid = (ymin + ymax) / 2.0
        ymin = float(ymin + r * (y_mid - ymin))
        ymax = float(ymax + r * (y_mid - ymax))
    return xmin, xmax, ymin, ymax


def _source_grid_cell_centers_and_jitter_radius(*, near_target_center_ratio: float = 0.0):
    """
    3x3 source-bin cell centers and per-cell jitter radius.

    X: ``linspace(xmin,xmax,3)`` with neighbor spacing = bin width / 2.
    Y: three rows symmetric about bin Y mid; row spacing = (bin height/2) * ``_SRC_GRID_Y_SPAN_FRAC`` (slightly tighter than full height).
    ``mo`` is derived from neighbor spacing and ``2*CUBE_HALF+MIN_GAP``.
    """
    xmin, xmax, ymin, ymax = _source_cube_center_xy_bounds(
        near_target_center_ratio=near_target_center_ratio
    )
    cx = np.linspace(xmin, xmax, 3, dtype=np.float64)
    y_mid = (ymin + ymax) / 2.0
    y_half = (float(ymax - ymin) / 2.0) * float(_SRC_GRID_Y_SPAN_FRAC)
    cy = np.array([y_mid - y_half, y_mid, y_mid + y_half], dtype=np.float64)
    spacing_x = float(xmax - xmin) / 2.0
    spacing_y = y_half

    def _mo_for_axis(spacing: float) -> float:
        # If both neighbors jitter mo toward each other: spacing - 2*CUBE_HALF - 2*mo >= MIN_GAP
        half = (spacing - 2.0 * CUBE_HALF - MIN_GAP) / 2.0
        return max(0.001, half)

    mo = min(_mo_for_axis(spacing_x), _mo_for_axis(spacing_y))
    return cx, cy, float(mo)


def _scatter_cubes(
    model,
    data,
    rng,
    *,
    uniform_xy: bool = False,
    num_cubes: int | None = None,
    near_target_center_ratio: float = 0.0,
):
    """
    Scatter cubes inside the source bin.

    - ``num_cubes``: only first K bodies (``ALL_CUBE_NAMES`` order) stay in the bin; others stashed aside.
      ``None`` means all ``TOTAL_CUBE_COUNT`` cubes.
    - ``uniform_xy=False`` (default): sample K distinct cells among 9; full X span of three columns; row spacing from
      ``_SRC_GRID_Y_SPAN_FRAC``; jitter radius ``mo`` bounded by X/Y neighbor spacing.
    - ``uniform_xy=True``: uniform random XY in allowed rectangle with pairwise center distance ≥ ``2*CUBE_HALF + MIN_GAP``;
      on failure fall back to cell centers + jitter.
    - ``near_target_center_ratio`` in [0,1]: larger values pull Y toward the middle row of the nominal 3x3 layout.
    """
    k = TOTAL_CUBE_COUNT if num_cubes is None else int(num_cubes)
    if k < 1 or k > TOTAL_CUBE_COUNT:
        raise ValueError(f"num_cubes must be in 1..{TOTAL_CUBE_COUNT}, got {k}")

    active = ALL_CUBE_NAMES[:k]
    for slot, name in enumerate(ALL_CUBE_NAMES[k:]):
        _stash_cube_off_table(model, data, name, slot)

    if not uniform_xy:
        cx, cy, mo = _source_grid_cell_centers_and_jitter_radius(
            near_target_center_ratio=near_target_center_ratio
        )
        slots = rng.choice(9, size=k, replace=False)
        for i, cn in enumerate(active):
            ci = int(slots[i])
            _set_body_pos(
                model,
                data,
                cn,
                np.array(
                    [
                        cx[ci % 3] + rng.uniform(-mo, mo),
                        cy[ci // 3] + rng.uniform(-mo, mo),
                        CUBE_Z,
                    ]
                ),
            )
        mujoco.mj_forward(model, data)
        return

    xmin, xmax, ymin, ymax = _source_cube_center_xy_bounds(
        near_target_center_ratio=near_target_center_ratio
    )
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("Source-bin XY range too tight relative to cube half-size for uniform_xy scatter")
    min_dist = 2.0 * CUBE_HALF + MIN_GAP
    placed_xy: list[tuple[float, float]] = []
    cx, cy, mo = _source_grid_cell_centers_and_jitter_radius(
        near_target_center_ratio=near_target_center_ratio
    )
    max_tries = 4000

    for i, cn in enumerate(active):
        ok = False
        for _ in range(max_tries):
            x = float(rng.uniform(xmin, xmax))
            y = float(rng.uniform(ymin, ymax))
            if all(np.hypot(x - px, y - py) >= min_dist for px, py in placed_xy):
                placed_xy.append((x, y))
                _set_body_pos(model, data, cn, np.array([x, y, CUBE_Z]))
                ok = True
                break
        if ok:
            continue
        ci = i % 9
        fx = float(cx[ci % 3] + rng.uniform(-mo, mo))
        fy = float(cy[ci // 3] + rng.uniform(-mo, mo))
        placed_xy.append((fx, fy))
        _set_body_pos(model, data, cn, np.array([fx, fy, CUBE_Z]))

    mujoco.mj_forward(model, data)


def scatter_random_single_cube(
    model,
    data,
    rng: np.random.Generator,
    *,
    uniform_xy: bool = False,
    near_target_center_ratio: float = 0.0,
) -> tuple[str, int, str]:
    """
    Uniform random color in ``COLOR_NAMES``, then uniform random body among that color,
    placed in the bin; all others stashed off-screen.

    - ``uniform_xy=False``: one of nine source cells with small jitter.
    - ``uniform_xy=True``: uniform XY in allowed rectangle.
    - ``near_target_center_ratio`` in [0,1]: larger pulls toward middle row of nominal 3x3 layout (Y only).

    Returns ``(target_color, cube_index, body_name)`` for task strings and ``run_episode``.
    """
    color_key = rng.choice(COLOR_NAMES)
    color = str(color_key)
    names = CUBE_BODIES[color]
    cube_index = int(rng.integers(0, len(names)))
    active_name = names[cube_index]
    slot = 0
    for name in ALL_CUBE_NAMES:
        if name == active_name:
            continue
        _stash_cube_off_table(model, data, name, slot)
        slot += 1

    if uniform_xy:
        xmin, xmax, ymin, ymax = _source_cube_center_xy_bounds(
            near_target_center_ratio=near_target_center_ratio
        )
        if xmin >= xmax or ymin >= ymax:
            raise ValueError("Source-bin XY range too tight relative to cube half-size")
        x = float(rng.uniform(xmin, xmax))
        y = float(rng.uniform(ymin, ymax))
        _set_body_pos(model, data, active_name, np.array([x, y, CUBE_Z]))
    else:
        cx, cy, mo = _source_grid_cell_centers_and_jitter_radius(
            near_target_center_ratio=near_target_center_ratio
        )
        ci = int(rng.integers(0, 9))
        _set_body_pos(
            model,
            data,
            active_name,
            np.array(
                [
                    cx[ci % 3] + rng.uniform(-mo, mo),
                    cy[ci // 3] + rng.uniform(-mo, mo),
                    CUBE_Z,
                ]
            ),
        )
    mujoco.mj_forward(model, data)
    # Re-derive color/index from actual body name to stay consistent with MuJoCo state
    color_out, idx_out = color_and_index_for_body(active_name)
    return color_out, idx_out, active_name


def relocate_random_subset_to_target_grid(
    model,
    data,
    rng: np.random.Generator,
    active_cube_names: list[str] | tuple[str, ...],
    *,
    exclude_bodies_from_grid: tuple[str, ...] | list[str] = (),
    exclude_cells: tuple[int, ...] | list[int] = (),
) -> tuple[int, list[tuple[str, int]]]:
    """
    Call after bin scatter: sample N ~ Uniform{0,…,9}, then set
    ``n = min(N, #eligible bodies, #free cells)``; sample ``n`` bodies without replacement among **non-excluded**
    active bodies, ``n`` cells without replacement among **non-excluded** cells, and place cubes at ``GRID_CENTERS``
    cell centers (Z=``CUBE_Z``).

    - ``exclude_bodies_from_grid``: bodies that must **not** be moved onto the grid (usually the task cube).
    - ``exclude_cells``: cells that must stay empty (usually the instruction target cell).

    Returns ``(n, [(body_name, cell_index), …])``.
    """
    ex_b = frozenset(str(x) for x in exclude_bodies_from_grid)
    ex_c = frozenset(int(x) for x in exclude_cells)
    names = [str(x) for x in active_cube_names if str(x) not in ex_b]
    cells_avail = [c for c in range(9) if c not in ex_c]
    if not names or not cells_avail:
        return 0, []
    n_want = int(rng.integers(0, 10))
    n = min(n_want, len(names), len(cells_avail))
    if n <= 0:
        return 0, []
    ix_b = rng.choice(len(names), size=n, replace=False)
    cells_arr = np.asarray(cells_avail, dtype=int)
    chosen_cells = rng.choice(cells_arr, size=n, replace=False)
    pairs: list[tuple[str, int]] = []
    for i in range(n):
        cn = names[int(ix_b[i])]
        cell = int(chosen_cells[i])
        pos = GRID_CENTERS[cell].copy()
        pos[2] = float(CUBE_Z)
        _set_body_pos(model, data, cn, pos)
        pairs.append((cn, cell))
    mujoco.mj_forward(model, data)
    return n, pairs


def _build_aligned_downward_R(current_R):
    """
    Build a downward-pointing tool rotation with continuity in the grasp approach direction.

    Y axis fixed to world +Z; pick X in {[0,1,0],[0,-1,0]} along the bin long axis to limit wrist twist.
    """
    y_axis = np.array([0, 0, 1.0])
    cur_x = current_R[:, 0]
    # Pick X sign from current dominant X to avoid large flips
    if cur_x[1] >= 0:
        x_axis = np.array([0.0, 1.0, 0.0])
    else:
        x_axis = np.array([0.0, -1.0, 0.0])
    z_axis = np.cross(x_axis, y_axis)
    return np.column_stack([x_axis, y_axis, z_axis])

def _make_se3(pos, R):
    """Pack position and rotation into an SE3 object for mink."""
    M = np.eye(4)
    M[:3,:3] = R
    M[:3, 3] = pos
    return mink.SE3.from_matrix(M)

def _check_grasp(model, data, cube_name):
    """
    Contact check: scan contacts and see if the target cube touches either jaw geom
    ('Fixed_Jaw', 'Moving_Jaw').
    """
    cg = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{cube_name}_geom")
    if cg < 0: return False
    jg = set()
    for jn in ("Fixed_Jaw", "Moving_Jaw"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, jn)
        if bid >= 0:
            for g in range(model.ngeom):
                if model.geom_bodyid[g] == bid:
                    jg.add(g)
    # Scan all contacts
    for ci in range(data.ncon):
        g1, g2 = data.contact[ci].geom1, data.contact[ci].geom2
        # True if one geom is the cube and the other is a jaw
        if (g1==cg and g2 in jg) or (g2==cg and g1 in jg):
            return True
    return False


def grasp_phases_3_to_4_and_check(
    model,
    data,
    cube_name: str,
    target_xy: np.ndarray,
    *,
    get_pad,
    ik_step,
    settle,
    jaw_ramp,
    step_sim,
    verbose: bool = True,
) -> bool:
    """
    Phase 3: descend to pre-grasp height; Phase 3b: sim-time hold; Phase 4: close gripper; finger contact check.

    Shared by ``run``, ``record_pick_place_dataset_v3.run_episode``, and ``vla_to_actions``.
    """
    if verbose:
        print("Phase 3: pre-grasp descent...")
    n_descent = 150
    jaw_vals = jaw_ramp(JAW_OPEN, JAW_PRE_GRASP, n_descent)
    t3 = np.array([target_xy[0], target_xy[1], PAD_GRASP_Z])
    start_pos = get_pad()
    for step_i in range(n_descent):
        frac = (step_i + 1) / n_descent
        t_pos = start_pos + (t3 - start_pos) * frac
        t_pos[:2] = target_xy
        ik_step(t_pos, jaw_vals[step_i], use_posture=False)
    settle(t3, JAW_PRE_GRASP, n=10, use_posture=False)

    if verbose:
        print(f"Phase 3b: hold {PRE_CLOSE_PAUSE_SIM_SEC:.1f}s sim time before close...")
    t_hold0 = data.time
    t3_hold = np.array([target_xy[0], target_xy[1], PAD_GRASP_Z])
    while data.time - t_hold0 < PRE_CLOSE_PAUSE_SIM_SEC:
        ik_step(t3_hold, JAW_PRE_GRASP, use_posture=False)

    if verbose:
        print("Phase 4: close gripper & grasp...")
    n_close = 100
    jaw_close = jaw_ramp(JAW_PRE_GRASP, JAW_CLOSED, n_close)
    t4 = np.array([target_xy[0], target_xy[1], PAD_GRASP_Z])
    for i in range(n_close):
        ik_step(t4, jaw_close[i], use_posture=False)

    if verbose:
        print("  -> checking finger-cube contact before lift...")
    for jv in jaw_ramp(jaw_close[-1], JAW_CLOSED, 20):
        data.ctrl[IDX_JAW] = jv
        ik_step(t4, jv, use_posture=False)
    for _ in range(30):
        step_sim()
    if not _check_grasp(model, data, cube_name):
        if verbose:
            print("  -> grasp failed (no finger contact on cube)")
        return False
    return True


def compute_source_grid_3x3():
    """Ideal uniform 3x3 source-bin cell centers (same inset bounds as scatter)."""
    xmin, xmax, ymin, ymax = _source_cube_center_xy_bounds()
    xe = np.linspace(xmin, xmax, 4)
    ye = np.linspace(ymin, ymax, 4)
    cx = (xe[:-1]+xe[1:])/2
    cy = (ye[:-1]+ye[1:])/2
    grid_centers = np.array([
        [cx[c], cy[r], CUBE_Z]
        for r in range(3) for c in range(3)
    ], dtype=np.float64)
    return grid_centers

def place_cubes_in_grid(model, data, grid_centers, seed):
    """Place cubes exactly on the preset 3x3 centers inside the source bin."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(9)
    for i, cn in enumerate(ALL_CUBE_NAMES):
        if i >= 9: break
        ci = perm[i]
        _set_body_pos(model, data, cn, grid_centers[ci])
    mujoco.mj_forward(model, data)


def apply_home_pose(model, data):
    """Set arm + gripper to ``HOME_QPOS`` (matches so100_puzzle keyframe)."""
    data.qpos[:6] = HOME_QPOS.copy()
    data.ctrl[:6] = HOME_QPOS.copy()
    mujoco.mj_forward(model, data)


# ========================= Main pick-place flow =========================

def run(
    target_color: str,
    cube_index: int = 0,
    target_cell: int = 4,
    seed: int = 42,
    viewer_on: bool = True,
    speed: float = 0.5,
) -> bool:
    """Full pick-and-place episode."""
    rng   = np.random.default_rng(seed)
    # Load MuJoCo model and data
    model = mujoco.MjModel.from_xml_path(str(_XML))
    data  = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)
    _scatter_cubes(model, data, rng)
    mujoco.mj_forward(model, data)

    # Resolve target cube and placement cell
    cube_name = CUBE_BODIES[target_color][cube_index]
    place_xy  = GRID_CENTERS[target_cell][:2].copy()
    
    # Lay out cubes on a tidy 3x3 grid inside the bin for easier grasping
    grid_centers = compute_source_grid_3x3()
    place_cubes_in_grid(model, data, grid_centers, seed=seed)
    apply_home_pose(model, data)

    site_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")


    # --- mink IK setup ---
    config = mink.Configuration(model)
    config.update(data.qpos)

    # Task 1: track ee_site pose
    ee_task = mink.FrameTask(
        frame_name="ee_site", frame_type="site",
        position_cost=1.0,
        orientation_cost=EE_ORIENTATION_COST_DEFAULT,
        lm_damping=1.0,
    )
    # Task 2: posture regularization for smooth motion
    posture_task = mink.PostureTask(model=model, cost=1e-2)

    viewer = None
    if viewer_on:
        # Passive MuJoCo viewer
        viewer = mujoco.viewer.launch_passive(
            model, data, show_left_ui=False, show_right_ui=False)
            
    def step_sim(n=CTRL_SUBSTEPS):
        """Advance physics by n substeps (no kinematic cube glue)."""
        for _ in range(n):
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
        if viewer:
            viewer.sync()  # sync render
            # Slow playback when speed < 1
            if speed < 1.0:
                time.sleep(CTRL_SUBSTEPS * 0.002 * (1.0/speed - 1.0))

    def get_pad():
        """Physical grasp pad center in world frame."""
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        ee_pos = data.site_xpos[sid]
        ee_mat = data.site_xmat[sid].reshape(3,3)
        return ee_pos + ee_mat @ GRASP_LOCAL_OFFSET

    def get_ee_R():
        """Current ee_site orientation as 3x3 rotation matrix."""
        return data.site_xmat[site_id].reshape(3,3).copy()

    def ik_step(target_pad_pos, jaw, use_posture=True):
        """
        One IK solve + control apply: move grasp pad toward ``target_pad_pos``, output ``ctrl``.
        """
        # Refresh configuration from sim
        config.update(data.qpos)

        cur_R = get_ee_R()
        # Downward tool orientation aligned with approach
        down_R = _build_aligned_downward_R(cur_R)
        
        # Map pad target back to ee_site frame
        target_ee = target_pad_pos - down_R @ GRASP_LOCAL_OFFSET
        ee_task.set_target(_make_se3(target_ee, down_R))

        if use_posture:
            posture_task.set_target(data.qpos.copy())

        # Assemble IK tasks
        tasks = [ee_task, posture_task] if use_posture else [ee_task]
        # Joint velocities from IK
        vel = mink.solve_ik(config, tasks, IK_DT, "daqp", damping=1e-3)

        vel_full = np.zeros(model.nv)
        vel_full[:ARM_DOF] = vel[:ARM_DOF]  # arm joints only
        vel_full[IDX_JAW] = 0.0            # jaw not driven by IK here

        # Integrate configuration one IK step
        config.integrate_inplace(vel_full, IK_DT)

        # Arm: send integrated targets to actuators
        arm_target = config.q[:ARM_DOF].copy()
        data.ctrl[:ARM_DOF] = arm_target
        # Gripper: direct qpos/ctrl for crisp open/close (avoid actuator lag on grasp)
        data.qpos[IDX_JAW] = jaw
        data.ctrl[IDX_JAW] = jaw

        # Physics step
        step_sim()

    # --- Smooth motion helpers ---
    SETTLE_STEPS = 20  # Hold at target to damp residual oscillation

    def settle(target_pos, jaw, n=SETTLE_STEPS, use_posture=True):
        """Hold IK at target for n steps."""
        for _ in range(n):
            ik_step(target_pos, jaw, use_posture=use_posture)

    def jaw_ramp(start, end, steps):
        """Linear jaw trajectory."""
        return [start + (end - start) * i / max(steps - 1, 1) for i in range(steps)]

    def move_to_smooth(target_pos, jaw, steps, use_posture=True, smooth=False):
        """Straight-line pad motion; smoothstep easing if ``smooth``."""
        start_pos = get_pad()
        for i in range(steps):
            t = (i + 1) / max(steps, 1)
            u = t * t * (3.0 - 2.0 * t) if smooth else t
            t_pos = start_pos + (target_pos - start_pos) * u
            ik_step(t_pos, jaw, use_posture=use_posture)

    def move_to_qpos_smooth(target_q, jaw, steps):
        start_q = data.qpos[:ARM_DOF].copy()
        start_jaw = float(data.qpos[IDX_JAW])
        end_jaw = float(jaw)
        for i in range(steps):
            frac = (i + 1) / max(steps, 1)
            t_q = start_q + (target_q - start_q) * frac
            t_jaw = start_jaw + (end_jaw - start_jaw) * frac
            data.ctrl[:ARM_DOF] = t_q
            data.ctrl[IDX_JAW] = t_jaw
            data.qpos[IDX_JAW] = t_jaw
            step_sim()
            config.update(data.qpos)

    def settle_qpos(target_q, jaw, n):
        for _ in range(n):
            data.ctrl[:ARM_DOF] = target_q
            data.ctrl[IDX_JAW] = jaw
            data.qpos[IDX_JAW] = jaw
            step_sim()
            config.update(data.qpos)

    # =========================================================
    # Phase 2: from HOME_QPOS — translate over the target cube
    # =========================================================
    cp = _body_pos(model, data, cube_name)  # target cube XY
    print(f"Phase 2: align above target cube at {cp[:2]}...")
    target_xy = cp[:2].copy()
    
    # Nudge along gripper open direction (world +Y) so the cube sits deeper in the jaw workspace
    target_xy[1] += 0.005
    
    t2 = np.array([target_xy[0], target_xy[1], PAD_SAFE_Z])
    move_to_smooth(t2, JAW_OPEN, steps=300)
    settle(t2, JAW_OPEN, n=30)

    if not grasp_phases_3_to_4_and_check(
        model,
        data,
        cube_name,
        target_xy,
        get_pad=get_pad,
        ik_step=ik_step,
        settle=settle,
        jaw_ramp=jaw_ramp,
        step_sim=step_sim,
        verbose=True,
    ):
        if viewer:
            time.sleep(2)
            viewer.close()
        return False

    # =========================================================
    # Phase 5: vertical lift out of the bin
    # =========================================================
    print("Phase 5: lift up...")
    ee_task.orientation_cost = EE_ORIENTATION_COST_CARRY
    cur = get_pad()
    t5 = np.array([cur[0], cur[1], PAD_CARRY_Z])  # carry height
    move_to_smooth(t5, JAW_CLOSED, steps=200)
    settle(t5, JAW_CLOSED)

    # =========================================================
    # Phase 6: translate over target cell
    # =========================================================
    print("Phase 6: move to target cell...")
    cpos = _body_pos(model, data, cube_name)
    pad = get_pad()
    off = cpos - pad
    t6_pad_xy = place_xy[:2] - off[:2]
    t6_pad_xy[1] += 0.003
    t6 = np.array([t6_pad_xy[0], t6_pad_xy[1], PAD_CARRY_Z])
    move_to_smooth(t6, JAW_CLOSED, steps=350, use_posture=False)
    settle(t6, JAW_CLOSED, n=30, use_posture=False)

    # =========================================================
    # Phase 7: lower to place height and release
    # =========================================================
    print("Phase 7: lower & release...")
    pad_place_z = CUBE_Z + 0.003 - off[2]
    t7 = np.array([t6_pad_xy[0], t6_pad_xy[1], pad_place_z])
    move_to_smooth(t7, JAW_CLOSED, steps=150, use_posture=False)
    settle(t7, JAW_CLOSED, n=15, use_posture=False)

    ee_task.orientation_cost = EE_ORIENTATION_COST_DEFAULT

    n_release = 90
    for jv in jaw_ramp(JAW_CLOSED, JAW_HOME, n_release):
        ik_step(t7, jv, use_posture=False)
    for _ in range(60):
        step_sim()

    # =========================================================
    # Phase 8: clear lift then return home for another pick
    # =========================================================
    print("Phase 8: lift clear, then return to HOME_QPOS for next pick...")

    # 8a: lift vertically first to avoid sweeping the placed cube
    t8_lift = np.array([t6_pad_xy[0], t6_pad_xy[1], PAD_SAFE_Z])
    move_to_smooth(t8_lift, JAW_HOME, steps=PHASE8_LIFT_STEPS, use_posture=False, smooth=True)
    settle(t8_lift, JAW_HOME, n=22, use_posture=False)

    # 8b: joint-space blend back to full home (including jaw) before next Phase 2
    move_to_qpos_smooth(HOME_QPOS[:ARM_DOF], float(HOME_QPOS[IDX_JAW]), steps=200)
    settle_qpos(HOME_QPOS[:ARM_DOF], float(HOME_QPOS[IDX_JAW]), n=30)

    # --- Success check and viewer teardown ---
    final = _body_pos(model, data, cube_name)
    dx = abs(final[0] - place_xy[0])
    dy = abs(final[1] - place_xy[1])
    # Success if XY error < 1.5 cm
    ok = dx <= 0.015 and dy <= 0.015
    print(f"\nDone! cube=[{final[0]:.4f}, {final[1]:.4f}, {final[2]:.4f}]  "
          f"target=[{place_xy[0]:.4f}, {place_xy[1]:.4f}]  "
          f"dx={dx:.4f} dy={dy:.4f}  {'OK' if ok else 'FAIL'}")

    if viewer:
        time.sleep(3); viewer.close()
    return ok


# ========================= CLI entry =========================

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Pick-and-place grid script")
    # Target cube color
    p.add_argument("--color", default="orange", choices=COLOR_NAMES)
    # Instance index within that color
    p.add_argument("--cube-idx", type=int, default=2)
    # Target 3x3 cell index (0..8)
    p.add_argument("--cell", type=int, default=2, choices=range(9))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-viewer", action="store_true")      # disable viewer
    p.add_argument("--speed", type=float, default=1)      # viewer playback speed multiplier
    a = p.parse_args()

    if a.cube_idx >= len(CUBE_BODIES[a.color]):
        p.error(f"'{a.color}' only has {len(CUBE_BODIES[a.color])} cubes")

    run(target_color=a.color, cube_index=a.cube_idx, target_cell=a.cell,
        seed=a.seed, viewer_on=not a.no_viewer, speed=a.speed)
