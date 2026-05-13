"""
vla_to_actions.py — pick-place helpers in one module:

- **Single-cube demo** (``run_single_fixed``): fixed grasp pose and fixed place cell on
  ``gym_so100/assets/so100_puzzle.xml``; other cubes are moved aside.
- **Generic task wording** (``build_task_text_generic_cube``): colorless XVLA-style line.
- **Passive viewer overlays** (pillars at target cube + target cell via ``user_scn`` / offscreen scene).

Edit ``FIXED_TARGET_COLOR`` / ``FIXED_CUBE_INDEX`` / ``FIXED_TARGET_CELL`` as needed, then run
``python vla_to_actions.py``.
"""

from __future__ import annotations

import re
import time

import mujoco
import mujoco.viewer
import mink
import numpy as np

from sim_scenes import (
    ALL_CUBE_NAMES,
    COLOR_NAMES,
    CUBE_BODIES,
    CUBE_Z,
    GRID_CENTERS,
    color_and_index_for_body,
    ARM_DOF,
    IDX_JAW,
    JAW_OPEN,
    JAW_PRE_GRASP,
    JAW_CLOSED,
    JAW_HOME,
    GRASP_LOCAL_OFFSET,
    PAD_SAFE_Z,
    PAD_CARRY_Z,
    IK_DT,
    CTRL_SUBSTEPS,
    _XML,
    _body_pos,
    _build_aligned_downward_R,
    _make_se3,
    compute_source_grid_3x3,
    HOME_QPOS,
    apply_home_pose,
    EE_ORIENTATION_COST_DEFAULT,
    EE_ORIENTATION_COST_CARRY,
    PHASE8_LIFT_STEPS,
    grasp_phases_3_to_4_and_check,
    _STASH_CUBES_X0,
    _STASH_CUBES_Y,
    _STASH_CUBES_Z,
    _STASH_CUBES_DX,
    active_cube_triplets,
    xy_in_source_bin,
    xy_near_grid_cell,
)


# ---------------------------------------------------------------------------
# Generic task line (XVLA training phrasing)
# ---------------------------------------------------------------------------


def build_task_text_generic_cube(target_cell: int) -> str:
    """Colorless instruction (when only one cube is in the source bin, it refers to that cube)."""
    return f"Pick up the cube and place it in grid cell {int(target_cell)}."


# ---------------------------------------------------------------------------
# MuJoCo passive viewer overlays (vertical pillars at cube + cell)
# ---------------------------------------------------------------------------

_Z_LINE_BOTTOM = 0.002
_Z_LINE_TOP = 0.48
_BLUE_RGBA = np.array([0.12, 0.42, 1.0, 0.9], dtype=np.float32)
_WIDTH_PILLAR = 0.005

_GENERIC_TASK_RE = re.compile(
    r"^Pick up the cube and place it in grid cell (\d+)\.$",
    re.IGNORECASE,
)
_COLORED_TASK_RE = re.compile(
    r"^Pick up the (\w+) cube and place it in grid cell (\d+)\.$",
    re.IGNORECASE,
)


def infer_target_cell_from_perfect_pos(perfect_pos: np.ndarray) -> int:
    xy = np.asarray(perfect_pos, dtype=np.float64)[:2]
    best_i = 0
    best_d = float("inf")
    for i in range(9):
        d = float(np.linalg.norm(xy - GRID_CENTERS[i, :2]))
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def _body_sort_key_for_guide(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> tuple[int, int]:
    """
    Prefer cubes in the source bin and not on a cell center; same-color distractors already on the grid go last.
    """
    pos = _body_pos(model, data, body_name)
    x, y = float(pos[0]), float(pos[1])
    on_grid = xy_near_grid_cell(x, y)
    in_src = xy_in_source_bin(x, y)
    if not on_grid and in_src:
        tier = 0
    elif not on_grid:
        tier = 1
    else:
        tier = 2
    try:
        ni = ALL_CUBE_NAMES.index(body_name)
    except ValueError:
        ni = 9999
    return (tier, ni)


def _pick_body_for_color(
    color: str,
    num_cubes: int | None,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> str:
    names = [t[2] for t in active_cube_triplets(num_cubes) if t[0] == color]
    if not names:
        raise ValueError(f"No active cube for color {color!r}")
    return min(names, key=lambda n: _body_sort_key_for_guide(model, data, n))


def _pick_body_generic(
    num_cubes: int | None,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> str:
    trips = active_cube_triplets(num_cubes)
    if not trips:
        raise ValueError("generic task: no active cubes")
    bodies = [t[2] for t in trips]
    return min(bodies, key=lambda n: _body_sort_key_for_guide(model, data, n))


def parse_task_for_visual_guide(
    task: str,
    num_cubes: int | None,
    model: mujoco.MjModel | None = None,
    data: mujoco.MjData | None = None,
) -> tuple[str, int]:
    """Parse task string to (body_name, target_cell); raises ValueError on failure.

    If ``model`` / ``data`` are given, when multiple bodies share a color prefer the instance
    **in the source bin and not on a cell center**, so the pillar does not land on a distractor pre-placed on the grid.
    """
    t = task.strip()
    mg = _GENERIC_TASK_RE.match(t)
    if mg:
        cell = int(mg.group(1))
        if not (0 <= cell <= 8):
            raise ValueError(f"cell must be in 0..8: {cell}")
        if model is not None and data is not None:
            return _pick_body_generic(num_cubes, model, data), cell
        trips = active_cube_triplets(num_cubes)
        if not trips:
            raise ValueError("generic task: no active cubes")
        return trips[0][2], cell
    mc = _COLORED_TASK_RE.match(t)
    if not mc:
        raise ValueError(f"cannot parse task string: {task!r}")
    color = mc.group(1).lower()
    cell = int(mc.group(2))
    if color not in CUBE_BODIES:
        raise ValueError(f"unknown color: {color!r}")
    if not (0 <= cell <= 8):
        raise ValueError(f"cell must be in 0..8: {cell}")
    if model is not None and data is not None:
        return _pick_body_for_color(color, num_cubes, model, data), cell
    for c, _i, name in active_cube_triplets(num_cubes):
        if c == color:
            return name, cell
    raise ValueError(f"No active cube for color {color!r}")


def _fill_task_guide_geoms(
    geoms,
    idx_start: int,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    cube_name: str,
    target_cell: int,
) -> None:
    """Write two vertical pillars into geoms[idx_start : idx_start+2] (cube center + cell center, matches viewer user_scn)."""
    pos = _body_pos(model, data, cube_name)
    px, py = float(pos[0]), float(pos[1])
    cube_lo = np.array([px, py, _Z_LINE_BOTTOM], dtype=np.float64)
    cube_hi = np.array([px, py, _Z_LINE_TOP], dtype=np.float64)
    gc = GRID_CENTERS[int(target_cell)]
    cx, cy = float(gc[0]), float(gc[1])
    cell_lo = np.array([cx, cy, _Z_LINE_BOTTOM], dtype=np.float64)
    cell_hi = np.array([cx, cy, _Z_LINE_TOP], dtype=np.float64)
    ident = np.eye(3, dtype=np.float64).flatten()

    def put_capsule(idx: int, p0: np.ndarray, p1: np.ndarray, rgba: np.ndarray, width: float) -> None:
        mujoco.mjv_initGeom(
            geoms[idx],
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            ident,
            rgba,
        )
        mujoco.mjv_connector(
            geoms[idx],
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            float(width),
            p0,
            p1,
        )

    put_capsule(idx_start, cube_lo, cube_hi, _BLUE_RGBA, _WIDTH_PILLAR)
    put_capsule(idx_start + 1, cell_lo, cell_hi, _BLUE_RGBA, _WIDTH_PILLAR)


def append_task_guide_geoms_to_scene(
    scn: mujoco.MjvScene,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    cube_name: str,
    target_cell: int,
    enabled: bool,
) -> None:
    """
    Call after ``mjv_updateScene`` and before ``mjr_render``: append two decorative capsules so
    offscreen ``Renderer`` output matches passive viewer guides.
    """
    if not enabled:
        return
    base = int(scn.ngeom)
    maxg = len(scn.geoms)
    if base + 2 > maxg:
        return
    _fill_task_guide_geoms(
        scn.geoms, base, model, data, cube_name=cube_name, target_cell=target_cell
    )
    scn.ngeom = base + 2


def sync_viewer_task_guides(
    viewer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    cube_name: str,
    target_cell: int,
    enabled: bool,
) -> None:
    """
    Call before ``viewer.sync()`` (this function uses ``with viewer.lock():`` internally).
    When ``enabled=False``, clear user_scn decorations to avoid stale overlays.
    """
    if not hasattr(viewer, "user_scn"):
        return
    with viewer.lock():
        scn = viewer.user_scn
        if not enabled:
            scn.ngeom = 0
            return
        _fill_task_guide_geoms(
            scn.geoms, 0, model, data, cube_name=cube_name, target_cell=target_cell
        )
        scn.ngeom = 2


# ---------------------------------------------------------------------------
# Single-cube fixed layout demo
# ---------------------------------------------------------------------------

# Cube to grasp: color must exist in CUBE_BODIES; index is instance index within that color.
FIXED_TARGET_COLOR = "green"
FIXED_CUBE_INDEX = 0

# Target 3x3 cell index 0..8 (same order as GRID_CENTERS: row-major).
FIXED_TARGET_CELL = 4

# Grasp pose: center of source-bin 3x3 slot; index 4 is the box center cell.
FIXED_PICK_XYZ = compute_source_grid_3x3()[4].copy()


def fixed_cube_body_name() -> str:
    return CUBE_BODIES[FIXED_TARGET_COLOR][FIXED_CUBE_INDEX]


def sample_random_single_fixed_target_body(rng: np.random.Generator) -> str:
    """
    Recording helper: uniform random among four colors, then uniform random body within that color;
    grasp world position stays ``FIXED_PICK_XYZ`` (same color sampling as ``scatter_random_single_cube``).
    """
    color = str(rng.choice(COLOR_NAMES))
    names = CUBE_BODIES[color]
    idx = int(rng.integers(0, len(names)))
    return names[idx]


def build_task_text_single_fixed_for_body(
    body_name: str, target_cell: int | None = None
) -> str:
    """Task sentence for single-fixed layout (grasp pose fixed by scene)."""
    color, _ = color_and_index_for_body(body_name)
    cell = FIXED_TARGET_CELL if target_cell is None else int(target_cell)
    return f"Pick up the {color} cube and place it in grid cell {cell}."


def build_task_text_single_fixed() -> str:
    return build_task_text_single_fixed_for_body(
        fixed_cube_body_name(), FIXED_TARGET_CELL
    )


def _set_free_body_pose(model, data, name: str, pos: np.ndarray, quat_wxyz=None):
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


def setup_single_fixed_cubes(model, data, *, target_body_name: str | None = None) -> None:
    """
    Place the target cube at ``FIXED_PICK_XYZ``; move all others off-screen.

    If ``target_body_name`` is ``None``, use ``fixed_cube_body_name()`` (default green demo).
    """
    target = fixed_cube_body_name() if target_body_name is None else target_body_name
    _set_free_body_pose(model, data, target, FIXED_PICK_XYZ.copy())

    others = [n for n in ALL_CUBE_NAMES if n != target]
    for i, name in enumerate(others):
        pos = np.array(
            [
                _STASH_CUBES_X0 + i * _STASH_CUBES_DX,
                _STASH_CUBES_Y,
                _STASH_CUBES_Z,
            ],
            dtype=np.float64,
        )
        _set_free_body_pose(model, data, name, pos)

    mujoco.mj_forward(model, data)


def run_single_fixed(seed: int = 0, viewer_on: bool = True, speed: float = 1.0) -> bool:
    """Same pipeline as ``sim_scenes.run``, but with a fixed single-cube layout (easy visual check)."""
    _ = seed  # Scene is deterministic; keep signature close to sim_scenes.run
    model = mujoco.MjModel.from_xml_path(str(_XML))
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)
    setup_single_fixed_cubes(model, data)
    apply_home_pose(model, data)

    cube_name = fixed_cube_body_name()
    place_xy = GRID_CENTERS[FIXED_TARGET_CELL][:2].copy()

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    config = mink.Configuration(model)
    config.update(data.qpos)

    ee_task = mink.FrameTask(
        frame_name="ee_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=EE_ORIENTATION_COST_DEFAULT,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model=model, cost=1e-2)

    viewer = None
    if viewer_on:
        viewer = mujoco.viewer.launch_passive(
            model, data, show_left_ui=False, show_right_ui=False
        )

    def step_sim(n=CTRL_SUBSTEPS):
        for _ in range(n):
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
        if viewer:
            viewer.sync()
            if speed < 1.0:
                time.sleep(CTRL_SUBSTEPS * 0.002 * (1.0 / speed - 1.0))

    def get_pad():
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        ee_pos = data.site_xpos[sid]
        ee_mat = data.site_xmat[sid].reshape(3, 3)
        return ee_pos + ee_mat @ GRASP_LOCAL_OFFSET

    def get_ee_R():
        return data.site_xmat[site_id].reshape(3, 3).copy()

    def ik_step(target_pad_pos, jaw, use_posture=True):
        config.update(data.qpos)
        cur_R = get_ee_R()
        down_R = _build_aligned_downward_R(cur_R)
        target_ee = target_pad_pos - down_R @ GRASP_LOCAL_OFFSET
        ee_task.set_target(_make_se3(target_ee, down_R))
        if use_posture:
            posture_task.set_target(data.qpos.copy())
        tasks = [ee_task, posture_task] if use_posture else [ee_task]
        vel = mink.solve_ik(config, tasks, IK_DT, "daqp", damping=1e-3)
        vel_full = np.zeros(model.nv)
        vel_full[:ARM_DOF] = vel[:ARM_DOF]
        vel_full[IDX_JAW] = 0.0
        config.integrate_inplace(vel_full, IK_DT)
        arm_target = config.q[:ARM_DOF].copy()
        data.ctrl[:ARM_DOF] = arm_target
        data.qpos[IDX_JAW] = jaw
        data.ctrl[IDX_JAW] = jaw
        step_sim()

    SETTLE_STEPS = 20

    def settle(target_pos, jaw, n=SETTLE_STEPS, use_posture=True):
        for _ in range(n):
            ik_step(target_pos, jaw, use_posture=use_posture)

    def jaw_ramp(start, end, steps):
        return [start + (end - start) * i / max(steps - 1, 1) for i in range(steps)]

    def move_to_smooth(target_pos, jaw, steps, use_posture=True, smooth=False):
        start_pos = get_pad()
        for i in range(steps):
            t = (i + 1) / max(steps, 1)
            u = t * t * (3.0 - 2.0 * t) if smooth else t
            t_pos = start_pos + (target_pos - start_pos) * u
            ik_step(t_pos, jaw, use_posture=use_posture)

    def move_to_smooth_jaw_interp(target_pos, jaw_start, jaw_end, steps, use_posture=True, smooth=False):
        start_pos = get_pad()
        jaw_start = float(jaw_start)
        jaw_end = float(jaw_end)
        for i in range(steps):
            t = (i + 1) / max(steps, 1)
            u = t * t * (3.0 - 2.0 * t) if smooth else t
            t_pos = start_pos + (target_pos - start_pos) * u
            t_jaw = jaw_start + (jaw_end - jaw_start) * u
            ik_step(t_pos, t_jaw, use_posture=use_posture)

    def move_to_qpos_smooth(target_q, jaw_target, steps, smooth_blend: bool = True):
        start_q = data.qpos[:ARM_DOF].copy()
        start_jaw = float(data.qpos[IDX_JAW])
        end_jaw = float(jaw_target)
        jaw_span = end_jaw - start_jaw
        tgt_q = np.asarray(target_q, dtype=np.float64)
        for i in range(steps):
            t_lin = (i + 1) / max(steps, 1)
            u = t_lin * t_lin * (3.0 - 2.0 * t_lin) if smooth_blend else t_lin
            t_q = start_q + (tgt_q - start_q) * u
            t_jaw = end_jaw if abs(jaw_span) < 1e-9 else start_jaw + jaw_span * u
            data.ctrl[:ARM_DOF] = t_q
            data.ctrl[IDX_JAW] = t_jaw
            data.qpos[IDX_JAW] = t_jaw
            step_sim()
            config.update(data.qpos)

    def settle_qpos(target_q, jaw, n):
        target_q = np.asarray(target_q, dtype=np.float64)
        jaw = float(jaw)
        for _ in range(n):
            data.ctrl[:ARM_DOF] = target_q
            data.ctrl[IDX_JAW] = jaw
            data.qpos[IDX_JAW] = jaw
            step_sim()
            config.update(data.qpos)

    cp = _body_pos(model, data, cube_name)
    print(f"Phase 2: align above target cube at {cp[:2]}...")
    target_xy = cp[:2].copy()
    target_xy[1] += 0.005

    t2 = np.array([target_xy[0], target_xy[1], PAD_SAFE_Z])
    jaw_at_phase2 = float(data.qpos[IDX_JAW])
    move_to_smooth_jaw_interp(t2, jaw_at_phase2, JAW_OPEN, steps=300, use_posture=True, smooth=False)
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

    print("Phase 5: lift up...")
    ee_task.orientation_cost = EE_ORIENTATION_COST_CARRY
    cur = get_pad()
    t5 = np.array([cur[0], cur[1], PAD_CARRY_Z])
    move_to_smooth(t5, JAW_CLOSED, steps=200)
    settle(t5, JAW_CLOSED)

    print("Phase 6: move to target cell...")
    cpos = _body_pos(model, data, cube_name)
    pad = get_pad()
    off = cpos - pad
    t6_pad_xy = place_xy[:2] - off[:2]
    t6_pad_xy[1] += 0.003
    t6 = np.array([t6_pad_xy[0], t6_pad_xy[1], PAD_CARRY_Z])
    move_to_smooth(t6, JAW_CLOSED, steps=350, use_posture=False)
    settle(t6, JAW_CLOSED, n=30, use_posture=False)

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

    print("Phase 8: lift clear, then return to HOME_QPOS...")
    t8_lift = np.array([t6_pad_xy[0], t6_pad_xy[1], PAD_SAFE_Z])
    move_to_smooth(t8_lift, JAW_HOME, steps=PHASE8_LIFT_STEPS, use_posture=False, smooth=True)
    settle(t8_lift, JAW_HOME, n=22, use_posture=False)

    home_arm = HOME_QPOS[:ARM_DOF].copy()
    home_jaw = float(HOME_QPOS[IDX_JAW])
    move_to_qpos_smooth(home_arm, home_jaw, steps=240, smooth_blend=True)
    settle_qpos(home_arm, home_jaw, n=30)

    final = _body_pos(model, data, cube_name)
    dx = abs(final[0] - place_xy[0])
    dy = abs(final[1] - place_xy[1])
    ok = dx <= 0.015 and dy <= 0.015
    print(
        f"\nDone! cube=[{final[0]:.4f}, {final[1]:.4f}, {final[2]:.4f}]  "
        f"target=[{place_xy[0]:.4f}, {place_xy[1]:.4f}]  "
        f"dx={dx:.4f} dy={dy:.4f}  {'OK' if ok else 'FAIL'}"
    )

    if viewer:
        time.sleep(3)
        viewer.close()
    return ok


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Fixed single-cube pick-and-place demo")
    p.add_argument("--no-viewer", action="store_true")
    p.add_argument("--speed", type=float, default=1.0)
    a = p.parse_args()
    run_single_fixed(viewer_on=not a.no_viewer, speed=a.speed)
