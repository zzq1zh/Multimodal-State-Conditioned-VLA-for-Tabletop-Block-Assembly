"""
Run a trained XVLA policy in the MuJoCo pick-place scene used for recording.

**Default mode** (no instruction flags): random per-episode tasks.

**Instruction-sequence mode**: exactly one of ``--instructions-file``, ``--instructions-json``, or
``--from-vlm-text`` (uses **OpenAI-compatible VLM** via ``vlm_to_actions.py``).

Requires: conda env ``lerobot``, mink, mujoco, xvla; **OPENAI_API_KEY** only for ``--from-vlm-text`` (OpenAI-compatible VLM planning).

Examples::

  python sim_eval.py --policy-id USER/HF_REPO --dataset-root /path/to/lerobot_dataset --render
  python sim_eval.py --instructions-file /tmp/plan.json --policy-id USER/HF_REPO --dataset-root ...

``--video-out`` uses ``--video-camera`` (default ``cam_global``). ``--visual-task-guides`` adds target pillars.

**JSON**: by default writes a pretty-printed summary to ``sim_eval_seed{seed}_episodes{N}.json`` (pick-place) or
``sim_eval_seed{seed}_instruction_sequence.json`` (instruction mode); prints one compact JSON line to stdout unless
``--no-json-summary-line``. Use ``--json-out PATH`` or ``--no-json-out-file`` to override.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

from sim_scenes import (
    ALL_CUBE_NAMES,
    CUBE_BODIES,
    CUBE_Z,
    GRID_CENTERS,
    TOTAL_CUBE_COUNT,
    _body_pos,
    active_color_counts_for_num_cubes,
    active_cube_triplets,
    _scatter_cubes,
    apply_home_pose,
    relocate_random_subset_to_target_grid,
    scatter_random_single_cube,
    build_task_text_for_body,
    _XML,
)
from vla_adapter import (
    GraspAssistState,
    apply_action_to_sim,
    build_pick_place_obs_dict_for_xvla,
    hub_or_posix_path,
    load_policy_and_processors,
    physics_substeps_with_grasp,
    policy_physics_substeps_per_decision,
    policy_vector_to_so100_action,
    resolve_local_policy_dir,
    resolve_lerobot_dataset_root_for_eval,
    xvla_image_step_presence,
)
from vla_to_actions import (
    append_task_guide_geoms_to_scene,
    build_task_text_generic_cube,
    parse_task_for_visual_guide,
    sync_viewer_task_guides,
)
from vlm_to_actions import (
    instruction_color_counts,
    load_env,
    plan_vla_instructions,
    validate_instructions,
    validate_instructions_against_vlm_inventory,
)


def _parse_action_slice(spec: str | None):
    if not spec:
        return None
    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid --action-slice {spec!r}, expected START:END")
    return slice(int(parts[0]), int(parts[1]))


def build_obs_dict(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    task: str,
    *,
    has_image_to_float: bool = True,
    has_imagenet_normalize: bool = True,
    policy_config: Any | None = None,
    visual_task_guides: bool = False,
    guide_cube_name: str | None = None,
    guide_target_cell: int | None = None,
) -> dict:
    """Delegates to ``vla_adapter.build_pick_place_obs_dict_for_xvla`` (pick_place qpos prefix + policy state dim)."""
    return build_pick_place_obs_dict_for_xvla(
        renderer,
        data,
        task,
        has_image_to_float=has_image_to_float,
        has_imagenet_normalize=has_imagenet_normalize,
        policy_config=policy_config,
        visual_task_guides=visual_task_guides,
        guide_cube_name=guide_cube_name,
        guide_target_cell=guide_target_cell,
    )


_PLACE_XY_TOL = 0.015
_PLACE_Z_TOL = 0.015
_EE_LEFT_GRID_MARGIN = 0.01


def _placement_xy_ok(model: mujoco.MjModel, data: mujoco.MjData, cube_name: str, target_cell: int) -> bool:
    """True when cube center is within XYZ tolerance of target grid cell on tabletop."""
    pos = _body_pos(model, data, cube_name)
    gc = GRID_CENTERS[target_cell]
    return bool(
        abs(float(pos[0]) - float(gc[0])) <= _PLACE_XY_TOL
        and abs(float(pos[1]) - float(gc[1])) <= _PLACE_XY_TOL
        and abs(float(pos[2]) - float(CUBE_Z)) <= _PLACE_Z_TOL
    )


def _cube_touches_jaw(model: mujoco.MjModel, data: mujoco.MjData, cube_name: str) -> bool:
    """True when target cube is currently in contact with either jaw."""
    cube_geoms: set[int] = set()
    cg = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{cube_name}_geom"))
    if cg >= 0:
        cube_geoms.add(cg)
    cube_bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cube_name))
    if cube_bid >= 0:
        for g in range(int(model.ngeom)):
            if int(model.geom_bodyid[g]) == cube_bid:
                cube_geoms.add(g)
    if not cube_geoms:
        return False

    jaw_geoms: set[int] = set()
    for jaw_body in ("Fixed_Jaw", "Moving_Jaw"):
        jaw_bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, jaw_body))
        if jaw_bid >= 0:
            for g in range(int(model.ngeom)):
                if int(model.geom_bodyid[g]) == jaw_bid:
                    jaw_geoms.add(g)
    if not jaw_geoms:
        return False

    for ci in range(int(data.ncon)):
        c = data.contact[ci]
        g1, g2 = int(c.geom1), int(c.geom2)
        if (g1 in cube_geoms and g2 in jaw_geoms) or (g2 in cube_geoms and g1 in jaw_geoms):
            return True
    return False


def _ee_has_left_grid(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    """
    True when ee_site XY has moved outside the 3x3 grid footprint (with a small margin).
    """
    sid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"))
    if sid < 0:
        # If ee_site is unavailable, do not block early-stop on this condition.
        return True
    ee = data.site_xpos[sid]
    ex, ey = float(ee[0]), float(ee[1])

    xs = [float(gc[0]) for gc in GRID_CENTERS]
    ys = [float(gc[1]) for gc in GRID_CENTERS]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    ux = sorted(set(xs))
    uy = sorted(set(ys))
    dx = min((abs(ux[i + 1] - ux[i]) for i in range(len(ux) - 1)), default=0.04)
    dy = min((abs(uy[i + 1] - uy[i]) for i in range(len(uy) - 1)), default=0.04)
    half_cell_x = 0.5 * float(dx)
    half_cell_y = 0.5 * float(dy)

    x_in = (min_x - half_cell_x - _EE_LEFT_GRID_MARGIN) <= ex <= (max_x + half_cell_x + _EE_LEFT_GRID_MARGIN)
    y_in = (min_y - half_cell_y - _EE_LEFT_GRID_MARGIN) <= ey <= (max_y + half_cell_y + _EE_LEFT_GRID_MARGIN)
    return not (x_in and y_in)


def _placement_xy_ok_and_released(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cube_name: str,
    target_cell: int,
) -> bool:
    # Early-stop success means:
    # 1) cube reached target cell, 2) gripper no longer touching cube, 3) ee_site has left the grid area.
    return (
        _placement_xy_ok(model, data, cube_name, target_cell)
        and (not _cube_touches_jaw(model, data, cube_name))
        and _ee_has_left_grid(model, data)
    )


def _min_num_cubes_for_color_needs(need: Counter[str]) -> int | None:
    """Smallest K with ``active_cube_triplets(K)`` satisfying ``need``; None if impossible."""
    if not need:
        return 1
    for k in range(1, TOTAL_CUBE_COUNT + 1):
        sup = Counter(c for c, _, _ in active_cube_triplets(k))
        if all(sup.get(col, 0) >= need[col] for col in need):
            return k
    return None


def _resolve_num_cubes_for_instructions(
    instructions: list[str],
    *,
    num_cubes: int | None,
    single_random_cube: bool,
    generic_cube_instruction: bool,
) -> tuple[str | None, int | None]:
    """
    In multi-cube mode, validate ``--num-cubes`` against instruction color needs; **does not** auto-adjust K.

    - If ``--num-cubes`` is omitted: return ``None`` for effective count so scatter uses all cubes (same as ``_scatter_cubes``).
    - Explicit K: must satisfy ``K >=`` minimum prefix length for color counts; otherwise return an error string.
    - Single-cube mode: leave ``num_cubes`` unchanged; multiple instructions still error.

    Returns ``(error_message, effective_num_cubes)``.
    """
    if single_random_cube or generic_cube_instruction:
        if len(instructions) > 1:
            return (
                "Single-cube mode (--single-random-cube / --generic-cube-instruction) starts with one cube in the source; "
                "for multiple instructions use multi-cube (drop those flags, default or explicit --num-cubes 9).",
                None,
            )
        return (None, num_cubes)

    need = instruction_color_counts(instructions)
    if not need:
        return (None, num_cubes)

    for c, n in need.items():
        mx = len(CUBE_BODIES.get(c, ()))
        if n > mx:
            return (
                f"Instructions need color {c!r} for {n} picks but the scene has at most {mx} cubes of that color; reduce steps for this color.",
                None,
            )

    min_k = _min_num_cubes_for_color_needs(need)
    if min_k is None:
        return (
            "Cannot satisfy instruction color mix under default cube order; check instructions vs ALL_CUBE_NAMES order.",
            None,
        )

    if num_cubes is None:
        return (None, None)

    rq = int(num_cubes)
    if rq < 1 or rq > TOTAL_CUBE_COUNT:
        return (f"--num-cubes must be in 1..{TOTAL_CUBE_COUNT}", None)

    if rq < min_k:
        return (
            f"Under ALL_CUBE_NAMES order, satisfying current instruction color counts needs at least the first {min_k} cubes in the source; "
            f"--num-cubes={rq} is too small; set >= {min_k} or change the plan.",
            None,
        )

    return (None, rq)


def _append_sim_inventory_to_prompt(
    user_text: str,
    sim_active_color_counts: dict[str, int] | None,
) -> str:
    """
    Append per-color inventory from the sim initial state to the user prompt so OpenAI-compatible VLM planning respects stock.
    """
    if not sim_active_color_counts:
        return str(user_text)
    text = str(user_text).strip()
    if "Current cube inventory" in text:
        return text
    cols = ("yellow", "green", "purple", "orange")
    inv = ", ".join(f"{c}={max(0, int(sim_active_color_counts.get(c, 0)))}" for c in cols)
    grid_desc = (
        "3x3 grid cell numbering (top view):\n"
        "- Top row: 0 (top-left), 1 (top-center), 2 (top-right)\n"
        "- Middle row: 3 (middle-left), 4 (center), 5 (middle-right)\n"
        "- Bottom row: 6 (bottom-left), 7 (bottom-center), 8 (bottom-right)"
    )
    suffix = (
        "HARD CONSTRAINTS FOR PLANNING (MUST FOLLOW):\n"
        f"- Current cube inventory in this scene: {inv}\n"
        "- The total number of pick instructions for each color MUST be <= its inventory above.\n"
        "- If any color usage exceeds inventory, the plan is INVALID.\n"
        "- You MUST revise the plan to satisfy the inventory limits before output.\n\n"
        f"{grid_desc}"
    )
    if not text:
        return suffix
    return f"{text}\n\n{suffix}"


_VLM_SNAPSHOT_CAMERAS: tuple[str, ...] = ("cam_global", "cam_oblique", "left_wrist")


def _cleanup_vlm_sim_image_paths(paths: list[Path]) -> None:
    for p in paths:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            if p.is_file():
                p.unlink()


def _write_vlm_sim_initial_images(
    renderer: mujoco.Renderer,
    mj_data: mujoco.MjData,
) -> list[Path]:
    """Same three cameras as XVLA observations: cam_global / cam_oblique / left_wrist → temporary PNG files."""
    tag = uuid.uuid4().hex[:12]
    tmp = Path(tempfile.gettempdir())
    paths: list[Path] = []
    for cam in _VLM_SNAPSHOT_CAMERAS:
        renderer.update_scene(mj_data, camera=cam)
        rgb = renderer.render()
        p = tmp / f"gym_so100_vlm_sim_{tag}_{cam}.png"
        imageio.imwrite(str(p), rgb)
        paths.append(p)
    return paths


def _extract_pattern_text_from_raw(raw: str) -> str | None:
    t = str(raw).strip()
    if t.startswith("```"):
        t = t.removeprefix("```json").removeprefix("```JSON").removeprefix("```").strip()
        if t.endswith("```"):
            t = t[:-3].strip()
    try:
        obj = json.loads(t)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    cand = obj.get("pattern")
    if isinstance(cand, str) and cand.strip():
        return cand.strip()
    cand2 = obj.get("pattern_text")
    if isinstance(cand2, str) and cand2.strip():
        return cand2.strip()
    return None


def _build_initial_scene_for_vlm(
    args: argparse.Namespace,
) -> tuple[mujoco.MjModel, mujoco.MjData, mujoco.Renderer, list[Path], int]:
    """
    Build the same style of initial scatter as episode step 1 before calling the VLM, then render three cameras.

    Uses ``--seed`` / ``--num-cubes`` / scatter flags; **does not** apply ``--random-target-grid-cubes`` pre-placement
    (that depends on the task string; skipped in preflight).

    Returns ``(model, data, renderer, temp_png_paths, k_used)``.
    """
    mj_model = mujoco.MjModel.from_xml_path(str(_XML))
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=480, width=640)
    rng = np.random.default_rng(int(args.seed))
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    k0 = int(args.num_cubes) if args.num_cubes is not None else TOTAL_CUBE_COUNT
    if args.single_random_cube or args.generic_cube_instruction:
        scatter_random_single_cube(
            mj_model,
            mj_data,
            rng,
            uniform_xy=args.scatter_uniform_xy,
            near_target_center_ratio=args.near_target_center_ratio,
        )
    else:
        _scatter_cubes(
            mj_model,
            mj_data,
            rng,
            uniform_xy=args.scatter_uniform_xy,
            num_cubes=k0,
            near_target_center_ratio=args.near_target_center_ratio,
        )
    apply_home_pose(mj_model, mj_data)
    mujoco.mj_forward(mj_model, mj_data)
    paths = _write_vlm_sim_initial_images(renderer, mj_data)
    return mj_model, mj_data, renderer, paths, k0


def _refresh_initial_scene_for_vlm(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    renderer: mujoco.Renderer,
    args: argparse.Namespace,
) -> tuple[list[Path], int]:
    """
    Same scatter + home + three-camera snapshots as ``_build_initial_scene_for_vlm``, but reuse an
    existing ``MjModel`` / ``MjData`` / ``Renderer`` (no XML reload). Returns ``(temp_png_paths, k_used)``.
    """
    rng = np.random.default_rng(int(args.seed))
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    k0 = int(args.num_cubes) if args.num_cubes is not None else TOTAL_CUBE_COUNT
    if args.single_random_cube or args.generic_cube_instruction:
        scatter_random_single_cube(
            mj_model,
            mj_data,
            rng,
            uniform_xy=args.scatter_uniform_xy,
            near_target_center_ratio=args.near_target_center_ratio,
        )
    else:
        _scatter_cubes(
            mj_model,
            mj_data,
            rng,
            uniform_xy=args.scatter_uniform_xy,
            num_cubes=k0,
            near_target_center_ratio=args.near_target_center_ratio,
        )
    apply_home_pose(mj_model, mj_data)
    mujoco.mj_forward(mj_model, mj_data)
    paths = _write_vlm_sim_initial_images(renderer, mj_data)
    return paths, k0


def _viewer_tick_between_instruction_batches(
    viewer,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    *,
    visual_task_guides: bool,
) -> None:
    """
    Keep ``launch_passive`` responsive across long CPU/network gaps (e.g. VLM HTTP calls).

    Without periodic ``sync()``, the passive viewer / internal simulate bridge can appear hung or
    block returning to the next instruction batch.
    """
    if viewer is None:
        return
    try:
        if not viewer.is_running():
            return
        if visual_task_guides:
            sync_viewer_task_guides(
                viewer,
                mj_model,
                mj_data,
                cube_name="",
                target_cell=0,
                enabled=False,
            )
        viewer.sync()
    except Exception:
        pass


def _load_instructions_from_file(path: Path) -> list[str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and "instructions" in obj:
        inst = obj["instructions"]
    elif isinstance(obj, list):
        inst = obj
    else:
        raise ValueError('JSON must be {"instructions": [...] } or a list of instruction strings')
    if not isinstance(inst, list) or not all(isinstance(x, str) for x in inst):
        raise ValueError("'instructions' must be a non-empty list of strings")
    return [x.strip() for x in inst if x.strip()]



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "XVLA MuJoCo pick-place: default random episodes, or instruction sequence "
            "(--instructions-file / --instructions-json / --from-vlm-text)."
        )
    )
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--instructions-file", type=Path, help='JSON with {"instructions": [...]}')
    src.add_argument("--instructions-json", type=str, help="Inline JSON (same schema)")
    src.add_argument(
        "--from-vlm-text",
        type=str,
        metavar="TEXT",
        help="Plan with an OpenAI-compatible VLM in-process (OPENAI_API_KEY / .env)",
    )
    p.add_argument(
        "--vlm-image",
        type=Path,
        action="append",
        dest="vlm_images",
        default=None,
        help="Images for --from-vlm-text (repeatable)",
    )
    p.add_argument("--dotenv", type=Path, default=None, help="Optional .env for load_env before VLM calls")
    p.add_argument(
        "--no-inventory-vlm",
        action="store_true",
        help="With --from-vlm-text + images: skip vision inventory pass",
    )
    p.add_argument(
        "--no-vlm-sim-render",
        action="store_true",
        help="With --from-vlm-text and no images: skip MuJoCo three-view snapshots",
    )
    p.add_argument(
        "--vlm-vision-model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI-compatible model id for vision/inventory calls",
    )
    p.add_argument(
        "--vlm-plan-model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI-compatible model id for text planning",
    )

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--policy-id", type=str)
    g.add_argument("--policy-path", type=str)
    p.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="LeRobot dataset local path or remote repo id (e.g. owner/name)",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-policy-steps", type=int, default=500)
    p.add_argument("--render", action="store_true")
    p.add_argument(
        "--visual-task-guides",
        action="store_true",
        default=True,
        help="Enable task indicator pillars (default: on)",
    )
    p.add_argument(
        "--no-visual-task-guides",
        action="store_false",
        dest="visual_task_guides",
        help="Disable task indicator pillars",
    )
    p.add_argument("--video-out", type=Path, default=None)
    p.add_argument("--video-fps", type=int, default=20)
    p.add_argument("--video-camera", type=str, default="cam_global", metavar="CAM_NAME")
    p.add_argument("--debug-extra-camera", type=str, default=None, metavar="CAM_NAME")
    p.add_argument("--debug-extra-camera-out-dir", type=Path, default=None)
    p.add_argument("--debug-extra-camera-every", type=int, default=50, metavar="N")
    p.add_argument("--debug-extra-camera-live", action="store_true")
    p.add_argument("--grasp-assist", action="store_true")
    p.add_argument("--action-slice", type=str, default=None)
    p.add_argument("--no-autocast", action="store_true")
    p.add_argument("--debug-actions", type=int, default=0, metavar="N")
    p.add_argument("--random-target-grid-cubes", action="store_true")
    p.add_argument("--scatter-uniform-xy", action="store_true")
    p.add_argument("--near-target-center-ratio", type=float, default=0.0, metavar="R")
    p.add_argument("--num-cubes", type=int, default=None, metavar="K")
    p.add_argument("--single-random-cube", action="store_true", dest="single_random_cube")
    p.add_argument("--generic-cube-instruction", action="store_true", dest="generic_cube_instruction")
    p.add_argument(
        "--early-stop-on-success",
        action="store_true",
        default=True,
        help="Instruction-sequence: stop when cube XY reaches target cell",
    )
    p.add_argument(
        "--no-early-stop-on-success",
        action="store_false",
        dest="early_stop_on_success",
        help="Disable early stop even after target cell success",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write eval summary JSON to this path (default: sim_eval_seed{seed}_episodes{N}.json unless --no-json-out-file)",
    )
    p.add_argument(
        "--no-json-out-file",
        action="store_true",
        help="Do not write the default summary JSON file unless --json-out is set",
    )
    p.add_argument(
        "--no-json-summary-line",
        action="store_true",
        help="Do not print the final summary JSON line to stdout",
    )
    args = p.parse_args()
    if args.single_random_cube and args.generic_cube_instruction:
        p.error("Cannot use --single-random-cube together with --generic-cube-instruction")
    inst_n = sum(
        1
        for x in (args.instructions_file, args.instructions_json, args.from_vlm_text)
        if x is not None
    )
    if inst_n > 1:
        p.error("At most one of --instructions-file, --instructions-json, --from-vlm-text")
    return args


def _sim_eval_emit_json_report(
    args: argparse.Namespace,
    summary: dict[str, Any],
    *,
    default_path_stem: str,
) -> None:
    write_file = args.json_out is not None or not args.no_json_out_file
    print_line = not args.no_json_summary_line
    if not write_file and not print_line:
        return
    if args.json_out is not None:
        out_path: Path | None = args.json_out
    elif args.no_json_out_file:
        out_path = None
    else:
        out_path = Path(f"sim_eval_seed{int(args.seed)}_{default_path_stem}.json")
    out_payload = {**summary, "json_out_file": out_path.as_posix() if out_path is not None else None}
    if write_file and out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if print_line:
        print(json.dumps(out_payload, ensure_ascii=False), flush=True)


def _instruction_sequence_mode(args: argparse.Namespace) -> bool:
    return (
        args.instructions_file is not None
        or args.instructions_json is not None
        or args.from_vlm_text is not None
    )


def main_instruction_sequence(args: argparse.Namespace) -> int:
    load_env(args.dotenv.resolve() if args.dotenv else None)

    if args.single_random_cube and args.generic_cube_instruction:
        print("Cannot use --single-random-cube together with --generic-cube-instruction", file=sys.stderr)
        return 1

    per_color_budget: dict[str, int] | None = None
    mj_preflight: tuple[mujoco.MjModel, mujoco.MjData, mujoco.Renderer] | None = None
    vlm_snapshot_k: int | None = None
    vlm_temp_images: list[Path] = []

    if args.from_vlm_text is not None:
        imgs = [Path(x) for x in (args.vlm_images or [])]
        if not imgs and not args.no_vlm_sim_render:
            _m, _d, _r, vlm_temp_images, vlm_snapshot_k = _build_initial_scene_for_vlm(args)
            imgs = list(vlm_temp_images)
            mj_preflight = (_m, _d, _r)
            print(
                f"[seq] sim initial state → VLM: seed={args.seed}, scatter K={vlm_snapshot_k}, "
                f"cameras {', '.join(_VLM_SNAPSHOT_CAMERAS)}",
                file=sys.stderr,
            )
        sim_counts: dict[str, int] | None = None
        if vlm_snapshot_k is not None:
            sim_counts = active_color_counts_for_num_cubes(vlm_snapshot_k)
        elif args.num_cubes is not None:
            sim_counts = active_color_counts_for_num_cubes(args.num_cubes)
        plan_text = _append_sim_inventory_to_prompt(args.from_vlm_text, sim_counts)
        if sim_counts is not None:
            print(
                f"[seq] appended initial inventory to prompt: {sim_counts}",
                file=sys.stderr,
            )
        try:
            instructions, _raw, per_color_budget = plan_vla_instructions(
                plan_text,
                image_paths=imgs or None,
                inventory_from_vlm=not args.no_inventory_vlm,
                sim_active_color_counts=sim_counts,
                plan_model=args.vlm_plan_model,
                inventory_model=args.vlm_vision_model,
            )
            ptxt = _extract_pattern_text_from_raw(_raw)
            if ptxt:
                print(f"[seq] pattern: {ptxt}", file=sys.stderr)
            else:
                print("[seq] pattern: (model did not return a pattern field)", file=sys.stderr)
        finally:
            _cleanup_vlm_sim_image_paths(vlm_temp_images)
    elif args.instructions_file is not None:
        instructions = _load_instructions_from_file(args.instructions_file.resolve())
    else:
        obj = json.loads(args.instructions_json)
        if isinstance(obj, dict) and "instructions" in obj:
            instructions = obj["instructions"]
        elif isinstance(obj, list):
            instructions = obj
        else:
            print('JSON must be {"instructions": [...] } or a list of strings', file=sys.stderr)
            return 1
        if not isinstance(instructions, list):
            print("'instructions' must be an array", file=sys.stderr)
            return 1
        instructions = [str(x).strip() for x in instructions if str(x).strip()]

    if not instructions:
        print("Instruction list is empty.", file=sys.stderr)
        return 1

    ok, errs = validate_instructions(instructions)
    if errs:
        for e in errs:
            print(e, file=sys.stderr)
        return 1
    instructions = ok

    total_inst = len(instructions)
    placement_ok_count = 0

    def _flush_placement_stats() -> None:
        setattr(
            args,
            "_sim_eval_instruction_placement_stats",
            {
                "placement_ok_released": int(placement_ok_count),
                "total_instructions": int(total_inst),
            },
        )

    if per_color_budget is not None:
        vk, vm = validate_instructions_against_vlm_inventory(instructions, per_color_budget)
        if not vk:
            print(vm, file=sys.stderr)
            _flush_placement_stats()
            return 1

    nc_err, effective_num_cubes = _resolve_num_cubes_for_instructions(
        instructions,
        num_cubes=args.num_cubes,
        single_random_cube=args.single_random_cube,
        generic_cube_instruction=args.generic_cube_instruction,
    )
    if nc_err:
        print(nc_err, file=sys.stderr)
        _flush_placement_stats()
        return 1

    if (
        vlm_snapshot_k is not None
        and effective_num_cubes is not None
        and int(effective_num_cubes) != int(vlm_snapshot_k)
    ):
        print(
            f"[seq] note: VLM snapshots used scatter K={vlm_snapshot_k}, "
            f"after parsing the sim uses num_cubes={effective_num_cubes}; step 1 will re-scatter with the latter — "
            "may mismatch what the vision model saw. Set --num-cubes explicitly before planning to align.",
            file=sys.stderr,
        )

    action_slice = _parse_action_slice(args.action_slice)

    reuse = getattr(args, "_sim_eval_reuse", None)
    if reuse is not None:
        policy = reuse["policy"]
        preprocessor = reuse["preprocessor"]
        postprocessor = reuse["postprocessor"]
        device = reuse["device"]
        has_float = bool(reuse["has_float"])
        has_imagenet = bool(reuse["has_imagenet"])
    else:
        if args.policy_id:
            policy_ref = hub_or_posix_path(args.policy_id)
        else:
            policy_ref = resolve_local_policy_dir(Path(args.policy_path)).as_posix()

        if args.device != "cpu" and not torch.cuda.is_available():
            print("CUDA requested but not available.", file=sys.stderr)
            _flush_placement_stats()
            return 1
        device = torch.device(args.device)

        dataset_root = resolve_lerobot_dataset_root_for_eval(
            args.dataset_root,
            script_parent=Path(__file__).resolve().parent,
        )
        policy, preprocessor, postprocessor, _fps = load_policy_and_processors(
            policy_ref, dataset_root, device
        )
        has_float, has_imagenet = xvla_image_step_presence(preprocessor)

    if reuse is not None and reuse.get("mj_model") is not None:
        mj_model = reuse["mj_model"]
        mj_data = reuse["mj_data"]
        renderer = reuse["renderer"]
    elif mj_preflight is not None:
        mj_model, mj_data, renderer = mj_preflight
    else:
        mj_model = mujoco.MjModel.from_xml_path(str(_XML))
        mj_data = mujoco.MjData(mj_model)
        renderer = mujoco.Renderer(mj_model, height=480, width=640)
    rng = np.random.default_rng(int(args.seed))

    viewer = None
    if args.render:
        from mujoco.viewer import launch_passive as _launch_passive_viewer

        if reuse is not None and reuse.get("viewer") is not None:
            v0 = reuse["viewer"]
            try:
                if v0.is_running():
                    viewer = v0
                else:
                    reuse["viewer"] = None
                    viewer = _launch_passive_viewer(mj_model, mj_data, show_left_ui=False, show_right_ui=False)
                    reuse["viewer"] = viewer
            except Exception:
                reuse["viewer"] = None
                viewer = _launch_passive_viewer(mj_model, mj_data, show_left_ui=False, show_right_ui=False)
                reuse["viewer"] = viewer
        else:
            viewer = _launch_passive_viewer(mj_model, mj_data, show_left_ui=False, show_right_ui=False)
            if reuse is not None:
                reuse["viewer"] = viewer

    video_writer = None
    video_cam = str(args.video_camera or "cam_global").strip()
    if not video_cam:
        video_cam = "cam_global"
    vid_cid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, video_cam)
    if int(vid_cid) < 0:
        print(f"[seq] video camera not found: {video_cam!r}", file=sys.stderr)
        _flush_placement_stats()
        return 1
    if args.video_out is not None:
        out = args.video_out.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(str(out), fps=args.video_fps)
        print(f"[seq] Writing video: {out} (fps={args.video_fps}, camera={video_cam})")

    dbg_cam = str(args.debug_extra_camera or "").strip()
    dbg_out_dir: Path | None = None
    dbg_live = bool(args.debug_extra_camera_live)
    dbg_win = "debug-extra-camera-seq"
    cv2_dbg = None
    if dbg_cam:
        cid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, dbg_cam)
        if int(cid) < 0:
            print(f"[seq] debug camera not found: {dbg_cam!r}", file=sys.stderr)
            _flush_placement_stats()
            return 1
        dbg_out_dir = (
            args.debug_extra_camera_out_dir.resolve()
            if args.debug_extra_camera_out_dir is not None
            else (Path.cwd() / "runs" / "debug_extra_camera_seq").resolve()
        )
        dbg_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[seq] debug camera enabled: {dbg_cam} -> {dbg_out_dir}", file=sys.stderr)
        if dbg_live:
            try:
                import cv2 as _cv2

                cv2_dbg = _cv2
                cv2_dbg.namedWindow(dbg_win, cv2_dbg.WINDOW_NORMAL)
            except Exception as e:
                print(f"[seq] live debug display unavailable, disabled: {e}", file=sys.stderr)
                dbg_live = False
    elif dbg_live:
        print("[seq] --debug-extra-camera-live requires --debug-extra-camera", file=sys.stderr)
        _flush_placement_stats()
        return 1

    substeps_per_policy = policy_physics_substeps_per_decision()
    num_cubes = effective_num_cubes
    multi_step = len(instructions) > 1

    for step_i, task in enumerate(instructions):
        policy.reset()
        # Multi-step: only step 0 resets/scatters; later steps keep physics and arm pose from the previous instruction.
        fresh_scene = not multi_step or step_i == 0
        if fresh_scene:
            mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)

        if fresh_scene:
            if args.single_random_cube or args.generic_cube_instruction:
                target_color, cube_index, cube_name = scatter_random_single_cube(
                    mj_model,
                    mj_data,
                    rng,
                    uniform_xy=args.scatter_uniform_xy,
                    near_target_center_ratio=args.near_target_center_ratio,
                )
            else:
                _scatter_cubes(
                    mj_model,
                    mj_data,
                    rng,
                    uniform_xy=args.scatter_uniform_xy,
                    num_cubes=num_cubes,
                    near_target_center_ratio=args.near_target_center_ratio,
                )
                mujoco.mj_forward(mj_model, mj_data)

        try:
            if fresh_scene and args.random_target_grid_cubes:
                cube_name, target_cell = parse_task_for_visual_guide(
                    task, num_cubes, mj_model, mj_data
                )
                if args.single_random_cube or args.generic_cube_instruction:
                    relocate_random_subset_to_target_grid(
                        mj_model,
                        mj_data,
                        rng,
                        [cube_name],
                        exclude_bodies_from_grid=(cube_name,),
                        exclude_cells=(target_cell,),
                    )
                else:
                    k = TOTAL_CUBE_COUNT if num_cubes is None else int(num_cubes)
                    k = max(1, min(k, TOTAL_CUBE_COUNT))
                    relocate_random_subset_to_target_grid(
                        mj_model,
                        mj_data,
                        rng,
                        list(ALL_CUBE_NAMES[:k]),
                        exclude_bodies_from_grid=(cube_name,),
                        exclude_cells=(target_cell,),
                    )
                cube_name, target_cell = parse_task_for_visual_guide(
                    task, num_cubes, mj_model, mj_data
                )
            else:
                cube_name, target_cell = parse_task_for_visual_guide(
                    task, num_cubes, mj_model, mj_data
                )
        except ValueError as e:
            print(f"[seq] step {step_i + 1}: could not parse task line: {e}", file=sys.stderr)
            _flush_placement_stats()
            return 1

        if fresh_scene:
            apply_home_pose(mj_model, mj_data)
        elif multi_step:
            print("[seq] continuing from previous sim state (no scene reset, no home pose)")

        gc = GRID_CENTERS[target_cell]
        perfect_pos = np.array([gc[0], gc[1], CUBE_Z], dtype=np.float64)
        gstate = (
            GraspAssistState(cube_name=cube_name, perfect_pos=perfect_pos, enabled=True)
            if args.grasp_assist
            else None
        )

        print(
            f"[seq] {step_i + 1}/{len(instructions)} task={task!r} "
            f"cube={cube_name} cell={target_cell} grasp_assist={gstate is not None}"
        )

        def _dump_dbg_camera(tag: str) -> None:
            nonlocal dbg_live
            if not dbg_cam or dbg_out_dir is None:
                return
            renderer.update_scene(mj_data, camera=dbg_cam)
            if args.visual_task_guides:
                append_task_guide_geoms_to_scene(
                    renderer.scene,
                    renderer.model,
                    mj_data,
                    cube_name=cube_name,
                    target_cell=target_cell,
                    enabled=True,
                )
            rgb_dbg = renderer.render()
            p = dbg_out_dir / f"step_{step_i:02d}_{tag}_{dbg_cam}.png"
            imageio.imwrite(str(p), rgb_dbg)
            if dbg_live and cv2_dbg is not None:
                bgr = rgb_dbg[..., ::-1].copy()
                cv2_dbg.imshow(dbg_win, bgr)
                k = cv2_dbg.waitKey(1) & 0xFF
                if k == 27:
                    # Esc closes the live debug window without affecting the main loop
                    cv2_dbg.destroyWindow(dbg_win)
                    dbg_live = False

        # Refresh guide pillars as soon as the new instruction is ready (avoid stale guides from the previous line).
        if args.visual_task_guides:
            if video_writer is not None:
                renderer.update_scene(mj_data, camera=video_cam)
                append_task_guide_geoms_to_scene(
                    renderer.scene,
                    renderer.model,
                    mj_data,
                    cube_name=cube_name,
                    target_cell=target_cell,
                    enabled=True,
                )
                video_writer.append_data(renderer.render())
            if viewer is not None:
                sync_viewer_task_guides(
                    viewer,
                    mj_model,
                    mj_data,
                    cube_name=cube_name,
                    target_cell=target_cell,
                    enabled=True,
                )
                viewer.sync()
        _dump_dbg_camera("init")

        stale_arm_steps = 0
        prev_arm = None
        steps_done = 0
        for t in range(args.max_policy_steps):
            obs = build_obs_dict(
                renderer,
                mj_data,
                task,
                has_image_to_float=has_float,
                has_imagenet_normalize=has_imagenet,
                policy_config=policy.config,
                visual_task_guides=args.visual_task_guides,
                guide_cube_name=cube_name,
                guide_target_cell=target_cell,
            )
            with torch.inference_mode():
                if device.type == "cuda" and not args.no_autocast:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        batch = preprocessor(obs)
                        act = policy.select_action(batch)
                        act = postprocessor(act)
                else:
                    batch = preprocessor(obs)
                    act = policy.select_action(batch)
                    act = postprocessor(act)

            if isinstance(act, dict) and "action" in act:
                act_t = act["action"]
            elif isinstance(act, torch.Tensor):
                act_t = act
            else:
                act_t = act
            if isinstance(act_t, torch.Tensor):
                raw = act_t.detach().float().cpu().numpy().reshape(-1)
            else:
                raw = np.asarray(act_t, dtype=np.float64).reshape(-1)

            a = policy_vector_to_so100_action(raw, action_slice=action_slice)
            if not np.all(np.isfinite(a)):
                if viewer is not None:
                    if args.visual_task_guides:
                        sync_viewer_task_guides(
                            viewer,
                            mj_model,
                            mj_data,
                            cube_name=cube_name,
                            target_cell=target_cell,
                            enabled=True,
                        )
                    viewer.sync()
                continue

            if args.debug_actions > 0 and t % args.debug_actions == 0:
                print(
                    f"  step {t}: |raw|={np.linalg.norm(raw):.4f} |a[:6]|={np.linalg.norm(a):.4f}"
                )

            apply_action_to_sim(mj_model, mj_data, a)
            physics_substeps_with_grasp(mj_model, mj_data, substeps_per_policy, gstate)

            arm = mj_data.qpos[:5].copy()
            if prev_arm is not None:
                dq = float(np.max(np.abs(arm - prev_arm)))
                stale_arm_steps = stale_arm_steps + 1 if dq < 1e-5 else 0
                if stale_arm_steps == 40:
                    stale_arm_steps = 0
            prev_arm = arm

            if video_writer is not None:
                renderer.update_scene(mj_data, camera=video_cam)
                if args.visual_task_guides:
                    append_task_guide_geoms_to_scene(
                        renderer.scene,
                        renderer.model,
                        mj_data,
                        cube_name=cube_name,
                        target_cell=target_cell,
                        enabled=True,
                    )
                video_writer.append_data(renderer.render())

            if dbg_cam:
                save_every = int(args.debug_extra_camera_every)
                if save_every > 0:
                    if (t + 1) % save_every == 0:
                        _dump_dbg_camera(f"t{t + 1:04d}")

            if viewer is not None:
                if args.visual_task_guides:
                    sync_viewer_task_guides(
                        viewer,
                        mj_model,
                        mj_data,
                        cube_name=cube_name,
                        target_cell=target_cell,
                        enabled=True,
                    )
                viewer.sync()

            if args.early_stop_on_success and _placement_xy_ok_and_released(
                mj_model, mj_data, cube_name, target_cell
            ):
                print(f"  [seq] early success (placed + released) @ policy step {t + 1}")
                _dump_dbg_camera(f"success_t{t + 1:04d}")
                break

        if _placement_xy_ok_and_released(mj_model, mj_data, cube_name, target_cell):
            placement_ok_count += 1

        print(f"  step block done (max {args.max_policy_steps})")

    keep_open = bool(getattr(args, "_sim_eval_keep_open", False))
    if keep_open and viewer is not None:
        _viewer_tick_between_instruction_batches(
            viewer, mj_model, mj_data, visual_task_guides=bool(args.visual_task_guides)
        )
    if not keep_open:
        renderer.close()
    if video_writer is not None:
        video_writer.close()
        print(f"Video saved: {args.video_out}")
    if viewer is not None and not keep_open:
        viewer.close()
        if reuse is not None:
            reuse["viewer"] = None
    if dbg_live and cv2_dbg is not None and not keep_open:
        try:
            cv2_dbg.destroyWindow(dbg_win)
        except Exception:
            pass
    _flush_placement_stats()
    return 0

def main_eval_pick_place(args: argparse.Namespace) -> None:
    action_slice = _parse_action_slice(args.action_slice)

    if args.policy_id:
        policy_ref = hub_or_posix_path(args.policy_id)
    else:
        local_dir = resolve_local_policy_dir(Path(args.policy_path))
        policy_ref = local_dir.as_posix()

    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"Device {args.device!r} requested but CUDA is not available.")
    device = torch.device(args.device)

    dataset_root = resolve_lerobot_dataset_root_for_eval(
        args.dataset_root,
        script_parent=Path(__file__).resolve().parent,
    )
    policy, preprocessor, postprocessor, _fps = load_policy_and_processors(
        policy_ref, dataset_root, device
    )
    has_float, has_imagenet = xvla_image_step_presence(preprocessor)

    mj_model = mujoco.MjModel.from_xml_path(str(_XML))
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=480, width=640)

    video_cam = str(args.video_camera or "cam_global").strip() or "cam_global"
    vid_cid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, video_cam)
    if int(vid_cid) < 0:
        raise ValueError(f"video camera not found: {video_cam!r}")

    rng = np.random.default_rng(int(args.seed))
    print(f"[sim_eval] seed={args.seed}")
    viewer = None
    if args.render:
        viewer = mujoco.viewer.launch_passive(mj_model, mj_data, show_left_ui=False, show_right_ui=False)

    video_writer = None
    if args.video_out is not None:
        args.video_out = args.video_out.resolve()
        args.video_out.parent.mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(str(args.video_out), fps=args.video_fps)
        print(f"Writing video: {args.video_out} (fps={args.video_fps})")

    dbg_cam = str(args.debug_extra_camera or "").strip()
    dbg_out_dir: Path | None = None
    dbg_live = bool(args.debug_extra_camera_live)
    dbg_win = "debug-extra-camera-sim_eval"
    cv2_dbg = None
    if dbg_cam:
        cid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, dbg_cam)
        if int(cid) < 0:
            raise ValueError(f"调试相机不存在: {dbg_cam!r}")
        dbg_out_dir = (
            args.debug_extra_camera_out_dir.resolve()
            if args.debug_extra_camera_out_dir is not None
            else (Path.cwd() / "runs" / "debug_extra_camera_eval").resolve()
        )
        dbg_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[sim_eval] 调试相机已启用: {dbg_cam} -> {dbg_out_dir}")
        if dbg_live:
            try:
                import cv2 as _cv2

                cv2_dbg = _cv2
                cv2_dbg.namedWindow(dbg_win, cv2_dbg.WINDOW_NORMAL)
            except Exception as e:
                print(f"[sim_eval] 调试实时显示不可用，已自动关闭: {e}")
                dbg_live = False
    elif dbg_live:
        raise ValueError("--debug-extra-camera-live 需要同时设置 --debug-extra-camera")

    substeps_per_policy = policy_physics_substeps_per_decision()

    episode_rows: list[dict[str, Any]] = []

    for ep in range(args.episodes):
        policy.reset()
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
        if args.single_random_cube or args.generic_cube_instruction:
            target_color, cube_index, cube_name = scatter_random_single_cube(
                mj_model,
                mj_data,
                rng,
                uniform_xy=args.scatter_uniform_xy,
                near_target_center_ratio=args.near_target_center_ratio,
            )
        else:
            _scatter_cubes(
                mj_model,
                mj_data,
                rng,
                uniform_xy=args.scatter_uniform_xy,
                num_cubes=args.num_cubes,
                near_target_center_ratio=args.near_target_center_ratio,
            )
            mujoco.mj_forward(mj_model, mj_data)
            triplets = active_cube_triplets(args.num_cubes)
            ti = int(rng.integers(0, len(triplets)))
            target_color, cube_index, cube_name = triplets[ti]
        target_cell = int(rng.integers(0, 9))
        if args.random_target_grid_cubes:
            if args.single_random_cube or args.generic_cube_instruction:
                relocate_random_subset_to_target_grid(
                    mj_model,
                    mj_data,
                    rng,
                    [cube_name],
                    exclude_bodies_from_grid=(cube_name,),
                    exclude_cells=(target_cell,),
                )
            else:
                k = (
                    args.num_cubes
                    if args.num_cubes is not None
                    else TOTAL_CUBE_COUNT
                )
                k = max(1, min(int(k), TOTAL_CUBE_COUNT))
                relocate_random_subset_to_target_grid(
                    mj_model,
                    mj_data,
                    rng,
                    list(ALL_CUBE_NAMES[:k]),
                    exclude_bodies_from_grid=(cube_name,),
                    exclude_cells=(target_cell,),
                )
        if args.generic_cube_instruction:
            task = build_task_text_generic_cube(target_cell)
        else:
            task = build_task_text_for_body(cube_name, target_cell)
        gc = GRID_CENTERS[target_cell]
        perfect_pos = np.array([gc[0], gc[1], CUBE_Z], dtype=np.float64)
        gstate = (
            GraspAssistState(cube_name=cube_name, perfect_pos=perfect_pos, enabled=True)
            if args.grasp_assist
            else None
        )
        print(f"Episode {ep + 1}: {task} (cube={cube_name}, grasp_assist={gstate is not None})")

        def _dump_dbg_camera(tag: str) -> None:
            nonlocal dbg_live
            if not dbg_cam or dbg_out_dir is None:
                return
            renderer.update_scene(mj_data, camera=dbg_cam)
            if args.visual_task_guides:
                append_task_guide_geoms_to_scene(
                    renderer.scene,
                    renderer.model,
                    mj_data,
                    cube_name=cube_name,
                    target_cell=target_cell,
                    enabled=True,
                )
            rgb_dbg = renderer.render()
            p = dbg_out_dir / f"ep_{ep:03d}_{tag}_{dbg_cam}.png"
            imageio.imwrite(str(p), rgb_dbg)
            if dbg_live and cv2_dbg is not None:
                bgr = rgb_dbg[..., ::-1].copy()
                cv2_dbg.imshow(dbg_win, bgr)
                k = cv2_dbg.waitKey(1) & 0xFF
                if k == 27:
                    cv2_dbg.destroyWindow(dbg_win)
                    dbg_live = False

        # 每回合任务确定后立刻刷新光柱，避免 viewer/视频仍显示上一回合的引导
        if args.visual_task_guides:
            if video_writer is not None:
                renderer.update_scene(mj_data, camera=video_cam)
                append_task_guide_geoms_to_scene(
                    renderer.scene,
                    renderer.model,
                    mj_data,
                    cube_name=cube_name,
                    target_cell=target_cell,
                    enabled=True,
                )
                video_writer.append_data(renderer.render())
            if viewer is not None:
                sync_viewer_task_guides(
                    viewer,
                    mj_model,
                    mj_data,
                    cube_name=cube_name,
                    target_cell=target_cell,
                    enabled=True,
                )
                viewer.sync()
        _dump_dbg_camera("init")

        stale_arm_steps = 0
        prev_arm = None
        steps_done = 0

        for t in range(args.max_policy_steps):
            obs = build_obs_dict(
                renderer,
                mj_data,
                task,
                has_image_to_float=has_float,
                has_imagenet_normalize=has_imagenet,
                policy_config=policy.config,
                visual_task_guides=args.visual_task_guides,
                guide_cube_name=cube_name,
                guide_target_cell=target_cell,
            )
            with torch.inference_mode():
                if device.type == "cuda" and not args.no_autocast:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        batch = preprocessor(obs)
                        act = policy.select_action(batch)
                        act = postprocessor(act)
                else:
                    batch = preprocessor(obs)
                    act = policy.select_action(batch)
                    act = postprocessor(act)

            if isinstance(act, dict) and "action" in act:
                act_t = act["action"]
            elif isinstance(act, torch.Tensor):
                act_t = act
            else:
                act_t = act
            if isinstance(act_t, torch.Tensor):
                raw = act_t.detach().float().cpu().numpy().reshape(-1)
            else:
                raw = np.asarray(act_t, dtype=np.float64).reshape(-1)

            a = policy_vector_to_so100_action(raw, action_slice=action_slice)
            if not np.all(np.isfinite(a)):
                print(f"[sim_eval] step {t}: non-finite action {a!r} — skip apply")
                stale_arm_steps += 1
                if viewer is not None:
                    if args.visual_task_guides:
                        sync_viewer_task_guides(
                            viewer,
                            mj_model,
                            mj_data,
                            cube_name=cube_name,
                            target_cell=target_cell,
                            enabled=True,
                        )
                    viewer.sync()
                continue
            if args.debug_actions > 0 and t % args.debug_actions == 0:
                print(
                    f"[sim_eval] step {t}: |raw|={np.linalg.norm(raw):.4f} |a[:6]|={np.linalg.norm(a):.4f} "
                    f"a[:3]={a[:3]} jaw={a[5]:.4f}"
                )

            apply_action_to_sim(mj_model, mj_data, a)
            physics_substeps_with_grasp(mj_model, mj_data, substeps_per_policy, gstate)

            arm = mj_data.qpos[:5].copy()
            if prev_arm is not None:
                dq = float(np.max(np.abs(arm - prev_arm)))
                stale_arm_steps = stale_arm_steps + 1 if dq < 1e-5 else 0
                if stale_arm_steps == 40:
                    print(
                        "[sim_eval] 臂关节约 40 步几乎不动。可试: --no-autocast，核对 --dataset-root 与训练 stats，"
                        "或 --action-slice（若策略前 6 维非关节）。"
                    )
                    stale_arm_steps = 0
            prev_arm = arm

            if video_writer is not None:
                renderer.update_scene(mj_data, camera=video_cam)
                if args.visual_task_guides:
                    append_task_guide_geoms_to_scene(
                        renderer.scene,
                        renderer.model,
                        mj_data,
                        cube_name=cube_name,
                        target_cell=target_cell,
                        enabled=True,
                    )
                video_writer.append_data(renderer.render())

            if dbg_cam:
                save_every = int(args.debug_extra_camera_every)
                if save_every > 0 and (t + 1) % save_every == 0:
                    _dump_dbg_camera(f"t{t + 1:04d}")

            if viewer is not None:
                if args.visual_task_guides:
                    sync_viewer_task_guides(
                        viewer,
                        mj_model,
                        mj_data,
                        cube_name=cube_name,
                        target_cell=target_cell,
                        enabled=True,
                    )
                viewer.sync()

            steps_done = t + 1
            if args.early_stop_on_success and _placement_xy_ok_and_released(
                mj_model, mj_data, cube_name, target_cell
            ):
                print(f"  [eval] early success (placed + released) @ policy step {t + 1}")
                _dump_dbg_camera(f"success_t{t + 1:04d}")
                break

        print(f"  done ({steps_done} policy steps, max {args.max_policy_steps})")

        placement_ok = bool(
            _placement_xy_ok_and_released(mj_model, mj_data, cube_name, target_cell)
        )
        episode_rows.append(
            {
                "episode_index": int(ep),
                "task": task,
                "cube_name": cube_name,
                "target_cell": int(target_cell),
                "placement_ok_released": placement_ok,
                "policy_steps": int(steps_done),
            }
        )

    ok_n = sum(1 for r in episode_rows if r["placement_ok_released"])
    rate = (float(ok_n) / float(len(episode_rows))) if episode_rows else None
    _sim_eval_emit_json_report(
        args,
        {
            "schema": "sim_eval/pick_place/v1",
            "seed": int(args.seed),
            "episodes": int(args.episodes),
            "policy_id": args.policy_id,
            "policy_path": args.policy_path,
            "dataset_root": args.dataset_root,
            "device": args.device,
            "render": bool(args.render),
            "grasp_assist": bool(args.grasp_assist),
            "early_stop_on_success": bool(args.early_stop_on_success),
            "num_cubes": args.num_cubes,
            "mean_placement_ok_released_rate": rate,
            "placement_ok_episodes": int(ok_n),
            "runs": episode_rows,
        },
        default_path_stem=f"episodes{int(args.episodes)}",
    )

    renderer.close()
    if video_writer is not None:
        video_writer.close()
        print(f"Video saved: {args.video_out}")
    if viewer is not None:
        viewer.close()
    if dbg_live and cv2_dbg is not None:
        try:
            cv2_dbg.destroyWindow(dbg_win)
        except Exception:
            pass

def main() -> int:
    args = parse_args()
    if _instruction_sequence_mode(args):
        rc = int(main_instruction_sequence(args))
        stats = getattr(args, "_sim_eval_instruction_placement_stats", None)
        _sim_eval_emit_json_report(
            args,
            {
                "schema": "sim_eval/instruction_sequence/v1",
                "seed": int(args.seed),
                "policy_id": args.policy_id,
                "policy_path": args.policy_path,
                "dataset_root": args.dataset_root,
                "device": args.device,
                "return_code": int(rc),
                "placement_stats": stats,
            },
            default_path_stem="instruction_sequence",
        )
        return rc
    main_eval_pick_place(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
