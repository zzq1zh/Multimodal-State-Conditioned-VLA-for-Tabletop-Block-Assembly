#!/usr/bin/env python3
"""
Run N random natural-language prompts through the VLM planner, then execute the
resulting XVLA-style instruction list in MuJoCo (same stepping logic as sim_eval).

**Task completion (one prompt)**::

    completion = (# instructions that end with the cube in the target cell) / (# VLM instructions)

Cube placement is checked with the same XY tolerance as ``sim_eval._placement_xy_ok``.
By default ``--early-stop-on-success`` is enabled so each instruction stops as soon as placement passes.

Example::

  python sim_vlm_to_actions_eval.py \\
    --n-prompts 30 \\
    --policy-id YOUR/HF_REPO \\
    --dataset-root /path/to/lerobot_dataset \\
    --device cuda \\
    --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np
import torch

from sim_scenes import (
    ALL_CUBE_NAMES,
    CUBE_Z,
    GRID_CENTERS,
    TOTAL_CUBE_COUNT,
    active_color_counts_for_num_cubes,
    _scatter_cubes,
    apply_home_pose,
    relocate_random_subset_to_target_grid,
    scatter_random_single_cube,
)
from sim_eval import (
    _append_sim_inventory_to_prompt,
    _build_initial_scene_for_vlm,
    _cleanup_vlm_sim_image_paths,
    main_instruction_sequence,
    _parse_action_slice,
    _placement_xy_ok,
    _resolve_num_cubes_for_instructions,
    _write_vlm_sim_initial_images,
    build_obs_dict,
)
from vla_adapter import (
    GraspAssistState,
    apply_action_to_sim,
    hub_or_posix_path,
    load_policy_and_processors,
    physics_substeps_with_grasp,
    policy_physics_substeps_per_decision,
    policy_vector_to_so100_action,
    resolve_local_policy_dir,
    resolve_lerobot_dataset_root_for_eval,
    xvla_image_step_presence,
)
from vla_to_actions import parse_task_for_visual_guide
from vlm_to_actions import (
    load_env,
    plan_vla_instructions,
    validate_instructions,
    validate_instructions_against_vlm_inventory,
)


def _random_vlm_prompt(rng: np.random.Generator) -> str:
    colors = ("yellow", "green", "purple", "orange")
    cells = np.arange(9, dtype=np.int64)
    roll = int(rng.integers(0, 6))
    if roll == 0:
        return "Form an X with cubes on the 3×3 grid."
    if roll == 1:
        return "Clear the source bin into the grid in a symmetric pattern."
    if roll == 2:
        c = colors[int(rng.integers(0, 4))]
        g = int(rng.integers(0, 9))
        return f"Pick up the {c} cube and place it in grid cell {g}."
    if roll == 3:
        c1, c2 = colors[int(rng.integers(0, 4))], colors[int(rng.integers(0, 4))]
        g1, g2 = [int(x) for x in rng.choice(cells, size=2, replace=False)]
        return f"First move the {c1} cube to cell {g1}, then the {c2} cube to cell {g2}."
    if roll == 4:
        c = colors[int(rng.integers(0, 4))]
        g = int(rng.integers(0, 9))
        return f"Only reposition the {c} cube; final goal is grid cell {g}."
    c1, c2, c3 = (
        colors[int(rng.integers(0, 4))],
        colors[int(rng.integers(0, 4))],
        colors[int(rng.integers(0, 4))],
    )
    g1, g2, g3 = [int(x) for x in rng.choice(cells, size=3, replace=False)]
    return (
        f"Sequence: {c1} to {g1}, then {c2} to {g2}, then {c3} to {g3}. "
        "Respect cube inventory and use valid pick-place lines."
    )


def _ns_for_vlm_preflight(args: argparse.Namespace, seed: int) -> argparse.Namespace:
    o = argparse.Namespace(**vars(args))
    o.seed = int(seed)
    return o


def _run_one_prompt(
    *,
    prompt: str,
    prompt_index: int,
    scatter_k: int,
    args: argparse.Namespace,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    renderer: mujoco.Renderer,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    device: torch.device,
    has_float: bool,
    has_imagenet: bool,
    action_slice: slice | None,
    rng: np.random.Generator,
) -> dict[str, Any]:
    vlm_temp_images: list[Path] = []
    instructions: list[str] = []
    per_color_budget: dict[str, int] | None = None
    try:
        imgs = [Path(x) for x in (args.vlm_images or [])]
        if not imgs and not args.no_vlm_sim_render:
            vlm_temp_images = _write_vlm_sim_initial_images(renderer, mj_data)
            imgs = list(vlm_temp_images)

        sim_counts = active_color_counts_for_num_cubes(int(scatter_k))
        plan_text = _append_sim_inventory_to_prompt(prompt, sim_counts)
        instructions, _raw, per_color_budget = plan_vla_instructions(
            plan_text,
            image_paths=imgs or None,
            inventory_from_vlm=not args.no_inventory_vlm,
            sim_active_color_counts=sim_counts,
            plan_model=args.vlm_plan_model,
            inventory_model=args.vlm_vision_model,
        )
    except Exception as e:
        return {
            "prompt_index": prompt_index,
            "prompt": prompt,
            "error": repr(e),
            "total_instructions": 0,
            "successes": 0,
            "completion": None,
        }
    finally:
        _cleanup_vlm_sim_image_paths(vlm_temp_images)

    if not instructions:
        return {
            "prompt_index": prompt_index,
            "prompt": prompt,
            "error": "empty_instruction_list",
            "total_instructions": 0,
            "successes": 0,
            "completion": None,
        }

    ok, errs = validate_instructions(instructions)
    if errs:
        return {
            "prompt_index": prompt_index,
            "prompt": prompt,
            "error": "validate_instructions: " + "; ".join(errs),
            "total_instructions": len(instructions) if instructions else 0,
            "successes": 0,
            "completion": None,
        }
    instructions = ok
    if per_color_budget is not None:
        vk, vm = validate_instructions_against_vlm_inventory(instructions, per_color_budget)
        if not vk:
            return {
                "prompt_index": prompt_index,
                "prompt": prompt,
                "error": vm,
                "total_instructions": len(instructions),
                "successes": 0,
                "completion": 0.0,
            }

    nc_err, effective_num_cubes = _resolve_num_cubes_for_instructions(
        instructions,
        num_cubes=args.num_cubes,
        single_random_cube=args.single_random_cube,
        generic_cube_instruction=args.generic_cube_instruction,
    )
    if nc_err:
        return {
            "prompt_index": prompt_index,
            "prompt": prompt,
            "error": nc_err,
            "total_instructions": len(instructions),
            "successes": 0,
            "completion": 0.0,
        }

    num_cubes = effective_num_cubes
    print(
        f"[vlm-eval] handoff instruction queue to sim_eval "
        f"(prompt_index={prompt_index}, n={len(instructions)})",
        flush=True,
    )
    sim_args = argparse.Namespace(
        instructions_file=None,
        instructions_json=json.dumps({"instructions": instructions}, ensure_ascii=False),
        from_vlm_text=None,
        vlm_images=None,
        dotenv=args.dotenv,
        no_inventory_vlm=args.no_inventory_vlm,
        no_vlm_sim_render=args.no_vlm_sim_render,
        vlm_vision_model=args.vlm_vision_model,
        vlm_plan_model=args.vlm_plan_model,
        policy_id=args.policy_id,
        policy_path=args.policy_path,
        dataset_root=args.dataset_root,
        device=args.device,
        seed=int(args.seed) + int(prompt_index) * 9973,
        episodes=1,
        max_policy_steps=args.max_policy_steps,
        render=args.render,
        visual_task_guides=args.visual_task_guides,
        video_out=None,
        video_fps=20,
        video_camera="cam_global",
        debug_extra_camera=None,
        debug_extra_camera_out_dir=None,
        debug_extra_camera_every=50,
        debug_extra_camera_live=False,
        grasp_assist=args.grasp_assist,
        action_slice=args.action_slice,
        no_autocast=args.no_autocast,
        debug_actions=0,
        random_target_grid_cubes=args.random_target_grid_cubes,
        scatter_uniform_xy=args.scatter_uniform_xy,
        near_target_center_ratio=args.near_target_center_ratio,
        num_cubes=num_cubes,
        single_random_cube=args.single_random_cube,
        generic_cube_instruction=args.generic_cube_instruction,
        early_stop_on_success=args.early_stop_on_success,
    )
    rc = int(main_instruction_sequence(sim_args))
    n = len(instructions)
    return {
        "prompt_index": prompt_index,
        "prompt": prompt,
        "error": None if rc == 0 else f"sim_eval_return_code={rc}",
        "delegated_to_sim_eval": True,
        "sim_eval_return_code": rc,
        "total_instructions": n,
        "successes": n if rc == 0 else 0,
        "completion": 1.0 if (n and rc == 0) else (0.0 if n else None),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-prompts", type=int, default=30, metavar="N")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dotenv", type=Path, default=None)
    p.add_argument("--vlm-image", type=Path, action="append", default=None, dest="vlm_images")
    p.add_argument("--no-inventory-vlm", action="store_true")
    p.add_argument("--no-vlm-sim-render", action="store_true")
    p.add_argument(
        "--vlm-vision-model",
        type=str,
        default=None,
        help="Override vision model; default uses OPENAI_VISION_MODEL/OPENAI_MODEL from env",
    )
    p.add_argument(
        "--vlm-plan-model",
        type=str,
        default=None,
        help="Override planning model; default uses OPENAI_PLAN_MODEL/OPENAI_MODEL from env",
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
    p.add_argument("--max-policy-steps", type=int, default=500)
    p.add_argument("--num-cubes", type=int, default=None, metavar="K")
    p.add_argument("--scatter-uniform-xy", action="store_true")
    p.add_argument("--near-target-center-ratio", type=float, default=0.0, metavar="R")
    p.add_argument("--single-random-cube", action="store_true")
    p.add_argument("--generic-cube-instruction", action="store_true")
    p.add_argument("--random-target-grid-cubes", action="store_true")
    p.add_argument("--grasp-assist", action="store_true")
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
    p.add_argument("--action-slice", type=str, default=None)
    p.add_argument("--no-autocast", action="store_true")
    p.add_argument("--early-stop-on-success", action="store_true", default=True)
    p.add_argument("--no-early-stop-on-success", action="store_false", dest="early_stop_on_success")
    p.add_argument("--json-out", type=Path, default=None, help="Write full summary JSON to this path")
    p.add_argument("--render", action="store_true", help="Pass through to sim_eval instruction viewer")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    load_env(args.dotenv.resolve() if args.dotenv else None)

    if args.single_random_cube and args.generic_cube_instruction:
        print("Cannot use --single-random-cube together with --generic-cube-instruction", file=sys.stderr)
        return 1

    if args.policy_id:
        policy_ref = hub_or_posix_path(args.policy_id)
    else:
        policy_ref = resolve_local_policy_dir(Path(args.policy_path)).as_posix()

    if args.device != "cpu" and not torch.cuda.is_available():
        print("CUDA requested but not available.", file=sys.stderr)
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
    action_slice = _parse_action_slice(args.action_slice)

    rng = np.random.default_rng(int(args.seed))
    runs: list[dict[str, Any]] = []
    ratios: list[float] = []

    for i in range(int(args.n_prompts)):
        prompt = _random_vlm_prompt(rng)
        seed_i = int(args.seed) + i * 9973
        ns = _ns_for_vlm_preflight(args, seed_i)
        mj_model, mj_data, renderer, vlm_paths, scatter_k = _build_initial_scene_for_vlm(ns)
        try:
            row = _run_one_prompt(
                prompt=prompt,
                prompt_index=i,
                scatter_k=int(scatter_k),
                args=args,
                mj_model=mj_model,
                mj_data=mj_data,
                renderer=renderer,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                device=device,
                has_float=has_float,
                has_imagenet=has_imagenet,
                action_slice=action_slice,
                rng=np.random.default_rng(seed_i + 7),
            )
        except Exception as e:
            row = {
                "prompt_index": i,
                "prompt": prompt,
                "error": repr(e),
                "total_instructions": 0,
                "successes": 0,
                "completion": None,
            }
        finally:
            _cleanup_vlm_sim_image_paths(vlm_paths)
            renderer.close()

        runs.append(row)
        c = row.get("completion")
        if isinstance(c, (int, float)) and not (isinstance(c, float) and np.isnan(c)):
            ratios.append(float(c))

        print(json.dumps(row, ensure_ascii=False), flush=True)

    summary = {
        "n_prompts": int(args.n_prompts),
        "mean_completion_over_nonnull": float(np.mean(ratios)) if ratios else None,
        "macro_completion": (
            (
                sum(int(r["successes"]) for r in runs if int(r.get("total_instructions") or 0) > 0)
                / sum(int(r["total_instructions"]) for r in runs if int(r.get("total_instructions") or 0) > 0)
            )
            if any(int(r.get("total_instructions") or 0) > 0 for r in runs)
            else None
        ),
        "runs": runs,
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary_mean_completion": summary["mean_completion_over_nonnull"],
                "summary_macro_completion": summary["macro_completion"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
