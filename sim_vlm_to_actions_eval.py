#!/usr/bin/env python3
"""
Run N random natural-language prompts through the VLM planner, then execute the
resulting XVLA-style instruction list in MuJoCo (same stepping logic as sim_eval).

**Task completion (one prompt)**::

    ``completion`` = (# instructions that satisfy ``_placement_xy_ok_and_released`` after their policy block)
    / (# VLM instructions). Counts come from ``sim_eval.main_instruction_sequence`` via
    ``_sim_eval_instruction_placement_stats`` on the ``sim_args`` namespace.

Each NDJSON ``runs[]`` row's ``prompt`` field is the **user template text only**; the string sent to the VLM
also appends simulation inventory constraints (see ``sim_eval._append_sim_inventory_to_prompt``) — that appended
block is not duplicated in ``prompt``. The same row includes ``vla_instructions``: the **decomposed** list of
per-step policy task strings (XVLA-style lines) produced by planning and passed to ``sim_eval`` when execution
proceeds (or the list available at the failure stage).

Between prompts, the **same** policy, ``MjModel``, ``Renderer``, and (with ``--render``) passive viewer stay
alive; only the pick-place scene is **refreshed** (re-keyframe, re-scatter, new VLM snapshots) before the next queue.

Prompts come from a **fixed set of 30** English templates (each a **single line** of text). None of them state
how many pick-place moves to use; longer tasks use chained ``First … then …`` wording. Each run shuffles the bank
in blocks of 30 so the first 30 evaluations (and every subsequent full block of 30) never reuse the same prompt
string within that block.

**Stdout JSON**: each prompt prints one NDJSON line; after all prompts, one final line prints the full summary
(including ``runs``). Use ``--no-json-summary-line`` to skip that final line.

By default a **pretty-printed copy** is also written to
``sim_vlm_eval_seed{seed}_n{n_prompts}.json`` in the current directory; override with ``--json-out PATH`` or
disable with ``--no-json-out-file``.

Progress and fatal configuration errors are one JSON object per line on **stderr** (``kind: "log"`` / ``"fatal"``)
so stdout stays machine-parseable.

Example (30 prompts: **render**, **sim MP4** per prompt, **JSON**). Default sim video uses **``cam_oblique``** (side-style view) and **burns the user prompt** into a bottom caption (OpenCV). With ``--n-prompts`` > 1, ``--sim-video-out runs/vlm_sim.mp4`` becomes ``runs/vlm_sim_p000.mp4``, ``runs/vlm_sim_p001.mp4``, …::

  mkdir -p runs && python sim_vlm_to_actions_eval.py \\
    --n-prompts 30 \\
    --policy-id YOUR/HF_REPO \\
    --dataset-root /path/to/lerobot_dataset \\
    --device cuda \\
    --seed 0 \\
    --render \\
    --sim-video-out runs/vlm_sim.mp4 \\
    --json-out runs/vlm_eval_seed0_n30.json

Use ``--sim-video-camera cam_global`` for the frontal training camera, or ``--sim-video-no-prompt-overlay`` to omit text on frames.

Omit ``--json-out`` to use the default pretty-printed file ``sim_vlm_eval_seed{seed}_n{n_prompts}.json`` in the cwd.
Use ``--no-json-out-file`` / ``--no-json-summary-line`` only if you want to suppress file or the final stdout summary line.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

from sim_scenes import active_color_counts_for_num_cubes
from sim_eval import (
    _append_sim_inventory_to_prompt,
    _build_initial_scene_for_vlm,
    _cleanup_vlm_sim_image_paths,
    _refresh_initial_scene_for_vlm,
    _resolve_num_cubes_for_instructions,
    _viewer_tick_between_instruction_batches,
    _write_vlm_sim_initial_images,
    main_instruction_sequence,
)
from vla_adapter import (
    hub_or_posix_path,
    load_policy_and_processors,
    resolve_local_policy_dir,
    resolve_lerobot_dataset_root_for_eval,
    xvla_image_step_presence,
)
from vlm_to_actions import (
    load_env,
    plan_vla_instructions,
    validate_instructions,
    validate_instructions_against_vlm_inventory,
)


# Thirty fixed natural-language prompts for VLM→sim eval. Each run shuffles this bank in blocks so
# every batch of up to 30 prompts has no string-level duplicates. Each prompt is a **single line** and
# describes **multiple** pick-place moves without stating how many. The first **15** use ``First … then …``
# chains of varying length (three prompts per band); the rest use ``Sequence:`` or similar multi-move wording.
_VLM_EVAL_PROMPT_BANK_30: tuple[str, ...] = (
    # Three prompts per chain-length band (short through long ``then`` chains); strings never state a move count.
    "Move the yellow cube to cell 0 first, and then place the green cube in cell 8.",
    "Arrange the cubes on the 3×3 grid so that they form an X shape.",
    "Start by putting the orange cube in cell 2, followed by the purple cube in cell 6.",
    "Create a diagonal line across the grid using available cubes.",
    "Place the green cube at cell 1 first; after that, move the yellow cube to cell 7.",
    "Form a simple vertical stripe on the grid with distinct cube colors.",
    "Begin with the yellow cube at cell 0, then move the green cube to cell 3, and finally place the purple cube in cell 8.",
    "Arrange the cubes to form a partial X shape centered on cell 4.",
    "First position the orange cube in cell 1, then put the purple cube in cell 4, and finish by moving the yellow cube to cell 7.",
    "Place cubes around the center cell to create a cross-like layout.",
    "Move the green cube into cell 2, then place the yellow cube at cell 5, and finally move the orange cube to cell 8.",
    "Create a right-leaning diagonal arrangement on the 3×3 grid.",
    "Place the yellow cube in cell 0 first, followed by the green cube in cell 2, the purple cube in cell 4, and the orange cube in cell 6.",
    "Form an X-like pattern by placing cubes along both diagonals where possible.",
    "Arrange the cubes in this order: orange to cell 1, purple to cell 3, yellow to cell 5, and green to cell 7.",
    "Create a symmetric edge pattern using cells on opposite sides of the grid.",
    "Fill the cells in reverse order by moving the green cube to cell 8, the yellow cube to cell 6, the purple cube to cell 4, and the orange cube to cell 2.",
    "Use the available cubes to mark the four corners and the center of the grid.",
    "Start with yellow at cell 0, continue with green at cell 2, then purple at cell 4, orange at cell 6, and finally move green again to cell 8.",
    "Create a zigzag path across the 3×3 grid using the available cubes.",
    "Follow this order: place yellow in cell 0, then green in cell 1, and finally purple in cell 2. Make sure the cube inventory is respected and each pick-place command is valid.",
    "Use three cubes to create a short horizontal bar across one row of the grid.",
    "Use valid pick-place actions to move orange to cell 3, yellow to cell 4, and green to cell 5 in sequence, while respecting the available cube inventory.",
    "Build a checkerboard-like layout using alternating occupied cells where possible.",
    "Place the cubes one by one: green in cell 6, purple in cell 7, and orange in cell 8. Ensure inventory limits and valid pick-place lines are followed.",
    "Empty the source bin by placing the cubes onto the grid in a symmetric layout.",
    "Using sequential pick-place commands, put the yellow cube in cell 0, the green cube in cell 3, and the purple cube in cell 6, while respecting inventory.",
    "Build a left-column pattern from top to bottom using available cubes.",
    "Place orange in cell 2, yellow in cell 4, and green in cell 8, executing the moves one at a time and observing inventory limits.",
    "Create a short vertical segment on the right column."
)


def _vlm_prompt_sequence(rng: np.random.Generator, n: int) -> list[str]:
    """Return ``n`` prompts; each consecutive block of up to 30 is a permutation of the bank (no repeats)."""
    bank = _VLM_EVAL_PROMPT_BANK_30
    if len(bank) != 30:
        raise RuntimeError("internal: _VLM_EVAL_PROMPT_BANK_30 must have length 30")
    out: list[str] = []
    while len(out) < n:
        perm = rng.permutation(30)
        for j in perm:
            out.append(bank[int(j)])
            if len(out) >= n:
                break
    return out


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
    rng: np.random.Generator,
    reuse_ctx: dict[str, Any],
    keep_open: bool,
) -> dict[str, Any]:
    vlm_temp_images: list[Path] = []
    instructions: list[str] = []
    per_color_budget: dict[str, int] | None = None
    try:
        imgs = [Path(x) for x in (args.vlm_images or [])]
        if not imgs and not args.no_vlm_sim_render:
            vlm_temp_images = _write_vlm_sim_initial_images(renderer, mj_data)
            imgs = list(vlm_temp_images)

        _viewer_tick_between_instruction_batches(
            reuse_ctx.get("viewer"),
            mj_model,
            mj_data,
            visual_task_guides=bool(args.visual_task_guides),
        )

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
            "vla_instructions": list(instructions),
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
            "vla_instructions": [],
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
            "vla_instructions": list(instructions),
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
                "vla_instructions": list(instructions),
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
            "vla_instructions": list(instructions),
            "error": nc_err,
            "total_instructions": len(instructions),
            "successes": 0,
            "completion": 0.0,
        }

    num_cubes = effective_num_cubes
    print(
        json.dumps(
            {
                "schema": "sim_vlm_to_actions_eval/v1",
                "kind": "log",
                "event": "sim_eval_handoff",
                "prompt_index": prompt_index,
                "n_instructions": len(instructions),
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
        flush=True,
    )
    _viewer_tick_between_instruction_batches(
        reuse_ctx.get("viewer"),
        mj_model,
        mj_data,
        visual_task_guides=bool(args.visual_task_guides),
    )
    video_out: Path | None = None
    if getattr(args, "sim_video_out", None) is not None:
        base = Path(args.sim_video_out)
        if base.suffix.lower() not in (".mp4", ".avi", ".mkv", ".webm", ".mov"):
            base = base.with_suffix(".mp4")
        if int(args.n_prompts) > 1:
            video_out = base.parent / f"{base.stem}_p{int(prompt_index):03d}{base.suffix}"
        else:
            video_out = base

    video_cam = str(getattr(args, "sim_video_camera", "cam_oblique") or "cam_oblique").strip() or "cam_oblique"
    overlay_prompt: str | None = None
    if video_out is not None and not bool(getattr(args, "sim_video_no_prompt_overlay", False)):
        overlay_prompt = (prompt or "").strip() or None

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
        video_out=video_out,
        video_fps=int(args.sim_video_fps),
        video_camera=video_cam,
        video_overlay_text=overlay_prompt,
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
    sim_args._sim_eval_reuse = reuse_ctx
    sim_args._sim_eval_keep_open = bool(keep_open)
    rc = int(main_instruction_sequence(sim_args))
    n = len(instructions)
    stats = getattr(sim_args, "_sim_eval_instruction_placement_stats", None)
    if stats is not None:
        successes = max(0, min(int(stats["placement_ok_released"]), n))
        completion = (float(successes) / float(n)) if n else None
    else:
        successes = 0
        completion = None if not n else 0.0
    return {
        "prompt_index": prompt_index,
        "prompt": prompt,
        "vla_instructions": list(instructions),
        "error": None if rc == 0 else f"sim_eval_return_code={rc}",
        "delegated_to_sim_eval": True,
        "sim_eval_return_code": rc,
        "total_instructions": n,
        "successes": successes,
        "completion": completion,
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
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write full summary JSON to this path (default: sim_vlm_eval_seed{seed}_n{n}.json unless --no-json-out-file)",
    )
    p.add_argument(
        "--no-json-out-file",
        action="store_true",
        help="Do not write the default auto-named summary file; overridden if --json-out is set",
    )
    p.add_argument(
        "--no-json-summary-line",
        action="store_true",
        help="Do not print the final full-summary JSON line to stdout (per-prompt lines still print)",
    )
    p.add_argument(
        "--sim-video-out",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "sim_eval: write MP4 per prompt (default camera: cam_oblique). "
            "If --n-prompts>1, uses stem_p000.ext, ..."
        ),
    )
    p.add_argument(
        "--sim-video-fps",
        type=int,
        default=30,
        metavar="FPS",
        help="FPS for --sim-video-out (default: 30)",
    )
    p.add_argument(
        "--sim-video-camera",
        type=str,
        default="cam_oblique",
        metavar="CAM_NAME",
        help="MuJoCo camera for --sim-video-out (default: cam_oblique, side-style table view)",
    )
    p.add_argument(
        "--sim-video-no-prompt-overlay",
        action="store_true",
        help="Do not draw the user prompt text on sim video frames",
    )
    p.add_argument("--render", action="store_true", help="Pass through to sim_eval instruction viewer")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    load_env(args.dotenv.resolve() if args.dotenv else None)

    if args.single_random_cube and args.generic_cube_instruction:
        print(
            json.dumps(
                {
                    "schema": "sim_vlm_to_actions_eval/v1",
                    "kind": "fatal",
                    "exit_code": 1,
                    "message": "Cannot use --single-random-cube together with --generic-cube-instruction",
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1

    if args.policy_id:
        policy_ref = hub_or_posix_path(args.policy_id)
    else:
        policy_ref = resolve_local_policy_dir(Path(args.policy_path)).as_posix()

    if args.device != "cpu" and not torch.cuda.is_available():
        print(
            json.dumps(
                {
                    "schema": "sim_vlm_to_actions_eval/v1",
                    "kind": "fatal",
                    "exit_code": 1,
                    "message": "CUDA requested but not available.",
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
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

    reuse_ctx: dict[str, Any] = {
        "policy": policy,
        "preprocessor": preprocessor,
        "postprocessor": postprocessor,
        "device": device,
        "has_float": has_float,
        "has_imagenet": has_imagenet,
        "mj_model": None,
        "mj_data": None,
        "renderer": None,
        "viewer": None,
    }

    rng = np.random.default_rng(int(args.seed))
    runs: list[dict[str, Any]] = []
    ratios: list[float] = []

    n = int(args.n_prompts)
    prompt_seq = _vlm_prompt_sequence(rng, n)
    for i in range(n):
        prompt = prompt_seq[i]
        seed_i = int(args.seed) + i * 9973
        ns = _ns_for_vlm_preflight(args, seed_i)
        if reuse_ctx["mj_model"] is None:
            mj_model, mj_data, renderer, vlm_paths, scatter_k = _build_initial_scene_for_vlm(ns)
            reuse_ctx["mj_model"] = mj_model
            reuse_ctx["mj_data"] = mj_data
            reuse_ctx["renderer"] = renderer
        else:
            vlm_paths, scatter_k = _refresh_initial_scene_for_vlm(
                reuse_ctx["mj_model"],
                reuse_ctx["mj_data"],
                reuse_ctx["renderer"],
                ns,
            )
            mj_model = reuse_ctx["mj_model"]
            mj_data = reuse_ctx["mj_data"]
            renderer = reuse_ctx["renderer"]

        _viewer_tick_between_instruction_batches(
            reuse_ctx.get("viewer"),
            mj_model,
            mj_data,
            visual_task_guides=bool(args.visual_task_guides),
        )

        try:
            row = _run_one_prompt(
                prompt=prompt,
                prompt_index=i,
                scatter_k=int(scatter_k),
                args=args,
                mj_model=mj_model,
                mj_data=mj_data,
                renderer=renderer,
                rng=np.random.default_rng(seed_i + 7),
                reuse_ctx=reuse_ctx,
                keep_open=(i < n - 1),
            )
        except Exception as e:
            row = {
                "prompt_index": i,
                "prompt": prompt,
                "vla_instructions": [],
                "error": repr(e),
                "total_instructions": 0,
                "successes": 0,
                "completion": None,
            }
        finally:
            _cleanup_vlm_sim_image_paths(vlm_paths)

        runs.append(row)
        c = row.get("completion")
        if isinstance(c, (int, float)) and not (isinstance(c, float) and np.isnan(c)):
            ratios.append(float(c))

        print(json.dumps(row, ensure_ascii=False), flush=True)

    r = reuse_ctx.get("renderer")
    if r is not None:
        try:
            r.close()
        except Exception:
            pass
    v = reuse_ctx.get("viewer")
    if v is not None:
        try:
            v.close()
        except Exception:
            pass

    if args.json_out is not None:
        summary_json_path: Path | None = args.json_out
    elif args.no_json_out_file:
        summary_json_path = None
    else:
        summary_json_path = Path(
            f"sim_vlm_eval_seed{int(args.seed)}_n{int(args.n_prompts)}.json"
        )

    summary: dict[str, Any] = {
        "schema": "sim_vlm_to_actions_eval/v1",
        "seed": int(args.seed),
        "n_prompts": int(args.n_prompts),
        "policy_id": args.policy_id,
        "policy_path": args.policy_path,
        "dataset_root": args.dataset_root,
        "device": args.device,
        "render": bool(args.render),
        "json_out_file": summary_json_path.as_posix() if summary_json_path is not None else None,
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
    if summary_json_path is not None:
        summary_json_path.parent.mkdir(parents=True, exist_ok=True)
        summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.no_json_summary_line:
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
