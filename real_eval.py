import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts, build_dataset_frame
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.processor import make_default_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import make_robot_from_config
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener, predict_action
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device, init_logging

from real_vision import (
    load_yolo_model,
    make_sam2_image_predictor,
    BeamOverlayState,
    load_calibration_json,
    run_yolo_board_observation_for_round,
    homography_front_to_side,
)
from vlm_to_actions import plan_vla_instructions, validate_instructions, load_env

import re

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("real_eval")


def _wrap_text_lines(text: str, *, max_chars: int, max_lines: int) -> list[str]:
    lines: list[str] = []
    for raw_line in (text or "").replace("\r", "").split("\n"):
        s = raw_line.strip()
        if not s:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        while s:
            if len(lines) >= max_lines:
                lines.append("...")
                return lines
            lines.append(s[:max_chars])
            s = s[max_chars:].lstrip()
    return lines[:max_lines] if lines else [""]


def _draw_prompt_strip_bgr(strip_h: int, strip_w: int, vlm_prompt: str, vla_prompt: str) -> np.ndarray:
    strip = np.zeros((strip_h, strip_w, 3), dtype=np.uint8)
    y = 16
    fs = 0.42
    gap = 15
    for label, body, color in (
        ("VLM user prompt", vlm_prompt or "(empty)", (180, 220, 255)),
        ("VLA prompt (policy task)", vla_prompt or "(empty)", (200, 255, 200)),
    ):
        cv2.putText(strip, label + ":", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        y += gap
        for line in _wrap_text_lines(body, max_chars=max(24, strip_w // 7), max_lines=8):
            cv2.putText(strip, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (230, 230, 230), 1, cv2.LINE_AA)
            y += 14
            if y > strip_h - 8:
                return strip
        y += 8
    return strip


def _compose_record_frame_bgr(
    front_rgb: np.ndarray,
    side_rgb: Optional[np.ndarray],
    *,
    vlm_prompt: str,
    vla_prompt: str,
    strip_h: int = 160,
) -> np.ndarray:
    """Side-by-side RGB cams + bottom strip with VLM / VLA text (BGR output for VideoWriter)."""
    front_bgr = cv2.cvtColor(front_rgb, cv2.COLOR_RGB2BGR)
    if side_rgb is not None:
        sh, sw = side_rgb.shape[:2]
        fh, fw = front_rgb.shape[:2]
        if (sh, sw) != (fh, fw):
            side_rgb = cv2.resize(side_rgb, (fw, fh), interpolation=cv2.INTER_AREA)
        side_bgr = cv2.cvtColor(side_rgb, cv2.COLOR_RGB2BGR)
        top = np.hstack([front_bgr, side_bgr])
    else:
        top = front_bgr
    tw = int(top.shape[1])
    strip = _draw_prompt_strip_bgr(strip_h, tw, vlm_prompt, vla_prompt)
    return np.vstack([top, strip])


def _overlay_prompt_banner_bgr(bgr: np.ndarray, vlm_one_line: str, vla_one_line: str) -> np.ndarray:
    """Light banner at bottom for live OpenCV windows."""
    out = bgr.copy()
    h, w = out.shape[:2]
    bar_h = 44
    cv2.rectangle(out, (0, h - bar_h), (w, h), (32, 32, 32), -1)
    cv2.putText(
        out,
        vla_one_line[: max(1, w // 8)],
        (6, h - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 255, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        vlm_one_line[: max(1, w // 8)],
        (6, h - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (200, 220, 255),
        1,
        cv2.LINE_AA,
    )
    return out


class _VideoSink:
    """Lazily opens cv2.VideoWriter on first frame."""

    def __init__(self, path: Path, fps: float) -> None:
        self.path = Path(path)
        self.fps = float(max(fps, 1e-3))
        self._writer: Optional[cv2.VideoWriter] = None

    def write(self, frame_bgr: np.ndarray) -> None:
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            h, w = frame_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (w, h))
            if not self._writer.isOpened():
                raise RuntimeError(f"Failed to open VideoWriter for {self.path}")
            log.info(f"Recording video -> {self.path} ({w}x{h} @ {self.fps} fps)")
        self._writer.write(frame_bgr)

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            log.info(f"Video saved: {self.path}")

def default_yolo_weights_str() -> str:
    local_path = Path(__file__).resolve().parent.parent / "runs" / "detect" / "train_manual_front_only_orange_purple" / "weights" / "best.pt"
    if local_path.is_file():
        return str(local_path)
    return "https://huggingface.co/zzq1zh/gym-so100-yolo-train/resolve/main/weights/best.pt"

def _parse_pick_instruction(line: str) -> Optional[Tuple[str, int]]:
    match = re.match(r"^Pick up the (\w+) cube and place it in grid cell (\d+)\.?$", line.strip(), re.IGNORECASE)
    if not match:
        return None
    color = match.group(1).lower()
    cell = int(match.group(2))
    return color, cell

def run_vlm_planning(
    front_bgr: np.ndarray,
    side_bgr: np.ndarray,
    calib_json: Path,
    yolo_weights: str,
    user_text: str,
    yolo_conf: float = 0.25,
) -> Tuple[List[str], np.ndarray, np.ndarray, Optional[Dict[str, Any]], str]:
    """
    Plan with VLM using ``user_text`` + YOLO board text merged for the API call.

    Returns ``vlm_user_prompt``: **only** ``user_text`` (stripped), for display/recording — not the merged planner string.
    """
    load_env()
    calib = load_calibration_json(calib_json)
    cf = calib["centers_front"]
    cs = calib["centers_side"]
    view_mapping = calib.get("view_mapping")

    yolo_text, _, _ = run_yolo_board_observation_for_round(
        weights=yolo_weights,
        front_bgr=front_bgr,
        side_bgr=side_bgr,
        centers_front=cf,
        centers_side=cs,
        yolo_conf=yolo_conf,
        yolo_imgsz=640,
        yolo_device=None,
    )

    merged = (user_text + "\n\n" + yolo_text).strip() if user_text else yolo_text
    vlm_user_prompt = (user_text or "").strip()

    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="real_eval_"))
    try:
        pf, ps = tmp / "first_front.png", tmp / "first_side.png"
        cv2.imwrite(str(pf), front_bgr)
        cv2.imwrite(str(ps), side_bgr)

        instructions, raw_text, budget = plan_vla_instructions(
            merged, image_paths=[pf, ps], inventory_from_vlm=True,
        )

        log.info(f"VLM Budget (Inventory): {budget}")
        log.info(f"VLM Raw Response:\n{raw_text}\n")

        ok, errs = validate_instructions(instructions)
        if errs:
            log.warning(f"Some instructions failed validation: {errs}")
        instructions = ok if ok else instructions

        log.info(f"VLM planned {len(instructions)} instructions:")
        for i, inst in enumerate(instructions):
            log.info(f"  [{i}] {inst}")

        return instructions, cf, cs, view_mapping, vlm_user_prompt
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def run_one_instruction(
    robot: Any,
    policy: PreTrainedPolicy,
    preprocessor: Any,
    postprocessor: Any,
    dataset: LeRobotDataset,
    instruction: str,
    events: dict,
    fps: int,
    timeout_s: float,
    beam_state: Optional[BeamOverlayState],
    centers_front: np.ndarray,
    centers_side: np.ndarray,
    front_key: str,
    side_key: str,
    calib: dict,
    show_vis: bool = False,
    debug_dir: Optional[Path] = None,
    *,
    vlm_prompt: str = "",
    show_prompt_overlay: bool = False,
    video_sink: Optional[_VideoSink] = None,
):
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    device = get_safe_torch_device(policy.config.device)

    parsed = _parse_pick_instruction(instruction)
    if not parsed:
        log.warning(f"Cannot parse instruction: {instruction}")
        return

    want_color, target_cell = parsed

    ref_h_f, ref_w_f = calib.get("_ref_h_f"), calib.get("_ref_w_f")
    ref_h_s, ref_w_s = calib.get("_ref_h_s"), calib.get("_ref_w_s")

    log.info("Generating SAM2 beam overlays...")
    obs = robot.get_observation()

    has_front = front_key in obs
    has_side = side_key in obs

    if (has_front or has_side) and beam_state is not None:
        front_rgb = obs[front_key] if has_front else None
        side_rgb = obs[side_key] if has_side else None

        front_bgr = cv2.cvtColor(front_rgb, cv2.COLOR_RGB2BGR) if front_rgb is not None else None
        side_bgr = cv2.cvtColor(side_rgb, cv2.COLOR_RGB2BGR) if side_rgb is not None else None

        if ref_h_f is None and front_bgr is not None:
            ref_h_f, ref_w_f = front_bgr.shape[:2]
        if ref_h_s is None and side_bgr is not None:
            ref_h_s, ref_w_s = side_bgr.shape[:2]
            
        beam_state.capture_reference_shapes((int(ref_h_f or 480), int(ref_w_f or 640)), (int(ref_h_s or 480), int(ref_w_s or 640)))
        beam_state.compute_statics(
            want_color=want_color,
            target_cell=target_cell,
            bgr_f=front_bgr if front_bgr is not None else side_bgr,
            bgr_s=side_bgr if side_bgr is not None else front_bgr,
        )

        bgr_f_for_overlay = front_bgr if front_bgr is not None else side_bgr
        bgr_s_for_overlay = side_bgr if side_bgr is not None else front_bgr
        front_bgr_overlaid, side_bgr_overlaid = beam_state.overlay(bgr_f_for_overlay, bgr_s_for_overlay)
        if front_bgr is not None:
            obs[front_key] = cv2.cvtColor(front_bgr_overlaid, cv2.COLOR_BGR2RGB)
        if side_bgr is not None:
            obs[side_key] = cv2.cvtColor(side_bgr_overlaid, cv2.COLOR_BGR2RGB)

        if debug_dir is not None:
            tag = re.sub(r'[^a-zA-Z0-9_-]', '_', instruction)[:60]
            debug_dir.mkdir(parents=True, exist_ok=True)
            if front_bgr is not None:
                cv2.imwrite(str(debug_dir / f"{tag}_front_before.png"), front_bgr)
                cv2.imwrite(str(debug_dir / f"{tag}_front_after.png"), front_bgr_overlaid)
            if side_bgr is not None:
                cv2.imwrite(str(debug_dir / f"{tag}_side_before.png"), side_bgr)
                cv2.imwrite(str(debug_dir / f"{tag}_side_after.png"), side_bgr_overlaid)

    start_t = time.perf_counter()
    deadline = start_t + timeout_s if timeout_s > 0 else float("inf")
    log.info(f"Executing instruction loop (timeout={timeout_s if timeout_s > 0 else 'none'})...")

    while time.perf_counter() < deadline:
        loop_start = time.perf_counter()

        if events.get("exit_early") or events.get("stop_recording"):
            events["exit_early"] = False
            break

        obs = robot.get_observation()

        if beam_state is not None:
            if has_front:
                front_bgr_cur = cv2.cvtColor(obs[front_key], cv2.COLOR_RGB2BGR)
            else:
                front_bgr_cur = None
            if has_side:
                side_bgr_cur = cv2.cvtColor(obs[side_key], cv2.COLOR_RGB2BGR)
            else:
                side_bgr_cur = None
            f_in = front_bgr_cur if front_bgr_cur is not None else side_bgr_cur
            s_in = side_bgr_cur if side_bgr_cur is not None else front_bgr_cur
            front_bgr_cur, side_bgr_cur = beam_state.overlay(f_in, s_in)
            if has_front:
                obs[front_key] = cv2.cvtColor(front_bgr_cur, cv2.COLOR_BGR2RGB)
            if has_side:
                obs[side_key] = cv2.cvtColor(side_bgr_cur, cv2.COLOR_BGR2RGB)

        obs_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)

        if show_vis:
            if front_key in obs:
                disp_f = cv2.cvtColor(obs[front_key], cv2.COLOR_RGB2BGR)
                if show_prompt_overlay:
                    disp_f = _overlay_prompt_banner_bgr(
                        disp_f,
                        "VLM user: " + (vlm_prompt.replace("\n", " ")[:200]),
                        "VLA: " + instruction,
                    )
                cv2.imshow("Front View (Beam Overlay)", disp_f)
            if side_key in obs:
                disp_s = cv2.cvtColor(obs[side_key], cv2.COLOR_RGB2BGR)
                if show_prompt_overlay:
                    disp_s = _overlay_prompt_banner_bgr(
                        disp_s,
                        "VLM user: " + (vlm_prompt.replace("\n", " ")[:200]),
                        "VLA: " + instruction,
                    )
                cv2.imshow("Side View (Beam Overlay)", disp_s)
            cv2.waitKey(1)

        if video_sink is not None and front_key in obs:
            fr = obs[front_key]
            sr = obs[side_key] if side_key in obs else None
            rec = _compose_record_frame_bgr(fr, sr, vlm_prompt=vlm_prompt, vla_prompt=instruction)
            video_sink.write(rec)

        if events.get("exit_early") or events.get("stop_recording"):
            events["exit_early"] = False
            break

        action_values = predict_action(
            observation=obs_frame,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
            task=instruction,
            robot_type=robot.robot_type,
        )

        if isinstance(action_values, torch.Tensor):
            action_values = action_values.to(torch.float32)

        act_dict = make_robot_action(action_values, dataset.features)
        robot.send_action(act_dict)

        action_frame = build_dataset_frame(dataset.features, act_dict, prefix=ACTION)
        frame = {
            **obs_frame,
            **action_frame,
            "task": instruction,
        }
        dataset.add_frame(frame)

        dt = time.perf_counter() - loop_start
        sleep_time = 1.0 / fps - dt
        precise_sleep(max(sleep_time, 0.0))

    log.info(f"Saving episode for instruction: {instruction}")
    dataset.save_episode()


def build_robot_config(args) -> SOFollowerRobotConfig:
    cameras = {}
    cam_specs = [
        (args.cam_front_key, args.cam_front_index),
        (args.cam_side_key, args.cam_side_index),
        (args.cam_third_key, args.cam_third_index),
    ]
    for key, idx in cam_specs:
        if idx is not None and idx >= 0:
            cameras[key] = OpenCVCameraConfig(
                index_or_path=idx, width=640, height=480, fps=30,
            )
    return SOFollowerRobotConfig(
        port=args.port,
        id=args.robot_id,
        cameras=cameras,
    )


def main():
    p = argparse.ArgumentParser(description="VLM + SAM2 Beam Overlay + xVLA Real Robot Inference")
    # Robot
    p.add_argument("--port", type=str, default="COM6")
    p.add_argument("--robot-id", type=str, default="my_follower_arm")
    p.add_argument("--cam-front-key", type=str, default="image")
    p.add_argument("--cam-front-index", type=int, default=1)
    p.add_argument("--cam-side-key", type=str, default="image2")
    p.add_argument("--cam-side-index", type=int, default=3)
    p.add_argument("--cam-third-key", type=str, default="image3")
    p.add_argument("--cam-third-index", type=int, default=0)

    # Policy
    p.add_argument("--policy-path", type=str, required=True)
    p.add_argument("--dataset-repo-id", type=str, default="yangxinye/eval_real_so101_record_v2_masked-10000steps")

    # VLM Planning
    p.add_argument("--user-text", type=str, default="Plan reasonable pick-and-place instructions (with colors and grid cells 0-8) based on the desktop state.")
    p.add_argument("--calib-json", type=Path, default=None)
    p.add_argument("--yolo-weights", type=str, default=None)
    p.add_argument("--yolo-conf", type=float, default=0.01)
    p.add_argument("--instructions-json", type=Path, default=None)

    # SAM2 Beam Overlay
    p.add_argument("--no-beam", action="store_true", help="Disable SAM2 beam overlay (run raw)")
    p.add_argument("--sam2-model", type=str, default="facebook/sam2-hiera-small")
    p.add_argument("--sam2-device", type=str, default="cuda")
    p.add_argument("--debug-beam-dir", type=Path, default=None, help="Save debug images of SAM2 beam overlays")

    # Execution
    p.add_argument("--per-instruction-timeout", type=float, default=0.0, help="Max execution seconds per step. <=0 means unlimited.")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--vis", action="store_true", help="Enable real-time OpenCV visualization")
    p.add_argument(
        "--record-video",
        type=Path,
        default=None,
        metavar="PATH",
        help="Record front|side + VLM/VLA prompt strip to this MP4 (e.g. runs/real_eval.mp4)",
    )
    p.add_argument(
        "--record-fps",
        type=float,
        default=None,
        help="FPS for --record-video (default: same as --fps)",
    )
    p.add_argument(
        "--no-show-prompts",
        action="store_true",
        help="Do not draw VLM/VLA prompt banners on --vis windows (prompts are still embedded in --record-video)",
    )
    p.add_argument("--push-to-hub", action="store_true", help="Push recorded dataset to Hub after execution")
    p.add_argument("--private", action="store_true", help="Upload dataset as private repo")

    args = p.parse_args()
    init_logging()

    dataset = None
    listener = None
    video_sink: Optional[_VideoSink] = None

    yolo_weights = args.yolo_weights if args.yolo_weights else default_yolo_weights_str()

    robot_cfg = build_robot_config(args)
    robot = make_robot_from_config(robot_cfg)
    robot.connect()
    log.info("Robot connected.")

    try:
        first_obs = robot.get_observation()
        front_rgb = first_obs[args.cam_front_key]
        side_rgb = first_obs[args.cam_side_key]
        
        front_bgr = cv2.cvtColor(front_rgb, cv2.COLOR_RGB2BGR)
        side_bgr = cv2.cvtColor(side_rgb, cv2.COLOR_RGB2BGR)

        if args.instructions_json and args.instructions_json.is_file():
            plan = json.loads(args.instructions_json.read_text(encoding="utf-8"))
            instructions = plan["instructions"]
            vlm_prompt = (
                plan.get("vlm_prompt")
                or plan.get("user_text")
                or f"(loaded instructions from {args.instructions_json})"
            )
            if "calibration" in plan:
                cf = np.array(plan["calibration"]["centers_front"], dtype=np.float32).reshape(9, 2)
                cs = np.array(plan["calibration"]["centers_side"], dtype=np.float32).reshape(9, 2)
                view_mapping = plan["calibration"].get("view_mapping")
            elif args.calib_json:
                calib = load_calibration_json(args.calib_json)
                cf, cs = calib["centers_front"], calib["centers_side"]
                view_mapping = calib.get("view_mapping")
            else:
                raise RuntimeError("Need calibration: provide --calib-json or include in instructions JSON")
        else:
            if not args.calib_json:
                raise RuntimeError("--calib-json must be provided for live VLM planning.")
            instructions, cf, cs, view_mapping, vlm_prompt = run_vlm_planning(
                front_bgr, side_bgr,
                calib_json=args.calib_json,
                yolo_weights=yolo_weights,
                user_text=args.user_text,
                yolo_conf=args.yolo_conf,
            )

        if not instructions:
            return 1

        log.info("======== VLM user prompt (input only; YOLO/system text not shown here) ========\n%s\n========", vlm_prompt)
        log.info("Planned %d instruction(s); VLA uses each line as the policy `task` string.", len(instructions))

        if args.record_video:
            rfps = float(args.record_fps) if args.record_fps is not None else float(args.fps)
            video_sink = _VideoSink(args.record_video, rfps)

        show_prompt_overlay = bool(args.vis) and not bool(args.no_show_prompts)

        if view_mapping is None or homography_front_to_side(view_mapping) is None:
            log.warning("Calibration missing 'view_mapping' (homography front->side).")

        calib_meta = {
            "_ref_h_f": front_bgr.shape[0],
            "_ref_w_f": front_bgr.shape[1],
            "_ref_h_s": side_bgr.shape[0],
            "_ref_w_s": side_bgr.shape[1],
            "view_mapping": view_mapping,
            "centers_front": cf,
            "centers_side": cs,
        }

        log.info(f"Loading xVLA policy from {args.policy_path}...")
        policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
        policy_cfg.pretrained_path = args.policy_path

        _, _, robot_obs_processor = make_default_processors()
        dataset_features = combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=make_default_processors()[0],
                initial_features=create_initial_features(action=robot.action_features),
                use_videos=True,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=robot_obs_processor,
                initial_features=create_initial_features(observation=robot.observation_features),
                use_videos=True,
            ),
        )
        dataset = LeRobotDataset.create(
            args.dataset_repo_id, args.fps,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=True,
        )
        policy = make_policy(policy_cfg, ds_meta=dataset.meta)
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=args.policy_path,
            dataset_stats=rename_stats(dataset.meta.stats, {}),
            preprocessor_overrides={
                "device_processor": {"device": policy_cfg.device},
                "rename_observations_processor": {"rename_map": {}},
            },
        )
        log.info("Policy loaded.")

        beam_state = None
        if not args.no_beam:
            log.info("Loading YOLO model...")
            yolo_model = load_yolo_model(yolo_weights)
            log.info(f"Loading SAM2 predictor ({args.sam2_model})...")
            sam2_predictor = make_sam2_image_predictor(args.sam2_model, args.sam2_device)
            beam_state = BeamOverlayState(
                calib=calib_meta,
                predictor=sam2_predictor,
                yolo_model=yolo_model,
                yolo_conf=args.yolo_conf,
            )
            log.info("SAM2 beam overlay ready.")

        listener, events = init_keyboard_listener()

        for i, instruction in enumerate(instructions):
            if events.get("stop_recording"):
                break
            log.info(f"\n[{i+1}/{len(instructions)}] {instruction}")
            log.info("-------- VLA prompt (policy `task`) --------\n%s\n--------", instruction)
            run_one_instruction(
                robot=robot,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                dataset=dataset,
                instruction=instruction,
                events=events,
                fps=args.fps,
                timeout_s=args.per_instruction_timeout,
                beam_state=beam_state,
                centers_front=cf,
                centers_side=cs,
                front_key=args.cam_front_key,
                side_key=args.cam_side_key,
                calib=calib_meta,
                show_vis=args.vis,
                debug_dir=args.debug_beam_dir,
                vlm_prompt=vlm_prompt,
                show_prompt_overlay=show_prompt_overlay,
                video_sink=video_sink,
            )
            if i < len(instructions) - 1:
                time.sleep(2.0)
        return 0
    except Exception as e:
        log.exception(e)
        return 1
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        if dataset:
            log.info("Finalizing dataset...")
            dataset.finalize()
            if args.push_to_hub:
                log.info("Pushing dataset to Hub...")
                try:
                    dataset.push_to_hub(
                        tags=["eval", "beam-overlay"],
                        private=args.private,
                        push_videos=True,
                        upload_large_folder=True,
                        license="apache-2.0",
                        robot_type=robot.name,
                        policy_path=args.policy_path,
                        tasks=", ".join(instructions) if 'instructions' in locals() else "",
                    )
                    log.info("Dataset pushed successfully.")
                except Exception as e:
                    log.error(f"Failed to push dataset: {e}")
        if robot.is_connected:
            robot.disconnect()
        if listener:
            listener.stop()
        if video_sink is not None:
            video_sink.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    sys.exit(main())
