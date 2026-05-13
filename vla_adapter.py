"""
LeRobot / VLA helpers (``vla_adapter.py``), aligned with ``sim_eval.py``:

- Load pre/post processors via ``make_pre_post_processors(..., pretrained_path=...)`` and the
  same ``preprocessor_overrides`` / ``postprocessor_overrides`` as sim_eval (device, stats,
  rename_map, unnormalizer).
- Build raw batches with **dataset-style keys** ``observation.images.cam_global`` / ``cam_oblique``
  / ``left_wrist`` so ``RenameObservationsProcessorStep`` in the checkpoint matches training.

Optional: ``load_lerobot_xvla_policy`` for config/weights split (base config + fine-tuned weights).

Constants and observation layout follow ``sim_scenes.py`` (``IDX_JAW`` + 1 qpos = arm + jaw) and
``sim_eval.build_obs_dict`` (``data.qpos[:6]``, Renderer 480×640).

Also includes **pick-place MuJoCo control** (merged from the former ``pick_place_xvla_sim.py``):
dataset-style 6-D actions → ``apply_action_to_sim``, optional ``GraspAssistState`` / ``physics_substeps_with_grasp``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

from sim_scenes import (
    ARM_DOF,
    CTRL_SUBSTEPS,
    CUBE_Z,
    GRASP_LOCAL_OFFSET,
    GRID_CENTERS,
    IDX_JAW,
    JAW_PRE_GRASP,
    _body_pos,
    _set_body_pos,
)

# -----------------------------------------------------------------------------
# sim_scenes.py: qpos indices 0..IDX_JAW are arm (ARM_DOF) + jaw — same slice as
# sim_eval ``data.qpos[:6]``.
# -----------------------------------------------------------------------------
PICK_PLACE_STATE_QPOS_PREFIX_LEN = int(IDX_JAW) + 1

# sim_eval.MjModel Renderer defaults (so100_puzzle / pick-place scene).
PICK_PLACE_OFFSCREEN_RENDER_HEIGHT = 480
PICK_PLACE_OFFSCREEN_RENDER_WIDTH = 640

# -----------------------------------------------------------------------------
# Pick-place MuJoCo control (same as former ``pick_place_xvla_sim.py``; shared with sim_eval).
# -----------------------------------------------------------------------------
RECORD_INTERVAL = 10

SO100_POLICY_ACTION_DIM = int(IDX_JAW) + 1  # arm + jaw = 6


def policy_physics_substeps_per_decision() -> int:
    """Physics substeps per policy command (sim_eval / dataset cadence)."""
    return int(CTRL_SUBSTEPS) * int(RECORD_INTERVAL)


def policy_vector_to_so100_action(
    act_np: np.ndarray,
    *,
    action_slice: slice | None = None,
    out_dim: int = SO100_POLICY_ACTION_DIM,
) -> np.ndarray:
    """
    Map a flat policy output vector to the **6** scalars expected by ``apply_action_to_sim``.

    Use ``action_slice`` when the policy outputs EE6D etc. and you intentionally select a window
    (experimental); trained SO100 policies usually need no slice.
    """
    a = np.asarray(act_np, dtype=np.float64).reshape(-1)
    if action_slice is not None:
        a = a[action_slice].copy()
    if a.shape[0] < out_dim:
        raise ValueError(
            f"Policy action length {a.shape[0]} < SO100 control dim {out_dim} "
            f"(after slice {action_slice!r}). Use --action-slice or --xvla-skip-mujoco-control."
        )
    return a[:out_dim].astype(np.float64)


def apply_action_to_sim(model: mujoco.MjModel, data: mujoco.MjData, action: np.ndarray) -> None:
    data.ctrl[:ARM_DOF] = action[:ARM_DOF]
    data.qpos[IDX_JAW] = float(action[IDX_JAW])
    data.ctrl[IDX_JAW] = float(action[IDX_JAW])


def get_grasp_pad(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    ee_pos = data.site_xpos[sid]
    ee_mat = data.site_xmat[sid].reshape(3, 3)
    return ee_pos + ee_mat @ GRASP_LOCAL_OFFSET


def _fix_cube_free_joint(model: mujoco.MjModel, data: mujoco.MjData, cname: str) -> None:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cname)
    ja = model.body_jntadr[bid]
    if ja < 0:
        return
    qa = model.jnt_qposadr[ja]
    data.qpos[qa + 3 : qa + 7] = [1.0, 0.0, 0.0, 0.0]
    da = model.jnt_dofadr[ja]
    data.qvel[da : da + 6] = 0.0


@dataclass
class GraspAssistState:
    """Mirrors ``record_pick_place_dataset_v3`` grasp_lock + locked_cubes after place."""

    cube_name: str
    perfect_pos: np.ndarray  # (3,) target placement (xy from grid, z = CUBE_Z)
    grasp_lock: dict = field(
        default_factory=lambda: {
            "active": False,
            "cube": None,
            "offset_x": 0.0,
            "offset_y": 0.0,
            "offset_z": 0.0,
        }
    )
    locked_cubes: dict[str, np.ndarray] = field(default_factory=dict)
    enabled: bool = True

    def maybe_engage(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if not self.enabled or self.grasp_lock["active"] or self.cube_name in self.locked_cubes:
            return
        jaw = float(data.qpos[IDX_JAW])
        if jaw > JAW_PRE_GRASP + 0.04:
            return
        pad = get_grasp_pad(model, data)
        cpos = _body_pos(model, data, self.cube_name)
        dxy = float(np.linalg.norm(pad[:2] - cpos[:2]))
        dz = float(abs(pad[2] - cpos[2]))
        if dxy < 0.030 and dz < 0.050:
            self.grasp_lock.update(
                {
                    "active": True,
                    "cube": self.cube_name,
                    "offset_x": cpos[0] - pad[0],
                    "offset_y": cpos[1] - pad[1],
                    "offset_z": cpos[2] - pad[2],
                }
            )

    def maybe_release(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if not self.enabled or not self.grasp_lock["active"]:
            return
        jaw = float(data.qpos[IDX_JAW])
        opening = jaw > JAW_PRE_GRASP + 0.10
        wide_open = jaw > 0.35
        if not opening:
            return
        pad = get_grasp_pad(model, data)
        near_place = float(np.linalg.norm(pad[:2] - self.perfect_pos[:2])) < 0.085
        if not (near_place or wide_open):
            return
        self.grasp_lock["active"] = False
        _set_body_pos(model, data, self.cube_name, self.perfect_pos.copy())
        self.locked_cubes[self.cube_name] = self.perfect_pos.copy()
        mujoco.mj_forward(model, data)

    def apply_glue_and_locks(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self.grasp_lock["active"]:
            cname = self.grasp_lock["cube"]
            pad = get_grasp_pad(model, data)
            pos = np.array(
                [
                    pad[0] + self.grasp_lock["offset_x"],
                    pad[1] + self.grasp_lock["offset_y"],
                    pad[2] + self.grasp_lock["offset_z"],
                ]
            )
            _set_body_pos(model, data, cname, pos)
            _fix_cube_free_joint(model, data, cname)

        for lname, lpos in self.locked_cubes.items():
            _set_body_pos(model, data, lname, lpos)
            _fix_cube_free_joint(model, data, lname)

        mujoco.mj_forward(model, data)


def physics_substeps_with_grasp(
    model: mujoco.MjModel, data: mujoco.MjData, n: int, gstate: GraspAssistState | None
) -> None:
    for _ in range(n):
        mujoco.mj_step(model, data)
        if gstate is not None:
            gstate.maybe_engage(model, data)
            gstate.maybe_release(model, data)
            gstate.apply_glue_and_locks(model, data)


def grasp_target_from_names(cube_body_name: str, target_cell_index: int) -> tuple[str, np.ndarray]:
    """Return ``(cube_name, perfect_pos)`` for ``GraspAssistState``."""
    if not (0 <= target_cell_index < len(GRID_CENTERS)):
        raise ValueError(f"target_cell_index must be in [0, {len(GRID_CENTERS)}), got {target_cell_index}")
    gc = GRID_CENTERS[target_cell_index]
    perfect_pos = np.array([gc[0], gc[1], CUBE_Z], dtype=np.float64)
    return cube_body_name, perfect_pos


# -----------------------------------------------------------------------------
# Same rename_map as sim_eval / record_pick_place_dataset_v3 training.
# -----------------------------------------------------------------------------
DEFAULT_RENAME_MAP: dict[str, str] = {
    "observation.images.cam_global": "observation.images.image",
    "observation.images.cam_oblique": "observation.images.image2",
    "observation.images.left_wrist": "observation.images.image3",
}

# Env camera names -> keys used *before* preprocessor rename (sim_eval-style).
TRAINING_IMAGE_KEYS_BY_ENV_CAMERA: dict[str, str] = {
    "cam_global": "observation.images.cam_global",
    "cam_oblique": "observation.images.cam_oblique",
    "left_wrist": "observation.images.left_wrist",
}

REQUIRED_ENV_CAMERAS: tuple[str, ...] = ("cam_global", "cam_oblique", "left_wrist")
SUPPORTED_POLICY_TYPES: tuple[str, ...] = ("xvla", "smolvla")

# When --dataset-root / --xvla-dataset-root is omitted, try these under the repo root (v2 first).
DEFAULT_LEROBOT_DATASET_DIR_CANDIDATES: tuple[str, ...] = (
    "dataset_large_merged_v2",
    "dataset_large_merged",
)


def lerobot_dataset_has_meta(root: Path) -> bool:
    return (root / "meta" / "info.json").is_file()


def resolve_lerobot_dataset_root_for_eval(
    explicit: Path | str | None,
    *,
    script_parent: Path,
) -> Path:
    """
    Required for sim_eval: return a LeRobot v3 dataset root that contains ``meta/info.json``.

    If ``explicit`` is set, use it; otherwise try ``DEFAULT_LEROBOT_DATASET_DIR_CANDIDATES`` under ``script_parent``.
    """
    if explicit is not None:
        explicit_s = str(explicit).strip()
        p = Path(explicit_s).expanduser()
        # Treat existing local paths as filesystem roots; otherwise try as a Hub repo id.
        if p.exists():
            root = p.resolve()
            if not lerobot_dataset_has_meta(root):
                raise FileNotFoundError(
                    f"Not a valid LeRobot dataset root (missing meta/info.json): {root}"
                )
            print(f"[sim_eval] dataset root: {root}")
            return root
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except Exception as e:
            raise FileNotFoundError(
                f"Dataset root does not exist locally: {explicit_s!r}. "
                "Also failed to import LeRobotDataset for remote repo resolution."
            ) from e
        try:
            ds = LeRobotDataset(repo_id=hub_or_posix_path(explicit_s))
            root = Path(ds.root).resolve()
            if not lerobot_dataset_has_meta(root):
                raise FileNotFoundError(
                    f"Resolved remote dataset but missing meta/info.json: {root}"
                )
            print(f"[sim_eval] dataset root (remote): {root} <- {explicit_s}")
            return root
        except Exception as e:
            raise FileNotFoundError(
                f"Not a valid local dataset path or remote repo id: {explicit_s!r}. "
                "Expected a local root containing meta/info.json, or a downloadable LeRobot repo id."
            ) from e
    for name in DEFAULT_LEROBOT_DATASET_DIR_CANDIDATES:
        c = (script_parent / name).resolve()
        if lerobot_dataset_has_meta(c):
            print(f"[sim_eval] dataset root (auto): {c}")
            return c
    raise FileNotFoundError(
        "No local LeRobot dataset found (needs meta/info.json). Either:\n"
        "  1) Download a LeRobot v3 dataset that matches training (e.g. from Hugging Face); or\n"
        "  2) Pass --dataset-root /path/to/dataset (must contain meta/info.json) when running sim_eval."
    )


def resolve_lerobot_dataset_root_optional(
    explicit: Path | str | None,
    *,
    script_parent: Path,
) -> Path | None:
    """
    Optional stats for visualization: explicit path errors if invalid; otherwise try candidates and return None if none match.
    """
    if explicit is not None:
        root = Path(explicit).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"LeRobot dataset root is not a directory: {root}")
        if not lerobot_dataset_has_meta(root):
            raise FileNotFoundError(f"LeRobot dataset root missing meta/info.json: {root}")
        return root
    for name in DEFAULT_LEROBOT_DATASET_DIR_CANDIDATES:
        c = (script_parent / name).resolve()
        if lerobot_dataset_has_meta(c):
            return c
    return None


def hub_or_posix_path(s: str) -> str:
    return str(s).replace("\\", "/")


def resolve_local_policy_dir(path: Path) -> Path:
    """
    ``lerobot-train`` checkpoints: ``checkpoints/005000/pretrained_model/config.json``.
    Accept step folder or ``pretrained_model`` directly.
    """
    path = path.resolve()
    if (path / "config.json").is_file():
        return path
    nested = path / "pretrained_model"
    if (nested / "config.json").is_file():
        return nested
    raise FileNotFoundError(
        f"Expected policy config at '{path / 'config.json'}' or '{nested / 'config.json'}'. "
        "Pass the step folder (e.g. .../checkpoints/005000) or .../pretrained_model directly."
    )


def resolve_policy_pretrained_path(ref: str | Path) -> str:
    """
    Match ``sim_eval`` / ``resolve_local_policy_dir``:

    LeRobot step folders often look like ``.../checkpoints/005000`` with ``config.json`` under
    nested ``pretrained_model/``. Loading processors or policy from the step folder **without**
    resolving breaks (wrong/missing ``policy_preprocessor.json``).

    Non-directory refs (Hub ids, files) are returned unchanged (posix slashes only).
    """
    raw = hub_or_posix_path(str(ref))
    p = Path(raw)
    if not p.is_dir():
        return raw
    try:
        return hub_or_posix_path(str(resolve_local_policy_dir(p)))
    except FileNotFoundError:
        return hub_or_posix_path(str(p.resolve()))


def _config_type_from_ref(ref: str) -> str | None:
    """Best-effort read of ``type`` from local or Hub ``config.json``."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError

    p = Path(ref)
    if p.is_dir():
        try:
            p = resolve_local_policy_dir(p)
        except FileNotFoundError:
            pass
    if p.is_dir() and (p / "config.json").is_file():
        try:
            meta = json.loads((p / "config.json").read_text())
            t = str(meta.get("type", "")).strip().lower()
            return t or None
        except (OSError, json.JSONDecodeError):
            return None
    try:
        cfg_path = hf_hub_download(repo_id=str(ref), filename="config.json")
        meta = json.loads(Path(cfg_path).read_text())
        t = str(meta.get("type", "")).strip().lower()
        return t or None
    except (HfHubHTTPError, OSError, json.JSONDecodeError):
        return None


def is_xvla_policy_ref(policy_path: str | None, policy_repo_id: str | None) -> bool:
    """
    Legacy name: returns whether the ref looks like a supported VLA policy (xvla/smolvla).
    """
    for ref in (policy_path, policy_repo_id):
        if ref and ("xvla" in ref.lower() or "smolvla" in ref.lower()):
            return True
    cfg_src = policy_repo_id or policy_path
    if not cfg_src:
        return False
    typ = _config_type_from_ref(str(cfg_src))
    return typ in SUPPORTED_POLICY_TYPES


def resolve_supported_policy_type(
    policy_path: str | None,
    policy_repo_id: str | None,
) -> str | None:
    """
    Return supported policy type (xvla/smolvla), or None if unknown.

    Prefer ``policy_repo_id``, then ``policy_path``.
    """
    for ref in (policy_repo_id, policy_path):
        if not ref:
            continue
        t = _config_type_from_ref(str(ref))
        if t in SUPPORTED_POLICY_TYPES:
            return t
    for ref in (policy_repo_id, policy_path):
        if not ref:
            continue
        low = str(ref).lower()
        for t in SUPPORTED_POLICY_TYPES:
            if t in low:
                return t
    return None


def ordered_visual_feature_keys(config: Any) -> list[str]:
    from lerobot.configs.types import FeatureType

    out: list[str] = []
    for name, ft in config.input_features.items():
        if ft.type == FeatureType.VISUAL:
            out.append(name)
    return out


def default_rename_map_for_policy(policy_cfg: Any) -> dict[str, str]:
    """
    Build a camera ``rename_map`` from policy input features.

    - XVLA often uses: image/image2/image3
    - SmolVLA often uses: camera1/camera2/camera3
    """
    vis = ordered_visual_feature_keys(policy_cfg)
    if len(vis) < 3:
        return dict(DEFAULT_RENAME_MAP)
    src = [
        "observation.images.cam_global",
        "observation.images.cam_oblique",
        "observation.images.left_wrist",
    ]
    dst = [str(vis[0]), str(vis[1]), str(vis[2])]
    if src == dst:
        return {}
    return {s: d for s, d in zip(src, dst)}


def _policy_class_for_type(policy_type: str):
    """Dynamically import the Policy class for ``policy_type``."""
    t = str(policy_type).lower().strip()
    if t == "xvla":
        import lerobot.policies.xvla.modeling_xvla  # noqa: F401
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

        return XVLAPolicy
    if t == "smolvla":
        import lerobot.policies.smolvla.modeling_smolvla  # noqa: F401
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        return SmolVLAPolicy
    raise ValueError(
        f"Unsupported policy type: {policy_type!r}; supported: {', '.join(SUPPORTED_POLICY_TYPES)}"
    )


def load_lerobot_xvla_policy(
    *,
    policy_path: str | None,
    policy_repo_id: str | None,
    device: str,
    strict_weights: bool = False,
) -> Any:
    """
    Legacy name: load a LeRobot VLA policy (xvla / smolvla).

    - Single of {path, repo_id}: ``XVLAPolicy.from_pretrained(that_id, strict=strict_weights)``.
    - Both set: load ``PreTrainedConfig`` from ``policy_path`` (base), weights from ``policy_repo_id``.
    """
    import safetensors.torch
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError

    from lerobot.configs.policies import PreTrainedConfig

    if not policy_path and not policy_repo_id:
        raise ValueError("Policy loading requires --policy.path and/or --policy.repo_id")

    policy_type = resolve_supported_policy_type(policy_path, policy_repo_id)
    if policy_type is None:
        raise ValueError(
            "Could not infer policy type (need config.type xvla/smolvla, or path/repo id containing xvla/smolvla)"
        )
    PolicyCls = _policy_class_for_type(policy_type)

    if policy_path and policy_repo_id and policy_path != policy_repo_id:
        config_ref = resolve_policy_pretrained_path(policy_path)
        config = PreTrainedConfig.from_pretrained(config_ref)
        config.device = device
        policy = PolicyCls(config)
        repo_p = Path(policy_repo_id)
        if repo_p.is_dir():
            model_file = str(repo_p / "model.safetensors")
            if not os.path.isfile(model_file):
                raise FileNotFoundError(f"Missing {model_file}")
        else:
            try:
                model_file = hf_hub_download(repo_id=str(policy_repo_id), filename="model.safetensors")
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"model.safetensors not found on the Hub at {policy_repo_id!r}"
                ) from e
        logging.info(
            "Loading %s weights from %s (config from %s)",
            policy_type,
            model_file,
            config_ref,
        )
        state_dict = safetensors.torch.load_file(model_file)
        encoder_key = "model.vlm.language_model.model.encoder.embed_tokens.weight"
        shared_key = "model.vlm.language_model.model.shared.weight"
        if encoder_key in state_dict:
            state_dict[shared_key] = state_dict[encoder_key]
        policy.load_state_dict(state_dict, strict=strict_weights)
        policy.model._apply_dtype()
        policy.to(device)
        policy.eval()
        return policy

    pretrained = policy_repo_id or policy_path
    assert pretrained is not None
    pretrained_resolved = resolve_policy_pretrained_path(pretrained)
    policy = PolicyCls.from_pretrained(
        pretrained_name_or_path=pretrained_resolved, strict=strict_weights
    )
    policy.config.device = device
    policy.to(device)
    policy.eval()
    return policy


def effective_xvla_tokenizer_max_length(config: Any, override: int | None = None) -> int:
    """
    Informative cap (used for logging / optional override). Hub configs may use 1024 while
    ``max_len_seq`` is smaller; ``load_xvla_pre_post_processors`` can apply ``override``.
    """
    if override is not None:
        return max(8, int(override))
    slack = 230
    return min(
        int(config.tokenizer_max_length),
        max(32, int(config.max_len_seq) - int(config.chunk_size) - slack),
    )


def xvla_image_step_presence(preprocessor: Any) -> tuple[bool, bool]:
    """Detect XVLA/SmolVLA-style image preprocessor steps (best effort)."""
    names: list[str | None] = []
    for step in preprocessor.steps:
        names.append(getattr(step.__class__, "_registry_name", None))
    names_l = [str(x).lower() for x in names if x is not None]
    return (
        any("image_to_float" in n for n in names_l),
        any("imagenet_normalize" in n for n in names_l),
    )


def imagenet_normalize_bchw(t: torch.Tensor) -> torch.Tensor:
    """Match ``XVLAImageNetNormalizeProcessorStep`` (values in ``[0, 1]``)."""
    from lerobot.datasets.factory import IMAGENET_STATS

    mean = torch.tensor(IMAGENET_STATS["mean"], dtype=t.dtype, device=t.device)
    std = torch.tensor(IMAGENET_STATS["std"], dtype=t.dtype, device=t.device)
    while mean.dim() < t.dim():
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    return (t - mean) / std


def pick_place_observation_state_tensor(
    joint_positions_for_state: np.ndarray,
    policy_config: Any,
) -> torch.Tensor:
    """
    Build ``observation.state`` like ``sim_eval.build_obs_dict``: take only the first
    ``PICK_PLACE_STATE_QPOS_PREFIX_LEN`` joint scalars (arm + jaw), then pad/truncate to
    ``robot_state_feature`` shape.
    """
    st = np.asarray(joint_positions_for_state, dtype=np.float32).reshape(-1)
    st = st[:PICK_PLACE_STATE_QPOS_PREFIX_LEN]
    state_dim = int(policy_config.robot_state_feature.shape[0])
    state = np.zeros(state_dim, dtype=np.float32)
    n = min(state_dim, st.shape[0])
    state[:n] = st[:n]
    return torch.from_numpy(state).unsqueeze(0)


def build_pick_place_obs_dict_for_xvla(
    renderer: Any,
    data: Any,
    task: str,
    *,
    has_image_to_float: bool = True,
    has_imagenet_normalize: bool = True,
    policy_config: Any | None = None,
    visual_task_guides: bool = False,
    guide_cube_name: str | None = None,
    guide_target_cell: int | None = None,
) -> dict[str, Any]:
    """
    Same observation layout as ``sim_eval.build_obs_dict`` (MuJoCo ``Renderer`` + ``MjData``).

    Joint prefix is ``data.qpos[:PICK_PLACE_STATE_QPOS_PREFIX_LEN]`` (``sim_scenes.py`` arm + jaw).
    If ``policy_config`` is set, state is padded/truncated to ``robot_state_feature.shape`` like
    ``build_xvla_raw_batch``; if ``None``, state is ``(1, PICK_PLACE_STATE_QPOS_PREFIX_LEN)`` only
    (legacy sim_eval shape).

    When ``visual_task_guides`` is true and ``guide_cube_name`` / ``guide_target_cell`` are set,
    the same capsule overlays as ``vla_to_actions.append_task_guide_geoms_to_scene`` are appended to ``renderer.scene``
    after each ``update_scene`` (three cameras), so policy inputs match recorded ``--visual-task-guides`` data.
    """
    if not isinstance(data, mujoco.MjData):
        raise TypeError(f"Expected mujoco.MjData, got {type(data)}")
    qprefix = data.qpos[:PICK_PLACE_STATE_QPOS_PREFIX_LEN].copy().astype(np.float32)
    if policy_config is not None:
        state_t = pick_place_observation_state_tensor(qprefix, policy_config)
    else:
        state_t = torch.from_numpy(qprefix).unsqueeze(0)

    def _append_renderer_guides() -> None:
        if not visual_task_guides or guide_cube_name is None or guide_target_cell is None:
            return
        from vla_to_actions import append_task_guide_geoms_to_scene

        append_task_guide_geoms_to_scene(
            renderer.scene,
            renderer.model,
            data,
            cube_name=guide_cube_name,
            target_cell=int(guide_target_cell),
            enabled=True,
        )

    renderer.update_scene(data, camera="cam_global")
    _append_renderer_guides()
    img_g = np.transpose(renderer.render(), (2, 0, 1)).copy()
    renderer.update_scene(data, camera="cam_oblique")
    _append_renderer_guides()
    img_o = np.transpose(renderer.render(), (2, 0, 1)).copy()
    renderer.update_scene(data, camera="left_wrist")
    _append_renderer_guides()
    img_w = np.transpose(renderer.render(), (2, 0, 1)).copy()
    return {
        "observation.state": state_t,
        "observation.images.cam_global": prep_visual_for_xvla_preprocessor(
            img_g,
            has_image_to_float=has_image_to_float,
            has_imagenet_normalize=has_imagenet_normalize,
            policy_config=policy_config,
        ),
        "observation.images.cam_oblique": prep_visual_for_xvla_preprocessor(
            img_o,
            has_image_to_float=has_image_to_float,
            has_imagenet_normalize=has_imagenet_normalize,
            policy_config=policy_config,
        ),
        "observation.images.left_wrist": prep_visual_for_xvla_preprocessor(
            img_w,
            has_image_to_float=has_image_to_float,
            has_imagenet_normalize=has_imagenet_normalize,
            policy_config=policy_config,
        ),
        "task": task,
    }


def prep_visual_for_xvla_preprocessor(
    chw_uint8: np.ndarray,
    *,
    has_image_to_float: bool,
    has_imagenet_normalize: bool,
    policy_config: Any | None = None,
) -> torch.Tensor:
    """
    MuJoCo CHW uint8 -> ``(1,C,H,W)`` tensor at the point expected after checkpoint image steps.

    If the saved ``policy_preprocessor.json`` omits xvla image steps, apply ``/255`` and ImageNet
    here so the model never sees Byte tensors in ``F.interpolate``.
    """
    ptype = str(getattr(policy_config, "type", "") or "").strip().lower()
    if ptype == "smolvla":
        return torch.from_numpy(chw_uint8).unsqueeze(0).float() / 255.0

    t = torch.from_numpy(chw_uint8).unsqueeze(0)
    if not has_image_to_float:
        t = t.float() / 255.0
    if not has_imagenet_normalize:
        if t.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            t = t.float() / 255.0
        t = imagenet_normalize_bchw(t)
    return t


def load_lerobot_dataset_stats(dataset_root: Path | str) -> dict[str, dict[str, torch.Tensor]]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"LeRobot dataset root is not a directory: {root}")
    ds = LeRobotDataset(repo_id=f"local/{root.name}", root=root)
    return ds.meta.stats


def _tensor_last_dim(val: Any) -> int | None:
    if val is None:
        return None
    if isinstance(val, torch.Tensor):
        return int(val.shape[-1])
    if isinstance(val, np.ndarray):
        return int(val.shape[-1])
    return None


def _dataset_action_stat_dim(stats: dict[str, Any]) -> int | None:
    from lerobot.utils.constants import ACTION

    block = stats.get(ACTION)
    if not isinstance(block, dict):
        return None
    for key in ("mean", "min", "max", "q01", "q99", "q10", "q90"):
        d = _tensor_last_dim(block.get(key))
        if d is not None:
            return d
    return None


def _policy_output_action_dim(config: Any) -> int | None:
    from lerobot.utils.constants import ACTION

    feat = config.output_features.get(ACTION)
    if feat is None or not getattr(feat, "shape", None):
        return None
    return int(feat.shape[0])


def _postprocessor_dataset_stats(
    dataset_stats: dict[str, Any] | None,
    config: Any,
) -> dict[str, Any] | None:
    if dataset_stats is None:
        return None
    from lerobot.utils.constants import ACTION

    p_dim = _policy_output_action_dim(config)
    ds_dim = _dataset_action_stat_dim(dataset_stats)
    if p_dim is None or ds_dim is None or p_dim == ds_dim:
        return dataset_stats
    logging.warning(
        "Dataset action stats dim=%s != policy output action dim=%s; "
        "omitting key %r from postprocessor stats (unnormalized policy actions).",
        ds_dim,
        p_dim,
        ACTION,
    )
    return {k: v for k, v in dataset_stats.items() if k != ACTION}


def load_xvla_pre_post_processors(
    policy_cfg: Any,
    *,
    pretrained_path: str,
    dataset_stats: dict[str, Any] | None,
    device: torch.device | str,
    rename_map: dict[str, str] | None = None,
    tokenizer_max_length_override: int | None = None,
) -> tuple[Any, Any]:
    """
    Load checkpoint preprocessor/postprocessor exactly like ``sim_eval.load_policy_and_processors``:
    ``make_pre_post_processors`` + device/stats/rename/unnormalizer overrides.
    """
    from lerobot.policies.factory import make_pre_post_processors

    pretrained_path = hub_or_posix_path(pretrained_path)
    device_t = torch.device(device) if isinstance(device, str) else device

    rmap = default_rename_map_for_policy(policy_cfg) if rename_map is None else rename_map
    features = {**policy_cfg.input_features, **policy_cfg.output_features}
    post_stats = _postprocessor_dataset_stats(dataset_stats, policy_cfg)

    preprocessor_overrides: dict[str, Any] = {
        "device_processor": {"device": device_t.type},
        "normalizer_processor": {
            "stats": dataset_stats,
            "features": features,
            "norm_map": policy_cfg.normalization_mapping,
        },
        "rename_observations_processor": {"rename_map": rmap},
    }
    if tokenizer_max_length_override is not None:
        preprocessor_overrides["tokenizer_processor"] = {
            "max_length": max(8, int(tokenizer_max_length_override)),
        }

    postprocessor_overrides = {
        "unnormalizer_processor": {
            "stats": post_stats,
            "features": policy_cfg.output_features,
            "norm_map": policy_cfg.normalization_mapping,
        },
    }

    return make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=pretrained_path,
        dataset_stats=dataset_stats,
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
    )


def load_policy_and_processors(
    policy_ref: str,
    dataset_root: Path | str,
    device: torch.device | str,
    *,
    rename_map: dict[str, str] | None = None,
    tokenizer_max_length_override: int | None = None,
) -> tuple[Any, Any, Any, float]:
    """
    Same as ``sim_eval.load_policy_and_processors``: policy from checkpoint,
    stats from ``LeRobotDataset``, processors from checkpoint + sim_eval-style overrides.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    policy_ref = resolve_policy_pretrained_path(hub_or_posix_path(policy_ref))
    policy_type = _config_type_from_ref(policy_ref)
    if policy_type is None:
        policy_type = "xvla"
    PolicyCls = _policy_class_for_type(policy_type)
    policy = PolicyCls.from_pretrained(policy_ref)
    policy.eval()
    device_t = torch.device(device) if isinstance(device, str) else device
    policy.to(device_t)

    root = Path(dataset_root).resolve()
    ds = LeRobotDataset(repo_id=f"local/{root.name}", root=root)
    stats = ds.meta.stats

    pre, post = load_xvla_pre_post_processors(
        policy.config,
        pretrained_path=policy_ref,
        dataset_stats=stats,
        device=device_t,
        rename_map=rename_map,
        tokenizer_max_length_override=tokenizer_max_length_override,
    )
    return policy, pre, post, float(ds.meta.fps)


def build_xvla_raw_batch(
    *,
    env_camera_rgb: dict[str, np.ndarray],
    task_prompt: str,
    joint_positions_for_state: np.ndarray,
    policy_config: Any,
    has_image_to_float: bool = True,
    has_imagenet_normalize: bool = True,
) -> dict[str, Any]:
    """
    Pre-preprocessor batch: **sim_eval-style keys** ``observation.images.cam_global`` / ``cam_oblique`` /
    ``left_wrist`` so checkpoint ``RenameObservationsProcessorStep`` matches training.

    ``env_camera_rgb`` values are HWC uint8 (MuJoCo ``Renderer.render()``).
    """
    missing = [c for c in REQUIRED_ENV_CAMERAS if c not in env_camera_rgb]
    if missing:
        raise ValueError(
            f"Missing required camera observations {missing}. "
            f"Have keys {sorted(env_camera_rgb.keys())}."
        )

    batch: dict[str, Any] = {}
    for cam in REQUIRED_ENV_CAMERAS:
        hwc = np.ascontiguousarray(env_camera_rgb[cam])
        if hwc.dtype != np.uint8:
            hwc = hwc.astype(np.uint8, copy=False)
        chw = np.transpose(hwc, (2, 0, 1)).copy()
        obs_key = TRAINING_IMAGE_KEYS_BY_ENV_CAMERA[cam]
        batch[obs_key] = prep_visual_for_xvla_preprocessor(
            chw,
            has_image_to_float=has_image_to_float,
            has_imagenet_normalize=has_imagenet_normalize,
            policy_config=policy_config,
        )

    batch["observation.state"] = pick_place_observation_state_tensor(
        joint_positions_for_state, policy_config
    )
    batch["task"] = task_prompt
    return batch


def action_tensor_to_numpy_float32(action: torch.Tensor) -> np.ndarray:
    if action.dtype in (torch.bfloat16, torch.float16):
        action = action.float()
    return action.detach().cpu().float().numpy()


def infer_xvla_action(
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    raw_batch: dict[str, Any],
    *,
    debug: bool = False,
    _debug_state: list[bool] | None = None,
    use_cuda_bf16_autocast: bool = True,
) -> Any:
    if _debug_state is None:
        _debug_state = [False]
    device = next(policy.parameters()).device
    with torch.inference_mode():
        if use_cuda_bf16_autocast and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                processed = preprocessor(raw_batch)
                action = policy.select_action(processed)
                action = postprocessor(action)
        else:
            processed = preprocessor(raw_batch)
            action = policy.select_action(processed)
            action = postprocessor(action)
    if debug and not _debug_state[0]:
        param_dev = device
        if isinstance(action, torch.Tensor):
            logging.info(
                "[debug_xvla] policy params device=%s | action tensor: dtype=%s shape=%s device=%s",
                param_dev,
                action.dtype,
                tuple(action.shape),
                action.device,
            )
        else:
            logging.info(
                "[debug_xvla] policy params device=%s | action output: type=%s",
                param_dev,
                type(action).__name__,
            )
        _debug_state[0] = True
    return action


def resolve_task_prompt(*, cli_task: str | None, env_instruction: str | None) -> str:
    if cli_task and cli_task.strip():
        return cli_task.strip()
    if env_instruction and env_instruction.strip():
        return env_instruction.strip()
    raise ValueError(
        "XVLA requires task text. Pass --task \"...\" or provide env_instruction from the environment."
    )


def processor_pretrained_ref_for_vis(*, policy_path: str | None, policy_repo_id: str | None) -> str:
    """
    Directory or Hub id holding ``policy_preprocessor.json``. Local step folders are resolved
    with ``resolve_policy_pretrained_path`` like ``sim_eval.resolve_local_policy_dir``.
    """
    for ref in (policy_path, policy_repo_id):
        if not ref:
            continue
        return resolve_policy_pretrained_path(ref)
    raise ValueError("processor_pretrained_ref_for_vis requires policy_path or policy_repo_id")
