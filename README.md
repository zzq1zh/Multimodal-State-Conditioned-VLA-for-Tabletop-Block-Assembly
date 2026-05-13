# gym-so100-sim

This repository provides **MuJoCo simulation** (Gymnasium single red cube task + 3×3 multicolor `so100_puzzle.xml` + mink IK), plus optional **OpenAI-compatible VLM planning → XVLA policy closed-loop** scripts.

**Not included**: LeRobot **record/merge datasets** or general robot pipelines outside this evaluation loop. `vlm_to_actions.py` is **library-only** (OpenAI-compatible `chat/completions`, `plan_vla_instructions`, validators). VLM planning runs **inside** `sim_eval.py` (for simulation) or `real_eval.py` (for physical deployment).

## Dependencies

- Python **3.10–3.12** (on 3.13, `dm-control`’s `labmaze` often has no prebuilt wheel; 3.12 is recommended).  
- **Simulation base**: `mujoco`, `dm-control`, `gymnasium`, `numpy`, `mink`  
- **XVLA + OpenAI-compatible VLM** (optional): `pip install -e ".[xvla]"` → `torch`, `lerobot[transformers-dep]`, `python-dotenv`, `imageio[ffmpeg]`

## Install

```bash
cd gym-so100-sim
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
# For XVLA / OpenAI-compatible VLM planning runs:
pip install -e ".[xvla]"
```

## Usage

**Gymnasium environment** (`so100_transfer_cube.xml`):

```python
import gymnasium as gym
import gym_so100  # noqa: F401

env = gym.make("gym_so100/SO100TouchCube-v0")
obs, _ = env.reset(seed=0)
```

**3×3 pick-and-place** (run from the repo root so `gym_so100/assets/so100_puzzle.xml` resolves):

```bash
# sim_scenes.py: --color {yellow|green|purple|orange} --cube-idx N --cell 0-8 [--seed] [--no-viewer] [--speed]
python sim_scenes.py --color orange --cube-idx 2 --cell 2 --seed 42
# vla_to_actions.py: fixed-layout demo [--no-viewer] [--speed]
python vla_to_actions.py
```

### OpenAI-compatible VLM → XVLA (simulation)

1. Set `OPENAI_API_KEY`. Optional: `OPENAI_BASE_URL` (default `https://api.openai.com/v1`), `OPENAI_MODEL`, `OPENAI_VISION_MODEL`, `OPENAI_PLAN_MODEL` (see `vlm_to_actions.py`), and/or a project-root `.env`; or pass `--dotenv /path/to/.env` on `sim_eval.py`.  
2. Prepare a **LeRobot v3 dataset root** consistent with training (including `meta/info.json`) for preprocessor stats.  
3. From the repo root (`sim_eval.py` requires **either** `--policy-id` **or** `--policy-path`):

```bash
# Random episodes (default): no instruction flags
python sim_eval.py --policy-id YOUR/HF_REPO --dataset-root /path/to/lerobot_dataset --render

# Instruction sequence: exactly one of --instructions-file | --instructions-json | --from-vlm-text
python sim_eval.py \
  --from-vlm-text "Form an X with cubes on the 3×3 grid." \
  --policy-id YOUR/HF_REPO \
  --dataset-root /path/to/lerobot_dataset \
  --seed 0 --render --visual-task-guides

# Or load instructions from JSON
python sim_eval.py \
  --instructions-file /tmp/plan.json \
  --policy-id YOUR/HF_REPO \
  --dataset-root /path/to/lerobot_dataset \
  --seed 0 --render --visual-task-guides
```

Key files: `sim_scenes.py` (3×3 puzzle / IK CLI), `vla_to_actions.py` (task text, viewer guides, fixed-layout demo), `vla_adapter.py` (LeRobot + pick-place sim control), `sim_eval.py` (XVLA closed-loop eval + instruction / VLM modes), `vlm_to_actions.py` (OpenAI-compatible VLM → instruction list).

### Dual-view calibration (nine-grid)

Before running real-robot evaluation, you need a calibration JSON that maps the 3×3 grid cells in both camera views. Use `calibrate_dual_view.py`:

```bash
python calibrate_dual_view.py \
  --front /path/to/first_cam_front.png \
  --side /path/to/first_cam_side.png \
  --out outputs/real101_nine_grid_calib.json
```

Interactive workflow:
1. Two images are displayed side-by-side (Front left, Side right).
2. Click the 9 grid cell centers on the **Front** view in order 0→8 (row-major: top-left is 0, center is 4). Press **Space** when done.
3. Click the same 9 points on the **Side** view in the same 0→8 order. Press **Space** to finish.
4. Then click corresponding cube centers — first all visible cubes on Front (Space to advance), then the same cubes in the same order on Side (Space to finish). This estimates a homography for cross-view mapping.
5. Press **R** to reset the current phase, **Esc** to abort.

Key bindings during calibration: **Left click** = place point, **Space** = confirm phase, **R** = reset current phase, **Esc** = abort.

Optional flags:
- `--cube-pairs N`: fix exactly N cube pairs (instead of dynamic: click all visible then Space).
- `--recalibrate`: overwrite an existing output file.
- `--skip-dialog`: skip the Tkinter popup instructions.

The output JSON contains `centers_front`, `centers_side` (9 grid points each) and optionally `cube_pairs_front`/`cube_pairs_side` plus a `view_mapping` with the estimated homography.

**Note on beam overlays (real vs sim):** The real-robot beam overlay in `real_vision.py` uses simplified SAM2 parameters (lower alpha, wider Gaussian sigma) compared to the simulation pipeline, trading fine-grained mask detail for faster rendering on live camera feeds. Beam appearance can be customized by adjusting the hardcoded parameters in `real_vision.py` `BeamOverlayState.compute_statics()`.

### OpenAI-compatible VLM → XVLA (real robot)

1. Connect your SO-ARM100 and set `OPENAI_API_KEY` (or `DASHSCOPE_API_KEY` if using Qwen). 
2. Generate a dual-view calibration JSON (see [Dual-view calibration](#dual-view-calibration-nine-grid) above).
3. From the repo root, run the `real_eval.py` entrypoint:

```bash
python real_eval.py \
  --port COM6 \
  --policy-path YOUR/HF_REPO \
  --dataset-repo-id YOUR/DATASET_REPO \
  --calib-json outputs/real101_nine_grid_calib.json \
  --user-text "Observe the cubes on the table. Generate at least 2 pick-and-place instructions." \
  --vis --push-to-hub
```

Key files: `real_eval.py` (real robot evaluation loop), `real_vision.py` (YOLO board observation, SAM2 attached beam overlays, dual-view calibration loading), `calibrate_dual_view.py` (interactive dual-view nine-grid calibration script).

## Tests

```bash
pytest -q
```
