# Multimodal-State-Conditioned-VLA-for-Tabletop-Block-Assembly

This repository provides **MuJoCo simulation and LeRobot SO-arm code** to run and evaluate **multimodal state-conditioned VLAs** for tabletop block assembly, **sub-instruction–conditioned** closed-loop pick-and-place with optional **VLM** decomposition and replanning.


## Dependencies and install

- **Python**: `pyproject.toml` requires `>=3.10,<3.13` (e.g. Conda `python=3.12`).
- **Core** (editable install from this repo): `mujoco`, `gymnasium`, `dm-control`, `mink`, `numpy`, `torch`, `lerobot[transformers-dep]`, `imageio`, `imageio-ffmpeg`, `python-dotenv`, …
- **Optional**: **OpenCV** (`cv2`) for optional text overlays on sim videos; **`ffmpeg`** on PATH for `tmp_merge_runs_videos.py`.
- **Tests**: `pip install -e ".[test]"` adds `pytest`.

Use **Miniconda / Anaconda** (or Mambaforge). Install PyTorch in the env first if you prefer the conda-forge CUDA stack; otherwise the commands below rely on `pip` inside the env to satisfy `pyproject.toml`.

```bash
cd Multimodal-State-Conditioned-VLA-for-Tabletop-Block-Assembly   # or your clone directory name

# New environment (pick a Python version in [3.10, 3.12]; env name can be anything short)
conda create -n msc-vla-tabletop python=3.12 -y
conda activate msc-vla-tabletop

# Optional: CUDA PyTorch from conda-forge/nvidia channels, then skip torch on pip if already satisfied
# conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia -y   # adjust CUDA version to your driver

pip install -e .
```

## Configuration

Copy **`.env.example`** to **`.env`** and fill in `OPENAI_API_KEY`, etc. Sim scripts also accept **`--dotenv`**. Use **`HF_TOKEN`** if you pull policies from the Hub.

## Simulation entrypoints

### `sim_scenes.py`

3×3 puzzle scene: pick color, cube index, target cell, optional viewer. See `--help`.

```bash
python sim_scenes.py --color orange --cube-idx 2 --cell 2 --seed 42
```

### `sim_eval.py`

- **Default**: random pick-place episodes (`--policy-id` or `--policy-path`, plus `--dataset-root`).
- **Instruction mode** (exactly one): `--instructions-file`, `--instructions-json`, or `--from-vlm-text` (single-shot VLM planning).
- Common flags: `--render`, `--video-out` / `--video-camera` / `--video-fps` / `--video-overlay-text`, `--visual-task-guides`, JSON output flags.

```bash
python sim_eval.py --policy-id OWNER/REPO --dataset-root /path/to/lerobot_dataset --render

python sim_eval.py \
  --instructions-file plan.json \
  --policy-id OWNER/REPO \
  --dataset-root /path/to/lerobot_dataset \
  --seed 0 --render
```

### `sim_vlm_to_actions_eval.py`

Draws prompts from a fixed bank of **30** templates (shuffled in blocks of 30). For each prompt: **render sim snapshots → VLM plan → validate → `sim_eval` instruction sequence**. **Stdout**: NDJSON (one object per prompt), optional final summary line; **`--json-out`** for a pretty-printed file; **`--sim-video-out`** writes one MP4 per prompt (`stem_p000.mp4`, …). Defaults: camera **`cam_oblique`**, video **30 FPS**, user prompt burned into frames (disable with **`--sim-video-no-prompt-overlay`**).

Each run row includes **`prompt`** (user template only) and **`vla_instructions`** (decomposed policy task lines).

Set **`OPENAI_API_KEY`** (see `.env.example`); pass **`--dotenv`** to load a specific env file.

---

## Planning and adapters (libraries)

| File | Role |
|------|------|
| `vlm_to_actions.py` | OpenAI-compatible API: `plan_vla_instructions`, validators, inventory alignment |
| `vla_adapter.py` | Hub/local policy, dataset root, observation batching, MuJoCo actuation |
| `vla_to_actions.py` | Task parsing, viewer guide geoms, small MuJoCo demo |

---

## Real robot and vision

| File | Role |
|------|------|
| `calibrate_dual_view.py` | Dual-view 3×3 grid clicks + cube pairs → calibration JSON |
| `real_vision.py` | YOLO, SAM2 beam overlay, calibration I/O |
| `real_eval.py` | LeRobot SO-Follower loop: `--policy-path`, `--dataset-repo-id`, `--user-text`, `--calib-json`, camera indices, `--vis`, `--record-video`, … |

Complete calibration and LeRobot/camera setup before real runs; see each script’s `--help`.

---

## Acknowledgments

The **MuJoCo simulation** and **scene / Gymnasium environment** portions of this project **reference and borrow** from **[gym-so100-c](https://github.com/ilonajulczuk/gym-so100-c)** (ilonajulczuk)—a Gym + MuJoCo setup for the SO-100/101 arm that informed our assets and env structure. Thanks to the original authors for sharing that work.
