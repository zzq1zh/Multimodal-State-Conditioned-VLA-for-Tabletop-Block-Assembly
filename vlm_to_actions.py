"""
Call a **vision-language model (VLM)** over an **OpenAI-compatible** ``/v1/chat/completions`` HTTP API
(from **text** and/or **images**) to produce **pick-place instruction lists** aligned with this repo's XVLA format.

This module (``vlm_to_actions.py``) exposes **Python APIs only** (e.g. ``plan_vla_instructions``, ``load_env``, validators) with
**no CLI entrypoint**; ``sim_eval.py`` (instruction-sequence flags) calls it in-process.

When images are used with the default inventory path: **first** run a VLM pass to count cubes by color
in the **source bin**, **then** plan under that merged per-color budget.

Environment:
  - ``OPENAI_API_KEY`` (required for API calls)

Optional:
  - ``OPENAI_BASE_URL`` — API base, default ``https://api.openai.com/v1`` (Azure / proxies: set to your deployment base).
  - ``OPENAI_MODEL`` — default model id when a call does not pass ``model=`` (repository default ``gpt-4o-mini``).
  - ``OPENAI_VISION_MODEL`` — overrides for vision/inventory calls (defaults to ``OPENAI_MODEL``).
  - ``OPENAI_PLAN_MODEL`` — overrides for planning calls (defaults to ``OPENAI_MODEL``).

``load_env()`` loads ``.env`` next to this file and in the current working directory (does not override
existing env vars). Callers may pass a ``Path`` to ``load_env(path)`` first to load and override keys.
If ``python-dotenv`` is installed, it is preferred over the minimal built-in parser.

Valid **XVLA task** lines (same checks as ``sim_eval.py`` instruction mode)::
  - **Must** include a color: ``Pick up the <yellow|green|purple|orange> cube and place it in grid cell <0-8>.``
  - Do **not** emit colorless ``Pick up the cube ...`` (rejected by model and local validation).
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def max_picks_per_color_physical() -> dict[str, int]:
    """
    Lazily import ``sim_scenes.py`` (depends on mujoco/mink).

    If the simulation module cannot be imported, return a permissive upper bound so
    import-only contexts do not fail.
    """
    try:
        from sim_scenes import max_picks_per_color_physical as _fn

        return _fn()
    except ImportError:
        return {c: 99 for c in ("yellow", "green", "purple", "orange")}


def _openai_chat_completions_url() -> str:
    base = (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    return f"{base}/chat/completions"


def _default_model() -> str:
    return os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"


# Same ordering as ``sim_scenes.GRID_CENTERS``: index = row*3+col, row/col in {0,1,2}
DEFAULT_GRID_BOARD_DESCRIPTION = """Placement board (3×3 grid on the desk, orthogonal to the table; like tic-tac-toe).
Cell indices use row-major order: index = row*3 + column, row and column each in {{0,1,2}}.
Top-down layout (column increases to the right, row increases downward in this diagram; matches simulation row-major indexing):

  column:   0      1      2
        +------+------+------+
  row 0 |  0   |  1   |  2   |
        +------+------+------+
  row 1 |  3   |  4   |  5   |
        +------+------+------+
  row 2 |  6   |  7   |  8   |
        +------+------+------+

Instructions must say "grid cell <0-8>" using these indices only (center is cell 4)."""

DEFAULT_SYSTEM_PROMPT = """You are a task planner for a simulated single-arm robot pick-and-place desk.

""" + DEFAULT_GRID_BOARD_DESCRIPTION + """

Source bin may contain up to nine cubes in four colors; use these exact spellings: yellow, green, purple, orange.

Each instruction must be EXACTLY one sentence ending with a period (.).
You MUST always name the cube color in every instruction. Use ONLY this template (repeat per step):
Pick up the <yellow|green|purple|orange> cube and place it in grid cell <0-8>.
Do NOT output "Pick up the cube" without a color word — that is invalid for the robot policy.

Hard limits from the MuJoCo asset (cannot exceed total uses of a color in ALL lines combined):
yellow ≤ 2, green ≤ 1, purple ≤ 3, orange ≤ 3.

Return ONLY valid JSON (no markdown fences), shape:
{"pattern":"<short text description>","instructions":["...","..."]}
The list order is the execution order: the robot completes instruction 1 fully before instruction 2, etc.
Use as few steps as needed; typical steps are 1 to 4."""

DEFAULT_PLAN_SYSTEM_PROMPT_WITH_INVENTORY = """You plan pick-and-place instructions for a simulated robot.

""" + DEFAULT_GRID_BOARD_DESCRIPTION + """

The user message lists **per-color budgets** already merged from vision (if any), the simulation's active cube set, and hard XML caps. Those numbers are the maximum total uses of each color across ALL instructions — never exceed them.

Rules:
- Across the whole instruction list, uses of color C must be <= the budget line for C. Budget 0 means do not use that color.
- Do not assume extra cubes beyond the budgets (especially purple ≤ 3, yellow ≤ 2, orange ≤ 3, green ≤ 1 in this asset).
- Each instruction is one sentence ending with a period (.):
  Pick up the <color> cube and place it in grid cell <0-8>.

Return ONLY valid JSON (no markdown fences):
{"pattern":"<short text description>","instructions":["...","..."]}
List order is execution order. If user intent asks for more than inventory allows, output only the steps that fit the per-color budget (omit impossible steps)."""

DEFAULT_SINGLE_SHOT_VISION_SUFFIX = """

Planning constraint: base the plan **only** on cube colors/counts you can see in the **source bin / tray** at the start (not cubes already on the 3x3 placement grid). Do not require a color that is not visibly available there."""

DEFAULT_USER_TEMPLATE = """User instructions (plain text; optional images attached above):

""" + DEFAULT_GRID_BOARD_DESCRIPTION + """

{user_text}

Produce {{"pattern":"...","instructions":[...]}} for the robot.
- pattern: one short sentence describing the intended spatial pattern.
- instructions: every line must include one of yellow/green/purple/orange
  (e.g. "Pick up the green cube and place it in grid cell 3.")."""

DEFAULT_EXTRA_BUDGET_FOR_TEXT_PLAN = """

=== Per-color maximum picks for THIS run (merged simulation + XML caps) ===
{budget_block}
The total uses of each color across ALL instruction lines must not exceed the numbers above (XML absolute caps: yellow≤2, green≤1, purple≤3, orange≤3)."""

DEFAULT_INVENTORY_SYSTEM_PROMPT = """You analyze images of a simulated robot desk at **task initial time**.
Your task is to count only small colored cubes that are **still in the SOURCE BIN / tray** (the pick-up region before any grasp — often a box or cluster on one side of the desk).

""" + DEFAULT_GRID_BOARD_DESCRIPTION + """

The 3×3 area above is the **placement target** on the desk — do NOT count cubes whose centers sit on those nine cell locations toward the source-bin inventory; only cubes clearly left in the source region (not on the placement board).

Color keys in JSON must be exactly: yellow, green, purple, orange.

Return ONLY valid JSON (no markdown fences), shape:
{"inventory":{"yellow":<int>,"green":<int>,"purple":<int>,"orange":<int>}}
Use non-negative integers. Each physical cube is counted once. If a color is absent, use 0.
If the view is unclear, give your best estimate and prefer under-counting over inventing cubes."""

DEFAULT_USER_TEMPLATE_WITH_INVENTORY = """=== Placement board (3×3 grid, cell indices 0–8) ===
""" + DEFAULT_GRID_BOARD_DESCRIPTION + """

=== Per-color budget for THIS run (already merged: min(vision, simulation active bodies, XML caps)) ===
{budget_block}

=== User instructions ===
{user_text}

=== Output ===
Produce {{"pattern":"...","instructions":[...]}}.
- pattern: one short sentence describing the intended spatial pattern.
- instructions: each line:
Pick up the <yellow|green|purple|orange> cube and place it in grid cell <0-8>.
The total times you use a color across ALL instruction lines must be <= the number shown for that color above (never exceed XML caps: yellow≤2, green≤1, purple≤3, orange≤3).
If user intent would exceed any budget, output only feasible steps (omit impossible picks)."""

_VALID_COLORS = frozenset({"yellow", "green", "purple", "orange"})
_GENERIC_TASK_RE = re.compile(
    r"^Pick up the cube and place it in grid cell (\d+)\.$",
    re.IGNORECASE,
)
_COLORED_TASK_RE = re.compile(
    r"^Pick up the (\w+) cube and place it in grid cell (\d+)\.$",
    re.IGNORECASE,
)
_ENV_LOADED = False


def _apply_env_file(path: Path, *, override: bool = False) -> None:
    """Parse a minimal ``.env`` (KEY=VALUE, ``#`` comments); do not override existing keys unless ``override=True``."""
    if not path.is_file():
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            continue
        k, _, v = s.partition("=")
        k = k.strip()
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        if not k:
            continue
        if not override and k in os.environ:
            continue
        os.environ[k] = v


def load_env(dotenv_path: Path | None = None) -> None:
    """
    Load ``.env`` into ``os.environ``.

    - If ``dotenv_path`` is given: load that file first (**override=True**, overwrites existing keys).
    - Then (once only) load ``.env`` next to this script and in the current working directory
      (**override=False**, does not override variables already set in the environment).

    If ``python-dotenv`` is installed, use it; otherwise use the built-in minimal parser
    (``KEY=VALUE``, ``#`` comments).
    """
    global _ENV_LOADED

    if dotenv_path is not None:
        p = dotenv_path.expanduser().resolve()
        try:
            from dotenv import load_dotenv as _ld

            _ld(p, override=True)
        except ImportError:
            _apply_env_file(p, override=True)

    if _ENV_LOADED:
        return

    repo = Path(__file__).resolve().parent
    paths = [repo / ".env", Path.cwd() / ".env"]
    try:
        from dotenv import load_dotenv as _ld

        for path in paths:
            if path.is_file():
                _ld(path, override=False)
    except ImportError:
        for path in paths:
            _apply_env_file(path, override=False)

    _ENV_LOADED = True


def _mime_for_path(path: Path) -> str:
    suf = path.suffix.lower()
    if suf in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suf == ".png":
        return "image/png"
    if suf == ".webp":
        return "image/webp"
    return "application/octet-stream"


def _image_part(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    b64 = base64.standard_b64encode(raw).decode("ascii")
    mime = _mime_for_path(path)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{b64}"},
    }


def validate_instruction_line(line: str) -> tuple[bool, str]:
    s = line.strip()
    if not s.endswith("."):
        return False, "Line must end with a period (.)"
    if _GENERIC_TASK_RE.match(s):
        return (
            False,
            "XVLA tasks must name a color; do not use colorless 'Pick up the cube ...'; "
            "use Pick up the <yellow|green|purple|orange> cube and place it in grid cell <0-8>.",
        )
    mc = _COLORED_TASK_RE.match(s)
    if mc:
        color = mc.group(1).lower()
        cell = int(mc.group(2))
        if color not in _VALID_COLORS:
            return False, f"Color must be one of yellow/green/purple/orange: {color!r}"
        if 0 <= cell <= 8:
            return True, ""
        return False, f"Grid cell must be in 0..8: {cell}"
    return (
        False,
        "Expected form: Pick up the <yellow|green|purple|orange> cube and place it in grid cell <0-8>.",
    )


def validate_instructions(lines: list[str]) -> tuple[list[str], list[str]]:
    """Return ``(valid instruction lines, error message lines)``."""
    ok: list[str] = []
    errs: list[str] = []
    for i, line in enumerate(lines):
        good, msg = validate_instruction_line(line)
        if good:
            ok.append(line.strip())
        else:
            errs.append(f"[{i}] {msg}: {line!r}")
    return ok, errs


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def parse_instructions_json(raw: str) -> list[str]:
    """Parse the ``instructions`` list from model JSON output."""
    t = _strip_json_fence(raw)
    obj = json.loads(t)
    if not isinstance(obj, dict) or "instructions" not in obj:
        raise ValueError("JSON root object must contain key 'instructions'")
    inst = obj["instructions"]
    if not isinstance(inst, list):
        raise ValueError("'instructions' must be an array")
    out = []
    for x in inst:
        if not isinstance(x, str):
            raise ValueError("each entry in 'instructions' must be a string")
        out.append(x.strip())
    return [x for x in out if x]


def parse_inventory_json(raw: str) -> dict[str, int]:
    """Parse VLM ``inventory`` JSON; keep only the four colors as non-negative integers."""
    t = _strip_json_fence(raw)
    obj = json.loads(t)
    if isinstance(obj, dict) and "inventory" in obj:
        inv = obj["inventory"]
    else:
        inv = obj
    if not isinstance(inv, dict):
        raise ValueError("inventory JSON must be an object")
    out: dict[str, int] = {}
    for k, v in inv.items():
        if not isinstance(k, str):
            continue
        key = k.lower().strip()
        if key not in _VALID_COLORS:
            continue
        try:
            n = int(v)
        except (TypeError, ValueError):
            continue
        if n >= 0:
            out[key] = n
    return out


def merge_per_color_budget(
    vlm_inventory: dict[str, int] | None,
    sim_active_color_counts: dict[str, int] | None,
) -> dict[str, int]:
    """
    Per-color pick upper bound = min(VLM source-bin counts, sim active instances for current K,
    XML body count for that color).

    If a source is missing it is omitted from the min (e.g. VLM-only vs physics-only).
    """
    phys = max_picks_per_color_physical()
    out: dict[str, int] = {}
    for c in _VALID_COLORS:
        cap = int(phys.get(c, 0))
        parts: list[int] = [cap]
        if vlm_inventory is not None:
            parts.append(max(0, int(vlm_inventory.get(c, 0))))
        if sim_active_color_counts is not None:
            parts.append(max(0, int(sim_active_color_counts.get(c, 0))))
        out[c] = min(parts)
    return out


def _format_budget_lines(budget: dict[str, int]) -> str:
    lines = []
    for c in sorted(_VALID_COLORS):
        n = int(budget.get(c, 0))
        lines.append(
            f"- {c}: {n}  (total uses of this color across ALL instructions must be ≤ {n})"
        )
    return "\n".join(lines)


def instruction_color_counts(lines: list[str]) -> Counter[str]:
    """Count one use per colored instruction line."""
    c: Counter[str] = Counter()
    for line in lines:
        m = _COLORED_TASK_RE.match(line.strip())
        if m:
            c[m.group(1).lower()] += 1
    return c


def validate_instructions_against_vlm_inventory(
    instructions: list[str], per_color_budget: dict[str, int]
) -> tuple[bool, str]:
    """
    Per-color counts in ``instructions`` must not exceed the merged per-color budget
    (VLM / simulation / XML caps).
    """
    phys = max_picks_per_color_physical()
    need = instruction_color_counts(instructions)
    for color, n in need.items():
        avail = int(per_color_budget.get(color, 0))
        if n > avail:
            pmax = int(phys.get(color, 0))
            return (
                False,
                f"Color {color!r} appears {n} times in instructions, over planning budget {avail} "
                f"(at most {pmax} cubes of this color in the scene; reduce steps for this color or check budget merge).",
            )
    return True, ""


def call_openai_compatible_vlm(
    *,
    system_prompt: str,
    user_content: list[dict[str, Any]] | str,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.2,
    timeout_s: float = 120.0,
) -> str:
    """
    Call an OpenAI-compatible ``POST .../chat/completions`` endpoint (VLM / chat); return assistant text.

    ``user_content`` may be a plain string or a list of OpenAI-style multimodal content parts.
    """
    load_env()
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set environment variable OPENAI_API_KEY")

    m = model or _default_model()

    if isinstance(user_content, str):
        user_payload: str | list[dict[str, Any]] = user_content
    else:
        user_payload = user_content

    body: dict[str, Any] = {
        "model": m,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ],
    }
    data = json.dumps(body).encode("utf-8")
    url = _openai_chat_completions_url()
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {err_body}") from e

    if "choices" not in payload or not payload["choices"]:
        raise RuntimeError(f"Unexpected response: {payload}")
    msg = payload["choices"][0].get("message") or {}
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
        return "".join(parts)
    raise RuntimeError(f"Could not parse assistant content: {payload}")


def _resolve_vision_model(explicit: str | None, fallback: str | None) -> str:
    return (
        explicit
        or fallback
        or os.environ.get("OPENAI_VISION_MODEL")
        or _default_model()
    )


def _resolve_plan_model(explicit: str | None, fallback: str | None) -> str:
    return (
        explicit
        or fallback
        or os.environ.get("OPENAI_PLAN_MODEL")
        or _default_model()
    )


def infer_initial_inventory_from_images(
    image_paths: list[Path],
    *,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.1,
) -> str:
    """
    Call a vision-capable VLM (via OpenAI-compatible chat) to count four-color cubes in the source bin from images only;
    return raw assistant text (JSON).
    """
    if not image_paths:
        raise ValueError("image_paths must not be empty")
    parts: list[dict[str, Any]] = []
    for p in image_paths:
        if not p.is_file():
            raise FileNotFoundError(p)
        parts.append(_image_part(p))
    parts.append(
        {
            "type": "text",
            "text": (
                "These images show the robot desk at task start. "
                "Count only cubes in the source bin / tray (not on the 3x3 grid). "
                "Reply with JSON only as in the system message."
            ),
        }
    )
    return call_openai_compatible_vlm(
        system_prompt=DEFAULT_INVENTORY_SYSTEM_PROMPT,
        user_content=parts,
        model=_resolve_vision_model(model, None),
        api_key=api_key,
        temperature=temperature,
    )


def plan_vla_instructions(
    user_text: str,
    *,
    image_paths: list[Path] | None = None,
    inventory_from_vlm: bool = True,
    sim_active_color_counts: dict[str, int] | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    plan_system_prompt_with_inventory: str | None = None,
    user_template: str = DEFAULT_USER_TEMPLATE,
    model: str | None = None,
    plan_model: str | None = None,
    inventory_model: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.2,
) -> tuple[list[str], str, dict[str, int]]:
    """
    Call an OpenAI-compatible VLM; return ``(instruction list, combined raw model text, per_color_budget)``.

    ``per_color_budget`` is the max picks per color (min(vision inventory, sim active, XML cap)) for validation and JSON.

    - If ``image_paths`` is set and ``inventory_from_vlm`` is True (default): VLM source-bin inventory first,
      then plan under the merged budget.
    - ``sim_active_color_counts``: per-color active instance counts from ``active_cube_triplets(K)``
      (aligned with ``--num-cubes``).

    ``user_text`` may be empty (for image-only runs, state intent clearly, e.g. "Plan one step from the image").
    """
    imgs = list(image_paths) if image_paths else []

    max_retries = 2

    if imgs and inventory_from_vlm:
        raw_inv = infer_initial_inventory_from_images(
            imgs,
            model=_resolve_vision_model(inventory_model, model),
            api_key=api_key,
            temperature=min(0.12, float(temperature)),
        )
        vlm_inv = parse_inventory_json(raw_inv)
        per_color_budget = merge_per_color_budget(vlm_inv, sim_active_color_counts)
        budget_block = _format_budget_lines(per_color_budget)
        text_for_plan = DEFAULT_USER_TEMPLATE_WITH_INVENTORY.format(
            budget_block=budget_block,
            user_text=user_text or "(no extra text; use inventory only)",
        )
        sp_plan = plan_system_prompt_with_inventory or DEFAULT_PLAN_SYSTEM_PROMPT_WITH_INVENTORY
        repair_hint = ""
        raw_plan = ""
        lines: list[str] = []
        for _ in range(max_retries + 1):
            prompt = text_for_plan + repair_hint
            raw_plan = call_openai_compatible_vlm(
                system_prompt=sp_plan,
                user_content=prompt,
                model=_resolve_plan_model(plan_model, model),
                api_key=api_key,
                temperature=temperature,
            )
            lines = parse_instructions_json(raw_plan)
            ok_budget, msg_budget = validate_instructions_against_vlm_inventory(
                lines, per_color_budget
            )
            if ok_budget:
                break
            repair_hint = (
                "\n\n[REPAIR REQUIRED]\n"
                f"{msg_budget}\n"
                "Regenerate a FULL new JSON response that satisfies all per-color budgets. "
                "Keep the same JSON schema with keys pattern and instructions."
            )
        combined = f"[inventory_assistant]\n{raw_inv}\n\n[plan_assistant]\n{raw_plan}"
        return lines, combined, per_color_budget

    per_color_budget = merge_per_color_budget(None, sim_active_color_counts)
    budget_append = DEFAULT_EXTRA_BUDGET_FOR_TEXT_PLAN.format(
        budget_block=_format_budget_lines(per_color_budget)
    )

    text_block = user_template.format(user_text=user_text or "(no extra text; use images only)")
    text_for_multimodal = (
        text_block + budget_append + (DEFAULT_SINGLE_SHOT_VISION_SUFFIX if imgs else "")
    )

    if imgs:
        parts: list[dict[str, Any]] = []
        for p in imgs:
            if not p.is_file():
                raise FileNotFoundError(p)
            parts.append(_image_part(p))
        parts.append({"type": "text", "text": text_for_multimodal})
        user_content: list[dict[str, Any]] | str = parts
    else:
        user_content = text_for_multimodal

    repair_hint = ""
    raw = ""
    lines: list[str] = []
    for _ in range(max_retries + 1):
        if isinstance(user_content, str):
            uc: str | list[dict[str, Any]] = user_content + repair_hint
        else:
            uc = list(user_content)
            if repair_hint:
                uc = uc + [{"type": "text", "text": repair_hint}]
        m_call = (
            _resolve_vision_model(inventory_model or plan_model, model)
            if imgs
            else _resolve_plan_model(plan_model, model)
        )
        raw = call_openai_compatible_vlm(
            system_prompt=system_prompt,
            user_content=uc,
            model=m_call,
            api_key=api_key,
            temperature=temperature,
        )
        lines = parse_instructions_json(raw)
        ok_budget, msg_budget = validate_instructions_against_vlm_inventory(
            lines, per_color_budget
        )
        if ok_budget:
            break
        repair_hint = (
            "\n\n[REPAIR REQUIRED]\n"
            f"{msg_budget}\n"
            "Regenerate a FULL new JSON response that satisfies all per-color budgets. "
            "Keep the same JSON schema with keys pattern and instructions."
        )
    return lines, raw, per_color_budget
