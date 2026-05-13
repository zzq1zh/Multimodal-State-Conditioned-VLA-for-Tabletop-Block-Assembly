import cv2
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

# Valid colors consistent with the simulation environment
VALID_COLORS: Set[str] = {"yellow", "green", "purple", "orange"}

def parse_class_id_to_color(spec: Optional[str]) -> Optional[Dict[int, str]]:
    if not spec or not str(spec).strip():
        return None
    out: Dict[int, str] = {}
    for part in str(spec).split(","):
        part = part.strip()
        if ":" not in part:
            continue
        a, b = part.split(":", 1)
        try:
            k = int(a.strip())
        except ValueError:
            continue
        v = b.strip().lower()
        if v in VALID_COLORS:
            out[k] = v
    return out or None

def load_yolo_model(weights: Path | str, device: Optional[str] = None) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError("ultralytics is required for real robot vision.") from e
    return YOLO(str(weights))

def _yolo_collect_detections(
    model: Any,
    img_bgr: np.ndarray,
    conf: float,
    imgsz: Optional[int],
    device: Optional[str],
) -> List[Dict[str, Any]]:
    kw: Dict[str, Any] = {"conf": float(conf), "verbose": False}
    if imgsz is not None:
        kw["imgsz"] = int(imgsz)
    if device is not None:
        kw["device"] = device
    
    res = model.predict(source=img_bgr, **kw)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return []
        
    xyxy = res.boxes.xyxy.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)
    cf = res.boxes.conf.cpu().numpy()
    names = getattr(model, "names", None) or {}
    
    out: List[Dict[str, Any]] = []
    for i in range(len(xyxy)):
        ci = int(cls[i])
        if isinstance(names, dict):
            raw = names.get(ci, "")
        elif isinstance(names, (list, tuple)) and 0 <= ci < len(names):
            raw = names[ci]
        else:
            raw = ""
        nm = str(raw).lower()
        out.append({
            "xyxy": xyxy[i].astype(np.float32),
            "cls": ci,
            "name": nm,
            "conf": float(cf[i]),
        })
    return out

def yolo_color_from_detection(
    det: Dict[str, Any],
    class_id_to_color: Optional[Dict[int, str]] = None,
) -> Optional[str]:
    if class_id_to_color:
        c = class_id_to_color.get(int(det["cls"]))
        if c and str(c).lower() in VALID_COLORS:
            return str(c).lower()
    nm = str(det.get("name", "")).lower()
    for c in VALID_COLORS:
        if c in nm:
            return c
    return None

def yolo_best_cube_center_xy(
    model: Any,
    bgr: np.ndarray,
    want_color: str,
    yolo_conf: float,
    yolo_imgsz: Optional[int],
    yolo_device: Optional[str],
    class_id_to_color: Optional[Dict[int, str]] = None,
) -> Optional[Tuple[float, float, float]]:
    dets = _yolo_collect_detections(
        model, bgr, conf=yolo_conf, imgsz=yolo_imgsz, device=yolo_device
    )
    best: Optional[Tuple[float, float, float]] = None
    best_c = -1.0
    wc = want_color.lower()
    for d in dets:
        c = yolo_color_from_detection(d, class_id_to_color=class_id_to_color)
        if c != wc:
            continue
        cf = float(d.get("conf", 0.0))
        if cf <= best_c:
            continue
        x1, y1, x2, y2 = [float(t) for t in d["xyxy"]]
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        best = (cx, cy, cf)
        best_c = cf
    return best

def collect_front_cube_candidates(
    model: Any,
    bgr: np.ndarray,
    yolo_conf: float,
    yolo_imgsz: Optional[int],
    yolo_device: Optional[str],
    class_id_to_color: Optional[Dict[int, str]] = None,
) -> List[Dict[str, Any]]:
    dets = _yolo_collect_detections(
        model, bgr, conf=yolo_conf, imgsz=yolo_imgsz, device=yolo_device
    )
    out: List[Dict[str, Any]] = []
    for d in dets:
        color = yolo_color_from_detection(d, class_id_to_color=class_id_to_color)
        if color is None:
            continue
        x1, y1, x2, y2 = [float(t) for t in d["xyxy"]]
        out.append({
            "color": str(color),
            "conf": float(d.get("conf", 0.0)),
            "xyxy": [x1, y1, x2, y2],
            "center_uv_front": [0.5 * (x1 + x2), 0.5 * (y1 + y2)],
        })
    out.sort(key=lambda x: float(x["conf"]), reverse=True)
    return out

# --- Calibration Utilities ---

def cell_id_nearest_center(xy: Tuple[float, float], centers9: np.ndarray) -> int:
    p = np.array(xy, dtype=np.float64).reshape(1, 2)
    d2 = np.sum((centers9.astype(np.float64) - p) ** 2, axis=1)
    return int(np.argmin(d2))

def format_yolo_round_observation(lines_front: List[str], lines_side: List[str]) -> str:
    parts = [
        "=== YOLO board observation (cells from user calibration, row-major 0..8) ===",
        "Each line: <color> cube center falls in grid cell <0-8> (front or side view)."
    ]
    if lines_front:
        parts.append("Front camera:")
        parts.extend("  - " + x for x in lines_front)
    else:
        parts.append("Front camera: (no qualifying detections)")
    
    if lines_side:
        parts.append("Side camera:")
        parts.extend("  - " + x for x in lines_side)
    else:
        parts.append("Side camera: (no qualifying detections)")
        
    parts.append(
        "Use these placements together with the images when planning; "
        "cell indices must match the calibrated grid."
    )
    return "\n".join(parts)

def yolo_detections_to_cell_lines(
    dets: List[Dict[str, Any]],
    centers9: np.ndarray,
    view_name: str,
    class_id_to_color: Optional[Dict[int, str]],
    min_conf: float,
) -> List[str]:
    lines: List[str] = []
    for d in dets:
        if float(d.get("conf", 0.0)) < float(min_conf):
            continue
        color = yolo_color_from_detection(d, class_id_to_color=class_id_to_color)
        if color is None:
            continue
        x1, y1, x2, y2 = [float(t) for t in d["xyxy"]]
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        cid = cell_id_nearest_center((cx, cy), centers9)
        lines.append(
            f"{color} @ cell {cid} (conf {float(d['conf']):.2f}; {view_name} bbox center)"
        )
    return lines

def run_yolo_board_observation_for_round(
    weights: Path | str,
    front_bgr: np.ndarray,
    side_bgr: np.ndarray,
    centers_front: np.ndarray,
    centers_side: np.ndarray,
    yolo_conf: float,
    yolo_imgsz: Optional[int],
    yolo_device: Optional[str],
    class_id_to_color: Optional[Dict[int, str]] = None,
) -> Tuple[str, List[str], List[str]]:
    ws = str(weights).strip()
    if not ws.startswith(("http://", "https://")):
        pt = Path(ws).expanduser().resolve()
        if not pt.is_file():
            raise FileNotFoundError(f"YOLO weights not found: {pt}")
        ws = str(pt)
        
    model = load_yolo_model(ws, device=yolo_device)
    df = _yolo_collect_detections(model, front_bgr, conf=yolo_conf, imgsz=yolo_imgsz, device=yolo_device)
    ds = _yolo_collect_detections(model, side_bgr, conf=yolo_conf, imgsz=yolo_imgsz, device=yolo_device)
    
    lf = yolo_detections_to_cell_lines(
        df, centers_front, view_name="front", class_id_to_color=class_id_to_color, min_conf=yolo_conf
    )
    ls = yolo_detections_to_cell_lines(
        ds, centers_side, view_name="side", class_id_to_color=class_id_to_color, min_conf=yolo_conf
    )
    text = format_yolo_round_observation(lf, ls)
    return text, lf, ls

def homography_front_to_side(view_mapping: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    if not isinstance(view_mapping, dict):
        return None
    cand = view_mapping.get("H_front_to_side")
    if isinstance(cand, np.ndarray) and cand.shape == (3, 3):
        return cand.astype(np.float64)
    raw = view_mapping.get("raw")
    if isinstance(raw, dict):
        cand = raw.get("homography_front_to_side")
        if isinstance(cand, list):
            arr = np.asarray(cand, dtype=np.float64)
            if arr.shape == (3, 3):
                return arr
    cand2 = view_mapping.get("homography_front_to_side")
    if isinstance(cand2, list):
        arr2 = np.asarray(cand2, dtype=np.float64)
        if arr2.shape == (3, 3):
            return arr2
    return None

def map_front_uv_to_side_uv(
    uv_front_cur: Tuple[float, float],
    H_front_to_side_ref: np.ndarray,
    ref_w_f: int,
    ref_h_f: int,
    ref_w_s: int,
    ref_h_s: int,
    cur_w_f: int,
    cur_h_f: int,
    cur_w_s: int,
    cur_h_s: int,
) -> Tuple[float, float]:
    u_ref = float(uv_front_cur[0]) * float(max(ref_w_f, 1)) / float(max(cur_w_f, 1))
    v_ref = float(uv_front_cur[1]) * float(max(ref_h_f, 1)) / float(max(cur_h_f, 1))
    p = np.array([u_ref, v_ref, 1.0], dtype=np.float64).reshape(3, 1)
    q = H_front_to_side_ref @ p
    z = float(q[2, 0]) if abs(float(q[2, 0])) > 1e-9 else 1e-9
    u_s_ref = float(q[0, 0] / z)
    v_s_ref = float(q[1, 0] / z)
    u_s = u_s_ref * float(max(cur_w_s, 1)) / float(max(ref_w_s, 1))
    v_s = v_s_ref * float(max(cur_h_s, 1)) / float(max(ref_h_s, 1))
    return u_s, v_s

def scale_centers_xy(
    centers: Any,
    ref_w: int,
    ref_h: int,
    cur_w: int,
    cur_h: int,
) -> np.ndarray:
    pts = np.asarray(centers, dtype=np.float64).reshape(9, 2)
    sx = float(cur_w) / float(max(ref_w, 1))
    sy = float(cur_h) / float(max(ref_h, 1))
    out = pts.copy()
    out[:, 0] *= sx
    out[:, 1] *= sy
    return out.astype(np.float32)

# --- SAM2 Overlay Utilities ---

def make_sam2_image_predictor(model_id: str, device: str) -> Any:
    try:
        from huggingface_hub import hf_hub_download
        _SAM2_VARIANTS = {
            "facebook/sam2-hiera-tiny": ("sam2_hiera_t.yaml", "sam2_hiera_tiny.pt"),
            "facebook/sam2-hiera-small": ("sam2_hiera_s.yaml", "sam2_hiera_small.pt"),
            "facebook/sam2-hiera-base-plus": ("sam2_hiera_b+.yaml", "sam2_hiera_base_plus.pt"),
            "facebook/sam2-hiera-large": ("sam2_hiera_l.yaml", "sam2_hiera_large.pt"),
        }
        config_file, ckpt_file = _SAM2_VARIANTS.get(model_id, ("sam2_hiera_s.yaml", "sam2_hiera_small.pt"))
        config_path = hf_hub_download(repo_id=model_id, filename=config_file)
        ckpt_path = hf_hub_download(repo_id=model_id, filename=ckpt_file)

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam = build_sam2(config_path, ckpt_path, device=device)
        return SAM2ImagePredictor(sam)
    except ImportError as e:
        raise RuntimeError("sam2 is required for visual grounding overlay.") from e

def sam2_mask_from_point(
    predictor: Any,
    rgb_hwc: np.ndarray,
    u: float,
    v: float,
) -> Tuple[np.ndarray, np.ndarray]:
    predictor.set_image(rgb_hwc)
    pts = np.array([[float(u), float(v)]], dtype=np.float32)
    labs = np.array([1], dtype=np.int32)
    masks, ious, _ = predictor.predict(
        point_coords=pts,
        point_labels=labs,
        multimask_output=True,
        return_logits=False,
        normalize_coords=True,
    )
    ci = int(np.argmax(ious))
    m = (masks[ci] > 0.0).astype(np.uint8)
    return m, ious.astype(np.float32)

def build_attached_beam_alpha_from_sam2_point(
    bgr: np.ndarray,
    predictor: Any,
    point_uv: Tuple[float, float],
    mask_alpha: float,
    beam_alpha: float,
    beam_height_ratio: float,
    beam_sigma_x: float,
    beam_sigma_y: float,
    mask_erode_px: int,
    no_gradient: bool,
) -> np.ndarray:
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    u, v = float(point_uv[0]), float(point_uv[1])
    m, _ = sam2_mask_from_point(predictor, rgb, u, v)
    if m.shape[:2] != (h, w):
        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    mu8 = (m > 0).astype(np.uint8)
    
    if int(mask_erode_px) > 0:
        k = int(mask_erode_px) * 2 + 1
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mu8 = cv2.erode(mu8, ker, iterations=1)
    mask = mu8.astype(np.float32)
    
    if float(mask.sum()) < 1.0:
        return np.zeros((h, w), dtype=np.float32)

    col_profile = mask.max(axis=0).astype(np.float32)
    col_profile = cv2.GaussianBlur(
        col_profile[np.newaxis, :], (0, 0), sigmaX=max(float(beam_sigma_x), 1e-3)
    )[0]
    top_y = int(np.min(np.where(mask > 0.5)[0]))
    beam_h = max(1, int(float(h) * float(np.clip(beam_height_ratio, 0.05, 1.0))))
    y0 = max(0, top_y - beam_h)
    ys = np.arange(h, dtype=np.float32)
    t = (float(top_y) - ys) / float(max(top_y - y0, 1))
    y_fall = np.clip(t, 0.0, 1.0)
    beam = np.outer(y_fall, col_profile).astype(np.float32)
    beam = cv2.GaussianBlur(
        beam,
        (0, 0),
        sigmaX=max(float(beam_sigma_x), 1e-3),
        sigmaY=max(float(beam_sigma_y), 1e-3),
    )
    beam /= float(max(beam.max(), 1e-6))

    if bool(no_gradient):
        flat = ((mask > 0.5) | (beam > 1e-3)).astype(np.float32)
        alpha_map = flat * float(max(mask_alpha, beam_alpha))
    else:
        alpha_map = (
            np.clip(mask, 0.0, 1.0) * float(np.clip(mask_alpha, 0.0, 1.0))
            + np.clip(beam, 0.0, 1.0) * float(np.clip(beam_alpha, 0.0, 1.0))
        )
    return np.clip(alpha_map, 0.0, 1.0).astype(np.float32)

def apply_color_alpha_map_bgr(
    bgr: np.ndarray,
    color_bgr: Tuple[int, int, int],
    alpha_map: np.ndarray,
) -> np.ndarray:
    if alpha_map.shape[:2] != bgr.shape[:2]:
        alpha_map = cv2.resize(
            alpha_map, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR
        )
    a = np.clip(alpha_map.astype(np.float32), 0.0, 1.0)[..., None]
    out = bgr.astype(np.float32)
    col = np.array(color_bgr, dtype=np.float32).reshape(1, 1, 3)
    out = out * (1.0 - a) + col * a
    return np.clip(out, 0, 255).astype(np.uint8)

class BeamOverlayState:
    """Caches SAM2 alpha maps to avoid running SAM2 continuously during real-time inference."""
    def __init__(self, calib: Dict[str, Any], predictor: Any, yolo_model: Any, yolo_conf: float):
        self.calib = calib
        self.predictor = predictor
        self.yolo_model = yolo_model
        self.yolo_conf = yolo_conf
        
        self.ref_w_f: int = -1
        self.ref_h_f: int = -1
        self.ref_w_s: int = -1
        self.ref_h_s: int = -1
        
        # Cache for alpha maps
        self.alpha_f_t: Optional[np.ndarray] = None
        self.alpha_s_t: Optional[np.ndarray] = None
        self.alpha_f_c: Optional[np.ndarray] = None
        self.alpha_s_c: Optional[np.ndarray] = None

    def capture_reference_shapes(self, shape_f: Tuple[int, int], shape_s: Tuple[int, int]) -> None:
        self.ref_h_f, self.ref_w_f = shape_f
        self.ref_h_s, self.ref_w_s = shape_s

    def compute_statics(self, want_color: str, target_cell: int, bgr_f: np.ndarray, bgr_s: np.ndarray) -> None:
        cf0 = np.asarray(self.calib["centers_front"], dtype=np.float64).reshape(9, 2)
        cs0 = np.asarray(self.calib["centers_side"], dtype=np.float64).reshape(9, 2)
        hf, wf = bgr_f.shape[:2]
        hs, ws = bgr_s.shape[:2]
        c_f = scale_centers_xy(cf0, self.ref_w_f, self.ref_h_f, wf, hf)
        c_s = scale_centers_xy(cs0, self.ref_w_s, self.ref_h_s, ws, hs)
        
        Hfs = homography_front_to_side(self.calib.get("view_mapping"))
        
        all_front = collect_front_cube_candidates(
            self.yolo_model, bgr_f, yolo_conf=self.yolo_conf, yolo_imgsz=640, yolo_device=None
        )
        # Exclude cubes already on the 3x3 grid (placed by previous instructions)
        grid_margin_px = 35
        all_front = [
            d for d in all_front
            if not any(
                np.hypot(float(d["center_uv_front"][0]) - c_f[k, 0],
                         float(d["center_uv_front"][1]) - c_f[k, 1]) < grid_margin_px
                for k in range(9)
            )
        ]
        for d in all_front:
            uf, vf = float(d["center_uv_front"][0]), float(d["center_uv_front"][1])
            if Hfs is not None:
                us, vs = map_front_uv_to_side_uv(
                    (uf, vf), Hfs, self.ref_w_f, self.ref_h_f, self.ref_w_s, self.ref_h_s,
                    wf, hf, ws, hs
                )
                d["center_uv_side_mapped"] = [us, vs]
            else:
                d["center_uv_side_mapped"] = None

        want_list = [d for d in all_front if str(d.get("color")) == want_color]
        chosen = want_list[0] if want_list else (all_front[0] if all_front else None)
        
        cube_front = (float(chosen["center_uv_front"][0]), float(chosen["center_uv_front"][1])) if chosen else None
        
        if chosen and chosen.get("center_uv_side_mapped"):
            cube_side = (float(chosen["center_uv_side_mapped"][0]), float(chosen["center_uv_side_mapped"][1]))
        else:
            cs3 = yolo_best_cube_center_xy(
                self.yolo_model, bgr_s, want_color=want_color, yolo_conf=self.yolo_conf,
                yolo_imgsz=640, yolo_device=None
            )
            cube_side = (cs3[0], cs3[1]) if cs3 else None

        target_front = (float(c_f[target_cell, 0]), float(c_f[target_cell, 1]))
        if Hfs is not None:
            target_side = map_front_uv_to_side_uv(
                target_front, Hfs, self.ref_w_f, self.ref_h_f, self.ref_w_s, self.ref_h_s,
                wf, hf, ws, hs
            )
        else:
            target_side = (float(c_s[target_cell, 0]), float(c_s[target_cell, 1]))

        # SAM2 Alphas for target cell
        self.alpha_f_t = build_attached_beam_alpha_from_sam2_point(
            bgr_f, self.predictor, target_front, 0.40, 0.40, 0.55, 11.0, 21.0, 2, False
        )
        self.alpha_s_t = build_attached_beam_alpha_from_sam2_point(
            bgr_s, self.predictor, target_side, 0.40, 0.40, 0.55, 11.0, 21.0, 2, False
        )

        # SAM2 Alphas for cube
        if cube_front:
            self.alpha_f_c = build_attached_beam_alpha_from_sam2_point(
                bgr_f, self.predictor, cube_front, 0.45, 0.45, 0.55, 11.0, 21.0, 2, False
            )
        else:
            self.alpha_f_c = None

        if cube_side:
            self.alpha_s_c = build_attached_beam_alpha_from_sam2_point(
                bgr_s, self.predictor, cube_side, 0.45, 0.45, 0.55, 11.0, 21.0, 2, False
            )
        else:
            self.alpha_s_c = None

    def overlay(self, bgr_f: np.ndarray, bgr_s: np.ndarray, cube_bgr: Tuple[int,int,int] = (0,255,0), target_bgr: Tuple[int,int,int] = (255,0,255)) -> Tuple[np.ndarray, np.ndarray]:
        of, osb = bgr_f.copy(), bgr_s.copy()
        
        # Apply target alpha first
        if self.alpha_f_t is not None:
            of = apply_color_alpha_map_bgr(of, target_bgr, self.alpha_f_t)
        if self.alpha_s_t is not None:
            osb = apply_color_alpha_map_bgr(osb, target_bgr, self.alpha_s_t)
            
        # Apply cube alpha second
        if self.alpha_f_c is not None:
            of = apply_color_alpha_map_bgr(of, cube_bgr, self.alpha_f_c)
        if self.alpha_s_c is not None:
            osb = apply_color_alpha_map_bgr(osb, cube_bgr, self.alpha_s_c)
            
        return of, osb

import json

def load_calibration_json(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    cf = np.array(obj["centers_front"], dtype=np.float32).reshape(9, 2)
    cs = np.array(obj["centers_side"], dtype=np.float32).reshape(9, 2)
    pf = obj.get("cube_pairs_front")
    ps = obj.get("cube_pairs_side")
    cpf = (
        np.array(pf, dtype=np.float32).reshape(-1, 2)
        if isinstance(pf, list) and len(pf) > 0
        else np.zeros((0, 2), dtype=np.float32)
    )
    cps = (
        np.array(ps, dtype=np.float32).reshape(-1, 2)
        if isinstance(ps, list) and len(ps) > 0
        else np.zeros((0, 2), dtype=np.float32)
    )
    vm = obj.get("view_mapping") if isinstance(obj.get("view_mapping"), dict) else {}
    h_fs = vm.get("homography_front_to_side")
    h_sf = vm.get("homography_side_to_front")
    Hfs = (
        np.array(h_fs, dtype=np.float64).reshape(3, 3)
        if isinstance(h_fs, list) and len(h_fs) == 3
        else None
    )
    Hsf = (
        np.array(h_sf, dtype=np.float64).reshape(3, 3)
        if isinstance(h_sf, list) and len(h_sf) == 3
        else None
    )
    return {
        "centers_front": cf,
        "centers_side": cs,
        "cube_pairs_front": cpf,
        "cube_pairs_side": cps,
        "view_mapping": {"H_front_to_side": Hfs, "H_side_to_front": Hsf, "raw": vm},
        "raw": obj,
    }
