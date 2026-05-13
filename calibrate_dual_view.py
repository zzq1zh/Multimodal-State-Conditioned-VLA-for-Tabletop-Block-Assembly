#!/usr/bin/env python3
import argparse
import sys
import json
from pathlib import Path
from typing import Any, Tuple

import cv2
import numpy as np

def calibrate_nine_cells_dual_view_interactive(
    front_bgr: np.ndarray,
    side_bgr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interactive calibration of the 3x3 grid across dual views.
    - Click 0-8 on the Front view (left), then press Space.
    - Click 0-8 on the Side view (right), then press Space.
    """
    state: dict[str, Any] = {"phase": "front", "front": [], "side": []}
    gap = 16
    h = max(front_bgr.shape[0], side_bgr.shape[0])
    w = front_bgr.shape[1] + gap + side_bgr.shape[1]
    x_side = front_bgr.shape[1] + gap

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if state["phase"] == "front":
            if not (0 <= x < front_bgr.shape[1] and 0 <= y < front_bgr.shape[0]):
                return
            if len(state["front"]) < 9:
                state["front"].append((float(x), float(y)))
        else:
            if not (x_side <= x < x_side + side_bgr.shape[1] and 0 <= y < side_bgr.shape[0]):
                return
            if len(state["side"]) < 9:
                state["side"].append((float(x - x_side), float(y)))

    wname = "CALIB NINE GRID FRONT|SIDE"
    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(wname, on_mouse)
    try:
        while True:
            vis = np.zeros((h + 56, w, 3), dtype=np.uint8)
            vis[: front_bgr.shape[0], : front_bgr.shape[1]] = front_bgr
            vis[: side_bgr.shape[0], x_side : x_side + side_bgr.shape[1]] = side_bgr

            for i, (px, py) in enumerate(state["front"]):
                cv2.circle(vis, (int(px), int(py)), 6, (0, 255, 0), 2, lineType=cv2.LINE_AA)
                cv2.putText(vis, str(i), (int(px) + 5, int(py) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, lineType=cv2.LINE_AA)
            for i, (px, py) in enumerate(state["side"]):
                xx = int(px) + x_side
                cv2.circle(vis, (xx, int(py)), 6, (0, 255, 0), 2, lineType=cv2.LINE_AA)
                cv2.putText(vis, str(i), (xx + 5, int(py) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, lineType=cv2.LINE_AA)

            cv2.putText(vis, "FRONT", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            cv2.putText(vis, "SIDE", (x_side + 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2, lineType=cv2.LINE_AA)

            if state["phase"] == "front":
                msg = f"Nine-grid FRONT: {len(state['front'])}/9 | click left by 0..8 | SPACE=next R=reset Esc=abort"
            else:
                msg = f"Nine-grid SIDE: {len(state['side'])}/9 | click right by 0..8 | SPACE=finish R=reset Esc=abort"
            cv2.putText(vis, msg, (10, h + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (40, 220, 40), 2, lineType=cv2.LINE_AA)

            cv2.imshow(wname, vis)
            key = cv2.waitKey(30) & 0xFF
            if key in (27,):
                raise SystemExit("Calibration aborted by user (Esc)")
            if key in (ord("r"), ord("R")):
                if state["phase"] == "front":
                    state["front"] = []
                else:
                    state["side"] = []
            if key in (ord(" "),):
                if state["phase"] == "front" and len(state["front"]) == 9:
                    state["phase"] = "side"
                elif state["phase"] == "side" and len(state["side"]) == 9:
                    break
    finally:
        cv2.destroyWindow(wname)

    cf = np.array(state["front"], dtype=np.float32).reshape(9, 2)
    cs = np.array(state["side"], dtype=np.float32).reshape(9, 2)
    return cf, cs


def calibrate_cube_pairs_dual_view_interactive(
    front_bgr: np.ndarray,
    side_bgr: np.ndarray,
    *,
    n_points: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dual view cube correspondence calibration.
    """
    dynamic_mode = int(n_points) <= 0
    target_n = int(max(0, n_points))
    state: dict[str, Any] = {"phase": "front", "front": [], "side": []}
    gap = 16
    h = max(front_bgr.shape[0], side_bgr.shape[0])
    w = front_bgr.shape[1] + gap + side_bgr.shape[1]
    x_side = front_bgr.shape[1] + gap

    def _can_advance_front() -> bool:
        if dynamic_mode:
            return len(state["front"]) >= 1
        return len(state["front"]) == target_n

    def _target_n() -> int:
        return len(state["front"]) if dynamic_mode else target_n

    def _can_finish_side() -> bool:
        return len(state["side"]) == _target_n() and _target_n() > 0

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if state["phase"] == "front":
            if not (0 <= x < front_bgr.shape[1] and 0 <= y < front_bgr.shape[0]):
                return
            if (not dynamic_mode) and len(state["front"]) >= target_n:
                return
            state["front"].append((float(x), float(y)))
        else:
            if not (x_side <= x < x_side + side_bgr.shape[1] and 0 <= y < side_bgr.shape[0]):
                return
            if len(state["side"]) >= _target_n():
                return
            state["side"].append((float(x - x_side), float(y)))

    wname = "CALIB cube pairs FRONT|SIDE"
    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(wname, on_mouse)
    try:
        while True:
            vis = np.zeros((h + 56, w, 3), dtype=np.uint8)
            vis[: front_bgr.shape[0], : front_bgr.shape[1]] = front_bgr
            vis[: side_bgr.shape[0], x_side : x_side + side_bgr.shape[1]] = side_bgr

            for i, (px, py) in enumerate(state["front"]):
                cv2.circle(vis, (int(px), int(py)), 6, (255, 120, 0), 2, lineType=cv2.LINE_AA)
                cv2.putText(vis, str(i), (int(px) + 5, int(py) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, lineType=cv2.LINE_AA)
            for i, (px, py) in enumerate(state["side"]):
                xx = int(px) + x_side
                cv2.circle(vis, (xx, int(py)), 6, (0, 220, 255), 2, lineType=cv2.LINE_AA)
                cv2.putText(vis, str(i), (xx + 5, int(py) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, lineType=cv2.LINE_AA)

            cv2.putText(vis, "FRONT", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            cv2.putText(vis, "SIDE", (x_side + 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2, lineType=cv2.LINE_AA)

            if state["phase"] == "front":
                req = f"{len(state['front'])}/{target_n}" if not dynamic_mode else f"{len(state['front'])} (dynamic)"
                msg = f"Phase FRONT: click points on left | count={req} | SPACE=next  R=reset  Esc=abort"
            else:
                msg = f"Phase SIDE: click same-order points on right | {len(state['side'])}/{_target_n()} | SPACE=finish  R=reset  Esc=abort"
            cv2.putText(vis, msg, (10, h + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (40, 220, 40), 2, lineType=cv2.LINE_AA)

            cv2.imshow(wname, vis)
            key = cv2.waitKey(30) & 0xFF
            if key in (27,):
                raise SystemExit("Calibration aborted by user (Esc)")
            if key in (ord("r"), ord("R")):
                if state["phase"] == "front":
                    state["front"] = []
                else:
                    state["side"] = []
            if key in (ord(" "),):
                if state["phase"] == "front" and _can_advance_front():
                    state["phase"] = "side"
                elif state["phase"] == "side" and _can_finish_side():
                    break
    finally:
        cv2.destroyWindow(wname)

    cf = np.array(state["front"], dtype=np.float32).reshape(-1, 2)
    cs = np.array(state["side"], dtype=np.float32).reshape(-1, 2)
    return cf, cs


def _estimate_dual_view_mapping(
    centers_front: np.ndarray,
    centers_side: np.ndarray,
    cube_pairs_front: np.ndarray,
    cube_pairs_side: np.ndarray,
) -> dict[str, Any] | None:
    cf = np.asarray(centers_front, dtype=np.float32).reshape(-1, 2)
    cs = np.asarray(centers_side, dtype=np.float32).reshape(-1, 2)
    pf = np.asarray(cube_pairs_front, dtype=np.float32).reshape(-1, 2)
    ps = np.asarray(cube_pairs_side, dtype=np.float32).reshape(-1, 2)

    pts_f = [cf]
    pts_s = [cs]
    if int(pf.shape[0]) > 0 and int(ps.shape[0]) > 0 and int(pf.shape[0]) == int(ps.shape[0]):
        pts_f.append(pf)
        pts_s.append(ps)
    src = np.concatenate(pts_f, axis=0).astype(np.float32)
    dst = np.concatenate(pts_s, axis=0).astype(np.float32)
    if int(src.shape[0]) < 4:
        return None

    Hfs, inliers = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if Hfs is None:
        return None
    Hsf, _ = cv2.findHomography(dst, src, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if Hsf is None:
        return None

    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), Hfs).reshape(-1, 2)
    err = np.linalg.norm(proj - dst, axis=1).astype(np.float64)
    inl = int(inliers.sum()) if inliers is not None else int(src.shape[0])
    return {
        "homography_front_to_side": np.asarray(Hfs, dtype=np.float64).tolist(),
        "homography_side_to_front": np.asarray(Hsf, dtype=np.float64).tolist(),
        "n_points_total": int(src.shape[0]),
        "n_points_grid": int(cf.shape[0]),
        "n_points_cube_pairs": int(min(pf.shape[0], ps.shape[0])),
        "n_inliers_ransac": int(inl),
        "reproj_error_mean_px": float(np.mean(err)),
        "reproj_error_median_px": float(np.median(err)),
        "reproj_error_max_px": float(np.max(err)),
    }

def save_calibration_json(
    path: Path,
    *,
    centers_front: np.ndarray,
    centers_side: np.ndarray,
    front_path: str,
    side_path: str,
    cube_pairs_front: np.ndarray | None = None,
    cube_pairs_side: np.ndarray | None = None,
    view_mapping: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "centers_front": centers_front.astype(float).tolist(),
        "centers_side": centers_side.astype(float).tolist(),
        "front_image": front_path,
        "side_image": side_path,
    }
    if cube_pairs_front is not None and cube_pairs_side is not None:
        cf = np.asarray(cube_pairs_front, dtype=np.float32).reshape(-1, 2)
        cs = np.asarray(cube_pairs_side, dtype=np.float32).reshape(-1, 2)
        if int(cf.shape[0]) == int(cs.shape[0]) and int(cf.shape[0]) > 0:
            obj["cube_pairs_front"] = cf.astype(float).tolist()
            obj["cube_pairs_side"] = cs.astype(float).tolist()
            obj["cube_pair_count"] = int(cf.shape[0])
    if isinstance(view_mapping, dict) and len(view_mapping) > 0:
        obj["view_mapping"] = view_mapping
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def _intro_dialog() -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(
            "Nine-Grid Calibration",
            "Front and Side images will be displayed side-by-side.\n\n"
            "In each window: Click the 9 grid cells in order 0->8 (row-major).\n"
            "After clicking 9 points, press SPACE to confirm. Press R to reset, Esc to abort.\n\n"
            "If cube-pairs is set, you will be prompted to click cube correspondence points as well.",
        )
        root.destroy()
    except Exception:
        pass


def load_or_calibrate_dual_views(
    front_bgr: np.ndarray,
    side_bgr: np.ndarray,
    *,
    calib_json: Path,
    recalibrate: bool,
    front_path: str,
    side_path: str,
    cube_pairs: int = 0,
) -> dict[str, Any]:
    if calib_json.is_file() and not recalibrate:
        import json
        return json.loads(calib_json.read_text(encoding="utf-8"))
    
    _intro_dialog()
    print("Calibrate nine grid: Click 0..8 on FRONT (left), then 0..8 on SIDE (right).", file=sys.stderr)
    cf, cs = calibrate_nine_cells_dual_view_interactive(front_bgr, side_bgr)
    
    cpf = np.zeros((0, 2), dtype=np.float32)
    cps = np.zeros((0, 2), dtype=np.float32)
    if int(cube_pairs) > 0:
        n = int(cube_pairs)
        print(f"Calibrate mapping: Click {n} cubes on FRONT, SPACE, then same order on SIDE.", file=sys.stderr)
    else:
        print("Calibrate mapping: Click all visible cubes on FRONT, SPACE, then same order on SIDE.", file=sys.stderr)
        
    cpf, cps = calibrate_cube_pairs_dual_view_interactive(front_bgr, side_bgr, n_points=int(cube_pairs))
    n = int(cpf.shape[0])
    if n <= 0 or int(cps.shape[0]) != n:
        raise SystemExit("Point capture failed: front/side count mismatch or empty")
        
    vm = _estimate_dual_view_mapping(cf, cs, cpf, cps)
    out = {
        "centers_front": cf,
        "centers_side": cs,
        "cube_pairs_front": cpf,
        "cube_pairs_side": cps,
        "view_mapping": vm,
    }
    if vm is not None:
        print(
            f"Joint homography estimated: mean={vm['reproj_error_mean_px']:.2f}px "
            f"median={vm['reproj_error_median_px']:.2f}px "
            f"inliers={vm['n_inliers_ransac']}/{vm['n_points_total']}",
            file=sys.stderr,
        )
    save_calibration_json(
        calib_json,
        centers_front=cf,
        centers_side=cs,
        front_path=front_path,
        side_path=side_path,
        cube_pairs_front=cpf,
        cube_pairs_side=cps,
        view_mapping=vm,
    )
    print(f"Saved calibration: {calib_json.resolve()}", file=sys.stderr)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Dual view calibration (nine grid + optional cube pairs)")
    p.add_argument("--front", type=Path, required=True, help="Front camera image")
    p.add_argument("--side", type=Path, required=True, help="Side camera image")
    p.add_argument("--out", type=Path, required=True, help="Output calibration JSON path")
    p.add_argument("--recalibrate", action="store_true", help="Recalibrate and overwrite even if --out exists")
    p.add_argument("--skip-dialog", action="store_true", help="Skip Tk interative dialog")
    p.add_argument("--cube-pairs", type=int, default=0, help="Number of cube pairs to align views. N=0 means click all visible cubes dynamically.")
    
    args = p.parse_args()

    out = args.out.expanduser().resolve()
    if out.is_file() and not args.recalibrate:
        print(f"File {out} already exists. Use --recalibrate to overwrite.", file=sys.stderr)
        return 1

    front_path = args.front.expanduser().resolve()
    side_path = args.side.expanduser().resolve()
    if not front_path.is_file() or not side_path.is_file():
        print(f"Images not found: {front_path} / {side_path}", file=sys.stderr)
        return 1

    fb = cv2.imread(str(front_path), cv2.IMREAD_COLOR)
    sb = cv2.imread(str(side_path), cv2.IMREAD_COLOR)
    if fb is None or sb is None:
        print("Failed to read images.", file=sys.stderr)
        return 1

    if not args.skip_dialog:
        _intro_dialog()

    load_or_calibrate_dual_views(
        fb,
        sb,
        calib_json=out,
        recalibrate=True,
        front_path=str(front_path),
        side_path=str(side_path),
        cube_pairs=int(args.cube_pairs),
    )
    return 0

if __name__ == "__main__":
    sys.exit(main())
