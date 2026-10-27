"""ZoomClick baseline (Jiang et al., arXiv:2512.05941) on a general VLM.

Follows the official implementation (github.com/Princeton-AI2-Lab/ZoomClick,
grounding/eval_sspro_zoomclick.py, PrezoomIterativeNarrowingRunner) with its
ScreenSpot-Pro settings (scripts/run_zoomclick_qwen3.slurm):
  in_depth 3, in_ratio 0.5, in_min_crop 768, patch_size 2, center_mode clip,
  prezoom_px_thresh 50.

Iteration 1 (pre-zoom): one click prediction on the full screenshot and one on
each 2x2 patch; if the patch point nearest the direct point is within 50 px it
is used, otherwise the direct point. Later iterations crop the current view to
max(0.5 * size, 768) centered on the last prediction (clipped to the view),
predict again, and map back. The click after the last iteration is the answer.

The only adaptation is the point predictor: the official backends are
grounding models with their own output formats; here the same runtime as the
other locators is asked for a pixel point in the displayed image, using the
official Qwen3-VL user instruction wording.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

from PIL import Image

from gui_lens.utils import parse_json

DEPTH = 3
SHRINK_RATIO = 0.5
MIN_CROP = 768
PATCH_SIZE = 2
PREZOOM_PX_THRESH = 50.0

_PROMPT = """Please click on the element described as: {instruction}. Respond with the exact pixel coordinates.
The image is {w}x{h} pixels; coordinates are pixels of this image.
Reply with ONLY JSON: {{"x": 0, "y": 0}}"""


def _log(message: str) -> None:
    print(f"  [zoomclick] {message}", file=sys.stderr)


def _patches() -> list[tuple[float, float, float, float]]:
    return [(c / PATCH_SIZE, r / PATCH_SIZE, (c + 1) / PATCH_SIZE, (r + 1) / PATCH_SIZE)
            for r in range(PATCH_SIZE) for c in range(PATCH_SIZE)]


def _crop_clip(image: Image.Image, center_px, target_w, target_h):
    """Official crop_center_px with center_mode='clip'."""
    W, H = image.size
    cx, cy = center_px
    tw = min(max(1, int(round(target_w))), W)
    th = min(max(1, int(round(target_h))), H)
    x1 = int(round(cx - tw / 2)); y1 = int(round(cy - th / 2))
    x2 = x1 + tw; y2 = y1 + th
    ix1, iy1, ix2, iy2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if ix2 <= ix1:
        ix1 = max(0, min(W - 1, int(round(cx)))); ix2 = min(W, ix1 + 1)
    if iy2 <= iy1:
        iy1 = max(0, min(H - 1, int(round(cy)))); iy2 = min(H, iy1 + 1)
    return image.crop((ix1, iy1, ix2, iy2)), (ix1, iy1, ix2, iy2)


def locate(task, img_path, img_w, img_h, runtime, out_dir, config, *, run) -> Optional[dict]:
    calls = {"n": 0}
    trace: list[dict] = []

    def predict(img: Image.Image, name: str):
        """Normalized (x, y) in img, or None. One model call."""
        path = str(Path(out_dir) / f"zoomclick_{name}.png")
        img.save(path)
        w, h = img.size
        calls["n"] += 1
        try:
            parsed = parse_json(run(runtime, content=[
                {"type": "text", "text": _PROMPT.format(instruction=task, w=w, h=h)},
                {"type": "image", "path": path},
            ])) or {}
            x, y = float(parsed["x"]), float(parsed["y"])
        except Exception as exc:
            _log(f"{name}: no point ({exc.__class__.__name__}: {str(exc)[:120]})")
            trace.append({"call": name, "size": [w, h], "point": None})
            return None
        trace.append({"call": name, "size": [w, h], "point": [x, y]})
        return (x / w, y / h)

    image = Image.open(img_path).convert("RGB")
    W, H = image.size
    viewport = (0.0, 0.0, 1.0, 1.0)
    curr = image
    last_real = None
    prezoom: dict = {}
    for it in range(DEPTH):
        if it == 0:
            direct = predict(curr, "direct")
            patch_points = []
            for k, (x1, y1, x2, y2) in enumerate(_patches()):
                patch = curr.crop((int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)))
                pt = predict(patch, f"patch{k + 1}")
                if pt is None:
                    patch_points.append(((x1 + x2) / 2, (y1 + y2) / 2))
                else:
                    patch_points.append((x1 + pt[0] * (x2 - x1), y1 + pt[1] * (y2 - y1)))
            if direct is not None:
                dists = sorted((math.hypot((p[0] - direct[0]) * W, (p[1] - direct[1]) * H), i)
                               for i, p in enumerate(patch_points))
                d, best = dists[0]
                pt_view = patch_points[best] if d < PREZOOM_PX_THRESH else direct
                prezoom = {"min_distance": round(d, 1), "choice": f"patch{best + 1}" if d < PREZOOM_PX_THRESH else "direct"}
            else:
                pt_view = patch_points[0]
                prezoom = {"choice": "patch1 (no direct point)"}
        else:
            pt_view = predict(curr, f"iter{it + 1}")
            if pt_view is None:
                break
        vx1, vy1, vx2, vy2 = viewport
        last_real = (min(1.0, max(0.0, vx1 + (vx2 - vx1) * pt_view[0])),
                     min(1.0, max(0.0, vy1 + (vy2 - vy1) * pt_view[1])))
        if it == DEPTH - 1:
            break
        cw, ch = curr.size
        tw = min(max(int(round(cw * SHRINK_RATIO)), MIN_CROP), cw)
        th = min(max(int(round(ch * SHRINK_RATIO)), MIN_CROP), ch)
        if tw == cw and th == ch:
            break
        center = (int(round(pt_view[0] * cw)), int(round(pt_view[1] * ch)))
        curr, (cx1, cy1, cx2, cy2) = _crop_clip(curr, center, tw, th)
        nb = (cx1 / cw, cy1 / ch, cx2 / cw, cy2 / ch)
        vw, vh = vx2 - vx1, vy2 - vy1
        viewport = (vx1 + vw * nb[0], vy1 + vh * nb[1], vx1 + vw * nb[2], vy1 + vh * nb[3])

    if last_real is None:
        return None
    cx, cy = int(round(last_real[0] * W)), int(round(last_real[1] * H))
    _log(f"click ({cx},{cy}) after {calls['n']} calls, prezoom={prezoom}")
    return {
        "id": "zoomclick", "label": "zoomclick", "name": "zoomclick",
        "source": "zoomclick", "grounding_type": "zoomclick",
        "cx": cx, "cy": cy, "x": cx - 24, "y": cy - 24, "w": 48, "h": 48,
        "confidence": 0.0, "reasoning": "",
        "zoomclick": {"model_calls": calls["n"], "prezoom": prezoom, "trace": trace,
                      "final_viewport": [round(v, 5) for v in viewport]},
    }
