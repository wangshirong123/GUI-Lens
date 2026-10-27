"""Direct baseline: one model call on the full screenshot, no zoom, OCR or detector.

With coords_normalized the model answers in [0, 1000] and the point is scaled to pixels.

GUILENS_FIT_MP (megapixels) sends a screenshot above that size already downscaled to it and states the
downscaled size in the prompt. An endpoint that downscales oversized images on its own does not tell the
model, so a pixel answer would be in the downscaled space while the prompt declares the original size.

Uses the same instruction wording as the ZoomClick baseline (official Qwen3-VL
grounding prompt) so the two differ only in the zoom procedure.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from gui_lens.planning.zoomclick_locator import _PROMPT
from gui_lens.planning.screenspot_locator import parse_json

# Models trained on relative coordinates (Qwen3-VL) answer in [0, 1000] whatever the prompt asks.
_PROMPT_NORM = """Please click on the element described as: {instruction}. Respond with the point to click, as relative coordinates in [0, 1000] of the image width and height.
Reply with ONLY JSON: {{"point_2d": [x, y]}}"""


def _point(parsed: dict) -> tuple[float, float]:
    """{"x": .., "y": ..}, {"point_2d": [x, y]}, or Qwen's {"x": [x, y]}."""
    for key in ("point_2d", "x"):
        value = parsed.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return float(value[0]), float(value[1])
    return float(parsed["x"]), float(parsed["y"])


def _fit(img_path, img_w, img_h, out_dir) -> tuple[str, int, int]:
    """Image path and size to send under the GUILENS_FIT_MP budget (unchanged when within it)."""
    budget = float(os.environ.get("GUILENS_FIT_MP") or 0) * 1e6
    if not budget or img_w * img_h <= budget:
        return img_path, img_w, img_h
    from PIL import Image
    k = (budget / (img_w * img_h)) ** 0.5
    w, h = round(img_w * k), round(img_h * k)
    path = str(Path(out_dir) / f"direct_fit_{w}x{h}_{Path(img_path).stem}.png")
    Image.open(img_path).convert("RGB").resize((w, h), Image.LANCZOS).save(path)
    return path, w, h


def locate(task, img_path, img_w, img_h, runtime, out_dir, config, *, run) -> Optional[dict]:
    send_path, w, h = _fit(img_path, img_w, img_h, out_dir)
    prompt = _PROMPT_NORM.format(instruction=task) if config.coords_normalized else _PROMPT.format(instruction=task, w=w, h=h)
    try:
        parsed = parse_json(run(runtime, content=[
            {"type": "text", "text": prompt},
            {"type": "image", "path": send_path},
        ])) or {}
        x, y = _point(parsed)
        if config.coords_normalized:
            x, y = x / 1000.0 * img_w, y / 1000.0 * img_h
        else:
            x, y = x * img_w / w, y * img_h / h
    except Exception as exc:
        print(f"  [direct] no point ({exc.__class__.__name__}: {str(exc)[:120]})", file=sys.stderr)
        return None
    cx, cy = int(round(min(max(x, 0), img_w - 1))), int(round(min(max(y, 0), img_h - 1)))
    return {
        "id": "direct", "label": "direct", "name": "direct",
        "source": "direct", "grounding_type": "direct",
        "cx": cx, "cy": cy, "x": cx - 24, "y": cy - 24, "w": 48, "h": 48,
        "confidence": 0.0, "reasoning": "",
    }
