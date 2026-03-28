"""Optional Set-of-Marks overlay, resize helpers, and real prior-state images for vlmsearch."""

from __future__ import annotations

import json
import os
from typing import Any, List, Optional

import ast
from PIL import Image, ImageDraw, ImageFont

from qwen_vl_utils import fetch_image


def parse_som_bboxes(data: Any) -> List[dict]:
    """Normalize JSONL field into a list of {id, xyxy: [x1,y1,x2,y2]} dicts."""
    if data is None:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, str):
        s = data.strip()
        if not s:
            return []
        return json.loads(s)
    return []


def parse_previous_state_paths(data: Any) -> List[str]:
    """JSON list of relative (to image_root) or absolute image paths, oldest → newest."""
    if data is None:
        return []
    if isinstance(data, list):
        return [str(p) for p in data]
    if isinstance(data, str):
        s = data.strip()
        if not s:
            return []
        return [str(p) for p in json.loads(s)]
    return []


def resize_image_for_rollout(
    input_image: Image.Image,
    max_image_side: Optional[int],
    max_pixels: Optional[int],
) -> Image.Image:
    """Same geometry as the main observation (no label scaling)."""
    width, height = input_image.size
    max_side = max(width, height)
    if max_image_side is not None and max_side > max_image_side:
        scale_factor = max_image_side / max_side
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        input_image = input_image.resize((new_width, new_height))
    width, height = input_image.size
    if max_pixels is not None and width * height > max_pixels:
        input_image = fetch_image({"image": input_image, "max_pixels": max_pixels})
    return input_image


def load_previous_state_images(
    relative_or_abs_paths: List[str],
    image_root: str,
    max_image_side: Optional[int],
    max_pixels: Optional[int],
) -> List[Image.Image]:
    """Load real screenshots from disk; same resize as current state. No synthesis."""
    out: List[Image.Image] = []
    for p in relative_or_abs_paths:
        full = p if os.path.isabs(p) else os.path.join(image_root, p)
        if not os.path.isfile(full):
            continue
        im = Image.open(full).convert("RGB")
        out.append(resize_image_for_rollout(im, max_image_side, max_pixels))
    return out


def apply_som_overlay(image: Image.Image, boxes: List[dict]) -> Image.Image:
    """
    Draw numbered rectangles like a lightweight SoM. Boxes must be in the same
    pixel coordinate system as the given image (after any resize you apply before calling).
    """
    if not boxes:
        return image
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    for item in boxes:
        bid = int(item["id"])
        x1, y1, x2, y2 = (int(x) for x in item["xyxy"])
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        label = str(bid)
        if hasattr(draw, "textbbox"):
            lx, ly, rx, ry = draw.textbbox((0, 0), label, font=font)
            tw, th = rx - lx, ry - ly
        else:
            tw, th = draw.textsize(label, font=font)
        tx = max(0, min(x1, img.size[0] - tw))
        ty = max(0, y1 - th - 2)
        draw.rectangle([tx, ty, tx + tw + 2, ty + th + 2], fill="yellow")
        draw.text((tx + 1, ty + 1), label, fill="black", font=font)
    return img


def preprocess_rollout_image(
    input_image_path: str,
    true_answer: str,
    *,
    max_image_side: Optional[int],
    max_pixels: Optional[int],
    judge_type: str,
    use_som: bool,
    som_bboxes: Any,
) -> tuple[Image.Image, str]:
    """
    Load image, apply max_image_side / max_pixels (scaling ground-truth for point judges),
    then optional SoM on the current state only.
    """
    input_image = Image.open(input_image_path).convert("RGB")

    width, height = input_image.size
    max_side = max(width, height)
    scale_factor = 1.0
    if max_image_side is not None and max_side > max_image_side:
        scale_factor = max_image_side / max_side
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        input_image = input_image.resize((new_width, new_height))
        if judge_type == "point_match":
            gt_point = tuple(ast.literal_eval(true_answer))
            gt_point = tuple(int(x * scale_factor) for x in gt_point)
            true_answer = str(gt_point)
        elif judge_type == "point_in_bbox":
            gt_bbox = tuple(ast.literal_eval(true_answer))
            gt_bbox = tuple(int(x * scale_factor) for x in gt_bbox)
            true_answer = str(gt_bbox)

    width, height = input_image.size
    if max_pixels is not None and width * height > max_pixels:
        input_image = fetch_image({"image": input_image, "max_pixels": max_pixels})
        scale_factor_width = input_image.size[0] / width
        scale_factor_height = input_image.size[1] / height
        if judge_type == "point_match":
            gt_point = tuple(ast.literal_eval(true_answer))
            gt_point = (
                int(gt_point[0] * scale_factor_width),
                int(gt_point[1] * scale_factor_height),
            )
            true_answer = str(gt_point)
        elif judge_type == "point_in_bbox":
            gt_bbox = tuple(ast.literal_eval(true_answer))
            gt_bbox = (
                int(gt_bbox[0] * scale_factor_width),
                int(gt_bbox[1] * scale_factor_height),
                int(gt_bbox[2] * scale_factor_width),
                int(gt_bbox[3] * scale_factor_height),
            )
            true_answer = str(gt_bbox)

    boxes = parse_som_bboxes(som_bboxes) if use_som else []
    if use_som and boxes:
        input_image = apply_som_overlay(input_image, boxes)

    return input_image, true_answer
