"""Optional Set-of-Marks overlay and mild color augmentation for vlmsearch rollouts."""

from __future__ import annotations

import json
import random
from typing import Any, List, Optional

import ast
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

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


def apply_mild_color_augmentation(
    image: Image.Image, strength: float, seed: Optional[int] = None
) -> Image.Image:
    """
    Non-geometric aug only (brightness / contrast / color). Safe for fixed (x,y) labels.
    strength in [0, 1]; 0 disables.
    """
    if strength <= 0:
        return image
    rng = random.Random(seed)
    img = image.convert("RGB")
    b = 1.0 + (rng.random() * 2 - 1) * 0.12 * strength
    c = 1.0 + (rng.random() * 2 - 1) * 0.12 * strength
    col = 1.0 + (rng.random() * 2 - 1) * 0.08 * strength
    img = ImageEnhance.Brightness(img).enhance(b)
    img = ImageEnhance.Contrast(img).enhance(c)
    img = ImageEnhance.Color(img).enhance(col)
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
    aug_strength: float,
    aug_seed: Optional[int],
) -> tuple[Image.Image, str]:
    """
    Load image, apply max_image_side / max_pixels (scaling ground-truth for point judges),
    then optional SoM and color augmentation.
    """
    input_image = Image.open(input_image_path)

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

    if aug_strength > 0:
        input_image = apply_mild_color_augmentation(
            input_image, aug_strength, seed=aug_seed
        )

    return input_image, true_answer
