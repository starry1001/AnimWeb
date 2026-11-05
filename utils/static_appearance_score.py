#!/usr/bin/env python3
"""Static Appearance Fidelity: Structure, Text, Color, and CLIP similarity."""

import argparse
import colorsys
import json
import math
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

try:
    import torch
except ImportError:
    torch = None


DEFAULT_MAX_WIDTH = 640
CONTAINED_BOX_THRESHOLD = 0.90
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GENERATED_ROOT = PROJECT_ROOT / "experiment"
DEFAULT_GROUND_TRUTH_ROOT = PROJECT_ROOT / "experiment" / "ground_truth"
GROUND_TRUTH_KEYFRAMES_DIRNAME = "keyframes"

CLIP_MODEL = None
CLIP_PREPROCESS = None
CLIP_DEVICE = None
EASYOCR_READER = None


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float
    area: float
    text: str = ""
    color: tuple[int, int, int] | None = None
    detection_id: int | None = None
    class_name: str = ""


@dataclass
class TextResult:
    text: str
    boxes: list[Box]
    method: str
    available: bool = True


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def load_rgb(image_path: Path) -> np.ndarray:
    return np.asarray(Image.open(image_path).convert("RGB"))


def get_clip_model():
    global CLIP_MODEL, CLIP_PREPROCESS, CLIP_DEVICE
    if torch is None:
        raise ImportError("torch is not installed")
    if CLIP_MODEL is None or CLIP_PREPROCESS is None:
        import clip

        CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        CLIP_MODEL, CLIP_PREPROCESS = clip.load("ViT-B/32", device=CLIP_DEVICE)
    return CLIP_MODEL, CLIP_PREPROCESS, CLIP_DEVICE


def clip_similarity(image_path1: Path, image_path2: Path) -> tuple[float, str]:
    model, preprocess, device = get_clip_model()
    image1 = preprocess(Image.open(image_path1).convert("RGB")).unsqueeze(0).to(device)
    image2 = preprocess(Image.open(image_path2).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        features1 = model.encode_image(image1)
        features2 = model.encode_image(image2)

    features1 = features1 / features1.norm(p=2, dim=-1, keepdim=True)
    features2 = features2 / features2.norm(p=2, dim=-1, keepdim=True)
    return float(torch.nn.functional.cosine_similarity(features1, features2).item()), "clip_vit_b_32"


def resize_for_analysis(rgb: np.ndarray, max_width: int) -> np.ndarray:
    height, width = rgb.shape[:2]
    if width <= max_width:
        return rgb
    scale = max_width / width
    return cv2.resize(rgb, (max_width, int(round(height * scale))), interpolation=cv2.INTER_AREA)


def get_easyocr_reader():
    global EASYOCR_READER
    if EASYOCR_READER is None:
        import easyocr

        EASYOCR_READER = easyocr.Reader(
            ["en"],
            gpu=torch.cuda.is_available() if torch is not None else False,
            download_enabled=True,
            verbose=False,
        )
    return EASYOCR_READER


def box_iou(left: Box, right: Box) -> float:
    x1 = max(left.x, right.x)
    y1 = max(left.y, right.y)
    x2 = min(left.x + left.w, right.x + right.w)
    y2 = min(left.y + left.h, right.y + right.h)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = left.area + right.area - inter
    return inter / union if union > 0 else 0.0


def intersection_area(left: Box, right: Box) -> float:
    x1 = max(left.x, right.x)
    y1 = max(left.y, right.y)
    x2 = min(left.x + left.w, right.x + right.w)
    y2 = min(left.y + left.h, right.y + right.h)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def contained_ratio(inner: Box, outer: Box) -> float:
    if inner.area <= 0:
        return 0.0
    return intersection_area(inner, outer) / inner.area


def remove_contained_boxes(boxes: list[Box], threshold: float = CONTAINED_BOX_THRESHOLD) -> list[Box]:
    kept: list[Box] = []
    for box in sorted(boxes, key=lambda item: item.area, reverse=True):
        if any(contained_ratio(box, kept_box) >= threshold for kept_box in kept):
            continue
        kept.append(box)
    return kept


def center_score(left: Box, right: Box) -> float:
    distance = math.hypot((left.x + left.w / 2) - (right.x + right.w / 2), (left.y + left.h / 2) - (right.y + right.h / 2))
    return clamp01(1.0 - distance / math.sqrt(2.0))


def area_score(left: Box, right: Box) -> float:
    if left.area <= 0 or right.area <= 0:
        return 0.0
    return min(left.area, right.area) / max(left.area, right.area)


def aspect_score(left: Box, right: Box) -> float:
    left_aspect = left.w / max(left.h, 1e-6)
    right_aspect = right.w / max(right.h, 1e-6)
    return min(left_aspect, right_aspect) / max(left_aspect, right_aspect)


def color_similarity_rgb(left: tuple[int, int, int], right: tuple[int, int, int]) -> float:
    diff = np.linalg.norm(np.asarray(left, dtype=np.float32) - np.asarray(right, dtype=np.float32))
    return clamp01(1.0 - diff / math.sqrt(3 * 255 * 255))


def mean_color(rgb: np.ndarray, box: Box) -> tuple[int, int, int]:
    height, width = rgb.shape[:2]
    x1 = int(clamp01(box.x) * width)
    y1 = int(clamp01(box.y) * height)
    x2 = int(clamp01(box.x + box.w) * width)
    y2 = int(clamp01(box.y + box.h) * height)
    crop = rgb[y1:max(y1 + 1, y2), x1:max(x1 + 1, x2)]
    if crop.size == 0:
        return (0, 0, 0)
    return tuple(int(v) for v in crop.reshape(-1, 3).mean(axis=0))


def dominant_rgb(pixels: np.ndarray) -> tuple[int, int, int]:
    """Find the most common 8-level RGB bin and return an actual sampled color."""
    pixels = pixels.reshape(-1, 3)
    if not len(pixels):
        raise ValueError("Cannot find a dominant color among zero pixels")
    bins = pixels // 8
    colors, inverse, counts = np.unique(bins, axis=0, return_inverse=True, return_counts=True)
    best_bin = int(np.argmax(counts))
    candidates = pixels[inverse == best_bin]
    center = colors[best_bin].astype(np.float32) * 8 + 3.5
    closest = candidates[np.argmin(np.sum((candidates.astype(np.float32) - center) ** 2, axis=1))]
    return tuple(int(value) for value in closest)


def gpa_component_mask(rgb: np.ndarray, components: list[Box]) -> np.ndarray:
    """Union of every GPA box, whether it contains OCR text or another UI element."""
    height, width = rgb.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for box in components:
        x1 = max(0, min(width, round(box.x * width)))
        y1 = max(0, min(height, round(box.y * height)))
        x2 = max(x1, min(width, round((box.x + box.w) * width)))
        y2 = max(y1, min(height, round((box.y + box.h) * height)))
        mask[y1:y2, x1:x2] = 255
    return mask


def page_background_color(rgb: np.ndarray, component_mask: np.ndarray) -> tuple[int, int, int] | None:
    """Take 100 evenly spaced non-component pixels and choose their modal color."""
    pixels = rgb[component_mask == 0]
    if not len(pixels):
        return None
    sample_indices = np.linspace(0, len(pixels) - 1, min(100, len(pixels)), dtype=int)
    return dominant_rgb(pixels[sample_indices])


def rgb_to_lab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    """OpenCV expects float RGB in [0, 1] for standard CIE Lab output."""
    pixel = np.asarray(rgb, dtype=np.float32).reshape(1, 1, 3) / 255.0
    return tuple(float(value) for value in cv2.cvtColor(pixel, cv2.COLOR_RGB2LAB)[0, 0])


def ciede2000(lab1: tuple[float, float, float], lab2: tuple[float, float, float]) -> float:
    """CIEDE2000 color difference with unit lightness/chroma/hue weights."""
    l1, a1, b1 = lab1
    l2, a2, b2 = lab2
    c1, c2 = math.hypot(a1, b1), math.hypot(a2, b2)
    mean_c = (c1 + c2) / 2
    g = (1 - math.sqrt(mean_c**7 / (mean_c**7 + 25**7))) / 2
    a1_prime, a2_prime = (1 + g) * a1, (1 + g) * a2
    c1_prime, c2_prime = math.hypot(a1_prime, b1), math.hypot(a2_prime, b2)
    h1 = math.degrees(math.atan2(b1, a1_prime)) % 360
    h2 = math.degrees(math.atan2(b2, a2_prime)) % 360

    delta_l = l2 - l1
    delta_c = c2_prime - c1_prime
    if c1_prime * c2_prime == 0:
        delta_h = 0.0
        mean_h = h1 + h2
    else:
        hue_difference = (h2 - h1 + 180) % 360 - 180
        delta_h = 2 * math.sqrt(c1_prime * c2_prime) * math.sin(math.radians(hue_difference / 2))
        mean_h = (h1 + h2) / 2
        if abs(h1 - h2) > 180:
            mean_h += 180
    mean_h %= 360
    mean_l = (l1 + l2) / 2
    mean_c_prime = (c1_prime + c2_prime) / 2
    t = (1 - 0.17 * math.cos(math.radians(mean_h - 30))
         + 0.24 * math.cos(math.radians(2 * mean_h))
         + 0.32 * math.cos(math.radians(3 * mean_h + 6))
         - 0.20 * math.cos(math.radians(4 * mean_h - 63)))
    delta_theta = 30 * math.exp(-((mean_h - 275) / 25) ** 2)
    r_c = 2 * math.sqrt(mean_c_prime**7 / (mean_c_prime**7 + 25**7))
    s_l = 1 + 0.015 * (mean_l - 50) ** 2 / math.sqrt(20 + (mean_l - 50) ** 2)
    s_c = 1 + 0.045 * mean_c_prime
    s_h = 1 + 0.015 * mean_c_prime * t
    r_t = -math.sin(math.radians(2 * delta_theta)) * r_c
    lightness = delta_l / s_l
    chroma = delta_c / s_c
    hue = delta_h / s_h
    return math.sqrt(max(0.0, lightness**2 + chroma**2 + hue**2 + r_t * chroma * hue))


def color_similarity_ciede2000(left: tuple[int, int, int], right: tuple[int, int, int]) -> float:
    # Dividing by 100 is a score normalization choice, not a bound on delta E.
    return clamp01(1 - ciede2000(rgb_to_lab(left), rgb_to_lab(right)) / 100)


def foreground_mask(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    bg = np.asarray(page_background_color(rgb, np.zeros(rgb.shape[:2], dtype=np.uint8)), dtype=np.float32)
    color_distance = np.linalg.norm(rgb.astype(np.float32) - bg, axis=2)
    edges = cv2.Canny(gray, 60, 160)
    mask = ((color_distance > 28) | (edges > 0) | ((hsv[:, :, 1] > 45) & (gray > 25))).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def detect_visual_boxes(rgb: np.ndarray) -> tuple[list[Box], int]:
    small = resize_for_analysis(rgb, DEFAULT_MAX_WIDTH)
    height, width = small.shape[:2]
    total_area = float(height * width)
    mask = foreground_mask(small)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[Box] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = (w * h) / total_area
        if area < 0.00035 or area > 0.85 or w < 4 or h < 4:
            continue
        boxes.append(Box(x / width, y / height, w / width, h / height, area))
    boxes.sort(key=lambda item: item.area, reverse=True)
    return remove_contained_boxes(boxes)[:80], len(boxes)


def extract_text_boxes_easyocr(rgb: np.ndarray) -> TextResult:
    reader = get_easyocr_reader()
    results = reader.readtext(rgb)
    height, width = rgb.shape[:2]
    boxes: list[Box] = []
    texts = ""
    for points, raw_text, _confidence in results:
        texts += " " + raw_text
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        left = max(0.0, min(xs))
        top = max(0.0, min(ys))
        right = min(float(width), max(xs))
        bottom = min(float(height), max(ys))
        box = Box(
            left / width,
            top / height,
            max(1.0, right - left) / width,
            max(1.0, bottom - top) / height,
            (max(1.0, right - left) * max(1.0, bottom - top)) / (width * height),
            raw_text,
        )
        # box.color = mean_color(rgb, box)  # Used only by the disabled color score.
        boxes.append(box)
    boxes.sort(key=lambda item: (item.y, item.x))
    return TextResult(texts, boxes, "easyocr")


def extract_text(rgb: np.ndarray) -> TextResult:
    return extract_text_boxes_easyocr(rgb)


def load_gpa_components(gpa_dir: Path, frame_index: int, rgb: np.ndarray) -> tuple[list[Box], Path]:
    """Load boxes for the exact source frame and normalize them to its image."""
    result_path = gpa_dir / "frame_results" / f"frame_{frame_index:06d}.json"
    if not result_path.is_file():
        raise FileNotFoundError(
            f"GPA result for source frame {frame_index} not found: {result_path}. "
            "Check that GPA used the same video and did not skip this frame."
        )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if int(payload["source_frame_index"]) != frame_index:
        raise ValueError(f"GPA frame index does not match {result_path}")
    height, width = rgb.shape[:2]
    summary_path = gpa_dir / "summary.json"
    if summary_path.is_file():
        source = json.loads(summary_path.read_text(encoding="utf-8"))["source"]
        if (int(source["width"]), int(source["height"])) != (width, height):
            raise ValueError(
                f"GPA source size differs from keyframe image: {result_path} "
                f"is {source['width']}x{source['height']}, image is {width}x{height}"
            )
    components = []
    for detection in payload["detections"]:
        x1, y1, x2, y2 = map(float, detection["bbox_xyxy"])
        x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
        y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
        if x2 <= x1 or y2 <= y1:
            continue
        components.append(Box(
            x1 / width, y1 / height, (x2 - x1) / width,
            (y2 - y1) / height, (x2 - x1) * (y2 - y1) / (width * height),
            detection_id=int(detection["id"]),
            class_name=str(detection.get("class_name", "")),
        ))
    return components, result_path


def validate_gpa_video(gpa_dir: Path, keyframes_manifest: Path) -> None:
    """Reject a GPA cache made from a different video when metadata exists."""
    summary_path = gpa_dir / "summary.json"
    if not keyframes_manifest.is_file() or not summary_path.is_file():
        return
    keyframes_video = json.loads(keyframes_manifest.read_text(encoding="utf-8"))[
        "video_path"
    ]
    gpa_video = json.loads(summary_path.read_text(encoding="utf-8"))["input_video"]
    if Path(keyframes_video).resolve() != Path(gpa_video).resolve():
        raise ValueError(
            f"GPA cache {gpa_dir} uses {gpa_video}, but keyframes "
            f"in {keyframes_manifest} use {keyframes_video}"
        )


def attach_ocr_text(components: list[Box], text_boxes: list[Box]) -> None:
    """Assign each OCR line to the smallest GPA component covering it."""
    assigned: dict[int, list[tuple[float, float, str]]] = {}
    for text_box in text_boxes:
        if not text_box.text.strip() or text_box.area <= 0:
            continue
        candidates = []
        for index, component in enumerate(components):
            coverage = intersection_area(text_box, component) / text_box.area
            if coverage >= 0.5:
                candidates.append((coverage, -component.area, index))
        if candidates:
            index = max(candidates)[2]
            assigned.setdefault(index, []).append(
                (text_box.y, text_box.x, text_box.text.strip())
            )
    for index, lines in assigned.items():
        components[index].text = " ".join(
            text for _, _, text in sorted(lines)
        )


def component_text_match_score(ground_truth: Box, rendered: Box) -> float:
    if not ground_truth.text.strip() or not rendered.text.strip():
        return 0.0
    return SequenceMatcher(
        None, ground_truth.text.strip().lower(), rendered.text.strip().lower()
    ).ratio()


def component_geometry_match_score(ground_truth: Box, rendered: Box) -> float:
    position = clamp01(1.0 - 2.0 * math.hypot(
        ground_truth.x + ground_truth.w / 2 - rendered.x - rendered.w / 2,
        ground_truth.y + ground_truth.h / 2 - rendered.y - rendered.h / 2,
    ))
    width = min(ground_truth.w, rendered.w) / max(ground_truth.w, rendered.w)
    height = min(ground_truth.h, rendered.h) / max(ground_truth.h, rendered.h)
    return 0.6 * position + 0.4 * (width + height) / 2


def _assign_component_pairs(
    ground_truth: list[Box],
    rendered: list[Box],
    ground_truth_indices: list[int],
    rendered_indices: list[int],
    score_function,
    threshold: float,
    *,
    text_stage: bool = False,
) -> list[tuple[int, int, float]]:
    """One-to-one assignment with a dummy option for each GT component."""
    if not ground_truth_indices or not rendered_indices:
        return []
    similarities = np.array([
        [score_function(ground_truth[i], rendered[j]) for j in rendered_indices]
        for i in ground_truth_indices
    ], dtype=np.float64)
    profit = similarities - threshold
    if text_stage:
        # Geometry resolves duplicate OCR strings without overriding text rank.
        geometry = np.array([
            [component_geometry_match_score(ground_truth[i], rendered[j])
             for j in rendered_indices]
            for i in ground_truth_indices
        ], dtype=np.float64)
        profit += 1e-6 * geometry
    profit = np.concatenate((profit, np.zeros((len(ground_truth_indices),
                                               len(ground_truth_indices)))), axis=1)
    rows, columns = linear_sum_assignment(-profit)
    return [
        (ground_truth_indices[int(i)], rendered_indices[int(j)], float(similarities[i, j]))
        for i, j in zip(rows, columns)
        if j < len(rendered_indices) and similarities[i, j] >= threshold
    ]


def match_gpa_components(
    ground_truth: list[Box], rendered: list[Box]
) -> tuple[float, dict]:
    """Match components one-to-one, then score only the matched geometry."""
    if not ground_truth and not rendered:
        return 1.0, {"ground_truth_count": 0, "rendered_count": 0,
                     "component_union_count": 0, "matches": [], "unmatched_ground_truth_ids": [],
                     "unmatched_rendered_ids": []}
    text_pairs = _assign_component_pairs(
        ground_truth, rendered, list(range(len(ground_truth))),
        list(range(len(rendered))), component_text_match_score, 0.3,
        text_stage=True,
    )
    text_gt = {i for i, _, _ in text_pairs}
    text_rendered = {j for _, j, _ in text_pairs}
    geometry_pairs = _assign_component_pairs(
        ground_truth, rendered,
        [i for i in range(len(ground_truth)) if i not in text_gt],
        [j for j in range(len(rendered)) if j not in text_rendered],
        component_geometry_match_score, 0.5,
    )
    pairs = [(i, j, similarity, "ocr") for i, j, similarity in text_pairs]
    pairs += [(i, j, similarity, "geometry") for i, j, similarity in geometry_pairs]
    matched_gt = {i for i, _, _, _ in pairs}
    matched_rendered = {j for _, j, _, _ in pairs}
    matches = []
    geometry_total = 0.0
    for i, j, match_score, match_method in pairs:
        gt, candidate = ground_truth[i], rendered[j]
        metrics = {
            "iou": box_iou(gt, candidate),
            "center_score": center_score(gt, candidate),
            "area_score": area_score(gt, candidate),
            "aspect_score": aspect_score(gt, candidate),
        }
        geometry = (0.35 * metrics["iou"] + 0.30 * metrics["center_score"]
                    + 0.25 * metrics["area_score"] + 0.10 * metrics["aspect_score"])
        geometry_total += geometry
        matches.append({
            "ground_truth_id": gt.detection_id,
            "rendered_id": candidate.detection_id,
            "ground_truth_class": gt.class_name,
            "rendered_class": candidate.class_name,
            "ground_truth_text": gt.text,
            "rendered_text": candidate.text,
            "ground_truth_box_normalized": {
                "x": gt.x, "y": gt.y, "w": gt.w, "h": gt.h,
            },
            "rendered_box_normalized": {
                "x": candidate.x, "y": candidate.y,
                "w": candidate.w, "h": candidate.h,
            },
            "match_score": match_score,
            "match_method": match_method,
            **metrics,
        })
    details = {
        "ground_truth_count": len(ground_truth),
        "rendered_count": len(rendered),
        "component_union_count": len(ground_truth) + len(rendered) - len(pairs),
        "matches": matches,
        "unmatched_ground_truth_ids": [box.detection_id for i, box in enumerate(ground_truth)
                                       if i not in matched_gt],
        "unmatched_rendered_ids": [box.detection_id for j, box in enumerate(rendered)
                                   if j not in matched_rendered],
    }
    return clamp01(geometry_total / details["component_union_count"]), details


def summarize_geometry_scores(details: dict) -> dict[str, float]:
    """Expose structure submetrics with the shared component-union penalty."""
    denominator = details["component_union_count"]
    if denominator == 0:
        return dict.fromkeys(
            ("iou_score", "center_score", "area_score", "aspect_score"),
            1.0,
        )
    return {
        "iou_score": sum(match["iou"] for match in details["matches"]) / denominator,
        "center_score": sum(match["center_score"] for match in details["matches"]) / denominator,
        "area_score": sum(match["area_score"] for match in details["matches"]) / denominator,
        "aspect_score": sum(match["aspect_score"] for match in details["matches"]) / denominator,
    }


def save_component_match_visualization(
    ground_truth_image: Path,
    rendered_image: Path,
    details: dict,
    output_path: Path,
) -> None:
    """Draw matched GPA components with shared colors in a side-by-side image."""
    max_panel_width = 960
    header_height = 64
    gap = 12
    panels = []
    source_sizes = []
    for image_path in (ground_truth_image, rendered_image):
        rgb = load_rgb(image_path)
        height, width = rgb.shape[:2]
        source_sizes.append((width, height))
        scale = min(1.0, max_panel_width / width)
        panel_width = max(1, round(width * scale))
        panel_height = max(1, round(height * scale))
        panels.append(cv2.resize(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            (panel_width, panel_height),
            interpolation=cv2.INTER_AREA,
        ))

    canvas_width = panels[0].shape[1] + gap + panels[1].shape[1]
    canvas_height = header_height + max(panel.shape[0] for panel in panels)
    canvas = np.full((canvas_height, canvas_width, 3), 32, dtype=np.uint8)
    panel_offsets = (0, panels[0].shape[1] + gap)
    for panel, x_offset, title in zip(
        panels, panel_offsets, ("GROUND TRUTH", "RENDERED")
    ):
        canvas[header_height:header_height + panel.shape[0],
               x_offset:x_offset + panel.shape[1]] = panel
        cv2.putText(canvas, title, (x_offset + 8, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Same color / M# = matched pair; red X = unmatched",
                    (x_offset + 8, 47), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (220, 220, 220), 1, cv2.LINE_AA)

    def draw_box(side: int, box: dict, label: str, color: tuple[int, int, int]) -> None:
        panel = panels[side]
        width, height = panel.shape[1], panel.shape[0]
        x1 = max(0, min(width - 1, round(box["x"] * width)))
        y1 = max(0, min(height - 1, round(box["y"] * height)))
        x2 = max(x1, min(width - 1, round((box["x"] + box["w"]) * width)))
        y2 = max(y1, min(height - 1, round((box["y"] + box["h"]) * height)))
        x_offset = panel_offsets[side]
        top = header_height
        cv2.rectangle(canvas, (x_offset + x1, top + y1),
                      (x_offset + x2, top + y2), color, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size, baseline = cv2.getTextSize(label, font, 0.45, 1)
        label_x = x_offset + x1
        label_y = top + y1 - 4
        if label_y - text_size[1] < top:
            label_y = min(top + height - 2, top + y1 + text_size[1] + 4)
        label_x = min(label_x, x_offset + width - text_size[0] - 4)
        label_x = max(x_offset, label_x)
        cv2.rectangle(canvas,
                      (label_x, label_y - text_size[1] - 2),
                      (label_x + text_size[0] + 3, label_y + baseline + 1),
                      (0, 0, 0), -1)
        cv2.putText(canvas, label, (label_x + 1, label_y), font,
                    0.45, color, 1, cv2.LINE_AA)

    for pair_number, match in enumerate(details["matches"], start=1):
        red, green, blue = colorsys.hsv_to_rgb(
            (0.08 + (pair_number - 1) * 0.61803398875) % 1.0, 0.85, 1.0
        )
        color = (round(blue * 255), round(green * 255), round(red * 255))
        label = f"M{pair_number:02d}"
        draw_box(0, match["ground_truth_box_normalized"],
                 f"{label} #{match['ground_truth_id']}", color)
        draw_box(1, match["rendered_box_normalized"],
                 f"{label} #{match['rendered_id']}", color)

    unmatched_color = (0, 0, 255)
    for side, gpa_path, unmatched_key in (
        (0, details["ground_truth_gpa"], "unmatched_ground_truth_ids"),
        (1, details["rendered_gpa"], "unmatched_rendered_ids"),
    ):
        unmatched_ids = set(details[unmatched_key])
        if not unmatched_ids:
            continue
        payload = json.loads(Path(gpa_path).read_text(encoding="utf-8"))
        source_width, source_height = source_sizes[side]
        for detection in payload["detections"]:
            if int(detection["id"]) not in unmatched_ids:
                continue
            x1, y1, x2, y2 = map(float, detection["bbox_xyxy"])
            x1, x2 = max(0.0, min(x1, source_width)), max(0.0, min(x2, source_width))
            y1, y2 = max(0.0, min(y1, source_height)), max(0.0, min(y2, source_height))
            if x2 <= x1 or y2 <= y1:
                continue
            draw_box(side, {
                "x": x1 / source_width, "y": y1 / source_height,
                "w": (x2 - x1) / source_width,
                "h": (y2 - y1) / source_height,
            }, f"X #{detection['id']}", unmatched_color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"Could not write component match visualization: {output_path}")


def hsv_histogram(rgb: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, [18, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = hist.astype(np.float32).reshape(-1)
    total = float(hist.sum())
    return hist / total if total else hist


def hist_intersection(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.minimum(left, right).sum())


def masked_hist_score(left_rgb: np.ndarray, left_mask: np.ndarray, right_rgb: np.ndarray, right_mask: np.ndarray) -> float:
    left_has_pixels = bool(np.any(left_mask))
    right_has_pixels = bool(np.any(right_mask))
    if not left_has_pixels and not right_has_pixels:
        return 1.0
    if not left_has_pixels or not right_has_pixels:
        return 0.0
    return hist_intersection(hsv_histogram(left_rgb, left_mask), hsv_histogram(right_rgb, right_mask))


def crop_normalized_box(rgb: np.ndarray, box: dict[str, float]) -> np.ndarray:
    """Crop a normalized GPA component box from a frame."""
    height, width = rgb.shape[:2]
    x1 = max(0, min(width - 1, round(box["x"] * width)))
    y1 = max(0, min(height - 1, round(box["y"] * height)))
    x2 = max(x1 + 1, min(width, round((box["x"] + box["w"]) * width)))
    y2 = max(y1 + 1, min(height, round((box["y"] + box["h"]) * height)))
    return rgb[y1:y2, x1:x2]


def score_foreground_components(
    matches: list[dict], ground_truth_count: int, rendered_count: int,
    gt_rgb: np.ndarray, rendered_rgb: np.ndarray,
) -> tuple[float, list[dict]]:
    """Compare matched box histograms; unmatched boxes score zero in the union."""
    union_count = ground_truth_count + rendered_count - len(matches)
    if union_count == 0:
        return 1.0, []
    color_matches = []
    total = 0.0
    for match in matches:
        gt_crop = crop_normalized_box(gt_rgb, match["ground_truth_box_normalized"])
        rendered_crop = crop_normalized_box(
            rendered_rgb, match["rendered_box_normalized"]
        )
        score = clamp01(hist_intersection(
            hsv_histogram(gt_crop), hsv_histogram(rendered_crop)
        ))
        total += score
        color_matches.append({
            "ground_truth_id": match["ground_truth_id"],
            "rendered_id": match["rendered_id"],
            "score": score,
        })
    return clamp01(total / union_count), color_matches


def match_boxes(left_boxes: list[Box], right_boxes: list[Box], text_aware: bool = False, include_boxes: bool = True) -> tuple[float, list[dict]]:
    if not left_boxes and not right_boxes:
        return 1.0, []
    if not left_boxes or not right_boxes:
        return 0.0, []

    matrix = np.zeros((len(left_boxes), len(right_boxes)), dtype=np.float32)
    for i, left in enumerate(left_boxes):
        for j, right in enumerate(right_boxes):
            geometry = 0.35 * box_iou(left, right) + 0.30 * center_score(left, right) + 0.25 * area_score(left, right) + 0.10 * aspect_score(left, right)
            if text_aware:
                text = SequenceMatcher(None, left.text, right.text).ratio()
                geometry = 0.55 * text + 0.45 * geometry
            matrix[i, j] = geometry

    row_ind, col_ind = linear_sum_assignment(-matrix)
    matches = []
    score_sum = 0.0
    for row, col in zip(row_ind, col_ind):
        score = float(matrix[row, col])
        if score < (0.25 if text_aware else 0.18):
            continue
        left = left_boxes[row]
        right = right_boxes[col]
        score_sum += score
        match = {
            "left_index": int(row),
            "right_index": int(col),
            "score": score,
            "iou": box_iou(left, right),
            "center_score": center_score(left, right),
            "area_score": area_score(left, right),
        }
        if include_boxes:
            match.update(
                {
                    "rendered_box": {
                        "x": left.x,
                        "y": left.y,
                        "w": left.w,
                        "h": left.h,
                    },
                    "ground_truth_box": {
                        "x": right.x,
                        "y": right.y,
                        "w": right.w,
                        "h": right.h,
                    },
                }
            )
        if text_aware:
            match.update({"rendered_text": left.text, "ground_truth_text": right.text})
        matches.append(match)
    return score_sum / max(len(left_boxes), len(right_boxes)), matches


def text_levenshtein_similarity(reference: TextResult, candidate: TextResult) -> float:
    """Score whole-frame OCR text by normalized Levenshtein edit distance."""
    from rapidfuzz.distance import Levenshtein

    reference_text = " ".join(reference.text.split())
    candidate_text = " ".join(candidate.text.split())
    if not reference_text and not candidate_text:
        return 1.0
    if not reference_text or not candidate_text:
        return 0.0
    distance = Levenshtein.distance(reference_text, candidate_text)
    return 1.0 - distance / max(len(reference_text), len(candidate_text))


def weighted_final_score(scores: dict[str, float | None], weights: dict[str, float]) -> float:
    available = [(name, score) for name, score in scores.items() if score is not None]
    total_weight = sum(weights[name] for name, _ in available)
    if total_weight == 0:
        return 0.0
    return sum(weights[name] * float(score) for name, score in available) / total_weight


def format_visual_matches(matches: list[dict]) -> list[dict]:
    formatted = []
    for match in matches:
        formatted.append(
            {
                "score": match["score"],
                "iou": match["iou"],
                "center_score": match["center_score"],
                "area_score": match["area_score"],
                "rendered_box": match["rendered_box"],
                "ground_truth_box": match["ground_truth_box"],
            }
        )
    return formatted


def evaluate_pair(
    rendered_image: Path,
    ground_truth_image: Path,
    rendered_gpa_dir: Path,
    ground_truth_gpa_dir: Path,
    rendered_frame_index: int,
    ground_truth_frame_index: int,
) -> dict:
    rendered_rgb = load_rgb(rendered_image)
    gt_original_rgb = load_rgb(ground_truth_image)

    rendered_text_result = extract_text(rendered_rgb)
    gt_text_result = extract_text(gt_original_rgb)
    text_score = text_levenshtein_similarity(gt_text_result, rendered_text_result)
    rendered_components, rendered_gpa_path = load_gpa_components(
        rendered_gpa_dir, rendered_frame_index, rendered_rgb
    )
    gt_components, gt_gpa_path = load_gpa_components(
        ground_truth_gpa_dir, ground_truth_frame_index, gt_original_rgb
    )
    attach_ocr_text(rendered_components, rendered_text_result.boxes)
    attach_ocr_text(gt_components, gt_text_result.boxes)
    structure_score, structure_details = match_gpa_components(
        gt_components, rendered_components
    )
    geometry_scores = summarize_geometry_scores(structure_details)
    gt_component_mask = gpa_component_mask(gt_original_rgb, gt_components)
    rendered_component_mask = gpa_component_mask(rendered_rgb, rendered_components)
    gt_background = page_background_color(gt_original_rgb, gt_component_mask)
    rendered_background = page_background_color(rendered_rgb, rendered_component_mask)
    background_color_score = (
        color_similarity_ciede2000(gt_background, rendered_background)
        if gt_background is not None and rendered_background is not None else 0.0
    )
    gt_small = resize_for_analysis(gt_original_rgb, DEFAULT_MAX_WIDTH)
    rendered_small = resize_for_analysis(rendered_rgb, DEFAULT_MAX_WIDTH)
    foreground_color_score, foreground_matches = score_foreground_components(
        structure_details["matches"], len(gt_components), len(rendered_components),
        gt_small, rendered_small,
    )
    color_score = clamp01((background_color_score + foreground_color_score) / 2)

    clip_score, _ = clip_similarity(rendered_image, ground_truth_image)

    return {
        "structure_score": clamp01(structure_score),
        **geometry_scores,
        "structure_details": {
            "ground_truth_gpa": str(gt_gpa_path),
            "rendered_gpa": str(rendered_gpa_path),
            **structure_details,
        },
        "text_score": text_score,
        "color_score": color_score,
        "background_color_score": clamp01(background_color_score),
        "foreground_color_score": clamp01(foreground_color_score),
        "color_details": {
            "ground_truth_background_rgb": gt_background,
            "rendered_background_rgb": rendered_background,
            "foreground_union_count": len(gt_components) + len(rendered_components) - len(foreground_matches),
            "foreground_matches": foreground_matches,
        },
        "clip_score": clamp01(clip_score),
    }


def load_match_payload(matches_path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(matches_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "matches" in payload:
        matches = payload["matches"]
        if not isinstance(matches, list):
            raise ValueError(f"'matches' must be a list in {matches_path}")
        return payload, matches
    if isinstance(payload, list):
        return {}, payload
    raise ValueError(f"Unsupported keyframe match format: {matches_path}")


def resolve_image_path(raw_path: str, matches_path: Path) -> Path:
    path = Path(raw_path).expanduser()
    candidates = [path] if path.is_absolute() else [
        Path.cwd() / path,
        matches_path.parent / path,
        PROJECT_ROOT / path,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    attempted = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Matched image not found: {raw_path}. Tried: {attempted}")


def load_ground_truth_images(ground_truth_dir: Path) -> dict[int, Path]:
    metadata_path = ground_truth_dir / "keyframes.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Shared ground-truth keyframes not found: {metadata_path}. "
            "Run tools/keyframe.py for this id first."
        )
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    images = {}
    for item in payload.get("keyframes", []):
        order = int(item["order"])
        image_path = resolve_image_path(str(item["image_path"]), metadata_path)
        images[order] = image_path
    if not images:
        raise ValueError(f"No keyframes recorded in {metadata_path}")
    return images


def match_value(match: dict, primary: str, fallback: str | None = None):
    value = match.get(primary)
    if value is None and fallback is not None:
        value = match.get(fallback)
    return value


def average_available(
    pair_results: list[dict],
    key: str,
    unmatched_ground_truth_count: int = 0,
) -> float | None:
    """Average a metric, treating every unmatched GT keyframe as a zero.

    A ``None`` value on a matched pair still means that the metric itself was
    unavailable (for example, OCR could not run), so it is omitted just as it
    was previously.  Unmatched ground-truth frames are different: the rendered
    animation failed to produce that state and must therefore contribute 0.
    """
    values = [item[key] for item in pair_results if item[key] is not None]
    if not values:
        return 0.0 if unmatched_ground_truth_count else None
    denominator = len(values) + unmatched_ground_truth_count
    return float(sum(values) / denominator)


def ground_truth_keyframe_count(
    source_payload: dict,
    shared_ground_truth: dict[int, Path] | None,
    matched_count: int,
) -> int:
    """Find the GT keyframe count used as the aggregate-score denominator."""
    if shared_ground_truth is not None:
        count = len(shared_ground_truth)
    else:
        summary = source_payload.get("summary")
        summary = summary if isinstance(summary, dict) else {}
        count = None
        for key in ("reference_keyframe_count", "ground_truth_count"):
            if summary.get(key) is not None:
                count = int(summary[key])
                break
        if count is None:
            # Older alignment files may only expose matched/unmatched counts.
            for unmatched_key in (
                "unmatched_reference_count",
                "unmatched_ground_truth_count",
            ):
                if summary.get(unmatched_key) is not None:
                    count = matched_count + int(summary[unmatched_key])
                    break
        if count is None:
            # With a bare list of matches there is no information about omitted
            # GT frames, so retain the historical matched-pairs-only behavior.
            count = matched_count

    if count < matched_count:
        raise ValueError(
            f"Ground-truth keyframe count ({count}) is smaller than matched "
            f"pair count ({matched_count})"
        )
    return count


def score_matches(
    matches_path: Path,
    output_path: Path | None,
    ground_truth_dir: Path | None = None,
    ground_truth_gpa_dir: Path | None = None,
    rendered_gpa_dir: Path | None = None,
    visualize_matches: bool = False,
) -> dict:
    source_payload, matches = load_match_payload(matches_path)
    shared_ground_truth = (
        load_ground_truth_images(ground_truth_dir) if ground_truth_dir else None
    )
    pair_results = []
    for index, match in enumerate(matches):
        reference_image_value = match_value(match, "ground_truth_image", "reference_image")
        rendered_image_value = match.get("rendered_image")
        if not rendered_image_value or (
            shared_ground_truth is None and not reference_image_value
        ):
            raise ValueError(
                f"Match {index} must contain reference_image/ground_truth_image "
                "and rendered_image"
            )
        if shared_ground_truth is not None:
            ground_truth_order = match_value(
                match, "ground_truth_order", "reference_order"
            )
            if ground_truth_order is None:
                raise ValueError(
                    f"Match {index} has no ground_truth_order/reference_order"
                )
            try:
                gt_image = shared_ground_truth[int(ground_truth_order)]
            except KeyError as exc:
                raise ValueError(
                    f"Ground-truth order {ground_truth_order} from match {index} "
                    f"does not exist in {ground_truth_dir / 'keyframes.json'}"
                ) from exc
        else:
            gt_image = resolve_image_path(str(reference_image_value), matches_path)
        rendered_image = resolve_image_path(str(rendered_image_value), matches_path)
        gt_frame_index = match_value(
            match, "ground_truth_frame_index", "reference_frame_index"
        )
        rendered_frame_index = match.get("rendered_frame_index")
        if gt_frame_index is None or rendered_frame_index is None:
            raise ValueError(
                f"Match {index} needs ground_truth_frame_index/reference_frame_index "
                "and rendered_frame_index to locate GPA results"
            )
        pair_rendered_gpa = rendered_gpa_dir or matches_path.parent.parent / "gpa"
        pair_gt_gpa = ground_truth_gpa_dir or gt_image.parent.parent / "gpa"
        validate_gpa_video(
            pair_rendered_gpa,
            matches_path.parent / "rendered_matched_frames" / "keyframes.json",
        )
        validate_gpa_video(
            pair_gt_gpa,
            (ground_truth_dir or gt_image.parent) / "keyframes.json",
        )
        result = evaluate_pair(
            rendered_image, gt_image,
            pair_rendered_gpa, pair_gt_gpa,
            int(rendered_frame_index), int(gt_frame_index),
        )
        if visualize_matches:
            visualization_root = (
                output_path.parent if output_path else matches_path.parent
            ) / "component_matches"
            visualization_path = visualization_root / f"pair_{index:03d}.png"
            save_component_match_visualization(
                gt_image, rendered_image,
                result["structure_details"], visualization_path,
            )
            result["component_match_visualization"] = str(visualization_path)
        pair_results.append(
            {
                "pair_index": index,
                "ground_truth_order": match_value(match, "ground_truth_order", "reference_order"),
                "rendered_order": match.get("rendered_order"),
                "ground_truth_frame_index": int(gt_frame_index),
                "rendered_frame_index": int(rendered_frame_index),
                "ground_truth_time_sec": match_value(
                    match, "ground_truth_time_sec", "reference_time_sec"
                ),
                "rendered_time_sec": match.get("rendered_time_sec"),
                "ground_truth_image": str(gt_image),
                "rendered_image": str(rendered_image),
                **result,
            }
        )

    total_ground_truth_count = ground_truth_keyframe_count(
        source_payload,
        shared_ground_truth,
        len(pair_results),
    )
    unmatched_ground_truth_count = total_ground_truth_count - len(pair_results)
    summary_keys = [
        "structure_score", "text_score", "color_score", "clip_score",
    ]
    summary = {
        "pair_count": len(pair_results),
        "ground_truth_keyframe_count": total_ground_truth_count,
        "matched_pair_count": len(pair_results),
        "unmatched_ground_truth_count": unmatched_ground_truth_count,
        **{
            key: average_available(
                pair_results,
                key,
                unmatched_ground_truth_count,
            )
            for key in summary_keys
        },
    }
    payload = {
        "summary": summary,
        "pairs": pair_results,
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score keyframe pairs produced by utils/keyframe_alignment.py without using HTML source."
    )
    parser.add_argument("--matches", type=Path, help="Path to keyframe_matches.json")
    parser.add_argument("--id", dest="sample_id", help="Dataset sample id")
    parser.add_argument("--run-name", default="gpt-6-astra")
    parser.add_argument("--generated-root", type=Path, default=DEFAULT_GENERATED_ROOT)
    parser.add_argument(
        "--ground-truth-root",
        type=Path,
        default=DEFAULT_GROUND_TRUTH_ROOT,
        help="Shared ground-truth keyframe cache root used by --id mode",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        help="Explicit shared ground-truth keyframe directory",
    )
    parser.add_argument(
        "--ground-truth-gpa-dir", type=Path,
        help="Override ground-truth GPA directory (default: beside ground-truth keyframes)",
    )
    parser.add_argument(
        "--rendered-gpa-dir", type=Path,
        help="Override rendered GPA directory (default: beside keyframe_alignment)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON output path; defaults to static_appearance_scores.json beside --matches",
    )
    parser.add_argument(
        "--visualize-matches", action="store_true",
        help="Save side-by-side colored GPA component matches in component_matches/ beside the output JSON",
    )
    return parser


def resolve_cli_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    cwd_candidate = (Path.cwd() / path).resolve()
    project_candidate = (PROJECT_ROOT / path).resolve()
    if cwd_candidate.exists() or not project_candidate.exists():
        return cwd_candidate
    return project_candidate


def resolve_cli_inputs(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path | None]:
    if args.matches and args.sample_id:
        raise ValueError("Use either --matches or --id, not both")
    if args.sample_id:
        run_dir = args.generated_root.expanduser().resolve() / str(args.sample_id) / args.run_name
        matches_path = run_dir / "keyframe_alignment" / "keyframe_matches.json"
        ground_truth_dir = (
            resolve_cli_path(args.ground_truth_dir.expanduser())
            if args.ground_truth_dir
            else (
                args.ground_truth_root.expanduser().resolve()
                / str(args.sample_id)
                / GROUND_TRUTH_KEYFRAMES_DIRNAME
            )
        )
    elif args.matches:
        matches_path = resolve_cli_path(args.matches.expanduser())
        ground_truth_dir = (
            resolve_cli_path(args.ground_truth_dir.expanduser())
            if args.ground_truth_dir
            else None
        )
    else:
        raise ValueError("Provide --matches, or use --id with --run-name")

    if not matches_path.is_file():
        raise FileNotFoundError(f"Keyframe matches not found: {matches_path}")
    output_path = (
        resolve_cli_path(args.output.expanduser())
        if args.output
        else matches_path.parent / "static_appearance_scores.json"
    )
    return matches_path, output_path, ground_truth_dir


def main() -> None:
    args = build_parser().parse_args()
    matches_path, output_path, ground_truth_dir = resolve_cli_inputs(args)
    payload = score_matches(
        matches_path,
        output_path,
        ground_truth_dir=ground_truth_dir,
        ground_truth_gpa_dir=(
            resolve_cli_path(args.ground_truth_gpa_dir.expanduser())
            if args.ground_truth_gpa_dir else None
        ),
        rendered_gpa_dir=(
            resolve_cli_path(args.rendered_gpa_dir.expanduser())
            if args.rendered_gpa_dir else None
        ),
        visualize_matches=args.visualize_matches,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
