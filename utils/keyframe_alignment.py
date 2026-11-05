#!/usr/bin/env python3
"""Align reference keyframes with frames in generated videos."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


UTILS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = UTILS_DIR.parent
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from keyframe_extraction import (  # noqa: E402
    DEFAULT_CHANGE_WINDOW_SEC,
    DEFAULT_MAX_KEYFRAMES,
    DEFAULT_MIN_KEYFRAME_GAP_SEC,
    DEFAULT_MIN_STABLE_SEC,
    get_reference_keyframes,
    load_video_info,
    require_binary,
    save_keyframe,
)



DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data"
DEFAULT_GENERATED_ROOT = PROJECT_ROOT / "experiment"
DEFAULT_GROUND_TRUTH_ROOT = PROJECT_ROOT / "experiment" / "ground_truth"
GROUND_TRUTH_KEYFRAMES_DIRNAME = "keyframes"
DEFAULT_TOP_CANDIDATES = 5
DEFAULT_OWNER_MARGIN = 0.01


DEFAULT_RESIZE = 96
DEFAULT_VISUAL_WEIGHT = 0.85
DEFAULT_TIME_WEIGHT = 0.15
DEFAULT_MATCH_THRESHOLD = 0.20
DEFAULT_SKIP_PENALTY = 0.12


@dataclass
class Keyframe:
    order: int
    frame_index: int
    time_sec: float
    image_path: Path




def load_image_rgb(image_path: Path, resize: int) -> np.ndarray:
    image = Image.open(image_path).convert("RGB").resize((resize, resize), Image.LANCZOS)
    return np.asarray(image, dtype=np.float32)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return vector
    return vector / norm


def histogram_intersection(hist1: np.ndarray, hist2: np.ndarray) -> float:
    return float(np.minimum(hist1, hist2).sum())


def summarize_mask(mask: np.ndarray) -> dict:
    total_pixels = mask.shape[0] * mask.shape[1]
    area_ratio = float(mask.sum() / total_pixels)
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

    components = []
    for index in range(1, num_labels):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < 5:
            continue
        left = int(stats[index, cv2.CC_STAT_LEFT])
        top = int(stats[index, cv2.CC_STAT_TOP])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        centroid_x, centroid_y = centroids[index]
        components.append(
            {
                "area_ratio": area / total_pixels,
                "bbox_area_ratio": (width * height) / total_pixels,
                "center_x": float(centroid_x / mask.shape[1]),
                "center_y": float(centroid_y / mask.shape[0]),
            }
        )

    components.sort(key=lambda item: item["area_ratio"], reverse=True)
    return {
        "area_ratio": area_ratio,
        "count": len(components),
        "components": components[:5],
    }


def compare_mask_summaries(left: dict, right: dict) -> float:
    area_score = max(0.0, 1.0 - abs(left["area_ratio"] - right["area_ratio"]) * 12.0)

    left_count = left["count"]
    right_count = right["count"]
    count_denominator = max(left_count, right_count, 1)
    count_score = 1.0 - abs(left_count - right_count) / count_denominator

    if left["components"] and right["components"]:
        component_scores = []
        pair_count = min(len(left["components"]), len(right["components"]), 3)
        for index in range(pair_count):
            left_component = left["components"][index]
            right_component = right["components"][index]
            center_score = 1.0 - max(
                abs(left_component["center_x"] - right_component["center_x"]),
                abs(left_component["center_y"] - right_component["center_y"]),
            )
            component_area_score = max(
                0.0,
                1.0 - abs(left_component["area_ratio"] - right_component["area_ratio"]) * 20.0,
            )
            component_scores.append(0.6 * center_score + 0.4 * component_area_score)
        layout_score = float(np.mean(component_scores))
    else:
        layout_score = 1.0 if left_count == 0 and right_count == 0 else 0.0

    return 0.35 * area_score + 0.25 * count_score + 0.40 * layout_score


def extract_feature(image_path: Path, resize: int) -> dict:
    rgb = load_image_rgb(image_path, resize)
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    edges = cv2.Canny(gray, 100, 200)

    white_mask = ((gray > 180) & (hsv[:, :, 1] < 40)).astype(np.uint8)
    color_mask = ((gray > 40) & (hsv[:, :, 1] > 50)).astype(np.uint8)
    foreground_mask = (gray > 25).astype(np.uint8)

    rgb_vec = normalize_vector((rgb / 255.0).reshape(-1))
    edge_vec = normalize_vector((edges.astype(np.float32) / 255.0).reshape(-1))

    hist_channels = []
    for channel in range(3):
        hist, _ = np.histogram(
            rgb[:, :, channel],
            bins=32,
            range=(0, 256),
            density=False,
        )
        hist = hist.astype(np.float32)
        hist_sum = float(hist.sum())
        hist_channels.append(hist / hist_sum if hist_sum else hist)

    return {
        "rgb_image": rgb / 255.0,
        "gray_image": gray.astype(np.float32) / 255.0,
        "edge_image": edges.astype(np.float32) / 255.0,
        "white_summary": summarize_mask(white_mask),
        "color_summary": summarize_mask(color_mask),
        "foreground_summary": summarize_mask(foreground_mask),
        "rgb_vec": rgb_vec,
        "edge_vec": edge_vec,
        "rgb_hist": np.concatenate(hist_channels),
    }


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0:
        return 0.0
    return float(np.dot(left, right) / denom)


def compute_visual_similarity(feature_a: dict, feature_b: dict) -> dict:
    rgb_score = max(0.0, cosine_similarity(feature_a["rgb_vec"], feature_b["rgb_vec"]))

    if feature_a["edge_image"].sum() == 0 and feature_b["edge_image"].sum() == 0:
        edge_score = 1.0
    else:
        edge_score = max(0.0, cosine_similarity(feature_a["edge_vec"], feature_b["edge_vec"]))

    hist_score = histogram_intersection(feature_a["rgb_hist"], feature_b["rgb_hist"]) / 3.0
    pixel_score = 1.0 - float(np.mean(np.abs(feature_a["rgb_image"] - feature_b["rgb_image"])))
    gray_score = 1.0 - float(np.mean(np.abs(feature_a["gray_image"] - feature_b["gray_image"])))
    white_structure_score = compare_mask_summaries(
        feature_a["white_summary"], feature_b["white_summary"]
    )
    color_structure_score = compare_mask_summaries(
        feature_a["color_summary"], feature_b["color_summary"]
    )
    foreground_structure_score = compare_mask_summaries(
        feature_a["foreground_summary"], feature_b["foreground_summary"]
    )
    structure_score = (
        0.20 * white_structure_score
        + 0.50 * color_structure_score
        + 0.30 * foreground_structure_score
    )
    score = (
        0.15 * pixel_score
        + 0.10 * gray_score
        + 0.10 * hist_score
        + 0.10 * edge_score
        + 0.10 * rgb_score
        + 0.45 * structure_score
    )
    return {
        "score": score,
        "pixel_score": pixel_score,
        "gray_score": gray_score,
        "structure_score": structure_score,
        "white_structure_score": white_structure_score,
        "color_structure_score": color_structure_score,
        "foreground_structure_score": foreground_structure_score,
        "rgb_score": rgb_score,
        "edge_score": edge_score,
        "hist_score": hist_score,
    }


def extract_feature_from_rgb(rgb: np.ndarray, resize: int) -> dict:
    """Equivalent to extract_feature, without a temporary JPG."""
    resized = Image.fromarray(rgb).resize((resize, resize), Image.Resampling.LANCZOS)
    rgb_float = np.asarray(resized, dtype=np.float32)
    rgb_uint8 = rgb_float.astype(np.uint8)
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)
    edges = cv2.Canny(gray, 100, 200)

    white_mask = ((gray > 180) & (hsv[:, :, 1] < 40)).astype(np.uint8)
    color_mask = ((gray > 40) & (hsv[:, :, 1] > 50)).astype(np.uint8)
    foreground_mask = (gray > 25).astype(np.uint8)

    hist_channels = []
    for channel in range(3):
        hist, _ = np.histogram(rgb_float[:, :, channel], bins=32, range=(0, 256))
        hist = hist.astype(np.float32)
        hist_sum = float(hist.sum())
        hist_channels.append(hist / hist_sum if hist_sum else hist)

    rgb_unit = rgb_float / 255.0
    edge_unit = edges.astype(np.float32) / 255.0
    return {
        "rgb_image": rgb_unit,
        "gray_image": gray.astype(np.float32) / 255.0,
        "edge_image": edge_unit,
        "white_summary": summarize_mask(white_mask),
        "color_summary": summarize_mask(color_mask),
        "foreground_summary": summarize_mask(foreground_mask),
        "rgb_vec": normalize_vector(rgb_unit.reshape(-1)),
        "edge_vec": normalize_vector(edge_unit.reshape(-1)),
        "rgb_hist": np.concatenate(hist_channels),
    }


def reference_keyframes_from_payload(payload: dict) -> list[Keyframe]:
    return [
        Keyframe(
            order=int(item["order"]),
            frame_index=int(item["frame_index"]),
            time_sec=float(item["time_sec"]),
            image_path=Path(item["image_path"]),
        )
        for item in payload["keyframes"]
    ]


def score_rendered_video(
    rendered_video: Path,
    reference_keyframes: list[Keyframe],
    reference_duration: float,
    resize: int,
    visual_weight: float,
    time_weight: float,
) -> tuple[list[list[dict]], np.ndarray, int]:
    """Return per-pair details and scores for reference keyframes x all frames."""
    rendered_info = load_video_info(rendered_video)
    reference_features = [
        extract_feature(frame.image_path, resize) for frame in reference_keyframes
    ]
    rows: list[list[dict]] = [[] for _ in reference_keyframes]

    capture = cv2.VideoCapture(str(rendered_video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open rendered video: {rendered_video}")

    frame_index = 0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            rendered_feature = extract_feature_from_rgb(
                cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), resize
            )
            rendered_time = frame_index / rendered_info.fps
            rendered_normalized_time = min(
                1.0, rendered_time / max(rendered_info.duration, 1e-12)
            )
            for row, reference, reference_feature in zip(
                rows, reference_keyframes, reference_features
            ):
                visual = compute_visual_similarity(reference_feature, rendered_feature)
                reference_normalized_time = min(
                    1.0, reference.time_sec / max(reference_duration, 1e-12)
                )
                time_score = 1.0 - abs(
                    reference_normalized_time - rendered_normalized_time
                )
                total_score = visual_weight * visual["score"] + time_weight * time_score
                row.append(
                    {
                        "rendered_frame_index": frame_index,
                        "rendered_time_sec": round(rendered_time, 6),
                        "score": float(total_score),
                        "visual_score": float(visual["score"]),
                        "time_score": float(time_score),
                        "pixel_score": float(visual["pixel_score"]),
                        "gray_score": float(visual["gray_score"]),
                        "structure_score": float(visual["structure_score"]),
                        "white_structure_score": float(visual["white_structure_score"]),
                        "color_structure_score": float(visual["color_structure_score"]),
                        "foreground_structure_score": float(
                            visual["foreground_structure_score"]
                        ),
                        "rgb_score": float(visual["rgb_score"]),
                        "edge_score": float(visual["edge_score"]),
                        "hist_score": float(visual["hist_score"]),
                    }
                )
            frame_index += 1
    finally:
        capture.release()

    if frame_index == 0:
        raise ValueError(f"Rendered video contains no decodable frames: {rendered_video}")
    scores = np.asarray(
        [[candidate["score"] for candidate in row] for row in rows],
        dtype=np.float32,
    )
    return rows, scores, frame_index


def build_rendered_match_eligibility(
    details: list[list[dict]], owner_margin: float
) -> tuple[np.ndarray, np.ndarray]:
    """Build visually competitive reference candidates for rendered frames.

    The best reference is recorded for diagnostics.  References within
    ``owner_margin`` of that best visual score remain eligible for global
    alignment.  Keeping near ties is important when adjacent reference states
    differ only in a small region: a global descriptor can rank the incomplete
    state ahead by a tiny, insignificant amount even after the new region has
    appeared.  The alignment still consumes each rendered frame at most once
    and enforces strict temporal order.

    Ownership eligibility uses ``visual_score`` rather than the combined
    visual/time score, so timing cannot turn an unrelated reference into a
    candidate.
    """
    if owner_margin < 0:
        raise ValueError("owner_margin cannot be negative")
    if not details:
        return np.empty(0, dtype=np.int64), np.zeros((0, 0), dtype=bool)

    rendered_count = len(details[0])
    if any(len(row) != rendered_count for row in details):
        raise ValueError("All reference candidate rows must have the same length")
    if rendered_count == 0:
        return np.empty(0, dtype=np.int64), np.zeros((len(details), 0), dtype=bool)

    visual_scores = np.asarray(
        [[candidate["visual_score"] for candidate in row] for row in details],
        dtype=np.float32,
    )
    best_scores = np.max(visual_scores, axis=0)
    owners = np.argmax(visual_scores, axis=0).astype(np.int64)
    allowed = visual_scores >= (best_scores[np.newaxis, :] - owner_margin)

    # Persist competition metadata so rejected matches are explainable in the
    # output without recomputing the full matrix.
    for reference_index, row in enumerate(details):
        for rendered_index, candidate in enumerate(row):
            candidate["best_reference_index"] = int(owners[rendered_index])
            candidate["visual_score_gap_from_best"] = float(
                best_scores[rendered_index]
                - visual_scores[reference_index, rendered_index]
            )
            candidate["within_owner_margin"] = bool(
                allowed[reference_index, rendered_index]
            )
    return owners, allowed


def align_to_all_frames(
    scores: np.ndarray,
    match_threshold: float,
    skip_penalty: float,
    allowed_matches: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Choose eligible matches with strictly increasing rendered indices.

    Skipping a rendered frame is free because all rendered frames are merely
    candidates.  Skipping a reference state costs ``skip_penalty``.

    ``allowed_matches`` can additionally restrict which visually competitive
    references may consume a rendered frame.
    """
    reference_count, rendered_count = scores.shape
    if allowed_matches is None:
        allowed_matches = np.ones(scores.shape, dtype=bool)
    elif allowed_matches.shape != scores.shape:
        raise ValueError("allowed_matches must have the same shape as scores")

    dp = np.full((reference_count + 1, rendered_count + 1), -np.inf, dtype=np.float64)
    back: list[list[tuple[str, int, int] | None]] = [
        [None] * (rendered_count + 1) for _ in range(reference_count + 1)
    ]
    dp[0, 0] = 0.0

    for i in range(reference_count + 1):
        for j in range(rendered_count + 1):
            current = dp[i, j]
            if np.isneginf(current):
                continue
            if j < rendered_count and current > dp[i, j + 1]:
                dp[i, j + 1] = current
                back[i][j + 1] = ("skip_rendered", i, j)
            if i < reference_count:
                skipped = current - skip_penalty
                if skipped > dp[i + 1, j]:
                    dp[i + 1, j] = skipped
                    back[i + 1][j] = ("skip_reference", i, j)
            if (
                i < reference_count
                and j < rendered_count
                and allowed_matches[i, j]
                and scores[i, j] >= match_threshold
            ):
                matched = current + float(scores[i, j]) - match_threshold
                if matched > dp[i + 1, j + 1]:
                    dp[i + 1, j + 1] = matched
                    back[i + 1][j + 1] = ("match", i, j)

    matches: list[tuple[int, int]] = []
    i, j = reference_count, rendered_count
    while i or j:
        step = back[i][j]
        if step is None:
            raise RuntimeError("Failed to reconstruct frame alignment")
        action, previous_i, previous_j = step
        if action == "match":
            matches.append((previous_i, previous_j))
        i, j = previous_i, previous_j
    matches.reverse()
    return matches


def validate_strict_alignment(matches: list[tuple[int, int]]) -> None:
    """Fail loudly if an alignment ever violates reference or video order."""
    for previous, current in zip(matches, matches[1:]):
        previous_reference, previous_rendered = previous
        current_reference, current_rendered = current
        if not (
            previous_reference < current_reference
            and previous_rendered < current_rendered
        ):
            raise RuntimeError(
                "Alignment must be strictly increasing in both reference and "
                "rendered frame order"
            )


def clean_keyframe_images(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for image_path in directory.glob("keyframe_*.jpg"):
        image_path.unlink()


def compact_candidate(candidate: dict) -> dict:
    return {key: candidate[key] for key in candidate}


def align_videos(
    reference_video: Path,
    rendered_video: Path,
    output_dir: Path,
    min_stable_sec: float,
    change_window_sec: float,
    min_keyframe_gap_sec: float,
    max_keyframes: int,
    resize: int,
    visual_weight: float,
    time_weight: float,
    match_threshold: float,
    skip_penalty: float,
    top_candidates: int,
    owner_margin: float,
    reference_dir: Path | None = None,
    refresh_reference: bool = False,
) -> dict:
    require_binary("ffmpeg")
    require_binary("ffprobe")
    if not reference_video.is_file():
        raise FileNotFoundError(f"Reference video not found: {reference_video}")
    if not rendered_video.is_file():
        raise FileNotFoundError(f"Rendered video not found: {rendered_video}")

    reference_dir = reference_dir or output_dir / "reference_keyframes"
    rendered_dir = output_dir / "rendered_matched_frames"
    reference_payload, reference_cache_reused = get_reference_keyframes(
        reference_video,
        reference_dir,
        min_stable_sec,
        change_window_sec,
        min_keyframe_gap_sec,
        max_keyframes,
        refresh_reference,
    )
    reference_keyframes = reference_keyframes_from_payload(reference_payload)
    details, scores, rendered_frame_count = score_rendered_video(
        rendered_video,
        reference_keyframes,
        float(reference_payload["video_info"]["duration_sec"]),
        resize,
        visual_weight,
        time_weight,
    )
    _, allowed_matches = build_rendered_match_eligibility(details, owner_margin)
    matched_pairs = align_to_all_frames(
        scores,
        match_threshold,
        skip_penalty,
        allowed_matches=allowed_matches,
    )
    validate_strict_alignment(matched_pairs)
    match_by_reference = {reference_index: frame_index for reference_index, frame_index in matched_pairs}

    clean_keyframe_images(rendered_dir)
    matches = []
    unmatched_reference = []
    rendered_keyframes_payload = []
    for reference_index, reference in enumerate(reference_keyframes):
        ranked = sorted(details[reference_index], key=lambda item: item["score"], reverse=True)
        top = [compact_candidate(item) for item in ranked[:top_candidates]]
        if reference_index not in match_by_reference:
            best = compact_candidate(ranked[0])
            owned_candidates = [
                candidate
                for candidate in details[reference_index]
                if candidate["within_owner_margin"]
                and candidate["score"] >= match_threshold
            ]
            unmatched_reference.append(
                {
                    "reference_order": reference.order,
                    "reference_frame_index": reference.frame_index,
                    "reference_time_sec": reference.time_sec,
                    "reference_image": str(reference.image_path),
                    "ground_truth_order": reference.order,
                    "ground_truth_frame_index": reference.frame_index,
                    "ground_truth_time_sec": reference.time_sec,
                    "ground_truth_image": str(reference.image_path),
                    "reason": (
                        "no_candidate_reached_threshold"
                        if best["score"] < match_threshold
                        else (
                            "no_rendered_frame_within_visual_owner_margin"
                            if not owned_candidates
                            else "skipped_by_order_preserving_alignment"
                        )
                    ),
                    "best_unconstrained_candidate": best,
                    "top_candidates": top,
                }
            )
            continue

        rendered_frame_index = match_by_reference[reference_index]
        candidate = details[reference_index][rendered_frame_index]
        rendered_time = float(candidate["rendered_time_sec"])
        filename = (
            f"keyframe_{reference.order:02d}_frame_{rendered_frame_index:05d}_"
            f"{rendered_time:06.3f}s.jpg"
        )
        rendered_image = rendered_dir / filename
        save_keyframe(rendered_video, rendered_frame_index, rendered_image)
        item = {
            "reference_order": reference.order,
            "reference_frame_index": reference.frame_index,
            "reference_time_sec": reference.time_sec,
            "reference_image": str(reference.image_path),
            # Keep the established output field names so the result
            # can feed static_appearance_score.py and animation_segment_score.py.
            "ground_truth_order": reference.order,
            "ground_truth_frame_index": reference.frame_index,
            "ground_truth_time_sec": reference.time_sec,
            "ground_truth_image": str(reference.image_path),
            "rendered_order": len(matches),
            **candidate,
            "rendered_image": str(rendered_image),
            "top_candidates": top,
        }
        matches.append(item)
        rendered_keyframes_payload.append(
            {
                "order": reference.order,
                "frame_index": rendered_frame_index,
                "time_sec": round(rendered_time, 6),
                "matched_reference_frame_index": reference.frame_index,
                "score": candidate["score"],
                "image_path": str(rendered_image),
            }
        )

    rendered_info = load_video_info(rendered_video)
    rendered_payload = {
        "video_path": str(rendered_video),
        "video_info": {
            "width": rendered_info.width,
            "height": rendered_info.height,
            "fps": rendered_info.fps,
            "duration_sec": rendered_info.duration,
            "total_frames": rendered_info.total_frames,
            "analyzed_frames": rendered_frame_count,
        },
        "keyframes": rendered_keyframes_payload,
    }
    (rendered_dir / "keyframes.json").write_text(
        json.dumps(rendered_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    result = {
        "reference_video": str(reference_video),
        "rendered_video": str(rendered_video),
        "reference_keyframes_dir": str(reference_dir),
        "rendered_matched_frames_dir": str(rendered_dir),
        "ground_truth_dir": str(reference_dir),
        "rendered_dir": str(rendered_dir),
        "parameters": {
            "min_stable_sec": min_stable_sec,
            "change_window_sec": change_window_sec,
            "min_keyframe_gap_sec": min_keyframe_gap_sec,
            "max_keyframes": max_keyframes,
            "resize": resize,
            "visual_weight": visual_weight,
            "time_weight": time_weight,
            "match_threshold": match_threshold,
            "skip_penalty": skip_penalty,
            "top_candidates": top_candidates,
            "strictly_increasing_rendered_frames": True,
            "rendered_frame_single_final_match": True,
            "rendered_frame_owner_metric": "visual_score",
            "rendered_frame_owner_margin": owner_margin,
            "reference_cache_reused": reference_cache_reused,
        },
        "summary": {
            "reference_keyframe_count": len(reference_keyframes),
            "reference_cache_reused": reference_cache_reused,
            "rendered_frames_scored": rendered_frame_count,
            "matched_count": len(matches),
            "unmatched_reference_count": len(unmatched_reference),
            "average_match_score": (
                float(np.mean([item["score"] for item in matches])) if matches else 0.0
            ),
        },
        "matches": matches,
        "unmatched_reference": unmatched_reference,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "keyframe_matches.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract only reference keyframes, then match them against every "
            "rendered-video frame with order-preserving alignment."
        )
    )
    parser.add_argument("reference_video", nargs="?", type=Path)
    parser.add_argument("rendered_video", nargs="?", type=Path)
    parser.add_argument("--task-id", "--id", dest="sample_id", help="Dataset sample id")
    parser.add_argument("--data", "--dataset-root", dest="dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--result", "--generated-root", dest="generated_root", type=Path, default=DEFAULT_GENERATED_ROOT)
    parser.add_argument(
        "--ground-truth-root",
        type=Path,
        default=DEFAULT_GROUND_TRUTH_ROOT,
        help="Shared ground-truth keyframe cache root used by --task-id mode",
    )
    parser.add_argument("--model", "--run-name", dest="run_name", help="Model name used during generation; required with --task-id")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output root; defaults to <rendered_dir>/keyframe_alignment",
    )
    parser.add_argument(
        "--refresh-ground-truth",
        action="store_true",
        help="Force regeneration of the shared ground-truth keyframe cache",
    )
    parser.add_argument("--min-stable-sec", type=float, default=DEFAULT_MIN_STABLE_SEC)
    parser.add_argument("--change-window-sec", type=float, default=DEFAULT_CHANGE_WINDOW_SEC)
    parser.add_argument(
        "--min-keyframe-gap-sec", type=float, default=DEFAULT_MIN_KEYFRAME_GAP_SEC
    )
    parser.add_argument("--max-keyframes", type=int, default=DEFAULT_MAX_KEYFRAMES)
    parser.add_argument("--resize", type=int, default=DEFAULT_RESIZE)
    parser.add_argument("--visual-weight", type=float, default=DEFAULT_VISUAL_WEIGHT)
    parser.add_argument("--time-weight", type=float, default=DEFAULT_TIME_WEIGHT)
    parser.add_argument("--match-threshold", type=float, default=DEFAULT_MATCH_THRESHOLD)
    parser.add_argument("--skip-penalty", type=float, default=DEFAULT_SKIP_PENALTY)
    parser.add_argument("--top-candidates", type=int, default=DEFAULT_TOP_CANDIDATES)
    parser.add_argument(
        "--owner-margin",
        type=float,
        default=DEFAULT_OWNER_MARGIN,
        help=(
            "Keep references whose visual score is within this margin of a "
            "rendered frame's best reference"
        ),
    )
    return parser


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path, Path | None]:
    if args.sample_id:
        if args.reference_video or args.rendered_video:
            raise ValueError("Use either --task-id mode or two positional video paths, not both")
        if not args.run_name or not args.run_name.strip():
            raise ValueError("--model is required with --task-id")
        model_dir = args.run_name.lower().replace("/", "__")
        run_dir = args.generated_root.expanduser().resolve() / str(args.sample_id) / model_dir
        reference_video = args.dataset_root.expanduser().resolve() / str(args.sample_id) / "ground_truth.mp4"
        rendered_video = run_dir / "rendered_video.mp4"
        if not rendered_video.is_file():
            rendered_video = run_dir / "rendered.mp4"
        reference_dir = (
            args.ground_truth_root.expanduser().resolve()
            / str(args.sample_id)
            / GROUND_TRUTH_KEYFRAMES_DIRNAME
        )
    else:
        if args.reference_video is None or args.rendered_video is None:
            raise ValueError("Provide <reference_video> <rendered_video>, or use --task-id")
        reference_video = args.reference_video.resolve()
        rendered_video = args.rendered_video.resolve()
        run_dir = rendered_video.parent
        reference_dir = None
    if args.output_dir:
        output_dir = args.output_dir.resolve()
    else:
        output_dir = run_dir / "keyframe_alignment"
    return reference_video, rendered_video, output_dir, reference_dir


def validate_args(args: argparse.Namespace) -> None:
    if args.max_keyframes < 2:
        raise ValueError("--max-keyframes must be at least 2")
    if args.resize <= 0:
        raise ValueError("--resize must be positive")
    if args.top_candidates < 1:
        raise ValueError("--top-candidates must be at least 1")
    if not 0.0 <= args.match_threshold <= 1.0:
        raise ValueError("--match-threshold must be between 0 and 1")
    if args.visual_weight < 0 or args.time_weight < 0:
        raise ValueError("Similarity weights cannot be negative")
    if not np.isclose(args.visual_weight + args.time_weight, 1.0):
        raise ValueError("--visual-weight and --time-weight must sum to 1")
    if args.skip_penalty < 0:
        raise ValueError("--skip-penalty cannot be negative")
    if not 0.0 <= args.owner_margin <= 1.0:
        raise ValueError("--owner-margin must be between 0 and 1")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    reference_video, rendered_video, output_dir, reference_dir = resolve_inputs(args)
    align_videos(
        reference_video=reference_video,
        rendered_video=rendered_video,
        output_dir=output_dir,
        reference_dir=reference_dir,
        refresh_reference=args.refresh_ground_truth,
        min_stable_sec=args.min_stable_sec,
        change_window_sec=args.change_window_sec,
        min_keyframe_gap_sec=args.min_keyframe_gap_sec,
        max_keyframes=args.max_keyframes,
        resize=args.resize,
        visual_weight=args.visual_weight,
        time_weight=args.time_weight,
        match_threshold=args.match_threshold,
        skip_penalty=args.skip_penalty,
        top_candidates=args.top_candidates,
        owner_margin=args.owner_margin,
    )


if __name__ == "__main__":
    main()
