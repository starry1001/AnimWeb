#!/usr/bin/env python3
"""Animation Fidelity: Position, Size, Velocity, and component CLIP similarity."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.keyframe_extraction import (
    DEFAULT_CHANGE_WINDOW_SEC,
    DEFAULT_MAX_KEYFRAMES,
    DEFAULT_MIN_KEYFRAME_GAP_SEC,
    DEFAULT_MIN_STABLE_SEC,
    get_reference_keyframes,
)

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # A deterministic greedy fallback is provided below.
    linear_sum_assignment = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GPA_ROOT = PROJECT_ROOT / "utils" / "GPA-GUI-Detector"
DEFAULT_MODEL = GPA_ROOT / "model.pt"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data"
DEFAULT_GENERATED_ROOT = PROJECT_ROOT / "experiment"
DEFAULT_GROUND_TRUTH_ROOT = DEFAULT_GENERATED_ROOT / "ground_truth"
GPA_CACHE_SCHEMA = 1
SEGMENT_CACHE_SCHEMA = 4
FULL_TRAJECTORY_CACHE_SCHEMA = 1
COMPARISON_CACHE_SCHEMA = 1
TRACKING_IMPLEMENTATION = "gpa_bbox_tracks_clip_center"
VISUALIZATION_IMPLEMENTATION = "gpa_trajectory_overlay"
OCR_IMPLEMENTATION = "track_ocr_three_observations"
SCORING_IMPLEMENTATION = "animation_fidelity_legacy_matching"
METRIC_KEYS = ("position_score", "size_score", "velocity_score", "clip_score")
ASSIGNMENT_WEIGHTS = {"position": 0.60, "size": 0.25, "velocity": 0.05, "appearance": 0.10}
OCR_MATCH_THRESHOLD = 0.82

_OCR_READER: Any = None


@dataclass(frozen=True)
class GpaConfig:
    device: str | None = None
    batch_size: int = 16
    frame_step: int = 1
    conf: float = 0.05
    iou: float = 0.1
    imgsz: int = 1280
    max_box_area_ratio: float = 0.5
    recover_large_boxes: bool = True
    refine_box_area_ratio: float = 0.12
    refine_box_width_ratio: float = 0.55
    refine_box_height_ratio: float = 0.35
    temporal_large_box_max_gap: int = 4


@dataclass(frozen=True)
class TrackingConfig:
    max_track_gap: int = 2
    within_video_threshold: float = 0.35
    min_track_observations: int = 2
    timeline_samples: int = 24
    trajectory_distance_scale: float = 0.20
    clip_batch_size: int = 8


@dataclass
class Observation:
    frame_index: int
    normalized_time: float
    bbox: np.ndarray
    confidence: float
    clip_embedding: np.ndarray | None = None


@dataclass
class Track:
    track_id: int
    class_name: str
    observations: list[Observation] = field(default_factory=list)
    gray: np.ndarray | None = None
    color_histogram: np.ndarray | None = None
    mean_bbox: np.ndarray | None = None
    weight: float = 0.0
    motion_extent: float = 0.0
    ocr_samples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.observations[0].normalized_time

    @property
    def end(self) -> float:
        return self.observations[-1].normalized_time


def track_build_parameters(config: TrackingConfig) -> dict[str, Any]:
    """Parameters that change tracks, excluding cross-video scoring knobs."""
    return {
        "max_track_gap": config.max_track_gap,
        "within_video_threshold": config.within_video_threshold,
        "min_track_observations": config.min_track_observations,
    }


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def file_signature(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def stable_fingerprint(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@contextmanager
def cache_lock(target_dir: Path, timeout_sec: float = 7200.0):
    """Serialize shared cache creation across simultaneous model evaluations."""
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target_dir.with_name(f"{target_dir.name}.lock")
    started = time.monotonic()
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, f"pid={os.getpid()}\n".encode())
        except FileExistsError:
            try:
                if time.time() - lock_path.stat().st_mtime > timeout_sec:
                    lock_path.unlink()
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() - started > timeout_sec:
                raise TimeoutError(f"Timed out waiting for cache lock: {lock_path}")
            time.sleep(0.25)
    try:
        yield
    finally:
        os.close(descriptor)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _load_gpa_api():
    """Load the detector only on a cache miss, avoiding eager CUDA startup."""
    module_name = "animweb_gpa_detect_video"
    if module_name in sys.modules:
        return sys.modules[module_name].detect_video
    detector_path = GPA_ROOT / "detect_video.py"
    if not detector_path.is_file():
        raise FileNotFoundError(f"GPA detector entry point not found: {detector_path}")
    sys.path.insert(0, str(GPA_ROOT))
    spec = importlib.util.spec_from_file_location(module_name, detector_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import GPA detector: {detector_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.detect_video


def _gpa_cache_expected(video: Path, model: Path, config: GpaConfig) -> dict[str, Any]:
    # Device and batch size only affect execution. Excluding them lets the
    # shared GT cache be reused by jobs running on another GPU or batch size.
    semantic_parameters = {
        "frame_step": config.frame_step,
        "conf": config.conf,
        "iou": config.iou,
        "imgsz": config.imgsz,
        "max_box_area_ratio": config.max_box_area_ratio,
        "recover_large_boxes": config.recover_large_boxes,
        "refine_box_area_ratio": config.refine_box_area_ratio,
        "refine_box_width_ratio": config.refine_box_width_ratio,
        "refine_box_height_ratio": config.refine_box_height_ratio,
        "temporal_large_box_max_gap": config.temporal_large_box_max_gap,
    }
    identity = {
        "schema_version": GPA_CACHE_SCHEMA,
        "input_video": file_signature(video),
        "model": file_signature(model),
        "detector_implementation": {
            name: file_signature(GPA_ROOT / name)
            for name in ("detect_video.py", "box_refinement.py", "temporal_refinement.py")
        },
        "parameters": semantic_parameters,
    }
    return {**identity, "fingerprint": stable_fingerprint(identity)}


def reusable_gpa_results(output_dir: Path) -> dict[str, Any] | None:
    """Reuse complete detection files, including standalone GPA runs without cache.json.

    Optional annotated videos are not required: visualization must not trigger
    expensive inference again. Explicit --refresh-* flags still force detection.
    """
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        return None
    try:
        summary = read_json(summary_path)
        frames = summary["result"]["frames"]
        if not isinstance(frames, list) or not frames:
            return None
        signatures = []
        for item in frames:
            path = output_dir / item["result_file"]
            if not path.is_file():
                return None
            payload = read_json(path)
            if not isinstance(payload["detections"], list):
                return None
            int(payload["source_frame_index"])
            float(payload["time_sec"])
            for detection in payload["detections"]:
                if len(detection["bbox_xyxy"]) != 4:
                    return None
            signatures.append(file_signature(path))
        identity = {"summary": file_signature(summary_path), "frame_results": signatures}
        return {"fingerprint": stable_fingerprint(identity), "existing_detection_files": True}
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None



def _repair_gpa_summary_paths(output_dir: Path) -> None:
    summary_path = output_dir / "summary.json"
    summary = read_json(summary_path)
    if summary.get("annotated_video"):
        summary["annotated_video"] = str((output_dir / "gpa_annotated.mp4").resolve())
        summary["annotated_frames_directory"] = str((output_dir / "annotated_frames").resolve())
    summary["frame_results_directory"] = str((output_dir / "frame_results").resolve())
    write_json(summary_path, summary)


def ensure_gpa_cache(
    video: Path,
    output_dir: Path,
    model: Path,
    config: GpaConfig,
    *,
    refresh: bool,
    visualize: bool = False,
) -> tuple[dict[str, Any], bool]:
    """Return cache metadata and whether an existing full-video cache was reused."""
    if not refresh:
        existing = reusable_gpa_results(output_dir)
        if existing is not None:
            return existing, True

    with cache_lock(output_dir):
        if not refresh:
            existing = reusable_gpa_results(output_dir)
            if existing is not None:
                return existing, True
        expected = _gpa_cache_expected(video, model, config)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{output_dir.name}.building-", dir=output_dir.parent)
        )
        try:
            detect_video = _load_gpa_api()
            detect_video(
                video,
                temporary,
                model=model,
                device=config.device,
                batch_size=config.batch_size,
                frame_step=config.frame_step,
                conf=config.conf,
                iou=config.iou,
                imgsz=config.imgsz,
                max_box_area_ratio=config.max_box_area_ratio,
                recover_large_boxes=config.recover_large_boxes,
                refine_box_area_ratio=config.refine_box_area_ratio,
                refine_box_width_ratio=config.refine_box_width_ratio,
                refine_box_height_ratio=config.refine_box_height_ratio,
                temporal_large_box_max_gap=config.temporal_large_box_max_gap,
                save_visualizations=visualize,
            )
            write_json(temporary / "cache.json", expected)
            if output_dir.exists():
                shutil.rmtree(output_dir)
            temporary.rename(output_dir)
            _repair_gpa_summary_paths(output_dir)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    generated = reusable_gpa_results(output_dir)
    if generated is None:
        raise RuntimeError(f"GPA did not produce complete detection files: {output_dir}")
    return generated, False


def load_gpa_frames(gpa_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = read_json(gpa_dir / "summary.json")
    frames = []
    for item in summary["result"]["frames"]:
        result = read_json(gpa_dir / item["result_file"])
        frames.append(
            {
                "source_frame_index": int(result["source_frame_index"]),
                "time_sec": float(result["time_sec"]),
                "detections": result.get("detections", []),
            }
        )
    frames.sort(key=lambda item: item["source_frame_index"])
    return summary, frames


def video_info(video: Path) -> tuple[int, int, float, int]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if width <= 0 or height <= 0 or fps <= 0 or count <= 0:
        raise RuntimeError(f"Invalid video metadata: {video}")
    return width, height, fps, count


def normalized_bbox(raw: list[float], width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = (float(value) for value in raw)
    return np.asarray([x1 / width, y1 / height, x2 / width, y2 / height]).clip(0, 1)





def bbox_iou(left: np.ndarray, right: np.ndarray) -> float:
    x1, y1 = np.maximum(left[:2], right[:2])
    x2, y2 = np.minimum(left[2:], right[2:])
    intersection = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    left_area = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
    right_area = max(0.0, float(right[2] - right[0])) * max(0.0, float(right[3] - right[1]))
    union = left_area + right_area - intersection
    return intersection / union if union > 1e-12 else 0.0


def bbox_center(bbox: np.ndarray) -> np.ndarray:
    return np.asarray([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])


def optimal_pairs(matrix: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """One-to-one maximum assignment with an explicit unmatched option."""
    if matrix.size == 0:
        return []
    left_count, right_count = matrix.shape
    if linear_sum_assignment is not None:
        benefit = np.zeros((left_count + right_count, left_count + right_count))
        benefit[:left_count, :right_count] = matrix - threshold
        rows, columns = linear_sum_assignment(-benefit)
        candidates = [
            (left, right)
            for left, right in zip(rows.tolist(), columns.tolist())
            if left < left_count and right < right_count
        ]
    else:
        candidates = []
        used_left: set[int] = set()
        used_right: set[int] = set()
        ranked = sorted(
            (
                (float(matrix[left, right]), left, right)
                for left in range(left_count)
                for right in range(right_count)
            ),
            reverse=True,
        )
        for score, left, right in ranked:
            if score < threshold:
                break
            if left not in used_left and right not in used_right:
                candidates.append((left, right))
                used_left.add(left)
                used_right.add(right)
    return [(left, right) for left, right in candidates if matrix[left, right] >= threshold]


def _within_track_score(track: Track, observation: Observation) -> float:
    previous = track.observations[-1]
    overlap = bbox_iou(previous.bbox, observation.bbox)
    center_distance = float(np.linalg.norm(bbox_center(previous.bbox) - bbox_center(observation.bbox)))
    center_score = clamp01(1.0 - center_distance / math.sqrt(2.0))
    if previous.clip_embedding is None or observation.clip_embedding is None:
        raise ValueError("Trajectory construction requires CLIP embeddings")
    appearance = float(np.clip(np.dot(previous.clip_embedding, observation.clip_embedding), -1.0, 1.0))
    if overlap < 0.01 and (center_distance > 0.15 or appearance < 0.45):
        return 0.0
    return 0.50 * overlap + 0.30 * appearance + 0.20 * center_score


def _finalize_track(track: Track, frame_slots: int) -> Track:
    boxes = np.stack([item.bbox for item in track.observations])
    track.mean_bbox = np.mean(boxes, axis=0)
    centers = np.stack([bbox_center(item.bbox) for item in track.observations])
    track.motion_extent = float(np.max(np.linalg.norm(centers - centers[0], axis=1)))
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    coverage = len(track.observations) / max(1, frame_slots)
    motion_emphasis = 0.25 + 0.75 * min(1.0, track.motion_extent / 0.10)
    track.weight = max(1e-4, math.sqrt(max(1e-8, float(np.mean(areas))))) * coverage * motion_emphasis
    return track


def build_tracks(
    video: Path,
    segment_frames: list[dict[str, Any]],
    start_frame: int,
    end_frame: int,
    config: TrackingConfig,
) -> list[Track]:
    width, height, _, _ = video_info(video)
    features = CropFeatures(config.clip_batch_size)
    def detection_key(entry, detection):
        box = normalized_bbox(detection["bbox_xyxy"], width, height)
        return (str(video), int(entry["source_frame_index"]), bbox_key(box))
    features.ensure({detection_key(entry, detection) for entry in segment_frames
                     for detection in entry["detections"]})
    tracks: list[Track] = []
    next_track_id = 0
    denominator = max(1, end_frame - start_frame)
    frame_differences = [
        int(right["source_frame_index"]) - int(left["source_frame_index"])
        for left, right in zip(segment_frames, segment_frames[1:])
        if int(right["source_frame_index"]) > int(left["source_frame_index"])
    ]
    selected_frame_step = int(round(float(np.median(frame_differences)))) if frame_differences else 1
    for frame_entry in segment_frames:
        frame_index = int(frame_entry["source_frame_index"])
        candidates: list[tuple[str, Observation]] = []
        for detection in frame_entry["detections"]:
            raw_bbox = detection["bbox_xyxy"]
            candidates.append(
                (
                    str(detection.get("class_name", "unknown")),
                    Observation(
                        frame_index=frame_index,
                        normalized_time=clamp01((frame_index - start_frame) / denominator),
                        bbox=normalized_bbox(raw_bbox, width, height),
                        confidence=float(detection.get("confidence", 0.0)),
                        clip_embedding=features.cache[detection_key(frame_entry, detection)],
                    ),
                )
            )

        active = [
            track
            for track in tracks
            if frame_index - track.observations[-1].frame_index
            <= config.max_track_gap * max(1, selected_frame_step)
        ]
        matrix = np.zeros((len(active), len(candidates)), dtype=np.float64)
        for left, track in enumerate(active):
            for right, (_, observation) in enumerate(candidates):
                matrix[left, right] = _within_track_score(track, observation)
        pairs = optimal_pairs(matrix, config.within_video_threshold)
        matched_candidates = set()
        for left, right in pairs:
            active[left].observations.append(candidates[right][1])
            matched_candidates.add(right)
        for index, (class_name, observation) in enumerate(candidates):
            if index in matched_candidates:
                continue
            tracks.append(
                Track(
                    track_id=next_track_id,
                    class_name=class_name,
                    observations=[observation],
                )
            )
            next_track_id += 1

    return [
        _finalize_track(track, len(segment_frames))
        for track in tracks
        if len(track.observations) >= config.min_track_observations
    ]


def track_to_json(track: Track) -> dict[str, Any]:
    return {
        "track_id": track.track_id,
        "class_name": track.class_name,
        "weight": track.weight,
        "motion_extent": track.motion_extent,
        "ocr_samples": track.ocr_samples,
        "mean_bbox_xyxy_normalized": track.mean_bbox.tolist(),
        "observations": [
            {
                "frame_index": item.frame_index,
                "normalized_time": item.normalized_time,
                "bbox_xyxy_normalized": item.bbox.tolist(),
                "center_xy_normalized": bbox_center(item.bbox).tolist(),
                "size_wh_normalized": [
                    float(item.bbox[2] - item.bbox[0]),
                    float(item.bbox[3] - item.bbox[1]),
                ],
                "confidence": item.confidence,
            }
            for item in track.observations
        ],
    }


def track_from_json(payload: dict[str, Any]) -> Track:
    observations = [
        Observation(
            frame_index=int(item["frame_index"]),
            normalized_time=float(item["normalized_time"]),
            bbox=np.asarray(item["bbox_xyxy_normalized"], dtype=np.float64),
            confidence=float(item["confidence"]),
        )
        for item in payload["observations"]
    ]
    return Track(
        track_id=int(payload["track_id"]),
        class_name=str(payload["class_name"]),
        observations=observations,
        mean_bbox=np.asarray(payload["mean_bbox_xyxy_normalized"], dtype=np.float64),
        weight=float(payload["weight"]),
        motion_extent=float(payload["motion_extent"]),
        ocr_samples=list(payload.get("ocr_samples", [])),
    )


def track_color(index: int) -> tuple[int, int, int]:
    """Return a bright deterministic BGR color."""
    hue = (29 + index * 47) % 180
    pixel = np.uint8([[[hue, 220, 255]]])
    bgr = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(value) for value in bgr)


def draw_tracks(
    frame: np.ndarray,
    tracks: list[Track],
    normalized_time: float,
    styles: dict[int, tuple[tuple[int, int, int], str]] | None = None,
) -> np.ndarray:
    """Draw the current boxes and their accumulated center trails."""
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    line_width = max(1, int(round(min(width, height) / 540)))
    font_scale = max(0.42, min(width, height) / 1800)
    for track in tracks:
        bbox = interpolate_bbox(track, normalized_time)
        if bbox is None:
            continue
        if styles is None:
            color = track_color(track.track_id)
            label = f"T{track.track_id}"
        else:
            color, label = styles.get(
                track.track_id, ((150, 150, 150), f"U:T{track.track_id}")
            )

        all_centers = [
            (
                int(round(float(bbox_center(item.bbox)[0]) * width)),
                int(round(float(bbox_center(item.bbox)[1]) * height)),
            )
            for item in track.observations
        ]
        history = [
            point
            for point, item in zip(all_centers, track.observations)
            if item.normalized_time <= normalized_time + 1e-7
        ]
        dim_color = tuple(max(25, int(value * 0.35)) for value in color)
        if len(all_centers) >= 2:
            cv2.polylines(
                canvas,
                [np.asarray(all_centers, dtype=np.int32)],
                False,
                dim_color,
                line_width,
                cv2.LINE_AA,
            )
        if len(history) >= 2:
            cv2.polylines(
                canvas,
                [np.asarray(history, dtype=np.int32)],
                False,
                color,
                line_width + 1,
                cv2.LINE_AA,
            )
        for point in history[-8:]:
            cv2.circle(canvas, point, line_width + 1, color, -1, cv2.LINE_AA)

        x1 = int(round(float(bbox[0]) * width))
        y1 = int(round(float(bbox[1]) * height))
        x2 = int(round(float(bbox[2]) * width))
        y2 = int(round(float(bbox[3]) * height))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, line_width, cv2.LINE_AA)
        text_position = (max(0, x1), max(14, y1 - 4))
        cv2.putText(
            canvas,
            label,
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            line_width + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            line_width,
            cv2.LINE_AA,
        )
    return canvas


def encode_h264(raw_video: Path, output_video: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to create trajectory videos")
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(raw_video),
        "-an",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_video),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Could not encode trajectory video {output_video}: "
            f"{completed.stderr.strip()}"
        )
    raw_video.unlink()


def render_trajectory_video(
    video: Path,
    tracks: list[Track],
    output: Path,
    start_frame: int,
    end_frame: int,
    *,
    styles: dict[int, tuple[tuple[int, int, int], str]] | None = None,
) -> None:
    width, height, fps, frame_count = video_info(video)
    start_frame = max(0, start_frame)
    end_frame = min(frame_count - 1, end_frame)
    if end_frame < start_frame:
        raise ValueError(f"Invalid trajectory interval: {start_frame}->{end_frame}")
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_video = output.with_name(f".{output.stem}.raw.mp4")
    writer = cv2.VideoWriter(
        str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create trajectory video: {raw_video}")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        writer.release()
        raise RuntimeError(f"Could not open video: {video}")
    denominator = max(1, end_frame - start_frame)
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_index in range(start_frame, end_frame + 1):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode frame {frame_index} from {video}")
            normalized_time = (frame_index - start_frame) / denominator
            writer.write(draw_tracks(frame, tracks, normalized_time, styles))
    finally:
        capture.release()
        writer.release()
    try:
        encode_h264(raw_video, output)
    finally:
        if raw_video.exists():
            raw_video.unlink()


def ensure_full_trajectory_artifacts(
    *,
    video: Path,
    all_frames: list[dict[str, Any]],
    gpa_dir: Path,
    gpa_fingerprint: str,
    tracking_config: TrackingConfig,
    refresh: bool,
    visualize: bool = False,
) -> tuple[list[Track], bool]:
    """Create a full-video track JSON and overlay, even if no segment is scoreable."""
    expected = {
        "schema_version": FULL_TRAJECTORY_CACHE_SCHEMA,
        "tracking_implementation": TRACKING_IMPLEMENTATION,
        "visualization_implementation": VISUALIZATION_IMPLEMENTATION,
        "video": file_signature(video),
        "gpa_fingerprint": gpa_fingerprint,
        "tracking_parameters": track_build_parameters(tracking_config),
    }
    cache_path = gpa_dir / "trajectory_cache.json"
    tracks_path = gpa_dir / "trajectory_tracks.json"
    video_path = gpa_dir / "trajectory.mp4"

    def load_reusable() -> list[Track] | None:
        if not cache_path.is_file() or not tracks_path.is_file() or (visualize and not video_path.is_file()):
            return None
        try:
            if read_json(cache_path) != expected:
                return None
            return [
                track_from_json(item)
                for item in read_json(tracks_path)["tracks"]
            ]
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None

    if not refresh:
        tracks = load_reusable()
        if tracks is not None:
            return tracks, True
    with cache_lock(gpa_dir / "trajectory"):
        if not refresh:
            tracks = load_reusable()
            if tracks is not None:
                return tracks, True
        _, _, _, frame_count = video_info(video)
        tracks = build_tracks(video, all_frames, 0, frame_count - 1, tracking_config)
        temporary = Path(
            tempfile.mkdtemp(prefix=".trajectory.building-", dir=gpa_dir)
        )
        try:
            write_json(
                temporary / "trajectory_tracks.json",
                {
                    "tracking_implementation": TRACKING_IMPLEMENTATION,
                    "track_count": len(tracks),
                    "tracks": [track_to_json(track) for track in tracks],
                },
            )
            if visualize:
                render_trajectory_video(
                    video, tracks, temporary / "trajectory.mp4", 0, frame_count - 1,
                )
            write_json(temporary / "trajectory_cache.json", expected)
            for name in ("trajectory_tracks.json", "trajectory_cache.json") + (("trajectory.mp4",) if visualize else ()):
                os.replace(temporary / name, gpa_dir / name)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return tracks, False


def ocr_cache_identity(video: Path, gpa_fingerprint: str) -> dict[str, Any]:
    """Identify OCR inputs without importing its heavyweight reader eagerly."""
    packages = {}
    for name in ("easyocr", "pytesseract"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    if packages["easyocr"] is None and (packages["pytesseract"] is None or
                                         shutil.which("tesseract") is None):
        raise RuntimeError("Track OCR requires EasyOCR or pytesseract with tesseract")
    return {
        "implementation": OCR_IMPLEMENTATION,
        "video": file_signature(video),
        "gpa_fingerprint": gpa_fingerprint,
        "packages": packages,
    }


def _ocr_lines(frame: np.ndarray) -> tuple[str, list[dict[str, Any]]]:
    """Recognize full-frame text once; return normalized line boxes."""
    global _OCR_READER
    height, width = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if importlib.util.find_spec("easyocr") is not None:
        try:
            if _OCR_READER is None:
                import easyocr
                try:
                    import torch
                    gpu = bool(torch.cuda.is_available())
                except ImportError:
                    gpu = False
                _OCR_READER = easyocr.Reader(
                    ["en"], gpu=gpu, download_enabled=False, verbose=False
                )
            lines = []
            for points, raw_text, confidence in _OCR_READER.readtext(rgb):
                if not raw_text.strip():
                    continue
                xs, ys = zip(*points)
                lines.append({
                    "bbox": [max(0.0, min(xs) / width), max(0.0, min(ys) / height),
                             min(1.0, max(xs) / width), min(1.0, max(ys) / height)],
                    "text": raw_text.strip(),
                    "confidence": clamp01(confidence),
                })
            return "easyocr", lines
        except (ImportError, OSError, RuntimeError):
            if importlib.util.find_spec("pytesseract") is None or shutil.which("tesseract") is None:
                raise

    import pytesseract
    data = pytesseract.image_to_data(rgb, output_type=pytesseract.Output.DICT)
    groups: dict[tuple[int, int, int], list[int]] = {}
    for index, raw_text in enumerate(data["text"]):
        if raw_text.strip() and float(data["conf"][index]) >= 0:
            key = (data["block_num"][index], data["par_num"][index], data["line_num"][index])
            groups.setdefault(key, []).append(index)
    lines = []
    for indices in groups.values():
        x1 = min(data["left"][i] for i in indices)
        y1 = min(data["top"][i] for i in indices)
        x2 = max(data["left"][i] + data["width"][i] for i in indices)
        y2 = max(data["top"][i] + data["height"][i] for i in indices)
        lines.append({
            "bbox": [clamp01(x1 / width), clamp01(y1 / height),
                     clamp01(x2 / width), clamp01(y2 / height)],
            "text": " ".join(data["text"][i].strip() for i in indices),
            "confidence": clamp01(sum(float(data["conf"][i]) for i in indices)
                                  / (100.0 * len(indices))),
        })
    return "tesseract", lines


def load_or_create_ocr_frame(
    video: Path, frame_index: int, cache_dir: Path, identity: dict[str, Any]
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"frame_{frame_index:06d}.json"
    expected = {**identity, "frame_index": frame_index}

    def reusable() -> dict[str, Any] | None:
        if not path.is_file():
            return None
        try:
            payload = read_json(path)
            return payload if payload.get("identity") == expected else None
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    cached = reusable()
    if cached is not None:
        return cached
    with cache_lock(path):
        cached = reusable()
        if cached is not None:
            return cached
        capture = cv2.VideoCapture(str(video))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode OCR frame {frame_index} from {video}")
        finally:
            capture.release()
        backend, lines = _ocr_lines(frame)
        payload = {"identity": expected, "backend": backend, "lines": lines}
        descriptor, temporary_name = tempfile.mkstemp(prefix=".ocr-", suffix=".json", dir=cache_dir)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            write_json(temporary, payload)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return payload


def _selected_ocr_frames(track: Track) -> list[int]:
    observations = track.observations
    largest = max(observations, key=lambda item: (
        float((item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])),
        -abs(item.normalized_time - (track.start + track.end) / 2.0),
    ))
    return sorted({observations[0].frame_index, largest.frame_index,
                   observations[-1].frame_index})


def attach_ocr_to_tracks(
    video: Path, tracks: list[Track], cache_dir: Path, identity: dict[str, Any]
) -> None:
    """Read each selected frame once and attach its text to the owning track."""
    selected = {track.track_id: set(_selected_ocr_frames(track)) for track in tracks}
    frame_indices = sorted(set().union(*selected.values())) if tracks else []
    for frame_index in frame_indices:
        frame_ocr = load_or_create_ocr_frame(video, frame_index, cache_dir, identity)
        observed = {
            track.track_id: (track, next((item for item in track.observations
                                        if item.frame_index == frame_index), None))
            for track in tracks
        }
        owned: dict[int, list[dict[str, Any]]] = {}
        for line in frame_ocr["lines"]:
            box = np.asarray(line["bbox"], dtype=np.float64)
            text_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
            if text_area <= 0:
                continue
            candidates = []
            for track_id, (_, observation) in observed.items():
                if observation is None:
                    continue
                detection = observation.bbox
                x1, y1 = np.maximum(box[:2], detection[:2])
                x2, y2 = np.minimum(box[2:], detection[2:])
                coverage = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1)) / text_area
                if coverage >= 0.5:
                    area = max(0.0, float(detection[2] - detection[0])) * max(
                        0.0, float(detection[3] - detection[1]))
                    candidates.append((coverage, -area, track_id))
            if candidates:
                owner = max(candidates)[2]
                owned.setdefault(owner, []).append(line)
        for track_id, selected_frames in selected.items():
            if frame_index not in selected_frames:
                continue
            track, observation = observed[track_id]
            assert observation is not None
            lines = sorted(owned.get(track_id, []), key=lambda item: (item["bbox"][1], item["bbox"][0]))
            track.ocr_samples.append({
                "frame_index": frame_index,
                "normalized_time": observation.normalized_time,
                "text": " ".join(item["text"] for item in lines),
                "confidence": min((item["confidence"] for item in lines), default=0.0),
                "backend": frame_ocr["backend"],
            })


def write_video_segment(video: Path, output: Path, start_frame: int, end_frame: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to write segment videos")
    video_filter = (
        f"trim=start_frame={start_frame}:end_frame={end_frame + 1},"
        "setpts=PTS-STARTPTS,pad=ceil(iw/2)*2:ceil(ih/2)*2"
    )
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-an",
        "-vf",
        video_filter,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Could not write {output}: {completed.stderr.strip()}")


def ensure_segment_artifacts(
    *,
    video: Path,
    all_frames: list[dict[str, Any]],
    output_dir: Path,
    start_frame: int,
    end_frame: int,
    start_order: int,
    end_order: int,
    gpa_fingerprint: str,
    ocr_cache_dir: Path,
    tracking_config: TrackingConfig,
    refresh: bool,
    visualize: bool = False,
) -> tuple[list[Track], bool]:
    ocr_identity = ocr_cache_identity(video, gpa_fingerprint)
    expected = {
        "schema_version": SEGMENT_CACHE_SCHEMA,
        "tracking_implementation": TRACKING_IMPLEMENTATION,
        "video": file_signature(video),
        "gpa_fingerprint": gpa_fingerprint,
        "start_order": start_order,
        "end_order": end_order,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "tracking_parameters": track_build_parameters(tracking_config),
        "ocr_identity": ocr_identity,
    }
    required = [
        "cache.json",
        "tracks.json",
    ]
    if visualize:
        required.extend(("segment.mp4", "trajectory.mp4"))

    def load_reusable() -> list[Track] | None:
        if not all((output_dir / name).is_file() for name in required):
            return None
        try:
            if read_json(output_dir / "cache.json") != expected:
                return None
            tracks_payload = read_json(output_dir / "tracks.json")
            return [track_from_json(item) for item in tracks_payload["tracks"]]
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None

    if not refresh:
        tracks = load_reusable()
        if tracks is not None:
            return tracks, True

    with cache_lock(output_dir):
        if not refresh:
            tracks = load_reusable()
            if tracks is not None:
                return tracks, True
        selected = [
            item
            for item in all_frames
            if start_frame <= int(item["source_frame_index"]) <= end_frame
        ]
        if not selected:
            raise ValueError(
                f"No GPA frames fall inside video interval {start_frame}->{end_frame}"
            )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{output_dir.name}.building-", dir=output_dir.parent)
        )
        try:
            if visualize:
                write_video_segment(video, temporary / "segment.mp4", start_frame, end_frame)
            tracks = build_tracks(video, selected, start_frame, end_frame, tracking_config)
            attach_ocr_to_tracks(video, tracks, ocr_cache_dir, ocr_identity)
            write_json(
                temporary / "tracks.json",
                {
                    "tracking_implementation": TRACKING_IMPLEMENTATION,
                    "track_count": len(tracks),
                    "tracks": [track_to_json(track) for track in tracks],
                },
            )
            if visualize:
                render_trajectory_video(
                    video, tracks, temporary / "trajectory.mp4", start_frame, end_frame,
                )
            write_json(temporary / "cache.json", expected)
            if output_dir.exists():
                shutil.rmtree(output_dir)
            temporary.rename(output_dir)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return tracks, False


def ratio_score(left: float, right: float) -> float:
    if left <= 0 or right <= 0:
        return 0.0
    return min(left, right) / max(left, right)


def interpolate_bbox(track: Track, normalized_time: float) -> np.ndarray | None:
    if normalized_time < track.start - 1e-7 or normalized_time > track.end + 1e-7:
        return None
    times = np.asarray([item.normalized_time for item in track.observations])
    boxes = np.stack([item.bbox for item in track.observations])
    return np.asarray(
        [np.interp(normalized_time, times, boxes[:, index]) for index in range(4)]
    )


def interpolate_velocity(track: Track, normalized_time: float) -> np.ndarray | None:
    """Center displacement per source-video frame, including across missed detections."""
    if len(track.observations) < 2 or interpolate_bbox(track, normalized_time) is None:
        return None
    observations = track.observations
    times = [item.normalized_time for item in observations]
    right = min(len(observations) - 1, int(np.searchsorted(times, normalized_time, side="right")))
    left = right - 1
    first, last = observations[left], observations[right]
    frame_gap = last.frame_index - first.frame_index
    if frame_gap <= 0:
        return None
    return (bbox_center(last.bbox) - bbox_center(first.bbox)) / frame_gap


def crop_features(
    frame: np.ndarray, raw_bbox: list[float]
) -> tuple[np.ndarray | None, np.ndarray | None]:
    height, width = frame.shape[:2]
    x1 = max(0, min(width - 1, int(math.floor(raw_bbox[0]))))
    y1 = max(0, min(height - 1, int(math.floor(raw_bbox[1]))))
    x2 = max(x1 + 1, min(width, int(math.ceil(raw_bbox[2]))))
    y2 = max(y1 + 1, min(height, int(math.ceil(raw_bbox[3]))))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32)
    gray -= float(gray.mean())
    norm = float(np.linalg.norm(gray))
    gray_descriptor = gray.reshape(-1) / norm if norm > 1e-6 else None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256]).ravel()
    histogram_sum = float(histogram.sum())
    color_histogram = histogram / histogram_sum if histogram_sum > 0 else None
    return gray_descriptor, color_histogram

def mean_unit(vectors: list[np.ndarray]) -> np.ndarray | None:
    if not vectors:
        return None
    result = np.mean(np.stack(vectors), axis=0)
    norm = float(np.linalg.norm(result))
    return result / norm if norm > 1e-8 else None

def appearance_similarity(
    left_gray: np.ndarray | None,
    left_histogram: np.ndarray | None,
    right_gray: np.ndarray | None,
    right_histogram: np.ndarray | None,
) -> float:
    scores = []
    if left_gray is not None and right_gray is not None:
        scores.append(clamp01((float(np.dot(left_gray, right_gray)) + 1.0) / 2.0))
    if left_histogram is not None and right_histogram is not None:
        scores.append(clamp01(float(np.minimum(left_histogram, right_histogram).sum())))
    return float(np.mean(scores)) if scores else 0.5

def velocity_similarity(left: np.ndarray | None, right: np.ndarray | None) -> float:
    """Equation (3)-style direction/speed agreement with explicit zero-motion cases."""
    if left is None or right is None:
        return 0.0
    left_speed = float(np.linalg.norm(left))
    right_speed = float(np.linalg.norm(right))
    if left_speed < 1e-8 and right_speed < 1e-8:
        return 1.0
    if left_speed < 1e-8 or right_speed < 1e-8:
        return 0.0
    direction = float(np.dot(left, right)) / (left_speed * right_speed)
    speed_ratio = min(left_speed, right_speed) / max(left_speed, right_speed)
    return clamp01((direction + speed_ratio) / 2.0)


def attach_assignment_features(video: Path, tracks: list[Track]) -> None:
    """Recreate original gray/HSV track descriptors without running GPA."""
    by_frame: dict[int, list[tuple[Track, Observation]]] = {}
    for track in tracks:
        for observation in track.observations:
            by_frame.setdefault(observation.frame_index, []).append((track, observation))
    if not by_frame:
        return
    width, height, _, _ = video_info(video)
    gray = {t.track_id: [] for t in tracks}
    hist = {t.track_id: [] for t in tracks}
    capture = cv2.VideoCapture(str(video))
    try:
        previous = None
        for index in sorted(by_frame):
            if previous is None or index != previous + 1:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode assignment frame {index}: {video}")
            previous = index
            for track, observation in by_frame[index]:
                raw_box = observation.bbox * np.array([width, height, width, height])
                g, h = crop_features(frame, raw_box.tolist())
                if g is not None:
                    gray[track.track_id].append(g)
                if h is not None:
                    hist[track.track_id].append(h)
    finally:
        capture.release()
    for track in tracks:
        track.gray = mean_unit(gray[track.track_id])
        values = hist[track.track_id]
        track.color_histogram = np.mean(np.stack(values), axis=0) if values else None



def trajectory_pair_cost(
    left: Track,
    right: Track,
    timeline_samples: int,
    distance_scale: float,
    *,
    include_samples: bool = False,
) -> dict[str, Any]:
    """Original assignment evidence; these similarities are NOT reported metrics."""
    shared_position_scores = []
    shared_size_scores = []
    shared_velocity_scores = []
    samples = []
    squared_tolerance = 2.0 * distance_scale**2
    for normalized_time in np.linspace(0.0, 1.0, timeline_samples):
        left_bbox = interpolate_bbox(left, float(normalized_time))
        right_bbox = interpolate_bbox(right, float(normalized_time))
        if left_bbox is None and right_bbox is None:
            continue
        squared_distance = None
        left_velocity = interpolate_velocity(left, float(normalized_time))
        right_velocity = interpolate_velocity(right, float(normalized_time))
        if left_bbox is None or right_bbox is None:
            position_score = size_score = velocity_score = 0.0
        else:
            difference = bbox_center(left_bbox) - bbox_center(right_bbox)
            squared_distance = float(np.dot(difference, difference))
            position_score = 1.0 - clamp01(squared_distance / squared_tolerance)
            left_size = left_bbox[2:] - left_bbox[:2]
            right_size = right_bbox[2:] - right_bbox[:2]
            size_score = (
                ratio_score(float(left_size[0]), float(right_size[0]))
                + ratio_score(float(left_size[1]), float(right_size[1]))
            ) / 2.0
            velocity_score = velocity_similarity(left_velocity, right_velocity)
            shared_position_scores.append(position_score)
            shared_size_scores.append(size_score)
            shared_velocity_scores.append(velocity_score)
        if include_samples:
            samples.append(
                {
                    "normalized_time": round(float(normalized_time), 8),
                    "ground_truth_velocity_per_frame": (
                        left_velocity.tolist() if left_velocity is not None else None
                    ),
                    "rendered_velocity_per_frame": (
                        right_velocity.tolist() if right_velocity is not None else None
                    ),
                    "ground_truth_bbox": left_bbox.tolist() if left_bbox is not None else None,
                    "rendered_bbox": right_bbox.tolist() if right_bbox is not None else None,
                }
            )

    appearance_score = appearance_similarity(
        left.gray, left.color_histogram, right.gray, right.color_histogram
    )
    if len(shared_position_scores) >= 2:
        shared_scores = {
            "position": float(np.mean(shared_position_scores)),
            "size": float(np.mean(shared_size_scores)),
            "velocity": float(np.mean(shared_velocity_scores)),
            "appearance": appearance_score,
        }
        cost = sum(
            weight * (1.0 - shared_scores[name])
            for name, weight in ASSIGNMENT_WEIGHTS.items()
        )
    else:
        cost = 1.0
    result: dict[str, Any] = {
        "cost": cost,
        "shared_sample_count": len(shared_position_scores),
        "shared_position_score": (
            float(np.mean(shared_position_scores)) if shared_position_scores else 0.0
        ),
    }
    if include_samples:
        result["samples"] = samples
    return result



def minimum_cost_assignment(cost: np.ndarray) -> list[tuple[int, int]]:
    """Exact Hungarian assignment, including when SciPy is unavailable."""
    if linear_sum_assignment is not None:
        rows, columns = linear_sum_assignment(cost)
        return list(zip(rows.tolist(), columns.tolist()))

    # The shortest augmenting path form of the Hungarian algorithm. `cost` is
    # square because the caller adds dummy tracks for unmatched assignments.
    size = cost.shape[0]
    row_potential = np.zeros(size + 1)
    column_potential = np.zeros(size + 1)
    column_to_row = np.zeros(size + 1, dtype=int)
    predecessor = np.zeros(size + 1, dtype=int)
    for row in range(1, size + 1):
        column_to_row[0] = row
        minimum = np.full(size + 1, np.inf)
        used = np.zeros(size + 1, dtype=bool)
        column = 0
        while True:
            used[column] = True
            current_row = column_to_row[column]
            change = np.inf
            next_column = 0
            for candidate in range(1, size + 1):
                if used[candidate]:
                    continue
                reduced = (
                    cost[current_row - 1, candidate - 1]
                    - row_potential[current_row]
                    - column_potential[candidate]
                )
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    predecessor[candidate] = column
                if minimum[candidate] < change:
                    change = minimum[candidate]
                    next_column = candidate
            for candidate in range(size + 1):
                if used[candidate]:
                    row_potential[column_to_row[candidate]] += change
                    column_potential[candidate] -= change
                else:
                    minimum[candidate] -= change
            column = next_column
            if column_to_row[column] == 0:
                break
        while True:
            previous = predecessor[column]
            column_to_row[column] = column_to_row[previous]
            column = previous
            if column == 0:
                break
    return [(int(column_to_row[column] - 1), column - 1) for column in range(1, size + 1)]


def match_valid_track_costs(cost: np.ndarray, valid: np.ndarray) -> list[tuple[int, int]]:
    """Match as many eligible pairs as possible, then minimize their total cost."""
    left_count, right_count = cost.shape
    if not left_count or not right_count:
        return []
    if valid.shape != cost.shape:
        raise ValueError("Pair eligibility must have the same shape as pair cost")
    size = left_count + right_count
    # A dummy assignment costs more than the sum of all normalized pair costs.
    # It permits unmatched tracks without rejecting a valid pair by cost.
    unmatched_cost = float(size + 1)
    assignment_cost = np.zeros((size, size), dtype=np.float64)
    assignment_cost[:left_count, :right_count] = np.where(
        valid, cost, 4.0 * unmatched_cost
    )
    assignment_cost[:left_count, right_count:] = unmatched_cost
    assignment_cost[left_count:, :right_count] = unmatched_cost
    return [
        (left, right)
        for left, right in minimum_cost_assignment(assignment_cost)
        if left < left_count and right < right_count and valid[left, right]
    ]


def _normalized_ocr_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def track_text_similarity(left: Track, right: Track) -> float | None:
    """Compare readable OCR observations at approximately corresponding times."""
    def readable(track: Track) -> list[tuple[float, str]]:
        return [
            (float(item["normalized_time"]), _normalized_ocr_text(item["text"]))
            for item in track.ocr_samples
            if float(item.get("confidence", 0.0)) >= 0.5
            and len(_normalized_ocr_text(item.get("text", ""))) >= 3
        ]

    a, b = readable(left), readable(right)
    if not a or not b:
        return None

    def one_direction(source: list[tuple[float, str]], target: list[tuple[float, str]]) -> float:
        values = []
        for source_time, source_text in source:
            candidates = [
                SequenceMatcher(None, source_text, target_text).ratio()
                * (1.0 - 0.25 * abs(source_time - target_time))
                for target_time, target_text in target
                if abs(source_time - target_time) <= 0.35
            ]
            values.append(max(candidates, default=0.0))
        return float(np.mean(values))

    return (one_direction(a, b) + one_direction(b, a)) / 2.0


def match_tracks_with_ocr(
    ground_truth_tracks: list[Track],
    rendered_tracks: list[Track],
    evidence: list[list[dict[str, Any]]],
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], dict[str, Any]]]:
    """Anchor confident text first, then assign the remaining tracks by motion."""
    left_count, right_count = len(ground_truth_tracks), len(rendered_tracks)
    if not left_count or not right_count:
        return [], {}
    text_cost = np.ones((left_count, right_count), dtype=np.float64)
    text_valid = np.zeros((left_count, right_count), dtype=bool)
    similarities: dict[tuple[int, int], float] = {}
    for i, left in enumerate(ground_truth_tracks):
        for j, right in enumerate(rendered_tracks):
            similarity = track_text_similarity(left, right)
            if similarity is not None and similarity >= OCR_MATCH_THRESHOLD:
                similarities[i, j] = similarity
                text_valid[i, j] = True
                # OCR determines this stage; trajectory cost only breaks ties
                # between repeated text labels.
                text_cost[i, j] = 1.0 - similarity + 1e-6 * evidence[i][j]["cost"]
    anchors = match_valid_track_costs(text_cost, text_valid)
    used_left = {i for i, _ in anchors}
    used_right = {j for _, j in anchors}
    remaining_left = [i for i in range(left_count) if i not in used_left]
    remaining_right = [j for j in range(right_count) if j not in used_right]
    residual = np.asarray(
        [[evidence[i][j]["cost"] for j in remaining_right] for i in remaining_left],
        dtype=np.float64,
    ).reshape(len(remaining_left), len(remaining_right))
    eligible = np.asarray(
        [[
            evidence[i][j]["shared_sample_count"] >= 2
            and evidence[i][j]["shared_position_score"] > 0.0
            for j in remaining_right
        ] for i in remaining_left],
        dtype=bool,
    ).reshape(len(remaining_left), len(remaining_right))
    geometry_pairs = [
        (remaining_left[i], remaining_right[j])
        for i, j in match_valid_track_costs(residual, eligible)
    ]
    reasons = {
        (i, j): {"method": "ocr", "text_similarity": similarities[i, j]}
        for i, j in anchors
    }
    reasons.update({
        (i, j): {"method": "trajectory", "text_similarity": track_text_similarity(
            ground_truth_tracks[i], rendered_tracks[j]
        )}
        for i, j in geometry_pairs
    })
    return anchors + geometry_pairs, reasons


def score_track_sets(
    ground_truth_tracks: list[Track],
    rendered_tracks: list[Track],
    config: TrackingConfig,
    *,
    ground_truth_video: Path,
    rendered_video: Path,
    ground_truth_interval: tuple[int, int],
    rendered_interval: tuple[int, int],
    features: CropFeatures,
) -> dict[str, Any]:
    """OCR-first assignment followed by the original gray/HSV trajectory cost."""
    attach_assignment_features(ground_truth_video, ground_truth_tracks)
    attach_assignment_features(rendered_video, rendered_tracks)
    evidence = [[trajectory_pair_cost(left, right, config.timeline_samples,
                                     config.trajectory_distance_scale)
                 for right in rendered_tracks] for left in ground_truth_tracks]
    pairs, match_reasons = match_tracks_with_ocr(
        ground_truth_tracks, rendered_tracks, evidence
    )
    matched_left = {left for left, _ in pairs}
    matched_right = {right for _, right in pairs}
    match_results = []
    for left_index, right_index in pairs:
        left = ground_truth_tracks[left_index]
        right = rendered_tracks[right_index]
        pair = trajectory_pair_cost(left, right, config.timeline_samples,
                                    config.trajectory_distance_scale, include_samples=True)
        match_results.append(
            {
                "ground_truth_track_id": left.track_id,
                "rendered_track_id": right.track_id,
                "ground_truth_class": left.class_name,
                "rendered_class": right.class_name,
                "matching": match_reasons[left_index, right_index],
                "ground_truth_ocr_samples": left.ocr_samples,
                "rendered_ocr_samples": right.ocr_samples,
                "pair_cost": pair,
            }
        )
    return {
        "matches": match_results,
        "unmatched_ground_truth_track_ids": [
            track.track_id
            for index, track in enumerate(ground_truth_tracks)
            if index not in matched_left
        ],
        "unmatched_rendered_track_ids": [
            track.track_id
            for index, track in enumerate(rendered_tracks)
            if index not in matched_right
        ],
    }


def comparison_styles(
    ground_truth_tracks: list[Track],
    rendered_tracks: list[Track],
    track_score: dict[str, Any],
) -> tuple[
    dict[int, tuple[tuple[int, int, int], str]],
    dict[int, tuple[tuple[int, int, int], str]],
]:
    ground_truth_styles = {
        track.track_id: ((145, 145, 145), f"U:G{track.track_id}")
        for track in ground_truth_tracks
    }
    rendered_styles = {
        track.track_id: ((145, 145, 145), f"U:R{track.track_id}")
        for track in rendered_tracks
    }
    for match_index, match in enumerate(track_score["matches"]):
        color = track_color(match_index)
        ground_truth_id = int(match["ground_truth_track_id"])
        rendered_id = int(match["rendered_track_id"])
        ground_truth_styles[ground_truth_id] = (color, f"M{match_index}:G{ground_truth_id}")
        rendered_styles[rendered_id] = (color, f"M{match_index}:R{rendered_id}")
    return ground_truth_styles, rendered_styles


def _read_frame(capture: cv2.VideoCapture, frame_index: int, video: Path) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Could not decode frame {frame_index} from {video}")
    return frame


def _letterbox_panel(frame: np.ndarray, panel_width: int, panel_height: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(panel_width / width, panel_height / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    x = (panel_width - resized_width) // 2
    y = (panel_height - resized_height) // 2
    panel[y : y + resized_height, x : x + resized_width] = resized
    return panel


def render_comparison_video(
    *,
    ground_truth_video: Path,
    ground_truth_tracks: list[Track],
    ground_truth_start: int,
    ground_truth_end: int,
    rendered_video: Path,
    rendered_tracks: list[Track],
    rendered_start: int,
    rendered_end: int,
    track_score: dict[str, Any],
    output: Path,
) -> None:
    gt_width, gt_height, gt_fps, _ = video_info(ground_truth_video)
    rendered_width, rendered_height, rendered_fps, _ = video_info(rendered_video)
    panel_height = min(720, max(gt_height, rendered_height))
    panel_width = max(
        int(round(gt_width * panel_height / gt_height)),
        int(round(rendered_width * panel_height / rendered_height)),
    )
    output_fps = max(gt_fps, rendered_fps)
    gt_duration = (ground_truth_end - ground_truth_start + 1) / gt_fps
    rendered_duration = (rendered_end - rendered_start + 1) / rendered_fps
    output_frames = max(2, int(round(max(gt_duration, rendered_duration) * output_fps)))
    raw_video = output.with_name(f".{output.stem}.raw.mp4")
    writer = cv2.VideoWriter(
        str(raw_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (panel_width * 2, panel_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create comparison video: {raw_video}")
    gt_capture = cv2.VideoCapture(str(ground_truth_video))
    rendered_capture = cv2.VideoCapture(str(rendered_video))
    if not gt_capture.isOpened() or not rendered_capture.isOpened():
        writer.release()
        gt_capture.release()
        rendered_capture.release()
        raise RuntimeError("Could not open a source video for trajectory comparison")
    gt_styles, rendered_styles = comparison_styles(
        ground_truth_tracks, rendered_tracks, track_score
    )
    try:
        for slot in range(output_frames):
            normalized_time = slot / max(1, output_frames - 1)
            gt_frame_index = int(
                round(ground_truth_start + normalized_time * (ground_truth_end - ground_truth_start))
            )
            rendered_frame_index = int(
                round(rendered_start + normalized_time * (rendered_end - rendered_start))
            )
            gt_frame = draw_tracks(
                _read_frame(gt_capture, gt_frame_index, ground_truth_video),
                ground_truth_tracks,
                normalized_time,
                gt_styles,
            )
            rendered_frame = draw_tracks(
                _read_frame(rendered_capture, rendered_frame_index, rendered_video),
                rendered_tracks,
                normalized_time,
                rendered_styles,
            )
            gt_panel = _letterbox_panel(gt_frame, panel_width, panel_height)
            rendered_panel = _letterbox_panel(rendered_frame, panel_width, panel_height)
            canvas = np.concatenate([gt_panel, rendered_panel], axis=1)
            cv2.rectangle(canvas, (0, 0), (panel_width * 2, 38), (0, 0, 0), -1)
            cv2.putText(
                canvas,
                f"GROUND TRUTH  frame {gt_frame_index}",
                (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"RENDERED  frame {rendered_frame_index}",
                (panel_width + 12, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(canvas)
    finally:
        gt_capture.release()
        rendered_capture.release()
        writer.release()
    try:
        encode_h264(raw_video, output)
    finally:
        if raw_video.exists():
            raw_video.unlink()


def ensure_comparison_video(
    *,
    output_dir: Path,
    ground_truth_video: Path,
    ground_truth_tracks: list[Track],
    ground_truth_start: int,
    ground_truth_end: int,
    rendered_video: Path,
    rendered_tracks: list[Track],
    rendered_start: int,
    rendered_end: int,
    track_score: dict[str, Any],
    refresh: bool,
) -> tuple[Path, bool]:
    pair_ids = [
        [int(item["ground_truth_track_id"]), int(item["rendered_track_id"])]
        for item in track_score["matches"]
    ]
    identity = {
        "schema_version": COMPARISON_CACHE_SCHEMA,
        "visualization_implementation": VISUALIZATION_IMPLEMENTATION,
        "ground_truth_video": file_signature(ground_truth_video),
        "rendered_video": file_signature(rendered_video),
        "ground_truth_interval": [ground_truth_start, ground_truth_end],
        "rendered_interval": [rendered_start, rendered_end],
        "ground_truth_tracks": stable_fingerprint(
            [track_to_json(track) for track in ground_truth_tracks]
        ),
        "rendered_tracks": stable_fingerprint(
            [track_to_json(track) for track in rendered_tracks]
        ),
        "matched_track_ids": pair_ids,
        "scoring_implementation": SCORING_IMPLEMENTATION,
    }
    expected = {**identity, "fingerprint": stable_fingerprint(identity)}
    video_path = output_dir / "trajectory_comparison.mp4"
    cache_path = output_dir / "trajectory_comparison.cache.json"
    if not refresh and video_path.is_file() and cache_path.is_file():
        try:
            if read_json(cache_path) == expected:
                return video_path, True
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    with cache_lock(output_dir / "trajectory_comparison"):
        if not refresh and video_path.is_file() and cache_path.is_file():
            try:
                if read_json(cache_path) == expected:
                    return video_path, True
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        temporary = Path(
            tempfile.mkdtemp(prefix=".comparison.building-", dir=output_dir)
        )
        try:
            render_comparison_video(
                ground_truth_video=ground_truth_video,
                ground_truth_tracks=ground_truth_tracks,
                ground_truth_start=ground_truth_start,
                ground_truth_end=ground_truth_end,
                rendered_video=rendered_video,
                rendered_tracks=rendered_tracks,
                rendered_start=rendered_start,
                rendered_end=rendered_end,
                track_score=track_score,
                output=temporary / "trajectory_comparison.mp4",
            )
            write_json(temporary / "trajectory_comparison.cache.json", expected)
            os.replace(temporary / "trajectory_comparison.mp4", video_path)
            os.replace(temporary / "trajectory_comparison.cache.json", cache_path)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return video_path, False


def frame_index(start: int, end: int, normalized_time: float) -> int:
    """Inverse of build_tracks: normalized_time = (frame - start) / (end - start)."""
    return int(round(start + normalized_time * max(1, end - start)))

def bbox_key(bbox: list[float]) -> tuple[float, ...]:
    return tuple(round(float(value), 6) for value in bbox)

def crop_rgb(frame_bgr: np.ndarray, bbox: tuple[float, ...]) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    x1 = max(0, min(width - 1, round(bbox[0] * width)))
    y1 = max(0, min(height - 1, round(bbox[1] * height)))
    x2 = max(x1 + 1, min(width, round(bbox[2] * width)))
    y2 = max(y1 + 1, min(height, round(bbox[3] * height)))
    return cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)

class CropFeatures:
    """Unit CLIP embeddings keyed by (video, frame, bbox).

    The cache lives for the whole process, so GT crops are shared by all models
    of one sample ID.
    """

    def __init__(self, batch_size: int) -> None:
        from utils import static_appearance_score as appearance

        if batch_size < 1:
            raise ValueError("CLIP batch size must be positive")
        self.appearance = appearance
        self.batch_size = batch_size
        self.cache: dict[tuple[str, int, tuple[float, ...]], np.ndarray] = {}

    def ensure(self, requests: set[tuple[str, int, tuple[float, ...]]]) -> None:
        missing = sorted(key for key in requests if key not in self.cache)
        by_video: dict[str, dict[int, list[tuple[float, ...]]]] = {}
        for video, index, bbox in missing:
            by_video.setdefault(video, {}).setdefault(index, []).append(bbox)
        keys: list[tuple[str, int, tuple[float, ...]]] = []
        crops: list[np.ndarray] = []

        def flush() -> None:
            if not crops:
                return
            for key, embedding in zip(keys, self._clip_embeddings(crops)):
                self.cache[key] = embedding
            keys.clear()
            crops.clear()

        for video, frames in by_video.items():
            capture = cv2.VideoCapture(video)
            if not capture.isOpened():
                raise RuntimeError(f"Could not open video: {video}")
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            try:
                for index in sorted(frames):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, min(index, max(0, frame_count - 1)))
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError(f"Could not decode frame {index} from {video}")
                    for bbox in frames[index]:
                        keys.append((video, index, bbox))
                        crops.append(crop_rgb(frame, bbox))
                        if len(crops) >= self.batch_size:
                            flush()
            finally:
                capture.release()
        flush()

    def _clip_embeddings(self, crops: list[np.ndarray]) -> list[np.ndarray]:
        import torch
        from PIL import Image

        model, preprocess, device = self.appearance.get_clip_model()
        embeddings = []
        for offset in range(0, len(crops), self.batch_size):
            batch = torch.stack([
                preprocess(Image.fromarray(crop)) for crop in crops[offset:offset + self.batch_size]
            ]).to(device)
            with torch.no_grad():
                features = model.encode_image(batch).float()
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            embeddings.extend(features.cpu().numpy())
        return embeddings

    def similarity(self, left, right) -> float:
        return clamp01(float(np.dot(self.cache[left], self.cache[right])))


def sample_pairs(report: dict[str, Any]):
    """Yield (segment, match, sample, gt_key, rendered_key); keys are None if invisible."""
    gt_video = report["inputs"]["ground_truth_video"]
    rendered_video = report["inputs"]["rendered_video"]
    for segment in report["segments"]:
        if segment["status"] != "scored":
            continue
        gt, rendered = segment["ground_truth"], segment["rendered"]
        for match in segment["track_score"]["matches"]:
            for sample in match["pair_cost"]["samples"]:
                t = float(sample["normalized_time"])
                gt_bbox, rendered_bbox = sample["ground_truth_bbox"], sample["rendered_bbox"]
                yield (
                    segment, match, sample,
                    (gt_video, frame_index(gt["start_frame"], gt["end_frame"], t),
                     bbox_key(gt_bbox)) if gt_bbox is not None else None,
                    (rendered_video, frame_index(rendered["start_frame"], rendered["end_frame"], t),
                     bbox_key(rendered_bbox)) if rendered_bbox is not None else None,
                )

def sample_metrics(sample: dict[str, Any]) -> dict[str, float] | None:
    """Paper metrics on frame-normalized boxes and per-frame velocities."""
    left, right = sample["ground_truth_bbox"], sample["rendered_bbox"]
    if left is None and right is None:
        return None
    values = {name: 0.0 for name in METRIC_KEYS[:3]}
    if left is None or right is None:
        return values
    dx = (left[0] + left[2] - right[0] - right[2]) / 2.0
    dy = (left[1] + left[3] - right[1] - right[3]) / 2.0
    values["position_score"] = max(0.0, 1.0 - math.hypot(dx, dy) / math.sqrt(2.0))
    values["size_score"] = math.sqrt(
        ratio_score(left[2] - left[0], right[2] - right[0])
        * ratio_score(left[3] - left[1], right[3] - right[1]))
    a, b = sample["ground_truth_velocity_per_frame"], sample["rendered_velocity_per_frame"]
    if a is not None and b is not None:
        sa, sb = math.hypot(*a), math.hypot(*b)
        if sa < 1e-8 and sb < 1e-8:
            values["velocity_score"] = 1.0
        elif sa >= 1e-8 and sb >= 1e-8:
            cosine = (a[0] * b[0] + a[1] * b[1]) / (sa * sb)
            values["velocity_score"] = clamp01((cosine + min(sa, sb) / max(sa, sb)) / 2.0)
    return values


def apply_animation_metrics(report: dict[str, Any], features: CropFeatures) -> None:
    """Use augment-script aggregation, including zero credit for missing tracks."""
    pairs = list(sample_pairs(report))
    features.ensure({key for *_, a, b in pairs if a is not None and b is not None
                     for key in (a, b)})
    per_match: dict[int, list[dict[str, float]]] = {}
    for _, match, sample, a, b in pairs:
        values = sample_metrics(sample)
        if values is None:
            continue
        values["clip_score"] = features.similarity(a, b) if a is not None and b is not None else 0.0
        sample.update(values)
        per_match.setdefault(id(match), []).append(values)
    for segment in report["segments"]:
        values = {name: 0.0 for name in METRIC_KEYS}
        if segment["status"] == "scored":
            matches = segment["track_score"]["matches"]
            union = segment["ground_truth"]["track_count"] + segment["rendered"]["track_count"] - len(matches)
            for match in matches:
                points = per_match.get(id(match), [])
                match["metrics"] = {
                    name: sum(p[name] for p in points) / len(points) if points else 0.0
                    for name in METRIC_KEYS}
            values = {name: sum(m["metrics"][name] for m in matches) / union if union else 1.0
                      for name in METRIC_KEYS}
            segment["track_score"].update(values)
        segment.update(values)
    report["summary"].update({
        name: float(np.mean([s[name] for s in report["segments"]])) if report["segments"] else 0.0
        for name in METRIC_KEYS})


def load_keyframes(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    keyframes = sorted(payload["keyframes"], key=lambda item: int(item["order"]))
    if len(keyframes) < 2:
        raise ValueError(f"At least two ground-truth keyframes are required: {path}")
    return keyframes


def prepare_ground_truth(
    sample_id: str,
    *,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    ground_truth_root: Path = DEFAULT_GROUND_TRUTH_ROOT,
    model: Path = DEFAULT_MODEL,
    gpa_config: GpaConfig | None = None,
    tracking_config: TrackingConfig | None = None,
    visualize: bool = False,
    refresh_ground_truth: bool = False,
    refresh_segments: bool = False,
) -> None:
    """Build and validate all shared inputs before model jobs start."""
    video = dataset_root.expanduser().resolve() / sample_id / "ground_truth.mp4"
    ground_truth_dir = ground_truth_root.expanduser().resolve() / sample_id
    model = model.expanduser().resolve()
    for required in (video, model):
        if not required.is_file():
            raise FileNotFoundError(f"Required input not found: {required}")
    keyframes_dir = ground_truth_dir / "keyframes"
    get_reference_keyframes(
        video, keyframes_dir, DEFAULT_MIN_STABLE_SEC, DEFAULT_CHANGE_WINDOW_SEC,
        DEFAULT_MIN_KEYFRAME_GAP_SEC, DEFAULT_MAX_KEYFRAMES, refresh_ground_truth,
    )
    keyframes = load_keyframes(keyframes_dir / "keyframes.json")
    gpa_config = gpa_config or GpaConfig()
    tracking_config = tracking_config or TrackingConfig()
    gpa_dir = ground_truth_dir / "gpa"
    gpa_cache, gpa_reused = ensure_gpa_cache(
        video, gpa_dir, model, gpa_config,
        refresh=refresh_ground_truth, visualize=visualize,
    )
    _, frames = load_gpa_frames(gpa_dir)
    if visualize:
        ensure_full_trajectory_artifacts(
            video=video, all_frames=frames, gpa_dir=gpa_dir,
            gpa_fingerprint=gpa_cache["fingerprint"],
            tracking_config=tracking_config,
            refresh=refresh_segments or not gpa_reused, visualize=True,
        )
    for start, end in zip(keyframes, keyframes[1:]):
        start_order, end_order = int(start["order"]), int(end["order"])
        segment_dir = (ground_truth_dir / "animation_segments" /
                       f"segment_{start_order:02d}_{end_order:02d}")
        ensure_segment_artifacts(
            video=video, all_frames=frames,
            output_dir=segment_dir,
            start_frame=int(start["frame_index"]),
            end_frame=int(end["frame_index"]),
            start_order=start_order, end_order=end_order,
            gpa_fingerprint=gpa_cache["fingerprint"],
            ocr_cache_dir=gpa_dir / "ocr_frames",
            tracking_config=tracking_config,
            refresh=refresh_segments or not gpa_reused, visualize=visualize,
        )


def score_animation_segments(
    sample_id: str,
    run_name: str,
    *,
    clip_batch_size: int = 32,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    generated_root: Path = DEFAULT_GENERATED_ROOT,
    ground_truth_root: Path = DEFAULT_GROUND_TRUTH_ROOT,
    model: Path = DEFAULT_MODEL,
    gpa_config: GpaConfig | None = None,
    tracking_config: TrackingConfig | None = None,
    refresh_ground_truth: bool = False,
    refresh_rendered: bool = False,
    refresh_segments: bool = False,
    visualize: bool = False,
) -> dict[str, Any]:
    gpa_config = gpa_config or GpaConfig()
    tracking_config = tracking_config or TrackingConfig()
    dataset_root = dataset_root.expanduser().resolve()
    generated_root = generated_root.expanduser().resolve()
    ground_truth_root = ground_truth_root.expanduser().resolve()
    model = model.expanduser().resolve()
    run_dir = generated_root / sample_id / run_name
    ground_truth_dir = ground_truth_root / sample_id
    ground_truth_video = dataset_root / sample_id / "ground_truth.mp4"
    rendered_video = run_dir / "rendered_video.mp4"
    if not rendered_video.is_file():
        rendered_video = run_dir / "rendered.mp4"
    matches_path = run_dir / "keyframe_alignment" / "keyframe_matches.json"
    keyframes_path = ground_truth_dir / "keyframes" / "keyframes.json"
    for required in (ground_truth_video, rendered_video, matches_path, keyframes_path, model):
        if not required.is_file():
            raise FileNotFoundError(f"Required input not found: {required}")
    ground_truth_fps = video_info(ground_truth_video)[2]
    rendered_fps = video_info(rendered_video)[2]
    if abs(ground_truth_fps - rendered_fps) > 0.05:
        raise ValueError(
            "Per-frame velocity comparison requires matching video FPS: "
            f"ground truth={ground_truth_fps:.3f}, rendered={rendered_fps:.3f}"
        )

    alignment = read_json(matches_path)
    keyframes = load_keyframes(keyframes_path)
    match_by_order = {
        int(
            item["ground_truth_order"]
            if "ground_truth_order" in item
            else item["reference_order"]
        ): item
        for item in alignment["matches"]
    }

    ground_truth_gpa_dir = ground_truth_dir / "gpa"
    rendered_gpa_dir = run_dir / "gpa"
    ground_truth_cache, ground_truth_reused = ensure_gpa_cache(
        ground_truth_video,
        ground_truth_gpa_dir,
        model,
        gpa_config,
        refresh=refresh_ground_truth,
        visualize=visualize,
    )
    rendered_cache, rendered_reused = ensure_gpa_cache(
        rendered_video,
        rendered_gpa_dir,
        model,
        gpa_config,
        refresh=refresh_rendered,
        visualize=visualize,
    )

    _, ground_truth_frames = load_gpa_frames(ground_truth_gpa_dir)
    _, rendered_frames = load_gpa_frames(rendered_gpa_dir)
    ground_truth_trajectory_reused = False
    rendered_trajectory_reused = False
    if visualize:
        _, ground_truth_trajectory_reused = ensure_full_trajectory_artifacts(
            video=ground_truth_video,
            all_frames=ground_truth_frames,
            gpa_dir=ground_truth_gpa_dir,
            gpa_fingerprint=ground_truth_cache["fingerprint"],
            tracking_config=tracking_config,
            refresh=refresh_segments or not ground_truth_reused,
            visualize=visualize,
        )
        _, rendered_trajectory_reused = ensure_full_trajectory_artifacts(
            video=rendered_video,
            all_frames=rendered_frames,
            gpa_dir=rendered_gpa_dir,
            gpa_fingerprint=rendered_cache["fingerprint"],
            tracking_config=tracking_config,
            refresh=refresh_segments or not rendered_reused,
            visualize=visualize,
        )
    ground_truth_segments_root = ground_truth_dir / "animation_segments"
    rendered_segments_root = run_dir / "animation_segments"
    rendered_segments_root.mkdir(parents=True, exist_ok=True)

    segment_results = []
    features = CropFeatures(clip_batch_size)
    for segment_index, (start, end) in enumerate(zip(keyframes, keyframes[1:])):
        start_order = int(start["order"])
        end_order = int(end["order"])
        segment_name = f"segment_{start_order:02d}_{end_order:02d}"
        ground_truth_start = int(start["frame_index"])
        ground_truth_end = int(end["frame_index"])
        ground_truth_segment_dir = ground_truth_segments_root / segment_name
        ground_truth_tracks, gt_segment_reused = ensure_segment_artifacts(
            video=ground_truth_video,
            all_frames=ground_truth_frames,
            output_dir=ground_truth_segment_dir,
            start_frame=ground_truth_start,
            end_frame=ground_truth_end,
            start_order=start_order,
            end_order=end_order,
            gpa_fingerprint=ground_truth_cache["fingerprint"],
            ocr_cache_dir=ground_truth_gpa_dir / "ocr_frames",
            tracking_config=tracking_config,
            refresh=refresh_segments or not ground_truth_reused,
            visualize=visualize,
        )
        entry: dict[str, Any] = {
            "segment_index": segment_index,
            "segment_name": segment_name,
            "ground_truth": {
                "start_order": start_order,
                "end_order": end_order,
                "start_frame": ground_truth_start,
                "end_frame": ground_truth_end,
                "directory": str(ground_truth_segment_dir),
                "trajectory_video": str(ground_truth_segment_dir / "trajectory.mp4") if visualize else None,
                "track_count": len(ground_truth_tracks),
                "cache_reused": gt_segment_reused,
            },
            **{name: 0.0 for name in METRIC_KEYS},
        }
        missing_orders = [
            order for order in (start_order, end_order) if order not in match_by_order
        ]
        if missing_orders:
            entry.update(
                {
                    "status": "unmatched_keyframe_endpoint",
                    "missing_ground_truth_orders": missing_orders,
                    "reason": "Both adjacent keyframes need rendered matches; this segment scores zero.",
                }
            )
            segment_results.append(entry)
            continue

        rendered_start = int(match_by_order[start_order]["rendered_frame_index"])
        rendered_end = int(match_by_order[end_order]["rendered_frame_index"])
        if rendered_end <= rendered_start:
            entry.update(
                {
                    "status": "invalid_rendered_interval",
                    "rendered": {"start_frame": rendered_start, "end_frame": rendered_end},
                    "reason": "Rendered endpoint frames must be strictly increasing.",
                }
            )
            segment_results.append(entry)
            continue

        rendered_segment_dir = rendered_segments_root / segment_name
        rendered_tracks, rendered_segment_reused = ensure_segment_artifacts(
            video=rendered_video,
            all_frames=rendered_frames,
            output_dir=rendered_segment_dir,
            start_frame=rendered_start,
            end_frame=rendered_end,
            start_order=start_order,
            end_order=end_order,
            gpa_fingerprint=rendered_cache["fingerprint"],
            ocr_cache_dir=rendered_gpa_dir / "ocr_frames",
            tracking_config=tracking_config,
            refresh=refresh_segments or not rendered_reused,
            visualize=visualize,
        )
        track_score = score_track_sets(
            ground_truth_tracks, rendered_tracks, tracking_config,
            ground_truth_video=ground_truth_video, rendered_video=rendered_video,
            ground_truth_interval=(ground_truth_start, ground_truth_end),
            rendered_interval=(rendered_start, rendered_end), features=features,
        )
        comparison_video, comparison_reused = (ensure_comparison_video(
            output_dir=rendered_segment_dir,
            ground_truth_video=ground_truth_video,
            ground_truth_tracks=ground_truth_tracks,
            ground_truth_start=ground_truth_start,
            ground_truth_end=ground_truth_end,
            rendered_video=rendered_video,
            rendered_tracks=rendered_tracks,
            rendered_start=rendered_start,
            rendered_end=rendered_end,
            track_score=track_score,
            refresh=(
                refresh_segments
                or not gt_segment_reused
                or not rendered_segment_reused
            ),
        ) if visualize else (None, False))
        entry.update(
            {
                "status": "scored",
                "rendered": {
                    "start_frame": rendered_start,
                    "end_frame": rendered_end,
                    "directory": str(rendered_segment_dir),
                    "trajectory_video": str(rendered_segment_dir / "trajectory.mp4") if visualize else None,
                    "trajectory_comparison_video": str(comparison_video) if comparison_video else None,
                    "trajectory_comparison_cache_reused": comparison_reused,
                    "track_count": len(rendered_tracks),
                    "cache_reused": rendered_segment_reused,
                },
                "track_score": track_score,
            }
        )
        segment_results.append(entry)

    result = {
        "mode": "gpa_full_video_atomic_segment_trajectory_score",
        "sample_id": sample_id,
        "run_name": run_name,
        "inputs": {
            "keyframe_matches": str(matches_path),
            "ground_truth_keyframes": str(keyframes_path),
            "ground_truth_video": str(ground_truth_video),
            "rendered_video": str(rendered_video),
            "video_fps": ground_truth_fps,
        },
        "artifacts": {
            "ground_truth_gpa": str(ground_truth_gpa_dir),
            "ground_truth_ocr_frames": str(ground_truth_gpa_dir / "ocr_frames"),
            "ground_truth_full_trajectory_video": str(ground_truth_gpa_dir / "trajectory.mp4") if visualize else None,
            "rendered_gpa": str(rendered_gpa_dir),
            "rendered_ocr_frames": str(rendered_gpa_dir / "ocr_frames"),
            "rendered_full_trajectory_video": str(rendered_gpa_dir / "trajectory.mp4") if visualize else None,
            "ground_truth_segments": str(ground_truth_segments_root),
            "rendered_segments": str(rendered_segments_root),
        },
        "policy": {
            "gpa_runs_on_complete_video_before_segmentation": True,
            "ground_truth_gpa_reused_across_models": True,
            "atomic_segment_requires_both_keyframe_matches": True,
            "missing_segment_score": 0.0,
            "segment_time_normalization": "each interval mapped to [0, 1]",
            "scoring_implementation": SCORING_IMPLEMENTATION,
            "within_video_association": "0.5*IoU + 0.3*CLIP cosine + 0.2*(1-normalized center distance/sqrt(2))",
            "within_video_gate": "reject if IoU<0.01 and (center distance>0.15 or CLIP cosine<0.45)",
            "ocr_sampling": "first observation, largest-area observation, last observation per track",
            "ocr_matching": "confident text anchors first; remaining tracks use trajectory assignment",
            "velocity_unit": "normalized screen displacement per source-video frame",
            "velocity_score": "clipped mean of (direction cosine + speed ratio)/2; both stationary=1, one stationary=0",
            "position_score": "mean of 1 - normalized center distance / sqrt(2)",
            "size_score": "mean geometric mean of width and height min/max ratios",
            "assignment_cost": "0.60*(1-P_match) + 0.25*(1-S_match) + 0.05*(1-V_match) + 0.10*(1-A_match); original shared-visible squared-position, arithmetic-size, velocity, gray/HSV appearance",
            "assignment": "OCR anchors then maximum-cardinality minimum-cost Hungarian for tracks with at least two shared samples and positive shared position similarity; no pair-cost cutoff",
            "segment_metric_score": "sum of matched pair means divided by GT/rendered track union; equal mean over all reference segments",
            "clip_score": "CLIP ViT-B/32 cosine of component crops, clipped to [0, 1]",
            "unmatched_tracks_are_penalized": True,
            "bbox_sizes_are_scored": True,
        },
        "parameters": {
            "gpa": asdict(gpa_config),
            "tracking": asdict(tracking_config),
            "clip_batch_size": clip_batch_size,
        },
        "cache": {
            "ground_truth_gpa_reused": ground_truth_reused,
            "rendered_gpa_reused": rendered_reused,
            "ground_truth_full_trajectory_reused": ground_truth_trajectory_reused,
            "rendered_full_trajectory_reused": rendered_trajectory_reused,
        },
        "summary": {
            "ground_truth_keyframe_count": len(keyframes),
            "atomic_segment_count": len(segment_results),
            "scored_segment_count": sum(item["status"] == "scored" for item in segment_results),
        },
        "segments": segment_results,
    }
    apply_animation_metrics(result, features)
    output_path = rendered_segments_root / "animation_scores.json"
    write_json(output_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score keyframe animation segments from cached full-video GPA detections."
    )
    parser.add_argument("--id", "--task-id", dest="sample_id", required=True)
    parser.add_argument("--prepare-ground-truth", action="store_true",
                        help="Build shared keyframes, GPA, and segment tracks before model scoring")
    parser.add_argument("--run-name", default="gpt-6-astra")
    parser.add_argument("--clip-batch-size", type=int, default=32)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--generated-root", type=Path, default=DEFAULT_GENERATED_ROOT)
    parser.add_argument("--ground-truth-root", type=Path, default=DEFAULT_GROUND_TRUTH_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.1)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--max-box-area-ratio", type=float, default=0.5)
    parser.add_argument("--refine-box-area-ratio", type=float, default=0.12)
    parser.add_argument("--refine-box-width-ratio", type=float, default=0.55)
    parser.add_argument("--refine-box-height-ratio", type=float, default=0.35)
    parser.add_argument("--temporal-large-box-max-gap", type=int, default=4)
    parser.add_argument(
        "--recover-large-boxes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-track-gap", type=int, default=2)
    parser.add_argument("--within-video-threshold", type=float, default=0.35)
    parser.add_argument("--min-track-observations", type=int, default=2)
    parser.add_argument("--timeline-samples", type=int, default=24)
    parser.add_argument("--trajectory-distance-scale", type=float, default=0.20)
    parser.add_argument("--refresh-ground-truth", action="store_true")
    parser.add_argument("--refresh-rendered", action="store_true")
    parser.add_argument("--refresh-segments", action="store_true")
    parser.add_argument("--visualize", action="store_true",
                        help="Save GPA, trajectory, segment, and comparison videos")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.clip_batch_size < 1 or args.batch_size < 1 or args.frame_step < 1 or args.timeline_samples < 2:
        raise ValueError("batch size/frame step must be positive; timeline samples must be >= 2")
    if not 0 < args.trajectory_distance_scale <= 1:
        raise ValueError("trajectory distance scale must be in (0, 1]")
    if args.max_track_gap < 1 or args.min_track_observations < 1:
        raise ValueError("track gap and minimum observations must be positive")
    for name in ("conf", "iou", "within_video_threshold"):
        if not 0 <= getattr(args, name) <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1")
    if args.imgsz < 32 or not 0 < args.max_box_area_ratio <= 1:
        raise ValueError("Invalid GPA image size or maximum box area ratio")
    for name in (
        "refine_box_area_ratio",
        "refine_box_width_ratio",
        "refine_box_height_ratio",
    ):
        if not 0 < getattr(args, name) <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be in (0, 1]")
    if args.temporal_large_box_max_gap < 1:
        raise ValueError("--temporal-large-box-max-gap must be positive")


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r".*Torch was not compiled with flash attention.*",
        category=UserWarning,
        module=r"torch\.nn\.functional",
    )
    args = build_parser().parse_args()
    validate_args(args)
    gpa_config = GpaConfig(
        device=args.device,
        batch_size=args.batch_size,
        frame_step=args.frame_step,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        max_box_area_ratio=args.max_box_area_ratio,
        recover_large_boxes=args.recover_large_boxes,
        refine_box_area_ratio=args.refine_box_area_ratio,
        refine_box_width_ratio=args.refine_box_width_ratio,
        refine_box_height_ratio=args.refine_box_height_ratio,
        temporal_large_box_max_gap=args.temporal_large_box_max_gap,
    )
    tracking_config = TrackingConfig(
        max_track_gap=args.max_track_gap,
        within_video_threshold=args.within_video_threshold,
        min_track_observations=args.min_track_observations,
        timeline_samples=args.timeline_samples,
        trajectory_distance_scale=args.trajectory_distance_scale,
        clip_batch_size=args.clip_batch_size,
    )
    if args.prepare_ground_truth:
        prepare_ground_truth(
            str(args.sample_id), dataset_root=args.dataset_root,
            ground_truth_root=args.ground_truth_root, model=args.model,
            gpa_config=gpa_config, tracking_config=tracking_config,
            visualize=args.visualize,
            refresh_ground_truth=args.refresh_ground_truth,
            refresh_segments=args.refresh_segments,
        )
        return
    score_animation_segments(
        str(args.sample_id),
        args.run_name,
        clip_batch_size=args.clip_batch_size,
        dataset_root=args.dataset_root,
        generated_root=args.generated_root,
        ground_truth_root=args.ground_truth_root,
        model=args.model,
        gpa_config=gpa_config,
        tracking_config=tracking_config,
        refresh_ground_truth=args.refresh_ground_truth,
        refresh_rendered=args.refresh_rendered,
        refresh_segments=args.refresh_segments,
        visualize=args.visualize,
    )


if __name__ == "__main__":
    main()
