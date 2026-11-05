#!/usr/bin/env python3
"""Extract stable-state keyframes from reference videos."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MIN_STABLE_SEC = 0.20
DEFAULT_CHANGE_WINDOW_SEC = 0.20
DEFAULT_MIN_KEYFRAME_GAP_SEC = 0.50
DEFAULT_MAX_KEYFRAMES = 12
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data"
DEFAULT_GROUND_TRUTH_ROOT = PROJECT_ROOT / "experiment" / "ground_truth"
LOWRES_WIDTH = 160
LOWRES_HEIGHT = 90
GRID_COLUMNS = 6
GRID_ROWS = 4
PIXEL_CHANGE_THRESHOLD = 8
TOP_ACTIVE_CELLS = 4
LOCAL_ACTIVITY_WEIGHT = 0.20
MIN_PEAK_PROMINENCE = 0.0015
MIN_STATE_NOVELTY = 0.001

PIXEL_COUNT = LOWRES_WIDTH * LOWRES_HEIGHT
CELL_WIDTH = math.ceil(LOWRES_WIDTH / GRID_COLUMNS)
CELL_HEIGHT = math.ceil(LOWRES_HEIGHT / GRID_ROWS)
GRID_CELL_BY_PIXEL = tuple(
    min(pixel // LOWRES_WIDTH // CELL_HEIGHT, GRID_ROWS - 1) * GRID_COLUMNS
    + min(pixel % LOWRES_WIDTH // CELL_WIDTH, GRID_COLUMNS - 1)
    for pixel in range(PIXEL_COUNT)
)
GRID_CELL_PIXELS = tuple(
    GRID_CELL_BY_PIXEL.count(cell) for cell in range(GRID_COLUMNS * GRID_ROWS)
)


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float
    total_frames: int


@dataclass
class FrameAnalysis:
    index: int
    time_sec: float
    activity: float
    local_activity: float
    changed_ratio: float
    grid_activity: tuple[float, ...]
    spatial_activity: tuple[float, ...]
    pixels: bytes


@dataclass
class KeyframeCandidate:
    index: int
    reason: str
    score: float
    interval_start: int | None = None
    interval_end: int | None = None
    detected_at: int | None = None


def run_command(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=True, text=True, capture_output=True)


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required binary not found in PATH: {name}")


def parse_ratio(value: str) -> float:
    if "/" in value:
        left, right = value.split("/", 1)
        denominator = float(right)
        return float(left) / denominator if denominator else 0.0
    return float(value)


def load_video_info(video_path: Path) -> VideoInfo:
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video_path),
        ]
    )
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    fmt = payload.get("format", {})

    fps = parse_ratio(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0")
    duration = float(stream.get("duration") or fmt.get("duration") or 0.0)
    total_frames = int(stream.get("nb_frames") or 0)
    if fps <= 0 or duration <= 0:
        raise RuntimeError(f"Invalid video metadata for {video_path}")
    if total_frames <= 0:
        total_frames = max(1, int(round(duration * fps)))

    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=fps,
        duration=duration,
        total_frames=total_frames,
    )


def iter_lowres_gray_frames(video_path: Path):
    """Decode every source frame; deliberately do not use ffmpeg's fps filter."""
    frame_size = LOWRES_WIDTH * LOWRES_HEIGHT
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"scale={LOWRES_WIDTH}:{LOWRES_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={LOWRES_WIDTH}:{LOWRES_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black,format=gray",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    try:
        while True:
            chunk = process.stdout.read(frame_size)
            if not chunk:
                break
            if len(chunk) != frame_size:
                raise RuntimeError("Unexpected truncated frame stream from ffmpeg")
            yield chunk
    finally:
        process.stdout.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed while decoding video: {stderr.strip()}")


def transition_descriptor(
    previous: bytes, current: bytes
) -> tuple[float, float, float, tuple[float, ...], tuple[float, ...]]:
    total_difference = 0
    changed_pixels = 0
    grid_sums = [0] * (GRID_COLUMNS * GRID_ROWS)
    cell_width = math.ceil(LOWRES_WIDTH / GRID_COLUMNS)
    cell_height = math.ceil(LOWRES_HEIGHT / GRID_ROWS)

    for offset, (left, right) in enumerate(zip(previous, current)):
        difference = abs(left - right)
        total_difference += difference
        if difference >= PIXEL_CHANGE_THRESHOLD:
            changed_pixels += 1
        x = offset % LOWRES_WIDTH
        y = offset // LOWRES_WIDTH
        grid_x = min(x // cell_width, GRID_COLUMNS - 1)
        grid_y = min(y // cell_height, GRID_ROWS - 1)
        grid_sums[grid_y * GRID_COLUMNS + grid_x] += difference

    pixel_count = len(current)
    activity = total_difference / (pixel_count * 255.0)
    changed_ratio = changed_pixels / pixel_count
    grid_total = sum(grid_sums)
    cell_pixel_count = cell_width * cell_height
    grid_activity = tuple(value / (cell_pixel_count * 255.0) for value in grid_sums)
    local_activity = max(grid_activity)
    if grid_total:
        spatial_activity = tuple(value / grid_total for value in grid_sums)
    else:
        spatial_activity = tuple(0.0 for _ in grid_sums)
    return activity, local_activity, changed_ratio, grid_activity, spatial_activity


def moving_average(values: list[float], radius: int = 1) -> list[float]:
    if not values:
        return []
    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + value)
    result = []
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        result.append((prefix[end] - prefix[start]) / (end - start))
    return result


def collect_frame_analysis(video_path: Path, fps: float) -> list[FrameAnalysis]:
    raw: list[FrameAnalysis] = []
    previous_frame: bytes | None = None
    zero_grid = tuple(0.0 for _ in range(GRID_COLUMNS * GRID_ROWS))

    for index, frame_bytes in enumerate(iter_lowres_gray_frames(video_path)):
        if previous_frame is None:
            activity, local_activity, changed_ratio = 0.0, 0.0, 0.0
            grid_activity, spatial_activity = zero_grid, zero_grid
        else:
            activity, local_activity, changed_ratio, grid_activity, spatial_activity = (
                transition_descriptor(previous_frame, frame_bytes)
            )
        raw.append(
            FrameAnalysis(
                index,
                index / fps,
                activity,
                local_activity,
                changed_ratio,
                grid_activity,
                spatial_activity,
                frame_bytes,
            )
        )
        previous_frame = frame_bytes

    # Smooth only scalar noise. Spatial descriptors stay unsmoothed and are averaged in windows later.
    activities = moving_average([frame.activity for frame in raw])
    local_activities = moving_average([frame.local_activity for frame in raw])
    changed_ratios = moving_average([frame.changed_ratio for frame in raw])
    smoothed_grid = [
        moving_average([frame.grid_activity[cell] for frame in raw])
        for cell in range(GRID_COLUMNS * GRID_ROWS)
    ]
    return [
        FrameAnalysis(
            frame.index,
            frame.time_sec,
            activities[i],
            local_activities[i],
            changed_ratios[i],
            tuple(smoothed_grid[cell][i] for cell in range(GRID_COLUMNS * GRID_ROWS)),
            frame.spatial_activity,
            frame.pixels,
        )
        for i, frame in enumerate(raw)
    ]


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def macro_frame_distance(left: bytes, right: bytes) -> dict:
    grid_sums = [0] * (GRID_COLUMNS * GRID_ROWS)
    total = 0
    changed_pixels = 0
    for pixel, (left_value, right_value) in enumerate(zip(left, right)):
        difference = abs(left_value - right_value)
        total += difference
        grid_sums[GRID_CELL_BY_PIXEL[pixel]] += difference
        if difference >= PIXEL_CHANGE_THRESHOLD:
            changed_pixels += 1

    cell_scores = [
        grid_sums[cell] / (GRID_CELL_PIXELS[cell] * 255.0)
        for cell in range(GRID_COLUMNS * GRID_ROWS)
    ]
    global_score = total / (len(left) * 255.0)
    top_local_score = (
        sum(sorted(cell_scores, reverse=True)[:TOP_ACTIVE_CELLS])
        / TOP_ACTIVE_CELLS
    )
    return {
        "macro_score": global_score + LOCAL_ACTIVITY_WEIGHT * top_local_score,
        "global_score": global_score,
        "top_local_score": top_local_score,
        "changed_ratio": changed_pixels / len(left),
    }


def compute_macro_motion_energy(
    frames: list[FrameAnalysis],
    fps: float,
    change_window_sec: float,
    smoothing_sec: float,
) -> tuple[list[float], int, int]:
    half_window = max(2, round(change_window_sec * fps))
    smoothing_radius = max(1, round(smoothing_sec * fps / 3.0))
    energy = [0.0] * len(frames)
    for index in range(half_window, len(frames) - half_window):
        energy[index] = macro_frame_distance(
            frames[index - half_window].pixels,
            frames[index + half_window].pixels,
        )["macro_score"]
    if len(frames) > 2 * half_window:
        energy[:half_window] = [energy[half_window]] * half_window
        energy[-half_window:] = [energy[-half_window - 1]] * half_window
    return moving_average(energy, smoothing_radius), half_window, smoothing_radius


def detect_macro_peaks(
    energy: list[float], fps: float, min_gap_sec: float
) -> tuple[list[dict], float]:
    quiet_level = percentile(energy, 0.20)
    high_level = percentile(energy, 0.90)
    prominence_threshold = max(
        MIN_PEAK_PROMINENCE,
        0.05 * max(0.0, high_level - quiet_level),
    )
    peak_radius = max(2, round(0.25 * fps))
    search_radius = max(peak_radius + 1, round(1.0 * fps))
    candidates: list[dict] = []

    for index in range(peak_radius, len(energy) - peak_radius):
        if energy[index] != max(
            energy[index - peak_radius : index + peak_radius + 1]
        ):
            continue
        left_min = min(energy[max(0, index - search_radius) : index])
        right_min = min(
            energy[index + 1 : min(len(energy), index + search_radius + 1)]
        )
        prominence = energy[index] - max(left_min, right_min)
        if prominence >= prominence_threshold:
            candidates.append(
                {
                    "frame_index": index,
                    "score": energy[index],
                    "prominence": prominence,
                    "boundary_peak": False,
                }
            )

    min_gap_frames = max(1, math.ceil(min_gap_sec * fps))
    selected: list[dict] = []
    for candidate in sorted(
        candidates, key=lambda item: item["prominence"], reverse=True
    ):
        if all(
            abs(candidate["frame_index"] - existing["frame_index"])
            >= min_gap_frames
            for existing in selected
        ):
            selected.append(candidate)
    return sorted(selected, key=lambda item: item["frame_index"]), prominence_threshold


def choose_settled_valley(
    energy: list[float], start: int, end: int, peak_score: float
) -> int:
    if end <= start:
        return start
    minimum = min(energy[start : end + 1])
    tolerance = max(0.0002, 0.02 * max(0.0, peak_score - minimum))
    target = minimum + tolerance
    for index in range(start, end + 1):
        if energy[index] <= target:
            return index
    return min(range(start, end + 1), key=lambda index: energy[index])


def limit_macro_states(
    states: list[KeyframeCandidate], max_keyframes: int
) -> list[KeyframeCandidate]:
    if len(states) <= max_keyframes:
        return states
    first = states[0]
    last = states[-1]
    interior = sorted(states[1:-1], key=lambda item: item.score, reverse=True)
    selected = [first, last] + interior[: max(0, max_keyframes - 2)]
    return sorted({item.index: item for item in selected}.values(), key=lambda item: item.index)


def consolidate_macro_states(
    states: list[KeyframeCandidate],
    frames: list[FrameAnalysis],
    fps: float,
    min_gap_sec: float,
) -> tuple[list[KeyframeCandidate], list[dict]]:
    """Collapse adjacent candidates that represent the same visual state.

    Peak spacing alone is insufficient: settled valleys derived from different
    peaks can still land a few frames apart, especially near the end of a
    video.  State selection therefore has its own visual-novelty pass.  For a
    near-duplicate pair we retain the later frame when the pair is temporally
    close (it is normally the more complete state), otherwise the earlier one
    already represents the persistent state and is retained.
    """
    if not states:
        return [], []

    min_gap_frames = max(1, math.ceil(min_gap_sec * fps))
    consolidated = [states[0]]
    rejected: list[dict] = []
    for candidate in states[1:]:
        previous = consolidated[-1]
        distance = macro_frame_distance(
            frames[previous.index].pixels,
            frames[candidate.index].pixels,
        )
        if distance["macro_score"] >= MIN_STATE_NOVELTY:
            consolidated.append(candidate)
            continue

        close_in_time = candidate.index - previous.index < min_gap_frames
        keep_candidate = close_in_time and previous.index != 0
        kept = candidate if keep_candidate else previous
        dropped = previous if keep_candidate else candidate
        if keep_candidate:
            consolidated[-1] = candidate
        rejected.append(
            {
                "dropped_frame_index": dropped.index,
                "kept_frame_index": kept.index,
                "frame_gap": candidate.index - previous.index,
                "macro_score": distance["macro_score"],
                "changed_ratio": distance["changed_ratio"],
                "reason": "adjacent_state_below_visual_novelty",
            }
        )
    return consolidated, rejected


def detect_keyframes(
    frames: list[FrameAnalysis],
    fps: float,
    min_stable_sec: float,
    change_window_sec: float,
    min_keyframe_gap_sec: float,
    max_keyframes: int,
) -> tuple[list[KeyframeCandidate], dict]:
    if not frames:
        return [], {}
    energy, half_window, smoothing_radius = compute_macro_motion_energy(
        frames, fps, change_window_sec, min_stable_sec
    )
    peaks, prominence_threshold = detect_macro_peaks(
        energy, fps, min_keyframe_gap_sec
    )

    states = [KeyframeCandidate(0, "video_initial_state", float("inf"))]
    for left_peak, right_peak in zip(peaks, peaks[1:]):
        valley = choose_settled_valley(
            energy,
            left_peak["frame_index"],
            right_peak["frame_index"],
            left_peak["score"],
        )
        states.append(
            KeyframeCandidate(
                valley,
                "settled_after_macro_change",
                left_peak["prominence"],
                detected_at=left_peak["frame_index"],
            )
        )
    if peaks:
        last_peak = peaks[-1]
        states.append(
            KeyframeCandidate(
                len(frames) - 1,
                "final_state_after_macro_change",
                last_peak["prominence"],
                detected_at=last_peak["frame_index"],
            )
        )

    deduplicated: list[KeyframeCandidate] = []
    for state in sorted(states, key=lambda item: item.index):
        if deduplicated and state.index == deduplicated[-1].index:
            if state.score > deduplicated[-1].score:
                deduplicated[-1] = state
            continue
        deduplicated.append(state)

    if deduplicated[-1].index != len(frames) - 1:
        final_novelty = macro_frame_distance(
            frames[deduplicated[-1].index].pixels, frames[-1].pixels
        )["macro_score"]
        if final_novelty >= max(0.006, 2.0 * prominence_threshold):
            deduplicated.append(
                KeyframeCandidate(
                    len(frames) - 1,
                    "persistent_final_state",
                    final_novelty,
                )
            )

    consolidated, rejected_states = consolidate_macro_states(
        deduplicated,
        frames,
        fps,
        min_keyframe_gap_sec,
    )
    chosen = limit_macro_states(consolidated, max_keyframes)
    if len(chosen) < 2 and len(frames) >= 2:
        # A static or subtle animation can have no detectable macro change.
        # Downstream segment scoring still needs a nonempty time interval.
        chosen = [
            KeyframeCandidate(0, "fallback_video_start", float("inf")),
            KeyframeCandidate(len(frames) - 1, "fallback_video_end", 0.0),
        ]
    return chosen, {
        "macro_motion_energy": energy,
        "macro_peaks": peaks,
        "peak_prominence_threshold": prominence_threshold,
        "half_window_frames": half_window,
        "smoothing_radius_frames": smoothing_radius,
        "min_gap_frames": max(1, math.ceil(min_keyframe_gap_sec * fps)),
        "min_state_novelty": MIN_STATE_NOVELTY,
        "rejected_states": rejected_states,
    }


def save_keyframe(video_path: Path, frame_index: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"select=eq(n\\,{frame_index})",
        "-frames:v",
        "1",
        "-q:v",
        "2",
        "-y",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def build_segments(
    keyframes: list[KeyframeCandidate],
    frames: list[FrameAnalysis],
    macro_motion_energy: list[float],
) -> list[dict]:
    segments = []
    for left, right in zip(keyframes, keyframes[1:]):
        peak_frame = max(
            range(left.index, right.index + 1),
            key=lambda index: macro_motion_energy[index],
        )
        visual_change = macro_frame_distance(
            frames[left.index].pixels, frames[right.index].pixels
        )
        segments.append(
            {
                "from_frame_index": left.index,
                "to_frame_index": right.index,
                "from_time_sec": round(frames[left.index].time_sec, 3),
                "to_time_sec": round(frames[right.index].time_sec, 3),
                "duration_sec": round(
                    frames[right.index].time_sec - frames[left.index].time_sec, 3
                ),
                "motion_peak_frame": peak_frame,
                "motion_peak_time_sec": round(frames[peak_frame].time_sec, 3),
                "motion_peak_energy": round(macro_motion_energy[peak_frame], 6),
                "macro_visual_change": round(visual_change["macro_score"], 6),
                "global_visual_change": round(visual_change["global_score"], 6),
                "changed_ratio": round(visual_change["changed_ratio"], 6),
            }
        )
    return segments


def extract_keyframes(
    video_path: Path,
    output_dir: Path,
    min_stable_sec: float,
    change_window_sec: float,
    min_keyframe_gap_sec: float,
    max_keyframes: int,
) -> dict:
    require_binary("ffmpeg")
    require_binary("ffprobe")
    video_info = load_video_info(video_path)
    frames = collect_frame_analysis(video_path, video_info.fps)
    keyframes, debug = detect_keyframes(
        frames,
        video_info.fps,
        min_stable_sec,
        change_window_sec,
        min_keyframe_gap_sec,
        max_keyframes,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for old_image in output_dir.glob("keyframe_*.jpg"):
        old_image.unlink()

    keyframes_payload = []
    for order, candidate in enumerate(keyframes):
        frame = frames[candidate.index]
        filename = f"keyframe_{order:02d}_frame_{frame.index:05d}_{frame.time_sec:06.3f}s.jpg"
        output_path = output_dir / filename
        save_keyframe(video_path, frame.index, output_path)
        item = {
            "order": order,
            "frame_index": frame.index,
            "time_sec": round(frame.time_sec, 3),
            "reason": candidate.reason,
            "activity": round(frame.activity, 6),
            "local_activity": round(frame.local_activity, 6),
            "selection_score": None if math.isinf(candidate.score) else round(candidate.score, 6),
            "motion_energy": round(debug["macro_motion_energy"][candidate.index], 6),
            "preceding_motion_peak_frame": candidate.detected_at,
            "image_path": str(output_path),
        }
        keyframes_payload.append(item)

    payload = {
        "video_path": str(video_path),
        "video_info": {
            "width": video_info.width,
            "height": video_info.height,
            "fps": video_info.fps,
            "duration_sec": video_info.duration,
            "total_frames": video_info.total_frames,
            "analyzed_frames": len(frames),
        },
        "parameters": {
            "analyze_every_frame": True,
            "selection_mode": "persistent_macro_states",
            "min_stable_sec": min_stable_sec,
            "change_window_sec": change_window_sec,
            "min_keyframe_gap_sec": min_keyframe_gap_sec,
            "max_keyframes": max_keyframes,
            "lowres_size": [LOWRES_WIDTH, LOWRES_HEIGHT],
            "macro_grid": [GRID_COLUMNS, GRID_ROWS],
            "top_active_cells": TOP_ACTIVE_CELLS,
            "local_activity_weight": LOCAL_ACTIVITY_WEIGHT,
            "min_state_novelty": debug["min_state_novelty"],
        },
        "analysis": {
            "half_window_frames": debug["half_window_frames"],
            "smoothing_radius_frames": debug["smoothing_radius_frames"],
            "min_gap_frames": debug["min_gap_frames"],
            "peak_prominence_threshold": round(
                debug["peak_prominence_threshold"], 6
            ),
            "motion_peaks": [
                {
                    "frame_index": peak["frame_index"],
                    "time_sec": round(frames[peak["frame_index"]].time_sec, 3),
                    "score": round(peak["score"], 6),
                    "prominence": round(peak["prominence"], 6),
                    "boundary_peak": peak["boundary_peak"],
                }
                for peak in debug["macro_peaks"]
            ],
            "rejected_states": [
                {
                    **item,
                    "macro_score": round(item["macro_score"], 6),
                    "changed_ratio": round(item["changed_ratio"], 6),
                }
                for item in debug["rejected_states"]
            ],
            "frames": [
                {
                    "frame_index": frame.index,
                    "time_sec": round(frame.time_sec, 3),
                    "activity": round(frame.activity, 6),
                    "local_activity": round(frame.local_activity, 6),
                    "changed_ratio": round(frame.changed_ratio, 6),
                    "macro_motion_energy": round(
                        debug["macro_motion_energy"][frame.index], 6
                    ),
                }
                for frame in frames
            ],
        },
        "keyframes": keyframes_payload,
        "segments": build_segments(
            keyframes, frames, debug["macro_motion_energy"]
        ),
    }
    (output_dir / "keyframes.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload


def reference_cache_signature(video_path: Path) -> dict:
    stat = video_path.stat()
    return {
        "schema_version": 1,
        "source_size_bytes": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
    }


@contextmanager
def reference_cache_lock(reference_dir: Path, timeout_sec: float = 600.0):
    """Serialize cache creation across extraction and matching processes."""
    reference_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = reference_dir.with_name(f"{reference_dir.name}.lock")
    started = time.monotonic()
    descriptor = None
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
                raise TimeoutError(f"Timed out waiting for ground-truth cache lock: {lock_path}")
            time.sleep(0.25)
    try:
        yield
    finally:
        os.close(descriptor)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def load_reference_cache(
    reference_video: Path,
    reference_dir: Path,
    min_stable_sec: float,
    change_window_sec: float,
    min_keyframe_gap_sec: float,
    max_keyframes: int,
) -> dict | None:
    """Return a complete cache compatible with keyframe_alignment.py, or None."""
    metadata_path = reference_dir / "keyframes.json"
    if not metadata_path.is_file():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        cached_video = Path(payload["video_path"]).expanduser().resolve()
        if cached_video != reference_video.resolve():
            return None
        if payload.get("cache") != reference_cache_signature(reference_video):
            return None
        parameters = payload["parameters"]
        expected_parameters = {
            "min_stable_sec": min_stable_sec,
            "change_window_sec": change_window_sec,
            "min_keyframe_gap_sec": min_keyframe_gap_sec,
            "max_keyframes": max_keyframes,
        }
        if any(parameters.get(key) != value for key, value in expected_parameters.items()):
            return None
        video_info = load_video_info(reference_video)
        cached_info = payload["video_info"]
        if (
            int(cached_info["width"]) != video_info.width
            or int(cached_info["height"]) != video_info.height
            or int(cached_info["total_frames"]) != video_info.total_frames
            or not math.isclose(
                float(cached_info["fps"]), video_info.fps, rel_tol=1e-5, abs_tol=1e-8
            )
            or not math.isclose(
                float(cached_info["duration_sec"]), video_info.duration,
                rel_tol=1e-5, abs_tol=1e-8,
            )
        ):
            return None
        keyframes = payload["keyframes"]
        if not isinstance(keyframes, list) or len(keyframes) < min(2, video_info.total_frames):
            return None
        for item in keyframes:
            image_path = Path(item["image_path"]).expanduser()
            if not image_path.is_absolute():
                image_path = reference_dir / image_path
            if not image_path.is_file():
                return None
            item["image_path"] = str(image_path.resolve())
        return payload
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def get_reference_keyframes(
    reference_video: Path,
    reference_dir: Path,
    min_stable_sec: float,
    change_window_sec: float,
    min_keyframe_gap_sec: float,
    max_keyframes: int,
    refresh: bool,
) -> tuple[dict, bool]:
    """Load a valid shared cache or extract and record a new one."""
    if not refresh:
        cached = load_reference_cache(
            reference_video, reference_dir, min_stable_sec, change_window_sec,
            min_keyframe_gap_sec, max_keyframes,
        )
        if cached is not None:
            return cached, True
    with reference_cache_lock(reference_dir):
        if not refresh:
            cached = load_reference_cache(
                reference_video, reference_dir, min_stable_sec, change_window_sec,
                min_keyframe_gap_sec, max_keyframes,
            )
            if cached is not None:
                return cached, True
        payload = extract_keyframes(
            reference_video, reference_dir, min_stable_sec, change_window_sec,
            min_keyframe_gap_sec, max_keyframes,
        )
        payload["cache"] = reference_cache_signature(reference_video)
        (reference_dir / "keyframes.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return payload, False


def parse_sample_ids(specification: str) -> list[str]:
    """Expand comma-separated IDs and inclusive ranges, preserving order."""
    ids: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"[,，]", specification):
        token = part.strip()
        match = re.fullmatch(r"([0-9]+)(?:\s*-\s*([0-9]+))?", token)
        if match is None:
            raise ValueError(f"Invalid --id item: {token!r}")
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) is not None else start
        if start < 1 or end < start:
            raise ValueError(f"Invalid --id range: {token!r}")
        for number in range(start, end + 1):
            sample_id = str(number)
            if sample_id not in seen:
                ids.append(sample_id)
                seen.add(sample_id)
    return ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract video keyframes, or precompute shared ground-truth keyframes by ID."
    )
    parser.add_argument("video", nargs="?", type=Path, help="Path to input video")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Folder for keyframe images and keyframes.json. Defaults to <video_dir>/keyframes",
    )
    parser.add_argument(
        "--min-stable-sec",
        type=float,
        default=DEFAULT_MIN_STABLE_SEC,
        help="Temporal smoothing horizon for settled-state detection",
    )
    parser.add_argument(
        "--change-window-sec",
        type=float,
        default=DEFAULT_CHANGE_WINDOW_SEC,
        help="Half-window used to compare the visual states before and after each frame",
    )
    parser.add_argument(
        "--min-keyframe-gap-sec",
        type=float,
        default=DEFAULT_MIN_KEYFRAME_GAP_SEC,
        help="Minimum separation between macro-motion peaks",
    )
    parser.add_argument("--max-keyframes", type=int, default=DEFAULT_MAX_KEYFRAMES)
    parser.add_argument(
        "--task-id", "--id",
        dest="sample_ids",
        help="Ground-truth IDs, e.g. 1, 1-124, or 1,3-5",
    )
    parser.add_argument(
        "--data", "--dataset-root",
        dest="dataset_root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root containing <id>/ground_truth.mp4",
    )
    parser.add_argument(
        "--ground-truth-root",
        type=Path,
        default=DEFAULT_GROUND_TRUTH_ROOT,
        help="Shared keyframe cache root; writes <id>/keyframes/",
    )
    parser.add_argument(
        "--refresh-ground-truth",
        action="store_true",
        help="Rebuild selected ground-truth caches even when valid",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.max_keyframes < 2:
        parser.error("--max-keyframes must be at least 2")

    if args.sample_ids is not None:
        if args.video is not None or args.output_dir is not None:
            parser.error("Use --task-id without a video path or --output-dir")
        try:
            sample_ids = parse_sample_ids(args.sample_ids)
        except ValueError as exc:
            parser.error(str(exc))
        dataset_root = args.dataset_root.expanduser().resolve()
        ground_truth_root = args.ground_truth_root.expanduser().resolve()
        failures = []
        for sample_id in sample_ids:
            video = dataset_root / sample_id / "ground_truth.mp4"
            output_dir = ground_truth_root / sample_id / "keyframes"
            try:
                if not video.is_file():
                    raise FileNotFoundError(f"Ground-truth video not found: {video}")
                payload, reused = get_reference_keyframes(
                    video, output_dir, args.min_stable_sec, args.change_window_sec,
                    args.min_keyframe_gap_sec, args.max_keyframes,
                    args.refresh_ground_truth,
                )
                print(
                    f"ID {sample_id}: {'reused' if reused else 'extracted'} "
                    f"{len(payload['keyframes'])} keyframes -> {output_dir}",
                    flush=True,
                )
            except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
                failures.append(sample_id)
                print(f"ID {sample_id}: failed: {exc}", flush=True)
        print(f"Completed {len(sample_ids) - len(failures)}/{len(sample_ids)} IDs", flush=True)
        return 1 if failures else 0

    if args.video is None:
        parser.error("Provide a video path or use --task-id")

    video_path = args.video.resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = args.output_dir.resolve() if args.output_dir else video_path.parent / "keyframes"
    payload = extract_keyframes(
        video_path,
        output_dir,
        args.min_stable_sec,
        args.change_window_sec,
        args.min_keyframe_gap_sec,
        args.max_keyframes,
    )

    print(f"Video: {video_path}")
    print(f"Output: {output_dir}")
    print(f"Analyzed every frame: {payload['video_info']['analyzed_frames']}")
    print(f"Keyframes: {len(payload['keyframes'])}")
    for item in payload["keyframes"]:
        print(
            f"  - #{item['order']:02d} frame={item['frame_index']:05d} "
            f"time={item['time_sec']:.3f}s reason={item['reason']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
