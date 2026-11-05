"""Bounding-box recovery, pixel-evidence refinement, and online stabilization."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

@dataclass(frozen=True)
class RecoveryConfig:
    color_distance: float = 22.0
    bgr_tolerance: int = 24
    min_border_inlier_ratio: float = 0.55
    min_global_background_ratio: float = 0.30
    padding_ratio: float = 0.20
    min_component_area_ratio: float = 0.00003
    min_foreground_fill_ratio: float = 0.01
    # Bold display text (for example a solid black ``00``) can legitimately
    # occupy most of its tight box.  Dense regions are still protected by the
    # area-shrink and temporal checks below, so this must not be used as the
    # only photo/text discriminator.
    max_foreground_fill_ratio: float = 0.85


PIPELINE_VERSION = "conservative_refinement_locked_stabilization_v7"


@dataclass
class LargeBoxTrack:
    track_id: int
    bbox: np.ndarray
    previous_bbox: np.ndarray | None
    last_frame_index: int
    confidence: float
    class_id: int
    class_name: str
    appearance: np.ndarray | None


@dataclass
class DetectionTrack:
    """Short-lived state used only to remove detector bbox jitter."""

    track_id: int
    raw_bbox: np.ndarray
    stable_bbox: np.ndarray
    velocity: np.ndarray
    last_frame_index: int
    class_id: int
    age: int = 1
    recent_raw_boxes: list[np.ndarray] = field(default_factory=list)
    locked_bbox: np.ndarray | None = None


def _flat_background_mask(
    frame: np.ndarray, config: RecoveryConfig
) -> tuple[np.ndarray | None, dict]:
    """Build a foreground mask when the image border supports one background."""
    height, width = frame.shape[:2]
    longest = max(height, width)
    scale = min(1.0, 384.0 / longest)
    sample = (
        cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else frame
    )
    sample_lab = cv2.cvtColor(sample, cv2.COLOR_BGR2LAB).astype(np.float32)
    sample_height, sample_width = sample.shape[:2]
    band = max(2, round(min(sample_height, sample_width) * 0.025))
    border = np.concatenate(
        (
            sample_lab[:band].reshape(-1, 3),
            sample_lab[-band:].reshape(-1, 3),
            sample_lab[band:-band, :band].reshape(-1, 3),
            sample_lab[band:-band, -band:].reshape(-1, 3),
        ),
        axis=0,
    )
    background_lab = np.median(border, axis=0)
    border_distance = np.linalg.norm(border - background_lab, axis=1)
    sample_distance = np.linalg.norm(sample_lab - background_lab, axis=2)
    border_inlier_ratio = float(np.mean(border_distance <= config.color_distance))
    global_background_ratio = float(
        np.mean(sample_distance <= config.color_distance)
    )
    background_bgr = cv2.cvtColor(
        np.uint8([[background_lab]]), cv2.COLOR_LAB2BGR
    )[0, 0]
    diagnostics = {
        "border_inlier_ratio": round(border_inlier_ratio, 4),
        "global_background_ratio": round(global_background_ratio, 4),
        "background_bgr": [int(value) for value in background_bgr],
    }
    if (
        border_inlier_ratio < config.min_border_inlier_ratio
        or global_background_ratio < config.min_global_background_ratio
    ):
        return None, diagnostics

    # The expensive classification above runs only on a small image. Once a
    # flat background is confirmed, an OpenCV-native BGR interval is enough to
    # recover exact full-resolution bounds without large float allocations.
    background_i16 = background_bgr.astype(np.int16)
    lower = np.clip(background_i16 - config.bgr_tolerance, 0, 255).astype(np.uint8)
    upper = np.clip(background_i16 + config.bgr_tolerance, 0, 255).astype(np.uint8)
    background_mask = cv2.inRange(frame, lower, upper)
    mask = cv2.bitwise_not(background_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask, diagnostics


def _bbox_metrics(box: np.ndarray, width: int, height: int) -> dict[str, float]:
    box_width = max(0.0, float(box[2] - box[0]))
    box_height = max(0.0, float(box[3] - box[1]))
    return {
        "width_ratio": box_width / width,
        "height_ratio": box_height / height,
        "area_ratio": box_width * box_height / (width * height),
        "aspect_ratio": box_width / max(1e-6, box_height),
    }


def _bbox_intersection(left: np.ndarray, right: np.ndarray) -> float:
    x1, y1 = np.maximum(left[:2], right[:2])
    x2, y2 = np.minimum(left[2:], right[2:])
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def _bbox_area(box: np.ndarray) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def _bbox_union_area(boxes: list[np.ndarray]) -> float:
    """Return the exact union area of a small set of axis-aligned boxes."""
    if not boxes:
        return 0.0
    x_edges = sorted({float(x) for box in boxes for x in (box[0], box[2])})
    area = 0.0
    for left, right in zip(x_edges, x_edges[1:]):
        if right <= left:
            continue
        intervals = sorted(
            (float(box[1]), float(box[3]))
            for box in boxes
            if float(box[0]) < right and float(box[2]) > left
        )
        if not intervals:
            continue
        interval_start, interval_end = intervals[0]
        covered_height = 0.0
        for start, end in intervals[1:]:
            if start > interval_end:
                covered_height += max(0.0, interval_end - interval_start)
                interval_start, interval_end = start, end
            else:
                interval_end = max(interval_end, end)
        covered_height += max(0.0, interval_end - interval_start)
        area += (right - left) * covered_height
    return area


def _bbox_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = _bbox_intersection(left, right)
    union = _bbox_area(left) + _bbox_area(right) - intersection
    return intersection / union if union > 1e-8 else 0.0


def _bbox_center(box: np.ndarray) -> np.ndarray:
    return np.asarray([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def _expand_bbox(
    box: np.ndarray, padding: float, width: int, height: int
) -> np.ndarray:
    box_width = max(1.0, float(box[2] - box[0]))
    box_height = max(1.0, float(box[3] - box[1]))
    return np.asarray(
        [
            max(0.0, box[0] - box_width * padding),
            max(0.0, box[1] - box_height * padding),
            min(float(width), box[2] + box_width * padding),
            min(float(height), box[3] + box_height * padding),
        ],
        dtype=np.float32,
    )


def _clip_bbox(box: np.ndarray, width: int, height: int) -> np.ndarray:
    result = box.astype(np.float32).copy()
    result[[0, 2]] = np.clip(result[[0, 2]], 0, width)
    result[[1, 3]] = np.clip(result[[1, 3]], 0, height)
    if result[2] < result[0]:
        result[0], result[2] = result[2], result[0]
    if result[3] < result[1]:
        result[1], result[3] = result[3], result[1]
    return result


def _appearance_histogram(frame: np.ndarray, box: np.ndarray) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1 = max(0, min(width - 1, int(np.floor(box[0]))))
    y1 = max(0, min(height - 1, int(np.floor(box[1]))))
    x2 = max(x1 + 1, min(width, int(np.ceil(box[2]))))
    y2 = max(y1 + 1, min(height, int(np.ceil(box[3]))))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256]).ravel()
    total = float(histogram.sum())
    return histogram / total if total > 0 else None


def _appearance_similarity(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return 0.5
    return float(np.minimum(left, right).sum())


def _component_gap(left: np.ndarray, right: np.ndarray) -> float:
    horizontal = max(0.0, float(max(left[0], right[0]) - min(left[2], right[2])))
    vertical = max(0.0, float(max(left[1], right[1]) - min(left[3], right[3])))
    return float(np.hypot(horizontal, vertical))


def _recover_from_local_mask(
    mask: np.ndarray,
    anchor_box: np.ndarray,
    search_box: np.ndarray,
    max_area_ratio: float,
    config: RecoveryConfig,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Tighten foreground around an anchor without merging the whole page."""
    height, width = mask.shape
    left, top, right, bottom = (
        int(np.floor(search_box[0])),
        int(np.floor(search_box[1])),
        int(np.ceil(search_box[2])),
        int(np.ceil(search_box[3])),
    )
    left, top = max(0, left), max(0, top)
    right, bottom = min(width, right), min(height, bottom)
    diagnostics: dict[str, Any] = {
        "search_bbox_xyxy": [left, top, right, bottom],
        "candidate_component_count": 0,
        "selected_component_count": 0,
    }
    if right <= left or bottom <= top:
        diagnostics["failure_reason"] = "empty_search_region"
        return None, diagnostics

    roi = mask[top:bottom, left:right]
    count, _, stats, _ = cv2.connectedComponentsWithStats(roi, 8)
    min_component_area = max(
        16, round(width * height * config.min_component_area_ratio)
    )
    components: list[tuple[np.ndarray, int]] = []
    for component_index in range(1, count):
        cx, cy, cw, ch, area = (int(value) for value in stats[component_index])
        if area < min_component_area or cw < 2 or ch < 2:
            continue
        component_box = np.asarray(
            [left + cx, top + cy, left + cx + cw, top + cy + ch],
            dtype=np.float32,
        )
        components.append((component_box, area))
    diagnostics["candidate_component_count"] = len(components)
    if not components:
        diagnostics["failure_reason"] = "no_foreground_components"
        return None, diagnostics

    # Components overlapping the unpadded anchor are seeds. Nearby components
    # are attached iteratively so a slightly clipped first/last glyph can be
    # recovered, while distant scrollbars and lower-page imagery stay out.
    seeds = [item for item in components if _bbox_intersection(item[0], anchor_box) > 0]
    if not seeds:
        diagnostics["failure_reason"] = "no_component_overlaps_anchor"
        return None, diagnostics
    selected = list(seeds)
    seed_ids = {id(item) for item in seeds}
    remaining = [item for item in components if id(item) not in seed_ids]
    anchor_short_side = min(
        max(1.0, float(anchor_box[2] - anchor_box[0])),
        max(1.0, float(anchor_box[3] - anchor_box[1])),
    )
    # A large image can have a very large short side.  Without an upper bound,
    # an unrelated nearby label (sample 10's ``(12)``) was attached to the
    # image once their gap became smaller than 14% of the image size.  Text
    # glyph gaps need a relative allowance, but never a page-scale one.
    attach_distance = max(10.0, min(48.0, anchor_short_side * 0.12))
    changed = True
    while changed and remaining:
        changed = False
        for item in list(remaining):
            compatible_neighbor = any(
                _component_gap(item[0], chosen[0]) <= attach_distance
                and min(item[1], chosen[1]) / max(item[1], chosen[1]) >= 0.04
                for chosen in selected
            )
            if compatible_neighbor:
                selected.append(item)
                remaining = [other for other in remaining if id(other) != id(item)]
                changed = True

    recovered_box = np.asarray(
        [
            min(item[0][0] for item in selected),
            min(item[0][1] for item in selected),
            max(item[0][2] for item in selected),
            max(item[0][3] for item in selected),
        ],
        dtype=np.float32,
    )
    recovered_box = _expand_bbox(recovered_box, 0.006, width, height)
    recovered_area = _bbox_area(recovered_box)
    recovered_area_ratio = recovered_area / (width * height)
    foreground_pixels = sum(item[1] for item in selected)
    foreground_fill_ratio = foreground_pixels / max(1.0, recovered_area)
    diagnostics.update(
        {
            "selected_component_count": len(selected),
            "recovered_bbox_xyxy": [round(float(value), 2) for value in recovered_box],
            "recovered_area_ratio": round(recovered_area_ratio, 6),
            "foreground_fill_ratio": round(float(foreground_fill_ratio), 6),
        }
    )
    if recovered_area_ratio > max_area_ratio:
        diagnostics["failure_reason"] = "recovered_box_still_oversized"
        return None, diagnostics
    if not (
        config.min_foreground_fill_ratio
        <= foreground_fill_ratio
        <= config.max_foreground_fill_ratio
    ):
        diagnostics["failure_reason"] = "foreground_fill_out_of_range"
        return None, diagnostics
    return recovered_box, diagnostics


class DetectionStabilizer:
    """Online bbox stabilizer that preserves deliberate motion.

    GPA is normally very consistent semantically, but its four regressed box
    edges move by a few pixels on otherwise identical frames.  Tracks are
    associated only by strong spatial overlap.  A velocity-aware low-pass
    filter is applied when the innovation is small; large changes pass through
    untouched, so reveal, scale and translation animations are not frozen.
    Missing detections are never synthesized here.
    """

    def __init__(self, *, frame_step: int = 1, max_track_gap: int = 2) -> None:
        self.frame_step = max(1, frame_step)
        self.max_track_gap = max(1, max_track_gap) * self.frame_step
        self.tracks: dict[int, DetectionTrack] = {}
        self.next_track_id = 0

    @staticmethod
    def _association_score(
        box: np.ndarray, track: DetectionTrack, width: int, height: int
    ) -> float:
        overlap = _bbox_iou(box, track.raw_bbox)
        center_distance = float(
            np.linalg.norm(
                (_bbox_center(box) - _bbox_center(track.raw_bbox)) / [width, height]
            )
        )
        if overlap < 0.25 and center_distance > 0.012:
            return 0.0
        return 0.85 * overlap + 0.15 * max(0.0, 1.0 - center_distance / 0.04)

    def _new_track(
        self, detection: dict, box: np.ndarray, frame_index: int
    ) -> DetectionTrack:
        track = DetectionTrack(
            track_id=self.next_track_id,
            raw_bbox=box.copy(),
            stable_bbox=box.copy(),
            velocity=np.zeros(4, dtype=np.float32),
            last_frame_index=frame_index,
            class_id=int(detection.get("class_id", -1)),
            recent_raw_boxes=[box.copy()],
        )
        self.tracks[track.track_id] = track
        self.next_track_id += 1
        return track

    def _stabilize(
        self,
        track: DetectionTrack,
        raw_box: np.ndarray,
        frame_index: int,
    ) -> tuple[np.ndarray, bool, float]:
        elapsed = max(1, frame_index - track.last_frame_index)
        raw_delta = (raw_box - track.raw_bbox) / elapsed
        previous_width = max(1.0, float(track.stable_bbox[2] - track.stable_bbox[0]))
        previous_height = max(1.0, float(track.stable_bbox[3] - track.stable_bbox[1]))
        short_side = min(previous_width, previous_height)
        edge_noise = min(6.0, max(2.0, short_side * 0.04))
        center_shift = float(np.linalg.norm(
            _bbox_center(raw_box) - _bbox_center(track.raw_bbox)
        ))
        width_change = abs(
            float((raw_box[2] - raw_box[0]) - previous_width)
        )
        height_change = abs(
            float((raw_box[3] - raw_box[1]) - previous_height)
        )

        # Once several measurements agree that an element is stationary, keep
        # an exact bbox instead of perpetually applying an EMA.  This removes
        # the remaining sub-pixel crawl visible in annotated videos.  A change
        # larger than the release gate immediately unlocks the track.
        if track.locked_bbox is not None:
            locked_width = max(
                1.0, float(track.locked_bbox[2] - track.locked_bbox[0])
            )
            locked_height = max(
                1.0, float(track.locked_bbox[3] - track.locked_bbox[1])
            )
            release_center = float(np.linalg.norm(
                _bbox_center(raw_box) - _bbox_center(track.locked_bbox)
            ))
            release_width = abs(float((raw_box[2] - raw_box[0]) - locked_width))
            release_height = abs(float((raw_box[3] - raw_box[1]) - locked_height))
            if (
                release_center <= edge_noise * 1.75
                and release_width <= edge_noise * 3.0
                and release_height <= edge_noise * 3.0
            ):
                stable = track.locked_bbox.copy()
                track.raw_bbox = raw_box.copy()
                track.stable_bbox = stable
                track.velocity.fill(0.0)
                track.last_frame_index = frame_index
                track.age += 1
                track.recent_raw_boxes.append(raw_box.copy())
                track.recent_raw_boxes = track.recent_raw_boxes[-5:]
                return stable, True, float(np.max(np.abs(stable - raw_box)))
            track.locked_bbox = None
            track.recent_raw_boxes = [track.raw_bbox.copy()]

        same_direction = np.sign(raw_delta) == np.sign(track.velocity)
        filtered_velocity = np.where(
            same_direction,
            0.65 * track.velocity + 0.35 * raw_delta,
            0.15 * track.velocity + 0.25 * raw_delta,
        ).astype(np.float32)
        predicted = track.stable_bbox + filtered_velocity * elapsed
        innovation = raw_box - predicted

        # This gate is deliberately expressed in element pixels rather than
        # frame pixels.  It smooths a 2--6 px regression wobble but immediately
        # releases an actually moving/scaling element.
        jitter_like = (
            center_shift <= edge_noise
            and width_change <= edge_noise * 2.5
            and height_change <= edge_noise * 2.5
            and float(np.max(np.abs(innovation))) <= edge_noise * 2.5
        )
        if jitter_like:
            alpha = 0.28 if track.age >= 2 else 0.45
            stable = predicted + alpha * innovation
            # Raw velocity carries true steady motion; an immediate direction
            # reversal is much more likely to be detector noise, so its old
            # momentum is discarded quickly instead of overshooting.
            track.velocity = filtered_velocity
            track.recent_raw_boxes.append(raw_box.copy())
            track.recent_raw_boxes = track.recent_raw_boxes[-5:]

            # Do not lock a genuinely slow animation.  Consistent monotonic
            # movement across the recent window is treated as motion even if
            # every individual step is small.
            if len(track.recent_raw_boxes) >= 4:
                recent = np.stack(track.recent_raw_boxes[-5:])
                centers = np.stack([_bbox_center(box) for box in recent])
                sizes = recent[:, 2:] - recent[:, :2]
                signals = np.concatenate((centers, sizes), axis=1)
                deltas = np.diff(signals, axis=0)
                displacement = np.abs(signals[-1] - signals[0])
                edge_spread = np.ptp(recent, axis=0)
                monotonic = np.logical_or(
                    np.all(deltas > 0.30, axis=0),
                    np.all(deltas < -0.30, axis=0),
                )
                trending = bool(
                    np.any(monotonic & (displacement > edge_noise * 0.75))
                )
                compact = bool(np.max(edge_spread) <= edge_noise * 0.75)
                if not trending and compact:
                    track.locked_bbox = np.median(recent, axis=0).astype(
                        np.float32
                    )
                    stable = track.locked_bbox.copy()
                    track.velocity.fill(0.0)
        else:
            stable = raw_box.copy()
            track.velocity = 0.35 * raw_delta
            track.recent_raw_boxes = [raw_box.copy()]

        track.raw_bbox = raw_box.copy()
        track.stable_bbox = stable.astype(np.float32)
        track.last_frame_index = frame_index
        track.age += 1
        adjustment = float(np.max(np.abs(stable - raw_box)))
        return track.stable_bbox.copy(), jitter_like, adjustment

    def process(
        self,
        detections: list[dict],
        frame_index: int,
        width: int,
        height: int,
    ) -> tuple[list[dict], dict[str, Any]]:
        active = [
            track
            for track in self.tracks.values()
            if frame_index - track.last_frame_index <= self.max_track_gap
        ]
        ranked: list[tuple[float, int, DetectionTrack]] = []
        boxes = [
            np.asarray(detection["bbox_xyxy"], dtype=np.float32)
            for detection in detections
        ]
        for detection_index, (detection, box) in enumerate(zip(detections, boxes)):
            class_id = int(detection.get("class_id", -1))
            for track in active:
                if track.class_id != class_id:
                    continue
                score = self._association_score(box, track, width, height)
                if score > 0.0:
                    ranked.append((score, detection_index, track))

        matches: dict[int, DetectionTrack] = {}
        used_tracks: set[int] = set()
        for _, detection_index, track in sorted(
            ranked, key=lambda item: item[0], reverse=True
        ):
            if detection_index in matches or track.track_id in used_tracks:
                continue
            matches[detection_index] = track
            used_tracks.add(track.track_id)

        output: list[dict] = []
        smoothed_count = 0
        max_adjustment = 0.0
        for detection_index, (detection, raw_box) in enumerate(zip(detections, boxes)):
            result = dict(detection)
            track = matches.get(detection_index)
            if track is None:
                track = self._new_track(result, raw_box, frame_index)
                stable_box = raw_box
                smoothed = False
                adjustment = 0.0
            else:
                stable_box, smoothed, adjustment = self._stabilize(
                    track, raw_box, frame_index
                )
            result["stabilization_track_id"] = track.track_id
            if smoothed and adjustment >= 0.05:
                result["bbox_before_stabilization_xyxy"] = [
                    round(float(value), 2) for value in raw_box
                ]
                result["bbox_xyxy"] = [
                    round(float(value), 2) for value in stable_box
                ]
                result["bbox_stabilized"] = True
                smoothed_count += 1
                max_adjustment = max(max_adjustment, adjustment)
            else:
                result["bbox_stabilized"] = False
            output.append(result)

        self.tracks = {
            track_id: track
            for track_id, track in self.tracks.items()
            if frame_index - track.last_frame_index <= self.max_track_gap
        }
        return output, {
            "active_track_count": len(self.tracks),
            "smoothed_detection_count": smoothed_count,
            "max_adjustment_px": round(max_adjustment, 4),
            "synthesized_detection_count": 0,
        }


def _suppress_conflicting_fragments(
    detections: list[dict],
) -> tuple[list[dict], list[dict[str, Any]]]:
    """Remove only clear low-confidence fragments of a refined element.

    A recovered title may coexist with a weak GPA box covering just a few of
    its letters.  Keeping both creates two trajectories for one visual object.
    Small controls inside a large region are intentionally retained: a
    suppressible fragment must occupy at least 8% of the refined box, overlap
    at least 78% of its own area, and have much lower confidence.
    """

    refined = [
        item
        for item in detections
        if item.get("detection_source", "gpa") != "gpa"
    ]
    if not refined:
        return detections, []

    kept: list[dict] = []
    suppressed: list[dict[str, Any]] = []
    for detection in detections:
        if detection.get("detection_source", "gpa") != "gpa":
            kept.append(detection)
            continue
        box = np.asarray(detection["bbox_xyxy"], dtype=np.float32)
        area = _bbox_area(box)
        confidence = float(detection.get("confidence", 0.0))
        conflict = None
        for parent in refined:
            if int(parent.get("class_id", -1)) != int(
                detection.get("class_id", -1)
            ):
                continue
            parent_box = np.asarray(parent["bbox_xyxy"], dtype=np.float32)
            parent_area = _bbox_area(parent_box)
            if area <= 0.0 or parent_area <= 0.0:
                continue
            overlap_of_fragment = _bbox_intersection(box, parent_box) / area
            relative_area = area / parent_area
            parent_confidence = float(parent.get("confidence", 0.0))
            if (
                overlap_of_fragment >= 0.78
                and 0.08 <= relative_area <= 0.65
                and confidence <= 0.15
                and confidence <= parent_confidence * 0.30
            ):
                conflict = {
                    "suppressed_detection_id": detection.get("id"),
                    "suppressed_bbox_xyxy": detection["bbox_xyxy"],
                    "suppressed_confidence": round(confidence, 6),
                    "refined_detection_id": parent.get("id"),
                    "refined_bbox_xyxy": parent["bbox_xyxy"],
                    "fragment_overlap_ratio": round(overlap_of_fragment, 6),
                    "fragment_to_refined_area_ratio": round(relative_area, 6),
                    "reason": "low_confidence_overlapping_fragment",
                }
                break
        if conflict is None:
            kept.append(detection)
        else:
            suppressed.append(conflict)
    return kept, suppressed


class TemporalLargeBoxRefiner:
    """Refine large flat-background boxes and bridge short GPA regressions."""

    def __init__(
        self,
        *,
        max_area_ratio: float = 0.5,
        refine_area_ratio: float = 0.12,
        refine_width_ratio: float = 0.55,
        refine_height_ratio: float = 0.35,
        frame_step: int = 1,
        max_track_gap: int = 4,
        config: RecoveryConfig | None = None,
    ) -> None:
        self.max_area_ratio = max_area_ratio
        self.refine_area_ratio = refine_area_ratio
        self.refine_width_ratio = refine_width_ratio
        self.refine_height_ratio = refine_height_ratio
        self.frame_step = max(1, frame_step)
        self.max_track_gap = max(1, max_track_gap) * self.frame_step
        self.config = config or RecoveryConfig()
        self.tracks: dict[int, LargeBoxTrack] = {}
        self.next_track_id = 0
        self.stabilizer = DetectionStabilizer(frame_step=self.frame_step)

    def _is_candidate(self, box: np.ndarray, width: int, height: int) -> bool:
        metrics = _bbox_metrics(box, width, height)
        return (
            metrics["area_ratio"] >= self.refine_area_ratio
            or metrics["width_ratio"] >= self.refine_width_ratio
            or metrics["height_ratio"] >= self.refine_height_ratio
        )

    def _active_tracks(self, frame_index: int) -> list[LargeBoxTrack]:
        return [
            track
            for track in self.tracks.values()
            if frame_index - track.last_frame_index <= self.max_track_gap
        ]

    def _predict(self, track: LargeBoxTrack, frame_index: int, width: int, height: int) -> np.ndarray:
        # The local recovery ROI already expands this box. Extrapolating all
        # four edges is unstable for scale/reveal animations: one rapidly
        # changing width can invert the next predicted box. The last refined
        # box is therefore the safer prior, and foreground pixels determine
        # the current extent inside its padded neighborhood.
        return _clip_bbox(track.bbox, width, height)

    def _match_score(
        self, raw_box: np.ndarray, predicted: np.ndarray, width: int, height: int
    ) -> float:
        intersection = _bbox_intersection(raw_box, predicted)
        predicted_coverage = intersection / max(1.0, _bbox_area(predicted))
        overlap = _bbox_iou(raw_box, predicted)
        center_distance = float(
            np.linalg.norm((_bbox_center(raw_box) - _bbox_center(predicted)) / [width, height])
        )
        center_score = max(0.0, 1.0 - center_distance / 0.35)
        if predicted_coverage < 0.25 and overlap < 0.10:
            return 0.0
        return 0.55 * predicted_coverage + 0.25 * overlap + 0.20 * center_score

    def _match_candidates(
        self,
        candidates: list[tuple[int, dict, np.ndarray]],
        active_tracks: list[LargeBoxTrack],
        frame_index: int,
        width: int,
        height: int,
    ) -> dict[int, LargeBoxTrack]:
        ranked = []
        for candidate_index, _, raw_box in candidates:
            for track in active_tracks:
                predicted = self._predict(track, frame_index, width, height)
                score = self._match_score(raw_box, predicted, width, height)
                if score >= 0.35:
                    ranked.append((score, candidate_index, track))
        matches: dict[int, LargeBoxTrack] = {}
        used_tracks: set[int] = set()
        for _, candidate_index, track in sorted(ranked, key=lambda item: item[0], reverse=True):
            if candidate_index not in matches and track.track_id not in used_tracks:
                matches[candidate_index] = track
                used_tracks.add(track.track_id)
        return matches

    def _is_regression_outlier(
        self, raw_box: np.ndarray, predicted: np.ndarray, width: int, height: int
    ) -> bool:
        raw_metrics = _bbox_metrics(raw_box, width, height)
        predicted_metrics = _bbox_metrics(predicted, width, height)
        area_change = raw_metrics["area_ratio"] / max(1e-8, predicted_metrics["area_ratio"])
        width_change = raw_metrics["width_ratio"] / max(
            1e-8, predicted_metrics["width_ratio"]
        )
        height_change = raw_metrics["height_ratio"] / max(
            1e-8, predicted_metrics["height_ratio"]
        )
        center_distance = float(
            np.linalg.norm((_bbox_center(raw_box) - _bbox_center(predicted)) / [width, height])
        )
        aspect_change = abs(
            math.log(
                max(1e-6, raw_metrics["aspect_ratio"])
                / max(1e-6, predicted_metrics["aspect_ratio"])
            )
        )
        return (
            area_change > 2.0
            or area_change < 0.40
            or width_change > 2.20
            or width_change < 0.45
            or height_change > 1.55
            or height_change < 0.50
            or center_distance > 0.16
            or aspect_change > 0.85
        )

    def _validate_recovered(
        self,
        frame: np.ndarray,
        recovered_box: np.ndarray,
        raw_box: np.ndarray,
        predicted: np.ndarray | None,
        track: LargeBoxTrack | None,
        outlier: bool,
    ) -> tuple[bool, dict[str, float]]:
        height, width = frame.shape[:2]
        recovered_metrics = _bbox_metrics(recovered_box, width, height)
        raw_metrics = _bbox_metrics(raw_box, width, height)
        center_distance = float(
            np.linalg.norm((_bbox_center(recovered_box) - _bbox_center(raw_box)) / [width, height])
        )
        width_ratio = recovered_metrics["width_ratio"] / max(1e-8, raw_metrics["width_ratio"])
        height_ratio = recovered_metrics["height_ratio"] / max(1e-8, raw_metrics["height_ratio"])
        temporal_iou = _bbox_iou(recovered_box, predicted) if predicted is not None else 0.0
        appearance = _appearance_histogram(frame, recovered_box)
        appearance_score = _appearance_similarity(
            track.appearance if track is not None else None, appearance
        )
        raw_area = _bbox_area(raw_box)
        recovered_area = _bbox_area(recovered_box)
        recovered_to_raw_area = recovered_area / max(1.0, raw_area)
        oversized = raw_metrics["area_ratio"] > self.max_area_ratio
        evidence = {
            "raw_center_distance": round(center_distance, 6),
            "raw_width_ratio": round(width_ratio, 6),
            "raw_height_ratio": round(height_ratio, 6),
            "recovered_to_raw_area_ratio": round(recovered_to_raw_area, 6),
            "temporal_iou": round(temporal_iou, 6),
            "appearance_similarity": round(appearance_score, 6),
        }
        if predicted is not None and outlier:
            # An anomalous GPA regression may be replaced only by foreground
            # that strongly agrees with the already established track.
            accepted = temporal_iou >= 0.60 and appearance_score >= 0.45
        elif oversized:
            # With no trustworthy raw box, accept only a compact result that
            # removes a substantial amount of empty background.  This keeps
            # bold display text such as ``00`` while rejecting page-sized
            # photos and masks.
            accepted = (
                recovered_metrics["area_ratio"] <= min(0.35, self.max_area_ratio)
                and recovered_to_raw_area <= 0.65
                and center_distance <= 0.18
            )
        elif predicted is not None:
            strict_boundary_fix = (
                0.88 <= width_ratio <= 1.12
                and 0.86 <= height_ratio <= 1.15
                and 0.65 <= recovered_to_raw_area <= 1.25
                and center_distance <= 0.045
            )
            # During a large-text reveal GPA sometimes returns a very wide but
            # vertically clipped box (or the inverse).  An established track
            # lets us accept that correction, but only while recovered width
            # stays below 1.12x.  Neighbour absorption such as CAIN and the
            # Contact Us case grows width to roughly 1.20x/1.24x and therefore
            # cannot pass this branch.
            track_supported_fix = (
                temporal_iou >= 0.55
                and appearance_score >= 0.65
                and 0.55 <= width_ratio <= 1.12
                and 0.72 <= height_ratio <= 1.30
                and 0.45 <= recovered_to_raw_area <= 1.40
                and center_distance <= 0.12
            )
            strong_track_agreement = (
                temporal_iou >= 0.85
                and appearance_score >= 0.80
                and 0.50 <= width_ratio <= 1.35
                and 0.65 <= height_ratio <= 1.35
                and center_distance <= 0.12
            )
            accepted = (
                strict_boundary_fix
                or track_supported_fix
                or strong_track_agreement
            )
        else:
            # A normal GPA box is the primary evidence.  Spatial recovery may
            # only make a small boundary correction.  In particular it may not
            # grow 20% and absorb a neighbouring image/text block (the CAIN
            # regression that motivated this conservative gate).
            accepted = (
                0.88 <= width_ratio <= 1.12
                and 0.86 <= height_ratio <= 1.15
                and 0.65 <= recovered_to_raw_area <= 1.25
                and center_distance <= 0.045
            )
        return accepted, evidence

    def _find_stronger_existing_detection(
        self,
        recovered_box: np.ndarray,
        candidate_index: int,
        candidate: dict,
        raw_detections: list[dict],
        width: int,
        height: int,
    ) -> dict[str, Any] | None:
        """Return a normal GPA box that already represents the recovery.

        GPA can emit both a precise element box and a weak page-sized box in
        the same frame.  Recovering the weak box produces a second, tighter
        box inside the already-correct one.  Prefer the original detection
        when it contains nearly all of the recovery.

        During the first frames of a reveal animation, the recovered visible
        foreground can be much smaller than GPA's box for the whole element.
        The relaxed area test below is therefore limited to recoveries whose
        *source* is itself oversized.  Requiring near-total containment, a
        centre in the existing box's core, and a stronger native confidence
        keeps this from removing ordinary nested controls.
        """

        recovered_area = _bbox_area(recovered_box)
        if recovered_area <= 0.0:
            return None
        candidate_confidence = float(candidate.get("confidence", 0.0))
        candidate_class = int(candidate.get("class_id", -1))
        candidate_box = np.asarray(candidate["bbox_xyxy"], dtype=np.float32)
        candidate_metrics = _bbox_metrics(candidate_box, width, height)
        source_is_oversized = candidate_metrics["area_ratio"] > self.max_area_ratio
        recovered_center = np.asarray(
            [
                (recovered_box[0] + recovered_box[2]) * 0.5,
                (recovered_box[1] + recovered_box[3]) * 0.5,
            ],
            dtype=np.float32,
        )
        strong_overlap_clips: list[np.ndarray] = []
        strong_overlap_detections: list[dict[str, Any]] = []
        for other_index, other in enumerate(raw_detections):
            if other_index == candidate_index:
                continue
            if int(other.get("class_id", -1)) != candidate_class:
                continue
            other_box = np.asarray(other["bbox_xyxy"], dtype=np.float32)
            other_metrics = _bbox_metrics(other_box, width, height)
            if other_metrics["area_ratio"] > self.max_area_ratio:
                continue
            other_area = _bbox_area(other_box)
            if other_area <= 0.0:
                continue
            other_confidence = float(other.get("confidence", 0.0))
            recovered_coverage = (
                _bbox_intersection(recovered_box, other_box) / recovered_area
            )
            recovered_to_other_area = recovered_area / other_area
            other_width = max(1.0, float(other_box[2] - other_box[0]))
            other_height = max(1.0, float(other_box[3] - other_box[1]))
            other_center = np.asarray(
                [
                    (other_box[0] + other_box[2]) * 0.5,
                    (other_box[1] + other_box[3]) * 0.5,
                ],
                dtype=np.float32,
            )
            normalized_center_offset = np.abs(
                recovered_center - other_center
            ) / np.asarray([other_width, other_height], dtype=np.float32)

            intersection = _bbox_intersection(recovered_box, other_box)
            intersection_coverage = intersection / recovered_area
            if (
                source_is_oversized
                and candidate_confidence <= 0.15
                and intersection_coverage >= 0.10
                and other_confidence >= 0.15
                and other_confidence >= candidate_confidence * 1.5
            ):
                clipped_box = np.asarray(
                    [
                        max(recovered_box[0], other_box[0]),
                        max(recovered_box[1], other_box[1]),
                        min(recovered_box[2], other_box[2]),
                        min(recovered_box[3], other_box[3]),
                    ],
                    dtype=np.float32,
                )
                strong_overlap_clips.append(clipped_box)
                strong_overlap_detections.append(
                    {
                        "detection_id": other.get("id", other_index),
                        "bbox_xyxy": other["bbox_xyxy"],
                        "confidence": round(other_confidence, 6),
                        "recovered_intersection_ratio": round(
                            intersection_coverage, 6
                        ),
                    }
                )

            comparable_box_duplicate = (
                recovered_coverage >= 0.85
                and 0.25 <= recovered_to_other_area <= 1.20
                and other_confidence >= 0.20
                and other_confidence >= candidate_confidence * 2.0
            )
            early_reveal_duplicate = (
                source_is_oversized
                and recovered_coverage >= 0.98
                and 0.05 <= recovered_to_other_area <= 1.20
                and float(normalized_center_offset[0]) <= 0.22
                and float(normalized_center_offset[1]) <= 0.22
                and other_confidence >= 0.18
                and other_confidence >= candidate_confidence * 1.05
            )
            if comparable_box_duplicate or early_reveal_duplicate:
                return {
                    "detection_id": other.get("id", other_index),
                    "bbox_xyxy": other["bbox_xyxy"],
                    "confidence": round(other_confidence, 6),
                    "duplicate_rule": (
                        "comparable_box"
                        if comparable_box_duplicate
                        else "oversized_source_early_reveal"
                    ),
                    "recovered_coverage_ratio": round(recovered_coverage, 6),
                    "recovered_to_existing_area_ratio": round(
                        recovered_to_other_area, 6
                    ),
                    "normalized_center_offset_xy": [
                        round(float(value), 6)
                        for value in normalized_center_offset
                    ],
                }
        if len(strong_overlap_clips) >= 2:
            combined_coverage = _bbox_union_area(strong_overlap_clips) / recovered_area
            if combined_coverage >= 0.55:
                return {
                    "detection_ids": [
                        item["detection_id"] for item in strong_overlap_detections
                    ],
                    "detections": strong_overlap_detections,
                    "duplicate_rule": "oversized_source_multi_detection_coverage",
                    "combined_recovered_coverage_ratio": round(
                        combined_coverage, 6
                    ),
                }
        return None

    def _make_refined_detection(
        self,
        detection: dict,
        recovered_box: np.ndarray,
        source: str,
        raw_metrics: dict[str, float],
        recovery_details: dict[str, Any],
        track: LargeBoxTrack | None,
        outlier: bool,
    ) -> dict:
        refined = dict(detection)
        refined.update(
            {
                "bbox_xyxy": [round(float(value), 2) for value in recovered_box],
                "detection_source": source,
                "original_gpa_bbox_xyxy": detection["bbox_xyxy"],
                "original_gpa_area_ratio": round(raw_metrics["area_ratio"], 6),
                "large_box_regression_outlier": outlier,
                "large_box_track_id": track.track_id if track is not None else None,
                "refinement": recovery_details,
            }
        )
        return refined

    def _update_track(
        self,
        track: LargeBoxTrack | None,
        detection: dict,
        frame: np.ndarray,
        frame_index: int,
    ) -> LargeBoxTrack:
        bbox = np.asarray(detection["bbox_xyxy"], dtype=np.float32)
        appearance = _appearance_histogram(frame, bbox)
        if track is None:
            track = LargeBoxTrack(
                track_id=self.next_track_id,
                bbox=bbox,
                previous_bbox=None,
                last_frame_index=frame_index,
                confidence=float(detection.get("confidence", 0.0)),
                class_id=int(detection.get("class_id", -1)),
                class_name=str(detection.get("class_name", "unknown")),
                appearance=appearance,
            )
            self.tracks[track.track_id] = track
            self.next_track_id += 1
        else:
            track.previous_bbox = track.bbox.copy()
            track.bbox = bbox
            track.last_frame_index = frame_index
            track.confidence = float(detection.get("confidence", track.confidence))
            if appearance is not None:
                track.appearance = appearance
        detection["large_box_track_id"] = track.track_id
        return track

    def process(
        self,
        frame: np.ndarray,
        detections: list[dict],
        frame_index: int,
        *,
        enabled: bool = True,
    ) -> tuple[list[dict], dict[str, Any]]:
        height, width = frame.shape[:2]
        raw = [dict(item) for item in detections]
        candidates = []
        for index, detection in enumerate(raw):
            box = np.asarray(detection["bbox_xyxy"], dtype=np.float32)
            if self._is_candidate(box, width, height):
                candidates.append((index, detection, box))

        active_tracks = self._active_tracks(frame_index)
        matches = self._match_candidates(
            candidates, active_tracks, frame_index, width, height
        )
        diagnostics: dict[str, Any] = {
            "pipeline_version": PIPELINE_VERSION,
            "attempted": len(candidates),
            "active_temporal_track_count": len(active_tracks),
            "spatial_refined_count": 0,
            "temporal_refined_count": 0,
            "temporal_recovered_missing_count": 0,
            "regression_outlier_count": 0,
            "suppressed_oversized_count": 0,
            "events": [],
        }
        if not enabled:
            kept = []
            for detection in raw:
                box = np.asarray(detection["bbox_xyxy"], dtype=np.float32)
                if _bbox_metrics(box, width, height)["area_ratio"] <= self.max_area_ratio:
                    kept.append(detection)
                else:
                    diagnostics["suppressed_oversized_count"] += 1
            diagnostics["background_type"] = "recovery_disabled"
            kept, stabilization = self.stabilizer.process(
                kept, frame_index, width, height
            )
            diagnostics["stabilization"] = stabilization
            return kept, diagnostics

        mask = None
        mask_diagnostics: dict[str, Any] = {}
        if candidates or active_tracks:
            mask, mask_diagnostics = _flat_background_mask(frame, self.config)
        diagnostics.update(mask_diagnostics)
        diagnostics["background_type"] = (
            "flat" if mask is not None else "textured_or_uncertain"
        )

        candidate_indexes = {item[0] for item in candidates}
        final_detections = [
            detection for index, detection in enumerate(raw) if index not in candidate_indexes
        ]
        matched_track_ids: set[int] = set()
        for candidate_index, detection, raw_box in candidates:
            raw_metrics = _bbox_metrics(raw_box, width, height)
            track = matches.get(candidate_index)
            predicted = (
                self._predict(track, frame_index, width, height)
                if track is not None
                else None
            )
            outlier = (
                self._is_regression_outlier(raw_box, predicted, width, height)
                if predicted is not None
                else False
            )
            if outlier:
                diagnostics["regression_outlier_count"] += 1
            anchor = predicted if outlier and predicted is not None else raw_box
            padding = 0.12 if outlier and predicted is not None else 0.25
            recovered_box = None
            recovery_details: dict[str, Any] = {}
            if mask is not None:
                recovered_box, recovery_details = _recover_from_local_mask(
                    mask,
                    anchor,
                    _expand_bbox(anchor, padding, width, height),
                    self.max_area_ratio,
                    self.config,
                )
            accepted = False
            if recovered_box is not None:
                accepted, validation = self._validate_recovered(
                    frame, recovered_box, raw_box, predicted, track, outlier
                )
                recovery_details.update(validation)
                if accepted:
                    stronger_existing = self._find_stronger_existing_detection(
                        recovered_box,
                        candidate_index,
                        detection,
                        raw,
                        width,
                        height,
                    )
                    if stronger_existing is not None:
                        accepted = False
                        recovery_details["failure_reason"] = (
                            "duplicate_of_stronger_gpa_detection"
                        )
                        recovery_details["stronger_existing_detection"] = (
                            stronger_existing
                        )
                recovery_details["accepted"] = accepted

            oversized = raw_metrics["area_ratio"] > self.max_area_ratio
            final_detection = None
            if accepted and recovered_box is not None:
                source = (
                    "temporal_foreground_refinement"
                    if predicted is not None
                    else "spatial_foreground_refinement"
                )
                final_detection = self._make_refined_detection(
                    detection,
                    recovered_box,
                    source,
                    raw_metrics,
                    recovery_details,
                    track,
                    outlier,
                )
                if predicted is not None:
                    diagnostics["temporal_refined_count"] += 1
                else:
                    diagnostics["spatial_refined_count"] += 1
            elif not oversized and not outlier:
                final_detection = detection
                final_detection["large_box_track_id"] = (
                    track.track_id if track is not None else None
                )
            else:
                diagnostics["suppressed_oversized_count"] += int(oversized)

            event = {
                "raw_detection_id": detection.get("id", candidate_index),
                "raw_bbox_xyxy": detection["bbox_xyxy"],
                "raw_area_ratio": round(raw_metrics["area_ratio"], 6),
                "matched_track_id": track.track_id if track is not None else None,
                "regression_outlier": outlier,
                "result": (
                    final_detection.get("detection_source", "gpa")
                    if final_detection is not None
                    else "suppressed"
                ),
                "recovery": recovery_details,
            }
            diagnostics["events"].append(event)
            if final_detection is not None:
                final_detections.append(final_detection)
                updated_track = self._update_track(
                    track, final_detection, frame, frame_index
                )
                matched_track_ids.add(updated_track.track_id)

        # Deliberately do not synthesize a box when GPA has no detection in the
        # current frame.  Appearance histograms over mostly-flat UI regions are
        # too ambiguous for that decision and previously created large phantom
        # boxes.  Oversized raw GPA boxes are still recoverable above, so this
        # does not remove the important large-title fallback.

        # Expire stale state so an unrelated later element cannot inherit it.
        self.tracks = {
            track_id: track
            for track_id, track in self.tracks.items()
            if frame_index - track.last_frame_index <= self.max_track_gap
        }
        final_detections, suppressed_fragments = _suppress_conflicting_fragments(
            final_detections
        )
        diagnostics["suppressed_conflicting_fragment_count"] = len(
            suppressed_fragments
        )
        diagnostics["suppressed_conflicting_fragments"] = suppressed_fragments
        for detection_id, detection in enumerate(final_detections):
            detection["id"] = detection_id
        final_detections, stabilization = self.stabilizer.process(
            final_detections, frame_index, width, height
        )
        diagnostics["stabilization"] = stabilization
        diagnostics["output_detection_count"] = len(final_detections)
        diagnostics["tracked_large_box_count"] = len(self.tracks)
        return final_detections, diagnostics


EVIDENCE_PIPELINE_VERSION = "pixel_evidence_text_line"


def _intersection(a: np.ndarray, b: np.ndarray) -> float:
    return float(max(0, min(a[2], b[2]) - max(a[0], b[0])) *
                 max(0, min(a[3], b[3]) - max(a[1], b[1])))


class EvidenceTemporalRefiner:
    def __init__(
        self, frame_step: int = 1, max_area_ratio: float = 0.5,
        max_track_gap: int = 4, min_recovery_confidence: float = 0.1,
        stabilize: bool = True,
    ) -> None:
        if not 0 < max_area_ratio <= 1:
            raise ValueError("max_area_ratio must be in (0, 1]")
        if not 0 <= min_recovery_confidence <= 1:
            raise ValueError("min_recovery_confidence must be between 0 and 1")
        # The online stabilizer is skipped when a global temporal pass runs
        # afterwards: smoothing the same boxes twice only adds lag.
        self.stabilizer = DetectionStabilizer(frame_step=frame_step) if stabilize else None
        self.max_area_ratio = max_area_ratio
        self.min_recovery_confidence = min_recovery_confidence
        # Run the older foreground recovery only on proposals which would
        # otherwise be discarded for being oversized.  Its broad replacement
        # and fragment-suppression rules are not applied to ordinary boxes.
        self.oversized_refiner = TemporalLargeBoxRefiner(
            frame_step=frame_step, max_area_ratio=max_area_ratio,
            max_track_gap=max_track_gap,
        )
        self.title_box: np.ndarray | None = None
        self.title_confidence = 0.0
        self.title_frame = -1000
        self.frame_step = frame_step

    @staticmethod
    def _text_line(frame: np.ndarray, envelope: np.ndarray) -> np.ndarray | None:
        height, width = frame.shape[:2]
        # A flat bright background is the only case where a dark-pixel mask is
        # reliable.  No crop, image, or dark-mode control is segmented this way.
        upper = frame[:int(height * 0.46)]
        if np.mean(np.all(upper > 235, axis=2)) < 0.68:
            return None
        gray = cv2.cvtColor(upper, cv2.COLOR_BGR2GRAY)
        mask = (gray < 125).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        margin = width * 0.11
        components = []
        for x, y, w, h, area in stats[1:count]:
            if (area < 120 or h < height * 0.037 or w < 12 or
                w > width * 0.46 or x + w < envelope[0] - margin or
                x > envelope[2] + margin or y > height * 0.30):
                continue
            components.append((x, y, x + w, y + h))
        if not components:
            return None
        # Join glyphs on the same large headline.  A second small text line
        # is excluded by the minimum component height above.
        components.sort(key=lambda box: box[0])
        groups: list[list[tuple[int, int, int, int]]] = []
        for component in components:
            if groups and component[0] - groups[-1][-1][2] < width * 0.08:
                groups[-1].append(component)
            else:
                groups.append([component])
        group = max(groups, key=lambda items: sum((b[2]-b[0])*(b[3]-b[1]) for b in items))
        x1 = min(b[0] for b in group)
        y1 = min(b[1] for b in group)
        x2 = max(b[2] for b in group)
        y2 = max(b[3] for b in group)
        if x2 - x1 < width * 0.055:
            return None
        pad = max(2, round(height * 0.004))
        return np.array([max(0, x1-pad), max(0, y1-pad),
                         min(width, x2+pad), min(height, y2+pad)], dtype=np.float32)

    @staticmethod
    def _blank_border_box(frame: np.ndarray, detection: dict) -> bool:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = map(int, detection["bbox_xyxy"])
        if not (y1 > height * 0.70 and
                (x1 < width * 0.025 or x2 > width * 0.975) and
                x2 - x1 < width * 0.25 and
                float(detection.get("confidence", 0)) < 0.16):
            return False
        crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
        if crop.size == 0:
            return True
        # Sampled fraction is more robust than variance near JPEG edges.
        sample = crop[::4, ::4]
        return float(np.mean(np.any(sample < 225, axis=2))) < 0.01

    def process(self, frame: np.ndarray, raw: list[dict], frame_index: int
                ) -> tuple[list[dict], dict[str, Any]]:
        height, width = frame.shape[:2]
        detections = [dict(item) for item in raw]
        title_candidates = []
        for item in detections:
            box = np.asarray(item["bbox_xyxy"], dtype=np.float32)
            if (box[1] < height * 0.14 and box[3] < height * 0.53 and
                box[2] - box[0] > width * 0.06 and
                box[3] - box[1] > height * 0.06 and
                box[0] < width * 0.96):
                title_candidates.append(item)
        recent = frame_index - self.title_frame <= self.frame_step * 4
        title = None
        if title_candidates or (recent and self.title_box is not None):
            boxes = [np.asarray(item["bbox_xyxy"], dtype=np.float32)
                     for item in title_candidates]
            if recent and self.title_box is not None:
                boxes.append(self.title_box)
            envelope = np.array([min(b[0] for b in boxes), min(b[1] for b in boxes),
                                 max(b[2] for b in boxes), max(b[3] for b in boxes)])
            title = self._text_line(frame, envelope)
            if title is not None and not title_candidates and self.title_box is not None:
                # History may request a pixel check, but unrelated later text
                # must not inherit the old headline's confidence floor.
                previous = self.title_box
                overlap = _intersection(title, previous)
                union = ((title[2] - title[0]) * (title[3] - title[1])
                         + (previous[2] - previous[0]) * (previous[3] - previous[1])
                         - overlap)
                if union <= 0 or overlap / union < 0.40:
                    title = None
        removed = []
        if title is not None:
            kept = []
            for item in detections:
                box = np.asarray(item["bbox_xyxy"], dtype=np.float32)
                area = max(1.0, float((box[2]-box[0]) * (box[3]-box[1])))
                # Replace only detections that genuinely touch the headline;
                # a control below it remains an independent element.
                if (item in title_candidates and
                    (_intersection(box, title) / area > 0.10 or
                     _intersection(box, title) / max(1.0, (title[2]-title[0])*(title[3]-title[1])) > 0.08)):
                    removed.append(item["id"])
                else:
                    kept.append(item)
            if removed or recent:
                confidence = max([float(item["confidence"]) for item in title_candidates] +
                                 [self.title_confidence * 0.85,
                                  self.min_recovery_confidence + 0.001])
                source = "pixel_supported_text_line"
                kept.append({"id": -1, "class_id": 0, "class_name": "text_line",
                             "confidence": round(confidence, 6),
                             "bbox_xyxy": [round(float(v), 2) for v in title],
                             "detection_source": source,
                             "raw_detection_ids": removed,
                             "track_anchor_xy": [round(float(title[0]), 2),
                                                 round(float(title[1]), 2)]})
                detections = kept
                self.title_box = title
                self.title_confidence = confidence
                self.title_frame = frame_index
        # Apply the original oversized-box limit to native GPA detections even
        # when pixel-evidence refinement is enabled.  A separately validated
        # text-line box can coexist with the discarded broad GPA proposal.
        oversized_ids = []
        oversized = []
        bounded = []
        frame_area = width * height
        for item in detections:
            box = item["bbox_xyxy"]
            area_ratio = (max(0.0, box[2] - box[0]) *
                          max(0.0, box[3] - box[1]) / frame_area)
            if (item.get("detection_source", "gpa") == "gpa" and
                    area_ratio > self.max_area_ratio):
                oversized_ids.append(item["id"])
                # A recovered box inherits the proposal score.  Proposals
                # below the final output cutoff cannot yield a visible box,
                # and seeding tracks with them increases false recoveries.
                if float(item["confidence"]) >= self.min_recovery_confidence:
                    item["original_raw_detection_id"] = item["id"]
                    oversized.append(item)
            else:
                bounded.append(item)
        detections = bounded
        recovered_ids = []
        duplicate_ids = []
        fallback_diagnostics: dict[str, Any] = {}
        if oversized:
            fallback, fallback_diagnostics = self.oversized_refiner.process(
                frame, oversized, frame_index,
            )
            for item in fallback:
                if item.get("detection_source", "gpa") == "gpa":
                    continue
                box = np.asarray(item["bbox_xyxy"], dtype=np.float32)
                area = max(1.0, float((box[2] - box[0]) * (box[3] - box[1])))
                original_id = item["original_raw_detection_id"]
                source_index = next(
                    index for index, source in enumerate(raw)
                    if source["id"] == original_id
                )
                stronger_existing = self.oversized_refiner._find_stronger_existing_detection(
                    box, source_index, raw[source_index], raw, width, height
                )
                # The bright-background headline detector may already have
                # made a tighter box for the same foreground.  Compare shared
                # area relative to the smaller box, so a broad duplicate is
                # rejected even when its IoU is modest.
                duplicate = False
                for existing in detections:
                    if existing.get("detection_source") != "pixel_supported_text_line":
                        continue
                    other = np.asarray(existing["bbox_xyxy"], dtype=np.float32)
                    other_area = max(1.0, float(
                        (other[2] - other[0]) * (other[3] - other[1])))
                    if _intersection(box, other) / min(area, other_area) >= 0.70:
                        duplicate = True
                        break
                if duplicate or stronger_existing is not None:
                    duplicate_ids.append(original_id)
                else:
                    detections.append(item)
                    recovered_ids.append(original_id)
        blank_ids = [item["id"] for item in detections
                     if self._blank_border_box(frame, item)]
        detections = [item for item in detections if item["id"] not in blank_ids]
        if self.stabilizer is not None:
            detections, stabilization = self.stabilizer.process(
                detections, frame_index, width, height)
        else:
            stabilization = {"skipped": True}
        for index, item in enumerate(detections):
            item["id"] = index
        diagnostics = {"pipeline_version": EVIDENCE_PIPELINE_VERSION,
                       "text_line_refined": title is not None and bool(removed or recent),
                       "replaced_detection_ids": removed,
                       "recovered_oversized_detection_ids": recovered_ids,
                       "duplicate_oversized_detection_ids": duplicate_ids,
                       "suppressed_oversized_detection_ids": [
                           item_id for item_id in oversized_ids
                           if item_id not in recovered_ids and item_id not in duplicate_ids
                       ],
                       "oversized_recovery": fallback_diagnostics,
                       "blank_border_detection_ids": blank_ids,
                       "stabilization": stabilization,
                       "output_detection_count": len(detections),
                       "suppressed_oversized_count": sum(
                           item_id not in recovered_ids and item_id not in duplicate_ids
                           for item_id in oversized_ids
                       )}
        return detections, diagnostics
