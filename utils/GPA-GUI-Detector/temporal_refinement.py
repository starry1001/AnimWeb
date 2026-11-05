"""Whole-video temporal consolidation and final detection-box refinement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

PIPELINE_VERSION = "global_temporal_consistency"


@dataclass
class TemporalConfig:
    """Thresholds for :func:`consolidate`.

    Pixel thresholds are expressed on the 0-255 grayscale of the analysis
    image; geometric thresholds are relative to each element's own size, so
    they behave the same on a small control and on a hero banner.
    """

    min_confidence: float = 0.1
    max_gap: int = 8
    # Blank-box rejection.
    blank_max_confidence: float = 0.45
    blank_std: float = 4.0
    blank_contrast: float = 6.0
    blank_relative: float = 0.15
    blank_flat_std: float = 1.50
    # Track pruning.
    weak_track_confidence: float = 0.30
    min_weak_track_frames: int = 3
    # Existence filling.
    fill_match_score: float = 0.45
    fill_static_iou: float = 0.80
    extend_iou: float = 0.50
    extend_match_score: float = 0.50
    # Group consolidation.
    group_merge: bool = True
    group_containment: float = 0.80
    group_same_scene_score: float = 0.40
    group_overlap_share: float = 0.50
    group_complementary: float = 0.50
    group_min_frames: int = 3
    group_min_share: float = 0.40
    group_drop_confidence: float = 0.35
    # Geometry smoothing.
    line_merge: bool = True
    line_gap_ratio: float = 0.20
    line_gap_floor: float = 10.0
    line_height_ratio: float = 0.70
    line_overlap: float = 0.60
    line_min_members: int = 3
    line_member_aspect: float = 1.40
    line_max_height_ratio: float = 0.15
    tighten: bool = True
    tighten_area_ratio: float = 0.20
    tighten_min_shrink: float = 0.85
    tighten_deviation: float = 18.0
    tighten_edge: float = 12.0
    tighten_window: int = 9
    tighten_density: float = 0.25
    duplicate_iou: float = 0.70
    smooth_window: int = 2
    smooth_noise_ratio: float = 0.015
    smooth_noise_floor: float = 2.0
    smooth_break_ratio: float = 0.18
    smooth_break_floor: float = 10.0


def _area(box: np.ndarray) -> float:
    return float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))


def _intersection(a: np.ndarray, b: np.ndarray) -> float:
    return float(max(0.0, min(a[2], b[2]) - max(a[0], b[0])) *
                 max(0.0, min(a[3], b[3]) - max(a[1], b[1])))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    overlap = _intersection(a, b)
    return overlap / max(1.0, _area(a) + _area(b) - overlap)


def _containment(inner: np.ndarray, outer: np.ndarray) -> float:
    """Fraction of ``inner`` that lies inside ``outer``."""
    return _intersection(inner, outer) / max(1.0, _area(inner))


def _center(box: np.ndarray) -> np.ndarray:
    return (box[:2] + box[2:]) / 2.0


def _size(box: np.ndarray) -> np.ndarray:
    return np.maximum(box[2:] - box[:2], 1.0)


def _rounded(box: np.ndarray) -> list[float]:
    return [round(float(value), 2) for value in box]


class _Pixels:
    """Grayscale views of the clip used for the pixel-evidence checks."""

    def __init__(self, grays: list[np.ndarray], scale: float) -> None:
        self.grays = grays
        self.scale = scale
        self.detail: dict[int, float] = {}

    def frame_detail(self, position: int) -> float:
        """How much edge a frame carries at all, so a dark fade-in is judged
        against itself rather than against a fully drawn page."""
        if position not in self.detail:
            gray = self.grays[position].astype(np.float32)
            self.detail[position] = float(
                np.percentile(np.abs(cv2.Laplacian(gray, cv2.CV_32F)), 99))
        return self.detail[position]

    def _crop(self, position: int, box: np.ndarray, pad: int = 0) -> np.ndarray:
        gray = self.grays[position]
        scaled = box * self.scale
        x1 = max(0, int(scaled[0]) - pad)
        y1 = max(0, int(scaled[1]) - pad)
        x2 = min(gray.shape[1], int(np.ceil(scaled[2])) + pad)
        y2 = min(gray.shape[0], int(np.ceil(scaled[3])) + pad)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return np.empty((0, 0), dtype=np.uint8)
        return gray[y1:y2, x1:x2]

    def is_blank(self, position: int, box: np.ndarray, config: TemporalConfig) -> bool:
        """True when the box covers no visible content.

        A flat interior alone is not enough: a solid-colour button is flat but
        contrasts with the page around it, so the surrounding ring is compared
        as well before the box is called empty.
        """
        patch = self._crop(position, box)
        if patch.size == 0:
            return False
        interior = patch.astype(np.float32)
        if float(interior.std()) >= config.blank_std:
            return False
        flat = float(interior.std()) < config.blank_flat_std
        # Judged against the frame's own detail: on a page that is itself
        # barely drawn, faint content is all there is, and that is not the
        # same thing as a box over nothing.  A patch with no variation at all
        # is exempt: on an entirely flat frame it would otherwise rate as
        # being as detailed as everything else.
        edge = float(cv2.Laplacian(interior, cv2.CV_32F).std())
        if not flat and edge >= config.blank_relative * self.frame_detail(position):
            return False
        ring = self._crop(position, box, pad=max(2, int(round(3 * self.scale))))
        if ring.size > patch.size:
            outer_mean = (float(ring.astype(np.float32).sum()) - float(interior.sum())
                          ) / max(1, ring.size - patch.size)
            if abs(outer_mean - float(interior.mean())) >= config.blank_contrast:
                return False
        return True

    def content_bounds(self, position: int, box: np.ndarray,
                       config: TemporalConfig) -> np.ndarray | None:
        """Bounding box of the visible content inside ``box``.

        Pixels that differ from the region's own background, or that carry an
        edge, are marked; the marks are then required to be locally dense, so
        a hairline rule or a scrollbar does not count as content on its own.
        """
        gray = self.grays[position]
        scaled = np.rint(box * self.scale).astype(int)
        x1 = max(0, scaled[0])
        y1 = max(0, scaled[1])
        x2 = min(gray.shape[1], scaled[2])
        y2 = min(gray.shape[0], scaled[3])
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        crop = gray[y1:y2, x1:x2].astype(np.float32)
        background = float(np.median(crop))
        marks = ((np.abs(crop - background) > config.tighten_deviation) |
                 (np.abs(cv2.Laplacian(crop, cv2.CV_32F)) > config.tighten_edge)
                 ).astype(np.float32)
        window = max(3, int(config.tighten_window))
        dense = cv2.blur(marks, (window, window)) >= config.tighten_density
        rows = np.flatnonzero(dense.sum(axis=1) > max(1.0, 0.02 * dense.shape[1]))
        columns = np.flatnonzero(dense.sum(axis=0) > max(1.0, 0.02 * dense.shape[0]))
        if rows.size == 0 or columns.size == 0:
            return None
        pad = 4
        return np.array([x1 + columns[0] - pad, y1 + rows[0] - pad,
                         x1 + columns[-1] + 1 + pad, y1 + rows[-1] + 1 + pad],
                        dtype=np.float32) / self.scale

    def looks_same(self, anchor: int, target: int, box: np.ndarray,
                   threshold: float) -> bool:
        """True when this very place still shows what it showed before.

        The comparison is deliberately pinned to the box's own coordinates.
        Searching a neighbourhood would answer a different question -- whether
        the element is anywhere nearby -- and would hold a stale box in place
        over a row of controls that has just scrolled past it.
        """
        here = self._crop(anchor, box).astype(np.float32)
        there = self._crop(target, box).astype(np.float32)
        if here.size == 0 or there.size == 0 or here.shape != there.shape:
            return False
        if float(here.std()) >= 3.0:
            return float(cv2.matchTemplate(there, here,
                                           cv2.TM_CCOEFF_NORMED)[0, 0]) >= threshold
        # A featureless patch correlates with anything, so its own statistics
        # are compared instead of a meaningless score.
        return (abs(float(here.mean()) - float(there.mean())) < 8.0 and
                abs(float(here.std()) - float(there.std())) < 4.0)

    def match(self, anchor: int, target: int, box: np.ndarray,
              expected: np.ndarray) -> tuple[float, np.ndarray]:
        """Locate ``box``'s appearance near ``expected`` on another frame."""
        source = self.grays[anchor]
        scaled = np.rint(box * self.scale).astype(int)
        x1 = max(0, scaled[0])
        y1 = max(0, scaled[1])
        x2 = min(source.shape[1], scaled[2])
        y2 = min(source.shape[0], scaled[3])
        if x2 - x1 < 4 or y2 - y1 < 4:
            return 0.0, expected
        template = source[y1:y2, x1:x2]
        if float(template.std()) < 3.0:
            # Featureless templates match everywhere; the caller falls back to
            # its geometric rule instead of trusting a meaningless score.
            return 0.0, expected
        target_gray = self.grays[target]
        want = np.rint(expected * self.scale).astype(int)
        margin = max(4, int(round(max(template.shape) * 0.25)))
        left = max(0, want[0] - margin)
        top = max(0, want[1] - margin)
        right = min(target_gray.shape[1], want[2] + margin)
        bottom = min(target_gray.shape[0], want[3] + margin)
        search = target_gray[top:bottom, left:right]
        if (search.shape[0] < template.shape[0] or
                search.shape[1] < template.shape[1]):
            return 0.0, expected
        response = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(response)
        x = left + location[0]
        y = top + location[1]
        matched = np.array([x, y, x + template.shape[1], y + template.shape[0]],
                           dtype=np.float32) / self.scale
        return float(score), matched


@dataclass
class _Track:
    ident: int
    class_id: int
    positions: list[int] = field(default_factory=list)
    boxes: dict[int, np.ndarray] = field(default_factory=dict)
    items: dict[int, dict] = field(default_factory=dict)
    # Frames the detector itself reported this box on.  Synthesised boxes are
    # never added here, so later stages can still tell evidence from inference.
    observed: set[int] = field(default_factory=set)
    dropped: bool = False

    @property
    def first(self) -> int:
        return self.positions[0]

    @property
    def last(self) -> int:
        return self.positions[-1]

    def add(self, position: int, box: np.ndarray, item: dict | None) -> None:
        if position not in self.boxes:
            self.positions.append(position)
            self.positions.sort()
        self.boxes[position] = box
        if item is not None:
            self.items[position] = item

    def remove(self, position: int) -> None:
        self.boxes.pop(position, None)
        self.items.pop(position, None)
        if position in self.positions:
            self.positions.remove(position)

    def confidence(self, position: int) -> float:
        item = self.items.get(position)
        return float(item["confidence"]) if item is not None else 0.0

    def median_box(self) -> np.ndarray:
        return np.median(np.stack([self.boxes[p] for p in self.positions]), axis=0)

    def predict(self, position: int) -> np.ndarray:
        last = self.last
        box = self.boxes[last]
        if len(self.positions) < 2:
            return box
        previous = self.positions[-2]
        velocity = (box - self.boxes[previous]) / max(1, last - previous)
        step = min(position - last, 3)
        return box + velocity * step


def _associate(frame_boxes: list[np.ndarray], frame_items: list[list[dict]],
               config: TemporalConfig) -> list[_Track]:
    """Link per-frame boxes into tracks using overlap and predicted motion."""
    tracks: list[_Track] = []
    active: list[_Track] = []
    for position, (boxes, items) in enumerate(zip(frame_boxes, frame_items)):
        active = [track for track in active
                  if position - track.last <= config.max_gap]
        candidates: list[tuple[float, int, int]] = []
        for index, (box, item) in enumerate(zip(boxes, items)):
            class_id = int(item.get("class_id", -1))
            for order, track in enumerate(active):
                if track.class_id != class_id:
                    continue
                # An element that translates is best matched against the
                # predicted box; one whose edges are merely re-regressed is
                # best matched against the last observation.  Both are tried.
                best = None
                for reference_box in (track.boxes[track.last],
                                      track.predict(position)):
                    ratio = _area(box) / max(1.0, _area(reference_box))
                    # A size change this large means a different element, not
                    # a re-regressed one, whatever the overlap says.
                    if not 0.4 <= ratio <= 2.5:
                        continue
                    overlap = _iou(box, reference_box)
                    containment = max(_containment(box, reference_box),
                                      _containment(reference_box, box))
                    extent = np.maximum(_size(reference_box), 12.0)
                    distance = float(np.linalg.norm(
                        (_center(box) - _center(reference_box)) / extent))
                    if (overlap < 0.35 and containment < 0.8 and
                            not (overlap >= 0.15 and distance <= 0.4)):
                        continue
                    score = overlap - 0.1 * distance
                    best = score if best is None else max(best, score)
                if best is None:
                    continue
                candidates.append((best - 0.02 * (position - track.last),
                                   order, index))
        used_tracks: set[int] = set()
        used_boxes: set[int] = set()
        for _, order, index in sorted(candidates, key=lambda item: -item[0]):
            if order in used_tracks or index in used_boxes:
                continue
            active[order].add(position, boxes[index], items[index])
            active[order].observed.add(position)
            used_tracks.add(order)
            used_boxes.add(index)
        for index, (box, item) in enumerate(zip(boxes, items)):
            if index in used_boxes:
                continue
            track = _Track(len(tracks), int(item.get("class_id", -1)))
            track.add(position, box, item)
            track.observed.add(position)
            tracks.append(track)
            active.append(track)
    return tracks


def _reject_blank(frame_boxes: list[np.ndarray], frame_items: list[list[dict]],
                  pixels: _Pixels, config: TemporalConfig) -> int:
    removed = 0
    for position, (boxes, items) in enumerate(zip(frame_boxes, frame_items)):
        keep_boxes: list[np.ndarray] = []
        keep_items: list[dict] = []
        for box, item in zip(boxes, items):
            if (float(item["confidence"]) < config.blank_max_confidence and
                    pixels.is_blank(position, box, config)):
                removed += 1
                continue
            keep_boxes.append(box)
            keep_items.append(item)
        frame_boxes[position] = keep_boxes
        frame_items[position] = keep_items
    return removed


class _Element:
    """One screen element, possibly reported by several broken tracks.

    The detector often restarts a track when it re-regresses an element's
    edges, so a single on-screen element shows up as a handful of short
    tracks in the same place.  Voting on a representation only makes sense
    once those are seen as one thing.
    """

    def __init__(self, tracks: list[_Track]) -> None:
        self.tracks = tracks
        self.dropped = False
        self.refresh()

    def refresh(self) -> None:
        self.frames = {position for track in self.tracks
                       for position in track.positions}
        boxes = [track.boxes[position] for track in self.tracks
                 for position in track.positions]
        self.canonical = (np.median(np.stack(boxes), axis=0) if boxes
                          else np.zeros(4, dtype=np.float32))
        self.area = _area(self.canonical)

    @property
    def observed(self) -> set[int]:
        return {position for track in self.tracks for position in track.observed
                if position in track.boxes}

    @property
    def observed(self) -> set[int]:
        return {position for track in self.tracks for position in track.observed
                if position in track.boxes}

    @property
    def first(self) -> int:
        return min(self.observed) if self.observed else min(self.frames)

    @property
    def last(self) -> int:
        return max(self.observed) if self.observed else max(self.frames)

    @property
    def host(self) -> _Track:
        return max(self.tracks, key=lambda track: len(track.positions))

    def drop(self) -> None:
        self.dropped = True
        for track in self.tracks:
            track.dropped = True

    def box_at(self, position: int) -> np.ndarray | None:
        for track in self.tracks:
            if position in track.boxes:
                return track.boxes[position]
        return None

    def item_at(self, position: int) -> dict | None:
        for track in self.tracks:
            if position in track.items:
                return track.items[position]
        return None

    def confidence(self, position: int) -> float:
        item = self.item_at(position)
        return float(item["confidence"]) if item is not None else 0.0

    def track_of(self, position: int) -> _Track | None:
        for track in self.tracks:
            if position in track.boxes:
                return track
        return None

    def nearest(self, position: int) -> tuple[_Track, int]:
        best: tuple[int, _Track, int] | None = None
        for track in self.tracks:
            for other in track.positions:
                distance = abs(other - position)
                if best is None or distance < best[0]:
                    best = (distance, track, other)
        assert best is not None
        return best[1], best[2]


def _elements(tracks: list[_Track], overlap: float = 0.55) -> list[_Element]:
    """Group tracks that occupy the same place into one element."""
    live = [track for track in tracks if not track.dropped and track.positions]
    parent = list(range(len(live)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    medians = [track.median_box() for track in live]
    for left in range(len(live)):
        for right in range(left + 1, len(live)):
            if live[left].class_id != live[right].class_id:
                continue
            if _iou(medians[left], medians[right]) < overlap:
                continue
            parent[find(left)] = find(right)
    groups: dict[int, list[_Track]] = {}
    for index, track in enumerate(live):
        groups.setdefault(find(index), []).append(track)
    return [_Element(members) for members in groups.values()]


def _confirm_elements(elements: list[_Element], pixels: _Pixels,
                      config: TemporalConfig) -> dict[str, int]:
    """Decide where each element exists, and give it a box on those frames.

    Confirmation is a property of the element rather than of a single track:
    the detector restarts a track whenever it re-regresses an element, and a
    restarted track is not a new element.
    """
    statistics = {"unconfirmed_elements": 0, "flickering_elements": 0,
                  "promoted_weak_boxes": 0, "extended_boxes": 0,
                  "trimmed_weak_boxes": 0, "gap_filled_boxes": 0}
    for element in elements:
        positions = sorted(element.frames)
        confirmed = [p for p in positions
                     if element.confidence(p) >= config.min_confidence]
        if not confirmed:
            # The output cutoff is applied first and is never overridden by an
            # element made only of sub-threshold boxes.
            element.drop()
            statistics["unconfirmed_elements"] += 1
            continue
        peak = max(element.confidence(p) for p in positions)
        if (peak < config.weak_track_confidence and
                len(positions) < config.min_weak_track_frames):
            element.drop()
            statistics["flickering_elements"] += 1
            continue
        support = round(float(np.median([element.confidence(p) for p in confirmed])), 6)
        start, end = confirmed[0], confirmed[-1]

        # An element the detector stops reporting is still on screen while the
        # box still shows what it showed on the last confirmed frame.  Its own
        # weak detections are preferred; where it has none, the box is held.
        host = element.host
        for direction, anchor in ((-1, start), (1, end)):
            template = element.box_at(anchor)
            reference = template
            source = dict(element.item_at(anchor))
            previous = anchor
            for step in range(1, config.max_gap + 1):
                position = anchor + direction * step
                if position < 0 or position >= len(pixels.grays):
                    break
                box = element.box_at(position)
                synthesised = box is None
                if synthesised:
                    box = reference
                elif _iou(box, reference) < config.extend_iou:
                    break
                if not pixels.looks_same(anchor, position, template,
                                         config.extend_match_score):
                    break
                if pixels.is_blank(position, box, config):
                    break
                if synthesised:
                    item = dict(source)
                    item.pop("temporal_raw_confidence", None)
                    item["confidence"] = support
                    item["detection_source"] = "temporal_extension"
                    item["bbox_xyxy"] = _rounded(box)
                    host.add(position, box.astype(np.float32), item)
                if direction < 0:
                    start = position
                else:
                    end = position
                previous = position
                reference = box
                statistics["extended_boxes"] += 1
        element.refresh()
        positions = sorted(element.frames)

        for position in positions:
            item = element.item_at(position)
            if item is None:
                continue
            if not start <= position <= end:
                if element.confidence(position) < config.min_confidence:
                    track = element.track_of(position)
                    if track is not None:
                        track.remove(position)
                        statistics["trimmed_weak_boxes"] += 1
                continue
            if float(item["confidence"]) < config.min_confidence:
                item["temporal_raw_confidence"] = item["confidence"]
                item["confidence"] = support
                item["detection_source"] = "temporal_weak_support"
                statistics["promoted_weak_boxes"] += 1

        element.refresh()
        anchors = [p for p in sorted(element.frames) if start <= p <= end]
        host = element.host
        for left, right in zip(anchors, anchors[1:]):
            # A hole wider than the tracking gap is a scene change, not a
            # missed detection, however well the two sides line up.
            if right - left < 2 or right - left - 1 > config.max_gap:
                continue
            left_box = element.box_at(left)
            right_box = element.box_at(right)
            static = _iou(left_box, right_box) >= config.fill_static_iou
            anchor = (left if element.confidence(left) >= element.confidence(right)
                      else right)
            template = element.box_at(anchor)
            source = dict(element.item_at(anchor))
            for position in range(left + 1, right):
                weight = (position - left) / (right - left)
                expected = left_box * (1 - weight) + right_box * weight
                score, matched = pixels.match(anchor, position, template, expected)
                if score > 0.0:
                    if score < config.fill_match_score:
                        # The template is informative and did not find the
                        # element: it has genuinely gone from this frame, so
                        # the hole in the track is real.
                        continue
                    if not static:
                        extent = np.maximum(_size(expected), 12.0)
                        drift = np.abs(_center(matched) - _center(expected))
                        if np.all(drift <= np.maximum(16.0, extent * 0.25)):
                            expected = matched
                elif not pixels.looks_same(anchor, position, template,
                                           config.fill_match_score):
                    continue
                if pixels.is_blank(position, expected, config):
                    continue
                synthetic = dict(source)
                synthetic.pop("temporal_raw_confidence", None)
                synthetic["confidence"] = support
                synthetic["detection_source"] = "temporal_gap_fill"
                synthetic["bbox_xyxy"] = _rounded(expected)
                host.add(position, expected.astype(np.float32), synthetic)
                statistics["gap_filled_boxes"] += 1
        element.refresh()
    return statistics


def _same_element(pixels: _Pixels, parent: _Element, child: _Element,
                  config: TemporalConfig) -> bool:
    """Reject a pair that merely shares screen space at different times.

    Two readings of one element alternate frame by frame.  A box from an
    earlier part of the clip and a box from a later part can look just as
    complementary while being unrelated content at the same place, so when
    their frames barely overlap the pixels decide.
    """
    # A track made entirely of synthetic gap fills has no detector frame to
    # compare or vote with. It cannot establish that a fragment is the same
    # element, and indexing its observations would fail.
    if not parent.observed or not child.observed:
        return False
    low = parent.first - config.max_gap
    high = parent.last + config.max_gap
    inside = sum(low <= position <= high for position in child.observed)
    if inside >= config.group_overlap_share * len(child.observed):
        return True
    elsewhere = sorted(child.observed - parent.observed)
    if not elsewhere:
        return True
    here = sorted(parent.observed)[len(parent.observed) // 2]
    there = elsewhere[len(elsewhere) // 2]
    return pixels.looks_same(here, there, parent.canonical,
                             config.group_same_scene_score)


def _consolidate_groups(elements: list[_Element], pixels: _Pixels,
                        config: TemporalConfig) -> dict[str, int]:
    """Reduce an element reported both as one box and as fragments.

    The clip decides which representation to keep: whichever covers more
    frames is applied to every frame of the shared span.
    """
    merged_frames = 0
    dropped_parents = 0
    suppressed_fragments = 0
    ranked = sorted((element for element in elements if not element.dropped),
                    key=lambda element: -element.area)
    for element in ranked:
        if element.dropped or element.area <= 0:
            continue
        # Every contained element is judged on its own.  Two readings of one
        # element alternate, so a fragment is reported mostly on the frames
        # the enclosing box is missing from, while siblings inside a container
        # are reported alongside it however neatly they tile it.
        children = []
        for other in ranked:
            if (other is element or other.dropped or
                    other.area > 0.7 * element.area or
                    _containment(other.canonical, element.canonical)
                    < config.group_containment):
                continue
            alone = len(other.observed - element.observed) / max(1, len(other.observed))
            if alone < config.group_complementary:
                continue
            if not _same_element(pixels, element, other, config):
                continue
            children.append(other)
        if not children:
            continue
        child_frames: set[int] = set()
        for child in children:
            child_frames |= child.frames
        if not child_frames:
            continue
        observed_children: set[int] = set()
        for child in children:
            observed_children |= child.observed
        if not observed_children:
            continue
        # The vote counts only what the detector reported by itself.
        observed_shared = element.observed | observed_children
        parent_wins = (len(element.observed) >= config.group_min_frames and
                       len(element.observed) >= config.group_min_share * len(observed_shared))
        shared = element.frames | child_frames
        if parent_wins:
            low, high = min(shared), max(shared)
            host = element.host
            for position in sorted(child_frames - element.frames):
                if not low <= position <= high:
                    continue
                source, anchor = element.nearest(position)
                before = [p for p in sorted(element.frames) if p < position]
                after = [p for p in sorted(element.frames) if p > position]
                if before and after:
                    left, right = before[-1], after[0]
                    weight = (position - left) / (right - left)
                    box = (element.box_at(left) * (1 - weight) +
                           element.box_at(right) * weight)
                else:
                    box = source.boxes[anchor].copy()
                synthetic = dict(source.items[anchor])
                synthetic["detection_source"] = "temporal_group_merge"
                synthetic["bbox_xyxy"] = _rounded(box)
                host.add(position, box.astype(np.float32), synthetic)
                element.frames.add(position)
                merged_frames += 1
            for child in children:
                for track in child.tracks:
                    for position in list(track.positions):
                        if not low <= position <= high:
                            continue
                        track.remove(position)
                        suppressed_fragments += 1
                    if not track.positions:
                        track.dropped = True
                child.refresh()
                if not child.frames:
                    child.drop()
            element.refresh()
        else:
            confidence = float(np.median([element.confidence(position)
                                          for position in element.observed]))
            # Losing the vote is not on its own a reason to delete a box the
            # detector keeps drawing with some confidence; only a weak box
            # that the fragments clearly outnumber is dropped.
            if (confidence < config.group_drop_confidence and
                    len(element.observed) <= 0.25 * len(observed_shared)):
                element.drop()
                dropped_parents += 1
    return {"group_merged_frames": merged_frames,
            "group_dropped_parents": dropped_parents,
            "group_suppressed_fragments": suppressed_fragments}


def _absorb_siblings(elements: list[_Element], config: TemporalConfig) -> dict[str, int]:
    """Use the wider box the detector itself draws when it draws one.

    The same element is regressed on some frames as a box that covers its
    neighbour and on others as a box that stops short of it.  The neighbour
    is reported separately only on the frames that stop short, which says the
    two readings are one element, so the wider one is used throughout.  A box
    that simply sits inside another on every frame is a real child element
    and is left alone.
    """
    widened = 0
    absorbed = 0
    live = [element for element in elements if not element.dropped and element.frames]
    for element in live:
        if element.dropped:
            continue
        boxes = {position: element.box_at(position) for position in element.observed}
        boxes = {position: box for position, box in boxes.items() if box is not None}
        if len(boxes) < 2 * config.group_min_frames:
            continue
        widest = max(boxes.values(), key=_area)
        for other in live:
            if (other is element or other.dropped or not other.frames or
                    _area(other.canonical) > 0.7 * _area(widest) or
                    _containment(other.canonical, widest) < config.group_containment):
                continue
            wide = [position for position, box in boxes.items()
                    if _containment(other.canonical, box) >= config.group_containment]
            narrow = [position for position, box in boxes.items()
                      if _containment(other.canonical, box) < config.group_containment]
            if len(wide) < config.group_min_frames or not narrow:
                continue
            # The neighbour has to be what the narrow reading leaves out: it
            # is reported where the box stops short and not where it does not.
            seen_narrow = sum(position in other.observed for position in narrow)
            seen_wide = sum(position in other.observed for position in wide)
            if seen_narrow < 0.5 * len(narrow) or seen_wide > 0.1 * len(wide):
                continue
            for position in sorted(set(other.frames) & set(element.frames)):
                box = element.box_at(position)
                small = other.box_at(position)
                if box is None or small is None:
                    continue
                sibling = other.track_of(position)
                if _containment(small, box) >= config.group_containment:
                    # The wider box is already drawn here, so a separate box
                    # for the neighbour is a second reading of the same thing.
                    if sibling is not None:
                        sibling.remove(position)
                        absorbed += 1
                    continue
                merged = np.array([min(box[0], small[0]), min(box[1], small[1]),
                                   max(box[2], small[2]), max(box[3], small[3])],
                                  dtype=np.float32)
                # The narrow reading has to sit inside the wide one and the
                # two together have to reproduce it.  A box that merely moved
                # is not a narrow reading of where it used to be.
                if _containment(box, widest) < 0.9 or _iou(merged, widest) < 0.8:
                    continue
                track = element.track_of(position)
                if track is None:
                    continue
                item = track.items[position]
                item["bbox_xyxy"] = _rounded(merged)
                item["detection_source"] = "temporal_sibling_absorbed"
                track.boxes[position] = merged
                widened += 1
                if sibling is not None:
                    sibling.remove(position)
                    absorbed += 1
            other.refresh()
            if not other.frames:
                other.drop()
            element.refresh()
    return {"widened_boxes": widened, "absorbed_siblings": absorbed}


def _text_lines(elements: list[_Element], height: int,
                config: TemporalConfig) -> list[list[_Element]]:
    """Chain elements that sit side by side on one tight row of glyphs.

    Character-level boxes on a line of text nearly touch, while separate
    controls on the same row are spaced out, so the gap between neighbours,
    measured against their own height, tells the two apart.
    """
    ceiling = config.line_max_height_ratio * height

    def glyph(element: _Element) -> bool:
        box = element.canonical
        box_height = box[3] - box[1]
        # A character is small and about as wide as it is tall.  A row of
        # tightly spaced controls, or a pair of full-height panels, is not a
        # line of text however neatly it lines up.
        return (0 < box_height <= ceiling and
                (box[2] - box[0]) <= config.line_member_aspect * box_height)

    live = sorted((element for element in elements
                   if not element.dropped and element.frames and glyph(element)),
                  key=lambda element: element.canonical[0])
    used: set[int] = set()
    lines: list[list[_Element]] = []
    for index, first in enumerate(live):
        if index in used:
            continue
        chain = [first]
        chain_indexes = [index]
        for other_index in range(index + 1, len(live)):
            if other_index in used:
                continue
            left = chain[-1].canonical
            right = live[other_index].canonical
            left_height = left[3] - left[1]
            right_height = right[3] - right[1]
            if min(left_height, right_height) <= 0:
                continue
            if (min(left_height, right_height) / max(left_height, right_height)
                    < config.line_height_ratio):
                continue
            overlap = min(left[3], right[3]) - max(left[1], right[1])
            if overlap < config.line_overlap * min(left_height, right_height):
                continue
            gap = right[0] - left[2]
            if gap < -0.3 * min(left_height, right_height):
                continue
            if gap > max(config.line_gap_floor,
                         config.line_gap_ratio * min(left_height, right_height)):
                continue
            chain.append(live[other_index])
            chain_indexes.append(other_index)
        if len(chain) >= config.line_min_members:
            used.update(chain_indexes)
            lines.append(chain)
    return lines


def _merge_text_lines(elements: list[_Element], height: int,
                      config: TemporalConfig) -> dict[str, int]:
    """Replace a row of glyph boxes with one box for the line."""
    merged = 0
    lines = 0
    for line in _text_lines(elements, height, config):
        host = max(line, key=lambda element: len(element.frames))
        others = [element for element in line if element is not host]
        frames = sorted({position for element in line for position in element.frames})
        touched = False
        for position in frames:
            present = [element for element in line
                       if element.box_at(position) is not None]
            if len(present) < 2:
                continue
            stacked = np.stack([element.box_at(position) for element in present])
            union = np.array([stacked[:, 0].min(), stacked[:, 1].min(),
                              stacked[:, 2].max(), stacked[:, 3].max()],
                             dtype=np.float32)
            source = max(present, key=lambda element: element.confidence(position))
            if host.box_at(position) is None:
                host.host.add(position, union, dict(source.item_at(position)))
            track = host.track_of(position)
            if track is None:
                continue
            item = track.items[position]
            item["bbox_xyxy"] = _rounded(union)
            item["detection_source"] = "temporal_line_merge"
            item["confidence"] = max(element.confidence(position)
                                     for element in present)
            track.boxes[position] = union
            for element in others:
                member = element.track_of(position)
                if member is not None:
                    member.remove(position)
            merged += 1
            touched = True
        for element in others:
            element.refresh()
            if not element.frames:
                element.drop()
        host.refresh()
        lines += touched
    return {"text_lines_merged": lines, "text_line_frames": merged}


def _tighten_large_boxes(tracks: list[_Track], pixels: _Pixels, frame_area: float,
                         config: TemporalConfig) -> int:
    """Pull a box that covers much of the page in to what it actually shows.

    The detector sometimes regresses a whole column of the page around one
    block of content.  Such a box is correct about where the content is and
    wrong about how far it reaches, so its own pixels decide its extent.
    """
    tightened = 0
    limit = config.tighten_area_ratio * frame_area
    for track in tracks:
        if track.dropped:
            continue
        for position in list(track.positions):
            box = track.boxes[position]
            if _area(box) < limit:
                continue
            content = pixels.content_bounds(position, box, config)
            if content is None:
                continue
            content = np.array([max(box[0], content[0]), max(box[1], content[1]),
                                min(box[2], content[2]), min(box[3], content[3])],
                               dtype=np.float32)
            if (content[2] - content[0] < 8 or content[3] - content[1] < 8 or
                    _area(content) > config.tighten_min_shrink * _area(box)):
                continue
            item = track.items[position]
            item["bbox_before_temporal_tighten_xyxy"] = item["bbox_xyxy"]
            item["bbox_xyxy"] = _rounded(content)
            track.boxes[position] = content
            tightened += 1
    return tightened


def _drop_duplicates(tracks: list[_Track], config: TemporalConfig) -> int:
    """Remove a second box drawn over one the pipeline already produced.

    Separate stages can arrive at the same element from different sides -- a
    line assembled from glyphs and the line box the detector drew itself --
    and two boxes on one element are worse than either alone.
    """
    removed = 0
    frames: dict[int, list[tuple[_Track, np.ndarray]]] = {}
    for track in tracks:
        if track.dropped:
            continue
        for position in track.positions:
            frames.setdefault(position, []).append((track, track.boxes[position]))
    for position, entries in frames.items():
        entries.sort(key=lambda entry: (
            -float(entry[0].items[position]["confidence"]),
            position not in entry[0].observed,
        ))
        kept: list[np.ndarray] = []
        for track, box in entries:
            if any(_iou(box, other) >= config.duplicate_iou for other in kept):
                track.remove(position)
                removed += 1
                continue
            kept.append(box)
    return removed


def _quiet_mask(values: np.ndarray, noise: float, window: int) -> np.ndarray:
    """Mark the frames where a coordinate only carries regression noise."""
    count = len(values)
    mask = np.zeros(count, dtype=bool)
    for index in range(count):
        low = max(0, index - window)
        high = min(count, index + window + 1)
        chunk = values[low:high]
        mask[index] = float(chunk.max() - chunk.min()) <= 2 * noise
    return mask


def _smooth_coordinate(values: np.ndarray, noise: float, window: int) -> np.ndarray:
    """Pin motionless stretches to one value; keep real motion linear."""
    smoothed = values.copy()
    quiet = _quiet_mask(values, noise, window)
    index = 0
    count = len(values)
    while index < count:
        if quiet[index]:
            # A whole motionless run takes a single value, so a static
            # element's box stops moving altogether instead of creeping.
            end = index
            while end + 1 < count and quiet[end + 1]:
                end += 1
            smoothed[index:end + 1] = np.median(values[index:end + 1])
            index = end + 1
            continue
        low = max(0, index - window)
        high = min(count, index + window + 1)
        chunk = values[low:high]
        x = np.arange(low, high, dtype=np.float32)
        centred = x - x.mean()
        denominator = float((centred ** 2).sum()) or 1.0
        slope = float((centred * (chunk - chunk.mean())).sum()) / denominator
        # A local linear fit leaves a steady ramp untouched, so a reveal or a
        # translation keeps its speed while its noise is averaged out.
        smoothed[index] = chunk.mean() + slope * (index - x.mean())
        index += 1
    return smoothed


def _smooth_track(track: _Track, config: TemporalConfig) -> int:
    """Remove regression wobble while keeping deliberate motion intact."""
    positions = track.positions
    if len(positions) < 3:
        return 0
    boxes = np.stack([track.boxes[p] for p in positions]).astype(np.float32)
    sizes = np.maximum(boxes[:, 2:] - boxes[:, :2], 1.0)
    typical = np.maximum(np.median(sizes, axis=0), 12.0)
    noise = np.maximum(config.smooth_noise_floor,
                       config.smooth_noise_ratio * np.tile(typical, 2))
    break_limit = max(config.smooth_break_floor,
                      config.smooth_break_ratio * float(typical.min()))
    steps = np.abs(np.diff(boxes, axis=0)).max(axis=1)
    frame_gaps = np.diff(positions)
    # A redefinition of the box (new text revealed, element replaced) must not
    # be averaged with what came before it.
    breaks = np.flatnonzero((steps > break_limit) | (frame_gaps > 1)) + 1
    segments = np.split(np.arange(len(positions)), breaks)
    changed = 0
    for segment in segments:
        if len(segment) < 3:
            continue
        values = boxes[segment]
        smoothed = np.stack([
            _smooth_coordinate(values[:, coordinate], float(noise[coordinate]),
                               config.smooth_window)
            for coordinate in range(4)
        ], axis=1)
        for offset, index in enumerate(segment):
            if float(np.max(np.abs(smoothed[offset] - values[offset]))) < 0.05:
                continue
            box = smoothed[offset]
            if box[2] - box[0] < 1 or box[3] - box[1] < 1:
                continue
            position = positions[index]
            item = track.items[position]
            item["bbox_before_temporal_smoothing_xyxy"] = item.get(
                "bbox_xyxy", _rounded(track.boxes[position]))
            item["bbox_xyxy"] = _rounded(box)
            track.boxes[position] = box
            changed += 1
    return changed


def consolidate(frame_detections: list[list[dict]], grays: list[np.ndarray],
                scale: float, width: int, height: int,
                config: TemporalConfig | None = None
                ) -> tuple[list[list[dict]], list[list[int]], dict[str, Any]]:
    """Return consolidated detections, the ids they dropped, and diagnostics.

    ``frame_detections`` holds the raw per-frame detections of the whole clip,
    before the output confidence cutoff.  ``grays`` holds one grayscale image
    per frame, scaled by ``scale``.
    """
    config = config or TemporalConfig()
    if len(frame_detections) != len(grays):
        raise ValueError("Each frame needs one grayscale image")
    if not 0 <= config.min_confidence <= 1:
        raise ValueError("min_confidence must be between 0 and 1")
    if config.max_gap < 1:
        raise ValueError("max_gap must be at least 1")

    frame_items = [[dict(item) for item in detections]
                   for detections in frame_detections]
    # Identity of the copied dicts, so dropped originals can be reported
    # without relying on ids that are renumbered at the end.
    origin = [{id(item): int(item["id"]) for item in items} for items in frame_items]
    frame_boxes = [[np.asarray(item["bbox_xyxy"], dtype=np.float32)
                    for item in detections] for detections in frame_items]
    pixels = _Pixels(grays, scale)

    blank_removed = _reject_blank(frame_boxes, frame_items, pixels, config)
    tracks = _associate(frame_boxes, frame_items, config)
    elements = _elements(tracks)
    statistics: dict[str, Any] = {"pipeline_version": PIPELINE_VERSION,
                                  "blank_boxes_removed": blank_removed,
                                  "track_count": len(tracks),
                                  "element_count": len(elements)}
    statistics.update(_confirm_elements(elements, pixels, config))
    if config.line_merge:
        statistics.update(_merge_text_lines(elements, height, config))
        # A line that was just assembled from glyphs is the same element as
        # the line box the detector draws on its better frames, so the tracks
        # are regrouped before the two readings are compared.
        elements = _elements([track for track in tracks if not track.dropped])
    if config.group_merge:
        statistics.update(_consolidate_groups(elements, pixels, config))
        statistics.update(_absorb_siblings(elements, config))

    if config.tighten:
        statistics["tightened_boxes"] = _tighten_large_boxes(
            tracks, pixels, float(width) * float(height), config)
    statistics["duplicate_boxes_removed"] = _drop_duplicates(tracks, config)
    for track in tracks:
        if not track.positions:
            track.dropped = True
    smoothed = 0
    for track in tracks:
        if not track.dropped:
            smoothed += _smooth_track(track, config)
    statistics["smoothed_boxes"] = smoothed

    output: list[list[dict]] = [[] for _ in frame_items]
    limit = np.array([width, height, width, height], dtype=np.float32)
    for track in tracks:
        if track.dropped:
            continue
        for position in track.positions:
            item = track.items.get(position)
            if item is None:
                continue
            box = np.clip(track.boxes[position], 0, limit)
            if box[2] - box[0] < 1 or box[3] - box[1] < 1:
                continue
            item["bbox_xyxy"] = _rounded(box)
            item["temporal_track_id"] = track.ident
            output[position].append(item)
    removed: list[list[int]] = []
    for position, items in enumerate(output):
        items.sort(key=lambda item: -float(item["confidence"]))
        surviving = {origin[position][id(item)] for item in items
                     if id(item) in origin[position]}
        removed.append(sorted(set(origin[position].values()) - surviving))
        for index, item in enumerate(items):
            item["id"] = index
    statistics["output_detection_count"] = sum(len(items) for items in output)
    statistics["removed_source_detection_count"] = sum(len(ids) for ids in removed)
    return output, removed, statistics


OUTPUT_CANONICALIZATION_VERSION = "final_pixel_text_rows_and_stale_fragments"


def _output_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _output_intersection(left: list[float], right: list[float]) -> float:
    return max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )


def _stale_nested_fragment(child: dict, detections: list[dict]) -> bool:
    # Native detections are preserved. A temporal extension is only a held
    # reading, and should not double-count a stronger enclosing native box.
    if child.get("detection_source") != "temporal_extension":
        return False
    box = child["bbox_xyxy"]
    area = _output_area(box)
    if area <= 0:
        return False
    for parent in detections:
        if parent is child or parent.get("detection_source", "gpa") != "gpa":
            continue
        if parent.get("class_name") != child.get("class_name"):
            continue
        parent_box = parent["bbox_xyxy"]
        parent_area = _output_area(parent_box)
        if (parent_area > 0
                and area / parent_area <= 0.20
                and _output_intersection(box, parent_box) / area >= 0.98
                and float(parent["confidence"]) >= 1.5 * float(child["confidence"])):
            return True
    return False


def _dark_text_rows(frame: np.ndarray, box: list[float]) -> list[list[float]]:
    frame_height, frame_width = frame.shape[:2]
    x1 = max(0, min(frame_width, int(np.floor(box[0]))))
    y1 = max(0, min(frame_height, int(np.floor(box[1]))))
    x2 = max(0, min(frame_width, int(np.ceil(box[2]))))
    y2 = max(0, min(frame_height, int(np.ceil(box[3]))))
    width, height = x2 - x1, y2 - y1
    if width < 100 or height < 35 or width < 3.0 * height:
        return []
    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Restrict this operation to dark text on a nearly white, neutral page.
    if (float(np.mean(gray > 235)) < 0.65
            or float(np.mean(crop.max(axis=2) - crop.min(axis=2) > 25)) > 0.03):
        return []
    dark = gray < 125
    active = np.count_nonzero(dark, axis=1) >= max(6, int(round(width * 0.025)))
    bands: list[tuple[int, int]] = []
    start = None
    for row, present in enumerate(active):
        if present and start is None:
            start = row
        elif not present and start is not None:
            bands.append((start, row))
            start = None
    if start is not None:
        bands.append((start, height))
    bands = [(top, bottom) for top, bottom in bands
             if bottom - top >= max(8, int(height * 0.18))]
    if len(bands) != 2 or bands[1][0] - bands[0][1] < max(6, int(height * 0.06)):
        return []

    rows = []
    x_pad = max(4, int(round(width * 0.02)))
    y_pad = max(3, int(round(height * 0.07)))
    for top, bottom in bands:
        columns = np.flatnonzero(np.any(dark[top:bottom], axis=0))
        if len(columns) == 0 or columns[-1] - columns[0] < width * 0.35:
            return []
        rows.append([
            float(max(0, x1 + int(columns[0]) - x_pad)),
            float(max(0, y1 + top - y_pad)),
            float(min(frame_width, x1 + int(columns[-1]) + 1 + x_pad)),
            float(min(frame_height, y1 + bottom + y_pad)),
        ])
    return rows


def canonicalize_output(frame: np.ndarray, detections: list[dict]) -> tuple[list[dict], dict]:
    """Return normalized final boxes and counts for per-frame diagnostics."""
    kept = [item for item in detections if not _stale_nested_fragment(item, detections)]
    suppressed = len(detections) - len(kept)
    output = []
    split = 0
    for item in kept:
        rows = _dark_text_rows(frame, item["bbox_xyxy"])
        if not rows:
            output.append(dict(item))
            continue
        for row_index, row in enumerate(rows):
            child = dict(item)
            child["bbox_xyxy"] = [round(value, 2) for value in row]
            child["detection_source"] = "pixel_text_row_split"
            child["split_from_detection_id"] = item["id"]
            child["split_row_index"] = row_index
            output.append(child)
        split += 1
    for index, item in enumerate(output):
        item["id"] = index
    return output, {
        "suppressed_stale_nested_fragments": suppressed,
        "split_two_row_text_boxes": split,
    }
