#!/usr/bin/env python3
"""Run GPA-GUI-Detector on a video and save video/frame/JSON artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from box_refinement import (
    PIPELINE_VERSION,
    EVIDENCE_PIPELINE_VERSION,
    TemporalLargeBoxRefiner,
    EvidenceTemporalRefiner,
)
from temporal_refinement import (
    PIPELINE_VERSION as TEMPORAL_PIPELINE_VERSION,
    OUTPUT_CANONICALIZATION_VERSION,
    TemporalConfig,
    consolidate,
    canonicalize_output,
)


ROOT = Path(__file__).resolve().parent

# Grayscale scale factor of the analysis images kept for the temporal pass.
ANALYSIS_SCALE = 0.5
# Below this factor the pixel checks lose small text, so a long clip caps the
# memory the cached analysis images take instead of scaling down further.
LONG_CLIP_ANALYSIS_SCALE = 0.25
LONG_CLIP_FRAMES = 600


def _encode_browser_video(raw_video: Path, output_video: Path) -> None:
    """Convert OpenCV's temporary MP4 to browser-compatible H.264."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg is required to create a browser-compatible H.264 video; "
            f"the temporary mp4v file was kept at {raw_video}"
        )
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
        "medium",
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
            "ffmpeg could not encode the browser-compatible video; "
            f"the temporary file was kept at {raw_video}: "
            f"{completed.stderr.strip()}"
        )
    raw_video.unlink()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run GPA-GUI-Detector on video frames and save an annotated H.264 "
            "video, every annotated frame, and per-frame JSON."
        )
    )
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Output directory (default: <video_stem>_gpa_debug beside the video)",
    )
    parser.add_argument("--model", type=Path, default=ROOT / "model.pt")
    parser.add_argument(
        "--device",
        default=None,
        help="Ultralytics device, e.g. 0, 1, 2, cpu (default: CUDA 0 if available)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of selected frames inferred together (default: 16)",
    )
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument(
        "--min-output-confidence", type=float, default=0.1,
        help="Remove final boxes with confidence below this value (default: 0.1)",
    )
    parser.add_argument("--iou", type=float, default=0.1)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument(
        "--max-box-area-ratio",
        type=float,
        default=0.5,
        help="Drop boxes larger than this fraction of the frame; 1.0 disables it",
    )
    parser.add_argument(
        "--recover-large-boxes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recover visible foreground inside oversized boxes on flat backgrounds",
    )
    parser.add_argument(
        "--refinement-mode", choices=("evidence", "legacy"), default="evidence",
        help="Evidence-based text-line refinement (default) or the old large-box patch",
    )
    parser.add_argument(
        "--refine-box-area-ratio",
        type=float,
        default=0.12,
        help="Try tightening boxes at or above this frame-area ratio",
    )
    parser.add_argument(
        "--refine-box-width-ratio",
        type=float,
        default=0.55,
        help="Try tightening boxes at or above this frame-width ratio",
    )
    parser.add_argument(
        "--refine-box-height-ratio",
        type=float,
        default=0.35,
        help="Try tightening boxes at or above this frame-height ratio",
    )
    parser.add_argument(
        "--temporal-large-box-max-gap",
        type=int,
        default=4,
        help="Maximum selected-frame gap bridged by temporal large-box recovery",
    )
    parser.add_argument(
        "--temporal-consistency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Consolidate detections across the whole clip after inference",
    )
    parser.add_argument(
        "--temporal-max-gap",
        type=int,
        default=8,
        help="Largest hole in frames bridged inside one track (default: 8)",
    )
    parser.add_argument(
        "--temporal-group-merge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Unify elements reported both as one box and as fragments",
    )
    parser.add_argument(
        "--temporal-line-merge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Join a row of single-character boxes into one box per text line",
    )
    parser.add_argument(
        "--temporal-tighten",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pull a box covering much of the page in to its visible content",
    )
    parser.add_argument("--line-width", type=int, default=1)
    parser.add_argument("--font-scale", type=float, default=0.28)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser


def _validate(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    video = args.video.expanduser().resolve()
    model = args.model.expanduser().resolve()
    if not video.is_file():
        raise ValueError(f"Video does not exist: {video}")
    if not model.is_file():
        raise ValueError(f"Model does not exist: {model}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.frame_step < 1:
        raise ValueError("--frame-step must be at least 1")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be at least 1")
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
        raise ValueError("--conf and --iou must be between 0 and 1")
    if not 0 <= args.min_output_confidence <= 1:
        raise ValueError("--min-output-confidence must be between 0 and 1")
    if args.imgsz < 32:
        raise ValueError("--imgsz must be at least 32")
    if not 0 < args.max_box_area_ratio <= 1:
        raise ValueError("--max-box-area-ratio must be in (0, 1]")
    for name in (
        "refine_box_area_ratio",
        "refine_box_width_ratio",
        "refine_box_height_ratio",
    ):
        if not 0 < getattr(args, name) <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be in (0, 1]")
    if args.temporal_large_box_max_gap < 1:
        raise ValueError("--temporal-large-box-max-gap must be at least 1")
    if args.temporal_max_gap < 1:
        raise ValueError("--temporal-max-gap must be at least 1")
    if args.line_width < 1 or args.font_scale <= 0:
        raise ValueError("--line-width and --font-scale must be positive")
    if not 0 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 0 and 100")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else video.with_name(f"{video.stem}_gpa_debug")
    )
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"Output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise ValueError(f"Output directory is not empty: {output_dir}")
    return video, model, output_dir


def _serialize_result(result) -> list[dict]:
    detections = []
    if result.boxes is None:
        return detections
    boxes = result.boxes.xyxy.detach().cpu().tolist()
    scores = result.boxes.conf.detach().cpu().tolist()
    classes = result.boxes.cls.detach().cpu().tolist()
    for index, (box, score, class_id) in enumerate(zip(boxes, scores, classes)):
        class_index = int(class_id)
        detections.append(
            {
                "id": index,
                "class_id": class_index,
                "class_name": result.names.get(class_index, str(class_index)),
                "confidence": round(float(score), 6),
                "bbox_xyxy": [round(float(value), 2) for value in box],
                "detection_source": "gpa",
            }
        )
    return detections


def _filter_output_confidence(
    detections: list[dict], threshold: float
) -> tuple[list[dict], list[int]]:
    """Filter rendered/serialized boxes after refinement and tracking."""
    removed_ids = [item["id"] for item in detections
                   if float(item["confidence"]) < threshold]
    kept = [dict(item) for item in detections
            if float(item["confidence"]) >= threshold]
    for index, item in enumerate(kept):
        item["id"] = index
    return kept, removed_ids


def _render_detections(
    frame, detections: list[dict], line_width: int, font_scale: float
):
    """Draw both native GPA boxes and recovered foreground boxes."""
    canvas = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = (
            int(round(value)) for value in detection["bbox_xyxy"]
        )
        recovered = detection.get("detection_source") != "gpa"
        color = (0, 140, 255) if recovered else (0, 0, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, line_width)
        prefix = "R:" if recovered else ""
        label = f"{prefix}{detection['confidence']:.2f}"
        text_y = y1 - 2 if y1 >= 9 else min(canvas.shape[0] - 2, y1 + 8)
        cv2.putText(
            canvas,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


def run(args: argparse.Namespace) -> dict:
    video, model_path, output_dir = _validate(args)
    frames_dir = output_dir / "annotated_frames"
    results_dir = output_dir / "frame_results"
    save_visualizations = getattr(args, "save_visualizations", True)
    if save_visualizations:
        frames_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Invalid video metadata: {video}")

    device = args.device if args.device is not None else ("0" if torch.cuda.is_available() else "cpu")
    detector = YOLO(str(model_path))
    temporal = bool(getattr(args, "temporal_consistency", True))
    expected_frames = source_frame_count // max(1, args.frame_step)
    if args.max_frames is not None:
        expected_frames = min(expected_frames, args.max_frames)
    analysis_scale = (LONG_CLIP_ANALYSIS_SCALE if expected_frames > LONG_CLIP_FRAMES
                      else ANALYSIS_SCALE)
    large_box_refiner = TemporalLargeBoxRefiner(
        max_area_ratio=args.max_box_area_ratio,
        refine_area_ratio=args.refine_box_area_ratio,
        refine_width_ratio=args.refine_box_width_ratio,
        refine_height_ratio=args.refine_box_height_ratio,
        frame_step=args.frame_step,
        max_track_gap=args.temporal_large_box_max_gap,
    )
    evidence_refiner = EvidenceTemporalRefiner(
        frame_step=args.frame_step,
        max_area_ratio=args.max_box_area_ratio,
        max_track_gap=args.temporal_large_box_max_gap,
        min_recovery_confidence=args.min_output_confidence,
        stabilize=not temporal,
    )
    refinement_mode = getattr(args, "refinement_mode", "evidence")
    frame_area = width * height
    started = time.perf_counter()

    # Pass 1: inference only.  Detections are kept unfiltered so the temporal
    # pass can use sub-threshold boxes as evidence, and one small grayscale
    # image per frame is kept for its pixel checks.
    source_indexes: list[int] = []
    frame_detections: list[list[dict]] = []
    frame_diagnostics: list[dict] = []
    frame_oversized: list[int] = []
    grays: list = []
    processed_count = 0
    next_source_index = 0
    try:
        reached_end = False
        while not reached_end:
            batch = []
            while len(batch) < args.batch_size:
                if (
                    args.max_frames is not None
                    and processed_count + len(batch) >= args.max_frames
                ):
                    reached_end = True
                    break
                ok, frame = capture.read()
                if not ok:
                    reached_end = True
                    break
                source_index = next_source_index
                next_source_index += 1
                if source_index % args.frame_step == 0:
                    batch.append((source_index, frame))
            if not batch:
                break

            results = detector.predict(
                source=[frame for _, frame in batch],
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=device,
                batch=args.batch_size,
                verbose=False,
            )
            if len(results) != len(batch):
                raise RuntimeError(
                    f"Detector returned {len(results)} results for {len(batch)} frames"
                )

            for (source_index, frame), result in zip(batch, results):
                raw_detections = _serialize_result(result)
                filtered_large_box_count = sum(
                    (
                        max(0.0, item["bbox_xyxy"][2] - item["bbox_xyxy"][0])
                        * max(0.0, item["bbox_xyxy"][3] - item["bbox_xyxy"][1])
                        / frame_area
                    )
                    > args.max_box_area_ratio
                    for item in raw_detections
                )
                if args.recover_large_boxes and refinement_mode == "evidence":
                    detections, recovery_diagnostics = evidence_refiner.process(
                        frame, raw_detections, source_index
                    )
                else:
                    detections, recovery_diagnostics = large_box_refiner.process(
                        frame, raw_detections, source_index,
                        enabled=args.recover_large_boxes,
                    )
                source_indexes.append(source_index)
                frame_detections.append(detections)
                frame_diagnostics.append(recovery_diagnostics)
                frame_oversized.append(filtered_large_box_count)
                if temporal:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    grays.append(cv2.resize(gray, None, fx=analysis_scale,
                                            fy=analysis_scale,
                                            interpolation=cv2.INTER_AREA))
                processed_count += 1

    finally:
        capture.release()

    if not source_indexes:
        raise ValueError("The video contains no decodable selected frames")

    # Pass 2: whole-clip consolidation.
    removed_per_frame: list[list[int]] = []
    if temporal:
        config = TemporalConfig(
            min_confidence=args.min_output_confidence,
            max_gap=args.temporal_max_gap,
            group_merge=bool(getattr(args, "temporal_group_merge", True)),
            line_merge=bool(getattr(args, "temporal_line_merge", True)),
            tighten=bool(getattr(args, "temporal_tighten", True)),
        )
        detection_done = time.perf_counter()
        frame_detections, removed_per_frame, temporal_statistics = consolidate(
            frame_detections, grays, analysis_scale, width, height, config,
        )
        grays.clear()
        temporal_statistics["enabled"] = True
        temporal_statistics["max_gap"] = args.temporal_max_gap
        temporal_statistics["analysis_scale"] = analysis_scale
        temporal_statistics["seconds"] = round(
            time.perf_counter() - detection_done, 3)
    else:
        temporal_statistics = {"enabled": False}
        filtered: list[list[dict]] = []
        for detections in frame_detections:
            kept, removed_ids = _filter_output_confidence(
                detections, args.min_output_confidence
            )
            filtered.append(kept)
            removed_per_frame.append(removed_ids)
        frame_detections = filtered

    # Pass 3: render the consolidated boxes and write every artifact.
    raw_video = output_dir / "gpa_annotated_raw.mp4"
    output_video = output_dir / "gpa_annotated.mp4"
    writer = cv2.VideoWriter(
        str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"),
        fps / args.frame_step, (width, height),
    ) if save_visualizations else None
    if save_visualizations and not writer.isOpened():
        raise RuntimeError(f"Could not create temporary output video: {raw_video}")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        if writer is not None:
            writer.release()
        raise RuntimeError(f"Could not reopen video: {video}")
    entries = []
    try:
        position = 0
        next_source_index = 0
        while position < len(source_indexes):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(
                    f"Could not reread frame {source_indexes[position]} of {video}"
                )
            source_index = next_source_index
            next_source_index += 1
            if source_index != source_indexes[position]:
                continue
            detections, canonicalization = canonicalize_output(
                frame, frame_detections[position]
            )
            recovery_diagnostics = frame_diagnostics[position]
            recovery_diagnostics["output_canonicalization"] = canonicalization
            low_confidence_ids = removed_per_frame[position]
            recovery_diagnostics["low_confidence_removed_count"] = len(
                low_confidence_ids
            )
            recovery_diagnostics["low_confidence_removed_ids"] = low_confidence_ids
            recovery_diagnostics["output_detection_count"] = len(detections)
            recovered_large_box_count = sum(
                item.get("detection_source", "gpa") not in ("gpa", "pixel_text_row_split")
                for item in detections
            )
            if save_visualizations:
                rendered = _render_detections(
                    frame, detections, args.line_width, args.font_scale
                )
                if rendered.shape[:2] != (height, width):
                    rendered = cv2.resize(rendered, (width, height),
                                          interpolation=cv2.INTER_LINEAR)
                writer.write(rendered)

            stem = f"frame_{source_index:06d}"
            frame_path = frames_dir / f"{stem}.jpg"
            result_path = results_dir / f"{stem}.json"
            if save_visualizations and not cv2.imwrite(
                str(frame_path), rendered,
                [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
            ):
                raise RuntimeError(f"Could not write frame: {frame_path}")

            entry = {
                "source_frame_index": source_index,
                "time_sec": source_index / fps,
                "detection_count": len(detections),
                "low_confidence_removed_count": len(low_confidence_ids),
                "filtered_large_box_count": frame_oversized[position],
                "recovered_large_box_count": recovered_large_box_count,
                "suppressed_large_box_count": recovery_diagnostics.get(
                    "suppressed_oversized_count", 0
                ),
                "large_box_recovery": recovery_diagnostics,
                "detections": detections,
                "annotated_frame": str(frame_path.relative_to(output_dir)) if save_visualizations else None,
                "result_file": str(result_path.relative_to(output_dir)),
            }
            result_path.write_text(
                json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            entries.append(entry)
            position += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if not entries:
        raise ValueError("The video contains no decodable selected frames")

    if save_visualizations:
        _encode_browser_video(raw_video, output_video)
    elapsed = time.perf_counter() - started
    summary = {
        "mode": "gpa_gui_detector_video_debug",
        "temporal_pipeline_version": (
            TEMPORAL_PIPELINE_VERSION if temporal else None
        ),
        "temporal_consistency": temporal_statistics,
        "output_canonicalization_version": OUTPUT_CANONICALIZATION_VERSION,
        "large_box_pipeline_version": (
            EVIDENCE_PIPELINE_VERSION
            if args.recover_large_boxes and refinement_mode == "evidence"
            else PIPELINE_VERSION
        ),
        "model": str(model_path),
        "input_video": str(video),
        "annotated_video": str(output_video) if save_visualizations else None,
        "video_codec": "h264",
        "annotated_frames_directory": str(frames_dir) if save_visualizations else None,
        "frame_results_directory": str(results_dir),
        "source": {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": source_frame_count,
        },
        "parameters": {
            "device": str(device),
            "batch_size": args.batch_size,
            "frame_step": args.frame_step,
            "max_frames": args.max_frames,
            "conf": args.conf,
            "min_output_confidence": args.min_output_confidence,
            "iou": args.iou,
            "imgsz": args.imgsz,
            "max_box_area_ratio": args.max_box_area_ratio,
            "recover_large_boxes": args.recover_large_boxes,
            "refinement_mode": refinement_mode,
            "refine_box_area_ratio": args.refine_box_area_ratio,
            "refine_box_width_ratio": args.refine_box_width_ratio,
            "refine_box_height_ratio": args.refine_box_height_ratio,
            "temporal_large_box_max_gap": args.temporal_large_box_max_gap,
            "temporal_consistency": temporal,
            "temporal_max_gap": args.temporal_max_gap,
            "temporal_group_merge": args.temporal_group_merge,
            "temporal_line_merge": args.temporal_line_merge,
            "temporal_tighten": args.temporal_tighten,
            "line_width": args.line_width,
            "font_scale": args.font_scale,
        },
        "result": {
            "processed_frame_count": processed_count,
            "output_fps": fps / args.frame_step,
            "processing_seconds": elapsed,
            "total_detection_count": sum(item["detection_count"] for item in entries),
            "total_low_confidence_removed_count": sum(
                item["low_confidence_removed_count"] for item in entries
            ),
            "total_filtered_large_box_count": sum(
                item["filtered_large_box_count"] for item in entries
            ),
            "total_recovered_large_box_count": sum(
                item["recovered_large_box_count"] for item in entries
            ),
            "total_suppressed_large_box_count": sum(
                item["suppressed_large_box_count"] for item in entries
            ),
            "frames": [
                {
                    "source_frame_index": item["source_frame_index"],
                    "time_sec": item["time_sec"],
                    "detection_count": item["detection_count"],
                    "low_confidence_removed_count": item["low_confidence_removed_count"],
                    "filtered_large_box_count": item["filtered_large_box_count"],
                    "recovered_large_box_count": item["recovered_large_box_count"],
                    "suppressed_large_box_count": item["suppressed_large_box_count"],
                    "annotated_frame": item["annotated_frame"],
                    "result_file": item["result_file"],
                }
                for item in entries
            ],
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def detect_video(
    video: str | Path,
    output_dir: str | Path,
    *,
    model: str | Path = ROOT / "model.pt",
    device: str | int | None = None,
    batch_size: int = 16,
    frame_step: int = 1,
    max_frames: int | None = None,
    conf: float = 0.05,
    min_output_confidence: float = 0.1,
    iou: float = 0.1,
    imgsz: int = 1280,
    max_box_area_ratio: float = 0.5,
    recover_large_boxes: bool = True,
    refinement_mode: str = "evidence",
    refine_box_area_ratio: float = 0.12,
    refine_box_width_ratio: float = 0.55,
    refine_box_height_ratio: float = 0.35,
    temporal_large_box_max_gap: int = 4,
    temporal_consistency: bool = True,
    temporal_max_gap: int = 8,
    temporal_group_merge: bool = True,
    temporal_line_merge: bool = True,
    temporal_tighten: bool = True,
    line_width: int = 1,
    font_scale: float = 0.28,
    jpeg_quality: int = 95,
    save_visualizations: bool = True,
) -> dict:
    """Library entry point used by downstream video evaluation scripts.

    The output layout and validation are identical to the command-line tool.
    ``output_dir`` must be absent or empty; callers that implement caching
    should build into a temporary directory and publish it after completion.
    """
    args = argparse.Namespace(
        video=Path(video),
        output_dir=Path(output_dir),
        model=Path(model),
        device=None if device is None else str(device),
        batch_size=batch_size,
        frame_step=frame_step,
        max_frames=max_frames,
        conf=conf,
        min_output_confidence=min_output_confidence,
        iou=iou,
        imgsz=imgsz,
        max_box_area_ratio=max_box_area_ratio,
        recover_large_boxes=recover_large_boxes,
        refinement_mode=refinement_mode,
        refine_box_area_ratio=refine_box_area_ratio,
        refine_box_width_ratio=refine_box_width_ratio,
        refine_box_height_ratio=refine_box_height_ratio,
        temporal_large_box_max_gap=temporal_large_box_max_gap,
        temporal_consistency=temporal_consistency,
        temporal_max_gap=temporal_max_gap,
        temporal_group_merge=temporal_group_merge,
        temporal_line_merge=temporal_line_merge,
        temporal_tighten=temporal_tighten,
        line_width=line_width,
        font_scale=font_scale,
        jpeg_quality=jpeg_quality,
        save_visualizations=save_visualizations,
    )
    return run(args)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
