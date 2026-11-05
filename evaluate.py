#!/usr/bin/env python3
"""Record generated webpages, evaluate fidelity, and export one CSV per model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_KEYS = ("structure_score", "text_score", "color_score", "clip_score")
ANIMATION_KEYS = ("position_score", "size_score", "velocity_score", "clip_score")
CSV_METRICS = tuple(f"static_{key}" for key in STATIC_KEYS) + tuple(
    f"animation_{key}" for key in ANIMATION_KEYS
)

# Run heavy stages in separate processes so GPU memory is released between them.
GPA_COMMAND = """
import sys
from pathlib import Path
from utils.animation_score import GpaConfig, ensure_gpa_cache
video, output, weights, device, batch, refresh = sys.argv[1:]
ensure_gpa_cache(
    Path(video), Path(output), Path(weights),
    GpaConfig(device=device or None, batch_size=int(batch)),
    refresh=refresh == '1', visualize=False,
)
"""


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def fresh(path: Path, inputs: list[Path]) -> bool:
    return nonempty(path) and all(
        nonempty(item) and path.stat().st_mtime_ns >= item.stat().st_mtime_ns
        for item in inputs
    )


def image_path(value: str, directory: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else directory / path


def keyframes_ready(path: Path, video: Path) -> bool:
    try:
        frames = read_json(path)["keyframes"]
        return fresh(path, [video]) and len(frames) >= 2 and all(
            nonempty(image_path(item["image_path"], path.parent)) for item in frames
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def alignment_ready(path: Path, inputs: list[Path]) -> bool:
    try:
        payload = read_json(path)
        matches = payload["matches"]
        return fresh(path, inputs) and isinstance(matches, list) and all(
            nonempty(image_path(item["ground_truth_image"], path.parent))
            and nonempty(image_path(item["rendered_image"], path.parent))
            for item in matches
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def gpa_ready(directory: Path) -> bool:
    """Standalone detector output is reusable; cache.json and videos are optional."""
    try:
        frames = read_json(directory / "summary.json")["result"]["frames"]
        if not isinstance(frames, list) or not frames:
            return False
        for item in frames:
            data = read_json(directory / item["result_file"])
            int(data["source_frame_index"])
            float(data["time_sec"])
            if not isinstance(data["detections"], list):
                return False
            if any(len(detection["bbox_xyxy"]) != 4 for detection in data["detections"]):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def metric_values(path: Path, keys: tuple[str, ...]) -> list[float]:
    summary = read_json(path)["summary"]
    values = []
    for key in keys:
        value = summary[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Missing numeric {key}: {path}")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid {key}: {path}")
        values.append(float(value))
    return values


def scores_ready(path: Path, keys: tuple[str, ...], inputs: list[Path]) -> bool:
    try:
        metric_values(path, keys)
        return fresh(path, inputs)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def comparison_videos_ready(path: Path) -> bool:
    try:
        segments = read_json(path)["segments"]
        return all(
            nonempty(path.parent / segment["segment_name"] / "trajectory_comparison.mp4")
            for segment in segments if segment["status"] == "scored"
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run_command(command: list[str]) -> None:
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", env=environment,
    )
    if completed.returncode:
        raise RuntimeError(completed.stdout.strip() or f"Command failed: {command[0]}")


def run_stage(sample_id: str, label: str, ready, command: list[str], force=False, report=True) -> bool:
    if not force and ready():
        if report:
            print(f"[{sample_id}] {label}: reused", flush=True)
        return False
    run_command(command)
    if not ready():
        raise RuntimeError(f"{label} did not produce complete output files")
    if report:
        print(f"[{sample_id}] {label}: done", flush=True)
    return True


def evaluate_sample(args: argparse.Namespace, sample_id: str) -> None:
    run_dir = args.result / sample_id / args.model
    reference_dir = args.ground_truth_root / sample_id
    reference_video = args.data / sample_id / "ground_truth.mp4"
    rendered_video = run_dir / "rendered_video.mp4"
    if not nonempty(reference_video):
        raise FileNotFoundError(f"Reference video not found: {reference_video}")
    python = sys.executable
    common = ["--id", sample_id, "--run-name", args.model,
              "--generated-root", str(args.result),
              "--ground-truth-root", str(args.ground_truth_root)]

    recording = [args.node, str(PROJECT_ROOT / "utils/record_video.mjs"),
                 "--task-id", sample_id, "--model", args.model,
                 "--data", str(args.data), "--result", str(args.result), "--overwrite"]
    recorded = run_stage(sample_id, "Video recording", lambda: nonempty(rendered_video),
                         recording, args.overwrite)

    keyframes = reference_dir / "keyframes/keyframes.json"
    extraction = [python, str(PROJECT_ROOT / "utils/keyframe_extraction.py"),
                  "--task-id", sample_id, "--data", str(args.data),
                  "--ground-truth-root", str(args.ground_truth_root), "--refresh-ground-truth"]
    extracted = run_stage(sample_id, "Reference keyframe extraction",
                          lambda: keyframes_ready(keyframes, reference_video),
                          extraction, args.overwrite, report=False)

    matches = run_dir / "keyframe_alignment/keyframe_matches.json"
    alignment_inputs = [reference_video, rendered_video, keyframes]
    alignment = [python, str(PROJECT_ROOT / "utils/keyframe_alignment.py"),
                 *common, "--data", str(args.data)]
    aligned = run_stage(sample_id, "Keyframe alignment",
                        lambda: alignment_ready(matches, alignment_inputs), alignment,
                        args.overwrite or recorded or extracted, report=False)
    print(f"[{sample_id}] Keyframe extraction: {'done' if extracted or aligned else 'reused'}", flush=True)

    gpa_changed = False
    for label, video, directory, changed in (
        ("Reference GPA detection", reference_video, reference_dir / "gpa", False),
        ("Generated GPA detection", rendered_video, run_dir / "gpa", recorded),
    ):
        force = args.overwrite or changed or (
            gpa_ready(directory) and not fresh(directory / "summary.json", [video])
        )
        command = [python, "-c", GPA_COMMAND, str(video), str(directory),
                   str(args.gpa_model), args.device or "", str(args.batch_size),
                   "1" if force else "0"]
        gpa_changed |= run_stage(sample_id, label, lambda d=directory: gpa_ready(d), command, force, report=False)
    print(f"[{sample_id}] GPA GUI detection: {'done' if gpa_changed else 'reused'}", flush=True)

    score_inputs = [matches, keyframes, rendered_video, reference_video,
                    reference_dir / "gpa/summary.json", run_dir / "gpa/summary.json"]
    changed = args.overwrite or aligned or gpa_changed
    static_path = run_dir / "keyframe_alignment/static_appearance_scores.json"
    static = [python, str(PROJECT_ROOT / "utils/static_appearance_score.py"), *common]
    run_stage(sample_id, "Static appearance scoring",
              lambda: scores_ready(static_path, STATIC_KEYS, score_inputs), static, changed)

    animation_path = run_dir / "animation_segments/animation_scores.json"
    animation = [python, str(PROJECT_ROOT / "utils/animation_score.py"), *common,
                 "--dataset-root", str(args.data), "--model", str(args.gpa_model),
                 "--batch-size", str(args.batch_size), "--clip-batch-size", str(args.clip_batch_size)]
    if args.device is not None:
        animation += ["--device", args.device]
    if args.overwrite:
        animation += ["--refresh-segments"]
    if args.visualize:
        animation += ["--visualize"]
    run_stage(sample_id, "Animation scoring",
              lambda: scores_ready(animation_path, ANIMATION_KEYS, score_inputs)
              and (not args.visualize or comparison_videos_ready(animation_path)), animation, changed)


def write_model_csv(result_root: Path, model: str, excluded: set[str]) -> tuple[Path, int]:
    """Include all completed IDs for this model, including earlier invocations."""
    rows = []
    for directory in sorted(result_root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else -1):
        if not directory.is_dir() or not directory.name.isdigit() or directory.name in excluded:
            continue
        run_dir = directory / model
        try:
            values = metric_values(run_dir / "keyframe_alignment/static_appearance_scores.json", STATIC_KEYS)
            values += metric_values(run_dir / "animation_segments/animation_scores.json", ANIMATION_KEYS)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        rows.append([directory.name, *values])
    output = result_root / f"{model}.csv"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{model}.", suffix=".csv", dir=result_root)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["task_id", *CSV_METRICS])
            for sample_id, *values in rows:
                writer.writerow([sample_id, *(format(v, ".10f") for v in values)])
            means = [sum(row[column] for row in rows) / len(rows) for column in range(1, 9)] if rows else []
            writer.writerow(["mean", *(format(v, ".10f") for v in means)] if rows else ["mean", *([""] * 8)])
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output, len(rows)


def parse_ids(specifications: list[str]) -> list[str]:
    ids = set()
    for part in re.split(r"[,，\s]+", " ".join(specifications).strip()):
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not match:
            raise ValueError(f"Invalid task ID or range: {part}")
        start, end = int(match[1]), int(match[2] or match[1])
        if start < 1 or end < start:
            raise ValueError(f"Invalid task ID or range: {part}")
        ids.update(range(start, end + 1))
    return [str(i) for i in sorted(ids)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--result", type=Path, default=PROJECT_ROOT / "experiment")
    parser.add_argument("--model", required=True, help="Model name used for generation")
    parser.add_argument("--task-id", "--id", nargs="+", required=True, help="IDs or ranges, e.g. 1,3-10")
    parser.add_argument("--ground-truth-root", type=Path, help="Default: <result>/ground_truth")
    parser.add_argument("--gpa-model", type=Path, default=PROJECT_ROOT / "utils/GPA-GUI-Detector/model.pt")
    parser.add_argument("--device", help="GPA device, e.g. 0 or cpu; default: automatic")
    parser.add_argument("--batch-size", type=int, default=1, help="GPA batch size")
    parser.add_argument("--clip-batch-size", type=int, default=8)
    parser.add_argument("--node", default="node", help="Node.js executable")
    parser.add_argument("--overwrite", action="store_true", help="Rerun all stages for the selected IDs")
    parser.add_argument("--visualize", action="store_true", help="Also save animation trajectory comparison videos")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        sample_ids = parse_ids(args.task_id)
    except ValueError as exc:
        parser.error(str(exc))
    args.model = args.model.strip().lower().replace("/", "__")
    if not args.model or args.model in (".", "..") or "\\" in args.model or ":" in args.model:
        parser.error("--model must be a model name, not a filesystem path")
    if args.batch_size < 1 or args.clip_batch_size < 1:
        parser.error("Batch sizes must be positive")
    args.data = args.data.expanduser().resolve()
    args.result = args.result.expanduser().resolve()
    args.ground_truth_root = (args.ground_truth_root or args.result / "ground_truth").expanduser().resolve()
    args.gpa_model = args.gpa_model.expanduser().resolve()
    args.result.mkdir(parents=True, exist_ok=True)
    failed = set()
    for sample_id in sample_ids:
        try:
            evaluate_sample(args, sample_id)
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            failed.add(sample_id)
            print(f"[{sample_id}] Failed: {exc}", file=sys.stderr, flush=True)
        output, count = write_model_csv(args.result, args.model, failed)
    print(f"CSV: {output} ({count} completed IDs; {len(failed)} failed)", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
