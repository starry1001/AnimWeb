#!/usr/bin/env python3
import argparse
import os
import shutil
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return False


load_dotenv()

from utils.config import DEFAULT_MODEL_NAME

utils = None
MAX_MLLM_ATTEMPTS = 3


def get_utils():
    global utils
    if utils is None:
        from utils import core as imported_utils
        utils = imported_utils
    return utils


def build_tasks(data_path: str) -> tuple[str, list[dict]]:
    utils = get_utils()
    if os.path.isdir(data_path):
        task_dirs = [path for path in Path(data_path).iterdir() if path.is_dir() and path.name.isdigit()]
        task_dirs = sorted(task_dirs, key=lambda path: int(path.name))
        tasks = [{"id": task_dir.name} for task_dir in task_dirs]
        return data_path, tasks

    if os.path.isfile(data_path) and data_path.endswith(".jsonl"):
        tasks = utils.load_examples(data_path)
        base = os.path.splitext(os.path.basename(data_path))[0]
        data_root = os.path.join(os.path.dirname(data_path), base)
        if not os.path.isdir(data_root):
            raise FileNotFoundError(f"Tasks directory not found next to {data_path}: expected {data_root}")
        return data_root, tasks

    raise FileNotFoundError(f"Unsupported --data input: {data_path}")


def process_task(task_item, data_dir: str, result_dir: str, model_name: str, coverage: bool = False, prompt: str | None = None):
    utils = get_utils()
    task_id = str(task_item.get("id"))
    task_path = os.path.join(data_dir, task_id)
    print(f"Processing task: {task_path}")

    if not os.path.exists(task_path):
        raise FileNotFoundError(f"Task {task_id} directory is missing: {task_path}")

    video_path, stitched_images = utils.find_files(task_path)

    model_dir = model_name.lower().replace("/", "__")
    output_dir = os.path.join(result_dir, task_id, model_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "index.html")
    raw_output_file = os.path.join(output_dir, "response.txt")

    if coverage and os.path.exists(output_file):
        try:
            os.remove(output_file)
        except Exception:
            pass

    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        return task_id, "Already exists"

    temp_folder = os.path.join(output_dir, "temp_frames")
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)

    print(f"Calling model for task {task_id} (model={model_name})")
    
    last_error = None
    raw = None
    for attempt in range(1, MAX_MLLM_ATTEMPTS + 1):
        try:
            raw = utils.call_mllm_html(
                model_name=model_name,
                project_path=task_path,
                image_paths=stitched_images,
                video_path=video_path,
                temp_folder=temp_folder,
                prompt=prompt,
            )
            if raw and raw.strip():
                break
            last_error = RuntimeError("Model returned an empty response.")
        except Exception as exc:
            last_error = exc

        if attempt < MAX_MLLM_ATTEMPTS:
            print(
                f"Task {task_id}: MLLM request attempt {attempt}/{MAX_MLLM_ATTEMPTS} "
                f"failed ({last_error}); retrying."
            )
    else:
        raise RuntimeError(
            f"Task {task_id}: MLLM request failed after {MAX_MLLM_ATTEMPTS} attempts."
        ) from last_error

    assets_dir = os.path.join(data_dir, task_id, "assets")
    assets_path = os.path.relpath(assets_dir, output_dir).replace("\\", "/")
    if not raw or not raw.strip():
        raise RuntimeError(f"Task {task_id} generated an empty model response.")

    html = utils.post_process(raw, assets_path)

    with open(raw_output_file, "w", encoding="utf-8") as f:
        f.write(raw)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)

    # Treat a missing or empty result as a failed task.  This check is kept
    # after the write so an interrupted/failed write cannot look successful.
    if not os.path.isfile(output_file) or os.path.getsize(output_file) == 0:
        raise RuntimeError(f"Task {task_id} did not produce a non-empty index.html: {output_file}")

    return task_id, "Done"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Dataset directory or jsonl file")
    parser.add_argument("--result", required=True, help="Path to write generated outputs")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Model name")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--coverage", action="store_true", help="Force regeneration by deleting existing index.html before generating")
    parser.add_argument("--task-id", help="Process task IDs separated by commas, including numeric ranges (e.g. 7,9,14-19)")
    args = parser.parse_args()

    if not args.model.strip():
        parser.error("Set --model or OPENAI_MODEL_NAME.")

    if args.workers < 1:
        parser.error("--workers must be at least 1")

    data_root, tasks = build_tasks(args.data)
    if args.task_id is not None:
        selected_ids = set()
        selected_ranges = []
        for part in args.task_id.split(","):
            part = part.strip()
            start, separator, end = part.partition("-")
            if separator:
                if not start.isdigit() or not end.isdigit():
                    parser.error(f"Invalid --task-id range: {part!r}")
                start_id, end_id = int(start), int(end)
                if start_id > end_id:
                    parser.error("--task-id range start must be less than or equal to its end")
                selected_ranges.append((start_id, end_id))
            elif part:
                selected_ids.add(part)
            else:
                parser.error("--task-id cannot contain an empty ID")
        tasks = [
            task for task in tasks
            if (task_id := str(task.get("id"))) in selected_ids
            or (task_id.isdigit() and any(start <= int(task_id) <= end for start, end in selected_ranges))
        ]

    if not tasks:
        print("No tasks found")
        return

    print(f"Processing {len(tasks)} tasks with {args.workers} workers using model {args.model}")

    ordered = sorted(tasks, key=lambda x: int(x.get("id")) if str(x.get("id")).isdigit() else str(x.get("id")))
    task_iterator = iter(ordered)
    executor = ThreadPoolExecutor(max_workers=args.workers)
    pending = {}

    def submit_next_task() -> bool:
        """Submit one task, returning False once every task has been submitted."""
        try:
            task_item = next(task_iterator)
        except StopIteration:
            return False
        future = executor.submit(
            process_task, task_item, data_root, args.result, args.model, args.coverage, None
        )
        pending[future] = task_item
        return True

    for _ in range(min(args.workers, len(ordered))):
        submit_next_task()

    completed = 0
    while pending:
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        failure = None

        for future in done:
            task_item = pending.pop(future)
            task_id = str(task_item.get("id"))
            try:
                task, status = future.result()
            except Exception as exc:
                failure = (task_id, exc)
                break
            completed += 1
            print(f"[{completed}/{len(tasks)}] {task}: {status}")

        if failure is not None:
            failed_task, error = failure
            print(f"ERROR: task {failed_task} failed; stopping generation: {error}")
            for future in pending:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            return 1

        for _ in done:
            if not submit_next_task():
                break

    executor.shutdown(wait=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
