import base64
import json
import math
import mimetypes
import os
import re
from pathlib import Path
from typing import List, Tuple

import cv2
import httpx
from openai import OpenAI

from .config import API_CONFIG, DEFAULT_API_KEY, DEFAULT_BASE_URL, PROMPT_TEMPLATE


REQUEST_TIMEOUT = 600  # Seconds per model request.


def load_examples(jsonl_path: str) -> list[dict]:
    entries = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except Exception:
                continue
    return entries


def get_config(model_name: str) -> dict:
    lowered = model_name.lower()
    for key, value in API_CONFIG.items():
        if key != "default" and key in lowered:
            return value
    return API_CONFIG["default"]


def get_response_text(response) -> str:
    if not response.choices:
        return ""
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text") or "")
            else:
                parts.append(getattr(item, "text", "") or "")
        return "".join(parts)
    return content or ""


def describe_response(response) -> str:
    finish_reason = response.choices[0].finish_reason if response.choices else "no choices"
    usage = getattr(response, "usage", None)
    return f"finish_reason={finish_reason}, usage={usage}"


def require_response_text(response, model_name: str) -> str:
    text = get_response_text(response)
    if text:
        return text
    raise RuntimeError(f"Model {model_name} returned an empty response ({describe_response(response)}).")


def encode_file(file_path: str | Path) -> str:
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_media_type(file_path: str | Path) -> str:
    file_path = str(file_path)
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type:
        return mime_type
    return "application/octet-stream"


def to_data_url(file_path: str | Path) -> str:
    return f"data:{get_media_type(file_path)};base64,{encode_file(file_path)}"


def extract_frames(video_path: str | Path, output_folder: str | Path, config_dict: dict) -> List[str]:
    video_fps = config_dict.get("video_fps", 2)
    if not math.isfinite(video_fps) or video_fps <= 0:
        raise ValueError("video_fps must be a positive finite number.")
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or total_frames <= 0:
        cap.release()
        return []

    video_duration = total_frames / fps
    frame_interval = fps / video_fps
    # Sample at 0, 1/video_fps, ... while the timestamp is inside the video.
    target_frame_count = math.ceil(video_duration * video_fps)

    frame_paths = []
    for i in range(target_frame_count):
        frame_index = int(i * frame_interval)
        if frame_index >= total_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()
        if success:
            frame_path = os.path.join(output_folder, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)

    cap.release()
    return frame_paths


def find_files(project_path: str | Path) -> Tuple[str, List[str]]:
    project_path = Path(project_path)
    video_path = project_path / "ground_truth.mp4"
    if not video_path.is_file():
        raise FileNotFoundError(f"Reference video not found: {video_path}")

    stitched_dir = project_path / "stitched_assets"
    image_paths = sorted(path for path in stitched_dir.glob("stitched_image*.jpg") if path.is_file())
    if not image_paths:
        raise FileNotFoundError(f"No stitched_image*.jpg files found in {stitched_dir}")

    return str(video_path), [str(path) for path in image_paths]


def get_asset_names(project_path: str | Path) -> list[str]:
    assets_dir = Path(project_path) / "assets"
    if not assets_dir.exists():
        return []
    return sorted(path.name for path in assets_dir.iterdir() if path.is_file())


def build_prompt(project_path: str | Path, prompt: str | None = None) -> str:
    prompt_text = prompt or PROMPT_TEMPLATE

    asset_names = get_asset_names(project_path)
    if asset_names:
        asset_list = "\n".join(f"- {name}" for name in asset_names)
        prompt_text += f"\n\nAvailable asset filenames in `assets/`:\n{asset_list}\n"
    return prompt_text


def post_process(raw_response: str, assets_path: str) -> str:
    if not raw_response:
        return ""
    code = raw_response.strip()
    html_match = re.search(r"```html\s*\n?(.*?)\n?```", code, re.DOTALL | re.IGNORECASE)
    if html_match:
        code = html_match.group(1).strip()
    code = re.sub(r"<iframe[^>]*>.*?</iframe>", "", code, flags=re.DOTALL | re.IGNORECASE)
    code = re.sub(r"<iframe[^>]*/?>", "", code, flags=re.IGNORECASE)
    code = code.replace("```html", "")
    code = code.replace("```", "")
    code = code.replace("__PLACEHOLDER_ASSETS_BASE_DIR__", assets_path)
    return code


def call_mllm(
    model_name: str,
    content_list: List[dict],
    timeout: int = REQUEST_TIMEOUT,
    extra_params: dict | None = None,
    http_client=None,
) -> str:
    """Call a Chat Completions-compatible endpoint using its exact model ID."""
    api_key = os.getenv("OPENAI_API_KEY", DEFAULT_API_KEY)
    base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    if not base_url.strip():
        raise RuntimeError("OPENAI_BASE_URL is not set.")
    if not model_name.strip():
        raise ValueError("Model name is required.")

    if http_client is None:
        try:
            proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
            if proxy:
                http_client = httpx.Client(proxy=proxy, transport=httpx.HTTPTransport(local_address="0.0.0.0"))
        except Exception as e:
            print(f"Warning: failed to configure httpx proxy client: {e}")

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, http_client=http_client)

    api_params = {
        "model": model_name,
        "messages": [{"role": "user", "content": content_list}],
    }
    if extra_params:
        api_params.update(extra_params)

    print(f"Calling model {model_name}, content count: {len(content_list)}, base_url: {base_url}")
    response = client.chat.completions.create(**api_params)
    return require_response_text(response, model_name)


def build_multimodal_content(
    project_path: str | Path,
    image_paths: List[str],
    video_path: str,
    temp_folder: str | Path,
    config_dict: dict,
    prompt: str | None = None,
) -> list[dict]:
    prompt_text = build_prompt(project_path, prompt)
    content_list = [{"type": "text", "text": prompt_text}]

    for img_path in image_paths:
        content_list.append({"type": "image_url", "image_url": {"url": to_data_url(img_path)}})

    if config_dict.get("use_video_directly") and video_path and os.path.exists(video_path):
        content_list.append({"type": "video_url", "video_url": {"url": to_data_url(video_path)}})
        return content_list

    if video_path and os.path.exists(video_path):
        frame_paths = extract_frames(video_path, temp_folder, config_dict)
        for frame_path in frame_paths:
            content_list.append({"type": "image_url", "image_url": {"url": to_data_url(frame_path)}})
    return content_list


def call_mllm_html(
    model_name: str,
    project_path: str | Path,
    image_paths: List[str],
    video_path: str,
    temp_folder: str | Path,
    prompt: str | None = None,
) -> str:
    config_dict = get_config(model_name)
    content_list = build_multimodal_content(project_path, image_paths, video_path, temp_folder, config_dict, prompt)
    return call_mllm(
        model_name,
        content_list,
        extra_params=config_dict.get("extra_params"),
    )


__all__ = [
    "build_prompt",
    "call_mllm",
    "call_mllm_html",
    "encode_file",
    "extract_frames",
    "find_files",
    "get_asset_names",
    "get_config",
    "get_media_type",
    "load_examples",
    "post_process",
]
