import os

API_CONFIG = {
    "gpt-6-astra": {
        "video_fps": 2,
        "use_video_directly": False,
    },
    "gpt-6-sol": {
        "video_fps": 2,
        "use_video_directly": False,
    },
    "gpt-5.6-sol": {
        "video_fps": 2,
        "use_video_directly": False,
    },
    "claude-fable-5.1": {
        "video_fps": 2,
        "use_video_directly": False,
    },
    "claude-opus-5": {
        "video_fps": 2,
        "use_video_directly": False,
    },
    "grok-4.7": {
        "video_fps": 2,
        "use_video_directly": False,
    },
    "qwen3.8-max-0902": {
        "use_video_directly": True,
    },
    "gemini-3.1-pro-preview": {
        "use_video_directly": True,
    },
    "gemini-3.8-flash": {
        "use_video_directly": True,
    },
    "doubao-seed-2-1-pro": {
        "use_video_directly": True,
    },
    "kimi-k3": {
        "use_video_directly": True,
    },
    "deepseek-v4.1-flash": {
        "video_fps": 2,
        "use_video_directly": False,
    },
    "glm-5.3-flash": {
        "use_video_directly": True,
    },
    "qwen3-vl-235b-a22b-thinking": {
        "use_video_directly": True,
    },
    "default": {
        "video_fps": 2,
        "use_video_directly": False,
    },
}


DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
DEFAULT_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "")


PROMPT_TEMPLATE = """
You are an expert front-end developer. Your task is to create a pixel-perfect animated website from the provided reference materials.

You will receive:
1. A set of stitched asset overview images summarizing the available static resources.
2. One animation video showing the target webpage behavior.

Instructions:
1. Single File Output: Provide a single, complete HTML document containing all CSS inside <style> tags and all JavaScript inside <script> tags.
2. Animation Fidelity: Reproduce the layout, styling, motion, timing, and easing as closely as possible. Ensure high accuracy for animation durations, timing functions, keyframes, transforms (rotate, scale, translate), colors, gradients.
3. Asset Usage:
   - All images must use the provided stitched image assets.
   - Every `src` attribute must start with the literal string `__PLACEHOLDER_ASSETS_BASE_DIR__/`, followed by the exact filename identified from the stitched image.
   - All `<img>` tags must include `width` and `height` attributes.
4. No External Dependencies: The generated code must be entirely self-contained without using external libraries, CDNs, or external fonts.
5. Output Format: Return ONLY the complete HTML code wrapped in a single ```html code block, with no explanations.
"""
