"""
ComfyUI-Gemini-Prompt-Studio

Enhance your ComfyUI prompts with Google Gemini's dynamic "artist persona" system.
One node generates both T2I & I2V positive/negative conditioning in a single click.

使用 Google Gemini 的「动态艺术人格」系统增强 ComfyUI 提示词。
一键生成文生图 + 图生视频的正反提示词与条件。
"""

__version__ = "1.0.0"
__author__ = "DaLongZhuaZi"
__license__ = "MIT"

from .gemini_studio import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']