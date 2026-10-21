"""Definitions shared by the framework and application tools."""

from dataclasses import dataclass, field
from typing import Callable

from ..images import normalize_images


class ToolError(Exception):
    """Expected business rejection, safe to return to the model."""


@dataclass(frozen=True)
class ToolResult:
    """Text for the model, optional images, plus message metadata kept only by the host.

    images are references in the same forms as user input images (reference
    string, Image or {ref, detail}); they are validated here, stored in the
    tool message's metadata["images"] and resolved through
    BaseAgent.resolve_image before each model call. "images" is therefore a
    reserved metadata key.
    """

    content: str
    metadata: dict = field(default_factory=dict)
    images: list = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.content, str) or not isinstance(self.metadata, dict):
            raise TypeError("ToolResult requires string content and dictionary metadata")
        if "images" in self.metadata:
            raise ValueError('metadata["images"] is reserved; pass images=[...] instead')
        # Frozen dataclass: store the normalized form through object.__setattr__.
        object.__setattr__(self, "images", normalize_images(self.images))


@dataclass(frozen=True)
class Tool:
    """Handler receives model arguments and framework-injected store, ctx and agent_cls."""

    handler: Callable
    description: str = ""
    parameters: dict = field(default_factory=lambda: {"type": "object", "properties": {}})
    result_type: str = "extra"

    def __post_init__(self):
        if self.result_type not in ("signal", "memory", "extra"):
            raise ValueError("Tool.result_type must be signal, memory or extra")
