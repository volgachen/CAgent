"""Definitions shared by the framework and application tools."""

from dataclasses import dataclass, field
from typing import Callable

from ..images import normalize_images, split_interleaved


class ToolError(Exception):
    """Expected business rejection, safe to return to the model."""


@dataclass(frozen=True)
class ToolResult:
    """Text for the model, optional images, plus message metadata kept only by the host.

    content is either a string or an interleaved list of text strings and
    Image objects, e.g. ["Page 1:", Image("file:1"), "Page 2:", Image("file:2")],
    in which case the model sees text and images in that order. After
    construction content is always the concatenated text and each inline image
    in images carries "at", its character offset in content.

    images are references in the same forms as user input images (reference
    string, Image or {ref, detail}) and follow all text; they cannot be
    combined with interleaved content. They are validated here, stored in the
    tool message's metadata["images"] and resolved through
    BaseAgent.resolve_image before each model call. "images" is therefore a
    reserved metadata key.
    """

    content: str | list
    metadata: dict = field(default_factory=dict)
    images: list = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.content, (str, list, tuple)) or not isinstance(self.metadata, dict):
            raise TypeError("ToolResult requires string or interleaved list content and dictionary metadata")
        if "images" in self.metadata:
            raise ValueError('metadata["images"] is reserved; pass images=[...] instead')
        # Frozen dataclass: store the normalized form through object.__setattr__.
        if isinstance(self.content, (list, tuple)):
            if self.images:
                raise ValueError("Interleaved content carries its images inline; do not also pass images=")
            content, images = split_interleaved(self.content)
            object.__setattr__(self, "content", content)
            object.__setattr__(self, "images", images)
        else:
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
