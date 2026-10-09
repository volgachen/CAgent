"""Definitions shared by the framework and application tools."""

from dataclasses import dataclass, field
from typing import Callable


class ToolError(Exception):
    """Expected business rejection, safe to return to the model."""


@dataclass(frozen=True)
class ToolResult:
    """Text for the model, plus message metadata kept only by the host."""

    content: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.content, str) or not isinstance(self.metadata, dict):
            raise TypeError("ToolResult requires string content and dictionary metadata")


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
