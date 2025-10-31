"""
MCP Tools Base Definitions

Provides lightweight base classes and schemas for MCP tools.
This module is intentionally dependency-light and used by tests and server.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolParameter:
    name: str
    type: str
    description: str
    required: bool = False
    enum: Optional[List[str]] = None


@dataclass
class ToolMetadata:
    name: str
    description: str
    version: str = "1.0.0"
    parameters: List[ToolParameter] = field(default_factory=list)


class MCPTool:
    """
    Minimal MCP Tool base.

    Tools should implement:
      - _get_metadata() -> ToolMetadata
      - get_parameters() -> List[ToolParameter]
      - execute(params: Dict[str, Any]) -> Any (async)
    """

    def _get_metadata(self) -> ToolMetadata:
        raise NotImplementedError

    def get_parameters(self) -> List[ToolParameter]:
        return []

    def get_schema(self) -> Dict[str, Any]:
        meta = self._get_metadata()
        return {
            "name": meta.name,
            "description": meta.description,
            "parameters": {
                "type": "object",
                "required": [p.name for p in meta.parameters if p.required],
                "properties": {
                    p.name: {
                        "type": p.type,
                        "description": p.description,
                        **({"enum": p.enum} if p.enum else {}),
                    }
                    for p in meta.parameters
                },
            },
        }

    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        meta = self._get_metadata()
        required = {p.name for p in meta.parameters if p.required}
        missing = [name for name in required if name not in params or params[name] in (None, "")]
        if missing:
            raise ValueError(f"Missing required parameters: {', '.join(missing)}")
        return params


