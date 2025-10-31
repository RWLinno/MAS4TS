"""
MCP Log Tools

Provide filesystem-based log retrieval and pattern scans to aid operations.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from oncall_agent.mcp.tools.base import MCPTool, ToolMetadata, ToolParameter


class LogTools(MCPTool):
    def _get_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="logs",
            description="Search and summarize logs from filesystem paths",
            parameters=[
                ToolParameter(name="action", type="string", description="log action", required=True, enum=[
                    "tail", "grep", "summary"
                ]),
                ToolParameter(name="path", type="string", description="log file path", required=True),
                ToolParameter(name="pattern", type="string", description="grep pattern (for action=grep)", required=False),
                ToolParameter(name="lines", type="integer", description="lines to tail (for action=tail)", required=False),
            ],
        )

    async def execute(self, params: Dict[str, Any]) -> Any:
        params = self.validate_params(params)
        action = params.get("action")
        path = params.get("path")
        if not os.path.exists(path):
            raise ValueError(f"Log path not found: {path}")

        if action == "tail":
            return self._tail(path, int(params.get("lines") or 200))
        if action == "grep":
            pattern = params.get("pattern")
            if not pattern:
                raise ValueError("'pattern' is required for grep action")
            return self._grep(path, pattern)
        if action == "summary":
            return self._summary(path)

        raise ValueError(f"Unknown action: {action}")

    def _tail(self, path: str, lines: int) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            data = f.readlines()
        return {"path": path, "lines": lines, "content": "".join(data[-lines:])}

    def _grep(self, path: str, pattern: str) -> Dict[str, Any]:
        regex = re.compile(pattern)
        matches: List[str] = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if regex.search(line):
                    matches.append(line.rstrip("\n"))
        return {"path": path, "pattern": pattern, "count": len(matches), "matches": matches[:500]}

    def _summary(self, path: str) -> Dict[str, Any]:
        # Simple heuristic summary: count levels and error patterns
        levels = {"ERROR": 0, "WARN": 0, "INFO": 0, "DEBUG": 0}
        total = 0
        first_ts = None
        last_ts = None
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                total += 1
                for level in levels:
                    if level in line:
                        levels[level] += 1
                # naive timestamp capture
                if first_ts is None and ("T" in line or ":" in line):
                    first_ts = line[:32]
                last_ts = line[:32]
        return {"path": path, "total": total, "levels": levels, "first_ts": first_ts, "last_ts": last_ts}


