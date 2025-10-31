"""
MCP Database Tools

Provide database access (PostgreSQL, MongoDB, Redis) via DatabaseManager.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from oncall_agent.database.connection_manager import DatabaseManager
from oncall_agent.mcp.tools.base import MCPTool, ToolMetadata, ToolParameter


def _load_config() -> Dict[str, Any]:
    # Try to load config.json from project root if present
    candidates = [
        os.getenv("ONCALL_CONFIG"),
        os.path.join(os.getcwd(), "config.json"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return {"database": {}}


class DatabaseTools(MCPTool):
    def __init__(self) -> None:
        self._db_manager: Optional[DatabaseManager] = None

    def _ensure_manager(self) -> DatabaseManager:
        if self._db_manager is None:
            config = _load_config()
            self._db_manager = DatabaseManager(config)
        return self._db_manager

    def _get_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="database",
            description="Execute SQL, MongoDB queries, and Redis cache ops",
            parameters=[
                ToolParameter(name="action", type="string", description="database action", required=True, enum=[
                    "sql", "mongo_find", "redis_get", "redis_set", "health"
                ]),
                ToolParameter(name="query", type="string", description="SQL query string (for action=sql)", required=False),
                ToolParameter(name="params", type="object", description="SQL params object (for action=sql)", required=False),
                ToolParameter(name="collection", type="string", description="MongoDB collection name (for action=mongo_find)", required=False),
                ToolParameter(name="filter", type="object", description="MongoDB filter object (for action=mongo_find)", required=False),
                ToolParameter(name="limit", type="integer", description="Result limit (for action=mongo_find)", required=False),
                ToolParameter(name="key", type="string", description="Redis key (for redis_get/set)", required=False),
                ToolParameter(name="value", type="string", description="Redis value (for redis_set)", required=False),
                ToolParameter(name="ttl", type="integer", description="TTL seconds (for redis_set)", required=False),
            ],
        )

    async def execute(self, params: Dict[str, Any]) -> Any:
        params = self.validate_params(params)
        action = params.get("action")
        db = self._ensure_manager()

        if action == "health":
            return db.health_check()

        if action == "sql":
            query = params.get("query")
            if not query:
                raise ValueError("'query' is required for sql action")
            rows = db.execute_sql(query, params=params.get("params"))
            result: List[Dict[str, Any]] = []
            for row in rows:
                try:
                    # SQLAlchemy Row exposes _mapping for dict-like access
                    result.append(dict(row._mapping))  # type: ignore[attr-defined]
                except Exception:
                    try:
                        result.append(dict(row))  # type: ignore[arg-type]
                    except Exception:
                        result.append({"values": list(row)})
            return {"rows": result, "count": len(result)}

        if action == "mongo_find":
            collection = params.get("collection")
            if not collection:
                raise ValueError("'collection' is required for mongo_find")
            mongo_filter = params.get("filter") or {}
            limit = int(params.get("limit") or 10)
            docs = db.find_documents(collection, mongo_filter, limit)
            # Convert ObjectId to string if present
            def _normalize(doc: Dict[str, Any]) -> Dict[str, Any]:
                out = {}
                for k, v in doc.items():
                    try:
                        from bson import ObjectId  # type: ignore
                        if isinstance(v, ObjectId):
                            out[k] = str(v)
                        else:
                            out[k] = v
                    except Exception:
                        out[k] = v
                return out
            return {"documents": [_normalize(d) for d in docs], "count": len(docs)}

        if action == "redis_get":
            key = params.get("key")
            if not key:
                raise ValueError("'key' is required for redis_get")
            return {"key": key, "value": db.cache_get(key)}

        if action == "redis_set":
            key = params.get("key")
            value = params.get("value")
            if not key or value is None:
                raise ValueError("'key' and 'value' are required for redis_set")
            ttl = int(params.get("ttl") or 3600)
            ok = db.cache_set(key, value, ttl)
            return {"success": bool(ok), "key": key, "ttl": ttl}

        raise ValueError(f"Unknown action: {action}")


