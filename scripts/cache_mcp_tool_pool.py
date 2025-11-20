#!/usr/bin/env python
"""Cache tool metadata for all local MCP servers into data/mcp_tool_pool.json."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from datagen.local_runtime import get_local_registry  # pylint: disable=wrong-import-position

CACHE_PATH = REPO_ROOT / "data" / "mcp_tool_pool.json"


def serialize_tool(tool) -> dict:
    description = getattr(tool, "description", "") or ""
    input_schema = getattr(tool, "input_schema", None)
    if not isinstance(input_schema, dict):
        input_schema = {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": getattr(tool, "name", ""),
            "description": description,
            "parameters": input_schema,
        },
    }


async def fetch_server_tools(server_name: str) -> list[dict]:
    registry = get_local_registry()
    connection = registry.create_connection(server_name)
    try:
        async with connection as session:
            tools = await connection.initialize_and_list_tools()
            return [serialize_tool(tool) for tool in tools]
    except Exception as exc:  # pragma: no cover - best effort cache
        print(f"⚠️  Skipping server '{server_name}' due to error: {exc}")
        return []


async def build_pool() -> dict[str, list[dict]]:
    registry = get_local_registry()
    servers = registry.list_available_servers()
    pool: dict[str, list[dict]] = {}
    for server_name in sorted(servers):
        pool[server_name] = await fetch_server_tools(server_name)
    return pool


def main() -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pool = asyncio.run(build_pool())
    CACHE_PATH.write_text(json.dumps(pool, indent=2, ensure_ascii=False))
    print(f"Cached {len(pool)} servers to {CACHE_PATH}")


if __name__ == "__main__":
    main()
