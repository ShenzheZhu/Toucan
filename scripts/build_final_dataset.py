#!/usr/bin/env python
"""Build final dataset jsonl (uuid, question, target_tools, tools, messages) from filtered results."""
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from datagen.local_runtime import get_local_registry  # pylint: disable=wrong-import-position


def load_tool_pool(pool_path: Path) -> dict[str, list[dict]]:
    if not pool_path.exists():
        raise FileNotFoundError(f"Tool pool not found: {pool_path}")
    return json.loads(pool_path.read_text())


def get_server_name(metadata: dict) -> str:
    for server in metadata.get("mcp_servers", []):
        connection = server.get("remote_server_response", {}).get("connection", {})
        candidate = connection.get("server") or server.get("server_name")
        if candidate:
            return candidate
    # Fallback: try server_info name
    return metadata.get("mcp_servers", [{}])[0].get("server_name", "unknown")


def sanitize_messages(messages: list[dict]) -> list[dict]:
    return [msg for msg in messages if msg.get("role") != "system"]


def build_records(src_path: Path, job_id: str, pool: dict[str, list[dict]]):
    for line in src_path.open():
        data = json.loads(line)
        metadata = data.get("metadata", {})
        prompt_id = metadata.get("prompt_id", str(uuid.uuid4()))
        row_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{job_id}-{prompt_id}"))
        question = data.get("question") or data.get("messages", [{}])[0].get("content", "")
        target_tools = data.get("target_tools", "")
        server_name = get_server_name(metadata)
        tools = pool.get(server_name)
        if tools is None:
            raise KeyError(f"Server '{server_name}' not present in tool pool.")
        clean_messages = sanitize_messages(data.get("messages", []))
        yield {
            "uuid": row_uuid,
            "question": question,
            "target_tools": target_tools,
            "tools": json.dumps(tools, ensure_ascii=False),
            "messages": json.dumps(clean_messages, ensure_ascii=False),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final dataset jsonl from filtered results.")
    parser.add_argument("--job_id", required=True, help="Job ID (used for deterministic uuid generation)")
    parser.add_argument("--input", required=True, help="Path to filtered results JSONL")
    parser.add_argument("--output", required=True, help="Path to write final dataset JSONL")
    parser.add_argument("--tool_pool", default=str(REPO_ROOT / "data" / "mcp_tool_pool.json"), help="Path to cached tool pool JSON")
    args = parser.parse_args()

    src_path = Path(args.input)
    out_path = Path(args.output)
    pool = load_tool_pool(Path(args.tool_pool))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w") as dst:
        for record in build_records(src_path, args.job_id, pool):
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    print(f"Wrote {count} rows to {out_path}")


if __name__ == "__main__":
    main()
