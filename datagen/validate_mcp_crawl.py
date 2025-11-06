import argparse
import asyncio
import json
from pathlib import Path
from datetime import datetime

from mcp.client.streamable_http import streamablehttp_client
import mcp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = PROJECT_ROOT / "mcp_crawl"
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / "mcp_crawl_validation.log"


def write_log(lines):
    timestamp = datetime.utcnow().isoformat() + "Z"
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"=== Validation run: {timestamp} ===\n")
        for line in lines:
            f.write(line + "\n")
        f.write("\n")


async def verify(url: str):
    try:
        async with streamablehttp_client(url) as (read_stream, write_stream, _):
            async with mcp.ClientSession(read_stream, write_stream) as session:
                await asyncio.wait_for(session.initialize(), timeout=8)
                tools = await asyncio.wait_for(session.list_tools(), timeout=8)
                tool_list = getattr(tools, "tools", []) or []
                call_summary = "skipped"
                if tool_list:
                    first_tool = tool_list[0]
                    try:
                        result = await asyncio.wait_for(
                            session.call_tool(first_tool.name, {}), timeout=10
                        )
                        call_summary = f"success: {str(result)[:80]}"
                    except Exception as exc:  # noqa: BLE001
                        call_summary = f"error: {exc}"
                return True, len(tool_list), call_summary
    except Exception as exc:  # noqa: BLE001
        return False, 0, f"error: {exc}"


def main(directory: Path):
    if not directory.exists():
        print(f"Directory {directory} does not exist")
        return

    all_files = sorted(directory.glob("*.json"))
    log_lines = []
    kept = []
    removed = []

    async def validate_file(path: Path):
        data = json.loads(path.read_text())
        url = data.get("metadata", {}).get("remote_server_response", {}).get("url")
        if not url:
            return False, 0, "missing URL"
        return await verify(url)

    async def run_all():
        tasks = []
        for path in all_files:
            tasks.append((path, asyncio.create_task(validate_file(path))))
        results = []
        for path, task in tasks:
            results.append((path, await task))
        return results

    results = asyncio.run(run_all())

    for path, (ok, tool_count, call_status) in results:
        entry = f"{path.name}: tools={tool_count}, call={call_status}"
        if not ok or tool_count == 0:
            removed.append(entry)
            path.unlink()
        else:
            kept.append(entry)

    log_lines.append("Kept entries:")
    log_lines.extend(["  " + line for line in kept] or ["  (none)"])
    log_lines.append("Removed entries:")
    log_lines.extend(["  " + line for line in removed] or ["  (none)"])

    write_log(log_lines)

    print("Validation complete")
    print(f" Kept: {len(kept)}")
    print(f" Removed: {len(removed)}")
    print(f" Log written to {LOG_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate crawled MCP servers")
    parser.add_argument("--dir", type=str, default=str(DEFAULT_DIR), help="Directory with MCP crawl JSON files")
    args = parser.parse_args()
    main(Path(args.dir))
