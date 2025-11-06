import torch
import os
import sys
import argparse
import copy
import json
import re
import requests
import concurrent.futures
import multiprocessing
import types
import asyncio
import base64
import threading
import queue
import signal
import atexit
from time import sleep, time
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
from tqdm import tqdm
from wrapt_timeout_decorator import timeout

from utils import load_dataset_from_file, save_dataset, make_api_request_with_retry, get_model_short_name, validate_api_pool_from_file, check_if_api_key_is_valid, safe_save_checkpoint, get_model_abbreviation

try:
    import mcp
    from mcp.client.streamable_http import streamablehttp_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# Global cleanup function for MCP resources
def cleanup_mcp_resources():
    """Clean up MCP resources on exit"""
    # Only cleanup if we're using agent mode
    try:
        # Check if args is available and agent mode is enabled
        if 'args' in globals() and hasattr(args, 'agent') and args.agent:
            # OpenAI Agent framework handles cleanup automatically
            pass
    except Exception as e:
        # print(f"⚠️ Warning: Emergency MCP cleanup failed: {e}")
        pass

# Signal handler for graceful shutdown
def signal_handler(signum, frame):
    """Handle termination signals gracefully"""
    # print(f"\n🛑 Received signal {signum}. Cleaning up...")
    cleanup_mcp_resources()
    # print("👋 Exiting gracefully.")
    os._exit(0)  # Use os._exit instead of sys.exit to avoid atexit conflicts

# Register cleanup functions
atexit.register(cleanup_mcp_resources)
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination

################
# Configurations
################
def get_args():
    # Experiment Settings
    parser = argparse.ArgumentParser(description="Response Generation Manager.")
    parser.add_argument("--model_path", type=str, default="openai/gpt-oss-120b",
                        help="Model path for inference")
    parser.add_argument("--input_file", type=str, default=None, help="Input dataset file name")
    parser.add_argument("--checkpoint_every", type=int, default=16, help="Save checkpoint every n completed items")
    parser.add_argument("--openrouter_url", type=str, default="https://openrouter.ai/api/v1", help="OpenRouter API URL")
    parser.add_argument("--openrouter_api_key", type=str, default="", help="OpenRouter API Key")
    parser.add_argument("--openai_api_key", type=str, default="", help="OpenAI API Key")
    parser.add_argument("--vllm_api_url", type=str, default="http://localhost:8000/v1", help="vLLM API URL")
    parser.add_argument("--vllm_api_key", type=str, default="EMPTY", help="vLLM API Key")
    parser.add_argument("--smithery_api_key", type=str, default="", help="Smithery API Key")
    parser.add_argument("--smithery_profile", type=str, default="", help="Smithery Profile")
    parser.add_argument("--smithery_api_pool", type=str, default="smithery_api_pool.json", help="Path to Smithery API pool JSON file")
    parser.add_argument("--max_workers", type=int, default=None, help="Maximum number of parallel workers (default: use API pool size)")
    parser.add_argument("--batch_size", type=int, default=None, help="Optional concurrent worker cap (alias for max_workers)")

    # Generation Parameters
    parser.add_argument('--engine', default="vllm_api", type=str, choices=["vllm_api", "openrouter_api", "openai"])
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--step", type=str, default="unknown", help="Processing step identifier.")
    parser.add_argument("--agent", type=str, default="openai_agent", help="Use agent inference for items with MCP server URLs")
    parser.add_argument("--timeout", type=int, default=90, help="Timeout in seconds for each item processing (default: 90 seconds)")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of retries for each item processing (default: 3)")
    parser.add_argument("--fncall_prompt_type", type=str, default="nous", help="Function call prompt type (default: nous)")
    parser.add_argument("--parallel_function_calls", type=bool, default=True, help="Parallel function calls (default: True)")
    parser.add_argument("--reasoning_effort", type=str, default="high", help="Reasoning effort (default: high)")
    parser.add_argument("--enable_tool_hint", action="store_true", help="Enable tool hint (default: off)")
    parser.add_argument("--enable_irrelevant_warning", action="store_true", help="Enable irrelevant warning (default: off)")
    parser.add_argument("--max_turns", type=int, default=10, help="Maximum number of dialogue turns.")
    parser.add_argument("--interaction_mode", type=str, default="single_turn",
                        choices=["single_turn", "multi_turn"],
                        help="Interaction strategy: single_turn stops after the first assistant response without tool calls; "
                             "multi_turn simulates user replies for clarification requests.")
    parser.add_argument("--limit_tools_to_targets", action="store_true",
                        help="Expose only the tools listed in target_tools to the agent.")
    return parser.parse_args()

args = get_args()
print(f"Response Generation Manager. Arguments: {args}") # For logging

def use_completion_token_param(model_name: str) -> bool:
    normalized = model_name.lower()
    return normalized.startswith("gpt-4.1") or normalized.startswith("gpt-5")

# Harmonize batch_size alias
if args.batch_size is not None and args.max_workers is None:
    args.max_workers = args.batch_size

if args.input_file is None:
    raise ValueError("Please specify the input file path.")
    
# Input check: check if ends with prepared.jsonl or prepared.json
if not args.input_file.endswith("prepared.jsonl") and not args.input_file.endswith("prepared.json"):
    print("Error: Input file must end with prepared.json(l) for completion pipeline. Please make sure you are using the correct input file.")
    exit(1)

# Resolve OpenAI API key when needed
if args.engine == "openai":
    resolved_openai_key = args.openai_api_key if args.openai_api_key else os.getenv("OPENAI_API_KEY")
    if not resolved_openai_key:
        raise ValueError("OpenAI API Key not provided. Please set OPENAI_API_KEY environment variable or provide --openai_api_key argument.")
    args.openai_api_key = resolved_openai_key

# Constants for the local vllm engine
MODEL_NAME = args.model_path
INPUT_FILE_NAME = args.input_file 
CHECKPOINT_EVERY = args.checkpoint_every

model_abbreviation = get_model_abbreviation(args.model_path)
config_str = f"{model_abbreviation}_{args.reasoning_effort}_pfc" if args.parallel_function_calls else f"{model_abbreviation}_{args.reasoning_effort}_sfc"

base_name = INPUT_FILE_NAME[:INPUT_FILE_NAME.rfind('.')]
if base_name.endswith("_4prepared"):
    base_name = base_name[:-10]  # Remove "_4prepared"

if args.num_trials > 1:
    checkpoint_files = [
        f"{base_name}_{config_str}_results{i}_checkpoint.json"
        for i in range(args.num_trials)
    ]
    saved_files = [
        f"{base_name}_{config_str}_results{i}.jsonl"
        for i in range(args.num_trials)
    ]
else:
    checkpoint_file = f"{base_name}_{config_str}_results_checkpoint.json"
    saved_file = f"{base_name}_{config_str}_results.jsonl"

# API Setups
if args.engine == "openrouter_api":
    API_ENDPOINT = args.openrouter_url + "/chat/completions"
    API_HEADERS = {
        "Authorization": f"Bearer {args.openrouter_api_key}",
        "Content-Type": "application/json"
    }
    API_PARAMS = {
        "model": args.model_path,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "parallel_tool_calls": args.parallel_function_calls,
        "reasoning": {"effort": args.reasoning_effort},
    }

elif args.engine == "vllm_api":
    API_ENDPOINT = args.vllm_api_url + "/chat/completions"
    API_HEADERS = {
        "Authorization": f"Bearer {args.vllm_api_key}",
        "Content-Type": "application/json"
    }
    API_PARAMS = {
        "model": args.model_path,
        # "max_tokens": args.max_tokens # If a user does not specify a max_tokens in their request, then the minimum of max_new_tokens and (max_model_len - prompt_tokens) will be used.
        "temperature": args.temperature,
        "top_p": args.top_p,
        "parallel_tool_calls": args.parallel_function_calls,
        "reasoning": {"effort": args.reasoning_effort},
    }

elif args.engine == "openai":
    API_ENDPOINT = "https://api.openai.com/v1/chat/completions"
    API_HEADERS = {
        "Authorization": f"Bearer {args.openai_api_key}",
        "Content-Type": "application/json"
    }
    API_PARAMS = {
        "model": args.model_path,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    if args.max_tokens:
        if use_completion_token_param(args.model_path):
            API_PARAMS["max_completion_tokens"] = args.max_tokens
        else:
            API_PARAMS["max_tokens"] = args.max_tokens

# Global API pool variable
smithery_api_pool = None

def load_and_validate_smithery_api_pool(pool_file_path):
    """Load and validate Smithery API pool from JSON file, keeping only valid keys"""
    global smithery_api_pool
    
    print("=" * 50)
    print("🔍 SMITHERY API POOL VALIDATION")
    print("=" * 50)
    
    # Check if pool file exists
    if not os.path.exists(pool_file_path):
        print(f"⚠️  API pool file {pool_file_path} not found!")
        print("🔍 Testing fallback API key from arguments...")
        
        # Validate the fallback API key
        fallback_result = check_if_api_key_is_valid(args.smithery_profile, args.smithery_api_key)
        
        if not fallback_result['valid']:
            raise ValueError(f"❌ Fallback API key is also invalid: {fallback_result['message']}")
        
        print(f"✅ Fallback API key is valid: {fallback_result['message']}")
        smithery_api_pool = [{
            "profile": args.smithery_profile,
            "api_key": args.smithery_api_key,
            "source": "fallback"
        }]
        print(f"✅ Using 1 valid API key (fallback)")
        print("=" * 50)
        return smithery_api_pool
    
    # Validate the entire API pool using the test logic
    print(f"📁 Validating all entries in {pool_file_path}...")
    
    try:
        results = validate_api_pool_from_file(pool_file_path)
        
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            raise ValueError(f"API pool validation failed: {results['error']}")
        
        # Display detailed results like in test file
        print("=" * 30)
        print("📊 VALIDATION SUMMARY")
        print("=" * 30)
        print(f"Total entries: {results['total_entries']}")
        print(f"Valid entries: {results['valid_entries']}")
        print(f"Invalid entries: {results['invalid_entries']}")
        print(f"Success rate: {results['valid_entries']/results['total_entries']*100:.1f}%")
        
        print(f"\n📋 DETAILED RESULTS")
        print("-" * 30)
        for result in results['results']:
            status = "✅" if result['valid'] else "❌"
            print(f"{status} {result['profile']} ({result['source']}): {result['message']}")
        
        # Check if we have any valid entries
        if results['valid_entries'] == 0:
            raise ValueError("❌ No valid API keys found in the pool! All API keys failed validation.")
        
        # Load original data to get valid entries with API keys
        with open(pool_file_path, 'r') as f:
            original_data = json.load(f)
            original_pool = original_data.get('api_pool', [])
        
        # Keep only valid entries
        valid_pool = []
        for result in results['results']:
            if result['valid']:
                # Find the original entry to get the API key
                for original_entry in original_pool:
                    if original_entry['profile'] == result['profile']:
                        valid_pool.append(original_entry)
                        break
        
        smithery_api_pool = valid_pool
        
        print(f"\n✅ SUCCESS: Using {len(smithery_api_pool)} valid API keys from pool")
        print("=" * 50)
        return smithery_api_pool
        
    except Exception as e:
        print(f"❌ Error during API pool validation: {e}")
        raise ValueError(f"API pool validation failed: {str(e)}")

def get_api_key_for_worker(worker_id):
    """Get API key and profile for a specific worker"""
    if smithery_api_pool and len(smithery_api_pool) > 0:
        # Round-robin assignment
        pool_entry = smithery_api_pool[worker_id % len(smithery_api_pool)]
        return pool_entry['api_key'], pool_entry['profile']
    else:
        return args.smithery_api_key, args.smithery_profile

def construct_mcp_server_url(server_info, api_key=None, profile=None, remote_response=None):
    """
    Construct MCP server URL from server info.
    """
    if not server_info:
        server_info = {}
    
    server_url = server_info.get('python_sdk_url', '')
    if not server_url and remote_response:
        server_url = remote_response.get('url', '')
    if not server_url:
        return None
    
    # Use provided api_key and profile, or fall back to args
    if api_key is None:
        api_key = args.smithery_api_key
    if profile is None:
        profile = args.smithery_profile
    
    # Get or create default config
    mcp_config = server_info.get('python_sdk_config', "")
    if not mcp_config:
        mcp_config = {"debug": False}
    elif isinstance(mcp_config, dict):
        mcp_config = mcp_config or {"debug": False}
    else:
        try:
            mcp_config = json.loads(mcp_config)
        except (TypeError, json.JSONDecodeError):
            mcp_config = {"debug": False}
    
    # Replace URL placeholders
    config_b64 = base64.b64encode(json.dumps(mcp_config).encode()).decode()
    if "{config_b64}" in server_url:
        server_url = server_url.replace("{config_b64}", config_b64)
    if "{smithery_api_key}" in server_url:
        server_url = server_url.replace("{smithery_api_key}", api_key)
    if "{smithery_profile}" in server_url:
        server_url = server_url.replace("{smithery_profile}", profile)
    
    parsed = urlparse(server_url)
    query_params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if api_key:
        query_params["api_key"] = api_key
    if profile:
        query_params["profile"] = profile
    rebuilt_query = urlencode(query_params, doseq=True)
    server_url = urlunparse(parsed._replace(query=rebuilt_query))
    
    return server_url

TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _sanitize_function_name(name: str) -> str:
    """
    Convert arbitrary tool identifiers into API-safe function names.
    """
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    if not sanitized:
        sanitized = "tool"
    if len(sanitized) > 64:
        sanitized = sanitized[:64]
    return sanitized.lower()


def format_tool_result(result) -> str:
    """
    Convert an MCP tool result into string content we can store in transcripts.
    """
    if result is None:
        return ""
    
    # Direct string result
    if isinstance(result, str):
        return result
    
    # If result behaves like a dict, JSON dump it
    if isinstance(result, dict):
        return json.dumps(result, ensure_ascii=False)
    
    segments = []
    content = getattr(result, "content", None)
    if content:
        for item in content:
            text = getattr(item, "text", None)
            if text:
                segments.append(text)
    structured = getattr(result, "structuredContent", None)
    if structured:
        try:
            segments.append(json.dumps(structured, ensure_ascii=False))
        except TypeError:
            segments.append(str(structured))
    if not segments:
        return str(result)
    return "\n".join(segments)


def parse_tool_call_blocks(text: str):
    """
    Parse <tool_call>...</tool_call> blocks emitted inline by the model.
    Returns a list of dictionaries with keys: name, arguments, call_id (optional).
    """
    if not text:
        return []
    
    calls = []
    for match in TOOL_CALL_PATTERN.finditer(text):
        block = match.group(1)
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            continue
        
        name = payload.get("name")
        arguments = payload.get("arguments", {})
        call_id = payload.get("id")
        
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                continue
        if not isinstance(arguments, dict):
            continue
        if not name:
            continue
        
        calls.append({
            "name": name,
            "arguments": arguments,
            "id": call_id
        })
    return calls


class MCPToolExecutor:
    """
    Manage MCP connections and provide metadata for tool prompting and invocation.
    """

    def __init__(self, mcp_servers, api_key, profile, timeout_seconds=90):
        self.mcp_servers = mcp_servers or []
        self.api_key = api_key
        self.profile = profile
        self.timeout_seconds = timeout_seconds
        self.connections = []
        self.tool_records = []
        self._api_name_lookup = {}
        self._original_name_lookup = {}
        self._tools_for_api = []

    async def __aenter__(self):
        if not MCP_AVAILABLE:
            raise RuntimeError("mcp package is required but not available.")
        await self._connect()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self._close()

    def get_tool_records(self):
        return self.tool_records

    def get_tools_for_api(self):
        return self._tools_for_api

    def resolve_tool(self, name):
        if name in self._api_name_lookup:
            return self._api_name_lookup[name]
        if name in self._original_name_lookup:
            return self._original_name_lookup[name]
        return None

    async def call_tool(self, api_name, arguments):
        record = self.resolve_tool(api_name)
        if not record:
            raise ValueError(f"Unknown tool name '{api_name}' requested.")
        
        session = record["session"]
        tool_name = record["base_name"]
        timeout = max(10, min(self.timeout_seconds, 180))
        return await asyncio.wait_for(session.call_tool(tool_name, arguments), timeout=timeout)

    async def _connect(self):
        if not self.mcp_servers:
            raise ValueError("No MCP servers provided in metadata.")
        
        # Count tool name occurrences to detect duplicates across servers
        name_counts = {}
        for server in self.mcp_servers:
            remote_tools = server.get("remote_server_response", {}).get("tools", [])
            for tool in remote_tools:
                tool_name = tool.get("name")
                if tool_name:
                    name_counts[tool_name] = name_counts.get(tool_name, 0) + 1

        used_api_names = set()

        for idx, server in enumerate(self.mcp_servers):
            server_info = server.get("server_info") or server.get("server_info_crawled") or {}
            remote_response = server.get("remote_server_response", {})
            server_name = server_info.get("name") or server.get("server_name") or f"server_{idx}"
            server_alias = re.sub(r"[^a-zA-Z0-9]+", "_", server_name).strip("_").lower() or f"server_{idx}"

            url = construct_mcp_server_url(server_info, self.api_key, self.profile, remote_response)
            if not url:
                raise ValueError(f"Unable to construct MCP URL for server '{server_name}'.")

            client_ctx = streamablehttp_client(url)
            read_stream, write_stream, _ = await client_ctx.__aenter__()
            session_ctx = mcp.ClientSession(read_stream, write_stream)
            session = await session_ctx.__aenter__()

            try:
                await asyncio.wait_for(session.initialize(), timeout=15)
                tools_result = await asyncio.wait_for(session.list_tools(), timeout=15)
            except Exception:
                await session_ctx.__aexit__(None, None, None)
                await client_ctx.__aexit__(None, None, None)
                raise

            available_tools = getattr(tools_result, "tools", []) or []
            remote_tool_lookup = {
                tool.get("name"): tool for tool in remote_response.get("tools", []) if isinstance(tool, dict)
            }

            for tool in available_tools:
                tool_name = getattr(tool, "name", None)
                if not tool_name:
                    continue

                base_identifier = tool_name
                if name_counts.get(tool_name, 0) > 1:
                    base_identifier = f"{server_alias}__{tool_name}"

                api_name = _sanitize_function_name(base_identifier)
                if api_name in used_api_names:
                    suffix = 2
                    while f"{api_name}_{suffix}" in used_api_names:
                        suffix += 1
                    api_name = f"{api_name}_{suffix}"
                used_api_names.add(api_name)

                remote_meta = remote_tool_lookup.get(tool_name, {})
                description = getattr(tool, "description", None) or remote_meta.get("description", "")
                input_schema = getattr(tool, "input_schema", None) or remote_meta.get("input_schema", None)
                if not isinstance(input_schema, dict):
                    input_schema = {"type": "object", "properties": {}}

                record = {
                    "api_name": api_name,
                    "original_name": tool_name,
                    "prompt_name": tool_name if base_identifier == tool_name else f"{server_name}::{tool_name}",
                    "base_name": tool_name,
                    "server_name": server_name,
                    "server_alias": server_alias,
                    "description": description.strip() if isinstance(description, str) else "",
                    "input_schema": input_schema,
                    "session": session,
                    "session_ctx": session_ctx,
                    "client_ctx": client_ctx,
                    "url": url,
                }

                self.tool_records.append(record)
                self._api_name_lookup[api_name] = record
                self._original_name_lookup[tool_name] = record
                if record["prompt_name"] not in (tool_name, api_name):
                    self._original_name_lookup[record["prompt_name"]] = record

                self._tools_for_api.append({
                    "type": "function",
                    "function": {
                        "name": api_name,
                        "description": record["description"] or f"{tool_name} from {server_name}",
                        "parameters": input_schema
                    }
                })

            self.connections.append({
                "session": session,
                "session_ctx": session_ctx,
                "client_ctx": client_ctx,
                "server_name": server_name
            })

    async def _close(self):
        while self.connections:
            conn = self.connections.pop()
            try:
                await conn["session_ctx"].__aexit__(None, None, None)
            except Exception as e:
                print(f"⚠️  Failed to close MCP session for {conn.get('server_name')}: {e}")
            try:
                await conn["client_ctx"].__aexit__(None, None, None)
            except Exception as e:
                print(f"⚠️  Failed to close MCP client for {conn.get('server_name')}: {e}")


def build_chat_payload(messages, tools_for_api=None):
    payload = copy.deepcopy(API_PARAMS)
    payload["messages"] = messages
    if tools_for_api:
        payload["tools"] = tools_for_api
        payload["tool_choice"] = "auto"
    return payload


def _post_chat_completion(payload):
    response = requests.post(API_ENDPOINT, json=payload, headers=API_HEADERS, timeout=120)
    if not response.ok:
        try:
            error_preview = response.text
        except Exception:
            error_preview = "<unable to read error body>"
        print(f"❌ Chat completion request failed ({response.status_code}): {error_preview}")
    response.raise_for_status()
    return response.json()


async def call_chat_completion(messages, tools_for_api=None, allow_tool_retry=True):
    """
    Wrapper around chat completion endpoint that optionally retries without the tools payload.
    """
    payload = build_chat_payload(messages, tools_for_api)
    try:
        return await asyncio.to_thread(_post_chat_completion, payload)
    except requests.RequestException as err:
        if tools_for_api and allow_tool_retry:
            print("⚠️  Chat completion request with tools failed, retrying without tools...")
            return await call_chat_completion(messages, None, allow_tool_retry=False)
        raise err

def convert_openai_agent_result_to_messages(result, original_messages, system_prompt=None):
    raise RuntimeError("convert_openai_agent_result_to_messages is deprecated in the direct tool pipeline.")

def create_agent_for_item(item, api_key=None, profile=None):
    return None


def qwen_compatible_system_prompt_generator(tool_records):
    """
    Build a system prompt that introduces the available tools and instructs the model
    on how to emit tool calls.
    """
    tool_sections = []
    for record in tool_records or []:
        schema_json = json.dumps(record.get("input_schema", {"type": "object", "properties": {}}), ensure_ascii=False)
        description = record.get("description") or "No description provided."
        prompt_name = record.get("prompt_name") or record.get("original_name") or record.get("api_name")
        original_name = record.get("original_name") or record.get("api_name")
        server_name = record.get("server_name") or "Unknown Server"
        section = [
            "<tool>",
            f"  <tool_name>{record.get('api_name')}</tool_name>",
            f"  <display_name>{prompt_name}</display_name>",
            f"  <original_name>{original_name}</original_name>",
            f"  <server>{server_name}</server>",
            f"  <description>{description}</description>",
            f"  <json_schema>{schema_json}</json_schema>",
            "</tool>",
        ]
        tool_sections.append("\n".join(section))

    tools_text = "\n".join(tool_sections) if tool_sections else "<tool><tool_name>none</tool_name></tool>"

    return (
        "You are an AI assistant who can call external tools to solve the user's request.\n"
        "You will not receive follow-up messages from the user, so do not ask clarifying questions; instead, make reasonable assumptions and proceed with the available information.\n"
        "Evaluate the available tools carefully and decide whether tool usage is necessary.\n"
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        f"<tools>\n{tools_text}\n</tools>\n\n"
        "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
        "</tool_call>\n\n"
        "Only use <tool_name> values when calling tools. The arguments must satisfy each tool's JSON schema.\n"
        "If a tool call is not needed, respond directly to the user in natural language.\n"
        "Do not include <tool_call> tags in your final answer once you are done with tool usage.\n"
        "When you have gathered enough information, stop calling tools and provide a single natural-language summary that answers the user.\n"
    )
SIMULATED_USER_SYSTEM_PROMPT = (
    "You are the same human user who issued the original task. "
    "First judge whether the assistant's latest response already contains a final answer and no further user input is required. "
    "If it is final, reply EXACTLY with 'REJECTION' (all caps, no explanation). "
    "Otherwise, craft the user’s next message using information from the original request and any previous user instructions you may have given. "
    "If you cannot provide additional information, explicitly say you have none but still encourage the assistant to continue with whatever it already has."
)

DEFAULT_FALLBACK_USER_RESPONSE = (
    "I don't have additional information beyond my original request. Please proceed with the details I already provided."
)

REJECTION_PREFIX = "REJECTION"


async def simulate_user_reply(
    original_request: str,
    assistant_followup: str,
    conversation_history: list,
) -> tuple[bool, str]:
    original_request = (original_request or "").strip()
    assistant_followup = (assistant_followup or "").strip()
    if not assistant_followup:
        return False, ""

    history_lines = []
    for idx, msg in enumerate(conversation_history):
        role = msg.get("role", "assistant")
        name = msg.get("name")
        content = msg.get("content", "")
        label = role
        if role in ("function", "tool") and name:
            label = f"{role}:{name}"
        history_lines.append(f"[{idx}] {label} => {content}")

    history_block = "\n".join(history_lines)

    prompt_body = (
        "=== Conversation so far ===\n"
        f"{history_block}\n\n"
        "=== Latest assistant response ===\n"
        f"{assistant_followup}\n\n"
        "Reply as the user:"
    )
    messages = [
        {"role": "system", "content": SIMULATED_USER_SYSTEM_PROMPT},
        {"role": "user", "content": prompt_body},
    ]
    try:
        response_json = await call_chat_completion(messages, tools_for_api=None, allow_tool_retry=False)
        choice = (response_json.get("choices") or [{}])[0]
        reply = ((choice.get("message") or {}).get("content") or "").strip()
        normalized = reply.strip().upper()
        if normalized == REJECTION_PREFIX:
            return True, ""
        return False, reply
    except Exception as exc:
        print(f"⚠️  Simulated user reply failed: {exc}")
        return False, ""

# Process a single item using agent inference
async def process_single_item_agent_async(item, api_key=None, profile=None):
    """Process a single item using direct chat completions with manual MCP tool calls."""
    prompt_id = item.get('metadata', {}).get('prompt_id', 'unknown')
    metadata = item.get('metadata', {})
    mcp_servers = metadata.get('mcp_servers', [])

    if not mcp_servers:
        raise ValueError("No MCP server metadata available for this item.")

    messages = copy.deepcopy(item.get("messages", []))
    if not messages:
        raise ValueError("Item does not contain any messages.")

    user_indices = [idx for idx, msg in enumerate(messages) if msg.get("role") == "user"]
    if not user_indices:
        raise ValueError("No user messages found in conversation.")
    last_user_idx = user_indices[-1]

    target_tools_raw = item.get("target_tools") or metadata.get("target_tools") or ""
    target_tools_list = [tool.strip() for tool in target_tools_raw.split(',') if tool.strip()]
    if target_tools_list:
        formatted_targets = ", ".join(target_tools_list)
        print(f"🔍 Target tools for item {prompt_id}: {formatted_targets}")
    else:
        formatted_targets = ""

    if args.enable_tool_hint and formatted_targets:
        hint = f"\n\nWe need to use the following tools: {formatted_targets}."
        messages[last_user_idx]["content"] += hint
    if args.enable_irrelevant_warning:
        warning = "\n\nUse the available tools only if they are relevant. Otherwise, reply directly."
        messages[last_user_idx]["content"] += warning

    print(f"🚀 Running direct tool-enabled inference for item {prompt_id}...")
    print(f"📝 User message passed to model:\n{messages[last_user_idx]['content']}")

    target_specs = []
    target_entries_lower = set()
    for raw_entry in target_tools_list:
        entry = raw_entry.strip()
        if not entry:
            continue
        target_entries_lower.add(entry.lower())
        if "::" in entry:
            server_part, tool_part = entry.split("::", 1)
            server_part = server_part.strip()
            tool_part = tool_part.strip()
            if tool_part:
                target_specs.append((server_part.lower(), tool_part.lower()))
        else:
            target_specs.append((None, entry.lower()))

    if args.limit_tools_to_targets and not target_specs:
        print("⚠️  limit_tools_to_targets requested, but no target_tools provided; using full tool list.")

    async with MCPToolExecutor(mcp_servers, api_key, profile, args.timeout) as executor:
        tool_records = executor.get_tool_records()
        if not tool_records:
            raise ValueError("Connected MCP servers expose no callable tools.")

        tools_for_api = executor.get_tools_for_api()
        filtered_records = tool_records
        filtered_tools_for_api = tools_for_api

        if args.limit_tools_to_targets and target_specs:
            allowed_api_names = set()
            matched_records = []

            for record in tool_records:
                record_tool = (record.get("original_name") or "").strip()
                record_server = (record.get("server_name") or "").strip()
                record_prompt_name = (record.get("prompt_name") or "").strip()

                record_tool_lower = record_tool.lower()
                record_server_lower = record_server.lower()
                record_prompt_lower = record_prompt_name.lower()

                matches_target = False

                combined_lower = ""
                if record_server_lower and record_tool_lower:
                    combined_lower = f"{record_server_lower}::{record_tool_lower}"

                if record_prompt_lower and record_prompt_lower in target_entries_lower:
                    matches_target = True
                elif combined_lower and combined_lower in target_entries_lower:
                    matches_target = True
                elif record_tool_lower in target_entries_lower:
                    matches_target = True
                else:
                    for server_target, tool_target in target_specs:
                        if record_tool_lower != tool_target:
                            continue
                        if server_target is None:
                            matches_target = True
                            break
                        if server_target == record_server_lower:
                            matches_target = True
                            break
                        if "::" in record_prompt_lower:
                            prompt_server = record_prompt_lower.split("::", 1)[0]
                            if server_target == prompt_server:
                                matches_target = True
                                break

                if matches_target:
                    matched_records.append(record)
                    allowed_api_names.add(record.get("api_name"))

            if matched_records:
                filtered_records = matched_records
                filtered_tools_for_api = [
                    tool for tool in tools_for_api
                    if tool.get("function", {}).get("name") in allowed_api_names
                ]
                if not filtered_tools_for_api:
                    print("⚠️  No API tools matched target_tools after filtering; falling back to full tool list.")
                    filtered_records = tool_records
                    filtered_tools_for_api = tools_for_api
            else:
                print("⚠️  No tools matched target_tools; falling back to full tool list.")

        system_prompt = qwen_compatible_system_prompt_generator(filtered_records)
        print(f"🧭 System prompt for item {prompt_id}:\n{system_prompt}")

        tool_names_for_log = ", ".join(sorted({record["prompt_name"] for record in filtered_records}))
        print(f"🔧 Available tools ({len(filtered_records)}): {tool_names_for_log}")

        conversation_for_model = [{"role": "system", "content": system_prompt}]
        transcript_messages = [{"role": "system", "content": system_prompt}]

        original_user_segments = []
        for msg in messages:
            cloned = copy.deepcopy(msg)
            conversation_for_model.append(cloned)
            transcript_messages.append(cloned)
            if cloned.get("role") == "user":
                original_user_segments.append(cloned.get("content", ""))

        original_user_request = "\n\n".join(segment for segment in original_user_segments if segment).strip()
        simulated_followups = 0
        fallback_used = False
        turn_usage = 0
        turn_idx = 0

        while turn_usage < args.max_turns:
            response_json = await call_chat_completion(conversation_for_model, filtered_tools_for_api)
            choice = (response_json.get("choices") or [{}])[0]
            response_message = choice.get("message", {})
            finish_reason = choice.get("finish_reason")

            raw_message_str = json.dumps(response_message, ensure_ascii=False)
            if len(raw_message_str) > 2000:
                raw_message_str = raw_message_str[:2000] + "... [truncated]"
            print(f"🤖 Model raw response (turn {turn_idx}): {raw_message_str}")

            assistant_model_msg = {
                "role": "assistant",
                "content": response_message.get("content") or ""
            }
            parsed_calls = []

            tool_calls_from_model = response_message.get("tool_calls") or []
            if tool_calls_from_model:
                normalized_calls = []
                for call_idx, call in enumerate(tool_calls_from_model):
                    func = call.get("function", {})
                    api_name = func.get("name")
                    arguments_payload = func.get("arguments", "{}")
                    if isinstance(arguments_payload, dict):
                        arguments = arguments_payload
                    else:
                        try:
                            arguments = json.loads(arguments_payload)
                        except json.JSONDecodeError as exc:
                            raise ValueError(f"Invalid JSON arguments for tool '{api_name}': {exc}") from exc
                    call_id = call.get("id") or f"call_{turn_idx}_{call_idx}"
                    normalized_calls.append({
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": api_name,
                            "arguments": json.dumps(arguments, ensure_ascii=False)
                        }
                    })
                    parsed_calls.append({"api_name": api_name, "arguments": arguments, "id": call_id})
                assistant_model_msg["tool_calls"] = normalized_calls
            else:
                inline_calls = parse_tool_call_blocks(response_message.get("content", ""))
                if inline_calls:
                    normalized_calls = []
                    for call_idx, call in enumerate(inline_calls):
                        api_name = call["name"]
                        arguments = call["arguments"]
                        call_id = call.get("id") or f"inline_{turn_idx}_{call_idx}"
                        normalized_calls.append({
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": api_name,
                                "arguments": json.dumps(arguments, ensure_ascii=False)
                            }
                        })
                        parsed_calls.append({"api_name": api_name, "arguments": arguments, "id": call_id})
                    assistant_model_msg["tool_calls"] = normalized_calls

            assistant_output_msg = {
                "role": "assistant",
                "content": assistant_model_msg.get("content", "")
            }

            turn_usage += 1
            turn_idx += 1

            if parsed_calls:
                output_calls = []
                for call in parsed_calls:
                    record = executor.resolve_tool(call["api_name"])
                    if not record:
                        raise ValueError(f"Model requested unknown tool '{call['api_name']}'.")
                    output_calls.append({
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": record["original_name"],
                            "arguments": json.dumps(call["arguments"], ensure_ascii=False)
                        }
                    })
                assistant_output_msg["tool_calls"] = output_calls

            conversation_for_model.append(assistant_model_msg)
            transcript_messages.append(assistant_output_msg)

            if not parsed_calls:
                assistant_text = assistant_output_msg["content"]
                if args.interaction_mode == "multi_turn":
                    is_rejection, simulated_reply = await simulate_user_reply(
                        original_user_request,
                        assistant_text,
                        transcript_messages,
                    )
                    if is_rejection:
                        print("🛑 Simulated user marked assistant response as final (REJECTION). Ending trajectory.")
                        break
                    multi_turn_handled = False
                    if simulated_reply:
                        if turn_usage >= args.max_turns:
                            print("ℹ️  Turn budget exhausted before injecting simulated user reply.")
                            break
                        simulated_followups += 1
                        sim_msg = {"role": "user", "content": simulated_reply}
                        conversation_for_model.append(sim_msg)
                        transcript_messages.append(sim_msg)
                        turn_usage += 1
                        print(f"🗣️  Injected simulated user reply (count {simulated_followups}): {simulated_reply}")
                        multi_turn_handled = True
                        if turn_usage >= args.max_turns:
                            print("ℹ️  Turn budget reached immediately after simulated user reply.")
                            break
                    else:
                        print("⚠️  Simulated user reply was empty, attempting fallback.")

                    if multi_turn_handled:
                        continue

                    if not fallback_used:
                        if turn_usage >= args.max_turns:
                            print("ℹ️  Turn budget exhausted before adding fallback reminder.")
                            break
                        fallback_used = True
                        fallback_msg = {"role": "user", "content": DEFAULT_FALLBACK_USER_RESPONSE}
                        conversation_for_model.append(fallback_msg)
                        transcript_messages.append(fallback_msg)
                        turn_usage += 1
                        print("ℹ️  Added fallback user reminder to proceed without extra info.")
                        if turn_usage >= args.max_turns:
                            print("ℹ️  Turn budget reached immediately after fallback reminder.")
                            break
                        continue
                    else:
                        print("⚠️  Unable to provide additional user replies; ending interaction.")

                if not assistant_text.strip():
                    print(f"⚠️  Assistant returned no content and no tool calls for item {prompt_id}.")
                else:
                    print(f"✅ Completed tool interaction loop for item {prompt_id}.")
                break

            for call in parsed_calls:
                record = executor.resolve_tool(call["api_name"])
                if not record:
                    raise ValueError(f"Unable to resolve tool '{call['api_name']}'.")
                print(f"🛠️  Executing tool '{record['original_name']}' (API name: {record['api_name']}) with arguments {json.dumps(call['arguments'], ensure_ascii=False)}")
                tool_result = await executor.call_tool(record["api_name"], call["arguments"])
                formatted_output = format_tool_result(tool_result)
                if len(formatted_output) > 2000:
                    log_output = formatted_output[:2000] + "... [truncated]"
                else:
                    log_output = formatted_output
                print(f"📨 Tool response:\n{log_output}")

                tool_message_model = {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": record["api_name"],
                    "content": formatted_output
                }
                tool_message_output = {
                    "role": "function",
                    "name": record["original_name"],
                    "content": formatted_output,
                    "tool_call_id": call["id"]
                }
                conversation_for_model.append(tool_message_model)
                transcript_messages.append(tool_message_output)
        else:
            print(f"⚠️  Reached max turns ({args.max_turns}) without final assistant answer for item {prompt_id}.")

    item["messages"] = transcript_messages
    metadata_ref = item.setdefault("metadata", {})
    metadata_ref["simulated_user_turns"] = simulated_followups
    return item

@timeout(args.timeout, use_signals=False)
def process_single_item_agent(item, api_key=None, profile=None):
    """Process a single item using agent inference with timeout"""
    prompt_id = item.get('metadata', {}).get('prompt_id', 'unknown')
    
    try:
        return asyncio.run(process_single_item_agent_async(item, api_key, profile))
    except Exception as e:
        print(f"Error processing item {prompt_id}: {str(e)}")
        raise


# Dynamic processing with timeout resilience
class DynamicProcessor:
    """
    Dynamic processor that handles individual items with timeout resilience.
    Each item is processed independently so timeouts don't block other items.
    """
    
    def __init__(self, max_workers=None, checkpoint_every=16):
        self.max_workers = max_workers or len(smithery_api_pool) if smithery_api_pool else 1
        self.checkpoint_every = checkpoint_every
        self.processed_count = 0
        self.lock = threading.Lock()
        self.completed_items_list = []  # Thread-safe list for completed items
        
    def process_single_item_with_fallback(self, item_data):
        """Process a single item; if the agent fails, surface the error without fallback."""
        item, item_index, api_key, profile = item_data
        prompt_id = item.get('metadata', {}).get('prompt_id', f'item_{item_index}')
        
        # Try agent processing first if available
        agent_failed = False
        agent_error = None
        
        if args.agent:
            try:
                processed_item = process_single_item_agent(item, api_key, profile)
                return processed_item, item_index, True, None  # success, no error
            except Exception as e:
                print(f"⚠️ Agent processing failed for item {prompt_id}: {str(e)}")
                agent_failed = True
                agent_error = str(e)
        else:
            print(f"ℹ️ No agent specified for item {prompt_id}, using direct API...")
            agent_failed = True
            agent_error = "No agent specified"
            
        # No fallback: propagate failure directly
        if agent_failed:
            return item, item_index, False, f"Agent failed: {agent_error}"
                
    def process_items_dynamically(self, items_to_process, processed_dataset, checkpoint_file, progress_bar):
        """
        Process items dynamically with individual timeouts and immediate checkpointing.
        Only saves completed items to checkpoint for proper resume functionality.
        """
        completed_items = {}
        
        # Prepare items with metadata for processing
        items_with_metadata = []
        for i, (item, original_index) in enumerate(items_to_process):
            api_key, profile = get_api_key_for_worker(i)
            items_with_metadata.append((item, original_index, api_key, profile))
        
        # Process items with ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all items
            future_to_data = {}
            for item_data in items_with_metadata:
                future = executor.submit(self.process_single_item_with_fallback, item_data)
                future_to_data[future] = item_data
            
            # Process completions as they arrive
            for future in concurrent.futures.as_completed(future_to_data):
                try:
                    processed_item, original_index, success, error_msg = future.result()
                    completed_items[original_index] = processed_item
                    
                    # Update the main dataset immediately
                    processed_dataset[original_index] = processed_item
                    
                    # Update progress and handle checkpoint saving atomically
                    with self.lock:
                        # Add to completed items list for checkpoint (thread-safe)
                        self.completed_items_list.append(processed_item)
                        
                        self.processed_count += 1
                        progress_bar.update(1)
                        
                        # Log completion status
                        prompt_id = processed_item.get('metadata', {}).get('prompt_id', f'item_{original_index}')
                        status = "✅" if success else "❌"
                        if error_msg:
                            print(f"{status} Completed item {prompt_id} (index {original_index}) - {error_msg}")
                        else:
                            print(f"{status} Completed item {prompt_id} (index {original_index})")
                        
                        # Save checkpoint periodically - ONLY completed items
                        if self.processed_count % self.checkpoint_every == 0:
                            self._save_checkpoint_safely(checkpoint_file)
                
                except Exception as e:
                    item_data = future_to_data[future]
                    original_item, original_index, _, _ = item_data
                    prompt_id = original_item.get('metadata', {}).get('prompt_id', f'item_{original_index}')
                    print(f"❌ Unexpected error processing item {prompt_id}: {str(e)}")
                    
                    # Create error item
                    message = original_item["messages"]
                    original_item['messages'] = message + [
                        {
                            "role": "assistant",
                            "content": f"[UNEXPECTED_ERROR: {str(e)}]"
                        }
                    ]
                    processed_dataset[original_index] = original_item
                    
                    with self.lock:
                        self.completed_items_list.append(original_item)
                        self.processed_count += 1
                        progress_bar.update(1)
        
        # Final checkpoint save for any remaining completed items
        with self.lock:
            if self.completed_items_list:
                self._save_checkpoint_safely(checkpoint_file, is_final=True)
        
        return len(completed_items)
    
    def _save_checkpoint_safely(self, checkpoint_file, is_final=False):
        """
        Thread-safe checkpoint saving method.
        Must be called within self.lock context.
        """
        try:
            # Load existing checkpoint and append new completions
            existing_completed = []
            if os.path.exists(checkpoint_file):
                try:
                    existing_completed = load_dataset_from_file(checkpoint_file)
                    if not isinstance(existing_completed, list):
                        existing_completed = [existing_completed]
                except Exception as e:
                    print(f"⚠️ Warning: Could not load existing checkpoint: {e}")
                    existing_completed = []
            
            # Create combined list and sort by row_id
            all_completed = existing_completed + self.completed_items_list
            all_completed_sorted = sort_dataset_by_row_id(all_completed)
            
            # Save checkpoint safely
            safe_save_checkpoint(all_completed_sorted, checkpoint_file, convert_to_jsonl=False)
            
            checkpoint_type = "Final" if is_final else "Periodic"
            print(f"💾 {checkpoint_type} checkpoint saved: {len(all_completed_sorted)} completed items total (sorted by row_id)")
            
            # Clear the completed items list since they're now saved
            self.completed_items_list = []
            
        except Exception as e:
            print(f"❌ Error saving checkpoint: {e}")
            # Don't clear the list if save failed - we'll try again next time

# Function to sort dataset by row_id from metadata
def sort_dataset_by_row_id(dataset):
    """Sort dataset by row_id from metadata, handling missing row_ids gracefully"""
    def get_sort_key(item):
        metadata = item.get('metadata', {})
        row_id = metadata.get('row_id')
        if row_id is not None:
            try:
                return int(row_id)
            except (ValueError, TypeError):
                # If row_id can't be converted to int, use as string
                return float('inf'), str(row_id)
        else:
            # Items without row_id go to the end
            return float('inf'), ''
    
    return sorted(dataset, key=get_sort_key)

# Function to add generation config to metadata
def add_generation_config_to_metadata(dataset, model_short_name, generation_params):
    """Add synthetic data generation config to each item's metadata"""
    config_entry = {
        "model": model_short_name,
        "generation_params": generation_params,
        "timestamp": int(time())
    }
    
    for item in dataset:
        if "metadata" not in item:
            item["metadata"] = {}
        
        if "synthetic_data_gen_configs" not in item["metadata"]:
            item["metadata"]["synthetic_data_gen_configs"] = []
        
        item["metadata"]["synthetic_data_gen_configs"].append(config_entry)
    
    return dataset

# Generate outputs using dynamic processing with timeout resilience
def generate_and_update(dataset, checkpoint_file):
    processed_dataset = copy.deepcopy(dataset)

    # Prepare generation parameters for metadata
    generation_params = {
        "engine": args.engine,
        "model_path": args.model_path,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_p": args.top_p,
        "num_trials": args.num_trials,
        "step": args.step,
        "agent": args.agent,
        "timeout": args.timeout,
        "max_workers": args.max_workers
    }

    # Determine which items need processing by comparing IDs/metadata
    items_to_process = []
    completed_item_ids = set()
    completed_count = 0
    
    if os.path.exists(checkpoint_file):
        try:
            checkpoint_data = load_dataset_from_file(checkpoint_file)
            if not isinstance(checkpoint_data, list):
                checkpoint_data = [checkpoint_data]
            
            print(f"Checkpoint file found with {len(checkpoint_data)} completed items.")
            
            # Extract completed item IDs from checkpoint
            for completed_item in checkpoint_data:
                # Use prompt_id from metadata if available, otherwise use a hash of the input
                metadata = completed_item.get('metadata', {})
                prompt_id = metadata.get('prompt_id')
                
                if prompt_id:
                    completed_item_ids.add(prompt_id)
                else:
                    # Fallback: use hash of the user message for identification
                    messages = completed_item.get('messages', [])
                    if messages:
                        user_msg = next((msg['content'] for msg in messages if msg.get('role') == 'user'), '')
                        if user_msg:
                            completed_item_ids.add(hash(user_msg))
            
            completed_count = len(checkpoint_data)
            
            # Update processed_dataset with completed items for those positions we can identify
            # This maintains compatibility with the old approach while being more robust
            checkpoint_index = 0
            for i, item in enumerate(processed_dataset):
                metadata = item.get('metadata', {})
                prompt_id = metadata.get('prompt_id')
                
                # Check if this item is completed
                is_completed = False
                if prompt_id and prompt_id in completed_item_ids:
                    is_completed = True
                else:
                    # Fallback check using message hash
                    messages = item.get('messages', [])
                    if messages:
                        user_msg = next((msg['content'] for msg in messages if msg.get('role') == 'user'), '')
                        if user_msg and hash(user_msg) in completed_item_ids:
                            is_completed = True
                
                if is_completed and checkpoint_index < len(checkpoint_data):
                    # Replace with completed version from checkpoint
                    processed_dataset[i] = checkpoint_data[checkpoint_index]
                    checkpoint_index += 1
                else:
                    # This item needs processing
                    items_to_process.append((item, i))
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting fresh...")
            completed_count = 0
            # Process all items if checkpoint is corrupted
            for i in range(len(processed_dataset)):
                items_to_process.append((processed_dataset[i], i))
    else:
        print("No checkpoint found. Processing all items.")
        # Process all items
        for i in range(len(processed_dataset)):
            items_to_process.append((processed_dataset[i], i))
    
    print(f"Total items in dataset: {len(processed_dataset)}")
    print(f"Already completed: {completed_count}")
    print(f"Remaining to process: {len(items_to_process)}")
    
    if len(items_to_process) == 0:
        print("All items already processed!")
        return processed_dataset

    # Create dynamic processor
    max_workers = args.max_workers or (len(smithery_api_pool) if smithery_api_pool else 8)
    processor = DynamicProcessor(
        max_workers=max_workers, 
        checkpoint_every=CHECKPOINT_EVERY
    )
    
    print(f"🚀 Starting dynamic processing with {max_workers} workers...")
    print(f"💾 Checkpoints will be saved every {CHECKPOINT_EVERY} completed items")
    print(f"⏱️ Individual item timeout: {args.timeout} seconds")
    
    # Create progress bar for remaining items
    with tqdm(total=len(items_to_process), 
              desc="Processing items", 
              unit="item",
              initial=0,
              leave=True, 
              dynamic_ncols=True,
              bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as progress_bar:
        
        start_time = time()
        
        # Process items dynamically (will use agent if available, otherwise direct API)
        completed_count = processor.process_items_dynamically(
            items_to_process, 
            processed_dataset, 
            checkpoint_file, 
            progress_bar
        )
        
        end_time = time()
        
        print(f"\n🎉 Dynamic processing completed!")
        print(f"📊 Items processed: {completed_count}/{len(items_to_process)}")
        print(f"⏱️ Total time: {end_time - start_time:.2f} seconds")
        print(f"⚡ Average time per item: {(end_time - start_time)/max(completed_count, 1):.2f} seconds")

    # Add generation config to metadata and sort by row_id before returning
    processed_dataset = add_generation_config_to_metadata(processed_dataset, model_abbreviation, generation_params)
    processed_dataset_sorted = sort_dataset_by_row_id(processed_dataset)
    
    return processed_dataset_sorted

# Main function to control workflow
def main():
    # Load and validate Smithery API pool
    api_pool = load_and_validate_smithery_api_pool(args.smithery_api_pool)
    
    # Display dynamic processing info
    effective_workers = args.max_workers or len(api_pool)
    print("=" * 50)
    print("🚀 DYNAMIC PROCESSING CONFIGURATION")
    print("=" * 50)
    print(f"Processing mode: Dynamic (individual item processing)")
    print(f"Workers: {effective_workers}")
    print(f"API pool size: {len(api_pool)}")
    print(f"Timeout per item: {args.timeout} seconds")
    print(f"Checkpoint frequency: Every {args.checkpoint_every} completed items")
    
    if args.max_workers is not None:
        print(f"Worker setting: Custom ({args.max_workers} workers)")
    else:
        print(f"Worker setting: Auto-detected from API pool size")
    
    print(f"Resilience: Individual timeouts prevent blocking")
    if args.agent:
        print(f"Processing: Agent mode with direct API fallback")
    else:
        print(f"Processing: Direct API mode (no agent)")
    print(f"Checkpoint format: Only completed items (compatible with old format)")
    print(f"Sorting: All outputs sorted by row_id from metadata")
    print("=" * 50)
    
    try:
        # Load instructions from the input file
        dataset = load_dataset_from_file(INPUT_FILE_NAME)
        
        # Ensure dataset is always a list (fix for single-item JSON files)
        if not isinstance(dataset, list):
            dataset = [dataset]

        if args.num_trials == 1:
            updated_dataset = generate_and_update(dataset, checkpoint_file)
            save_dataset(updated_dataset, saved_file, convert_to_jsonl=True)

            # Optionally remove the checkpoint file after completion
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
            print("Final dataset saved. Checkpoint removed.")
        else:
            for i in range(args.num_trials):
                updated_dataset = generate_and_update(dataset, checkpoint_files[i])
                save_dataset(updated_dataset, saved_files[i], convert_to_jsonl=True)

                # Optionally remove the checkpoint file after completion
                if os.path.exists(checkpoint_files[i]):
                    os.remove(checkpoint_files[i])
                print(f"Dataset for trial {i} saved. Checkpoint {i} removed.")
    
    finally:
        # Clean up MCP resources to ensure proper program exit
        if args.agent:
            try:
                print("🧹 Cleaning up MCP resources...")
                # OpenAI Agent framework handles cleanup automatically via context managers
                print("✅ MCP cleanup completed.")
            except Exception as e:
                print(f"⚠️ Warning: MCP cleanup failed: {e}")
        
        print("🎯 Program execution completed.")
        os._exit(0)  # Use os._exit to avoid atexit conflicts with multiprocessing


# Run the main function
if __name__ == "__main__":
    main()
