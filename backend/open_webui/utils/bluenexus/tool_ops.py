"""
BlueNexus Tool Operations

Async helper functions for tool operations using BlueNexus storage.
"""

import asyncio
import logging
import time
from typing import Optional

from open_webui.utils.bluenexus.collections import Collections
from open_webui.utils.bluenexus.factory import get_bluenexus_client_for_user
from open_webui.utils.bluenexus.types import QueryOptions
from open_webui.utils.bluenexus.cache import (
    get_cached_record_id,
    set_cached_record_id,
)

log = logging.getLogger(__name__)

# Simple in-memory cache for tool data (longer TTL since tools change less frequently)
_tool_cache: dict[str, tuple[dict, str, float]] = {}  # key -> (data, record_id, timestamp)
_TOOL_CACHE_TTL = 60  # 60 seconds


def _cache_tool(user_id: str, tool_id: str, data: dict, record_id: str) -> None:
    """Cache tool data with record ID."""
    key = f"{user_id}:tools:{tool_id}"
    _tool_cache[key] = (data, record_id, time.time())
    set_cached_record_id(user_id, "tools", tool_id, record_id)


def _get_cached_tool(user_id: str, tool_id: str) -> tuple[Optional[dict], Optional[str]]:
    """Get cached tool data and record ID."""
    key = f"{user_id}:tools:{tool_id}"
    if key in _tool_cache:
        data, record_id, ts = _tool_cache[key]
        if time.time() - ts < _TOOL_CACHE_TTL:
            return data, record_id
        else:
            del _tool_cache[key]
    # Try to get just record ID from persistent cache
    record_id = get_cached_record_id(user_id, "tools", tool_id)
    return None, record_id


def _invalidate_tool_cache(user_id: str, tool_id: str) -> None:
    """Invalidate tool cache."""
    key = f"{user_id}:tools:{tool_id}"
    if key in _tool_cache:
        del _tool_cache[key]


def _run_async(coro):
    """Run an async coroutine from sync code."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


async def get_tool_by_id(user_id: str, tool_id: str) -> Optional[dict]:
    """Get a tool by its ID. Uses cache when available."""
    # Check cache first
    cached_data, record_id = _get_cached_tool(user_id, tool_id)
    if cached_data:
        return cached_data

    client = get_bluenexus_client_for_user(user_id)
    if not client:
        log.warning(f"No BlueNexus client for user {user_id}")
        return None

    try:
        response = await client.query(
            Collections.TOOLS,
            QueryOptions(filter={"owui_id": tool_id}, limit=1)
        )

        if response.data and len(response.data) > 0:
            record = response.get_records()[0]
            result = record.model_dump()
            # Cache for future operations
            _cache_tool(user_id, tool_id, result, record.id)
            return result
        return None
    except Exception as e:
        log.error(f"Error getting tool {tool_id}: {e}")
        return None


async def get_tool_valves_by_id(user_id: str, tool_id: str) -> Optional[dict]:
    """Get tool valves by tool ID."""
    tool = await get_tool_by_id(user_id, tool_id)
    if tool:
        return tool.get("valves")
    return None


async def get_user_valves_by_id_and_user_id(
    user_id: str, tool_id: str, target_user_id: str
) -> dict:
    """Get user-specific valves for a tool."""
    client = get_bluenexus_client_for_user(target_user_id)
    if not client:
        log.warning(f"No BlueNexus client for user {target_user_id}")
        return {}

    try:
        # User valves could be stored separately or within the tool
        tool = await get_tool_by_id(user_id, tool_id)
        if tool:
            user_valves = tool.get("user_valves", {})
            return user_valves.get(target_user_id, {})
        return {}
    except Exception as e:
        log.error(f"Error getting user valves for tool {tool_id}: {e}")
        return {}


async def update_tool_by_id(user_id: str, tool_id: str, update_data: dict) -> Optional[dict]:
    """Update a tool by its ID. Optimized with record ID caching."""
    client = get_bluenexus_client_for_user(user_id)
    if not client:
        log.warning(f"No BlueNexus client for user {user_id}")
        return None

    try:
        # Try to get cached data and record ID
        cached_data, record_id = _get_cached_tool(user_id, tool_id)

        if record_id and cached_data:
            # FAST PATH: Use cached record ID - only 1 API call
            log.debug(f"[BlueNexus] Fast update for tool {tool_id} using cached record_id")
            tool_data = cached_data.copy()
            for key, value in update_data.items():
                tool_data[key] = value

            updated_record = await client.update(Collections.TOOLS, record_id, tool_data)
            result = updated_record.model_dump()
            _cache_tool(user_id, tool_id, result, record_id)
            return result

        # SLOW PATH: Need to query for record ID first - 2 API calls
        response = await client.query(
            Collections.TOOLS,
            QueryOptions(filter={"owui_id": tool_id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            return None

        record = response.get_records()[0]
        tool_data = record.model_dump()

        # Update fields
        for key, value in update_data.items():
            tool_data[key] = value

        updated_record = await client.update(Collections.TOOLS, record.id, tool_data)
        result = updated_record.model_dump()
        # Cache for future updates
        _cache_tool(user_id, tool_id, result, record.id)
        return result
    except Exception as e:
        log.error(f"Error updating tool {tool_id}: {e}")
        return None


async def get_tools(user_id: str) -> list[dict]:
    """Get all tools for a user."""
    client = get_bluenexus_client_for_user(user_id)
    if not client:
        log.warning(f"No BlueNexus client for user {user_id}")
        return []

    try:
        response = await client.query(
            Collections.TOOLS,
            QueryOptions(limit=1000)  # Get all tools
        )

        if response.data:
            return [r.model_dump() for r in response.get_records()]
        return []
    except Exception as e:
        log.error(f"Error getting tools for user {user_id}: {e}")
        return []


# Sync wrappers
def get_tool_by_id_sync(user_id: str, tool_id: str) -> Optional[dict]:
    """Get a tool by ID (sync wrapper)."""
    return _run_async(get_tool_by_id(user_id, tool_id))


def get_tool_valves_by_id_sync(user_id: str, tool_id: str) -> Optional[dict]:
    """Get tool valves (sync wrapper)."""
    return _run_async(get_tool_valves_by_id(user_id, tool_id))


def get_user_valves_by_id_and_user_id_sync(
    user_id: str, tool_id: str, target_user_id: str
) -> dict:
    """Get user valves (sync wrapper)."""
    return _run_async(get_user_valves_by_id_and_user_id(user_id, tool_id, target_user_id))


def update_tool_by_id_sync(user_id: str, tool_id: str, update_data: dict) -> Optional[dict]:
    """Update a tool (sync wrapper)."""
    return _run_async(update_tool_by_id(user_id, tool_id, update_data))


def get_tools_sync(user_id: str) -> list[dict]:
    """Get all tools for a user (sync wrapper)."""
    return _run_async(get_tools(user_id))
