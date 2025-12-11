"""
BlueNexus Tool Operations

Async helper functions for tool operations using BlueNexus storage.
"""

import asyncio
import logging
from typing import Optional

from open_webui.utils.bluenexus.collections import Collections
from open_webui.utils.bluenexus.factory import get_bluenexus_client_for_user
from open_webui.utils.bluenexus.types import QueryOptions

log = logging.getLogger(__name__)


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
    """Get a tool by its ID."""
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
            return record.model_dump()
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
    """Update a tool by its ID."""
    client = get_bluenexus_client_for_user(user_id)
    if not client:
        log.warning(f"No BlueNexus client for user {user_id}")
        return None

    try:
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
        return updated_record.model_dump()
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
