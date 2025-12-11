"""
BlueNexus Message Operations

Async helper functions for channel message operations using BlueNexus storage.
These replace the sync Messages model methods for channel-based communication.

Note: Channel messages are shared resources. This implementation stores messages
with the user who created them but enables cross-user queries for channels.
"""

import asyncio
import logging
import time
import uuid
from typing import Optional

from open_webui.utils.bluenexus.collections import Collections
from open_webui.utils.bluenexus.factory import get_bluenexus_client_for_user
from open_webui.utils.bluenexus.types import QueryOptions
from open_webui.models.users import Users, UserNameResponse
from open_webui.models.messages import (
    MessageModel,
    MessageForm,
    MessageResponse,
    MessageUserResponse,
    MessageReplyToResponse,
    Reactions,
)

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


async def insert_new_message(
    user_id: str, form_data: MessageForm, channel_id: str
) -> Optional[dict]:
    """Insert a new message into a channel."""
    client = get_bluenexus_client_for_user(user_id)
    if not client:
        log.warning(f"No BlueNexus client for user {user_id}")
        return None

    try:
        msg_id = str(uuid.uuid4())
        ts = int(time.time_ns())

        message_data = {
            "owui_id": msg_id,
            "id": msg_id,
            "user_id": user_id,
            "channel_id": channel_id,
            "reply_to_id": form_data.reply_to_id,
            "parent_id": form_data.parent_id,
            "content": form_data.content,
            "data": form_data.data,
            "meta": form_data.meta,
            "reactions": [],  # Store reactions inline
            "created_at": ts,
            "updated_at": ts,
        }

        record = await client.create(Collections.MESSAGES, message_data)
        return record.model_dump()
    except Exception as e:
        log.error(f"Error inserting message: {e}")
        return None


async def get_message_by_id(user_id: str, message_id: str) -> Optional[dict]:
    """Get a message by its ID."""
    client = get_bluenexus_client_for_user(user_id)
    if not client:
        log.warning(f"No BlueNexus client for user {user_id}")
        return None

    try:
        response = await client.query(
            Collections.MESSAGES,
            QueryOptions(filter={"owui_id": message_id}, limit=1)
        )

        if response.data and len(response.data) > 0:
            record = response.get_records()[0]
            return record.model_dump()
        return None
    except Exception as e:
        log.error(f"Error getting message {message_id}: {e}")
        return None


async def get_messages_by_channel_id(
    user_id: str, channel_id: str, skip: int = 0, limit: int = 50
) -> list[dict]:
    """Get messages for a channel (top-level messages only)."""
    client = get_bluenexus_client_for_user(user_id)
    if not client:
        log.warning(f"No BlueNexus client for user {user_id}")
        return []

    try:
        response = await client.query(
            Collections.MESSAGES,
            QueryOptions(
                filter={"channel_id": channel_id, "parent_id": None},
                sort=[{"field": "created_at", "order": "desc"}],
                skip=skip,
                limit=limit
            )
        )

        if response.data:
            return [r.model_dump() for r in response.get_records()]
        return []
    except Exception as e:
        log.error(f"Error getting messages for channel {channel_id}: {e}")
        return []


async def get_messages_by_parent_id(
    user_id: str, channel_id: str, parent_id: str, skip: int = 0, limit: int = 50
) -> list[dict]:
    """Get thread replies for a message."""
    client = get_bluenexus_client_for_user(user_id)
    if not client:
        log.warning(f"No BlueNexus client for user {user_id}")
        return []

    try:
        # Get the parent message first
        parent_response = await client.query(
            Collections.MESSAGES,
            QueryOptions(filter={"owui_id": parent_id}, limit=1)
        )

        parent = None
        if parent_response.data and len(parent_response.data) > 0:
            parent = parent_response.get_records()[0].model_dump()

        if not parent:
            return []

        # Get replies
        response = await client.query(
            Collections.MESSAGES,
            QueryOptions(
                filter={"channel_id": channel_id, "parent_id": parent_id},
                sort=[{"field": "created_at", "order": "desc"}],
                skip=skip,
                limit=limit
            )
        )

        messages = []
        if response.data:
            messages = [r.model_dump() for r in response.get_records()]

        # Add parent message if we have room
        if len(messages) < limit:
            messages.append(parent)

        return messages
    except Exception as e:
        log.error(f"Error getting thread messages for {parent_id}: {e}")
        return []


async def get_thread_replies_by_message_id(user_id: str, message_id: str) -> list[dict]:
    """Get all thread replies for a message."""
    client = get_bluenexus_client_for_user(user_id)
    if not client:
        log.warning(f"No BlueNexus client for user {user_id}")
        return []

    try:
        response = await client.query(
            Collections.MESSAGES,
            QueryOptions(
                filter={"parent_id": message_id},
                sort=[{"field": "created_at", "order": "desc"}]
            )
        )

        if response.data:
            return [r.model_dump() for r in response.get_records()]
        return []
    except Exception as e:
        log.error(f"Error getting thread replies for {message_id}: {e}")
        return []


async def get_reactions_by_message_id(user_id: str, message_id: str) -> list[dict]:
    """Get reactions for a message."""
    message = await get_message_by_id(user_id, message_id)
    if not message:
        return []

    # Reactions are stored inline in the message
    raw_reactions = message.get("reactions", [])

    # Group reactions by name
    reactions_map = {}
    for reaction in raw_reactions:
        name = reaction.get("name")
        if name not in reactions_map:
            reactions_map[name] = {"name": name, "user_ids": [], "count": 0}
        reactions_map[name]["user_ids"].append(reaction.get("user_id"))
        reactions_map[name]["count"] += 1

    return list(reactions_map.values())


async def update_message_by_id(
    user_id: str, message_id: str, form_data: MessageForm
) -> Optional[dict]:
    """Update a message."""
    client = get_bluenexus_client_for_user(user_id)
    if not client:
        log.warning(f"No BlueNexus client for user {user_id}")
        return None

    try:
        # Find the message
        response = await client.query(
            Collections.MESSAGES,
            QueryOptions(filter={"owui_id": message_id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            return None

        record = response.get_records()[0]
        message_data = record.model_dump()

        # Update fields
        message_data["content"] = form_data.content
        message_data["data"] = {
            **(message_data.get("data") or {}),
            **(form_data.data or {}),
        }
        message_data["meta"] = {
            **(message_data.get("meta") or {}),
            **(form_data.meta or {}),
        }
        message_data["updated_at"] = int(time.time_ns())

        # Update in BlueNexus
        updated_record = await client.update(Collections.MESSAGES, record.id, message_data)
        return updated_record.model_dump()
    except Exception as e:
        log.error(f"Error updating message {message_id}: {e}")
        return None


async def add_reaction_to_message(
    user_id: str, message_id: str, reactor_user_id: str, name: str
) -> Optional[dict]:
    """Add a reaction to a message."""
    client = get_bluenexus_client_for_user(user_id)
    if not client:
        log.warning(f"No BlueNexus client for user {user_id}")
        return None

    try:
        response = await client.query(
            Collections.MESSAGES,
            QueryOptions(filter={"owui_id": message_id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            return None

        record = response.get_records()[0]
        message_data = record.model_dump()

        # Add reaction
        reactions = message_data.get("reactions", [])
        reactions.append({
            "id": str(uuid.uuid4()),
            "user_id": reactor_user_id,
            "name": name,
            "created_at": int(time.time_ns()),
        })
        message_data["reactions"] = reactions

        updated_record = await client.update(Collections.MESSAGES, record.id, message_data)
        return updated_record.model_dump()
    except Exception as e:
        log.error(f"Error adding reaction to message {message_id}: {e}")
        return None


async def remove_reaction_by_id_and_user_id_and_name(
    user_id: str, message_id: str, reactor_user_id: str, name: str
) -> bool:
    """Remove a reaction from a message."""
    client = get_bluenexus_client_for_user(user_id)
    if not client:
        log.warning(f"No BlueNexus client for user {user_id}")
        return False

    try:
        response = await client.query(
            Collections.MESSAGES,
            QueryOptions(filter={"owui_id": message_id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            return False

        record = response.get_records()[0]
        message_data = record.model_dump()

        # Remove matching reaction
        reactions = message_data.get("reactions", [])
        message_data["reactions"] = [
            r for r in reactions
            if not (r.get("user_id") == reactor_user_id and r.get("name") == name)
        ]

        await client.update(Collections.MESSAGES, record.id, message_data)
        return True
    except Exception as e:
        log.error(f"Error removing reaction from message {message_id}: {e}")
        return False


async def delete_message_by_id(user_id: str, message_id: str) -> bool:
    """Delete a message and its reactions."""
    client = get_bluenexus_client_for_user(user_id)
    if not client:
        log.warning(f"No BlueNexus client for user {user_id}")
        return False

    try:
        response = await client.query(
            Collections.MESSAGES,
            QueryOptions(filter={"owui_id": message_id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            return False

        record = response.get_records()[0]
        await client.delete(Collections.MESSAGES, record.id)
        return True
    except Exception as e:
        log.error(f"Error deleting message {message_id}: {e}")
        return False


# Helper functions to build response objects
def build_message_user_response(message_data: dict) -> dict:
    """Build a MessageUserResponse from message data."""
    user = Users.get_user_by_id(message_data.get("user_id", ""))
    return {
        **message_data,
        "user": user.model_dump() if user else None,
    }


async def build_message_response(user_id: str, message_data: dict) -> dict:
    """Build a full MessageResponse from message data."""
    message_id = message_data.get("owui_id", message_data.get("id"))

    # Get reply_to message if exists
    reply_to_message = None
    if message_data.get("reply_to_id"):
        reply_to = await get_message_by_id(user_id, message_data["reply_to_id"])
        if reply_to:
            reply_to_message = build_message_user_response(reply_to)

    # Get thread replies
    thread_replies = await get_thread_replies_by_message_id(user_id, message_id)

    # Get reactions
    reactions = await get_reactions_by_message_id(user_id, message_id)

    # Get user info
    user = Users.get_user_by_id(message_data.get("user_id", ""))

    return {
        **message_data,
        "user": user.model_dump() if user else None,
        "reply_to_message": reply_to_message,
        "latest_reply_at": thread_replies[0].get("created_at") if thread_replies else None,
        "reply_count": len(thread_replies),
        "reactions": reactions,
    }


# Sync wrappers for compatibility with sync code
def insert_new_message_sync(user_id: str, form_data: MessageForm, channel_id: str) -> Optional[dict]:
    """Insert a new message (sync wrapper)."""
    return _run_async(insert_new_message(user_id, form_data, channel_id))


def get_message_by_id_sync(user_id: str, message_id: str) -> Optional[dict]:
    """Get a message by ID (sync wrapper)."""
    return _run_async(get_message_by_id(user_id, message_id))


def get_messages_by_channel_id_sync(
    user_id: str, channel_id: str, skip: int = 0, limit: int = 50
) -> list[dict]:
    """Get messages for a channel (sync wrapper)."""
    return _run_async(get_messages_by_channel_id(user_id, channel_id, skip, limit))


def get_messages_by_parent_id_sync(
    user_id: str, channel_id: str, parent_id: str, skip: int = 0, limit: int = 50
) -> list[dict]:
    """Get thread replies (sync wrapper)."""
    return _run_async(get_messages_by_parent_id(user_id, channel_id, parent_id, skip, limit))


def get_thread_replies_by_message_id_sync(user_id: str, message_id: str) -> list[dict]:
    """Get thread replies (sync wrapper)."""
    return _run_async(get_thread_replies_by_message_id(user_id, message_id))


def get_reactions_by_message_id_sync(user_id: str, message_id: str) -> list[dict]:
    """Get reactions (sync wrapper)."""
    return _run_async(get_reactions_by_message_id(user_id, message_id))


def update_message_by_id_sync(
    user_id: str, message_id: str, form_data: MessageForm
) -> Optional[dict]:
    """Update a message (sync wrapper)."""
    return _run_async(update_message_by_id(user_id, message_id, form_data))


def add_reaction_to_message_sync(
    user_id: str, message_id: str, reactor_user_id: str, name: str
) -> Optional[dict]:
    """Add reaction (sync wrapper)."""
    return _run_async(add_reaction_to_message(user_id, message_id, reactor_user_id, name))


def remove_reaction_sync(user_id: str, message_id: str, reactor_user_id: str, name: str) -> bool:
    """Remove reaction (sync wrapper)."""
    return _run_async(remove_reaction_by_id_and_user_id_and_name(user_id, message_id, reactor_user_id, name))


def delete_message_by_id_sync(user_id: str, message_id: str) -> bool:
    """Delete a message (sync wrapper)."""
    return _run_async(delete_message_by_id(user_id, message_id))


def build_message_response_sync(user_id: str, message_data: dict) -> dict:
    """Build message response (sync wrapper)."""
    return _run_async(build_message_response(user_id, message_data))
