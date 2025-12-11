"""
BlueNexus Chat Operations

Async helper functions for chat operations using BlueNexus storage.
These replace the sync Chats model methods for real-time socket operations.
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
            # If there's already a running loop, create a new one
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop exists, create one
        return asyncio.run(coro)


async def get_chat_by_id(user_id: str, chat_id: str) -> Optional[dict]:
    """Get a chat by its owui_id."""
    client = get_bluenexus_client_for_user(user_id)
    if not client:
        log.warning(f"No BlueNexus client for user {user_id}")
        return None

    try:
        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"owui_id": chat_id, "user_id": user_id}, limit=1)
        )

        if response.data and len(response.data) > 0:
            record = response.get_records()[0]
            return record.model_dump()
        return None
    except Exception as e:
        log.error(f"Error getting chat {chat_id}: {e}")
        return None


async def get_chat_by_id_and_user_id(user_id: str, chat_id: str) -> Optional[dict]:
    """Get a chat by its owui_id and user_id (alias for get_chat_by_id)."""
    return await get_chat_by_id(user_id, chat_id)


async def update_chat_by_id(user_id: str, chat_id: str, chat_data: dict) -> Optional[dict]:
    """Update a chat by its owui_id."""
    client = get_bluenexus_client_for_user(user_id)
    if not client:
        log.warning(f"No BlueNexus client for user {user_id}")
        return None

    try:
        # First find the record to get the BlueNexus record ID
        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"owui_id": chat_id, "user_id": user_id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            log.warning(f"Chat {chat_id} not found for user {user_id}")
            return None

        record = response.get_records()[0]
        existing_data = record.model_dump()

        # Merge chat content
        existing_data["chat"] = chat_data

        # Update in BlueNexus
        updated_record = await client.update(Collections.CHATS, record.id, existing_data)
        return updated_record.model_dump()

    except Exception as e:
        log.error(f"Error updating chat {chat_id}: {e}")
        return None


async def get_message_by_id_and_message_id(
    user_id: str, chat_id: str, message_id: str
) -> Optional[dict]:
    """Get a specific message from a chat."""
    chat_data = await get_chat_by_id(user_id, chat_id)
    if chat_data is None:
        return None

    chat = chat_data.get("chat", {})
    return chat.get("history", {}).get("messages", {}).get(message_id, {})


async def upsert_message_to_chat_by_id_and_message_id(
    user_id: str, chat_id: str, message_id: str, message: dict
) -> Optional[dict]:
    """Update or insert a message in a chat."""
    chat_data = await get_chat_by_id(user_id, chat_id)
    if chat_data is None:
        return None

    # Sanitize message content for null characters
    if isinstance(message.get("content"), str):
        message["content"] = message["content"].replace("\x00", "")

    chat = chat_data.get("chat", {})
    history = chat.get("history", {})
    messages = history.get("messages", {})

    if message_id in messages:
        messages[message_id] = {**messages[message_id], **message}
    else:
        messages[message_id] = message

    history["messages"] = messages
    history["currentId"] = message_id
    chat["history"] = history

    return await update_chat_by_id(user_id, chat_id, chat)


async def add_message_status_to_chat_by_id_and_message_id(
    user_id: str, chat_id: str, message_id: str, status: dict
) -> Optional[dict]:
    """Add a status entry to a message's status history."""
    chat_data = await get_chat_by_id(user_id, chat_id)
    if chat_data is None:
        return None

    chat = chat_data.get("chat", {})
    history = chat.get("history", {})
    messages = history.get("messages", {})

    if message_id in messages:
        status_history = messages[message_id].get("statusHistory", [])
        status_history.append(status)
        messages[message_id]["statusHistory"] = status_history

    history["messages"] = messages
    chat["history"] = history

    return await update_chat_by_id(user_id, chat_id, chat)


async def get_messages_map_by_chat_id(user_id: str, chat_id: str) -> Optional[dict]:
    """Get the messages map from a chat."""
    chat_data = await get_chat_by_id(user_id, chat_id)
    if chat_data is None:
        return None

    chat = chat_data.get("chat", {})
    return chat.get("history", {}).get("messages", {})


async def update_chat_title_by_id(user_id: str, chat_id: str, title: str) -> Optional[dict]:
    """Update the title of a chat."""
    client = get_bluenexus_client_for_user(user_id)
    if not client:
        log.warning(f"No BlueNexus client for user {user_id}")
        return None

    try:
        # First find the record
        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"owui_id": chat_id, "user_id": user_id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            log.warning(f"Chat {chat_id} not found for user {user_id}")
            return None

        record = response.get_records()[0]
        chat_data = record.model_dump()

        # Update title
        chat_data["title"] = title
        if "chat" in chat_data:
            chat_data["chat"]["title"] = title

        # Update in BlueNexus
        updated_record = await client.update(Collections.CHATS, record.id, chat_data)
        return updated_record.model_dump()

    except Exception as e:
        log.error(f"Error updating chat title {chat_id}: {e}")
        return None


async def get_chat_title_by_id(user_id: str, chat_id: str) -> Optional[str]:
    """Get the title of a chat."""
    chat_data = await get_chat_by_id(user_id, chat_id)
    if chat_data is None:
        return None
    return chat_data.get("title")


async def update_chat_tags_by_id(user_id: str, chat_id: str, tags: list) -> Optional[dict]:
    """Update the tags of a chat."""
    client = get_bluenexus_client_for_user(user_id)
    if not client:
        log.warning(f"No BlueNexus client for user {user_id}")
        return None

    try:
        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"owui_id": chat_id, "user_id": user_id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            log.warning(f"Chat {chat_id} not found for user {user_id}")
            return None

        record = response.get_records()[0]
        chat_data = record.model_dump()

        # Update tags
        chat_data["meta"] = chat_data.get("meta", {})
        chat_data["meta"]["tags"] = tags

        # Update in BlueNexus
        updated_record = await client.update(Collections.CHATS, record.id, chat_data)
        return updated_record.model_dump()

    except Exception as e:
        log.error(f"Error updating chat tags {chat_id}: {e}")
        return None


async def get_chat_by_share_id(user_id: str, share_id: str) -> Optional[dict]:
    """Get a chat by its share_id (for shared chat lookups)."""
    client = get_bluenexus_client_for_user(user_id)
    if not client:
        log.warning(f"No BlueNexus client for user {user_id}")
        return None

    try:
        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"share_id": share_id}, limit=1)
        )

        if response.data and len(response.data) > 0:
            record = response.get_records()[0]
            return record.model_dump()
        return None
    except Exception as e:
        log.error(f"Error getting chat by share_id {share_id}: {e}")
        return None


async def count_chats_by_folder_id_and_user_id(user_id: str, folder_id: str) -> int:
    """Count chats in a folder for a user."""
    client = get_bluenexus_client_for_user(user_id)
    if not client:
        log.warning(f"No BlueNexus client for user {user_id}")
        return 0

    try:
        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"folder_id": folder_id, "user_id": user_id})
        )
        return len(response.data) if response.data else 0
    except Exception as e:
        log.error(f"Error counting chats in folder {folder_id}: {e}")
        return 0


async def delete_chats_by_user_id_and_folder_id(user_id: str, folder_id: str) -> bool:
    """Delete all chats in a folder for a user."""
    client = get_bluenexus_client_for_user(user_id)
    if not client:
        log.warning(f"No BlueNexus client for user {user_id}")
        return False

    try:
        # First get all chats in the folder
        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"folder_id": folder_id, "user_id": user_id})
        )

        if not response.data:
            return True

        # Delete each chat
        for record in response.get_records():
            await client.delete(Collections.CHATS, record.id)

        return True
    except Exception as e:
        log.error(f"Error deleting chats in folder {folder_id}: {e}")
        return False


async def delete_chats_by_user_id_async(user_id: str) -> bool:
    """Delete all chats for a user (async version)."""
    client = get_bluenexus_client_for_user(user_id)
    if not client:
        log.warning(f"No BlueNexus client for user {user_id}")
        return False

    try:
        # Get all chats for the user
        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"user_id": user_id})
        )

        if not response.data:
            return True

        # Delete each chat
        for record in response.get_records():
            await client.delete(Collections.CHATS, record.id)

        return True
    except Exception as e:
        log.error(f"Error deleting chats for user {user_id}: {e}")
        return False


def delete_chats_by_user_id(user_id: str) -> bool:
    """Delete all chats for a user (sync wrapper)."""
    return _run_async(delete_chats_by_user_id_async(user_id))


def get_chat_by_id_sync(user_id: str, chat_id: str) -> Optional[dict]:
    """Get a chat by its owui_id (sync wrapper)."""
    return _run_async(get_chat_by_id(user_id, chat_id))
