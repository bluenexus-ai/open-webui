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
from open_webui.utils.bluenexus.cache import (
    chat_cache,
    messages_cache,
    make_cache_key,
    invalidate_chat_cache,
    get_cached_record_id,
    set_cached_record_id,
)

log = logging.getLogger(__name__)


# Store record IDs alongside chat data in cache for fast updates
# Format: {"data": chat_data, "record_id": bluenexus_record_id}
def _cache_chat_with_record_id(user_id: str, chat_id: str, data: dict, record_id: str) -> None:
    """Cache chat data along with its BlueNexus record ID."""
    cache_key = make_cache_key(user_id, "chats", chat_id)
    chat_cache.set(cache_key, {"data": data, "record_id": record_id})
    set_cached_record_id(user_id, "chats", chat_id, record_id)


def _get_cached_chat(user_id: str, chat_id: str) -> tuple[Optional[dict], Optional[str]]:
    """Get cached chat data and record ID. Returns (data, record_id)."""
    cache_key = make_cache_key(user_id, "chats", chat_id)
    cached = chat_cache.get(cache_key)
    if cached is not None:
        # Handle both old format (just data) and new format (data + record_id)
        if isinstance(cached, dict) and "data" in cached and "record_id" in cached:
            return cached["data"], cached["record_id"]
        else:
            # Old format - try to get record_id from separate cache
            record_id = get_cached_record_id(user_id, "chats", chat_id)
            return cached, record_id
    return None, None


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


async def get_chat_by_id(user_id: str, chat_id: str, use_cache: bool = True) -> Optional[dict]:
    """Get a chat by its owui_id."""
    # Check cache first
    if use_cache:
        cached_data, cached_record_id = _get_cached_chat(user_id, chat_id)
        if cached_data is not None:
            log.debug(f"[BlueNexus Cache] HIT for chat {chat_id}")
            return cached_data

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
            result = record.model_dump()
            # Cache the result WITH record ID for fast updates
            _cache_chat_with_record_id(user_id, chat_id, result, record.id)
            return result
        return None
    except Exception as e:
        log.error(f"Error getting chat {chat_id}: {e}")
        return None


async def get_chat_by_id_and_user_id(user_id: str, chat_id: str) -> Optional[dict]:
    """Get a chat by its owui_id and user_id (alias for get_chat_by_id)."""
    return await get_chat_by_id(user_id, chat_id)


async def update_chat_by_id(user_id: str, chat_id: str, chat_data: dict) -> Optional[dict]:
    """Update a chat by its owui_id. Optimized to use cached record ID."""
    client = get_bluenexus_client_for_user(user_id)
    if not client:
        log.warning(f"No BlueNexus client for user {user_id}")
        return None

    try:
        # Try to get record ID and existing data from cache first
        cached_data, record_id = _get_cached_chat(user_id, chat_id)

        if record_id and cached_data:
            # FAST PATH: Use cached record ID - only 1 API call (update)
            log.debug(f"[BlueNexus] Fast update for chat {chat_id} using cached record_id")
            existing_data = cached_data.copy()
            existing_data["chat"] = chat_data

            updated_record = await client.update(Collections.CHATS, record_id, existing_data)
            result = updated_record.model_dump()

            # Update cache (don't invalidate - just update)
            _cache_chat_with_record_id(user_id, chat_id, result, record_id)
            return result

        # SLOW PATH: Need to query for record ID first - 2 API calls (query + update)
        log.debug(f"[BlueNexus] Slow update for chat {chat_id} - querying for record_id")
        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"owui_id": chat_id, "user_id": user_id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            log.warning(f"Chat {chat_id} not found for user {user_id}")
            return None

        record = response.get_records()[0]
        existing_data = record.model_dump()
        record_id = record.id

        # Merge chat content
        existing_data["chat"] = chat_data

        # Update in BlueNexus
        updated_record = await client.update(Collections.CHATS, record_id, existing_data)
        result = updated_record.model_dump()

        # Cache with record ID for future fast updates
        _cache_chat_with_record_id(user_id, chat_id, result, record_id)
        return result

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
    """Update the title of a chat. Optimized to use cached record ID."""
    client = get_bluenexus_client_for_user(user_id)
    if not client:
        log.warning(f"No BlueNexus client for user {user_id}")
        return None

    try:
        # Try to get record ID and existing data from cache first
        cached_data, record_id = _get_cached_chat(user_id, chat_id)

        if record_id and cached_data:
            # FAST PATH: Use cached data and record ID
            chat_data = cached_data.copy()
            chat_data["title"] = title
            if "chat" in chat_data:
                chat_data["chat"]["title"] = title

            updated_record = await client.update(Collections.CHATS, record_id, chat_data)
            result = updated_record.model_dump()
            _cache_chat_with_record_id(user_id, chat_id, result, record_id)
            return result

        # SLOW PATH: Query for record first
        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"owui_id": chat_id, "user_id": user_id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            log.warning(f"Chat {chat_id} not found for user {user_id}")
            return None

        record = response.get_records()[0]
        chat_data = record.model_dump()
        record_id = record.id

        # Update title
        chat_data["title"] = title
        if "chat" in chat_data:
            chat_data["chat"]["title"] = title

        updated_record = await client.update(Collections.CHATS, record_id, chat_data)
        result = updated_record.model_dump()
        _cache_chat_with_record_id(user_id, chat_id, result, record_id)
        return result

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
    """Update the tags of a chat. Optimized to use cached record ID."""
    client = get_bluenexus_client_for_user(user_id)
    if not client:
        log.warning(f"No BlueNexus client for user {user_id}")
        return None

    try:
        # Try to get record ID and existing data from cache first
        cached_data, record_id = _get_cached_chat(user_id, chat_id)

        if record_id and cached_data:
            # FAST PATH: Use cached data and record ID
            chat_data = cached_data.copy()
            chat_data["meta"] = chat_data.get("meta", {})
            chat_data["meta"]["tags"] = tags

            updated_record = await client.update(Collections.CHATS, record_id, chat_data)
            result = updated_record.model_dump()
            _cache_chat_with_record_id(user_id, chat_id, result, record_id)
            return result

        # SLOW PATH: Query for record first
        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"owui_id": chat_id, "user_id": user_id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            log.warning(f"Chat {chat_id} not found for user {user_id}")
            return None

        record = response.get_records()[0]
        chat_data = record.model_dump()
        record_id = record.id

        # Update tags
        chat_data["meta"] = chat_data.get("meta", {})
        chat_data["meta"]["tags"] = tags

        updated_record = await client.update(Collections.CHATS, record_id, chat_data)
        result = updated_record.model_dump()
        _cache_chat_with_record_id(user_id, chat_id, result, record_id)
        return result

    except Exception as e:
        log.error(f"Error updating chat tags {chat_id}: {e}")
        return None


async def update_chat_title_and_tags(
    user_id: str, chat_id: str, title: str = None, tags: list = None
) -> Optional[dict]:
    """
    Update chat title and/or tags in a single API call.

    This combines update_chat_title_by_id and update_chat_tags_by_id
    to reduce API calls from 2 to 1.
    """
    if title is None and tags is None:
        return None

    client = get_bluenexus_client_for_user(user_id)
    if not client:
        log.warning(f"No BlueNexus client for user {user_id}")
        return None

    try:
        # Try to get record ID and existing data from cache first
        cached_data, record_id = _get_cached_chat(user_id, chat_id)

        if record_id and cached_data:
            # FAST PATH: Use cached data and record ID - 1 API call
            log.debug(f"[BlueNexus] Fast update title/tags for chat {chat_id} using cached record_id")
            chat_data = cached_data.copy()

            if title is not None:
                chat_data["title"] = title
                if "chat" in chat_data:
                    chat_data["chat"]["title"] = title

            if tags is not None:
                chat_data["meta"] = chat_data.get("meta", {})
                chat_data["meta"]["tags"] = tags

            updated_record = await client.update(Collections.CHATS, record_id, chat_data)
            result = updated_record.model_dump()
            _cache_chat_with_record_id(user_id, chat_id, result, record_id)
            return result

        # SLOW PATH: Query for record first - 2 API calls
        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"owui_id": chat_id, "user_id": user_id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            log.warning(f"Chat {chat_id} not found for user {user_id}")
            return None

        record = response.get_records()[0]
        chat_data = record.model_dump()
        record_id = record.id

        if title is not None:
            chat_data["title"] = title
            if "chat" in chat_data:
                chat_data["chat"]["title"] = title

        if tags is not None:
            chat_data["meta"] = chat_data.get("meta", {})
            chat_data["meta"]["tags"] = tags

        updated_record = await client.update(Collections.CHATS, record_id, chat_data)
        result = updated_record.model_dump()
        _cache_chat_with_record_id(user_id, chat_id, result, record_id)
        return result

    except Exception as e:
        log.error(f"Error updating chat title/tags {chat_id}: {e}")
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


# ============================================================================
# Batched Message Updates - reduces API calls during streaming
# ============================================================================

class MessageUpdateBatcher:
    """
    Batches message updates during streaming to reduce BlueNexus API calls.

    Instead of updating on every chunk, collects updates and flushes periodically.
    This can reduce 20-50 API calls per message to just 3-5.

    Usage:
        batcher = MessageUpdateBatcher(user_id, chat_id)
        async with batcher:
            # During streaming:
            await batcher.update_message(message_id, {"content": "chunk1"})
            await batcher.update_message(message_id, {"content": "chunk1chunk2"})
            # ... more updates
        # Auto-flushes on exit
    """

    def __init__(
        self,
        user_id: str,
        chat_id: str,
        flush_interval: float = 2.0,  # Flush every 2 seconds
        max_pending: int = 10,  # Or after 10 pending updates
    ):
        self.user_id = user_id
        self.chat_id = chat_id
        self.flush_interval = flush_interval
        self.max_pending = max_pending
        self._pending_updates: dict[str, dict] = {}  # message_id -> latest message data
        self._pending_count = 0
        self._last_flush = 0.0
        self._lock = asyncio.Lock()
        self._chat_data: Optional[dict] = None
        self._record_id: Optional[str] = None

    async def __aenter__(self):
        """Load chat data on entry."""
        self._chat_data, self._record_id = _get_cached_chat(self.user_id, self.chat_id)
        if not self._chat_data:
            self._chat_data = await get_chat_by_id(self.user_id, self.chat_id)
            if self._chat_data:
                _, self._record_id = _get_cached_chat(self.user_id, self.chat_id)
        self._last_flush = asyncio.get_event_loop().time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Flush any pending updates on exit."""
        await self.flush()
        return False

    async def update_message(self, message_id: str, message: dict) -> None:
        """Queue a message update. May trigger flush if threshold reached."""
        async with self._lock:
            if self._chat_data is None:
                return

            # Sanitize content
            if isinstance(message.get("content"), str):
                message["content"] = message["content"].replace("\x00", "")

            # Merge with existing pending update for this message
            if message_id in self._pending_updates:
                self._pending_updates[message_id] = {
                    **self._pending_updates[message_id],
                    **message
                }
            else:
                self._pending_updates[message_id] = message
                self._pending_count += 1

            # Check if we should flush
            now = asyncio.get_event_loop().time()
            should_flush = (
                self._pending_count >= self.max_pending or
                (now - self._last_flush) >= self.flush_interval
            )

            if should_flush:
                await self._flush_locked()

    async def add_status(self, message_id: str, status: dict) -> None:
        """Queue a status update for a message."""
        async with self._lock:
            if self._chat_data is None:
                return

            # Get or create pending update for this message
            if message_id not in self._pending_updates:
                self._pending_updates[message_id] = {}
                self._pending_count += 1

            # Append to status history
            status_history = self._pending_updates[message_id].get("statusHistory", [])
            status_history.append(status)
            self._pending_updates[message_id]["statusHistory"] = status_history

    async def flush(self) -> Optional[dict]:
        """Flush all pending updates to BlueNexus."""
        async with self._lock:
            return await self._flush_locked()

    async def _flush_locked(self) -> Optional[dict]:
        """Internal flush (must hold lock)."""
        if not self._pending_updates or self._chat_data is None:
            return self._chat_data

        log.debug(f"[BlueNexus Batcher] Flushing {len(self._pending_updates)} message updates for chat {self.chat_id}")

        # Apply all pending updates to chat data
        chat = self._chat_data.get("chat", {})
        history = chat.get("history", {})
        messages = history.get("messages", {})

        for message_id, update in self._pending_updates.items():
            if message_id in messages:
                messages[message_id] = {**messages[message_id], **update}
            else:
                messages[message_id] = update
            history["currentId"] = message_id

        history["messages"] = messages
        chat["history"] = history

        # Update in BlueNexus
        result = await update_chat_by_id(self.user_id, self.chat_id, chat)

        # Clear pending updates
        self._pending_updates.clear()
        self._pending_count = 0
        self._last_flush = asyncio.get_event_loop().time()

        if result:
            self._chat_data = result

        return result


# Global batchers for active chats (auto-cleanup on completion)
_active_batchers: dict[str, MessageUpdateBatcher] = {}


def get_message_batcher(user_id: str, chat_id: str) -> MessageUpdateBatcher:
    """Get or create a message batcher for a chat."""
    key = f"{user_id}:{chat_id}"
    if key not in _active_batchers:
        _active_batchers[key] = MessageUpdateBatcher(user_id, chat_id)
    return _active_batchers[key]


def remove_message_batcher(user_id: str, chat_id: str) -> None:
    """Remove a message batcher (call after streaming completes)."""
    key = f"{user_id}:{chat_id}"
    if key in _active_batchers:
        del _active_batchers[key]
