"""
BlueNexus Chat Storage

This module provides a BlueNexus-backed implementation of chat storage
for Open WebUI, allowing users who authenticate via BlueNexus to store
their chat data in the BlueNexus User-Data API.
"""

import logging
import time
import uuid
from typing import Any, Optional

from open_webui.env import SRC_LOG_LEVELS
from open_webui.utils.bluenexus.client import BlueNexusDataClient
from open_webui.utils.bluenexus.collections import Collections
from open_webui.utils.bluenexus.types import (
    BlueNexusRecord,
    BlueNexusNotFoundError,
    QueryOptions,
    SortBy,
    SortOrder,
)

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("BLUENEXUS", SRC_LOG_LEVELS.get("MAIN", logging.INFO)))


class ChatData:
    """
    Represents chat data in BlueNexus format.

    This class handles transformation between Open WebUI's ChatModel
    and BlueNexus record format.
    """

    # Fields stored in BlueNexus
    FIELDS = [
        "owui_id",        # Original Open WebUI ID
        "user_id",        # User who owns the chat
        "title",          # Chat title
        "chat",           # Full chat data (messages, history)
        "share_id",       # Share link ID
        "archived",       # Is archived
        "pinned",         # Is pinned
        "meta",           # Metadata (tags, etc.)
        "folder_id",      # Folder organization
        "owui_created_at",  # Original creation timestamp
        "owui_updated_at",  # Original update timestamp
    ]

    @staticmethod
    def to_bluenexus(chat_model: dict) -> dict:
        """
        Convert Open WebUI ChatModel to BlueNexus record format.

        Args:
            chat_model: Dictionary from ChatModel

        Returns:
            Dictionary for BlueNexus storage
        """
        return {
            "owui_id": chat_model.get("id"),
            "user_id": chat_model.get("user_id"),
            "title": chat_model.get("title", "New Chat"),
            "chat": chat_model.get("chat", {}),
            "share_id": chat_model.get("share_id"),
            "archived": chat_model.get("archived", False),
            "pinned": chat_model.get("pinned", False),
            "meta": chat_model.get("meta", {}),
            "folder_id": chat_model.get("folder_id"),
            "owui_created_at": chat_model.get("created_at", int(time.time())),
            "owui_updated_at": chat_model.get("updated_at", int(time.time())),
        }

    @staticmethod
    def from_bluenexus(record: BlueNexusRecord) -> dict:
        """
        Convert BlueNexus record to Open WebUI ChatModel format.

        Args:
            record: BlueNexusRecord from API

        Returns:
            Dictionary compatible with ChatModel
        """
        data = record.to_dict()
        return {
            "id": data.get("owui_id", record.id),
            "user_id": data.get("user_id"),
            "title": data.get("title", "New Chat"),
            "chat": data.get("chat", {}),
            "share_id": data.get("share_id"),
            "archived": data.get("archived", False),
            "pinned": data.get("pinned", False),
            "meta": data.get("meta", {}),
            "folder_id": data.get("folder_id"),
            "created_at": data.get("owui_created_at", int(record.createdAt.timestamp())),
            "updated_at": data.get("owui_updated_at", int(record.updatedAt.timestamp())),
            # Store BlueNexus record ID for reference
            "_bluenexus_id": record.id,
        }

    @staticmethod
    def from_bluenexus_dict(data: dict) -> dict:
        """
        Convert BlueNexus record dict to Open WebUI ChatModel format.

        Args:
            data: Raw dictionary from BlueNexus API

        Returns:
            Dictionary compatible with ChatModel
        """
        # Parse timestamps
        created_at = data.get("owui_created_at")
        if not created_at and "createdAt" in data:
            from datetime import datetime
            try:
                dt = datetime.fromisoformat(data["createdAt"].replace("Z", "+00:00"))
                created_at = int(dt.timestamp())
            except (ValueError, TypeError):
                created_at = int(time.time())

        updated_at = data.get("owui_updated_at")
        if not updated_at and "updatedAt" in data:
            from datetime import datetime
            try:
                dt = datetime.fromisoformat(data["updatedAt"].replace("Z", "+00:00"))
                updated_at = int(dt.timestamp())
            except (ValueError, TypeError):
                updated_at = int(time.time())

        return {
            "id": data.get("owui_id", data.get("id")),
            "user_id": data.get("user_id"),
            "title": data.get("title", "New Chat"),
            "chat": data.get("chat", {}),
            "share_id": data.get("share_id"),
            "archived": data.get("archived", False),
            "pinned": data.get("pinned", False),
            "meta": data.get("meta", {}),
            "folder_id": data.get("folder_id"),
            "created_at": created_at or int(time.time()),
            "updated_at": updated_at or int(time.time()),
            "_bluenexus_id": data.get("id"),
        }


class BlueNexusChatStorage:
    """
    BlueNexus-backed chat storage for Open WebUI.

    This class provides methods to store and retrieve chats using
    the BlueNexus User-Data API. It mirrors the interface of ChatTable
    but stores data in BlueNexus instead of the local database.
    """

    def __init__(self, client: BlueNexusDataClient):
        """
        Initialize BlueNexus chat storage.

        Args:
            client: BlueNexusDataClient instance with valid auth
        """
        self.client = client
        self.collection = Collections.CHATS.value
        log.info(f"[BlueNexus Chat Storage] Initialized with collection='{self.collection}'")

    # =========================================================================
    # ID Mapping
    # =========================================================================

    async def _get_bluenexus_id_by_owui_id(self, owui_id: str) -> Optional[str]:
        """
        Find BlueNexus record ID by Open WebUI ID.

        Args:
            owui_id: Open WebUI chat ID

        Returns:
            BlueNexus record ID if found, None otherwise
        """
        log.debug(f"[BlueNexus Chat Storage] Looking up BlueNexus ID for owui_id={owui_id}")
        response = await self.client.query(
            self.collection,
            QueryOptions(filter={"owui_id": owui_id}, limit=1),
        )
        if response.data:
            bn_id = response.data[0].get("id")
            log.debug(f"[BlueNexus Chat Storage] Found BlueNexus ID={bn_id} for owui_id={owui_id}")
            return bn_id
        log.debug(f"[BlueNexus Chat Storage] No BlueNexus record found for owui_id={owui_id}")
        return None

    # =========================================================================
    # Create Operations
    # =========================================================================

    async def insert_new_chat(
        self,
        user_id: str,
        chat_data: dict,
        folder_id: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Create a new chat.

        Args:
            user_id: Owner user ID
            chat_data: Chat content (messages, history)
            folder_id: Optional folder ID

        Returns:
            Created chat as dictionary (ChatModel compatible)
        """
        owui_id = str(uuid.uuid4())
        now = int(time.time())

        chat_model = {
            "id": owui_id,
            "user_id": user_id,
            "title": chat_data.get("title", "New Chat"),
            "chat": chat_data,
            "folder_id": folder_id,
            "created_at": now,
            "updated_at": now,
            "archived": False,
            "pinned": False,
            "meta": {},
        }

        bn_data = ChatData.to_bluenexus(chat_model)

        log.info(f"[BlueNexus Chat Storage] Creating chat in collection='{self.collection}' owui_id={owui_id} for user={user_id}")
        log.debug(f"[BlueNexus Chat Storage] Chat data: {bn_data}")

        record = await self.client.create(self.collection, bn_data)
        result = ChatData.from_bluenexus(record)

        log.info(f"[BlueNexus Chat Storage] Created chat bn_id={record.id}, owui_id={owui_id} in collection='{self.collection}'")

        return result

    async def import_chat(
        self,
        user_id: str,
        chat_data: dict,
        meta: Optional[dict] = None,
        pinned: bool = False,
        folder_id: Optional[str] = None,
        created_at: Optional[int] = None,
        updated_at: Optional[int] = None,
    ) -> Optional[dict]:
        """
        Import an existing chat (e.g., from export).

        Args:
            user_id: Owner user ID
            chat_data: Chat content
            meta: Chat metadata
            pinned: Is pinned
            folder_id: Folder ID
            created_at: Original creation timestamp
            updated_at: Original update timestamp

        Returns:
            Imported chat as dictionary
        """
        owui_id = str(uuid.uuid4())
        now = int(time.time())

        chat_model = {
            "id": owui_id,
            "user_id": user_id,
            "title": chat_data.get("title", "New Chat"),
            "chat": chat_data,
            "meta": meta or {},
            "pinned": pinned,
            "folder_id": folder_id,
            "created_at": created_at or now,
            "updated_at": updated_at or now,
            "archived": False,
        }

        bn_data = ChatData.to_bluenexus(chat_model)

        log.info(f"[BlueNexus Chat Storage] Importing chat to collection='{self.collection}' for user={user_id}")

        record = await self.client.create(self.collection, bn_data)
        log.info(f"[BlueNexus Chat Storage] Imported chat bn_id={record.id} to collection='{self.collection}'")
        return ChatData.from_bluenexus(record)

    # =========================================================================
    # Read Operations
    # =========================================================================

    async def get_chat_by_id(self, chat_id: str) -> Optional[dict]:
        """
        Get a chat by its Open WebUI ID.

        Args:
            chat_id: Open WebUI chat ID

        Returns:
            Chat as dictionary or None if not found
        """
        log.info(f"[BlueNexus Chat Storage] Getting chat by id={chat_id} from collection='{self.collection}'")
        try:
            # First try to find by owui_id
            bn_id = await self._get_bluenexus_id_by_owui_id(chat_id)
            if bn_id:
                log.debug(f"[BlueNexus Chat Storage] Found by owui_id, fetching bn_id={bn_id}")
                record = await self.client.get(self.collection, bn_id)
                log.info(f"[BlueNexus Chat Storage] Retrieved chat bn_id={bn_id} from collection='{self.collection}'")
                return ChatData.from_bluenexus(record)

            # Try direct BlueNexus ID lookup
            log.debug(f"[BlueNexus Chat Storage] Trying direct BlueNexus ID lookup for {chat_id}")
            record = await self.client.get(self.collection, chat_id)
            log.info(f"[BlueNexus Chat Storage] Retrieved chat by direct ID={chat_id} from collection='{self.collection}'")
            return ChatData.from_bluenexus(record)

        except BlueNexusNotFoundError:
            log.info(f"[BlueNexus Chat Storage] Chat not found: {chat_id} in collection='{self.collection}'")
            return None
        except Exception as e:
            log.error(f"[BlueNexus Chat Storage] Error getting chat {chat_id} from collection='{self.collection}': {e}")
            return None

    async def get_chat_by_id_and_user_id(
        self,
        chat_id: str,
        user_id: str,
    ) -> Optional[dict]:
        """
        Get a chat by ID with user ownership check.

        Args:
            chat_id: Chat ID
            user_id: Expected owner user ID

        Returns:
            Chat if found and owned by user, None otherwise
        """
        chat = await self.get_chat_by_id(chat_id)
        if chat and chat.get("user_id") == user_id:
            return chat
        return None

    async def get_chats_by_user_id(
        self,
        user_id: str,
        include_archived: bool = False,
        skip: int = 0,
        limit: int = 50,
    ) -> list[dict]:
        """
        Get all chats for a user.

        Args:
            user_id: User ID
            include_archived: Include archived chats
            skip: Number of records to skip
            limit: Maximum records to return

        Returns:
            List of chat dictionaries
        """
        filter_query = {"user_id": user_id}
        if not include_archived:
            filter_query["archived"] = False

        # Calculate page from skip/limit
        page = (skip // limit) + 1 if limit > 0 else 1

        log.info(f"[BlueNexus Chat Storage] Querying chats from collection='{self.collection}' for user={user_id}, filter={filter_query}, page={page}, limit={limit}")

        response = await self.client.query(
            self.collection,
            QueryOptions(
                filter=filter_query,
                sort_by=SortBy.UPDATED_AT,
                sort_order=SortOrder.DESC,
                limit=limit,
                page=page,
            ),
        )

        log.info(f"[BlueNexus Chat Storage] Query returned {len(response.data)} chats from collection='{self.collection}', total={response.pagination.total}")

        return [ChatData.from_bluenexus_dict(data) for data in response.data]

    async def get_chat_list_by_user_id(
        self,
        user_id: str,
        include_archived: bool = False,
        filter: Optional[dict] = None,
        skip: int = 0,
        limit: int = 50,
    ) -> list[dict]:
        """
        Get paginated chat list for a user with optional filtering.

        Args:
            user_id: User ID
            include_archived: Include archived chats
            filter: Additional filter options (query, order_by, direction)
            skip: Records to skip
            limit: Max records

        Returns:
            List of chat dictionaries
        """
        filter_query = {"user_id": user_id}
        if not include_archived:
            filter_query["archived"] = False

        # Handle additional filters
        sort_by = SortBy.UPDATED_AT
        sort_order = SortOrder.DESC

        if filter:
            if filter.get("query"):
                # BlueNexus supports regex for text search
                filter_query["title"] = {"$regex": filter["query"], "$options": "i"}

            order_by = filter.get("order_by")
            direction = filter.get("direction")

            if order_by == "created_at":
                sort_by = SortBy.CREATED_AT
            elif order_by == "updated_at":
                sort_by = SortBy.UPDATED_AT

            if direction and direction.lower() == "asc":
                sort_order = SortOrder.ASC

        page = (skip // limit) + 1 if limit > 0 else 1

        response = await self.client.query(
            self.collection,
            QueryOptions(
                filter=filter_query,
                sort_by=sort_by,
                sort_order=sort_order,
                limit=limit,
                page=page,
            ),
        )

        return [ChatData.from_bluenexus_dict(data) for data in response.data]

    async def get_pinned_chats_by_user_id(self, user_id: str) -> list[dict]:
        """Get pinned, non-archived chats for a user."""
        response = await self.client.query(
            self.collection,
            QueryOptions(
                filter={"user_id": user_id, "pinned": True, "archived": False},
                sort_by=SortBy.UPDATED_AT,
                sort_order=SortOrder.DESC,
                limit=100,
            ),
        )
        return [ChatData.from_bluenexus_dict(data) for data in response.data]

    async def get_archived_chats_by_user_id(self, user_id: str) -> list[dict]:
        """Get archived chats for a user."""
        response = await self.client.query(
            self.collection,
            QueryOptions(
                filter={"user_id": user_id, "archived": True},
                sort_by=SortBy.UPDATED_AT,
                sort_order=SortOrder.DESC,
                limit=100,
            ),
        )
        return [ChatData.from_bluenexus_dict(data) for data in response.data]

    async def get_chats_by_folder_id_and_user_id(
        self,
        folder_id: str,
        user_id: str,
        skip: int = 0,
        limit: int = 60,
    ) -> list[dict]:
        """Get chats in a specific folder."""
        page = (skip // limit) + 1 if limit > 0 else 1

        response = await self.client.query(
            self.collection,
            QueryOptions(
                filter={
                    "user_id": user_id,
                    "folder_id": folder_id,
                    "archived": False,
                    "$or": [{"pinned": False}, {"pinned": None}],
                },
                sort_by=SortBy.UPDATED_AT,
                sort_order=SortOrder.DESC,
                limit=limit,
                page=page,
            ),
        )
        return [ChatData.from_bluenexus_dict(data) for data in response.data]

    # =========================================================================
    # Update Operations
    # =========================================================================

    async def update_chat_by_id(self, chat_id: str, chat_data: dict) -> Optional[dict]:
        """
        Update chat content.

        Args:
            chat_id: Chat ID (Open WebUI ID)
            chat_data: New chat content

        Returns:
            Updated chat or None
        """
        try:
            bn_id = await self._get_bluenexus_id_by_owui_id(chat_id)
            if not bn_id:
                log.warning(f"[BlueNexus Chat] Chat not found for update: {chat_id}")
                return None

            # Get existing record to preserve fields
            existing = await self.client.get(self.collection, bn_id)
            existing_data = existing.to_dict()

            # Update fields
            existing_data["chat"] = chat_data
            existing_data["title"] = chat_data.get("title", "New Chat")
            existing_data["owui_updated_at"] = int(time.time())

            record = await self.client.update(self.collection, bn_id, existing_data)

            log.info(f"[BlueNexus Chat] Updated chat {chat_id}")
            return ChatData.from_bluenexus(record)

        except Exception as e:
            log.error(f"[BlueNexus Chat] Error updating chat {chat_id}: {e}")
            return None

    async def update_chat_title_by_id(
        self,
        chat_id: str,
        title: str,
    ) -> Optional[dict]:
        """Update chat title."""
        chat = await self.get_chat_by_id(chat_id)
        if not chat:
            return None

        chat_data = chat.get("chat", {})
        chat_data["title"] = title

        return await self.update_chat_by_id(chat_id, chat_data)

    async def toggle_chat_pinned_by_id(self, chat_id: str) -> Optional[dict]:
        """Toggle chat pinned status."""
        try:
            bn_id = await self._get_bluenexus_id_by_owui_id(chat_id)
            if not bn_id:
                return None

            existing = await self.client.get(self.collection, bn_id)
            existing_data = existing.to_dict()

            existing_data["pinned"] = not existing_data.get("pinned", False)
            existing_data["owui_updated_at"] = int(time.time())

            record = await self.client.update(self.collection, bn_id, existing_data)
            return ChatData.from_bluenexus(record)

        except Exception as e:
            log.error(f"[BlueNexus Chat] Error toggling pinned {chat_id}: {e}")
            return None

    async def toggle_chat_archive_by_id(self, chat_id: str) -> Optional[dict]:
        """Toggle chat archived status."""
        try:
            bn_id = await self._get_bluenexus_id_by_owui_id(chat_id)
            if not bn_id:
                return None

            existing = await self.client.get(self.collection, bn_id)
            existing_data = existing.to_dict()

            existing_data["archived"] = not existing_data.get("archived", False)
            existing_data["owui_updated_at"] = int(time.time())

            record = await self.client.update(self.collection, bn_id, existing_data)
            return ChatData.from_bluenexus(record)

        except Exception as e:
            log.error(f"[BlueNexus Chat] Error toggling archive {chat_id}: {e}")
            return None

    async def update_chat_folder_id_by_id_and_user_id(
        self,
        chat_id: str,
        user_id: str,
        folder_id: str,
    ) -> Optional[dict]:
        """Move chat to a folder."""
        try:
            bn_id = await self._get_bluenexus_id_by_owui_id(chat_id)
            if not bn_id:
                return None

            existing = await self.client.get(self.collection, bn_id)
            existing_data = existing.to_dict()

            # Verify ownership
            if existing_data.get("user_id") != user_id:
                return None

            existing_data["folder_id"] = folder_id
            existing_data["pinned"] = False
            existing_data["owui_updated_at"] = int(time.time())

            record = await self.client.update(self.collection, bn_id, existing_data)
            return ChatData.from_bluenexus(record)

        except Exception as e:
            log.error(f"[BlueNexus Chat] Error updating folder {chat_id}: {e}")
            return None

    async def update_chat_meta_by_id(
        self,
        chat_id: str,
        meta: dict,
    ) -> Optional[dict]:
        """Update chat metadata (tags, etc.)."""
        try:
            bn_id = await self._get_bluenexus_id_by_owui_id(chat_id)
            if not bn_id:
                return None

            existing = await self.client.get(self.collection, bn_id)
            existing_data = existing.to_dict()

            existing_data["meta"] = meta
            existing_data["owui_updated_at"] = int(time.time())

            record = await self.client.update(self.collection, bn_id, existing_data)
            return ChatData.from_bluenexus(record)

        except Exception as e:
            log.error(f"[BlueNexus Chat] Error updating meta {chat_id}: {e}")
            return None

    # =========================================================================
    # Delete Operations
    # =========================================================================

    async def delete_chat_by_id(self, chat_id: str) -> bool:
        """
        Delete a chat by ID.

        Args:
            chat_id: Chat ID (Open WebUI ID)

        Returns:
            True if deleted, False otherwise
        """
        try:
            bn_id = await self._get_bluenexus_id_by_owui_id(chat_id)
            if not bn_id:
                log.warning(f"[BlueNexus Chat] Chat not found for delete: {chat_id}")
                return False

            await self.client.delete(self.collection, bn_id)
            log.info(f"[BlueNexus Chat] Deleted chat {chat_id}")
            return True

        except Exception as e:
            log.error(f"[BlueNexus Chat] Error deleting chat {chat_id}: {e}")
            return False

    async def delete_chat_by_id_and_user_id(
        self,
        chat_id: str,
        user_id: str,
    ) -> bool:
        """Delete a chat with ownership verification."""
        chat = await self.get_chat_by_id(chat_id)
        if not chat or chat.get("user_id") != user_id:
            return False

        return await self.delete_chat_by_id(chat_id)

    async def delete_chats_by_user_id(self, user_id: str) -> bool:
        """Delete all chats for a user."""
        try:
            # Get all user's chats
            all_chats = await self.client.query_all(
                self.collection,
                filter={"user_id": user_id},
            )

            # Delete each one
            for record in all_chats:
                await self.client.delete(self.collection, record.id)

            log.info(f"[BlueNexus Chat] Deleted {len(all_chats)} chats for user {user_id}")
            return True

        except Exception as e:
            log.error(f"[BlueNexus Chat] Error deleting user chats: {e}")
            return False

    # =========================================================================
    # Bulk Operations
    # =========================================================================

    async def archive_all_chats_by_user_id(self, user_id: str) -> bool:
        """Archive all chats for a user."""
        try:
            all_chats = await self.client.query_all(
                self.collection,
                filter={"user_id": user_id, "archived": False},
            )

            for record in all_chats:
                data = record.to_dict()
                data["archived"] = True
                data["owui_updated_at"] = int(time.time())
                await self.client.update(self.collection, record.id, data)

            log.info(f"[BlueNexus Chat] Archived {len(all_chats)} chats for user {user_id}")
            return True

        except Exception as e:
            log.error(f"[BlueNexus Chat] Error archiving chats: {e}")
            return False

    async def unarchive_all_chats_by_user_id(self, user_id: str) -> bool:
        """Unarchive all chats for a user."""
        try:
            all_chats = await self.client.query_all(
                self.collection,
                filter={"user_id": user_id, "archived": True},
            )

            for record in all_chats:
                data = record.to_dict()
                data["archived"] = False
                data["owui_updated_at"] = int(time.time())
                await self.client.update(self.collection, record.id, data)

            log.info(f"[BlueNexus Chat] Unarchived {len(all_chats)} chats for user {user_id}")
            return True

        except Exception as e:
            log.error(f"[BlueNexus Chat] Error unarchiving chats: {e}")
            return False

    # =========================================================================
    # Count Operations
    # =========================================================================

    async def count_chats_by_user_id(
        self,
        user_id: str,
        include_archived: bool = False,
    ) -> int:
        """Count chats for a user."""
        filter_query = {"user_id": user_id}
        if not include_archived:
            filter_query["archived"] = False

        return await self.client.count(self.collection, filter_query)

    async def count_chats_by_folder_id_and_user_id(
        self,
        folder_id: str,
        user_id: str,
    ) -> int:
        """Count chats in a folder."""
        return await self.client.count(
            self.collection,
            {"user_id": user_id, "folder_id": folder_id},
        )
