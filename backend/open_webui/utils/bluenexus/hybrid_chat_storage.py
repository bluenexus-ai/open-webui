"""
Hybrid Chat Storage

This module provides a hybrid storage facade that automatically switches
between local database storage and BlueNexus cloud storage based on user
authentication status.

For users logged in via BlueNexus OAuth, chats are stored in BlueNexus.
For all other users, chats are stored in the local database.
"""

import logging
from typing import Optional

from open_webui.env import SRC_LOG_LEVELS
from open_webui.models.chats import ChatModel, ChatForm, ChatImportForm, Chats
from open_webui.utils.bluenexus.factory import (
    get_bluenexus_client_for_user,
    has_bluenexus_session,
)
from open_webui.utils.bluenexus.chat_storage import BlueNexusChatStorage

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("BLUENEXUS", SRC_LOG_LEVELS.get("MAIN", logging.INFO)))


def _dict_to_chat_model(data: dict) -> Optional[ChatModel]:
    """Convert dict to ChatModel, filtering out internal fields."""
    if not data:
        return None
    # Remove internal fields before validation
    clean_data = {k: v for k, v in data.items() if not k.startswith("_")}
    return ChatModel.model_validate(clean_data)


class HybridChatStorage:
    """
    Hybrid storage facade for chats.

    This class provides a unified interface for chat operations that
    automatically routes to either BlueNexus or local storage based
    on the user's authentication method.

    Usage:
        storage = HybridChatStorage()

        # Create a chat - automatically uses correct backend
        chat = await storage.insert_new_chat(user_id, form_data)

        # Get chats - automatically uses correct backend
        chats = await storage.get_chats_by_user_id(user_id)
    """

    def __init__(self):
        """Initialize hybrid storage."""
        self._local_storage = Chats

    def _get_bluenexus_storage(self, user_id: str) -> Optional[BlueNexusChatStorage]:
        """
        Get BlueNexus storage for a user if they have a valid session.

        Args:
            user_id: User ID to check

        Returns:
            BlueNexusChatStorage if user has BlueNexus session, None otherwise
        """
        log.info(f"[Hybrid Chat Storage] Checking BlueNexus session for user={user_id}")
        client = get_bluenexus_client_for_user(user_id)
        if client:
            log.info(f"[Hybrid Chat Storage] User {user_id} has BlueNexus session, using BlueNexus storage")
            return BlueNexusChatStorage(client)
        log.info(f"[Hybrid Chat Storage] User {user_id} has no BlueNexus session, using local storage")
        return None

    def _uses_bluenexus(self, user_id: str) -> bool:
        """Check if a user should use BlueNexus storage."""
        uses_bn = has_bluenexus_session(user_id)
        log.debug(f"[Hybrid Chat Storage] User {user_id} uses_bluenexus={uses_bn}")
        return uses_bn

    # =========================================================================
    # Create Operations
    # =========================================================================

    async def insert_new_chat(
        self,
        user_id: str,
        form_data: ChatForm,
    ) -> Optional[ChatModel]:
        """
        Create a new chat.

        Routes to BlueNexus or local storage based on user's auth.
        """
        bn_storage = self._get_bluenexus_storage(user_id)
        if bn_storage:
            log.debug(f"[Hybrid Chat] Using BlueNexus for insert_new_chat user={user_id}")
            result = await bn_storage.insert_new_chat(
                user_id=user_id,
                chat_data=form_data.chat,
                folder_id=form_data.folder_id,
            )
            return _dict_to_chat_model(result)

        log.debug(f"[Hybrid Chat] Using local DB for insert_new_chat user={user_id}")
        return self._local_storage.insert_new_chat(user_id, form_data)

    async def import_chat(
        self,
        user_id: str,
        form_data: ChatImportForm,
    ) -> Optional[ChatModel]:
        """Import an existing chat."""
        bn_storage = self._get_bluenexus_storage(user_id)
        if bn_storage:
            log.debug(f"[Hybrid Chat] Using BlueNexus for import_chat user={user_id}")
            result = await bn_storage.import_chat(
                user_id=user_id,
                chat_data=form_data.chat,
                meta=form_data.meta,
                pinned=form_data.pinned,
                folder_id=form_data.folder_id,
                created_at=form_data.created_at,
                updated_at=form_data.updated_at,
            )
            return _dict_to_chat_model(result)

        log.debug(f"[Hybrid Chat] Using local DB for import_chat user={user_id}")
        return self._local_storage.import_chat(user_id, form_data)

    # =========================================================================
    # Read Operations
    # =========================================================================

    async def get_chat_by_id(self, chat_id: str, user_id: str) -> Optional[ChatModel]:
        """
        Get a chat by ID.

        Note: We need user_id to determine which storage to use.
        """
        bn_storage = self._get_bluenexus_storage(user_id)
        if bn_storage:
            result = await bn_storage.get_chat_by_id(chat_id)
            return _dict_to_chat_model(result)

        return self._local_storage.get_chat_by_id(chat_id)

    async def get_chat_by_id_and_user_id(
        self,
        chat_id: str,
        user_id: str,
    ) -> Optional[ChatModel]:
        """Get a chat by ID with user ownership verification."""
        bn_storage = self._get_bluenexus_storage(user_id)
        if bn_storage:
            result = await bn_storage.get_chat_by_id_and_user_id(chat_id, user_id)
            return _dict_to_chat_model(result)

        return self._local_storage.get_chat_by_id_and_user_id(chat_id, user_id)

    async def get_chats_by_user_id(self, user_id: str) -> list[ChatModel]:
        """Get all chats for a user."""
        bn_storage = self._get_bluenexus_storage(user_id)
        if bn_storage:
            results = await bn_storage.get_chats_by_user_id(user_id)
            return [_dict_to_chat_model(r) for r in results if r]

        return self._local_storage.get_chats_by_user_id(user_id)

    async def get_chat_list_by_user_id(
        self,
        user_id: str,
        include_archived: bool = False,
        filter: Optional[dict] = None,
        skip: int = 0,
        limit: int = 50,
    ) -> list[ChatModel]:
        """Get paginated chat list for a user."""
        bn_storage = self._get_bluenexus_storage(user_id)
        if bn_storage:
            results = await bn_storage.get_chat_list_by_user_id(
                user_id=user_id,
                include_archived=include_archived,
                filter=filter,
                skip=skip,
                limit=limit,
            )
            return [_dict_to_chat_model(r) for r in results if r]

        return self._local_storage.get_chat_list_by_user_id(
            user_id, include_archived, filter, skip, limit
        )

    async def get_pinned_chats_by_user_id(self, user_id: str) -> list[ChatModel]:
        """Get pinned chats for a user."""
        bn_storage = self._get_bluenexus_storage(user_id)
        if bn_storage:
            results = await bn_storage.get_pinned_chats_by_user_id(user_id)
            return [_dict_to_chat_model(r) for r in results if r]

        return self._local_storage.get_pinned_chats_by_user_id(user_id)

    async def get_archived_chats_by_user_id(self, user_id: str) -> list[ChatModel]:
        """Get archived chats for a user."""
        bn_storage = self._get_bluenexus_storage(user_id)
        if bn_storage:
            results = await bn_storage.get_archived_chats_by_user_id(user_id)
            return [_dict_to_chat_model(r) for r in results if r]

        return self._local_storage.get_archived_chats_by_user_id(user_id)

    async def get_chats_by_folder_id_and_user_id(
        self,
        folder_id: str,
        user_id: str,
        skip: int = 0,
        limit: int = 60,
    ) -> list[ChatModel]:
        """Get chats in a specific folder."""
        bn_storage = self._get_bluenexus_storage(user_id)
        if bn_storage:
            results = await bn_storage.get_chats_by_folder_id_and_user_id(
                folder_id, user_id, skip, limit
            )
            return [_dict_to_chat_model(r) for r in results if r]

        return self._local_storage.get_chats_by_folder_id_and_user_id(
            folder_id, user_id, skip, limit
        )

    # =========================================================================
    # Update Operations
    # =========================================================================

    async def update_chat_by_id(
        self,
        chat_id: str,
        chat: dict,
        user_id: str,
    ) -> Optional[ChatModel]:
        """
        Update chat content.

        Note: user_id is needed to determine storage backend.
        """
        bn_storage = self._get_bluenexus_storage(user_id)
        if bn_storage:
            result = await bn_storage.update_chat_by_id(chat_id, chat)
            return _dict_to_chat_model(result)

        return self._local_storage.update_chat_by_id(chat_id, chat)

    async def update_chat_title_by_id(
        self,
        chat_id: str,
        title: str,
        user_id: str,
    ) -> Optional[ChatModel]:
        """Update chat title."""
        bn_storage = self._get_bluenexus_storage(user_id)
        if bn_storage:
            result = await bn_storage.update_chat_title_by_id(chat_id, title)
            return _dict_to_chat_model(result)

        return self._local_storage.update_chat_title_by_id(chat_id, title)

    async def toggle_chat_pinned_by_id(
        self,
        chat_id: str,
        user_id: str,
    ) -> Optional[ChatModel]:
        """Toggle chat pinned status."""
        bn_storage = self._get_bluenexus_storage(user_id)
        if bn_storage:
            result = await bn_storage.toggle_chat_pinned_by_id(chat_id)
            return _dict_to_chat_model(result)

        return self._local_storage.toggle_chat_pinned_by_id(chat_id)

    async def toggle_chat_archive_by_id(
        self,
        chat_id: str,
        user_id: str,
    ) -> Optional[ChatModel]:
        """Toggle chat archived status."""
        bn_storage = self._get_bluenexus_storage(user_id)
        if bn_storage:
            result = await bn_storage.toggle_chat_archive_by_id(chat_id)
            return _dict_to_chat_model(result)

        return self._local_storage.toggle_chat_archive_by_id(chat_id)

    async def update_chat_folder_id_by_id_and_user_id(
        self,
        chat_id: str,
        user_id: str,
        folder_id: str,
    ) -> Optional[ChatModel]:
        """Move chat to a folder."""
        bn_storage = self._get_bluenexus_storage(user_id)
        if bn_storage:
            result = await bn_storage.update_chat_folder_id_by_id_and_user_id(
                chat_id, user_id, folder_id
            )
            return _dict_to_chat_model(result)

        return self._local_storage.update_chat_folder_id_by_id_and_user_id(
            chat_id, user_id, folder_id
        )

    async def update_chat_meta_by_id(
        self,
        chat_id: str,
        meta: dict,
        user_id: str,
    ) -> Optional[ChatModel]:
        """Update chat metadata."""
        bn_storage = self._get_bluenexus_storage(user_id)
        if bn_storage:
            result = await bn_storage.update_chat_meta_by_id(chat_id, meta)
            return _dict_to_chat_model(result)

        # Local storage doesn't have this method directly, update via chat update
        chat = self._local_storage.get_chat_by_id(chat_id)
        if chat:
            chat_dict = chat.chat
            # Meta is stored separately in local DB, need direct update
            from open_webui.internal.db import get_db
            from open_webui.models.chats import Chat
            with get_db() as db:
                chat_obj = db.get(Chat, chat_id)
                if chat_obj:
                    chat_obj.meta = meta
                    db.commit()
                    db.refresh(chat_obj)
                    return ChatModel.model_validate(chat_obj)
        return None

    # =========================================================================
    # Delete Operations
    # =========================================================================

    async def delete_chat_by_id(self, chat_id: str, user_id: str) -> bool:
        """Delete a chat by ID."""
        bn_storage = self._get_bluenexus_storage(user_id)
        if bn_storage:
            return await bn_storage.delete_chat_by_id(chat_id)

        return self._local_storage.delete_chat_by_id(chat_id)

    async def delete_chat_by_id_and_user_id(
        self,
        chat_id: str,
        user_id: str,
    ) -> bool:
        """Delete a chat with ownership verification."""
        bn_storage = self._get_bluenexus_storage(user_id)
        if bn_storage:
            return await bn_storage.delete_chat_by_id_and_user_id(chat_id, user_id)

        return self._local_storage.delete_chat_by_id_and_user_id(chat_id, user_id)

    async def delete_chats_by_user_id(self, user_id: str) -> bool:
        """Delete all chats for a user."""
        bn_storage = self._get_bluenexus_storage(user_id)
        if bn_storage:
            return await bn_storage.delete_chats_by_user_id(user_id)

        return self._local_storage.delete_chats_by_user_id(user_id)

    # =========================================================================
    # Bulk Operations
    # =========================================================================

    async def archive_all_chats_by_user_id(self, user_id: str) -> bool:
        """Archive all chats for a user."""
        bn_storage = self._get_bluenexus_storage(user_id)
        if bn_storage:
            return await bn_storage.archive_all_chats_by_user_id(user_id)

        return self._local_storage.archive_all_chats_by_user_id(user_id)

    async def unarchive_all_chats_by_user_id(self, user_id: str) -> bool:
        """Unarchive all chats for a user."""
        bn_storage = self._get_bluenexus_storage(user_id)
        if bn_storage:
            return await bn_storage.unarchive_all_chats_by_user_id(user_id)

        return self._local_storage.unarchive_all_chats_by_user_id(user_id)

    # =========================================================================
    # Count Operations
    # =========================================================================

    async def count_chats_by_user_id(
        self,
        user_id: str,
        include_archived: bool = False,
    ) -> int:
        """Count chats for a user."""
        bn_storage = self._get_bluenexus_storage(user_id)
        if bn_storage:
            return await bn_storage.count_chats_by_user_id(user_id, include_archived)

        # Local storage doesn't have count method, use len
        chats = self._local_storage.get_chats_by_user_id(user_id)
        if include_archived:
            return len(chats)
        return len([c for c in chats if not c.archived])

    async def count_chats_by_folder_id_and_user_id(
        self,
        folder_id: str,
        user_id: str,
    ) -> int:
        """Count chats in a folder."""
        bn_storage = self._get_bluenexus_storage(user_id)
        if bn_storage:
            return await bn_storage.count_chats_by_folder_id_and_user_id(
                folder_id, user_id
            )

        return self._local_storage.count_chats_by_folder_id_and_user_id(
            folder_id, user_id
        )


# Global instance
HybridChats = HybridChatStorage()
