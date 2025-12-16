"""
Chat Repository implementations for PostgreSQL and BlueNexus storage.
"""

import logging
import time
import uuid
from typing import Optional, List

from open_webui.repositories.base import BaseChatRepository
from open_webui.models.chats import (
    Chat,
    ChatModel,
    ChatForm,
    ChatResponse,
    ChatTitleIdResponse,
    Chats,
    SharedChatMappings,
)
from open_webui.internal.db import get_db
from open_webui.env import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("REPOSITORIES", logging.INFO))


class PostgresChatRepository(BaseChatRepository):
    """PostgreSQL implementation of chat repository using existing Chats model."""

    async def get_list(
        self,
        user_id: str,
        page: int = 1,
        limit: int = 60,
        include_archived: bool = False,
        include_pinned: bool = False,
        include_folders: bool = False,
    ) -> List[dict]:
        skip = (page - 1) * limit
        chats = Chats.get_chat_title_id_list_by_user_id(
            user_id,
            include_archived=include_archived,
            include_pinned=include_pinned,
            include_folders=include_folders,
            skip=skip,
            limit=limit,
        )
        return [chat.model_dump() for chat in chats]

    async def get_by_id(self, chat_id: str, user_id: str) -> Optional[dict]:
        chat = Chats.get_chat_by_id_and_user_id(chat_id, user_id)
        return chat.model_dump() if chat else None

    async def get_by_share_id(self, share_id: str) -> Optional[dict]:
        chat = Chats.get_chat_by_share_id(share_id)
        return chat.model_dump() if chat else None

    async def create(self, user_id: str, data: dict) -> dict:
        form = ChatForm(chat=data.get("chat", {}), folder_id=data.get("folder_id"))
        chat = Chats.insert_new_chat(user_id, form)
        return chat.model_dump() if chat else {}

    async def update(self, chat_id: str, user_id: str, data: dict) -> Optional[dict]:
        # Verify ownership first
        existing = Chats.get_chat_by_id_and_user_id(chat_id, user_id)
        if not existing:
            return None

        chat = Chats.update_chat_by_id(chat_id, data.get("chat", {}))
        return chat.model_dump() if chat else None

    async def delete(self, chat_id: str, user_id: str) -> bool:
        return Chats.delete_chat_by_id_and_user_id(chat_id, user_id)

    async def delete_all_by_user(self, user_id: str) -> bool:
        return Chats.delete_chats_by_user_id(user_id)

    async def share(self, chat_id: str, user_id: str) -> Optional[dict]:
        # PostgreSQL uses duplicate record approach
        chat = Chats.get_chat_by_id_and_user_id(chat_id, user_id)
        if not chat:
            return None

        shared = Chats.update_shared_chat_by_chat_id(chat_id)
        if shared:
            # Also create mapping for cross-storage compatibility
            if chat.share_id:
                SharedChatMappings.create_mapping(chat.share_id, user_id, chat_id)

        # Return the original chat with share_id
        chat = Chats.get_chat_by_id(chat_id)
        return chat.model_dump() if chat else None

    async def unshare(self, chat_id: str, user_id: str) -> bool:
        chat = Chats.get_chat_by_id_and_user_id(chat_id, user_id)
        if not chat or not chat.share_id:
            return False

        old_share_id = chat.share_id
        Chats.delete_shared_chat_by_chat_id(chat_id)
        Chats.update_chat_share_id_by_id(chat_id, None)
        SharedChatMappings.delete_mapping(old_share_id)
        return True

    async def archive(self, chat_id: str, user_id: str) -> Optional[dict]:
        """Toggle archived state - frontend expects this to toggle, not just archive."""
        chat = Chats.get_chat_by_id_and_user_id(chat_id, user_id)
        if not chat:
            return None
        chat = Chats.toggle_chat_archive_by_id(chat_id)
        return chat.model_dump() if chat else None

    async def unarchive(self, chat_id: str, user_id: str) -> Optional[dict]:
        """Explicitly unarchive - only unarchive if currently archived."""
        chat = Chats.get_chat_by_id_and_user_id(chat_id, user_id)
        if not chat:
            return None
        if chat.archived:
            chat = Chats.toggle_chat_archive_by_id(chat_id)
        return chat.model_dump() if chat else None

    async def pin(self, chat_id: str, user_id: str) -> Optional[dict]:
        """Toggle pinned state - frontend expects this to toggle, not just pin."""
        chat = Chats.get_chat_by_id_and_user_id(chat_id, user_id)
        if not chat:
            return None
        chat = Chats.toggle_chat_pinned_by_id(chat_id)
        return chat.model_dump() if chat else None

    async def unpin(self, chat_id: str, user_id: str) -> Optional[dict]:
        """Explicitly unpin - only unpin if currently pinned."""
        chat = Chats.get_chat_by_id_and_user_id(chat_id, user_id)
        if not chat:
            return None
        if chat.pinned:
            chat = Chats.toggle_chat_pinned_by_id(chat_id)
        return chat.model_dump() if chat else None

    async def clone(self, chat_id: str, user_id: str) -> Optional[dict]:
        chat = Chats.get_chat_by_id_and_user_id(chat_id, user_id)
        if not chat:
            return None

        # Create a cloned chat
        cloned_data = {
            "chat": {
                **chat.chat,
                "originalChatId": chat_id,
                "branchPointMessageId": chat.chat.get("history", {}).get("currentId"),
                "title": f"Clone of {chat.title}",
            },
            "folder_id": chat.folder_id,
        }
        return await self.create(user_id, cloned_data)

    async def get_archived(self, user_id: str, page: int = 1, limit: int = 60) -> List[dict]:
        skip = (page - 1) * limit
        chats = Chats.get_archived_chat_list_by_user_id(user_id, skip=skip, limit=limit)
        return [chat.model_dump() for chat in chats]

    async def get_pinned(self, user_id: str) -> List[dict]:
        chats = Chats.get_pinned_chats_by_user_id(user_id)
        return [chat.model_dump() for chat in chats]

    async def search(self, user_id: str, query: str, page: int = 1, limit: int = 60) -> List[dict]:
        skip = (page - 1) * limit
        chats = Chats.get_chats_by_user_id_and_search_text(
            user_id, query, include_archived=False, skip=skip, limit=limit
        )
        return [chat.model_dump() for chat in chats]

    # Additional methods for full endpoint support

    async def get_all(self) -> List[dict]:
        """Get all chats (admin only)."""
        chats = Chats.get_chats()
        return [chat.model_dump() for chat in chats]

    async def get_by_user_id_admin(self, user_id: str, page: int = 1, limit: int = 60, query: str = None) -> List[dict]:
        """Get chats for a specific user (admin only)."""
        skip = (page - 1) * limit
        if query:
            chats = Chats.get_chats_by_user_id_and_search_text(user_id, query, skip=skip, limit=limit)
        else:
            chats = Chats.get_chats_by_user_id(user_id, page, limit)
        return [chat.model_dump() for chat in chats]

    async def get_by_folder_id(self, user_id: str, folder_id: str, page: int = 1, limit: int = 60) -> List[dict]:
        """Get chats in a specific folder."""
        chats = Chats.get_chats_by_folder_id_and_user_id(folder_id, user_id, page, limit)
        return [chat.model_dump() for chat in chats]

    async def get_by_folder_ids(self, user_id: str, folder_ids: List[str]) -> List[dict]:
        """Get chats in multiple folders."""
        chats = Chats.get_chats_by_folder_ids_and_user_id(folder_ids, user_id)
        return [chat.model_dump() for chat in chats]

    async def archive_all(self, user_id: str) -> bool:
        """Archive all chats for a user."""
        Chats.archive_all_chats_by_user_id(user_id)
        return True

    async def unarchive_all(self, user_id: str) -> bool:
        """Unarchive all chats for a user."""
        Chats.unarchive_all_chats_by_user_id(user_id)
        return True

    async def get_by_tag(self, user_id: str, tag_name: str, skip: int = 0, limit: int = 50) -> List[dict]:
        """Get chats with a specific tag."""
        chats = Chats.get_chats_by_user_id_and_tag_name(user_id, tag_name, skip, limit)
        return [chat.model_dump() for chat in chats]

    async def update_folder_id(self, chat_id: str, user_id: str, folder_id: str) -> Optional[dict]:
        """Update chat folder."""
        chat = Chats.update_chat_folder_id_by_id_and_user_id(chat_id, user_id, folder_id)
        return chat.model_dump() if chat else None

    async def get_tags(self, chat_id: str, user_id: str) -> List[str]:
        """Get tags for a chat."""
        chat = Chats.get_chat_by_id_and_user_id(chat_id, user_id)
        if not chat:
            return []
        return chat.meta.get("tags", []) if chat.meta else []

    async def add_tag(self, chat_id: str, user_id: str, tag_name: str) -> List[str]:
        """Add a tag to a chat."""
        chat = Chats.get_chat_by_id_and_user_id(chat_id, user_id)
        if not chat:
            return []
        tags = chat.meta.get("tags", []) if chat.meta else []
        if tag_name not in tags:
            tags.append(tag_name)
            Chats.update_chat_tags_by_id_and_user_id(chat_id, user_id, tags)
        return tags

    async def remove_tag(self, chat_id: str, user_id: str, tag_name: str) -> List[str]:
        """Remove a tag from a chat."""
        chat = Chats.get_chat_by_id_and_user_id(chat_id, user_id)
        if not chat:
            return []
        tags = chat.meta.get("tags", []) if chat.meta else []
        if tag_name in tags:
            tags.remove(tag_name)
            Chats.update_chat_tags_by_id_and_user_id(chat_id, user_id, tags)
        return tags

    async def clear_tags(self, chat_id: str, user_id: str) -> bool:
        """Clear all tags from a chat."""
        chat = Chats.get_chat_by_id_and_user_id(chat_id, user_id)
        if not chat:
            return False
        Chats.update_chat_tags_by_id_and_user_id(chat_id, user_id, [])
        return True

    async def update_message(self, chat_id: str, user_id: str, message_id: str, content: str) -> Optional[dict]:
        """Update a message in a chat."""
        chat = Chats.get_chat_by_id(chat_id)
        if not chat:
            return None
        # Check access
        if chat.user_id != user_id:
            return None

        chat_content = chat.chat or {}
        history = chat_content.get("history", {})
        messages = history.get("messages", {})

        if message_id in messages:
            messages[message_id]["content"] = content
        else:
            messages[message_id] = {"content": content}

        history["messages"] = messages
        chat_content["history"] = history

        updated_chat = Chats.update_chat_by_id(chat_id, {"chat": chat_content})
        return updated_chat.model_dump() if updated_chat else None


class BlueNexusChatRepository(BaseChatRepository):
    """BlueNexus implementation of chat repository."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self._client = None

    def _get_client(self):
        """Get BlueNexus client for the user."""
        if self._client is None:
            from open_webui.utils.bluenexus.factory import get_bluenexus_client_for_user
            self._client = get_bluenexus_client_for_user(self.user_id)
        return self._client

    def _normalize_chat_data(self, chat_data: dict) -> dict:
        """Normalize BlueNexus data to Open WebUI format."""
        # Map owui_id to id
        if "owui_id" in chat_data:
            chat_data["id"] = chat_data.get("owui_id", chat_data.get("id"))

        # Handle timestamp conversion
        if "createdAt" in chat_data and chat_data["createdAt"]:
            created = chat_data["createdAt"]
            if hasattr(created, "timestamp"):
                chat_data["created_at"] = int(created.timestamp())
            elif isinstance(created, (int, float)):
                chat_data["created_at"] = int(created)
        if "created_at" not in chat_data:
            chat_data["created_at"] = int(time.time())

        if "updatedAt" in chat_data and chat_data["updatedAt"]:
            updated = chat_data["updatedAt"]
            if hasattr(updated, "timestamp"):
                chat_data["updated_at"] = int(updated.timestamp())
            elif isinstance(updated, (int, float)):
                chat_data["updated_at"] = int(updated)
        if "updated_at" not in chat_data:
            chat_data["updated_at"] = int(time.time())

        # Ensure chat object has proper history structure
        # Frontend expects history.messages to be a dict (object) that can be iterated
        chat_content = chat_data.get("chat", {})
        if not isinstance(chat_content, dict):
            chat_content = {}
            chat_data["chat"] = chat_content

        # Ensure history exists with proper structure
        if "history" not in chat_content or not isinstance(chat_content.get("history"), dict):
            chat_content["history"] = {"currentId": None, "messages": {}}
        else:
            # Ensure history.messages is a dict
            history = chat_content["history"]
            if "messages" not in history or not isinstance(history.get("messages"), dict):
                history["messages"] = {}
            if "currentId" not in history:
                history["currentId"] = None

        # Ensure messages array exists
        if "messages" not in chat_content or not isinstance(chat_content.get("messages"), list):
            chat_content["messages"] = []

        return chat_data

    async def get_list(
        self,
        user_id: str,
        page: int = 1,
        limit: int = 60,
        include_archived: bool = False,
        include_pinned: bool = False,
        include_folders: bool = False,
    ) -> List[dict]:
        from open_webui.utils.bluenexus.collections import Collections
        from open_webui.utils.bluenexus.types import QueryOptions, SortBy, SortOrder

        client = self._get_client()
        if not client:
            log.warning(f"[BlueNexusChatRepository.get_list] No client for user {user_id}, returning empty list")
            return []

        # Simplified filter - only filter by user_id and archived status
        # Don't filter by pinned/folder_id as these fields may not exist in all records
        filter_params = {"user_id": user_id}
        if not include_archived:
            filter_params["archived"] = False

        response = await client.query(
            Collections.CHATS,
            QueryOptions(
                filter=filter_params,
                sort_by=SortBy.UPDATED_AT,
                sort_order=SortOrder.DESC,
                limit=limit,
                page=page,
            )
        )

        chats = []
        for record in response.get_records():
            chat_data = record.model_dump()
            self._normalize_chat_data(chat_data)
            chats.append(chat_data)
        return chats

    async def get_by_id(self, chat_id: str, user_id: str) -> Optional[dict]:
        from open_webui.utils.bluenexus.collections import Collections
        from open_webui.utils.bluenexus.types import QueryOptions

        client = self._get_client()
        if not client:
            return None

        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"owui_id": chat_id, "user_id": user_id}, limit=1)
        )

        if response.data and len(response.data) > 0:
            record = response.get_records()[0]
            chat_data = record.model_dump()
            self._normalize_chat_data(chat_data)
            return chat_data
        return None

    async def get_by_share_id(self, share_id: str) -> Optional[dict]:
        from open_webui.utils.bluenexus.collections import Collections
        from open_webui.utils.bluenexus.types import QueryOptions
        from open_webui.utils.bluenexus.factory import get_bluenexus_client_for_user

        # Look up owner from mapping
        owner_user_id = SharedChatMappings.get_owner_user_id(share_id)

        if owner_user_id:
            client = get_bluenexus_client_for_user(owner_user_id)
        else:
            client = self._get_client()

        if not client:
            return None

        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"share_id": share_id}, limit=1)
        )

        if response.data and len(response.data) > 0:
            record = response.get_records()[0]
            chat_data = record.model_dump()
            self._normalize_chat_data(chat_data)
            return chat_data
        return None

    async def create(self, user_id: str, data: dict) -> dict:
        from open_webui.utils.bluenexus.collections import Collections

        client = self._get_client()
        if not client:
            return {}

        chat_content = data.get("chat", {})
        new_chat_id = str(uuid.uuid4())
        new_chat_data = {
            "owui_id": new_chat_id,
            "user_id": user_id,
            "title": chat_content.get("title", "New Chat"),
            "chat": chat_content,
            "meta": data.get("meta", {}),
            "pinned": data.get("pinned", False),
            "archived": False,
            "folder_id": data.get("folder_id"),
            "share_id": None,
        }

        record = await client.create(Collections.CHATS, new_chat_data)
        result_data = record.model_dump()
        self._normalize_chat_data(result_data)
        return result_data

    async def update(self, chat_id: str, user_id: str, data: dict) -> Optional[dict]:
        from open_webui.utils.bluenexus.collections import Collections
        from open_webui.utils.bluenexus.types import QueryOptions

        client = self._get_client()
        if not client:
            return None

        # Find existing chat
        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"owui_id": chat_id, "user_id": user_id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            return None

        record = response.get_records()[0]
        chat_data = record.model_dump()

        # Update fields - MERGE chat data to preserve history/messages
        if "chat" in data:
            existing_chat = chat_data.get("chat", {})
            incoming_chat = data["chat"]
            # Merge incoming chat into existing (preserves history, messages, etc.)
            if isinstance(existing_chat, dict) and isinstance(incoming_chat, dict):
                existing_chat.update(incoming_chat)
                chat_data["chat"] = existing_chat
            else:
                chat_data["chat"] = incoming_chat
            chat_data["title"] = chat_data["chat"].get("title", chat_data.get("title", "New Chat"))

        updated_record = await client.update(Collections.CHATS, record.id, chat_data)
        result_data = updated_record.model_dump()
        self._normalize_chat_data(result_data)
        return result_data

    async def delete(self, chat_id: str, user_id: str) -> bool:
        from open_webui.utils.bluenexus.collections import Collections
        from open_webui.utils.bluenexus.types import QueryOptions

        client = self._get_client()
        if not client:
            return False

        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"owui_id": chat_id, "user_id": user_id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            return False

        record = response.get_records()[0]
        await client.delete(Collections.CHATS, record.id)
        return True

    async def delete_all_by_user(self, user_id: str) -> bool:
        from open_webui.utils.bluenexus.collections import Collections
        from open_webui.utils.bluenexus.types import QueryOptions

        client = self._get_client()
        if not client:
            return False

        while True:
            response = await client.query(
                Collections.CHATS,
                QueryOptions(filter={"user_id": user_id}, limit=100)
            )

            if not response.data or len(response.data) == 0:
                break

            for record in response.get_records():
                await client.delete(Collections.CHATS, record.id)

        return True

    async def share(self, chat_id: str, user_id: str) -> Optional[dict]:
        from open_webui.utils.bluenexus.collections import Collections
        from open_webui.utils.bluenexus.types import QueryOptions

        client = self._get_client()
        if not client:
            return None

        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"owui_id": chat_id, "user_id": user_id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            return None

        record = response.get_records()[0]
        chat_data = record.model_dump()

        # Generate share_id if not exists
        if not chat_data.get("share_id"):
            chat_data["share_id"] = str(uuid.uuid4())

        updated_record = await client.update(Collections.CHATS, record.id, chat_data)

        # Create mapping for other users to access
        SharedChatMappings.create_mapping(
            share_id=chat_data["share_id"],
            owner_user_id=user_id,
            chat_id=chat_id,
        )

        result_data = updated_record.model_dump()
        self._normalize_chat_data(result_data)
        return result_data

    async def unshare(self, chat_id: str, user_id: str) -> bool:
        from open_webui.utils.bluenexus.collections import Collections
        from open_webui.utils.bluenexus.types import QueryOptions

        client = self._get_client()
        if not client:
            return False

        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"owui_id": chat_id, "user_id": user_id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            return False

        record = response.get_records()[0]
        chat_data = record.model_dump()

        old_share_id = chat_data.get("share_id")
        if not old_share_id:
            return False

        chat_data["share_id"] = None
        await client.update(Collections.CHATS, record.id, chat_data)

        # Delete mapping
        SharedChatMappings.delete_mapping(old_share_id)

        return True

    async def archive(self, chat_id: str, user_id: str) -> Optional[dict]:
        """Toggle archived state - frontend expects this to toggle, not just archive."""
        from open_webui.utils.bluenexus.collections import Collections
        from open_webui.utils.bluenexus.types import QueryOptions

        client = self._get_client()
        if not client:
            return None

        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"owui_id": chat_id, "user_id": user_id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            return None

        record = response.get_records()[0]
        chat_data = record.model_dump()
        # Toggle the archived state
        chat_data["archived"] = not chat_data.get("archived", False)

        updated_record = await client.update(Collections.CHATS, record.id, chat_data)
        result_data = updated_record.model_dump()
        self._normalize_chat_data(result_data)
        return result_data

    async def unarchive(self, chat_id: str, user_id: str) -> Optional[dict]:
        """Explicitly unarchive - only unarchive if currently archived."""
        return await self._set_field(chat_id, user_id, "archived", False)

    async def pin(self, chat_id: str, user_id: str) -> Optional[dict]:
        """Toggle pinned state - frontend expects this to toggle, not just pin."""
        from open_webui.utils.bluenexus.collections import Collections
        from open_webui.utils.bluenexus.types import QueryOptions

        client = self._get_client()
        if not client:
            return None

        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"owui_id": chat_id, "user_id": user_id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            return None

        record = response.get_records()[0]
        chat_data = record.model_dump()
        # Toggle the pinned state
        chat_data["pinned"] = not chat_data.get("pinned", False)

        updated_record = await client.update(Collections.CHATS, record.id, chat_data)
        result_data = updated_record.model_dump()
        self._normalize_chat_data(result_data)
        return result_data

    async def unpin(self, chat_id: str, user_id: str) -> Optional[dict]:
        """Explicitly unpin - only unpin if currently pinned."""
        return await self._set_field(chat_id, user_id, "pinned", False)

    async def _set_field(self, chat_id: str, user_id: str, field: str, value: bool) -> Optional[dict]:
        from open_webui.utils.bluenexus.collections import Collections
        from open_webui.utils.bluenexus.types import QueryOptions

        client = self._get_client()
        if not client:
            return None

        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"owui_id": chat_id, "user_id": user_id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            return None

        record = response.get_records()[0]
        chat_data = record.model_dump()
        chat_data[field] = value

        updated_record = await client.update(Collections.CHATS, record.id, chat_data)
        result_data = updated_record.model_dump()
        self._normalize_chat_data(result_data)
        return result_data

    async def clone(self, chat_id: str, user_id: str) -> Optional[dict]:
        chat = await self.get_by_id(chat_id, user_id)
        if not chat:
            return None

        cloned_data = {
            "chat": {
                **chat.get("chat", {}),
                "originalChatId": chat_id,
                "branchPointMessageId": chat.get("chat", {}).get("history", {}).get("currentId"),
                "title": f"Clone of {chat.get('title', 'Chat')}",
            },
            "meta": chat.get("meta", {}),
            "folder_id": chat.get("folder_id"),
        }
        return await self.create(user_id, cloned_data)

    async def get_archived(self, user_id: str, page: int = 1, limit: int = 60) -> List[dict]:
        from open_webui.utils.bluenexus.collections import Collections
        from open_webui.utils.bluenexus.types import QueryOptions, SortBy, SortOrder

        client = self._get_client()
        if not client:
            return []

        response = await client.query(
            Collections.CHATS,
            QueryOptions(
                filter={"user_id": user_id, "archived": True},
                sort_by=SortBy.UPDATED_AT,
                sort_order=SortOrder.DESC,
                limit=limit,
                page=page,
            )
        )

        chats = []
        for record in response.get_records():
            chat_data = record.model_dump()
            self._normalize_chat_data(chat_data)
            chats.append(chat_data)
        return chats

    async def get_pinned(self, user_id: str) -> List[dict]:
        from open_webui.utils.bluenexus.collections import Collections
        from open_webui.utils.bluenexus.types import QueryOptions, SortBy, SortOrder

        client = self._get_client()
        if not client:
            return []

        response = await client.query(
            Collections.CHATS,
            QueryOptions(
                filter={"user_id": user_id, "pinned": True, "archived": False},
                sort_by=SortBy.UPDATED_AT,
                sort_order=SortOrder.DESC,
            )
        )

        chats = []
        for record in response.get_records():
            chat_data = record.model_dump()
            self._normalize_chat_data(chat_data)
            chats.append(chat_data)
        return chats

    async def search(self, user_id: str, query: str, page: int = 1, limit: int = 60) -> List[dict]:
        from open_webui.utils.bluenexus.collections import Collections
        from open_webui.utils.bluenexus.types import QueryOptions, SortBy, SortOrder

        client = self._get_client()
        if not client:
            return []

        # Parse special search prefixes like PostgreSQL does
        query = query.replace("\u0000", "").lower().strip()
        if not query:
            return await self.get_list(user_id, page, limit, include_archived=False)

        search_words = query.split(" ")

        # Extract tags
        tag_ids = [
            word.replace("tag:", "").replace(" ", "_").lower()
            for word in search_words
            if word.startswith("tag:")
        ]

        # Extract pinned filter
        is_pinned = None
        if "pinned:true" in search_words:
            is_pinned = True
        elif "pinned:false" in search_words:
            is_pinned = False

        # Extract archived filter
        is_archived = None
        if "archived:true" in search_words:
            is_archived = True
        elif "archived:false" in search_words:
            is_archived = False

        # Remove special keywords from search text
        title_search = " ".join([
            word for word in search_words
            if not word.startswith("tag:")
            and not word.startswith("folder:")
            and word not in ["pinned:true", "pinned:false", "archived:true", "archived:false"]
        ]).strip()

        # Build filter - BlueNexus doesn't support $contains, so we filter client-side
        filter_dict = {"user_id": user_id}

        if is_archived is not None:
            filter_dict["archived"] = is_archived
        else:
            filter_dict["archived"] = False

        if is_pinned is not None:
            filter_dict["pinned"] = is_pinned

        # Get more results for client-side title filtering (max 100 per QueryOptions)
        fetch_limit = min(limit * 3, 100) if title_search else min(limit, 100)

        response = await client.query(
            Collections.CHATS,
            QueryOptions(
                filter=filter_dict,
                sort_by=SortBy.UPDATED_AT,
                sort_order=SortOrder.DESC,
                limit=fetch_limit,
                page=page,
            )
        )

        chats = []
        for record in response.get_records():
            chat_data = record.model_dump()
            self._normalize_chat_data(chat_data)

            # Client-side tag filtering (if tag search)
            if tag_ids:
                chat_tags = chat_data.get("meta", {}).get("tags", [])
                # Normalize tag format - could be list of strings or list of dicts
                if chat_tags and isinstance(chat_tags[0], dict):
                    chat_tag_names = [t.get("name", "").lower() for t in chat_tags]
                else:
                    chat_tag_names = [str(t).lower() for t in chat_tags]

                if not any(tag_id in chat_tag_names for tag_id in tag_ids):
                    continue

            # Client-side title filtering (BlueNexus doesn't support $contains)
            if title_search:
                title = chat_data.get("title", "").lower()
                if title_search not in title:
                    continue

            chats.append(chat_data)

            # Stop once we have enough results
            if len(chats) >= limit:
                break

        return chats

    # Additional methods for full endpoint support

    async def get_all(self) -> List[dict]:
        """Get all chats (admin only) - BlueNexus doesn't support cross-user queries."""
        # BlueNexus is per-user, so admin export falls back to PostgreSQL
        log.warning("[BlueNexusChatRepository.get_all] Admin export not supported in BlueNexus mode")
        return []

    async def get_by_user_id_admin(self, user_id: str, page: int = 1, limit: int = 60, query: str = None) -> List[dict]:
        """Get chats for a specific user (admin only)."""
        # For admin access to other users, we need a special client
        from open_webui.utils.bluenexus.factory import get_bluenexus_client_for_user
        client = get_bluenexus_client_for_user(user_id)
        if not client:
            return []

        from open_webui.utils.bluenexus.collections import Collections
        from open_webui.utils.bluenexus.types import QueryOptions, SortBy, SortOrder

        filter_params = {"user_id": user_id}
        if query:
            filter_params["title"] = {"$contains": query}

        response = await client.query(
            Collections.CHATS,
            QueryOptions(
                filter=filter_params,
                sort_by=SortBy.UPDATED_AT,
                sort_order=SortOrder.DESC,
                limit=limit,
                page=page,
            )
        )

        chats = []
        for record in response.get_records():
            chat_data = record.model_dump()
            self._normalize_chat_data(chat_data)
            chats.append(chat_data)
        return chats

    async def get_by_folder_id(self, user_id: str, folder_id: str, page: int = 1, limit: int = 60) -> List[dict]:
        """Get chats in a specific folder."""
        from open_webui.utils.bluenexus.collections import Collections
        from open_webui.utils.bluenexus.types import QueryOptions, SortBy, SortOrder

        client = self._get_client()
        if not client:
            return []

        response = await client.query(
            Collections.CHATS,
            QueryOptions(
                filter={"user_id": user_id, "folder_id": folder_id},
                sort_by=SortBy.UPDATED_AT,
                sort_order=SortOrder.DESC,
                limit=limit,
                page=page,
            )
        )

        chats = []
        for record in response.get_records():
            chat_data = record.model_dump()
            self._normalize_chat_data(chat_data)
            chats.append(chat_data)
        return chats

    async def get_by_folder_ids(self, user_id: str, folder_ids: List[str]) -> List[dict]:
        """Get chats in multiple folders."""
        from open_webui.utils.bluenexus.collections import Collections
        from open_webui.utils.bluenexus.types import QueryOptions, SortBy, SortOrder

        client = self._get_client()
        if not client:
            return []

        response = await client.query(
            Collections.CHATS,
            QueryOptions(
                filter={"user_id": user_id, "folder_id": {"$in": folder_ids}},
                sort_by=SortBy.UPDATED_AT,
                sort_order=SortOrder.DESC,
                limit=100,
            )
        )

        chats = []
        for record in response.get_records():
            chat_data = record.model_dump()
            self._normalize_chat_data(chat_data)
            chats.append(chat_data)
        return chats

    async def archive_all(self, user_id: str) -> bool:
        """Archive all chats for a user."""
        from open_webui.utils.bluenexus.collections import Collections
        from open_webui.utils.bluenexus.types import QueryOptions

        client = self._get_client()
        if not client:
            return False

        while True:
            response = await client.query(
                Collections.CHATS,
                QueryOptions(filter={"user_id": user_id, "archived": False}, limit=100)
            )

            if not response.data or len(response.data) == 0:
                break

            for record in response.get_records():
                chat_data = record.model_dump()
                chat_data["archived"] = True
                await client.update(Collections.CHATS, record.id, chat_data)

        return True

    async def unarchive_all(self, user_id: str) -> bool:
        """Unarchive all chats for a user."""
        from open_webui.utils.bluenexus.collections import Collections
        from open_webui.utils.bluenexus.types import QueryOptions

        client = self._get_client()
        if not client:
            return False

        while True:
            response = await client.query(
                Collections.CHATS,
                QueryOptions(filter={"user_id": user_id, "archived": True}, limit=100)
            )

            if not response.data or len(response.data) == 0:
                break

            for record in response.get_records():
                chat_data = record.model_dump()
                chat_data["archived"] = False
                await client.update(Collections.CHATS, record.id, chat_data)

        return True

    async def get_by_tag(self, user_id: str, tag_name: str, skip: int = 0, limit: int = 50) -> List[dict]:
        """Get chats with a specific tag."""
        from open_webui.utils.bluenexus.collections import Collections
        from open_webui.utils.bluenexus.types import QueryOptions, SortBy, SortOrder

        client = self._get_client()
        if not client:
            return []

        # Query chats that have the tag in meta.tags array
        response = await client.query(
            Collections.CHATS,
            QueryOptions(
                filter={"user_id": user_id, "meta.tags": {"$contains": tag_name}},
                sort_by=SortBy.UPDATED_AT,
                sort_order=SortOrder.DESC,
                limit=limit,
            )
        )

        chats = []
        for record in response.get_records():
            chat_data = record.model_dump()
            self._normalize_chat_data(chat_data)
            chats.append(chat_data)
        return chats[skip:] if skip > 0 else chats

    async def update_folder_id(self, chat_id: str, user_id: str, folder_id: str) -> Optional[dict]:
        """Update chat folder."""
        from open_webui.utils.bluenexus.collections import Collections
        from open_webui.utils.bluenexus.types import QueryOptions

        client = self._get_client()
        if not client:
            return None

        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"owui_id": chat_id, "user_id": user_id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            return None

        record = response.get_records()[0]
        chat_data = record.model_dump()
        chat_data["folder_id"] = folder_id

        updated_record = await client.update(Collections.CHATS, record.id, chat_data)
        result_data = updated_record.model_dump()
        self._normalize_chat_data(result_data)
        return result_data

    async def get_tags(self, chat_id: str, user_id: str) -> List[str]:
        """Get tags for a chat."""
        chat = await self.get_by_id(chat_id, user_id)
        if not chat:
            return []
        return chat.get("meta", {}).get("tags", [])

    async def add_tag(self, chat_id: str, user_id: str, tag_name: str) -> List[str]:
        """Add a tag to a chat."""
        from open_webui.utils.bluenexus.collections import Collections
        from open_webui.utils.bluenexus.types import QueryOptions

        client = self._get_client()
        if not client:
            return []

        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"owui_id": chat_id, "user_id": user_id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            return []

        record = response.get_records()[0]
        chat_data = record.model_dump()

        meta = chat_data.get("meta", {})
        tags = meta.get("tags", [])
        if tag_name not in tags:
            tags.append(tag_name)
            meta["tags"] = tags
            chat_data["meta"] = meta
            await client.update(Collections.CHATS, record.id, chat_data)

        return tags

    async def remove_tag(self, chat_id: str, user_id: str, tag_name: str) -> List[str]:
        """Remove a tag from a chat."""
        from open_webui.utils.bluenexus.collections import Collections
        from open_webui.utils.bluenexus.types import QueryOptions

        client = self._get_client()
        if not client:
            return []

        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"owui_id": chat_id, "user_id": user_id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            return []

        record = response.get_records()[0]
        chat_data = record.model_dump()

        meta = chat_data.get("meta", {})
        tags = meta.get("tags", [])
        if tag_name in tags:
            tags.remove(tag_name)
            meta["tags"] = tags
            chat_data["meta"] = meta
            await client.update(Collections.CHATS, record.id, chat_data)

        return tags

    async def clear_tags(self, chat_id: str, user_id: str) -> bool:
        """Clear all tags from a chat."""
        from open_webui.utils.bluenexus.collections import Collections
        from open_webui.utils.bluenexus.types import QueryOptions

        client = self._get_client()
        if not client:
            return False

        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"owui_id": chat_id, "user_id": user_id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            return False

        record = response.get_records()[0]
        chat_data = record.model_dump()

        meta = chat_data.get("meta", {})
        meta["tags"] = []
        chat_data["meta"] = meta
        await client.update(Collections.CHATS, record.id, chat_data)

        return True

    async def update_message(self, chat_id: str, user_id: str, message_id: str, content: str) -> Optional[dict]:
        """Update a message in a chat."""
        from open_webui.utils.bluenexus.collections import Collections
        from open_webui.utils.bluenexus.types import QueryOptions

        client = self._get_client()
        if not client:
            return None

        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"owui_id": chat_id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            return None

        record = response.get_records()[0]
        chat_data = record.model_dump()

        # Check access
        if chat_data.get("user_id") != user_id:
            return None

        chat_content = chat_data.get("chat", {})
        history = chat_content.get("history", {})
        messages = history.get("messages", {})

        if message_id in messages:
            messages[message_id]["content"] = content
        else:
            messages[message_id] = {"content": content}

        history["messages"] = messages
        chat_content["history"] = history
        chat_data["chat"] = chat_content

        updated_record = await client.update(Collections.CHATS, record.id, chat_data)
        result_data = updated_record.model_dump()
        self._normalize_chat_data(result_data)
        return result_data