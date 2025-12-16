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
        chat = Chats.get_chat_by_id_and_user_id(chat_id, user_id)
        if not chat:
            return None
        if not chat.archived:
            chat = Chats.toggle_chat_archive_by_id(chat_id)
        return chat.model_dump() if chat else None

    async def unarchive(self, chat_id: str, user_id: str) -> Optional[dict]:
        chat = Chats.get_chat_by_id_and_user_id(chat_id, user_id)
        if not chat:
            return None
        if chat.archived:
            chat = Chats.toggle_chat_archive_by_id(chat_id)
        return chat.model_dump() if chat else None

    async def pin(self, chat_id: str, user_id: str) -> Optional[dict]:
        chat = Chats.get_chat_by_id_and_user_id(chat_id, user_id)
        if not chat:
            return None
        if not chat.pinned:
            chat = Chats.toggle_chat_pinned_by_id(chat_id)
        return chat.model_dump() if chat else None

    async def unpin(self, chat_id: str, user_id: str) -> Optional[dict]:
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

        # Update fields
        if "chat" in data:
            chat_data["chat"] = data["chat"]
            chat_data["title"] = data["chat"].get("title", chat_data.get("title", "New Chat"))

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
        return await self._toggle_field(chat_id, user_id, "archived", True)

    async def unarchive(self, chat_id: str, user_id: str) -> Optional[dict]:
        return await self._toggle_field(chat_id, user_id, "archived", False)

    async def pin(self, chat_id: str, user_id: str) -> Optional[dict]:
        return await self._toggle_field(chat_id, user_id, "pinned", True)

    async def unpin(self, chat_id: str, user_id: str) -> Optional[dict]:
        return await self._toggle_field(chat_id, user_id, "pinned", False)

    async def _toggle_field(self, chat_id: str, user_id: str, field: str, value: bool) -> Optional[dict]:
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

        # BlueNexus search - filter by title containing query
        response = await client.query(
            Collections.CHATS,
            QueryOptions(
                filter={"user_id": user_id, "archived": False, "title": {"$contains": query}},
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
