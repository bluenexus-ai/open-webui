"""
Tool Repository implementations for PostgreSQL and BlueNexus storage.
"""

import logging
import time
from typing import Optional, List

from open_webui.repositories.base import BaseToolRepository
from open_webui.models.tools import Tools, ToolForm, ToolMeta
from open_webui.env import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("REPOSITORIES", logging.INFO))


class PostgresToolRepository(BaseToolRepository):
    """PostgreSQL implementation of tool repository using existing Tools model."""

    async def get_list(self, user_id: str) -> List[dict]:
        tools = Tools.get_tools_by_user_id(user_id)
        return [tool.model_dump() for tool in tools]

    async def get_all(self) -> List[dict]:
        tools = Tools.get_tools()
        return [tool.model_dump() for tool in tools]

    async def get_by_id(self, tool_id: str) -> Optional[dict]:
        tool = Tools.get_tool_by_id(tool_id)
        return tool.model_dump() if tool else None

    async def create(self, user_id: str, data: dict) -> dict:
        form = ToolForm(
            id=data.get("id"),
            name=data.get("name"),
            content=data.get("content"),
            meta=ToolMeta(**data.get("meta", {})),
            access_control=data.get("access_control"),
        )
        specs = data.get("specs", [])
        tool = Tools.insert_new_tool(user_id, form, specs)
        return tool.model_dump() if tool else {}

    async def update(self, tool_id: str, data: dict) -> Optional[dict]:
        tool = Tools.update_tool_by_id(tool_id, data)
        return tool.model_dump() if tool else None

    async def delete(self, tool_id: str) -> bool:
        return Tools.delete_tool_by_id(tool_id)


class BlueNexusToolRepository(BaseToolRepository):
    """BlueNexus implementation of tool repository."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self._client = None

    def _get_client(self):
        """Get BlueNexus client for the user."""
        if self._client is None:
            from open_webui.utils.bluenexus.factory import get_bluenexus_client_for_user
            self._client = get_bluenexus_client_for_user(self.user_id)
        return self._client

    def _normalize_tool_data(self, tool_data: dict) -> dict:
        """Normalize BlueNexus data to Open WebUI format."""
        # Handle timestamp conversion
        if "createdAt" in tool_data and tool_data["createdAt"]:
            created = tool_data["createdAt"]
            if hasattr(created, "timestamp"):
                tool_data["created_at"] = int(created.timestamp())
            elif isinstance(created, (int, float)):
                tool_data["created_at"] = int(created)
        if "created_at" not in tool_data:
            tool_data["created_at"] = int(time.time())

        if "updatedAt" in tool_data and tool_data["updatedAt"]:
            updated = tool_data["updatedAt"]
            if hasattr(updated, "timestamp"):
                tool_data["updated_at"] = int(updated.timestamp())
            elif isinstance(updated, (int, float)):
                tool_data["updated_at"] = int(updated)
        if "updated_at" not in tool_data:
            tool_data["updated_at"] = int(time.time())

        return tool_data

    async def get_list(self, user_id: str) -> List[dict]:
        from open_webui.utils.bluenexus.collections import Collections
        from open_webui.utils.bluenexus.types import QueryOptions, SortBy, SortOrder

        client = self._get_client()
        if not client:
            return []

        response = await client.query(
            Collections.TOOLS,
            QueryOptions(
                filter={"user_id": user_id},
                sort_by=SortBy.UPDATED_AT,
                sort_order=SortOrder.DESC,
            )
        )

        tools = []
        for record in response.get_records():
            tool_data = record.model_dump()
            self._normalize_tool_data(tool_data)
            tools.append(tool_data)
        return tools

    async def get_all(self) -> List[dict]:
        from open_webui.utils.bluenexus.collections import Collections
        from open_webui.utils.bluenexus.types import QueryOptions, SortBy, SortOrder

        client = self._get_client()
        if not client:
            return []

        response = await client.query(
            Collections.TOOLS,
            QueryOptions(
                sort_by=SortBy.UPDATED_AT,
                sort_order=SortOrder.DESC,
            )
        )

        tools = []
        for record in response.get_records():
            tool_data = record.model_dump()
            self._normalize_tool_data(tool_data)
            tools.append(tool_data)
        return tools

    async def get_by_id(self, tool_id: str) -> Optional[dict]:
        from open_webui.utils.bluenexus.collections import Collections
        from open_webui.utils.bluenexus.types import QueryOptions

        client = self._get_client()
        if not client:
            return None

        response = await client.query(
            Collections.TOOLS,
            QueryOptions(filter={"id": tool_id}, limit=1)
        )

        if response.data and len(response.data) > 0:
            record = response.get_records()[0]
            tool_data = record.model_dump()
            self._normalize_tool_data(tool_data)
            return tool_data
        return None

    async def create(self, user_id: str, data: dict) -> dict:
        from open_webui.utils.bluenexus.collections import Collections

        client = self._get_client()
        if not client:
            return {}

        new_tool_data = {
            "id": data.get("id"),
            "user_id": user_id,
            "name": data.get("name"),
            "content": data.get("content"),
            "specs": data.get("specs", []),
            "meta": data.get("meta", {}),
            "valves": data.get("valves"),
            "access_control": data.get("access_control"),
            "created_at": int(time.time()),
            "updated_at": int(time.time()),
        }

        record = await client.create(Collections.TOOLS, new_tool_data)
        result_data = record.model_dump()
        self._normalize_tool_data(result_data)
        return result_data

    async def update(self, tool_id: str, data: dict) -> Optional[dict]:
        from open_webui.utils.bluenexus.collections import Collections
        from open_webui.utils.bluenexus.types import QueryOptions

        client = self._get_client()
        if not client:
            return None

        # Find existing tool
        response = await client.query(
            Collections.TOOLS,
            QueryOptions(filter={"id": tool_id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            return None

        record = response.get_records()[0]
        tool_data = record.model_dump()

        # Update fields
        for key in ["name", "content", "specs", "meta", "valves", "access_control"]:
            if key in data:
                tool_data[key] = data[key]
        tool_data["updated_at"] = int(time.time())

        updated_record = await client.update(Collections.TOOLS, record.id, tool_data)
        result_data = updated_record.model_dump()
        self._normalize_tool_data(result_data)
        return result_data

    async def delete(self, tool_id: str) -> bool:
        from open_webui.utils.bluenexus.collections import Collections
        from open_webui.utils.bluenexus.types import QueryOptions

        client = self._get_client()
        if not client:
            return False

        response = await client.query(
            Collections.TOOLS,
            QueryOptions(filter={"id": tool_id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            return False

        record = response.get_records()[0]
        await client.delete(Collections.TOOLS, record.id)
        return True
