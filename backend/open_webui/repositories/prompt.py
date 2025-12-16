"""
Prompt Repository implementations for PostgreSQL and BlueNexus storage.
"""

import logging
import time
import uuid
from typing import Optional, List

from open_webui.repositories.base import BasePromptRepository
from open_webui.models.prompts import Prompts, PromptForm
from open_webui.env import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("REPOSITORIES", logging.INFO))


class PostgresPromptRepository(BasePromptRepository):
    """PostgreSQL implementation of prompt repository using existing Prompts model."""

    async def get_list(self, user_id: str) -> List[dict]:
        prompts = Prompts.get_prompts_by_user_id(user_id)
        return [prompt.model_dump() for prompt in prompts]

    async def get_by_command(self, command: str, user_id: str) -> Optional[dict]:
        prompt = Prompts.get_prompt_by_command(command)
        return prompt.model_dump() if prompt else None

    async def create(self, user_id: str, data: dict) -> dict:
        form = PromptForm(
            command=data.get("command"),
            title=data.get("title"),
            content=data.get("content"),
            access_control=data.get("access_control"),
        )
        prompt = Prompts.insert_new_prompt(user_id, form)
        return prompt.model_dump() if prompt else {}

    async def update(self, command: str, user_id: str, data: dict) -> Optional[dict]:
        form = PromptForm(
            command=command,
            title=data.get("title"),
            content=data.get("content"),
            access_control=data.get("access_control"),
        )
        prompt = Prompts.update_prompt_by_command(command, form)
        return prompt.model_dump() if prompt else None

    async def delete(self, command: str, user_id: str) -> bool:
        return Prompts.delete_prompt_by_command(command)


class BlueNexusPromptRepository(BasePromptRepository):
    """BlueNexus implementation of prompt repository."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self._client = None

    def _get_client(self):
        """Get BlueNexus client for the user."""
        if self._client is None:
            from open_webui.utils.bluenexus.factory import get_bluenexus_client_for_user
            self._client = get_bluenexus_client_for_user(self.user_id)
        return self._client

    def _normalize_prompt_data(self, prompt_data: dict) -> dict:
        """Normalize BlueNexus data to Open WebUI format."""
        # Handle timestamp conversion
        if "createdAt" in prompt_data and prompt_data["createdAt"]:
            created = prompt_data["createdAt"]
            if hasattr(created, "timestamp"):
                prompt_data["timestamp"] = int(created.timestamp())
            elif isinstance(created, (int, float)):
                prompt_data["timestamp"] = int(created)
        if "timestamp" not in prompt_data:
            prompt_data["timestamp"] = int(time.time())

        return prompt_data

    async def get_list(self, user_id: str) -> List[dict]:
        from open_webui.utils.bluenexus.collections import Collections
        from open_webui.utils.bluenexus.types import QueryOptions, SortBy, SortOrder

        client = self._get_client()
        if not client:
            return []

        response = await client.query(
            Collections.PROMPTS,
            QueryOptions(
                filter={"user_id": user_id},
                sort_by=SortBy.CREATED_AT,
                sort_order=SortOrder.DESC,
            )
        )

        prompts = []
        for record in response.get_records():
            prompt_data = record.model_dump()
            self._normalize_prompt_data(prompt_data)
            prompts.append(prompt_data)
        return prompts

    async def get_by_command(self, command: str, user_id: str) -> Optional[dict]:
        from open_webui.utils.bluenexus.collections import Collections
        from open_webui.utils.bluenexus.types import QueryOptions

        client = self._get_client()
        if not client:
            return None

        response = await client.query(
            Collections.PROMPTS,
            QueryOptions(filter={"command": command}, limit=1)
        )

        if response.data and len(response.data) > 0:
            record = response.get_records()[0]
            prompt_data = record.model_dump()
            self._normalize_prompt_data(prompt_data)
            return prompt_data
        return None

    async def create(self, user_id: str, data: dict) -> dict:
        from open_webui.utils.bluenexus.collections import Collections

        client = self._get_client()
        if not client:
            return {}

        new_prompt_data = {
            "command": data.get("command"),
            "user_id": user_id,
            "title": data.get("title"),
            "content": data.get("content"),
            "access_control": data.get("access_control"),
            "timestamp": int(time.time()),
        }

        record = await client.create(Collections.PROMPTS, new_prompt_data)
        result_data = record.model_dump()
        self._normalize_prompt_data(result_data)
        return result_data

    async def update(self, command: str, user_id: str, data: dict) -> Optional[dict]:
        from open_webui.utils.bluenexus.collections import Collections
        from open_webui.utils.bluenexus.types import QueryOptions

        client = self._get_client()
        if not client:
            return None

        # Find existing prompt
        response = await client.query(
            Collections.PROMPTS,
            QueryOptions(filter={"command": command}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            return None

        record = response.get_records()[0]
        prompt_data = record.model_dump()

        # Update fields
        prompt_data["title"] = data.get("title", prompt_data.get("title"))
        prompt_data["content"] = data.get("content", prompt_data.get("content"))
        prompt_data["access_control"] = data.get("access_control", prompt_data.get("access_control"))
        prompt_data["timestamp"] = int(time.time())

        updated_record = await client.update(Collections.PROMPTS, record.id, prompt_data)
        result_data = updated_record.model_dump()
        self._normalize_prompt_data(result_data)
        return result_data

    async def delete(self, command: str, user_id: str) -> bool:
        from open_webui.utils.bluenexus.collections import Collections
        from open_webui.utils.bluenexus.types import QueryOptions

        client = self._get_client()
        if not client:
            return False

        response = await client.query(
            Collections.PROMPTS,
            QueryOptions(filter={"command": command}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            return False

        record = response.get_records()[0]
        await client.delete(Collections.PROMPTS, record.id)
        return True
