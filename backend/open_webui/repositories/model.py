"""
Model Repository implementations for PostgreSQL and BlueNexus storage.
"""

import logging
import time
from typing import Optional, List

from open_webui.repositories.base import BaseModelRepository
from open_webui.models.models import Models, ModelForm
from open_webui.env import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("REPOSITORIES", logging.INFO))


class PostgresModelRepository(BaseModelRepository):
    """PostgreSQL implementation of model repository using existing Models model."""

    async def get_list(self, user_id: str) -> List[dict]:
        models = Models.get_models_by_user_id(user_id)
        return [model.model_dump() for model in models]

    async def get_all(self) -> List[dict]:
        models = Models.get_all_models()
        return [model.model_dump() for model in models]

    async def get_by_id(self, model_id: str) -> Optional[dict]:
        model = Models.get_model_by_id(model_id)
        return model.model_dump() if model else None

    async def create(self, user_id: str, data: dict) -> dict:
        form = ModelForm(
            id=data.get("id"),
            base_model_id=data.get("base_model_id"),
            name=data.get("name"),
            meta=data.get("meta", {}),
            params=data.get("params", {}),
            access_control=data.get("access_control"),
            is_active=data.get("is_active", True),
        )
        model = Models.insert_new_model(form, user_id)
        return model.model_dump() if model else {}

    async def update(self, model_id: str, data: dict) -> Optional[dict]:
        form = ModelForm(
            id=model_id,
            base_model_id=data.get("base_model_id"),
            name=data.get("name"),
            meta=data.get("meta", {}),
            params=data.get("params", {}),
            access_control=data.get("access_control"),
            is_active=data.get("is_active", True),
        )
        model = Models.update_model_by_id(model_id, form)
        return model.model_dump() if model else None

    async def delete(self, model_id: str) -> bool:
        return Models.delete_model_by_id(model_id)


class BlueNexusModelRepository(BaseModelRepository):
    """BlueNexus implementation of model repository."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self._client = None

    def _get_client(self):
        """Get BlueNexus client for the user."""
        if self._client is None:
            from open_webui.utils.bluenexus.factory import get_bluenexus_client_for_user
            self._client = get_bluenexus_client_for_user(self.user_id)
        return self._client

    def _normalize_model_data(self, model_data: dict) -> dict:
        """Normalize BlueNexus data to Open WebUI format."""
        # Handle timestamp conversion
        if "createdAt" in model_data and model_data["createdAt"]:
            created = model_data["createdAt"]
            if hasattr(created, "timestamp"):
                model_data["created_at"] = int(created.timestamp())
            elif isinstance(created, (int, float)):
                model_data["created_at"] = int(created)
        if "created_at" not in model_data:
            model_data["created_at"] = int(time.time())

        if "updatedAt" in model_data and model_data["updatedAt"]:
            updated = model_data["updatedAt"]
            if hasattr(updated, "timestamp"):
                model_data["updated_at"] = int(updated.timestamp())
            elif isinstance(updated, (int, float)):
                model_data["updated_at"] = int(updated)
        if "updated_at" not in model_data:
            model_data["updated_at"] = int(time.time())

        return model_data

    async def get_list(self, user_id: str) -> List[dict]:
        from open_webui.utils.bluenexus.collections import Collections
        from open_webui.utils.bluenexus.types import QueryOptions, SortBy, SortOrder

        client = self._get_client()
        if not client:
            return []

        response = await client.query(
            Collections.MODELS,
            QueryOptions(
                filter={"user_id": user_id},
                sort_by=SortBy.UPDATED_AT,
                sort_order=SortOrder.DESC,
            )
        )

        models = []
        for record in response.get_records():
            model_data = record.model_dump()
            self._normalize_model_data(model_data)
            models.append(model_data)
        return models

    async def get_all(self) -> List[dict]:
        from open_webui.utils.bluenexus.collections import Collections
        from open_webui.utils.bluenexus.types import QueryOptions, SortBy, SortOrder

        client = self._get_client()
        if not client:
            return []

        response = await client.query(
            Collections.MODELS,
            QueryOptions(
                sort_by=SortBy.UPDATED_AT,
                sort_order=SortOrder.DESC,
            )
        )

        models = []
        for record in response.get_records():
            model_data = record.model_dump()
            self._normalize_model_data(model_data)
            models.append(model_data)
        return models

    async def get_by_id(self, model_id: str) -> Optional[dict]:
        from open_webui.utils.bluenexus.collections import Collections
        from open_webui.utils.bluenexus.types import QueryOptions

        client = self._get_client()
        if not client:
            return None

        response = await client.query(
            Collections.MODELS,
            QueryOptions(filter={"id": model_id}, limit=1)
        )

        if response.data and len(response.data) > 0:
            record = response.get_records()[0]
            model_data = record.model_dump()
            self._normalize_model_data(model_data)
            return model_data
        return None

    async def create(self, user_id: str, data: dict) -> dict:
        from open_webui.utils.bluenexus.collections import Collections

        client = self._get_client()
        if not client:
            return {}

        new_model_data = {
            "id": data.get("id"),
            "user_id": user_id,
            "base_model_id": data.get("base_model_id"),
            "name": data.get("name"),
            "meta": data.get("meta", {}),
            "params": data.get("params", {}),
            "access_control": data.get("access_control"),
            "is_active": data.get("is_active", True),
            "created_at": int(time.time()),
            "updated_at": int(time.time()),
        }

        record = await client.create(Collections.MODELS, new_model_data)
        result_data = record.model_dump()
        self._normalize_model_data(result_data)
        return result_data

    async def update(self, model_id: str, data: dict) -> Optional[dict]:
        from open_webui.utils.bluenexus.collections import Collections
        from open_webui.utils.bluenexus.types import QueryOptions

        client = self._get_client()
        if not client:
            return None

        # Find existing model
        response = await client.query(
            Collections.MODELS,
            QueryOptions(filter={"id": model_id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            return None

        record = response.get_records()[0]
        model_data = record.model_dump()

        # Update fields
        for key in ["base_model_id", "name", "meta", "params", "access_control", "is_active"]:
            if key in data:
                model_data[key] = data[key]
        model_data["updated_at"] = int(time.time())

        updated_record = await client.update(Collections.MODELS, record.id, model_data)
        result_data = updated_record.model_dump()
        self._normalize_model_data(result_data)
        return result_data

    async def delete(self, model_id: str) -> bool:
        from open_webui.utils.bluenexus.collections import Collections
        from open_webui.utils.bluenexus.types import QueryOptions

        client = self._get_client()
        if not client:
            return False

        response = await client.query(
            Collections.MODELS,
            QueryOptions(filter={"id": model_id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            return False

        record = response.get_records()[0]
        await client.delete(Collections.MODELS, record.id)
        return True
