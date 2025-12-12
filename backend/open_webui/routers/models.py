from typing import Optional
import io
import base64
import json
import asyncio
import logging
import time
from datetime import datetime

from open_webui.models.models import (
    ModelForm,
    ModelModel,
    ModelResponse,
    ModelUserResponse,
    ModelParams,
    ModelMeta,
)

from pydantic import BaseModel
from open_webui.constants import ERROR_MESSAGES
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    status,
    Response,
)
from fastapi.responses import FileResponse, StreamingResponse


from open_webui.utils.auth import get_admin_user, get_verified_user
from open_webui.utils.access_control import has_access, has_permission
from open_webui.utils.bluenexus.collections import Collections
from open_webui.utils.bluenexus.factory import get_bluenexus_client_for_user
from open_webui.utils.bluenexus.types import QueryOptions, BlueNexusError
from open_webui.utils.bluenexus.cache import (
    get_cached_record_id,
    set_cached_record_id,
)
from open_webui.config import BYPASS_ADMIN_ACCESS_CONTROL, STATIC_DIR

log = logging.getLogger(__name__)

router = APIRouter()

# Simple in-memory cache for model data
_model_cache: dict[str, tuple[dict, str, float]] = {}  # key -> (data, record_id, timestamp)
_MODEL_CACHE_TTL = 60  # 60 seconds


def _cache_model(user_id: str, model_id: str, data: dict, record_id: str) -> None:
    """Cache model data with record ID."""
    key = f"{user_id}:models:{model_id}"
    _model_cache[key] = (data, record_id, time.time())
    set_cached_record_id(user_id, "models", model_id, record_id)


def _get_cached_model(user_id: str, model_id: str) -> tuple[Optional[dict], Optional[str]]:
    """Get cached model data and record ID."""
    key = f"{user_id}:models:{model_id}"
    if key in _model_cache:
        data, record_id, ts = _model_cache[key]
        if time.time() - ts < _MODEL_CACHE_TTL:
            return data, record_id
        else:
            del _model_cache[key]
    # Try to get just record ID from persistent cache
    record_id = get_cached_record_id(user_id, "models", model_id)
    return None, record_id


def _invalidate_model_cache(user_id: str, model_id: str) -> None:
    """Invalidate model cache."""
    key = f"{user_id}:models:{model_id}"
    if key in _model_cache:
        del _model_cache[key]


def get_client_or_raise(user_id: str):
    """Get BlueNexus client for user or raise 401 error."""
    client = get_bluenexus_client_for_user(user_id)
    if not client:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="BlueNexus session required. Please log in with BlueNexus.",
        )
    return client


def validate_model_id(model_id: str) -> bool:
    return model_id and len(model_id) <= 256


def _convert_bluenexus_to_model(data: dict) -> dict:
    """
    Convert BlueNexus record data to ModelModel format.

    Handles:
    - createdAt/updatedAt (ISO string) → created_at/updated_at (epoch ns)
    - owui_id → id
    - Ensures params and meta have defaults if missing
    """
    result = dict(data)

    # Map owui_id to id
    if "owui_id" in result:
        result["id"] = result["owui_id"]

    # Convert timestamps from ISO string to epoch nanoseconds
    if "createdAt" in result:
        created_at = result.pop("createdAt")
        if isinstance(created_at, str):
            # Parse ISO string to datetime, then to epoch ns
            try:
                dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                result["created_at"] = int(dt.timestamp() * 1_000_000_000)
            except (ValueError, AttributeError):
                result["created_at"] = int(time.time_ns())
        elif isinstance(created_at, (int, float)):
            result["created_at"] = int(created_at)
        else:
            result["created_at"] = int(time.time_ns())

    if "updatedAt" in result:
        updated_at = result.pop("updatedAt")
        if isinstance(updated_at, str):
            try:
                dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                result["updated_at"] = int(dt.timestamp() * 1_000_000_000)
            except (ValueError, AttributeError):
                result["updated_at"] = int(time.time_ns())
        elif isinstance(updated_at, (int, float)):
            result["updated_at"] = int(updated_at)
        else:
            result["updated_at"] = int(time.time_ns())

    # Ensure created_at and updated_at exist
    if "created_at" not in result:
        result["created_at"] = int(time.time_ns())
    if "updated_at" not in result:
        result["updated_at"] = int(time.time_ns())

    # Ensure params has default value
    if "params" not in result or result["params"] is None:
        result["params"] = ModelParams().model_dump()

    # Ensure meta has default value
    if "meta" not in result or result["meta"] is None:
        result["meta"] = ModelMeta(description="").model_dump()

    # Ensure is_active has default
    if "is_active" not in result:
        result["is_active"] = True

    return result


###########################
# GetModels
###########################


@router.get(
    "/list", response_model=list[ModelUserResponse]
)  # do NOT use "/" as path, conflicts with main.py
async def get_models(id: Optional[str] = None, user=Depends(get_verified_user)):
    client = get_client_or_raise(user.id)

    try:
        # Query all models from BlueNexus for this user
        response = await client.query(
            Collections.MODELS,
            QueryOptions(filter={"user_id": user.id})
        )

        records = response.get_records()
        models = []

        for record in records:
            model_data = _convert_bluenexus_to_model(record.model_dump())

            # Apply access control
            if user.role == "admin" and BYPASS_ADMIN_ACCESS_CONTROL:
                models.append(ModelUserResponse(**model_data))
            elif has_access(user.id, "read", model_data.get("access_control")):
                models.append(ModelUserResponse(**model_data))

        return models

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


###########################
# GetBaseModels
###########################


@router.get("/base", response_model=list[ModelResponse])
async def get_base_models(user=Depends(get_admin_user)):
    client = get_client_or_raise(user.id)

    try:
        # Query all base models from BlueNexus (models with base_model_id set)
        response = await client.query(
            Collections.MODELS,
            QueryOptions(filter={"user_id": user.id})
        )

        base_models = []
        for record in response.get_records():
            model_data = _convert_bluenexus_to_model(record.model_dump())
            # Filter for base models (those with base_model_id set)
            if model_data.get("base_model_id"):
                base_models.append(ModelResponse(**model_data))

        return base_models

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# CreateNewModel
############################


@router.post("/create", response_model=Optional[ModelModel])
async def create_new_model(
    request: Request,
    form_data: ModelForm,
    user=Depends(get_verified_user),
):
    if user.role != "admin" and not has_permission(
        user.id, "workspace.models", request.app.state.config.USER_PERMISSIONS
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.UNAUTHORIZED,
        )

    if not validate_model_id(form_data.id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.MODEL_ID_TOO_LONG,
        )

    client = get_client_or_raise(user.id)

    try:
        # Check if model with this ID already exists
        existing = await client.query(
            Collections.MODELS,
            QueryOptions(filter={"owui_id": form_data.id, "user_id": user.id}, limit=1)
        )

        if existing.data and len(existing.data) > 0:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.MODEL_ID_TAKEN,
            )

        # Create new model in BlueNexus
        model_data = form_data.model_dump()
        model_data["owui_id"] = form_data.id
        model_data["user_id"] = user.id

        record = await client.create(Collections.MODELS, model_data)

        # Convert BlueNexus response to ModelModel format
        result_data = _convert_bluenexus_to_model(record.model_dump())
        return ModelModel(**result_data)

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# ExportModels
############################


@router.get("/export", response_model=list[ModelModel])
async def export_models(user=Depends(get_admin_user)):
    client = get_client_or_raise(user.id)

    try:
        # Query all models from BlueNexus for export
        response = await client.query(
            Collections.MODELS,
            QueryOptions(filter={"user_id": user.id})
        )

        models = []
        for record in response.get_records():
            model_data = _convert_bluenexus_to_model(record.model_dump())
            models.append(ModelModel(**model_data))

        return models

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# ImportModels
############################


class ModelsImportForm(BaseModel):
    models: list[dict]


@router.post("/import", response_model=bool)
async def import_models(
    user=Depends(get_admin_user), form_data: ModelsImportForm = (...)
):
    client = get_client_or_raise(user.id)

    try:
        data = form_data.models
        if not isinstance(data, list):
            raise HTTPException(status_code=400, detail="Invalid JSON format")

        for model_data in data:
            model_id = model_data.get("id")

            if not model_id or not validate_model_id(model_id):
                continue

            # Check if model already exists in BlueNexus
            existing = await client.query(
                Collections.MODELS,
                QueryOptions(filter={"owui_id": model_id, "user_id": user.id}, limit=1)
            )

            model_data["meta"] = model_data.get("meta", {})
            model_data["params"] = model_data.get("params", {})

            if existing.data and len(existing.data) > 0:
                # Update existing model
                record = existing.get_records()[0]
                existing_data = record.model_dump()
                updated_data = {**existing_data, **model_data}
                updated_data["owui_id"] = model_id
                updated_data["user_id"] = user.id
                await client.update(Collections.MODELS, record.id, updated_data)
            else:
                # Insert new model
                model_data["owui_id"] = model_id
                model_data["user_id"] = user.id
                await client.create(Collections.MODELS, model_data)

        return True

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )
    except Exception as e:
        log.exception(e)
        raise HTTPException(status_code=500, detail=str(e))


############################
# SyncModels
############################


class SyncModelsForm(BaseModel):
    models: list[ModelModel] = []


@router.post("/sync", response_model=list[ModelModel])
async def sync_models(
    request: Request, form_data: SyncModelsForm, user=Depends(get_admin_user)
):
    client = get_client_or_raise(user.id)

    try:
        synced_models = []

        for model in form_data.models:
            model_data = model.model_dump()
            model_id = model_data.get("id")

            if not model_id:
                continue

            # Check if model exists
            existing = await client.query(
                Collections.MODELS,
                QueryOptions(filter={"owui_id": model_id, "user_id": user.id}, limit=1)
            )

            model_data["owui_id"] = model_id
            model_data["user_id"] = user.id

            if existing.data and len(existing.data) > 0:
                # Update existing model
                record = existing.get_records()[0]
                updated = await client.update(Collections.MODELS, record.id, model_data)
                result_data = _convert_bluenexus_to_model(updated.model_dump())
                synced_models.append(ModelModel(**result_data))
            else:
                # Create new model
                created = await client.create(Collections.MODELS, model_data)
                result_data = _convert_bluenexus_to_model(created.model_dump())
                synced_models.append(ModelModel(**result_data))

        return synced_models

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


###########################
# GetModelById
###########################


# Note: We're not using the typical url path param here, but instead using a query parameter to allow '/' in the id
@router.get("/model", response_model=Optional[ModelResponse])
async def get_model_by_id(id: str, user=Depends(get_verified_user)):
    # Check cache first
    cached_data, record_id = _get_cached_model(user.id, id)
    if cached_data:
        model_data = _convert_bluenexus_to_model(cached_data)
        if (
            (user.role == "admin" and BYPASS_ADMIN_ACCESS_CONTROL)
            or model_data.get("user_id") == user.id
            or has_access(user.id, "read", model_data.get("access_control"))
        ):
            return ModelResponse(**model_data)
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

    client = get_client_or_raise(user.id)

    try:
        # Query BlueNexus for model with this ID
        response = await client.query(
            Collections.MODELS,
            QueryOptions(filter={"owui_id": id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

        record = response.get_records()[0]
        raw_data = record.model_dump()

        # Cache for future operations
        _cache_model(user.id, id, raw_data, record.id)

        model_data = _convert_bluenexus_to_model(raw_data)

        # Check access
        if (
            (user.role == "admin" and BYPASS_ADMIN_ACCESS_CONTROL)
            or model_data.get("user_id") == user.id
            or has_access(user.id, "read", model_data.get("access_control"))
        ):
            return ModelResponse(**model_data)
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


###########################
# GetModelById
###########################


@router.get("/model/profile/image")
async def get_model_profile_image(id: str, user=Depends(get_verified_user)):
    client = get_client_or_raise(user.id)

    try:
        # Query BlueNexus for model with this ID
        response = await client.query(
            Collections.MODELS,
            QueryOptions(filter={"owui_id": id}, limit=1)
        )

        if response.data and len(response.data) > 0:
            model_data = response.get_records()[0].model_dump()
            meta = model_data.get("meta", {})
            profile_image_url = meta.get("profile_image_url") if isinstance(meta, dict) else None

            if profile_image_url:
                if profile_image_url.startswith("http"):
                    return Response(
                        status_code=status.HTTP_302_FOUND,
                        headers={"Location": profile_image_url},
                    )
                elif profile_image_url.startswith("data:image"):
                    try:
                        header, base64_data = profile_image_url.split(",", 1)
                        image_data = base64.b64decode(base64_data)
                        image_buffer = io.BytesIO(image_data)

                        return StreamingResponse(
                            image_buffer,
                            media_type="image/png",
                            headers={"Content-Disposition": "inline; filename=image.png"},
                        )
                    except Exception:
                        pass

        return FileResponse(f"{STATIC_DIR}/favicon.png")

    except BlueNexusError:
        return FileResponse(f"{STATIC_DIR}/favicon.png")


############################
# ToggleModelById
############################


@router.post("/model/toggle", response_model=Optional[ModelResponse])
async def toggle_model_by_id(id: str, user=Depends(get_verified_user)):
    client = get_client_or_raise(user.id)

    try:
        # Try to get cached data and record ID
        cached_data, record_id = _get_cached_model(user.id, id)

        if record_id and cached_data:
            # FAST PATH: Use cached record ID - only 1 API call
            log.debug(f"[BlueNexus] Fast toggle for model {id} using cached record_id")

            # Check access first
            if not (
                user.role == "admin"
                or cached_data.get("user_id") == user.id
                or has_access(user.id, "write", cached_data.get("access_control"))
            ):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=ERROR_MESSAGES.UNAUTHORIZED,
                )

            # Toggle is_active field
            model_data = cached_data.copy()
            model_data["is_active"] = not model_data.get("is_active", True)

            updated = await client.update(Collections.MODELS, record_id, model_data)
            raw_data = updated.model_dump()
            _cache_model(user.id, id, raw_data, record_id)
            result_data = _convert_bluenexus_to_model(raw_data)
            return ModelResponse(**result_data)

        # SLOW PATH: Need to query for record ID first - 2 API calls
        response = await client.query(
            Collections.MODELS,
            QueryOptions(filter={"owui_id": id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

        record = response.get_records()[0]
        model_data = record.model_dump()

        # Check access
        if not (
            user.role == "admin"
            or model_data.get("user_id") == user.id
            or has_access(user.id, "write", model_data.get("access_control"))
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.UNAUTHORIZED,
            )

        # Toggle is_active field
        model_data["is_active"] = not model_data.get("is_active", True)

        # Update in BlueNexus
        updated = await client.update(Collections.MODELS, record.id, model_data)
        raw_data = updated.model_dump()
        _cache_model(user.id, id, raw_data, record.id)
        result_data = _convert_bluenexus_to_model(raw_data)
        return ModelResponse(**result_data)

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# UpdateModelById
############################


@router.post("/model/update", response_model=Optional[ModelModel])
async def update_model_by_id(
    id: str,
    form_data: ModelForm,
    user=Depends(get_verified_user),
):
    client = get_client_or_raise(user.id)

    try:
        # Try to get cached data and record ID
        cached_data, record_id = _get_cached_model(user.id, id)

        if record_id and cached_data:
            # FAST PATH: Use cached record ID - only 1 API call
            log.debug(f"[BlueNexus] Fast update for model {id} using cached record_id")

            # Check access first
            if (
                cached_data.get("user_id") != user.id
                and not has_access(user.id, "write", cached_data.get("access_control"))
                and user.role != "admin"
            ):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
                )

            # Update in BlueNexus
            updated_data = form_data.model_dump()
            updated_data["owui_id"] = id
            updated_data["user_id"] = cached_data.get("user_id")

            updated_record = await client.update(Collections.MODELS, record_id, updated_data)
            raw_data = updated_record.model_dump()
            _cache_model(user.id, id, raw_data, record_id)
            result_data = _convert_bluenexus_to_model(raw_data)
            return ModelModel(**result_data)

        # SLOW PATH: Need to query for record ID first - 2 API calls
        response = await client.query(
            Collections.MODELS,
            QueryOptions(filter={"owui_id": id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

        record = response.get_records()[0]
        model_data = record.model_dump()

        # Check access
        if (
            model_data.get("user_id") != user.id
            and not has_access(user.id, "write", model_data.get("access_control"))
            and user.role != "admin"
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
            )

        # Update in BlueNexus
        updated_data = form_data.model_dump()
        updated_data["owui_id"] = id
        updated_data["user_id"] = model_data.get("user_id")

        updated_record = await client.update(Collections.MODELS, record.id, updated_data)

        # Convert back to ModelModel and cache
        raw_data = updated_record.model_dump()
        _cache_model(user.id, id, raw_data, record.id)
        result_data = _convert_bluenexus_to_model(raw_data)
        return ModelModel(**result_data)

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# DeleteModelById
############################


@router.delete("/model/delete", response_model=bool)
async def delete_model_by_id(id: str, user=Depends(get_verified_user)):
    client = get_client_or_raise(user.id)

    try:
        # Try to get cached data and record ID
        cached_data, record_id = _get_cached_model(user.id, id)

        if record_id and cached_data:
            # FAST PATH: Use cached record ID - only 1 API call
            log.debug(f"[BlueNexus] Fast delete for model {id} using cached record_id")

            # Check access first
            if (
                user.role != "admin"
                and cached_data.get("user_id") != user.id
                and not has_access(user.id, "write", cached_data.get("access_control"))
            ):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=ERROR_MESSAGES.UNAUTHORIZED,
                )

            # Delete from BlueNexus
            await client.delete(Collections.MODELS, record_id)
            _invalidate_model_cache(user.id, id)
            return True

        # SLOW PATH: Need to query for record ID first - 2 API calls
        response = await client.query(
            Collections.MODELS,
            QueryOptions(filter={"owui_id": id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

        record = response.get_records()[0]
        model_data = record.model_dump()

        # Check access
        if (
            user.role != "admin"
            and model_data.get("user_id") != user.id
            and not has_access(user.id, "write", model_data.get("access_control"))
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.UNAUTHORIZED,
            )

        # Delete from BlueNexus
        await client.delete(Collections.MODELS, record.id)
        _invalidate_model_cache(user.id, id)
        return True

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


@router.delete("/delete/all", response_model=bool)
async def delete_all_models(user=Depends(get_admin_user)):
    client = get_client_or_raise(user.id)

    try:
        # Delete all models for this user using pagination loop
        while True:
            response = await client.query(
                Collections.MODELS,
                QueryOptions(filter={"user_id": user.id}, limit=100)
            )

            if not response.data or len(response.data) == 0:
                break

            # Delete each model in this batch
            for record in response.get_records():
                await client.delete(Collections.MODELS, record.id)

        return True

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )
