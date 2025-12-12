import logging
import time
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request

from open_webui.models.prompts import (
    PromptForm,
    PromptUserResponse,
    PromptModel,
)
from open_webui.constants import ERROR_MESSAGES
from open_webui.utils.auth import get_admin_user, get_verified_user
from open_webui.utils.access_control import has_access, has_permission
from open_webui.utils.bluenexus.collections import Collections
from open_webui.utils.bluenexus.factory import get_bluenexus_client_for_user
from open_webui.utils.bluenexus.types import QueryOptions, BlueNexusError
from open_webui.utils.bluenexus.cache import (
    get_cached_record_id,
    set_cached_record_id,
)
from open_webui.config import BYPASS_ADMIN_ACCESS_CONTROL

log = logging.getLogger(__name__)

router = APIRouter()

# Simple in-memory cache for prompt data
_prompt_cache: dict[str, tuple[dict, str, float]] = {}  # key -> (data, record_id, timestamp)
_PROMPT_CACHE_TTL = 60  # 60 seconds


def _cache_prompt(user_id: str, command: str, data: dict, record_id: str) -> None:
    """Cache prompt data with record ID."""
    key = f"{user_id}:prompts:{command}"
    _prompt_cache[key] = (data, record_id, time.time())
    set_cached_record_id(user_id, "prompts", command, record_id)


def _get_cached_prompt(user_id: str, command: str) -> tuple[Optional[dict], Optional[str]]:
    """Get cached prompt data and record ID."""
    key = f"{user_id}:prompts:{command}"
    if key in _prompt_cache:
        data, record_id, ts = _prompt_cache[key]
        if time.time() - ts < _PROMPT_CACHE_TTL:
            return data, record_id
        else:
            del _prompt_cache[key]
    # Try to get just record ID from persistent cache
    record_id = get_cached_record_id(user_id, "prompts", command)
    return None, record_id


def _invalidate_prompt_cache(user_id: str, command: str) -> None:
    """Invalidate prompt cache."""
    key = f"{user_id}:prompts:{command}"
    if key in _prompt_cache:
        del _prompt_cache[key]


def get_client_or_raise(user_id: str):
    """Get BlueNexus client for user or raise 401 error."""
    client = get_bluenexus_client_for_user(user_id)
    if not client:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="BlueNexus session required. Please log in with BlueNexus.",
        )
    return client


def _ensure_prompt_fields(prompt_data: dict) -> dict:
    """Ensure required fields exist in prompt data from BlueNexus."""
    # Map owui_id to command
    if "command" not in prompt_data and "owui_id" in prompt_data:
        prompt_data["command"] = prompt_data["owui_id"]

    # Ensure timestamp exists (use createdAt from BlueNexus or current time)
    if "timestamp" not in prompt_data:
        if "createdAt" in prompt_data:
            created_at = prompt_data["createdAt"]
            # BlueNexus may return either Unix timestamp (int) or ISO string
            if isinstance(created_at, (int, float)):
                prompt_data["timestamp"] = int(created_at)
            elif isinstance(created_at, str):
                from datetime import datetime
                try:
                    dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    prompt_data["timestamp"] = int(dt.timestamp())
                except (ValueError, AttributeError):
                    prompt_data["timestamp"] = int(time.time())
            else:
                prompt_data["timestamp"] = int(time.time())
        else:
            prompt_data["timestamp"] = int(time.time())

    return prompt_data


############################
# GetPrompts
############################


@router.get("/", response_model=list[PromptModel])
async def get_prompts(user=Depends(get_verified_user)):
    client = get_client_or_raise(user.id)

    try:
        # Query all prompts from BlueNexus for this user
        response = await client.query(
            Collections.PROMPTS,
            QueryOptions(filter={"user_id": user.id})
        )

        records = response.get_records()
        prompts = []

        for record in records:
            # Convert BlueNexus record to PromptModel
            prompt_data = _ensure_prompt_fields(record.model_dump())

            # Apply access control - user always has access to their own prompts
            if user.role == "admin" and BYPASS_ADMIN_ACCESS_CONTROL:
                prompts.append(PromptModel(**prompt_data))
            elif prompt_data.get("user_id") == user.id:
                prompts.append(PromptModel(**prompt_data))
            elif has_access(user.id, "read", prompt_data.get("access_control")):
                prompts.append(PromptModel(**prompt_data))

        return prompts
    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


@router.get("/list", response_model=list[PromptUserResponse])
async def get_prompt_list(user=Depends(get_verified_user)):
    client = get_client_or_raise(user.id)

    try:
        # Query all prompts from BlueNexus for this user
        response = await client.query(
            Collections.PROMPTS,
            QueryOptions(filter={"user_id": user.id})
        )

        records = response.get_records()
        prompts = []

        for record in records:
            # Convert BlueNexus record to PromptUserResponse
            prompt_data = _ensure_prompt_fields(record.model_dump())

            # Apply access control for write permission - user always has access to their own prompts
            if user.role == "admin" and BYPASS_ADMIN_ACCESS_CONTROL:
                prompts.append(PromptUserResponse(**prompt_data))
            elif prompt_data.get("user_id") == user.id:
                prompts.append(PromptUserResponse(**prompt_data))
            elif has_access(user.id, "write", prompt_data.get("access_control")):
                prompts.append(PromptUserResponse(**prompt_data))

        return prompts
    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# CreateNewPrompt
############################


@router.post("/create", response_model=Optional[PromptModel])
async def create_new_prompt(
    request: Request, form_data: PromptForm, user=Depends(get_verified_user)
):
    if user.role != "admin" and not has_permission(
        user.id, "workspace.prompts", request.app.state.config.USER_PERMISSIONS
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.UNAUTHORIZED,
        )

    client = get_client_or_raise(user.id)

    try:
        # Normalize command to always have leading slash
        command = form_data.command if form_data.command.startswith("/") else f"/{form_data.command}"

        # Check if prompt with this command already exists
        existing = await client.query(
            Collections.PROMPTS,
            QueryOptions(filter={"owui_id": command, "user_id": user.id}, limit=1)
        )

        if existing.data and len(existing.data) > 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.COMMAND_TAKEN,
            )

        # Create new prompt in BlueNexus
        prompt_data = form_data.model_dump()
        prompt_data["owui_id"] = command
        prompt_data["command"] = command
        prompt_data["user_id"] = user.id
        prompt_data["timestamp"] = int(time.time())

        record = await client.create(Collections.PROMPTS, prompt_data)

        # Convert back to PromptModel
        result_data = _ensure_prompt_fields(record.model_dump())
        return PromptModel(**result_data)

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# GetPromptByCommand
############################


@router.get("/command/{command}", response_model=Optional[PromptModel])
async def get_prompt_by_command(command: str, user=Depends(get_verified_user)):
    full_command = f"/{command}"

    # Check cache first
    cached_data, record_id = _get_cached_prompt(user.id, full_command)
    if cached_data:
        prompt_data = _ensure_prompt_fields(cached_data)
        if (
            user.role == "admin"
            or prompt_data.get("user_id") == user.id
            or has_access(user.id, "read", prompt_data.get("access_control"))
        ):
            return PromptModel(**prompt_data)
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

    client = get_client_or_raise(user.id)

    try:
        # Query BlueNexus for prompt with this command
        response = await client.query(
            Collections.PROMPTS,
            QueryOptions(filter={"owui_id": full_command}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

        record = response.get_records()[0]
        prompt_data = _ensure_prompt_fields(record.model_dump())

        # Cache for future operations
        _cache_prompt(user.id, full_command, prompt_data, record.id)

        # Check access
        if (
            user.role == "admin"
            or prompt_data.get("user_id") == user.id
            or has_access(user.id, "read", prompt_data.get("access_control"))
        ):
            return PromptModel(**prompt_data)
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


############################
# UpdatePromptByCommand
############################


@router.post("/command/{command}/update", response_model=Optional[PromptModel])
async def update_prompt_by_command(
    command: str,
    form_data: PromptForm,
    user=Depends(get_verified_user),
):
    client = get_client_or_raise(user.id)
    full_command = f"/{command}"

    try:
        # Try to get cached data and record ID
        cached_data, record_id = _get_cached_prompt(user.id, full_command)

        if record_id and cached_data:
            # FAST PATH: Use cached record ID - only 1 API call
            log.debug(f"[BlueNexus] Fast update for prompt {full_command} using cached record_id")

            # Check access first
            if (
                cached_data.get("user_id") != user.id
                and not has_access(user.id, "write", cached_data.get("access_control"))
                and user.role != "admin"
            ):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
                )

            # Update in BlueNexus
            updated_data = form_data.model_dump()
            updated_data["owui_id"] = full_command
            updated_data["user_id"] = cached_data.get("user_id")

            updated_record = await client.update(Collections.PROMPTS, record_id, updated_data)
            result_data = _ensure_prompt_fields(updated_record.model_dump())
            _cache_prompt(user.id, full_command, result_data, record_id)
            return PromptModel(**result_data)

        # SLOW PATH: Need to query for record ID first - 2 API calls
        response = await client.query(
            Collections.PROMPTS,
            QueryOptions(filter={"owui_id": full_command}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

        record = response.get_records()[0]
        prompt_data = record.model_dump()

        # Check access
        if (
            prompt_data.get("user_id") != user.id
            and not has_access(user.id, "write", prompt_data.get("access_control"))
            and user.role != "admin"
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
            )

        # Update in BlueNexus
        updated_data = form_data.model_dump()
        updated_data["owui_id"] = full_command
        updated_data["user_id"] = prompt_data.get("user_id")

        updated_record = await client.update(Collections.PROMPTS, record.id, updated_data)

        # Convert back to PromptModel and cache
        result_data = _ensure_prompt_fields(updated_record.model_dump())
        _cache_prompt(user.id, full_command, result_data, record.id)
        return PromptModel(**result_data)

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# DeletePromptByCommand
############################


@router.delete("/command/{command}/delete", response_model=bool)
async def delete_prompt_by_command(command: str, user=Depends(get_verified_user)):
    client = get_client_or_raise(user.id)
    full_command = f"/{command}"

    try:
        # Try to get cached data and record ID
        cached_data, record_id = _get_cached_prompt(user.id, full_command)

        if record_id and cached_data:
            # FAST PATH: Use cached record ID - only 1 API call
            log.debug(f"[BlueNexus] Fast delete for prompt {full_command} using cached record_id")

            # Check access first
            if (
                cached_data.get("user_id") != user.id
                and not has_access(user.id, "write", cached_data.get("access_control"))
                and user.role != "admin"
            ):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
                )

            # Delete from BlueNexus
            await client.delete(Collections.PROMPTS, record_id)
            _invalidate_prompt_cache(user.id, full_command)
            return True

        # SLOW PATH: Need to query for record ID first - 2 API calls
        response = await client.query(
            Collections.PROMPTS,
            QueryOptions(filter={"owui_id": full_command}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

        record = response.get_records()[0]
        prompt_data = record.model_dump()

        # Check access
        if (
            prompt_data.get("user_id") != user.id
            and not has_access(user.id, "write", prompt_data.get("access_control"))
            and user.role != "admin"
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
            )

        # Delete from BlueNexus
        await client.delete(Collections.PROMPTS, record.id)
        _invalidate_prompt_cache(user.id, full_command)
        return True

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )
