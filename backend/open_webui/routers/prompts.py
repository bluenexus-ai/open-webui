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
from open_webui.config import BYPASS_ADMIN_ACCESS_CONTROL

router = APIRouter()


def get_client_or_raise(user_id: str):
    """Get BlueNexus client for user or raise 401 error."""
    client = get_bluenexus_client_for_user(user_id)
    if not client:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="BlueNexus session required. Please log in with BlueNexus.",
        )
    return client

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
            prompt_data = record.model_dump()
            prompt_data["command"] = prompt_data.get("owui_id", prompt_data.get("command"))

            # Apply access control
            if user.role == "admin" and BYPASS_ADMIN_ACCESS_CONTROL:
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
            prompt_data = record.model_dump()
            prompt_data["command"] = prompt_data.get("owui_id", prompt_data.get("command"))

            # Apply access control for write permission
            if user.role == "admin" and BYPASS_ADMIN_ACCESS_CONTROL:
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

        record = await client.create(Collections.PROMPTS, prompt_data)

        # Convert back to PromptModel
        result_data = record.model_dump()
        result_data["command"] = result_data.get("owui_id", result_data.get("command"))
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
    client = get_client_or_raise(user.id)

    try:
        # Query BlueNexus for prompt with this command
        response = await client.query(
            Collections.PROMPTS,
            QueryOptions(filter={"owui_id": f"/{command}"}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

        record = response.get_records()[0]
        prompt_data = record.model_dump()
        prompt_data["command"] = prompt_data.get("owui_id", prompt_data.get("command"))

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

    try:
        # Find the prompt by command
        response = await client.query(
            Collections.PROMPTS,
            QueryOptions(filter={"owui_id": f"/{command}"}, limit=1)
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
        updated_data["owui_id"] = f"/{command}"
        updated_data["user_id"] = prompt_data.get("user_id")

        updated_record = await client.update(Collections.PROMPTS, record.id, updated_data)

        # Convert back to PromptModel
        result_data = updated_record.model_dump()
        result_data["command"] = result_data.get("owui_id", result_data.get("command"))
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

    try:
        # Find the prompt by command
        response = await client.query(
            Collections.PROMPTS,
            QueryOptions(filter={"owui_id": f"/{command}"}, limit=1)
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
        return True

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )
