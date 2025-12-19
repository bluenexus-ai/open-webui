import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request

from open_webui.models.prompts import (
    PromptForm,
    PromptUserResponse,
    PromptModel,
    Prompts,
)
from open_webui.models.users import Users
from open_webui.constants import ERROR_MESSAGES
from open_webui.utils.auth import get_admin_user, get_verified_user
from open_webui.utils.access_control import has_access, has_permission
from open_webui.utils.bluenexus.config import is_bluenexus_data_storage_enabled
from open_webui.repositories import get_prompt_repository
from open_webui.config import BYPASS_ADMIN_ACCESS_CONTROL

log = logging.getLogger(__name__)

router = APIRouter()


############################
# GetPrompts
############################


@router.get("/", response_model=list[PromptModel])
async def get_prompts(user=Depends(get_verified_user)):
    if is_bluenexus_data_storage_enabled():
        repo = get_prompt_repository(user.id)
        prompts_data = await repo.get_list(user.id)

        prompts = []
        for prompt_data in prompts_data:
            # Apply access control - user always has access to their own prompts
            if user.role == "admin" and BYPASS_ADMIN_ACCESS_CONTROL:
                prompts.append(PromptModel(**prompt_data))
            elif prompt_data.get("user_id") == user.id:
                prompts.append(PromptModel(**prompt_data))
            elif has_access(user.id, "read", prompt_data.get("access_control")):
                prompts.append(PromptModel(**prompt_data))

        return prompts
    else:
        # PostgreSQL path - use existing Prompts model
        if user.role == "admin" and BYPASS_ADMIN_ACCESS_CONTROL:
            return Prompts.get_prompts()
        else:
            return Prompts.get_prompts_by_user_id(user.id)


@router.get("/list", response_model=list[PromptUserResponse])
async def get_prompt_list(user=Depends(get_verified_user)):
    if is_bluenexus_data_storage_enabled():
        repo = get_prompt_repository(user.id)
        prompts_data = await repo.get_list(user.id)

        # Collect all unique user_ids and batch fetch users
        user_ids = list(set(p.get("user_id") for p in prompts_data if p.get("user_id")))
        users = Users.get_users_by_user_ids(user_ids) if user_ids else []
        users_dict = {u.id: u for u in users}

        prompts = []
        for prompt_data in prompts_data:
            # Get user info for this prompt
            prompt_user = users_dict.get(prompt_data.get("user_id"))
            prompt_data["user"] = prompt_user.model_dump() if prompt_user else None

            # Apply access control for write permission - user always has access to their own prompts
            if user.role == "admin" and BYPASS_ADMIN_ACCESS_CONTROL:
                prompts.append(PromptUserResponse(**prompt_data))
            elif prompt_data.get("user_id") == user.id:
                prompts.append(PromptUserResponse(**prompt_data))
            elif has_access(user.id, "write", prompt_data.get("access_control")):
                prompts.append(PromptUserResponse(**prompt_data))

        return prompts
    else:
        # PostgreSQL path
        if user.role == "admin" and BYPASS_ADMIN_ACCESS_CONTROL:
            prompts = Prompts.get_prompts()
        else:
            prompts = Prompts.get_prompts_by_user_id(user.id)

        user_ids = list(set(p.user_id for p in prompts if p.user_id))
        users = Users.get_users_by_user_ids(user_ids) if user_ids else []
        users_dict = {u.id: u for u in users}

        result = []
        for prompt in prompts:
            prompt_user = users_dict.get(prompt.user_id)
            prompt_data = prompt.model_dump()
            prompt_data["user"] = prompt_user.model_dump() if prompt_user else None
            result.append(PromptUserResponse(**prompt_data))

        return result


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

    # Normalize command to always have leading slash
    command = form_data.command if form_data.command.startswith("/") else f"/{form_data.command}"

    if is_bluenexus_data_storage_enabled():
        repo = get_prompt_repository(user.id)

        # Check if prompt with this command already exists
        existing = await repo.get_by_command(command, user.id)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.COMMAND_TAKEN,
            )

        # Create new prompt
        prompt_data = form_data.model_dump()
        prompt_data["command"] = command

        result = await repo.create(user.id, prompt_data)
        return PromptModel(**result) if result else None
    else:
        # PostgreSQL path
        form_data.command = command
        prompt = Prompts.get_prompt_by_command(command)
        if prompt:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.COMMAND_TAKEN,
            )

        prompt = Prompts.insert_new_prompt(user.id, form_data)
        return prompt


############################
# GetPromptByCommand
############################


@router.get("/command/{command}", response_model=Optional[PromptModel])
async def get_prompt_by_command(command: str, user=Depends(get_verified_user)):
    full_command = f"/{command}"

    if is_bluenexus_data_storage_enabled():
        repo = get_prompt_repository(user.id)
        prompt_data = await repo.get_by_command(full_command, user.id)

        if not prompt_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

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
    else:
        # PostgreSQL path
        prompt = Prompts.get_prompt_by_command(full_command)
        if not prompt:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

        if (
            user.role == "admin"
            or prompt.user_id == user.id
            or has_access(user.id, "read", prompt.access_control)
        ):
            return prompt
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
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
    full_command = f"/{command}"

    if is_bluenexus_data_storage_enabled():
        repo = get_prompt_repository(user.id)

        # Get existing prompt first to check access
        existing = await repo.get_by_command(full_command, user.id)
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

        # Check access
        if (
            existing.get("user_id") != user.id
            and not has_access(user.id, "write", existing.get("access_control"))
            and user.role != "admin"
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
            )

        # Update prompt
        update_data = form_data.model_dump()
        result = await repo.update(full_command, user.id, update_data)
        return PromptModel(**result) if result else None
    else:
        # PostgreSQL path
        prompt = Prompts.get_prompt_by_command(full_command)
        if not prompt:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

        if (
            prompt.user_id != user.id
            and not has_access(user.id, "write", prompt.access_control)
            and user.role != "admin"
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
            )

        prompt = Prompts.update_prompt_by_command(full_command, form_data)
        return prompt


############################
# DeletePromptByCommand
############################


@router.delete("/command/{command}/delete", response_model=bool)
async def delete_prompt_by_command(command: str, user=Depends(get_verified_user)):
    full_command = f"/{command}"

    if is_bluenexus_data_storage_enabled():
        repo = get_prompt_repository(user.id)

        # Get existing prompt first to check access
        existing = await repo.get_by_command(full_command, user.id)
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

        # Check access
        if (
            existing.get("user_id") != user.id
            and not has_access(user.id, "write", existing.get("access_control"))
            and user.role != "admin"
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
            )

        # Delete prompt
        return await repo.delete(full_command, user.id)
    else:
        # PostgreSQL path
        prompt = Prompts.get_prompt_by_command(full_command)
        if not prompt:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

        if (
            prompt.user_id != user.id
            and not has_access(user.id, "write", prompt.access_control)
            and user.role != "admin"
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
            )

        return Prompts.delete_prompt_by_command(full_command)
