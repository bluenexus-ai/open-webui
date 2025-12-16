"""
Chat Router - Refactored to use Repository Pattern

Uses BLUENEXUS_DATA_STORAGE env to switch between PostgreSQL and BlueNexus storage.
"""

import logging
from typing import Optional

from open_webui.socket.main import get_event_emitter
from open_webui.models.chats import (
    ChatForm,
    ChatImportForm,
    ChatResponse,
    ChatTitleIdResponse,
    SharedChatMappings,
    Chats,
)
from open_webui.models.tags import TagModel, Tags

from open_webui.config import ENABLE_ADMIN_CHAT_ACCESS, ENABLE_ADMIN_EXPORT
from open_webui.constants import ERROR_MESSAGES
from open_webui.env import SRC_LOG_LEVELS
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from open_webui.utils.auth import get_admin_user, get_verified_user
from open_webui.utils.access_control import has_permission
from open_webui.repositories import get_chat_repository

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MODELS"])

router = APIRouter()


############################
# GetChatList
############################


@router.get("/", response_model=list[ChatTitleIdResponse])
@router.get("/list", response_model=list[ChatTitleIdResponse])
async def get_session_user_chat_list(
    user=Depends(get_verified_user),
    page: Optional[int] = None,
    include_pinned: Optional[bool] = False,
    include_folders: Optional[bool] = False,
):
    try:
        log.info(f"[get_session_user_chat_list] user={user.id}, page={page}, include_pinned={include_pinned}, include_folders={include_folders}")
        repo = get_chat_repository(user.id)
        page_num = page if page is not None else 1

        chats = await repo.get_list(
            user_id=user.id,
            page=page_num,
            limit=60,
            include_archived=False,
            include_pinned=include_pinned,
            include_folders=include_folders,
        )

        log.info(f"[get_session_user_chat_list] returning {len(chats)} chats")
        return [ChatTitleIdResponse(**chat) for chat in chats]

    except Exception as e:
        log.exception(f"[get_session_user_chat_list] ERROR: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# DeleteAllChats
############################


@router.delete("/", response_model=bool)
async def delete_all_user_chats(request: Request, user=Depends(get_verified_user)):
    if user.role == "user" and not has_permission(
        user.id, "chat.delete", request.app.state.config.USER_PERMISSIONS
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    try:
        repo = get_chat_repository(user.id)
        return await repo.delete_all_by_user(user.id)
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(),
        )


############################
# GetArchivedChats
############################


@router.get("/archived", response_model=list[ChatTitleIdResponse])
async def get_user_archived_chats(
    user=Depends(get_verified_user),
    page: Optional[int] = None,
):
    try:
        repo = get_chat_repository(user.id)
        page_num = page if page is not None else 1

        chats = await repo.get_archived(user.id, page=page_num, limit=60)
        return [ChatTitleIdResponse(**chat) for chat in chats]

    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# GetPinnedChats
############################


@router.get("/pinned", response_model=list[ChatTitleIdResponse])
async def get_user_pinned_chats(user=Depends(get_verified_user)):
    try:
        repo = get_chat_repository(user.id)
        chats = await repo.get_pinned(user.id)
        return [ChatTitleIdResponse(**chat) for chat in chats]

    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# SearchChats
############################


@router.get("/search", response_model=list[ChatTitleIdResponse])
async def search_user_chats(
    user=Depends(get_verified_user),
    query: str = "",
    page: Optional[int] = None,
):
    try:
        repo = get_chat_repository(user.id)
        page_num = page if page is not None else 1

        chats = await repo.search(user.id, query, page=page_num, limit=60)
        return [ChatTitleIdResponse(**chat) for chat in chats]

    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# GetSharedChatById
############################


@router.get("/share/{share_id}", response_model=Optional[ChatResponse])
async def get_shared_chat_by_id(share_id: str, user=Depends(get_verified_user)):
    if user.role == "pending":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=ERROR_MESSAGES.NOT_FOUND
        )

    try:
        # Look up owner from mapping and use their repository
        owner_user_id = SharedChatMappings.get_owner_user_id(share_id)

        if owner_user_id:
            repo = get_chat_repository(owner_user_id)
        else:
            repo = get_chat_repository(user.id)

        chat = await repo.get_by_share_id(share_id)

        if chat:
            return ChatResponse(**chat)

        # Admin fallback
        if user.role == "admin" and ENABLE_ADMIN_CHAT_ACCESS:
            repo = get_chat_repository(user.id)
            chat = await repo.get_by_id(share_id, user.id)
            if chat:
                return ChatResponse(**chat)

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=ERROR_MESSAGES.NOT_FOUND
        )

    except HTTPException:
        raise
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unavailable: {str(e)}",
        )


############################
# GetChatById
############################


@router.get("/{id}", response_model=Optional[ChatResponse])
async def get_chat_by_id(id: str, user=Depends(get_verified_user)):
    try:
        repo = get_chat_repository(user.id)
        chat = await repo.get_by_id(id, user.id)

        if chat:
            return ChatResponse(**chat)

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=ERROR_MESSAGES.NOT_FOUND
        )

    except HTTPException:
        raise
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# CreateNewChat
############################


@router.post("/new", response_model=Optional[ChatResponse])
async def create_new_chat(form_data: ChatForm, user=Depends(get_verified_user)):
    try:
        repo = get_chat_repository(user.id)

        chat_data = {
            "chat": form_data.chat,
            "folder_id": form_data.folder_id,
        }

        chat = await repo.create(user.id, chat_data)

        if chat:
            return ChatResponse(**chat)

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(),
        )

    except HTTPException:
        raise
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# UpdateChatById
############################


@router.post("/{id}", response_model=Optional[ChatResponse])
async def update_chat_by_id(id: str, form_data: ChatForm, user=Depends(get_verified_user)):
    try:
        repo = get_chat_repository(user.id)

        chat_data = {
            "chat": form_data.chat,
        }

        chat = await repo.update(id, user.id, chat_data)

        if chat:
            return ChatResponse(**chat)

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    except HTTPException:
        raise
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# DeleteChatById
############################


@router.delete("/{id}", response_model=bool)
async def delete_chat_by_id(request: Request, id: str, user=Depends(get_verified_user)):
    if user.role == "user" and not has_permission(
        user.id, "chat.delete", request.app.state.config.USER_PERMISSIONS
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    try:
        repo = get_chat_repository(user.id)
        result = await repo.delete(id, user.id)

        if result:
            return True

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    except HTTPException:
        raise
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# CloneChatById
############################


@router.post("/{id}/clone", response_model=Optional[ChatResponse])
async def clone_chat_by_id(id: str, user=Depends(get_verified_user)):
    try:
        repo = get_chat_repository(user.id)
        chat = await repo.clone(id, user.id)

        if chat:
            return ChatResponse(**chat)

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=ERROR_MESSAGES.DEFAULT()
        )

    except HTTPException:
        raise
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# CloneSharedChatById
############################


@router.post("/{id}/clone/shared", response_model=Optional[ChatResponse])
async def clone_shared_chat_by_id(id: str, user=Depends(get_verified_user)):
    try:
        # Look up owner from mapping
        owner_user_id = SharedChatMappings.get_owner_user_id(id)

        if owner_user_id:
            read_repo = get_chat_repository(owner_user_id)
        else:
            read_repo = get_chat_repository(user.id)

        # Get shared chat
        shared_chat = await read_repo.get_by_share_id(id)

        if not shared_chat:
            # Admin fallback
            if user.role == "admin" and ENABLE_ADMIN_CHAT_ACCESS:
                shared_chat = await read_repo.get_by_id(id, owner_user_id or user.id)

        if not shared_chat:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail=ERROR_MESSAGES.DEFAULT()
            )

        # Clone to user's own storage
        write_repo = get_chat_repository(user.id)
        original_id = shared_chat.get("id")
        chat_content = shared_chat.get("chat", {})

        clone_data = {
            "chat": {
                **chat_content,
                "originalChatId": original_id,
                "branchPointMessageId": chat_content.get("history", {}).get("currentId"),
                "title": f"Clone of {shared_chat.get('title', 'Chat')}",
            },
            "meta": shared_chat.get("meta", {}),
            "folder_id": shared_chat.get("folder_id"),
        }

        chat = await write_repo.create(user.id, clone_data)

        if chat:
            return ChatResponse(**chat)

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )

    except HTTPException:
        raise
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# ArchiveChatById
############################


@router.post("/{id}/archive", response_model=Optional[ChatResponse])
async def archive_chat_by_id(id: str, user=Depends(get_verified_user)):
    try:
        repo = get_chat_repository(user.id)
        chat = await repo.archive(id, user.id)

        if chat:
            return ChatResponse(**chat)

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=ERROR_MESSAGES.DEFAULT()
        )

    except HTTPException:
        raise
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# UnarchiveChatById
############################


@router.post("/{id}/unarchive", response_model=Optional[ChatResponse])
async def unarchive_chat_by_id(id: str, user=Depends(get_verified_user)):
    try:
        repo = get_chat_repository(user.id)
        chat = await repo.unarchive(id, user.id)

        if chat:
            return ChatResponse(**chat)

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=ERROR_MESSAGES.DEFAULT()
        )

    except HTTPException:
        raise
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# PinChatById
############################


@router.post("/{id}/pin", response_model=Optional[ChatResponse])
async def pin_chat_by_id(id: str, user=Depends(get_verified_user)):
    try:
        repo = get_chat_repository(user.id)
        chat = await repo.pin(id, user.id)

        if chat:
            return ChatResponse(**chat)

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=ERROR_MESSAGES.DEFAULT()
        )

    except HTTPException:
        raise
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# UnpinChatById
############################


@router.post("/{id}/unpin", response_model=Optional[ChatResponse])
async def unpin_chat_by_id(id: str, user=Depends(get_verified_user)):
    try:
        repo = get_chat_repository(user.id)
        chat = await repo.unpin(id, user.id)

        if chat:
            return ChatResponse(**chat)

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=ERROR_MESSAGES.DEFAULT()
        )

    except HTTPException:
        raise
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# ShareChatById
############################


@router.post("/{id}/share", response_model=Optional[ChatResponse])
async def share_chat_by_id(request: Request, id: str, user=Depends(get_verified_user)):
    if (user.role != "admin") and (
        not has_permission(
            user.id, "chat.share", request.app.state.config.USER_PERMISSIONS
        )
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    try:
        repo = get_chat_repository(user.id)
        chat = await repo.share(id, user.id)

        if chat:
            return ChatResponse(**chat)

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    except HTTPException:
        raise
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# DeleteSharedChatById
############################


@router.delete("/{id}/share", response_model=Optional[bool])
async def delete_shared_chat_by_id(id: str, user=Depends(get_verified_user)):
    try:
        repo = get_chat_repository(user.id)
        result = await repo.unshare(id, user.id)
        return result

    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# ImportChat (simplified)
############################


@router.post("/import", response_model=Optional[ChatResponse])
async def import_chat(form_data: ChatImportForm, user=Depends(get_verified_user)):
    try:
        repo = get_chat_repository(user.id)

        chat_data = {
            "chat": form_data.chat,
            "folder_id": form_data.folder_id,
            "meta": form_data.meta,
            "pinned": form_data.pinned,
        }

        chat = await repo.create(user.id, chat_data)

        if chat:
            return ChatResponse(**chat)

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(),
        )

    except HTTPException:
        raise
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )
