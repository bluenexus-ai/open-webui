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
from open_webui.models.folders import Folders

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MODELS"])

router = APIRouter()


############################
# Form Classes
############################


class TagForm(BaseModel):
    name: str


class TagFilterForm(TagForm):
    skip: Optional[int] = 0
    limit: Optional[int] = 50


class MessageForm(BaseModel):
    content: str


class EventForm(BaseModel):
    type: str
    data: dict


class ChatFolderIdForm(BaseModel):
    folder_id: Optional[str] = None


############################
# GetAllChats
############################


@router.get("/all", response_model=list[ChatResponse])
async def get_user_chats(user=Depends(get_verified_user)):
    try:
        repo = get_chat_repository(user.id)
        chats = await repo.get_list(
            user_id=user.id,
            page=1,
            limit=100,
            include_archived=False,
            include_pinned=True,
            include_folders=True,
        )
        return [ChatResponse(**chat) for chat in chats]

    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# GetAllArchivedChats
############################


@router.get("/all/archived", response_model=list[ChatResponse])
async def get_all_user_archived_chats(user=Depends(get_verified_user)):
    try:
        repo = get_chat_repository(user.id)
        chats = await repo.get_archived(user.id, page=1, limit=100)
        return [ChatResponse(**chat) for chat in chats]

    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# GetAllDbChats (Admin)
############################


@router.get("/all/db", response_model=list[ChatResponse])
async def get_all_user_chats_in_db(user=Depends(get_admin_user)):
    if not ENABLE_ADMIN_EXPORT:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    try:
        repo = get_chat_repository(user.id)
        chats = await repo.get_all()
        return [ChatResponse(**chat) for chat in chats]

    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# GetAllTags
############################


@router.get("/all/tags", response_model=list[TagModel])
async def get_all_user_tags(user=Depends(get_verified_user)):
    try:
        tags = Tags.get_tags_by_user_id(user.id)
        return tags

    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


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
# GetUserChatListByUserId (Admin)
############################


@router.get("/list/user/{user_id}", response_model=list[ChatTitleIdResponse])
async def get_user_chat_list_by_user_id(
    user_id: str,
    page: Optional[int] = None,
    query: Optional[str] = None,
    order_by: Optional[str] = None,
    direction: Optional[str] = None,
    user=Depends(get_admin_user),
):
    if not ENABLE_ADMIN_CHAT_ACCESS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    try:
        page_num = page if page is not None else 1
        repo = get_chat_repository(user.id)
        chats = await repo.get_by_user_id_admin(user_id, page_num, 60, query)

        return [ChatTitleIdResponse(**chat) for chat in chats]

    except Exception as e:
        log.exception(e)
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
    text: str = "",
    page: Optional[int] = None,
):
    try:
        repo = get_chat_repository(user.id)
        page_num = page if page is not None else 1

        chats = await repo.search(user.id, text, page=page_num, limit=60)
        return [ChatTitleIdResponse(**chat) for chat in chats]

    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# GetChatsByFolderId
############################


@router.get("/folder/{folder_id}", response_model=list[ChatResponse])
async def get_chats_by_folder_id(folder_id: str, user=Depends(get_verified_user)):
    try:
        # Get children folders
        folder_ids = [folder_id]
        children = Folders.get_children_folders_by_id_and_user_id(folder_id, user.id)
        folder_ids.extend([f.id for f in children])

        # Get chats in these folders using repository
        repo = get_chat_repository(user.id)
        chats = await repo.get_by_folder_ids(user.id, folder_ids)
        return [ChatResponse(**chat) for chat in chats]

    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# GetChatListByFolderId
############################


@router.get("/folder/{folder_id}/list")
async def get_chat_list_by_folder_id(
    folder_id: str, page: Optional[int] = 1, user=Depends(get_verified_user)
):
    try:
        limit = 60
        repo = get_chat_repository(user.id)
        chats = await repo.get_by_folder_id(user.id, folder_id, page, limit)

        return [
            {
                "title": chat.get("title"),
                "id": chat.get("id"),
                "updated_at": chat.get("updated_at"),
            }
            for chat in chats
        ]

    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# ArchiveAllChats
############################


@router.post("/archive/all", response_model=bool)
async def archive_all_chats(user=Depends(get_verified_user)):
    try:
        repo = get_chat_repository(user.id)
        return await repo.archive_all(user.id)

    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# UnarchiveAllChats
############################


@router.post("/unarchive/all", response_model=bool)
async def unarchive_all_chats(user=Depends(get_verified_user)):
    try:
        repo = get_chat_repository(user.id)
        return await repo.unarchive_all(user.id)

    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# GetChatsByTags
############################


@router.post("/tags", response_model=list[ChatTitleIdResponse])
async def get_user_chats_by_tag_name(
    form_data: TagFilterForm, user=Depends(get_verified_user)
):
    try:
        repo = get_chat_repository(user.id)
        chats = await repo.get_by_tag(user.id, form_data.name, form_data.skip, form_data.limit)
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
            chat_content = chat.get("chat", {})
            has_history = "history" in chat_content
            has_messages = "messages" in chat_content
            history_obj = chat_content.get("history", {})
            history_messages_type = type(history_obj.get("messages")).__name__ if history_obj else "NoHistory"
            log.info(f"[get_chat_by_id] chat_id={id}, has_history={has_history}, has_messages={has_messages}, history_messages_type={history_messages_type}")
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

        log.info(f"[update_chat_by_id] chat_id={id}, title={form_data.chat.get('title') if isinstance(form_data.chat, dict) else 'N/A'}")
        chat = await repo.update(id, user.id, chat_data)

        if chat:
            log.info(f"[update_chat_by_id] Updated chat_id={id}, new_title={chat.get('title')}")
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
            log.info(f"[share_chat_by_id] chat_id={id}, share_id={chat.get('share_id')}")
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


############################
# UpdateChatFolderById
############################


@router.post("/{id}/folder", response_model=Optional[ChatResponse])
async def update_chat_folder_id_by_id(
    id: str, form_data: ChatFolderIdForm, user=Depends(get_verified_user)
):
    try:
        repo = get_chat_repository(user.id)
        chat = await repo.update_folder_id(id, user.id, form_data.folder_id)
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
# GetChatPinnedStatusById
############################


@router.get("/{id}/pinned", response_model=Optional[bool])
async def get_chat_pinned_status_by_id(id: str, user=Depends(get_verified_user)):
    try:
        repo = get_chat_repository(user.id)
        chat = await repo.get_by_id(id, user.id)
        if chat:
            return chat.get("pinned", False)

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
# GetChatTagsById
############################


@router.get("/{id}/tags", response_model=list[TagModel])
async def get_chat_tags_by_id(id: str, user=Depends(get_verified_user)):
    try:
        repo = get_chat_repository(user.id)
        tag_names = await repo.get_tags(id, user.id)

        if tag_names is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail=ERROR_MESSAGES.NOT_FOUND
            )

        tags = []
        for tag_name in tag_names:
            tag = Tags.get_tag_by_name_and_user_id(tag_name, user.id)
            if tag:
                tags.append(tag)

        return tags

    except HTTPException:
        raise
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# AddChatTagById
############################


@router.post("/{id}/tags", response_model=list[TagModel])
async def add_tag_by_id_and_tag_name(
    id: str, form_data: TagForm, user=Depends(get_verified_user)
):
    try:
        tag_id = form_data.name.replace(" ", "_").lower()

        if tag_id == "none":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.DEFAULT("Tag name cannot be 'None'"),
            )

        # Get or create tag in Tags table
        tag = Tags.get_tag_by_name_and_user_id(tag_id, user.id)
        if not tag:
            tag = Tags.insert_new_tag(tag_id, user.id)

        # Add tag to chat via repository
        repo = get_chat_repository(user.id)
        tags = await repo.add_tag(id, user.id, tag_id)

        if not tags:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail=ERROR_MESSAGES.DEFAULT()
            )

        # Return updated tags as TagModel objects
        result_tags = []
        for tag_name in tags:
            t = Tags.get_tag_by_name_and_user_id(tag_name, user.id)
            if t:
                result_tags.append(t)

        return result_tags

    except HTTPException:
        raise
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# DeleteChatTagById
############################


@router.delete("/{id}/tags", response_model=list[TagModel])
async def delete_tag_by_id_and_tag_name(
    id: str, form_data: TagForm, user=Depends(get_verified_user)
):
    try:
        tag_id = form_data.name.replace(" ", "_").lower()

        # Remove tag from chat via repository
        repo = get_chat_repository(user.id)
        tags = await repo.remove_tag(id, user.id, tag_id)

        # Return updated tags as TagModel objects
        result_tags = []
        for tag_name in tags:
            t = Tags.get_tag_by_name_and_user_id(tag_name, user.id)
            if t:
                result_tags.append(t)

        return result_tags

    except HTTPException:
        raise
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# DeleteAllChatTagsById
############################


@router.delete("/{id}/tags/all", response_model=Optional[bool])
async def delete_all_tags_by_id(id: str, user=Depends(get_verified_user)):
    try:
        repo = get_chat_repository(user.id)
        result = await repo.clear_tags(id, user.id)

        if not result:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail=ERROR_MESSAGES.DEFAULT()
            )

        return True

    except HTTPException:
        raise
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# UpdateChatMessageById
############################


@router.post("/{id}/messages/{message_id}", response_model=Optional[ChatResponse])
async def update_chat_message_by_id(
    id: str, message_id: str, form_data: MessageForm, user=Depends(get_verified_user)
):
    try:
        repo = get_chat_repository(user.id)
        updated_chat = await repo.update_message(id, user.id, message_id, form_data.content)

        if not updated_chat:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
            )

        # Emit event
        event_emitter = get_event_emitter(
            {
                "user_id": user.id,
                "chat_id": id,
            }
        )
        if event_emitter:
            await event_emitter(
                {
                    "type": "chat:message:update",
                    "data": {
                        "message_id": message_id,
                        "content": form_data.content,
                    },
                }
            )

        return ChatResponse(**updated_chat)

    except HTTPException:
        raise
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# UpdateChatMessageEventById
############################


@router.post("/{id}/messages/{message_id}/event", response_model=Optional[bool])
async def update_chat_message_event_by_id(
    id: str,
    message_id: str,
    form_data: EventForm,
    user=Depends(get_verified_user),
):
    try:
        # Verify access via repository
        repo = get_chat_repository(user.id)
        chat = await repo.get_by_id(id, user.id)

        if not chat:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
            )

        # Emit event
        event_emitter = get_event_emitter(
            {
                "user_id": user.id,
                "chat_id": id,
            }
        )
        if event_emitter:
            await event_emitter(
                {
                    "type": form_data.type,
                    "data": form_data.data,
                }
            )
            return True

        return False

    except HTTPException:
        raise
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )
