import logging
from typing import Optional


from open_webui.socket.main import get_event_emitter
from open_webui.models.chats import (
    ChatForm,
    ChatImportForm,
    ChatResponse,
    ChatTitleIdResponse,
)
from open_webui.models.tags import TagModel

from open_webui.config import ENABLE_ADMIN_CHAT_ACCESS, ENABLE_ADMIN_EXPORT
from open_webui.constants import ERROR_MESSAGES
from open_webui.env import SRC_LOG_LEVELS
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel


from open_webui.utils.auth import get_admin_user, get_verified_user
from open_webui.utils.access_control import has_permission
from open_webui.utils.bluenexus.collections import Collections
from open_webui.utils.bluenexus.factory import get_bluenexus_client_for_user
from open_webui.utils.bluenexus.types import QueryOptions, BlueNexusError, SortBy, SortOrder

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MODELS"])

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


def normalize_chat_data(chat_data: dict) -> dict:
    """
    Normalize BlueNexus record data to Open WebUI format.
    - Maps owui_id → id
    - Maps createdAt → created_at (converts datetime to int timestamp)
    - Maps updatedAt → updated_at (converts datetime to int timestamp)
    """
    import time

    # Map owui_id to id
    if "owui_id" in chat_data:
        chat_data["id"] = chat_data.get("owui_id", chat_data.get("id"))

    # Handle timestamp conversion (BlueNexus uses camelCase datetime, OWUI uses snake_case int)
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
    client = get_client_or_raise(user.id)

    try:
        # Build filter
        filter_params = {"user_id": user.id, "archived": False}
        if not include_pinned:
            filter_params["pinned"] = False
        if not include_folders:
            filter_params["folder_id"] = None

        # Calculate pagination
        limit = 60
        page_num = page if page is not None else 1

        response = await client.query(
            Collections.CHATS,
            QueryOptions(
                filter=filter_params,
                sort_by=SortBy.UPDATED_AT,
                sort_order=SortOrder.DESC,
                limit=limit,
                page=page_num,
            )
        )

        chats = []
        for record in response.get_records():
            chat_data = record.model_dump()
            normalize_chat_data(chat_data)
            chats.append(ChatTitleIdResponse(**chat_data))

        return chats

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )
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

    client = get_client_or_raise(user.id)

    try:
        # Delete all chats for this user using pagination loop
        while True:
            response = await client.query(
                Collections.CHATS,
                QueryOptions(filter={"user_id": user.id}, limit=100)
            )

            if not response.data or len(response.data) == 0:
                break

            # Delete each chat in this batch
            for record in response.get_records():
                await client.delete(Collections.CHATS, record.id)

        return True

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# GetUserChatList
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

    # Admin uses their own client to query another user's data
    client = get_client_or_raise(user.id)

    try:
        page_num = page if page is not None else 1
        limit = 60

        # Build filter - include archived for admin view
        filter_params = {"user_id": user_id}
        if query:
            filter_params["title"] = {"$contains": query}

        # Determine sort order
        sort_by = SortBy.UPDATED_AT
        if order_by == "created_at":
            sort_by = SortBy.CREATED_AT

        sort_order = SortOrder.DESC
        if direction == "asc":
            sort_order = SortOrder.ASC

        response = await client.query(
            Collections.CHATS,
            QueryOptions(
                filter=filter_params,
                sort_by=sort_by,
                sort_order=sort_order,
                limit=limit,
                page=page_num,
            )
        )

        chats = []
        for record in response.get_records():
            chat_data = record.model_dump()
            normalize_chat_data(chat_data)
            chats.append(ChatTitleIdResponse(**chat_data))

        return chats

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# CreateNewChat
############################


@router.post("/new", response_model=Optional[ChatResponse])
async def create_new_chat(form_data: ChatForm, user=Depends(get_verified_user)):
    import uuid

    client = get_client_or_raise(user.id)

    try:
        # Generate a new chat ID
        chat_id = str(uuid.uuid4())

        # Build chat data
        chat_data = {
            "owui_id": chat_id,
            "user_id": user.id,
            "title": form_data.chat.get("title", "New Chat") if form_data.chat else "New Chat",
            "chat": form_data.chat or {},
            "meta": {},
            "pinned": False,
            "archived": False,
            "folder_id": None,
            "share_id": None,
        }

        record = await client.create(Collections.CHATS, chat_data)

        # Convert back to ChatResponse
        result_data = record.model_dump()
        normalize_chat_data(result_data)
        return ChatResponse(**result_data)

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# ImportChat
############################


@router.post("/import", response_model=Optional[ChatResponse])
async def import_chat(form_data: ChatImportForm, user=Depends(get_verified_user)):
    import uuid

    client = get_client_or_raise(user.id)

    try:
        # Generate a new chat ID
        chat_id = str(uuid.uuid4())

        # Build chat data from import form
        chat_data = {
            "owui_id": chat_id,
            "user_id": user.id,
            "title": form_data.chat.get("title", "Imported Chat") if form_data.chat else "Imported Chat",
            "chat": form_data.chat or {},
            "meta": form_data.meta or {},
            "pinned": form_data.pinned if form_data.pinned is not None else False,
            "archived": False,
            "folder_id": form_data.folder_id,
            "share_id": None,
        }

        record = await client.create(Collections.CHATS, chat_data)

        # Handle tags - create them in BlueNexus if needed
        result_data = record.model_dump()
        tags = result_data.get("meta", {}).get("tags", [])
        for tag_id in tags:
            tag_id = tag_id.replace(" ", "_").lower()
            tag_name = " ".join([word.capitalize() for word in tag_id.split("_")])
            if tag_id != "none":
                # Check if tag exists
                existing_tag = await client.query(
                    Collections.TAGS,
                    QueryOptions(filter={"name": tag_name, "user_id": user.id}, limit=1)
                )
                if not existing_tag.data or len(existing_tag.data) == 0:
                    # Create new tag
                    await client.create(Collections.TAGS, {
                        "owui_id": tag_id,
                        "name": tag_name,
                        "user_id": user.id,
                    })

        # Convert back to ChatResponse
        normalize_chat_data(result_data)
        return ChatResponse(**result_data)

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# GetChats
############################


@router.get("/search", response_model=list[ChatTitleIdResponse])
async def search_user_chats(
    text: str, page: Optional[int] = None, user=Depends(get_verified_user)
):
    client = get_client_or_raise(user.id)

    try:
        page_num = page if page is not None else 1
        limit = 60

        # Build filter for search
        filter_params = {"user_id": user.id}

        # Check if searching by tag
        words = text.strip().split(" ")
        if len(words) == 1 and words[0].startswith("tag:"):
            tag_id = words[0].replace("tag:", "")
            filter_params["meta.tags"] = {"$contains": tag_id}
        else:
            # Search in title
            filter_params["title"] = {"$contains": text}

        response = await client.query(
            Collections.CHATS,
            QueryOptions(
                filter=filter_params,
                sort_by=SortBy.UPDATED_AT,
                sort_order=SortOrder.DESC,
                limit=limit,
                page=page_num,
            )
        )

        chats = []
        for record in response.get_records():
            chat_data = record.model_dump()
            normalize_chat_data(chat_data)
            chats.append(ChatTitleIdResponse(**chat_data))

        # Delete tag if no chat is found for tag search
        if page_num == 1 and len(words) == 1 and words[0].startswith("tag:"):
            tag_id = words[0].replace("tag:", "")
            if len(chats) == 0:
                # Check if tag exists and delete it
                existing_tag = await client.query(
                    Collections.TAGS,
                    QueryOptions(filter={"name": tag_id, "user_id": user.id}, limit=1)
                )
                if existing_tag.data and len(existing_tag.data) > 0:
                    log.debug(f"deleting tag: {tag_id}")
                    tag_record = existing_tag.get_records()[0]
                    await client.delete(Collections.TAGS, tag_record.id)

        return chats

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# GetChatsByFolderId
############################


@router.get("/folder/{folder_id}", response_model=list[ChatResponse])
async def get_chats_by_folder_id(folder_id: str, user=Depends(get_verified_user)):
    client = get_client_or_raise(user.id)

    try:
        # Get children folders from BlueNexus
        folder_ids = [folder_id]
        folders_response = await client.query(
            Collections.FOLDERS,
            QueryOptions(filter={"parent_id": folder_id, "user_id": user.id}, limit=100)
        )
        for folder_record in folders_response.get_records():
            folder_ids.append(folder_record.model_dump().get("owui_id", folder_record.id))

        # Query chats in these folders
        response = await client.query(
            Collections.CHATS,
            QueryOptions(
                filter={"user_id": user.id, "folder_id": {"$in": folder_ids}},
                sort_by=SortBy.UPDATED_AT,
                sort_order=SortOrder.DESC,
                limit=100,
            )
        )

        chats = []
        for record in response.get_records():
            chat_data = record.model_dump()
            normalize_chat_data(chat_data)
            chats.append(ChatResponse(**chat_data))

        return chats

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


@router.get("/folder/{folder_id}/list")
async def get_chat_list_by_folder_id(
    folder_id: str, page: Optional[int] = 1, user=Depends(get_verified_user)
):
    client = get_client_or_raise(user.id)

    try:
        limit = 60

        response = await client.query(
            Collections.CHATS,
            QueryOptions(
                filter={"user_id": user.id, "folder_id": folder_id},
                sort_by=SortBy.UPDATED_AT,
                sort_order=SortOrder.DESC,
                limit=limit,
                page=page,
            )
        )

        chats = []
        for record in response.get_records():
            chat_data = record.model_dump()
            chats.append({
                "title": chat_data.get("title"),
                "id": chat_data.get("owui_id", chat_data.get("id")),
                "updated_at": chat_data.get("updatedAt") or chat_data.get("updated_at"),
            })

        return chats

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )
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
    client = get_client_or_raise(user.id)

    try:
        response = await client.query(
            Collections.CHATS,
            QueryOptions(
                filter={"user_id": user.id, "pinned": True},
                sort_by=SortBy.UPDATED_AT,
                sort_order=SortOrder.DESC,
                limit=100,
            )
        )

        chats = []
        for record in response.get_records():
            chat_data = record.model_dump()
            normalize_chat_data(chat_data)
            chats.append(ChatTitleIdResponse(**chat_data))

        return chats

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# GetChats
############################


@router.get("/all", response_model=list[ChatResponse])
async def get_user_chats(user=Depends(get_verified_user)):
    client = get_client_or_raise(user.id)

    try:
        response = await client.query(
            Collections.CHATS,
            QueryOptions(
                filter={"user_id": user.id, "archived": False},
                sort_by=SortBy.UPDATED_AT,
                sort_order=SortOrder.DESC,
                limit=100,
            )
        )

        chats = []
        for record in response.get_records():
            chat_data = record.model_dump()
            normalize_chat_data(chat_data)
            chats.append(ChatResponse(**chat_data))

        return chats

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# GetArchivedChats
############################


@router.get("/all/archived", response_model=list[ChatResponse])
async def get_user_archived_chats(user=Depends(get_verified_user)):
    client = get_client_or_raise(user.id)

    try:
        response = await client.query(
            Collections.CHATS,
            QueryOptions(
                filter={"user_id": user.id, "archived": True},
                sort_by=SortBy.UPDATED_AT,
                sort_order=SortOrder.DESC,
                limit=100,
            )
        )

        chats = []
        for record in response.get_records():
            chat_data = record.model_dump()
            normalize_chat_data(chat_data)
            chats.append(ChatResponse(**chat_data))

        return chats

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# GetAllTags
############################


@router.get("/all/tags", response_model=list[TagModel])
async def get_all_user_tags(user=Depends(get_verified_user)):
    client = get_client_or_raise(user.id)

    try:
        response = await client.query(
            Collections.TAGS,
            QueryOptions(
                filter={"user_id": user.id},
                limit=100,
            )
        )

        tags = []
        for record in response.get_records():
            tag_data = record.model_dump()
            tag_data["id"] = tag_data.get("owui_id", tag_data.get("id"))
            tags.append(TagModel(**tag_data))

        return tags

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=ERROR_MESSAGES.DEFAULT()
        )


############################
# GetAllChatsInDB
############################


@router.get("/all/db", response_model=list[ChatResponse])
async def get_all_user_chats_in_db(user=Depends(get_admin_user)):
    if not ENABLE_ADMIN_EXPORT:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    client = get_client_or_raise(user.id)

    try:
        # Admin can query all chats (no user_id filter)
        response = await client.query(
            Collections.CHATS,
            QueryOptions(
                sort_by=SortBy.UPDATED_AT,
                sort_order=SortOrder.DESC,
                limit=100,
            )
        )

        chats = []
        for record in response.get_records():
            chat_data = record.model_dump()
            normalize_chat_data(chat_data)
            chats.append(ChatResponse(**chat_data))

        return chats

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# GetArchivedChats
############################


@router.get("/archived", response_model=list[ChatTitleIdResponse])
async def get_archived_session_user_chat_list(
    page: Optional[int] = None,
    query: Optional[str] = None,
    order_by: Optional[str] = None,
    direction: Optional[str] = None,
    user=Depends(get_verified_user),
):
    client = get_client_or_raise(user.id)

    try:
        page_num = page if page is not None else 1
        limit = 60

        # Build filter
        filter_params = {"user_id": user.id, "archived": True}
        if query:
            filter_params["title"] = {"$contains": query}

        # Determine sort order
        sort_by = SortBy.UPDATED_AT
        if order_by == "created_at":
            sort_by = SortBy.CREATED_AT

        sort_order = SortOrder.DESC
        if direction == "asc":
            sort_order = SortOrder.ASC

        response = await client.query(
            Collections.CHATS,
            QueryOptions(
                filter=filter_params,
                sort_by=sort_by,
                sort_order=sort_order,
                limit=limit,
                page=page_num,
            )
        )

        chats = []
        for record in response.get_records():
            chat_data = record.model_dump()
            normalize_chat_data(chat_data)
            chats.append(ChatTitleIdResponse(**chat_data))

        return chats

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# ArchiveAllChats
############################


@router.post("/archive/all", response_model=bool)
async def archive_all_chats(user=Depends(get_verified_user)):
    client = get_client_or_raise(user.id)

    try:
        # Archive all chats for this user using pagination loop
        while True:
            response = await client.query(
                Collections.CHATS,
                QueryOptions(filter={"user_id": user.id, "archived": False}, limit=100)
            )

            if not response.data or len(response.data) == 0:
                break

            # Update each chat in this batch to archived
            for record in response.get_records():
                chat_data = record.model_dump()
                chat_data["archived"] = True
                await client.update(Collections.CHATS, record.id, chat_data)

        return True

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# UnarchiveAllChats
############################


@router.post("/unarchive/all", response_model=bool)
async def unarchive_all_chats(user=Depends(get_verified_user)):
    client = get_client_or_raise(user.id)

    try:
        # Unarchive all chats for this user using pagination loop
        while True:
            response = await client.query(
                Collections.CHATS,
                QueryOptions(filter={"user_id": user.id, "archived": True}, limit=100)
            )

            if not response.data or len(response.data) == 0:
                break

            # Update each chat in this batch to unarchived
            for record in response.get_records():
                chat_data = record.model_dump()
                chat_data["archived"] = False
                await client.update(Collections.CHATS, record.id, chat_data)

        return True

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
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

    client = get_client_or_raise(user.id)

    try:
        # Build query based on user role
        if user.role == "user" or (user.role == "admin" and not ENABLE_ADMIN_CHAT_ACCESS):
            # Query by share_id
            response = await client.query(
                Collections.CHATS,
                QueryOptions(filter={"share_id": share_id}, limit=1)
            )
        elif user.role == "admin" and ENABLE_ADMIN_CHAT_ACCESS:
            # Admin can query by owui_id directly
            response = await client.query(
                Collections.CHATS,
                QueryOptions(filter={"owui_id": share_id}, limit=1)
            )

        if response.data and len(response.data) > 0:
            record = response.get_records()[0]
            chat_data = record.model_dump()
            normalize_chat_data(chat_data)
            return ChatResponse(**chat_data)
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail=ERROR_MESSAGES.NOT_FOUND
            )

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# GetChatsByTags
############################


class TagForm(BaseModel):
    name: str


class TagFilterForm(TagForm):
    skip: Optional[int] = 0
    limit: Optional[int] = 50


@router.post("/tags", response_model=list[ChatTitleIdResponse])
async def get_user_chat_list_by_tag_name(
    form_data: TagFilterForm, user=Depends(get_verified_user)
):
    client = get_client_or_raise(user.id)

    try:
        # Calculate page from skip/limit
        page = (form_data.skip // form_data.limit) + 1 if form_data.limit > 0 else 1

        response = await client.query(
            Collections.CHATS,
            QueryOptions(
                filter={"user_id": user.id, "meta.tags": {"$contains": form_data.name}},
                sort_by=SortBy.UPDATED_AT,
                sort_order=SortOrder.DESC,
                limit=form_data.limit,
                page=page,
            )
        )

        chats = []
        for record in response.get_records():
            chat_data = record.model_dump()
            normalize_chat_data(chat_data)
            chats.append(ChatTitleIdResponse(**chat_data))

        # Delete tag if no chats found
        if len(chats) == 0:
            existing_tag = await client.query(
                Collections.TAGS,
                QueryOptions(filter={"name": form_data.name, "user_id": user.id}, limit=1)
            )
            if existing_tag.data and len(existing_tag.data) > 0:
                tag_record = existing_tag.get_records()[0]
                await client.delete(Collections.TAGS, tag_record.id)

        return chats

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# GetChatById
############################


@router.get("/{id}", response_model=Optional[ChatResponse])
async def get_chat_by_id(id: str, user=Depends(get_verified_user)):
    client = get_client_or_raise(user.id)

    try:
        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"owui_id": id, "user_id": user.id}, limit=1)
        )

        if response.data and len(response.data) > 0:
            record = response.get_records()[0]
            chat_data = record.model_dump()
            normalize_chat_data(chat_data)
            return ChatResponse(**chat_data)
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail=ERROR_MESSAGES.NOT_FOUND
            )

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# UpdateChatById
############################


@router.post("/{id}", response_model=Optional[ChatResponse])
async def update_chat_by_id(
    id: str, form_data: ChatForm, user=Depends(get_verified_user)
):
    client = get_client_or_raise(user.id)

    try:
        # Find the chat
        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"owui_id": id, "user_id": user.id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
            )

        record = response.get_records()[0]
        chat_data = record.model_dump()

        # Merge chat content
        updated_chat_content = {**chat_data.get("chat", {}), **form_data.chat}
        chat_data["chat"] = updated_chat_content

        # Update title if present in chat content
        if "title" in updated_chat_content:
            chat_data["title"] = updated_chat_content["title"]

        # Update in BlueNexus
        updated_record = await client.update(Collections.CHATS, record.id, chat_data)

        # Convert back to ChatResponse
        result_data = updated_record.model_dump()
        normalize_chat_data(result_data)
        return ChatResponse(**result_data)

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# UpdateChatMessageById
############################
class MessageForm(BaseModel):
    content: str


@router.post("/{id}/messages/{message_id}", response_model=Optional[ChatResponse])
async def update_chat_message_by_id(
    id: str, message_id: str, form_data: MessageForm, user=Depends(get_verified_user)
):
    client = get_client_or_raise(user.id)

    try:
        # Find the chat
        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"owui_id": id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
            )

        record = response.get_records()[0]
        chat_data = record.model_dump()

        # Check access
        if chat_data.get("user_id") != user.id and user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
            )

        # Update message in chat history
        chat_content = chat_data.get("chat", {})
        history = chat_content.get("history", {})
        messages = history.get("messages", {})

        if message_id in messages:
            messages[message_id]["content"] = form_data.content
        else:
            messages[message_id] = {"content": form_data.content}

        history["messages"] = messages
        chat_content["history"] = history
        chat_data["chat"] = chat_content

        # Update in BlueNexus
        updated_record = await client.update(Collections.CHATS, record.id, chat_data)

        # Emit event
        event_emitter = get_event_emitter(
            {
                "user_id": user.id,
                "chat_id": id,
                "message_id": message_id,
            },
            False,
        )

        if event_emitter:
            await event_emitter(
                {
                    "type": "chat:message",
                    "data": {
                        "chat_id": id,
                        "message_id": message_id,
                        "content": form_data.content,
                    },
                }
            )

        # Convert back to ChatResponse
        result_data = updated_record.model_dump()
        normalize_chat_data(result_data)
        return ChatResponse(**result_data)

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# SendChatMessageEventById
############################
class EventForm(BaseModel):
    type: str
    data: dict


@router.post("/{id}/messages/{message_id}/event", response_model=Optional[bool])
async def send_chat_message_event_by_id(
    id: str, message_id: str, form_data: EventForm, user=Depends(get_verified_user)
):
    client = get_client_or_raise(user.id)

    try:
        # Find the chat
        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"owui_id": id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
            )

        record = response.get_records()[0]
        chat_data = record.model_dump()

        # Check access
        if chat_data.get("user_id") != user.id and user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
            )

        event_emitter = get_event_emitter(
            {
                "user_id": user.id,
                "chat_id": id,
                "message_id": message_id,
            }
        )

        try:
            if event_emitter:
                await event_emitter(form_data.model_dump())
            else:
                return False
            return True
        except:
            return False

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# DeleteChatById
############################


@router.delete("/{id}", response_model=bool)
async def delete_chat_by_id(request: Request, id: str, user=Depends(get_verified_user)):
    client = get_client_or_raise(user.id)

    try:
        # Build query based on role
        if user.role == "admin":
            response = await client.query(
                Collections.CHATS,
                QueryOptions(filter={"owui_id": id}, limit=1)
            )
        else:
            if not has_permission(
                user.id, "chat.delete", request.app.state.config.USER_PERMISSIONS
            ):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
                )

            response = await client.query(
                Collections.CHATS,
                QueryOptions(filter={"owui_id": id, "user_id": user.id}, limit=1)
            )

        if not response.data or len(response.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )

        record = response.get_records()[0]
        chat_data = record.model_dump()

        # Clean up tags
        for tag in chat_data.get("meta", {}).get("tags", []):
            # Count chats with this tag
            tag_chats = await client.query(
                Collections.CHATS,
                QueryOptions(filter={"user_id": user.id, "meta.tags": {"$contains": tag}}, limit=2)
            )
            # If only one chat has this tag (the one being deleted), delete the tag
            if tag_chats.data and len(tag_chats.data) == 1:
                existing_tag = await client.query(
                    Collections.TAGS,
                    QueryOptions(filter={"name": tag, "user_id": user.id}, limit=1)
                )
                if existing_tag.data and len(existing_tag.data) > 0:
                    tag_record = existing_tag.get_records()[0]
                    await client.delete(Collections.TAGS, tag_record.id)

        # Delete the chat
        await client.delete(Collections.CHATS, record.id)
        return True

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# GetPinnedStatusById
############################


@router.get("/{id}/pinned", response_model=Optional[bool])
async def get_pinned_status_by_id(id: str, user=Depends(get_verified_user)):
    client = get_client_or_raise(user.id)

    try:
        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"owui_id": id, "user_id": user.id}, limit=1)
        )

        if response.data and len(response.data) > 0:
            record = response.get_records()[0]
            return record.model_dump().get("pinned", False)
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail=ERROR_MESSAGES.DEFAULT()
            )

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# PinChatById
############################


@router.post("/{id}/pin", response_model=Optional[ChatResponse])
async def pin_chat_by_id(id: str, user=Depends(get_verified_user)):
    client = get_client_or_raise(user.id)

    try:
        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"owui_id": id, "user_id": user.id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail=ERROR_MESSAGES.DEFAULT()
            )

        record = response.get_records()[0]
        chat_data = record.model_dump()

        # Toggle pinned status
        chat_data["pinned"] = not chat_data.get("pinned", False)

        # Update in BlueNexus
        updated_record = await client.update(Collections.CHATS, record.id, chat_data)

        # Convert back to ChatResponse
        result_data = updated_record.model_dump()
        normalize_chat_data(result_data)
        return ChatResponse(**result_data)

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# CloneChat
############################


class CloneForm(BaseModel):
    title: Optional[str] = None


@router.post("/{id}/clone", response_model=Optional[ChatResponse])
async def clone_chat_by_id(
    form_data: CloneForm, id: str, user=Depends(get_verified_user)
):
    import uuid

    client = get_client_or_raise(user.id)

    try:
        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"owui_id": id, "user_id": user.id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail=ERROR_MESSAGES.DEFAULT()
            )

        record = response.get_records()[0]
        original_chat = record.model_dump()
        original_id = original_chat.get("owui_id", original_chat.get("id"))

        # Build cloned chat content
        chat_content = original_chat.get("chat", {})
        updated_chat_content = {
            **chat_content,
            "originalChatId": original_id,
            "branchPointMessageId": chat_content.get("history", {}).get("currentId"),
            "title": form_data.title if form_data.title else f"Clone of {original_chat.get('title', 'Chat')}",
        }

        # Create new chat
        new_chat_id = str(uuid.uuid4())
        new_chat_data = {
            "owui_id": new_chat_id,
            "user_id": user.id,
            "title": updated_chat_content.get("title"),
            "chat": updated_chat_content,
            "meta": original_chat.get("meta", {}),
            "pinned": original_chat.get("pinned", False),
            "archived": False,
            "folder_id": original_chat.get("folder_id"),
            "share_id": None,
        }

        new_record = await client.create(Collections.CHATS, new_chat_data)

        # Convert back to ChatResponse
        result_data = new_record.model_dump()
        normalize_chat_data(result_data)
        return ChatResponse(**result_data)

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# CloneSharedChatById
############################


@router.post("/{id}/clone/shared", response_model=Optional[ChatResponse])
async def clone_shared_chat_by_id(id: str, user=Depends(get_verified_user)):
    import uuid

    client = get_client_or_raise(user.id)

    try:
        # Build query based on role
        if user.role == "admin":
            response = await client.query(
                Collections.CHATS,
                QueryOptions(filter={"owui_id": id}, limit=1)
            )
        else:
            response = await client.query(
                Collections.CHATS,
                QueryOptions(filter={"share_id": id}, limit=1)
            )

        if not response.data or len(response.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail=ERROR_MESSAGES.DEFAULT()
            )

        record = response.get_records()[0]
        original_chat = record.model_dump()
        original_id = original_chat.get("owui_id", original_chat.get("id"))

        # Build cloned chat content
        chat_content = original_chat.get("chat", {})
        updated_chat_content = {
            **chat_content,
            "originalChatId": original_id,
            "branchPointMessageId": chat_content.get("history", {}).get("currentId"),
            "title": f"Clone of {original_chat.get('title', 'Chat')}",
        }

        # Create new chat for the current user
        new_chat_id = str(uuid.uuid4())
        new_chat_data = {
            "owui_id": new_chat_id,
            "user_id": user.id,
            "title": updated_chat_content.get("title"),
            "chat": updated_chat_content,
            "meta": original_chat.get("meta", {}),
            "pinned": original_chat.get("pinned", False),
            "archived": False,
            "folder_id": original_chat.get("folder_id"),
            "share_id": None,
        }

        new_record = await client.create(Collections.CHATS, new_chat_data)

        # Convert back to ChatResponse
        result_data = new_record.model_dump()
        normalize_chat_data(result_data)
        return ChatResponse(**result_data)

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# ArchiveChat
############################


@router.post("/{id}/archive", response_model=Optional[ChatResponse])
async def archive_chat_by_id(id: str, user=Depends(get_verified_user)):
    client = get_client_or_raise(user.id)

    try:
        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"owui_id": id, "user_id": user.id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail=ERROR_MESSAGES.DEFAULT()
            )

        record = response.get_records()[0]
        chat_data = record.model_dump()

        # Toggle archived status
        chat_data["archived"] = not chat_data.get("archived", False)

        # Handle tags based on archive status
        if chat_data["archived"]:
            # Delete tags if no other chats use them
            for tag_id in chat_data.get("meta", {}).get("tags", []):
                tag_chats = await client.query(
                    Collections.CHATS,
                    QueryOptions(
                        filter={"user_id": user.id, "archived": False, "meta.tags": {"$contains": tag_id}},
                        limit=1
                    )
                )
                if not tag_chats.data or len(tag_chats.data) == 0:
                    log.debug(f"deleting tag: {tag_id}")
                    existing_tag = await client.query(
                        Collections.TAGS,
                        QueryOptions(filter={"name": tag_id, "user_id": user.id}, limit=1)
                    )
                    if existing_tag.data and len(existing_tag.data) > 0:
                        tag_record = existing_tag.get_records()[0]
                        await client.delete(Collections.TAGS, tag_record.id)
        else:
            # Restore tags if needed
            for tag_id in chat_data.get("meta", {}).get("tags", []):
                existing_tag = await client.query(
                    Collections.TAGS,
                    QueryOptions(filter={"name": tag_id, "user_id": user.id}, limit=1)
                )
                if not existing_tag.data or len(existing_tag.data) == 0:
                    log.debug(f"inserting tag: {tag_id}")
                    await client.create(Collections.TAGS, {
                        "owui_id": tag_id,
                        "name": tag_id,
                        "user_id": user.id,
                    })

        # Update in BlueNexus
        updated_record = await client.update(Collections.CHATS, record.id, chat_data)

        # Convert back to ChatResponse
        result_data = updated_record.model_dump()
        normalize_chat_data(result_data)
        return ChatResponse(**result_data)

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# ShareChatById
############################


@router.post("/{id}/share", response_model=Optional[ChatResponse])
async def share_chat_by_id(request: Request, id: str, user=Depends(get_verified_user)):
    import uuid

    if (user.role != "admin") and (
        not has_permission(
            user.id, "chat.share", request.app.state.config.USER_PERMISSIONS
        )
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    client = get_client_or_raise(user.id)

    try:
        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"owui_id": id, "user_id": user.id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
            )

        record = response.get_records()[0]
        chat_data = record.model_dump()

        # Generate or keep share_id
        if not chat_data.get("share_id"):
            chat_data["share_id"] = str(uuid.uuid4())

        # Update in BlueNexus
        updated_record = await client.update(Collections.CHATS, record.id, chat_data)

        # Convert back to ChatResponse
        result_data = updated_record.model_dump()
        normalize_chat_data(result_data)
        return ChatResponse(**result_data)

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# DeletedSharedChatById
############################


@router.delete("/{id}/share", response_model=Optional[bool])
async def delete_shared_chat_by_id(id: str, user=Depends(get_verified_user)):
    client = get_client_or_raise(user.id)

    try:
        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"owui_id": id, "user_id": user.id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
            )

        record = response.get_records()[0]
        chat_data = record.model_dump()

        if not chat_data.get("share_id"):
            return False

        # Remove share_id
        chat_data["share_id"] = None

        # Update in BlueNexus
        await client.update(Collections.CHATS, record.id, chat_data)
        return True

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# UpdateChatFolderIdById
############################


class ChatFolderIdForm(BaseModel):
    folder_id: Optional[str] = None


@router.post("/{id}/folder", response_model=Optional[ChatResponse])
async def update_chat_folder_id_by_id(
    id: str, form_data: ChatFolderIdForm, user=Depends(get_verified_user)
):
    client = get_client_or_raise(user.id)

    try:
        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"owui_id": id, "user_id": user.id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail=ERROR_MESSAGES.DEFAULT()
            )

        record = response.get_records()[0]
        chat_data = record.model_dump()

        # Update folder_id
        chat_data["folder_id"] = form_data.folder_id

        # Update in BlueNexus
        updated_record = await client.update(Collections.CHATS, record.id, chat_data)

        # Convert back to ChatResponse
        result_data = updated_record.model_dump()
        normalize_chat_data(result_data)
        return ChatResponse(**result_data)

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# GetChatTagsById
############################


@router.get("/{id}/tags", response_model=list[TagModel])
async def get_chat_tags_by_id(id: str, user=Depends(get_verified_user)):
    client = get_client_or_raise(user.id)

    try:
        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"owui_id": id, "user_id": user.id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail=ERROR_MESSAGES.NOT_FOUND
            )

        record = response.get_records()[0]
        chat_data = record.model_dump()
        tag_names = chat_data.get("meta", {}).get("tags", [])

        # Get tag models from BlueNexus
        tags = []
        for tag_name in tag_names:
            tag_response = await client.query(
                Collections.TAGS,
                QueryOptions(filter={"name": tag_name, "user_id": user.id}, limit=1)
            )
            if tag_response.data and len(tag_response.data) > 0:
                tag_record = tag_response.get_records()[0]
                tag_data = tag_record.model_dump()
                tag_data["id"] = tag_data.get("owui_id", tag_data.get("id"))
                tags.append(TagModel(**tag_data))

        return tags

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# AddChatTagById
############################


@router.post("/{id}/tags", response_model=list[TagModel])
async def add_tag_by_id_and_tag_name(
    id: str, form_data: TagForm, user=Depends(get_verified_user)
):
    client = get_client_or_raise(user.id)

    try:
        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"owui_id": id, "user_id": user.id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail=ERROR_MESSAGES.DEFAULT()
            )

        record = response.get_records()[0]
        chat_data = record.model_dump()

        tag_id = form_data.name.replace(" ", "_").lower()

        if tag_id == "none":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.DEFAULT("Tag name cannot be 'None'"),
            )

        # Get current tags
        meta = chat_data.get("meta", {})
        tags = meta.get("tags", [])

        if tag_id not in tags:
            tags.append(tag_id)
            meta["tags"] = tags
            chat_data["meta"] = meta

            # Update chat in BlueNexus
            await client.update(Collections.CHATS, record.id, chat_data)

            # Create tag if it doesn't exist
            existing_tag = await client.query(
                Collections.TAGS,
                QueryOptions(filter={"name": tag_id, "user_id": user.id}, limit=1)
            )
            if not existing_tag.data or len(existing_tag.data) == 0:
                await client.create(Collections.TAGS, {
                    "owui_id": tag_id,
                    "name": tag_id,
                    "user_id": user.id,
                })

        # Return current tags
        result_tags = []
        for tag_name in tags:
            tag_response = await client.query(
                Collections.TAGS,
                QueryOptions(filter={"name": tag_name, "user_id": user.id}, limit=1)
            )
            if tag_response.data and len(tag_response.data) > 0:
                tag_record = tag_response.get_records()[0]
                tag_data = tag_record.model_dump()
                tag_data["id"] = tag_data.get("owui_id", tag_data.get("id"))
                result_tags.append(TagModel(**tag_data))

        return result_tags

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# DeleteChatTagById
############################


@router.delete("/{id}/tags", response_model=list[TagModel])
async def delete_tag_by_id_and_tag_name(
    id: str, form_data: TagForm, user=Depends(get_verified_user)
):
    client = get_client_or_raise(user.id)

    try:
        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"owui_id": id, "user_id": user.id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail=ERROR_MESSAGES.NOT_FOUND
            )

        record = response.get_records()[0]
        chat_data = record.model_dump()

        tag_id = form_data.name.replace(" ", "_").lower()

        # Remove tag from chat
        meta = chat_data.get("meta", {})
        tags = meta.get("tags", [])
        if tag_id in tags:
            tags.remove(tag_id)
            meta["tags"] = tags
            chat_data["meta"] = meta
            await client.update(Collections.CHATS, record.id, chat_data)

        # Check if any other chats use this tag
        tag_chats = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"user_id": user.id, "meta.tags": {"$contains": form_data.name}}, limit=1)
        )
        if not tag_chats.data or len(tag_chats.data) == 0:
            # Delete the tag
            existing_tag = await client.query(
                Collections.TAGS,
                QueryOptions(filter={"name": form_data.name, "user_id": user.id}, limit=1)
            )
            if existing_tag.data and len(existing_tag.data) > 0:
                tag_record = existing_tag.get_records()[0]
                await client.delete(Collections.TAGS, tag_record.id)

        # Return current tags
        result_tags = []
        for tag_name in tags:
            tag_response = await client.query(
                Collections.TAGS,
                QueryOptions(filter={"name": tag_name, "user_id": user.id}, limit=1)
            )
            if tag_response.data and len(tag_response.data) > 0:
                tag_record = tag_response.get_records()[0]
                tag_data = tag_record.model_dump()
                tag_data["id"] = tag_data.get("owui_id", tag_data.get("id"))
                result_tags.append(TagModel(**tag_data))

        return result_tags

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )


############################
# DeleteAllTagsById
############################


@router.delete("/{id}/tags/all", response_model=Optional[bool])
async def delete_all_tags_by_id(id: str, user=Depends(get_verified_user)):
    client = get_client_or_raise(user.id)

    try:
        response = await client.query(
            Collections.CHATS,
            QueryOptions(filter={"owui_id": id, "user_id": user.id}, limit=1)
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail=ERROR_MESSAGES.NOT_FOUND
            )

        record = response.get_records()[0]
        chat_data = record.model_dump()

        # Get current tags before clearing
        current_tags = chat_data.get("meta", {}).get("tags", [])

        # Clear tags from chat
        meta = chat_data.get("meta", {})
        meta["tags"] = []
        chat_data["meta"] = meta
        await client.update(Collections.CHATS, record.id, chat_data)

        # Clean up unused tags
        for tag_name in current_tags:
            tag_chats = await client.query(
                Collections.CHATS,
                QueryOptions(filter={"user_id": user.id, "meta.tags": {"$contains": tag_name}}, limit=1)
            )
            if not tag_chats.data or len(tag_chats.data) == 0:
                existing_tag = await client.query(
                    Collections.TAGS,
                    QueryOptions(filter={"name": tag_name, "user_id": user.id}, limit=1)
                )
                if existing_tag.data and len(existing_tag.data) > 0:
                    tag_record = existing_tag.get_records()[0]
                    await client.delete(Collections.TAGS, tag_record.id)

        return True

    except BlueNexusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BlueNexus service unavailable: {str(e)}",
        )
