"""
BlueNexus API Client for Open WebUI

This module provides a client for interacting with the BlueNexus User-Data API,
enabling Open WebUI to store and retrieve user data in BlueNexus.

Usage:
    from open_webui.utils.bluenexus import (
        BlueNexusDataClient,
        get_bluenexus_client_for_user,
        Collections,
        QueryOptions,
    )

    # Get client for a user
    client = get_bluenexus_client_for_user(user_id)

    # Create a chat record
    chat = await client.create(Collections.CHATS, {
        "title": "My Chat",
        "messages": []
    })

    # Query chats
    response = await client.query(Collections.CHATS, QueryOptions(limit=10))
"""

from open_webui.utils.bluenexus.client import BlueNexusDataClient
from open_webui.utils.bluenexus.types import (
    BlueNexusRecord,
    BlueNexusError,
    BlueNexusAuthError,
    BlueNexusNotFoundError,
    BlueNexusValidationError,
    BlueNexusConnectionError,
    PaginatedResponse,
    PaginationInfo,
    QueryOptions,
    SortBy,
    SortOrder,
    ValidationError,
    VerifyResponse,
)
from open_webui.utils.bluenexus.collections import Collections, MODEL_TO_COLLECTION
from open_webui.utils.bluenexus.factory import (
    get_bluenexus_client_for_user,
    has_bluenexus_session,
    get_or_create_bluenexus_client,
    BlueNexusClientContext,
)
from open_webui.utils.bluenexus.chat_storage import BlueNexusChatStorage, ChatData
from open_webui.utils.bluenexus.hybrid_chat_storage import HybridChatStorage, HybridChats

__all__ = [
    # Client
    "BlueNexusDataClient",
    # Types
    "BlueNexusRecord",
    "BlueNexusError",
    "BlueNexusAuthError",
    "BlueNexusNotFoundError",
    "BlueNexusValidationError",
    "BlueNexusConnectionError",
    "PaginatedResponse",
    "PaginationInfo",
    "QueryOptions",
    "SortBy",
    "SortOrder",
    "ValidationError",
    "VerifyResponse",
    # Collections
    "Collections",
    "MODEL_TO_COLLECTION",
    # Factory
    "get_bluenexus_client_for_user",
    "has_bluenexus_session",
    "get_or_create_bluenexus_client",
    "BlueNexusClientContext",
    # Chat Storage
    "BlueNexusChatStorage",
    "ChatData",
    "HybridChatStorage",
    "HybridChats",
]
