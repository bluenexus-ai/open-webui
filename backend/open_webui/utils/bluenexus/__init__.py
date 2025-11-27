"""
BlueNexus Integration for Open WebUI

This module provides a complete BlueNexus integration as an extension plugin.
All functionality is gated by the ENABLE_BLUENEXUS flag.

Core Features:
- OAuth authentication with BlueNexus
- LLM provider integration (OpenAI-compatible)
- MCP (Model Context Protocol) server integration
- Data synchronization service
- User data storage API client

Usage:
    from open_webui.utils.bluenexus import (
        # Data Client
        BlueNexusDataClient,
        get_bluenexus_client_for_user,
        Collections,
        QueryOptions,
        # OAuth & Auth
        ensure_bluenexus_provider,
        refresh_oauth_token,
        # MCP
        get_bluenexus_mcp_servers,
        # Sync
        BlueNexusSync,
    )
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from open_webui.utils.bluenexus.config import (  # noqa: F401
        ENABLE_BLUENEXUS,
        ENABLE_BLUENEXUS_SYNC,
        is_bluenexus_enabled,
        is_bluenexus_sync_enabled,
        is_bluenexus_configured,
    )
    from open_webui.utils.bluenexus.oauth import (  # noqa: F401
        register_bluenexus_oauth,
        get_bluenexus_oauth_provider_config,
        should_disable_ssl_for_provider,
    )
    from open_webui.utils.bluenexus.llm import (  # noqa: F401
        ensure_bluenexus_provider,
        get_ssl_context_for_url,
        get_bluenexus_oauth_token_for_headers,
    )
    from open_webui.utils.bluenexus.mcp import (  # noqa: F401
        get_bluenexus_mcp_servers,
        get_bluenexus_mcp_oauth_token,
        BlueNexusMCPServer,
        BlueNexusMCPServersResponse,
    )
    from open_webui.utils.bluenexus.auth import (  # noqa: F401
        refresh_oauth_token,
        OAuthTokenStatusResponse,
    )
    from open_webui.utils.bluenexus.client import BlueNexusDataClient  # noqa: F401
    from open_webui.utils.bluenexus.types import (  # noqa: F401
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
    from open_webui.utils.bluenexus.collections import (  # noqa: F401
        Collections,
        MODEL_TO_COLLECTION,
    )
    from open_webui.utils.bluenexus.factory import (  # noqa: F401
        get_bluenexus_client_for_user,
        has_bluenexus_session,
        get_or_create_bluenexus_client,
        BlueNexusClientContext,
    )
    from open_webui.utils.bluenexus.chat_storage import (  # noqa: F401
        BlueNexusChatStorage,
        ChatData,
    )
    from open_webui.utils.bluenexus.hybrid_chat_storage import (  # noqa: F401
        HybridChatStorage,
        HybridChats,
    )
    from open_webui.utils.bluenexus.sync_service import (  # noqa: F401
        BlueNexusSyncService,
        BlueNexusSync,
    )

__all__ = [
    "ENABLE_BLUENEXUS",
    "ENABLE_BLUENEXUS_SYNC",
    "is_bluenexus_enabled",
    "is_bluenexus_sync_enabled",
    "is_bluenexus_configured",
    "register_bluenexus_oauth",
    "get_bluenexus_oauth_provider_config",
    "should_disable_ssl_for_provider",
    "ensure_bluenexus_provider",
    "get_ssl_context_for_url",
    "get_bluenexus_oauth_token_for_headers",
    "get_bluenexus_mcp_servers",
    "get_bluenexus_mcp_oauth_token",
    "BlueNexusMCPServer",
    "BlueNexusMCPServersResponse",
    "refresh_oauth_token",
    "OAuthTokenStatusResponse",
    "BlueNexusDataClient",
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
    "Collections",
    "MODEL_TO_COLLECTION",
    "get_bluenexus_client_for_user",
    "has_bluenexus_session",
    "get_or_create_bluenexus_client",
    "BlueNexusClientContext",
    "BlueNexusChatStorage",
    "ChatData",
    "HybridChatStorage",
    "HybridChats",
    "BlueNexusSyncService",
    "BlueNexusSync",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)

    # Import on demand to avoid circular imports at package init time.
    if name in {
        "ENABLE_BLUENEXUS",
        "ENABLE_BLUENEXUS_SYNC",
        "is_bluenexus_enabled",
        "is_bluenexus_sync_enabled",
        "is_bluenexus_configured",
    }:
        from open_webui.utils.bluenexus import config as _cfg
        return getattr(_cfg, name)

    if name in {
        "register_bluenexus_oauth",
        "get_bluenexus_oauth_provider_config",
        "should_disable_ssl_for_provider",
    }:
        from open_webui.utils.bluenexus import oauth as _oauth
        return getattr(_oauth, name)

    if name in {
        "ensure_bluenexus_provider",
        "get_ssl_context_for_url",
        "get_bluenexus_oauth_token_for_headers",
    }:
        from open_webui.utils.bluenexus import llm as _llm
        return getattr(_llm, name)

    if name in {
        "get_bluenexus_mcp_servers",
        "get_bluenexus_mcp_oauth_token",
        "BlueNexusMCPServer",
        "BlueNexusMCPServersResponse",
    }:
        from open_webui.utils.bluenexus import mcp as _mcp
        return getattr(_mcp, name)

    if name in {"refresh_oauth_token", "OAuthTokenStatusResponse"}:
        from open_webui.utils.bluenexus import auth as _auth
        return getattr(_auth, name)

    if name in {"BlueNexusDataClient"}:
        from open_webui.utils.bluenexus import client as _client
        return getattr(_client, name)

    if name in {
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
    }:
        from open_webui.utils.bluenexus import types as _types
        return getattr(_types, name)

    if name in {"Collections", "MODEL_TO_COLLECTION"}:
        from open_webui.utils.bluenexus import collections as _collections
        return getattr(_collections, name)

    if name in {
        "get_bluenexus_client_for_user",
        "has_bluenexus_session",
        "get_or_create_bluenexus_client",
        "BlueNexusClientContext",
    }:
        from open_webui.utils.bluenexus import factory as _factory
        return getattr(_factory, name)

    if name in {"BlueNexusChatStorage", "ChatData"}:
        from open_webui.utils.bluenexus import chat_storage as _cs
        return getattr(_cs, name)

    if name in {"HybridChatStorage", "HybridChats"}:
        from open_webui.utils.bluenexus import hybrid_chat_storage as _hcs
        return getattr(_hcs, name)

    if name in {"BlueNexusSyncService", "BlueNexusSync"}:
        from open_webui.utils.bluenexus import sync_service as _sync
        return getattr(_sync, name)

    raise AttributeError(name)
