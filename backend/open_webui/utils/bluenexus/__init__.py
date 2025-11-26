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

# Configuration
from open_webui.utils.bluenexus.config import (
    ENABLE_BLUENEXUS,
    ENABLE_BLUENEXUS_SYNC,
    is_bluenexus_enabled,
    is_bluenexus_sync_enabled,
    is_bluenexus_configured,
)

# OAuth Integration
from open_webui.utils.bluenexus.oauth import (
    register_bluenexus_oauth,
    get_bluenexus_oauth_provider_config,
    should_disable_ssl_for_provider,
)

# LLM Provider Integration
from open_webui.utils.bluenexus.llm import (
    ensure_bluenexus_provider,
    get_ssl_context_for_url,
    get_bluenexus_oauth_token_for_headers,
)

# MCP Integration
from open_webui.utils.bluenexus.mcp import (
    get_bluenexus_mcp_servers,
    get_bluenexus_mcp_oauth_token,
    BlueNexusMCPServer,
    BlueNexusMCPServersResponse,
)

# Auth Management
from open_webui.utils.bluenexus.auth import (
    refresh_oauth_token,
    OAuthTokenStatusResponse,
)

# Data Client
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
from open_webui.utils.bluenexus.sync_service import BlueNexusSyncService, BlueNexusSync

__all__ = [
    # Configuration
    "ENABLE_BLUENEXUS",
    "ENABLE_BLUENEXUS_SYNC",
    "is_bluenexus_enabled",
    "is_bluenexus_sync_enabled",
    "is_bluenexus_configured",
    # OAuth Integration
    "register_bluenexus_oauth",
    "get_bluenexus_oauth_provider_config",
    "should_disable_ssl_for_provider",
    # LLM Provider Integration
    "ensure_bluenexus_provider",
    "get_ssl_context_for_url",
    "get_bluenexus_oauth_token_for_headers",
    # MCP Integration
    "get_bluenexus_mcp_servers",
    "get_bluenexus_mcp_oauth_token",
    "BlueNexusMCPServer",
    "BlueNexusMCPServersResponse",
    # Auth Management
    "refresh_oauth_token",
    "OAuthTokenStatusResponse",
    # Data Client
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
    # Sync Service
    "BlueNexusSyncService",
    "BlueNexusSync",
]
