"""
BlueNexus MCP (Model Context Protocol) Integration Module

This module handles fetching and managing MCP servers from BlueNexus.
Includes caching for MCP server lists and tool specs to avoid repeated API calls.
"""

import asyncio
import logging
import time
from threading import Lock
from typing import Optional, List, Dict, Any
import aiohttp

from pydantic import BaseModel, ConfigDict
from open_webui.models.oauth_sessions import OAuthSessions

from open_webui.utils.bluenexus.config import (
    BLUENEXUS_API_BASE_URL,
    is_bluenexus_enabled,
)
from open_webui.utils.bluenexus.llm import get_ssl_context_for_url

log = logging.getLogger(__name__)


# ============================================================================
# MCP Server List Cache
# ============================================================================
# Cache format: user_id -> (response, access_token, timestamp)
_mcp_servers_cache: Dict[str, tuple] = {}
_mcp_servers_cache_lock = Lock()
_MCP_SERVERS_CACHE_TTL = 300  # 5 minutes

# Tool specs cache format: server_url -> (tool_specs, timestamp)
_tool_specs_cache: Dict[str, tuple] = {}
_tool_specs_cache_lock = Lock()
_TOOL_SPECS_CACHE_TTL = 300  # 5 minutes


def invalidate_mcp_servers_cache(user_id: Optional[str] = None) -> None:
    """
    Invalidate MCP servers cache.

    Args:
        user_id: If provided, only invalidate for this user. Otherwise invalidate all.
    """
    global _mcp_servers_cache
    with _mcp_servers_cache_lock:
        if user_id:
            if user_id in _mcp_servers_cache:
                del _mcp_servers_cache[user_id]
                log.info(f"[MCP Cache] Invalidated server list cache for user {user_id}")
        else:
            _mcp_servers_cache.clear()
            log.info("[MCP Cache] Invalidated all server list caches")


def invalidate_tool_specs_cache(server_url: Optional[str] = None) -> None:
    """
    Invalidate tool specs cache.

    Args:
        server_url: If provided, only invalidate for this server. Otherwise invalidate all.
    """
    global _tool_specs_cache
    with _tool_specs_cache_lock:
        if server_url:
            if server_url in _tool_specs_cache:
                del _tool_specs_cache[server_url]
                log.info(f"[MCP Cache] Invalidated tool specs cache for {server_url}")
        else:
            _tool_specs_cache.clear()
            log.info("[MCP Cache] Invalidated all tool specs caches")


def invalidate_all_mcp_caches(user_id: Optional[str] = None) -> None:
    """Invalidate all MCP-related caches for a user or globally."""
    invalidate_mcp_servers_cache(user_id)
    invalidate_tool_specs_cache()
    log.info(f"[MCP Cache] Invalidated all MCP caches" + (f" for user {user_id}" if user_id else ""))


def get_cached_tool_specs(server_url: str) -> Optional[List[Dict[str, Any]]]:
    """
    Get cached tool specs for a server if still valid.

    Args:
        server_url: The MCP server URL

    Returns:
        Cached tool specs list or None if not cached/expired
    """
    with _tool_specs_cache_lock:
        if server_url in _tool_specs_cache:
            tool_specs, timestamp = _tool_specs_cache[server_url]
            if time.time() - timestamp < _TOOL_SPECS_CACHE_TTL:
                log.debug(f"[MCP Cache] Tool specs cache HIT for {server_url}")
                return tool_specs
            else:
                # Expired, remove from cache
                del _tool_specs_cache[server_url]
                log.debug(f"[MCP Cache] Tool specs cache EXPIRED for {server_url}")
    return None


def set_cached_tool_specs(server_url: str, tool_specs: List[Dict[str, Any]]) -> None:
    """
    Cache tool specs for a server.

    Args:
        server_url: The MCP server URL
        tool_specs: List of tool specifications
    """
    with _tool_specs_cache_lock:
        _tool_specs_cache[server_url] = (tool_specs, time.time())
        log.debug(f"[MCP Cache] Cached {len(tool_specs)} tool specs for {server_url}")


class BlueNexusMCPServer(BaseModel):
    """BlueNexus MCP Server model."""
    model_config = ConfigDict(extra="allow")
    slug: str
    label: str
    description: Optional[str] = None
    isActive: Optional[bool] = None
    url: str


class BlueNexusMCPServersResponse(BaseModel):
    """Response model for BlueNexus MCP servers."""
    data: List[BlueNexusMCPServer]


async def get_bluenexus_mcp_servers(
    user_id: str,
    force_refresh: bool = False
) -> BlueNexusMCPServersResponse:
    """
    Get available BlueNexus MCP servers for the authenticated user.
    Requires BlueNexus OAuth connection with mcp-proxy scope.

    Results are cached per user for 5 minutes to avoid repeated API calls.
    Cache is automatically invalidated when access token changes.

    Args:
        user_id: User ID
        force_refresh: If True, bypass cache and fetch fresh data

    Returns:
        Response with list of MCP servers
    """
    global _mcp_servers_cache

    if not is_bluenexus_enabled():
        log.debug(f"BlueNexus MCP servers request skipped - BlueNexus disabled for user {user_id}")
        return BlueNexusMCPServersResponse(data=[])

    if not BLUENEXUS_API_BASE_URL.value:
        log.warning("BLUENEXUS_API_BASE_URL not configured")
        return BlueNexusMCPServersResponse(data=[])

    try:
        # Get BlueNexus OAuth session token
        oauth_session = OAuthSessions.get_session_by_provider_and_user_id(
            provider="bluenexus", user_id=user_id
        )

        if not oauth_session:
            log.warning(f"No BlueNexus OAuth session found for user {user_id}")
            return BlueNexusMCPServersResponse(data=[])

        access_token = oauth_session.token.get("access_token")
        if not access_token:
            log.warning(f"No access token found in BlueNexus session for user {user_id}")
            return BlueNexusMCPServersResponse(data=[])

        # Check cache first (unless force_refresh)
        if not force_refresh:
            with _mcp_servers_cache_lock:
                if user_id in _mcp_servers_cache:
                    cached_response, cached_token, timestamp = _mcp_servers_cache[user_id]
                    # Cache is valid if: same token AND within TTL
                    if cached_token == access_token and time.time() - timestamp < _MCP_SERVERS_CACHE_TTL:
                        log.info(f"[MCP Cache] Server list cache HIT for user {user_id} ({len(cached_response.data)} servers)")
                        return cached_response
                    elif cached_token != access_token:
                        log.info(f"[MCP Cache] Token changed for user {user_id}, invalidating cache")
                        del _mcp_servers_cache[user_id]

        # Fetch available MCP servers from BlueNexus API with retry logic
        mcp_servers_url = f"{BLUENEXUS_API_BASE_URL.value}/api/v1/mcps"
        headers = {"Authorization": f"Bearer {access_token}"}

        max_retries = 3
        retry_delay = 1.0
        last_error = None

        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession(trust_env=True) as http_session:
                    ssl_context = get_ssl_context_for_url(mcp_servers_url)
                    async with http_session.get(
                        mcp_servers_url, headers=headers, ssl=ssl_context
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            mcp_servers = result.get("data", [])

                            # Transform to include full proxy URL
                            for server in mcp_servers:
                                server["url"] = (
                                    f"{BLUENEXUS_API_BASE_URL.value}/api/v1/mcps/{server['slug']}"
                                )

                            # Create response and cache it
                            mcp_response = BlueNexusMCPServersResponse(data=mcp_servers)
                            with _mcp_servers_cache_lock:
                                _mcp_servers_cache[user_id] = (mcp_response, access_token, time.time())

                            log.info(
                                f"[MCP Cache] Fetched and cached {len(mcp_servers)} BlueNexus MCP servers for user {user_id}"
                            )
                            return mcp_response
                        else:
                            log.error(
                                f"Failed to fetch BlueNexus MCP servers: HTTP {response.status}"
                            )
                            return BlueNexusMCPServersResponse(data=[])

            except (aiohttp.ClientConnectorError, aiohttp.ServerDisconnectedError, asyncio.TimeoutError) as conn_error:
                last_error = conn_error
                if attempt < max_retries - 1:
                    delay = retry_delay * (2 ** attempt)
                    log.warning(
                        f"BlueNexus MCP server fetch connection error (attempt {attempt + 1}/{max_retries}): {conn_error}. Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    log.error(
                        f"BlueNexus MCP server fetch failed after {max_retries} attempts: {conn_error}"
                    )

        # All retries exhausted
        return BlueNexusMCPServersResponse(data=[])

    except Exception as e:
        log.error(f"Error fetching BlueNexus MCP servers: {e}")
        return BlueNexusMCPServersResponse(data=[])


def get_bluenexus_mcp_oauth_token(user_id: str, server_id: str) -> Optional[Dict[str, Any]]:
    """
    Get BlueNexus OAuth token for MCP server authentication.

    Args:
        user_id: User ID
        server_id: MCP server ID (used to identify BlueNexus servers)

    Returns:
        OAuth token dict or None if not available
    """
    if not is_bluenexus_enabled():
        return None

    # Only return token for BlueNexus MCP servers
    if "bluenexus" not in server_id:
        return None

    try:
        session = OAuthSessions.get_session_by_provider_and_user_id(
            provider="bluenexus", user_id=user_id
        )
        if session:
            return session.token
    except Exception as e:
        log.warning(f"Failed to get BlueNexus OAuth token for MCP server: {e}")

    return None
