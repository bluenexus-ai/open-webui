"""
BlueNexus MCP (Model Context Protocol) Integration Module

This module handles fetching and managing MCP servers from BlueNexus.
"""

import logging
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


async def get_bluenexus_mcp_servers(user_id: str) -> BlueNexusMCPServersResponse:
    """
    Get available BlueNexus MCP servers for the authenticated user.
    Requires BlueNexus OAuth connection with mcp-proxy scope.

    Args:
        user_id: User ID

    Returns:
        Response with list of MCP servers
    """
    if not is_bluenexus_enabled():
        log.debug(f"BlueNexus MCP servers request skipped - BlueNexus disabled for user {user_id}")
        return BlueNexusMCPServersResponse(data=[])

    if not BLUENEXUS_API_BASE_URL.value:
        log.warning("BLUENEXUS_API_BASE_URL not configured")
        return BlueNexusMCPServersResponse(data=[])

    try:
        # Get BlueNexus OAuth session token
        session = OAuthSessions.get_session_by_provider_and_user_id(
            provider="bluenexus", user_id=user_id
        )

        if not session:
            log.warning(f"No BlueNexus OAuth session found for user {user_id}")
            return BlueNexusMCPServersResponse(data=[])

        access_token = session.token.get("access_token")
        if not access_token:
            log.warning(f"No access token found in BlueNexus session for user {user_id}")
            return BlueNexusMCPServersResponse(data=[])

        # Fetch available MCP servers from BlueNexus API
        mcp_servers_url = f"{BLUENEXUS_API_BASE_URL.value}/api/v1/mcps"
        headers = {"Authorization": f"Bearer {access_token}"}

        async with aiohttp.ClientSession(trust_env=True) as session:
            ssl_context = get_ssl_context_for_url(mcp_servers_url)
            async with session.get(
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

                    log.info(
                        f"Fetched {len(mcp_servers)} BlueNexus MCP servers for user {user_id}"
                    )
                    return BlueNexusMCPServersResponse(data=mcp_servers)
                else:
                    log.error(
                        f"Failed to fetch BlueNexus MCP servers: HTTP {response.status}"
                    )
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
