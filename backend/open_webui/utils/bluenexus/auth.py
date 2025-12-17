"""
BlueNexus Authentication Module

This module handles OAuth token refresh and session management for BlueNexus.
"""

import asyncio
import logging
import time
from typing import Optional
from pydantic import BaseModel

from fastapi import Request
from open_webui.models.users import UserModel
from open_webui.models.oauth_sessions import OAuthSessions

from open_webui.utils.bluenexus.config import is_bluenexus_enabled
from open_webui.utils.bluenexus.mcp import invalidate_mcp_servers_cache

log = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1


class OAuthTokenStatusResponse(BaseModel):
    """OAuth token status response model."""
    has_session: bool
    provider: Optional[str] = None
    expires_at: Optional[int] = None
    expires_in: Optional[int] = None
    refreshed: bool = False
    requires_reauth: bool = False  # True if user needs to re-login


async def refresh_oauth_token(request: Request, user: UserModel) -> OAuthTokenStatusResponse:
    """
    Refresh the OAuth token if it's close to expiring.
    This endpoint is designed to be called periodically by the frontend
    to keep the OAuth session alive for active users.

    Args:
        request: FastAPI request object
        user: Current user model

    Returns:
        OAuthTokenStatusResponse with token status
    """
    if not is_bluenexus_enabled():
        log.debug(f"OAuth refresh skipped - BlueNexus disabled for user {user.id}")
        return OAuthTokenStatusResponse(has_session=False)

    log.info(f"OAuth refresh requested for user {user.id}")

    oauth_session_id = request.cookies.get("oauth_session_id")

    # Try to get session - first by cookie session ID, then fallback to provider lookup
    session = None
    if oauth_session_id:
        log.info(f"Found oauth_session_id: {oauth_session_id[:8]}... for user {user.id}")
        session = OAuthSessions.get_session_by_id_and_user_id(oauth_session_id, user.id)

    # Fallback: lookup by provider if cookie session not found
    if not session:
        log.info(f"Session not found by ID, trying provider lookup for user {user.id}")
        session = OAuthSessions.get_session_by_provider_and_user_id("bluenexus", user.id)
        if session:
            oauth_session_id = session.id
            log.info(f"Found BlueNexus session by provider lookup: {oauth_session_id[:8]}...")

    if not session or not oauth_session_id:
        log.info(f"No BlueNexus OAuth session found for user {user.id} - requires re-authentication")
        # User has oauth_session_id cookie but no valid session - needs to re-login
        return OAuthTokenStatusResponse(has_session=False, requires_reauth=True)

    # Retry logic for token refresh
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            # Get token with force_refresh=False - this will auto-refresh if needed
            token = await request.app.state.oauth_manager.get_oauth_token(
                user.id,
                oauth_session_id,
                force_refresh=(attempt > 0),  # Force refresh on retry
            )

            if token:
                expires_at = token.get("expires_at")
                expires_in = None
                if expires_at:
                    expires_in = max(0, int(expires_at - time.time()))

                provider = session.provider if session else "bluenexus"

                log.info(f"OAuth token valid for user {user.id}, provider: {provider}, expires_in: {expires_in}s")

                # Invalidate MCP cache when token was refreshed (new token = new capabilities)
                if attempt > 0:
                    log.info(f"Token was refreshed, invalidating MCP cache for user {user.id}")
                    invalidate_mcp_servers_cache(user.id)

                return OAuthTokenStatusResponse(
                    has_session=True,
                    provider=provider,
                    expires_at=expires_at,
                    expires_in=expires_in,
                    refreshed=(attempt > 0),
                )
            else:
                log.warning(f"OAuth token not found (attempt {attempt + 1}/{MAX_RETRIES}) for user {user.id}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY_SECONDS)
                    continue
                # Session may have been deleted, user needs to re-authenticate
                return OAuthTokenStatusResponse(has_session=False, requires_reauth=True)

        except Exception as e:
            last_error = e
            log.warning(f"Error refreshing OAuth token (attempt {attempt + 1}/{MAX_RETRIES}) for user {user.id}: {e}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY_SECONDS)
                continue

    log.error(f"Failed to refresh OAuth token after {MAX_RETRIES} attempts for user {user.id}: {last_error}")
    # Session may have been deleted by oauth manager, user needs to re-authenticate
    return OAuthTokenStatusResponse(has_session=False, requires_reauth=True)
