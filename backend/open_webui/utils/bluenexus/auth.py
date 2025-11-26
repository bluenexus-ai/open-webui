"""
BlueNexus Authentication Module

This module handles OAuth token refresh and session management for BlueNexus.
"""

import logging
import time
from typing import Optional
from pydantic import BaseModel

from fastapi import Request
from open_webui.models.users import UserModel
from open_webui.models.oauth_sessions import OAuthSessions

from open_webui.utils.bluenexus.config import is_bluenexus_enabled

log = logging.getLogger(__name__)


class OAuthTokenStatusResponse(BaseModel):
    """OAuth token status response model."""
    has_session: bool
    provider: Optional[str] = None
    expires_at: Optional[int] = None
    expires_in: Optional[int] = None
    refreshed: bool = False


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
    if not oauth_session_id:
        log.info(f"No oauth_session_id cookie found for user {user.id}")
        return OAuthTokenStatusResponse(has_session=False)

    log.info(f"Found oauth_session_id: {oauth_session_id[:8]}... for user {user.id}")

    try:
        # Get token with force_refresh=False - this will auto-refresh if needed
        token = await request.app.state.oauth_manager.get_oauth_token(
            user.id,
            oauth_session_id,
            force_refresh=False,
        )

        if token:
            expires_at = token.get("expires_at")
            expires_in = None
            if expires_at:
                expires_in = max(0, int(expires_at - time.time()))

            # Get session to check provider
            session = OAuthSessions.get_session_by_id(oauth_session_id)
            provider = session.provider if session else None

            log.info(f"OAuth token valid for user {user.id}, provider: {provider}, expires_in: {expires_in}s")

            return OAuthTokenStatusResponse(
                has_session=True,
                provider=provider,
                expires_at=expires_at,
                expires_in=expires_in,
                refreshed=False,
            )
        else:
            log.warning(f"OAuth token not found or expired for user {user.id}, session {oauth_session_id[:8]}...")
            return OAuthTokenStatusResponse(has_session=False)

    except Exception as e:
        log.error(f"Error refreshing OAuth token for user {user.id}: {e}")
        return OAuthTokenStatusResponse(has_session=False)
