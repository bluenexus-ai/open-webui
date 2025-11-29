"""
Factory functions for creating BlueNexus data clients.

Provides methods to create clients from OAuth sessions and user context.
"""

import logging
from typing import Optional

from open_webui.env import SRC_LOG_LEVELS
from open_webui.utils.bluenexus.config import ENABLE_BLUENEXUS, BLUENEXUS_API_BASE_URL
from open_webui.models.oauth_sessions import OAuthSessions
from open_webui.utils.bluenexus.client import BlueNexusDataClient
from open_webui.utils.bluenexus.types import BlueNexusAuthError

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("BLUENEXUS", SRC_LOG_LEVELS.get("MAIN", logging.INFO)))


def get_bluenexus_client_for_user(
    user_id: str,
    base_url: Optional[str] = None,
) -> Optional[BlueNexusDataClient]:
    """
    Get a BlueNexus data client for a user using their OAuth session.

    This retrieves the user's BlueNexus OAuth session and creates a client
    with their access token. The token must have the 'user-data' scope.

    Args:
        user_id: The Open WebUI user ID
        base_url: Optional BlueNexus API base URL. If not provided,
                 uses the configured BLUENEXUS_API_BASE_URL.

    Returns:
        BlueNexusDataClient if user has a valid BlueNexus session, None otherwise

    Example:
        client = get_bluenexus_client_for_user("user-123")
        if client:
            chats = await client.query("owui-chats")
    """
    # Check if BlueNexus is enabled
    if not ENABLE_BLUENEXUS.value:
        log.debug(f"[BlueNexus Factory] BlueNexus disabled, returning None for user {user_id}")
        return None

    # Get BlueNexus OAuth session for user
    log.debug(f"[BlueNexus Factory] Looking up BlueNexus OAuth session for user {user_id}")
    session = OAuthSessions.get_session_by_provider_and_user_id(
        provider="bluenexus",
        user_id=user_id,
    )

    if not session:
        log.debug(f"[BlueNexus Factory] No BlueNexus session found for user {user_id}")
        return None

    log.debug(f"[BlueNexus Factory] Found BlueNexus session for user {user_id}")

    # Extract access token
    token_data = session.token
    if not token_data or not isinstance(token_data, dict):
        log.warning(f"[BlueNexus Factory] Invalid token data for user {user_id}")
        return None

    access_token = token_data.get("access_token")
    if not access_token:
        log.warning(f"[BlueNexus Factory] No access token in session for user {user_id}")
        return None

    # Get base URL from config if not provided
    if base_url is None:
        base_url = BLUENEXUS_API_BASE_URL.value

    if not base_url:
        log.error("[BlueNexus Factory] BLUENEXUS_API_BASE_URL not configured")
        return None

    log.debug(f"[BlueNexus Factory] Creating BlueNexusDataClient for user {user_id}")

    return BlueNexusDataClient(
        base_url=base_url,
        access_token=access_token,
    )


def has_bluenexus_session(user_id: str) -> bool:
    """
    Check if a user has a valid BlueNexus OAuth session.

    Args:
        user_id: The Open WebUI user ID

    Returns:
        True if user has a BlueNexus session with access token
    """
    # Check if BlueNexus is enabled
    if not ENABLE_BLUENEXUS.value:
        return False

    log.debug(f"[BlueNexus Factory] Checking if user {user_id} has BlueNexus session")
    session = OAuthSessions.get_session_by_provider_and_user_id(
        provider="bluenexus",
        user_id=user_id,
    )

    if not session:
        log.debug(f"[BlueNexus Factory] User {user_id} has no BlueNexus session")
        return False

    token_data = session.token
    if not token_data or not isinstance(token_data, dict):
        log.debug(f"[BlueNexus Factory] User {user_id} has invalid token data")
        return False

    has_token = bool(token_data.get("access_token"))
    log.debug(f"[BlueNexus Factory] User {user_id} has_bluenexus_session={has_token}")
    return has_token


async def get_or_create_bluenexus_client(
    user_id: str,
    request=None,
    base_url: Optional[str] = None,
) -> Optional[BlueNexusDataClient]:
    """
    Get a BlueNexus client, refreshing the token if needed.

    This is the preferred method for getting a client in request handlers,
    as it can use the request's OAuth manager to refresh expired tokens.

    Args:
        user_id: The Open WebUI user ID
        request: Optional FastAPI request object (for token refresh)
        base_url: Optional BlueNexus API base URL

    Returns:
        BlueNexusDataClient if successful, None otherwise
    """
    # First try to get a simple client
    client = get_bluenexus_client_for_user(user_id, base_url)

    if client:
        return client

    # If request is provided and has oauth_manager, try to refresh
    if request and hasattr(request.app.state, "oauth_manager"):
        session = OAuthSessions.get_session_by_provider_and_user_id(
            provider="bluenexus",
            user_id=user_id,
        )

        if session:
            try:
                # Try to get a refreshed token
                token = await request.app.state.oauth_manager.get_oauth_token(
                    user_id,
                    session.id,
                    force_refresh=True,
                )

                if token and token.get("access_token"):
                    if base_url is None:
                        base_url = BLUENEXUS_API_BASE_URL.value

                    return BlueNexusDataClient(
                        base_url=base_url,
                        access_token=token["access_token"],
                    )
            except Exception as e:
                log.error(f"[BlueNexus Factory] Token refresh failed: {e}", exc_info=True)

    return None


class BlueNexusClientContext:
    """
    Context manager for BlueNexus client with automatic error handling.

    Usage:
        async with BlueNexusClientContext(user_id) as client:
            if client:
                await client.create("owui-chats", data)
    """

    def __init__(self, user_id: str, base_url: Optional[str] = None):
        self.user_id = user_id
        self.base_url = base_url
        self.client: Optional[BlueNexusDataClient] = None

    async def __aenter__(self) -> Optional[BlueNexusDataClient]:
        self.client = get_bluenexus_client_for_user(self.user_id, self.base_url)
        return self.client

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Handle BlueNexus auth errors specially
        if exc_type is not None and issubclass(exc_type, BlueNexusAuthError):
            log.warning(f"[BlueNexus Context] Auth error for user {self.user_id}: {exc_val}")
            # Don't suppress the exception
            return False
        return False
