"""
BlueNexus LLM Provider Integration Module

This module handles automatic registration of BlueNexus as an OpenAI-compatible LLM provider.
"""

import logging
import ssl
from typing import Optional
from urllib.parse import urlparse

from fastapi import Request
from open_webui.models.users import UserModel
from open_webui.models.oauth_sessions import OAuthSessions

from open_webui.utils.bluenexus.config import (
    BLUENEXUS_LLM_API_BASE_URL,
    BLUENEXUS_LLM_AUTO_ENABLE,
    is_bluenexus_enabled,
)

log = logging.getLogger(__name__)


def get_ssl_context_for_url(url: str):
    """
    Get SSL context for a URL. Disables SSL verification for localhost.

    Args:
        url: URL to get SSL context for

    Returns:
        SSL context or False for httpx compatibility
    """
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname or ""

    # Disable SSL verification for localhost
    if hostname in ["localhost", "127.0.0.1", "::1"]:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        return ssl_context

    # Return None to use default SSL settings
    return None


def ensure_bluenexus_provider(request: Request, user: UserModel) -> None:
    """
    Dynamically register BlueNexus as an OpenAI-compatible provider when the user
    authenticates via BlueNexus OAuth. Uses per-user OAuth tokens via `system_oauth`.

    Args:
        request: FastAPI request object
        user: Current user model
    """
    if not is_bluenexus_enabled():
        return

    if not BLUENEXUS_LLM_AUTO_ENABLE.value:
        return

    session_id = request.cookies.get("oauth_session_id")
    if not session_id or not user:
        return

    try:
        session = OAuthSessions.get_session_by_id_and_user_id(session_id, user.id)
        if not session or session.provider != "bluenexus":
            return

        base_url = BLUENEXUS_LLM_API_BASE_URL.value
        if not base_url:
            return

        base_url = base_url.rstrip("/")

        # Check if already registered
        base_urls = request.app.state.config.OPENAI_API_BASE_URLS
        if base_url not in base_urls:
            request.app.state.config.OPENAI_API_BASE_URLS = [*base_urls, base_url]
            request.app.state.config.OPENAI_API_KEYS = [
                *request.app.state.config.OPENAI_API_KEYS,
                "",
            ]
            request.app.state.BASE_MODELS = None
            request.app.state.MODELS = None

        # Configure API settings
        idx = request.app.state.config.OPENAI_API_BASE_URLS.index(base_url)
        api_configs = request.app.state.config.OPENAI_API_CONFIGS

        if (str(idx) not in api_configs) and (base_url not in api_configs):
            updated_configs = {**api_configs}
            updated_configs[str(idx)] = {
                "name": "BlueNexus",
                "auth_type": "system_oauth",
                "oauth_provider": "bluenexus",
                "enable": True,
                "connection_type": "external",
                "tags": ["bluenexus"],
            }
            request.app.state.config.OPENAI_API_CONFIGS = updated_configs

    except Exception as e:
        log.error(f"Error enabling BlueNexus LLM provider: {e}")


def get_bluenexus_oauth_token_for_headers(user_id: str) -> Optional[str]:
    """
    Get BlueNexus OAuth access token for authorization headers.

    Args:
        user_id: User ID

    Returns:
        Access token string or None if not found
    """
    if not is_bluenexus_enabled():
        return None

    try:
        session = OAuthSessions.get_session_by_provider_and_user_id(
            provider="bluenexus",
            user_id=user_id
        )
        if session:
            return session.token.get("access_token")
    except Exception as e:
        log.warning(f"Failed to get BlueNexus OAuth token for user {user_id}: {e}")

    return None
