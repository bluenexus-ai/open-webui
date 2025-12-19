"""
BlueNexus OAuth Integration Module

This module handles OAuth provider registration and authentication for BlueNexus.
"""

import httpx
import ssl
import warnings
import urllib3
from urllib.parse import urlparse
from authlib.integrations.starlette_client import OAuth

from open_webui.utils.bluenexus.config import (
    BLUENEXUS_CLIENT_ID,
    BLUENEXUS_CLIENT_SECRET,
    BLUENEXUS_OAUTH_SCOPE,
    BLUENEXUS_REDIRECT_URI,
    BLUENEXUS_API_BASE_URL,
    BLUENEXUS_AUTHORIZATION_URL,
    BLUENEXUS_TOKEN_URL,
    is_bluenexus_enabled,
    is_bluenexus_configured,
)


def _is_localhost_url(url: str) -> bool:
    """Check if a URL points to localhost or Docker host."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        return hostname in ["localhost", "127.0.0.1", "::1", "host.docker.internal"]
    except Exception:
        return False


def _should_disable_ssl_for_bluenexus() -> bool:
    """
    Check if SSL should be disabled for BlueNexus based on URL.
    Only disables SSL for localhost development.
    """
    api_url = BLUENEXUS_API_BASE_URL.value
    token_url = BLUENEXUS_TOKEN_URL.value
    return _is_localhost_url(api_url) or _is_localhost_url(token_url)


def get_bluenexus_ssl_context():
    """
    Get SSL context for BlueNexus connections.
    Disables SSL verification only for localhost development.
    """
    if not _should_disable_ssl_for_bluenexus():
        return None  # Use default SSL verification

    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    return ssl_context


def register_bluenexus_oauth(oauth: OAuth, oauth_timeout: str = ""):
    """
    Register BlueNexus as an OAuth provider.

    Args:
        oauth: OAuth instance to register with
        oauth_timeout: Optional timeout value

    Returns:
        Registered OAuth client or None if not configured
    """
    if not is_bluenexus_enabled():
        return None

    if not is_bluenexus_configured():
        return None

    # Only disable SSL for localhost development
    disable_ssl = _should_disable_ssl_for_bluenexus()

    if disable_ssl:
        # Filter SSL warnings for localhost development only (module-scoped, not global)
        warnings.filterwarnings(
            "ignore",
            message="Unverified HTTPS request",
            category=urllib3.exceptions.InsecureRequestWarning,
            module="urllib3"
        )

    # Create httpx client with SSL verification based on environment
    httpx_client = httpx.AsyncClient(
        verify=not disable_ssl,
        trust_env=not disable_ssl
    )

    client = oauth.register(
        name="bluenexus",
        client_id=BLUENEXUS_CLIENT_ID.value,
        client_secret=BLUENEXUS_CLIENT_SECRET.value,
        access_token_url=BLUENEXUS_TOKEN_URL.value,
        authorize_url=BLUENEXUS_AUTHORIZATION_URL.value,
        api_base_url=BLUENEXUS_API_BASE_URL.value,
        userinfo_endpoint=f"{BLUENEXUS_API_BASE_URL.value}/api/v1/accounts/me",
        server_metadata_url=None,
        client_kwargs={
            "scope": BLUENEXUS_OAUTH_SCOPE.value,
            "code_challenge_method": "S256",
            "token_endpoint_auth_method": "client_secret_post",
            "verify": not disable_ssl,
            **(
                {"timeout": int(oauth_timeout)}
                if oauth_timeout
                else {}
            ),
        },
        redirect_uri=BLUENEXUS_REDIRECT_URI.value,
        client=httpx_client,
    )
    return client


def get_bluenexus_oauth_provider_config(oauth_timeout: str = "") -> dict:
    """
    Get BlueNexus OAuth provider configuration.

    Returns:
        Provider configuration dict or None if not enabled/configured
    """
    if not is_bluenexus_enabled():
        return None

    if not is_bluenexus_configured():
        return None

    return {
        "redirect_uri": BLUENEXUS_REDIRECT_URI.value,
        "register": lambda oauth: register_bluenexus_oauth(oauth, oauth_timeout),
        "sub_claim": "id",
        "email_claim": "email",
        "username_claim": "name",
        "picture_claim": "avatar",
    }


def should_disable_ssl_for_provider(provider: str) -> bool:
    """
    Check if SSL verification should be disabled for a provider.
    Only returns True for BlueNexus when connecting to localhost.

    Args:
        provider: OAuth provider name

    Returns:
        True if SSL should be disabled, False otherwise
    """
    if not is_bluenexus_enabled():
        return False

    if provider != "bluenexus":
        return False

    # Only disable SSL for localhost development
    return _should_disable_ssl_for_bluenexus()
