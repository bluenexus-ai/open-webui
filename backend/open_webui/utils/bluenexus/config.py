"""
BlueNexus Configuration Module

This module contains all BlueNexus-specific configuration variables.
All configuration is gated by the ENABLE_BLUENEXUS flag.
"""

import os
import sys
from typing import TYPE_CHECKING

# Avoid circular import when open_webui.config imports this module
if TYPE_CHECKING:
    from open_webui.config import PersistentConfig
else:
    _config_mod = sys.modules.get("open_webui.config")
    if _config_mod and hasattr(_config_mod, "PersistentConfig"):
        PersistentConfig = getattr(_config_mod, "PersistentConfig")
    else:
        from open_webui.config import PersistentConfig


####################################
# BlueNexus OAuth Configuration
####################################

BLUENEXUS_CLIENT_ID = PersistentConfig(
    "BLUENEXUS_CLIENT_ID",
    "oauth.bluenexus.client_id",
    os.environ.get("BLUENEXUS_CLIENT_ID", ""),
)

BLUENEXUS_CLIENT_SECRET = PersistentConfig(
    "BLUENEXUS_CLIENT_SECRET",
    "oauth.bluenexus.client_secret",
    os.environ.get("BLUENEXUS_CLIENT_SECRET", ""),
)

BLUENEXUS_OAUTH_SCOPE = PersistentConfig(
    "BLUENEXUS_OAUTH_SCOPE",
    "oauth.bluenexus.scope",
    os.environ.get("BLUENEXUS_OAUTH_SCOPE", "account auth-sessions llm-all mcp-proxy user-data connections providers"),
)

BLUENEXUS_REDIRECT_URI = PersistentConfig(
    "BLUENEXUS_REDIRECT_URI",
    "oauth.bluenexus.redirect_uri",
    os.environ.get("BLUENEXUS_REDIRECT_URI", ""),
)

BLUENEXUS_API_BASE_URL = PersistentConfig(
    "BLUENEXUS_API_BASE_URL",
    "oauth.bluenexus.api_base_url",
    os.environ.get("BLUENEXUS_API_BASE_URL", "https://localhost:3000"),
)

BLUENEXUS_AUTHORIZATION_URL = PersistentConfig(
    "BLUENEXUS_AUTHORIZATION_URL",
    "oauth.bluenexus.authorization_url",
    os.environ.get("BLUENEXUS_AUTHORIZATION_URL", "http://localhost:3001/oauth/authorize"),
)

BLUENEXUS_TOKEN_URL = PersistentConfig(
    "BLUENEXUS_TOKEN_URL",
    "oauth.bluenexus.token_url",
    os.environ.get("BLUENEXUS_TOKEN_URL", "https://localhost:3000/api/v1/auth/token"),
)

BLUENEXUS_LLM_API_BASE_URL = PersistentConfig(
    "BLUENEXUS_LLM_API_BASE_URL",
    "oauth.bluenexus.llm_api_base_url",
    os.environ.get(
        "BLUENEXUS_LLM_API_BASE_URL",
        f"{BLUENEXUS_API_BASE_URL.value.rstrip('/')}/api/v1",
    ),
)

BLUENEXUS_LLM_AUTO_ENABLE = PersistentConfig(
    "BLUENEXUS_LLM_AUTO_ENABLE",
    "oauth.bluenexus.llm_auto_enable",
    os.environ.get("BLUENEXUS_LLM_AUTO_ENABLE", "True").lower() == "true",
)

####################################
# BlueNexus Master Enable Flag
####################################

ENABLE_BLUENEXUS = PersistentConfig(
    "ENABLE_BLUENEXUS",
    "bluenexus.enable",
    os.environ.get("ENABLE_BLUENEXUS", "True").lower() == "true",
)

####################################
# BlueNexus Data Sync
####################################

ENABLE_BLUENEXUS_SYNC = PersistentConfig(
    "ENABLE_BLUENEXUS_SYNC",
    "bluenexus.sync.enable",
    os.environ.get("ENABLE_BLUENEXUS_SYNC", "True").lower() == "true",
)


def is_bluenexus_enabled() -> bool:
    """Check if BlueNexus is enabled."""
    return ENABLE_BLUENEXUS.value


def is_bluenexus_sync_enabled() -> bool:
    """Check if BlueNexus sync is enabled."""
    return ENABLE_BLUENEXUS.value and ENABLE_BLUENEXUS_SYNC.value


def is_bluenexus_configured() -> bool:
    """Check if BlueNexus OAuth is properly configured."""
    return bool(BLUENEXUS_CLIENT_ID.value and BLUENEXUS_CLIENT_SECRET.value)
