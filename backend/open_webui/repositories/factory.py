"""
Repository Factory

Returns the correct repository implementation based on BLUENEXUS_DATA_STORAGE config.

Usage:
    from open_webui.repositories import get_chat_repository

    # Get repository for current user
    repo = get_chat_repository(user_id)

    # Use async methods
    chats = await repo.get_list(user_id, page=1)
"""

import logging
from typing import Union

from open_webui.utils.bluenexus.config import is_bluenexus_data_storage_enabled
from open_webui.env import SRC_LOG_LEVELS

from open_webui.repositories.base import (
    BaseChatRepository,
    BasePromptRepository,
    BaseModelRepository,
    BaseToolRepository,
)
from open_webui.repositories.chat import PostgresChatRepository, BlueNexusChatRepository
from open_webui.repositories.prompt import PostgresPromptRepository, BlueNexusPromptRepository
from open_webui.repositories.model import PostgresModelRepository, BlueNexusModelRepository
from open_webui.repositories.tool import PostgresToolRepository, BlueNexusToolRepository

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("REPOSITORIES", logging.INFO))


def get_chat_repository(user_id: str) -> BaseChatRepository:
    """
    Get the appropriate chat repository based on storage configuration.

    Args:
        user_id: The user ID for BlueNexus client initialization

    Returns:
        BaseChatRepository: Either PostgresChatRepository or BlueNexusChatRepository
    """
    if is_bluenexus_data_storage_enabled():
        log.debug(f"[Repository Factory] Using BlueNexus chat repository for user {user_id}")
        return BlueNexusChatRepository(user_id)
    else:
        log.debug(f"[Repository Factory] Using PostgreSQL chat repository")
        return PostgresChatRepository()


def get_prompt_repository(user_id: str) -> BasePromptRepository:
    """
    Get the appropriate prompt repository based on storage configuration.

    Args:
        user_id: The user ID for BlueNexus client initialization

    Returns:
        BasePromptRepository: Either PostgresPromptRepository or BlueNexusPromptRepository
    """
    if is_bluenexus_data_storage_enabled():
        log.debug(f"[Repository Factory] Using BlueNexus prompt repository for user {user_id}")
        return BlueNexusPromptRepository(user_id)
    else:
        log.debug(f"[Repository Factory] Using PostgreSQL prompt repository")
        return PostgresPromptRepository()


def get_model_repository(user_id: str) -> BaseModelRepository:
    """
    Get the appropriate model repository based on storage configuration.

    Args:
        user_id: The user ID for BlueNexus client initialization

    Returns:
        BaseModelRepository: Either PostgresModelRepository or BlueNexusModelRepository
    """
    if is_bluenexus_data_storage_enabled():
        log.debug(f"[Repository Factory] Using BlueNexus model repository for user {user_id}")
        return BlueNexusModelRepository(user_id)
    else:
        log.debug(f"[Repository Factory] Using PostgreSQL model repository")
        return PostgresModelRepository()


def get_tool_repository(user_id: str) -> BaseToolRepository:
    """
    Get the appropriate tool repository based on storage configuration.

    Args:
        user_id: The user ID for BlueNexus client initialization

    Returns:
        BaseToolRepository: Either PostgresToolRepository or BlueNexusToolRepository
    """
    if is_bluenexus_data_storage_enabled():
        log.debug(f"[Repository Factory] Using BlueNexus tool repository for user {user_id}")
        return BlueNexusToolRepository(user_id)
    else:
        log.debug(f"[Repository Factory] Using PostgreSQL tool repository")
        return PostgresToolRepository()
