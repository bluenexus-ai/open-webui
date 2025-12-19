"""
Repository pattern for data storage abstraction.

This module provides an abstraction layer between routers and data storage,
allowing seamless switching between PostgreSQL and BlueNexus storage backends.

Usage:
    from open_webui.repositories import get_chat_repository

    repo = get_chat_repository(user_id)
    chats = await repo.get_list(page=1, limit=60)
"""

from open_webui.repositories.factory import (
    get_chat_repository,
    get_prompt_repository,
    get_model_repository,
    get_tool_repository,
)

__all__ = [
    "get_chat_repository",
    "get_prompt_repository",
    "get_model_repository",
    "get_tool_repository",
]
