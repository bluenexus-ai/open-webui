"""
Abstract base classes for repository pattern.

These define the interface that both PostgreSQL and BlueNexus implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Any


class BaseChatRepository(ABC):
    """Abstract base class for chat data operations."""

    @abstractmethod
    async def get_list(
        self,
        user_id: str,
        page: int = 1,
        limit: int = 60,
        include_archived: bool = False,
        include_pinned: bool = False,
        include_folders: bool = False,
    ) -> List[dict]:
        """Get list of chats for a user."""
        pass

    @abstractmethod
    async def get_by_id(self, chat_id: str, user_id: str) -> Optional[dict]:
        """Get a specific chat by ID."""
        pass

    @abstractmethod
    async def get_by_share_id(self, share_id: str) -> Optional[dict]:
        """Get a chat by its share ID."""
        pass

    @abstractmethod
    async def create(self, user_id: str, data: dict) -> dict:
        """Create a new chat."""
        pass

    @abstractmethod
    async def update(self, chat_id: str, user_id: str, data: dict) -> Optional[dict]:
        """Update an existing chat."""
        pass

    @abstractmethod
    async def delete(self, chat_id: str, user_id: str) -> bool:
        """Delete a chat."""
        pass

    @abstractmethod
    async def delete_all_by_user(self, user_id: str) -> bool:
        """Delete all chats for a user."""
        pass

    @abstractmethod
    async def share(self, chat_id: str, user_id: str) -> Optional[dict]:
        """Share a chat and return updated chat with share_id."""
        pass

    @abstractmethod
    async def unshare(self, chat_id: str, user_id: str) -> bool:
        """Unshare a chat."""
        pass

    @abstractmethod
    async def archive(self, chat_id: str, user_id: str) -> Optional[dict]:
        """Archive a chat."""
        pass

    @abstractmethod
    async def unarchive(self, chat_id: str, user_id: str) -> Optional[dict]:
        """Unarchive a chat."""
        pass

    @abstractmethod
    async def pin(self, chat_id: str, user_id: str) -> Optional[dict]:
        """Pin a chat."""
        pass

    @abstractmethod
    async def unpin(self, chat_id: str, user_id: str) -> Optional[dict]:
        """Unpin a chat."""
        pass

    @abstractmethod
    async def clone(self, chat_id: str, user_id: str) -> Optional[dict]:
        """Clone a chat."""
        pass

    @abstractmethod
    async def get_archived(self, user_id: str, page: int = 1, limit: int = 60) -> List[dict]:
        """Get archived chats for a user."""
        pass

    @abstractmethod
    async def get_pinned(self, user_id: str) -> List[dict]:
        """Get pinned chats for a user."""
        pass

    @abstractmethod
    async def search(self, user_id: str, query: str, page: int = 1, limit: int = 60) -> List[dict]:
        """Search chats by title or content."""
        pass


class BasePromptRepository(ABC):
    """Abstract base class for prompt data operations."""

    @abstractmethod
    async def get_list(self, user_id: str) -> List[dict]:
        """Get list of prompts for a user."""
        pass

    @abstractmethod
    async def get_by_command(self, command: str, user_id: str) -> Optional[dict]:
        """Get a prompt by its command."""
        pass

    @abstractmethod
    async def create(self, user_id: str, data: dict) -> dict:
        """Create a new prompt."""
        pass

    @abstractmethod
    async def update(self, command: str, user_id: str, data: dict) -> Optional[dict]:
        """Update an existing prompt."""
        pass

    @abstractmethod
    async def delete(self, command: str, user_id: str) -> bool:
        """Delete a prompt."""
        pass


class BaseModelRepository(ABC):
    """Abstract base class for model data operations."""

    @abstractmethod
    async def get_list(self, user_id: str) -> List[dict]:
        """Get list of models for a user."""
        pass

    @abstractmethod
    async def get_all(self) -> List[dict]:
        """Get all models (admin)."""
        pass

    @abstractmethod
    async def get_by_id(self, model_id: str) -> Optional[dict]:
        """Get a model by ID."""
        pass

    @abstractmethod
    async def create(self, user_id: str, data: dict) -> dict:
        """Create a new model."""
        pass

    @abstractmethod
    async def update(self, model_id: str, data: dict) -> Optional[dict]:
        """Update an existing model."""
        pass

    @abstractmethod
    async def delete(self, model_id: str) -> bool:
        """Delete a model."""
        pass


class BaseToolRepository(ABC):
    """Abstract base class for tool data operations."""

    @abstractmethod
    async def get_list(self, user_id: str) -> List[dict]:
        """Get list of tools for a user."""
        pass

    @abstractmethod
    async def get_all(self) -> List[dict]:
        """Get all tools (admin)."""
        pass

    @abstractmethod
    async def get_by_id(self, tool_id: str) -> Optional[dict]:
        """Get a tool by ID."""
        pass

    @abstractmethod
    async def create(self, user_id: str, data: dict) -> dict:
        """Create a new tool."""
        pass

    @abstractmethod
    async def update(self, tool_id: str, data: dict) -> Optional[dict]:
        """Update an existing tool."""
        pass

    @abstractmethod
    async def delete(self, tool_id: str) -> bool:
        """Delete a tool."""
        pass
