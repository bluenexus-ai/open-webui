"""
Collection constants and helpers for BlueNexus data storage.

Defines the collection names used by Open WebUI when storing data in BlueNexus.
"""

from enum import Enum


class Collections(str, Enum):
    """
    Collection names for Open WebUI data in BlueNexus.

    Each collection stores a specific type of user data.
    The prefix 'owui-' is used to namespace Open WebUI data.
    """

    # Core chat data
    CHATS = "owui-chats"
    MESSAGES = "owui-messages"

    # Organization
    FOLDERS = "owui-folders"
    TAGS = "owui-tags"

    # Files and knowledge
    FILES = "owui-files"
    KNOWLEDGE = "owui-knowledge"

    # Customization
    PROMPTS = "owui-prompts"
    TOOLS = "owui-tools"
    FUNCTIONS = "owui-functions"
    MODELS = "owui-models"

    # Personal data
    NOTES = "owui-notes"
    MEMORIES = "owui-memories"

    # User settings and preferences
    SETTINGS = "owui-settings"

    # Collaboration
    GROUPS = "owui-groups"
    CHANNELS = "owui-channels"

    # Feedback
    FEEDBACKS = "owui-feedbacks"


# Mapping from Open WebUI model names to BlueNexus collection names
MODEL_TO_COLLECTION = {
    "chat": Collections.CHATS,
    "message": Collections.MESSAGES,
    "folder": Collections.FOLDERS,
    "tag": Collections.TAGS,
    "file": Collections.FILES,
    "knowledge": Collections.KNOWLEDGE,
    "prompt": Collections.PROMPTS,
    "tool": Collections.TOOLS,
    "function": Collections.FUNCTIONS,
    "model": Collections.MODELS,
    "note": Collections.NOTES,
    "memory": Collections.MEMORIES,
    "group": Collections.GROUPS,
    "channel": Collections.CHANNELS,
    "feedback": Collections.FEEDBACKS,
}
