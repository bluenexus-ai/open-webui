"""
BlueNexus Sync Service

This module provides a sync-based approach for backing up Open WebUI data to BlueNexus.
Instead of replacing the local database, this service syncs data bidirectionally:

1. On user login: Pull latest data from BlueNexus
2. On data change: Push changes to BlueNexus (async, non-blocking)
3. Background sync: Periodic sync for conflict resolution

This approach preserves the local database as the source of truth while providing
cloud backup and cross-device sync capabilities.
"""

import logging
import asyncio

from open_webui.env import SRC_LOG_LEVELS
from open_webui.config import ENABLE_BLUENEXUS, ENABLE_BLUENEXUS_SYNC
from open_webui.utils.bluenexus.chat_storage import BlueNexusChatStorage
from open_webui.utils.bluenexus.factory import get_bluenexus_client_for_user
from open_webui.utils.bluenexus.collections import Collections
from open_webui.models.chats import Chats

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("BLUENEXUS", SRC_LOG_LEVELS.get("MAIN", logging.INFO)))


class BlueNexusSyncService:
    """
    Service for syncing Open WebUI data with BlueNexus.

    This service provides event hooks and sync methods to keep local data
    backed up to BlueNexus without replacing the local database.
    """

    def __init__(self):
        """Initialize sync service."""
        self._sync_tasks = {}  # Track background sync tasks

    # =========================================================================
    # Login Sync
    # =========================================================================

    async def sync_on_login(self, user_id: str) -> dict:
        """
        Sync user data from BlueNexus on login.

        This pulls the latest chats from BlueNexus and merges them with
        local data, using timestamps to resolve conflicts.

        Args:
            user_id: User ID to sync

        Returns:
            Dictionary with sync statistics:
            {
                "pulled": 5,      # Chats pulled from BlueNexus
                "updated": 2,     # Local chats updated
                "created": 3,     # New chats created locally
                "conflicts": 0,   # Conflicts resolved
            }
        """
        # Check if BlueNexus is enabled
        if not ENABLE_BLUENEXUS.value:
            log.debug(f"[BlueNexus Sync] BlueNexus disabled via ENABLE_BLUENEXUS, skipping login sync for user={user_id}")
            return {"pulled": 0, "updated": 0, "created": 0, "conflicts": 0}

        # Check if BlueNexus sync is enabled
        if not ENABLE_BLUENEXUS_SYNC.value:
            log.debug(f"[BlueNexus Sync] Sync disabled via ENABLE_BLUENEXUS_SYNC, skipping login sync for user={user_id}")
            return {"pulled": 0, "updated": 0, "created": 0, "conflicts": 0}

        log.info(f"[BlueNexus Sync] Starting login sync for user={user_id}")

        client = get_bluenexus_client_for_user(user_id)
        if not client:
            log.info(f"[BlueNexus Sync] User {user_id} has no BlueNexus session, skipping sync")
            return {"pulled": 0, "updated": 0, "created": 0, "conflicts": 0}

        try:
            bn_storage = BlueNexusChatStorage(client)

            # Get all chats from BlueNexus
            remote_chats = await bn_storage.get_chats_by_user_id(
                user_id=user_id,
                include_archived=True,
                skip=0,
                limit=1000,  # Pull all chats
            )

            log.info(f"[BlueNexus Sync] Found {len(remote_chats)} chats in BlueNexus for user={user_id}")

            stats = {
                "pulled": len(remote_chats),
                "updated": 0,
                "created": 0,
                "conflicts": 0,
            }

            # Sync each remote chat
            for remote_chat in remote_chats:
                result = await self._sync_chat_from_remote(user_id, remote_chat)
                if result == "created":
                    stats["created"] += 1
                elif result == "updated":
                    stats["updated"] += 1
                elif result == "conflict":
                    stats["conflicts"] += 1

            log.info(f"[BlueNexus Sync] Login sync complete for user={user_id}: {stats}")
            return stats

        except Exception as e:
            log.error(f"[BlueNexus Sync] Error during login sync for user={user_id}: {e}")
            return {"pulled": 0, "updated": 0, "created": 0, "conflicts": 0, "error": str(e)}

    async def _sync_chat_from_remote(self, user_id: str, remote_chat: dict) -> str:
        """
        Sync a single chat from BlueNexus to local database.

        Args:
            user_id: User ID
            remote_chat: Chat data from BlueNexus

        Returns:
            "created", "updated", "conflict", or "skipped"
        """
        chat_id = remote_chat.get("id")
        remote_updated_at = remote_chat.get("updated_at", 0)

        # Check if chat exists locally
        local_chat = Chats.get_chat_by_id(chat_id)

        if not local_chat:
            # Chat doesn't exist locally, create it
            log.debug(f"[BlueNexus Sync] Creating new local chat {chat_id} from BlueNexus")
            try:
                from open_webui.models.chats import ChatImportForm

                form = ChatImportForm(
                    chat=remote_chat.get("chat", {}),
                    meta=remote_chat.get("meta", {}),
                    pinned=remote_chat.get("pinned", False),
                    folder_id=remote_chat.get("folder_id"),
                    created_at=remote_chat.get("created_at"),
                    updated_at=remote_chat.get("updated_at"),
                )

                # Import chat with original ID
                Chats.import_chat(user_id, form)
                return "created"
            except Exception as e:
                log.error(f"[BlueNexus Sync] Error creating local chat {chat_id}: {e}")
                return "skipped"

        # Chat exists locally, check if remote is newer
        local_updated_at = local_chat.updated_at

        if remote_updated_at > local_updated_at:
            # Remote is newer, update local
            log.debug(f"[BlueNexus Sync] Updating local chat {chat_id} from BlueNexus (remote newer)")
            try:
                Chats.update_chat_by_id(chat_id, remote_chat.get("chat", {}))

                # Update metadata
                if remote_chat.get("pinned") != local_chat.pinned:
                    Chats.toggle_chat_pinned_by_id(chat_id)
                if remote_chat.get("archived") != local_chat.archived:
                    Chats.toggle_chat_archive_by_id(chat_id)

                return "updated"
            except Exception as e:
                log.error(f"[BlueNexus Sync] Error updating local chat {chat_id}: {e}")
                return "skipped"
        elif local_updated_at > remote_updated_at:
            # Local is newer, push to BlueNexus
            log.debug(f"[BlueNexus Sync] Local chat {chat_id} is newer, will push to BlueNexus")
            await self.sync_chat_to_bluenexus(chat_id, user_id)
            return "conflict"
        else:
            # Same timestamp, skip
            log.debug(f"[BlueNexus Sync] Chat {chat_id} is in sync, skipping")
            return "skipped"

    # =========================================================================
    # Push Sync (Local → BlueNexus)
    # =========================================================================

    async def sync_chat_to_bluenexus(
        self,
        chat_id: str,
        user_id: str,
        operation: str = "update",
    ) -> bool:
        """
        Push a chat to BlueNexus (async, non-blocking).

        This is called after local chat operations to backup data to BlueNexus.

        Args:
            chat_id: Chat ID to sync
            user_id: User ID
            operation: "create", "update", or "delete"

        Returns:
            True if sync was successful, False otherwise
        """
        # Check if BlueNexus is enabled
        if not ENABLE_BLUENEXUS.value:
            log.debug(f"[BlueNexus Sync] BlueNexus disabled via ENABLE_BLUENEXUS, skipping sync for chat {chat_id}")
            return False

        # Check if BlueNexus sync is enabled
        if not ENABLE_BLUENEXUS_SYNC.value:
            log.debug(f"[BlueNexus Sync] Sync disabled via ENABLE_BLUENEXUS_SYNC, skipping sync for chat {chat_id}")
            return False

        log.debug(f"[BlueNexus Sync] Syncing chat {chat_id} to BlueNexus (operation={operation})")

        client = get_bluenexus_client_for_user(user_id)
        if not client:
            log.debug(f"[BlueNexus Sync] User {user_id} has no BlueNexus session, skipping sync")
            return False

        try:
            bn_storage = BlueNexusChatStorage(client)

            if operation == "delete":
                # Delete from BlueNexus
                result = await bn_storage.delete_chat_by_id_and_user_id(chat_id, user_id)
                log.info(f"[BlueNexus Sync] Deleted chat {chat_id} from BlueNexus: {result}")
                return result

            # Get local chat
            local_chat = Chats.get_chat_by_id_and_user_id(chat_id, user_id)
            if not local_chat:
                log.warning(f"[BlueNexus Sync] Chat {chat_id} not found locally, skipping sync")
                return False

            # Check if chat exists in BlueNexus
            remote_chat = await bn_storage.get_chat_by_id_and_user_id(chat_id, user_id)

            if operation == "create" or not remote_chat:
                # Create new chat in BlueNexus
                log.debug(f"[BlueNexus Sync] Creating chat {chat_id} in BlueNexus")
                chat_data = local_chat.chat
                chat_data["id"] = chat_id  # Preserve original ID

                result = await bn_storage.insert_new_chat(
                    user_id=user_id,
                    chat_data=chat_data,
                    folder_id=local_chat.folder_id,
                )
                log.info(f"[BlueNexus Sync] Created chat {chat_id} in BlueNexus")
                return result is not None

            # Update existing chat in BlueNexus
            log.debug(f"[BlueNexus Sync] Updating chat {chat_id} in BlueNexus")
            result = await bn_storage.update_chat_by_id(chat_id, local_chat.chat)
            log.info(f"[BlueNexus Sync] Updated chat {chat_id} in BlueNexus")
            return result is not None

        except Exception as e:
            log.error(f"[BlueNexus Sync] Error syncing chat {chat_id} to BlueNexus: {e}")
            return False

    def sync_chat_to_bluenexus_background(
        self,
        chat_id: str,
        user_id: str,
        operation: str = "update",
    ):
        """
        Schedule a background sync to BlueNexus (fire-and-forget).

        This method is non-blocking and returns immediately.
        The actual sync happens in the background.

        Args:
            chat_id: Chat ID to sync
            user_id: User ID
            operation: "create", "update", or "delete"
        """
        # Create background task
        task_key = f"{user_id}:{chat_id}"

        # Cancel existing task if any
        if task_key in self._sync_tasks:
            self._sync_tasks[task_key].cancel()

        # Create new task
        try:
            loop = asyncio.get_event_loop()
            task = loop.create_task(
                self.sync_chat_to_bluenexus(chat_id, user_id, operation)
            )
            self._sync_tasks[task_key] = task

            # Clean up task when done
            def cleanup(_task):
                if task_key in self._sync_tasks:
                    del self._sync_tasks[task_key]

            task.add_done_callback(cleanup)

        except RuntimeError:
            # No event loop, sync will happen on next request
            log.debug(f"[BlueNexus Sync] No event loop, skipping background sync for {chat_id}")

    # =========================================================================
    # Bulk Sync
    # =========================================================================

    async def sync_all_chats_to_bluenexus(self, user_id: str) -> dict:
        """
        Sync all local chats to BlueNexus.

        This is useful for initial migration or manual sync.

        Args:
            user_id: User ID to sync

        Returns:
            Dictionary with sync statistics
        """
        # Check if BlueNexus is enabled
        if not ENABLE_BLUENEXUS.value:
            log.debug(f"[BlueNexus Sync] BlueNexus disabled via ENABLE_BLUENEXUS, skipping bulk sync for user={user_id}")
            return {"synced": 0, "failed": 0}

        # Check if BlueNexus sync is enabled
        if not ENABLE_BLUENEXUS_SYNC.value:
            log.debug(f"[BlueNexus Sync] Sync disabled via ENABLE_BLUENEXUS_SYNC, skipping bulk sync for user={user_id}")
            return {"synced": 0, "failed": 0}

        log.info(f"[BlueNexus Sync] Starting bulk sync for user={user_id}")

        client = get_bluenexus_client_for_user(user_id)
        if not client:
            log.info(f"[BlueNexus Sync] User {user_id} has no BlueNexus session, skipping sync")
            return {"synced": 0, "failed": 0}

        # Get all local chats
        local_chats = Chats.get_chats_by_user_id(user_id)

        stats = {"synced": 0, "failed": 0}

        for chat in local_chats:
            try:
                result = await self.sync_chat_to_bluenexus(chat.id, user_id, "create")
                if result:
                    stats["synced"] += 1
                else:
                    stats["failed"] += 1
            except Exception as e:
                log.error(f"[BlueNexus Sync] Error syncing chat {chat.id}: {e}")
                stats["failed"] += 1

        log.info(f"[BlueNexus Sync] Bulk sync complete for user={user_id}: {stats}")
        return stats

    # =========================================================================
    # Generic Model Sync Methods
    # =========================================================================

    async def sync_model_to_bluenexus(
        self,
        collection: str,
        record_id: str,
        user_id: str,
        data: dict,
        operation: str = "update",
    ) -> bool:
        """
        Generic method to sync any model to BlueNexus.

        Args:
            collection: BlueNexus collection name (e.g., Collections.PROMPTS)
            record_id: Record ID
            user_id: User ID
            data: Record data to sync
            operation: "create", "update", or "delete"

        Returns:
            True if sync was successful, False otherwise
        """
        log.info(f"[BlueNexus Sync] sync_model_to_bluenexus ASYNC called: collection={collection}, record_id={record_id}, operation={operation}")

        # Check if BlueNexus is enabled
        if not ENABLE_BLUENEXUS.value or not ENABLE_BLUENEXUS_SYNC.value:
            log.warning(f"[BlueNexus Sync] Sync disabled: ENABLE_BLUENEXUS={ENABLE_BLUENEXUS.value}, ENABLE_BLUENEXUS_SYNC={ENABLE_BLUENEXUS_SYNC.value}")
            return False

        log.info(f"[BlueNexus Sync] Syncing {collection}/{record_id} (operation={operation})")

        client = get_bluenexus_client_for_user(user_id)
        if not client:
            log.warning(f"[BlueNexus Sync] User {user_id} has no BlueNexus session, skipping sync")
            return False

        log.info(f"[BlueNexus Sync] Got BlueNexus client for user {user_id}")

        try:
            if operation == "delete":
                # Delete from BlueNexus
                result = await client.delete(collection, record_id)
                log.info(f"[BlueNexus Sync] Deleted {collection}/{record_id} from BlueNexus")
                return result is not None

            # Ensure data has required fields
            sync_data = {
                **data,
                "id": record_id,
                "user_id": user_id,
            }

            # Check if record exists in BlueNexus
            existing = await client.get(collection, record_id)

            if operation == "create" or not existing:
                # Create new record in BlueNexus
                result = await client.create(collection, sync_data)
                log.info(f"[BlueNexus Sync] Created {collection}/{record_id} in BlueNexus")
                return result is not None

            # Update existing record in BlueNexus
            result = await client.update(collection, record_id, sync_data)
            log.info(f"[BlueNexus Sync] Updated {collection}/{record_id} in BlueNexus")
            return result is not None

        except Exception as e:
            log.error(f"[BlueNexus Sync] Error syncing {collection}/{record_id} to BlueNexus: {e}")
            return False

    def sync_model_to_bluenexus_background(
        self,
        collection: str,
        record_id: str,
        user_id: str,
        data: dict = None,
        operation: str = "update",
    ):
        """
        Schedule a background sync for any model (fire-and-forget).

        Args:
            collection: BlueNexus collection name
            record_id: Record ID
            user_id: User ID
            data: Record data (required for create/update)
            operation: "create", "update", or "delete"
        """
        try:
            log.info(f"[BlueNexus Sync] sync_model_to_bluenexus_background called: collection={collection}, record_id={record_id}, user_id={user_id}, operation={operation}, data_exists={data is not None}")

            if operation != "delete" and not data:
                log.warning(f"[BlueNexus Sync] Cannot sync {collection}/{record_id}: data required for {operation}")
                return

            log.info(f"[BlueNexus Sync] Data validation passed, creating task key")
            task_key = f"{collection}:{user_id}:{record_id}"
            log.info(f"[BlueNexus Sync] Creating background task with key: {task_key}")

            # Cancel existing task if any
            if task_key in self._sync_tasks:
                self._sync_tasks[task_key].cancel()

            try:
                loop = asyncio.get_event_loop()
                log.info(f"[BlueNexus Sync] Got event loop, creating task")
                task = loop.create_task(
                    self.sync_model_to_bluenexus(collection, record_id, user_id, data or {}, operation)
                )
                self._sync_tasks[task_key] = task
                log.info(f"[BlueNexus Sync] Task created and registered")

                def cleanup(_task):
                    # Log task result or exception
                    try:
                        if _task.exception():
                            log.error(f"[BlueNexus Sync] Task failed with exception: {_task.exception()}", exc_info=_task.exception())
                        else:
                            result = _task.result()
                            log.info(f"[BlueNexus Sync] Task completed successfully: {result}")
                    except Exception as e:
                        log.error(f"[BlueNexus Sync] Error in cleanup callback: {e}")
                    finally:
                        if task_key in self._sync_tasks:
                            del self._sync_tasks[task_key]

                task.add_done_callback(cleanup)

            except RuntimeError as e:
                log.warning(f"[BlueNexus Sync] No event loop available: {e}, skipping background sync for {collection}/{record_id}")
        except Exception as e:
            log.error(f"[BlueNexus Sync] Unexpected error in sync_model_to_bluenexus_background: {e}", exc_info=True)

    # =========================================================================
    # Convenience Methods for Specific Models
    # =========================================================================

    def sync_message_to_bluenexus_background(
        self, message_id: str, user_id: str, data: dict = None, operation: str = "update"
    ):
        """Sync a message to BlueNexus (background)."""
        self.sync_model_to_bluenexus_background(
            Collections.MESSAGES, message_id, user_id, data, operation
        )

    def sync_prompt_to_bluenexus_background(
        self, command: str, user_id: str, data: dict = None, operation: str = "update"
    ):
        """Sync a prompt to BlueNexus (background)."""
        self.sync_model_to_bluenexus_background(
            Collections.PROMPTS, command, user_id, data, operation
        )

    def sync_memory_to_bluenexus_background(
        self, memory_id: str, user_id: str, data: dict = None, operation: str = "update"
    ):
        """Sync a memory to BlueNexus (background)."""
        self.sync_model_to_bluenexus_background(
            Collections.MEMORIES, memory_id, user_id, data, operation
        )

    def sync_note_to_bluenexus_background(
        self, note_id: str, user_id: str, data: dict = None, operation: str = "update"
    ):
        """Sync a note to BlueNexus (background)."""
        self.sync_model_to_bluenexus_background(
            Collections.NOTES, note_id, user_id, data, operation
        )


# Global singleton instance
BlueNexusSync = BlueNexusSyncService()
