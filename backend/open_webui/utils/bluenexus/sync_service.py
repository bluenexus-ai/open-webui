"""
BlueNexus Sync Service

This module provides push-based sync for backing up Open WebUI data to BlueNexus.
The local database is the source of truth, with changes automatically pushed to BlueNexus:

1. On data change: Push changes to BlueNexus (async, non-blocking)
2. All operations (create/update/delete) trigger background sync
3. Generic sync methods support all model types (chats, notes, prompts, memories, messages)

This approach preserves the local database as the source of truth while providing
cloud backup capabilities.
"""

import logging
import asyncio

from open_webui.env import SRC_LOG_LEVELS
from open_webui.config import ENABLE_BLUENEXUS, ENABLE_BLUENEXUS_SYNC
from open_webui.utils.bluenexus.factory import get_bluenexus_client_for_user
from open_webui.utils.bluenexus.types import QueryOptions
from open_webui.utils.bluenexus.collections import Collections

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("BLUENEXUS", SRC_LOG_LEVELS.get("MAIN", logging.INFO)))


class BlueNexusSyncService:
    """
    Service for syncing Open WebUI data to BlueNexus.

    This service provides push-based sync to backup local data to BlueNexus.
    The local database remains the source of truth.
    """

    def __init__(self):
        """Initialize sync service."""
        self._sync_tasks = {}  # Track background sync tasks

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
        collection_name = getattr(collection, "value", collection)
        log.info(f"[BlueNexus Sync] sync_model_to_bluenexus ASYNC called: collection={collection_name}, record_id={record_id}, operation={operation}")

        # Check if BlueNexus is enabled
        if not ENABLE_BLUENEXUS.value or not ENABLE_BLUENEXUS_SYNC.value:
            log.warning(f"[BlueNexus Sync] Sync disabled: ENABLE_BLUENEXUS={ENABLE_BLUENEXUS.value}, ENABLE_BLUENEXUS_SYNC={ENABLE_BLUENEXUS_SYNC.value}")
            return False

        log.info(f"[BlueNexus Sync] Syncing {collection_name}/{record_id} (operation={operation})")

        client = get_bluenexus_client_for_user(user_id)
        if not client:
            log.warning(f"[BlueNexus Sync] User {user_id} has no BlueNexus session, skipping sync")
            return False

        log.info(f"[BlueNexus Sync] Got BlueNexus client for user {user_id}")

        try:
            # Helper function to find BlueNexus record ID by Open WebUI ID
            async def find_bluenexus_id(owui_id: str) -> str | None:
                """Query BlueNexus to find the MongoDB ID for a given Open WebUI ID."""
                try:
                    log.debug(f"[BlueNexus Sync] Querying for record with owui_id={owui_id}")
                    response = await client.query(
                        collection_name,
                        QueryOptions(filter={"owui_id": owui_id}, limit=1)
                    )
                    if response.data and len(response.data) > 0:
                        # BlueNexus returns records with MongoDB's _id as "id"
                        bluenexus_id = response.get_records()[0].id
                        log.info(f"[BlueNexus Sync] Found BlueNexus ID {bluenexus_id} for Open WebUI ID {owui_id}")
                        return bluenexus_id
                    log.debug(f"[BlueNexus Sync] No record found with id={owui_id}")
                    return None
                except Exception as e:
                    log.error(f"[BlueNexus Sync] Error querying for record: {e}")
                    return None

            if operation == "delete":
                # Find the BlueNexus ID first by querying with Open WebUI ID
                bluenexus_id = await find_bluenexus_id(record_id)
                if not bluenexus_id:
                    log.debug(f"[BlueNexus Sync] Delete skipped for {collection_name}/{record_id}: record not found in BlueNexus (may not have been synced yet)")
                    return True

                # Delete using the BlueNexus MongoDB ID
                await client.delete(collection_name, bluenexus_id)
                log.info(f"[BlueNexus Sync] Deleted {collection_name}/{record_id} (BlueNexus ID: {bluenexus_id}) from BlueNexus")
                return True

            # Ensure data has required fields
            # Note: We use "owui_id" instead of "id" because "id" is reserved
            # for MongoDB's _id in the BlueNexus schema transform
            sync_data = {
                **data,
                "owui_id": record_id,  # Open WebUI's original ID
                "user_id": user_id,
            }

            if operation == "create":
                # Create new record in BlueNexus
                result = await client.create(collection_name, sync_data)
                log.info(f"[BlueNexus Sync] Created {collection_name}/{record_id} in BlueNexus (BlueNexus ID: {result.id})")
                return result is not None

            # For UPDATE operation, find existing record by Open WebUI ID
            bluenexus_id = await find_bluenexus_id(record_id)

            if not bluenexus_id:
                # Record doesn't exist, create it instead
                log.info(f"[BlueNexus Sync] Record {collection_name}/{record_id} not found, creating instead of updating")
                result = await client.create(collection_name, sync_data)
                log.info(f"[BlueNexus Sync] Created {collection_name}/{record_id} in BlueNexus (BlueNexus ID: {result.id})")
                return result is not None

            # Update existing record using BlueNexus MongoDB ID
            result = await client.update(collection_name, bluenexus_id, sync_data)
            log.info(f"[BlueNexus Sync] Updated {collection_name}/{record_id} (BlueNexus ID: {bluenexus_id}) in BlueNexus")
            return result is not None

        except Exception as e:
            log.error(f"[BlueNexus Sync] Error syncing {collection_name}/{record_id} to BlueNexus: {e}")
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
    # High-Level Helper Methods
    # =========================================================================

    def sync_create(self, collection: str, model_instance, record_id: str = None):
        """
        Sync a newly created model instance to BlueNexus.

        Args:
            collection: Collection name (e.g., Collections.CHATS)
            model_instance: Model instance with user_id and model_dump() method
            record_id: Optional explicit record ID (uses model_instance.id if not provided)

        Example:
            chat = Chats.insert_new_chat(user.id, form_data)
            BlueNexusSync.sync_create(Collections.CHATS, chat)

            # For models with different ID fields:
            prompt = Prompts.insert_new_prompt(user.id, form_data)
            BlueNexusSync.sync_create(Collections.PROMPTS, prompt, prompt.command)
        """
        # Use explicit record_id if provided, otherwise try to get 'id' attribute
        if record_id is None:
            record_id = getattr(model_instance, 'id', None)
            if record_id is None:
                # Try 'command' for prompts
                record_id = getattr(model_instance, 'command', None)

        self.sync_model_to_bluenexus_background(
            collection,
            record_id,
            model_instance.user_id,
            model_instance.model_dump(),
            operation="create"
        )

    def sync_update(self, collection: str, model_instance, record_id: str = None):
        """
        Sync an updated model instance to BlueNexus.

        Args:
            collection: Collection name (e.g., Collections.NOTES)
            model_instance: Model instance with user_id and model_dump() method
            record_id: Optional explicit record ID (uses model_instance.id if not provided)

        Example:
            note = Notes.update_note_by_id(id, form_data)
            BlueNexusSync.sync_update(Collections.NOTES, note)

            # For models with different ID fields:
            prompt = Prompts.update_prompt_by_command(command, form_data)
            BlueNexusSync.sync_update(Collections.PROMPTS, prompt, prompt.command)
        """
        # Use explicit record_id if provided, otherwise try to get 'id' attribute
        if record_id is None:
            record_id = getattr(model_instance, 'id', None)
            if record_id is None:
                # Try 'command' for prompts
                record_id = getattr(model_instance, 'command', None)

        self.sync_model_to_bluenexus_background(
            collection,
            record_id,
            model_instance.user_id,
            model_instance.model_dump(),
            operation="update"
        )

    def sync_delete(self, collection: str, record_id: str, user_id: str):
        """
        Sync a deletion to BlueNexus.

        Args:
            collection: Collection name (e.g., Collections.PROMPTS)
            record_id: ID of the deleted record
            user_id: User ID who owns the record

        Example:
            # Capture user_id before deletion
            user_id = prompt.user_id
            Prompts.delete_prompt_by_command(command)
            BlueNexusSync.sync_delete(Collections.PROMPTS, command, user_id)
        """
        self.sync_model_to_bluenexus_background(
            collection,
            record_id,
            user_id,
            None,
            operation="delete"
        )

    # =========================================================================
    # Convenience Methods for Specific Models
    # =========================================================================

    def sync_chat_to_bluenexus_background(
        self, chat_id: str, user_id: str, data: dict = None, operation: str = "update"
    ):
        """Sync a message to BlueNexus (background)."""
        self.sync_model_to_bluenexus_background(
            Collections.CHATS, chat_id, user_id, data, operation
        )

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
