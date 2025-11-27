import time
import uuid
from typing import Optional

from open_webui.internal.db import Base, get_db
from pydantic import BaseModel, ConfigDict
from sqlalchemy import BigInteger, Column, String, Text

####################
# Memory DB Schema
####################


class Memory(Base):
    __tablename__ = "memory"

    id = Column(String, primary_key=True)
    user_id = Column(String)
    content = Column(Text)
    updated_at = Column(BigInteger)
    created_at = Column(BigInteger)


class MemoryModel(BaseModel):
    id: str
    user_id: str
    content: str
    updated_at: int  # timestamp in epoch
    created_at: int  # timestamp in epoch

    model_config = ConfigDict(from_attributes=True)


####################
# Forms
####################


class MemoriesTable:
    def insert_new_memory(
        self,
        user_id: str,
        content: str,
    ) -> Optional[MemoryModel]:
        with get_db() as db:
            id = str(uuid.uuid4())

            memory = MemoryModel(
                **{
                    "id": id,
                    "user_id": user_id,
                    "content": content,
                    "created_at": int(time.time()),
                    "updated_at": int(time.time()),
                }
            )
            result = Memory(**memory.model_dump())
            db.add(result)
            db.commit()
            db.refresh(result)
            if result:
                validated_memory = MemoryModel.model_validate(result)

                # Sync to BlueNexus (non-blocking)
                try:
                    from open_webui.utils.bluenexus.sync_service import BlueNexusSync
                    from open_webui.env import log
                    log.info(f"[Memory Sync] Syncing memory {id} to BlueNexus")
                    BlueNexusSync.sync_memory_to_bluenexus_background(
                        id,
                        user_id,
                        validated_memory.model_dump(),
                        operation="create"
                    )
                except Exception as e:
                    from open_webui.env import log
                    log.error(f"[Memory Sync] Failed to sync memory {id}: {e}")
                    pass  # Don't fail if sync fails

                return validated_memory
            else:
                return None

    def update_memory_by_id_and_user_id(
        self,
        id: str,
        user_id: str,
        content: str,
    ) -> Optional[MemoryModel]:
        with get_db() as db:
            try:
                memory = db.get(Memory, id)
                if not memory or memory.user_id != user_id:
                    return None

                memory.content = content
                memory.updated_at = int(time.time())

                db.commit()
                validated_memory = self.get_memory_by_id(id)

                # Sync to BlueNexus (non-blocking)
                if validated_memory:
                    try:
                        from open_webui.utils.bluenexus.sync_service import BlueNexusSync
                        BlueNexusSync.sync_memory_to_bluenexus_background(
                            id,
                            user_id,
                            validated_memory.model_dump(),
                            operation="update"
                        )
                    except Exception:
                        pass  # Don't fail if sync fails

                return validated_memory
            except Exception:
                return None

    def get_memories(self) -> list[MemoryModel]:
        with get_db() as db:
            try:
                memories = db.query(Memory).all()
                return [MemoryModel.model_validate(memory) for memory in memories]
            except Exception:
                return None

    def get_memories_by_user_id(self, user_id: str) -> list[MemoryModel]:
        with get_db() as db:
            try:
                memories = db.query(Memory).filter_by(user_id=user_id).all()
                return [MemoryModel.model_validate(memory) for memory in memories]
            except Exception:
                return None

    def get_memory_by_id(self, id: str) -> Optional[MemoryModel]:
        with get_db() as db:
            try:
                memory = db.get(Memory, id)
                return MemoryModel.model_validate(memory)
            except Exception:
                return None

    def delete_memory_by_id(self, id: str) -> bool:
        with get_db() as db:
            try:
                db.query(Memory).filter_by(id=id).delete()
                db.commit()

                return True

            except Exception:
                return False

    def delete_memories_by_user_id(self, user_id: str) -> bool:
        with get_db() as db:
            try:
                db.query(Memory).filter_by(user_id=user_id).delete()
                db.commit()

                return True
            except Exception:
                return False

    def delete_memory_by_id_and_user_id(self, id: str, user_id: str) -> bool:
        with get_db() as db:
            try:
                memory = db.get(Memory, id)
                if not memory or memory.user_id != user_id:
                    return None

                # Delete the memory
                db.delete(memory)
                db.commit()

                # Sync to BlueNexus (non-blocking)
                try:
                    from open_webui.utils.bluenexus.sync_service import BlueNexusSync
                    BlueNexusSync.sync_memory_to_bluenexus_background(
                        id,
                        user_id,
                        None,
                        operation="delete"
                    )
                except Exception:
                    pass  # Don't fail if sync fails

                return True
            except Exception:
                return False


Memories = MemoriesTable()
