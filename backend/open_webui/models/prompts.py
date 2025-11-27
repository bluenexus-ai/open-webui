import time
from typing import Optional

from open_webui.internal.db import Base, get_db
from open_webui.models.groups import Groups
from open_webui.models.users import Users, UserResponse

from pydantic import BaseModel, ConfigDict
from sqlalchemy import BigInteger, Column, String, Text, JSON

from open_webui.utils.access_control import has_access

####################
# Prompts DB Schema
####################


class Prompt(Base):
    __tablename__ = "prompt"

    command = Column(String, primary_key=True)
    user_id = Column(String)
    title = Column(Text)
    content = Column(Text)
    timestamp = Column(BigInteger)

    access_control = Column(JSON, nullable=True)  # Controls data access levels.
    # Defines access control rules for this entry.
    # - `None`: Public access, available to all users with the "user" role.
    # - `{}`: Private access, restricted exclusively to the owner.
    # - Custom permissions: Specific access control for reading and writing;
    #   Can specify group or user-level restrictions:
    #   {
    #      "read": {
    #          "group_ids": ["group_id1", "group_id2"],
    #          "user_ids":  ["user_id1", "user_id2"]
    #      },
    #      "write": {
    #          "group_ids": ["group_id1", "group_id2"],
    #          "user_ids":  ["user_id1", "user_id2"]
    #      }
    #   }


class PromptModel(BaseModel):
    command: str
    user_id: str
    title: str
    content: str
    timestamp: int  # timestamp in epoch

    access_control: Optional[dict] = None
    model_config = ConfigDict(from_attributes=True)


####################
# Forms
####################


class PromptUserResponse(PromptModel):
    user: Optional[UserResponse] = None


class PromptForm(BaseModel):
    command: str
    title: str
    content: str
    access_control: Optional[dict] = None


class PromptsTable:
    def insert_new_prompt(
        self, user_id: str, form_data: PromptForm
    ) -> Optional[PromptModel]:
        prompt = PromptModel(
            **{
                "user_id": user_id,
                **form_data.model_dump(),
                "timestamp": int(time.time()),
            }
        )

        try:
            with get_db() as db:
                result = Prompt(**prompt.model_dump())
                db.add(result)
                db.commit()
                db.refresh(result)
                if result:
                    validated_prompt = PromptModel.model_validate(result)

                    # Sync to BlueNexus (non-blocking)
                    try:
                        from open_webui.utils.bluenexus.sync_service import BlueNexusSync
                        from open_webui.env import log
                        log.info(f"[Prompt Sync] Syncing prompt {validated_prompt.command} to BlueNexus")
                        BlueNexusSync.sync_prompt_to_bluenexus_background(
                            validated_prompt.command,
                            user_id,
                            validated_prompt.model_dump(),
                            operation="create"
                        )
                    except Exception as e:
                        from open_webui.env import log
                        log.error(f"[Prompt Sync] Failed to sync prompt {validated_prompt.command}: {e}")
                        pass  # Don't fail if sync fails

                    return validated_prompt
                else:
                    return None
        except Exception:
            return None

    def get_prompt_by_command(self, command: str) -> Optional[PromptModel]:
        try:
            with get_db() as db:
                prompt = db.query(Prompt).filter_by(command=command).first()
                return PromptModel.model_validate(prompt)
        except Exception:
            return None

    def get_prompts(self) -> list[PromptUserResponse]:
        with get_db() as db:
            all_prompts = db.query(Prompt).order_by(Prompt.timestamp.desc()).all()

            user_ids = list(set(prompt.user_id for prompt in all_prompts))

            users = Users.get_users_by_user_ids(user_ids) if user_ids else []
            users_dict = {user.id: user for user in users}

            prompts = []
            for prompt in all_prompts:
                user = users_dict.get(prompt.user_id)
                prompts.append(
                    PromptUserResponse.model_validate(
                        {
                            **PromptModel.model_validate(prompt).model_dump(),
                            "user": user.model_dump() if user else None,
                        }
                    )
                )

            return prompts

    def get_prompts_by_user_id(
        self, user_id: str, permission: str = "write"
    ) -> list[PromptUserResponse]:
        prompts = self.get_prompts()
        user_group_ids = {group.id for group in Groups.get_groups_by_member_id(user_id)}

        return [
            prompt
            for prompt in prompts
            if prompt.user_id == user_id
            or has_access(user_id, permission, prompt.access_control, user_group_ids)
        ]

    def update_prompt_by_command(
        self, command: str, form_data: PromptForm
    ) -> Optional[PromptModel]:
        try:
            with get_db() as db:
                prompt = db.query(Prompt).filter_by(command=command).first()
                prompt.title = form_data.title
                prompt.content = form_data.content
                prompt.access_control = form_data.access_control
                prompt.timestamp = int(time.time())
                db.commit()

                validated_prompt = PromptModel.model_validate(prompt)

                # Sync to BlueNexus (non-blocking)
                try:
                    from open_webui.utils.bluenexus.sync_service import BlueNexusSync
                    from open_webui.env import log
                    log.info(f"[Prompt Sync] Syncing prompt update {command} to BlueNexus")
                    BlueNexusSync.sync_prompt_to_bluenexus_background(
                        command,
                        validated_prompt.user_id,
                        validated_prompt.model_dump(),
                        operation="update"
                    )
                except Exception as e:
                    from open_webui.env import log
                    log.error(f"[Prompt Sync] Failed to sync prompt update {command}: {e}")
                    pass  # Don't fail if sync fails

                return validated_prompt
        except Exception:
            return None

    def delete_prompt_by_command(self, command: str) -> bool:
        try:
            with get_db() as db:
                # Get prompt before deleting to extract user_id
                prompt = db.query(Prompt).filter_by(command=command).first()
                if not prompt:
                    return False

                user_id = prompt.user_id
                db.query(Prompt).filter_by(command=command).delete()
                db.commit()

                # Sync to BlueNexus (non-blocking)
                try:
                    from open_webui.utils.bluenexus.sync_service import BlueNexusSync
                    from open_webui.env import log
                    log.info(f"[Prompt Sync] Syncing prompt deletion {command} to BlueNexus")
                    BlueNexusSync.sync_prompt_to_bluenexus_background(
                        command,
                        user_id,
                        None,
                        operation="delete"
                    )
                except Exception as e:
                    from open_webui.env import log
                    log.error(f"[Prompt Sync] Failed to sync prompt deletion {command}: {e}")
                    pass  # Don't fail if sync fails

                return True
        except Exception:
            return False


Prompts = PromptsTable()
