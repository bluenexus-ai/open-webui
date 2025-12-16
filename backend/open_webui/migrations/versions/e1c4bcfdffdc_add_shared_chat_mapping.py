"""add_shared_chat_mapping

Revision ID: e1c4bcfdffdc
Revises: a5c220713937
Create Date: 2024-12-17

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from open_webui.migrations.util import get_existing_tables

# revision identifiers, used by Alembic.
revision: str = "e1c4bcfdffdc"
down_revision: Union[str, None] = "a5c220713937"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Create shared_chat_mapping table.

    This table maps share_id to owner_user_id for BlueNexus shared content access.
    It allows looking up who owns a shared chat so we can use their
    BlueNexus session to fetch the chat data.
    """
    existing_tables = set(get_existing_tables())

    if "shared_chat_mapping" not in existing_tables:
        op.create_table(
            "shared_chat_mapping",
            sa.Column("share_id", sa.String(), nullable=False),
            sa.Column("owner_user_id", sa.String(), nullable=False),
            sa.Column("chat_id", sa.String(), nullable=False),
            sa.Column("created_at", sa.BigInteger(), nullable=False),
            sa.PrimaryKeyConstraint("share_id"),
        )

        # Create index on owner_user_id for faster lookups
        op.create_index(
            "ix_shared_chat_mapping_owner_user_id",
            "shared_chat_mapping",
            ["owner_user_id"],
        )

        # Create index on chat_id for reverse lookups
        op.create_index(
            "ix_shared_chat_mapping_chat_id",
            "shared_chat_mapping",
            ["chat_id"],
        )


def downgrade() -> None:
    """Drop shared_chat_mapping table."""
    op.drop_index("ix_shared_chat_mapping_chat_id", table_name="shared_chat_mapping")
    op.drop_index("ix_shared_chat_mapping_owner_user_id", table_name="shared_chat_mapping")
    op.drop_table("shared_chat_mapping")
