"""Начальная схема: pgvector, источники, вопросы и вхождения.

Revision ID: 0001
Revises:
Create Date: 2026-09-17
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

from chgk_agent.config import get_settings

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

HNSW_MAX_DIM = 2000


def _embedding_dim() -> int:
    return get_settings().gigachat.embedding_dim


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    dim = _embedding_dim()

    op.create_table(
        "sources",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("location", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("kind", "location", name="uq_sources_kind_location"),
    )

    op.create_table(
        "questions",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("normalized_text", sa.Text(), nullable=False),
        sa.Column("question_text", sa.Text(), nullable=False),
        sa.Column("answer_text", sa.Text(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("text_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("embedding", Vector(dim), nullable=True),
        sa.Column("embedding_model", sa.String(128), nullable=True),
        sa.Column("embedding_dim", sa.Integer(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )

    op.create_table(
        "question_occurrences",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column(
            "question_id",
            sa.BigInteger(),
            sa.ForeignKey("questions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "source_id",
            sa.Integer(),
            sa.ForeignKey("sources.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("source_key", sa.Text(), nullable=False),
        sa.Column("raw_question_text", sa.Text(), nullable=False),
        sa.Column("raw_answer_text", sa.Text(), nullable=False),
        sa.Column("external_url", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint(
            "source_id", "source_key", name="uq_question_occurrences_source_key"
        ),
    )
    op.create_index(
        "ix_question_occurrences_question_id", "question_occurrences", ["question_id"]
    )
    op.create_index("ix_question_occurrences_source_id", "question_occurrences", ["source_id"])

    op.create_index(
        "ix_questions_normalized_text_fts",
        "questions",
        [sa.text("to_tsvector('russian', normalized_text)")],
        postgresql_using="gin",
    )

    if dim <= HNSW_MAX_DIM:
        op.execute(
            "CREATE INDEX ix_questions_embedding_hnsw ON questions "
            "USING hnsw (embedding vector_cosine_ops)"
        )
    else:
        op.execute(
            "CREATE INDEX ix_questions_embedding_ivfflat ON questions "
            "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
        )


def downgrade() -> None:
    op.drop_index("ix_questions_embedding_hnsw", table_name="questions", if_exists=True)
    op.drop_index("ix_questions_embedding_ivfflat", table_name="questions", if_exists=True)
    op.drop_index("ix_questions_normalized_text_fts", table_name="questions")
    op.drop_index("ix_question_occurrences_source_id", table_name="question_occurrences")
    op.drop_index("ix_question_occurrences_question_id", table_name="question_occurrences")
    op.drop_table("question_occurrences")
    op.drop_table("questions")
    op.drop_table("sources")
