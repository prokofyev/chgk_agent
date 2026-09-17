"""Базовый класс ORM и таблицы схемы."""

from datetime import datetime
from functools import lru_cache

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from chgk_agent.config import get_settings


@lru_cache(maxsize=1)
def get_embedding_dim() -> int:
    """Вернуть размерность вектора эмбеддинга из конфигурации."""

    return get_settings().gigachat.embedding_dim


class Base(DeclarativeBase):
    """Базовый класс декларативных моделей."""


class Source(Base):
    """Зарегистрированный источник вопросов."""

    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    location: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    occurrences: Mapped[list["QuestionOccurrence"]] = relationship(
        back_populates="source", cascade="all, delete-orphan"
    )

    __table_args__ = (UniqueConstraint("kind", "location", name="uq_sources_kind_location"),)


class Question(Base):
    """Канонический вопрос, независимый от источника."""

    __tablename__ = "questions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    normalized_text: Mapped[str] = mapped_column(Text, nullable=False)
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    answer_text: Mapped[str] = mapped_column(Text, nullable=False)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    text_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(get_embedding_dim()), nullable=True
    )
    embedding_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    embedding_dim: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    occurrences: Mapped[list["QuestionOccurrence"]] = relationship(
        back_populates="question", cascade="all, delete-orphan"
    )


class QuestionOccurrence(Base):
    """Вхождение канонического вопроса в конкретный источник."""

    __tablename__ = "question_occurrences"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    question_id: Mapped[int] = mapped_column(
        ForeignKey("questions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    source_id: Mapped[int] = mapped_column(
        ForeignKey("sources.id", ondelete="CASCADE"), nullable=False, index=True
    )
    source_key: Mapped[str] = mapped_column(Text, nullable=False)
    raw_question_text: Mapped[str] = mapped_column(Text, nullable=False)
    raw_answer_text: Mapped[str] = mapped_column(Text, nullable=False)
    external_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    question: Mapped[Question] = relationship(back_populates="occurrences")
    source: Mapped[Source] = relationship(back_populates="occurrences")

    __table_args__ = (
        UniqueConstraint("source_id", "source_key", name="uq_question_occurrences_source_key"),
    )
