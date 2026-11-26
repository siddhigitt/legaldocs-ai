import os
from contextlib import contextmanager
from datetime import datetime
from functools import lru_cache

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
    create_engine,
    delete,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class DatabaseUnavailable(Exception):
    """The database cannot be reached or is not configured."""


class EmailTaken(Exception):
    pass


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    __table_args__ = (CheckConstraint("email = lower(email)", name="users_email_lowercase"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(254), unique=True)
    full_name: Mapped[str] = mapped_column(String(120))
    password_hash: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Analysis(Base):
    __tablename__ = "analyses"
    __table_args__ = (
        CheckConstraint("kind IN ('single', 'comparison')", name="analyses_kind"),
        Index("ix_analyses_user_created", "user_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(16))
    title: Mapped[str] = mapped_column(String(600))
    doc_type: Mapped[str] = mapped_column(String(40))
    risk_score: Mapped[int | None]
    overall_risk: Mapped[str | None] = mapped_column(String(10))
    high_count: Mapped[int] = mapped_column(default=0)
    medium_count: Mapped[int] = mapped_column(default=0)
    low_count: Mapped[int] = mapped_column(default=0)
    details: Mapped[dict] = mapped_column(JSON().with_variant(JSONB(), "postgresql"), default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


@lru_cache(maxsize=1)
def _engine():
    url = os.getenv("DATABASE_URL")
    if not url:
        raise DatabaseUnavailable("DATABASE_URL is not set in the .env file.")
    # Without a connect timeout, psycopg on Windows waits indefinitely on a stopped server.
    return create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 3})


@contextmanager
def _session():
    try:
        with Session(_engine()) as session, session.begin():
            yield session
    except OperationalError as exc:
        raise DatabaseUnavailable("The database server is not reachable.") from exc


def init_db():
    try:
        Base.metadata.create_all(_engine())
    except OperationalError as exc:
        raise DatabaseUnavailable("The database server is not reachable.") from exc


def create_user(email, full_name, password_hash):
    try:
        with _session() as session:
            user = User(email=email, full_name=full_name, password_hash=password_hash)
            session.add(user)
            session.flush()
            return {"id": user.id, "email": user.email, "full_name": user.full_name}
    except IntegrityError as exc:
        raise EmailTaken from exc


def find_user(email):
    with _session() as session:
        user = session.scalar(select(User).where(User.email == email))
        if user is None:
            return None
        return {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "password_hash": user.password_hash,
        }


def record_analysis(user_id, kind, title, doc_type, risk_score, overall_risk, counts, details):
    with _session() as session:
        session.add(Analysis(
            user_id=user_id,
            kind=kind,
            title=title[:600],
            doc_type=doc_type,
            risk_score=risk_score,
            overall_risk=overall_risk,
            high_count=counts.get("high", 0),
            medium_count=counts.get("medium", 0),
            low_count=counts.get("low", 0),
            details=details,
        ))


def list_history(user_id, limit=20):
    with _session() as session:
        rows = session.scalars(
            select(Analysis)
            .where(Analysis.user_id == user_id)
            .order_by(Analysis.created_at.desc(), Analysis.id.desc())
            .limit(limit)
        )
        return [
            {
                "id": row.id,
                "kind": row.kind,
                "title": row.title,
                "doc_type": row.doc_type,
                "risk_score": row.risk_score,
                "overall_risk": row.overall_risk,
                "high": row.high_count,
                "medium": row.medium_count,
                "low": row.low_count,
                "details": row.details,
                "created_at": row.created_at,
            }
            for row in rows
        ]


def clear_history(user_id):
    with _session() as session:
        session.execute(delete(Analysis).where(Analysis.user_id == user_id))
