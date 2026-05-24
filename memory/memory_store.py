"""
memory/memory_store.py — SQLite-backed memory for the clinical assistant.
Stores conversation history (short-term) and clinical notes (long-term).
No external services needed.
"""
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from config.settings import MEMORY_DB_PATH, MAX_HISTORY_TURNS

logger = logging.getLogger(__name__)

engine = create_engine(f"sqlite:///{MEMORY_DB_PATH}", echo=False)
SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


class ConversationTurn(Base):
    """Short-term: individual conversation turns per session."""
    __tablename__ = "conversation_turns"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), index=True, nullable=False)
    role = Column(String(20), nullable=False)        # "user" or "assistant"
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


class ClinicalNote(Base):
    """Long-term: persisted clinical facts, diagnoses, and plans."""
    __tablename__ = "clinical_notes"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), index=True, nullable=False)
    note_type = Column(String(50), nullable=False)   # "diagnosis", "drug", "plan"
    content = Column(Text, nullable=False)
    metadata_json = Column(Text, default="{}")
    timestamp = Column(DateTime, default=datetime.utcnow)


class AgentTrace(Base):
    """Audit: trace of which agents fired and what tools were called."""
    __tablename__ = "agent_traces"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), index=True, nullable=False)
    agent_name = Column(String(50), nullable=False)
    tool_calls = Column(Text, default="[]")          # JSON list
    result_summary = Column(Text, default="")
    timestamp = Column(DateTime, default=datetime.utcnow)


def init_db() -> None:
    """Create all tables. Safe to call multiple times."""
    Base.metadata.create_all(engine)
    logger.info(f"Memory DB initialised at {MEMORY_DB_PATH}")


# ─── Conversation history ────────────────────────────────────────────────────

def add_turn(session_id: str, role: str, content: str) -> None:
    with SessionLocal() as db:
        db.add(ConversationTurn(session_id=session_id, role=role, content=content))
        db.commit()


def get_history(session_id: str, last_n: int = MAX_HISTORY_TURNS) -> List[Dict[str, str]]:
    """
    Return last N turns as a list of {role, content} dicts.
    Suitable for direct injection into LangChain message history.
    """
    with SessionLocal() as db:
        rows = (
            db.query(ConversationTurn)
            .filter(ConversationTurn.session_id == session_id)
            .order_by(ConversationTurn.id.desc())
            .limit(last_n)
            .all()
        )
    return [{"role": r.role, "content": r.content} for r in reversed(rows)]


def clear_history(session_id: str) -> None:
    with SessionLocal() as db:
        db.query(ConversationTurn).filter(
            ConversationTurn.session_id == session_id
        ).delete()
        db.commit()


# ─── Clinical notes (long-term) ──────────────────────────────────────────────

def save_clinical_note(
    session_id: str,
    note_type: str,
    content: str,
    metadata: Optional[Dict] = None,
) -> None:
    with SessionLocal() as db:
        db.add(ClinicalNote(
            session_id=session_id,
            note_type=note_type,
            content=content,
            metadata_json=json.dumps(metadata or {}),
        ))
        db.commit()


def get_clinical_notes(session_id: str, note_type: Optional[str] = None) -> List[Dict[str, Any]]:
    with SessionLocal() as db:
        q = db.query(ClinicalNote).filter(ClinicalNote.session_id == session_id)
        if note_type:
            q = q.filter(ClinicalNote.note_type == note_type)
        rows = q.order_by(ClinicalNote.timestamp.desc()).all()
    return [
        {
            "type": r.note_type,
            "content": r.content,
            "metadata": json.loads(r.metadata_json),
            "timestamp": r.timestamp.isoformat(),
        }
        for r in rows
    ]


# ─── Agent tracing ───────────────────────────────────────────────────────────

def log_agent_trace(
    session_id: str,
    agent_name: str,
    tool_calls: List[str],
    result_summary: str,
) -> None:
    with SessionLocal() as db:
        db.add(AgentTrace(
            session_id=session_id,
            agent_name=agent_name,
            tool_calls=json.dumps(tool_calls),
            result_summary=result_summary[:500],
        ))
        db.commit()


def get_traces(session_id: str) -> List[Dict[str, Any]]:
    with SessionLocal() as db:
        rows = (
            db.query(AgentTrace)
            .filter(AgentTrace.session_id == session_id)
            .order_by(AgentTrace.timestamp)
            .all()
        )
    return [
        {
            "agent": r.agent_name,
            "tools": json.loads(r.tool_calls),
            "summary": r.result_summary,
            "timestamp": r.timestamp.isoformat(),
        }
        for r in rows
    ]


# Initialise on import
init_db()