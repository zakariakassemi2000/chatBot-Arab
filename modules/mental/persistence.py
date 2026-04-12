# -*- coding: utf-8 -*-
"""
SHIFA-Mental · Persistence Layer (SQLite)
Sauvegarde persistante du mood, journal, scores PHQ-9/GAD-7, et alertes de crise.
Rien ne se perd au refresh de page.
"""

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

logger = logging.getLogger("shifa.mental.persistence")

DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "mental_health.db"


class MentalPersistence:
    """SQLite-backed persistence for mental health data."""

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = str(db_path or DEFAULT_DB_PATH)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        try:
            conn = self._get_conn()
            c = conn.cursor()

            c.execute("""CREATE TABLE IF NOT EXISTS mood_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                mood_label TEXT NOT NULL,
                score INTEGER NOT NULL,
                note TEXT DEFAULT ''
            )""")

            c.execute("""CREATE TABLE IF NOT EXISTS thought_journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                negative_thought TEXT NOT NULL,
                positive_thought TEXT DEFAULT ''
            )""")

            c.execute("""CREATE TABLE IF NOT EXISTS assessment_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                test_type TEXT NOT NULL,
                score INTEGER NOT NULL,
                severity TEXT NOT NULL,
                answers TEXT NOT NULL
            )""")

            c.execute("""CREATE TABLE IF NOT EXISTS crisis_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                level INTEGER NOT NULL,
                matched_keywords TEXT NOT NULL,
                user_text_hash TEXT DEFAULT ''
            )""")

            c.execute("""CREATE TABLE IF NOT EXISTS journal_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                content TEXT NOT NULL,
                sentiment_score REAL DEFAULT NULL,
                sentiment_label TEXT DEFAULT ''
            )""")

            conn.commit()
            conn.close()
            logger.info("[MentalPersistence] Database initialized at %s", self._db_path)
        except Exception as e:
            logger.error("[MentalPersistence] DB init error: %s", e)

    # ── Mood ─────────────────────────────────────────────────────
    def save_mood(self, mood_label: str, score: int, note: str = "") -> None:
        try:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO mood_entries (created_at, mood_label, score, note) VALUES (?, ?, ?, ?)",
                (datetime.now().strftime("%Y-%m-%d %H:%M"), mood_label, score, note)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error("[MentalPersistence] save_mood error: %s", e)

    def get_moods(self, limit: int = 30) -> list[dict]:
        try:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT created_at, mood_label, score, note FROM mood_entries ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
            conn.close()
            return [{"date": r[0], "mood_label": r[1], "score": r[2], "note": r[3]} for r in rows]
        except Exception as e:
            logger.error("[MentalPersistence] get_moods error: %s", e)
            return []

    def clear_moods(self) -> None:
        try:
            conn = self._get_conn()
            conn.execute("DELETE FROM mood_entries")
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error("[MentalPersistence] clear_moods error: %s", e)

    # ── Thought Journal ──────────────────────────────────────────
    def save_thought(self, negative: str, positive: str = "") -> None:
        try:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO thought_journal (created_at, negative_thought, positive_thought) VALUES (?, ?, ?)",
                (datetime.now().strftime("%Y-%m-%d %H:%M"), negative, positive)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error("[MentalPersistence] save_thought error: %s", e)

    def get_thoughts(self, limit: int = 20) -> list[dict]:
        try:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT created_at, negative_thought, positive_thought FROM thought_journal ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
            conn.close()
            return [{"date": r[0], "negative": r[1], "positive": r[2]} for r in rows]
        except Exception as e:
            logger.error("[MentalPersistence] get_thoughts error: %s", e)
            return []

    # ── Assessment Results ────────────────────────────────────────
    def save_assessment(self, test_type: str, score: int, severity: str, answers: list[int]) -> None:
        try:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO assessment_results (created_at, test_type, score, severity, answers) VALUES (?, ?, ?, ?, ?)",
                (datetime.now().strftime("%Y-%m-%d %H:%M"), test_type, score, severity, json.dumps(answers))
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error("[MentalPersistence] save_assessment error: %s", e)

    def get_assessments(self, test_type: str = "", limit: int = 10) -> list[dict]:
        try:
            conn = self._get_conn()
            if test_type:
                rows = conn.execute(
                    "SELECT created_at, test_type, score, severity, answers FROM assessment_results WHERE test_type=? ORDER BY id DESC LIMIT ?",
                    (test_type, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT created_at, test_type, score, severity, answers FROM assessment_results ORDER BY id DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            conn.close()
            return [{"date": r[0], "test_type": r[1], "score": r[2], "severity": r[3], "answers": json.loads(r[4])} for r in rows]
        except Exception as e:
            logger.error("[MentalPersistence] get_assessments error: %s", e)
            return []

    # ── Crisis Logs ───────────────────────────────────────────────
    def log_crisis(self, level: int, matched_keywords: list[str], text_hash: str = "") -> None:
        try:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO crisis_logs (created_at, level, matched_keywords, user_text_hash) VALUES (?, ?, ?, ?)",
                (datetime.now().strftime("%Y-%m-%d %H:%M"), level, json.dumps(matched_keywords, ensure_ascii=False), text_hash)
            )
            conn.commit()
            conn.close()
            logger.warning("[MentalPersistence] Crisis logged — level=%d", level)
        except Exception as e:
            logger.error("[MentalPersistence] log_crisis error: %s", e)

    def has_recent_crisis(self, hours: int = 24) -> bool:
        """Check if there's been a crisis in the last N hours."""
        try:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT COUNT(*) FROM crisis_logs WHERE level >= 3 AND created_at >= datetime('now', ?)",
                (f"-{hours} hours",)
            ).fetchone()
            conn.close()
            return row[0] > 0 if row else False
        except Exception as e:
            logger.error("[MentalPersistence] has_recent_crisis error: %s", e)
            return False

    # ── Journal Intime ────────────────────────────────────────────
    def save_journal(self, content: str, sentiment_score: float = None, sentiment_label: str = "") -> None:
        try:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO journal_entries (created_at, content, sentiment_score, sentiment_label) VALUES (?, ?, ?, ?)",
                (datetime.now().strftime("%Y-%m-%d %H:%M"), content, sentiment_score, sentiment_label)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error("[MentalPersistence] save_journal error: %s", e)

    def get_journals(self, limit: int = 15) -> list[dict]:
        try:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT created_at, content, sentiment_score, sentiment_label FROM journal_entries ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
            conn.close()
            return [{"date": r[0], "content": r[1], "sentiment_score": r[2], "sentiment_label": r[3]} for r in rows]
        except Exception as e:
            logger.error("[MentalPersistence] get_journals error: %s", e)
            return []
