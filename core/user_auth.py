# -*- coding: utf-8 -*-
"""
SHIFA AI — Gestionnaire d'authentification local (SQLite + bcrypt)
Gère : inscription, connexion, mode invité
"""

import os
import sqlite3
import hashlib
import secrets
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("shifa.auth")

DB_PATH = Path(__file__).parent.parent / "data" / "shifa_users.db"

# ─────────────────────────────────────────────────────────────
# DB INIT
# ─────────────────────────────────────────────────────────────
def _get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Crée les tables si elles n'existent pas encore."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                username    TEXT    NOT NULL UNIQUE,
                email       TEXT    UNIQUE,
                password_hash TEXT  NOT NULL,
                full_name   TEXT    DEFAULT '',
                role        TEXT    DEFAULT 'patient',
                created_at  TEXT    DEFAULT CURRENT_TIMESTAMP,
                last_login  TEXT
            )
        """)
        conn.commit()
    logger.info("[Auth] Base de données initialisée.")

# ─────────────────────────────────────────────────────────────
# PASSWORD HELPERS
# ─────────────────────────────────────────────────────────────
def _hash_password(password: str) -> str:
    """Hash SHA-256 + sel aléatoire (sans bcrypt pour ne pas dépendre de la lib)."""
    salt = secrets.token_hex(16)
    h = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    return f"{salt}${h}"

def _verify_password(password: str, stored: str) -> bool:
    """Vérifie un mot de passe contre le hash stocké."""
    try:
        salt, h = stored.split("$", 1)
        expected = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
        return secrets.compare_digest(h, expected)
    except Exception:
        return False

# ─────────────────────────────────────────────────────────────
# AUTH OPERATIONS
# ─────────────────────────────────────────────────────────────
def register_user(username: str, password: str, email: str = "", full_name: str = "") -> dict:
    """
    Inscrit un nouvel utilisateur.
    Retourne {"success": bool, "message": str}
    """
    username = username.strip()
    email = email.strip()

    if len(username) < 3:
        return {"success": False, "message": "Le nom d'utilisateur doit contenir au moins 3 caractères."}
    if len(password) < 6:
        return {"success": False, "message": "Le mot de passe doit contenir au moins 6 caractères."}

    try:
        init_db()
        with _get_conn() as conn:
            existing = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
            if existing:
                return {"success": False, "message": "Ce nom d'utilisateur est déjà pris."}
            if email:
                existing_email = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
                if existing_email:
                    return {"success": False, "message": "Cet email est déjà utilisé."}

            pw_hash = _hash_password(password)
            conn.execute(
                "INSERT INTO users (username, email, password_hash, full_name) VALUES (?, ?, ?, ?)",
                (username, email or None, pw_hash, full_name)
            )
            conn.commit()
        logger.info(f"[Auth] Nouvel utilisateur inscrit: {username}")
        return {"success": True, "message": "Compte créé avec succès ! Vous pouvez maintenant vous connecter."}
    except Exception as e:
        logger.error(f"[Auth] Erreur inscription: {e}")
        return {"success": False, "message": f"Erreur lors de l'inscription: {str(e)}"}


def login_user(username: str, password: str) -> dict:
    """
    Authentifie un utilisateur.
    Retourne {"success": bool, "user": dict | None, "message": str}
    """
    username = username.strip()
    try:
        init_db()
        with _get_conn() as conn:
            row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
            if not row:
                return {"success": False, "message": "Nom d'utilisateur ou mot de passe incorrect."}
            if not _verify_password(password, row["password_hash"]):
                return {"success": False, "message": "Nom d'utilisateur ou mot de passe incorrect."}
            # Mise à jour dernière connexion
            conn.execute("UPDATE users SET last_login = ? WHERE id = ?", (datetime.now().isoformat(), row["id"]))
            conn.commit()

        user_data = {
            "id": row["id"],
            "username": row["username"],
            "email": row["email"] or "",
            "full_name": row["full_name"] or username,
            "role": row["role"],
        }
        logger.info(f"[Auth] Connexion réussie: {username}")
        return {"success": True, "user": user_data}
    except Exception as e:
        logger.error(f"[Auth] Erreur connexion: {e}")
        return {"success": False, "message": f"Erreur de connexion: {str(e)}"}


def guest_session() -> dict:
    """Retourne un profil invité (sans accès base de données)."""
    return {
        "id": None,
        "username": "زائر / Invité",
        "email": "",
        "full_name": "Invité",
        "role": "guest",
    }
