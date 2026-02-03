import sqlite3
import os
from typing import List, Dict
from datetime import datetime

class MigrationService:
    def __init__(self, db_path: str = "learning_data.db"):
        self.db_path = db_path
        self._ensure_migration_table()

    def _ensure_migration_table(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS _migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def get_applied_migrations(self) -> List[int]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT version FROM _migrations ORDER BY version")
            return [row[0] for row in cursor.fetchall()]

    def apply_migrations(self, migrations: Dict[int, str]):
        """
        migrations: { 1: "CREATE TABLE...", 2: "ALTER TABLE..." }
        """
        applied = self.get_applied_migrations()
        with sqlite3.connect(self.db_path) as conn:
            for version in sorted(migrations.keys()):
                if version not in applied:
                    print(f"Applying migration version {version}...")
                    try:
                        conn.executescript(migrations[version])
                        conn.execute("INSERT INTO _migrations (version, name) VALUES (?, ?)", (version, f"migration_{version}"))
                    except Exception as e:
                        print(f"Error applying migration {version}: {e}")
                        conn.rollback()
                        raise
            conn.commit()

PHASE_3_MIGRATIONS = {
    1: """
    CREATE TABLE IF NOT EXISTS mentor_profiles (
        id TEXT PRIMARY KEY,
        name TEXT,
        expertise TEXT,
        industry TEXT,
        years_experience INTEGER
    );
    """,
    2: """
    CREATE TABLE IF NOT EXISTS shared_decisions (
        short_code TEXT PRIMARY KEY,
        decision_data TEXT,
        expires_at TIMESTAMP
    );
    """,
    3: """
    CREATE TABLE IF NOT EXISTS journal_entries (
        id TEXT PRIMARY KEY,
        user_id TEXT,
        decision_type TEXT,
        title TEXT,
        description TEXT,
        status TEXT,
        predicted_regret REAL,
        predicted_confidence REAL,
        emotions TEXT,
        factors TEXT,
        nlp_analysis TEXT,
        chosen_option TEXT,
        alternatives TEXT,
        notes TEXT,
        created_at TIMESTAMP,
        updated_at TIMESTAMP,
        decided_at TIMESTAMP
    );
    -- Check if column exists is not easy in SQLite executescript,
    -- but we can use a separate check or just try/except in apply_migrations.
    -- For now, let's just make the table exist.
    ALTER TABLE journal_entries ADD COLUMN sync_status TEXT DEFAULT 'synced';
    """
}

migration_service = MigrationService()

if __name__ == "__main__":
    migration_service.apply_migrations(PHASE_3_MIGRATIONS)
