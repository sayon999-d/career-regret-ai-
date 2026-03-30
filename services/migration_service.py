import os
import sqlite3
from typing import List, Dict
from datetime import datetime

try:
    import psycopg2
except ImportError:
    import sys
    vendor_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'vendor')
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)
    try:
        import psycopg2
    except ImportError:
        psycopg2 = None

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/stepwise_ai")

def _resolve_sqlite_path():
    primary = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'stepwise_ai.db')
    try:
        conn = sqlite3.connect(primary)
        conn.execute("SELECT 1")
        conn.close()
        return primary
    except Exception:
        fallback = os.path.join('/tmp', 'stepwise_ai.db')
        return fallback

SQLITE_PATH = _resolve_sqlite_path()


class MigrationService:

    def __init__(self, database_url: str = DATABASE_URL, sqlite_path: str = SQLITE_PATH):
        self.database_url = database_url
        self.sqlite_path = sqlite_path
        self.backend = None  
        self._pg_available = False

        if psycopg2 is not None:
            try:
                conn = psycopg2.connect(self.database_url)
                conn.close()
                self._pg_available = True
                self.backend = 'postgres'
            except Exception:
                self._pg_available = False

        if not self._pg_available:
            self.backend = 'sqlite'

        self._ensure_migration_table()

    def _get_conn(self):
        if self.backend == 'postgres':
            return psycopg2.connect(self.database_url)
        else:
            conn = sqlite3.connect(self.sqlite_path)
            conn.execute("PRAGMA journal_mode=WAL")
            return conn

    def _q(self, sql: str) -> str:
        if self.backend == 'postgres':
            return sql
        sql = sql.replace('%s', '?')
        sql = sql.replace('SERIAL PRIMARY KEY', 'INTEGER PRIMARY KEY AUTOINCREMENT')
        sql = sql.replace('TIMESTAMP DEFAULT CURRENT_TIMESTAMP', 'TEXT DEFAULT CURRENT_TIMESTAMP')
        sql = sql.replace('TIMESTAMP', 'TEXT')
        return sql

    def _ensure_migration_table(self):
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(self._q("""
                CREATE TABLE IF NOT EXISTS _migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.commit()
        finally:
            conn.close()

    def get_applied_migrations(self) -> List[int]:
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT version FROM _migrations ORDER BY version")
            return [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()

    def apply_migrations(self, migrations: Dict[int, str]):
        applied = self.get_applied_migrations()
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            for version in sorted(migrations.keys()):
                if version not in applied:
                    print(f"Applying migration version {version}...")
                    try:
                        cursor.execute(self._q(migrations[version]))
                        cursor.execute(
                            self._q("INSERT INTO _migrations (version, name) VALUES (%s, %s)"),
                            (version, f"migration_{version}")
                        )
                        conn.commit()
                    except Exception as e:
                        print(f"Error applying migration {version}: {e}")
                        conn.rollback()
                        raise
        finally:
            conn.close()


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
        decided_at TIMESTAMP,
        sync_status TEXT DEFAULT 'synced'
    );
    """
}

try:
    migration_service = MigrationService()
except Exception as e:
    import warnings
    import traceback
    error_msg = f"MigrationService failed to initialise: {e}\n{traceback.format_exc()}"
    warnings.warn(error_msg)
    migration_service = None

if __name__ == "__main__":
    if migration_service:
        migration_service.apply_migrations(PHASE_3_MIGRATIONS)
