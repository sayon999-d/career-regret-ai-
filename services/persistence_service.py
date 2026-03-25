from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import os
import sqlite3
from contextlib import contextmanager

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

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/career_regret_ai")

def _resolve_sqlite_path():
    primary = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'career_ai.db')
    try:
        conn = sqlite3.connect(primary)
        conn.execute("SELECT 1")
        conn.close()
        return primary
    except Exception:
        fallback = os.path.join('/tmp', 'career_ai.db')
        print(f"[PersistenceService] Primary SQLite path unavailable, using fallback: {fallback}")
        return fallback

SQLITE_PATH = _resolve_sqlite_path()


class PersistenceService:

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
            print(f"[PersistenceService] PostgreSQL unavailable — using SQLite at {self.sqlite_path}")

        self._initialize_database()

    @contextmanager
    def _get_connection(self):
        if self.backend == 'postgres':
            conn = psycopg2.connect(self.database_url)
            conn.autocommit = False
            try:
                yield conn
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        else:
            conn = sqlite3.connect(self.sqlite_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            try:
                yield conn
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _ph(self, count: int = 1) -> str:
        p = "%s" if self.backend == 'postgres' else "?"
        return ", ".join([p] * count)

    def _q(self, sql: str) -> str:
        if self.backend == 'postgres':
            return sql
        sql = sql.replace('%s', '?')
        sql = sql.replace('SERIAL PRIMARY KEY', 'INTEGER PRIMARY KEY AUTOINCREMENT')
        sql = sql.replace('BOOLEAN DEFAULT FALSE', 'INTEGER DEFAULT 0')
        sql = sql.replace('BOOLEAN DEFAULT TRUE', 'INTEGER DEFAULT 1')
        sql = sql.replace('BOOLEAN', 'INTEGER')
        sql = sql.replace('= TRUE', '= 1')
        sql = sql.replace('= FALSE', '= 0')
        return sql

    def _initialize_database(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(self._q('''
                CREATE TABLE IF NOT EXISTS learning_profiles (
                    user_id TEXT PRIMARY KEY,
                    total_outcomes INTEGER DEFAULT 0,
                    prediction_accuracy REAL DEFAULT 0,
                    avg_prediction_error REAL DEFAULT 0,
                    optimism_bias REAL DEFAULT 0,
                    decision_style TEXT DEFAULT 'balanced',
                    risk_profile TEXT DEFAULT 'moderate',
                    patterns TEXT DEFAULT '[]',
                    created_at TEXT,
                    last_updated TEXT
                )
            '''))

            cursor.execute(self._q('''
                CREATE TABLE IF NOT EXISTS decision_outcomes (
                    id SERIAL PRIMARY KEY,
                    decision_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    predicted_regret REAL,
                    actual_regret REAL,
                    satisfaction_score REAL,
                    outcome_notes TEXT,
                    surprises TEXT,
                    lessons_learned TEXT,
                    time_since_decision_days INTEGER,
                    recorded_at TEXT
                )
            '''))

            cursor.execute(self._q('''
                CREATE TABLE IF NOT EXISTS coaching_profiles (
                    user_id TEXT PRIMARY KEY,
                    decision_style TEXT DEFAULT 'balanced',
                    risk_profile TEXT DEFAULT 'moderate',
                    primary_biases TEXT DEFAULT '[]',
                    strengths TEXT DEFAULT '[]',
                    growth_areas TEXT DEFAULT '[]',
                    total_sessions INTEGER DEFAULT 0,
                    progress_score REAL DEFAULT 0.5,
                    created_at TEXT,
                    last_session TEXT
                )
            '''))

            cursor.execute(self._q('''
                CREATE TABLE IF NOT EXISTS future_self_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    timeframe TEXT,
                    decision_path TEXT,
                    emotional_state TEXT,
                    conversation_history TEXT,
                    insights TEXT,
                    emotional_impact_score REAL,
                    started_at TEXT,
                    ended_at TEXT
                )
            '''))

            cursor.execute(self._q('''
                CREATE TABLE IF NOT EXISTS scout_profiles (
                    user_id TEXT PRIMARY KEY,
                    "current_role" TEXT,
                    industry TEXT,
                    skills TEXT,
                    interests TEXT,
                    risk_tolerance REAL,
                    salary_target REAL,
                    location_preferences TEXT,
                    low_regret_factors TEXT,
                    career_goals TEXT,
                    created_at TEXT,
                    last_scan TEXT
                )
            '''))

            cursor.execute(self._q('''
                CREATE TABLE IF NOT EXISTS scout_opportunities (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    type TEXT,
                    title TEXT,
                    description TEXT,
                    source TEXT,
                    match_score REAL,
                    match_reasons TEXT,
                    regret_score REAL,
                    action_items TEXT,
                    discovered_at TEXT,
                    viewed BOOLEAN DEFAULT FALSE,
                    saved BOOLEAN DEFAULT FALSE,
                    dismissed BOOLEAN DEFAULT FALSE
                )
            '''))

            cursor.execute(self._q('''
                CREATE TABLE IF NOT EXISTS bias_history (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT,
                    bias_type TEXT,
                    confidence REAL,
                    trigger_phrase TEXT,
                    detected_at TEXT,
                    intervention_shown BOOLEAN DEFAULT FALSE,
                    intervention_accepted BOOLEAN DEFAULT FALSE
                )
            '''))

            cursor.execute(self._q('''
                CREATE TABLE IF NOT EXISTS global_outcomes (
                    outcome_hash TEXT PRIMARY KEY,
                    decision_type TEXT,
                    industry TEXT,
                    experience_bucket TEXT,
                    predicted_regret REAL,
                    actual_regret REAL,
                    satisfaction_score REAL,
                    time_to_outcome_days INTEGER,
                    key_factors TEXT,
                    created_at TEXT
                )
            '''))

            cursor.execute('CREATE INDEX IF NOT EXISTS idx_p_outcomes_user ON decision_outcomes(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_p_outcomes_decision ON decision_outcomes(decision_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_p_scout_user ON scout_opportunities(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_p_bias_user ON bias_history(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_p_global_type ON global_outcomes(decision_type)')

            conn.commit()

    def _row_to_dict(self, cursor, row):
        if row is None:
            return None
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, row))

    def save_learning_profile(self, user_id: str, profile_data: Dict[str, Any]) -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.utcnow().isoformat()

            cursor.execute(self._q('''
                INSERT INTO learning_profiles
                (user_id, total_outcomes, prediction_accuracy, avg_prediction_error,
                 optimism_bias, decision_style, risk_profile, patterns, created_at, last_updated)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE SET
                    total_outcomes = EXCLUDED.total_outcomes,
                    prediction_accuracy = EXCLUDED.prediction_accuracy,
                    avg_prediction_error = EXCLUDED.avg_prediction_error,
                    optimism_bias = EXCLUDED.optimism_bias,
                    decision_style = EXCLUDED.decision_style,
                    risk_profile = EXCLUDED.risk_profile,
                    patterns = EXCLUDED.patterns,
                    last_updated = EXCLUDED.last_updated
            '''), (
                user_id,
                profile_data.get('total_outcomes', 0),
                profile_data.get('prediction_accuracy', 0),
                profile_data.get('avg_prediction_error', 0),
                profile_data.get('optimism_bias', 0),
                profile_data.get('decision_style', 'balanced'),
                profile_data.get('risk_profile', 'moderate'),
                json.dumps(profile_data.get('patterns', [])),
                now, now
            ))
            conn.commit()
            return True

    def get_learning_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(self._q('SELECT * FROM learning_profiles WHERE user_id = %s'), (user_id,))
            row = cursor.fetchone()

            if row:
                d = self._row_to_dict(cursor, row)
                d['patterns'] = json.loads(d['patterns']) if d.get('patterns') else []
                return d
            return None

    def save_outcome(self, user_id: str, outcome_data: Dict[str, Any]) -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(self._q('''
                INSERT INTO decision_outcomes
                (decision_id, user_id, predicted_regret, actual_regret, satisfaction_score,
                 outcome_notes, surprises, lessons_learned, time_since_decision_days, recorded_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            '''), (
                outcome_data.get('decision_id'),
                user_id,
                outcome_data.get('predicted_regret'),
                outcome_data.get('actual_regret'),
                outcome_data.get('satisfaction_score'),
                outcome_data.get('outcome_notes', ''),
                json.dumps(outcome_data.get('surprises', [])),
                outcome_data.get('lessons_learned', ''),
                outcome_data.get('time_since_decision_days', 0),
                datetime.utcnow().isoformat()
            ))
            conn.commit()
            return True

    def get_user_outcomes(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(self._q('''
                SELECT * FROM decision_outcomes WHERE user_id = %s
                ORDER BY recorded_at DESC LIMIT %s
            '''), (user_id, limit))

            outcomes = []
            for row in cursor.fetchall():
                d = self._row_to_dict(cursor, row)
                d['surprises'] = json.loads(d['surprises']) if d.get('surprises') else []
                outcomes.append(d)
            return outcomes

    def save_scout_profile(self, user_id: str, profile_data: Dict[str, Any]) -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.utcnow().isoformat()

            cursor.execute(self._q('''
                INSERT INTO scout_profiles
                (user_id, "current_role", industry, skills, interests, risk_tolerance,
                 salary_target, location_preferences, low_regret_factors, career_goals,
                 created_at, last_scan)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE SET
                    "current_role" = EXCLUDED."current_role",
                    industry = EXCLUDED.industry,
                    skills = EXCLUDED.skills,
                    interests = EXCLUDED.interests,
                    risk_tolerance = EXCLUDED.risk_tolerance,
                    salary_target = EXCLUDED.salary_target,
                    location_preferences = EXCLUDED.location_preferences,
                    low_regret_factors = EXCLUDED.low_regret_factors,
                    career_goals = EXCLUDED.career_goals,
                    last_scan = EXCLUDED.last_scan
            '''), (
                user_id,
                profile_data.get('current_role', ''),
                profile_data.get('industry', ''),
                json.dumps(profile_data.get('skills', [])),
                json.dumps(profile_data.get('interests', [])),
                profile_data.get('risk_tolerance', 0.5),
                profile_data.get('salary_target', 0),
                json.dumps(profile_data.get('location_preferences', [])),
                json.dumps(profile_data.get('low_regret_factors', [])),
                json.dumps(profile_data.get('career_goals', [])),
                now, now
            ))
            conn.commit()
            return True

    def save_opportunity(self, user_id: str, opportunity: Dict[str, Any]) -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(self._q('''
                INSERT INTO scout_opportunities
                (id, user_id, type, title, description, source, match_score,
                 match_reasons, regret_score, action_items, discovered_at, viewed, saved, dismissed)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    match_score = EXCLUDED.match_score,
                    match_reasons = EXCLUDED.match_reasons,
                    regret_score = EXCLUDED.regret_score,
                    action_items = EXCLUDED.action_items
            '''), (
                opportunity.get('id'),
                user_id,
                opportunity.get('type'),
                opportunity.get('title'),
                opportunity.get('description'),
                opportunity.get('source'),
                opportunity.get('match_score'),
                json.dumps(opportunity.get('match_reasons', [])),
                opportunity.get('regret_score'),
                json.dumps(opportunity.get('action_items', [])),
                opportunity.get('discovered_at', datetime.utcnow().isoformat()),
                opportunity.get('viewed', False),
                opportunity.get('saved', False),
                opportunity.get('dismissed', False)
            ))
            conn.commit()
            return True

    def save_bias_detection(self, user_id: str, detection: Dict[str, Any]) -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(self._q('''
                INSERT INTO bias_history
                (user_id, bias_type, confidence, trigger_phrase, detected_at,
                 intervention_shown, intervention_accepted)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            '''), (
                user_id,
                detection.get('bias_type'),
                detection.get('confidence'),
                detection.get('trigger_phrase'),
                datetime.utcnow().isoformat(),
                detection.get('intervention_shown', False),
                detection.get('intervention_accepted', False)
            ))
            conn.commit()
            return True

    def get_user_bias_stats(self, user_id: str) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(self._q('''
                SELECT bias_type, COUNT(*) as count, AVG(confidence) as avg_confidence
                FROM bias_history WHERE user_id = %s
                GROUP BY bias_type ORDER BY count DESC
            '''), (user_id,))

            biases = []
            for row in cursor.fetchall():
                d = self._row_to_dict(cursor, row)
                biases.append({
                    'type': d['bias_type'],
                    'count': d['count'],
                    'avg_confidence': d['avg_confidence']
                })

            cursor.execute(self._q('''
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN intervention_accepted = TRUE THEN 1 ELSE 0 END) as accepted
                FROM bias_history WHERE user_id = %s AND intervention_shown = TRUE
            '''), (user_id,))

            intervention_row = cursor.fetchone()
            ir = self._row_to_dict(cursor, intervention_row) if intervention_row else {}

            return {
                'common_biases': biases[:5],
                'total_detections': sum(b['count'] for b in biases),
                'intervention_acceptance_rate': (
                    (ir.get('accepted') or 0) / max(1, ir.get('total') or 1)
                )
            }

    def save_global_outcome(self, outcome: Dict[str, Any]) -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(self._q('''
                INSERT INTO global_outcomes
                (outcome_hash, decision_type, industry, experience_bucket,
                 predicted_regret, actual_regret, satisfaction_score,
                 time_to_outcome_days, key_factors, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (outcome_hash) DO UPDATE SET
                    actual_regret = EXCLUDED.actual_regret,
                    satisfaction_score = EXCLUDED.satisfaction_score,
                    time_to_outcome_days = EXCLUDED.time_to_outcome_days
            '''), (
                outcome.get('outcome_hash'),
                outcome.get('decision_type'),
                outcome.get('industry'),
                outcome.get('experience_bucket'),
                outcome.get('predicted_regret'),
                outcome.get('actual_regret'),
                outcome.get('satisfaction_score'),
                outcome.get('time_to_outcome_days'),
                json.dumps(outcome.get('key_factors', [])),
                datetime.utcnow().isoformat()
            ))
            conn.commit()
            return True

    def get_global_stats(self, decision_type: str = None) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if decision_type:
                cursor.execute(self._q('''
                    SELECT COUNT(*) as count,
                           AVG(actual_regret) as avg_regret,
                           AVG(satisfaction_score) as avg_satisfaction,
                           AVG(ABS(predicted_regret - actual_regret)) as avg_error
                    FROM global_outcomes WHERE decision_type = %s
                '''), (decision_type,))
            else:
                cursor.execute('''
                    SELECT COUNT(*) as count,
                           AVG(actual_regret) as avg_regret,
                           AVG(satisfaction_score) as avg_satisfaction,
                           AVG(ABS(predicted_regret - actual_regret)) as avg_error
                    FROM global_outcomes
                ''')

            row = cursor.fetchone()
            d = self._row_to_dict(cursor, row) if row else {}
            return {
                'sample_count': d.get('count', 0) or 0,
                'avg_regret': d.get('avg_regret', 0) or 0,
                'avg_satisfaction': d.get('avg_satisfaction', 0) or 0,
                'prediction_accuracy': 100 - (d.get('avg_error', 0) or 0)
            }

    def cleanup_old_data(self, days: int = 365) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

            cursor.execute(self._q('DELETE FROM bias_history WHERE detected_at < %s'), (cutoff,))
            deleted = cursor.rowcount

            conn.commit()
            return deleted


try:
    persistence_service = PersistenceService()
except Exception as e:
    import warnings
    import traceback
    error_msg = f"PersistenceService failed to initialise: {e}\n{traceback.format_exc()}"
    warnings.warn(
        f"{error_msg}\nRunning in degraded mode – persistence features disabled."
    )
    persistence_service = None
