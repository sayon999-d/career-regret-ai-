from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import sqlite3
import json
import os
from contextlib import contextmanager

class PersistenceService:
    """
    Persistent storage for user learning profiles, outcomes, and decision history.
    Ensures data survives server restarts.
    """

    def __init__(self, db_path: str = "learning_data.db"):
        self.db_path = db_path
        self._initialize_database()

    def _initialize_database(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
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
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS decision_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    predicted_regret REAL,
                    actual_regret REAL,
                    satisfaction_score REAL,
                    outcome_notes TEXT,
                    surprises TEXT,
                    lessons_learned TEXT,
                    time_since_decision_days INTEGER,
                    recorded_at TEXT,
                    FOREIGN KEY (user_id) REFERENCES learning_profiles(user_id)
                )
            ''')

            cursor.execute('''
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
            ''')

            cursor.execute('''
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
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS scout_profiles (
                    user_id TEXT PRIMARY KEY,
                    current_role TEXT,
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
            ''')

            cursor.execute('''
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
                    viewed BOOLEAN DEFAULT 0,
                    saved BOOLEAN DEFAULT 0,
                    dismissed BOOLEAN DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES scout_profiles(user_id)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS bias_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    bias_type TEXT,
                    confidence REAL,
                    trigger_phrase TEXT,
                    detected_at TEXT,
                    intervention_shown BOOLEAN DEFAULT 0,
                    intervention_accepted BOOLEAN DEFAULT 0
                )
            ''')

            cursor.execute('''
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
            ''')

            cursor.execute('CREATE INDEX IF NOT EXISTS idx_outcomes_user ON decision_outcomes(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_outcomes_decision ON decision_outcomes(decision_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_scout_user ON scout_opportunities(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_bias_user ON bias_history(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_global_type ON global_outcomes(decision_type)')

            conn.commit()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def save_learning_profile(self, user_id: str, profile_data: Dict[str, Any]) -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.utcnow().isoformat()

            cursor.execute('''
                INSERT OR REPLACE INTO learning_profiles
                (user_id, total_outcomes, prediction_accuracy, avg_prediction_error,
                 optimism_bias, decision_style, risk_profile, patterns, created_at, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?,
                        COALESCE((SELECT created_at FROM learning_profiles WHERE user_id = ?), ?), ?)
            ''', (
                user_id,
                profile_data.get('total_outcomes', 0),
                profile_data.get('prediction_accuracy', 0),
                profile_data.get('avg_prediction_error', 0),
                profile_data.get('optimism_bias', 0),
                profile_data.get('decision_style', 'balanced'),
                profile_data.get('risk_profile', 'moderate'),
                json.dumps(profile_data.get('patterns', [])),
                user_id, now, now
            ))
            conn.commit()
            return True

    def get_learning_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM learning_profiles WHERE user_id = ?', (user_id,))
            row = cursor.fetchone()

            if row:
                return {
                    'user_id': row['user_id'],
                    'total_outcomes': row['total_outcomes'],
                    'prediction_accuracy': row['prediction_accuracy'],
                    'avg_prediction_error': row['avg_prediction_error'],
                    'optimism_bias': row['optimism_bias'],
                    'decision_style': row['decision_style'],
                    'risk_profile': row['risk_profile'],
                    'patterns': json.loads(row['patterns']) if row['patterns'] else [],
                    'created_at': row['created_at'],
                    'last_updated': row['last_updated']
                }
            return None

    def save_outcome(self, user_id: str, outcome_data: Dict[str, Any]) -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO decision_outcomes
                (decision_id, user_id, predicted_regret, actual_regret, satisfaction_score,
                 outcome_notes, surprises, lessons_learned, time_since_decision_days, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
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
            cursor.execute('''
                SELECT * FROM decision_outcomes WHERE user_id = ?
                ORDER BY recorded_at DESC LIMIT ?
            ''', (user_id, limit))

            outcomes = []
            for row in cursor.fetchall():
                outcomes.append({
                    'decision_id': row['decision_id'],
                    'predicted_regret': row['predicted_regret'],
                    'actual_regret': row['actual_regret'],
                    'satisfaction_score': row['satisfaction_score'],
                    'outcome_notes': row['outcome_notes'],
                    'surprises': json.loads(row['surprises']) if row['surprises'] else [],
                    'lessons_learned': row['lessons_learned'],
                    'time_since_decision_days': row['time_since_decision_days'],
                    'recorded_at': row['recorded_at']
                })
            return outcomes

    def save_scout_profile(self, user_id: str, profile_data: Dict[str, Any]) -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.utcnow().isoformat()

            cursor.execute('''
                INSERT OR REPLACE INTO scout_profiles
                (user_id, current_role, industry, skills, interests, risk_tolerance,
                 salary_target, location_preferences, low_regret_factors, career_goals,
                 created_at, last_scan)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        COALESCE((SELECT created_at FROM scout_profiles WHERE user_id = ?), ?), ?)
            ''', (
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
                user_id, now, now
            ))
            conn.commit()
            return True

    def save_opportunity(self, user_id: str, opportunity: Dict[str, Any]) -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO scout_opportunities
                (id, user_id, type, title, description, source, match_score,
                 match_reasons, regret_score, action_items, discovered_at, viewed, saved, dismissed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
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

            cursor.execute('''
                INSERT INTO bias_history
                (user_id, bias_type, confidence, trigger_phrase, detected_at,
                 intervention_shown, intervention_accepted)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
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

            cursor.execute('''
                SELECT bias_type, COUNT(*) as count, AVG(confidence) as avg_confidence
                FROM bias_history WHERE user_id = ?
                GROUP BY bias_type ORDER BY count DESC
            ''', (user_id,))

            biases = []
            for row in cursor.fetchall():
                biases.append({
                    'type': row['bias_type'],
                    'count': row['count'],
                    'avg_confidence': row['avg_confidence']
                })

            cursor.execute('''
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN intervention_accepted = 1 THEN 1 ELSE 0 END) as accepted
                FROM bias_history WHERE user_id = ? AND intervention_shown = 1
            ''', (user_id,))

            intervention_row = cursor.fetchone()

            return {
                'common_biases': biases[:5],
                'total_detections': sum(b['count'] for b in biases),
                'intervention_acceptance_rate': (
                    intervention_row['accepted'] / max(1, intervention_row['total'])
                    if intervention_row else 0
                )
            }

    def save_global_outcome(self, outcome: Dict[str, Any]) -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO global_outcomes
                (outcome_hash, decision_type, industry, experience_bucket,
                 predicted_regret, actual_regret, satisfaction_score,
                 time_to_outcome_days, key_factors, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
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
                cursor.execute('''
                    SELECT COUNT(*) as count,
                           AVG(actual_regret) as avg_regret,
                           AVG(satisfaction_score) as avg_satisfaction,
                           AVG(ABS(predicted_regret - actual_regret)) as avg_error
                    FROM global_outcomes WHERE decision_type = ?
                ''', (decision_type,))
            else:
                cursor.execute('''
                    SELECT COUNT(*) as count,
                           AVG(actual_regret) as avg_regret,
                           AVG(satisfaction_score) as avg_satisfaction,
                           AVG(ABS(predicted_regret - actual_regret)) as avg_error
                    FROM global_outcomes
                ''')

            row = cursor.fetchone()
            return {
                'sample_count': row['count'] if row else 0,
                'avg_regret': row['avg_regret'] if row else 0,
                'avg_satisfaction': row['avg_satisfaction'] if row else 0,
                'prediction_accuracy': 100 - (row['avg_error'] if row else 0)
            }

    def cleanup_old_data(self, days: int = 365) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

            cursor.execute('DELETE FROM bias_history WHERE detected_at < ?', (cutoff,))
            deleted = cursor.rowcount

            conn.commit()
            return deleted

persistence_service = PersistenceService()
