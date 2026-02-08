import sqlite3
import json
import hashlib
import secrets
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from contextlib import contextmanager
import bcrypt
import os

DATABASE_PATH = os.getenv("DATABASE_PATH", "/tmp/career_regret_ai.db")

class DatabaseService:
    """Central database service for all persistent storage needs"""
    
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self._initialize_database()
    
    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _initialize_database(self):
        """Create all required tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    full_name TEXT,
                    avatar_url TEXT,
                    bio TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    email_verified BOOLEAN DEFAULT 0,
                    is_active BOOLEAN DEFAULT 1,
                    preferences TEXT DEFAULT '{}',
                    theme TEXT DEFAULT 'dark',
                    notification_settings TEXT DEFAULT '{}'
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    token TEXT UNIQUE NOT NULL,
                    refresh_token TEXT UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    device_info TEXT,
                    ip_address TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS decisions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    decision_type TEXT,
                    status TEXT DEFAULT 'pending',
                    predicted_regret REAL,
                    actual_regret REAL,
                    confidence REAL,
                    emotions TEXT DEFAULT '[]',
                    factors TEXT DEFAULT '[]',
                    pros TEXT DEFAULT '[]',
                    cons TEXT DEFAULT '[]',
                    chosen_option TEXT,
                    alternatives TEXT DEFAULT '[]',
                    notes TEXT,
                    tags TEXT DEFAULT '[]',
                    template_id TEXT,
                    nlp_analysis TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    decided_at TIMESTAMP,
                    follow_up_date TIMESTAMP,
                    outcome_recorded_at TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS decision_outcomes (
                    id TEXT PRIMARY KEY,
                    decision_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    actual_regret REAL,
                    satisfaction_score REAL,
                    outcome_notes TEXT,
                    surprises TEXT DEFAULT '[]',
                    lessons_learned TEXT,
                    would_do_again BOOLEAN,
                    time_since_decision_days INTEGER,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (decision_id) REFERENCES decisions(id),
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS calendar_events (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    event_type TEXT NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    location TEXT,
                    reminder_minutes TEXT DEFAULT '[30, 1440]',
                    recurrence TEXT,
                    color_id TEXT DEFAULT '1',
                    google_event_id TEXT,
                    synced BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_documents (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    content TEXT,
                    doc_type TEXT DEFAULT 'general',
                    summary TEXT,
                    embedding_id TEXT,
                    size_bytes INTEGER,
                    char_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT,
                    preview TEXT,
                    messages TEXT DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS notifications (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT,
                    notification_type TEXT DEFAULT 'info',
                    action_url TEXT,
                    is_read BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_analytics (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    metadata TEXT DEFAULT '{}',
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_feedback (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    message_id TEXT,
                    feedback_type TEXT,
                    rating INTEGER,
                    comment TEXT,
                    context TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS backups (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    backup_type TEXT DEFAULT 'full',
                    file_path TEXT,
                    size_bytes INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_decisions_user ON decisions(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_decisions_status ON decisions(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_decisions_type ON decisions(decision_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_calendar_user ON calendar_events(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_calendar_date ON calendar_events(start_time)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_notifications_user ON notifications(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)')
            
            conn.commit()
            print(f"Database initialized at {self.db_path}")
    
    def create_user(self, email: str, username: str, password: str, full_name: str = None) -> Optional[Dict]:
        """Create a new user with hashed password"""
        user_id = secrets.token_hex(16)
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('''
                    INSERT INTO users (id, email, username, password_hash, full_name, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (user_id, email.lower(), username, password_hash, full_name, datetime.utcnow().isoformat()))
                conn.commit()
                return self.get_user_by_id(user_id)
            except sqlite3.IntegrityError:
                return None
    
    def authenticate_user(self, email: str, password: str) -> Optional[Dict]:
        """Authenticate user with email and password"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE email = ? AND is_active = 1', (email.lower(),))
            row = cursor.fetchone()
            
            if row and bcrypt.checkpw(password.encode('utf-8'), row['password_hash'].encode('utf-8')):
                cursor.execute('UPDATE users SET last_login = ? WHERE id = ?', 
                             (datetime.utcnow().isoformat(), row['id']))
                conn.commit()
                return dict(row)
            return None
    
    def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_user_by_email(self, email: str) -> Optional[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE email = ?', (email.lower(),))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def update_user(self, user_id: str, **updates) -> bool:
        allowed_fields = ['full_name', 'bio', 'avatar_url', 'preferences', 'theme', 'notification_settings']
        updates = {k: v for k, v in updates.items() if k in allowed_fields}
        
        if not updates:
            return False
        
        set_clause = ', '.join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [user_id]
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'UPDATE users SET {set_clause} WHERE id = ?', values)
            conn.commit()
            return cursor.rowcount > 0
    
    def change_password(self, user_id: str, old_password: str, new_password: str) -> bool:
        user = self.get_user_by_id(user_id)
        if not user:
            return False
        
        if not bcrypt.checkpw(old_password.encode('utf-8'), user['password_hash'].encode('utf-8')):
            return False
        
        new_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE users SET password_hash = ? WHERE id = ?', (new_hash, user_id))
            conn.commit()
            return True
    
    def create_session(self, user_id: str, device_info: str = None, ip_address: str = None, 
                       expires_hours: int = 24) -> Dict:
        session_id = secrets.token_hex(16)
        token = secrets.token_urlsafe(64)
        refresh_token = secrets.token_urlsafe(64)
        expires_at = datetime.utcnow() + timedelta(hours=expires_hours)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO sessions (id, user_id, token, refresh_token, expires_at, device_info, ip_address)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (session_id, user_id, token, refresh_token, expires_at.isoformat(), device_info, ip_address))
            conn.commit()
        
        return {
            'session_id': session_id,
            'token': token,
            'refresh_token': refresh_token,
            'expires_at': expires_at.isoformat()
        }
    
    def validate_session(self, token: str) -> Optional[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT s.*, u.email, u.username, u.full_name 
                FROM sessions s 
                JOIN users u ON s.user_id = u.id
                WHERE s.token = ? AND s.is_active = 1 AND s.expires_at > ?
            ''', (token, datetime.utcnow().isoformat()))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def invalidate_session(self, token: str) -> bool:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE sessions SET is_active = 0 WHERE token = ?', (token,))
            conn.commit()
            return cursor.rowcount > 0
    
    def refresh_session(self, refresh_token: str) -> Optional[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM sessions WHERE refresh_token = ? AND is_active = 1', (refresh_token,))
            row = cursor.fetchone()
            
            if row:
                new_token = secrets.token_urlsafe(64)
                new_expires = datetime.utcnow() + timedelta(hours=24)
                cursor.execute('''
                    UPDATE sessions SET token = ?, expires_at = ? WHERE id = ?
                ''', (new_token, new_expires.isoformat(), row['id']))
                conn.commit()
                return {
                    'token': new_token,
                    'expires_at': new_expires.isoformat()
                }
            return None
    
    def create_decision(self, user_id: str, data: Dict) -> Dict:
        decision_id = secrets.token_hex(8)
        now = datetime.utcnow().isoformat()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO decisions (
                    id, user_id, title, description, decision_type, status,
                    predicted_regret, confidence, emotions, factors, pros, cons,
                    chosen_option, alternatives, notes, tags, template_id, nlp_analysis,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                decision_id, user_id,
                data.get('title', ''),
                data.get('description', ''),
                data.get('decision_type', 'general'),
                data.get('status', 'pending'),
                data.get('predicted_regret'),
                data.get('confidence'),
                json.dumps(data.get('emotions', [])),
                json.dumps(data.get('factors', [])),
                json.dumps(data.get('pros', [])),
                json.dumps(data.get('cons', [])),
                data.get('chosen_option'),
                json.dumps(data.get('alternatives', [])),
                data.get('notes'),
                json.dumps(data.get('tags', [])),
                data.get('template_id'),
                json.dumps(data.get('nlp_analysis', {})),
                now, now
            ))
            conn.commit()
        
        return self.get_decision(user_id, decision_id)
    
    def get_decision(self, user_id: str, decision_id: str) -> Optional[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM decisions WHERE id = ? AND user_id = ?', (decision_id, user_id))
            row = cursor.fetchone()
            if row:
                d = dict(row)
                for field in ['emotions', 'factors', 'pros', 'cons', 'alternatives', 'tags', 'nlp_analysis']:
                    if d.get(field):
                        try:
                            d[field] = json.loads(d[field])
                        except:
                            pass
                return d
            return None
    
    def get_decisions(self, user_id: str, status: str = None, decision_type: str = None,
                      search: str = None, limit: int = 50, offset: int = 0,
                      sort_by: str = 'created_at', sort_order: str = 'desc') -> Tuple[List[Dict], int]:
        """Get decisions with filtering, searching, and pagination"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = 'SELECT * FROM decisions WHERE user_id = ?'
            count_query = 'SELECT COUNT(*) as count FROM decisions WHERE user_id = ?'
            params = [user_id]
            
            if status:
                query += ' AND status = ?'
                count_query += ' AND status = ?'
                params.append(status)
            
            if decision_type:
                query += ' AND decision_type = ?'
                count_query += ' AND decision_type = ?'
                params.append(decision_type)
            
            if search:
                query += ' AND (title LIKE ? OR description LIKE ? OR notes LIKE ?)'
                count_query += ' AND (title LIKE ? OR description LIKE ? OR notes LIKE ?)'
                search_param = f'%{search}%'
                params.extend([search_param, search_param, search_param])
            
            cursor.execute(count_query, params)
            total = cursor.fetchone()['count']
            
            allowed_sort = ['created_at', 'updated_at', 'predicted_regret', 'title', 'decision_type']
            if sort_by not in allowed_sort:
                sort_by = 'created_at'
            sort_order = 'DESC' if sort_order.lower() == 'desc' else 'ASC'
            
            query += f' ORDER BY {sort_by} {sort_order} LIMIT ? OFFSET ?'
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            decisions = []
            for row in rows:
                d = dict(row)
                for field in ['emotions', 'factors', 'pros', 'cons', 'alternatives', 'tags', 'nlp_analysis']:
                    if d.get(field):
                        try:
                            d[field] = json.loads(d[field])
                        except:
                            pass
                decisions.append(d)
            
            return decisions, total
    
    def update_decision(self, user_id: str, decision_id: str, data: Dict) -> bool:
        allowed_fields = ['title', 'description', 'decision_type', 'status', 'predicted_regret',
                         'actual_regret', 'confidence', 'emotions', 'factors', 'pros', 'cons',
                         'chosen_option', 'alternatives', 'notes', 'tags', 'decided_at', 'follow_up_date']
        
        updates = {}
        for k, v in data.items():
            if k in allowed_fields:
                if isinstance(v, (list, dict)):
                    updates[k] = json.dumps(v)
                else:
                    updates[k] = v
        
        if not updates:
            return False
        
        updates['updated_at'] = datetime.utcnow().isoformat()
        set_clause = ', '.join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [decision_id, user_id]
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'UPDATE decisions SET {set_clause} WHERE id = ? AND user_id = ?', values)
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_decision(self, user_id: str, decision_id: str) -> bool:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM decisions WHERE id = ? AND user_id = ?', (decision_id, user_id))
            conn.commit()
            return cursor.rowcount > 0
    
    def record_outcome(self, user_id: str, decision_id: str, data: Dict) -> Dict:
        outcome_id = secrets.token_hex(8)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO decision_outcomes (
                    id, decision_id, user_id, actual_regret, satisfaction_score,
                    outcome_notes, surprises, lessons_learned, would_do_again, time_since_decision_days
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                outcome_id, decision_id, user_id,
                data.get('actual_regret'),
                data.get('satisfaction_score'),
                data.get('outcome_notes'),
                json.dumps(data.get('surprises', [])),
                data.get('lessons_learned'),
                data.get('would_do_again'),
                data.get('time_since_decision_days')
            ))
            
            if data.get('actual_regret') is not None:
                cursor.execute('''
                    UPDATE decisions SET actual_regret = ?, outcome_recorded_at = ? WHERE id = ?
                ''', (data['actual_regret'], datetime.utcnow().isoformat(), decision_id))
            
            conn.commit()
        
        return {'outcome_id': outcome_id, 'decision_id': decision_id}
    
    def create_calendar_event(self, user_id: str, data: Dict) -> Dict:
        event_id = secrets.token_hex(8)
        now = datetime.utcnow().isoformat()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO calendar_events (
                    id, user_id, title, description, event_type, start_time, end_time,
                    location, reminder_minutes, recurrence, color_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event_id, user_id,
                data.get('title'),
                data.get('description', ''),
                data.get('event_type', 'general'),
                data.get('start_time'),
                data.get('end_time'),
                data.get('location', ''),
                json.dumps(data.get('reminder_minutes', [30, 1440])),
                data.get('recurrence'),
                data.get('color_id', '1'),
                now, now
            ))
            conn.commit()
        
        return self.get_calendar_event(user_id, event_id)
    
    def get_calendar_event(self, user_id: str, event_id: str) -> Optional[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM calendar_events WHERE id = ? AND user_id = ?', (event_id, user_id))
            row = cursor.fetchone()
            if row:
                d = dict(row)
                if d.get('reminder_minutes'):
                    try:
                        d['reminder_minutes'] = json.loads(d['reminder_minutes'])
                    except:
                        pass
                return d
            return None
    
    def get_calendar_events(self, user_id: str, start_date: str = None, end_date: str = None,
                           event_type: str = None) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = 'SELECT * FROM calendar_events WHERE user_id = ?'
            params = [user_id]
            
            if start_date:
                query += ' AND start_time >= ?'
                params.append(start_date)
            
            if end_date:
                query += ' AND start_time <= ?'
                params.append(end_date)
            
            if event_type:
                query += ' AND event_type = ?'
                params.append(event_type)
            
            query += ' ORDER BY start_time ASC'
            
            cursor.execute(query, params)
            events = []
            for row in cursor.fetchall():
                d = dict(row)
                if d.get('reminder_minutes'):
                    try:
                        d['reminder_minutes'] = json.loads(d['reminder_minutes'])
                    except:
                        pass
                events.append(d)
            return events
    
    def update_calendar_event(self, user_id: str, event_id: str, data: Dict) -> bool:
        allowed_fields = ['title', 'description', 'event_type', 'start_time', 'end_time',
                         'location', 'reminder_minutes', 'recurrence', 'color_id', 'synced']
        
        updates = {}
        for k, v in data.items():
            if k in allowed_fields:
                if isinstance(v, list):
                    updates[k] = json.dumps(v)
                else:
                    updates[k] = v
        
        if not updates:
            return False
        
        updates['updated_at'] = datetime.utcnow().isoformat()
        set_clause = ', '.join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [event_id, user_id]
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'UPDATE calendar_events SET {set_clause} WHERE id = ? AND user_id = ?', values)
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_calendar_event(self, user_id: str, event_id: str) -> bool:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM calendar_events WHERE id = ? AND user_id = ?', (event_id, user_id))
            conn.commit()
            return cursor.rowcount > 0
    
    def add_document(self, user_id: str, filename: str, content: str, doc_type: str = 'general') -> Dict:
        doc_id = secrets.token_hex(8)
        summary = f"Document about {filename} with {len(content)} characters of specialized knowledge."
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO knowledge_documents (
                    id, user_id, filename, content, doc_type, summary, char_count, size_bytes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (doc_id, user_id, filename, content, doc_type, summary, len(content), len(content.encode('utf-8'))))
            conn.commit()
        
        return {
            'id': doc_id,
            'filename': filename,
            'type': doc_type,
            'chars': len(content),
            'summary': summary,
            'added_at': datetime.utcnow().isoformat()
        }
    
    def get_documents(self, user_id: str) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, filename, doc_type as type, char_count as chars, summary, created_at as added_at
                FROM knowledge_documents WHERE user_id = ? ORDER BY created_at DESC
            ''', (user_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    def delete_document(self, user_id: str, doc_id: str) -> bool:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM knowledge_documents WHERE id = ? AND user_id = ?', (doc_id, user_id))
            conn.commit()
            return cursor.rowcount > 0
    
    # ============ CONVERSATIONS ============
    
    def save_conversation(self, user_id: str, title: str, messages: List[Dict]) -> Dict:
        conv_id = secrets.token_hex(8)
        preview = messages[0].get('content', '')[:50] if messages else ''
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO conversations (id, user_id, title, preview, messages)
                VALUES (?, ?, ?, ?, ?)
            ''', (conv_id, user_id, title, preview, json.dumps(messages)))
            conn.commit()
        
        return {'id': conv_id, 'title': title, 'preview': preview}
    
    def get_conversations(self, user_id: str, limit: int = 50) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, title, preview, created_at as timestamp, 
                       json_array_length(messages) as message_count
                FROM conversations WHERE user_id = ? ORDER BY created_at DESC LIMIT ?
            ''', (user_id, limit))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_conversation(self, user_id: str, conv_id: str) -> Optional[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM conversations WHERE id = ? AND user_id = ?', (conv_id, user_id))
            row = cursor.fetchone()
            if row:
                d = dict(row)
                d['messages'] = json.loads(d.get('messages', '[]'))
                return d
            return None
    
    def delete_conversation(self, user_id: str, conv_id: str) -> bool:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM conversations WHERE id = ? AND user_id = ?', (conv_id, user_id))
            conn.commit()
            return cursor.rowcount > 0
    
    def create_notification(self, user_id: str, title: str, message: str, 
                           notification_type: str = 'info', action_url: str = None) -> Dict:
        notif_id = secrets.token_hex(8)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO notifications (id, user_id, title, message, notification_type, action_url)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (notif_id, user_id, title, message, notification_type, action_url))
            conn.commit()
        
        return {'id': notif_id, 'title': title, 'message': message, 'type': notification_type}
    
    def get_notifications(self, user_id: str, unread_only: bool = False, limit: int = 50) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = 'SELECT * FROM notifications WHERE user_id = ?'
            params = [user_id]
            
            if unread_only:
                query += ' AND is_read = 0'
            
            query += ' ORDER BY created_at DESC LIMIT ?'
            params.append(limit)
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def mark_notification_read(self, user_id: str, notification_id: str) -> bool:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE notifications SET is_read = 1 WHERE id = ? AND user_id = ?', 
                          (notification_id, user_id))
            conn.commit()
            return cursor.rowcount > 0
    
    def mark_all_notifications_read(self, user_id: str) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE notifications SET is_read = 1 WHERE user_id = ? AND is_read = 0', (user_id,))
            conn.commit()
            return cursor.rowcount
    
    def record_metric(self, user_id: str, metric_name: str, value: float, metadata: Dict = None):
        metric_id = secrets.token_hex(8)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO user_analytics (id, user_id, metric_name, metric_value, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (metric_id, user_id, metric_name, value, json.dumps(metadata or {})))
            conn.commit()
    
    def get_analytics_summary(self, user_id: str) -> Dict:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT COUNT(*) as total_decisions,
                       AVG(predicted_regret) as avg_predicted_regret,
                       AVG(actual_regret) as avg_actual_regret,
                       SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_decisions
                FROM decisions WHERE user_id = ?
            ''', (user_id,))
            decision_stats = dict(cursor.fetchone())
            
            cursor.execute('''
                SELECT decision_type, COUNT(*) as count 
                FROM decisions WHERE user_id = ? 
                GROUP BY decision_type ORDER BY count DESC LIMIT 5
            ''', (user_id,))
            decision_types = [dict(row) for row in cursor.fetchall()]
            
            cursor.execute('''
                SELECT DATE(created_at) as date, COUNT(*) as count
                FROM decisions WHERE user_id = ? AND created_at > date('now', '-30 days')
                GROUP BY DATE(created_at) ORDER BY date DESC
            ''', (user_id,))
            activity = [dict(row) for row in cursor.fetchall()]
            
            return {
                'decision_stats': decision_stats,
                'decision_types': decision_types,
                'recent_activity': activity
            }
    
    def record_ai_feedback(self, user_id: str, message_id: str, feedback_type: str, 
                          rating: int = None, comment: str = None, context: Dict = None):
        feedback_id = secrets.token_hex(8)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO ai_feedback (id, user_id, message_id, feedback_type, rating, comment, context)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (feedback_id, user_id, message_id, feedback_type, rating, comment, json.dumps(context or {})))
            conn.commit()
    
    def export_user_data(self, user_id: str) -> Dict:
        """Export all user data for backup/download"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, email, username, full_name, bio, created_at, preferences, theme
                FROM users WHERE id = ?
            ''', (user_id,))
            user = dict(cursor.fetchone()) if cursor.fetchone() else {}
            
            decisions, _ = self.get_decisions(user_id, limit=10000)
            
            events = self.get_calendar_events(user_id)
            
            documents = self.get_documents(user_id)
            
            conversations = self.get_conversations(user_id, limit=10000)
            
            return {
                'exported_at': datetime.utcnow().isoformat(),
                'version': '1.0',
                'user': user,
                'decisions': decisions,
                'calendar_events': events,
                'documents': documents,
                'conversations': conversations
            }
    
    def import_user_data(self, user_id: str, data: Dict) -> Dict:
        """Import user data from backup"""
        imported = {'decisions': 0, 'events': 0, 'documents': 0}
        
        for decision in data.get('decisions', []):
            try:
                decision.pop('id', None) 
                self.create_decision(user_id, decision)
                imported['decisions'] += 1
            except:
                pass
        
        for event in data.get('calendar_events', []):
            try:
                event.pop('id', None)
                self.create_calendar_event(user_id, event)
                imported['events'] += 1
            except:
                pass
        
        return imported

db_service = DatabaseService()
