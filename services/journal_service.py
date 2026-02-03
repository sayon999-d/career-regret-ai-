import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import json

class DecisionStatus(str, Enum):
    PENDING = "pending"
    DECIDED = "decided"
    COMPLETED = "completed"
    ABANDONED = "abandoned"

class FollowUpType(str, Enum):
    DAYS_30 = "30_days"
    DAYS_90 = "90_days"
    DAYS_180 = "180_days"
    CUSTOM = "custom"

@dataclass
class FollowUp:

    id: str
    decision_id: str
    follow_up_type: FollowUpType
    scheduled_date: datetime
    completed: bool = False
    completed_date: Optional[datetime] = None
    notes: Optional[str] = None

@dataclass
class DecisionOutcome:

    actual_regret: float
    satisfaction: float
    would_decide_same: bool
    lessons_learned: str
    unexpected_outcomes: List[str]
    recorded_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class JournalEntry:

    id: str
    user_id: str
    decision_type: str
    title: str
    description: str
    status: DecisionStatus
    predicted_regret: float
    predicted_confidence: float
    emotions: List[str]
    factors: Dict[str, float]
    nlp_analysis: Optional[Dict] = None
    chosen_option: Optional[str] = None
    alternatives: List[str] = field(default_factory=list)
    outcome: Optional[DecisionOutcome] = None
    follow_ups: List[FollowUp] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    decided_at: Optional[datetime] = None

class JournalService:


    def __init__(self):
        self.entries: Dict[str, JournalEntry] = {}
        self.user_entries: Dict[str, List[str]] = defaultdict(list)
        self.follow_up_queue: List[FollowUp] = []

    def create_entry(
        self,
        user_id: str,
        decision_type: str,
        title: str,
        description: str,
        predicted_regret: float,
        predicted_confidence: float,
        emotions: List[str] = None,
        factors: Dict[str, float] = None,
        nlp_analysis: Dict = None,
        alternatives: List[str] = None,
        tags: List[str] = None
    ) -> JournalEntry:

        entry_id = f"journal_{uuid.uuid4().hex[:12]}"

        entry = JournalEntry(
            id=entry_id,
            user_id=user_id,
            decision_type=decision_type,
            title=title,
            description=description,
            status=DecisionStatus.PENDING,
            predicted_regret=predicted_regret,
            predicted_confidence=predicted_confidence,
            emotions=emotions or [],
            factors=factors or {},
            nlp_analysis=nlp_analysis,
            alternatives=alternatives or [],
            tags=tags or []
        )

        self._create_default_followups(entry)

        self.entries[entry_id] = entry
        self.user_entries[user_id].append(entry_id)

        return entry

    def _create_default_followups(self, entry: JournalEntry):

        now = datetime.utcnow()

        follow_up_configs = [
            (FollowUpType.DAYS_30, timedelta(days=30)),
            (FollowUpType.DAYS_90, timedelta(days=90)),
            (FollowUpType.DAYS_180, timedelta(days=180))
        ]

        for fu_type, delta in follow_up_configs:
            follow_up = FollowUp(
                id=f"fu_{uuid.uuid4().hex[:8]}",
                decision_id=entry.id,
                follow_up_type=fu_type,
                scheduled_date=now + delta
            )
            entry.follow_ups.append(follow_up)
            self.follow_up_queue.append(follow_up)

    def record_decision(
        self,
        entry_id: str,
        chosen_option: str,
        notes: str = ""
    ) -> Optional[JournalEntry]:

        entry = self.entries.get(entry_id)
        if not entry:
            return None

        entry.status = DecisionStatus.DECIDED
        entry.chosen_option = chosen_option
        entry.decided_at = datetime.utcnow()
        entry.notes = notes
        entry.updated_at = datetime.utcnow()

        return entry

    def record_outcome(
        self,
        entry_id: str,
        actual_regret: float,
        satisfaction: float,
        would_decide_same: bool,
        lessons_learned: str = "",
        unexpected_outcomes: List[str] = None
    ) -> Optional[JournalEntry]:

        entry = self.entries.get(entry_id)
        if not entry:
            return None

        entry.outcome = DecisionOutcome(
            actual_regret=actual_regret,
            satisfaction=satisfaction,
            would_decide_same=would_decide_same,
            lessons_learned=lessons_learned,
            unexpected_outcomes=unexpected_outcomes or []
        )
        entry.status = DecisionStatus.COMPLETED
        entry.updated_at = datetime.utcnow()

        return entry

    def get_entry(self, entry_id: str) -> Optional[JournalEntry]:

        return self.entries.get(entry_id)

    def get_user_entries(
        self,
        user_id: str,
        status: DecisionStatus = None,
        limit: int = 50
    ) -> List[JournalEntry]:

        entry_ids = self.user_entries.get(user_id, [])
        entries = [self.entries[eid] for eid in entry_ids if eid in self.entries]

        if status:
            entries = [e for e in entries if e.status == status]

        entries.sort(key=lambda e: e.created_at, reverse=True)

        return entries[:limit]

    def get_pending_followups(self, user_id: str = None) -> List[Dict]:

        now = datetime.utcnow()
        pending = []

        for follow_up in self.follow_up_queue:
            if follow_up.completed:
                continue
            if follow_up.scheduled_date > now:
                continue

            entry = self.entries.get(follow_up.decision_id)
            if not entry:
                continue

            if user_id and entry.user_id != user_id:
                continue

            pending.append({
                'follow_up_id': follow_up.id,
                'decision_id': follow_up.decision_id,
                'title': entry.title,
                'type': follow_up.follow_up_type,
                'scheduled_date': follow_up.scheduled_date.isoformat(),
                'days_overdue': (now - follow_up.scheduled_date).days
            })

        return pending

    def complete_followup(self, follow_up_id: str, notes: str = "") -> bool:

        for follow_up in self.follow_up_queue:
            if follow_up.id == follow_up_id:
                follow_up.completed = True
                follow_up.completed_date = datetime.utcnow()
                follow_up.notes = notes
                return True
        return False

    def get_accuracy_metrics(self, user_id: str = None) -> Dict[str, Any]:

        completed_entries = [
            e for e in self.entries.values()
            if e.status == DecisionStatus.COMPLETED and e.outcome
            and (user_id is None or e.user_id == user_id)
        ]

        if not completed_entries:
            return {
                'total_completed': 0,
                'accuracy': None,
                'avg_prediction_error': None,
                'satisfaction_correlation': None
            }

        prediction_errors = []
        satisfaction_scores = []
        correct_predictions = 0

        for entry in completed_entries:
            error = abs(entry.predicted_regret - entry.outcome.actual_regret)
            prediction_errors.append(error)
            satisfaction_scores.append(entry.outcome.satisfaction)

            if error <= 0.2:
                correct_predictions += 1

        avg_error = sum(prediction_errors) / len(prediction_errors)
        accuracy = correct_predictions / len(completed_entries)
        avg_satisfaction = sum(satisfaction_scores) / len(satisfaction_scores)

        would_repeat = sum(1 for e in completed_entries if e.outcome.would_decide_same)
        repeat_rate = would_repeat / len(completed_entries)

        return {
            'total_completed': len(completed_entries),
            'accuracy': accuracy,
            'avg_prediction_error': avg_error,
            'avg_satisfaction': avg_satisfaction,
            'repeat_decision_rate': repeat_rate,
            'total_pending': len([e for e in self.entries.values() if e.status == DecisionStatus.PENDING]),
            'total_decided': len([e for e in self.entries.values() if e.status == DecisionStatus.DECIDED])
        }

    def get_timeline(self, user_id: str, days: int = 365) -> List[Dict]:

        cutoff = datetime.utcnow() - timedelta(days=days)
        entries = self.get_user_entries(user_id)

        timeline = []
        for entry in entries:
            if entry.created_at < cutoff:
                continue

            item = {
                'id': entry.id,
                'title': entry.title,
                'type': entry.decision_type,
                'status': entry.status.value,
                'predicted_regret': entry.predicted_regret,
                'created_at': entry.created_at.isoformat(),
                'decided_at': entry.decided_at.isoformat() if entry.decided_at else None
            }

            if entry.outcome:
                item['actual_regret'] = entry.outcome.actual_regret
                item['satisfaction'] = entry.outcome.satisfaction

            timeline.append(item)

        return timeline

    def search_entries(
        self,
        user_id: str,
        query: str = None,
        decision_type: str = None,
        tags: List[str] = None,
        date_from: datetime = None,
        date_to: datetime = None
    ) -> List[JournalEntry]:

        entries = self.get_user_entries(user_id, limit=1000)

        if query:
            query_lower = query.lower()
            entries = [
                e for e in entries
                if query_lower in e.title.lower() or query_lower in e.description.lower()
            ]

        if decision_type:
            entries = [e for e in entries if e.decision_type == decision_type]

        if tags:
            entries = [e for e in entries if any(t in e.tags for t in tags)]

        if date_from:
            entries = [e for e in entries if e.created_at >= date_from]

        if date_to:
            entries = [e for e in entries if e.created_at <= date_to]

        return entries

    def get_statistics(self, user_id: str) -> Dict[str, Any]:

        entries = self.get_user_entries(user_id, limit=1000)

        if not entries:
            return {
                'total_entries': 0,
                'by_status': {},
                'by_type': {},
                'avg_regret_predicted': 0,
                'most_common_emotions': []
            }

        by_status = defaultdict(int)
        by_type = defaultdict(int)
        emotions_count = defaultdict(int)
        total_regret = 0

        for entry in entries:
            by_status[entry.status.value] += 1
            by_type[entry.decision_type] += 1
            total_regret += entry.predicted_regret
            for emotion in entry.emotions:
                emotions_count[emotion] += 1

        top_emotions = sorted(emotions_count.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            'total_entries': len(entries),
            'by_status': dict(by_status),
            'by_type': dict(by_type),
            'avg_regret_predicted': total_regret / len(entries),
            'most_common_emotions': [{'emotion': e, 'count': c} for e, c in top_emotions],
            'oldest_entry': min(e.created_at for e in entries).isoformat(),
            'newest_entry': max(e.created_at for e in entries).isoformat()
        }

    def to_dict(self, entry: JournalEntry) -> Dict[str, Any]:

        result = {
            'id': entry.id,
            'user_id': entry.user_id,
            'decision_type': entry.decision_type,
            'title': entry.title,
            'description': entry.description,
            'status': entry.status.value,
            'predicted_regret': entry.predicted_regret,
            'predicted_confidence': entry.predicted_confidence,
            'emotions': entry.emotions,
            'factors': entry.factors,
            'chosen_option': entry.chosen_option,
            'alternatives': entry.alternatives,
            'tags': entry.tags,
            'notes': entry.notes,
            'created_at': entry.created_at.isoformat(),
            'updated_at': entry.updated_at.isoformat(),
            'decided_at': entry.decided_at.isoformat() if entry.decided_at else None,
            'follow_ups': [
                {
                    'id': fu.id,
                    'type': fu.follow_up_type.value,
                    'scheduled_date': fu.scheduled_date.isoformat(),
                    'completed': fu.completed
                }
                for fu in entry.follow_ups
            ]
        }

        if entry.outcome:
            result['outcome'] = {
                'actual_regret': entry.outcome.actual_regret,
                'satisfaction': entry.outcome.satisfaction,
                'would_decide_same': entry.outcome.would_decide_same,
                'lessons_learned': entry.outcome.lessons_learned,
                'unexpected_outcomes': entry.outcome.unexpected_outcomes,
                'recorded_at': entry.outcome.recorded_at.isoformat()
            }

        return result
