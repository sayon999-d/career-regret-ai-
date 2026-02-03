from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import asyncio


class CheckInType(str, Enum):
    DECISION_FOLLOW_UP = "decision_follow_up"
    GOAL_REVIEW = "goal_review"
    WEEKLY_REFLECTION = "weekly_reflection"
    OUTCOME_VERIFICATION = "outcome_verification"
    ENGAGEMENT_NUDGE = "engagement_nudge"
    BIAS_PATTERN_REVIEW = "bias_pattern_review"


class CheckInFrequency(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    CUSTOM = "custom"


@dataclass
class ScheduledCheckIn:
    id: str
    user_id: str
    check_in_type: CheckInType
    title: str
    description: str
    frequency: CheckInFrequency
    next_due: datetime
    last_completed: Optional[datetime] = None
    is_active: bool = True
    custom_interval_days: int = 0
    related_decision_id: Optional[str] = None
    related_goal_id: Optional[str] = None
    questions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CheckInResponse:
    check_in_id: str
    user_id: str
    responses: Dict[str, Any]
    mood_score: int
    completed_at: datetime = field(default_factory=datetime.utcnow)
    insights_generated: List[str] = field(default_factory=list)


class ScheduledCheckInService:
    """
    Manages automated check-ins and follow-ups for user engagement
    """

    DEFAULT_QUESTIONS = {
        CheckInType.DECISION_FOLLOW_UP: [
            "How do you feel about this decision now?",
            "Have there been any unexpected outcomes?",
            "Would you make the same choice again?",
            "What have you learned from this decision?"
        ],
        CheckInType.GOAL_REVIEW: [
            "What progress have you made toward this goal?",
            "What obstacles are you facing?",
            "Do you need to adjust your timeline?",
            "What's your next action step?"
        ],
        CheckInType.WEEKLY_REFLECTION: [
            "What was your biggest win this week?",
            "What decision are you currently pondering?",
            "What would you do differently next week?",
            "How would you rate your decision-making this week?"
        ],
        CheckInType.OUTCOME_VERIFICATION: [
            "Did the outcome match your expectations?",
            "On a scale of 1-10, how satisfied are you?",
            "What contributed most to this outcome?",
            "What insights can we learn from this?"
        ],
        CheckInType.BIAS_PATTERN_REVIEW: [
            "Have you noticed any recurring thought patterns?",
            "Were there decisions where emotions overrode logic?",
            "What triggered your biggest stress this period?",
            "How can you improve your decision process?"
        ]
    }

    FREQUENCY_DAYS = {
        CheckInFrequency.DAILY: 1,
        CheckInFrequency.WEEKLY: 7,
        CheckInFrequency.BIWEEKLY: 14,
        CheckInFrequency.MONTHLY: 30,
        CheckInFrequency.QUARTERLY: 90
    }

    def __init__(self):
        self.check_ins: Dict[str, List[ScheduledCheckIn]] = {}
        self.responses: Dict[str, List[CheckInResponse]] = {}
        self.user_preferences: Dict[str, Dict[str, Any]] = {}

    def create_check_in(
        self,
        user_id: str,
        check_in_type: CheckInType,
        title: str,
        description: str = "",
        frequency: CheckInFrequency = CheckInFrequency.WEEKLY,
        custom_interval_days: int = 0,
        start_date: datetime = None,
        related_decision_id: str = None,
        related_goal_id: str = None,
        custom_questions: List[str] = None
    ) -> ScheduledCheckIn:
        """Create a new scheduled check-in"""
        check_in_id = hashlib.md5(
            f"{user_id}{check_in_type.value}{datetime.utcnow().timestamp()}".encode()
        ).hexdigest()[:12]

        interval_days = (
            custom_interval_days if frequency == CheckInFrequency.CUSTOM
            else self.FREQUENCY_DAYS.get(frequency, 7)
        )

        next_due = start_date or (datetime.utcnow() + timedelta(days=interval_days))

        questions = custom_questions or self.DEFAULT_QUESTIONS.get(check_in_type, [])

        check_in = ScheduledCheckIn(
            id=check_in_id,
            user_id=user_id,
            check_in_type=check_in_type,
            title=title,
            description=description,
            frequency=frequency,
            next_due=next_due,
            custom_interval_days=interval_days,
            related_decision_id=related_decision_id,
            related_goal_id=related_goal_id,
            questions=questions
        )

        if user_id not in self.check_ins:
            self.check_ins[user_id] = []
        self.check_ins[user_id].append(check_in)

        return check_in

    def get_due_check_ins(
        self,
        user_id: str,
        include_overdue: bool = True
    ) -> List[Dict[str, Any]]:
        """Get all check-ins that are due or overdue"""
        if user_id not in self.check_ins:
            return []

        now = datetime.utcnow()
        due_check_ins = []

        for check_in in self.check_ins[user_id]:
            if not check_in.is_active:
                continue

            if check_in.next_due <= now:
                days_overdue = (now - check_in.next_due).days
                due_check_ins.append({
                    "id": check_in.id,
                    "type": check_in.check_in_type.value,
                    "title": check_in.title,
                    "description": check_in.description,
                    "due_date": check_in.next_due.isoformat(),
                    "is_overdue": days_overdue > 0,
                    "days_overdue": days_overdue,
                    "questions": check_in.questions,
                    "related_decision_id": check_in.related_decision_id,
                    "related_goal_id": check_in.related_goal_id
                })
            elif not include_overdue:
                due_check_ins.append({
                    "id": check_in.id,
                    "type": check_in.check_in_type.value,
                    "title": check_in.title,
                    "due_date": check_in.next_due.isoformat(),
                    "is_overdue": False,
                    "days_until_due": (check_in.next_due - now).days
                })

        return sorted(due_check_ins, key=lambda x: x.get("days_overdue", 0), reverse=True)

    def get_all_check_ins(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all check-ins for a user"""
        if user_id not in self.check_ins:
            return []

        return [{
            "id": c.id,
            "type": c.check_in_type.value,
            "title": c.title,
            "frequency": c.frequency.value,
            "next_due": c.next_due.isoformat(),
            "is_active": c.is_active,
            "last_completed": c.last_completed.isoformat() if c.last_completed else None
        } for c in self.check_ins[user_id]]

    def complete_check_in(
        self,
        user_id: str,
        check_in_id: str,
        responses: Dict[str, Any],
        mood_score: int = 5
    ) -> Dict[str, Any]:
        """Complete a check-in with responses"""
        if user_id not in self.check_ins:
            return {"error": "No check-ins found for user"}

        check_in = None
        for c in self.check_ins[user_id]:
            if c.id == check_in_id:
                check_in = c
                break

        if not check_in:
            return {"error": "Check-in not found"}

        response = CheckInResponse(
            check_in_id=check_in_id,
            user_id=user_id,
            responses=responses,
            mood_score=mood_score,
            insights_generated=self._generate_insights(check_in, responses, mood_score)
        )

        if user_id not in self.responses:
            self.responses[user_id] = []
        self.responses[user_id].append(response)

        check_in.last_completed = datetime.utcnow()
        interval_days = check_in.custom_interval_days or self.FREQUENCY_DAYS.get(
            check_in.frequency, 7
        )
        check_in.next_due = datetime.utcnow() + timedelta(days=interval_days)

        return {
            "completed": True,
            "check_in_id": check_in_id,
            "next_due": check_in.next_due.isoformat(),
            "insights": response.insights_generated,
            "streak": self._calculate_streak(user_id)
        }

    def _generate_insights(
        self,
        check_in: ScheduledCheckIn,
        responses: Dict[str, Any],
        mood_score: int
    ) -> List[str]:
        """Generate insights from check-in responses"""
        insights = []

        if mood_score <= 3:
            insights.append("Your mood seems lower than usual. Consider revisiting your support strategies.")
        elif mood_score >= 8:
            insights.append("Great mood! This is a good time for important decisions.")

        if check_in.check_in_type == CheckInType.DECISION_FOLLOW_UP:
            insights.append("Tracking decision outcomes improves future predictions by 23%.")
        elif check_in.check_in_type == CheckInType.GOAL_REVIEW:
            insights.append("Regular goal reviews increase success rate by 42%.")

        return insights

    def _calculate_streak(self, user_id: str) -> int:
        """Calculate the user's check-in completion streak"""
        if user_id not in self.responses:
            return 1

        responses = sorted(
            self.responses[user_id],
            key=lambda x: x.completed_at,
            reverse=True
        )

        streak = 0
        last_date = None

        for response in responses:
            response_date = response.completed_at.date()
            if last_date is None:
                streak = 1
                last_date = response_date
            elif (last_date - response_date).days <= 7:
                streak += 1
                last_date = response_date
            else:
                break

        return streak

    def pause_check_in(self, user_id: str, check_in_id: str) -> bool:
        """Pause a scheduled check-in"""
        if user_id not in self.check_ins:
            return False

        for check_in in self.check_ins[user_id]:
            if check_in.id == check_in_id:
                check_in.is_active = False
                return True
        return False

    def resume_check_in(self, user_id: str, check_in_id: str) -> bool:
        """Resume a paused check-in"""
        if user_id not in self.check_ins:
            return False

        for check_in in self.check_ins[user_id]:
            if check_in.id == check_in_id:
                check_in.is_active = True
                check_in.next_due = datetime.utcnow() + timedelta(days=1)
                return True
        return False

    def delete_check_in(self, user_id: str, check_in_id: str) -> bool:
        """Delete a scheduled check-in"""
        if user_id not in self.check_ins:
            return False

        self.check_ins[user_id] = [
            c for c in self.check_ins[user_id] if c.id != check_in_id
        ]
        return True

    def setup_default_check_ins(self, user_id: str) -> List[Dict[str, Any]]:
        """Set up default check-ins for a new user"""
        default_check_ins = [
            {
                "type": CheckInType.WEEKLY_REFLECTION,
                "title": "Weekly Reflection",
                "description": "Reflect on your week and decision-making",
                "frequency": CheckInFrequency.WEEKLY
            },
            {
                "type": CheckInType.BIAS_PATTERN_REVIEW,
                "title": "Monthly Bias Review",
                "description": "Review and improve your thinking patterns",
                "frequency": CheckInFrequency.MONTHLY
            }
        ]

        created = []
        for config in default_check_ins:
            check_in = self.create_check_in(
                user_id=user_id,
                check_in_type=config["type"],
                title=config["title"],
                description=config["description"],
                frequency=config["frequency"]
            )
            created.append({
                "id": check_in.id,
                "title": check_in.title,
                "frequency": check_in.frequency.value
            })

        return created

    def create_decision_follow_up(
        self,
        user_id: str,
        decision_id: str,
        decision_title: str,
        follow_up_days: int = 30
    ) -> ScheduledCheckIn:
        """Create a follow-up check-in for a specific decision"""
        return self.create_check_in(
            user_id=user_id,
            check_in_type=CheckInType.DECISION_FOLLOW_UP,
            title=f"Follow-up: {decision_title}",
            description=f"How did '{decision_title}' turn out?",
            frequency=CheckInFrequency.CUSTOM,
            custom_interval_days=follow_up_days,
            related_decision_id=decision_id
        )

    def create_goal_check_in(
        self,
        user_id: str,
        goal_id: str,
        goal_title: str,
        frequency: CheckInFrequency = CheckInFrequency.WEEKLY
    ) -> ScheduledCheckIn:
        """Create a check-in for goal progress tracking"""
        return self.create_check_in(
            user_id=user_id,
            check_in_type=CheckInType.GOAL_REVIEW,
            title=f"Goal Review: {goal_title}",
            description=f"Track your progress toward '{goal_title}'",
            frequency=frequency,
            related_goal_id=goal_id
        )

    def get_check_in_stats(self, user_id: str) -> Dict[str, Any]:
        """Get check-in statistics for a user"""
        check_ins = self.check_ins.get(user_id, [])
        responses = self.responses.get(user_id, [])

        total_completed = len(responses)
        avg_mood = sum(r.mood_score for r in responses) / max(1, len(responses))

        return {
            "total_check_ins": len(check_ins),
            "active_check_ins": len([c for c in check_ins if c.is_active]),
            "total_completed": total_completed,
            "average_mood": round(avg_mood, 1),
            "current_streak": self._calculate_streak(user_id),
            "completion_rate": round(total_completed / max(1, len(check_ins)) * 100, 1)
        }


scheduled_checkin_service = ScheduledCheckInService()
