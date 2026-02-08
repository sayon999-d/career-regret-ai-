import os
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass
import json

from .database_service import db_service


class NotificationType(str, Enum):
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    REMINDER = "reminder"
    DECISION_DUE = "decision_due"
    FOLLOW_UP = "follow_up"
    WEEKLY_DIGEST = "weekly_digest"
    OUTCOME_REVIEW = "outcome_review"


@dataclass
class NotificationSettings:
    email_notifications: bool = True
    push_notifications: bool = True
    decision_reminders: bool = True
    weekly_digest: bool = True
    outcome_reviews: bool = True
    reminder_days_before: int = 1
    digest_day: str = "monday"


class NotificationService:
    """Manages all notification functionality"""
    
    def __init__(self):
        self.default_settings = NotificationSettings()
        self._pending_notifications = {}
    
    def create_notification(
        self,
        user_id: str,
        title: str,
        message: str,
        notification_type: NotificationType = NotificationType.INFO,
        action_url: str = None,
        persist: bool = True
    ) -> Dict:
        """Create a new notification"""
        if persist:
            result = db_service.create_notification(
                user_id=user_id,
                title=title,
                message=message,
                notification_type=notification_type.value,
                action_url=action_url
            )
        else:
            result = {
                "id": secrets.token_hex(8),
                "title": title,
                "message": message,
                "type": notification_type.value,
                "action_url": action_url,
                "created_at": datetime.utcnow().isoformat(),
                "is_read": False
            }
        
        if user_id not in self._pending_notifications:
            self._pending_notifications[user_id] = []
        self._pending_notifications[user_id].append(result)
        
        return result
    
    def get_notifications(
        self,
        user_id: str,
        unread_only: bool = False,
        limit: int = 50
    ) -> List[Dict]:
        """Get user notifications"""
        return db_service.get_notifications(user_id, unread_only, limit)
    
    def get_unread_count(self, user_id: str) -> int:
        """Get count of unread notifications"""
        notifications = db_service.get_notifications(user_id, unread_only=True)
        return len(notifications)
    
    def mark_read(self, user_id: str, notification_id: str) -> bool:
        """Mark a notification as read"""
        return db_service.mark_notification_read(user_id, notification_id)
    
    def mark_all_read(self, user_id: str) -> int:
        """Mark all notifications as read"""
        return db_service.mark_all_notifications_read(user_id)
    
    def get_pending_realtime(self, user_id: str) -> List[Dict]:
        """Get pending notifications for real-time delivery and clear them"""
        pending = self._pending_notifications.get(user_id, [])
        self._pending_notifications[user_id] = []
        return pending
    
    
    def create_decision_reminder(
        self,
        user_id: str,
        decision_id: str,
        decision_title: str,
        due_date: datetime,
        days_before: int = 1
    ) -> Dict:
        """Create a reminder for an upcoming decision deadline"""
        reminder_date = due_date - timedelta(days=days_before)
        
        if reminder_date <= datetime.utcnow():
            return self.create_notification(
                user_id=user_id,
                title="Decision Due Soon!",
                message=f"Your decision '{decision_title}' is due {'today' if due_date.date() == datetime.utcnow().date() else 'soon'}!",
                notification_type=NotificationType.DECISION_DUE,
                action_url=f"/decisions/{decision_id}"
            )
        else:
            return {
                "scheduled": True,
                "reminder_date": reminder_date.isoformat(),
                "decision_id": decision_id
            }
    
    def create_outcome_review_reminder(
        self,
        user_id: str,
        decision_id: str,
        decision_title: str,
        review_days: int = 30
    ) -> Dict:
        """Create a reminder to review decision outcome"""
        return self.create_notification(
            user_id=user_id,
            title="Time to Review Your Decision",
            message=f"It's been {review_days} days since you made the decision '{decision_title}'. How did it turn out?",
            notification_type=NotificationType.OUTCOME_REVIEW,
            action_url=f"/decisions/{decision_id}/outcome"
        )
    
    def create_follow_up_notification(
        self,
        user_id: str,
        decision_id: str,
        decision_title: str
    ) -> Dict:
        """Create a follow-up notification"""
        return self.create_notification(
            user_id=user_id,
            title="Decision Follow-up",
            message=f"Reminder to follow up on your decision: '{decision_title}'",
            notification_type=NotificationType.FOLLOW_UP,
            action_url=f"/decisions/{decision_id}"
        )
    
    
    def create_event_reminder(
        self,
        user_id: str,
        event_id: str,
        event_title: str,
        event_time: datetime,
        minutes_before: int = 30
    ) -> Dict:
        """Create a reminder for an upcoming calendar event"""
        return self.create_notification(
            user_id=user_id,
            title=f"Upcoming: {event_title}",
            message=f"Your event starts in {minutes_before} minutes at {event_time.strftime('%I:%M %p')}",
            notification_type=NotificationType.REMINDER,
            action_url=f"/calendar/{event_id}"
        )
    
    
    def generate_weekly_digest(self, user_id: str) -> Dict:
        """Generate weekly digest notification"""
        analytics = db_service.get_analytics_summary(user_id)
        decision_stats = analytics.get('decision_stats', {})
        
        total = decision_stats.get('total_decisions', 0)
        completed = decision_stats.get('completed_decisions', 0)
        avg_regret = decision_stats.get('avg_predicted_regret', 0) or 0
        
        message = f"""
Your weekly career decision summary:
• {total} total decisions tracked
• {completed} decisions completed
• Average predicted regret: {avg_regret:.1f}%

Keep making thoughtful decisions!
""".strip()
        
        return self.create_notification(
            user_id=user_id,
            title="Your Weekly Career Report",
            message=message,
            notification_type=NotificationType.WEEKLY_DIGEST,
            action_url="/analytics"
        )
    
    # ============ SYSTEM NOTIFICATIONS ============
    
    def welcome_notification(self, user_id: str, username: str) -> Dict:
        """Create welcome notification for new users"""
        return self.create_notification(
            user_id=user_id,
            title=f"Welcome, {username}! 🎉",
            message="Start by exploring decision templates or chat with the AI advisor.",
            notification_type=NotificationType.SUCCESS,
            action_url="/templates"
        )
    
    def achievement_notification(
        self,
        user_id: str,
        achievement: str,
        description: str
    ) -> Dict:
        """Create achievement notification"""
        return self.create_notification(
            user_id=user_id,
            title=f"Achievement Unlocked: {achievement}",
            message=description,
            notification_type=NotificationType.SUCCESS,
            action_url="/profile/achievements"
        )
    
    
    def send_email_notification(
        self,
        user_id: str,
        subject: str,
        body: str,
        html_body: str = None
    ) -> bool:
        """Send email notification (placeholder for actual email service)"""
        print(f"[EMAIL] To: {user_id} | Subject: {subject}")
        return True
    
    def send_email_digest(self, user_id: str) -> bool:
        """Send weekly email digest"""
        user = db_service.get_user_by_id(user_id)
        if not user:
            return False
        
        analytics = db_service.get_analytics_summary(user_id)
        
        subject = "Your Weekly Career Decision Report"
        body = f"""
Hi {user.get('username', 'there')},

Here's your weekly career decision summary:

{json.dumps(analytics, indent=2)}

Keep making thoughtful decisions!

- Career Decision Regret AI
"""
        return self.send_email_notification(user_id, subject, body)
    
    def send_push_notification(
        self,
        user_id: str,
        title: str,
        body: str,
        data: Dict = None
    ) -> bool:
        """Send push notification (placeholder for actual push service)"""
        print(f"[PUSH] To: {user_id} | Title: {title}")
        return True


notification_service = NotificationService()
