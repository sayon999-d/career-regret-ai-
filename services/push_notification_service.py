from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json
import asyncio


class NotificationType(str, Enum):
    OPPORTUNITY_ALERT = "opportunity_alert"
    DECISION_REMINDER = "decision_reminder"
    BIAS_WARNING = "bias_warning"
    GOAL_PROGRESS = "goal_progress"
    CHECK_IN = "check_in"
    ACHIEVEMENT = "achievement"
    SYSTEM = "system"


class NotificationPriority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class PushSubscription:
    user_id: str
    endpoint: str
    p256dh_key: str
    auth_key: str
    subscribed_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True
    preferences: Dict[str, bool] = field(default_factory=dict)


@dataclass
class Notification:
    id: str
    user_id: str
    type: NotificationType
    title: str
    body: str
    data: Dict[str, Any]
    priority: NotificationPriority
    created_at: datetime = field(default_factory=datetime.utcnow)
    sent_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    clicked_at: Optional[datetime] = None


class PushNotificationService:
    """
    Manages push notifications for real-time user engagement
    """

    DEFAULT_PREFERENCES = {
        NotificationType.OPPORTUNITY_ALERT.value: True,
        NotificationType.DECISION_REMINDER.value: True,
        NotificationType.BIAS_WARNING.value: True,
        NotificationType.GOAL_PROGRESS.value: True,
        NotificationType.CHECK_IN.value: True,
        NotificationType.ACHIEVEMENT.value: True,
        NotificationType.SYSTEM.value: True
    }

    def __init__(self):
        self.subscriptions: Dict[str, PushSubscription] = {}
        self.notifications: Dict[str, List[Notification]] = {}
        self.notification_queue: asyncio.Queue = asyncio.Queue()
        self._worker_running = False

    def subscribe(
        self,
        user_id: str,
        endpoint: str,
        p256dh_key: str,
        auth_key: str,
        preferences: Dict[str, bool] = None
    ) -> Dict[str, Any]:
        """Register a push subscription for a user"""
        subscription = PushSubscription(
            user_id=user_id,
            endpoint=endpoint,
            p256dh_key=p256dh_key,
            auth_key=auth_key,
            preferences=preferences or self.DEFAULT_PREFERENCES.copy()
        )

        self.subscriptions[user_id] = subscription

        return {
            "subscribed": True,
            "user_id": user_id,
            "preferences": subscription.preferences,
            "message": "Push notifications enabled successfully"
        }

    def unsubscribe(self, user_id: str) -> Dict[str, Any]:
        """Unsubscribe a user from push notifications"""
        if user_id in self.subscriptions:
            self.subscriptions[user_id].is_active = False
            return {"unsubscribed": True, "user_id": user_id}
        return {"unsubscribed": False, "error": "Subscription not found"}

    def update_preferences(
        self,
        user_id: str,
        preferences: Dict[str, bool]
    ) -> Dict[str, Any]:
        """Update notification preferences for a user"""
        if user_id not in self.subscriptions:
            return {"error": "User not subscribed"}

        self.subscriptions[user_id].preferences.update(preferences)
        return {
            "updated": True,
            "preferences": self.subscriptions[user_id].preferences
        }

    def get_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get notification preferences for a user"""
        if user_id not in self.subscriptions:
            return {
                "subscribed": False,
                "preferences": self.DEFAULT_PREFERENCES
            }

        sub = self.subscriptions[user_id]
        return {
            "subscribed": sub.is_active,
            "preferences": sub.preferences,
            "subscribed_at": sub.subscribed_at.isoformat()
        }

    async def send_notification(
        self,
        user_id: str,
        notification_type: NotificationType,
        title: str,
        body: str,
        data: Dict[str, Any] = None,
        priority: NotificationPriority = NotificationPriority.MEDIUM
    ) -> Dict[str, Any]:
        """Queue a notification for sending"""
        if user_id not in self.subscriptions:
            return {"sent": False, "reason": "User not subscribed"}

        sub = self.subscriptions[user_id]
        if not sub.is_active:
            return {"sent": False, "reason": "Subscription inactive"}

        if not sub.preferences.get(notification_type.value, True):
            return {"sent": False, "reason": "Notification type disabled by user"}

        notification_id = hashlib.sha256(
            f"{user_id}{datetime.utcnow().timestamp()}".encode()
        ).hexdigest()[:12]

        notification = Notification(
            id=notification_id,
            user_id=user_id,
            type=notification_type,
            title=title,
            body=body,
            data=data or {},
            priority=priority
        )

        if user_id not in self.notifications:
            self.notifications[user_id] = []
        self.notifications[user_id].append(notification)

        await self.notification_queue.put(notification)

        return {
            "sent": True,
            "notification_id": notification_id,
            "queued_at": datetime.utcnow().isoformat()
        }

    def get_notifications(
        self,
        user_id: str,
        unread_only: bool = False,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get notifications for a user"""
        if user_id not in self.notifications:
            return []

        notifications = self.notifications[user_id]

        if unread_only:
            notifications = [n for n in notifications if n.read_at is None]

        notifications = sorted(
            notifications,
            key=lambda x: x.created_at,
            reverse=True
        )[:limit]

        return [{
            "id": n.id,
            "type": n.type.value,
            "title": n.title,
            "body": n.body,
            "data": n.data,
            "priority": n.priority.value,
            "created_at": n.created_at.isoformat(),
            "read": n.read_at is not None
        } for n in notifications]

    def mark_as_read(self, user_id: str, notification_id: str) -> bool:
        """Mark a notification as read"""
        if user_id not in self.notifications:
            return False

        for notification in self.notifications[user_id]:
            if notification.id == notification_id:
                notification.read_at = datetime.utcnow()
                return True
        return False

    def mark_all_read(self, user_id: str) -> int:
        """Mark all notifications as read"""
        if user_id not in self.notifications:
            return 0

        count = 0
        for notification in self.notifications[user_id]:
            if notification.read_at is None:
                notification.read_at = datetime.utcnow()
                count += 1
        return count

    def get_unread_count(self, user_id: str) -> int:
        """Get count of unread notifications"""
        if user_id not in self.notifications:
            return 0
        return len([n for n in self.notifications[user_id] if n.read_at is None])

    async def send_opportunity_alert(
        self,
        user_id: str,
        opportunity_title: str,
        match_score: float,
        opportunity_id: str
    ):
        """Send an opportunity alert notification"""
        await self.send_notification(
            user_id=user_id,
            notification_type=NotificationType.OPPORTUNITY_ALERT,
            title="New Opportunity Match!",
            body=f"{opportunity_title} - {int(match_score * 100)}% match",
            data={"opportunity_id": opportunity_id, "match_score": match_score},
            priority=NotificationPriority.HIGH if match_score > 0.8 else NotificationPriority.MEDIUM
        )

    async def send_decision_reminder(
        self,
        user_id: str,
        decision_title: str,
        days_pending: int
    ):
        """Send a reminder about a pending decision"""
        await self.send_notification(
            user_id=user_id,
            notification_type=NotificationType.DECISION_REMINDER,
            title="Decision Pending",
            body=f"'{decision_title}' has been pending for {days_pending} days",
            data={"decision_title": decision_title},
            priority=NotificationPriority.MEDIUM
        )

    async def send_bias_warning(
        self,
        user_id: str,
        bias_type: str,
        context: str
    ):
        """Send a bias detection warning"""
        await self.send_notification(
            user_id=user_id,
            notification_type=NotificationType.BIAS_WARNING,
            title="Bias Detected",
            body=f"Potential {bias_type.replace('_', ' ')} detected in your recent analysis",
            data={"bias_type": bias_type, "context": context},
            priority=NotificationPriority.HIGH
        )

    async def send_goal_update(
        self,
        user_id: str,
        goal_title: str,
        progress: int,
        milestone: str = None
    ):
        """Send a goal progress update"""
        body = f"'{goal_title}' is now at {progress}%"
        if milestone:
            body += f" - Milestone reached: {milestone}"

        await self.send_notification(
            user_id=user_id,
            notification_type=NotificationType.GOAL_PROGRESS,
            title="Goal Progress",
            body=body,
            data={"goal_title": goal_title, "progress": progress},
            priority=NotificationPriority.LOW
        )

    async def send_achievement(
        self,
        user_id: str,
        achievement_name: str,
        achievement_description: str
    ):
        """Send an achievement unlocked notification"""
        await self.send_notification(
            user_id=user_id,
            notification_type=NotificationType.ACHIEVEMENT,
            title="Achievement Unlocked!",
            body=f"{achievement_name}: {achievement_description}",
            data={"achievement": achievement_name},
            priority=NotificationPriority.MEDIUM
        )


push_notification_service = PushNotificationService()
