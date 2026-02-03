from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum
import random

class AchievementCategory(str, Enum):
    DECISIONS = "decisions"
    STREAKS = "streaks"
    LEARNING = "learning"
    ACCURACY = "accuracy"
    ENGAGEMENT = "engagement"
    MILESTONES = "milestones"

class BadgeLevel(str, Enum):
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
    PLATINUM = "platinum"

@dataclass
class Achievement:

    id: str
    name: str
    description: str
    category: AchievementCategory
    level: BadgeLevel
    icon: str
    requirement: str
    points: int
    unlocked: bool = False
    unlocked_at: Optional[datetime] = None

@dataclass
class Streak:

    streak_type: str
    current_count: int
    longest_count: int
    last_activity: datetime
    is_active: bool

@dataclass
class DailyChallenge:

    id: str
    title: str
    description: str
    reward_points: int
    completed: bool = False
    expires_at: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(days=1))

@dataclass
class UserGamification:

    user_id: str
    total_points: int = 0
    level: int = 1
    current_xp: int = 0
    xp_to_next_level: int = 100
    achievements: List[Achievement] = field(default_factory=list)
    streaks: Dict[str, Streak] = field(default_factory=dict)
    daily_challenges: List[DailyChallenge] = field(default_factory=list)
    decision_skill_score: float = 50.0
    weekly_activity: Dict[str, int] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

class GamificationService:


    ACHIEVEMENTS = {
        'first_decision': {
            'name': 'First Step',
            'description': 'Analyzed your first career decision',
            'category': AchievementCategory.DECISIONS,
            'level': BadgeLevel.BRONZE,
            'icon': '',
            'requirement': 'Analyze 1 decision',
            'points': 10
        },
        'decision_maker': {
            'name': 'Decision Maker',
            'description': 'Analyzed 10 career decisions',
            'category': AchievementCategory.DECISIONS,
            'level': BadgeLevel.SILVER,
            'icon': '',
            'requirement': 'Analyze 10 decisions',
            'points': 50
        },
        'decision_master': {
            'name': 'Decision Master',
            'description': 'Analyzed 50 career decisions',
            'category': AchievementCategory.DECISIONS,
            'level': BadgeLevel.GOLD,
            'icon': '',
            'requirement': 'Analyze 50 decisions',
            'points': 200
        },

        'week_warrior': {
            'name': 'Week Warrior',
            'description': 'Maintained a 7-day reflection streak',
            'category': AchievementCategory.STREAKS,
            'level': BadgeLevel.BRONZE,
            'icon': '',
            'requirement': '7-day streak',
            'points': 30
        },
        'month_master': {
            'name': 'Month Master',
            'description': 'Maintained a 30-day reflection streak',
            'category': AchievementCategory.STREAKS,
            'level': BadgeLevel.GOLD,
            'icon': '',
            'requirement': '30-day streak',
            'points': 150
        },

        'self_aware': {
            'name': 'Self Aware',
            'description': 'Identified 3 cognitive biases in your decisions',
            'category': AchievementCategory.LEARNING,
            'level': BadgeLevel.SILVER,
            'icon': '',
            'requirement': 'Detect 3 biases',
            'points': 40
        },
        'bias_buster': {
            'name': 'Bias Buster',
            'description': 'Completed 5 bias mitigation action items',
            'category': AchievementCategory.LEARNING,
            'level': BadgeLevel.GOLD,
            'icon': '',
            'requirement': 'Complete 5 bias actions',
            'points': 100
        },

        'sharp_predictor': {
            'name': 'Sharp Predictor',
            'description': 'Achieved 70%+ prediction accuracy',
            'category': AchievementCategory.ACCURACY,
            'level': BadgeLevel.SILVER,
            'icon': '',
            'requirement': '70% accuracy',
            'points': 75
        },
        'oracle': {
            'name': 'Oracle',
            'description': 'Achieved 90%+ prediction accuracy with 10+ decisions',
            'category': AchievementCategory.ACCURACY,
            'level': BadgeLevel.PLATINUM,
            'icon': '',
            'requirement': '90% accuracy, 10+ decisions',
            'points': 250
        },

        'contributor': {
            'name': 'Community Contributor',
            'description': 'Shared an outcome to help others',
            'category': AchievementCategory.ENGAGEMENT,
            'level': BadgeLevel.BRONZE,
            'icon': '',
            'requirement': 'Share 1 outcome',
            'points': 25
        },
        'explorer': {
            'name': 'Explorer',
            'description': 'Used 5 different features of the system',
            'category': AchievementCategory.ENGAGEMENT,
            'level': BadgeLevel.SILVER,
            'icon': '',
            'requirement': 'Use 5 features',
            'points': 35
        },

        'level_5': {
            'name': 'Rising Star',
            'description': 'Reached Level 5',
            'category': AchievementCategory.MILESTONES,
            'level': BadgeLevel.SILVER,
            'icon': '',
            'requirement': 'Reach Level 5',
            'points': 50
        },
        'level_10': {
            'name': 'Decision Pro',
            'description': 'Reached Level 10',
            'category': AchievementCategory.MILESTONES,
            'level': BadgeLevel.GOLD,
            'icon': '',
            'requirement': 'Reach Level 10',
            'points': 100
        }
    }

    DAILY_CHALLENGES = [
        {'title': 'Morning Reflection', 'description': 'Start your day by reflecting on a pending decision', 'points': 10},
        {'title': 'Decision Review', 'description': 'Review and update the status of a past decision', 'points': 15},
        {'title': 'Bias Check', 'description': 'Identify potential biases in your current thinking', 'points': 20},
        {'title': 'Future Visualization', 'description': 'Run a career simulation for your current situation', 'points': 15},
        {'title': 'Learn from Others', 'description': 'Read 3 community stories about similar decisions', 'points': 10},
        {'title': 'Action Completion', 'description': 'Complete one of your coaching action items', 'points': 20},
        {'title': 'Gratitude Log', 'description': 'Write down 3 things you are grateful for in your career', 'points': 10},
        {'title': 'Network Outreach', 'description': 'Reach out to someone for career advice', 'points': 25}
    ]

    LEVEL_XP = [100, 250, 500, 800, 1200, 1700, 2300, 3000, 4000, 5000]

    def __init__(self):
        self.user_states: Dict[str, UserGamification] = {}
        self.activity_log: Dict[str, List[Dict]] = defaultdict(list)

    def get_or_create_user(self, user_id: str) -> UserGamification:

        if user_id not in self.user_states:
            self.user_states[user_id] = UserGamification(
                user_id=user_id,
                streaks={
                    'daily_reflection': Streak('daily_reflection', 0, 0, datetime.utcnow(), True),
                    'weekly_decision': Streak('weekly_decision', 0, 0, datetime.utcnow(), True)
                }
            )
            self._initialize_achievements(user_id)

        return self.user_states[user_id]

    def _initialize_achievements(self, user_id: str):

        user = self.user_states[user_id]

        for ach_id, ach_data in self.ACHIEVEMENTS.items():
            user.achievements.append(Achievement(
                id=ach_id,
                name=ach_data['name'],
                description=ach_data['description'],
                category=ach_data['category'],
                level=ach_data['level'],
                icon=ach_data['icon'],
                requirement=ach_data['requirement'],
                points=ach_data['points']
            ))

    def award_points(self, user_id: str, points: int, reason: str) -> Dict[str, Any]:

        user = self.get_or_create_user(user_id)
        old_level = user.level

        user.total_points += points
        user.current_xp += points

        level_ups = []
        while user.current_xp >= user.xp_to_next_level and user.level < 50:
            user.current_xp -= user.xp_to_next_level
            user.level += 1
            level_idx = min(user.level - 1, len(self.LEVEL_XP) - 1)
            user.xp_to_next_level = self.LEVEL_XP[level_idx]
            level_ups.append(user.level)

        self.activity_log[user_id].append({
            'type': 'points_awarded',
            'points': points,
            'reason': reason,
            'timestamp': datetime.utcnow().isoformat()
        })

        self._check_level_achievements(user)

        return {
            'points_awarded': points,
            'total_points': user.total_points,
            'current_level': user.level,
            'current_xp': user.current_xp,
            'xp_to_next_level': user.xp_to_next_level,
            'level_ups': level_ups
        }

    def record_activity(self, user_id: str, activity_type: str, data: Dict = None) -> Dict[str, Any]:

        user = self.get_or_create_user(user_id)
        now = datetime.utcnow()

        points_map = {
            'decision_analyzed': 15,
            'chat_message': 2,
            'outcome_recorded': 25,
            'action_completed': 20,
            'simulation_run': 10,
            'report_generated': 10,
            'story_shared': 30,
            'reflection_made': 10
        }

        points = points_map.get(activity_type, 5)

        week_key = now.strftime('%Y-W%W')
        user.weekly_activity[week_key] = user.weekly_activity.get(week_key, 0) + 1

        streak_updates = self._update_streaks(user, activity_type, now)

        point_result = self.award_points(user_id, points, activity_type)

        new_achievements = self._check_achievements(user, activity_type, data)

        self._update_skill_score(user, activity_type, data)

        return {
            'activity_recorded': activity_type,
            'points': point_result,
            'streak_updates': streak_updates,
            'new_achievements': new_achievements
        }

    def _update_streaks(self, user: UserGamification, activity_type: str, now: datetime) -> List[Dict]:

        updates = []

        if activity_type in ['decision_analyzed', 'reflection_made', 'outcome_recorded']:
            streak = user.streaks.get('daily_reflection')
            if streak:
                last = streak.last_activity
                days_diff = (now.date() - last.date()).days

                if days_diff == 1:
                    streak.current_count += 1
                    streak.is_active = True
                    if streak.current_count > streak.longest_count:
                        streak.longest_count = streak.current_count
                    updates.append({'streak': 'daily_reflection', 'count': streak.current_count, 'action': 'continued'})
                elif days_diff == 0:
                    pass
                else:
                    if streak.current_count > 0:
                        updates.append({'streak': 'daily_reflection', 'count': streak.current_count, 'action': 'broken'})
                    streak.current_count = 1
                    streak.is_active = True

                streak.last_activity = now

        return updates

    def _check_achievements(self, user: UserGamification, activity_type: str, data: Dict = None) -> List[Dict]:

        newly_unlocked = []

        activities = self.activity_log.get(user.user_id, [])
        decision_count = sum(1 for a in activities if a.get('type') == 'points_awarded' and 'decision' in a.get('reason', ''))

        for achievement in user.achievements:
            if achievement.unlocked:
                continue

            unlocked = False

            if achievement.id == 'first_decision' and decision_count >= 1:
                unlocked = True
            elif achievement.id == 'decision_maker' and decision_count >= 10:
                unlocked = True
            elif achievement.id == 'decision_master' and decision_count >= 50:
                unlocked = True
            elif achievement.id == 'week_warrior' and user.streaks.get('daily_reflection', Streak('', 0, 0, datetime.utcnow(), False)).current_count >= 7:
                unlocked = True
            elif achievement.id == 'month_master' and user.streaks.get('daily_reflection', Streak('', 0, 0, datetime.utcnow(), False)).longest_count >= 30:
                unlocked = True
            elif achievement.id == 'contributor' and activity_type == 'story_shared':
                unlocked = True

            if unlocked:
                achievement.unlocked = True
                achievement.unlocked_at = datetime.utcnow()
                newly_unlocked.append({
                    'id': achievement.id,
                    'name': achievement.name,
                    'icon': achievement.icon,
                    'points': achievement.points,
                    'level': achievement.level.value
                })

                user.total_points += achievement.points

        return newly_unlocked

    def _check_level_achievements(self, user: UserGamification):

        for achievement in user.achievements:
            if achievement.unlocked:
                continue

            if achievement.id == 'level_5' and user.level >= 5:
                achievement.unlocked = True
                achievement.unlocked_at = datetime.utcnow()
            elif achievement.id == 'level_10' and user.level >= 10:
                achievement.unlocked = True
                achievement.unlocked_at = datetime.utcnow()

    def _update_skill_score(self, user: UserGamification, activity_type: str, data: Dict = None):

        increases = {
            'decision_analyzed': 0.5,
            'outcome_recorded': 1.0,
            'action_completed': 0.8,
            'reflection_made': 0.3
        }

        if activity_type in increases:
            user.decision_skill_score = min(100, user.decision_skill_score + increases[activity_type])

    def get_daily_challenges(self, user_id: str) -> List[DailyChallenge]:

        user = self.get_or_create_user(user_id)

        now = datetime.utcnow()
        user.daily_challenges = [c for c in user.daily_challenges if c.expires_at > now and not c.completed]

        while len(user.daily_challenges) < 3:
            template = random.choice(self.DAILY_CHALLENGES)
            challenge = DailyChallenge(
                id=f"challenge_{random.randint(1000, 9999)}",
                title=template['title'],
                description=template['description'],
                reward_points=template['points'],
                expires_at=now.replace(hour=23, minute=59, second=59) + timedelta(days=1)
            )
            user.daily_challenges.append(challenge)

        return user.daily_challenges

    def complete_challenge(self, user_id: str, challenge_id: str) -> Dict[str, Any]:

        user = self.get_or_create_user(user_id)

        for challenge in user.daily_challenges:
            if challenge.id == challenge_id and not challenge.completed:
                challenge.completed = True
                points_result = self.award_points(user_id, challenge.reward_points, f"challenge_{challenge_id}")

                return {
                    'success': True,
                    'challenge': challenge.title,
                    'points_awarded': challenge.reward_points,
                    'result': points_result
                }

        return {'success': False, 'message': 'Challenge not found or already completed'}

    def get_leaderboard(self, limit: int = 10) -> List[Dict[str, Any]]:

        users = list(self.user_states.values())
        users.sort(key=lambda u: u.total_points, reverse=True)

        leaderboard = []
        for i, user in enumerate(users[:limit]):
            leaderboard.append({
                'rank': i + 1,
                'user_id': user.user_id[:8] + '...',
                'level': user.level,
                'points': user.total_points,
                'achievements_count': sum(1 for a in user.achievements if a.unlocked),
                'skill_score': user.decision_skill_score
            })

        return leaderboard

    def get_user_stats(self, user_id: str) -> Dict[str, Any]:

        user = self.get_or_create_user(user_id)

        unlocked_achievements = [a for a in user.achievements if a.unlocked]
        locked_achievements = [a for a in user.achievements if not a.unlocked]

        return {
            'user_id': user_id,
            'level': user.level,
            'total_points': user.total_points,
            'current_xp': user.current_xp,
            'xp_to_next_level': user.xp_to_next_level,
            'progress_to_next_level': user.current_xp / user.xp_to_next_level * 100,
            'decision_skill_score': user.decision_skill_score,
            'streaks': {
                name: {
                    'current': streak.current_count,
                    'longest': streak.longest_count,
                    'active': streak.is_active
                }
                for name, streak in user.streaks.items()
            },
            'achievements': {
                'unlocked_count': len(unlocked_achievements),
                'total_count': len(user.achievements),
                'recent_unlocks': [
                    {
                        'name': a.name,
                        'icon': a.icon,
                        'unlocked_at': a.unlocked_at.isoformat() if a.unlocked_at else None
                    }
                    for a in sorted(unlocked_achievements, key=lambda x: x.unlocked_at or datetime.min, reverse=True)[:5]
                ],
                'next_achievements': [
                    {
                        'name': a.name,
                        'icon': a.icon,
                        'requirement': a.requirement,
                        'points': a.points
                    }
                    for a in locked_achievements[:3]
                ]
            },
            'activity_summary': {
                'this_week': sum(
                    v for k, v in user.weekly_activity.items()
                    if k == datetime.utcnow().strftime('%Y-W%W')
                ),
                'total_activities': len(self.activity_log.get(user_id, []))
            },
            'member_since': user.created_at.isoformat()
        }

    def get_motivational_message(self, user_id: str) -> str:

        user = self.get_or_create_user(user_id)

        messages = []

        if user.level < 5:
            messages.append("You're building strong decision-making habits. Keep going!")
        elif user.level < 10:
            messages.append("You're becoming a skilled decision-maker. Your future self will thank you!")
        else:
            messages.append("You're a decision-making pro! Share your wisdom with others.")

        streak = user.streaks.get('daily_reflection', Streak('', 0, 0, datetime.utcnow(), False))
        if streak.current_count >= 7:
            messages.append(f"Amazing! {streak.current_count}-day streak! You're on fire!")
        elif streak.current_count >= 3:
            messages.append(f"{streak.current_count}-day streak going! Consistency is key.")

        if user.decision_skill_score >= 80:
            messages.append("Your decision skill score is excellent!")
        elif user.decision_skill_score >= 60:
            messages.append("Your decision-making skills are improving steadily.")

        return random.choice(messages) if messages else "Every decision is a step forward!"

    def get_reflection_prompts(self) -> List[str]:

        prompts = [
            "What decision have you been avoiding? Why?",
            "What would you do if you weren't afraid?",
            "What's one thing you learned from a past decision?",
            "How does your current role align with your 5-year vision?",
            "What decision would your future self thank you for making today?",
            "What's holding you back from that career move you've been considering?",
            "If money weren't a factor, what would you choose?",
            "What advice would you give to someone in your exact situation?",
            "What's the worst realistic outcome? Could you handle it?",
            "What opportunity cost are you paying by not deciding?"
        ]
        return random.sample(prompts, min(3, len(prompts)))

    def to_dict(self, user: UserGamification) -> Dict[str, Any]:

        return {
            'user_id': user.user_id,
            'level': user.level,
            'total_points': user.total_points,
            'current_xp': user.current_xp,
            'xp_to_next_level': user.xp_to_next_level,
            'decision_skill_score': user.decision_skill_score,
            'achievements': [
                {
                    'id': a.id,
                    'name': a.name,
                    'description': a.description,
                    'category': a.category.value,
                    'level': a.level.value,
                    'icon': a.icon,
                    'points': a.points,
                    'unlocked': a.unlocked,
                    'unlocked_at': a.unlocked_at.isoformat() if a.unlocked_at else None
                }
                for a in user.achievements
            ],
            'streaks': {
                name: {
                    'current': s.current_count,
                    'longest': s.longest_count,
                    'active': s.is_active
                }
                for name, s in user.streaks.items()
            },
            'daily_challenges': [
                {
                    'id': c.id,
                    'title': c.title,
                    'description': c.description,
                    'points': c.reward_points,
                    'completed': c.completed,
                    'expires_at': c.expires_at.isoformat()
                }
                for c in user.daily_challenges
            ]
        }
