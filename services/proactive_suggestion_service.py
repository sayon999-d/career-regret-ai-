from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import random
import hashlib


class SuggestionType(str, Enum):
    SKILL_DEVELOPMENT = "skill_development"
    CAREER_MOVE = "career_move"
    NETWORKING = "networking"
    LEARNING_RESOURCE = "learning_resource"
    DECISION_PROMPT = "decision_prompt"
    REFLECTION = "reflection"
    MARKET_INSIGHT = "market_insight"
    GOAL_NUDGE = "goal_nudge"
    BIAS_AWARENESS = "bias_awareness"
    WELLNESS = "wellness"


class SuggestionPriority(str, Enum):
    URGENT = "urgent"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class UserContext:
    user_id: str
    current_role: str = ""
    target_role: str = ""
    skills: List[str] = field(default_factory=list)
    recent_decisions: int = 0
    pending_decisions: int = 0
    goals: List[str] = field(default_factory=list)
    last_active: datetime = field(default_factory=datetime.utcnow)
    engagement_score: float = 0.5
    bias_patterns: List[str] = field(default_factory=list)
    mood_trend: str = "neutral"


@dataclass
class Suggestion:
    id: str
    user_id: str
    type: SuggestionType
    title: str
    body: str
    reasoning: str
    actions: List[Dict[str, str]]
    priority: SuggestionPriority
    relevance_score: float
    expires_at: datetime
    created_at: datetime = field(default_factory=datetime.utcnow)
    dismissed_at: Optional[datetime] = None
    acted_on_at: Optional[datetime] = None
    feedback: Optional[str] = None


class ProactiveSuggestionService:
    """
    Generates personalized, AI-driven suggestions proactively
    """

    SUGGESTION_TEMPLATES = {
        SuggestionType.SKILL_DEVELOPMENT: [
            {
                "title": "Skill Growth Opportunity",
                "body": "Based on your goal of becoming a {target_role}, consider learning {skill}. It's trending in your industry.",
                "actions": [
                    {"label": "Find Courses", "action": "search_courses"},
                    {"label": "Add to Goals", "action": "add_goal"}
                ]
            },
            {
                "title": "Skill Assessment Time",
                "body": "It's been a while since you assessed your {skill} proficiency. Take 5 minutes to reflect on your growth.",
                "actions": [
                    {"label": "Self-Assess", "action": "self_assess"},
                    {"label": "Remind Later", "action": "snooze"}
                ]
            }
        ],
        SuggestionType.CAREER_MOVE: [
            {
                "title": "Career Opportunity Detected",
                "body": "Based on market trends and your profile, now might be a good time to explore {opportunity}.",
                "actions": [
                    {"label": "Explore", "action": "view_opportunity"},
                    {"label": "Not Now", "action": "dismiss"}
                ]
            }
        ],
        SuggestionType.NETWORKING: [
            {
                "title": "Networking Reminder",
                "body": "You haven't connected with your network recently. Consider reaching out to someone in your field.",
                "actions": [
                    {"label": "Get Suggestions", "action": "suggest_contacts"},
                    {"label": "Skip", "action": "dismiss"}
                ]
            }
        ],
        SuggestionType.DECISION_PROMPT: [
            {
                "title": "Time to Decide?",
                "body": "You've been pondering a decision for a while. Would you like help analyzing it?",
                "actions": [
                    {"label": "Analyze Now", "action": "start_analysis"},
                    {"label": "I Decided", "action": "record_outcome"}
                ]
            }
        ],
        SuggestionType.REFLECTION: [
            {
                "title": "Weekly Reflection",
                "body": "Take a moment to reflect on your decisions this week. What went well? What could improve?",
                "actions": [
                    {"label": "Start Reflection", "action": "open_journal"},
                    {"label": "Later", "action": "snooze"}
                ]
            },
            {
                "title": "Celebrate Your Progress",
                "body": "You've made {decision_count} decisions this month. That takes courage! How do you feel about them?",
                "actions": [
                    {"label": "Review Decisions", "action": "view_history"},
                    {"label": "Share Thoughts", "action": "open_journal"}
                ]
            }
        ],
        SuggestionType.BIAS_AWARENESS: [
            {
                "title": "Bias Pattern Noticed",
                "body": "We've detected a pattern of {bias_type} in your recent decisions. Would you like to explore this?",
                "actions": [
                    {"label": "Learn More", "action": "view_bias_info"},
                    {"label": "Mindful Mode", "action": "enable_mindful"}
                ]
            }
        ],
        SuggestionType.GOAL_NUDGE: [
            {
                "title": "Goal Check-In",
                "body": "Your goal '{goal_title}' might need attention. You're at {progress}% - let's keep momentum!",
                "actions": [
                    {"label": "Update Progress", "action": "update_goal"},
                    {"label": "Adjust Goal", "action": "edit_goal"}
                ]
            }
        ],
        SuggestionType.WELLNESS: [
            {
                "title": "Stress Check",
                "body": "Your recent decisions seem to involve high stress. Remember to take breaks and practice self-care.",
                "actions": [
                    {"label": "Breathing Exercise", "action": "open_wellness"},
                    {"label": "I'm Fine", "action": "dismiss"}
                ]
            }
        ],
        SuggestionType.MARKET_INSIGHT: [
            {
                "title": "Market Update",
                "body": "{insight} This could affect your career plans.",
                "actions": [
                    {"label": "Read More", "action": "view_insight"},
                    {"label": "Got It", "action": "dismiss"}
                ]
            }
        ]
    }

    def __init__(self):
        self.user_contexts: Dict[str, UserContext] = {}
        self.suggestions: Dict[str, List[Suggestion]] = {}
        self.suggestion_history: Dict[str, List[str]] = {}

    def update_user_context(
        self,
        user_id: str,
        **kwargs
    ) -> UserContext:
        """Update or create user context for personalization"""
        if user_id in self.user_contexts:
            context = self.user_contexts[user_id]
            for key, value in kwargs.items():
                if hasattr(context, key):
                    setattr(context, key, value)
        else:
            context = UserContext(user_id=user_id, **kwargs)
            self.user_contexts[user_id] = context

        context.last_active = datetime.utcnow()
        return context

    def generate_suggestions(
        self,
        user_id: str,
        max_suggestions: int = 3
    ) -> List[Dict[str, Any]]:
        """Generate personalized suggestions for a user"""
        context = self.user_contexts.get(user_id, UserContext(user_id=user_id))

        candidates = []

        if context.pending_decisions > 0:
            candidates.extend(self._generate_decision_suggestions(context))

        if context.target_role and context.skills:
            candidates.extend(self._generate_skill_suggestions(context))

        if context.goals:
            candidates.extend(self._generate_goal_suggestions(context))

        if context.bias_patterns:
            candidates.extend(self._generate_bias_suggestions(context))

        days_inactive = (datetime.utcnow() - context.last_active).days
        if days_inactive > 3:
            candidates.extend(self._generate_engagement_suggestions(context))

        candidates.extend(self._generate_reflection_suggestions(context))

        candidates.sort(key=lambda x: x.relevance_score, reverse=True)
        selected = candidates[:max_suggestions]

        if user_id not in self.suggestions:
            self.suggestions[user_id] = []
        self.suggestions[user_id].extend(selected)

        return [self._to_dict(s) for s in selected]

    def _generate_decision_suggestions(
        self,
        context: UserContext
    ) -> List[Suggestion]:
        """Generate suggestions related to pending decisions"""
        suggestions = []

        template = self.SUGGESTION_TEMPLATES[SuggestionType.DECISION_PROMPT][0]
        suggestion = self._create_suggestion(
            context.user_id,
            SuggestionType.DECISION_PROMPT,
            template,
            {"pending": context.pending_decisions},
            relevance=0.8,
            priority=SuggestionPriority.HIGH
        )
        suggestions.append(suggestion)

        return suggestions

    def _generate_skill_suggestions(
        self,
        context: UserContext
    ) -> List[Suggestion]:
        """Generate skill development suggestions"""
        suggestions = []

        trending_skills = ["AI/ML", "Cloud Computing", "Data Analysis", "Leadership"]
        skill_to_suggest = random.choice([
            s for s in trending_skills if s not in context.skills
        ] or trending_skills)

        template = self.SUGGESTION_TEMPLATES[SuggestionType.SKILL_DEVELOPMENT][0]
        suggestion = self._create_suggestion(
            context.user_id,
            SuggestionType.SKILL_DEVELOPMENT,
            template,
            {"target_role": context.target_role, "skill": skill_to_suggest},
            relevance=0.7
        )
        suggestions.append(suggestion)

        return suggestions

    def _generate_goal_suggestions(
        self,
        context: UserContext
    ) -> List[Suggestion]:
        """Generate goal-related suggestions"""
        suggestions = []

        if context.goals:
            goal = context.goals[0]
            template = self.SUGGESTION_TEMPLATES[SuggestionType.GOAL_NUDGE][0]
            suggestion = self._create_suggestion(
                context.user_id,
                SuggestionType.GOAL_NUDGE,
                template,
                {"goal_title": goal, "progress": random.randint(20, 60)},
                relevance=0.75
            )
            suggestions.append(suggestion)

        return suggestions

    def _generate_bias_suggestions(
        self,
        context: UserContext
    ) -> List[Suggestion]:
        """Generate bias awareness suggestions"""
        suggestions = []

        if context.bias_patterns:
            bias = context.bias_patterns[0].replace("_", " ")
            template = self.SUGGESTION_TEMPLATES[SuggestionType.BIAS_AWARENESS][0]
            suggestion = self._create_suggestion(
                context.user_id,
                SuggestionType.BIAS_AWARENESS,
                template,
                {"bias_type": bias},
                relevance=0.85,
                priority=SuggestionPriority.HIGH
            )
            suggestions.append(suggestion)

        return suggestions

    def _generate_engagement_suggestions(
        self,
        context: UserContext
    ) -> List[Suggestion]:
        """Generate re-engagement suggestions"""
        suggestions = []

        template = self.SUGGESTION_TEMPLATES[SuggestionType.REFLECTION][0]
        suggestion = self._create_suggestion(
            context.user_id,
            SuggestionType.REFLECTION,
            template,
            {},
            relevance=0.6,
            priority=SuggestionPriority.MEDIUM
        )
        suggestions.append(suggestion)

        return suggestions

    def _generate_reflection_suggestions(
        self,
        context: UserContext
    ) -> List[Suggestion]:
        """Generate reflection suggestions"""
        suggestions = []

        if context.recent_decisions > 3:
            template = self.SUGGESTION_TEMPLATES[SuggestionType.REFLECTION][1]
            suggestion = self._create_suggestion(
                context.user_id,
                SuggestionType.REFLECTION,
                template,
                {"decision_count": context.recent_decisions},
                relevance=0.65
            )
            suggestions.append(suggestion)

        return suggestions

    def _create_suggestion(
        self,
        user_id: str,
        suggestion_type: SuggestionType,
        template: Dict[str, Any],
        context_vars: Dict[str, Any],
        relevance: float = 0.5,
        priority: SuggestionPriority = SuggestionPriority.MEDIUM
    ) -> Suggestion:
        """Create a suggestion from a template"""
        suggestion_id = hashlib.sha256(
            f"{user_id}{suggestion_type.value}{datetime.utcnow().timestamp()}".encode()
        ).hexdigest()[:12]

        title = template["title"]
        body = template["body"]
        for key, value in context_vars.items():
            body = body.replace(f"{{{key}}}", str(value))
            title = title.replace(f"{{{key}}}", str(value))

        return Suggestion(
            id=suggestion_id,
            user_id=user_id,
            type=suggestion_type,
            title=title,
            body=body,
            reasoning=f"Based on your profile and recent activity",
            actions=template.get("actions", []),
            priority=priority,
            relevance_score=relevance,
            expires_at=datetime.utcnow() + timedelta(days=7)
        )

    def _to_dict(self, suggestion: Suggestion) -> Dict[str, Any]:
        """Convert suggestion to dictionary"""
        return {
            "id": suggestion.id,
            "type": suggestion.type.value,
            "title": suggestion.title,
            "body": suggestion.body,
            "reasoning": suggestion.reasoning,
            "actions": suggestion.actions,
            "priority": suggestion.priority.value,
            "relevance_score": round(suggestion.relevance_score, 2),
            "expires_at": suggestion.expires_at.isoformat(),
            "created_at": suggestion.created_at.isoformat()
        }

    def get_active_suggestions(
        self,
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Get active (non-dismissed) suggestions for a user"""
        if user_id not in self.suggestions:
            return self.generate_suggestions(user_id)

        active = [
            s for s in self.suggestions[user_id]
            if s.dismissed_at is None and s.expires_at > datetime.utcnow()
        ]

        if len(active) < 2:
            new_suggestions = self.generate_suggestions(user_id, max_suggestions=2)
            return [self._to_dict(s) for s in active] + new_suggestions

        return [self._to_dict(s) for s in active]

    def dismiss_suggestion(
        self,
        user_id: str,
        suggestion_id: str,
        feedback: str = None
    ) -> bool:
        """Dismiss a suggestion"""
        if user_id not in self.suggestions:
            return False

        for suggestion in self.suggestions[user_id]:
            if suggestion.id == suggestion_id:
                suggestion.dismissed_at = datetime.utcnow()
                suggestion.feedback = feedback
                return True
        return False

    def act_on_suggestion(
        self,
        user_id: str,
        suggestion_id: str,
        action: str
    ) -> Dict[str, Any]:
        """Record that user acted on a suggestion"""
        if user_id not in self.suggestions:
            return {"error": "No suggestions found"}

        for suggestion in self.suggestions[user_id]:
            if suggestion.id == suggestion_id:
                suggestion.acted_on_at = datetime.utcnow()
                return {
                    "acted": True,
                    "action": action,
                    "suggestion_type": suggestion.type.value
                }

        return {"error": "Suggestion not found"}

    def get_suggestion_stats(self, user_id: str) -> Dict[str, Any]:
        """Get suggestion engagement statistics"""
        if user_id not in self.suggestions:
            return {"total": 0, "acted_on": 0, "dismissed": 0}

        all_suggestions = self.suggestions[user_id]
        acted = len([s for s in all_suggestions if s.acted_on_at])
        dismissed = len([s for s in all_suggestions if s.dismissed_at and not s.acted_on_at])

        return {
            "total": len(all_suggestions),
            "acted_on": acted,
            "dismissed": dismissed,
            "engagement_rate": round(acted / max(1, len(all_suggestions)) * 100, 1),
            "type_breakdown": self._get_type_breakdown(all_suggestions)
        }

    def _get_type_breakdown(
        self,
        suggestions: List[Suggestion]
    ) -> Dict[str, int]:
        """Get breakdown of suggestions by type"""
        breakdown = {}
        for s in suggestions:
            type_name = s.type.value
            breakdown[type_name] = breakdown.get(type_name, 0) + 1
        return breakdown


proactive_suggestion_service = ProactiveSuggestionService()
