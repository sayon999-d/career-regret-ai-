import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum


class GoalStatus(Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    PAUSED = "paused"
    ABANDONED = "abandoned"
    OVERDUE = "overdue"


class GoalPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class GoalCategory(Enum):
    SKILLS = "skills"
    CAREER_MOVE = "career_move"
    COMPENSATION = "compensation"
    LEADERSHIP = "leadership"
    NETWORKING = "networking"
    EDUCATION = "education"
    WORK_LIFE = "work_life"
    PERSONAL_BRAND = "personal_brand"


@dataclass
class SubTask:
    id: str
    title: str
    description: str = ""
    completed: bool = False
    due_date: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class ProgressCheckIn:
    id: str
    date: datetime
    progress_pct: float
    notes: str
    blockers: List[str] = field(default_factory=list)
    mood: str = "neutral"


@dataclass
class Goal:
    id: str
    user_id: str
    title: str
    description: str
    category: GoalCategory
    priority: GoalPriority
    status: GoalStatus
    target_date: datetime
    measurable_target: str
    success_criteria: List[str]
    sub_tasks: List[SubTask] = field(default_factory=list)
    check_ins: List[ProgressCheckIn] = field(default_factory=list)
    progress_pct: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)


class GoalTrackingService:
    GOAL_TEMPLATES = {
        GoalCategory.SKILLS: [
            {"title": "Learn {skill}", "measurable": "Complete {course/certification} and build {project}",
             "sub_tasks": ["Research learning resources", "Set up study schedule",
                           "Complete first module", "Build practice project", "Pass certification"]},
            {"title": "Master {technology}", "measurable": "Ship {N} features using {technology}",
             "sub_tasks": ["Read official docs", "Complete tutorial project",
                           "Use in production feature", "Share knowledge with team"]}
        ],
        GoalCategory.CAREER_MOVE: [
            {"title": "Transition to {role}", "measurable": "Receive offer for {role} position",
             "sub_tasks": ["Update resume for target role", "Network with 10 people in field",
                           "Apply to 20 positions", "Prepare for interviews", "Negotiate offer"]},
        ],
        GoalCategory.COMPENSATION: [
            {"title": "Increase compensation by {X}%", "measurable": "Achieve {X}% raise or new offer",
             "sub_tasks": ["Research market rates", "Document achievements",
                           "Schedule review meeting", "Prepare negotiation points", "Practice negotiation"]},
        ],
        GoalCategory.NETWORKING: [
            {"title": "Build professional network in {industry}",
             "measurable": "Connect with {N} professionals",
             "sub_tasks": ["Identify target connections", "Attend 2 industry events",
                           "Schedule 5 coffee chats", "Publish 3 articles/posts"]}
        ],
        GoalCategory.LEADERSHIP: [
            {"title": "Develop leadership skills",
             "measurable": "Mentor {N} people and lead {N} projects",
             "sub_tasks": ["Find a mentee", "Lead a cross-team initiative",
                           "Take a leadership course", "Get 360 feedback"]}
        ]
    }

    def __init__(self):
        self.goals: Dict[str, List[Goal]] = defaultdict(list)
        self.accountability_settings: Dict[str, Dict] = {}

    def create_goal(self, user_id: str, title: str, description: str,
                    category: str, priority: str = "medium",
                    target_date: str = None, measurable_target: str = "",
                    success_criteria: List[str] = None,
                    auto_decompose: bool = True) -> Dict:
        try:
            cat = GoalCategory(category)
        except ValueError:
            cat = GoalCategory.SKILLS
        try:
            pri = GoalPriority(priority)
        except ValueError:
            pri = GoalPriority.MEDIUM

        if target_date:
            td = datetime.fromisoformat(target_date)
        else:
            td = datetime.utcnow() + timedelta(days=90)

        goal = Goal(
            id=str(uuid.uuid4()),
            user_id=user_id,
            title=title,
            description=description,
            category=cat,
            priority=pri,
            status=GoalStatus.ACTIVE,
            target_date=td,
            measurable_target=measurable_target or title,
            success_criteria=success_criteria or [f"Complete: {title}"]
        )

        if auto_decompose:
            goal.sub_tasks = self._decompose_goal(goal)

        self.goals[user_id].append(goal)

        return {
            "goal_id": goal.id,
            "title": goal.title,
            "category": cat.value,
            "priority": pri.value,
            "target_date": td.isoformat(),
            "sub_tasks": [{"id": st.id, "title": st.title} for st in goal.sub_tasks],
            "days_remaining": (td - datetime.utcnow()).days,
            "status": "active"
        }

    def _decompose_goal(self, goal: Goal) -> List[SubTask]:
        templates = self.GOAL_TEMPLATES.get(goal.category, [])
        sub_tasks = []

        if templates:
            template = templates[0]
            for i, task_title in enumerate(template.get("sub_tasks", [])):
                days_per_task = max(1, (goal.target_date - datetime.utcnow()).days // max(len(template.get("sub_tasks", [])), 1))
                sub_tasks.append(SubTask(
                    id=str(uuid.uuid4()),
                    title=task_title,
                    due_date=datetime.utcnow() + timedelta(days=days_per_task * (i + 1))
                ))
        else:
            total_days = max(1, (goal.target_date - datetime.utcnow()).days)
            generic_steps = [
                "Research and planning",
                "Set up resources and tools",
                "Begin execution / first milestone",
                "Mid-point review and adjustment",
                "Final push and completion"
            ]
            for i, step in enumerate(generic_steps):
                sub_tasks.append(SubTask(
                    id=str(uuid.uuid4()),
                    title=f"{step} — {goal.title}",
                    due_date=datetime.utcnow() + timedelta(days=int(total_days * (i + 1) / len(generic_steps)))
                ))

        return sub_tasks

    def complete_subtask(self, user_id: str, goal_id: str, subtask_id: str) -> Dict:
        goal = self._find_goal(user_id, goal_id)
        if not goal:
            return {"error": "Goal not found"}

        for st in goal.sub_tasks:
            if st.id == subtask_id:
                st.completed = True
                st.completed_at = datetime.utcnow()
                break
        else:
            return {"error": "Sub-task not found"}

        completed = sum(1 for st in goal.sub_tasks if st.completed)
        total = len(goal.sub_tasks) if goal.sub_tasks else 1
        goal.progress_pct = round((completed / total) * 100, 1)
        if goal.progress_pct >= 100:
            goal.status = GoalStatus.COMPLETED
            goal.completed_at = datetime.utcnow()

        return {
            "goal_id": goal_id,
            "subtask_id": subtask_id,
            "progress_pct": goal.progress_pct,
            "status": goal.status.value,
            "completed_count": completed,
            "total_count": total
        }

    def check_in(self, user_id: str, goal_id: str, progress_pct: float,
                 notes: str = "", blockers: List[str] = None,
                 mood: str = "neutral") -> Dict:
        goal = self._find_goal(user_id, goal_id)
        if not goal:
            return {"error": "Goal not found"}

        checkin = ProgressCheckIn(
            id=str(uuid.uuid4()),
            date=datetime.utcnow(),
            progress_pct=max(0, min(100, progress_pct)),
            notes=notes,
            blockers=blockers or [],
            mood=mood
        )
        goal.check_ins.append(checkin)
        goal.progress_pct = progress_pct

        if datetime.utcnow() > goal.target_date and goal.status == GoalStatus.ACTIVE:
            goal.status = GoalStatus.OVERDUE

        report = self._generate_checkin_feedback(goal, checkin)

        return {
            "checkin_id": checkin.id,
            "goal_id": goal_id,
            "progress_pct": progress_pct,
            "status": goal.status.value,
            "days_remaining": max(0, (goal.target_date - datetime.utcnow()).days),
            "feedback": report,
            "on_track": self._is_on_track(goal)
        }

    def _generate_checkin_feedback(self, goal: Goal, checkin: ProgressCheckIn) -> Dict:
        days_elapsed = (datetime.utcnow() - goal.created_at).days
        days_total = max(1, (goal.target_date - goal.created_at).days)
        time_pct = (days_elapsed / days_total) * 100
        on_track = goal.progress_pct >= time_pct * 0.8

        feedback = {
            "on_track": on_track,
            "time_elapsed_pct": round(time_pct, 1),
            "progress_vs_time": round(goal.progress_pct - time_pct, 1),
        }

        if on_track:
            feedback["message"] = "Great progress! You're on track to meet your goal."
            if goal.progress_pct > time_pct + 10:
                feedback["message"] = "Excellent! You're ahead of schedule. Keep up the momentum!"
        else:
            gap = time_pct - goal.progress_pct
            if gap > 30:
                feedback["message"] = (
                    f"You're significantly behind schedule ({gap:.0f}% gap). "
                    "Consider breaking down the remaining work into smaller daily tasks "
                    "or adjusting the timeline."
                )
            else:
                feedback["message"] = (
                    f"You're slightly behind ({gap:.0f}% gap). "
                    "A focused effort this week can get you back on track."
                )

        if checkin.blockers:
            feedback["blocker_advice"] = [
                f"Blocker: '{b}' — Consider: Can you delegate, eliminate, or find a workaround?"
                for b in checkin.blockers[:3]
            ]

        if checkin.mood == "stuck":
            feedback["mood_advice"] = (
                "Feeling stuck is normal during goal pursuit. Try: "
                "1) Break the next step into 15-minute chunks, "
                "2) Ask someone who's done this for advice, "
                "3) Take a short break and return fresh."
            )

        return feedback

    def _is_on_track(self, goal: Goal) -> bool:
        days_elapsed = (datetime.utcnow() - goal.created_at).days
        days_total = max(1, (goal.target_date - goal.created_at).days)
        expected = (days_elapsed / days_total) * 100
        return goal.progress_pct >= expected * 0.8

    def get_goals(self, user_id: str, status: str = None,
                  category: str = None) -> List[Dict]:
        goals = self.goals.get(user_id, [])
        if status:
            goals = [g for g in goals if g.status.value == status]
        if category:
            goals = [g for g in goals if g.category.value == category]

        return [{
            "id": g.id,
            "title": g.title,
            "description": g.description,
            "category": g.category.value,
            "priority": g.priority.value,
            "status": g.status.value,
            "progress_pct": g.progress_pct,
            "target_date": g.target_date.isoformat(),
            "days_remaining": max(0, (g.target_date - datetime.utcnow()).days),
            "on_track": self._is_on_track(g),
            "sub_tasks_completed": sum(1 for st in g.sub_tasks if st.completed),
            "sub_tasks_total": len(g.sub_tasks),
            "check_ins": len(g.check_ins),
            "created_at": g.created_at.isoformat()
        } for g in goals]

    def get_goal_detail(self, user_id: str, goal_id: str) -> Dict:
        goal = self._find_goal(user_id, goal_id)
        if not goal:
            return {"error": "Goal not found"}

        return {
            "id": goal.id,
            "title": goal.title,
            "description": goal.description,
            "category": goal.category.value,
            "priority": goal.priority.value,
            "status": goal.status.value,
            "progress_pct": goal.progress_pct,
            "target_date": goal.target_date.isoformat(),
            "measurable_target": goal.measurable_target,
            "success_criteria": goal.success_criteria,
            "on_track": self._is_on_track(goal),
            "days_remaining": max(0, (goal.target_date - datetime.utcnow()).days),
            "sub_tasks": [{
                "id": st.id,
                "title": st.title,
                "completed": st.completed,
                "due_date": st.due_date.isoformat() if st.due_date else None,
                "completed_at": st.completed_at.isoformat() if st.completed_at else None
            } for st in goal.sub_tasks],
            "check_ins": [{
                "id": ci.id,
                "date": ci.date.isoformat(),
                "progress_pct": ci.progress_pct,
                "notes": ci.notes,
                "blockers": ci.blockers,
                "mood": ci.mood
            } for ci in goal.check_ins],
            "created_at": goal.created_at.isoformat()
        }

    def update_goal_status(self, user_id: str, goal_id: str,
                           status: str) -> Dict:
        goal = self._find_goal(user_id, goal_id)
        if not goal:
            return {"error": "Goal not found"}

        try:
            new_status = GoalStatus(status)
        except ValueError:
            return {"error": f"Invalid status: {status}"}

        goal.status = new_status
        if new_status == GoalStatus.COMPLETED:
            goal.completed_at = datetime.utcnow()
            goal.progress_pct = 100

        return {
            "goal_id": goal_id,
            "status": new_status.value,
            "updated": True
        }

    def get_accountability_report(self, user_id: str) -> Dict:
        goals = self.goals.get(user_id, [])
        active = [g for g in goals if g.status in (GoalStatus.ACTIVE, GoalStatus.OVERDUE)]
        completed_recently = [g for g in goals if g.status == GoalStatus.COMPLETED
                              and g.completed_at
                              and (datetime.utcnow() - g.completed_at).days <= 7]

        overdue = [g for g in goals if g.status == GoalStatus.OVERDUE]
        on_track = [g for g in active if self._is_on_track(g)]
        behind = [g for g in active if not self._is_on_track(g)]

        upcoming_tasks = []
        now = datetime.utcnow()
        week_ahead = now + timedelta(days=7)
        for g in active:
            for st in g.sub_tasks:
                if not st.completed and st.due_date and now <= st.due_date <= week_ahead:
                    upcoming_tasks.append({
                        "goal": g.title,
                        "task": st.title,
                        "due": st.due_date.isoformat(),
                        "days_until": (st.due_date - now).days
                    })

        return {
            "report_date": now.isoformat(),
            "summary": {
                "total_active": len(active),
                "on_track": len(on_track),
                "behind": len(behind),
                "overdue": len(overdue),
                "completed_this_week": len(completed_recently)
            },
            "on_track_goals": [{"id": g.id, "title": g.title, "progress": g.progress_pct}
                               for g in on_track],
            "behind_goals": [{"id": g.id, "title": g.title, "progress": g.progress_pct,
                              "gap": round((datetime.utcnow() - g.created_at).days /
                                           max(1, (g.target_date - g.created_at).days) * 100 - g.progress_pct)}
                             for g in behind],
            "overdue_goals": [{"id": g.id, "title": g.title, "days_overdue": (now - g.target_date).days}
                              for g in overdue],
            "completed_this_week": [{"id": g.id, "title": g.title} for g in completed_recently],
            "upcoming_tasks": upcoming_tasks[:10],
            "overall_health": "excellent" if len(on_track) > len(behind) * 2
                             else "good" if len(on_track) >= len(behind)
                             else "needs_attention",
            "motivation": self._get_motivation(active, completed_recently)
        }

    def _get_motivation(self, active: List[Goal], completed: List[Goal]) -> str:
        if completed:
            return f"Great work this week! You completed {len(completed)} goal(s). Keep the momentum going."
        if active:
            closest = min(active, key=lambda g: g.target_date)
            days = (closest.target_date - datetime.utcnow()).days
            return f"Your nearest deadline is in {days} days for '{closest.title}'. Focus on progress, not perfection."
        return "Set your first goal to start building career momentum. Small steps lead to big changes."

    def get_templates(self, category: str = None) -> List[Dict]:
        templates = []
        for cat, cat_templates in self.GOAL_TEMPLATES.items():
            if category and cat.value != category:
                continue
            for t in cat_templates:
                templates.append({
                    "category": cat.value,
                    "title": t["title"],
                    "measurable": t["measurable"],
                    "sub_task_count": len(t.get("sub_tasks", []))
                })
        return templates

    def _find_goal(self, user_id: str, goal_id: str) -> Optional[Goal]:
        for goal in self.goals.get(user_id, []):
            if goal.id == goal_id:
                return goal
        return None


goal_tracking_service = GoalTrackingService()
