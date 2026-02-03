from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import statistics
import math

@dataclass
class PredictionAccuracyMetrics:
    total_predictions: int
    correct_predictions: int
    accuracy_percentage: float
    average_error: float
    error_trend: str
    by_decision_type: Dict[str, Dict]

@dataclass
class BiasPatternAnalysis:
    most_common_biases: List[Dict]
    bias_frequency_over_time: List[Dict]
    improvement_rate: float
    trigger_patterns: List[Dict]
    recommendations: List[str]

@dataclass
class CareerGoal:
    id: str
    user_id: str
    title: str
    description: str
    target_date: Optional[datetime]
    category: str
    progress: float
    milestones: List[Dict]
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)

@dataclass
class GoalProgress:
    goal_id: str
    current_progress: float
    milestones_completed: int
    milestones_total: int
    on_track: bool
    days_remaining: Optional[int]
    next_action: str

class AdvancedAnalyticsService:
    def __init__(self):
        self.prediction_history: Dict[str, List[Dict]] = {}
        self.bias_history: Dict[str, List[Dict]] = {}
        self.decision_timeline: Dict[str, List[Dict]] = {}
        self.user_goals: Dict[str, List[CareerGoal]] = {}
        self.goal_updates: Dict[str, List[Dict]] = {}

    def record_prediction(
        self,
        user_id: str,
        decision_id: str,
        decision_type: str,
        predicted_regret: float,
        factors: List[str] = None
    ):
        if user_id not in self.prediction_history:
            self.prediction_history[user_id] = []

        self.prediction_history[user_id].append({
            "decision_id": decision_id,
            "decision_type": decision_type,
            "predicted_regret": predicted_regret,
            "actual_regret": None,
            "factors": factors or [],
            "recorded_at": datetime.utcnow().isoformat()
        })

    def record_actual_outcome(
        self,
        user_id: str,
        decision_id: str,
        actual_regret: float,
        satisfaction: float
    ):
        if user_id not in self.prediction_history:
            return False

        for pred in self.prediction_history[user_id]:
            if pred["decision_id"] == decision_id:
                pred["actual_regret"] = actual_regret
                pred["satisfaction"] = satisfaction
                pred["outcome_recorded_at"] = datetime.utcnow().isoformat()
                return True
        return False

    def get_prediction_accuracy(self, user_id: str) -> PredictionAccuracyMetrics:
        if user_id not in self.prediction_history:
            return PredictionAccuracyMetrics(
                total_predictions=0,
                correct_predictions=0,
                accuracy_percentage=0,
                average_error=0,
                error_trend="no_data",
                by_decision_type={}
            )

        predictions = [
            p for p in self.prediction_history[user_id]
            if p.get("actual_regret") is not None
        ]

        if not predictions:
            return PredictionAccuracyMetrics(
                total_predictions=len(self.prediction_history[user_id]),
                correct_predictions=0,
                accuracy_percentage=0,
                average_error=0,
                error_trend="pending",
                by_decision_type={}
            )

        errors = []
        correct = 0
        by_type = defaultdict(lambda: {"count": 0, "errors": []})

        for p in predictions:
            error = abs(p["predicted_regret"] - p["actual_regret"])
            errors.append(error)

            if error <= 20:
                correct += 1

            by_type[p["decision_type"]]["count"] += 1
            by_type[p["decision_type"]]["errors"].append(error)

        avg_error = statistics.mean(errors)
        accuracy = (correct / len(predictions)) * 100

        if len(errors) > 5:
            first_half = errors[:len(errors)//2]
            second_half = errors[len(errors)//2:]
            if statistics.mean(second_half) < statistics.mean(first_half):
                trend = "improving"
            elif statistics.mean(second_half) > statistics.mean(first_half):
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        by_decision_type = {}
        for dt, data in by_type.items():
            by_decision_type[dt] = {
                "count": data["count"],
                "average_error": statistics.mean(data["errors"]),
                "accuracy": (len([e for e in data["errors"] if e <= 20]) / data["count"]) * 100
            }

        return PredictionAccuracyMetrics(
            total_predictions=len(self.prediction_history[user_id]),
            correct_predictions=correct,
            accuracy_percentage=round(accuracy, 1),
            average_error=round(avg_error, 1),
            error_trend=trend,
            by_decision_type=by_decision_type
        )

    def record_bias_detection(
        self,
        user_id: str,
        bias_type: str,
        context: str,
        intervention_accepted: bool
    ):
        if user_id not in self.bias_history:
            self.bias_history[user_id] = []

        self.bias_history[user_id].append({
            "bias_type": bias_type,
            "context": context[:200],
            "intervention_accepted": intervention_accepted,
            "detected_at": datetime.utcnow().isoformat()
        })

    def get_bias_pattern_analysis(self, user_id: str) -> BiasPatternAnalysis:
        if user_id not in self.bias_history or not self.bias_history[user_id]:
            return BiasPatternAnalysis(
                most_common_biases=[],
                bias_frequency_over_time=[],
                improvement_rate=0,
                trigger_patterns=[],
                recommendations=["Start journaling to track your decision-making patterns"]
            )

        history = self.bias_history[user_id]

        bias_counts = defaultdict(int)
        for entry in history:
            bias_counts[entry["bias_type"]] += 1

        sorted_biases = sorted(bias_counts.items(), key=lambda x: x[1], reverse=True)
        most_common = [
            {"bias_type": bt, "count": count, "percentage": (count / len(history)) * 100}
            for bt, count in sorted_biases[:5]
        ]

        weekly_data = defaultdict(lambda: defaultdict(int))
        for entry in history:
            week = datetime.fromisoformat(entry["detected_at"]).strftime("%Y-W%W")
            weekly_data[week][entry["bias_type"]] += 1

        frequency_over_time = [
            {"week": week, "biases": dict(biases)}
            for week, biases in sorted(weekly_data.items())
        ]

        accepted = len([e for e in history if e.get("intervention_accepted")])
        improvement_rate = (accepted / len(history)) * 100 if history else 0

        recommendations = self._generate_bias_recommendations(most_common, improvement_rate)

        return BiasPatternAnalysis(
            most_common_biases=most_common,
            bias_frequency_over_time=frequency_over_time[-12:],
            improvement_rate=round(improvement_rate, 1),
            trigger_patterns=[],
            recommendations=recommendations
        )

    def _generate_bias_recommendations(self, common_biases: List[Dict], improvement_rate: float) -> List[str]:
        recommendations = []

        if not common_biases:
            return ["Continue journaling to identify patterns in your decision-making"]

        top_bias = common_biases[0]["bias_type"]

        bias_tips = {
            "sunk_cost": "Practice asking 'Would I start this today?' before continuing any commitment",
            "loss_aversion": "Reframe decisions by focusing on potential gains, not just avoiding losses",
            "confirmation_bias": "Actively seek out opinions that disagree with your initial view",
            "overconfidence": "Before major decisions, list 3 things that could go wrong",
            "status_quo": "Schedule quarterly reviews to question your current path",
            "anchoring": "Research multiple data points before forming opinions on numbers"
        }

        if top_bias in bias_tips:
            recommendations.append(f"Focus area: {bias_tips[top_bias]}")

        if improvement_rate < 30:
            recommendations.append("Try pausing for 24 hours when a bias is detected before deciding")
        elif improvement_rate > 70:
            recommendations.append("Great progress! You're responding well to bias interventions")

        if len(common_biases) > 3:
            recommendations.append("You show diverse bias patterns - consider working with a mentor or coach")

        return recommendations

    def add_to_decision_timeline(
        self,
        user_id: str,
        decision_id: str,
        decision_type: str,
        title: str,
        description: str,
        outcome: str = None
    ):
        if user_id not in self.decision_timeline:
            self.decision_timeline[user_id] = []

        self.decision_timeline[user_id].append({
            "decision_id": decision_id,
            "decision_type": decision_type,
            "title": title,
            "description": description[:500],
            "outcome": outcome,
            "timestamp": datetime.utcnow().isoformat()
        })

    def get_decision_timeline(
        self,
        user_id: str,
        limit: int = 50,
        decision_type: str = None
    ) -> List[Dict]:
        if user_id not in self.decision_timeline:
            return []

        timeline = self.decision_timeline[user_id]

        if decision_type:
            timeline = [d for d in timeline if d["decision_type"] == decision_type]

        timeline.sort(key=lambda x: x["timestamp"], reverse=True)
        return timeline[:limit]

    def create_career_goal(
        self,
        user_id: str,
        title: str,
        description: str,
        category: str,
        target_date: datetime = None,
        milestones: List[Dict] = None
    ) -> CareerGoal:
        goal = CareerGoal(
            id=f"goal_{user_id}_{datetime.utcnow().timestamp()}",
            user_id=user_id,
            title=title,
            description=description,
            target_date=target_date,
            category=category,
            progress=0,
            milestones=milestones or []
        )

        if user_id not in self.user_goals:
            self.user_goals[user_id] = []
        self.user_goals[user_id].append(goal)

        return goal

    def update_goal_progress(
        self,
        user_id: str,
        goal_id: str,
        progress: float = None,
        milestone_completed: str = None
    ) -> Optional[GoalProgress]:
        if user_id not in self.user_goals:
            return None

        goal = None
        for g in self.user_goals[user_id]:
            if g.id == goal_id:
                goal = g
                break

        if not goal:
            return None

        if progress is not None:
            goal.progress = min(100, max(0, progress))

        if milestone_completed:
            for m in goal.milestones:
                if m.get("id") == milestone_completed:
                    m["completed"] = True
                    m["completed_at"] = datetime.utcnow().isoformat()

        goal.last_updated = datetime.utcnow()

        if goal_id not in self.goal_updates:
            self.goal_updates[goal_id] = []
        self.goal_updates[goal_id].append({
            "progress": goal.progress,
            "updated_at": datetime.utcnow().isoformat()
        })

        return self._calculate_goal_progress(goal)

    def _calculate_goal_progress(self, goal: CareerGoal) -> GoalProgress:
        completed = len([m for m in goal.milestones if m.get("completed")])
        total = len(goal.milestones)

        days_remaining = None
        on_track = True

        if goal.target_date:
            days_remaining = (goal.target_date - datetime.utcnow()).days
            if days_remaining > 0:
                expected_progress = ((datetime.utcnow() - goal.created_at).days /
                                   (goal.target_date - goal.created_at).days) * 100
                on_track = goal.progress >= expected_progress * 0.8

        next_actions = {
            "career_growth": "Review your skill development plan",
            "salary": "Research market rates and prepare negotiation points",
            "leadership": "Seek out a mentorship opportunity",
            "skills": "Complete the next learning module",
            "network": "Attend an industry event or reach out to a connection"
        }

        return GoalProgress(
            goal_id=goal.id,
            current_progress=goal.progress,
            milestones_completed=completed,
            milestones_total=total,
            on_track=on_track,
            days_remaining=days_remaining,
            next_action=next_actions.get(goal.category, "Review your goal progress")
        )

    def get_user_goals(self, user_id: str, category: str = None) -> List[Dict]:
        if user_id not in self.user_goals:
            return []

        goals = self.user_goals[user_id]

        if category:
            goals = [g for g in goals if g.category == category]

        return [
            {
                "id": g.id,
                "title": g.title,
                "description": g.description,
                "category": g.category,
                "progress": g.progress,
                "target_date": g.target_date.isoformat() if g.target_date else None,
                "milestones": g.milestones,
                "created_at": g.created_at.isoformat(),
                **self._calculate_goal_progress(g).__dict__
            }
            for g in goals
        ]

    def get_analytics_dashboard(self, user_id: str) -> Dict[str, Any]:
        prediction_metrics = self.get_prediction_accuracy(user_id)
        bias_analysis = self.get_bias_pattern_analysis(user_id)
        goals = self.get_user_goals(user_id)
        timeline = self.get_decision_timeline(user_id, limit=10)

        active_goals = len([g for g in goals if g.get("progress", 0) < 100])
        on_track_goals = len([g for g in goals if g.get("on_track", False)])

        return {
            "user_id": user_id,
            "prediction_accuracy": {
                "total": prediction_metrics.total_predictions,
                "accuracy": prediction_metrics.accuracy_percentage,
                "average_error": prediction_metrics.average_error,
                "trend": prediction_metrics.error_trend
            },
            "bias_insights": {
                "most_common": bias_analysis.most_common_biases[:3],
                "improvement_rate": bias_analysis.improvement_rate,
                "recommendations": bias_analysis.recommendations[:2]
            },
            "goals_summary": {
                "total": len(goals),
                "active": active_goals,
                "on_track": on_track_goals,
                "completion_rate": ((len(goals) - active_goals) / max(1, len(goals))) * 100
            },
            "recent_decisions": len(timeline),
            "decision_types": list(set(d["decision_type"] for d in timeline)) if timeline else [],
            "generated_at": datetime.utcnow().isoformat()
        }

    def export_analytics_data(self, user_id: str) -> Dict[str, Any]:
        return {
            "user_id": user_id,
            "export_date": datetime.utcnow().isoformat(),
            "predictions": self.prediction_history.get(user_id, []),
            "bias_history": self.bias_history.get(user_id, []),
            "decision_timeline": self.decision_timeline.get(user_id, []),
            "goals": self.get_user_goals(user_id),
            "summary": self.get_analytics_dashboard(user_id)
        }

advanced_analytics_service = AdvancedAnalyticsService()
