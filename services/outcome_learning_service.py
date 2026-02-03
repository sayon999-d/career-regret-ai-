from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

@dataclass
class DecisionOutcome:
    decision_id: str
    user_id: str
    predicted_regret: float
    actual_regret: float
    satisfaction_score: float
    outcome_notes: str
    surprises: List[str]
    lessons_learned: str
    recorded_at: datetime = field(default_factory=datetime.utcnow)
    time_since_decision_days: int = 0

@dataclass
class UserLearningProfile:
    user_id: str
    total_outcomes: int = 0
    prediction_accuracy: float = 0.0
    avg_prediction_error: float = 0.0
    optimism_bias: float = 0.0
    risk_perception_accuracy: float = 0.0
    best_decision_types: List[str] = field(default_factory=list)
    worst_decision_types: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)

class OutcomeLearningService:
    def __init__(self):
        self.outcomes: Dict[str, List[DecisionOutcome]] = {}
        self.learning_profiles: Dict[str, UserLearningProfile] = {}
        self.prediction_adjustments: Dict[str, Dict[str, float]] = {}

    def record_outcome(
        self,
        decision_id: str,
        user_id: str,
        predicted_regret: float,
        actual_regret: float,
        satisfaction_score: float,
        outcome_notes: str = "",
        surprises: List[str] = None,
        lessons_learned: str = "",
        decision_date: datetime = None
    ) -> Dict[str, Any]:
        time_since = 0
        if decision_date:
            time_since = (datetime.utcnow() - decision_date).days

        outcome = DecisionOutcome(
            decision_id=decision_id,
            user_id=user_id,
            predicted_regret=predicted_regret,
            actual_regret=actual_regret,
            satisfaction_score=satisfaction_score,
            outcome_notes=outcome_notes,
            surprises=surprises or [],
            lessons_learned=lessons_learned,
            time_since_decision_days=time_since
        )

        if user_id not in self.outcomes:
            self.outcomes[user_id] = []
        self.outcomes[user_id].append(outcome)

        self._update_learning_profile(user_id)

        prediction_error = abs(predicted_regret - actual_regret)
        accuracy = max(0, 100 - prediction_error)

        return {
            "success": True,
            "prediction_error": prediction_error,
            "accuracy": accuracy,
            "message": self._generate_outcome_message(predicted_regret, actual_regret, satisfaction_score)
        }

    def _generate_outcome_message(self, predicted: float, actual: float, satisfaction: float) -> str:
        error = predicted - actual

        if abs(error) < 10:
            accuracy_msg = "Our prediction was quite accurate!"
        elif error > 20:
            accuracy_msg = "You experienced less regret than predicted - that's great!"
        elif error < -20:
            accuracy_msg = "The outcome was more challenging than expected. We'll learn from this."
        else:
            accuracy_msg = "The prediction was reasonably close to reality."

        if satisfaction >= 80:
            satisfaction_msg = "You seem very satisfied with this decision."
        elif satisfaction >= 60:
            satisfaction_msg = "Overall, this decision worked out well for you."
        elif satisfaction >= 40:
            satisfaction_msg = "This decision had mixed results."
        else:
            satisfaction_msg = "This decision was challenging, but every experience teaches us something valuable."

        return f"{accuracy_msg} {satisfaction_msg}"

    def _update_learning_profile(self, user_id: str):
        if user_id not in self.outcomes or not self.outcomes[user_id]:
            return

        outcomes = self.outcomes[user_id]

        if user_id not in self.learning_profiles:
            self.learning_profiles[user_id] = UserLearningProfile(user_id=user_id)

        profile = self.learning_profiles[user_id]
        profile.total_outcomes = len(outcomes)

        errors = [abs(o.predicted_regret - o.actual_regret) for o in outcomes]
        profile.avg_prediction_error = sum(errors) / len(errors)
        profile.prediction_accuracy = max(0, 100 - profile.avg_prediction_error)

        biases = [o.predicted_regret - o.actual_regret for o in outcomes]
        profile.optimism_bias = sum(biases) / len(biases)

        profile.patterns = self._detect_patterns(outcomes)
        profile.last_updated = datetime.utcnow()

        if user_id not in self.prediction_adjustments:
            self.prediction_adjustments[user_id] = {}

        self.prediction_adjustments[user_id]["bias_correction"] = -profile.optimism_bias * 0.5

    def _detect_patterns(self, outcomes: List[DecisionOutcome]) -> List[str]:
        patterns = []

        if len(outcomes) < 3:
            return ["Keep tracking outcomes to discover your decision patterns."]

        avg_error = sum(abs(o.predicted_regret - o.actual_regret) for o in outcomes) / len(outcomes)
        if avg_error < 15:
            patterns.append("You have good self-awareness - predictions closely match your actual experiences.")

        over_estimates = sum(1 for o in outcomes if o.predicted_regret > o.actual_regret + 10)
        under_estimates = sum(1 for o in outcomes if o.predicted_regret < o.actual_regret - 10)

        if over_estimates > len(outcomes) * 0.6:
            patterns.append("You tend to worry more than necessary - reality often turns out better than expected.")
        elif under_estimates > len(outcomes) * 0.6:
            patterns.append("Consider being more cautious - outcomes sometimes have unexpected challenges.")

        high_satisfaction = [o for o in outcomes if o.satisfaction_score >= 70]
        if len(high_satisfaction) > len(outcomes) * 0.7:
            patterns.append("Most of your decisions lead to high satisfaction - trust your judgment!")

        recent = outcomes[-3:] if len(outcomes) >= 3 else outcomes
        recent_satisfaction = sum(o.satisfaction_score for o in recent) / len(recent)
        if recent_satisfaction >= 75:
            patterns.append("Your recent decisions are showing excellent outcomes!")
        elif recent_satisfaction < 50:
            patterns.append("Recent decisions have been challenging - consider slowing down and gathering more information.")

        return patterns if patterns else ["Continue tracking to reveal more patterns."]

    def get_learning_profile(self, user_id: str) -> Dict[str, Any]:
        if user_id not in self.learning_profiles:
            return {
                "user_id": user_id,
                "total_outcomes": 0,
                "prediction_accuracy": 0,
                "message": "Start recording outcomes to build your learning profile.",
                "patterns": []
            }

        profile = self.learning_profiles[user_id]

        return {
            "user_id": user_id,
            "total_outcomes": profile.total_outcomes,
            "prediction_accuracy": round(profile.prediction_accuracy, 1),
            "avg_prediction_error": round(profile.avg_prediction_error, 1),
            "optimism_bias": round(profile.optimism_bias, 1),
            "bias_interpretation": self._interpret_bias(profile.optimism_bias),
            "patterns": profile.patterns,
            "last_updated": profile.last_updated.isoformat() if profile.last_updated else None
        }

    def _interpret_bias(self, bias: float) -> str:
        if bias > 15:
            return "You tend to overestimate regret - things usually turn out better than you expect."
        elif bias < -15:
            return "You may underestimate challenges - consider being more cautious."
        else:
            return "Your risk perception is well-calibrated."

    def get_adjusted_prediction(self, user_id: str, base_prediction: float, decision_type: str = None) -> float:
        if user_id not in self.prediction_adjustments:
            return base_prediction

        adjustments = self.prediction_adjustments[user_id]
        bias_correction = adjustments.get("bias_correction", 0)

        adjusted = base_prediction + bias_correction
        return max(0, min(100, adjusted))

    def get_outcome_history(self, user_id: str) -> List[Dict[str, Any]]:
        if user_id not in self.outcomes:
            return []

        return [
            {
                "decision_id": o.decision_id,
                "predicted_regret": o.predicted_regret,
                "actual_regret": o.actual_regret,
                "prediction_error": abs(o.predicted_regret - o.actual_regret),
                "satisfaction_score": o.satisfaction_score,
                "outcome_notes": o.outcome_notes,
                "lessons_learned": o.lessons_learned,
                "recorded_at": o.recorded_at.isoformat(),
                "time_since_decision_days": o.time_since_decision_days
            }
            for o in self.outcomes[user_id]
        ]

    def get_prediction_vs_reality_data(self, user_id: str) -> Dict[str, Any]:
        if user_id not in self.outcomes or not self.outcomes[user_id]:
            return {"data": [], "summary": None}

        outcomes = self.outcomes[user_id]

        data = [
            {
                "index": i,
                "predicted": o.predicted_regret,
                "actual": o.actual_regret,
                "satisfaction": o.satisfaction_score,
                "date": o.recorded_at.isoformat()
            }
            for i, o in enumerate(outcomes)
        ]

        return {
            "data": data,
            "summary": {
                "total_outcomes": len(outcomes),
                "avg_predicted": sum(o.predicted_regret for o in outcomes) / len(outcomes),
                "avg_actual": sum(o.actual_regret for o in outcomes) / len(outcomes),
                "avg_satisfaction": sum(o.satisfaction_score for o in outcomes) / len(outcomes),
                "best_outcome": max(outcomes, key=lambda o: o.satisfaction_score).decision_id if outcomes else None,
                "most_accurate_prediction": min(outcomes, key=lambda o: abs(o.predicted_regret - o.actual_regret)).decision_id if outcomes else None
            }
        }

    def cleanup(self):
        self.outcomes.clear()
        self.learning_profiles.clear()
        self.prediction_adjustments.clear()
