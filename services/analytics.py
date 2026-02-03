from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

@dataclass
class DecisionRecord:
    id: str
    user_id: str
    decision_type: str
    description: str
    predicted_regret: float
    actual_outcome: Optional[str] = None
    actual_regret: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)

class AnalyticsService:
    def __init__(self):
        self.decisions: Dict[str, List[DecisionRecord]] = defaultdict(list)
        self.insights_cache: Dict[str, Dict] = {}

    def record_decision(self, user_id: str, decision_id: str, decision_type: str,
                       description: str, predicted_regret: float,
                       tags: List[str] = None) -> DecisionRecord:
        record = DecisionRecord(
            id=decision_id, user_id=user_id, decision_type=decision_type,
            description=description, predicted_regret=predicted_regret,
            tags=tags or []
        )
        self.decisions[user_id].append(record)
        return record

    def update_outcome(self, user_id: str, decision_id: str,
                       outcome: str, actual_regret: float):
        for record in self.decisions.get(user_id, []):
            if record.id == decision_id:
                record.actual_outcome = outcome
                record.actual_regret = actual_regret
                break

    def get_user_analytics(self, user_id: str) -> Dict:
        records = self.decisions.get(user_id, [])
        if not records:
            return {"user_id": user_id, "total_decisions": 0}

        type_counts = defaultdict(int)
        for r in records:
            type_counts[r.decision_type] += 1

        predictions = [r.predicted_regret for r in records]
        actuals = [r.actual_regret for r in records if r.actual_regret is not None]

        accuracy = None
        if actuals:
            paired = [(r.predicted_regret, r.actual_regret)
                     for r in records if r.actual_regret is not None]
            errors = [abs(p - a) for p, a in paired]
            accuracy = 1 - (sum(errors) / len(errors))

        now = datetime.utcnow()
        last_30_days = [r for r in records if (now - r.timestamp).days <= 30]
        last_90_days = [r for r in records if (now - r.timestamp).days <= 90]

        return {
            "user_id": user_id,
            "total_decisions": len(records),
            "decision_types": dict(type_counts),
            "avg_predicted_regret": sum(predictions) / len(predictions),
            "avg_actual_regret": sum(actuals) / len(actuals) if actuals else None,
            "prediction_accuracy": accuracy,
            "decisions_last_30_days": len(last_30_days),
            "decisions_last_90_days": len(last_90_days),
            "outcomes_recorded": len(actuals),
            "most_common_type": max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else None
        }

    def get_decision_history(self, user_id: str, limit: int = 20) -> List[Dict]:
        records = self.decisions.get(user_id, [])
        sorted_records = sorted(records, key=lambda r: r.timestamp, reverse=True)
        return [
            {
                "id": r.id, "type": r.decision_type, "description": r.description[:100],
                "predicted_regret": r.predicted_regret, "actual_outcome": r.actual_outcome,
                "actual_regret": r.actual_regret, "date": r.timestamp.isoformat(), "tags": r.tags
            }
            for r in sorted_records[:limit]
        ]

    def get_trends(self, user_id: str) -> Dict:
        records = self.decisions.get(user_id, [])
        if len(records) < 3:
            return {"insufficient_data": True}

        sorted_records = sorted(records, key=lambda r: r.timestamp)
        recent = sorted_records[-5:]
        older = sorted_records[:-5] if len(sorted_records) > 5 else []

        recent_avg = sum(r.predicted_regret for r in recent) / len(recent)
        older_avg = sum(r.predicted_regret for r in older) / len(older) if older else recent_avg

        trend = "improving" if recent_avg < older_avg else "declining" if recent_avg > older_avg else "stable"

        return {
            "trend": trend, "recent_avg_regret": recent_avg, "historical_avg_regret": older_avg,
            "change_percent": ((older_avg - recent_avg) / older_avg * 100) if older_avg > 0 else 0
        }

    def generate_report(self, user_id: str) -> Dict:
        analytics = self.get_user_analytics(user_id)
        history = self.get_decision_history(user_id, limit=10)
        trends = self.get_trends(user_id)

        insights = []
        if analytics.get("prediction_accuracy") and analytics["prediction_accuracy"] > 0.8:
            insights.append("Your self-awareness is strong - predictions closely match outcomes.")
        if trends.get("trend") == "improving":
            insights.append("Your decision quality is improving over time.")
        if analytics.get("most_common_type"):
            insights.append(f"You frequently make {analytics['most_common_type'].replace('_', ' ')} decisions.")

        return {
            "generated_at": datetime.utcnow().isoformat(), "user_id": user_id,
            "summary": analytics, "trends": trends, "recent_decisions": history,
            "insights": insights, "recommendations": self._generate_recommendations(analytics, trends)
        }

    def _generate_recommendations(self, analytics: Dict, trends: Dict) -> List[str]:
        recs = []
        if analytics.get("outcomes_recorded", 0) < analytics.get("total_decisions", 0) * 0.5:
            recs.append("Record more outcomes to improve prediction accuracy over time.")
        if trends.get("trend") == "declining":
            recs.append("Consider slowing down and analyzing decisions more thoroughly.")
        if analytics.get("avg_predicted_regret", 0) > 0.6:
            recs.append("Your decisions tend toward higher risk - ensure this aligns with your goals.")
        return recs or ["Keep tracking your decisions for personalized insights."]
