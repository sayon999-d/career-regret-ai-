from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib
import random
import math

@dataclass
class AnonymizedOutcome:
    outcome_hash: str
    decision_type: str
    industry: str
    experience_bucket: str
    predicted_regret: float
    actual_regret: float
    satisfaction_score: float
    time_to_outcome_days: int
    key_factors: List[str]
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class GlobalPattern:
    pattern_id: str
    decision_type: str
    sample_count: int
    avg_prediction_error: float
    avg_regret: float
    avg_satisfaction: float
    success_rate: float
    common_surprises: List[str]
    risk_factors: List[str]
    protective_factors: List[str]
    last_updated: datetime = field(default_factory=datetime.utcnow)

@dataclass
class PredictionAdjustment:
    decision_type: str
    industry: str
    experience_level: str
    bias_correction: float
    confidence_adjustment: float
    sample_count: int
    last_updated: datetime = field(default_factory=datetime.utcnow)

class GlobalRegretDatabase:
    """
    Privacy-preserving database that aggregates anonymized decision outcomes
    to improve predictions without storing identifiable user data.
    """

    EXPERIENCE_BUCKETS = ["0-2", "3-5", "6-10", "11-15", "16-20", "20+"]

    PROTECTIVE_FACTORS = {
        "thorough_research": 0.85,
        "consulted_mentor": 0.80,
        "financial_buffer": 0.78,
        "gradual_transition": 0.82,
        "clear_criteria": 0.88,
        "backup_plan": 0.75,
        "family_support": 0.72,
        "skill_preparation": 0.83
    }

    RISK_FACTORS = {
        "time_pressure": 1.25,
        "emotional_decision": 1.35,
        "financial_stress": 1.40,
        "relationship_conflict": 1.30,
        "no_alternatives_considered": 1.45,
        "influenced_by_others": 1.20,
        "status_seeking": 1.28,
        "escape_motivation": 1.38
    }

    def __init__(self):
        self.outcomes: List[AnonymizedOutcome] = []
        self.patterns: Dict[str, GlobalPattern] = {}
        self.adjustments: Dict[str, PredictionAdjustment] = {}
        self.decision_type_stats: Dict[str, Dict] = defaultdict(lambda: {
            "count": 0, "total_regret": 0, "total_satisfaction": 0,
            "prediction_errors": [], "successes": 0
        })
        self._initialize_seed_data()

    def _initialize_seed_data(self):
        seed_patterns = [
            {"type": "job_change", "regret": 32, "satisfaction": 68, "success": 0.72},
            {"type": "career_switch", "regret": 38, "satisfaction": 65, "success": 0.65},
            {"type": "promotion", "regret": 28, "satisfaction": 72, "success": 0.78},
            {"type": "entrepreneurship", "regret": 45, "satisfaction": 60, "success": 0.55},
            {"type": "education", "regret": 25, "satisfaction": 75, "success": 0.82},
            {"type": "relocation", "regret": 35, "satisfaction": 67, "success": 0.68}
        ]

        for pattern in seed_patterns:
            self.patterns[pattern["type"]] = GlobalPattern(
                pattern_id=f"seed_{pattern['type']}",
                decision_type=pattern["type"],
                sample_count=1000,
                avg_prediction_error=12.5,
                avg_regret=pattern["regret"],
                avg_satisfaction=pattern["satisfaction"],
                success_rate=pattern["success"],
                common_surprises=["Timeline was longer than expected", "Emotional impact underestimated"],
                risk_factors=["time_pressure", "emotional_decision"],
                protective_factors=["thorough_research", "financial_buffer"],
                last_updated=datetime.utcnow()
            )

    def contribute_outcome(
        self,
        user_id: str,
        decision_type: str,
        industry: str,
        years_experience: int,
        predicted_regret: float,
        actual_regret: float,
        satisfaction_score: float,
        decision_date: datetime,
        factors: List[str] = None
    ) -> Dict[str, Any]:
        outcome_hash = hashlib.sha256(
            f"{user_id}{decision_type}{datetime.utcnow().timestamp()}".encode()
        ).hexdigest()[:16]

        exp_bucket = self._get_experience_bucket(years_experience)
        time_to_outcome = (datetime.utcnow() - decision_date).days

        anonymized = AnonymizedOutcome(
            outcome_hash=outcome_hash,
            decision_type=decision_type,
            industry=industry,
            experience_bucket=exp_bucket,
            predicted_regret=predicted_regret,
            actual_regret=actual_regret,
            satisfaction_score=satisfaction_score,
            time_to_outcome_days=time_to_outcome,
            key_factors=factors or []
        )

        self.outcomes.append(anonymized)

        stats = self.decision_type_stats[decision_type]
        stats["count"] += 1
        stats["total_regret"] += actual_regret
        stats["total_satisfaction"] += satisfaction_score
        stats["prediction_errors"].append(abs(predicted_regret - actual_regret))
        if satisfaction_score >= 60:
            stats["successes"] += 1

        self._update_pattern(decision_type)
        self._update_adjustment(decision_type, industry, exp_bucket, predicted_regret, actual_regret)

        return {
            "contributed": True,
            "outcome_hash": outcome_hash,
            "current_sample_size": len(self.outcomes),
            "pattern_updated": decision_type in self.patterns
        }

    def _get_experience_bucket(self, years: int) -> str:
        if years <= 2: return "0-2"
        elif years <= 5: return "3-5"
        elif years <= 10: return "6-10"
        elif years <= 15: return "11-15"
        elif years <= 20: return "16-20"
        else: return "20+"

    def _update_pattern(self, decision_type: str):
        relevant = [o for o in self.outcomes if o.decision_type == decision_type]
        if len(relevant) < 5:
            return

        avg_regret = sum(o.actual_regret for o in relevant) / len(relevant)
        avg_satisfaction = sum(o.satisfaction_score for o in relevant) / len(relevant)
        avg_error = sum(abs(o.predicted_regret - o.actual_regret) for o in relevant) / len(relevant)
        success_rate = len([o for o in relevant if o.satisfaction_score >= 60]) / len(relevant)

        all_factors = []
        for o in relevant:
            all_factors.extend(o.key_factors)
        factor_counts = defaultdict(int)
        for f in all_factors:
            factor_counts[f] += 1

        risk_factors = [f for f in factor_counts if f in self.RISK_FACTORS][:5]
        protective = [f for f in factor_counts if f in self.PROTECTIVE_FACTORS][:5]

        self.patterns[decision_type] = GlobalPattern(
            pattern_id=f"pattern_{decision_type}_{len(relevant)}",
            decision_type=decision_type,
            sample_count=len(relevant),
            avg_prediction_error=avg_error,
            avg_regret=avg_regret,
            avg_satisfaction=avg_satisfaction,
            success_rate=success_rate,
            common_surprises=["Timeline longer than expected", "Emotional adjustment needed"],
            risk_factors=risk_factors,
            protective_factors=protective,
            last_updated=datetime.utcnow()
        )

    def _update_adjustment(self, decision_type: str, industry: str,
                          exp_level: str, predicted: float, actual: float):
        key = f"{decision_type}_{industry}_{exp_level}"

        if key not in self.adjustments:
            self.adjustments[key] = PredictionAdjustment(
                decision_type=decision_type,
                industry=industry,
                experience_level=exp_level,
                bias_correction=0,
                confidence_adjustment=1.0,
                sample_count=0
            )

        adj = self.adjustments[key]
        error = predicted - actual

        adj.sample_count += 1
        weight = 1 / adj.sample_count
        adj.bias_correction = adj.bias_correction * (1 - weight) + error * weight
        adj.last_updated = datetime.utcnow()

    def get_adjusted_prediction(
        self,
        base_prediction: float,
        decision_type: str,
        industry: str = "technology",
        years_experience: int = 5
    ) -> Dict[str, Any]:
        exp_bucket = self._get_experience_bucket(years_experience)
        key = f"{decision_type}_{industry}_{exp_bucket}"

        adjustment = self.adjustments.get(key)
        pattern = self.patterns.get(decision_type)

        adjusted = base_prediction
        adjustments_applied = []

        if adjustment and adjustment.sample_count >= 10:
            bias_factor = adjustment.bias_correction * 0.5
            adjusted -= bias_factor
            adjustments_applied.append(f"Bias correction: {bias_factor:+.1f}")

        if pattern and pattern.sample_count >= 50:
            if base_prediction > pattern.avg_regret * 1.3:
                adjusted = adjusted * 0.9 + pattern.avg_regret * 0.1
                adjustments_applied.append("Calibrated to global average")

        adjusted = max(0, min(100, adjusted))

        confidence = 0.5
        if adjustment:
            confidence += min(0.3, adjustment.sample_count / 100)
        if pattern:
            confidence += min(0.2, pattern.sample_count / 500)

        return {
            "original_prediction": base_prediction,
            "adjusted_prediction": round(adjusted, 1),
            "confidence": round(confidence, 2),
            "adjustments_applied": adjustments_applied,
            "sample_size": pattern.sample_count if pattern else 0,
            "global_average": pattern.avg_regret if pattern else None,
            "success_rate": pattern.success_rate if pattern else None
        }

    def get_global_insights(self, decision_type: str) -> Dict[str, Any]:
        pattern = self.patterns.get(decision_type)

        if not pattern:
            return {
                "decision_type": decision_type,
                "data_available": False,
                "message": "Not enough data for this decision type yet"
            }

        return {
            "decision_type": decision_type,
            "data_available": True,
            "sample_size": pattern.sample_count,
            "average_regret": round(pattern.avg_regret, 1),
            "average_satisfaction": round(pattern.avg_satisfaction, 1),
            "success_rate": f"{pattern.success_rate * 100:.0f}%",
            "prediction_accuracy": f"{100 - pattern.avg_prediction_error:.0f}%",
            "risk_factors": pattern.risk_factors,
            "protective_factors": pattern.protective_factors,
            "common_surprises": pattern.common_surprises,
            "recommendation": self._generate_recommendation(pattern)
        }

    def _generate_recommendation(self, pattern: GlobalPattern) -> str:
        if pattern.success_rate >= 0.75:
            return f"This decision type generally leads to positive outcomes. Focus on {pattern.protective_factors[0] if pattern.protective_factors else 'thorough preparation'}."
        elif pattern.success_rate >= 0.6:
            return f"Mixed outcomes observed. Mitigate risks by addressing {pattern.risk_factors[0] if pattern.risk_factors else 'common pitfalls'}."
        else:
            return f"Higher regret rates observed. Consider extra preparation and {pattern.protective_factors[0] if pattern.protective_factors else 'backup plans'}."

    def compare_decision_types(self) -> List[Dict[str, Any]]:
        comparisons = []
        for dt, pattern in self.patterns.items():
            comparisons.append({
                "decision_type": dt,
                "sample_size": pattern.sample_count,
                "success_rate": pattern.success_rate,
                "avg_regret": pattern.avg_regret,
                "avg_satisfaction": pattern.avg_satisfaction,
                "risk_level": "low" if pattern.avg_regret < 30 else "medium" if pattern.avg_regret < 45 else "high"
            })

        comparisons.sort(key=lambda x: x["success_rate"], reverse=True)
        return comparisons

    def get_similar_outcomes(
        self,
        decision_type: str,
        industry: str,
        years_experience: int,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        exp_bucket = self._get_experience_bucket(years_experience)

        similar = [
            o for o in self.outcomes
            if o.decision_type == decision_type
            and (o.industry == industry or o.experience_bucket == exp_bucket)
        ]

        similar.sort(key=lambda x: x.created_at, reverse=True)

        return [{
            "satisfaction": o.satisfaction_score,
            "actual_regret": o.actual_regret,
            "prediction_error": abs(o.predicted_regret - o.actual_regret),
            "time_to_outcome_days": o.time_to_outcome_days,
            "was_successful": o.satisfaction_score >= 60
        } for o in similar[:limit]]

    def get_database_stats(self) -> Dict[str, Any]:
        return {
            "total_outcomes": len(self.outcomes),
            "decision_types_tracked": len(self.patterns),
            "patterns": {dt: p.sample_count for dt, p in self.patterns.items()},
            "adjustments_calculated": len(self.adjustments),
            "overall_success_rate": self._calculate_overall_success_rate(),
            "last_contribution": self.outcomes[-1].created_at.isoformat() if self.outcomes else None
        }

    def _calculate_overall_success_rate(self) -> float:
        if not self.outcomes:
            return 0.7
        successful = len([o for o in self.outcomes if o.satisfaction_score >= 60])
        return round(successful / len(self.outcomes), 2)

global_regret_db = GlobalRegretDatabase()
