import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import numpy as np
import hashlib

@dataclass
class FeedbackEntry:
    id: str
    user_id: str
    analysis_id: Optional[str]
    feedback_type: str
    content: Dict
    timestamp: datetime = field(default_factory=datetime.utcnow)
    processed: bool = False
    applied_weight: float = 0.0
    ab_variant: Optional[str] = None
    demographic_group: Optional[str] = None

@dataclass
class FeedbackStats:
    total_feedback: int = 0
    positive_feedback: int = 0
    negative_feedback: int = 0
    corrections_applied: int = 0
    average_rating: float = 0.0
    model_updates: int = 0
    last_update: Optional[datetime] = None
    nps_score: float = 0.0

@dataclass
class ABTestConfig:
    test_id: str
    name: str
    variants: List[str]
    weights: List[float]
    start_date: datetime = field(default_factory=datetime.utcnow)
    end_date: Optional[datetime] = None
    active: bool = True
    metrics_tracked: List[str] = field(default_factory=list)

@dataclass
class ABTestResult:
    test_id: str
    variant: str
    user_id: str
    metric: str
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

class AdvancedFeedbackLoop:
    def __init__(self, ml_predictor=None, decision_graph=None, learning_rate: float = 0.01,
                 feedback_weight: float = 0.3, batch_size: int = 10, enable_ab_testing: bool = True):
        self.ml_predictor = ml_predictor
        self.decision_graph = decision_graph
        self.learning_rate = learning_rate
        self.feedback_weight = feedback_weight
        self.batch_size = batch_size
        self.enable_ab_testing = enable_ab_testing

        self.pending_feedback: deque = deque(maxlen=1000)
        self.processed_feedback: List[FeedbackEntry] = []
        self.feedback_history: Dict[str, List[FeedbackEntry]] = {}
        self.stats = FeedbackStats()
        self.ab_tests: Dict[str, ABTestConfig] = {}
        self.ab_results: List[ABTestResult] = []
        self.user_variants: Dict[str, Dict[str, str]] = {}
        self.cohorts: Dict[str, List[str]] = defaultdict(list)
        self.cohort_metrics: Dict[str, Dict] = {}
        self.demographic_metrics: Dict[str, List[float]] = defaultdict(list)
        self.bias_alerts: List[Dict] = []

        self.sentiment_keywords = {
            'positive': ['helpful', 'accurate', 'great', 'useful', 'thanks', 'perfect', 'exactly', 'love', 'excellent'],
            'negative': ['wrong', 'inaccurate', 'unhelpful', 'bad', 'terrible', 'useless', 'confused', 'frustrated']
        }

        self.outcome_mappings = {
            'very_satisfied': 1.0, 'satisfied': 0.8, 'somewhat_satisfied': 0.6, 'neutral': 0.5,
            'somewhat_dissatisfied': 0.4, 'dissatisfied': 0.2, 'very_dissatisfied': 0.0
        }

        if enable_ab_testing:
            self._initialize_default_tests()

    def _initialize_default_tests(self):
        self.create_ab_test(
            test_id="response_style_v1", name="Response Style Test",
            variants=["empathetic", "analytical", "balanced"], weights=[0.33, 0.33, 0.34],
            metrics_tracked=["rating", "engagement", "follow_up"]
        )
        self.create_ab_test(
            test_id="rec_count_v1", name="Recommendation Count Test",
            variants=["few", "moderate", "many"], weights=[0.33, 0.33, 0.34],
            metrics_tracked=["rating", "action_taken"]
        )

    def create_ab_test(self, test_id: str, name: str, variants: List[str],
                       weights: List[float], metrics_tracked: List[str],
                       duration_days: int = 30) -> ABTestConfig:
        config = ABTestConfig(
            test_id=test_id, name=name, variants=variants, weights=weights,
            end_date=datetime.utcnow() + timedelta(days=duration_days), metrics_tracked=metrics_tracked
        )
        self.ab_tests[test_id] = config
        return config

    def get_user_variant(self, user_id: str, test_id: str) -> str:
        if not self.enable_ab_testing or test_id not in self.ab_tests:
            return "control"
        test = self.ab_tests[test_id]
        if not test.active or (test.end_date and datetime.utcnow() > test.end_date):
            return test.variants[0]
        if user_id not in self.user_variants:
            self.user_variants[user_id] = {}
        if test_id not in self.user_variants[user_id]:
            hash_val = int(hashlib.sha256(f"{user_id}_{test_id}".encode()).hexdigest(), 16)
            rand_val = (hash_val % 1000) / 1000.0
            cumulative = 0
            for variant, weight in zip(test.variants, test.weights):
                cumulative += weight
                if rand_val < cumulative:
                    self.user_variants[user_id][test_id] = variant
                    break
            else:
                self.user_variants[user_id][test_id] = test.variants[-1]
        return self.user_variants[user_id][test_id]

    def record_ab_metric(self, test_id: str, user_id: str, metric: str, value: float):
        variant = self.get_user_variant(user_id, test_id)
        result = ABTestResult(test_id=test_id, variant=variant, user_id=user_id, metric=metric, value=value)
        self.ab_results.append(result)

    def analyze_ab_test(self, test_id: str) -> Dict:
        if test_id not in self.ab_tests:
            return {"error": "Test not found"}
        test = self.ab_tests[test_id]
        results = [r for r in self.ab_results if r.test_id == test_id]
        if not results:
            return {"test_id": test_id, "status": "no_data"}

        variant_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        for result in results:
            variant_metrics[result.variant][result.metric].append(result.value)

        analysis = {"test_id": test_id, "name": test.name, "variants": {}, "sample_sizes": {}, "winner": None, "confidence": 0.0}
        for variant in test.variants:
            metrics = variant_metrics[variant]
            variant_data = {}
            for metric_name in test.metrics_tracked:
                values = metrics.get(metric_name, [])
                if values:
                    variant_data[metric_name] = {"mean": np.mean(values), "std": np.std(values), "count": len(values)}
            analysis["variants"][variant] = variant_data
            analysis["sample_sizes"][variant] = len(set(r.user_id for r in results if r.variant == variant))

        if test.metrics_tracked and all(v for v in analysis["variants"].values()):
            primary_metric = test.metrics_tracked[0]
            best_variant = None
            best_value = -float('inf')
            for variant, data in analysis["variants"].items():
                if primary_metric in data:
                    mean = data[primary_metric]["mean"]
                    if mean > best_value:
                        best_value = mean
                        best_variant = variant
            analysis["winner"] = best_variant
            if len(analysis["variants"]) >= 2:
                all_means = [v.get(primary_metric, {}).get("mean", 0) for v in analysis["variants"].values()]
                if max(all_means) > 0:
                    analysis["confidence"] = (max(all_means) - np.mean(all_means)) / max(all_means)
        return analysis

    def add_feedback(self, user_id: str, feedback_type: str, content: Dict,
                     analysis_id: Optional[str] = None, demographic_group: Optional[str] = None) -> FeedbackEntry:
        ab_variant = self.get_user_variant(user_id, "response_style_v1") if self.enable_ab_testing else None
        entry = FeedbackEntry(
            id=f"fb_{datetime.utcnow().timestamp()}_{user_id[:8]}", user_id=user_id,
            analysis_id=analysis_id, feedback_type=feedback_type, content=content,
            ab_variant=ab_variant, demographic_group=demographic_group
        )
        self.pending_feedback.append(entry)
        if user_id not in self.feedback_history:
            self.feedback_history[user_id] = []
        self.feedback_history[user_id].append(entry)
        self.stats.total_feedback += 1

        if feedback_type == 'rating':
            self._process_rating(entry)
            rating = content.get('rating', 3)
            self.record_ab_metric("response_style_v1", user_id, "rating", rating)

        if feedback_type == 'comment' and 'text' in content:
            sentiment = self._analyze_sentiment(content['text'])
            entry.content['sentiment'] = sentiment
            if sentiment > 0.3:
                self.stats.positive_feedback += 1
            elif sentiment < -0.3:
                self.stats.negative_feedback += 1

        if demographic_group:
            self._track_demographic_metric(demographic_group, entry)

        self._assign_to_cohort(user_id, entry)

        if len(self.pending_feedback) >= self.batch_size:
            asyncio.create_task(self.process_batch())
        return entry

    def _process_rating(self, entry: FeedbackEntry):
        rating = entry.content.get('rating', 3)
        normalized = (rating - 1) / 4
        n = self.stats.total_feedback
        old_avg = self.stats.average_rating
        self.stats.average_rating = (old_avg * (n - 1) + normalized) / n if n > 0 else normalized
        if rating >= 5:
            nps_contribution = 1
        elif rating >= 3:
            nps_contribution = 0
        else:
            nps_contribution = -1
        self.stats.nps_score = (self.stats.nps_score * (n - 1) + nps_contribution) / n if n > 0 else nps_contribution

    def _analyze_sentiment(self, text: str) -> float:
        text_lower = text.lower()
        positive_count = sum(1 for word in self.sentiment_keywords['positive'] if word in text_lower)
        negative_count = sum(1 for word in self.sentiment_keywords['negative'] if word in text_lower)
        total = positive_count + negative_count
        return (positive_count - negative_count) / total if total > 0 else 0.0

    def _track_demographic_metric(self, group: str, entry: FeedbackEntry):
        if entry.feedback_type == 'rating':
            rating = entry.content.get('rating', 3)
            self.demographic_metrics[group].append(rating)
            self._check_demographic_bias()

    def _check_demographic_bias(self):
        if len(self.demographic_metrics) < 2:
            return
        group_means = {}
        for group, ratings in self.demographic_metrics.items():
            if len(ratings) >= 10:
                group_means[group] = np.mean(ratings)
        if len(group_means) < 2:
            return
        overall_mean = np.mean(list(group_means.values()))
        for group, mean in group_means.items():
            deviation = abs(mean - overall_mean) / overall_mean if overall_mean > 0 else 0
            if deviation > 0.15:
                alert = {"type": "demographic_bias", "group": group, "deviation": deviation,
                        "group_mean": mean, "overall_mean": overall_mean,
                        "timestamp": datetime.utcnow().isoformat(), "sample_size": len(self.demographic_metrics[group])}
                if not any(a["group"] == group and a["type"] == "demographic_bias" for a in self.bias_alerts[-10:]):
                    self.bias_alerts.append(alert)

    def _assign_to_cohort(self, user_id: str, entry: FeedbackEntry):
        week_cohort = f"week_{datetime.utcnow().isocalendar()[1]}_{datetime.utcnow().year}"
        if user_id not in self.cohorts[week_cohort]:
            self.cohorts[week_cohort].append(user_id)
        user_feedback = self.feedback_history.get(user_id, [])
        if len(user_feedback) >= 5:
            self.cohorts["engaged_users"].append(user_id)
        if entry.feedback_type == 'correction':
            self.cohorts["correctors"].append(user_id)

    async def process_batch(self) -> Dict:
        if not self.pending_feedback:
            return {"processed": 0}
        batch = []
        while self.pending_feedback and len(batch) < self.batch_size:
            batch.append(self.pending_feedback.popleft())

        results = {"processed": len(batch), "model_updates": 0, "graph_updates": 0, "rlhf_updates": 0, "errors": []}

        corrections = [e for e in batch if e.feedback_type == 'correction']
        outcomes = [e for e in batch if e.feedback_type == 'outcome_update']

        if corrections and self.ml_predictor:
            rlhf_result = await self._apply_rlhf_updates(corrections)
            results["rlhf_updates"] = rlhf_result.get("updates", 0)

        for entry in outcomes:
            try:
                await self._process_outcome_update(entry)
                results["model_updates"] += 1
                results["graph_updates"] += 1
            except Exception as e:
                results["errors"].append({"id": entry.id, "error": str(e)})

        for entry in batch:
            entry.processed = True
            self.processed_feedback.append(entry)

        self.stats.model_updates += results["model_updates"]
        self.stats.last_update = datetime.utcnow()
        return results

    async def _apply_rlhf_updates(self, corrections: List[FeedbackEntry]) -> Dict:
        if not self.ml_predictor:
            return {"updates": 0}
        updates = 0
        for entry in corrections:
            content = entry.content
            decision_data = content.get('decision_data', {})
            actual_regret = content.get('actual_regret')
            if actual_regret is not None:
                predicted = content.get('predicted_regret', 0.5)
                error = abs(predicted - actual_regret)
                reward = 1.0 - error
                adjusted_lr = self.learning_rate * (1 + reward * self.feedback_weight)
                self.ml_predictor.update_with_feedback(decision_data=decision_data, actual_regret=actual_regret, learning_rate=adjusted_lr)
                updates += 1
                entry.applied_weight = adjusted_lr
        return {"updates": updates}

    async def _process_outcome_update(self, entry: FeedbackEntry):
        content = entry.content
        decision_id = content.get('decision_id')
        outcome = content.get('outcome')
        actual_regret = self.outcome_mappings.get(outcome, 0.5)
        if self.ml_predictor and 'decision_data' in content:
            self.ml_predictor.update_with_feedback(decision_data=content['decision_data'], actual_regret=actual_regret, learning_rate=self.learning_rate)
        if self.decision_graph and decision_id:
            outcome_node = content.get('outcome_node')
            if outcome_node:
                satisfaction = 1.0 - actual_regret
                self.decision_graph.update_from_feedback(decision_id=decision_id, actual_outcome=outcome_node, satisfaction=satisfaction)

    def get_cohort_analysis(self, cohort_id: str) -> Dict:
        user_ids = self.cohorts.get(cohort_id, [])
        if not user_ids:
            return {"cohort_id": cohort_id, "status": "empty"}
        cohort_feedback = []
        for user_id in user_ids:
            cohort_feedback.extend(self.feedback_history.get(user_id, []))
        if not cohort_feedback:
            return {"cohort_id": cohort_id, "status": "no_feedback"}
        ratings = [e.content.get('rating', 3) for e in cohort_feedback if e.feedback_type == 'rating']
        corrections = [e for e in cohort_feedback if e.feedback_type == 'correction']
        return {
            "cohort_id": cohort_id, "user_count": len(set(user_ids)), "total_feedback": len(cohort_feedback),
            "average_rating": np.mean(ratings) if ratings else None, "rating_std": np.std(ratings) if ratings else None,
            "correction_rate": len(corrections) / len(cohort_feedback) if cohort_feedback else 0,
            "time_range": {"first": min(e.timestamp for e in cohort_feedback).isoformat(), "last": max(e.timestamp for e in cohort_feedback).isoformat()}
        }

    def get_improvement_insights(self) -> Dict:
        if not self.processed_feedback:
            return {"message": "Insufficient feedback for insights"}

        corrections = [e for e in self.processed_feedback if e.feedback_type == 'correction']
        correction_analysis = defaultdict(lambda: {"count": 0, "total_error": 0})
        for correction in corrections:
            decision_type = correction.content.get('decision_data', {}).get('decision_type', 'unknown')
            predicted = correction.content.get('predicted_regret', 0.5)
            actual = correction.content.get('actual_regret', 0.5)
            correction_analysis[decision_type]["count"] += 1
            correction_analysis[decision_type]["total_error"] += abs(predicted - actual)

        for dt, data in correction_analysis.items():
            if data["count"] > 0:
                data["avg_error"] = data["total_error"] / data["count"]

        ab_insights = {test_id: self.analyze_ab_test(test_id) for test_id in self.ab_tests}
        recommendations = self._generate_improvement_recommendations(correction_analysis, ab_insights)

        return {
            "total_processed": len(self.processed_feedback), "correction_analysis": dict(correction_analysis),
            "ab_test_insights": ab_insights, "bias_alerts": self.bias_alerts[-5:],
            "cohort_summaries": {cohort: len(users) for cohort, users in list(self.cohorts.items())[:10]},
            "nps_score": self.stats.nps_score, "average_rating": self.stats.average_rating, "recommendations": recommendations
        }

    def _generate_improvement_recommendations(self, correction_analysis: Dict, ab_insights: Dict) -> List[str]:
        recommendations = []
        if correction_analysis:
            worst_type = max(correction_analysis.items(), key=lambda x: x[1].get("avg_error", 0))
            if worst_type[1].get("avg_error", 0) > 0.2:
                recommendations.append(f"Improve predictions for '{worst_type[0]}' decisions - average error of {worst_type[1]['avg_error']:.1%}.")

        for test_id, analysis in ab_insights.items():
            if analysis.get("winner") and analysis.get("confidence", 0) > 0.3:
                recommendations.append(f"Consider adopting '{analysis['winner']}' variant for {analysis.get('name', test_id)} (confidence: {analysis['confidence']:.0%}).")

        if self.bias_alerts:
            latest_bias = self.bias_alerts[-1]
            recommendations.append(f"Investigate potential bias in '{latest_bias['group']}' group ({latest_bias['deviation']:.1%} deviation from mean).")

        if self.stats.nps_score < 0:
            recommendations.append("User satisfaction is declining. Review recent negative feedback for patterns.")
        elif self.stats.nps_score > 0.5:
            recommendations.append("Strong user satisfaction. Consider collecting success stories.")

        return recommendations or ["System performing well. Continue monitoring feedback trends."]

    def get_stats(self) -> Dict:
        return {
            "total_feedback": self.stats.total_feedback, "positive_feedback": self.stats.positive_feedback,
            "negative_feedback": self.stats.negative_feedback, "corrections_applied": self.stats.corrections_applied,
            "average_rating": round(self.stats.average_rating, 3), "nps_score": round(self.stats.nps_score, 3),
            "model_updates": self.stats.model_updates, "last_update": self.stats.last_update.isoformat() if self.stats.last_update else None,
            "pending_feedback": len(self.pending_feedback), "active_ab_tests": len([t for t in self.ab_tests.values() if t.active]),
            "cohorts_tracked": len(self.cohorts), "bias_alerts": len(self.bias_alerts)
        }

    def reset_stats(self):
        self.stats = FeedbackStats()
        self.pending_feedback.clear()
        self.processed_feedback.clear()
        self.feedback_history.clear()
        self.ab_results.clear()
        self.user_variants.clear()
        self.cohorts.clear()
        self.demographic_metrics.clear()
        self.bias_alerts.clear()

FeedbackLoop = AdvancedFeedbackLoop
