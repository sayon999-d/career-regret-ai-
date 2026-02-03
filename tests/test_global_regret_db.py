import pytest
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.global_regret_db import GlobalRegretDatabase

class TestGlobalRegretDatabase:
    @pytest.fixture
    def service(self):
        return GlobalRegretDatabase()

    def test_contribute_outcome(self, service, sample_user_id):
        result = service.contribute_outcome(
            user_id=sample_user_id,
            decision_type="job_change",
            industry="technology",
            years_experience=5,
            predicted_regret=35.0,
            actual_regret=28.0,
            satisfaction_score=72.0,
            decision_date=datetime.utcnow() - timedelta(days=180),
            factors=["salary", "growth"]
        )

        assert result["contributed"] == True
        assert "outcome_hash" in result

    def test_outcome_anonymization(self, service, sample_user_id):
        result = service.contribute_outcome(
            user_id=sample_user_id,
            decision_type="promotion",
            industry="finance",
            years_experience=8,
            predicted_regret=20.0,
            actual_regret=15.0,
            satisfaction_score=85.0,
            decision_date=datetime.utcnow(),
            factors=["leadership"]
        )

        outcome_hash = result["outcome_hash"]

        assert sample_user_id not in outcome_hash

    def test_get_global_insights(self, service, sample_user_id):
        insights = service.get_global_insights("job_change")

        assert "decision_type" in insights
        assert insights["data_available"] == True
        assert "average_regret" in insights
        assert "sample_size" in insights

    def test_adjusted_prediction(self, service):
        base_prediction = 40.0

        result = service.get_adjusted_prediction(
            base_prediction=base_prediction,
            decision_type="job_change",
            industry="technology",
            years_experience=5
        )

        assert "adjusted_prediction" in result
        assert "original_prediction" in result
        assert "confidence" in result

    def test_compare_decision_types(self, service):
        comparisons = service.compare_decision_types()

        assert isinstance(comparisons, list)
        assert len(comparisons) > 0

        for comp in comparisons:
            assert "decision_type" in comp
            assert "success_rate" in comp

    def test_database_stats(self, service, sample_user_id):
        service.contribute_outcome(
            user_id=sample_user_id,
            decision_type="stats_test",
            industry="technology",
            years_experience=5,
            predicted_regret=30.0,
            actual_regret=25.0,
            satisfaction_score=75.0,
            decision_date=datetime.utcnow(),
            factors=[]
        )

        stats = service.get_database_stats()

        assert "total_outcomes" in stats
        assert "decision_types_tracked" in stats

    def test_privacy_preservation(self, service):
        user_id = "private_user@email.com"

        result = service.contribute_outcome(
            user_id=user_id,
            decision_type="privacy_test",
            industry="healthcare",
            years_experience=10,
            predicted_regret=25.0,
            actual_regret=20.0,
            satisfaction_score=80.0,
            decision_date=datetime.utcnow(),
            factors=["work_life_balance"]
        )

        insights = service.get_global_insights("privacy_test")

        assert "private_user" not in str(insights)
        assert "email" not in str(insights)

    def test_experience_bucket_aggregation(self, service):
        experience_levels = [2, 7, 12, 18, 25]

        for exp in experience_levels:
            service.contribute_outcome(
                user_id=f"exp_user_{exp}",
                decision_type="experience_test",
                industry="technology",
                years_experience=exp,
                predicted_regret=30.0,
                actual_regret=25.0,
                satisfaction_score=75.0,
                decision_date=datetime.utcnow(),
                factors=[]
            )

        insights = service.get_global_insights("experience_test")
        assert insights is not None

    def test_get_similar_outcomes(self, service):
        for i in range(3):
            service.contribute_outcome(
                user_id=f"similar_user_{i}",
                decision_type="similar_test",
                industry="technology",
                years_experience=5,
                predicted_regret=30.0,
                actual_regret=25.0 + i,
                satisfaction_score=70.0 + i,
                decision_date=datetime.utcnow(),
                factors=[]
            )

        similar = service.get_similar_outcomes(
            decision_type="similar_test",
            industry="technology",
            years_experience=5,
            limit=5
        )

        assert isinstance(similar, list)


class TestGlobalPatternDetection:
    @pytest.fixture
    def service(self):
        return GlobalRegretDatabase()

    def test_pattern_update(self, service):
        for i in range(10):
            service.contribute_outcome(
                user_id=f"pattern_user_{i}",
                decision_type="pattern_test",
                industry="technology",
                years_experience=5,
                predicted_regret=40.0,
                actual_regret=35.0 if i < 7 else 50.0,
                satisfaction_score=70.0,
                decision_date=datetime.utcnow(),
                factors=["factor_a"]
            )

        insights = service.get_global_insights("pattern_test")

        assert insights["data_available"] == True

    def test_seeded_patterns_exist(self, service):
        patterns = service.compare_decision_types()

        assert len(patterns) >= 6

        decision_types = [p["decision_type"] for p in patterns]
        assert "job_change" in decision_types
        assert "career_switch" in decision_types
        assert "promotion" in decision_types
