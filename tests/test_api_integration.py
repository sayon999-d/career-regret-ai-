import pytest
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBiasInterceptorAPI:
    """Test Bias Interceptor service directly"""

    def test_analyze_bias(self):
        from services.bias_interceptor_service import BiasInterceptorService
        service = BiasInterceptorService()

        detections = service.analyze_text("I've already invested 5 years, I cannot give up now")

        assert isinstance(detections, list)
        if detections:
            assert "type" in detections[0]

    def test_realtime_feedback(self):
        from services.bias_interceptor_service import BiasInterceptorService
        service = BiasInterceptorService()

        feedback = service.get_real_time_feedback("I'm afraid to lose what I have")

        assert "has_bias" in feedback

    def test_bias_profile(self):
        from services.bias_interceptor_service import BiasInterceptorService
        service = BiasInterceptorService()

        service.analyze_text("I cannot give up after all this time", "profile_test")
        profile = service.get_user_bias_profile("profile_test")

        assert "user_id" in profile

    def test_bias_explanation(self):
        from services.bias_interceptor_service import BiasInterceptorService
        service = BiasInterceptorService()

        explanation = service.get_bias_explanation("sunk_cost")

        assert "explanation" in explanation


class TestFutureSelfAPI:
    """Test Future Self service directly"""

    def test_start_conversation(self):
        from services.future_self_service import FutureSelfService
        service = FutureSelfService()

        result = service.start_conversation(
            user_id="api_test",
            decision_type="career_switch",
            decision_desc="Switching to product management",
            timeframe="10_years",
            scenario="optimistic"
        )

        assert "session_id" in result
        assert "opening_message" in result

    def test_send_message(self):
        from services.future_self_service import FutureSelfService
        service = FutureSelfService()

        start_result = service.start_conversation(
            user_id="message_test",
            decision_type="job_change",
            decision_desc="Changing jobs",
            timeframe="5_years",
            scenario="realistic"
        )
        session_id = start_result["session_id"]

        result = service.send_message(session_id, "Do you have any regrets?")

        assert "response" in result

    def test_end_conversation(self):
        from services.future_self_service import FutureSelfService
        service = FutureSelfService()

        start_result = service.start_conversation(
            user_id="end_test",
            decision_type="promotion",
            decision_desc="Taking a leadership role",
            timeframe="5_years",
            scenario="optimistic"
        )
        session_id = start_result["session_id"]

        result = service.end_conversation(session_id)

        assert "session_id" in result


class TestOpportunityScoutAPI:
    """Test Opportunity Scout service directly"""

    def test_register_profile(self):
        from services.opportunity_scout_service import OpportunityScoutService
        service = OpportunityScoutService()

        profile = service.register_user_profile(
            "scout_api_test",
            {
                "current_role": "Software Engineer",
                "industry": "technology",
                "skills": ["Python", "JavaScript"],
                "salary_target": 150000
            }
        )

        assert profile is not None
        assert profile.user_id == "scout_api_test"

    @pytest.mark.asyncio
    async def test_scan_opportunities(self):
        from services.opportunity_scout_service import OpportunityScoutService
        service = OpportunityScoutService()

        service.register_user_profile(
            "scan_test",
            {
                "current_role": "Engineer",
                "skills": ["Python"]
            }
        )

        opportunities = await service.scan_opportunities("scan_test")

        assert isinstance(opportunities, list)

    def test_get_summary(self):
        from services.opportunity_scout_service import OpportunityScoutService
        service = OpportunityScoutService()

        service.register_user_profile("summary_test", {"current_role": "Developer"})

        summary = service.get_scout_summary("summary_test")

        assert "profile_active" in summary


class TestGlobalRegretAPI:
    """Test Global Regret Database service directly"""

    def test_contribute_outcome(self):
        from services.global_regret_db import GlobalRegretDatabase
        service = GlobalRegretDatabase()

        result = service.contribute_outcome(
            user_id="api_test",
            decision_type="job_change",
            industry="technology",
            years_experience=5,
            predicted_regret=35.0,
            actual_regret=28.0,
            satisfaction_score=72.0,
            decision_date=datetime.utcnow(),
            factors=["salary"]
        )

        assert result["contributed"] == True
        assert "outcome_hash" in result

    def test_get_insights(self):
        from services.global_regret_db import GlobalRegretDatabase
        service = GlobalRegretDatabase()

        insights = service.get_global_insights("job_change")

        assert "decision_type" in insights

    def test_adjusted_prediction(self):
        from services.global_regret_db import GlobalRegretDatabase
        service = GlobalRegretDatabase()

        result = service.get_adjusted_prediction(
            base_prediction=40.0,
            decision_type="job_change",
            industry="technology",
            years_experience=5
        )

        assert "adjusted_prediction" in result

    def test_compare(self):
        from services.global_regret_db import GlobalRegretDatabase
        service = GlobalRegretDatabase()

        comparisons = service.compare_decision_types()

        assert isinstance(comparisons, list)

    def test_stats(self):
        from services.global_regret_db import GlobalRegretDatabase
        service = GlobalRegretDatabase()

        stats = service.get_database_stats()

        assert "total_outcomes" in stats


class TestAdvancedAnalyticsAPI:
    """Test Advanced Analytics service directly"""

    def test_analytics_dashboard(self):
        from services.advanced_analytics_service import AdvancedAnalyticsService
        service = AdvancedAnalyticsService()

        dashboard = service.get_analytics_dashboard("test_user")

        assert "user_id" in dashboard
        assert "prediction_accuracy" in dashboard

    def test_create_goal(self):
        from services.advanced_analytics_service import AdvancedAnalyticsService
        service = AdvancedAnalyticsService()

        goal = service.create_career_goal(
            user_id="goal_test",
            title="Become Tech Lead",
            description="Transition to a leadership role",
            category="leadership"
        )

        assert goal.user_id == "goal_test"
        assert goal.title == "Become Tech Lead"

    def test_get_goals(self):
        from services.advanced_analytics_service import AdvancedAnalyticsService
        service = AdvancedAnalyticsService()

        service.create_career_goal("goals_user", "Test Goal", "Description", "career_growth")

        goals = service.get_user_goals("goals_user")

        assert isinstance(goals, list)
        assert len(goals) >= 1


class TestDataPrivacyAPI:
    """Test Data Privacy service directly"""

    def test_privacy_dashboard(self):
        from services.data_privacy_service import DataPrivacyService
        service = DataPrivacyService()

        dashboard = service.get_privacy_dashboard("test_user")

        assert "consents" in dashboard
        assert "rights" in dashboard

    def test_consent_management(self):
        from services.data_privacy_service import DataPrivacyService, ConsentType
        service = DataPrivacyService()

        consent = service.record_consent("consent_user", ConsentType.DATA_COLLECTION, True)

        assert consent.granted == True

    def test_export_request(self):
        from services.data_privacy_service import DataPrivacyService
        service = DataPrivacyService()

        request = service.request_data_export("export_user")

        assert request.user_id == "export_user"
        assert request.status == "pending"

    def test_deletion_request(self):
        from services.data_privacy_service import DataPrivacyService
        service = DataPrivacyService()

        request = service.request_account_deletion("delete_user", "Testing")

        assert request.status == "pending"
        assert request.scheduled_at is not None
