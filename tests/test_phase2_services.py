import pytest
from datetime import datetime, timedelta
from services.push_notification_service import (
    PushNotificationService, NotificationType, NotificationPriority
)
from services.scheduled_checkin_service import (
    ScheduledCheckInService, CheckInType, CheckInFrequency
)
from services.resume_parser_service import ResumeParserService
from services.proactive_suggestion_service import ProactiveSuggestionService, SuggestionType
from services.monitoring_dashboard_service import MonitoringDashboardService
from services.calendar_sync_service import GoogleCalendarSyncService, CalendarEventType


class TestPushNotificationService:
    @pytest.fixture
    def service(self):
        return PushNotificationService()

    def test_subscribe(self, service):
        result = service.subscribe(
            user_id="test_user",
            endpoint="https://push.example.com/test",
            p256dh_key="test_p256dh",
            auth_key="test_auth"
        )

        assert result["subscribed"] is True
        assert result["user_id"] == "test_user"
        assert "preferences" in result

    def test_unsubscribe(self, service):
        service.subscribe(
            user_id="test_user",
            endpoint="https://push.example.com/test",
            p256dh_key="test_p256dh",
            auth_key="test_auth"
        )

        result = service.unsubscribe("test_user")
        assert result["unsubscribed"] is True

    def test_update_preferences(self, service):
        service.subscribe(
            user_id="test_user",
            endpoint="https://push.example.com/test",
            p256dh_key="test_p256dh",
            auth_key="test_auth"
        )

        result = service.update_preferences(
            "test_user",
            {NotificationType.OPPORTUNITY_ALERT.value: False}
        )

        assert result["updated"] is True
        assert result["preferences"][NotificationType.OPPORTUNITY_ALERT.value] is False

    def test_get_preferences(self, service):
        service.subscribe(
            user_id="test_user",
            endpoint="https://push.example.com/test",
            p256dh_key="test_p256dh",
            auth_key="test_auth"
        )

        result = service.get_preferences("test_user")
        assert result["subscribed"] is True
        assert "preferences" in result

    def test_get_unread_count(self, service):
        count = service.get_unread_count("new_user")
        assert count == 0

    def test_mark_all_read(self, service):
        count = service.mark_all_read("test_user")
        assert count >= 0


class TestScheduledCheckInService:
    """Tests for scheduled check-in functionality"""

    @pytest.fixture
    def service(self):
        return ScheduledCheckInService()

    def test_create_check_in(self, service):
        check_in = service.create_check_in(
            user_id="test_user",
            check_in_type=CheckInType.WEEKLY_REFLECTION,
            title="Weekly Reflection",
            description="Reflect on your week",
            frequency=CheckInFrequency.WEEKLY
        )

        assert check_in.id is not None
        assert check_in.title == "Weekly Reflection"
        assert check_in.frequency == CheckInFrequency.WEEKLY

    def test_get_due_check_ins(self, service):
        service.create_check_in(
            user_id="test_user",
            check_in_type=CheckInType.WEEKLY_REFLECTION,
            title="Due Check-in",
            frequency=CheckInFrequency.WEEKLY,
            start_date=datetime.utcnow() - timedelta(days=1)
        )

        due = service.get_due_check_ins("test_user")
        assert len(due) >= 1

    def test_complete_check_in(self, service):
        check_in = service.create_check_in(
            user_id="test_user",
            check_in_type=CheckInType.WEEKLY_REFLECTION,
            title="Test Check-in",
            frequency=CheckInFrequency.WEEKLY,
            start_date=datetime.utcnow() - timedelta(days=1)
        )

        result = service.complete_check_in(
            user_id="test_user",
            check_in_id=check_in.id,
            responses={"question1": "answer1"},
            mood_score=7
        )

        assert result["completed"] is True
        assert "streak" in result

    def test_get_check_in_stats(self, service):
        service.create_check_in(
            user_id="test_user",
            check_in_type=CheckInType.WEEKLY_REFLECTION,
            title="Stats Test",
            frequency=CheckInFrequency.WEEKLY
        )

        stats = service.get_check_in_stats("test_user")
        assert "total_check_ins" in stats
        assert stats["total_check_ins"] >= 1

    def test_setup_default_check_ins(self, service):
        created = service.setup_default_check_ins("new_user")
        assert len(created) == 2

    def test_pause_resume_check_in(self, service):
        check_in = service.create_check_in(
            user_id="test_user",
            check_in_type=CheckInType.GOAL_REVIEW,
            title="Goal Review",
            frequency=CheckInFrequency.WEEKLY
        )

        paused = service.pause_check_in("test_user", check_in.id)
        assert paused is True

        resumed = service.resume_check_in("test_user", check_in.id)
        assert resumed is True


class TestResumeParserService:
    """Tests for resume parsing functionality"""

    @pytest.fixture
    def service(self):
        return ResumeParserService()

    @pytest.fixture
    def sample_resume(self):
        return """
        John Smith
        john.smith@email.com
        +1 (555) 123-4567
        San Francisco, CA

        PROFESSIONAL SUMMARY
        Experienced software engineer with 5+ years of experience in Python,
        JavaScript, and cloud technologies. Strong problem-solving skills
        and passion for building scalable applications.

        EXPERIENCE

        Senior Software Engineer
        Tech Company Inc.
        Jan 2020 - Present
        - Led development of microservices architecture
        - Mentored junior developers

        Software Engineer
        Startup LLC
        Jun 2017 - Dec 2019
        - Built REST APIs using Python and FastAPI
        - Implemented CI/CD pipelines

        EDUCATION

        Bachelor's Degree in Computer Science
        University of California
        2017

        SKILLS
        Python, JavaScript, React, Docker, Kubernetes, AWS, PostgreSQL,
        Redis, Machine Learning, Agile, Scrum

        CERTIFICATIONS
        AWS Certified Solutions Architect
        """

    def test_parse_resume(self, service, sample_resume):
        result = service.parse_resume(
            user_id="test_user",
            text_content=sample_resume
        )

        assert result["id"] is not None
        assert "skills" in result
        assert len(result["skills"]) > 0
        assert "years_of_experience" in result

    def test_extract_email(self, service, sample_resume):
        result = service.parse_resume(
            user_id="test_user",
            text_content=sample_resume
        )

        assert result["email"] == "john.smith@email.com"

    def test_extract_skills(self, service, sample_resume):
        result = service.parse_resume(
            user_id="test_user",
            text_content=sample_resume
        )

        skills_lower = [s.lower() for s in result["skills"]]
        assert "python" in skills_lower or "Python" in result["skills"]

    def test_get_skill_gaps(self, service, sample_resume):
        parsed = service.parse_resume(
            user_id="test_user",
            text_content=sample_resume
        )

        gaps = service.get_skill_gaps(parsed["id"], "software engineer")

        assert "target_role" in gaps
        assert "matching_skills" in gaps
        assert "missing_skills" in gaps
        assert "match_percentage" in gaps

    def test_empty_resume(self, service):
        result = service.parse_resume(
            user_id="test_user",
            text_content=""
        )

        assert result["id"] is not None
        assert result["confidence_score"] < 0.5


class TestProactiveSuggestionService:
    """Tests for proactive suggestion functionality"""

    @pytest.fixture
    def service(self):
        return ProactiveSuggestionService()

    def test_update_user_context(self, service):
        context = service.update_user_context(
            user_id="test_user",
            current_role="Software Developer",
            target_role="Engineering Manager",
            skills=["Python", "Leadership"],
            pending_decisions=2
        )

        assert context.user_id == "test_user"
        assert context.current_role == "Software Developer"
        assert context.target_role == "Engineering Manager"

    def test_generate_suggestions(self, service):
        service.update_user_context(
            user_id="test_user",
            target_role="Senior Developer",
            skills=["Python"],
            pending_decisions=1
        )

        suggestions = service.generate_suggestions("test_user")

        assert len(suggestions) > 0
        assert all("id" in s for s in suggestions)
        assert all("title" in s for s in suggestions)

    def test_dismiss_suggestion(self, service):
        service.update_user_context(
            user_id="test_user",
            pending_decisions=1
        )

        suggestions = service.generate_suggestions("test_user")
        if suggestions:
            success = service.dismiss_suggestion(
                "test_user",
                suggestions[0]["id"],
                "Not relevant"
            )
            assert success is True

    def test_get_suggestion_stats(self, service):
        service.generate_suggestions("test_user")

        stats = service.get_suggestion_stats("test_user")

        assert "total" in stats
        assert "acted_on" in stats
        assert "engagement_rate" in stats


class TestMonitoringDashboardService:
    """Tests for monitoring dashboard functionality"""

    @pytest.fixture
    def service(self):
        return MonitoringDashboardService()

    def test_record_request(self, service):
        service.record_request(
            endpoint="/api/test",
            method="GET",
            latency_ms=50.0,
            status_code=200
        )

        assert service.request_count >= 1

    def test_get_system_metrics(self, service):
        metrics = service.get_system_metrics()

        assert "cpu" in metrics
        assert "memory" in metrics
        assert "disk" in metrics

    def test_get_application_metrics(self, service):
        service.record_request("/api/test", "GET", 100.0, 200)

        metrics = service.get_application_metrics()

        assert "uptime" in metrics
        assert "requests" in metrics
        assert "latency" in metrics

    def test_check_health(self, service):
        health = service.check_health()

        assert "status" in health
        assert "checks" in health
        assert health["status"] in ["healthy", "degraded", "unhealthy"]

    def test_get_dashboard_summary(self, service):
        dashboard = service.get_dashboard_summary()

        assert "health" in dashboard
        assert "application" in dashboard
        assert "system" in dashboard

    def test_add_and_get_alerts(self, service):
        service.add_alert(
            severity="warning",
            title="Test Alert",
            message="This is a test alert"
        )

        alerts = service.get_active_alerts()
        assert len(alerts) >= 1
        assert alerts[0]["title"] == "Test Alert"

    def test_acknowledge_alert(self, service):
        service.add_alert(
            severity="info",
            title="Ack Test Alert",
            message="Test"
        )

        alerts = service.get_active_alerts()
        if alerts:
            success = service.acknowledge_alert(alerts[0]["id"])
            assert success is True


class TestGoogleCalendarSyncService:
    """Tests for calendar sync functionality"""

    @pytest.fixture
    def service(self):
        return GoogleCalendarSyncService()

    def test_get_connection_status_disconnected(self, service):
        status = service.get_connection_status("new_user")

        assert status["connected"] is False
        assert "oauth_url" in status

    def test_create_event(self, service):
        result = service.create_event(
            user_id="test_user",
            event_type=CalendarEventType.DECISION_DEADLINE,
            title="Test Decision",
            start_time=datetime.utcnow() + timedelta(days=7)
        )

        assert "event_id" in result
        assert result["title"] == "Test Decision"

    def test_create_decision_deadline(self, service):
        result = service.create_decision_deadline(
            user_id="test_user",
            decision_title="Accept Job Offer",
            deadline=datetime.utcnow() + timedelta(days=3)
        )

        assert "event_id" in result
        assert "Accept Job Offer" in result["title"]

    def test_create_follow_up(self, service):
        result = service.create_follow_up(
            user_id="test_user",
            decision_title="Career Change",
            follow_up_date=datetime.utcnow() + timedelta(days=30)
        )

        assert "event_id" in result

    def test_get_upcoming_events(self, service):
        service.create_event(
            user_id="test_user",
            event_type=CalendarEventType.CHECK_IN,
            title="Weekly Check-in",
            start_time=datetime.utcnow() + timedelta(days=3)
        )

        events = service.get_upcoming_events("test_user", days=7)
        assert len(events) >= 1

    def test_get_today_agenda(self, service):
        agenda = service.get_today_agenda("test_user")

        assert "date" in agenda
        assert "event_count" in agenda
        assert "summary" in agenda

    def test_update_sync_settings(self, service):
        service.connect_calendar(
            user_id="test_user",
            access_token="test_token",
            refresh_token="test_refresh",
            token_expiry=datetime.utcnow() + timedelta(hours=1)
        )

        result = service.update_sync_settings(
            user_id="test_user",
            auto_create=False
        )

        assert result["updated"] is True
        assert result["auto_create"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
