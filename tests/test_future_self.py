import pytest
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.future_self_service import FutureSelfService

class TestFutureSelfService:
    @pytest.fixture
    def service(self):
        return FutureSelfService()

    def test_start_conversation(self, service, sample_user_id):
        result = service.start_conversation(
            user_id=sample_user_id,
            decision_type="career_switch",
            decision_desc="Considering switching to product management",
            timeframe="10_years",
            scenario="optimistic"
        )

        assert "session_id" in result
        assert "opening_message" in result
        assert "persona" in result
        assert "suggested_questions" in result

    def test_persona_generation(self, service, sample_user_id):
        result = service.start_conversation(
            user_id=sample_user_id,
            decision_type="job_change",
            decision_desc="Moving to a new company",
            timeframe="5_years",
            scenario="realistic"
        )

        persona = result["persona"]

        assert "timeframe" in persona
        assert "traits" in persona
        assert "wisdom" in persona
        assert "achievements" in persona
        assert len(persona["traits"]) > 0
        assert len(persona["wisdom"]) > 0

    def test_send_message(self, service, sample_user_id):
        start_result = service.start_conversation(
            user_id=sample_user_id,
            decision_type="promotion",
            decision_desc="Considering a management role",
            timeframe="5_years",
            scenario="optimistic"
        )

        session_id = start_result["session_id"]

        message_result = service.send_message(
            session_id=session_id,
            message="What was the hardest part of the journey?"
        )

        assert "response" in message_result
        assert len(message_result["response"]) > 0

    def test_conversation_history(self, service, sample_user_id):
        start_result = service.start_conversation(
            user_id=sample_user_id,
            decision_type="entrepreneurship",
            decision_desc="Starting my own business",
            timeframe="10_years",
            scenario="optimistic"
        )

        session_id = start_result["session_id"]

        service.send_message(session_id, "Do you have any regrets?")
        service.send_message(session_id, "Was it worth it?")

        session = service.get_session(session_id)

        assert session is not None
        assert "history" in session
        assert len(session["history"]) >= 2

    def test_end_conversation(self, service, sample_user_id):
        start_result = service.start_conversation(
            user_id=sample_user_id,
            decision_type="relocation",
            decision_desc="Moving to a new city",
            timeframe="5_years",
            scenario="realistic"
        )

        session_id = start_result["session_id"]

        service.send_message(session_id, "Was it a good decision?")

        end_result = service.end_conversation(session_id)

        assert "closing" in end_result or "session_id" in end_result

    def test_different_scenarios(self, service, sample_user_id):
        scenarios = ["optimistic", "realistic", "pessimistic"]
        results = []

        for scenario in scenarios:
            result = service.start_conversation(
                user_id=f"{sample_user_id}_{scenario}",
                decision_type="career_switch",
                decision_desc="Changing careers",
                timeframe="10_years",
                scenario=scenario
            )
            results.append(result)

        assert all("session_id" in r for r in results)

    def test_different_timeframes(self, service, sample_user_id):
        timeframes = ["5_years", "10_years", "15_years"]

        for timeframe in timeframes:
            result = service.start_conversation(
                user_id=f"{sample_user_id}_{timeframe}",
                decision_type="education",
                decision_desc="Going back to school",
                timeframe=timeframe,
                scenario="realistic"
            )

            assert result["persona"]["timeframe"] == timeframe

    def test_invalid_session(self, service):
        result = service.send_message(
            session_id="invalid_session_id",
            message="Hello?"
        )

        assert "error" in result

    def test_message_context_awareness(self, service, sample_user_id):
        start_result = service.start_conversation(
            user_id=sample_user_id,
            decision_type="job_change",
            decision_desc="Changing jobs for better pay",
            timeframe="5_years",
            scenario="optimistic"
        )

        session_id = start_result["session_id"]

        response = service.send_message(session_id, "Do you have any regrets?")

        assert "response" in response
        assert len(response["response"]) > 10


class TestFutureSelfPersona:
    @pytest.fixture
    def service(self):
        return FutureSelfService()

    def test_persona_has_traits(self, service, sample_user_id):
        result = service.start_conversation(
            user_id=sample_user_id,
            decision_type="career_switch",
            decision_desc="Test decision",
            timeframe="10_years",
            scenario="optimistic"
        )

        persona = result["persona"]
        assert len(persona["traits"]) >= 3

    def test_persona_has_wisdom(self, service, sample_user_id):
        result = service.start_conversation(
            user_id=sample_user_id,
            decision_type="career_switch",
            decision_desc="Test decision",
            timeframe="10_years",
            scenario="realistic"
        )

        persona = result["persona"]
        assert len(persona["wisdom"]) >= 2

    def test_opening_message_exists(self, service, sample_user_id):
        result = service.start_conversation(
            user_id=sample_user_id,
            decision_type="career_switch",
            decision_desc="Moving from engineering to product management",
            timeframe="10_years",
            scenario="optimistic"
        )

        assert len(result["opening_message"]) > 20
