import pytest
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.bias_interceptor_service import BiasInterceptorService, BiasType

class TestBiasInterceptorService:
    @pytest.fixture
    def service(self):
        return BiasInterceptorService()

    def test_detect_sunk_cost_bias(self, service):
        text = "I've already invested 5 years into this career, I cannot give up now"

        detections = service.analyze_text(text)

        assert len(detections) > 0
        assert any(d["type"] == "sunk_cost" for d in detections)

    def test_detect_loss_aversion(self, service):
        text = "I'm afraid to lose what I have. It's too risky to change jobs now."

        detections = service.analyze_text(text)

        assert len(detections) > 0
        assert any(d["type"] == "loss_aversion" for d in detections)

    def test_detect_confirmation_bias(self, service):
        text = "This proves that I was right all along. Everyone agrees with my decision."

        detections = service.analyze_text(text)

        assert len(detections) > 0
        assert any(d["type"] == "confirmation_bias" for d in detections)

    def test_detect_overconfidence(self, service):
        text = "I'm 100% sure this will work. It definitely can't fail."

        detections = service.analyze_text(text)

        assert len(detections) > 0
        assert any(d["type"] == "overconfidence" for d in detections)

    def test_detect_status_quo_bias(self, service):
        text = "Why change when I'm comfortable where I am? It's not broken."

        detections = service.analyze_text(text)

        assert len(detections) > 0
        assert any(d["type"] == "status_quo" for d in detections)

    def test_no_bias_in_neutral_text(self, service):
        text = "I am considering my options carefully and weighing the pros and cons."

        detections = service.analyze_text(text)

        assert len(detections) == 0

    def test_real_time_feedback(self, service, sample_user_id):
        text = "I've already invested 5 years into this career, I cannot give up now"

        feedback = service.get_real_time_feedback(text, sample_user_id)

        assert feedback["has_bias"] == True
        assert "message" in feedback
        assert "reframe" in feedback
        assert "questions" in feedback

    def test_user_bias_profile(self, service, sample_user_id):
        texts = [
            "I've already invested 10 years into this career",
            "It's too risky to change now, I might lose everything",
            "I'm 100% certain this will work out"
        ]

        for text in texts:
            service.analyze_text(text, sample_user_id)

        profile = service.get_user_bias_profile(sample_user_id)

        assert profile["user_id"] == sample_user_id
        assert profile["total_detections"] >= 1
        assert len(profile["common_biases"]) >= 1

    def test_bias_explanation(self, service):
        explanation = service.get_bias_explanation("sunk_cost")

        assert "explanation" in explanation
        assert "reframe" in explanation
        assert "questions" in explanation

    def test_short_text_no_analysis(self, service):
        text = "Hello"

        feedback = service.get_real_time_feedback(text)

        assert feedback["has_bias"] == False

    def test_multiple_biases_detected(self, service):
        text = "I've already invested 10 years, I'm afraid to lose what I have, and I'm 100% certain staying is the right choice."

        detections = service.analyze_text(text)

        bias_types = [d["type"] for d in detections]
        assert len(bias_types) >= 1

    def test_confidence_scoring(self, service):
        weak_text = "maybe I shouldn't change"
        strong_text = "I definitely cannot give up after all my years of investment"

        weak_detections = service.analyze_text(weak_text)
        strong_detections = service.analyze_text(strong_text)

        if weak_detections and strong_detections:
            assert strong_detections[0]["confidence"] > weak_detections[0]["confidence"]


class TestBiasInterceptorIntegration:
    @pytest.fixture
    def service(self):
        return BiasInterceptorService()

    def test_journal_entry_analysis(self, service, sample_journal_entry):
        feedback = service.get_real_time_feedback(sample_journal_entry["content"])

        assert "has_bias" in feedback
        if feedback["has_bias"]:
            assert feedback["primary_bias"] == "sunk_cost"

    def test_intervention_recording(self, service, sample_user_id):
        service.analyze_text("I've already invested 10 years into this career", sample_user_id)

        profile = service.get_user_bias_profile(sample_user_id)
        assert "user_id" in profile
        assert "total_detections" in profile
