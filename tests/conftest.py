import pytest
import asyncio
from typing import Generator, AsyncGenerator
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def sample_user_id():
    return "test_user_123"

@pytest.fixture
def sample_decision():
    return {
        "decision_type": "job_change",
        "description": "Considering switching from Company A to Company B",
        "predicted_regret": 35.0,
        "factors": ["salary", "growth", "culture"]
    }

@pytest.fixture
def sample_journal_entry():
    return {
        "title": "Career Decision Reflection",
        "content": "I've already invested 5 years into this company, I cannot give up now.",
        "emotions": ["anxious", "hopeful"],
        "decision_type": "job_change"
    }

@pytest.fixture
def sample_profile():
    return {
        "current_role": "Software Engineer",
        "industry": "technology",
        "skills": ["Python", "JavaScript", "Machine Learning"],
        "interests": ["AI", "Startups"],
        "risk_tolerance": 0.6,
        "salary_target": 180000,
        "locations": ["Remote", "San Francisco"],
        "career_goals": ["leadership", "expertise"]
    }
