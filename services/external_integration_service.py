from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import os
import json
import hashlib
import asyncio

class IntegrationType(str, Enum):
    LINKEDIN = "linkedin"
    INDEED = "indeed"
    GLASSDOOR = "glassdoor"
    NEWS = "news"
    CALENDAR = "calendar"

@dataclass
class IntegrationCredentials:
    integration_type: IntegrationType
    access_token: str
    refresh_token: Optional[str] = None
    expires_at: Optional[datetime] = None
    user_id: str = ""
    connected_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class LinkedInProfile:
    name: str
    headline: str
    current_company: str
    current_role: str
    industry: str
    skills: List[str]
    experience_years: int
    education: List[Dict]
    connections_count: int

@dataclass
class JobListing:
    id: str
    title: str
    company: str
    location: str
    salary_range: Optional[str]
    description: str
    posted_date: datetime
    source: str
    url: str
    match_score: float = 0

@dataclass
class NewsArticle:
    id: str
    title: str
    summary: str
    source: str
    url: str
    published_at: datetime
    relevance_score: float
    category: str

@dataclass
class CalendarEvent:
    id: str
    title: str
    description: str
    start_time: datetime
    end_time: datetime
    event_type: str
    reminder_set: bool = False

class ExternalIntegrationService:
    MOCK_JOBS = [
        {"title": "Senior Software Engineer", "company": "Tech Corp", "salary": "$150k-$180k", "location": "Remote"},
        {"title": "Staff Engineer", "company": "Startup Inc", "salary": "$180k-$220k", "location": "San Francisco"},
        {"title": "Engineering Manager", "company": "Big Tech", "salary": "$200k-$250k", "location": "New York"},
        {"title": "Principal Engineer", "company": "FAANG", "salary": "$250k-$350k", "location": "Seattle"},
        {"title": "Tech Lead", "company": "Unicorn Startup", "salary": "$170k-$200k", "location": "Austin"},
    ]

    MOCK_NEWS = [
        {"title": "AI Reshaping Tech Careers", "category": "technology", "source": "TechCrunch"},
        {"title": "Remote Work Trends 2026", "category": "workplace", "source": "Forbes"},
        {"title": "Salary Trends in Software Engineering", "category": "career", "source": "Bloomberg"},
        {"title": "Top Skills Employers Want", "category": "skills", "source": "LinkedIn News"},
        {"title": "Career Pivots: Success Stories", "category": "career", "source": "Harvard Business Review"},
    ]

    def __init__(self):
        self.credentials: Dict[str, Dict[IntegrationType, IntegrationCredentials]] = {}
        self.cached_data: Dict[str, Dict] = {}
        self.cache_expiry: Dict[str, datetime] = {}
        self.api_keys = {
            "linkedin": os.getenv("LINKEDIN_API_KEY", ""),
            "indeed": os.getenv("INDEED_API_KEY", ""),
            "news": os.getenv("NEWS_API_KEY", ""),
            "google_calendar": os.getenv("GOOGLE_CALENDAR_API_KEY", ""),
        }

    def get_oauth_url(self, integration_type: IntegrationType, user_id: str) -> str:
        state = hashlib.sha256(f"{user_id}{datetime.utcnow()}".encode()).hexdigest()[:16]

        oauth_urls = {
            IntegrationType.LINKEDIN: f"https://www.linkedin.com/oauth/v2/authorization?client_id=YOUR_CLIENT_ID&redirect_uri=YOUR_REDIRECT&state={state}&scope=r_liteprofile%20r_emailaddress",
            IntegrationType.CALENDAR: f"https://accounts.google.com/o/oauth2/auth?client_id=YOUR_CLIENT_ID&redirect_uri=YOUR_REDIRECT&state={state}&scope=https://www.googleapis.com/auth/calendar",
        }

        return oauth_urls.get(integration_type, "")

    async def handle_oauth_callback(
        self,
        integration_type: IntegrationType,
        user_id: str,
        auth_code: str
    ) -> Dict[str, Any]:
        creds = IntegrationCredentials(
            integration_type=integration_type,
            access_token=f"mock_token_{auth_code[:8]}",
            refresh_token=f"mock_refresh_{auth_code[:8]}",
            expires_at=datetime.utcnow() + timedelta(hours=1),
            user_id=user_id
        )

        if user_id not in self.credentials:
            self.credentials[user_id] = {}
        self.credentials[user_id][integration_type] = creds

        return {
            "success": True,
            "integration": integration_type.value,
            "connected_at": creds.connected_at.isoformat()
        }

    def is_connected(self, user_id: str, integration_type: IntegrationType) -> bool:
        if user_id not in self.credentials:
            return False
        if integration_type not in self.credentials[user_id]:
            return False
        creds = self.credentials[user_id][integration_type]
        if creds.expires_at and creds.expires_at < datetime.utcnow():
            return False
        return True

    def get_connected_integrations(self, user_id: str) -> List[str]:
        if user_id not in self.credentials:
            return []
        return [
            it.value for it in self.credentials[user_id].keys()
            if self.is_connected(user_id, it)
        ]

    async def fetch_linkedin_profile(self, user_id: str) -> Optional[LinkedInProfile]:
        if not self.is_connected(user_id, IntegrationType.LINKEDIN):
            return LinkedInProfile(
                name="Demo User",
                headline="Software Engineer at Tech Company",
                current_company="Tech Company",
                current_role="Senior Software Engineer",
                industry="Technology",
                skills=["Python", "JavaScript", "Machine Learning", "Cloud Computing", "System Design"],
                experience_years=5,
                education=[{"school": "University", "degree": "B.S. Computer Science", "year": 2019}],
                connections_count=500
            )

        return LinkedInProfile(
            name="Demo User",
            headline="Software Engineer at Tech Company",
            current_company="Tech Company",
            current_role="Senior Software Engineer",
            industry="Technology",
            skills=["Python", "JavaScript", "Machine Learning", "Cloud Computing", "System Design"],
            experience_years=5,
            education=[{"school": "University", "degree": "B.S. Computer Science", "year": 2019}],
            connections_count=500
        )

    async def search_jobs(
        self,
        user_id: str,
        query: str = "",
        location: str = "",
        salary_min: int = 0,
        limit: int = 10
    ) -> List[JobListing]:
        jobs = []

        for i, job_data in enumerate(self.MOCK_JOBS[:limit]):
            job = JobListing(
                id=f"job_{hashlib.md5(job_data['title'].encode()).hexdigest()[:8]}",
                title=job_data["title"],
                company=job_data["company"],
                location=job_data["location"],
                salary_range=job_data["salary"],
                description=f"Exciting opportunity at {job_data['company']}. We're looking for talented engineers...",
                posted_date=datetime.utcnow() - timedelta(days=i),
                source="Indeed",
                url=f"https://indeed.com/job/{i}",
                match_score=0.9 - (i * 0.1)
            )
            jobs.append(job)

        return jobs

    async def fetch_industry_news(
        self,
        industry: str = "technology",
        categories: List[str] = None,
        limit: int = 10
    ) -> List[NewsArticle]:
        articles = []

        for i, news_data in enumerate(self.MOCK_NEWS[:limit]):
            article = NewsArticle(
                id=f"news_{hashlib.md5(news_data['title'].encode()).hexdigest()[:8]}",
                title=news_data["title"],
                summary=f"Read about {news_data['title'].lower()} and how it affects your career decisions.",
                source=news_data["source"],
                url=f"https://news.example.com/{i}",
                published_at=datetime.utcnow() - timedelta(hours=i * 2),
                relevance_score=0.95 - (i * 0.1),
                category=news_data["category"]
            )
            articles.append(article)

        return articles

    async def get_calendar_events(
        self,
        user_id: str,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> List[CalendarEvent]:
        if not start_date:
            start_date = datetime.utcnow()
        if not end_date:
            end_date = start_date + timedelta(days=30)

        events = [
            CalendarEvent(
                id="event_1",
                title="Decision Review: Career Path",
                description="Review the job change decision made last month",
                start_time=datetime.utcnow() + timedelta(days=7),
                end_time=datetime.utcnow() + timedelta(days=7, hours=1),
                event_type="decision_review",
                reminder_set=True
            ),
            CalendarEvent(
                id="event_2",
                title="Skill Assessment",
                description="Quarterly skill assessment and learning plan review",
                start_time=datetime.utcnow() + timedelta(days=14),
                end_time=datetime.utcnow() + timedelta(days=14, hours=1),
                event_type="assessment",
                reminder_set=True
            )
        ]

        return events

    async def create_decision_reminder(
        self,
        user_id: str,
        decision_id: str,
        reminder_date: datetime,
        title: str,
        description: str = ""
    ) -> CalendarEvent:
        event = CalendarEvent(
            id=f"reminder_{decision_id}",
            title=title,
            description=description or f"Follow up on decision {decision_id}",
            start_time=reminder_date,
            end_time=reminder_date + timedelta(hours=1),
            event_type="decision_reminder",
            reminder_set=True
        )

        return event

    async def sync_profile_from_linkedin(self, user_id: str) -> Dict[str, Any]:
        profile = await self.fetch_linkedin_profile(user_id)

        if not profile:
            return {"error": "Could not fetch LinkedIn profile"}

        return {
            "success": True,
            "profile": {
                "name": profile.name,
                "current_role": profile.current_role,
                "industry": profile.industry,
                "skills": profile.skills,
                "experience_years": profile.experience_years
            }
        }

    def disconnect_integration(self, user_id: str, integration_type: IntegrationType) -> bool:
        if user_id in self.credentials and integration_type in self.credentials[user_id]:
            del self.credentials[user_id][integration_type]
            return True
        return False

    def get_integration_status(self, user_id: str) -> Dict[str, Any]:
        connected = self.get_connected_integrations(user_id)

        return {
            "user_id": user_id,
            "connected_integrations": connected,
            "available_integrations": [it.value for it in IntegrationType],
            "api_keys_configured": {
                key: bool(value) for key, value in self.api_keys.items()
            }
        }

external_integration_service = ExternalIntegrationService()
