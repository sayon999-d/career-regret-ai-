from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import random
import asyncio
import hashlib

class OpportunityType(str, Enum):
    JOB = "job"
    SKILL = "skill"
    NETWORK = "network"
    EDUCATION = "education"
    MARKET_TREND = "market_trend"
    SALARY_INSIGHT = "salary_insight"

class MatchConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class UserProfile:
    user_id: str
    current_role: str
    industry: str
    skills: List[str]
    interests: List[str]
    risk_tolerance: float
    salary_target: float
    location_preferences: List[str]
    low_regret_factors: List[str]
    career_goals: List[str]
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class Opportunity:
    id: str
    type: OpportunityType
    title: str
    description: str
    source: str
    match_score: float
    match_reasons: List[str]
    potential_regret_score: float
    action_items: List[str]
    expires_at: Optional[datetime]
    discovered_at: datetime = field(default_factory=datetime.utcnow)
    viewed: bool = False
    saved: bool = False
    dismissed: bool = False

@dataclass
class ScoutAlert:
    id: str
    user_id: str
    opportunities: List[Opportunity]
    priority: str
    message: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    read: bool = False

class OpportunityScoutService:
    MARKET_TRENDS = [
        {"name": "AI/ML Engineering", "growth": 0.35, "demand": "very high"},
        {"name": "Cloud Architecture", "growth": 0.28, "demand": "high"},
        {"name": "Data Science", "growth": 0.22, "demand": "high"},
        {"name": "Cybersecurity", "growth": 0.32, "demand": "very high"},
        {"name": "Product Management", "growth": 0.18, "demand": "high"},
        {"name": "DevOps/SRE", "growth": 0.25, "demand": "high"},
        {"name": "UX Design", "growth": 0.15, "demand": "medium"},
        {"name": "Blockchain", "growth": 0.12, "demand": "medium"},
    ]

    SKILL_RECOMMENDATIONS = {
        "technology": ["Python", "Cloud (AWS/GCP)", "Machine Learning", "System Design"],
        "finance": ["Python", "Risk Modeling", "Data Analysis", "Regulatory Compliance"],
        "healthcare": ["Data Analytics", "HIPAA Compliance", "EHR Systems", "Telehealth"],
        "consulting": ["Strategy", "Data Visualization", "Client Management", "Industry Expertise"],
    }

    SALARY_BENCHMARKS = {
        "software_engineer": {"junior": 80000, "mid": 120000, "senior": 160000, "staff": 220000},
        "data_scientist": {"junior": 85000, "mid": 130000, "senior": 175000, "staff": 240000},
        "product_manager": {"junior": 90000, "mid": 135000, "senior": 180000, "director": 250000},
        "engineering_manager": {"mid": 150000, "senior": 200000, "director": 280000},
    }

    def __init__(self):
        self.user_profiles: Dict[str, UserProfile] = {}
        self.opportunities: Dict[str, List[Opportunity]] = {}
        self.alerts: Dict[str, List[ScoutAlert]] = {}
        self.applications: Dict[str, List[Dict]] = {}
        self.scan_history: Dict[str, datetime] = {}
        self._is_scanning = False

    def register_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> UserProfile:
        profile = UserProfile(
            user_id=user_id,
            current_role=profile_data.get('current_role', 'Software Engineer'),
            industry=profile_data.get('industry', 'technology'),
            skills=profile_data.get('skills', []),
            interests=profile_data.get('interests', []),
            risk_tolerance=profile_data.get('risk_tolerance', 0.5),
            salary_target=profile_data.get('salary_target', 150000),
            location_preferences=profile_data.get('locations', ['Remote']),
            low_regret_factors=profile_data.get('low_regret_factors', ['growth', 'stability']),
            career_goals=profile_data.get('career_goals', ['leadership', 'expertise'])
        )
        self.user_profiles[user_id] = profile
        return profile

    async def scan_opportunities(self, user_id: str) -> List[Opportunity]:
        if user_id not in self.user_profiles:
            return []

        profile = self.user_profiles[user_id]
        opportunities = []

        opportunities.extend(self._scan_job_market(profile))
        opportunities.extend(self._scan_skill_gaps(profile))
        opportunities.extend(self._scan_market_trends(profile))
        opportunities.extend(self._scan_salary_insights(profile))
        opportunities.extend(self._scan_network_opportunities(profile))

        opportunities.sort(key=lambda x: x.match_score, reverse=True)

        if user_id not in self.opportunities:
            self.opportunities[user_id] = []

        existing_titles = {o.title for o in self.opportunities[user_id]}
        for o in opportunities:
            if o.title not in existing_titles:
                self.opportunities[user_id].append(o)

        self.scan_history[user_id] = datetime.utcnow()
        if opportunities:
            self._create_alert(user_id, opportunities[:5])

        return opportunities[:10]

    def _scan_job_market(self, profile: UserProfile) -> List[Opportunity]:
        opportunities = []
        job_templates = [
            {"title": f"Senior {profile.current_role}", "company": "Growth Startup", "match": 0.85},
            {"title": f"Staff {profile.current_role}", "company": "FAANG Company", "match": 0.78},
            {"title": f"{profile.current_role} Lead", "company": "Mid-size Tech", "match": 0.82},
            {"title": f"Principal {profile.current_role}", "company": "Enterprise Corp", "match": 0.75},
        ]

        for job in job_templates:
            if random.random() > 0.3:
                opp_id = hashlib.md5(f"{job['title']}{datetime.utcnow()}".encode()).hexdigest()[:12]
                regret_score = self._calculate_regret_score(profile, job)
                opportunities.append(Opportunity(
                    id=opp_id,
                    type=OpportunityType.JOB,
                    title=f"{job['title']} at {job['company']}",
                    description=f"Role matching your {', '.join(profile.skills[:3])} skills",
                    source="Job Market Analysis",
                    match_score=job['match'] * (1 - regret_score/100),
                    match_reasons=self._get_match_reasons(profile, job),
                    potential_regret_score=regret_score,
                    action_items=["Update resume", "Research company culture", "Prepare for interviews"],
                    expires_at=datetime.utcnow() + timedelta(days=30)
                ))
        return opportunities

    def _scan_skill_gaps(self, profile: UserProfile) -> List[Opportunity]:
        opportunities = []
        industry_skills = self.SKILL_RECOMMENDATIONS.get(profile.industry, ["Python", "Data Analysis"])
        gap_skills = [s for s in industry_skills if s not in profile.skills]

        for skill in gap_skills[:2]:
            opp_id = hashlib.md5(f"skill_{skill}{profile.user_id}".encode()).hexdigest()[:12]
            opportunities.append(Opportunity(
                id=opp_id,
                type=OpportunityType.SKILL,
                title=f"Skill Gap: {skill}",
                description=f"Learning {skill} could increase your market value by 15-25%",
                source="Skills Analysis",
                match_score=0.75,
                match_reasons=[f"High demand skill in {profile.industry}", "Aligns with your career goals"],
                potential_regret_score=15,
                action_items=[f"Take an online course in {skill}", f"Build a project using {skill}"],
                expires_at=None
            ))
        return opportunities

    def _scan_market_trends(self, profile: UserProfile) -> List[Opportunity]:
        opportunities = []
        relevant_trends = [t for t in self.MARKET_TRENDS if t['demand'] in ['high', 'very high']]
        for trend in relevant_trends[:2]:
            opp_id = hashlib.md5(f"trend_{trend['name']}".encode()).hexdigest()[:12]
            opportunities.append(Opportunity(
                id=opp_id,
                type=OpportunityType.MARKET_TREND,
                title=f"Growing Field: {trend['name']}",
                description=f"{int(trend['growth']*100)}% growth rate, {trend['demand']} demand",
                source="Market Intelligence",
                match_score=0.70 + trend['growth']/2,
                match_reasons=[f"High growth potential", f"{trend['demand'].title()} market demand"],
                potential_regret_score=20,
                action_items=[f"Research {trend['name']} fundamentals"],
                expires_at=datetime.utcnow() + timedelta(days=90)
            ))
        return opportunities

    def _scan_salary_insights(self, profile: UserProfile) -> List[Opportunity]:
        opportunities = []
        role_key = profile.current_role.lower().replace(' ', '_')
        benchmarks = self.SALARY_BENCHMARKS.get(role_key, self.SALARY_BENCHMARKS['software_engineer'])
        current_estimate = benchmarks.get('mid', 120000)

        if profile.salary_target > current_estimate:
            gap = profile.salary_target - current_estimate
            opp_id = hashlib.md5(f"salary_{profile.user_id}".encode()).hexdigest()[:12]
            opportunities.append(Opportunity(
                id=opp_id,
                type=OpportunityType.SALARY_INSIGHT,
                title=f"Salary Growth Path: +${gap:,.0f}",
                description=f"Reaching ${profile.salary_target:,.0f} is achievable within 2-3 years",
                source="Compensation Analysis",
                match_score=0.80,
                match_reasons=[f"Market demand is in your favor"],
                potential_regret_score=10,
                action_items=["Research compensation at target companies", "Build negotiation skills"],
                expires_at=None
            ))
        return opportunities

    def _scan_network_opportunities(self, profile: UserProfile) -> List[Opportunity]:
        opportunities = []
        if random.random() > 0.6:
            opp_id = hashlib.md5(f"network_{profile.user_id}{datetime.utcnow()}".encode()).hexdigest()[:12]
            opportunities.append(Opportunity(
                id=opp_id,
                type=OpportunityType.NETWORK,
                title="Networking Opportunity: Industry Event",
                description=f"Connect with leaders in {profile.industry}",
                source="Professional Network Analysis",
                match_score=0.65,
                match_reasons=["Expands your professional network", "Access to hidden job market"],
                potential_regret_score=5,
                action_items=["Prepare elevator pitch", "Set networking goals"],
                expires_at=datetime.utcnow() + timedelta(days=14)
            ))
        return opportunities

    def _calculate_regret_score(self, profile: UserProfile, job: Dict) -> float:
        base_score = 30
        if 'growth' in profile.low_regret_factors: base_score -= 10
        if 'stability' in profile.low_regret_factors and 'Startup' in job.get('company', ''): base_score += 15
        if profile.risk_tolerance > 0.7: base_score -= 10
        elif profile.risk_tolerance < 0.3: base_score += 10
        return max(5, min(80, base_score))

    def _get_match_reasons(self, profile: UserProfile, job: Dict) -> List[str]:
        reasons = [f"Matches your {profile.current_role} experience", f"Aligns with {profile.industry} industry expertise"]
        if profile.skills: reasons.append(f"Leverages your {profile.skills[0]} skills")
        return reasons

    def _create_alert(self, user_id: str, opportunities: List[Opportunity]):
        alert_id = hashlib.md5(f"alert_{user_id}{datetime.utcnow()}".encode()).hexdigest()[:12]
        high_match = [o for o in opportunities if o.match_score > 0.75]
        priority = "high" if len(high_match) >= 2 else "medium" if high_match else "low"
        alert = ScoutAlert(id=alert_id, user_id=user_id, opportunities=opportunities, priority=priority, message=f"Found {len(opportunities)} opportunities matching your profile")
        if user_id not in self.alerts: self.alerts[user_id] = []
        self.alerts[user_id].append(alert)

    def get_opportunities(self, user_id: str, type_filter: str = None) -> List[Dict]:
        if user_id not in self.opportunities: return []
        opps = self.opportunities[user_id]
        if type_filter: opps = [o for o in opps if o.type.value == type_filter]
        return [self._opp_to_dict(o) for o in opps if not o.dismissed]

    def get_alerts(self, user_id: str, unread_only: bool = True) -> List[Dict]:
        if user_id not in self.alerts: return []
        alerts = self.alerts[user_id]
        if unread_only: alerts = [a for a in alerts if not a.read]
        return [{"id": a.id, "priority": a.priority, "message": a.message, "opportunity_count": len(a.opportunities), "created_at": a.created_at.isoformat(), "read": a.read} for a in alerts]

    def apply_for_opportunity(self, user_id: str, opp_id: str, notes: str = "") -> Dict:
        if user_id not in self.opportunities: return {"success": False, "error": "No opportunities found"}
        target_opp = next((o for o in self.opportunities[user_id] if o.id == opp_id), None)
        if not target_opp: return {"success": False, "error": "Opportunity not found"}
        application = {
            "application_id": hashlib.md5(f"app_{opp_id}{datetime.utcnow()}".encode()).hexdigest()[:8],
            "opportunity_id": opp_id,
            "title": target_opp.title,
            "status": "applied",
            "applied_at": datetime.utcnow().isoformat(),
            "notes": notes,
            "next_step": target_opp.action_items[0] if target_opp.action_items else "Wait for feedback"
        }
        if user_id not in self.applications: self.applications[user_id] = []
        self.applications[user_id].append(application)
        return {"success": True, "application": application}

    def get_applications(self, user_id: str) -> List[Dict]:
        return self.applications.get(user_id, [])

    def mark_opportunity(self, user_id: str, opp_id: str, action: str) -> bool:
        if user_id not in self.opportunities: return False
        for opp in self.opportunities[user_id]:
            if opp.id == opp_id:
                if action == "save": opp.saved = True
                elif action == "unsave": opp.saved = False
                elif action == "dismiss": opp.dismissed = True
                elif action == "view": opp.viewed = True
                return True
        return False

    def get_saved_opportunities(self, user_id: str) -> List[Dict]:
        if user_id not in self.opportunities: return []
        return [self._opp_to_dict(o) for o in self.opportunities[user_id] if o.saved and not o.dismissed]

    def get_scout_summary(self, user_id: str) -> Dict[str, Any]:
        if user_id not in self.user_profiles: return {"error": "Profile not found"}
        opps = self.opportunities.get(user_id, [])
        alerts = self.alerts.get(user_id, [])
        return {
            "user_id": user_id,
            "profile_active": True,
            "last_scan": self.scan_history.get(user_id, datetime.utcnow()).isoformat(),
            "total_opportunities": len([o for o in opps if not o.dismissed]),
            "saved_opportunities": len([o for o in opps if o.saved]),
            "unread_alerts": len([a for a in alerts if not a.read]),
            "top_opportunity": self._opp_to_dict(opps[0]) if opps else None,
            "market_outlook": "positive" if any(t['growth'] > 0.2 for t in self.MARKET_TRENDS) else "stable"
        }

    def _opp_to_dict(self, o: Opportunity) -> Dict:
        return {
            "id": o.id, "type": o.type.value, "title": o.title, "description": o.description,
            "source": o.source, "match_score": round(o.match_score, 2), "match_reasons": o.match_reasons,
            "regret_score": o.potential_regret_score, "action_items": o.action_items,
            "discovered_at": o.discovered_at.isoformat(), "saved": o.saved, "viewed": o.viewed
        }

opportunity_scout = OpportunityScoutService()
