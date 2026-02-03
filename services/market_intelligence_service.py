from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import random

@dataclass
class SalaryData:

    role: str
    location: str
    experience_years: int
    percentile_25: float
    percentile_50: float
    percentile_75: float
    percentile_90: float
    currency: str = "USD"
    last_updated: datetime = field(default_factory=datetime.utcnow)

@dataclass
class IndustryTrend:

    industry: str
    trend_direction: str
    growth_rate: float
    hiring_activity: str
    emerging_roles: List[str]
    declining_roles: List[str]
    key_skills: List[str]
    last_updated: datetime = field(default_factory=datetime.utcnow)

@dataclass
class JobMarketHealth:

    industry: str
    location: str
    demand_score: float
    supply_score: float
    competition_level: str
    avg_time_to_hire_days: int
    remote_friendly_pct: float
    salary_trend: str

@dataclass
class SkillDemand:

    skill: str
    demand_score: float
    trend: str
    avg_salary_premium: float
    related_roles: List[str]

class MarketIntelligenceService:


    BASE_SALARIES = {
        'software_engineer': {'entry': 85000, 'mid': 130000, 'senior': 175000, 'lead': 220000},
        'product_manager': {'entry': 90000, 'mid': 140000, 'senior': 180000, 'lead': 230000},
        'data_scientist': {'entry': 95000, 'mid': 145000, 'senior': 190000, 'lead': 240000},
        'ux_designer': {'entry': 70000, 'mid': 100000, 'senior': 140000, 'lead': 175000},
        'devops_engineer': {'entry': 90000, 'mid': 135000, 'senior': 175000, 'lead': 210000},
        'data_analyst': {'entry': 65000, 'mid': 90000, 'senior': 120000, 'lead': 150000},
        'project_manager': {'entry': 70000, 'mid': 95000, 'senior': 130000, 'lead': 160000},
        'marketing_manager': {'entry': 60000, 'mid': 85000, 'senior': 120000, 'lead': 160000},
        'sales_manager': {'entry': 65000, 'mid': 100000, 'senior': 150000, 'lead': 200000},
        'hr_manager': {'entry': 55000, 'mid': 80000, 'senior': 110000, 'lead': 145000},
        'finance_analyst': {'entry': 70000, 'mid': 100000, 'senior': 140000, 'lead': 180000},
        'frontend_developer': {'entry': 75000, 'mid': 115000, 'senior': 155000, 'lead': 195000},
        'backend_developer': {'entry': 80000, 'mid': 125000, 'senior': 170000, 'lead': 215000},
        'machine_learning_engineer': {'entry': 105000, 'mid': 160000, 'senior': 210000, 'lead': 270000},
        'cloud_architect': {'entry': 110000, 'mid': 165000, 'senior': 215000, 'lead': 275000}
    }

    LOCATION_MULTIPLIERS = {
        'san_francisco': 1.35,
        'new_york': 1.25,
        'seattle': 1.20,
        'boston': 1.15,
        'los_angeles': 1.12,
        'austin': 1.05,
        'denver': 1.00,
        'chicago': 0.98,
        'atlanta': 0.95,
        'remote_us': 0.92,
        'london': 0.95,
        'berlin': 0.75,
        'bangalore': 0.35,
        'singapore': 0.85,
        'toronto': 0.80,
        'sydney': 0.90
    }

    INDUSTRY_TRENDS = {
        'technology': {
            'trend': 'growing', 'growth_rate': 0.08,
            'hiring': 'high',
            'emerging': ['AI/ML Engineer', 'Cloud Security', 'DevOps', 'Data Engineer'],
            'declining': ['System Admin', 'Network Technician'],
            'skills': ['Python', 'Cloud', 'AI/ML', 'Kubernetes', 'React']
        },
        'fintech': {
            'trend': 'growing', 'growth_rate': 0.12,
            'hiring': 'high',
            'emerging': ['Blockchain Developer', 'Risk Analyst', 'Compliance Tech'],
            'declining': ['Traditional Banking Roles'],
            'skills': ['Python', 'Blockchain', 'Risk Modeling', 'RegTech']
        },
        'healthcare': {
            'trend': 'growing', 'growth_rate': 0.06,
            'hiring': 'high',
            'emerging': ['Health Informatics', 'Telemedicine', 'AI Diagnostics'],
            'declining': ['Manual Records Management'],
            'skills': ['Healthcare IT', 'HIPAA', 'Data Analytics', 'Telehealth']
        },
        'retail': {
            'trend': 'stable', 'growth_rate': 0.02,
            'hiring': 'moderate',
            'emerging': ['E-commerce', 'Supply Chain Analyst', 'Customer Data'],
            'declining': ['Traditional Retail', 'Store Management'],
            'skills': ['E-commerce', 'Supply Chain', 'Customer Analytics']
        },
        'manufacturing': {
            'trend': 'stable', 'growth_rate': 0.01,
            'hiring': 'moderate',
            'emerging': ['Automation Engineer', 'IoT Specialist', 'Robotics'],
            'declining': ['Manual Assembly', 'Traditional QA'],
            'skills': ['Automation', 'IoT', 'Lean Manufacturing', 'PLC']
        },
        'consulting': {
            'trend': 'growing', 'growth_rate': 0.05,
            'hiring': 'high',
            'emerging': ['Digital Transformation', 'AI Strategy', 'ESG Consulting'],
            'declining': ['Traditional IT Consulting'],
            'skills': ['Digital Strategy', 'Change Management', 'Data Analytics']
        },
        'startup': {
            'trend': 'growing', 'growth_rate': 0.15,
            'hiring': 'high',
            'emerging': ['Full-Stack', 'Growth', 'Product', 'AI/ML'],
            'declining': [],
            'skills': ['Full-Stack', 'Growth Hacking', 'Agile', 'Fundraising']
        }
    }

    SKILL_DEMAND = {
        'python': {'demand': 95, 'trend': 'rising', 'premium': 15},
        'javascript': {'demand': 90, 'trend': 'stable', 'premium': 10},
        'machine_learning': {'demand': 88, 'trend': 'rising', 'premium': 25},
        'cloud_computing': {'demand': 92, 'trend': 'rising', 'premium': 20},
        'kubernetes': {'demand': 85, 'trend': 'rising', 'premium': 22},
        'react': {'demand': 88, 'trend': 'stable', 'premium': 12},
        'sql': {'demand': 82, 'trend': 'stable', 'premium': 8},
        'agile': {'demand': 75, 'trend': 'stable', 'premium': 5},
        'leadership': {'demand': 80, 'trend': 'stable', 'premium': 18},
        'communication': {'demand': 85, 'trend': 'stable', 'premium': 10},
        'data_analysis': {'demand': 87, 'trend': 'rising', 'premium': 15},
        'ai': {'demand': 90, 'trend': 'rising', 'premium': 30},
        'blockchain': {'demand': 65, 'trend': 'stable', 'premium': 20},
        'cybersecurity': {'demand': 88, 'trend': 'rising', 'premium': 25}
    }

    def __init__(self):
        self.cache: Dict[str, Any] = {}

    def get_salary_benchmark(
        self,
        role: str,
        location: str = 'remote_us',
        experience_years: int = 5
    ) -> SalaryData:

        role_key = role.lower().replace(' ', '_').replace('-', '_')

        base_salaries = self.BASE_SALARIES.get(role_key)
        if not base_salaries:
            for key in self.BASE_SALARIES:
                if role_key in key or key in role_key:
                    base_salaries = self.BASE_SALARIES[key]
                    break

        if not base_salaries:
            base_salaries = self.BASE_SALARIES['software_engineer']

        if experience_years < 2:
            level = 'entry'
        elif experience_years < 5:
            level = 'mid'
        elif experience_years < 10:
            level = 'senior'
        else:
            level = 'lead'

        base = base_salaries[level]

        location_key = location.lower().replace(' ', '_')
        multiplier = self.LOCATION_MULTIPLIERS.get(location_key, 1.0)

        median = base * multiplier

        return SalaryData(
            role=role,
            location=location,
            experience_years=experience_years,
            percentile_25=median * 0.85,
            percentile_50=median,
            percentile_75=median * 1.18,
            percentile_90=median * 1.35
        )

    def compare_salaries(
        self,
        role: str,
        current_location: str,
        target_location: str,
        experience_years: int = 5
    ) -> Dict[str, Any]:

        current = self.get_salary_benchmark(role, current_location, experience_years)
        target = self.get_salary_benchmark(role, target_location, experience_years)

        difference = target.percentile_50 - current.percentile_50
        pct_change = (difference / current.percentile_50) * 100

        col_index = {
            'san_francisco': 1.5, 'new_york': 1.4, 'seattle': 1.25,
            'boston': 1.2, 'los_angeles': 1.3, 'austin': 1.0,
            'denver': 1.05, 'chicago': 1.0, 'atlanta': 0.95,
            'bangalore': 0.3, 'london': 1.35, 'berlin': 0.9,
            'remote_us': 1.0, 'singapore': 1.1, 'toronto': 1.0
        }

        current_col = col_index.get(current_location.lower().replace(' ', '_'), 1.0)
        target_col = col_index.get(target_location.lower().replace(' ', '_'), 1.0)

        adjusted_current = current.percentile_50 / current_col
        adjusted_target = target.percentile_50 / target_col
        real_difference = adjusted_target - adjusted_current

        return {
            'role': role,
            'current': {
                'location': current_location,
                'salary_median': current.percentile_50,
                'salary_range': f"${current.percentile_25:,.0f} - ${current.percentile_75:,.0f}"
            },
            'target': {
                'location': target_location,
                'salary_median': target.percentile_50,
                'salary_range': f"${target.percentile_25:,.0f} - ${target.percentile_75:,.0f}"
            },
            'nominal_difference': difference,
            'percentage_change': pct_change,
            'cost_of_living_adjusted_difference': real_difference,
            'recommendation': self._salary_recommendation(pct_change, real_difference)
        }

    def _salary_recommendation(self, pct_change: float, real_diff: float) -> str:

        if real_diff > 15000:
            return "Strong financial incentive to relocate"
        elif real_diff > 5000:
            return "Moderate financial advantage for the move"
        elif real_diff > -5000:
            return "Similar purchasing power - consider other factors"
        else:
            return "Financial disadvantage - ensure other benefits justify the move"

    def get_industry_trend(self, industry: str) -> IndustryTrend:

        industry_key = industry.lower().replace(' ', '_')
        data = self.INDUSTRY_TRENDS.get(industry_key, self.INDUSTRY_TRENDS['technology'])

        return IndustryTrend(
            industry=industry,
            trend_direction=data['trend'],
            growth_rate=data['growth_rate'],
            hiring_activity=data['hiring'],
            emerging_roles=data['emerging'],
            declining_roles=data['declining'],
            key_skills=data['skills']
        )

    def get_job_market_health(self, industry: str, location: str = 'remote_us') -> JobMarketHealth:

        trend = self.INDUSTRY_TRENDS.get(industry.lower(), self.INDUSTRY_TRENDS['technology'])

        base_demand = 70 if trend['hiring'] == 'high' else (55 if trend['hiring'] == 'moderate' else 40)
        location_factor = self.LOCATION_MULTIPLIERS.get(location.lower().replace(' ', '_'), 1.0)

        demand = min(100, base_demand * (0.8 + location_factor * 0.3))
        supply = 100 - demand * 0.7

        if demand > supply + 20:
            competition = 'low'
        elif demand > supply:
            competition = 'moderate'
        else:
            competition = 'high'

        return JobMarketHealth(
            industry=industry,
            location=location,
            demand_score=demand,
            supply_score=supply,
            competition_level=competition,
            avg_time_to_hire_days=int(30 - (demand - 50) * 0.3),
            remote_friendly_pct=0.6 if 'tech' in industry.lower() else 0.35,
            salary_trend='increasing' if trend['growth_rate'] > 0.05 else 'stable'
        )

    def get_skill_demand(self, skill: str) -> SkillDemand:

        skill_key = skill.lower().replace(' ', '_').replace('-', '_')
        data = self.SKILL_DEMAND.get(skill_key)

        if not data:
            data = {'demand': 50, 'trend': 'stable', 'premium': 5}

        related_roles = self._get_related_roles(skill_key)

        return SkillDemand(
            skill=skill,
            demand_score=data['demand'],
            trend=data['trend'],
            avg_salary_premium=data['premium'],
            related_roles=related_roles
        )

    def _get_related_roles(self, skill: str) -> List[str]:

        role_skills = {
            'python': ['Data Scientist', 'ML Engineer', 'Backend Developer'],
            'javascript': ['Frontend Developer', 'Full Stack Developer', 'Web Developer'],
            'machine_learning': ['ML Engineer', 'Data Scientist', 'AI Engineer'],
            'cloud_computing': ['Cloud Architect', 'DevOps Engineer', 'SRE'],
            'react': ['Frontend Developer', 'Full Stack Developer', 'UI Engineer'],
            'data_analysis': ['Data Analyst', 'Business Analyst', 'Product Analyst']
        }
        return role_skills.get(skill, ['Software Engineer', 'Developer'])

    def get_skills_gap_analysis(self, current_skills: List[str], target_role: str) -> Dict[str, Any]:

        role_requirements = {
            'software_engineer': ['python', 'javascript', 'sql', 'git', 'agile'],
            'data_scientist': ['python', 'machine_learning', 'sql', 'statistics', 'data_analysis'],
            'product_manager': ['agile', 'communication', 'data_analysis', 'leadership', 'strategy'],
            'devops_engineer': ['cloud_computing', 'kubernetes', 'python', 'linux', 'ci_cd'],
            'machine_learning_engineer': ['python', 'machine_learning', 'tensorflow', 'cloud_computing', 'sql']
        }

        role_key = target_role.lower().replace(' ', '_')
        required = role_requirements.get(role_key, role_requirements['software_engineer'])

        current_set = set(s.lower().replace(' ', '_') for s in current_skills)
        required_set = set(required)

        have = current_set & required_set
        missing = required_set - current_set
        extra = current_set - required_set

        missing_with_demand = []
        for skill in missing:
            demand = self.get_skill_demand(skill)
            missing_with_demand.append({
                'skill': skill.replace('_', ' ').title(),
                'demand': demand.demand_score,
                'trend': demand.trend,
                'priority': 'high' if demand.demand_score > 80 else 'medium'
            })

        missing_with_demand.sort(key=lambda x: x['demand'], reverse=True)

        return {
            'target_role': target_role,
            'match_percentage': len(have) / len(required_set) * 100 if required_set else 0,
            'skills_have': list(have),
            'skills_needed': missing_with_demand,
            'transferable_skills': list(extra),
            'recommendation': self._skills_recommendation(len(have), len(required_set))
        }

    def _skills_recommendation(self, have: int, need: int) -> str:

        ratio = have / need if need > 0 else 1.0
        if ratio >= 0.8:
            return "Strong skill match - you're well-prepared for this role"
        elif ratio >= 0.6:
            return "Good foundation - focus on filling the gaps within 3-6 months"
        elif ratio >= 0.4:
            return "Significant gaps - consider a structured learning plan over 6-12 months"
        else:
            return "Major transition - this may require substantial upskilling or education"

    def get_market_summary(self, industry: str = 'technology', location: str = 'remote_us') -> Dict[str, Any]:

        trend = self.get_industry_trend(industry)
        health = self.get_job_market_health(industry, location)

        top_skills = sorted(
            self.SKILL_DEMAND.items(),
            key=lambda x: x[1]['demand'],
            reverse=True
        )[:5]

        return {
            'industry': industry,
            'location': location,
            'market_health': {
                'demand_score': health.demand_score,
                'competition': health.competition_level,
                'salary_trend': health.salary_trend,
                'remote_friendly': f"{health.remote_friendly_pct * 100:.0f}%"
            },
            'industry_outlook': {
                'trend': trend.trend_direction,
                'growth_rate': f"{trend.growth_rate * 100:.1f}%",
                'hiring_activity': trend.hiring_activity
            },
            'emerging_roles': trend.emerging_roles,
            'declining_roles': trend.declining_roles,
            'top_skills_in_demand': [
                {
                    'skill': skill.replace('_', ' ').title(),
                    'demand': data['demand'],
                    'trend': data['trend'],
                    'salary_premium': f"+{data['premium']}%"
                }
                for skill, data in top_skills
            ],
            'key_takeaway': self._market_takeaway(trend, health)
        }

    def _market_takeaway(self, trend: IndustryTrend, health: JobMarketHealth) -> str:

        if trend.trend_direction == 'growing' and health.competition_level == 'low':
            return "Excellent market conditions - high demand and low competition. Good time to make moves."
        elif trend.trend_direction == 'growing':
            return "Growing industry with good opportunities, but competition exists. Differentiate with skills."
        elif trend.trend_direction == 'stable':
            return "Stable market - focus on building skills and network for long-term positioning."
        else:
            return "Challenging market conditions - consider adjacent industries or skill pivots."

    def to_dict(self, salary: SalaryData) -> Dict[str, Any]:

        return {
            'role': salary.role,
            'location': salary.location,
            'experience_years': salary.experience_years,
            'salary_range': {
                'percentile_25': salary.percentile_25,
                'median': salary.percentile_50,
                'percentile_75': salary.percentile_75,
                'percentile_90': salary.percentile_90
            },
            'currency': salary.currency,
            'last_updated': salary.last_updated.isoformat()
        }
