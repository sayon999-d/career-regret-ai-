import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class CulturalDimension(Enum):
    INDIVIDUALISM = "individualism" 
    POWER_DISTANCE = "power_distance" 
    UNCERTAINTY_AVOIDANCE = "uncertainty_avoidance" 
    LONG_TERM_ORIENTATION = "long_term_orientation"
    WORK_LIFE_PRIORITY = "work_life_priority"


@dataclass
class CulturalProfile:
    region: str
    country: str
    language: str
    dimensions: Dict[str, float] 
    career_norms: Dict[str, str]
    salary_ppp_factor: float 
    currency: str
    typical_work_hours: int
    vacation_days: int
    cultural_notes: List[str]


@dataclass
class LocalizedAdvice:
    original: str
    adapted: str
    cultural_context: str
    adjustments_made: List[str]


class MultilingualService:
    CULTURAL_PROFILES = {
        "us": CulturalProfile(
            region="North America", country="United States", language="en",
            dimensions={
                "individualism": 91, "power_distance": 40,
                "uncertainty_avoidance": 46, "long_term_orientation": 26,
                "work_life_priority": 45
            },
            career_norms={
                "job_hopping": "Accepted and often encouraged, especially in tech",
                "salary_negotiation": "Expected; candidates who don't negotiate leave money on the table",
                "resume_gaps": "Increasingly accepted, but prepare an explanation",
                "career_switching": "Common and valued; diverse experience seen as strength",
                "work_hours": "Varies widely; startup culture often means longer hours",
                "referrals": "Critical for hiring; networking is essential"
            },
            salary_ppp_factor=1.0, currency="USD",
            typical_work_hours=40, vacation_days=15,
            cultural_notes=[
                "At-will employment means both employer and employee can end the relationship",
                "Health insurance is typically tied to employment",
                "401(k) matching is a significant benefit to evaluate"
            ]
        ),
        "uk": CulturalProfile(
            region="Europe", country="United Kingdom", language="en",
            dimensions={
                "individualism": 89, "power_distance": 35,
                "uncertainty_avoidance": 35, "long_term_orientation": 51,
                "work_life_priority": 60
            },
            career_norms={
                "job_hopping": "Less common than US; 2-3 years minimum expected",
                "salary_negotiation": "Acceptable but more restrained than US",
                "resume_gaps": "Somewhat accepted with explanation",
                "career_switching": "Possible but slower; credentials matter more",
                "work_hours": "Typically 37.5-40 hours; overtime less common",
                "referrals": "Important but formal applications also valued"
            },
            salary_ppp_factor=0.85, currency="GBP",
            typical_work_hours=38, vacation_days=28,
            cultural_notes=[
                "Statutory minimum 28 days paid holiday (including bank holidays)",
                "NHS provides healthcare regardless of employment",
                "Notice periods are typically 1-3 months"
            ]
        ),
        "de": CulturalProfile(
            region="Europe", country="Germany", language="de",
            dimensions={
                "individualism": 67, "power_distance": 35,
                "uncertainty_avoidance": 65, "long_term_orientation": 83,
                "work_life_priority": 75
            },
            career_norms={
                "job_hopping": "Stigmatized; stability is valued, 3-5 year stints expected",
                "salary_negotiation": "Expected but formal; come with market data",
                "resume_gaps": "Must be explained; structure and planning valued",
                "career_switching": "Possible but formal retraining often expected",
                "work_hours": "Strictly regulated; 35-40 hours typical",
                "referrals": "Important but formal qualifications are paramount"
            },
            salary_ppp_factor=0.80, currency="EUR",
            typical_work_hours=38, vacation_days=30,
            cultural_notes=[
                "Works councils (Betriebsrat) have significant employee protection power",
                "Formal qualifications and certifications carry heavy weight",
                "30 days paid vacation is standard"
            ]
        ),
        "jp": CulturalProfile(
            region="Asia", country="Japan", language="ja",
            dimensions={
                "individualism": 46, "power_distance": 54,
                "uncertainty_avoidance": 92, "long_term_orientation": 88,
                "work_life_priority": 30
            },
            career_norms={
                "job_hopping": "Historically very stigmatized; changing rapidly among younger workers",
                "salary_negotiation": "Uncommon; raises tied to seniority and company performance",
                "resume_gaps": "Difficult to explain; continuous employment expected",
                "career_switching": "Rare and challenging; specialization is valued",
                "work_hours": "Long hours culturally expected; work reform laws improving this",
                "referrals": "Very important; personal connections (人脈) are critical"
            },
            salary_ppp_factor=0.65, currency="JPY",
            typical_work_hours=45, vacation_days=20,
            cultural_notes=[
                "Traditional lifetime employment (終身雇用) is declining but still influential",
                "Seniority-based promotions are common in traditional companies",
                "Working at a foreign company (外資系) carries different expectations"
            ]
        ),
        "in": CulturalProfile(
            region="South Asia", country="India", language="hi",
            dimensions={
                "individualism": 48, "power_distance": 77,
                "uncertainty_avoidance": 40, "long_term_orientation": 51,
                "work_life_priority": 35
            },
            career_norms={
                "job_hopping": "Common in IT/tech; 1-2 years is normal for salary growth",
                "salary_negotiation": "Expected and common; aggressive negotiation is normal",
                "resume_gaps": "Somewhat flexible; career breaks for education acceptable",
                "career_switching": "Common in IT; certifications help transition",
                "work_hours": "Often 45-55 hours; startup culture means more",
                "referrals": "Extremely important; personal networks drive hiring"
            },
            salary_ppp_factor=0.25, currency="INR",
            typical_work_hours=48, vacation_days=15,
            cultural_notes=[
                "Family influence on career decisions is significant",
                "Government/public sector jobs carry prestige beyond salary",
                "Notice periods are typically 30-90 days and enforced"
            ]
        ),
        "br": CulturalProfile(
            region="South America", country="Brazil", language="pt",
            dimensions={
                "individualism": 38, "power_distance": 69,
                "uncertainty_avoidance": 76, "long_term_orientation": 44,
                "work_life_priority": 55
            },
            career_norms={
                "job_hopping": "Increasingly accepted in tech; traditional sectors prefer stability",
                "salary_negotiation": "Common; Brazilian labor law provides strong employee protections",
                "resume_gaps": "Flexible; personal relationships matter more than resume gaps",
                "career_switching": "Possible through networking; less formal barriers",
                "work_hours": "44 hours legal maximum; overtime common",
                "referrals": "Very important; personal relationships (jeitinho) drive careers"
            },
            salary_ppp_factor=0.30, currency="BRL",
            typical_work_hours=44, vacation_days=30,
            cultural_notes=[
                "CLT (labor law) provides 13th salary, FGTS, and strong protections",
                "PJ (freelance contract) vs CLT is a critical career decision",
                "Relationships and personal connections are paramount"
            ]
        ),
        "sg": CulturalProfile(
            region="Southeast Asia", country="Singapore", language="en",
            dimensions={
                "individualism": 20, "power_distance": 74,
                "uncertainty_avoidance": 8, "long_term_orientation": 72,
                "work_life_priority": 40
            },
            career_norms={
                "job_hopping": "Common; 2 year stints are normal",
                "salary_negotiation": "Expected; market-driven economy",
                "resume_gaps": "Should be explained; meritocracy is valued",
                "career_switching": "Supported by government through SkillsFuture",
                "work_hours": "44 hours legally; often more in practice",
                "referrals": "Important but formal qualifications also valued"
            },
            salary_ppp_factor=0.70, currency="SGD",
            typical_work_hours=44, vacation_days=14,
            cultural_notes=[
                "Government-backed SkillsFuture credits for career development",
                "CPF (Central Provident Fund) is a major benefit consideration",
                "Multinational companies offer different dynamics than local firms"
            ]
        ),
        "remote": CulturalProfile(
            region="Global", country="Remote/Distributed", language="en",
            dimensions={
                "individualism": 75, "power_distance": 30,
                "uncertainty_avoidance": 40, "long_term_orientation": 50,
                "work_life_priority": 70
            },
            career_norms={
                "job_hopping": "Common; remote workers switch for better opportunities freely",
                "salary_negotiation": "Expected; geographic arbitrage is common",
                "resume_gaps": "Very flexible; outcomes matter more than tenure",
                "career_switching": "Highly supported; remote work enables experimentation",
                "work_hours": "Flexible; async work common; output over hours",
                "referrals": "Important through online communities and open source"
            },
            salary_ppp_factor=0.85, currency="USD",
            typical_work_hours=40, vacation_days=20,
            cultural_notes=[
                "Time zone management is a critical skill",
                "Self-motivation and communication are paramount",
                "Consider tax implications of remote work across jurisdictions"
            ]
        )
    }

    SUPPORTED_LANGUAGES = {
        "en": "English", "de": "German", "ja": "Japanese",
        "hi": "Hindi", "pt": "Portuguese", "zh": "Chinese",
        "es": "Spanish", "fr": "French", "ko": "Korean", "ar": "Arabic"
    }

    LANGUAGE_DETECTION_PATTERNS = {
        "ja": r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf]',
        "zh": r'[\u4e00-\u9fff]',
        "ko": r'[\uac00-\ud7af]',
        "ar": r'[\u0600-\u06ff]',
        "hi": r'[\u0900-\u097f]',
        "de": r'\b(der|die|das|und|ist|ein|ich|nicht|mit|auf)\b',
        "fr": r'\b(le|la|les|de|des|un|une|et|est|que|pour)\b',
        "es": r'\b(el|la|los|las|de|en|un|una|que|por|con)\b',
        "pt": r'\b(o|a|os|as|de|em|um|uma|que|por|com)\b',
    }

    def __init__(self):
        self.user_preferences: Dict[str, Dict] = {}

    def detect_language(self, text: str) -> str:
        for lang, pattern in self.LANGUAGE_DETECTION_PATTERNS.items():
            if re.search(pattern, text):
                return lang
        return "en"

    def get_cultural_profile(self, country_code: str) -> Dict:
        profile = self.CULTURAL_PROFILES.get(country_code.lower())
        if not profile:
            return {"error": f"Country code '{country_code}' not found",
                    "available": list(self.CULTURAL_PROFILES.keys())}
        return self._profile_to_dict(profile)

    def adapt_advice(self, advice: str, country_code: str,
                     decision_type: str = None) -> Dict:
        profile = self.CULTURAL_PROFILES.get(country_code.lower())
        if not profile:
            return {"original": advice, "adapted": advice,
                    "note": "No cultural adaptation available for this region"}

        adjustments = []
        adapted = advice

        if any(w in advice.lower() for w in ['switch', 'change job', 'new role', 'hop']):
            norm = profile.career_norms.get("job_hopping", "")
            if norm:
                adjustments.append(f"Job mobility context: {norm}")
        if any(w in advice.lower() for w in ['salary', 'negotiate', 'compensation', 'raise']):
            norm = profile.career_norms.get("salary_negotiation", "")
            if norm:
                adjustments.append(f"Negotiation culture: {norm}")
        if any(w in advice.lower() for w in ['hours', 'overtime', 'work-life', 'balance']):
            adjustments.append(
                f"Typical work hours in {profile.country}: {profile.typical_work_hours}/week, "
                f"{profile.vacation_days} vacation days"
            )

        cultural_context = (
            f"In {profile.country}, career decisions are influenced by "
            f"{'high' if profile.dimensions.get('individualism', 50) > 60 else 'moderate' if profile.dimensions.get('individualism', 50) > 40 else 'low'} individualism "
            f"and {'high' if profile.dimensions.get('uncertainty_avoidance', 50) > 60 else 'moderate' if profile.dimensions.get('uncertainty_avoidance', 50) > 40 else 'low'} uncertainty avoidance."
        )

        return {
            "original": advice,
            "adapted": adapted,
            "cultural_context": cultural_context,
            "adjustments": adjustments,
            "career_norms": profile.career_norms,
            "cultural_notes": profile.cultural_notes,
            "country": profile.country
        }

    def adjust_salary(self, salary_usd: float, from_country: str,
                      to_country: str) -> Dict:
        from_profile = self.CULTURAL_PROFILES.get(from_country.lower())
        to_profile = self.CULTURAL_PROFILES.get(to_country.lower())

        if not from_profile or not to_profile:
            return {"error": "One or both country codes not found"}

        relative_factor = to_profile.salary_ppp_factor / from_profile.salary_ppp_factor
        adjusted_salary = salary_usd * relative_factor

        currency_factors = {
            "USD": 1.0, "GBP": 0.79, "EUR": 0.92, "JPY": 149.0,
            "INR": 83.0, "BRL": 4.97, "SGD": 1.34
        }
        from_rate = currency_factors.get(from_profile.currency, 1.0)
        to_rate = currency_factors.get(to_profile.currency, 1.0)
        nominal_in_local = salary_usd / from_rate * to_rate

        return {
            "original_salary": salary_usd,
            "original_currency": from_profile.currency,
            "original_country": from_profile.country,
            "ppp_adjusted": round(adjusted_salary, 2),
            "nominal_local_currency": round(nominal_in_local, 2),
            "target_currency": to_profile.currency,
            "target_country": to_profile.country,
            "ppp_factor": round(relative_factor, 3),
            "purchasing_power_note": (
                f"${salary_usd:,.0f} {from_profile.currency} in {from_profile.country} is equivalent to "
                f"~${adjusted_salary:,.0f} {from_profile.currency} purchasing power in {to_profile.country}."
            ),
            "cost_of_living_comparison": {
                "from_work_hours": from_profile.typical_work_hours,
                "to_work_hours": to_profile.typical_work_hours,
                "from_vacation_days": from_profile.vacation_days,
                "to_vacation_days": to_profile.vacation_days
            }
        }

    def get_system_prompt_for_locale(self, country_code: str) -> str:
        profile = self.CULTURAL_PROFILES.get(country_code.lower())
        if not profile:
            return ""

        norms = "\n".join(f"- {k}: {v}" for k, v in profile.career_norms.items())
        notes = "\n".join(f"- {n}" for n in profile.cultural_notes[:3])

        return (
            f"The user is based in {profile.country}. Adapt your career advice to local context.\n\n"
            f"Key career norms in {profile.country}:\n{norms}\n\n"
            f"Important cultural notes:\n{notes}\n\n"
            f"Typical work: {profile.typical_work_hours}h/week, {profile.vacation_days} vacation days.\n"
            f"Be culturally sensitive and avoid assuming US-centric career norms."
        )

    def get_supported_countries(self) -> List[Dict]:
        return [{
            "code": code,
            "country": p.country,
            "region": p.region,
            "language": p.language,
            "currency": p.currency
        } for code, p in self.CULTURAL_PROFILES.items()]

    def set_user_locale(self, user_id: str, country_code: str,
                        preferred_language: str = None) -> Dict:
        profile = self.CULTURAL_PROFILES.get(country_code.lower())
        if not profile:
            return {"error": "Country code not found"}

        self.user_preferences[user_id] = {
            "country_code": country_code.lower(),
            "language": preferred_language or profile.language,
            "set_at": datetime.utcnow().isoformat()
        }
        return {
            "user_id": user_id,
            "country": profile.country,
            "language": self.SUPPORTED_LANGUAGES.get(
                preferred_language or profile.language, "English"
            ),
            "currency": profile.currency
        }

    def compare_work_cultures(self, country_a: str, country_b: str) -> Dict:
        a = self.CULTURAL_PROFILES.get(country_a.lower())
        b = self.CULTURAL_PROFILES.get(country_b.lower())
        if not a or not b:
            return {"error": "One or both country codes not found"}

        comparisons = {}
        for dim in CulturalDimension:
            val_a = a.dimensions.get(dim.value, 50)
            val_b = b.dimensions.get(dim.value, 50)
            diff = val_a - val_b
            comparisons[dim.value] = {
                f"{a.country}": val_a,
                f"{b.country}": val_b,
                "difference": round(abs(diff)),
                "insight": self._dimension_insight(dim, val_a, val_b, a.country, b.country)
            }

        return {
            "country_a": {"code": country_a, "name": a.country},
            "country_b": {"code": country_b, "name": b.country},
            "dimensions": comparisons,
            "work_hours_diff": abs(a.typical_work_hours - b.typical_work_hours),
            "vacation_days_diff": abs(a.vacation_days - b.vacation_days),
            "career_norm_differences": self._norm_differences(a, b)
        }

    def _dimension_insight(self, dim: CulturalDimension, val_a: float,
                            val_b: float, name_a: str, name_b: str) -> str:
        insights = {
            CulturalDimension.INDIVIDUALISM: (
                f"{name_a if val_a > val_b else name_b} has a more individualistic work culture. "
                f"Career decisions in individualistic cultures focus on personal growth; "
                f"collectivist cultures consider group/family impact more."
            ),
            CulturalDimension.POWER_DISTANCE: (
                f"{name_a if val_a > val_b else name_b} has higher power distance. "
                f"In higher power distance cultures, hierarchical advancement is more structured."
            ),
            CulturalDimension.UNCERTAINTY_AVOIDANCE: (
                f"{name_a if val_a > val_b else name_b} has higher uncertainty avoidance. "
                f"Job stability and established career paths are more valued there."
            ),
            CulturalDimension.WORK_LIFE_PRIORITY: (
                f"{name_a if val_a > val_b else name_b} places more emphasis on work-life balance."
            )
        }
        return insights.get(dim, "")

    def _norm_differences(self, a: CulturalProfile, b: CulturalProfile) -> List[Dict]:
        diffs = []
        for key in a.career_norms:
            if key in b.career_norms and a.career_norms[key] != b.career_norms[key]:
                diffs.append({
                    "aspect": key.replace('_', ' ').title(),
                    a.country: a.career_norms[key],
                    b.country: b.career_norms[key]
                })
        return diffs

    def _profile_to_dict(self, profile: CulturalProfile) -> Dict:
        return {
            "region": profile.region,
            "country": profile.country,
            "language": profile.language,
            "dimensions": profile.dimensions,
            "career_norms": profile.career_norms,
            "salary_ppp_factor": profile.salary_ppp_factor,
            "currency": profile.currency,
            "typical_work_hours": profile.typical_work_hours,
            "vacation_days": profile.vacation_days,
            "cultural_notes": profile.cultural_notes
        }


multilingual_service = MultilingualService()
