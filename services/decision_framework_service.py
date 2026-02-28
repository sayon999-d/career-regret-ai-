import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from enum import Enum


class FrameworkType(Enum):
    JOB_OFFER = "job_offer_evaluator"
    CAREER_PIVOT = "career_pivot_readiness"
    NEGOTIATE_OR_WALK = "negotiate_or_walk"
    STARTUP_VS_CORPORATE = "startup_vs_corporate"
    PROMOTION_READINESS = "promotion_readiness"
    EDUCATION_ROI = "education_roi"
    RELOCATION = "relocation_decision"
    FREELANCE_TRANSITION = "freelance_transition"


@dataclass
class FrameworkDimension:
    name: str
    description: str
    weight: float
    score: Optional[float] = None
    notes: str = ""
    sub_questions: List[Dict] = field(default_factory=list)


@dataclass
class FrameworkResult:
    id: str
    user_id: str
    framework_type: FrameworkType
    dimensions: List[FrameworkDimension]
    total_score: float
    max_possible: float
    score_pct: float
    recommendation: str
    risk_level: str
    strengths: List[str]
    concerns: List[str]
    action_items: List[str]
    created_at: datetime = field(default_factory=datetime.utcnow)


class DecisionFrameworkService:
    FRAMEWORKS = {
        FrameworkType.JOB_OFFER: {
            "name": "Job Offer Evaluator",
            "description": "Systematically evaluate a job offer across 12 key dimensions",
            "dimensions": [
                {"name": "Compensation", "weight": 0.15,
                 "description": "Base salary, bonuses, equity, benefits",
                 "sub_questions": [
                     "How does the base salary compare to your current pay?",
                     "What's the equity/bonus structure?",
                     "How do benefits (health, 401k match, PTO) compare?"
                 ]},
                {"name": "Career Growth", "weight": 0.14,
                 "description": "Promotion path, skill development, mentorship",
                 "sub_questions": [
                     "Is there a clear promotion path?",
                     "Will you learn new, marketable skills?",
                     "Are there senior mentors in your domain?"
                 ]},
                {"name": "Work-Life Balance", "weight": 0.12,
                 "description": "Hours, flexibility, remote policy, PTO",
                 "sub_questions": [
                     "What are typical working hours?",
                     "Is remote/hybrid an option?",
                     "How much PTO, and is it actually used?"
                 ]},
                {"name": "Company Stability", "weight": 0.10,
                 "description": "Financial health, market position, leadership",
                 "sub_questions": [
                     "Is the company profitable or well-funded?",
                     "What's the employee retention rate?",
                     "Do you trust the leadership team?"
                 ]},
                {"name": "Team & Culture", "weight": 0.10,
                 "description": "Team dynamics, company values, diversity",
                 "sub_questions": [
                     "Did you connect with your potential teammates?",
                     "Do company values align with yours?",
                     "Is there a culture of feedback and learning?"
                 ]},
                {"name": "Impact & Purpose", "weight": 0.08,
                 "description": "Meaningful work, product impact, mission alignment",
                 "sub_questions": [
                     "Will your work have visible impact?",
                     "Do you believe in the company's mission?",
                     "Will you be proud to tell people where you work?"
                 ]},
                {"name": "Technical Challenge", "weight": 0.07,
                 "description": "Complexity, scale, innovation opportunity",
                 "sub_questions": [
                     "Will the technical problems challenge you?",
                     "Is the tech stack modern and well-maintained?",
                     "Is there room for innovation?"
                 ]},
                {"name": "Manager Quality", "weight": 0.07,
                 "description": "Direct manager's style, support, reputation",
                 "sub_questions": [
                     "Did you get a good impression of your future manager?",
                     "Do they support career development?",
                     "What's their management style?"
                 ]},
                {"name": "Location & Commute", "weight": 0.05,
                 "description": "Office location, commute time, relocation needs",
                 "sub_questions": [
                     "Is the commute reasonable?",
                     "Would you need to relocate?",
                     "Is the office in a desirable area?"
                 ]},
                {"name": "Brand & Network", "weight": 0.05,
                 "description": "Company reputation, resume value, professional network",
                 "sub_questions": [
                     "Will this company name strengthen your resume?",
                     "Will you build a valuable professional network?",
                     "Is the company respected in the industry?"
                 ]},
                {"name": "Autonomy", "weight": 0.04,
                 "description": "Decision-making power, ownership, flexibility",
                 "sub_questions": [
                     "How much ownership will you have over projects?",
                     "Can you influence technical decisions?",
                     "Is there room to shape your role?"
                 ]},
                {"name": "Gut Feeling", "weight": 0.03,
                 "description": "Overall intuition and excitement level",
                 "sub_questions": [
                     "Are you genuinely excited about this opportunity?",
                     "Can you see yourself happy here in 2 years?",
                     "Does anything feel 'off' that you can't articulate?"
                 ]}
            ]
        },
        FrameworkType.CAREER_PIVOT: {
            "name": "Career Pivot Readiness Assessment",
            "description": "Evaluate your readiness to switch careers",
            "dimensions": [
                {"name": "Skills Transferability", "weight": 0.18,
                 "description": "How many of your current skills apply to the new career?",
                 "sub_questions": [
                     "What percentage of your hard skills transfer directly?",
                     "Are your soft skills (leadership, communication) valued?",
                     "Do you have any certifications needed?"
                 ]},
                {"name": "Financial Runway", "weight": 0.16,
                 "description": "Can you absorb a potential salary cut or gap?",
                 "sub_questions": [
                     "How many months of expenses do you have saved?",
                     "Can you handle a 20-30% salary cut initially?",
                     "Do you have financial obligations (mortgage, dependents)?"
                 ]},
                {"name": "Market Demand", "weight": 0.14,
                 "description": "Is the target career growing?",
                 "sub_questions": [
                     "Are there open positions in your target field?",
                     "Is the industry growing at 5%+?",
                     "Is it harder or easier than your current field to get hired?"
                 ]},
                {"name": "Domain Knowledge", "weight": 0.13,
                 "description": "How much do you know about the target career?",
                 "sub_questions": [
                     "Have you talked to 3+ people in the target field?",
                     "Have you done any projects or coursework?",
                     "Do you understand the daily realities?"
                 ]},
                {"name": "Network in Target Field", "weight": 0.10,
                 "description": "Do you know people who can help you transition?",
                 "sub_questions": [
                     "Do you have connections in the target industry?",
                     "Have you attended events/meetups?",
                     "Could someone refer you for a position?"
                 ]},
                {"name": "Motivation Clarity", "weight": 0.10,
                 "description": "Are you running toward something or away from something?",
                 "sub_questions": [
                     "Can you articulate why you want to switch?",
                     "Is this a long-held passion or recent frustration?",
                     "Have you tried fixing what's wrong in your current role?"
                 ]},
                {"name": "Support System", "weight": 0.08,
                 "description": "Does your personal network support this change?",
                 "sub_questions": [
                     "Does your family/partner support the transition?",
                     "Do you have a mentor in the target field?",
                     "Can you handle the stress of uncertainty?"
                 ]},
                {"name": "Timing", "weight": 0.06,
                 "description": "Is this the right moment?",
                 "sub_questions": [
                     "Are there major life events coming up?",
                     "Is the job market favorable right now?",
                     "Have you completed any necessary prerequisites?"
                 ]},
                {"name": "Risk Tolerance", "weight": 0.05,
                 "description": "How comfortable are you with uncertainty?",
                 "sub_questions": [
                     "Can you handle ambiguity and setbacks?",
                     "Have you navigated major changes before?",
                     "What's your worst-case scenario plan?"
                 ]}
            ]
        },
        FrameworkType.NEGOTIATE_OR_WALK: {
            "name": "Negotiate or Walk Away (BATNA Analyzer)",
            "description": "Determine your bargaining position and whether to negotiate or walk",
            "dimensions": [
                {"name": "BATNA Strength", "weight": 0.20,
                 "description": "How strong is your Best Alternative To Negotiated Agreement?",
                 "sub_questions": [
                     "Do you have other offers?",
                     "Could you stay in your current role?",
                     "How quickly could you find another opportunity?"
                 ]},
                {"name": "Leverage", "weight": 0.18,
                 "description": "How much does the company need you vs. you need them?",
                 "sub_questions": [
                     "Are your skills rare in the market?",
                     "Is the team understaffed?",
                     "Did they pursue you, or did you apply?"
                 ]},
                {"name": "Relationship Value", "weight": 0.12,
                 "description": "How important is this relationship long-term?",
                 "sub_questions": [
                     "Will you work closely with the negotiation counterpart?",
                     "Is this company one you might return to?",
                     "Could aggressive negotiation damage the relationship?"
                 ]},
                {"name": "Gap Size", "weight": 0.15,
                 "description": "How far apart are your positions?",
                 "sub_questions": [
                     "What's the difference between offer and target?",
                     "Is the gap in salary, equity, title, or all?",
                     "What's the maximum they've offered similar candidates?"
                 ]},
                {"name": "Market Data", "weight": 0.15,
                 "description": "Does market data support your ask?",
                 "sub_questions": [
                     "What do salary surveys say for this role?",
                     "What are competitors paying?",
                     "Can you cite specific data points?"
                 ]},
                {"name": "Non-Monetary Value", "weight": 0.10,
                 "description": "Are there non-salary elements to negotiate?",
                 "sub_questions": [
                     "Can you negotiate remote work, title, or PTO?",
                     "Are there signing bonuses or relocation packages?",
                     "Is there room for accelerated review cycles?"
                 ]},
                {"name": "Walk-Away Cost", "weight": 0.10,
                 "description": "What do you lose if you walk?",
                 "sub_questions": [
                     "How much time have you invested in this process?",
                     "Are there unique benefits you'd lose?",
                     "Would walking delay your career plans?"
                 ]}
            ]
        },
        FrameworkType.STARTUP_VS_CORPORATE: {
            "name": "Startup vs. Corporate Decision Matrix",
            "description": "Compare startup and corporate paths across lifestyle, financial, and growth dimensions",
            "dimensions": [
                {"name": "Financial Security", "weight": 0.15,
                 "description": "Stability of income and benefits",
                 "sub_questions": ["Guaranteed salary?", "Health/retirement benefits?", "Financial cushion?"]},
                {"name": "Equity Upside", "weight": 0.12,
                 "description": "Potential financial upside from equity",
                 "sub_questions": ["Equity offered?", "Realistic exit timeline?", "Dilution risk?"]},
                {"name": "Learning Speed", "weight": 0.13,
                 "description": "Rate of skill development",
                 "sub_questions": ["Breadth vs depth?", "Mentorship quality?", "Challenge level?"]},
                {"name": "Autonomy & Impact", "weight": 0.12,
                 "description": "Control over your work and visible impact",
                 "sub_questions": ["Decision authority?", "Direct user impact?", "Shape the product?"]},
                {"name": "Work-Life Integration", "weight": 0.12,
                 "description": "Hours, flexibility, boundary clarity",
                 "sub_questions": ["Expected hours?", "Weekend work?", "Vacation culture?"]},
                {"name": "Career Progression", "weight": 0.10,
                 "description": "Title advancement and career capital",
                 "sub_questions": ["Promotion timeline?", "Resume value?", "Network quality?"]},
                {"name": "Risk Tolerance Fit", "weight": 0.10,
                 "description": "Does the risk level match your personality?",
                 "sub_questions": ["Can you handle ambiguity?", "Comfortable with failure?", "Know when to pivot?"]},
                {"name": "Mission Alignment", "weight": 0.08,
                 "description": "Do you believe in what the company is building?",
                 "sub_questions": ["Passionate about the problem?", "Trust the founders?", "Proud of the product?"]},
                {"name": "Team Quality", "weight": 0.08,
                 "description": "Caliber and chemistry of the team",
                 "sub_questions": ["Impressed by colleagues?", "Culture fit?", "Trust the leadership?"]}
            ]
        },
        FrameworkType.PROMOTION_READINESS: {
            "name": "Promotion Readiness Assessment",
            "description": "Evaluate whether you're ready for the next level",
            "dimensions": [
                {"name": "Already Operating at Next Level", "weight": 0.20,
                 "description": "Are you already doing the job above yours?",
                 "sub_questions": ["Do you take on responsibilities beyond your role?", "Has your scope expanded?"]},
                {"name": "Visibility", "weight": 0.15,
                 "description": "Do decision-makers know your contributions?",
                 "sub_questions": ["Does your manager advocate for you?", "Have you presented to leadership?"]},
                {"name": "Impact Evidence", "weight": 0.18,
                 "description": "Can you quantify your impact?",
                 "sub_questions": ["Do you have metrics showing business impact?", "Can you tell a story with data?"]},
                {"name": "Peer Support", "weight": 0.10,
                 "description": "Do peers see you as a leader?",
                 "sub_questions": ["Do others seek your advice?", "Have you mentored anyone?"]},
                {"name": "Skills Gap", "weight": 0.15,
                 "description": "Do you have the skills needed at the next level?",
                 "sub_questions": ["What skills does the next level require?", "Have you developed them?"]},
                {"name": "Timing & Budget", "weight": 0.12,
                 "description": "Is the organizational context right?",
                 "sub_questions": ["Are promotions happening now?", "Is headcount available?"]},
                {"name": "Alternative Paths", "weight": 0.10,
                 "description": "If not promoted, what else can you do?",
                 "sub_questions": ["Would you leave for this title elsewhere?", "Can you lateral to a faster track?"]}
            ]
        },
        FrameworkType.EDUCATION_ROI: {
            "name": "Education Investment ROI Calculator",
            "description": "Evaluate the return on investment for education (degree, bootcamp, certification)",
            "dimensions": [
                {"name": "Salary Impact", "weight": 0.22,
                 "description": "Expected salary increase from the credential",
                 "sub_questions": ["What's the average salary boost?", "Is this credential required for target roles?"]},
                {"name": "Cost", "weight": 0.18,
                 "description": "Total cost including opportunity cost",
                 "sub_questions": ["Tuition costs?", "Lost income during study?", "Living expenses?"]},
                {"name": "Time Commitment", "weight": 0.14,
                 "description": "Duration and flexibility",
                 "sub_questions": ["Full-time or part-time?", "Can you work while studying?", "Total months?"]},
                {"name": "Career Unlock", "weight": 0.16,
                 "description": "Does it open doors that are currently closed?",
                 "sub_questions": ["Are certain roles gated by this credential?", "Would it unlock promotions?"]},
                {"name": "Network Value", "weight": 0.10,
                 "description": "Professional network gained from the program",
                 "sub_questions": ["Alumni network quality?", "Industry connections?"]},
                {"name": "Alternatives", "weight": 0.10,
                 "description": "Could you achieve the same result without this education?",
                 "sub_questions": ["Self-study viable?", "On-the-job learning possible?"]},
                {"name": "Market Trend", "weight": 0.10,
                 "description": "Is this credential's value increasing or decreasing?",
                 "sub_questions": ["Are employers valuing it more or less?", "Is the field growing?"]}
            ]
        }
    }

    def __init__(self):
        self.user_sessions: Dict[str, List[FrameworkResult]] = defaultdict(list)
        self.in_progress: Dict[str, Dict] = {}

    def list_frameworks(self) -> List[Dict]:
        return [{
            "type": ftype.value,
            "name": config["name"],
            "description": config["description"],
            "dimension_count": len(config["dimensions"]),
            "dimensions": [d["name"] for d in config["dimensions"]]
        } for ftype, config in self.FRAMEWORKS.items()]

    def start_framework(self, user_id: str, framework_type: str) -> Dict:
        try:
            ftype = FrameworkType(framework_type)
        except ValueError:
            return {"error": f"Unknown framework: {framework_type}",
                    "available": [f.value for f in FrameworkType]}

        config = self.FRAMEWORKS[ftype]
        session_id = str(uuid.uuid4())

        session = {
            "session_id": session_id,
            "framework_type": ftype,
            "config": config,
            "scores": {},
            "notes": {},
            "current_step": 0,
            "started_at": datetime.utcnow()
        }
        self.in_progress[f"{user_id}:{session_id}"] = session

        first_dim = config["dimensions"][0]
        return {
            "session_id": session_id,
            "framework": config["name"],
            "total_steps": len(config["dimensions"]),
            "current_step": 1,
            "dimension": {
                "name": first_dim["name"],
                "description": first_dim["description"],
                "weight": first_dim["weight"],
                "sub_questions": first_dim.get("sub_questions", [])
            },
            "instruction": "Rate this dimension from 1 (very poor) to 10 (excellent)"
        }

    def score_dimension(self, user_id: str, session_id: str,
                        score: float, notes: str = "") -> Dict:
        key = f"{user_id}:{session_id}"
        session = self.in_progress.get(key)
        if not session:
            return {"error": "Session not found or expired"}

        score = max(1, min(10, score))
        config = session["config"]
        step = session["current_step"]
        dim_name = config["dimensions"][step]["name"]

        session["scores"][dim_name] = score
        session["notes"][dim_name] = notes
        session["current_step"] += 1

        if session["current_step"] >= len(config["dimensions"]):
            return self._complete_framework(user_id, session_id)

        next_dim = config["dimensions"][session["current_step"]]
        return {
            "session_id": session_id,
            "scored": {"dimension": dim_name, "score": score},
            "current_step": session["current_step"] + 1,
            "total_steps": len(config["dimensions"]),
            "dimension": {
                "name": next_dim["name"],
                "description": next_dim["description"],
                "weight": next_dim["weight"],
                "sub_questions": next_dim.get("sub_questions", [])
            },
            "progress_pct": round(session["current_step"] / len(config["dimensions"]) * 100)
        }

    def quick_score(self, user_id: str, framework_type: str,
                    scores: Dict[str, float]) -> Dict:
        try:
            ftype = FrameworkType(framework_type)
        except ValueError:
            return {"error": f"Unknown framework: {framework_type}"}

        config = self.FRAMEWORKS[ftype]
        session_id = str(uuid.uuid4())
        key = f"{user_id}:{session_id}"

        session = {
            "session_id": session_id,
            "framework_type": ftype,
            "config": config,
            "scores": {},
            "notes": {},
            "current_step": len(config["dimensions"]),
            "started_at": datetime.utcnow()
        }

        for dim in config["dimensions"]:
            dim_name = dim["name"]
            score = scores.get(dim_name, scores.get(dim_name.lower(), 5))
            session["scores"][dim_name] = max(1, min(10, float(score)))

        self.in_progress[key] = session
        return self._complete_framework(user_id, session_id)

    def _complete_framework(self, user_id: str, session_id: str) -> Dict:
        key = f"{user_id}:{session_id}"
        session = self.in_progress.get(key)
        if not session:
            return {"error": "Session not found"}

        config = session["config"]
        ftype = session["framework_type"]
        dimensions = []
        weighted_total = 0
        max_possible = 0

        for dim_config in config["dimensions"]:
            name = dim_config["name"]
            weight = dim_config["weight"]
            score = session["scores"].get(name, 5)
            notes = session["notes"].get(name, "")

            dimension = FrameworkDimension(
                name=name,
                description=dim_config["description"],
                weight=weight,
                score=score,
                notes=notes,
                sub_questions=dim_config.get("sub_questions", [])
            )
            dimensions.append(dimension)
            weighted_total += score * weight
            max_possible += 10 * weight

        score_pct = (weighted_total / max(max_possible, 0.01)) * 100

        strengths = [d.name for d in sorted(dimensions, key=lambda x: x.score or 0, reverse=True)
                     if (d.score or 0) >= 7][:3]
        concerns = [d.name for d in sorted(dimensions, key=lambda x: x.score or 0)
                    if (d.score or 0) <= 4][:3]

        recommendation = self._generate_recommendation(ftype, score_pct, strengths, concerns)
        risk_level = "low" if score_pct >= 70 else "medium" if score_pct >= 50 else "high"
        action_items = self._generate_action_items(ftype, dimensions, concerns)

        result = FrameworkResult(
            id=session_id,
            user_id=user_id,
            framework_type=ftype,
            dimensions=dimensions,
            total_score=round(weighted_total, 2),
            max_possible=round(max_possible, 2),
            score_pct=round(score_pct, 1),
            recommendation=recommendation,
            risk_level=risk_level,
            strengths=strengths,
            concerns=concerns,
            action_items=action_items
        )
        self.user_sessions[user_id].append(result)

        del self.in_progress[key]

        return {
            "session_id": session_id,
            "framework": config["name"],
            "total_score": round(weighted_total, 2),
            "max_possible": round(max_possible, 2),
            "score_pct": round(score_pct, 1),
            "recommendation": recommendation,
            "risk_level": risk_level,
            "strengths": strengths,
            "concerns": concerns,
            "action_items": action_items,
            "dimensions": [{
                "name": d.name,
                "score": d.score,
                "weight": d.weight,
                "weighted_score": round((d.score or 0) * d.weight, 2)
            } for d in dimensions],
            "verdict": self._get_verdict(score_pct)
        }

    def _generate_recommendation(self, ftype: FrameworkType, score_pct: float,
                                  strengths: List, concerns: List) -> str:
        recs = {
            FrameworkType.JOB_OFFER: {
                "high": "This offer scores strongly. The fundamentals are solid—consider accepting.",
                "medium": "This offer has merit but some areas need attention. Consider negotiating on the weaker dimensions.",
                "low": "This offer has significant concerns. Unless key areas improve, it may not be the right move."
            },
            FrameworkType.CAREER_PIVOT: {
                "high": "You show strong readiness for a career pivot. The foundations are in place.",
                "medium": "You're partially ready. Focus on closing gaps in the weaker areas before committing.",
                "low": "Consider building more foundation before pivoting. Address the key concerns first."
            },
            FrameworkType.NEGOTIATE_OR_WALK: {
                "high": "You're in a strong negotiating position. Push confidently for your terms.",
                "medium": "You have some leverage. Negotiate strategically but be prepared to compromise.",
                "low": "Your bargaining position is weak. Consider whether the current offer is good enough."
            }
        }
        default = {
            "high": "Strong overall assessment. The data supports moving forward.",
            "medium": "Mixed signals. Focus on improving weak areas before deciding.",
            "low": "The analysis reveals significant concerns. Proceed with caution."
        }
        level = "high" if score_pct >= 70 else "medium" if score_pct >= 50 else "low"
        return recs.get(ftype, default)[level]

    def _generate_action_items(self, ftype: FrameworkType,
                                dimensions: List[FrameworkDimension],
                                concerns: List[str]) -> List[str]:
        actions = []
        for dim in dimensions:
            if dim.name in concerns:
                if "Compensation" in dim.name or "Salary" in dim.name:
                    actions.append(f"Research market salary data for your role and location to strengthen your position on {dim.name}.")
                elif "Skills" in dim.name or "Gap" in dim.name:
                    actions.append(f"Create a 90-day learning plan to address the {dim.name} gap.")
                elif "Network" in dim.name:
                    actions.append(f"Attend 2-3 industry events or reach out to 5 contacts in the target field.")
                elif "Financial" in dim.name or "Runway" in dim.name:
                    actions.append(f"Build a 6-month emergency fund before making this transition.")
                else:
                    actions.append(f"Address the '{dim.name}' concern before proceeding—gather more information or take action to improve your score.")

        if not actions:
            actions.append("Your assessment looks strong. Review your top priorities and set a timeline for action.")

        return actions[:5]

    def _get_verdict(self, score_pct: float) -> str:
        if score_pct >= 80:
            return "Strong Go — Proceed with confidence"
        elif score_pct >= 65:
            return "Conditional Go — Address key concerns first"
        elif score_pct >= 50:
            return "Proceed with Caution — Significant areas need work"
        return "Not Ready — Major gaps need to be closed"

    def get_user_history(self, user_id: str) -> List[Dict]:
        return [{
            "id": r.id,
            "framework": r.framework_type.value,
            "score_pct": r.score_pct,
            "recommendation": r.recommendation,
            "risk_level": r.risk_level,
            "created_at": r.created_at.isoformat()
        } for r in self.user_sessions.get(user_id, [])]


decision_framework_service = DecisionFrameworkService()
