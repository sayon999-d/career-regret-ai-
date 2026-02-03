from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

class DecisionCategory(str, Enum):
    JOB_OFFER = "job_offer"
    CAREER_SWITCH = "career_switch"
    PROMOTION = "promotion"
    FREELANCE = "freelance"
    EDUCATION = "education"
    STARTUP = "startup"
    RELOCATION = "relocation"
    SALARY_NEGOTIATION = "salary_negotiation"

@dataclass
class TemplateQuestion:
    id: str
    question: str
    question_type: str
    weight: float
    options: List[str] = field(default_factory=list)
    min_value: int = 0
    max_value: int = 100
    help_text: str = ""
    category: str = ""

@dataclass
class DecisionTemplate:
    id: str
    name: str
    category: DecisionCategory
    description: str
    icon: str
    questions: List[TemplateQuestion]
    pros_factors: List[str]
    cons_factors: List[str]
    key_considerations: List[str]
    estimated_time: str

@dataclass
class TemplateResult:
    template_id: str
    user_id: str
    answers: Dict[str, Any]
    score: float
    recommendation: str
    pros: List[Dict[str, Any]]
    cons: List[Dict[str, Any]]
    analysis: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)

class DecisionTemplateService:
    def __init__(self):
        self.templates: Dict[str, DecisionTemplate] = {}
        self.user_results: Dict[str, List[TemplateResult]] = {}
        self._initialize_templates()

    def _initialize_templates(self):
        self.templates["job_offer"] = DecisionTemplate(
            id="job_offer",
            name="Job Offer Analysis",
            category=DecisionCategory.JOB_OFFER,
            description="Evaluate a new job opportunity with structured analysis",
            icon="briefcase",
            estimated_time="5-7 minutes",
            questions=[
                TemplateQuestion(
                    id="salary_increase",
                    question="What is the salary increase compared to your current position?",
                    question_type="slider",
                    weight=0.20,
                    min_value=-20,
                    max_value=100,
                    help_text="Enter percentage change (e.g., 25 for 25% increase)",
                    category="financial"
                ),
                TemplateQuestion(
                    id="benefits_quality",
                    question="How do the benefits compare? (health, PTO, retirement)",
                    question_type="slider",
                    weight=0.10,
                    min_value=0,
                    max_value=100,
                    help_text="0 = much worse, 50 = similar, 100 = much better",
                    category="financial"
                ),
                TemplateQuestion(
                    id="growth_potential",
                    question="Rate the career growth potential at this new role",
                    question_type="slider",
                    weight=0.20,
                    min_value=0,
                    max_value=100,
                    help_text="Consider promotion paths and skill development",
                    category="growth"
                ),
                TemplateQuestion(
                    id="company_stability",
                    question="How stable is the new company?",
                    question_type="choice",
                    weight=0.15,
                    options=["Startup (risky)", "Growing company", "Established", "Industry leader"],
                    category="stability"
                ),
                TemplateQuestion(
                    id="culture_fit",
                    question="How well does the company culture match your values?",
                    question_type="slider",
                    weight=0.15,
                    min_value=0,
                    max_value=100,
                    help_text="Based on interviews and research",
                    category="culture"
                ),
                TemplateQuestion(
                    id="commute_impact",
                    question="How does the commute/location compare?",
                    question_type="choice",
                    weight=0.10,
                    options=["Much worse", "Slightly worse", "Similar", "Better", "Remote option"],
                    category="lifestyle"
                ),
                TemplateQuestion(
                    id="work_life_balance",
                    question="Expected work-life balance at the new role",
                    question_type="slider",
                    weight=0.10,
                    min_value=0,
                    max_value=100,
                    help_text="Based on role expectations and company reputation",
                    category="lifestyle"
                )
            ],
            pros_factors=["Higher salary", "Better growth", "Improved benefits", "Better culture", "Remote work"],
            cons_factors=["Job security risk", "Longer commute", "Unknown environment", "Leaving current team"],
            key_considerations=[
                "Consider the total compensation, not just base salary",
                "Research the company's financial health and recent news",
                "Talk to current or former employees if possible",
                "Ensure the role aligns with your 5-year career plan"
            ]
        )

        self.templates["career_switch"] = DecisionTemplate(
            id="career_switch",
            name="Career Change Evaluation",
            category=DecisionCategory.CAREER_SWITCH,
            description="Assess whether to switch to a completely different career path",
            icon="refresh",
            estimated_time="8-10 minutes",
            questions=[
                TemplateQuestion(
                    id="passion_level",
                    question="How passionate are you about the new career?",
                    question_type="slider",
                    weight=0.20,
                    min_value=0,
                    max_value=100,
                    help_text="Be honest - is this a dream or just an escape?",
                    category="motivation"
                ),
                TemplateQuestion(
                    id="transferable_skills",
                    question="What percentage of your current skills transfer to the new career?",
                    question_type="slider",
                    weight=0.15,
                    min_value=0,
                    max_value=100,
                    help_text="Consider both hard and soft skills",
                    category="skills"
                ),
                TemplateQuestion(
                    id="financial_runway",
                    question="How many months of expenses can you cover during transition?",
                    question_type="choice",
                    weight=0.20,
                    options=["Less than 3 months", "3-6 months", "6-12 months", "12+ months"],
                    category="financial"
                ),
                TemplateQuestion(
                    id="market_demand",
                    question="How strong is the job market for the new career?",
                    question_type="choice",
                    weight=0.15,
                    options=["Declining", "Stable", "Growing", "Booming"],
                    category="market"
                ),
                TemplateQuestion(
                    id="education_needed",
                    question="Additional education/certification required?",
                    question_type="choice",
                    weight=0.10,
                    options=["None", "Short courses", "Bootcamp/Certificate", "Degree required"],
                    category="preparation"
                ),
                TemplateQuestion(
                    id="support_system",
                    question="How supportive is your personal network about this change?",
                    question_type="slider",
                    weight=0.10,
                    min_value=0,
                    max_value=100,
                    help_text="Family, friends, mentors",
                    category="support"
                ),
                TemplateQuestion(
                    id="current_satisfaction",
                    question="How unsatisfied are you with your current career? (higher = more unsatisfied)",
                    question_type="slider",
                    weight=0.10,
                    min_value=0,
                    max_value=100,
                    help_text="Are you running TO something or FROM something?",
                    category="motivation"
                )
            ],
            pros_factors=["Follow passion", "Fresh start", "New challenges", "Industry growth", "Better alignment"],
            cons_factors=["Starting over", "Income drop", "Learning curve", "Network rebuilding", "Uncertainty"],
            key_considerations=[
                "Can you test the new career before fully committing?",
                "Consider transitional roles that bridge both careers",
                "Build network in target field before switching",
                "Have a realistic timeline - most switches take 1-2 years"
            ]
        )

        self.templates["promotion"] = DecisionTemplate(
            id="promotion",
            name="Promotion Decision",
            category=DecisionCategory.PROMOTION,
            description="Should you pursue or accept a promotion?",
            icon="arrow-up",
            estimated_time="5 minutes",
            questions=[
                TemplateQuestion(
                    id="readiness",
                    question="How ready do you feel for the new responsibilities?",
                    question_type="slider",
                    weight=0.20,
                    min_value=0,
                    max_value=100,
                    category="skills"
                ),
                TemplateQuestion(
                    id="salary_bump",
                    question="What is the salary increase?",
                    question_type="choice",
                    weight=0.15,
                    options=["Less than 10%", "10-20%", "20-30%", "More than 30%"],
                    category="financial"
                ),
                TemplateQuestion(
                    id="work_hours",
                    question="How much will your work hours increase?",
                    question_type="choice",
                    weight=0.15,
                    options=["Same hours", "Slightly more", "Significantly more", "Much more"],
                    category="lifestyle"
                ),
                TemplateQuestion(
                    id="management_interest",
                    question="How interested are you in managing people?",
                    question_type="slider",
                    weight=0.20,
                    min_value=0,
                    max_value=100,
                    help_text="Only if this is a management role",
                    category="preference"
                ),
                TemplateQuestion(
                    id="stress_tolerance",
                    question="Can you handle increased stress and pressure?",
                    question_type="slider",
                    weight=0.15,
                    min_value=0,
                    max_value=100,
                    category="personal"
                ),
                TemplateQuestion(
                    id="alternative_path",
                    question="Are there other growth paths you'd prefer?",
                    question_type="choice",
                    weight=0.15,
                    options=["This is my preferred path", "I'd consider alternatives", "I prefer IC track"],
                    category="growth"
                )
            ],
            pros_factors=["Higher salary", "Career advancement", "More influence", "New challenges", "Recognition"],
            cons_factors=["More stress", "Less hands-on work", "Longer hours", "Management challenges"],
            key_considerations=[
                "Promotion often means less of what you love doing",
                "Consider if you want to manage people or just projects",
                "Negotiate the salary - promotions are often under-compensated",
                "Ask about support and training for the new role"
            ]
        )

        self.templates["freelance"] = DecisionTemplate(
            id="freelance",
            name="Go Freelance/Independent",
            category=DecisionCategory.FREELANCE,
            description="Evaluate transitioning to freelance or consulting",
            icon="user",
            estimated_time="6 minutes",
            questions=[
                TemplateQuestion(
                    id="financial_buffer",
                    question="How many months of savings do you have?",
                    question_type="choice",
                    weight=0.25,
                    options=["Less than 3", "3-6 months", "6-12 months", "12+ months"],
                    category="financial"
                ),
                TemplateQuestion(
                    id="client_base",
                    question="Do you have potential clients lined up?",
                    question_type="choice",
                    weight=0.20,
                    options=["None", "A few leads", "1-2 confirmed", "Multiple confirmed"],
                    category="business"
                ),
                TemplateQuestion(
                    id="self_discipline",
                    question="Rate your self-discipline and motivation",
                    question_type="slider",
                    weight=0.15,
                    min_value=0,
                    max_value=100,
                    category="personal"
                ),
                TemplateQuestion(
                    id="market_rate",
                    question="Can you charge 1.5-2x your current hourly equivalent?",
                    question_type="choice",
                    weight=0.15,
                    options=["Unlikely", "Maybe with time", "Yes", "Already have offers"],
                    category="financial"
                ),
                TemplateQuestion(
                    id="benefits_plan",
                    question="Have you planned for health insurance and benefits?",
                    question_type="boolean",
                    weight=0.10,
                    category="planning"
                ),
                TemplateQuestion(
                    id="risk_tolerance",
                    question="How comfortable are you with income uncertainty?",
                    question_type="slider",
                    weight=0.15,
                    min_value=0,
                    max_value=100,
                    category="personal"
                )
            ],
            pros_factors=["Freedom", "Higher earning potential", "Choose your work", "Flexibility", "No boss"],
            cons_factors=["Income instability", "No benefits", "Self-employment taxes", "Finding clients", "Isolation"],
            key_considerations=[
                "Most freelancers underestimate time spent on non-billable work",
                "Consider starting part-time while employed",
                "Build 6-12 months runway before jumping",
                "Network and build your brand before leaving"
            ]
        )

        self.templates["education"] = DecisionTemplate(
            id="education",
            name="Further Education Decision",
            category=DecisionCategory.EDUCATION,
            description="Should you pursue additional education or degree?",
            icon="book",
            estimated_time="5 minutes",
            questions=[
                TemplateQuestion(
                    id="career_requirement",
                    question="Is this education required for your career goals?",
                    question_type="choice",
                    weight=0.25,
                    options=["Nice to have", "Helpful", "Strongly preferred", "Required"],
                    category="necessity"
                ),
                TemplateQuestion(
                    id="roi_timeline",
                    question="Expected time to recoup the investment?",
                    question_type="choice",
                    weight=0.20,
                    options=["5+ years", "3-5 years", "1-3 years", "Less than 1 year"],
                    category="financial"
                ),
                TemplateQuestion(
                    id="program_quality",
                    question="Rate the quality/reputation of the program",
                    question_type="slider",
                    weight=0.15,
                    min_value=0,
                    max_value=100,
                    category="quality"
                ),
                TemplateQuestion(
                    id="time_commitment",
                    question="Can you manage the time commitment?",
                    question_type="slider",
                    weight=0.15,
                    min_value=0,
                    max_value=100,
                    help_text="Consider work, family, and personal time",
                    category="feasibility"
                ),
                TemplateQuestion(
                    id="employer_support",
                    question="Will your employer support this (time, money)?",
                    question_type="choice",
                    weight=0.15,
                    options=["No support", "Time flexibility only", "Partial funding", "Full support"],
                    category="support"
                ),
                TemplateQuestion(
                    id="alternative_learning",
                    question="Could you learn this through alternatives (online, bootcamp)?",
                    question_type="choice",
                    weight=0.10,
                    options=["No - need formal degree", "Partially", "Yes but degree preferred", "Yes completely"],
                    category="alternatives"
                )
            ],
            pros_factors=["Career advancement", "Higher salary", "New skills", "Network", "Credential"],
            cons_factors=["Cost", "Time commitment", "Opportunity cost", "No guaranteed ROI"],
            key_considerations=[
                "Calculate the true total cost including lost income",
                "Talk to graduates about actual career outcomes",
                "Consider if the credential is truly required or just preferred",
                "Part-time and online options may offer better ROI"
            ]
        )

        self.templates["startup"] = DecisionTemplate(
            id="startup",
            name="Start a Business",
            category=DecisionCategory.STARTUP,
            description="Evaluate starting your own business or joining an early startup",
            icon="rocket",
            estimated_time="7 minutes",
            questions=[
                TemplateQuestion(
                    id="idea_validation",
                    question="Have you validated your idea with potential customers?",
                    question_type="choice",
                    weight=0.20,
                    options=["Just an idea", "Some research", "Talked to customers", "Have paying users"],
                    category="validation"
                ),
                TemplateQuestion(
                    id="financial_runway",
                    question="How long can you survive without income?",
                    question_type="choice",
                    weight=0.20,
                    options=["Less than 6 months", "6-12 months", "12-18 months", "18+ months"],
                    category="financial"
                ),
                TemplateQuestion(
                    id="domain_expertise",
                    question="Your expertise level in this industry/domain",
                    question_type="slider",
                    weight=0.15,
                    min_value=0,
                    max_value=100,
                    category="skills"
                ),
                TemplateQuestion(
                    id="co_founder",
                    question="Do you have co-founders or a team?",
                    question_type="choice",
                    weight=0.15,
                    options=["Solo", "1 co-founder", "Full founding team", "Team + advisors"],
                    category="team"
                ),
                TemplateQuestion(
                    id="market_timing",
                    question="How is the market timing for this?",
                    question_type="choice",
                    weight=0.15,
                    options=["Too early", "Competitive", "Good timing", "Perfect timing"],
                    category="market"
                ),
                TemplateQuestion(
                    id="failure_tolerance",
                    question="How would you handle failure?",
                    question_type="slider",
                    weight=0.15,
                    min_value=0,
                    max_value=100,
                    help_text="Emotionally and financially",
                    category="personal"
                )
            ],
            pros_factors=["Ownership", "Unlimited upside", "Build something", "Freedom", "Impact"],
            cons_factors=["High failure rate", "Financial risk", "Stress", "No stability", "Long hours"],
            key_considerations=[
                "80% of startups fail - plan for this possibility",
                "Validate before you build",
                "Consider starting as a side project first",
                "Your network and ability to recruit is critical"
            ]
        )

    def get_all_templates(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": t.id,
                "name": t.name,
                "category": t.category.value,
                "description": t.description,
                "icon": t.icon,
                "estimated_time": t.estimated_time,
                "question_count": len(t.questions)
            }
            for t in self.templates.values()
        ]

    def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        template = self.templates.get(template_id)
        if not template:
            return None

        return {
            "id": template.id,
            "name": template.name,
            "category": template.category.value,
            "description": template.description,
            "icon": template.icon,
            "estimated_time": template.estimated_time,
            "questions": [
                {
                    "id": q.id,
                    "question": q.question,
                    "type": q.question_type,
                    "weight": q.weight,
                    "options": q.options,
                    "min_value": q.min_value,
                    "max_value": q.max_value,
                    "help_text": q.help_text,
                    "category": q.category
                }
                for q in template.questions
            ],
            "pros_factors": template.pros_factors,
            "cons_factors": template.cons_factors,
            "key_considerations": template.key_considerations
        }

    def analyze_template(self, template_id: str, answers: Dict[str, Any], user_id: str = None) -> Dict[str, Any]:
        template = self.templates.get(template_id)
        if not template:
            return {"error": "Template not found"}

        total_score = 0
        max_score = 0
        category_scores: Dict[str, List[float]] = {}
        pros_analysis = []
        cons_analysis = []

        for question in template.questions:
            answer = answers.get(question.id)
            if answer is None:
                continue

            score = self._calculate_question_score(question, answer)
            weighted_score = score * question.weight
            total_score += weighted_score
            max_score += question.weight

            if question.category not in category_scores:
                category_scores[question.category] = []
            category_scores[question.category].append(score)

            if score >= 70:
                pros_analysis.append({
                    "factor": question.question[:50] + "...",
                    "score": score,
                    "impact": "positive"
                })
            elif score <= 30:
                cons_analysis.append({
                    "factor": question.question[:50] + "...",
                    "score": score,
                    "impact": "negative"
                })

        final_score = (total_score / max_score * 100) if max_score > 0 else 50
        final_score = max(0, min(100, final_score))

        recommendation = self._generate_recommendation(final_score, template.category)

        category_analysis = {
            cat: sum(scores) / len(scores)
            for cat, scores in category_scores.items()
            if scores
        }

        result = TemplateResult(
            template_id=template_id,
            user_id=user_id or "anonymous",
            answers=answers,
            score=final_score,
            recommendation=recommendation,
            pros=pros_analysis,
            cons=cons_analysis,
            analysis={
                "category_scores": category_analysis,
                "strongest_area": max(category_analysis.items(), key=lambda x: x[1])[0] if category_analysis else None,
                "weakest_area": min(category_analysis.items(), key=lambda x: x[1])[0] if category_analysis else None
            }
        )

        if user_id:
            if user_id not in self.user_results:
                self.user_results[user_id] = []
            self.user_results[user_id].append(result)

        return {
            "template_id": template_id,
            "template_name": template.name,
            "score": round(final_score, 1),
            "recommendation": recommendation,
            "decision_indicator": self._get_decision_indicator(final_score),
            "pros": pros_analysis[:5],
            "cons": cons_analysis[:5],
            "category_scores": {k: round(v, 1) for k, v in category_analysis.items()},
            "strongest_area": result.analysis["strongest_area"],
            "weakest_area": result.analysis["weakest_area"],
            "key_considerations": template.key_considerations,
            "next_steps": self._generate_next_steps(final_score, template.category)
        }

    def _calculate_question_score(self, question: TemplateQuestion, answer: Any) -> float:
        if question.question_type == "slider":
            value = float(answer)
            min_val = question.min_value
            max_val = question.max_value
            if max_val != min_val:
                normalized = ((value - min_val) / (max_val - min_val)) * 100
            else:
                normalized = 50
            return max(0, min(100, normalized))

        elif question.question_type == "boolean":
            return 100 if answer else 0

        elif question.question_type == "choice":
            if not question.options:
                return 50
            try:
                index = question.options.index(answer)
                return (index / (len(question.options) - 1)) * 100
            except (ValueError, ZeroDivisionError):
                return 50

        return 50

    def _generate_recommendation(self, score: float, category: DecisionCategory) -> str:
        if score >= 80:
            return f"Strong indicators suggest this {category.value.replace('_', ' ')} could be a great move. The analysis shows favorable conditions across most factors."
        elif score >= 65:
            return f"Good indicators for this {category.value.replace('_', ' ')}. Most factors are positive, but pay attention to the areas that scored lower."
        elif score >= 50:
            return f"Mixed indicators for this {category.value.replace('_', ' ')}. Consider addressing the weaker areas before making a decision."
        elif score >= 35:
            return f"Caution advised for this {category.value.replace('_', ' ')}. Several factors suggest this might not be the right time or opportunity."
        else:
            return f"Strong indicators suggest waiting or reconsidering this {category.value.replace('_', ' ')}. Multiple factors are unfavorable."

    def _get_decision_indicator(self, score: float) -> Dict[str, Any]:
        if score >= 80:
            return {"label": "Strong Yes", "color": "success", "confidence": "high"}
        elif score >= 65:
            return {"label": "Lean Yes", "color": "success", "confidence": "moderate"}
        elif score >= 50:
            return {"label": "Neutral", "color": "warning", "confidence": "low"}
        elif score >= 35:
            return {"label": "Lean No", "color": "danger", "confidence": "moderate"}
        else:
            return {"label": "Strong No", "color": "danger", "confidence": "high"}

    def _generate_next_steps(self, score: float, category: DecisionCategory) -> List[str]:
        steps = []

        if score >= 65:
            steps.append("Create a detailed transition plan with timeline")
            steps.append("Have conversations with key stakeholders")
            steps.append("Prepare for the change financially and emotionally")
        elif score >= 50:
            steps.append("Address the areas that scored below 50%")
            steps.append("Gather more information before deciding")
            steps.append("Consider setting a deadline for your decision")
        else:
            steps.append("Identify what would need to change for this to be viable")
            steps.append("Explore alternative options")
            steps.append("Revisit this decision in 3-6 months")

        if category == DecisionCategory.JOB_OFFER:
            steps.append("Negotiate the offer before accepting")
        elif category == DecisionCategory.CAREER_SWITCH:
            steps.append("Build skills and network in target field")
        elif category == DecisionCategory.STARTUP:
            steps.append("Validate your idea with 10 potential customers")

        return steps[:4]

    def get_user_history(self, user_id: str) -> List[Dict[str, Any]]:
        results = self.user_results.get(user_id, [])
        return [
            {
                "template_id": r.template_id,
                "score": r.score,
                "recommendation": r.recommendation[:100] + "...",
                "created_at": r.created_at.isoformat()
            }
            for r in results[-10:]
        ]
