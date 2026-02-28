import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum


class ReversibilityLevel(Enum):
    FULLY_REVERSIBLE = "fully_reversible"
    PARTIALLY_REVERSIBLE = "partially_reversible"
    DIFFICULT_TO_REVERSE = "difficult_to_reverse"
    IRREVERSIBLE = "irreversible"


class ReversalCostType(Enum):
    FINANCIAL = "financial"
    TIME = "time"
    REPUTATION = "reputation"
    EMOTIONAL = "emotional"
    OPPORTUNITY = "opportunity"
    NETWORK = "network"


@dataclass
class ReversalCost:
    cost_type: ReversalCostType
    severity: str 
    description: str
    estimated_value: Optional[str] = None 
    mitigation: str = ""


@dataclass
class ReversalStep:
    order: int
    action: str
    timeline: str
    difficulty: str 
    notes: str = ""


@dataclass
class ReversalAnalysis:
    id: str
    user_id: str
    decision_id: str
    decision_description: str
    decision_type: str
    reversibility: ReversibilityLevel
    reversal_costs: List[ReversalCost]
    reversal_roadmap: List[ReversalStep]
    optimal_timing: str
    recommendation: str 
    confidence: float
    alternatives: List[str]
    created_at: datetime = field(default_factory=datetime.utcnow)


class ReversalAnalyzerService:
    DECISION_REVERSIBILITY = {
        "job_change": {
            "base_reversibility": ReversibilityLevel.PARTIALLY_REVERSIBLE,
            "typical_costs": [
                ReversalCost(ReversalCostType.REPUTATION, "medium",
                             "Short tenure may raise questions in future interviews",
                             mitigation="Frame as a learning experience; emphasize skills gained"),
                ReversalCost(ReversalCostType.FINANCIAL, "low",
                             "Potential gap in benefits during transition",
                             mitigation="Negotiate start dates to minimize coverage gaps"),
                ReversalCost(ReversalCostType.EMOTIONAL, "medium",
                             "Stress of another job search and transition",
                             mitigation="Take time between roles if financially possible"),
                ReversalCost(ReversalCostType.NETWORK, "low",
                             "May strain relationship with current employer",
                             mitigation="Leave professionally; maintain connections")
            ],
            "reversal_steps": [
                ReversalStep(1, "Assess what specifically isn't working", "Week 1", "easy",
                             "List tangible issues vs. adjustment period discomfort"),
                ReversalStep(2, "Attempt internal resolution (talk to manager, transfer)", "Weeks 2-4", "moderate",
                             "Many issues can be resolved without leaving"),
                ReversalStep(3, "Reach out to former employer (if applicable)", "Week 3", "moderate",
                             "If within 6 months, doors may still be open"),
                ReversalStep(4, "Begin external job search", "Weeks 3-6", "hard",
                             "Target roles that address root cause of dissatisfaction"),
                ReversalStep(5, "Negotiate exit and transition", "Weeks 6-8", "moderate",
                             "Give proper notice, protect relationships"),
            ],
            "timing_guidance": "Best within first 3-6 months. After 6 months, consider staying to reach 1 year."
        },
        "career_switch": {
            "base_reversibility": ReversibilityLevel.DIFFICULT_TO_REVERSE,
            "typical_costs": [
                ReversalCost(ReversalCostType.TIME, "high",
                             "Lost months/years of career progression in original field",
                             mitigation="Highlight transferable skills; position as growth"),
                ReversalCost(ReversalCostType.FINANCIAL, "medium",
                             "May need to accept lower salary returning to original field",
                             mitigation="Your experience in both fields is uniquely valuable"),
                ReversalCost(ReversalCostType.REPUTATION, "medium",
                             "May be perceived as indecisive",
                             mitigation="Frame as strategic exploration with clear learnings"),
                ReversalCost(ReversalCostType.EMOTIONAL, "high",
                             "Feeling of sunk cost and wasted effort",
                             mitigation="No career experience is truly wasted; skills compound")
            ],
            "reversal_steps": [
                ReversalStep(1, "Document all skills learned in new career", "Week 1", "easy"),
                ReversalStep(2, "Reconnect with your original professional network", "Weeks 1-2", "moderate"),
                ReversalStep(3, "Update resume highlighting cross-domain expertise", "Week 2", "moderate"),
                ReversalStep(4, "Position yourself as a bridge between both domains", "Weeks 2-4", "hard"),
                ReversalStep(5, "Target roles that value multi-domain experience", "Weeks 3-8", "hard"),
            ],
            "timing_guidance": "If within 1 year, reversal is straightforward. After 2+ years, consider hybrid roles instead."
        },
        "startup": {
            "base_reversibility": ReversibilityLevel.PARTIALLY_REVERSIBLE,
            "typical_costs": [
                ReversalCost(ReversalCostType.FINANCIAL, "high",
                             "Investment of savings, lost guaranteed income",
                             mitigation="Many employers value startup experience; your risk tolerance is an asset"),
                ReversalCost(ReversalCostType.TIME, "medium",
                             "Gap in traditional career progression",
                             mitigation="Startup experience demonstrates leadership and adaptability"),
                ReversalCost(ReversalCostType.EMOTIONAL, "high",
                             "Sense of failure if startup didn't succeed",
                             mitigation="Most successful founders failed before; frame as education"),
                ReversalCost(ReversalCostType.OPPORTUNITY, "medium",
                             "Equity that may become worthless",
                             mitigation="The learning and network are the real returns")
            ],
            "reversal_steps": [
                ReversalStep(1, "Assess whether pivoting the startup vs. returning to employment", "Week 1", "hard"),
                ReversalStep(2, "If returning: document your startup achievements quantifiably", "Week 1-2", "moderate"),
                ReversalStep(3, "Leverage your founder network for employment opportunities", "Weeks 2-3", "easy"),
                ReversalStep(4, "Target companies that value entrepreneurial experience", "Weeks 2-6", "moderate"),
                ReversalStep(5, "Negotiate for senior roles that match your true experience level", "Weeks 4-8", "hard"),
            ],
            "timing_guidance": "12-18 months is a reasonable runway. If no traction, returning to industry is healthy and respected."
        },
        "education": {
            "base_reversibility": ReversibilityLevel.PARTIALLY_REVERSIBLE,
            "typical_costs": [
                ReversalCost(ReversalCostType.FINANCIAL, "high",
                             "Tuition already spent, student loans",
                             mitigation="Partial completion still adds credential value"),
                ReversalCost(ReversalCostType.TIME, "medium",
                             "Months/years in program",
                             mitigation="Knowledge gained is permanent regardless of credential"),
            ],
            "reversal_steps": [
                ReversalStep(1, "Assess whether to pause vs. drop out", "Week 1", "moderate"),
                ReversalStep(2, "Explore leave-of-absence options", "Week 2", "easy"),
                ReversalStep(3, "Evaluate if credits transfer or partial credential is available", "Weeks 2-3", "moderate"),
                ReversalStep(4, "If dropping out: update career narrative", "Week 3", "moderate"),
                ReversalStep(5, "Re-enter job market with skills-based positioning", "Weeks 3-6", "moderate"),
            ],
            "timing_guidance": "Consider pausing before dropping out. A leave of absence buys time without burning bridges."
        },
        "relocation": {
            "base_reversibility": ReversibilityLevel.FULLY_REVERSIBLE,
            "typical_costs": [
                ReversalCost(ReversalCostType.FINANCIAL, "medium",
                             "Moving costs, lease breaks, housing market losses",
                             estimated_value="$3,000-$15,000 typical",
                             mitigation="Factor as a known cost of the decision experiment"),
                ReversalCost(ReversalCostType.EMOTIONAL, "medium",
                             "Disruption of establishing new roots",
                             mitigation="Each location gives you a broader perspective"),
            ],
            "reversal_steps": [
                ReversalStep(1, "Give the new location a fair trial (6+ months)", "Months 1-6", "easy"),
                ReversalStep(2, "If relocating back: secure job/income in original location", "Weeks 1-4", "moderate"),
                ReversalStep(3, "Handle logistics (lease, moving, etc.)", "Weeks 4-8", "moderate"),
                ReversalStep(4, "Reconnect with local network", "Ongoing", "easy"),
            ],
            "timing_guidance": "Give 6-12 months before deciding to reverse. Adjustment period is real."
        },
        "promotion": {
            "base_reversibility": ReversibilityLevel.DIFFICULT_TO_REVERSE,
            "typical_costs": [
                ReversalCost(ReversalCostType.REPUTATION, "high",
                             "Stepping back from a management/senior role is stigmatized",
                             mitigation="Frame as deliberate choice, not failure"),
                ReversalCost(ReversalCostType.FINANCIAL, "medium",
                             "Salary reduction if moving back to IC track",
                             mitigation="Some companies offer parallel IC tracks at similar comp"),
            ],
            "reversal_steps": [
                ReversalStep(1, "Clarify whether the issue is the role or the specific context", "Week 1", "moderate"),
                ReversalStep(2, "Explore lateral moves that preserve level but change scope", "Weeks 2-4", "moderate"),
                ReversalStep(3, "If returning to IC: find companies with strong IC ladders", "Weeks 3-6", "hard"),
                ReversalStep(4, "Negotiate title and comp that reflect your true experience", "Weeks 5-8", "hard"),
            ],
            "timing_guidance": "Within 6 months, transition back is smoother. The management experience adds value regardless."
        }
    }

    def __init__(self):
        self.analyses: Dict[str, List[ReversalAnalysis]] = defaultdict(list)

    def analyze_reversal(self, user_id: str, decision_id: str,
                         decision_description: str,
                         decision_type: str,
                         months_since_decision: int = 0,
                         current_regret_score: float = 50,
                         specific_issues: List[str] = None) -> Dict:
        template = self.DECISION_REVERSIBILITY.get(
            decision_type,
            self.DECISION_REVERSIBILITY.get("job_change") 
        )
        reversibility = template["base_reversibility"]
        if months_since_decision > 24:
            if reversibility == ReversibilityLevel.FULLY_REVERSIBLE:
                reversibility = ReversibilityLevel.PARTIALLY_REVERSIBLE
            elif reversibility == ReversibilityLevel.PARTIALLY_REVERSIBLE:
                reversibility = ReversibilityLevel.DIFFICULT_TO_REVERSE

        recommendation = self._generate_recommendation(
            reversibility, current_regret_score, months_since_decision, specific_issues
        )
        confidence = self._calculate_confidence(
            current_regret_score, months_since_decision, len(specific_issues or [])
        )

        alternatives = self._generate_alternatives(
            decision_type, specific_issues or []
        )
        optimal_timing = template["timing_guidance"]
        if months_since_decision > 18:
            optimal_timing = (
                "You're past the typical reversal window, but it's never too late to course-correct. "
                "Consider partial reversals or pivot strategies rather than full reversal."
            )

        analysis = ReversalAnalysis(
            id=str(uuid.uuid4()),
            user_id=user_id,
            decision_id=decision_id,
            decision_description=decision_description,
            decision_type=decision_type,
            reversibility=reversibility,
            reversal_costs=template["typical_costs"],
            reversal_roadmap=template["reversal_steps"],
            optimal_timing=optimal_timing,
            recommendation=recommendation,
            confidence=confidence,
            alternatives=alternatives
        )
        self.analyses[user_id].append(analysis)

        return {
            "analysis_id": analysis.id,
            "decision": decision_description,
            "decision_type": decision_type,
            "reversibility": reversibility.value,
            "recommendation": recommendation,
            "confidence": round(confidence, 2),
            "optimal_timing": optimal_timing,
            "costs": [{
                "type": c.cost_type.value,
                "severity": c.severity,
                "description": c.description,
                "estimated_value": c.estimated_value,
                "mitigation": c.mitigation
            } for c in template["typical_costs"]],
            "roadmap": [{
                "step": s.order,
                "action": s.action,
                "timeline": s.timeline,
                "difficulty": s.difficulty,
                "notes": s.notes
            } for s in template["reversal_steps"]],
            "alternatives": alternatives,
            "months_since_decision": months_since_decision,
            "regret_score": current_regret_score
        }

    def _generate_recommendation(self, reversibility: ReversibilityLevel,
                                  regret_score: float, months: int,
                                  issues: List[str] = None) -> str:
        if regret_score < 30:
            return (
                "Your regret is relatively low. Consider whether this is a temporary adjustment period. "
                "Give it more time before making a drastic change."
            )

        if regret_score > 70 and months < 6:
            if reversibility in (ReversibilityLevel.FULLY_REVERSIBLE, ReversibilityLevel.PARTIALLY_REVERSIBLE):
                return (
                    "High regret with a short time investment suggests reversal is advisable. "
                    "The costs are manageable and you haven't yet built deep commitments to this path."
                )
            return (
                "High regret, but reversal is difficult. Focus on adapting: "
                "identify the specific issues and address them within your current situation."
            )

        if regret_score > 70 and months >= 12:
            return (
                "Significant regret over a sustained period suggests a change is needed. "
                "However, consider a strategic pivot rather than full reversal. "
                "Your experience in this path has value—find a role that leverages both."
            )

        return (
            "Moderate regret suggests mixed feelings. Before reversing, try to isolate "
            "the specific factors causing dissatisfaction. Often, targeted changes "
            "(new team, adjusted responsibilities, better boundaries) can address the root cause."
        )

    def _calculate_confidence(self, regret_score: float, months: int,
                               issues_count: int) -> float:
        base = 0.6
        if months >= 6:
            base += 0.1  
        if issues_count >= 3:
            base += 0.1 
        if regret_score > 70 or regret_score < 20:
            base += 0.1 
        return min(0.95, base)

    def _generate_alternatives(self, decision_type: str,
                                issues: List[str]) -> List[str]:
        alternatives = {
            "job_change": [
                "Request internal transfer to a different team",
                "Negotiate role scope or responsibility changes",
                "Set a 6-month timeline to reassess with specific improvement criteria",
                "Seek mentorship within the new organization",
                "Propose a pilot project in your preferred area"
            ],
            "career_switch": [
                "Find a hybrid role bridging both careers",
                "Do side projects in your original field while staying in the new one",
                "Become a consultant leveraging expertise from both domains",
                "Transition gradually rather than cold-turkey reverting"
            ],
            "startup": [
                "Pivot the startup idea rather than abandoning it",
                "Find a co-founder to share the burden",
                "Go part-time on the startup while taking a day job",
                "Transition to a startup-adjacent role (VC, accelerator, advisor)"
            ],
            "education": [
                "Switch to part-time or online format",
                "Take a leave of absence to test the alternative",
                "Transfer credits to a different program",
                "Consider completing for the credential even if career plans change"
            ],
            "relocation": [
                "Try working remotely for your previous employer",
                "Give the new location 3 more months with deliberate community building",
                "Explore a third location that might be better than both",
                "Negotiate a hybrid arrangement with periodic travel"
            ],
            "promotion": [
                "Negotiate to keep the title but shift responsibilities",
                "Ask for an executive coach to develop in weak areas",
                "Propose a 'player-coach' role combining IC and management",
                "Move to a company with stronger IC career ladders"
            ]
        }
        return alternatives.get(decision_type, [
            "Identify the specific issue and address it directly",
            "Set a timeline for reassessment before making changes",
            "Seek advice from someone who's navigated a similar situation",
            "Consider incremental adjustments rather than full reversal"
        ])

    def get_analysis_history(self, user_id: str) -> List[Dict]:
        return [{
            "id": a.id,
            "decision": a.decision_description,
            "type": a.decision_type,
            "reversibility": a.reversibility.value,
            "recommendation": a.recommendation[:100] + "..." if len(a.recommendation) > 100 else a.recommendation,
            "confidence": a.confidence,
            "created_at": a.created_at.isoformat()
        } for a in self.analyses.get(user_id, [])]


reversal_analyzer_service = ReversalAnalyzerService()
