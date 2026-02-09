from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import random
import hashlib

@dataclass
class AggregatePattern:

    decision_type: str
    sample_size: int
    avg_regret_predicted: float
    avg_actual_regret: float
    success_rate: float
    common_emotions: List[str]
    common_concerns: List[str]
    common_outcomes: List[str]
    top_factors: List[str]

@dataclass
class AnonymizedStory:

    id: str
    decision_type: str
    industry: str
    summary: str
    outcome: str
    regret_level: str
    key_learnings: List[str]
    time_since_decision: str
    would_decide_same: bool
    upvotes: int = 0

@dataclass
class SocialProofData:

    decision_type: str
    total_similar_decisions: int
    success_percentage: float
    avg_regret: float
    common_choice: str
    top_considerations: List[str]
    sample_stories: List[AnonymizedStory]
    confidence_boost: str

class CommunityInsightsService:


    COMMUNITY_PATTERNS = {
        'job_change': {
            'sample_size': 15420,
            'success_rate': 0.72,
            'avg_predicted_regret': 0.38,
            'avg_actual_regret': 0.28,
            'common_emotions': ['excited', 'anxious', 'hopeful'],
            'common_concerns': ['salary negotiation', 'culture fit', 'career growth'],
            'common_outcomes': ['higher salary', 'new skills', 'better work-life balance'],
            'top_factors': ['compensation', 'growth opportunities', 'company culture']
        },
        'career_switch': {
            'sample_size': 8350,
            'success_rate': 0.65,
            'avg_predicted_regret': 0.52,
            'avg_actual_regret': 0.35,
            'common_emotions': ['uncertain', 'excited', 'fearful'],
            'common_concerns': ['starting over', 'income drop', 'skill gaps'],
            'common_outcomes': ['personal fulfillment', 'new network', 'learning curve'],
            'top_factors': ['passion alignment', 'long-term potential', 'transferable skills']
        },
        'startup': {
            'sample_size': 4280,
            'success_rate': 0.55,
            'avg_predicted_regret': 0.58,
            'avg_actual_regret': 0.42,
            'common_emotions': ['excited', 'overwhelmed', 'motivated'],
            'common_concerns': ['financial risk', 'work-life balance', 'uncertainty'],
            'common_outcomes': ['major learning', 'network growth', 'flexibility'],
            'top_factors': ['financial runway', 'market timing', 'team strength']
        },
        'education': {
            'sample_size': 6120,
            'success_rate': 0.78,
            'avg_predicted_regret': 0.32,
            'avg_actual_regret': 0.22,
            'common_emotions': ['hopeful', 'motivated', 'anxious'],
            'common_concerns': ['cost', 'time commitment', 'ROI'],
            'common_outcomes': ['career advancement', 'higher salary', 'new opportunities'],
            'top_factors': ['program reputation', 'career outcomes', 'networking']
        },
        'freelance': {
            'sample_size': 3890,
            'success_rate': 0.62,
            'avg_predicted_regret': 0.45,
            'avg_actual_regret': 0.38,
            'common_emotions': ['excited', 'uncertain', 'motivated'],
            'common_concerns': ['income stability', 'finding clients', 'isolation'],
            'common_outcomes': ['flexibility', 'varied work', 'income volatility'],
            'top_factors': ['savings runway', 'existing network', 'marketable skills']
        },
        'promotion': {
            'sample_size': 12450,
            'success_rate': 0.75,
            'avg_predicted_regret': 0.30,
            'avg_actual_regret': 0.25,
            'common_emotions': ['excited', 'confident', 'stressed'],
            'common_concerns': ['increased responsibility', 'work-life balance', 'expectations'],
            'common_outcomes': ['higher pay', 'new challenges', 'leadership skills'],
            'top_factors': ['readiness', 'team support', 'growth potential']
        },
        'relocation': {
            'sample_size': 5670,
            'success_rate': 0.68,
            'avg_predicted_regret': 0.42,
            'avg_actual_regret': 0.32,
            'common_emotions': ['excited', 'anxious', 'hopeful'],
            'common_concerns': ['leaving network', 'cost of living', 'family impact'],
            'common_outcomes': ['new experiences', 'career growth', 'lifestyle change'],
            'top_factors': ['career opportunity', 'quality of life', 'family considerations']
        }
    }

    SAMPLE_STORIES = {
        'job_change': [
            {
                'summary': 'Left a stable corporate role for a fast-growing startup',
                'outcome': 'Tripled responsibilities but also salary within 2 years',
                'regret': 'none',
                'learnings': ['Taking calculated risks paid off', 'Culture fit matters more than I thought'],
                'would_same': True,
                'industry': 'technology'
            },
            {
                'summary': 'Switched from toxic workplace despite salary cut',
                'outcome': 'Mental health improved dramatically, salary recovered in 1 year',
                'regret': 'none',
                'learnings': ['Never underestimate workplace culture impact', 'Salary isnt everything'],
                'would_same': True,
                'industry': 'finance'
            },
            {
                'summary': 'Jumped for 40% raise without researching culture',
                'outcome': 'Left after 8 months due to poor management',
                'regret': 'moderate',
                'learnings': ['Do thorough due diligence', 'Talk to current employees before accepting'],
                'would_same': False,
                'industry': 'technology'
            }
        ],
        'career_switch': [
            {
                'summary': 'Engineer to Product Manager transition after 5 years',
                'outcome': 'Took 1 year to feel competent, now thriving',
                'regret': 'low',
                'learnings': ['Transferable skills are undervalued', 'Patience during learning curve is key'],
                'would_same': True,
                'industry': 'technology'
            },
            {
                'summary': 'Lawyer to UX Designer via bootcamp',
                'outcome': 'Income dropped 30% initially, recovered in 3 years',
                'regret': 'none',
                'learnings': ['Following passion is worth the temporary setback', 'Prior career experience adds unique value'],
                'would_same': True,
                'industry': 'technology'
            },
            {
                'summary': 'Finance to teaching without proper preparation',
                'outcome': 'Struggled with income and workload for 2 years',
                'regret': 'moderate',
                'learnings': ['Prepare financially before major switches', 'Shadow the new career first'],
                'would_same': False,
                'industry': 'education'
            }
        ],
        'startup': [
            {
                'summary': 'Left FAANG to join Series A startup as early employee',
                'outcome': 'IPO after 4 years, significant equity payout',
                'regret': 'none',
                'learnings': ['Timing and team selection crucial', 'Equity can be life-changing'],
                'would_same': True,
                'industry': 'technology'
            },
            {
                'summary': 'Founded a startup after 3 years in corporate',
                'outcome': 'Failed after 18 months but learned immensely',
                'regret': 'low',
                'learnings': ['Failure is a powerful teacher', 'Network built during startup is valuable'],
                'would_same': True,
                'industry': 'technology'
            },
            {
                'summary': 'Joined pre-seed startup without runway research',
                'outcome': 'Company ran out of money in 6 months',
                'regret': 'high',
                'learnings': ['Always verify funding and runway', 'Get offers in writing'],
                'would_same': False,
                'industry': 'fintech'
            }
        ]
    }

    def __init__(self):
        self.user_contributions: Dict[str, List[Dict]] = defaultdict(list)
        self.story_upvotes: Dict[str, int] = defaultdict(int)

    def get_social_proof(self, decision_type: str) -> SocialProofData:

        pattern = self.COMMUNITY_PATTERNS.get(decision_type, self.COMMUNITY_PATTERNS['job_change'])
        stories = self._get_sample_stories(decision_type)

        return SocialProofData(
            decision_type=decision_type,
            total_similar_decisions=pattern['sample_size'],
            success_percentage=pattern['success_rate'] * 100,
            avg_regret=pattern['avg_actual_regret'],
            common_choice=self._get_common_choice(decision_type),
            top_considerations=pattern['top_factors'],
            sample_stories=stories,
            confidence_boost=self._get_confidence_message(pattern)
        )

    def _get_sample_stories(self, decision_type: str, limit: int = 3) -> List[AnonymizedStory]:

        story_data = self.SAMPLE_STORIES.get(decision_type, self.SAMPLE_STORIES['job_change'])

        stories = []
        for i, s in enumerate(story_data[:limit]):
            story_id = hashlib.sha256(f"{decision_type}_{i}".encode()).hexdigest()[:8]
            stories.append(AnonymizedStory(
                id=story_id,
                decision_type=decision_type,
                industry=s.get('industry', 'technology'),
                summary=s['summary'],
                outcome=s['outcome'],
                regret_level=s['regret'],
                key_learnings=s['learnings'],
                time_since_decision=f"{random.randint(6, 24)} months ago",
                would_decide_same=s['would_same'],
                upvotes=random.randint(20, 200)
            ))

        return stories

    def _get_common_choice(self, decision_type: str) -> str:

        choices = {
            'job_change': 'Accept offers with 15%+ salary increase and clear growth path',
            'career_switch': 'Prepare for 6-12 months before making the jump',
            'startup': 'Join startups at Series A+ stage with 18+ months runway',
            'education': 'Part-time programs while working to minimize risk',
            'freelance': 'Build side income to 50% of salary before going full-time',
            'promotion': 'Accept when feeling 70% ready',
            'relocation': 'Negotiate relocation package covering at least 1 month expenses'
        }
        return choices.get(decision_type, choices['job_change'])

    def _get_confidence_message(self, pattern: Dict) -> str:

        success_rate = pattern['success_rate']
        sample_size = pattern['sample_size']

        if success_rate >= 0.75:
            return f"Great news! {int(success_rate * 100)}% of {sample_size:,} people who made similar decisions would do it again."
        elif success_rate >= 0.65:
            return f"You're not alone! {int(success_rate * 100)}% of {sample_size:,} similar decisions turned out well."
        else:
            return f"This is a significant decision. {int(success_rate * 100)}% of people in similar situations found success with proper preparation."

    def get_pattern_comparison(
        self,
        decision_type: str,
        user_data: Dict[str, Any]
    ) -> Dict[str, Any]:

        pattern = self.COMMUNITY_PATTERNS.get(decision_type, self.COMMUNITY_PATTERNS['job_change'])

        user_regret = user_data.get('predicted_regret', 0.5)
        community_regret = pattern['avg_predicted_regret']

        user_emotions = set(user_data.get('emotions', []))
        community_emotions = set(pattern['common_emotions'])
        emotion_overlap = len(user_emotions & community_emotions) / max(len(community_emotions), 1)

        comparison = {
            'decision_type': decision_type,
            'community_sample_size': pattern['sample_size'],
            'regret_comparison': {
                'your_predicted': user_regret,
                'community_avg_predicted': community_regret,
                'community_avg_actual': pattern['avg_actual_regret'],
                'interpretation': self._interpret_regret_comparison(user_regret, community_regret, pattern['avg_actual_regret'])
            },
            'emotion_alignment': {
                'overlap_percentage': emotion_overlap * 100,
                'common_in_community': pattern['common_emotions'],
                'your_unique': list(user_emotions - community_emotions)
            },
            'success_rate': {
                'community': pattern['success_rate'] * 100,
                'your_estimated': self._estimate_user_success(user_data, pattern)
            },
            'common_concerns': pattern['common_concerns'],
            'key_factors_to_consider': pattern['top_factors']
        }

        return comparison

    def _interpret_regret_comparison(self, user: float, community_pred: float, community_actual: float) -> str:

        gap = community_pred - community_actual

        if user < community_pred:
            return f"Your predicted regret is lower than average. Historically, actual regret tends to be {gap*100:.0f}% lower than predicted."
        elif user > community_pred + 0.1:
            return f"Your predicted regret is higher than average. Consider that actual regret is often {gap*100:.0f}% lower than predicted."
        else:
            return f"Your situation is typical. Actual regret tends to be about {gap*100:.0f}% lower than initially predicted."

    def _estimate_user_success(self, user_data: Dict, pattern: Dict) -> float:

        base_rate = pattern['success_rate']

        adjustments = 0

        risk_tolerance = user_data.get('risk_tolerance', 0.5)
        if risk_tolerance > 0.6:
            adjustments += 0.05
        elif risk_tolerance < 0.3:
            adjustments -= 0.05

        financial = user_data.get('financial_stability', 0.5)
        if financial > 0.7:
            adjustments += 0.08
        elif financial < 0.3:
            adjustments -= 0.08

        positive_emotions = ['excited', 'confident', 'hopeful', 'motivated']
        user_emotions = user_data.get('emotions', [])
        positive_count = sum(1 for e in user_emotions if e in positive_emotions)
        adjustments += min(0.05, positive_count * 0.02)

        return min(0.95, max(0.25, (base_rate + adjustments) * 100))

    def get_similar_decisions_stats(self, decision_type: str, filters: Dict = None) -> Dict[str, Any]:

        pattern = self.COMMUNITY_PATTERNS.get(decision_type, self.COMMUNITY_PATTERNS['job_change'])

        base_size = pattern['sample_size']

        if filters:
            if filters.get('industry'):
                base_size = int(base_size * 0.3)
            if filters.get('experience_range'):
                base_size = int(base_size * 0.4)

        return {
            'total_decisions': base_size,
            'regret_distribution': {
                'none': int(base_size * 0.35),
                'low': int(base_size * 0.30),
                'moderate': int(base_size * 0.25),
                'high': int(base_size * 0.10)
            },
            'average_time_to_outcome': f"{random.randint(3, 12)} months",
            'top_success_factors': pattern['top_factors'],
            'common_pitfalls': self._get_common_pitfalls(decision_type),
            'outcome_satisfaction': {
                'very_satisfied': 35,
                'satisfied': 37,
                'neutral': 15,
                'unsatisfied': 10,
                'very_unsatisfied': 3
            }
        }

    def _get_common_pitfalls(self, decision_type: str) -> List[str]:

        pitfalls = {
            'job_change': [
                'Not researching company culture thoroughly',
                'Focusing only on salary, ignoring growth',
                'Leaving too quickly during difficult times'
            ],
            'career_switch': [
                'Underestimating the learning curve',
                'Not building skills before transitioning',
                'Romanticizing the new field'
            ],
            'startup': [
                'Not verifying funding/runway',
                'Ignoring red flags about founders',
                'Underestimating equity dilution'
            ],
            'education': [
                'Choosing prestige over practical outcomes',
                'Not researching alumni career paths',
                'Underestimating time commitment'
            ],
            'freelance': [
                'Not having enough savings',
                'Underpricing services initially',
                'Poor client boundary setting'
            ]
        }
        return pitfalls.get(decision_type, pitfalls['job_change'])

    def contribute_outcome(
        self,
        user_id: str,
        decision_type: str,
        outcome_data: Dict[str, Any]
    ) -> Dict[str, Any]:

        contribution = {
            'decision_type': decision_type,
            'outcome': outcome_data.get('outcome', ''),
            'regret_level': outcome_data.get('regret_level', 'low'),
            'would_decide_same': outcome_data.get('would_decide_same', True),
            'learnings': outcome_data.get('learnings', []),
            'contributed_at': datetime.utcnow().isoformat()
        }

        anon_id = hashlib.sha256(f"{user_id}_{decision_type}_{datetime.utcnow().isoformat()}".encode()).hexdigest()[:12]
        self.user_contributions[anon_id] = contribution

        return {
            'contribution_id': anon_id,
            'status': 'accepted',
            'message': 'Thank you for contributing! Your anonymized story will help others.',
            'community_impact': f"You're helping {self.COMMUNITY_PATTERNS.get(decision_type, {}).get('sample_size', 0) + 1:,} people make better decisions."
        }

    def get_wisdom_nuggets(self, decision_type: str, count: int = 3) -> List[Dict[str, str]]:

        nuggets = {
            'job_change': [
                {'wisdom': 'The best job offers come when you are not desperate.', 'source': 'Community insight from 1,200+ decisions'},
                {'wisdom': 'Culture fit issues become apparent within the first 90 days.', 'source': 'Pattern analysis'},
                {'wisdom': 'Negotiating is expected - the worst they can say is no.', 'source': 'Survey of 5,000+ job changers'},
                {'wisdom': 'Always get the offer in writing before resigning.', 'source': 'Community wisdom'}
            ],
            'career_switch': [
                {'wisdom': 'Your previous experience is always more transferable than you think.', 'source': 'Analysis of 3,000+ career switches'},
                {'wisdom': 'The imposter syndrome is worst in months 3-6, then improves.', 'source': 'Community survey'},
                {'wisdom': 'Building skills while employed reduces risk significantly.', 'source': 'Success pattern analysis'},
                {'wisdom': 'Network in your target field before making the jump.', 'source': 'Top success factor'}
            ],
            'startup': [
                {'wisdom': 'Ask about runway in months, not vague terms like "well-funded".', 'source': 'Failure pattern analysis'},
                {'wisdom': 'Reference check the founders as thoroughly as they check you.', 'source': 'Community wisdom'},
                {'wisdom': 'Equity means nothing if you do not understand the cap table.', 'source': 'Financial analysis'},
                {'wisdom': 'The team matters more than the idea.', 'source': 'Success pattern analysis'}
            ]
        }

        type_nuggets = nuggets.get(decision_type, nuggets['job_change'])
        return random.sample(type_nuggets, min(count, len(type_nuggets)))

    def to_dict(self, social_proof: SocialProofData) -> Dict[str, Any]:

        return {
            'decision_type': social_proof.decision_type,
            'total_similar_decisions': social_proof.total_similar_decisions,
            'success_percentage': social_proof.success_percentage,
            'avg_regret': social_proof.avg_regret,
            'common_choice': social_proof.common_choice,
            'top_considerations': social_proof.top_considerations,
            'confidence_boost': social_proof.confidence_boost,
            'sample_stories': [
                {
                    'id': s.id,
                    'summary': s.summary,
                    'outcome': s.outcome,
                    'regret_level': s.regret_level,
                    'key_learnings': s.key_learnings,
                    'time_since': s.time_since_decision,
                    'would_decide_same': s.would_decide_same,
                    'upvotes': s.upvotes
                }
                for s in social_proof.sample_stories
            ]
        }
