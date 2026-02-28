import uuid
import math
import random
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum


@dataclass
class AnonymousProfile:
    profile_hash: str
    role: str
    industry: str
    experience_years: int
    experience_bracket: str
    risk_tolerance: str
    location_tier: str
    career_stage: str


@dataclass
class PeerDecisionOutcome:
    decision_type: str
    satisfaction_score: float
    regret_score: float
    salary_change_pct: float
    time_to_satisfaction_months: int
    profile: AnonymousProfile


@dataclass
class PeerBenchmark:
    metric: str
    user_value: float
    peer_avg: float
    peer_median: float
    percentile: float
    sample_size: int
    insight: str


class PeerComparisonService:
    EXPERIENCE_BRACKETS = {
        (0, 2): "0-2 years",
        (3, 5): "3-5 years",
        (5, 7): "5-7 years",
        (8, 10): "8-10 years",
        (11, 15): "11-15 years",
        (16, 99): "16+ years"
    }

    LOCATION_TIERS = {
        'san_francisco': 'tier1', 'new_york': 'tier1', 'seattle': 'tier1',
        'boston': 'tier1', 'los_angeles': 'tier1', 'london': 'tier1',
        'austin': 'tier2', 'chicago': 'tier2', 'denver': 'tier2',
        'atlanta': 'tier2', 'toronto': 'tier2', 'berlin': 'tier2',
        'remote_us': 'remote', 'remote': 'remote',
    }

    def __init__(self):
        self.anonymous_profiles: Dict[str, AnonymousProfile] = {}
        self.decision_outcomes: List[PeerDecisionOutcome] = []
        self.user_profile_map: Dict[str, str] = {}
        self._seed_sample_data()

    def _seed_sample_data(self):
        roles = ['software_engineer', 'product_manager', 'data_scientist',
                 'designer', 'manager', 'devops_engineer']
        industries = ['technology', 'finance', 'healthcare', 'ecommerce', 'saas']
        decision_types = ['job_change', 'promotion', 'career_switch',
                          'education', 'startup', 'relocation']

        for i in range(200):
            exp = random.randint(1, 25)
            bracket = self._get_experience_bracket(exp)
            risk = random.choice(['low', 'medium', 'high'])
            role = random.choice(roles)
            industry = random.choice(industries)
            stage = self._get_career_stage(exp)
            location = random.choice(list(self.LOCATION_TIERS.keys()))

            profile_hash = hashlib.sha256(f"sample_{i}".encode()).hexdigest()[:12]
            profile = AnonymousProfile(
                profile_hash=profile_hash,
                role=role,
                industry=industry,
                experience_years=exp,
                experience_bracket=bracket,
                risk_tolerance=risk,
                location_tier=self.LOCATION_TIERS.get(location, 'tier2'),
                career_stage=stage
            )
            self.anonymous_profiles[profile_hash] = profile

            for _ in range(random.randint(1, 3)):
                dtype = random.choice(decision_types)
                sat = max(0, min(100, random.gauss(65, 20)))
                regret = max(0, min(100, random.gauss(35, 20)))
                sal_change = random.gauss(15, 25)
                time_months = random.randint(1, 18)

                outcome = PeerDecisionOutcome(
                    decision_type=dtype,
                    satisfaction_score=round(sat, 1),
                    regret_score=round(regret, 1),
                    salary_change_pct=round(sal_change, 1),
                    time_to_satisfaction_months=time_months,
                    profile=profile
                )
                self.decision_outcomes.append(outcome)

    def _get_experience_bracket(self, years: int) -> str:
        for (low, high), bracket in self.EXPERIENCE_BRACKETS.items():
            if low <= years <= high:
                return bracket
        return "5-7 years"

    def _get_career_stage(self, years: int) -> str:
        if years <= 3:
            return "early"
        elif years <= 8:
            return "mid"
        elif years <= 15:
            return "senior"
        return "executive"

    def register_user_profile(self, user_id: str, role: str, industry: str,
                               experience_years: int, risk_tolerance: str = "medium",
                               location: str = "remote") -> Dict:
        profile_hash = hashlib.sha256(
            f"{user_id}:{datetime.utcnow().date()}".encode()
        ).hexdigest()[:12]

        profile = AnonymousProfile(
            profile_hash=profile_hash,
            role=role,
            industry=industry,
            experience_years=experience_years,
            experience_bracket=self._get_experience_bracket(experience_years),
            risk_tolerance=risk_tolerance,
            location_tier=self.LOCATION_TIERS.get(location, 'tier2'),
            career_stage=self._get_career_stage(experience_years)
        )
        self.anonymous_profiles[profile_hash] = profile
        self.user_profile_map[user_id] = profile_hash

        return {
            "profile_hash": profile_hash,
            "bracket": profile.experience_bracket,
            "career_stage": profile.career_stage,
            "peer_group_size": self._count_peers(profile)
        }

    def _count_peers(self, profile: AnonymousProfile) -> int:
        return sum(1 for p in self.anonymous_profiles.values()
                   if p.role == profile.role
                   and p.experience_bracket == profile.experience_bracket
                   and p.profile_hash != profile.profile_hash)

    def get_peer_comparison(self, user_id: str, decision_type: str = None) -> Dict:
        profile_hash = self.user_profile_map.get(user_id)
        if not profile_hash:
            return {"error": "User profile not registered. Call register first."}

        profile = self.anonymous_profiles[profile_hash]
        peers = self._find_similar_peers(profile)
        peer_outcomes = self._get_peer_outcomes(peers, decision_type)

        if not peer_outcomes:
            return {
                "peer_group_size": 0,
                "message": "Not enough peer data yet. More data will be available as the community grows.",
                "profile": self._profile_to_dict(profile)
            }

        satisfactions = [o.satisfaction_score for o in peer_outcomes]
        regrets = [o.regret_score for o in peer_outcomes]
        salary_changes = [o.salary_change_pct for o in peer_outcomes]
        type_stats = defaultdict(lambda: {"count": 0, "avg_satisfaction": 0, "avg_regret": 0})
        for o in peer_outcomes:
            ts = type_stats[o.decision_type]
            ts["count"] += 1
            ts["avg_satisfaction"] += o.satisfaction_score
            ts["avg_regret"] += o.regret_score
        for dtype, stats in type_stats.items():
            if stats["count"] > 0:
                stats["avg_satisfaction"] = round(stats["avg_satisfaction"] / stats["count"], 1)
                stats["avg_regret"] = round(stats["avg_regret"] / stats["count"], 1)

        decision_counts = defaultdict(int)
        for o in peer_outcomes:
            decision_counts[o.decision_type] += 1
        most_common = sorted(decision_counts.items(), key=lambda x: x[1], reverse=True)

        return {
            "profile": self._profile_to_dict(profile),
            "peer_group_size": len(peers),
            "total_decisions": len(peer_outcomes),
            "decision_filter": decision_type,
            "satisfaction": {
                "mean": round(sum(satisfactions) / len(satisfactions), 1),
                "median": round(sorted(satisfactions)[len(satisfactions) // 2], 1),
                "pct_satisfied": round(sum(1 for s in satisfactions if s >= 60) / len(satisfactions) * 100, 1),
                "pct_very_satisfied": round(sum(1 for s in satisfactions if s >= 80) / len(satisfactions) * 100, 1)
            },
            "regret": {
                "mean": round(sum(regrets) / len(regrets), 1),
                "pct_low_regret": round(sum(1 for r in regrets if r < 30) / len(regrets) * 100, 1),
                "pct_high_regret": round(sum(1 for r in regrets if r >= 60) / len(regrets) * 100, 1)
            },
            "salary_change": {
                "mean_pct": round(sum(salary_changes) / len(salary_changes), 1),
                "pct_increase": round(sum(1 for s in salary_changes if s > 0) / len(salary_changes) * 100, 1),
                "median_change_pct": round(sorted(salary_changes)[len(salary_changes) // 2], 1)
            },
            "most_common_decisions": [
                {"type": dtype, "count": count, "pct": round(count / len(peer_outcomes) * 100, 1)}
                for dtype, count in most_common[:5]
            ],
            "decision_type_stats": dict(type_stats),
            "insights": self._generate_peer_insights(profile, peer_outcomes, most_common)
        }

    def _find_similar_peers(self, profile: AnonymousProfile) -> List[AnonymousProfile]:
        peers = []
        for p in self.anonymous_profiles.values():
            if p.profile_hash == profile.profile_hash:
                continue
            score = 0
            if p.role == profile.role:
                score += 3
            if p.experience_bracket == profile.experience_bracket:
                score += 2
            if p.industry == profile.industry:
                score += 2
            if p.career_stage == profile.career_stage:
                score += 1
            if p.risk_tolerance == profile.risk_tolerance:
                score += 1

            if score >= 4:
                peers.append(p)
        return peers

    def _get_peer_outcomes(self, peers: List[AnonymousProfile],
                           decision_type: str = None) -> List[PeerDecisionOutcome]:
        peer_hashes = {p.profile_hash for p in peers}
        outcomes = [o for o in self.decision_outcomes
                    if o.profile.profile_hash in peer_hashes]
        if decision_type:
            outcomes = [o for o in outcomes if o.decision_type == decision_type]
        return outcomes

    def _generate_peer_insights(self, profile: AnonymousProfile,
                                 outcomes: List[PeerDecisionOutcome],
                                 most_common: List[Tuple]) -> List[str]:
        insights = []
        role_display = profile.role.replace('_', ' ').title()
        bracket = profile.experience_bracket

        if most_common:
            top_type = most_common[0][0].replace('_', ' ')
            top_pct = round(most_common[0][1] / len(outcomes) * 100)
            insights.append(
                f"{top_pct}% of {role_display}s with {bracket} experience recently made a '{top_type}' decision."
            )

        satisfied = sum(1 for o in outcomes if o.satisfaction_score >= 60)
        sat_pct = round(satisfied / max(len(outcomes), 1) * 100)
        insights.append(
            f"{sat_pct}% of your peers report satisfaction with their career decisions."
        )

        salary_increases = [o for o in outcomes if o.salary_change_pct > 0]
        if salary_increases:
            avg_increase = sum(o.salary_change_pct for o in salary_increases) / len(salary_increases)
            insights.append(
                f"Among peers who changed roles, the average salary increase was {avg_increase:.1f}%."
            )

        low_regret = [o for o in outcomes if o.regret_score < 30]
        if low_regret:
            avg_time = sum(o.time_to_satisfaction_months for o in low_regret) / len(low_regret)
            insights.append(
                f"Those with low regret typically reached satisfaction within {avg_time:.0f} months."
            )

        return insights

    def contribute_outcome(self, user_id: str, decision_type: str,
                           satisfaction: float, regret: float,
                           salary_change_pct: float,
                           time_to_satisfaction_months: int = 6) -> Dict:
        profile_hash = self.user_profile_map.get(user_id)
        if not profile_hash or profile_hash not in self.anonymous_profiles:
            return {"error": "User profile not registered"}

        profile = self.anonymous_profiles[profile_hash]
        outcome = PeerDecisionOutcome(
            decision_type=decision_type,
            satisfaction_score=max(0, min(100, satisfaction)),
            regret_score=max(0, min(100, regret)),
            salary_change_pct=round(salary_change_pct, 1),
            time_to_satisfaction_months=max(0, time_to_satisfaction_months),
            profile=profile
        )
        self.decision_outcomes.append(outcome)
        return {"status": "contributed", "total_outcomes": len(self.decision_outcomes)}

    def get_decision_distribution(self, user_id: str, decision_type: str) -> Dict:
        profile_hash = self.user_profile_map.get(user_id)
        if not profile_hash:
            return {"error": "User profile not registered"}

        profile = self.anonymous_profiles[profile_hash]
        peers = self._find_similar_peers(profile)
        outcomes = self._get_peer_outcomes(peers, decision_type)

        if not outcomes:
            return {
                "decision_type": decision_type,
                "sample_size": 0,
                "message": "Not enough data for this decision type"
            }

        satisfactions = sorted([o.satisfaction_score for o in outcomes])
        regrets = sorted([o.regret_score for o in outcomes])

        return {
            "decision_type": decision_type,
            "sample_size": len(outcomes),
            "satisfaction_distribution": {
                "0-20": sum(1 for s in satisfactions if s < 20),
                "20-40": sum(1 for s in satisfactions if 20 <= s < 40),
                "40-60": sum(1 for s in satisfactions if 40 <= s < 60),
                "60-80": sum(1 for s in satisfactions if 60 <= s < 80),
                "80-100": sum(1 for s in satisfactions if s >= 80)
            },
            "regret_distribution": {
                "0-20": sum(1 for r in regrets if r < 20),
                "20-40": sum(1 for r in regrets if 20 <= r < 40),
                "40-60": sum(1 for r in regrets if 40 <= r < 60),
                "60-80": sum(1 for r in regrets if 60 <= r < 80),
                "80-100": sum(1 for r in regrets if r >= 80)
            },
            "avg_time_to_satisfaction_months": round(
                sum(o.time_to_satisfaction_months for o in outcomes) / len(outcomes), 1
            )
        }

    def _profile_to_dict(self, profile: AnonymousProfile) -> Dict:
        return {
            "role": profile.role,
            "industry": profile.industry,
            "experience_bracket": profile.experience_bracket,
            "career_stage": profile.career_stage,
            "risk_tolerance": profile.risk_tolerance,
            "location_tier": profile.location_tier
        }


peer_comparison_service = PeerComparisonService()
