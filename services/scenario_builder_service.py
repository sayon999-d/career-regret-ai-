import uuid
import re
import random
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict


@dataclass
class ScenarioParameter:
    name: str
    value: Any
    unit: str = ""
    confidence: float = 0.8


@dataclass
class Scenario:
    id: str
    user_id: str
    description: str
    parameters: Dict[str, ScenarioParameter]
    parent_id: Optional[str] = None 
    simulation_results: Optional[Dict] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ScenarioComparison:
    scenario_a: Scenario
    scenario_b: Scenario
    comparison: Dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""


class ScenarioBuilderService:
    SALARY_PATTERNS = [
        r'(\$[\d,]+(?:k)?)', r'(\d+)k?\s*(?:salary|pay|compensation|per\s+year|annually)',
        r'(?:earning|making|paid)\s*(\$?[\d,]+k?)', r'(\d+)%?\s*(?:more|raise|increase|equity)',
    ]
    TIME_PATTERNS = [
        r'(\d+)\s*(?:year|yr)s?', r'(\d+)\s*(?:month)s?',
        r'(\d+)\s*(?:week)s?',
    ]
    ROLE_KEYWORDS = {
        'software_engineer': ['software', 'developer', 'engineer', 'coding', 'programming', 'swe'],
        'product_manager': ['product manager', 'pm', 'product lead'],
        'data_scientist': ['data scientist', 'data science', 'ml engineer', 'machine learning'],
        'designer': ['designer', 'ux', 'ui', 'design'],
        'manager': ['manager', 'management', 'director', 'lead', 'vp'],
        'startup_founder': ['startup', 'founder', 'entrepreneur', 'my own company'],
        'freelance': ['freelance', 'consultant', 'contracting', 'independent'],
    }
    COMPANY_SCALE = {
        'startup': {'growth_rate': 0.25, 'volatility': 0.35, 'risk': 'high'},
        'midsize': {'growth_rate': 0.12, 'volatility': 0.15, 'risk': 'medium'},
        'enterprise': {'growth_rate': 0.08, 'volatility': 0.08, 'risk': 'low'},
        'faang': {'growth_rate': 0.15, 'volatility': 0.12, 'risk': 'medium'},
    }
    COMPANY_KEYWORDS = {
        'faang': ['google', 'meta', 'amazon', 'apple', 'netflix', 'microsoft', 'faang', 'big tech'],
        'startup': ['startup', 'early-stage', 'seed', 'series a', 'small company'],
        'midsize': ['mid-size', 'midsize', 'medium company', 'growing company'],
        'enterprise': ['enterprise', 'corporate', 'large company', 'fortune 500', 'big company'],
    }

    def __init__(self):
        self.scenarios: Dict[str, Scenario] = {}
        self.user_scenarios: Dict[str, List[str]] = defaultdict(list)

    def parse_scenario(self, description: str, user_id: str,
                       current_context: Dict = None) -> Dict:
        text = description.lower().strip()
        params = {}

        salary = self._extract_salary(text, current_context)
        if salary:
            params['base_salary'] = ScenarioParameter('base_salary', salary, 'USD/year')

        years = self._extract_time_horizon(text)
        params['time_horizon'] = ScenarioParameter('time_horizon', years, 'years')

        role = self._extract_role(text)
        params['role'] = ScenarioParameter('role', role)

        company = self._extract_company_type(text)
        params['company_type'] = ScenarioParameter('company_type', company)

        modifiers = self._extract_modifiers(text)
        for key, value in modifiers.items():
            params[key] = ScenarioParameter(key, value)

        scenario_type = self._classify_scenario(text)
        params['scenario_type'] = ScenarioParameter('scenario_type', scenario_type)

        scenario = Scenario(
            id=str(uuid.uuid4()),
            user_id=user_id,
            description=description,
            parameters=params
        )

        self.scenarios[scenario.id] = scenario
        self.user_scenarios[user_id].append(scenario.id)

        sim_result = self._run_simulation(scenario)
        scenario.simulation_results = sim_result

        return {
            "scenario_id": scenario.id,
            "parsed_parameters": {k: {"value": v.value, "unit": v.unit}
                                  for k, v in params.items()},
            "scenario_type": scenario_type,
            "simulation": sim_result
        }

    def _extract_salary(self, text: str, context: Dict = None) -> Optional[float]:
        for pattern in self.SALARY_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                val = match.group(1).replace('$', '').replace(',', '')
                if 'k' in val.lower():
                    return float(val.lower().replace('k', '')) * 1000
                num = float(val)
                if num < 1000: 
                    base = context.get('current_salary', 100000) if context else 100000
                    return base * (1 + num / 100)
                return num

        if context and 'current_salary' in context:
            return context['current_salary']
        return 100000 

    def _extract_time_horizon(self, text: str) -> int:
        match = re.search(r'(\d+)\s*(?:year|yr)s?', text)
        if match:
            return min(int(match.group(1)), 30)
        match = re.search(r'(\d+)\s*months?', text)
        if match:
            return max(1, int(match.group(1)) // 12)
        return 5 

    def _extract_role(self, text: str) -> str:
        for role, keywords in self.ROLE_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    return role
        return 'general'

    def _extract_company_type(self, text: str) -> str:
        for ctype, keywords in self.COMPANY_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    return ctype
        return 'midsize'

    def _extract_modifiers(self, text: str) -> Dict[str, Any]:
        modifiers = {}

        match = re.search(r'(\d+)%?\s*(?:more|raise|increase|higher|bump)', text)
        if match:
            modifiers['salary_increase_pct'] = float(match.group(1))

        match = re.search(r'(\d+)%?\s*(?:equity|stock|options|shares|rsu)', text)
        if match:
            modifiers['equity_pct'] = float(match.group(1))

        
        if any(w in text for w in ['remote', 'work from home', 'wfh']):
            modifiers['remote'] = True

        if any(w in text for w in ['move to', 'relocate', 'relocation']):
            modifiers['relocation'] = True

        if any(w in text for w in ['manage', 'management', 'lead a team', 'people management']):
            modifiers['management'] = True

        if any(w in text for w in ['stay', 'remain', 'keep', 'current', 'same']):
            modifiers['status_quo'] = True

        return modifiers

    def _classify_scenario(self, text: str) -> str:
        if any(w in text for w in ['offer', 'new job', 'switch', 'leave']):
            return 'job_change'
        elif any(w in text for w in ['startup', 'found', 'entrepreneur']):
            return 'startup'
        elif any(w in text for w in ['freelance', 'consultant', 'independent']):
            return 'freelance'
        elif any(w in text for w in ['stay', 'remain', 'current']):
            return 'status_quo'
        elif any(w in text for w in ['promote', 'promotion', 'advance']):
            return 'promotion'
        elif any(w in text for w in ['learn', 'study', 'degree', 'course', 'education']):
            return 'education'
        elif any(w in text for w in ['negotiate', 'raise', 'counter']):
            return 'negotiation'
        elif any(w in text for w in ['relocate', 'move']):
            return 'relocation'
        return 'career_decision'

    def _run_simulation(self, scenario: Scenario) -> Dict:
        params = scenario.parameters
        base_salary = params.get('base_salary', ScenarioParameter('', 100000)).value
        years = params.get('time_horizon', ScenarioParameter('', 5)).value
        company = params.get('company_type', ScenarioParameter('', 'midsize')).value
        scenario_type = params.get('scenario_type', ScenarioParameter('', 'career_decision')).value

        company_config = self.COMPANY_SCALE.get(company, self.COMPANY_SCALE['midsize'])
        growth_rate = company_config['growth_rate']
        volatility = company_config['volatility']

        if 'salary_increase_pct' in params:
            pct = params['salary_increase_pct'].value
            base_salary *= (1 + pct / 100)
        if params.get('status_quo', ScenarioParameter('', False)).value:
            growth_rate *= 0.5
            volatility *= 0.3

        if scenario_type == 'startup':
            growth_rate = 0.35
            volatility = 0.45

        iterations = 500
        trajectories = []
        for _ in range(iterations):
            trajectory = [base_salary]
            current = base_salary
            for year in range(years):
                annual_change = np.random.normal(growth_rate, volatility)
                current *= (1 + annual_change)
                current = max(current * 0.3, current) 
                trajectory.append(round(current, 2))
            trajectories.append(trajectory)

        trajectories_arr = np.array(trajectories)
        final_values = trajectories_arr[:, -1]

        base_satisfaction = 65
        if scenario_type == 'startup':
            base_satisfaction = 55
        elif scenario_type == 'status_quo':
            base_satisfaction = 50
        elif scenario_type == 'promotion':
            base_satisfaction = 70

        satisfaction_sims = np.random.normal(base_satisfaction, 15, iterations)
        satisfaction_sims = np.clip(satisfaction_sims, 0, 100)

        if scenario_type == 'status_quo':
            regret_base = 45
        elif scenario_type == 'startup':
            regret_base = 35
        else:
            regret_base = 25
        regret_sims = np.random.normal(regret_base, 20, iterations)
        regret_sims = np.clip(regret_sims, 0, 100)

        return {
            "scenario_type": scenario_type,
            "time_horizon_years": years,
            "base_salary": round(base_salary, 2),
            "company_type": company,
            "salary_projections": {
                "year_by_year_median": [round(float(np.median(trajectories_arr[:, i])), 2)
                                        for i in range(years + 1)],
                "percentile_10": round(float(np.percentile(final_values, 10)), 2),
                "percentile_25": round(float(np.percentile(final_values, 25)), 2),
                "median": round(float(np.median(final_values)), 2),
                "percentile_75": round(float(np.percentile(final_values, 75)), 2),
                "percentile_90": round(float(np.percentile(final_values, 90)), 2),
                "max_upside": round(float(np.max(final_values)), 2),
                "min_downside": round(float(np.min(final_values)), 2)
            },
            "satisfaction": {
                "mean": round(float(np.mean(satisfaction_sims)), 1),
                "std": round(float(np.std(satisfaction_sims)), 1),
                "p_high_satisfaction": round(float(np.mean(satisfaction_sims > 70)) * 100, 1)
            },
            "regret": {
                "mean": round(float(np.mean(regret_sims)), 1),
                "std": round(float(np.std(regret_sims)), 1),
                "p_low_regret": round(float(np.mean(regret_sims < 30)) * 100, 1),
                "p_high_regret": round(float(np.mean(regret_sims > 60)) * 100, 1)
            },
            "risk_level": company_config['risk'],
            "iterations": iterations
        }

    def chain_scenario(self, parent_id: str, description: str,
                       user_id: str) -> Dict:
        parent = self.scenarios.get(parent_id)
        if not parent:
            return {"error": "Parent scenario not found"}

        parent_result = parent.simulation_results or {}
        context = {
            'current_salary': parent_result.get('salary_projections', {}).get('median', 100000),
        }

        result = self.parse_scenario(description, user_id, current_context=context)

        new_scenario = self.scenarios[result['scenario_id']]
        new_scenario.parent_id = parent_id

        result['parent_scenario_id'] = parent_id
        result['chain_depth'] = self._get_chain_depth(result['scenario_id'])
        return result

    def _get_chain_depth(self, scenario_id: str) -> int:
        depth = 0
        current = self.scenarios.get(scenario_id)
        while current and current.parent_id:
            depth += 1
            current = self.scenarios.get(current.parent_id)
            if depth > 10:
                break
        return depth

    def compare_scenarios(self, scenario_id_a: str, scenario_id_b: str) -> Dict:
        a = self.scenarios.get(scenario_id_a)
        b = self.scenarios.get(scenario_id_b)
        if not a or not b:
            return {"error": "One or both scenarios not found"}

        sim_a = a.simulation_results or {}
        sim_b = b.simulation_results or {}

        salary_a = sim_a.get('salary_projections', {}).get('median', 0)
        salary_b = sim_b.get('salary_projections', {}).get('median', 0)

        sat_a = sim_a.get('satisfaction', {}).get('mean', 50)
        sat_b = sim_b.get('satisfaction', {}).get('mean', 50)

        regret_a = sim_a.get('regret', {}).get('mean', 50)
        regret_b = sim_b.get('regret', {}).get('mean', 50)

        score_a = (salary_a / max(salary_a, salary_b, 1)) * 40 + (sat_a / 100) * 35 + ((100 - regret_a) / 100) * 25
        score_b = (salary_b / max(salary_a, salary_b, 1)) * 40 + (sat_b / 100) * 35 + ((100 - regret_b) / 100) * 25

        winner = "A" if score_a >= score_b else "B"
        winning_scenario = a if winner == "A" else b

        return {
            "scenario_a": {
                "id": a.id,
                "description": a.description,
                "salary_median": salary_a,
                "satisfaction": sat_a,
                "regret": regret_a,
                "composite_score": round(score_a, 2)
            },
            "scenario_b": {
                "id": b.id,
                "description": b.description,
                "salary_median": salary_b,
                "satisfaction": sat_b,
                "regret": regret_b,
                "composite_score": round(score_b, 2)
            },
            "winner": winner,
            "winner_description": winning_scenario.description,
            "salary_difference": round(abs(salary_a - salary_b), 2),
            "satisfaction_difference": round(abs(sat_a - sat_b), 1),
            "recommendation": self._generate_recommendation(a, b, winner, sim_a, sim_b)
        }

    def _generate_recommendation(self, a, b, winner, sim_a, sim_b) -> str:
        winning = a if winner == "A" else b
        losing = b if winner == "A" else a
        w_sim = sim_a if winner == "A" else sim_b
        l_sim = sim_b if winner == "A" else sim_a

        risk_w = w_sim.get('risk_level', 'medium')
        risk_l = l_sim.get('risk_level', 'medium')

        parts = [f"'{winning.description}' scores higher overall."]

        sal_w = w_sim.get('salary_projections', {}).get('median', 0)
        sal_l = l_sim.get('salary_projections', {}).get('median', 0)
        if sal_w > sal_l:
            parts.append(f"It offers ~${abs(sal_w - sal_l):,.0f} higher median salary over the projection period.")
        elif sal_l > sal_w:
            parts.append(f"However, '{losing.description}' shows ${abs(sal_w - sal_l):,.0f} more salary potential.")

        if risk_w == 'high':
            parts.append("Note: this path carries higher risk and volatility.")
        elif risk_w == 'low':
            parts.append("This is a stable, lower-risk path.")

        return " ".join(parts)

    def get_user_scenarios(self, user_id: str) -> List[Dict]:
        scenario_ids = self.user_scenarios.get(user_id, [])
        results = []
        for sid in scenario_ids:
            s = self.scenarios.get(sid)
            if s:
                results.append({
                    "id": s.id,
                    "description": s.description,
                    "scenario_type": s.parameters.get('scenario_type',
                                                       ScenarioParameter('', 'unknown')).value,
                    "parent_id": s.parent_id,
                    "created_at": s.created_at.isoformat(),
                    "has_simulation": s.simulation_results is not None
                })
        return results


scenario_builder_service = ScenarioBuilderService()
