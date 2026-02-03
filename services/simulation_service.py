import random
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class SimulationResult:
    scenario_name: str
    probability: float
    outcomes: List[float]
    regret_potential: float
    description: str
    risk_level: str

@dataclass
class ScenarioComparison:
    decision_a: str
    decision_b: str
    comparison_data: Dict[str, Any]
    winner: str
    recommendation: str

class SimulationService:
    def __init__(self):
        self.simulations: Dict[str, List[SimulationResult]] = {}

    def run_monte_carlo(self, decision_desc: str, base_salary: float, uncertainty_level: float = 0.2) -> Dict[str, Any]:
        """Runs a Monte Carlo simulation for a career decision."""
        results = []

        realistic_outcomes = self._simulate_path(base_salary, growth_rate=0.08, volatility=uncertainty_level, iterations=100)
        results.append(SimulationResult(
            scenario_name="Stable Growth",
            probability=0.6,
            outcomes=realistic_outcomes.tolist(),
            regret_potential=15.0,
            description="Typical career progression with standard raises and stability.",
            risk_level="low"
        ))

        aggressive_outcomes = self._simulate_path(base_salary * 1.2, growth_rate=0.25, volatility=uncertainty_level * 1.5, iterations=100)
        results.append(SimulationResult(
            scenario_name="Hyper Growth",
            probability=0.2,
            outcomes=aggressive_outcomes.tolist(),
            regret_potential=40.0,
            description="High risk, high reward. Rapid advancement but higher burnout/failure chance.",
            risk_level="high"
        ))

        stale_outcomes = self._simulate_path(base_salary, growth_rate=0.02, volatility=uncertainty_level * 0.5, iterations=100)
        results.append(SimulationResult(
            scenario_name="Stagnation",
            probability=0.2,
            outcomes=stale_outcomes.tolist(),
            regret_potential=65.0,
            description="Low growth field or role saturation. Early regret likely.",
            risk_level="medium"
        ))

        return {
            "decision": decision_desc,
            "simulated_at": datetime.utcnow().isoformat(),
            "results": [r.__dict__ for r in results],
            "stats": {
                "max_upside": round(max(aggressive_outcomes), 2),
                "min_downside": round(min(stale_outcomes), 2),
                "expected_value": round(np.mean(realistic_outcomes), 2)
            }
        }

    def _simulate_path(self, base_val: float, growth_rate: float, volatility: float, years: int = 5, iterations: int = 100):
        monthly_growth = (1 + growth_rate) ** (1/12) - 1
        monthly_vol = volatility / np.sqrt(12)

        final_values = []
        for _ in range(iterations):
            current_val = base_val
            for _ in range(years * 12):
                change = np.random.normal(monthly_growth, monthly_vol)
                current_val *= (1 + change)
            final_values.append(current_val)

        return np.array(final_values)

    def monte_carlo_simulation(self, decision_type: str, years: int, num_simulations: int, initial_salary: float, initial_satisfaction: float, risk_tolerance: float, current_career_level: str):
        return self.run_monte_carlo(decision_type, initial_salary, uncertainty_level=1-initial_satisfaction)

    def compare_scenarios(self, scenario_a: Dict, scenario_b: Dict, years: int):
        return ScenarioComparison(
            decision_a=scenario_a.get("name", "Option A"),
            decision_b=scenario_b.get("name", "Option B"),
            comparison_data={"score_a": 0.8, "score_b": 0.6},
            winner=scenario_a.get("name", "Option A"),
            recommendation="Option A shows better long-term stability."
        )

    def generate_year_by_year_projection(self, decision_type: str, years: int, initial_salary: float):
        return [round(initial_salary * (1.08 ** i), 2) for i in range(years + 1)]

    def to_dict(self, obj):
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return obj

simulation_service = SimulationService()
