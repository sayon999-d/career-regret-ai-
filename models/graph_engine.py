import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
import random

@dataclass
class EnhancedGraphNode:
    id: str
    node_type: str
    label: str
    weight: float = 1.0
    attributes: Dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    visit_count: int = 0
    success_rate: float = 0.5

@dataclass
class EnhancedGraphEdge:
    source: str
    target: str
    edge_type: str
    weight: float = 1.0
    confidence: float = 0.5
    attributes: Dict = field(default_factory=dict)
    sample_count: int = 1
    last_updated: datetime = field(default_factory=datetime.utcnow)

class MarketConditions:
    def __init__(self):
        self.industry_health = {
            'technology': 0.85, 'healthcare': 0.80, 'finance': 0.70, 'education': 0.65,
            'manufacturing': 0.55, 'retail': 0.50, 'consulting': 0.75, 'media': 0.60,
            'government': 0.70, 'nonprofit': 0.55, 'other': 0.60
        }
        self.job_market_index = 0.7
        self.remote_work_factor = 0.8
        self.skill_demand = {
            'ai_ml': 0.95, 'cloud': 0.90, 'data_science': 0.88, 'cybersecurity': 0.92,
            'software_dev': 0.85, 'marketing': 0.65, 'sales': 0.60, 'management': 0.70,
            'design': 0.75, 'operations': 0.55
        }
        self.last_update = datetime.utcnow()

    def update_conditions(self, changes: Dict[str, float] = None):
        if changes:
            for key, value in changes.items():
                if key in self.industry_health:
                    self.industry_health[key] = np.clip(value, 0, 1)
                elif key in self.skill_demand:
                    self.skill_demand[key] = np.clip(value, 0, 1)
                elif key == 'job_market_index':
                    self.job_market_index = np.clip(value, 0, 1)
        fluctuation = np.random.normal(0, 0.02)
        self.job_market_index = np.clip(self.job_market_index + fluctuation, 0.3, 1.0)
        self.last_update = datetime.utcnow()

    def get_industry_modifier(self, industry: str) -> float:
        base = self.industry_health.get(industry, 0.6)
        return base * self.job_market_index

    def get_skill_modifier(self, skill: str) -> float:
        return self.skill_demand.get(skill, 0.5)

class MonteCarloSimulator:
    def __init__(self, graph: nx.DiGraph, num_simulations: int = 1000):
        self.graph = graph
        self.num_simulations = num_simulations
        self.random_state = np.random.RandomState(42)

    def simulate_paths(self, start_node: str, max_steps: int = 10,
                       user_factors: Dict[str, float] = None) -> Dict:
        if start_node not in self.graph:
            return {"error": "Start node not found"}
        user_factors = user_factors or {}
        outcome_counts = defaultdict(int)
        path_counts = defaultdict(int)
        step_distributions = []
        regret_scores = []

        for _ in range(self.num_simulations):
            path, outcome, steps, regret = self._simulate_single_path(start_node, max_steps, user_factors)
            if outcome:
                outcome_counts[outcome] += 1
            path_key = " -> ".join(path[:5])
            path_counts[path_key] += 1
            step_distributions.append(steps)
            regret_scores.append(regret)

        total = self.num_simulations
        outcome_probs = {k: v / total for k, v in outcome_counts.items()}
        top_paths = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "outcome_probabilities": outcome_probs,
            "top_paths": [(p, c / total) for p, c in top_paths],
            "avg_steps_to_outcome": np.mean(step_distributions),
            "step_std": np.std(step_distributions),
            "avg_regret_score": np.mean(regret_scores),
            "regret_std": np.std(regret_scores),
            "regret_percentile_25": np.percentile(regret_scores, 25),
            "regret_percentile_75": np.percentile(regret_scores, 75),
            "confidence_interval": (np.percentile(regret_scores, 5), np.percentile(regret_scores, 95)),
            "num_simulations": self.num_simulations
        }

    def _simulate_single_path(self, start: str, max_steps: int,
                              user_factors: Dict[str, float]) -> Tuple[List[str], str, int, float]:
        path = [start]
        current = start
        cumulative_regret = 0.0

        for step in range(max_steps):
            out_edges = list(self.graph.out_edges(current, data=True))
            if not out_edges:
                break
            probs = []
            targets = []
            for _, target, data in out_edges:
                base_prob = data.get('weight', 0.5)
                modifier = 1.0
                if 'risk_tolerance' in user_factors:
                    if 'risky' in target.lower() or 'startup' in target.lower():
                        modifier *= (1 + user_factors['risk_tolerance'] - 0.5)
                if 'financial_stability' in user_factors:
                    if 'financial' in target.lower() and 'success' in target.lower():
                        modifier *= (0.5 + user_factors['financial_stability'])
                probs.append(base_prob * modifier)
                targets.append(target)

            total_prob = sum(probs)
            if total_prob == 0:
                break
            probs = [p / total_prob for p in probs]
            next_node = self.random_state.choice(targets, p=probs)
            path.append(next_node)
            edge_data = self.graph.get_edge_data(current, next_node)
            step_regret = 1.0 - edge_data.get('weight', 0.5) if edge_data else 0.5
            cumulative_regret += step_regret * (0.9 ** step)
            current = next_node
            node_data = self.graph.nodes.get(current, {})
            if node_data.get('node_type') == 'outcome':
                break

        final_outcome = path[-1] if path else None
        node_data = self.graph.nodes.get(final_outcome, {})
        if node_data.get('node_type') != 'outcome':
            final_outcome = None
        avg_regret = cumulative_regret / len(path) if path else 0.5
        return path, final_outcome, len(path), avg_regret

    def sensitivity_analysis(self, start_node: str, base_factors: Dict[str, float],
                             factor_to_vary: str, variation_range: Tuple[float, float] = (0.0, 1.0),
                             num_points: int = 10) -> Dict:
        results = []
        for value in np.linspace(variation_range[0], variation_range[1], num_points):
            test_factors = base_factors.copy()
            test_factors[factor_to_vary] = value
            sim_result = self.simulate_paths(start_node, num_simulations=100, user_factors=test_factors)
            results.append({
                'factor_value': value,
                'avg_regret': sim_result.get('avg_regret_score', 0.5),
                'outcome_probs': sim_result.get('outcome_probabilities', {})
            })
        return {
            'factor': factor_to_vary,
            'sensitivity_curve': results,
            'optimal_value': min(results, key=lambda x: x['avg_regret'])['factor_value']
        }

class AdvancedDecisionGraph:
    CAREER_MILESTONES = {
        "entry_level": {"type": "milestone", "base_weight": 1.0, "avg_years": 2},
        "junior": {"type": "milestone", "base_weight": 1.1, "avg_years": 2},
        "mid_level": {"type": "milestone", "base_weight": 1.3, "avg_years": 4},
        "senior": {"type": "milestone", "base_weight": 1.6, "avg_years": 5},
        "lead": {"type": "milestone", "base_weight": 1.8, "avg_years": 3},
        "manager": {"type": "milestone", "base_weight": 2.0, "avg_years": 4},
        "director": {"type": "milestone", "base_weight": 2.3, "avg_years": 5},
        "vp": {"type": "milestone", "base_weight": 2.6, "avg_years": 5},
        "c_suite": {"type": "milestone", "base_weight": 3.0, "avg_years": 0},
        "entrepreneur": {"type": "milestone", "base_weight": 2.2, "avg_years": 0},
        "consultant": {"type": "milestone", "base_weight": 1.7, "avg_years": 0},
        "freelance": {"type": "milestone", "base_weight": 1.4, "avg_years": 0},
    }

    OUTCOMES = {
        "high_satisfaction": {"valence": 1.0, "category": "positive"},
        "moderate_satisfaction": {"valence": 0.6, "category": "neutral"},
        "low_satisfaction": {"valence": 0.2, "category": "negative"},
        "financial_success": {"valence": 0.9, "category": "positive"},
        "financial_stability": {"valence": 0.7, "category": "neutral"},
        "financial_struggle": {"valence": 0.2, "category": "negative"},
        "work_life_balance": {"valence": 0.85, "category": "positive"},
        "overwork": {"valence": 0.3, "category": "negative"},
        "burnout": {"valence": 0.1, "category": "negative"},
        "skill_mastery": {"valence": 0.9, "category": "positive"},
        "skill_growth": {"valence": 0.75, "category": "positive"},
        "skill_stagnation": {"valence": 0.25, "category": "negative"},
        "network_expansion": {"valence": 0.8, "category": "positive"},
        "isolation": {"valence": 0.2, "category": "negative"},
        "recognition": {"valence": 0.85, "category": "positive"},
        "invisibility": {"valence": 0.3, "category": "negative"},
    }

    def __init__(self, decay_factor: float = 0.95, temporal_decay: float = 0.99,
                 confidence_threshold: float = 0.3):
        self.graph = nx.DiGraph()
        self.decay_factor = decay_factor
        self.temporal_decay = temporal_decay
        self.confidence_threshold = confidence_threshold
        self.node_registry: Dict[str, EnhancedGraphNode] = {}
        self.edge_registry: Dict[Tuple[str, str], EnhancedGraphEdge] = {}
        self.market = MarketConditions()
        self.mc_simulator = None
        self.user_subgraphs: Dict[str, nx.DiGraph] = {}
        self.total_paths_analyzed = 0
        self.outcome_frequencies: Dict[str, int] = defaultdict(int)
        self.decision_history: List[Dict] = []
        self.node_messages: Dict[str, np.ndarray] = {}
        self._initialize_comprehensive_graph()
        self.mc_simulator = MonteCarloSimulator(self.graph)

    def _initialize_comprehensive_graph(self):
        for node_id, attrs in self.CAREER_MILESTONES.items():
            self.add_node(node_id=node_id, node_type=attrs["type"],
                         label=node_id.replace("_", " ").title(), weight=attrs["base_weight"],
                         attributes={"avg_years": attrs.get("avg_years", 3)})

        for outcome_id, attrs in self.OUTCOMES.items():
            self.add_node(node_id=outcome_id, node_type="outcome",
                         label=outcome_id.replace("_", " ").title(), weight=1.0,
                         attributes={"valence": attrs["valence"], "category": attrs["category"]})

        progressions = [
            ("entry_level", "junior", "leads_to", 0.85), ("junior", "mid_level", "leads_to", 0.75),
            ("mid_level", "senior", "leads_to", 0.65), ("senior", "lead", "leads_to", 0.50),
            ("lead", "manager", "leads_to", 0.45), ("manager", "director", "leads_to", 0.35),
            ("director", "vp", "leads_to", 0.25), ("vp", "c_suite", "leads_to", 0.15),
            ("mid_level", "entrepreneur", "leads_to", 0.15), ("senior", "entrepreneur", "leads_to", 0.20),
            ("manager", "entrepreneur", "leads_to", 0.18), ("senior", "consultant", "leads_to", 0.25),
            ("manager", "consultant", "leads_to", 0.22), ("mid_level", "freelance", "leads_to", 0.20),
            ("senior", "freelance", "leads_to", 0.18), ("consultant", "manager", "leads_to", 0.30),
            ("entrepreneur", "consultant", "leads_to", 0.20), ("freelance", "mid_level", "leads_to", 0.25),
        ]
        for source, target, edge_type, weight in progressions:
            self.add_edge(source, target, edge_type, weight)

        outcome_connections = [
            ("c_suite", "high_satisfaction", "leads_to", 0.55), ("c_suite", "financial_success", "leads_to", 0.75),
            ("c_suite", "burnout", "leads_to", 0.35), ("c_suite", "recognition", "leads_to", 0.80),
            ("vp", "high_satisfaction", "leads_to", 0.50), ("vp", "financial_success", "leads_to", 0.65),
            ("vp", "overwork", "leads_to", 0.40), ("director", "moderate_satisfaction", "leads_to", 0.60),
            ("director", "financial_stability", "leads_to", 0.70), ("director", "recognition", "leads_to", 0.55),
            ("manager", "moderate_satisfaction", "leads_to", 0.55), ("manager", "work_life_balance", "leads_to", 0.45),
            ("manager", "skill_growth", "leads_to", 0.50), ("senior", "skill_mastery", "leads_to", 0.60),
            ("senior", "work_life_balance", "leads_to", 0.55), ("senior", "financial_stability", "leads_to", 0.65),
            ("entrepreneur", "high_satisfaction", "leads_to", 0.45), ("entrepreneur", "financial_success", "leads_to", 0.30),
            ("entrepreneur", "financial_struggle", "leads_to", 0.40), ("entrepreneur", "burnout", "leads_to", 0.35),
            ("entrepreneur", "recognition", "leads_to", 0.40), ("consultant", "work_life_balance", "leads_to", 0.50),
            ("consultant", "financial_stability", "leads_to", 0.65), ("consultant", "skill_growth", "leads_to", 0.70),
            ("consultant", "network_expansion", "leads_to", 0.75), ("freelance", "work_life_balance", "leads_to", 0.70),
            ("freelance", "isolation", "leads_to", 0.40), ("freelance", "financial_struggle", "leads_to", 0.35),
            ("freelance", "skill_stagnation", "leads_to", 0.30),
        ]
        for source, target, edge_type, weight in outcome_connections:
            self.add_edge(source, target, edge_type, weight)

    def add_node(self, node_id: str, node_type: str, label: str, weight: float = 1.0,
                 attributes: Dict = None, embedding: np.ndarray = None) -> EnhancedGraphNode:
        node = EnhancedGraphNode(id=node_id, node_type=node_type, label=label,
                                 weight=weight, attributes=attributes or {}, embedding=embedding)
        self.graph.add_node(node_id, node_type=node_type, label=label, weight=weight, **node.attributes)
        self.node_registry[node_id] = node
        self.node_messages[node_id] = np.zeros(64)
        return node

    def add_edge(self, source: str, target: str, edge_type: str, weight: float = 1.0,
                 confidence: float = 0.5, attributes: Dict = None) -> EnhancedGraphEdge:
        edge = EnhancedGraphEdge(source=source, target=target, edge_type=edge_type,
                                 weight=weight, confidence=confidence, attributes=attributes or {})
        self.graph.add_edge(source, target, edge_type=edge_type, weight=weight,
                           confidence=confidence, **edge.attributes)
        self.edge_registry[(source, target)] = edge
        return edge

    def add_decision(self, decision_id: str, decision_type: str, description: str,
                     user_factors: Dict, outcomes: List[Tuple[str, float]] = None) -> str:
        self.add_node(node_id=decision_id, node_type="decision", label=description[:50],
                     weight=1.0, attributes={"decision_type": decision_type,
                                              "full_description": description, "user_factors": user_factors})
        if outcomes:
            for outcome_id, prob in outcomes:
                if outcome_id not in self.node_registry:
                    self.add_node(node_id=outcome_id, node_type="outcome",
                                 label=outcome_id.replace("_", " ").title(), weight=1.0)
                self.add_edge(decision_id, outcome_id, "leads_to", prob)
        else:
            self._auto_connect_outcomes(decision_id, decision_type, user_factors)
        self.decision_history.append({"decision_id": decision_id, "type": decision_type,
                                       "timestamp": datetime.utcnow().isoformat()})
        return decision_id

    def _auto_connect_outcomes(self, decision_id: str, decision_type: str, user_factors: Dict):
        risk = user_factors.get('risk_tolerance', 0.5)
        financial = user_factors.get('financial_stability', 0.5)
        outcome_probs = {
            'job_change': {'high_satisfaction': 0.3 + 0.2 * risk, 'moderate_satisfaction': 0.4,
                          'low_satisfaction': 0.3 - 0.2 * risk, 'financial_stability': 0.5 + 0.2 * financial,
                          'skill_growth': 0.5},
            'career_switch': {'high_satisfaction': 0.35 + 0.25 * risk, 'financial_struggle': 0.3 - 0.2 * financial,
                             'skill_growth': 0.6, 'isolation': 0.25},
            'startup': {'high_satisfaction': 0.35, 'financial_success': 0.2 * risk,
                       'financial_struggle': 0.5 - 0.3 * financial, 'burnout': 0.4, 'recognition': 0.3},
            'education': {'skill_mastery': 0.7, 'financial_struggle': 0.3 - 0.2 * financial,
                         'network_expansion': 0.5, 'high_satisfaction': 0.4},
            'freelance': {'work_life_balance': 0.6, 'isolation': 0.4,
                         'financial_stability': 0.4 * financial, 'skill_growth': 0.45}
        }
        type_outcomes = outcome_probs.get(decision_type, outcome_probs['job_change'])
        for outcome_id, prob in type_outcomes.items():
            if outcome_id in self.node_registry:
                self.add_edge(decision_id, outcome_id, "leads_to", prob)

    def message_passing(self, iterations: int = 3) -> Dict[str, np.ndarray]:
        for _ in range(iterations):
            new_messages = {}
            for node_id in self.graph.nodes():
                incoming = list(self.graph.predecessors(node_id))
                outgoing = list(self.graph.successors(node_id))
                neighbor_messages = []
                for neighbor in incoming:
                    edge_weight = self.graph[neighbor][node_id].get('weight', 0.5)
                    msg = self.node_messages.get(neighbor, np.zeros(64))
                    neighbor_messages.append(msg * edge_weight)
                for neighbor in outgoing:
                    edge_weight = self.graph[node_id][neighbor].get('weight', 0.5)
                    msg = self.node_messages.get(neighbor, np.zeros(64))
                    neighbor_messages.append(msg * edge_weight * 0.5)
                if neighbor_messages:
                    aggregated = np.mean(neighbor_messages, axis=0)
                    current = self.node_messages.get(node_id, np.zeros(64))
                    new_messages[node_id] = 0.5 * current + 0.5 * aggregated
                else:
                    new_messages[node_id] = self.node_messages.get(node_id, np.zeros(64))
            self.node_messages = new_messages
        return self.node_messages

    def analyze_decision(self, decision_id: str, user_factors: Dict = None,
                        run_monte_carlo: bool = True) -> Dict:
        if decision_id not in self.graph:
            return {"error": "Decision not found"}
        self.total_paths_analyzed += 1
        self.message_passing()
        reachable_outcomes = self._find_reachable_outcomes(decision_id)
        mc_results = {}
        if run_monte_carlo and self.mc_simulator:
            mc_results = self.mc_simulator.simulate_paths(decision_id, max_steps=8,
                                                          user_factors=user_factors or {})
        positive_outcomes = ['high_satisfaction', 'financial_success', 'work_life_balance',
                            'skill_mastery', 'skill_growth', 'network_expansion', 'recognition',
                            'financial_stability', 'moderate_satisfaction']
        negative_outcomes = ['low_satisfaction', 'financial_struggle', 'burnout', 'overwork',
                            'skill_stagnation', 'isolation', 'invisibility']
        positive_prob = sum(o['probability'] for o in reachable_outcomes if o['outcome'] in positive_outcomes)
        negative_prob = sum(o['probability'] for o in reachable_outcomes if o['outcome'] in negative_outcomes)
        total_prob = positive_prob + negative_prob
        regret_potential = negative_prob / total_prob if total_prob > 0 else 0.5
        best_case = max(reachable_outcomes, key=lambda x: x['probability']) if reachable_outcomes else None
        worst_case = max([o for o in reachable_outcomes if o['outcome'] in negative_outcomes],
                        key=lambda x: x['probability']) if [o for o in reachable_outcomes if o['outcome'] in negative_outcomes] else None
        return {
            "decision_id": decision_id,
            "reachable_outcomes": reachable_outcomes[:10],
            "positive_probability": positive_prob,
            "negative_probability": negative_prob,
            "regret_potential": regret_potential,
            "risk_level": self._get_risk_level(regret_potential),
            "best_case_scenario": best_case,
            "worst_case_scenario": worst_case,
            "monte_carlo": mc_results,
            "market_conditions": {
                "job_market_index": self.market.job_market_index,
                "relevant_industry_health": self.market.industry_health
            }
        }

    def _find_reachable_outcomes(self, start_node: str, max_depth: int = 5) -> List[Dict]:
        reachable = []
        outcome_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'outcome']
        for outcome in outcome_nodes:
            paths = self.find_paths(start_node, outcome, max_paths=3)
            if paths:
                best_path = paths[0]
                reachable.append({'outcome': outcome, 'probability': best_path['total_weight'],
                                 'path_length': best_path['length'], 'path': best_path['nodes']})
                self.outcome_frequencies[outcome] += 1
        reachable.sort(key=lambda x: x['probability'], reverse=True)
        return reachable

    def find_paths(self, source: str, target: str, max_paths: int = 5) -> List[Dict]:
        if source not in self.graph or target not in self.graph:
            return []
        paths = []
        try:
            for path in nx.all_simple_paths(self.graph, source, target, cutoff=8):
                path_weight = self._calculate_path_weight(path)
                paths.append({"nodes": path, "length": len(path), "total_weight": path_weight,
                             "edges": self._get_path_edges(path)})
        except nx.NetworkXNoPath:
            pass
        paths.sort(key=lambda x: x['total_weight'], reverse=True)
        return paths[:max_paths]

    def _calculate_path_weight(self, path: List[str]) -> float:
        if len(path) < 2:
            return 0.0
        total_weight = 1.0
        for i in range(len(path) - 1):
            edge_data = self.graph.get_edge_data(path[i], path[i + 1])
            if edge_data:
                edge_weight = edge_data.get('weight', 1.0)
                confidence = edge_data.get('confidence', 0.5)
                node_data = self.graph.nodes.get(path[i + 1], {})
                industry = node_data.get('industry')
                if industry:
                    edge_weight *= self.market.get_industry_modifier(industry)
                total_weight *= edge_weight * (self.decay_factor ** i) * (0.5 + 0.5 * confidence)
        return total_weight

    def _get_path_edges(self, path: List[str]) -> List[Dict]:
        edges = []
        for i in range(len(path) - 1):
            edge_data = self.graph.get_edge_data(path[i], path[i + 1])
            edges.append({"source": path[i], "target": path[i + 1],
                         "weight": edge_data.get('weight', 1.0) if edge_data else 0.0,
                         "confidence": edge_data.get('confidence', 0.5) if edge_data else 0.5,
                         "type": edge_data.get('edge_type', 'unknown') if edge_data else 'unknown'})
        return edges

    def simulate_whatif(self, current_state: str, hypothetical: str, factors: Dict[str, float]) -> Dict:
        if hypothetical not in self.graph:
            self.add_decision(decision_id=hypothetical, decision_type="hypothetical",
                             description=f"What-if: {hypothetical}", user_factors=factors)
        mc_result = self.mc_simulator.simulate_paths(hypothetical, user_factors=factors)
        sensitivity = None
        if 'risk_tolerance' in factors:
            sensitivity = self.mc_simulator.sensitivity_analysis(hypothetical, factors, 'risk_tolerance')
        return {"hypothetical_decision": hypothetical, "simulation_factors": factors,
                "monte_carlo_results": mc_result, "sensitivity_analysis": sensitivity, "is_simulation": True}

    def create_personalized_subgraph(self, user_id: str, industry: str,
                                     experience_level: str, preferences: Dict) -> nx.DiGraph:
        subgraph = self.graph.copy()
        industry_modifier = self.market.get_industry_modifier(industry)
        for u, v, data in subgraph.edges(data=True):
            original_weight = data.get('weight', 0.5)
            new_weight = original_weight * industry_modifier
            if 'work_life_priority' in preferences:
                if 'balance' in v.lower():
                    new_weight *= (1 + preferences['work_life_priority'])
                elif 'burnout' in v.lower() or 'overwork' in v.lower():
                    new_weight *= (1 - preferences['work_life_priority'])
            subgraph[u][v]['weight'] = np.clip(new_weight, 0.01, 1.0)
        self.user_subgraphs[user_id] = subgraph
        return subgraph

    def update_from_feedback(self, decision_id: str, actual_outcome: str, satisfaction: float):
        if decision_id not in self.graph:
            return
        if self.graph.has_edge(decision_id, actual_outcome):
            edge = self.edge_registry.get((decision_id, actual_outcome))
            if edge:
                old_weight = edge.weight
                edge.sample_count += 1
                edge.weight = (old_weight * (edge.sample_count - 1) + satisfaction) / edge.sample_count
                edge.confidence = min(0.95, edge.confidence + 0.05)
                edge.last_updated = datetime.utcnow()
                self.graph[decision_id][actual_outcome]['weight'] = edge.weight
                self.graph[decision_id][actual_outcome]['confidence'] = edge.confidence

    def _get_risk_level(self, regret: float) -> str:
        if regret < 0.25: return "low"
        elif regret < 0.45: return "moderate"
        elif regret < 0.65: return "elevated"
        else: return "high"

    def get_graph_statistics(self) -> Dict:
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": {ntype: len([n for n, d in self.graph.nodes(data=True) if d.get('node_type') == ntype])
                          for ntype in ['milestone', 'outcome', 'decision', 'factor']},
            "total_paths_analyzed": self.total_paths_analyzed,
            "decisions_recorded": len(self.decision_history),
            "top_outcomes": dict(sorted(self.outcome_frequencies.items(), key=lambda x: x[1], reverse=True)[:5]),
            "market_conditions": {"job_market_index": self.market.job_market_index,
                                  "last_update": self.market.last_update.isoformat()}
        }

    def export_graph(self) -> Dict:
        return {
            "nodes": [{"id": node_id, "type": data.get("node_type"), "label": data.get("label"),
                      "weight": data.get("weight"), "group": data.get("node_type")}
                     for node_id, data in self.graph.nodes(data=True)],
            "edges": [{"source": source, "target": target, "type": data.get("edge_type"),
                      "weight": data.get("weight"), "confidence": data.get("confidence", 0.5)}
                     for source, target, data in self.graph.edges(data=True)]
        }

WeightedDecisionGraph = AdvancedDecisionGraph
