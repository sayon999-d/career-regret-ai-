from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import math
import random
import hashlib

class PathOutcome(str, Enum):
    SUCCESS = "success"
    NEUTRAL = "neutral"
    CHALLENGE = "challenge"
    UNKNOWN = "unknown"

@dataclass
class DecisionNode3D:
    id: str
    label: str
    description: str
    x: float
    y: float
    z: float
    color: str
    size: float
    decision_type: str
    regret_probability: float
    success_probability: float
    is_current: bool = False
    is_chosen: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DecisionEdge3D:
    source_id: str
    target_id: str
    weight: float
    color: str
    probability: float
    label: str

@dataclass
class Timeline:
    id: str
    name: str
    nodes: List[DecisionNode3D]
    edges: List[DecisionEdge3D]
    probability: float
    final_outcome: PathOutcome
    total_regret: float
    total_satisfaction: float

class MultiverseVisualizationService:
    """
    Generates 3D coordinate data for visualizing decision trees as a multiverse.
    """

    COLORS = {
        "success": "#22c55e",
        "neutral": "#60a5fa",
        "challenge": "#ef4444",
        "current": "#ffffff",
        "chosen": "#a78bfa",
        "opportunity": "#fbbf24",
        "default": "#707070"
    }

    DECISION_TYPES = {
        "job_change": {"base_branches": 4, "depth": 5},
        "career_switch": {"base_branches": 5, "depth": 6},
        "promotion": {"base_branches": 3, "depth": 4},
        "entrepreneurship": {"base_branches": 6, "depth": 7},
        "education": {"base_branches": 4, "depth": 5},
        "relocation": {"base_branches": 4, "depth": 5}
    }

    def __init__(self):
        self.timelines: Dict[str, List[Timeline]] = {}
        self.node_cache: Dict[str, DecisionNode3D] = {}

    def generate_decision_forest(
        self,
        user_id: str,
        current_decision: Dict[str, Any],
        historical_decisions: List[Dict[str, Any]] = None,
        simulation_results: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate a complete 3D decision forest visualization data.
        """
        decision_type = current_decision.get("decision_type", "job_change")
        config = self.DECISION_TYPES.get(decision_type, self.DECISION_TYPES["job_change"])

        timelines = self._generate_timelines(
            user_id=user_id,
            root_decision=current_decision,
            num_branches=config["base_branches"],
            depth=config["depth"],
            simulation_results=simulation_results
        )

        self.timelines[user_id] = timelines

        all_nodes = []
        all_edges = []
        timeline_summaries = []

        for timeline in timelines:
            for node in timeline.nodes:
                all_nodes.append(self._node_to_dict(node))
            for edge in timeline.edges:
                all_edges.append(self._edge_to_dict(edge))
            timeline_summaries.append({
                "id": timeline.id,
                "name": timeline.name,
                "probability": timeline.probability,
                "outcome": timeline.final_outcome.value,
                "total_regret": timeline.total_regret,
                "total_satisfaction": timeline.total_satisfaction
            })

        return {
            "user_id": user_id,
            "decision_type": decision_type,
            "nodes": all_nodes,
            "edges": all_edges,
            "timelines": timeline_summaries,
            "stats": self._calculate_forest_stats(timelines),
            "camera_position": {"x": 0, "y": 50, "z": 100},
            "recommended_timeline": self._get_recommended_timeline(timelines)
        }

    def _generate_timelines(
        self,
        user_id: str,
        root_decision: Dict[str, Any],
        num_branches: int,
        depth: int,
        simulation_results: Dict[str, Any] = None
    ) -> List[Timeline]:
        """Generate multiple possible timeline branches."""
        timelines = []

        root_node = DecisionNode3D(
            id=f"root_{user_id}",
            label="Current Decision",
            description=root_decision.get("description", "Your current decision point"),
            x=0, y=0, z=0,
            color=self.COLORS["current"],
            size=2.0,
            decision_type=root_decision.get("decision_type", "job_change"),
            regret_probability=root_decision.get("predicted_regret", 30) / 100,
            success_probability=1 - root_decision.get("predicted_regret", 30) / 100,
            is_current=True
        )

        branch_names = self._get_branch_names(root_decision.get("decision_type", "job_change"))

        for i in range(num_branches):
            angle = (2 * math.pi * i) / num_branches
            branch_probability = self._generate_branch_probability(i, num_branches)

            timeline = self._generate_single_timeline(
                timeline_id=f"timeline_{user_id}_{i}",
                name=branch_names[i] if i < len(branch_names) else f"Path {i+1}",
                root_node=root_node,
                branch_index=i,
                total_branches=num_branches,
                depth=depth,
                base_angle=angle,
                probability=branch_probability
            )
            timelines.append(timeline)

        return timelines

    def _generate_single_timeline(
        self,
        timeline_id: str,
        name: str,
        root_node: DecisionNode3D,
        branch_index: int,
        total_branches: int,
        depth: int,
        base_angle: float,
        probability: float
    ) -> Timeline:
        """Generate a single timeline branch."""
        nodes = [root_node]
        edges = []

        trajectory = self._determine_trajectory(branch_index, total_branches)

        current_x, current_y, current_z = 0, 0, 0
        parent_id = root_node.id
        cumulative_regret = 0
        cumulative_satisfaction = 0

        for year in range(1, depth + 1):
            spread = 20 + year * 5
            current_x = math.cos(base_angle) * spread
            current_z = math.sin(base_angle) * spread
            current_y = year * 15

            noise_x = random.uniform(-3, 3)
            noise_z = random.uniform(-3, 3)

            regret, satisfaction = self._calculate_year_metrics(year, trajectory)
            cumulative_regret += regret
            cumulative_satisfaction += satisfaction

            node_id = f"{timeline_id}_y{year}"
            outcome = self._determine_node_outcome(regret, satisfaction)

            node = DecisionNode3D(
                id=node_id,
                label=f"Year {year}",
                description=self._generate_year_description(year, trajectory, outcome),
                x=current_x + noise_x,
                y=current_y,
                z=current_z + noise_z,
                color=self._get_outcome_color(outcome),
                size=1.0 + satisfaction / 100,
                decision_type=root_node.decision_type,
                regret_probability=regret / 100,
                success_probability=satisfaction / 100,
                metadata={
                    "year": year,
                    "trajectory": trajectory,
                    "cumulative_regret": cumulative_regret,
                    "cumulative_satisfaction": cumulative_satisfaction
                }
            )

            nodes.append(node)

            edge = DecisionEdge3D(
                source_id=parent_id,
                target_id=node_id,
                weight=1.0 - regret / 200,
                color=self._get_outcome_color(outcome),
                probability=probability * (1 - regret / 200),
                label=f"Year {year}" if year <= 2 else ""
            )

            edges.append(edge)
            parent_id = node_id

        final_outcome = self._determine_final_outcome(trajectory, cumulative_regret / depth)

        return Timeline(
            id=timeline_id,
            name=name,
            nodes=nodes,
            edges=edges,
            probability=probability,
            final_outcome=final_outcome,
            total_regret=cumulative_regret / depth,
            total_satisfaction=cumulative_satisfaction / depth
        )

    def _determine_trajectory(self, branch_index: int, total: int) -> str:
        if branch_index == 0:
            return "optimistic"
        elif branch_index == total - 1:
            return "pessimistic"
        else:
            return random.choice(["realistic_positive", "realistic_neutral", "realistic_negative"])

    def _calculate_year_metrics(self, year: int, trajectory: str) -> Tuple[float, float]:
        base_regret = 30
        base_satisfaction = 60

        if trajectory == "optimistic":
            regret = max(5, base_regret - year * 5 + random.uniform(-5, 5))
            satisfaction = min(95, base_satisfaction + year * 5 + random.uniform(-5, 5))
        elif trajectory == "pessimistic":
            regret = min(80, base_regret + year * 8 + random.uniform(-5, 5))
            satisfaction = max(20, base_satisfaction - year * 6 + random.uniform(-5, 5))
        elif trajectory == "realistic_positive":
            regret = max(10, base_regret - year * 3 + random.uniform(-8, 8))
            satisfaction = min(85, base_satisfaction + year * 3 + random.uniform(-8, 8))
        elif trajectory == "realistic_negative":
            regret = min(60, base_regret + year * 4 + random.uniform(-8, 8))
            satisfaction = max(35, base_satisfaction - year * 3 + random.uniform(-8, 8))
        else:
            regret = base_regret + random.uniform(-10, 10)
            satisfaction = base_satisfaction + random.uniform(-10, 10)

        return regret, satisfaction

    def _determine_node_outcome(self, regret: float, satisfaction: float) -> PathOutcome:
        if satisfaction > 70 and regret < 30:
            return PathOutcome.SUCCESS
        elif satisfaction < 40 or regret > 60:
            return PathOutcome.CHALLENGE
        else:
            return PathOutcome.NEUTRAL

    def _determine_final_outcome(self, trajectory: str, avg_regret: float) -> PathOutcome:
        if trajectory == "optimistic" or avg_regret < 25:
            return PathOutcome.SUCCESS
        elif trajectory == "pessimistic" or avg_regret > 50:
            return PathOutcome.CHALLENGE
        else:
            return PathOutcome.NEUTRAL

    def _get_outcome_color(self, outcome: PathOutcome) -> str:
        return self.COLORS.get(outcome.value, self.COLORS["default"])

    def _generate_year_description(self, year: int, trajectory: str, outcome: PathOutcome) -> str:
        descriptions = {
            ("optimistic", PathOutcome.SUCCESS): f"Year {year}: Thriving with strong momentum",
            ("optimistic", PathOutcome.NEUTRAL): f"Year {year}: Steady progress with clear direction",
            ("pessimistic", PathOutcome.CHALLENGE): f"Year {year}: Facing obstacles, building resilience",
            ("pessimistic", PathOutcome.NEUTRAL): f"Year {year}: Navigating uncertainty",
            ("realistic_positive", PathOutcome.SUCCESS): f"Year {year}: Good progress despite challenges",
            ("realistic_negative", PathOutcome.CHALLENGE): f"Year {year}: Learning through difficulties",
        }

        key = (trajectory, outcome)
        return descriptions.get(key, f"Year {year}: Path unfolding")

    def _get_branch_names(self, decision_type: str) -> List[str]:
        names = {
            "job_change": ["Best Case", "Growth Path", "Stable Path", "Challenge Path"],
            "career_switch": ["Passion Path", "Gradual Transition", "Hybrid Approach", "Bold Leap", "Safe Exit"],
            "promotion": ["Leadership Track", "Expert Track", "Balanced Growth"],
            "entrepreneurship": ["Rapid Scale", "Steady Build", "Pivot Ready", "Bootstrap", "Partnership", "Exit Plan"],
            "education": ["Full Commit", "Part-time Balance", "Self-Directed", "Hybrid"],
            "relocation": ["Full Immersion", "Remote Hybrid", "Trial Period", "Network First"]
        }
        return names.get(decision_type, ["Path A", "Path B", "Path C", "Path D"])

    def _generate_branch_probability(self, index: int, total: int) -> float:
        if total <= 2:
            return 0.5
        if index == 0:
            return 0.25
        elif index == total - 1:
            return 0.15
        else:
            remaining = 0.60
            middle_branches = total - 2
            return remaining / middle_branches

    def _calculate_forest_stats(self, timelines: List[Timeline]) -> Dict[str, Any]:
        if not timelines:
            return {}

        avg_regret = sum(t.total_regret for t in timelines) / len(timelines)
        avg_satisfaction = sum(t.total_satisfaction for t in timelines) / len(timelines)
        success_paths = len([t for t in timelines if t.final_outcome == PathOutcome.SUCCESS])

        return {
            "total_timelines": len(timelines),
            "success_paths": success_paths,
            "challenge_paths": len([t for t in timelines if t.final_outcome == PathOutcome.CHALLENGE]),
            "average_regret": round(avg_regret, 1),
            "average_satisfaction": round(avg_satisfaction, 1),
            "best_path_name": min(timelines, key=lambda t: t.total_regret).name,
            "success_rate": f"{(success_paths / len(timelines)) * 100:.0f}%"
        }

    def _get_recommended_timeline(self, timelines: List[Timeline]) -> Optional[Dict[str, Any]]:
        if not timelines:
            return None

        best = min(timelines, key=lambda t: t.total_regret - t.total_satisfaction * 0.5)

        return {
            "id": best.id,
            "name": best.name,
            "probability": best.probability,
            "outcome": best.final_outcome.value,
            "total_regret": best.total_regret,
            "recommendation": f"The '{best.name}' path shows the best balance of low regret and high satisfaction."
        }

    def get_timeline_details(self, user_id: str, timeline_id: str) -> Optional[Dict[str, Any]]:
        if user_id not in self.timelines:
            return None

        for timeline in self.timelines[user_id]:
            if timeline.id == timeline_id:
                return {
                    "id": timeline.id,
                    "name": timeline.name,
                    "nodes": [self._node_to_dict(n) for n in timeline.nodes],
                    "edges": [self._edge_to_dict(e) for e in timeline.edges],
                    "probability": timeline.probability,
                    "outcome": timeline.final_outcome.value,
                    "total_regret": timeline.total_regret,
                    "total_satisfaction": timeline.total_satisfaction
                }
        return None

    def _node_to_dict(self, node: DecisionNode3D) -> Dict[str, Any]:
        return {
            "id": node.id,
            "label": node.label,
            "description": node.description,
            "position": {"x": node.x, "y": node.y, "z": node.z},
            "color": node.color,
            "size": node.size,
            "decision_type": node.decision_type,
            "regret_probability": node.regret_probability,
            "success_probability": node.success_probability,
            "is_current": node.is_current,
            "is_chosen": node.is_chosen,
            "metadata": node.metadata
        }

    def _edge_to_dict(self, edge: DecisionEdge3D) -> Dict[str, Any]:
        return {
            "source": edge.source_id,
            "target": edge.target_id,
            "weight": edge.weight,
            "color": edge.color,
            "probability": edge.probability,
            "label": edge.label
        }

multiverse_viz = MultiverseVisualizationService()
