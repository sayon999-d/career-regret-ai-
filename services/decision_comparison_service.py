from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass
import statistics


@dataclass
class ComparisonOption:
    """Represents an option in a decision comparison"""
    id: str
    title: str
    description: str
    pros: List[str]
    cons: List[str]
    importance_weights: Dict[str, float]
    scores: Dict[str, float]
    total_score: float = 0
    predicted_regret: float = 50


class DecisionComparisonService:
    """Service for comparing multiple decision options"""
    
    def __init__(self):
        self.default_criteria = [
            {"name": "career_growth", "label": "Career Growth Potential", "weight": 0.2},
            {"name": "work_life_balance", "label": "Work-Life Balance", "weight": 0.15},
            {"name": "compensation", "label": "Compensation & Benefits", "weight": 0.2},
            {"name": "learning", "label": "Learning Opportunities", "weight": 0.15},
            {"name": "job_security", "label": "Job Security", "weight": 0.1},
            {"name": "impact", "label": "Impact & Meaning", "weight": 0.1},
            {"name": "culture_fit", "label": "Culture Fit", "weight": 0.1}
        ]
    
    def create_comparison(
        self,
        decision_title: str,
        options: List[Dict],
        criteria: List[Dict] = None
    ) -> Dict:
        """Create a new decision comparison"""
        comparison_id = f"cmp_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        if not criteria:
            criteria = self.default_criteria
        
        total_weight = sum(c.get('weight', 1) for c in criteria)
        for c in criteria:
            c['weight'] = c.get('weight', 1) / total_weight
        
        return {
            "id": comparison_id,
            "title": decision_title,
            "created_at": datetime.utcnow().isoformat(),
            "options": options,
            "criteria": criteria,
            "status": "pending"
        }
    
    def evaluate_options(
        self,
        comparison: Dict,
        option_scores: Dict[str, Dict[str, float]]
    ) -> Dict:
        """Evaluate options based on criteria scores"""
        criteria = comparison.get("criteria", self.default_criteria)
        options = comparison.get("options", [])
        
        results = []
        
        for option in options:
            option_id = option.get("id")
            scores = option_scores.get(option_id, {})
            
            weighted_total = 0
            score_breakdown = []
            
            for criterion in criteria:
                criterion_name = criterion.get("name")
                weight = criterion.get("weight", 0.1)
                score = scores.get(criterion_name, 5)
                
                weighted_score = score * weight
                weighted_total += weighted_score
                
                score_breakdown.append({
                    "criterion": criterion_name,
                    "label": criterion.get("label", criterion_name),
                    "score": score,
                    "weight": weight,
                    "weighted_score": round(weighted_score, 2)
                })
            
            normalized_score = weighted_total / 10
            predicted_regret = max(0, min(100, (1 - normalized_score) * 100))
            
            results.append({
                "option_id": option_id,
                "option_title": option.get("title"),
                "total_score": round(weighted_total, 2),
                "normalized_score": round(normalized_score * 100, 1),
                "predicted_regret": round(predicted_regret, 1),
                "score_breakdown": score_breakdown
            })
        
        results.sort(key=lambda x: x["total_score"], reverse=True)
        
        for i, result in enumerate(results):
            result["rank"] = i + 1
        
        insights = self._generate_comparison_insights(results, criteria)
        
        return {
            "comparison_id": comparison.get("id"),
            "title": comparison.get("title"),
            "evaluated_at": datetime.utcnow().isoformat(),
            "results": results,
            "insights": insights,
            "recommendation": results[0] if results else None
        }
    
    def _generate_comparison_insights(
        self,
        results: List[Dict],
        criteria: List[Dict]
    ) -> List[Dict]:
        """Generate insights from comparison results"""
        insights = []
        
        if len(results) < 2:
            return insights
        
        top_option = results[0]
        runner_up = results[1]
        
        score_diff = top_option["total_score"] - runner_up["total_score"]
        if score_diff < 0.5:
            insights.append({
                "type": "close_call",
                "title": "Close Decision",
                "description": f"The top two options ({top_option['option_title']} and {runner_up['option_title']}) are very close. Consider your gut feeling.",
                "importance": "high"
            })
        elif score_diff > 2:
            insights.append({
                "type": "clear_winner",
                "title": "Clear Recommendation",
                "description": f"{top_option['option_title']} significantly outscores other options.",
                "importance": "medium"
            })
        
        for criterion in criteria:
            crit_name = criterion["name"]
            
            crit_scores = []
            for result in results:
                for breakdown in result["score_breakdown"]:
                    if breakdown["criterion"] == crit_name:
                        crit_scores.append({
                            "option": result["option_title"],
                            "score": breakdown["score"]
                        })
            
            if crit_scores:
                min_score = min(c["score"] for c in crit_scores)
                max_score = max(c["score"] for c in crit_scores)
                
                if max_score - min_score > 3:
                    insights.append({
                        "type": "criterion_variance",
                        "title": f"High Variance in {criterion['label']}",
                        "description": f"Options differ significantly in {criterion['label']}. Consider how important this is to you.",
                        "importance": "medium"
                    })
        
        return insights
    
    def what_if_analysis(
        self,
        comparison: Dict,
        option_scores: Dict[str, Dict[str, float]],
        scenario_changes: Dict[str, float]
    ) -> Dict:
        """Perform what-if analysis by adjusting criteria weights"""
        modified_comparison = comparison.copy()
        modified_criteria = []
        
        for criterion in comparison.get("criteria", []):
            new_criterion = criterion.copy()
            if criterion["name"] in scenario_changes:
                new_criterion["weight"] = scenario_changes[criterion["name"]]
            modified_criteria.append(new_criterion)
        
        total_weight = sum(c["weight"] for c in modified_criteria)
        for c in modified_criteria:
            c["weight"] = c["weight"] / total_weight
        
        modified_comparison["criteria"] = modified_criteria
        
        return self.evaluate_options(modified_comparison, option_scores)
    
    def sensitivity_analysis(
        self,
        comparison: Dict,
        option_scores: Dict[str, Dict[str, float]]
    ) -> Dict:
        """Analyze how sensitive the results are to weight changes"""
        base_results = self.evaluate_options(comparison, option_scores)
        base_winner = base_results["results"][0]["option_id"] if base_results["results"] else None
        
        sensitivity = []
        
        for criterion in comparison.get("criteria", []):
            crit_name = criterion["name"]
            
            scenario_high = {crit_name: criterion["weight"] * 2}
            high_results = self.what_if_analysis(comparison, option_scores, scenario_high)
            high_winner = high_results["results"][0]["option_id"] if high_results["results"] else None
            
            scenario_low = {crit_name: criterion["weight"] * 0.5}
            low_results = self.what_if_analysis(comparison, option_scores, scenario_low)
            low_winner = low_results["results"][0]["option_id"] if low_results["results"] else None
            
            sensitivity.append({
                "criterion": crit_name,
                "label": criterion.get("label", crit_name),
                "base_weight": criterion["weight"],
                "winner_changes": high_winner != base_winner or low_winner != base_winner,
                "sensitivity_level": "high" if high_winner != base_winner or low_winner != base_winner else "low"
            })
        
        return {
            "base_winner": base_winner,
            "sensitivity_analysis": sensitivity,
            "robust_decision": all(s["sensitivity_level"] == "low" for s in sensitivity)
        }
    
    def compare_pros_cons(self, options: List[Dict]) -> Dict:
        """Compare pros and cons across options"""
        all_pros = []
        all_cons = []
        
        for option in options:
            option_id = option.get("id")
            option_title = option.get("title")
            
            for pro in option.get("pros", []):
                all_pros.append({
                    "option_id": option_id,
                    "option_title": option_title,
                    "text": pro
                })
            
            for con in option.get("cons", []):
                all_cons.append({
                    "option_id": option_id,
                    "option_title": option_title,
                    "text": con
                })
        
        pro_texts = [p["text"].lower() for p in all_pros]
        con_texts = [c["text"].lower() for c in all_cons]
        
        return {
            "total_pros": len(all_pros),
            "total_cons": len(all_cons),
            "pros_by_option": {
                opt["id"]: len([p for p in all_pros if p["option_id"] == opt["id"]])
                for opt in options
            },
            "cons_by_option": {
                opt["id"]: len([c for c in all_cons if c["option_id"] == opt["id"]])
                for opt in options
            },
            "all_pros": all_pros,
            "all_cons": all_cons
        }
    
    def generate_recommendation(self, evaluation: Dict) -> Dict:
        """Generate a final recommendation based on evaluation"""
        results = evaluation.get("results", [])
        
        if not results:
            return {"recommendation": None, "confidence": 0}
        
        top_option = results[0]
        
        if len(results) >= 2:
            score_margin = top_option["total_score"] - results[1]["total_score"]
            confidence = min(100, 50 + score_margin * 10)
        else:
            confidence = 75
        
        if top_option["predicted_regret"] > 40:
            confidence = confidence * 0.8
        
        return {
            "recommended_option": top_option["option_title"],
            "option_id": top_option["option_id"],
            "confidence": round(confidence, 1),
            "predicted_regret": top_option["predicted_regret"],
            "rationale": self._generate_rationale(top_option, results, evaluation.get("insights", [])),
            "caveats": self._generate_caveats(top_option, results)
        }
    
    def _generate_rationale(
        self,
        top_option: Dict,
        all_results: List[Dict],
        insights: List[Dict]
    ) -> str:
        """Generate human-readable rationale"""
        strengths = []
        
        for breakdown in top_option.get("score_breakdown", []):
            if breakdown["score"] >= 7:
                strengths.append(breakdown["label"])
        
        rationale = f"**{top_option['option_title']}** is recommended based on its strong performance"
        
        if strengths:
            rationale += f" in {', '.join(strengths[:3])}"
        
        rationale += f". It has a normalized score of {top_option['normalized_score']}% with an estimated regret of {top_option['predicted_regret']}%."
        
        return rationale
    
    def _generate_caveats(self, top_option: Dict, all_results: List[Dict]) -> List[str]:
        """Generate caveats for the recommendation"""
        caveats = []
        
        if top_option["predicted_regret"] > 30:
            caveats.append("Consider that this option still has a moderate regret potential.")
        
        for breakdown in top_option.get("score_breakdown", []):
            if breakdown["score"] <= 4:
                caveats.append(f"Low score in {breakdown['label']} - ensure this is acceptable.")
        
        if len(all_results) >= 2 and all_results[0]["total_score"] - all_results[1]["total_score"] < 0.5:
            caveats.append("The top options are very close - trust your intuition.")
        
        return caveats[:3]


decision_comparison_service = DecisionComparisonService()
