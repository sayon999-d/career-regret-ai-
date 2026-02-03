from typing import List, Dict, Any
from datetime import datetime

class RoadmapService:
    def __init__(self):
        self.roadmaps: Dict[str, Dict] = {}

    def generate_roadmap(self, user_id: str, target_role: str, gap_skills: List[str]) -> Dict[str, Any]:
        """Generates a career progression roadmap based on identified skill gaps."""

        milestones = []

        if gap_skills:
            milestones.append({
                "phase": "Foundation",
                "weeks": "1-4",
                "tasks": [f"Learn fundamentals of {gap_skills[0]}", "Complete introductory certification"],
                "resources": ["Official Documentation", "Coursera/Udemy Foundational Course"]
            })

        if len(gap_skills) > 1:
            milestones.append({
                "phase": "Active Application",
                "weeks": "5-8",
                "tasks": [f"Build a portfolio project using {gap_skills[1]}", f"Integrate {gap_skills[0]} with existing projects"],
                "resources": ["GitHub Repositories", "Project-based Tutorials"]
            })
        else:
            milestones.append({
                "phase": "Intermediate Mastery",
                "weeks": "5-8",
                "tasks": [f"Build specialized tool with {gap_skills[0]}", "Deep dive into advanced concepts"],
                "resources": ["Advanced Workshops", "Technical Blogs"]
            })

        milestones.append({
            "phase": "Market Readiness",
            "weeks": "9-12",
            "tasks": [f"Update resume with {', '.join(gap_skills)}", "Mock interviews for target role"],
            "resources": ["Career Decision AI Interview Practice", "Peer Networking"]
        })

        roadmap = {
            "user_id": user_id,
            "target_role": target_role,
            "generated_at": datetime.utcnow().isoformat(),
            "milestones": milestones,
            "estimated_completion": "3 Months"
        }

        self.roadmaps[user_id] = roadmap
        return roadmap

    def get_roadmap(self, user_id: str) -> Dict:
        return self.roadmaps.get(user_id, {})

roadmap_service = RoadmapService()
