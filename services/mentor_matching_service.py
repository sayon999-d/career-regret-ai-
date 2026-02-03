import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class MentorProfile:
    id: str
    name: str
    expertise: List[str]
    industry: str
    years_experience: int
    availability: bool = True
    match_score: float = 0.0
    bio: str = ""
    video_recommendations: List[Dict] = field(default_factory=list)

@dataclass
class MentorMatch:
    user_id: str
    mentor_id: str
    score: float
    matched_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "pending"
    messages: List[Dict] = field(default_factory=list)
    recommended_videos: List[Dict] = field(default_factory=list)
    learning_resources: List[Dict] = field(default_factory=list)

class MentorMatchingService:
    def __init__(self):
        self.mentors: Dict[str, MentorProfile] = self._seed_mentors()
        self.matches: List[MentorMatch] = []

    def _seed_mentors(self) -> Dict[str, MentorProfile]:
        profiles = [
            MentorProfile("m1", "Dr. Sarah Chen", ["AI", "Research", "Academia"], "Technology", 12, bio="Former lead at DeepMind, expert in NLP."),
            MentorProfile("m2", "James Wilson", ["Management", "Product", "SaaS"], "Technology", 15, bio="Product director at a top 10 SaaS company."),
            MentorProfile("m3", "Elena Rodriguez", ["Fintech", "Investment", "Banking"], "Finance", 10, bio="Ex-Goldman Sachs, specialized in fintech startups."),
            MentorProfile("m4", "Marcus Thorne", ["Engineering", "Startup", "Scaling"], "Technology", 20, bio="Serial entrepreneur and CTO mentor."),
            MentorProfile("m5", "Linda Park", ["Marketing", "Growth", "Brand"], "Creative", 8, bio="Award-winning brand strategist.")
        ]
        return {m.id: m for m in profiles}

    def get_matches(self, user_requirements: Dict[str, Any]) -> List[MentorProfile]:
        results = []
        user_industry = user_requirements.get("industry", "")
        user_expertise = user_requirements.get("expertise", [])

        for mentor in self.mentors.values():
            if not mentor.availability:
                continue

            score = 0.0
            if mentor.industry == user_industry:
                score += 0.4

            overlap = set(mentor.expertise) & set(user_expertise)
            if overlap:
                score += (len(overlap) / max(len(user_expertise), 1)) * 0.5

            if mentor.years_experience > 10:
                score += 0.1

            if score > 0.3:
                m_copy = MentorProfile(**mentor.__dict__)
                m_copy.match_score = round(score, 2)
                results.append(m_copy)

        return sorted(results, key=lambda x: x.match_score, reverse=True)

    def request_match(self, user_id: str, mentor_id: str) -> bool:
        if mentor_id not in self.mentors:
            return False

        if any(m.user_id == user_id and m.mentor_id == mentor_id for m in self.matches):
            return True

        match = MentorMatch(user_id=user_id, mentor_id=mentor_id, score=0.0)
        match.status = "accepted"
        match.messages.append({
            "sender": "mentor",
            "text": f"Hi! I'm {self.mentors[mentor_id].name}. I've reviewed your profile and would love to help with your career journey.",
            "timestamp": datetime.utcnow().isoformat()
        })
        self.matches.append(match)
        return True

    def send_message(self, user_id: str, mentor_id: str, text: str) -> bool:
        match = next((m for m in self.matches if m.user_id == user_id and m.mentor_id == mentor_id), None)
        if not match: return False

        match.messages.append({
            "sender": "user",
            "text": text,
            "timestamp": datetime.utcnow().isoformat()
        })
        return True

    def get_user_mentors(self, user_id: str) -> List[Dict]:
        user_matches = [m for m in self.matches if m.user_id == user_id]
        return [
            {
                "mentor": self.mentors[m.mentor_id].__dict__,
                "status": m.status,
                "matched_at": m.matched_at.isoformat(),
                "messages": m.messages,
                "recommended_videos": m.recommended_videos,
                "learning_resources": m.learning_resources
            }
            for m in user_matches if m.mentor_id in self.mentors
        ]
    
    def get_mentor_profile_with_resources(self, mentor_id: str) -> Optional[Dict]:
        """Get mentor profile with recommended videos and learning resources"""
        if mentor_id not in self.mentors:
            return None
        
        mentor = self.mentors[mentor_id]
        return {
            "id": mentor.id,
            "name": mentor.name,
            "expertise": mentor.expertise,
            "industry": mentor.industry,
            "years_experience": mentor.years_experience,
            "availability": mentor.availability,
            "bio": mentor.bio,
            "video_recommendations": mentor.video_recommendations,
            "expertise_videos": self._get_expertise_videos(mentor)
        }
    
    def _get_expertise_videos(self, mentor: MentorProfile) -> List[Dict]:
        """Get videos related to mentor's expertise (placeholder for integration)"""
        return []
    
    def add_recommended_videos_to_match(
        self,
        user_id: str,
        mentor_id: str,
        videos: List[Dict]
    ) -> bool:
        """Add recommended videos to a mentor match"""
        match = next((m for m in self.matches if m.user_id == user_id and m.mentor_id == mentor_id), None)
        if not match:
            return False
        
        match.recommended_videos = videos
        return True
    
    def add_learning_resources(
        self,
        user_id: str,
        mentor_id: str,
        resources: List[Dict]
    ) -> bool:
        """Add learning resources to a mentor match"""
        match = next((m for m in self.matches if m.user_id == user_id and m.mentor_id == mentor_id), None)
        if not match:
            return False
        
        match.learning_resources = resources
        return True

mentor_matching_service = MentorMatchingService()
