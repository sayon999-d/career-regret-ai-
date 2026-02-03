import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class VideoCategory(Enum):
    """YouTube video categories for mentor matching"""
    CAREER_DEVELOPMENT = "career_development"
    TECHNICAL_SKILLS = "technical_skills"
    LEADERSHIP = "leadership"
    INDUSTRY_INSIGHTS = "industry_insights"
    PERSONAL_GROWTH = "personal_growth"
    NEGOTIATION = "negotiation"
    ENTREPRENEURSHIP = "entrepreneurship"
    WORK_LIFE_BALANCE = "work_life_balance"
    INNOVATION = "innovation"
    FINANCIAL_PLANNING = "financial_planning"


@dataclass
class YouTubeVideo:
    """Represents a YouTube video recommendation"""
    video_id: str
    title: str
    channel: str
    duration_minutes: int
    category: VideoCategory
    keywords: List[str]
    relevance_score: float = 0.0
    description: str = ""
    thumbnail_url: str = ""
    published_date: Optional[str] = None
    view_count: int = 0
    watch_later: bool = False
    watched: bool = False
    rating: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class UserProfile:
    """User profile for recommendation engine"""
    user_id: str
    expertise: List[str]
    industry: str
    years_experience: int
    career_goals: List[str] = field(default_factory=list)
    interests: List[str] = field(default_factory=list)
    learning_history: List[str] = field(default_factory=list)
    watched_videos: List[str] = field(default_factory=list)
    skill_gaps: List[str] = field(default_factory=list)
    preferred_content_length: str = "medium"


class YouTubeRecommendationService:
    """
    Service for recommending YouTube videos based on mentor matching,
    user profile, usage patterns, and career development needs.
    """
    
    def __init__(self):
        self.videos_db: Dict[str, YouTubeVideo] = self._initialize_video_database()
        self.user_profiles: Dict[str, UserProfile] = {}
        self.recommendation_cache: Dict[str, List[YouTubeVideo]] = {}
        self.cache_ttl: timedelta = timedelta(hours=24)
        
    def _initialize_video_database(self) -> Dict[str, YouTubeVideo]:
        """Initialize database with sample YouTube videos"""
        videos = [
            YouTubeVideo(
                video_id="vid_001",
                title="Career Pivoting: From Technical to Management",
                channel="Career Masters",
                duration_minutes=18,
                category=VideoCategory.CAREER_DEVELOPMENT,
                keywords=["career", "management", "transition", "leadership"],
                description="Learn how to successfully transition from technical roles to management positions.",
                view_count=524000
            ),
            YouTubeVideo(
                video_id="vid_002",
                title="5 Years in Tech: What I Wished I Knew",
                channel="Dev Life Diaries",
                duration_minutes=22,
                category=VideoCategory.CAREER_DEVELOPMENT,
                keywords=["tech", "career", "lessons", "mistakes"],
                description="Reflections on crucial lessons learned during 5 years in the technology industry.",
                view_count=312000
            ),
            YouTubeVideo(
                video_id="vid_003",
                title="Building Your Personal Brand as a Professional",
                channel="Professional Growth Hub",
                duration_minutes=25,
                category=VideoCategory.PERSONAL_GROWTH,
                keywords=["personal brand", "marketing", "networking", "visibility"],
                description="Strategic approach to building and maintaining your professional personal brand.",
                view_count=487000
            ),
            
            YouTubeVideo(
                video_id="vid_004",
                title="Advanced Machine Learning: From Theory to Practice",
                channel="AI Insider",
                duration_minutes=42,
                category=VideoCategory.TECHNICAL_SKILLS,
                keywords=["machine learning", "AI", "deep learning", "neural networks"],
                description="Advanced ML concepts and their practical applications in real-world scenarios.",
                view_count=156000
            ),
            YouTubeVideo(
                video_id="vid_005",
                title="Cloud Architecture Best Practices 2024",
                channel="Cloud Architects",
                duration_minutes=35,
                category=VideoCategory.TECHNICAL_SKILLS,
                keywords=["cloud", "AWS", "architecture", "scalability"],
                description="Latest cloud architecture patterns and best practices for scalable systems.",
                view_count=89000
            ),
            YouTubeVideo(
                video_id="vid_006",
                title="Full-Stack Development: RESTful APIs to React",
                channel="Code Academy Pro",
                duration_minutes=38,
                category=VideoCategory.TECHNICAL_SKILLS,
                keywords=["full-stack", "REST API", "React", "backend"],
                description="Complete guide to building modern full-stack applications.",
                view_count=234000
            ),
            
            YouTubeVideo(
                video_id="vid_007",
                title="Leading High-Performance Teams",
                channel="Leadership Institute",
                duration_minutes=28,
                category=VideoCategory.LEADERSHIP,
                keywords=["leadership", "team management", "motivation", "performance"],
                description="Strategies for building and leading teams that consistently exceed expectations.",
                view_count=412000
            ),
            YouTubeVideo(
                video_id="vid_008",
                title="Emotional Intelligence in the Workplace",
                channel="Soft Skills Academy",
                duration_minutes=19,
                category=VideoCategory.LEADERSHIP,
                keywords=["emotional intelligence", "EQ", "soft skills", "communication"],
                description="How emotional intelligence impacts leadership effectiveness and team dynamics.",
                view_count=567000
            ),
            YouTubeVideo(
                video_id="vid_009",
                title="Strategic Decision Making for Leaders",
                channel="Executive Coach",
                duration_minutes=31,
                category=VideoCategory.LEADERSHIP,
                keywords=["decision making", "strategy", "risk management", "leadership"],
                description="Frameworks and techniques for making strategic business decisions.",
                view_count=298000
            ),
            
            YouTubeVideo(
                video_id="vid_010",
                title="The Future of Finance: FinTech Revolution",
                channel="Finance Forward",
                duration_minutes=26,
                category=VideoCategory.INDUSTRY_INSIGHTS,
                keywords=["fintech", "finance", "blockchain", "cryptocurrency"],
                description="How fintech is transforming traditional financial services.",
                view_count=378000
            ),
            YouTubeVideo(
                video_id="vid_011",
                title="AI in Healthcare: Opportunities and Challenges",
                channel="Health Tech Weekly",
                duration_minutes=23,
                category=VideoCategory.INDUSTRY_INSIGHTS,
                keywords=["AI", "healthcare", "medical technology", "innovation"],
                description="Current state and future prospects of AI applications in healthcare.",
                view_count=445000
            ),
            YouTubeVideo(
                video_id="vid_012",
                title="E-commerce Evolution: DTC Brands in 2024",
                channel="Commerce Insights",
                duration_minutes=29,
                category=VideoCategory.INDUSTRY_INSIGHTS,
                keywords=["e-commerce", "DTC", "marketing", "retail"],
                description="Trends and strategies for Direct-to-Consumer e-commerce success.",
                view_count=203000
            ),
            
            YouTubeVideo(
                video_id="vid_013",
                title="Salary Negotiation: Getting What You Deserve",
                channel="Career Coach Pro",
                duration_minutes=24,
                category=VideoCategory.NEGOTIATION,
                keywords=["salary", "negotiation", "compensation", "career"],
                description="Practical techniques for negotiating better compensation packages.",
                view_count=892000
            ),
            YouTubeVideo(
                video_id="vid_014",
                title="Business Negotiation Strategies",
                channel="B2B Masters",
                duration_minutes=37,
                category=VideoCategory.NEGOTIATION,
                keywords=["negotiation", "business", "contracts", "deals"],
                description="Advanced negotiation tactics for business professionals.",
                view_count=156000
            ),
            
            YouTubeVideo(
                video_id="vid_015",
                title="From Idea to MVP: Launching Your Startup",
                channel="Startup Hub",
                duration_minutes=41,
                category=VideoCategory.ENTREPRENEURSHIP,
                keywords=["startup", "MVP", "entrepreneur", "founding"],
                description="Step-by-step guide to launching your first startup product.",
                view_count=567000
            ),
            YouTubeVideo(
                video_id="vid_016",
                title="Fundraising 101: Pitching to VCs",
                channel="Venture Capital Insider",
                duration_minutes=33,
                category=VideoCategory.ENTREPRENEURSHIP,
                keywords=["fundraising", "venture capital", "pitching", "startup"],
                description="How to successfully pitch your startup to venture capitalists.",
                view_count=434000
            ),
            
            YouTubeVideo(
                video_id="vid_017",
                title="Preventing Burnout: Sustainable Career Growth",
                channel="Wellness at Work",
                duration_minutes=20,
                category=VideoCategory.WORK_LIFE_BALANCE,
                keywords=["burnout", "work-life balance", "wellness", "mental health"],
                description="Strategies for maintaining work-life balance while advancing your career.",
                view_count=623000
            ),
            YouTubeVideo(
                video_id="vid_018",
                title="Remote Work Excellence: Setup and Mindset",
                channel="Remote Pro",
                duration_minutes=27,
                category=VideoCategory.WORK_LIFE_BALANCE,
                keywords=["remote work", "productivity", "focus", "setup"],
                description="Optimizing your remote work environment and mindset for productivity.",
                view_count=512000
            ),
            
            YouTubeVideo(
                video_id="vid_019",
                title="Design Thinking: Innovation Framework",
                channel="Innovation Lab",
                duration_minutes=30,
                category=VideoCategory.INNOVATION,
                keywords=["design thinking", "innovation", "problem solving", "creativity"],
                description="How to apply design thinking principles to solve complex problems.",
                view_count=389000
            ),
            YouTubeVideo(
                video_id="vid_020",
                title="Building an Innovation Culture",
                channel="Culture First",
                duration_minutes=25,
                category=VideoCategory.INNOVATION,
                keywords=["innovation", "culture", "team", "creativity"],
                description="Creating organizational culture that fosters continuous innovation.",
                view_count=287000
            ),
            
            YouTubeVideo(
                video_id="vid_021",
                title="Investment Basics for Professionals",
                channel="Money Matters",
                duration_minutes=35,
                category=VideoCategory.FINANCIAL_PLANNING,
                keywords=["investment", "finance", "stocks", "portfolio"],
                description="Building wealth through smart investment strategies.",
                view_count=678000
            ),
            YouTubeVideo(
                video_id="vid_022",
                title="Financial Independence: The Early Exit Strategy",
                channel="FIRE Movement",
                duration_minutes=28,
                category=VideoCategory.FINANCIAL_PLANNING,
                keywords=["financial independence", "FIRE", "retirement", "planning"],
                description="Path to financial independence and early retirement.",
                view_count=834000
            ),
        ]
        
        return {v.video_id: v for v in videos}
    
    def get_personalized_recommendations(
        self,
        user_id: str,
        mentor_expertise: List[str],
        user_goals: Optional[List[str]] = None,
        limit: int = 8
    ) -> List[YouTubeVideo]:
        """
        Get personalized YouTube video recommendations based on:
        - Mentor's expertise areas
        - User's career goals
        - User's skill gaps
        - User's learning history
        """
        cache_key = f"{user_id}_mentor_rec"
        
        if cache_key in self.recommendation_cache:
            cached_videos = self.recommendation_cache[cache_key]
            if cached_videos:
                return cached_videos[:limit]
        
        recommendations: List[YouTubeVideo] = []
        keywords_to_match = set(mentor_expertise or [])
        
        if user_goals:
            keywords_to_match.update(user_goals)
        if user_id in self.user_profiles:
            keywords_to_match.update(self.user_profiles[user_id].skill_gaps)
            keywords_to_match.update(self.user_profiles[user_id].interests)
        video_scores: Dict[str, tuple] = {}
        
        for video in self.videos_db.values():
            score = self._calculate_relevance_score(
                video,
                keywords_to_match,
                user_id
            )
            video_scores[video.video_id] = (video, score)
        
        sorted_videos = sorted(
            video_scores.values(),
            key=lambda x: x[1],
            reverse=True
        )
        
        recommendations = [video for video, _ in sorted_videos[:limit]]
        
        self.recommendation_cache[cache_key] = recommendations
        
        return recommendations
    
    def get_mentor_specialty_videos(
        self,
        mentor_expertise: List[str],
        mentor_industry: str,
        limit: int = 6
    ) -> List[YouTubeVideo]:
        """Get videos specifically related to mentor's expertise"""
        keywords_to_match = set(mentor_expertise)
        keywords_to_match.add(mentor_industry.lower())
        
        video_scores: Dict[str, tuple] = {}
        
        for video in self.videos_db.values():
            score = 0.0
            matched_keywords = set(video.keywords) & keywords_to_match
            
            if matched_keywords:
                score = len(matched_keywords) / len(set(video.keywords))
                score += 0.2
            
            if score > 0:
                video_scores[video.video_id] = (video, score)
        
        sorted_videos = sorted(
            video_scores.values(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [video for video, _ in sorted_videos[:limit]]
    
    def get_skill_gap_videos(
        self,
        current_skills: List[str],
        target_skills: List[str],
        limit: int = 6
    ) -> List[YouTubeVideo]:
        """Get videos to help bridge skill gaps"""
        skill_gaps = set(target_skills) - set(current_skills)
        keywords_to_match = skill_gaps
        
        video_scores: Dict[str, tuple] = {}
        
        for video in self.videos_db.values():
            matched_keywords = set(video.keywords) & keywords_to_match
            
            if matched_keywords:
                score = len(matched_keywords) / max(len(video.keywords), 1)
                video_scores[video.video_id] = (video, score)
        
        sorted_videos = sorted(
            video_scores.values(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [video for video, _ in sorted_videos[:limit]]
    
    def get_industry_trending_videos(
        self,
        industry: str,
        limit: int = 6
    ) -> List[YouTubeVideo]:
        """Get trending videos in a specific industry"""
        industry_keywords = {industry.lower(), "industry", "trends", "future"}
        
        video_scores: Dict[str, tuple] = {}
        
        for video in self.videos_db.values():
            if video.category in [VideoCategory.INDUSTRY_INSIGHTS, VideoCategory.INNOVATION]:
                matched_keywords = set(video.keywords) & industry_keywords
                score = (
                    (len(matched_keywords) * 0.5) +
                    (video.view_count / 1000000) * 0.5
                )
                video_scores[video.video_id] = (video, score)
        
        sorted_videos = sorted(
            video_scores.values(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [video for video, _ in sorted_videos[:limit]]
    
    def get_learning_path_videos(
        self,
        career_goal: str,
        current_level: str = "beginner",
        limit: int = 10
    ) -> List[YouTubeVideo]:
        """Get structured learning path videos for a career goal"""
        goal_keywords = set(career_goal.lower().split())
        
        video_scores: Dict[str, tuple] = {}
        
        for video in self.videos_db.values():
            matched_keywords = set(video.keywords) & goal_keywords
            
            if matched_keywords or any(k in video.title.lower() for k in goal_keywords):
                score = len(matched_keywords)
                
                if current_level == "beginner" and video.duration_minutes < 30:
                    score += 1.0
                elif current_level == "intermediate" and 20 < video.duration_minutes < 45:
                    score += 1.0
                elif current_level == "advanced" and video.duration_minutes > 30:
                    score += 1.0
                
                video_scores[video.video_id] = (video, score)
        
        sorted_videos = sorted(
            video_scores.values(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [video for video, _ in sorted_videos[:limit]]
    
    def _calculate_relevance_score(
        self,
        video: YouTubeVideo,
        keywords: set,
        user_id: str
    ) -> float:
        """Calculate relevance score for a video"""
        score = 0.0
        
        matched_keywords = set(video.keywords) & keywords
        if matched_keywords:
            keyword_score = (len(matched_keywords) / max(len(keywords), 1))
            score += keyword_score * 0.4
        
        popularity_score = min(video.view_count / 1000000, 1.0)
        score += popularity_score * 0.2
        
        if user_id in self.user_profiles:
            if video.video_id not in self.user_profiles[user_id].watched_videos:
                score += 0.15
        else:
            score += 0.15
        
        preferred_categories = [
            VideoCategory.CAREER_DEVELOPMENT,
            VideoCategory.TECHNICAL_SKILLS,
            VideoCategory.LEADERSHIP
        ]
        if video.category in preferred_categories:
            score += 0.25 * (1 / len(preferred_categories))
        
        return score
    
    def mark_video_watched(self, user_id: str, video_id: str) -> bool:
        """Mark a video as watched by user"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                expertise=[],
                industry="",
                years_experience=0
            )
        
        if video_id in self.videos_db:
            self.videos_db[video_id].watched = True
            self.user_profiles[user_id].watched_videos.append(video_id)
            # Invalidate cache
            self._invalidate_user_cache(user_id)
            return True
        
        return False
    
    def save_video_for_later(self, user_id: str, video_id: str) -> bool:
        """Save video to watch later"""
        if video_id in self.videos_db:
            self.videos_db[video_id].watch_later = True
            return True
        return False
    
    def rate_video(self, user_id: str, video_id: str, rating: float) -> bool:
        """Rate a video (1-5 stars)"""
        if video_id in self.videos_db and 1.0 <= rating <= 5.0:
            self.videos_db[video_id].rating = rating
            self._invalidate_user_cache(user_id)
            return True
        return False
    
    def get_watch_history(self, user_id: str, limit: int = 10) -> List[YouTubeVideo]:
        """Get user's watch history"""
        if user_id not in self.user_profiles:
            return []
        
        watched_ids = self.user_profiles[user_id].watched_videos
        watched_videos = [
            self.videos_db[vid] for vid in watched_ids
            if vid in self.videos_db
        ]
        
        return watched_videos[-limit:]
    
    def get_watch_later_list(self, user_id: str) -> List[YouTubeVideo]:
        """Get user's watch later list"""
        if user_id not in self.user_profiles:
            return []
        return [
            v for v in self.videos_db.values()
            if v.watch_later
        ]
    
    def _invalidate_user_cache(self, user_id: str):
        """Invalidate cache for a user"""
        cache_key = f"{user_id}_mentor_rec"
        if cache_key in self.recommendation_cache:
            del self.recommendation_cache[cache_key]
    
    def get_video_by_id(self, video_id: str) -> Optional[YouTubeVideo]:
        """Get a specific video by ID"""
        return self.videos_db.get(video_id)
    
    def get_all_categories(self) -> List[str]:
        """Get all available video categories"""
        return [cat.value for cat in VideoCategory]
    
    def search_videos(self, query: str, limit: int = 10) -> List[YouTubeVideo]:
        """Search videos by keyword or title"""
        query_lower = query.lower()
        results: List[YouTubeVideo] = []
        
        for video in self.videos_db.values():
            if (query_lower in video.title.lower() or
                query_lower in video.description.lower() or
                any(query_lower in kw.lower() for kw in video.keywords)):
                results.append(video)
        
        return results[:limit]

youtube_recommendation_service = YouTubeRecommendationService()
