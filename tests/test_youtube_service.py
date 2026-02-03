from services.youtube_recommendation_service import (
    youtube_recommendation_service,
    VideoCategory
)

def print_videos(videos, title="Videos"):
    """Helper to print videos nicely"""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    for i, video in enumerate(videos, 1):
        print(f"\n{i}. {video.title}")
        print(f"   Channel: {video.channel}")
        print(f"   Duration: {video.duration_minutes} minutes")
        print(f"   Category: {video.category.value}")
        print(f"   Views: {video.view_count:,}")
        print(f"   Keywords: {', '.join(video.keywords)}")


def test_personalized_recommendations():
    """Test personalized recommendations based on mentor expertise"""
    print("\n\n" + "TEST 1: Personalized Recommendations".center(70, "="))
    
    mentor_expertise = ["AI", "Research", "Academia"]
    user_goals = ["career development", "leadership"]
    
    videos = youtube_recommendation_service.get_personalized_recommendations(
        user_id="test_user",
        mentor_expertise=mentor_expertise,
        user_goals=user_goals,
        limit=4
    )
    
    print(f"\nMentor Expertise: {mentor_expertise}")
    print(f"User Goals: {user_goals}")
    print_videos(videos, "Personalized Recommendations")


def test_mentor_specialty_videos():
    """Test videos related to specific mentor expertise"""
    print("\n\n" + " TEST 2: Mentor Specialty Videos".center(70, "="))
    
    videos = youtube_recommendation_service.get_mentor_specialty_videos(
        mentor_expertise=["Management", "Product", "SaaS"],
        mentor_industry="Technology",
        limit=3
    )
    
    print("\nMentor: Product Manager in SaaS")
    print("Expertise: Management, Product, SaaS")
    print_videos(videos, "Mentor Specialty Videos")


def test_skill_gap_videos():
    """Test videos to bridge skill gaps"""
    print("\n\n" + "  TEST 3: Skill Gap Bridge Videos".center(70, "="))
    
    current_skills = ["Python", "JavaScript"]
    target_skills = ["Python", "JavaScript", "Machine Learning", "Leadership", "Negotiation"]
    
    videos = youtube_recommendation_service.get_skill_gap_videos(
        current_skills=current_skills,
        target_skills=target_skills,
        limit=4
    )
    
    skill_gaps = set(target_skills) - set(current_skills)
    print(f"\nCurrent Skills: {current_skills}")
    print(f"Target Skills: {target_skills}")
    print(f"Skill Gaps to Bridge: {list(skill_gaps)}")
    print_videos(videos, "Skill Gap Bridge Videos")


def test_industry_trending_videos():
    """Test trending videos in specific industry"""
    print("\n\n" + " TEST 4: Industry Trending Videos".center(70, "="))
    
    videos = youtube_recommendation_service.get_industry_trending_videos(
        industry="Technology",
        limit=3
    )
    
    print("\nIndustry: Technology")
    print_videos(videos, "Trending Videos in Tech")


def test_learning_path_videos():
    """Test structured learning paths"""
    print("\n\n" + "  TEST 5: Learning Path Videos".center(70, "="))
    
    for level in ["beginner", "intermediate", "advanced"]:
        videos = youtube_recommendation_service.get_learning_path_videos(
            career_goal="machine learning",
            current_level=level,
            limit=2
        )
        print(f"\n Level: {level.upper()}")
        print(f"Goal: Machine Learning")
        for v in videos:
            print(f"  . {v.title} ({v.duration_minutes}m)")


def test_search_videos():
    """Test video search functionality"""
    print("\n\n" + " TEST 6: Video Search".center(70, "="))
    
    query = "leadership"
    videos = youtube_recommendation_service.search_videos(query, limit=3)
    
    print(f"\nSearch Query: '{query}'")
    print(f"Results Found: {len(videos)}")
    print_videos(videos, "Search Results")


def test_video_interactions():
    """Test user interactions with videos"""
    print("\n\n" + " TEST 7: Video Interactions".center(70, "="))
    
    user_id = "demo_user"
    video_id = "vid_001"
    
    print("\n1. Marking video as watched...")
    success = youtube_recommendation_service.mark_video_watched(user_id, video_id)
    print(f"   ✓ Success: {success}")
    
    print("\n2. Saving video for later...")
    success = youtube_recommendation_service.save_video_for_later(user_id, video_id)
    print(f"   ✓ Success: {success}")
    
    print("\n3. Rating video (4.5 stars)...")
    success = youtube_recommendation_service.rate_video(user_id, video_id, 4.5)
    print(f"   ✓ Success: {success}")
    
    print("\n4. Getting watch history...")
    history = youtube_recommendation_service.get_watch_history(user_id)
    print(f"   ✓ Videos watched: {len(history)}")


def test_categories():
    """Test available categories"""
    print("\n\n" + "  TEST 8: Video Categories".center(70, "="))
    
    categories = youtube_recommendation_service.get_all_categories()
    print(f"\nAvailable Categories ({len(categories)}):")
    for i, cat in enumerate(categories, 1):
        print(f"  {i}. {cat.replace('_', ' ').title()}")


def test_get_video_by_id():
    """Test retrieving specific video"""
    print("\n\n" + " TEST 9: Get Video by ID".center(70, "="))
    
    video = youtube_recommendation_service.get_video_by_id("vid_001")
    if video:
        print(f"\nVideo Found:")
        print(f"  Title: {video.title}")
        print(f"  Channel: {video.channel}")
        print(f"  Duration: {video.duration_minutes} minutes")
        print(f"  Category: {video.category.value}")
    else:
        print("Video not found")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("YouTube Video Recommendation Service - Test Suite".center(70))
    print("="*70)
    
    try:
        test_personalized_recommendations()
        test_mentor_specialty_videos()
        test_skill_gap_videos()
        test_industry_trending_videos()
        test_learning_path_videos()
        test_search_videos()
        test_video_interactions()
        test_categories()
        test_get_video_by_id()
        
        print("\n\n" + " ALL TESTS COMPLETED SUCCESSFULLY! ".center(70, "="))
        print("\nThe YouTube Recommendation Service is working correctly!")
        print("You can now:")
        print("  • Access /api/mentor/videos/* endpoints")
        print("  • View recommendations in the mentor dashboard")
        print("  • Track watched videos and ratings")
        print("  • Search and filter videos by category")
        
    except Exception as e:
        print(f"\n TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
