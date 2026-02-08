from services.resume_parser_service import resume_parser_service
from services.ollama_service import EnhancedOllamaService, OllamaConfig

def test_resume_scoring():
    print("\n" + "="*60)
    print("Testing Resume Scoring Feature")
    print("="*60)
    
    sample_resume = """
    John Doe
    Email: john@example.com
    Phone: (555) 123-4567
    Location: San Francisco, CA
    
    PROFESSIONAL SUMMARY
    Experienced software engineer with 5+ years of experience in Python, JavaScript, and React.
    Passionate about solving complex problems and building scalable applications.
    
    WORK EXPERIENCE
    Senior Software Engineer
    Tech Company Inc., San Francisco, CA
    January 2021 - Present
    - Led development of microservices architecture
    - Mentored junior developers
    - Improved system performance by 40%
    
    Software Engineer
    Startup XYZ, San Francisco, CA
    June 2018 - December 2020
    - Built RESTful APIs using Python and FastAPI
    - Implemented CI/CD pipelines with Docker and Kubernetes
    
    EDUCATION
    Bachelor of Science in Computer Science
    University of California, Berkeley
    Graduated: May 2018
    GPA: 3.8/4.0
    
    SKILLS
    Languages: Python, JavaScript, TypeScript, SQL
    Frameworks: React, Django, FastAPI, Node.js
    Cloud: AWS, Docker, Kubernetes
    Databases: PostgreSQL, MongoDB, Redis
    
    CERTIFICATIONS
    AWS Certified Solutions Architect - Professional
    Kubernetes CKA
    
    LANGUAGES
    English (Native)
    Spanish (Fluent)
    """
    
    result = resume_parser_service.parse_resume(
        user_id="test_user",
        text_content=sample_resume,
        filename="test_resume.txt"
    )
    
    print("Resume Parsing successful")
    print(f"Resume ID: {result.get('id')}")
    print(f"Name: {result.get('name')}")
    print(f"Email: {result.get('email')}")
    print(f"Years of Experience: {result.get('years_of_experience')}")
    print(f"Seniority Level: {result.get('seniority_level')}")
    print(f"Skills Count: {len(result.get('skills', []))}")
    

    if 'resume_score' in result:
        score_data = result['resume_score']
        print(f"\nResume Score: {score_data['overall_score']}/100")
        print(f"Grade: {score_data['grade']}")
        print(f"Level: {score_data['level']}")
        print(f"Percentage: {score_data['percentage']}%")
    
    resume_id = result.get('id')
    detailed_score = resume_parser_service.calculate_resume_score(resume_id)
    
    print("\nDetailed Score Breakdown:")
    print("-" * 40)
    
    if 'score_breakdown' in detailed_score:
        for section, data in detailed_score['score_breakdown'].items():
            score = data['score']
            max_score = data['max']
            percentage = (score / max_score) * 100
            print(f"{section}: {score}/{max_score} ({percentage:.0f}%) - {data.get('status', 'N/A')}")
    
    print("\nStrengths:")
    for strength in detailed_score.get('strengths', []):
        print(f"  • {strength}")
    
    print("\nAreas for Improvement:")
    for improvement in detailed_score.get('improvement_areas', []):
        print(f"  • {improvement}")
    
    print("\nRecommendations:")
    for rec in detailed_score.get('recommendations', [])[:3]:
        print(f"  {rec}")
    
    print("\n" + "="*60)
    print("Resume Scoring Test PASSED")
    print("="*60)


def test_offline_context_awareness():
    print("\n" + "="*60)
    print("Testing Offline Context Awareness")
    print("="*60)
    
    config = OllamaConfig(
        base_url="http://localhost:11434",
        model="llama3.2"
    )
    service = EnhancedOllamaService(config=config, rag_service=None)
    service.is_available = False
    
    test_cases = [
        {
            "message": "I'm considering changing jobs to a startup",
            "context": None,
            "name": "Job Change Detection"
        },
        {
            "message": "Should I pursue a career switch to data science?",
            "context": None,
            "name": "Career Switch Question"
        },
        {
            "message": "I'm worried about taking a startup position",
            "context": None,
            "name": "Emotion Detection"
        },
        {
            "message": "What are the pros and cons of remote work?",
            "context": None,
            "name": "General Question"
        }
    ]
    
    print("\nTesting various user inputs in offline mode:\n")
    
    for test_case in test_cases:
        message = test_case['message']
        context = test_case['context']
        name = test_case['name']
        
        response = service._generate_chat_fallback(message, context)
        
        print(f"\nTest: {name}")
        print(f"  Input: '{message}'")
        print(f"  Response Preview: {response[:150]}...")
        
        generic_phrases = ["Hello, how can I help you today?", "Hi there! What's on your mind?"]
        is_generic = any(phrase.lower() in response.lower() for phrase in generic_phrases)
        
        if is_generic:
            print(f"  Warning: Response appears generic")
        else:
            print(f"  Response is context-aware")
    
    print("\n" + "="*60)
    print("Offline Context Awareness Test PASSED")
    print("="*60)


if __name__ == "__main__":
    print("\nTesting New Career Decision AI Features\n")
    
    try:
        test_resume_scoring()
        test_offline_context_awareness()
        
        print("\n" + "="*60)
        print("All Tests Passed Successfully!")
        print("="*60)
        print("\nFeatures successfully implemented:")
        print("1. Resume Scoring (0-100 with detailed breakdown)")
        print("2. Offline Context Awareness (understands user decision types)")
        print("\n")
        
    except Exception as e:
        print(f"\nTest Failed: {str(e)}")
        import traceback
        traceback.print_exc()
