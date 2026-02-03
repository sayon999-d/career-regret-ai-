from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re
import hashlib


class ResumeSection(str, Enum):
    CONTACT = "contact"
    SUMMARY = "summary"
    EXPERIENCE = "experience"
    EDUCATION = "education"
    SKILLS = "skills"
    CERTIFICATIONS = "certifications"
    PROJECTS = "projects"
    LANGUAGES = "languages"
    AWARDS = "awards"


@dataclass
class WorkExperience:
    company: str
    title: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    is_current: bool = False
    description: str = ""
    location: str = ""
    responsibilities: List[str] = field(default_factory=list)


@dataclass
class Education:
    institution: str
    degree: str
    field_of_study: str = ""
    graduation_date: Optional[str] = None
    gpa: Optional[float] = None
    honors: List[str] = field(default_factory=list)


@dataclass
class ParsedResume:
    id: str
    user_id: str
    raw_text: str
    name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    summary: str = ""
    experiences: List[WorkExperience] = field(default_factory=list)
    education: List[Education] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    years_of_experience: float = 0.0
    seniority_level: str = ""
    industry_keywords: List[str] = field(default_factory=list)
    parsed_at: datetime = field(default_factory=datetime.utcnow)
    confidence_score: float = 0.0


class ResumeParserService:
    """
    Parses resumes and extracts structured career data
    """

    SKILL_KEYWORDS = {
        "programming": [
            "python", "javascript", "java", "c++", "c#",
            "typescript", "swift", "kotlin", "php", "scala", "r", "matlab"
        ],
        "frameworks": [
            "react", "angular", "vue", "django", "flask", "fastapi", "spring",
            "express", "nextjs", "rails", "laravel", "node.js", "tensorflow",
            "pytorch", "keras", "pandas", "numpy", "scikit-learn"
        ],
        "databases": [
            "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
            "cassandra", "dynamodb", "oracle", "sqlite", "neo4j"
        ],
        "cloud": [
            "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
            "jenkins", "ci/cd", "devops", "linux", "microservices"
        ],
        "soft_skills": [
            "leadership", "communication", "teamwork", "problem-solving",
            "analytical", "project management", "agile", "scrum", "mentoring"
        ],
        "certifications": [
            "pmp", "aws certified", "azure certified", "gcp certified",
            "cissp", "cka", "ckad", "ceh", "comptia", "ccna", "ccnp"
        ]
    }

    SENIORITY_KEYWORDS = {
        "entry": ["intern", "junior", "entry-level", "associate", "trainee"],
        "mid": ["mid-level", "mid-senior", "software engineer", "developer"],
        "senior": ["senior", "lead", "staff", "principal", "architect"],
        "executive": ["director", "vp", "vice president", "cto", "ceo", "cfo", "head of"]
    }

    def __init__(self):
        self.parsed_resumes: Dict[str, ParsedResume] = {}

    def parse_resume(
        self,
        user_id: str,
        text_content: str,
        filename: str = ""
    ) -> Dict[str, Any]:
        """Parse a resume from text content"""
        resume_id = hashlib.md5(
            f"{user_id}{datetime.utcnow().timestamp()}".encode()
        ).hexdigest()[:12]

        text_lower = text_content.lower()

        name = self._extract_name(text_content)
        email = self._extract_email(text_content)
        phone = self._extract_phone(text_content)
        location = self._extract_location(text_content)
        summary = self._extract_summary(text_content)
        experiences = self._extract_experience(text_content)
        education = self._extract_education(text_content)
        skills = self._extract_skills(text_lower)
        certifications = self._extract_certifications(text_lower)
        languages = self._extract_languages(text_content)

        years_exp = self._calculate_years_experience(experiences)
        seniority = self._determine_seniority(text_lower, experiences, years_exp)
        industry_keywords = self._extract_industry_keywords(text_lower)

        sections_found = sum([
            bool(name), bool(email), bool(experiences),
            bool(education), bool(skills)
        ])
        confidence = min(1.0, sections_found / 5 * 0.8 + 0.2)

        parsed = ParsedResume(
            id=resume_id,
            user_id=user_id,
            raw_text=text_content,
            name=name,
            email=email,
            phone=phone,
            location=location,
            summary=summary,
            experiences=experiences,
            education=education,
            skills=skills,
            certifications=certifications,
            languages=languages,
            years_of_experience=years_exp,
            seniority_level=seniority,
            industry_keywords=industry_keywords,
            confidence_score=confidence
        )

        self.parsed_resumes[resume_id] = parsed

        return self._to_dict(parsed)

    def _extract_name(self, text: str) -> str:
        """Extract candidate name from resume"""
        lines = text.strip().split('\n')
        for line in lines[:5]:
            line = line.strip()
            words = line.split()
            if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w):
                if not any(c.isdigit() for c in line) and '@' not in line:
                    return line
        return ""

    def _extract_email(self, text: str) -> str:
        """Extract email address"""
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        match = re.search(email_pattern, text)
        return match.group(0) if match else ""

    def _extract_phone(self, text: str) -> str:
        """Extract phone number"""
        phone_patterns = [
            r'\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            r'\+\d{1,3}[-.\s]?\d{8,12}'
        ]
        for pattern in phone_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return ""

    def _extract_location(self, text: str) -> str:
        """Extract location from resume"""
        location_patterns = [
            r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*,\s*[A-Z]{2})',
            r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*,\s*[A-Z][a-z]+)',
        ]
        for pattern in location_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return ""

    def _extract_summary(self, text: str) -> str:
        """Extract professional summary"""
        summary_markers = [
            "summary", "professional summary", "profile", "objective",
            "about me", "career objective"
        ]

        text_lower = text.lower()
        for marker in summary_markers:
            idx = text_lower.find(marker)
            if idx != -1:
                start = idx + len(marker)
                lines = text[start:start+500].split('\n')
                summary_lines = []
                for line in lines[1:4]:
                    line = line.strip()
                    if line and len(line) > 20:
                        summary_lines.append(line)
                if summary_lines:
                    return ' '.join(summary_lines)[:300]
        return ""

    def _extract_experience(self, text: str) -> List[WorkExperience]:
        """Extract work experience entries"""
        experiences = []

        exp_markers = ["experience", "work history", "employment", "professional experience"]
        text_lower = text.lower()

        start_idx = -1
        for marker in exp_markers:
            idx = text_lower.find(marker)
            if idx != -1:
                start_idx = idx
                break

        if start_idx == -1:
            return experiences

        end_markers = ["education", "skills", "certifications", "projects"]
        end_idx = len(text)
        for marker in end_markers:
            idx = text_lower.find(marker, start_idx + 20)
            if idx != -1 and idx < end_idx:
                end_idx = idx

        exp_section = text[start_idx:end_idx]

        job_title_patterns = [
            r'((?:Senior|Junior|Lead|Staff|Principal)?\s*(?:Software|Data|Product|Project|Marketing|Sales|HR|Finance)?\s*(?:Engineer|Developer|Manager|Analyst|Designer|Director|Specialist|Consultant))',
        ]

        lines = exp_section.split('\n')
        current_exp = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            date_match = re.search(
                r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{4})\s*[-–to]+\s*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{4}|Present|Current)',
                line, re.I
            )

            if date_match:
                if current_exp:
                    experiences.append(current_exp)

                current_exp = WorkExperience(
                    company="",
                    title="",
                    start_date=date_match.group(1),
                    end_date=date_match.group(2),
                    is_current="present" in date_match.group(2).lower() or "current" in date_match.group(2).lower()
                )
            elif current_exp and len(line) > 10:
                if not current_exp.title:
                    current_exp.title = line[:60]
                elif not current_exp.company:
                    current_exp.company = line[:60]

        if current_exp:
            experiences.append(current_exp)

        return experiences[:10]

    def _extract_education(self, text: str) -> List[Education]:
        """Extract education entries"""
        education = []

        edu_markers = ["education", "academic background", "qualifications"]
        text_lower = text.lower()

        start_idx = -1
        for marker in edu_markers:
            idx = text_lower.find(marker)
            if idx != -1:
                start_idx = idx
                break

        if start_idx == -1:
            return education

        end_markers = ["experience", "skills", "certifications", "projects"]
        end_idx = len(text)
        for marker in end_markers:
            idx = text_lower.find(marker, start_idx + 20)
            if idx != -1 and idx < end_idx:
                end_idx = idx

        edu_section = text[start_idx:end_idx]

        degree_patterns = [
            r"(Bachelor's?|Master's?|Ph\.?D\.?|MBA|B\.?S\.?|M\.?S\.?|B\.?A\.?|M\.?A\.?)",
        ]

        lines = edu_section.split('\n')
        current_edu = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            for pattern in degree_patterns:
                if re.search(pattern, line, re.I):
                    if current_edu:
                        education.append(current_edu)
                    current_edu = Education(
                        institution="",
                        degree=line[:100]
                    )
                    break

        if current_edu:
            education.append(current_edu)

        return education[:5]

    def _extract_skills(self, text_lower: str) -> List[str]:
        """Extract skills from resume"""
        found_skills = set()

        for category, skills in self.SKILL_KEYWORDS.items():
            for skill in skills:
                if skill in text_lower:
                    found_skills.add(skill.title() if len(skill) > 3 else skill.upper())

        return list(found_skills)[:30]

    def _extract_certifications(self, text_lower: str) -> List[str]:
        """Extract certifications from resume"""
        certs = []
        for cert in self.SKILL_KEYWORDS.get("certifications", []):
            if cert in text_lower:
                certs.append(cert.upper())
        return certs[:10]

    def _extract_languages(self, text: str) -> List[str]:
        """Extract spoken languages"""
        common_languages = [
            "English", "Spanish", "French", "German", "Chinese", "Japanese",
            "Korean", "Portuguese", "Italian", "Russian", "Arabic", "Hindi"
        ]
        found = []
        for lang in common_languages:
            if lang.lower() in text.lower():
                found.append(lang)
        return found

    def _calculate_years_experience(self, experiences: List[WorkExperience]) -> float:
        """Calculate total years of experience"""
        if not experiences:
            return 0.0

        return min(20.0, len(experiences) * 2.5)

    def _determine_seniority(
        self,
        text_lower: str,
        experiences: List[WorkExperience],
        years_exp: float
    ) -> str:
        """Determine seniority level"""
        for level, keywords in self.SENIORITY_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                if level == "executive":
                    return "Executive"
                if level == "senior" and years_exp >= 5:
                    return "Senior"

        if years_exp >= 10:
            return "Senior"
        elif years_exp >= 5:
            return "Mid-Level"
        elif years_exp >= 2:
            return "Junior"
        return "Entry-Level"

    def _extract_industry_keywords(self, text_lower: str) -> List[str]:
        """Extract industry-specific keywords"""
        industries = [
            "technology", "healthcare", "finance", "fintech", "e-commerce",
            "saas", "startup", "enterprise", "consulting", "manufacturing",
            "retail", "education", "government", "non-profit", "media"
        ]
        return [ind.title() for ind in industries if ind in text_lower]

    def _to_dict(self, parsed: ParsedResume) -> Dict[str, Any]:
        """Convert ParsedResume to dictionary"""
        return {
            "id": parsed.id,
            "name": parsed.name,
            "email": parsed.email,
            "phone": parsed.phone,
            "location": parsed.location,
            "summary": parsed.summary,
            "skills": parsed.skills,
            "certifications": parsed.certifications,
            "languages": parsed.languages,
            "years_of_experience": parsed.years_of_experience,
            "seniority_level": parsed.seniority_level,
            "industry_keywords": parsed.industry_keywords,
            "experience_count": len(parsed.experiences),
            "education_count": len(parsed.education),
            "confidence_score": round(parsed.confidence_score, 2),
            "parsed_at": parsed.parsed_at.isoformat(),
            "experiences": [{
                "company": exp.company,
                "title": exp.title,
                "start_date": exp.start_date,
                "end_date": exp.end_date,
                "is_current": exp.is_current
            } for exp in parsed.experiences],
            "education": [{
                "institution": edu.institution,
                "degree": edu.degree,
                "field_of_study": edu.field_of_study
            } for edu in parsed.education]
        }

    def get_resume(self, resume_id: str) -> Optional[Dict[str, Any]]:
        """Get a parsed resume by ID"""
        parsed = self.parsed_resumes.get(resume_id)
        return self._to_dict(parsed) if parsed else None

    def get_skill_gaps(
        self,
        resume_id: str,
        target_role: str
    ) -> Dict[str, Any]:
        """Identify skill gaps for a target role"""
        parsed = self.parsed_resumes.get(resume_id)
        if not parsed:
            return {"error": "Resume not found"}

        role_skills = {
            "software engineer": ["Python", "Javascript", "SQL", "Git", "Docker"],
            "data scientist": ["Python", "SQL", "Machine Learning", "Statistics", "TensorFlow"],
            "product manager": ["Agile", "Scrum", "Communication", "Analytics", "Leadership"],
            "devops engineer": ["Docker", "Kubernetes", "AWS", "CI/CD", "Linux"],
            "frontend developer": ["Javascript", "React", "CSS", "HTML", "TypeScript"]
        }

        target_skills = role_skills.get(target_role.lower(), [])
        current_skills = set(s.lower() for s in parsed.skills)

        missing = [s for s in target_skills if s.lower() not in current_skills]
        matching = [s for s in target_skills if s.lower() in current_skills]

        return {
            "target_role": target_role,
            "matching_skills": matching,
            "missing_skills": missing,
            "match_percentage": round(len(matching) / max(1, len(target_skills)) * 100, 1),
            "recommendations": [
                f"Consider learning {skill}" for skill in missing[:3]
            ]
        }


resume_parser_service = ResumeParserService()
