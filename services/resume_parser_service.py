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
        resume_id = hashlib.sha256(
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

    def calculate_resume_score(self, resume_id: str) -> Dict[str, Any]:
        """Calculate a detailed resume score with breakdown and recommendations"""
        parsed = self.parsed_resumes.get(resume_id)
        if not parsed:
            return {"error": "Resume not found"}

        score_data = self._score_resume(parsed)
        breakdown = score_data["score_breakdown"]

        strengths = []
        improvement_areas = []
        recommendations = []

        for section, data in breakdown.items():
            max_score = data["max"]
            if max_score <= 0:
                continue
            percent = (data["score"] / max_score) * 100
            if percent >= 80:
                strengths.append(f"Strong {section.lower()} section")
            elif percent < 50:
                improvement_areas.append(f"Improve {section.lower()} details")

        if not parsed.summary:
            recommendations.append("Add a concise professional summary highlighting your impact.")
        if len(parsed.skills) < 10:
            recommendations.append("Expand your skills section with relevant tools and frameworks.")
        if not parsed.education:
            recommendations.append("Include education details to strengthen credibility.")
        if not parsed.certifications:
            recommendations.append("Add certifications if you have any relevant to your role.")
        if not parsed.experiences:
            recommendations.append("Add work experience with measurable achievements.")

        return {
            "resume_id": parsed.id,
            "overall_score": score_data["overall_score"],
            "grade": score_data["grade"],
            "level": score_data["level"],
            "percentage": score_data["percentage"],
            "score_breakdown": breakdown,
            "strengths": strengths,
            "improvement_areas": improvement_areas,
            "recommendations": recommendations
        }

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
        score_data = self._score_resume(parsed)
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
            "resume_score": {
                "overall_score": score_data["overall_score"],
                "grade": score_data["grade"],
                "level": score_data["level"],
                "percentage": score_data["percentage"]
            },
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

    def _score_resume(self, parsed: ParsedResume) -> Dict[str, Any]:
        score_breakdown = {
            "Contact": {"score": 0, "max": 10},
            "Summary": {"score": 0, "max": 10},
            "Experience": {"score": 0, "max": 30},
            "Education": {"score": 0, "max": 15},
            "Skills": {"score": 0, "max": 20},
            "Certifications": {"score": 0, "max": 10},
            "Languages": {"score": 0, "max": 5}
        }

        contact_score = 0
        contact_score += 2.5 if parsed.name else 0
        contact_score += 2.5 if parsed.email else 0
        contact_score += 2.5 if parsed.phone else 0
        contact_score += 2.5 if parsed.location else 0
        score_breakdown["Contact"]["score"] = contact_score

        summary_score = 10 if len(parsed.summary) >= 40 else (5 if parsed.summary else 0)
        score_breakdown["Summary"]["score"] = summary_score

        exp_score = min(30.0, parsed.years_of_experience * 3.0)
        if parsed.experiences and exp_score < 5:
            exp_score = 5.0
        score_breakdown["Experience"]["score"] = round(exp_score, 1)

        edu_score = 10 if parsed.education else 0
        if parsed.education and any(e.degree for e in parsed.education):
            edu_score = 15
        score_breakdown["Education"]["score"] = edu_score

        skills_score = min(20.0, len(parsed.skills) * 1.5)
        score_breakdown["Skills"]["score"] = round(skills_score, 1)

        cert_score = min(10.0, len(parsed.certifications) * 5.0)
        score_breakdown["Certifications"]["score"] = cert_score

        lang_score = min(5.0, len(parsed.languages) * 2.0)
        score_breakdown["Languages"]["score"] = lang_score

        overall_score = sum(section["score"] for section in score_breakdown.values())
        overall_score = round(min(100.0, overall_score), 1)

        if overall_score >= 90:
            grade = "A"
        elif overall_score >= 80:
            grade = "B"
        elif overall_score >= 70:
            grade = "C"
        elif overall_score >= 60:
            grade = "D"
        else:
            grade = "F"

        level = parsed.seniority_level or "Unspecified"

        return {
            "overall_score": overall_score,
            "grade": grade,
            "level": level,
            "percentage": overall_score,
            "score_breakdown": score_breakdown
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

        # Normalize role (allow underscores / hyphens from UI values)
        normalized = target_role.replace('_', ' ').replace('-', ' ').lower().strip()

        role_skills = {
            "software engineer": ["Python", "Javascript", "SQL", "Git", "Docker"],
            "backend developer": ["Python", "Java", "SQL", "APIs", "Docker"],
            "frontend developer": ["Javascript", "React", "CSS", "HTML", "TypeScript"],
            "full stack developer": ["Javascript", "Node.js", "React", "SQL", "Docker"],
            "mobile developer": ["Swift", "Kotlin", "React Native", "Flutter", "Mobile UI"],
            "embedded systems engineer": ["C", "C++", "RTOS", "Microcontrollers", "Firmware"],
            "game developer": ["Unity", "C#", "C++", "3D Math", "Game Design"],
            "devops engineer": ["Docker", "Kubernetes", "AWS", "CI/CD", "Linux"],
            "site reliability engineer": ["Kubernetes", "Monitoring", "Incident Response", "Linux", "Automation"],
            "cloud architect": ["AWS", "Azure", "GCP", "Terraform", "Microservices"],
            "security engineer": ["Penetration Testing", "SIEM", "Network Security", "Cryptography", "Compliance"],
            "qa engineer": ["Testing", "Selenium", "Automation", "CI/CD", "Bug Tracking"],
            "solutions architect": ["System Design", "Cloud Architecture", "APIs", "Integration", "Communication"],
            "blockchain developer": ["Solidity", "Smart Contracts", "Web3", "Ethereum", "Cryptography"],
            "data scientist": ["Python", "SQL", "Machine Learning", "Statistics", "TensorFlow"],
            "data analyst": ["SQL", "Excel", "Python", "Tableau", "Statistics"],
            "data engineer": ["SQL", "Spark", "Airflow", "ETL", "Data Modeling"],
            "machine learning engineer": ["Python", "TensorFlow", "PyTorch", "ML Ops", "Statistics"],
            "ai researcher": ["Deep Learning", "Research Methods", "Python", "Mathematics", "NLP"],
            "business intelligence analyst": ["SQL", "Tableau", "Power BI", "Data Modeling", "Reporting"],
            "nlp engineer": ["NLP", "Python", "Transformers", "Text Processing", "Deep Learning"],
            "computer vision engineer": ["OpenCV", "PyTorch", "Image Processing", "Deep Learning", "Python"],
            "ux designer": ["UX", "Prototyping", "Figma", "User Research", "Interaction Design"],
            "ui designer": ["Visual Design", "Figma", "Typography", "Color Theory", "Design Systems"],
            "product designer": ["UX", "UI", "Prototyping", "User Research", "Figma"],
            "graphic designer": ["Adobe Photoshop", "Illustrator", "Typography", "Branding", "Print Design"],
            "motion designer": ["After Effects", "Animation", "Cinema 4D", "Motion Graphics", "Video Editing"],
            "ux researcher": ["User Research", "Usability Testing", "Data Analysis", "Survey Design", "Interviewing"],
            "content designer": ["UX Writing", "Content Strategy", "Information Architecture", "Copywriting", "A/B Testing"],
            "creative director": ["Brand Strategy", "Art Direction", "Team Leadership", "Visual Design", "Storytelling"],
            "product manager": ["Agile", "Scrum", "Communication", "Analytics", "Leadership"],
            "technical program manager": ["Program Management", "Stakeholder Management", "Roadmapping", "Risk Management", "Communication"],
            "project manager": ["Project Planning", "Risk Management", "Agile", "Stakeholder Communication", "Budgeting"],
            "engineering manager": ["Technical Leadership", "People Management", "Agile", "Mentoring", "System Design"],
            "scrum master": ["Scrum", "Agile", "Facilitation", "Coaching", "Kanban"],
            "cto": ["Technical Strategy", "Architecture", "Leadership", "Business Acumen", "Innovation"],
            "product owner": ["Backlog Management", "Agile", "Stakeholder Communication", "Prioritization", "User Stories"],
            "digital marketing manager": ["SEO", "SEM", "Google Analytics", "Social Media", "Content Marketing"],
            "seo specialist": ["SEO", "Google Analytics", "Keyword Research", "Content Optimization", "Technical SEO"],
            "content marketing manager": ["Content Strategy", "Copywriting", "SEO", "Analytics", "Social Media"],
            "growth hacker": ["A/B Testing", "Analytics", "Funnel Optimization", "Marketing Automation", "Data-Driven"],
            "social media manager": ["Social Media Strategy", "Content Creation", "Analytics", "Community Management", "Advertising"],
            "brand manager": ["Brand Strategy", "Market Research", "Positioning", "Campaign Management", "Communication"],
            "marketing analyst": ["Google Analytics", "SQL", "Excel", "A/B Testing", "Reporting"],
            "email marketing specialist": ["Email Automation", "Copywriting", "A/B Testing", "Segmentation", "Analytics"],
            "sales representative": ["CRM", "Negotiation", "Communication", "Lead Generation", "Closing"],
            "account executive": ["Sales Strategy", "CRM", "Negotiation", "Pipeline Management", "Presentation"],
            "business development manager": ["Strategic Partnerships", "Market Analysis", "Negotiation", "Communication", "Revenue Growth"],
            "sales engineer": ["Technical Demos", "Solution Architecture", "Communication", "CRM", "Product Knowledge"],
            "customer success manager": ["Customer Onboarding", "Relationship Management", "Churn Analysis", "CRM", "Communication"],
            "account manager": ["Relationship Management", "Upselling", "CRM", "Communication", "Account Planning"],
            "financial analyst": ["Financial Modeling", "Excel", "Valuation", "Forecasting", "SQL"],
            "accountant": ["GAAP", "Tax Preparation", "Bookkeeping", "Auditing", "Excel"],
            "investment banker": ["Financial Modeling", "Valuation", "M&A", "Due Diligence", "Excel"],
            "financial advisor": ["Financial Planning", "Investment Management", "Risk Assessment", "Communication", "Compliance"],
            "risk analyst": ["Risk Modeling", "Statistics", "Python", "Compliance", "Financial Analysis"],
            "controller": ["Financial Reporting", "Budgeting", "GAAP", "Team Management", "Audit"],
            "hr manager": ["Talent Management", "Employee Relations", "Recruitment", "Labor Law", "HRIS"],
            "recruiter": ["Sourcing", "ATS", "Interviewing", "Employer Branding", "Negotiation"],
            "hr business partner": ["Strategic HR", "Change Management", "Employee Engagement", "Data Analysis", "Leadership Development"],
            "compensation analyst": ["Compensation Analysis", "Benchmarking", "Excel", "HRIS", "Market Research"],
            "training specialist": ["Instructional Design", "LMS", "Facilitation", "Needs Assessment", "E-Learning"],
            "operations manager": ["Process Improvement", "Budgeting", "Team Management", "KPIs", "Supply Chain"],
            "supply chain manager": ["Logistics", "Procurement", "ERP", "Inventory Management", "Vendor Management"],
            "management consultant": ["Strategy", "Problem Solving", "Data Analysis", "Presentation", "Stakeholder Management"],
            "strategy analyst": ["Strategic Planning", "Market Research", "Financial Modeling", "Data Analysis", "Communication"],
            "chief of staff": ["Executive Communication", "Project Management", "Strategic Planning", "Cross-functional Leadership", "Operations"],
            "doctor": ["Clinical Diagnosis", "Patient Care", "Medical Knowledge", "Communication", "Evidence-Based Medicine"],
            "nurse": ["Patient Assessment", "Clinical Skills", "Medication Administration", "Communication", "Critical Thinking"],
            "pharmacist": ["Pharmacology", "Drug Interactions", "Patient Counseling", "Regulatory Compliance", "Healthcare"],
            "biomedical engineer": ["Medical Devices", "Biocompatibility", "Signal Processing", "CAD", "Regulatory"],
            "clinical researcher": ["Clinical Trials", "Statistical Analysis", "GCP", "Protocol Design", "Data Management"],
            "healthcare administrator": ["Healthcare Management", "Compliance", "Budgeting", "Quality Improvement", "Leadership"],
            "lawyer": ["Legal Research", "Litigation", "Contract Drafting", "Negotiation", "Legal Writing"],
            "paralegal": ["Legal Research", "Document Preparation", "Case Management", "Filing", "Legal Writing"],
            "compliance officer": ["Regulatory Compliance", "Risk Assessment", "Policy Development", "Auditing", "Communication"],
            "legal counsel": ["Contract Review", "Legal Advisory", "Corporate Law", "Risk Management", "Negotiation"],
            "teacher": ["Curriculum Design", "Classroom Management", "Assessment", "Communication", "Subject Expertise"],
            "instructional designer": ["E-Learning", "LMS", "Instructional Design Models", "Multimedia", "Assessment Design"],
            "academic advisor": ["Advising", "Student Development", "Program Knowledge", "Communication", "Career Guidance"],
            "education administrator": ["Institutional Management", "Budgeting", "Compliance", "Leadership", "Strategic Planning"],
            "entrepreneur": ["Business Strategy", "Fundraising", "Leadership", "Product Development", "Marketing"],
            "freelancer": ["Self-Management", "Client Communication", "Negotiation", "Specialized Skill", "Marketing"],
            "technical writer": ["Technical Writing", "Documentation", "Information Architecture", "Markdown", "API Documentation"],
            "customer support": ["Communication", "Problem Solving", "CRM", "Empathy", "Ticketing Systems"],
            "real estate agent": ["Market Analysis", "Negotiation", "Client Relations", "Sales", "Property Knowledge"],
            "journalist": ["Writing", "Research", "Interviewing", "Ethics", "Storytelling"],
            "photographer videographer": ["Photography", "Video Editing", "Lighting", "Composition", "Adobe Suite"],
            "designer": ["UX", "Prototyping", "Figma", "User Research", "Interaction Design"]
        }

        target_skills = role_skills.get(normalized, [])
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
