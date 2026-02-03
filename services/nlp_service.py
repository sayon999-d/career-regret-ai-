import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import Counter
import asyncio

@dataclass
class SentimentResult:

    sentiment: str
    confidence: float
    emotions: List[str]
    emotion_scores: Dict[str, float]

@dataclass
class IntentResult:

    intent: str
    confidence: float
    sub_intents: List[str]

@dataclass
class EntityResult:

    job_titles: List[str]
    companies: List[str]
    skills: List[str]
    industries: List[str]
    locations: List[str]
    time_references: List[str]
    monetary_values: List[str]

@dataclass
class TextAnalysisResult:

    word_count: int
    sentence_count: int
    avg_sentence_length: float
    confidence_level: float
    uncertainty_markers: List[str]
    action_words: List[str]
    question_count: int
    exclamation_count: int
    complexity_score: float

@dataclass
class NLPAnalysis:

    sentiment: SentimentResult
    intent: IntentResult
    entities: EntityResult
    text_analysis: TextAnalysisResult
    keywords: List[str]
    summary: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

class SentimentAnalyzer:


    EMOTION_WORDS = {
        'excited': ['excited', 'thrilled', 'eager', 'enthusiastic', 'pumped', 'stoked', 'ecstatic'],
        'anxious': ['anxious', 'worried', 'nervous', 'stressed', 'tense', 'uneasy', 'apprehensive'],
        'hopeful': ['hopeful', 'optimistic', 'positive', 'encouraged', 'promising', 'bright'],
        'fearful': ['afraid', 'scared', 'fearful', 'terrified', 'frightened', 'dreading'],
        'confident': ['confident', 'sure', 'certain', 'convinced', 'assured', 'self-assured'],
        'uncertain': ['uncertain', 'unsure', 'doubtful', 'hesitant', 'indecisive', 'ambivalent'],
        'overwhelmed': ['overwhelmed', 'swamped', 'drowning', 'buried', 'overloaded', 'exhausted'],
        'motivated': ['motivated', 'driven', 'inspired', 'determined', 'ambitious', 'energized'],
        'stressed': ['stressed', 'pressured', 'strained', 'burned out', 'overworked', 'frazzled'],
        'curious': ['curious', 'interested', 'intrigued', 'wondering', 'exploring'],
        'conflicted': ['conflicted', 'torn', 'divided', 'mixed feelings', 'ambivalent'],
        'calm': ['calm', 'peaceful', 'relaxed', 'serene', 'composed', 'at ease'],
        'frustrated': ['frustrated', 'annoyed', 'irritated', 'fed up', 'exasperated'],
        'grateful': ['grateful', 'thankful', 'appreciative', 'blessed'],
        'disappointed': ['disappointed', 'let down', 'discouraged', 'disheartened']
    }

    POSITIVE_WORDS = {
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'happy',
        'excited', 'thrilled', 'opportunity', 'growth', 'success', 'achieve', 'perfect',
        'best', 'awesome', 'incredible', 'outstanding', 'brilliant', 'positive', 'hope',
        'confident', 'grateful', 'blessed', 'fortunate', 'lucky', 'proud', 'satisfied'
    }

    NEGATIVE_WORDS = {
        'bad', 'terrible', 'awful', 'horrible', 'hate', 'sad', 'worried', 'anxious',
        'stressed', 'frustrated', 'disappointed', 'angry', 'upset', 'confused', 'lost',
        'stuck', 'trapped', 'hopeless', 'worthless', 'failure', 'regret', 'mistake',
        'wrong', 'problem', 'issue', 'difficult', 'hard', 'struggle', 'suffering'
    }

    def analyze(self, text: str) -> SentimentResult:

        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)

        positive_count = sum(1 for w in words if w in self.POSITIVE_WORDS)
        negative_count = sum(1 for w in words if w in self.NEGATIVE_WORDS)
        total = len(words) if words else 1

        pos_ratio = positive_count / total
        neg_ratio = negative_count / total

        if pos_ratio > neg_ratio + 0.02:
            sentiment = 'positive'
            confidence = min(0.9, 0.5 + (pos_ratio - neg_ratio) * 5)
        elif neg_ratio > pos_ratio + 0.02:
            sentiment = 'negative'
            confidence = min(0.9, 0.5 + (neg_ratio - pos_ratio) * 5)
        else:
            sentiment = 'neutral'
            confidence = 0.6

        emotion_scores = {}
        detected_emotions = []

        for emotion, keywords in self.EMOTION_WORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                emotion_scores[emotion] = min(1.0, score * 0.3)
                detected_emotions.append(emotion)

        detected_emotions = sorted(detected_emotions,
                                   key=lambda e: emotion_scores.get(e, 0),
                                   reverse=True)[:5]

        return SentimentResult(
            sentiment=sentiment,
            confidence=confidence,
            emotions=detected_emotions,
            emotion_scores=emotion_scores
        )

class IntentClassifier:


    INTENT_PATTERNS = {
        'seeking_advice': [
            r'\bshould i\b', r'\bwhat do you think\b', r'\badvice\b', r'\brecommend\b',
            r'\bhelp me decide\b', r'\bwhat would you\b', r'\bwhich is better\b',
            r'\bpros and cons\b', r'\boption\b', r'\bchoose\b', r'\bpick\b'
        ],
        'expressing_concern': [
            r'\bworried\b', r'\bconcerned\b', r'\bafraid\b', r'\bscared\b',
            r'\bnervous\b', r'\banxious\b', r'\bstressed\b', r'\bproblem\b',
            r'\bissue\b', r'\bdifficult\b', r'\bstruggling\b'
        ],
        'requesting_analysis': [
            r'\banalyze\b', r'\bassess\b', r'\bevaluate\b', r'\breview\b',
            r'\bexamine\b', r'\blook at\b', r'\bcheck\b', r'\bwhat are the\b',
            r'\bhow risky\b', r'\bprediction\b', r'\bforecast\b'
        ],
        'sharing_update': [
            r'\bi decided\b', r'\bi chose\b', r'\bi accepted\b', r'\bi rejected\b',
            r'\bupdate\b', r'\blet you know\b', r'\bjust wanted to say\b',
            r'\bgood news\b', r'\bbad news\b', r'\bhappened\b'
        ],
        'asking_question': [
            r'\bwhat\b.*\?', r'\bhow\b.*\?', r'\bwhy\b.*\?', r'\bwhen\b.*\?',
            r'\bwhere\b.*\?', r'\bwho\b.*\?', r'\bcan you\b', r'\bcould you\b',
            r'\bis it\b.*\?', r'\bare there\b.*\?'
        ],
        'venting': [
            r'\bi hate\b', r'\bso frustrat\b', r'\bi can\'t believe\b',
            r'\bthis is ridiculous\b', r'\bunfair\b', r'\bterrible\b',
            r'\bawful\b', r'\bworst\b'
        ],
        'celebration': [
            r'\bi got\b', r'\bthey offered\b', r'\baccepted\b', r'\bhired\b',
            r'\bexciting\b', r'\bgreat news\b', r'\bhappy to\b', r'\bthrilled\b'
        ],
        'general_chat': []
    }

    def classify(self, text: str) -> IntentResult:

        text_lower = text.lower()

        intent_scores = {}
        for intent, patterns in self.INTENT_PATTERNS.items():
            score = sum(1 for pattern in patterns if re.search(pattern, text_lower))
            if score > 0:
                intent_scores[intent] = score

        if not intent_scores:
            if '?' in text:
                return IntentResult(
                    intent='asking_question',
                    confidence=0.6,
                    sub_intents=[]
                )
            return IntentResult(
                intent='general_chat',
                confidence=0.5,
                sub_intents=[]
            )

        primary_intent = max(intent_scores, key=intent_scores.get)
        max_score = intent_scores[primary_intent]

        sub_intents = [intent for intent, score in intent_scores.items()
                       if intent != primary_intent and score > 0]

        confidence = min(0.95, 0.5 + max_score * 0.15)

        return IntentResult(
            intent=primary_intent,
            confidence=confidence,
            sub_intents=sub_intents[:3]
        )

class EntityExtractor:


    JOB_TITLES = {
        'engineer', 'developer', 'manager', 'director', 'analyst', 'designer',
        'architect', 'consultant', 'specialist', 'coordinator', 'administrator',
        'executive', 'officer', 'lead', 'senior', 'junior', 'intern', 'associate',
        'vp', 'ceo', 'cto', 'cfo', 'founder', 'co-founder', 'president',
        'programmer', 'scientist', 'researcher', 'professor', 'teacher',
        'accountant', 'lawyer', 'doctor', 'nurse', 'therapist', 'coach',
        'engineering', 'management', 'leadership', 'role', 'position'
    }

    JOB_PREFIXES = {
        'software', 'data', 'product', 'project', 'program', 'marketing',
        'sales', 'hr', 'human resources', 'finance', 'operations', 'business',
        'ux', 'ui', 'frontend', 'backend', 'fullstack', 'full-stack', 'devops',
        'cloud', 'security', 'network', 'system', 'machine learning', 'ml', 'ai',
        'qa', 'quality', 'technical', 'engineering', 'general', 'account',
        'customer', 'success', 'support', 'content', 'brand', 'growth'
    }

    COMPOUND_ROLES = [
        'software engineer', 'software developer', 'software engineering',
        'product manager', 'product management', 'product owner',
        'project manager', 'project management', 'program manager',
        'data scientist', 'data analyst', 'data engineer', 'data engineering',
        'machine learning engineer', 'ml engineer', 'ai engineer',
        'ux designer', 'ui designer', 'ux/ui designer', 'product designer',
        'frontend developer', 'backend developer', 'fullstack developer',
        'full-stack developer', 'full stack developer',
        'devops engineer', 'site reliability engineer', 'sre',
        'technical lead', 'tech lead', 'engineering manager',
        'cto', 'ceo', 'cfo', 'coo', 'vp of engineering',
        'director of engineering', 'head of product', 'head of engineering',
        'business analyst', 'systems analyst', 'solutions architect',
        'cloud architect', 'security engineer', 'qa engineer',
        'quality assurance', 'customer success manager',
        'account manager', 'sales manager', 'marketing manager',
        'content writer', 'technical writer', 'copywriter',
        'hr manager', 'human resources', 'recruiter', 'talent acquisition'
    ]

    SKILLS = {
        'python', 'java', 'javascript', 'typescript', 'react', 'angular', 'vue',
        'node', 'sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'aws', 'azure',
        'gcp', 'docker', 'kubernetes', 'git', 'agile', 'scrum', 'leadership',
        'management', 'communication', 'presentation', 'excel', 'powerpoint',
        'photoshop', 'figma', 'sketch', 'analytics', 'marketing', 'seo',
        'machine learning', 'deep learning', 'nlp', 'data analysis', 'statistics'
    }

    INDUSTRIES = {
        'technology', 'tech', 'finance', 'fintech', 'healthcare', 'biotech',
        'pharmaceutical', 'education', 'edtech', 'retail', 'ecommerce', 'e-commerce',
        'manufacturing', 'automotive', 'aerospace', 'defense', 'consulting',
        'real estate', 'hospitality', 'entertainment', 'media', 'advertising',
        'telecommunications', 'energy', 'utilities', 'construction', 'agriculture',
        'government', 'nonprofit', 'startup', 'enterprise'
    }

    TIME_PATTERNS = [
        r'\b\d+\s*(?:year|month|week|day)s?\b',
        r'\b(?:next|last|this)\s+(?:year|month|week)\b',
        r'\bin\s+\d+\s*(?:year|month|week)s?\b',
        r'\bby\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\b',
        r'\b20\d{2}\b'
    ]

    MONEY_PATTERNS = [
        r'\$[\d,]+(?:\.\d{2})?(?:k|K|m|M)?\b',
        r'\b[\d,]+\s*(?:dollars|usd|inr|rupees)\b',
        r'\b\d+(?:\.\d+)?\s*(?:lpa|lakh|lakhs|crore|crores)\b',
        r'\b\d+k\b'
    ]

    def extract(self, text: str) -> EntityResult:

        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)

        job_titles = []
        for role in self.COMPOUND_ROLES:
            if role in text_lower:
                job_titles.append(role)

        for i, word in enumerate(words):
            if word in self.JOB_TITLES:
                if i > 0 and words[i-1] in self.JOB_PREFIXES:
                    compound = f"{words[i-1]} {word}"
                    if compound not in job_titles:
                        job_titles.append(compound)
                elif word not in ['role', 'position', 'management', 'engineering', 'leadership']:
                    if word not in job_titles:
                        job_titles.append(word)

        skills = [w for w in words if w in self.SKILLS]
        for skill in ['machine learning', 'deep learning', 'data analysis', 'full-stack', 'full stack']:
            if skill in text_lower:
                if skill not in skills:
                    skills.append(skill)

        industries = [w for w in words if w in self.INDUSTRIES]

        companies = []
        company_patterns = re.findall(r'\b(?:at|from|with|join(?:ing)?|left|leaving)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)', text)
        companies.extend(company_patterns)

        locations = []
        location_patterns = re.findall(r'\b(?:in|to|from|at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', text)
        locations.extend(location_patterns[:5])

        time_refs = []
        for pattern in self.TIME_PATTERNS:
            matches = re.findall(pattern, text_lower)
            time_refs.extend(matches)

        money_values = []
        for pattern in self.MONEY_PATTERNS:
            matches = re.findall(pattern, text_lower)
            money_values.extend(matches)

        return EntityResult(
            job_titles=list(dict.fromkeys(job_titles))[:5],
            companies=list(set(companies))[:5],
            skills=list(dict.fromkeys(skills))[:10],
            industries=list(set(industries))[:3],
            locations=list(set(locations))[:5],
            time_references=list(set(time_refs))[:5],
            monetary_values=list(set(money_values))[:5]
        )

class TextAnalyzer:


    UNCERTAINTY_MARKERS = [
        'maybe', 'perhaps', 'possibly', 'might', 'could be', 'not sure',
        'i think', 'i guess', 'probably', 'uncertain', 'unsure', "don't know",
        'hard to say', 'it depends', 'on one hand', 'on the other hand'
    ]

    CONFIDENCE_MARKERS = [
        'definitely', 'certainly', 'absolutely', 'sure', 'confident',
        'no doubt', 'clearly', 'obviously', 'i know', 'i believe',
        'i am certain', "i'm sure", 'without question'
    ]

    ACTION_WORDS = [
        'decide', 'choose', 'accept', 'reject', 'start', 'quit', 'leave',
        'join', 'apply', 'interview', 'negotiate', 'ask', 'move', 'change',
        'switch', 'pursue', 'explore', 'consider', 'evaluate', 'compare'
    ]

    def analyze(self, text: str) -> TextAnalysisResult:

        text_lower = text.lower()

        words = re.findall(r'\b\w+\b', text)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        word_count = len(words)
        sentence_count = len(sentences) if sentences else 1
        avg_sentence_length = word_count / sentence_count if sentence_count else 0

        question_count = text.count('?')
        exclamation_count = text.count('!')

        uncertainty_found = [m for m in self.UNCERTAINTY_MARKERS if m in text_lower]
        confidence_found = [m for m in self.CONFIDENCE_MARKERS if m in text_lower]

        uncertainty_score = len(uncertainty_found) * 0.15
        confidence_score = len(confidence_found) * 0.2

        confidence_level = 0.5 + confidence_score - uncertainty_score
        confidence_level = max(0.1, min(0.95, confidence_level))

        action_found = [w for w in self.ACTION_WORDS if w in text_lower]

        unique_words = len(set(w.lower() for w in words))
        vocab_richness = unique_words / word_count if word_count else 0

        complexity_score = min(1.0, (avg_sentence_length / 25) * 0.5 + vocab_richness * 0.5)

        return TextAnalysisResult(
            word_count=word_count,
            sentence_count=sentence_count,
            avg_sentence_length=avg_sentence_length,
            confidence_level=confidence_level,
            uncertainty_markers=uncertainty_found,
            action_words=action_found,
            question_count=question_count,
            exclamation_count=exclamation_count,
            complexity_score=complexity_score
        )

class KeywordExtractor:


    STOP_WORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
        'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'under', 'again', 'further', 'then', 'once',
        'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few',
        'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but',
        'if', 'or', 'because', 'as', 'until', 'while', 'this', 'that', 'these',
        'those', 'am', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'you',
        'your', 'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them',
        'what', 'which', 'who', 'whom', 'about', 'get', 'got', 'like', 'know',
        'think', 'want', 'going', "i'm", "i've", "i'll", "don't", "it's"
    }

    def extract(self, text: str, top_n: int = 10) -> List[str]:

        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

        meaningful_words = [w for w in words if w not in self.STOP_WORDS]

        word_counts = Counter(meaningful_words)

        keywords = [word for word, count in word_counts.most_common(top_n)]

        return keywords

class Summarizer:


    def summarize(self, text: str, max_sentences: int = 3) -> str:

        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s) > 20]

        if not sentences:
            return text[:200] if len(text) > 200 else text

        if len(sentences) <= max_sentences:
            return ' '.join(sentences)

        keyword_extractor = KeywordExtractor()
        keywords = set(keyword_extractor.extract(text, top_n=15))

        scored_sentences = []
        for i, sentence in enumerate(sentences):
            words = set(re.findall(r'\b\w+\b', sentence.lower()))
            keyword_overlap = len(words & keywords)

            position_score = 1.0
            if i == 0:
                position_score = 1.5
            elif i == len(sentences) - 1:
                position_score = 1.2

            score = keyword_overlap * position_score
            scored_sentences.append((score, i, sentence))

        scored_sentences.sort(reverse=True)
        top_indices = sorted([idx for _, idx, _ in scored_sentences[:max_sentences]])

        summary_sentences = [sentences[i] for i in top_indices]
        return ' '.join(summary_sentences)

class NLPService:


    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.text_analyzer = TextAnalyzer()
        self.keyword_extractor = KeywordExtractor()
        self.summarizer = Summarizer()
        self.is_initialized = False

    async def initialize(self):

        self.is_initialized = True
        print("NLP Service initialized (rule-based mode)")

    def analyze(self, text: str, include_summary: bool = False, user_context: Dict = None) -> NLPAnalysis:

        if not text or not text.strip():
            return self._empty_analysis()

        if user_context:
            text = self._normalize_context(text, user_context)

        sentiment = self.sentiment_analyzer.analyze(text)
        intent = self.intent_classifier.classify(text)
        entities = self.entity_extractor.extract(text)
        text_analysis = self.text_analyzer.analyze(text)
        keywords = self.keyword_extractor.extract(text)

        summary = None
        if include_summary and len(text) > 200:
            summary = self.summarizer.summarize(text)

        return NLPAnalysis(
            sentiment=sentiment,
            intent=intent,
            entities=entities,
            text_analysis=text_analysis,
            keywords=keywords,
            summary=summary
        )

    def _normalize_context(self, text: str, context: Dict) -> str:
        """Context normalization for improved understanding"""
        industry = context.get('industry', 'general')
        role = context.get('role', 'person')

        if industry == 'tech':
            text = text.replace('the product', 'the software')
            text = text.replace('users', 'customers')
        elif industry == 'finance':
            text = text.replace('the product', 'the portfolio')

        return text

    def get_tone_adaptation(self, emotion_state: str) -> str:
        """Adapt tone based on emotional state"""
        if emotion_state in ['stress', 'fear', 'anxiety']:
            return "empathetic_and_calm"
        elif emotion_state in ['anger', 'frustration']:
            return "objective_and_concise"
        elif emotion_state in ['joy', 'excitement']:
            return "enthusiastic_and_celebratory"
        else:
            return "professional_and_balanced"

    def analyze_ambiguity(self, text: str) -> Dict[str, Any]:
        """Handle incomplete or noisy inputs by detecting ambiguity"""
        ambiguous_phrases = ["maybe", "sort of", "kinda", "I think", "not sure"]
        found = [p for p in ambiguous_phrases if p in text.lower()]

        return {
            "is_ambiguous": len(found) > 0,
            "confidence_penalty": len(found) * 0.1,
            "clarifying_questions": [f"Can you clarify what you mean by '{f}'?" for f in found]
        }

    def analyze_sentiment(self, text: str) -> SentimentResult:

        return self.sentiment_analyzer.analyze(text)

    def classify_intent(self, text: str) -> IntentResult:

        return self.intent_classifier.classify(text)

    def extract_entities(self, text: str) -> EntityResult:

        return self.entity_extractor.extract(text)

    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:

        return self.keyword_extractor.extract(text, top_n)

    def summarize(self, text: str, max_sentences: int = 3) -> str:

        return self.summarizer.summarize(text, max_sentences)

    def get_emotional_insights(self, emotions: List[str]) -> Dict[str, Any]:

        if not emotions:
            return {'summary': 'No strong emotions detected.', 'recommendations': []}

        positive_emotions = {'excited', 'hopeful', 'confident', 'motivated', 'curious', 'calm', 'grateful'}
        negative_emotions = {'anxious', 'fearful', 'uncertain', 'overwhelmed', 'stressed', 'conflicted', 'frustrated', 'disappointed'}

        pos_count = sum(1 for e in emotions if e in positive_emotions)
        neg_count = sum(1 for e in emotions if e in negative_emotions)

        explanations = []

        if pos_count > neg_count:
            summary = "Your emotional state is predominantly positive, which often correlates with better decision-making."
            explanations.append("High frequency of positive emotion keywords detected.")
            recommendations = [
                "Channel your positive energy into thorough research",
                "Use your confidence to explore bold options",
                "Document your current mindset for future reflection"
            ]
        elif neg_count > pos_count:
            summary = "You're experiencing some challenging emotions. It's important to acknowledge these before deciding."
            explanations.append("Significant presence of stress/anxiety related markers.")
            recommendations = [
                "Take time to identify the root cause of your concerns",
                "Consider discussing your feelings with a trusted person",
                "Avoid making hasty decisions while emotionally charged"
            ]
        else:
            summary = "Your emotions are balanced, showing thoughtful consideration of multiple perspectives."
            explanations.append("Balanced mix of positive and negative emotional indicators.")
            recommendations = [
                "Use this balanced state for objective evaluation",
                "List pros and cons while you're feeling neutral",
                "Trust your analytical process"
            ]

        return {
            'summary': summary,
            'recommendations': recommendations,
            'positive_emotions': [e for e in emotions if e in positive_emotions],
            'negative_emotions': [e for e in emotions if e in negative_emotions],
            'explainability': {
                'logic': explanations,
                'confidence': 'High' if (pos_count + neg_count) > 2 else 'Moderate'
            }
        }

    def to_dict(self, analysis: NLPAnalysis) -> Dict[str, Any]:

        return {
            'sentiment': {
                'sentiment': analysis.sentiment.sentiment,
                'confidence': analysis.sentiment.confidence,
                'emotions': analysis.sentiment.emotions,
                'emotion_scores': analysis.sentiment.emotion_scores
            },
            'intent': {
                'intent': analysis.intent.intent,
                'confidence': analysis.intent.confidence,
                'sub_intents': analysis.intent.sub_intents
            },
            'entities': {
                'job_titles': analysis.entities.job_titles,
                'companies': analysis.entities.companies,
                'skills': analysis.entities.skills,
                'industries': analysis.entities.industries,
                'locations': analysis.entities.locations,
                'time_references': analysis.entities.time_references,
                'monetary_values': analysis.entities.monetary_values
            },
            'text_analysis': {
                'word_count': analysis.text_analysis.word_count,
                'sentence_count': analysis.text_analysis.sentence_count,
                'avg_sentence_length': analysis.text_analysis.avg_sentence_length,
                'confidence_level': analysis.text_analysis.confidence_level,
                'uncertainty_markers': analysis.text_analysis.uncertainty_markers,
                'action_words': analysis.text_analysis.action_words,
                'question_count': analysis.text_analysis.question_count,
                'complexity_score': analysis.text_analysis.complexity_score
            },
            'keywords': analysis.keywords,
            'summary': analysis.summary,
            'timestamp': analysis.timestamp.isoformat()
        }

    def _empty_analysis(self) -> NLPAnalysis:

        return NLPAnalysis(
            sentiment=SentimentResult('neutral', 0.5, [], {}),
            intent=IntentResult('unknown', 0.0, []),
            entities=EntityResult([], [], [], [], [], [], []),
            text_analysis=TextAnalysisResult(0, 0, 0.0, 0.5, [], [], 0, 0, 0.0),
            keywords=[],
            summary=None
        )
