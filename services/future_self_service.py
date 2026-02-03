from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import random

class FutureTimeframe(str, Enum):
    FIVE_YEARS = "5_years"
    TEN_YEARS = "10_years"
    FIFTEEN_YEARS = "15_years"

@dataclass
class FutureSelfPersona:
    user_id: str
    timeframe: FutureTimeframe
    decision_path: str
    personality_traits: List[str]
    life_circumstances: Dict[str, Any]
    emotional_state: str
    wisdom_gained: List[str]
    regrets: List[str]
    achievements: List[str]
    advice_themes: List[str]
    voice_tone: str
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ConversationTurn:
    role: str
    content: str
    emotion: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class FutureSelfSession:
    session_id: str
    user_id: str
    persona: FutureSelfPersona
    conversation_history: List[ConversationTurn] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    insights_generated: List[str] = field(default_factory=list)
    emotional_impact_score: float = 0.0

class FutureSelfService:
    PERSONALITY_EVOLUTION = {
        'job_change': {
            'positive': ['confident', 'professionally fulfilled', 'financially secure'],
            'negative': ['burned out', 'questioning choices', 'nostalgic'],
            'neutral': ['adapted', 'pragmatic', 'experienced']
        },
        'career_switch': {
            'positive': ['reinvented', 'passionate', 'grateful'],
            'negative': ['still adjusting', 'rebuilding', 'learning'],
            'neutral': ['evolving', 'open-minded', 'resilient']
        },
        'promotion': {
            'positive': ['accomplished', 'influential', 'mentoring'],
            'negative': ['overwhelmed', 'lonely at top', 'stressed'],
            'neutral': ['strategic', 'seasoned', 'growth-oriented']
        },
        'entrepreneurship': {
            'positive': ['free', 'impactful', 'wealth-building'],
            'negative': ['stressed', 'uncertain', 'isolated'],
            'neutral': ['autonomous', 'resilient', 'learning']
        }
    }

    WISDOM_TEMPLATES = [
        "The job title matters less than the daily experience.",
        "Your network becomes your net worth.",
        "Skills compound over time - start learning now.",
        "The best career move is often the one that scares you.",
        "Don't optimize for salary alone. Time and health matter.",
        "Regrets of inaction weigh heavier than regrets of action.",
        "The small daily choices matter more than big decisions.",
        "Start investing earlier than you think you should.",
        "Quality of relationships matters more than quantity."
    ]

    def __init__(self):
        self.active_sessions: Dict[str, FutureSelfSession] = {}
        self.user_personas: Dict[str, List[FutureSelfPersona]] = {}
        self.conversation_logs: Dict[str, List[Dict]] = {}

    def create_persona(self, user_id: str, decision_type: str,
                       decision_desc: str, timeframe: str = "5_years",
                       scenario: str = "realistic") -> FutureSelfPersona:
        evolution = self.PERSONALITY_EVOLUTION.get(decision_type,
                    self.PERSONALITY_EVOLUTION['job_change'])

        if scenario == 'optimistic':
            traits = evolution['positive'] + [evolution['neutral'][0]]
            emotional_state = random.choice(['content', 'fulfilled', 'grateful'])
            regrets = ["Not starting sooner"]
        elif scenario == 'pessimistic':
            traits = evolution['negative'] + [evolution['neutral'][0]]
            emotional_state = random.choice(['reflective', 'wistful', 'hopeful'])
            regrets = ["Could have considered more alternatives", "Underestimated challenges"]
        else:
            traits = [evolution['positive'][0], evolution['neutral'][0], evolution['negative'][0]]
            emotional_state = random.choice(['balanced', 'content', 'thoughtful'])
            regrets = ["Minor missed opportunities"]

        years = int(timeframe.split('_')[0])
        wisdom = random.sample(self.WISDOM_TEMPLATES, 4)
        achievements = [f"Built expertise over {years} years", "Developed strong network", "Achieved stability"]

        persona = FutureSelfPersona(
            user_id=user_id,
            timeframe=FutureTimeframe(timeframe),
            decision_path=decision_desc,
            personality_traits=traits,
            life_circumstances={'career_level': 'senior', 'balance': 'managed'},
            emotional_state=emotional_state,
            wisdom_gained=wisdom,
            regrets=regrets,
            achievements=achievements,
            advice_themes=['timing', 'growth', 'patience'],
            voice_tone='thoughtful'
        )

        if user_id not in self.user_personas:
            self.user_personas[user_id] = []
        self.user_personas[user_id].append(persona)
        return persona

    def start_conversation(self, user_id: str, decision_type: str,
                          decision_desc: str, timeframe: str = "5_years",
                          scenario: str = "realistic") -> Dict[str, Any]:
        persona = self.create_persona(user_id, decision_type, decision_desc, timeframe, scenario)
        session_id = f"fs_{user_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        years = int(timeframe.split('_')[0])
        opening = f"Hey... it's you. I mean, it's me - {years} years from now. I know you're wrestling with {decision_desc}. I've been where you are. {random.choice(persona.wisdom_gained)} What would you like to know?"

        session = FutureSelfSession(
            session_id=session_id, user_id=user_id, persona=persona
        )
        session.conversation_history.append(ConversationTurn(
            role="future_self", content=opening, emotion=persona.emotional_state
        ))

        self.active_sessions[session_id] = session

        return {
            "session_id": session_id,
            "opening_message": opening,
            "persona": self._persona_dict(persona),
            "suggested_questions": [
                "What do you wish you had known?",
                "What was the hardest part?",
                "Do you have any regrets?",
                "Was it worth it?"
            ]
        }

    def send_message(self, session_id: str, message: str) -> Dict[str, Any]:
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}

        session = self.active_sessions[session_id]
        persona = session.persona
        session.conversation_history.append(ConversationTurn(role="user", content=message))

        response = self._generate_response(persona, message)
        session.conversation_history.append(ConversationTurn(
            role="future_self", content=response, emotion=persona.emotional_state
        ))

        session.emotional_impact_score = min(100, len(session.conversation_history) * 10)

        return {
            "response": response,
            "emotion": persona.emotional_state,
            "impact_score": session.emotional_impact_score
        }

    def _generate_response(self, persona: FutureSelfPersona, message: str) -> str:
        msg = message.lower()
        wisdom = random.choice(persona.wisdom_gained)

        if any(w in msg for w in ['regret', 'wish', 'mistake']):
            regret = persona.regrets[0] if persona.regrets else "few things"
            return f"My biggest regret? {regret}. But {wisdom} The fact you're asking now shows wisdom."
        elif any(w in msg for w in ['advice', 'suggest', 'should']):
            return f"Here's my advice: {wisdom} Trust your preparation, but stay flexible."
        elif any(w in msg for w in ['hard', 'difficult', 'challenge']):
            return f"The hardest part was the uncertainty. But {wisdom} You're stronger than you think."
        elif any(w in msg for w in ['worth', 'satisfied', 'happy']):
            return f"I feel {persona.emotional_state}. {wisdom} The journey taught me more than I expected."
        else:
            return f"That's thoughtful. {wisdom} What's really driving that question?"

    def end_conversation(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}

        session = self.active_sessions[session_id]
        persona = session.persona
        wisdom = random.choice(persona.wisdom_gained)

        summary = {
            "session_id": session_id,
            "turns": len(session.conversation_history),
            "impact_score": session.emotional_impact_score,
            "closing": f"Before we part... {wisdom} Trust yourself. I'm proof you navigate this well.",
            "key_wisdom": persona.wisdom_gained[:3]
        }

        if session.user_id not in self.conversation_logs:
            self.conversation_logs[session.user_id] = []
        self.conversation_logs[session.user_id].append(summary)

        del self.active_sessions[session_id]
        return summary

    def _persona_dict(self, p: FutureSelfPersona) -> Dict:
        return {
            "timeframe": p.timeframe.value,
            "traits": p.personality_traits,
            "emotional_state": p.emotional_state,
            "wisdom": p.wisdom_gained,
            "achievements": p.achievements
        }

    def get_session(self, session_id: str) -> Optional[Dict]:
        if session_id not in self.active_sessions:
            return None
        s = self.active_sessions[session_id]
        return {
            "session_id": session_id,
            "persona": self._persona_dict(s.persona),
            "history": [{"role": t.role, "content": t.content} for t in s.conversation_history]
        }

future_self_service = FutureSelfService()
