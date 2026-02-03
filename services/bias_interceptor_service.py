from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re

class BiasType(str, Enum):
    SUNK_COST = "sunk_cost"
    LOSS_AVERSION = "loss_aversion"
    CONFIRMATION_BIAS = "confirmation_bias"
    ANCHORING = "anchoring"
    OVERCONFIDENCE = "overconfidence"
    STATUS_QUO = "status_quo"
    RECENCY_BIAS = "recency_bias"
    AVAILABILITY_HEURISTIC = "availability"
    PLANNING_FALLACY = "planning_fallacy"
    BANDWAGON = "bandwagon"
    HINDSIGHT = "hindsight"
    OPTIMISM_BIAS = "optimism_bias"

class InterventionLevel(str, Enum):
    SUBTLE = "subtle"
    MODERATE = "moderate"
    STRONG = "strong"

@dataclass
class BiasDetection:
    bias_type: BiasType
    confidence: float
    trigger_phrase: str
    explanation: str
    reframe_suggestion: str
    questions_to_ask: List[str]
    detected_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class InterventionHistory:
    user_id: str
    detections: List[BiasDetection] = field(default_factory=list)
    interventions_shown: int = 0
    interventions_accepted: int = 0
    common_biases: List[BiasType] = field(default_factory=list)

class BiasInterceptorService:
    BIAS_PATTERNS = {
        BiasType.SUNK_COST: {
            "patterns": [
                r"already (invested|spent|put in)",
                r"(years|months) of (work|effort|time)",
                r"can't (give up|quit|stop) now",
                r"too late to (change|turn back)",
                r"wasted? (time|money|effort)",
                r"come (this|so) far",
            ],
            "explanation": "You may be valuing past investments over future outcomes. Past costs are irreversible - only future benefits should guide decisions.",
            "reframe": "Instead of thinking about what you've invested, ask: 'Starting fresh today, would I make this same choice?'",
            "questions": [
                "If you were starting from scratch, would you make this choice?",
                "What future benefits does this path offer, independent of past investment?",
                "Are you continuing because it's right, or because you've already started?"
            ]
        },
        BiasType.LOSS_AVERSION: {
            "patterns": [
                r"afraid (of|to) lose",
                r"can't (afford|risk) losing",
                r"(might|could|will) lose",
                r"too risky",
                r"playing it safe",
                r"what if .* fails?",
                r"protect what I have",
            ],
            "explanation": "You may be weighing potential losses more heavily than equivalent gains. This can lead to excessive caution.",
            "reframe": "Try reframing in terms of what you might GAIN rather than lose. What opportunities might you miss by not acting?",
            "questions": [
                "What could you GAIN from this decision?",
                "What's the cost of NOT taking this risk?",
                "Is the fear of loss preventing you from growth?"
            ]
        },
        BiasType.CONFIRMATION_BIAS: {
            "patterns": [
                r"proves (that|my point)",
                r"confirms (what|that)",
                r"knew (it|I was right)",
                r"(supports|validates) my",
                r"everyone (agrees|thinks)",
                r"obvious(ly)?",
            ],
            "explanation": "You may be seeking information that confirms existing beliefs while ignoring contradicting evidence.",
            "reframe": "Actively seek out opposing viewpoints. Ask yourself: 'What evidence would change my mind?'",
            "questions": [
                "What evidence would change your mind?",
                "Have you genuinely considered opposing viewpoints?",
                "Who might disagree with you, and why?"
            ]
        },
        BiasType.ANCHORING: {
            "patterns": [
                r"first (offer|number|amount)",
                r"original (plan|idea|price)",
                r"started (at|with)",
                r"initial (thought|estimate)",
                r"compared to (before|the original)",
            ],
            "explanation": "You may be overly influenced by the first piece of information encountered (the 'anchor').",
            "reframe": "Step back and evaluate based on current information only. What would a fresh perspective suggest?",
            "questions": [
                "If you hadn't seen the initial number, what would you estimate?",
                "Are you evaluating this on its own merits or relative to a starting point?",
                "What does current market data suggest, ignoring prior numbers?"
            ]
        },
        BiasType.OVERCONFIDENCE: {
            "patterns": [
                r"definitely (will|going to)",
                r"(100%|completely|absolutely) (sure|certain)",
                r"no (doubt|way|chance)",
                r"guaranteed",
                r"can't (fail|go wrong)",
                r"easy|simple|obvious",
            ],
            "explanation": "You may be overestimating your knowledge, abilities, or the predictability of outcomes.",
            "reframe": "Consider: What could go wrong? What are you assuming that might not be true?",
            "questions": [
                "What assumptions are you making that might be wrong?",
                "What would need to happen for this to fail?",
                "Have you sought out critical feedback?"
            ]
        },
        BiasType.STATUS_QUO: {
            "patterns": [
                r"comfortable (with|where)",
                r"always (done|been)",
                r"why (change|fix)",
                r"not broken",
                r"(safe|stable|secure) (option|choice)",
                r"known (quantity|entity)",
            ],
            "explanation": "You may be preferring the current state simply because it's familiar, not because it's optimal.",
            "reframe": "Imagine you were choosing for the first time - would you choose your current situation?",
            "questions": [
                "If you were starting fresh, would you choose this path?",
                "What opportunities are you missing by staying put?",
                "Is 'comfortable' the same as 'best'?"
            ]
        },
        BiasType.RECENCY_BIAS: {
            "patterns": [
                r"just (happened|saw|heard)",
                r"recently",
                r"last (week|month|time)",
                r"latest (news|trend|data)",
                r"everyone's (talking|doing)",
            ],
            "explanation": "You may be overweighting recent events while underweighting historical patterns.",
            "reframe": "Look at longer-term trends. How does this fit into the bigger picture?",
            "questions": [
                "What does the historical pattern look like?",
                "Is this a trend or an anomaly?",
                "How would you view this if it happened a year ago?"
            ]
        },
        BiasType.PLANNING_FALLACY: {
            "patterns": [
                r"(should|will) be (quick|easy|fast)",
                r"(won't|can't) take (long|much)",
                r"just (need|have) to",
                r"(simple|straightforward|no big deal)",
                r"(couple|few) (weeks|months)",
            ],
            "explanation": "You may be underestimating the time, costs, and risks while overestimating benefits.",
            "reframe": "Add a buffer to your estimates. What do similar past projects teach you about timelines?",
            "questions": [
                "What does your track record suggest about timeline accuracy?",
                "What could cause delays that you haven't considered?",
                "Would you bet money on this timeline?"
            ]
        },
        BiasType.BANDWAGON: {
            "patterns": [
                r"everyone (is|else)",
                r"(popular|trending|hot)",
                r"(industry|market) is moving",
                r"don't want to (miss|be left)",
                r"FOMO|fear of missing",
            ],
            "explanation": "You may be influenced by what others are doing rather than what's right for you.",
            "reframe": "What's right for others may not be right for you. Focus on your unique situation.",
            "questions": [
                "Would you make this choice if no one else was doing it?",
                "Does this align with YOUR goals and values?",
                "Are you acting on information or social pressure?"
            ]
        },
        BiasType.OPTIMISM_BIAS: {
            "patterns": [
                r"best case",
                r"hopefully",
                r"should work out",
                r"(will|going to) be (fine|okay|great)",
                r"(lucky|blessed|fortunate)",
                r"things always",
            ],
            "explanation": "You may be assuming positive outcomes are more likely than the base rate suggests.",
            "reframe": "Consider multiple scenarios - best, realistic, and worst case. Plan for each.",
            "questions": [
                "What's your contingency plan if things don't go well?",
                "What do success rates look like for similar situations?",
                "Are you planning for what you hope or what's likely?"
            ]
        }
    }

    def __init__(self):
        self.user_history: Dict[str, InterventionHistory] = {}
        self.session_detections: Dict[str, List[BiasDetection]] = {}
        self.intervention_settings: Dict[str, Dict] = {}

    def analyze_text(self, text: str, user_id: str = None) -> List[Dict[str, Any]]:
        detections = []
        text_lower = text.lower()

        for bias_type, config in self.BIAS_PATTERNS.items():
            for pattern in config["patterns"]:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    confidence = self._calculate_confidence(text, match, bias_type)

                    if confidence > 0.4:
                        detection = BiasDetection(
                            bias_type=bias_type,
                            confidence=confidence,
                            trigger_phrase=match.group(0),
                            explanation=config["explanation"],
                            reframe_suggestion=config["reframe"],
                            questions_to_ask=config["questions"]
                        )
                        detections.append(detection)

                        if user_id:
                            self._record_detection(user_id, detection)
                        break

        detections.sort(key=lambda x: x.confidence, reverse=True)
        unique_detections = self._deduplicate(detections[:3])

        return [self._detection_to_dict(d) for d in unique_detections]

    def _calculate_confidence(self, full_text: str, match: re.Match, bias_type: BiasType) -> float:
        base_confidence = 0.5

        context_start = max(0, match.start() - 50)
        context_end = min(len(full_text), match.end() + 50)
        context = full_text[context_start:context_end]

        emphasis_words = ['really', 'definitely', 'absolutely', 'certainly', 'very', 'so']
        if any(word in context.lower() for word in emphasis_words):
            base_confidence += 0.15

        emotion_words = ['feel', 'afraid', 'worried', 'excited', 'anxious', 'stressed']
        if any(word in context.lower() for word in emotion_words):
            base_confidence += 0.1

        if match.group(0).lower() in ['definitely', 'guaranteed', 'already invested', 'too late']:
            base_confidence += 0.2

        return min(0.95, base_confidence)

    def _deduplicate(self, detections: List[BiasDetection]) -> List[BiasDetection]:
        seen_types = set()
        unique = []
        for d in detections:
            if d.bias_type not in seen_types:
                unique.append(d)
                seen_types.add(d.bias_type)
        return unique

    def _record_detection(self, user_id: str, detection: BiasDetection):
        if user_id not in self.user_history:
            self.user_history[user_id] = InterventionHistory(user_id=user_id)

        history = self.user_history[user_id]
        history.detections.append(detection)

        bias_counts = {}
        for d in history.detections:
            bias_counts[d.bias_type] = bias_counts.get(d.bias_type, 0) + 1

        history.common_biases = sorted(bias_counts.keys(), key=lambda x: bias_counts[x], reverse=True)[:3]

    def get_real_time_feedback(self, text: str, user_id: str = None) -> Dict[str, Any]:
        detections = self.analyze_text(text, user_id)

        if not detections:
            return {
                "has_bias": False,
                "message": None,
                "detections": []
            }

        top_detection = detections[0]
        intervention_level = self._get_intervention_level(top_detection['confidence'])

        if intervention_level == InterventionLevel.SUBTLE:
            message = f"Consider: {top_detection['questions'][0]}"
        elif intervention_level == InterventionLevel.MODERATE:
            message = f"Potential {top_detection['type'].replace('_', ' ')}: {top_detection['reframe']}"
        else:
            message = f"{top_detection['explanation']} {top_detection['reframe']}"

        return {
            "has_bias": True,
            "message": message,
            "intervention_level": intervention_level.value,
            "primary_bias": top_detection['type'],
            "confidence": top_detection['confidence'],
            "trigger_phrase": top_detection['trigger_phrase'],
            "reframe": top_detection['reframe'],
            "questions": top_detection['questions'],
            "detections": detections
        }

    def _get_intervention_level(self, confidence: float) -> InterventionLevel:
        if confidence < 0.5:
            return InterventionLevel.SUBTLE
        elif confidence < 0.75:
            return InterventionLevel.MODERATE
        else:
            return InterventionLevel.STRONG

    def get_user_bias_profile(self, user_id: str) -> Dict[str, Any]:
        if user_id not in self.user_history:
            return {
                "user_id": user_id,
                "total_detections": 0,
                "common_biases": [],
                "improvement_areas": []
            }

        history = self.user_history[user_id]

        bias_tips = {
            BiasType.SUNK_COST: "Practice asking 'Would I start this today?' before continuing projects",
            BiasType.LOSS_AVERSION: "Reframe decisions in terms of gains, not just avoiding losses",
            BiasType.OVERCONFIDENCE: "Seek out disconfirming evidence before making decisions",
            BiasType.STATUS_QUO: "Schedule regular reviews to question current approaches"
        }

        improvement_areas = [bias_tips.get(b, f"Be mindful of {b.value}") for b in history.common_biases[:3]]

        return {
            "user_id": user_id,
            "total_detections": len(history.detections),
            "common_biases": [b.value for b in history.common_biases],
            "interventions_shown": history.interventions_shown,
            "interventions_accepted": history.interventions_accepted,
            "acceptance_rate": history.interventions_accepted / max(1, history.interventions_shown),
            "improvement_areas": improvement_areas,
            "recent_detections": [
                self._detection_to_dict(d) for d in history.detections[-5:]
            ]
        }

    def record_intervention_response(self, user_id: str, accepted: bool):
        if user_id in self.user_history:
            history = self.user_history[user_id]
            history.interventions_shown += 1
            if accepted:
                history.interventions_accepted += 1

    def get_bias_explanation(self, bias_type: str) -> Dict[str, Any]:
        try:
            bias = BiasType(bias_type)
            config = self.BIAS_PATTERNS.get(bias, {})
            return {
                "bias_type": bias_type,
                "explanation": config.get("explanation", "No explanation available"),
                "reframe": config.get("reframe", ""),
                "questions": config.get("questions", []),
                "examples": config.get("patterns", [])[:3]
            }
        except ValueError:
            return {"error": f"Unknown bias type: {bias_type}"}

    def _detection_to_dict(self, d: BiasDetection) -> Dict[str, Any]:
        return {
            "type": d.bias_type.value,
            "confidence": round(d.confidence, 2),
            "trigger_phrase": d.trigger_phrase,
            "explanation": d.explanation,
            "reframe": d.reframe_suggestion,
            "questions": d.questions_to_ask,
            "detected_at": d.detected_at.isoformat()
        }

bias_interceptor = BiasInterceptorService()
