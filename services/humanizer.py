from typing import Dict, List, Optional, Tuple
import random
from dataclasses import dataclass
from datetime import datetime

@dataclass
class HumanizedResponse:
    main_message: str
    supporting_points: List[str]
    emotional_tone: str
    call_to_action: str
    encouragement: str

class ResponseHumanizer:
    GREETINGS = {
        "morning": ["Good morning.", "Hello, hope your morning is off to a great start.", "Hi, great to hear from you this morning."],
        "afternoon": ["Good afternoon.", "Hi there, hope you're having a good day.", "Hello, thanks for reaching out."],
        "evening": ["Good evening.", "Hi, hope you've had a productive day.", "Hello there, nice to connect with you."],
        "default": ["Hello.", "Hi there.", "Hey, great to hear from you."]
    }

    ACKNOWLEDGMENTS = {
        "high_stress": ["I can sense this decision is weighing heavily on you, and that's completely understandable.", "Career decisions can feel overwhelming - you're not alone in feeling this way.", "It takes courage to even think critically about a decision like this."],
        "moderate_stress": ["It's clear you're putting real thought into this, which is admirable.", "Taking time to reflect on your choices shows maturity and self-awareness.", "You're asking the right questions about your career path."],
        "low_stress": ["It's great that you're being proactive about your career planning.", "I appreciate you thinking ahead about these decisions.", "You seem to have a good handle on the situation."]
    }

    TRANSITIONS = ["Looking at your situation,", "Based on what you've shared,", "Here's what I'm seeing:", "Let me share some thoughts:", "Here's my perspective on this:", "Diving into the details,"]

    ENCOURAGEMENTS = {
        "low": ["You're in a good position here.", "The outlook looks positive for this path.", "This seems like a solid decision overall."],
        "moderate": ["There are challenges ahead, but they're manageable.", "With some careful planning, you can navigate this successfully.", "This is a balanced choice with both opportunities and considerations."],
        "high": ["This is a significant decision, but that doesn't mean it's wrong.", "Sometimes the riskier paths lead to the most growth.", "Let's work through this together to minimize potential downsides."]
    }

    CALLS_TO_ACTION = ["What aspects would you like to explore further?", "Would you like me to dive deeper into any of these points?", "Is there a specific concern I can help address?", "What's the most pressing question on your mind right now?", "Shall we look at some specific scenarios together?"]

    TONE_MODIFIERS = {
        "supportive": {"prefix": "I want you to know that ", "connectors": ["and remember, ", "also, ", "importantly, "]},
        "analytical": {"prefix": "Looking at the data, ", "connectors": ["furthermore, ", "additionally, ", "also noteworthy: "]},
        "encouraging": {"prefix": "Here's the good news: ", "connectors": ["what's more, ", "even better, ", "plus, "]},
        "cautious": {"prefix": "It's important to consider that ", "connectors": ["keep in mind, ", "be aware that ", "also consider, "]}
    }

    def __init__(self):
        self.last_greeting_index = -1
        self.conversation_context = {}

    def humanize_regret_analysis(self, prediction: Dict, decision_data: Dict, user_name: Optional[str] = None) -> HumanizedResponse:
        regret_score = prediction.get('predicted_regret', 0.5)
        confidence = prediction.get('confidence', 0.5)
        risk_level = prediction.get('risk_level', 'moderate')
        top_factors = prediction.get('top_factors', [])

        stress_level = 'high_stress' if regret_score > 0.6 else ('moderate_stress' if regret_score > 0.3 else 'low_stress')

        greeting = self._get_greeting(user_name)
        acknowledgment = random.choice(self.ACKNOWLEDGMENTS[stress_level])
        transition = random.choice(self.TRANSITIONS)

        if regret_score < 0.3:
            core_message = self._build_low_regret_message(prediction, decision_data)
            emotional_tone = "encouraging"
        elif regret_score < 0.6:
            core_message = self._build_moderate_regret_message(prediction, decision_data)
            emotional_tone = "supportive"
        else:
            core_message = self._build_high_regret_message(prediction, decision_data)
            emotional_tone = "cautious"

        supporting_points = self._build_supporting_points(top_factors, decision_data)
        encouragement = random.choice(self.ENCOURAGEMENTS[risk_level])
        call_to_action = random.choice(self.CALLS_TO_ACTION)

        main_message = f"{greeting}\n\n{acknowledgment}\n\n{transition} {core_message}\n\n{encouragement}\n\n{call_to_action}"

        return HumanizedResponse(main_message=main_message, supporting_points=supporting_points, emotional_tone=emotional_tone, call_to_action=call_to_action, encouragement=encouragement)

    def humanize_graph_analysis(self, graph_analysis: Dict, user_profile: Optional[Dict] = None) -> str:
        outcomes = graph_analysis.get('reachable_outcomes', [])
        risk_level = graph_analysis.get('risk_level', 'moderate')
        dominant = graph_analysis.get('dominant_outcome', {})

        intro = random.choice(self.TRANSITIONS)

        if outcomes:
            top_outcomes = outcomes[:3]
            outcome_text = self._narrativize_outcomes(top_outcomes)
        else:
            outcome_text = "I'm seeing multiple potential paths ahead for you."

        dominant_insight = self._get_outcome_insight(dominant['outcome']) if dominant and dominant.get('outcome') else ""
        risk_perspective = self._get_risk_perspective(risk_level)

        return f"{intro}\n\n{outcome_text}\n\n{dominant_insight}\n\n{risk_perspective}\n\nLet me know if you'd like to explore any of these paths in more detail."

    def humanize_recommendation(self, recommendations: List[str]) -> str:
        if not recommendations:
            return "I don't have specific recommendations right now, but I'm here to help you think through your options."
        intro = "Here are some things I'd suggest thinking about:"
        humanized_recs = []
        for i, rec in enumerate(recommendations[:5], 1):
            humanized = self._make_recommendation_conversational(rec)
            humanized_recs.append(f"{i}. {humanized}")
        outro = "\nRemember, these are starting points, not rules. Trust your instincts too."
        return f"{intro}\n\n" + "\n\n".join(humanized_recs) + outro

    def create_empathetic_response(self, user_input: str, analysis_result: Optional[Dict] = None) -> str:
        emotion = self._detect_emotion(user_input)

        if emotion == 'anxious':
            opening = "I hear the uncertainty in what you're sharing, and that's okay. Career decisions are inherently uncertain."
        elif emotion == 'frustrated':
            opening = "I understand the frustration. Navigating career decisions can feel like you're hitting walls sometimes."
        elif emotion == 'excited':
            opening = "I love the energy. It's exciting to be at a point where you're considering new possibilities."
        elif emotion == 'confused':
            opening = "Let's work through this together. It's completely normal to feel unsure about which direction to take."
        else:
            opening = "Thanks for sharing this with me. Let's think through it together."

        if analysis_result:
            analysis_section = self._summarize_analysis_naturally(analysis_result)
            return f"{opening}\n\n{analysis_section}"
        return f"{opening}\n\nTell me more about what's on your mind, and I'll do my best to help you think it through."

    def format_final_response(self, llm_response: str, prediction_data: Optional[Dict] = None) -> str:
        response = self._clean_robotic_language(llm_response)
        if prediction_data:
            insight = self._create_data_insight(prediction_data)
            if insight:
                response += f"\n\nQuick insight: {insight}"
        if not any(response.rstrip().endswith(c) for c in '?!'):
            response += "\n\nWhat else would you like to explore?"
        return response

    def _get_greeting(self, user_name: Optional[str] = None) -> str:
        hour = datetime.now().hour
        if 5 <= hour < 12:
            period = "morning"
        elif 12 <= hour < 17:
            period = "afternoon"
        elif 17 <= hour < 22:
            period = "evening"
        else:
            period = "default"
        greeting = random.choice(self.GREETINGS[period])
        if user_name:
            greeting = greeting.replace(".", f", {user_name}.")
        return greeting

    def _build_low_regret_message(self, prediction: Dict, decision_data: Dict) -> str:
        decision_type = decision_data.get('decision_type', 'this decision')
        confidence = prediction.get('confidence', 0.5)
        messages = [f"Your {decision_type} appears to align well with your values and circumstances.", "The analysis suggests this is a thoughtful choice with good potential outcomes.", "Based on what you've shared, you seem to be making a well-considered decision."]
        base = random.choice(messages)
        if confidence > 0.7:
            base += " I'm fairly confident in this assessment based on the information provided."
        return base

    def _build_moderate_regret_message(self, prediction: Dict, decision_data: Dict) -> str:
        decision_type = decision_data.get('decision_type', 'this decision')
        messages = [f"Your {decision_type} has both promising aspects and some areas that warrant attention.", "There's a balanced mix of opportunities and considerations here.", "This decision has good potential, though there are some factors worth examining more closely."]
        return random.choice(messages)

    def _build_high_regret_message(self, prediction: Dict, decision_data: Dict) -> str:
        decision_type = decision_data.get('decision_type', 'this decision')
        messages = [f"Your {decision_type} raises some flags that I think are worth discussing.", "There are some significant considerations that I want to make sure you've thought through.", "Based on the analysis, there are aspects of this decision that might benefit from more exploration."]
        base = random.choice(messages)
        base += " That doesn't mean it's the wrong choice - sometimes the most meaningful decisions carry more risk."
        return base

    def _build_supporting_points(self, factors: List[Tuple[str, float]], decision_data: Dict) -> List[str]:
        points = []
        factor_templates = {
            'financial': "The financial implications seem {assessment}",
            'work_life_balance': "Work-life balance looks {assessment}",
            'career_growth': "Career growth potential appears {assessment}",
            'personal_satisfaction': "Personal satisfaction could be {assessment}",
            'family': "Family considerations seem {assessment}",
            'health': "Health impact looks {assessment}",
            'location': "Location factors appear {assessment}",
            'skills': "Skills development seems {assessment}",
            'network': "Network impact appears {assessment}",
            'risk_tolerance': "Risk level seems {assessment}"
        }
        for factor_name, importance in factors[:5]:
            if factor_name in factor_templates:
                assessment = "like a key consideration" if importance > 0.15 else "manageable"
                point = factor_templates[factor_name].format(assessment=assessment)
                points.append(point)
        return points

    def _narrativize_outcomes(self, outcomes: List[Dict]) -> str:
        if not outcomes:
            return ""
        primary = outcomes[0]
        primary_name = primary.get('outcome', 'positive outcome').replace('_', ' ')
        primary_prob = primary.get('probability', 0.5)
        narrative = f"Based on your current trajectory, the most likely path leads toward {primary_name} with about a {primary_prob:.0%} probability. "
        if len(outcomes) > 1:
            alternatives = [o.get('outcome', '').replace('_', ' ') for o in outcomes[1:3]]
            narrative += f"Other possible directions include {' and '.join(alternatives)}."
        return narrative

    def _get_outcome_insight(self, outcome: str) -> str:
        insights = {
            'high_satisfaction': "People who reach this outcome often describe it as deeply fulfilling.",
            'moderate_satisfaction': "This outcome typically provides solid stability and contentment.",
            'low_satisfaction': "This path might require some adjustments to feel more fulfilling.",
            'financial_success': "Financial security can provide freedom to pursue what matters most.",
            'financial_struggle': "Financial challenges are often temporary setbacks, not permanent states.",
            'work_life_balance': "Balance looks different for everyone - it's about finding your rhythm.",
            'burnout': "Burnout is recoverable with the right support and boundaries.",
            'skill_growth': "Skill development opens doors you might not even see yet.",
            'skill_stagnation': "Feeling stagnant often precedes significant growth - it's a signal, not a sentence.",
            'network_expansion': "Relationships built along the way often prove more valuable than expected.",
            'isolation': "Professional isolation can be addressed - it just requires intentional connection."
        }
        return insights.get(outcome, "")

    def _get_risk_perspective(self, risk_level: str) -> str:
        perspectives = {
            'low': "From a risk perspective, this looks like a relatively safe path with manageable challenges.",
            'moderate': "There's some risk involved here, but it's within what most people would consider acceptable for meaningful career moves.",
            'high': "This path carries notable risk - but risk isn't inherently bad. The question is whether it aligns with your goals and whether you're prepared for various outcomes."
        }
        return perspectives.get(risk_level, perspectives['moderate'])

    def _make_recommendation_conversational(self, rec: str) -> str:
        rec = rec.lstrip('-* ').strip()
        conversational_starters = ["Consider ", "It might help to ", "You could try ", "Think about ", "One idea is to "]
        if not any(rec.lower().startswith(s.lower()) for s in conversational_starters):
            rec = random.choice(conversational_starters) + rec.lower()
        return rec

    def _detect_emotion(self, text: str) -> str:
        text_lower = text.lower()
        anxiety_words = ['worried', 'anxious', 'scared', 'nervous', 'uncertain', 'afraid']
        frustration_words = ['frustrated', 'stuck', 'annoying', 'tired of', 'hate']
        excitement_words = ['excited', 'thrilled', 'amazing', 'opportunity', "can't wait"]
        confusion_words = ['confused', "don't know", 'not sure', 'unclear', 'which']

        if any(word in text_lower for word in anxiety_words):
            return 'anxious'
        elif any(word in text_lower for word in frustration_words):
            return 'frustrated'
        elif any(word in text_lower for word in excitement_words):
            return 'excited'
        elif any(word in text_lower for word in confusion_words):
            return 'confused'
        return 'neutral'

    def _summarize_analysis_naturally(self, analysis: Dict) -> str:
        regret = analysis.get('predicted_regret', 0.5)
        if regret < 0.3:
            return "Looking at this holistically, the signs are positive. This seems like a well-aligned choice for you."
        elif regret < 0.6:
            return "There's a balanced picture here - some promising elements and some areas to be mindful of."
        return "There are some considerations worth exploring further before moving forward."

    def _clean_robotic_language(self, text: str) -> str:
        replacements = {
            "Based on my analysis": "Looking at this",
            "It is recommended that": "You might want to",
            "The data indicates": "From what I can see",
            "In conclusion": "All in all",
            "It should be noted": "One thing to keep in mind",
            "The following factors": "A few things",
            "Users should": "You could",
            "It is important to": "Don't forget to",
            "As per the analysis": "Based on what we've discussed"
        }
        for robotic, human in replacements.items():
            text = text.replace(robotic, human)
            text = text.replace(robotic.lower(), human.lower())
        return text

    def _create_data_insight(self, prediction_data: Dict) -> str:
        regret = prediction_data.get('predicted_regret', 0.5)
        confidence = prediction_data.get('confidence', 0.5)
        if confidence < 0.5:
            return "I'd like more information to give you a more confident assessment."
        if regret < 0.3:
            return f"Your decision scores in the low regret range ({regret:.0%} regret likelihood)."
        elif regret < 0.6:
            return f"This falls in the moderate consideration zone ({regret:.0%} regret potential)."
        return f"This registers as worth careful thought ({regret:.0%} regret potential)."
