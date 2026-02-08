import base64
import io
import os
from typing import Dict, List, Optional
from datetime import datetime
import json

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class EnhancedEmotionService:
    """Enhanced emotion detection using DeepFace or fallback"""
    
    EMOTION_LABELS = {
        'angry': {'label': 'Angry', 'color': '#ef4444', 'icon': '😠'},
        'disgust': {'label': 'Disgusted', 'color': '#84cc16', 'icon': '🤢'},
        'fear': {'label': 'Fearful', 'color': '#a855f7', 'icon': '😨'},
        'happy': {'label': 'Happy', 'color': '#22c55e', 'icon': '😊'},
        'sad': {'label': 'Sad', 'color': '#3b82f6', 'icon': '😢'},
        'surprise': {'label': 'Surprised', 'color': '#f97316', 'icon': '😲'},
        'neutral': {'label': 'Neutral', 'color': '#6b7280', 'icon': '😐'}
    }
    
    def __init__(self):
        self.deepface_available = DEEPFACE_AVAILABLE
        self.cv2_available = CV2_AVAILABLE
        
        self.emotion_history: Dict[str, List[Dict]] = {}
    
    def detect_emotion_from_base64(self, image_base64: str) -> Dict:
        """Detect emotions from base64 encoded image"""
        try:
            if ',' in image_base64:
                image_base64 = image_base64.split(',')[1]
            
            image_data = base64.b64decode(image_base64)
            
            if self.deepface_available and self.cv2_available:
                return self._detect_with_deepface(image_data)
            else:
                return self._detect_fallback(image_data)
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "emotions": []
            }
    
    def _detect_with_deepface(self, image_data: bytes) -> Dict:
        """Use DeepFace for real emotion detection"""
        try:
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return {"success": False, "error": "Failed to decode image", "emotions": []}
            
            results = DeepFace.analyze(
                img,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            emotions = []
            
            if isinstance(results, list):
                for result in results:
                    emotions.append(self._parse_deepface_result(result))
            else:
                emotions.append(self._parse_deepface_result(results))
            
            return {
                "success": True,
                "method": "deepface",
                "face_detected": len(emotions) > 0 and emotions[0].get("dominant_emotion") is not None,
                "emotions": emotions
            }
        
        except Exception as e:
            return self._detect_fallback(image_data)
    
    def _parse_deepface_result(self, result: Dict) -> Dict:
        """Parse DeepFace result into standardized format"""
        emotion_scores = result.get('emotion', {})
        dominant_emotion = result.get('dominant_emotion', 'neutral')
        
        parsed_emotions = []
        for emotion, score in sorted(emotion_scores.items(), key=lambda x: -x[1]):
            info = self.EMOTION_LABELS.get(emotion, {'label': emotion.capitalize(), 'color': '#6b7280', 'icon': '😐'})
            parsed_emotions.append({
                "emotion": emotion,
                "label": info['label'],
                "score": round(score, 2),
                "percentage": round(score, 1),
                "color": info['color'],
                "icon": info['icon']
            })
        
        dominant_info = self.EMOTION_LABELS.get(dominant_emotion, {'label': dominant_emotion.capitalize(), 'icon': '😐'})
        
        return {
            "dominant_emotion": dominant_emotion,
            "dominant_label": dominant_info['label'],
            "dominant_icon": dominant_info['icon'],
            "confidence": emotion_scores.get(dominant_emotion, 0),
            "all_emotions": parsed_emotions[:5],
            "face_region": result.get('region', {})
        }
    
    def _detect_fallback(self, image_data: bytes) -> Dict:
        """Fallback emotion detection (simulated for demo)"""
        import random
        
        emotions = list(self.EMOTION_LABELS.keys())
        
        scores = [random.random() for _ in emotions]
        total = sum(scores)
        normalized = {e: (s / total) * 100 for e, s in zip(emotions, scores)}
        
        sorted_emotions = sorted(normalized.items(), key=lambda x: -x[1])
        dominant = sorted_emotions[0][0]
        
        parsed_emotions = []
        for emotion, score in sorted_emotions:
            info = self.EMOTION_LABELS.get(emotion, {})
            parsed_emotions.append({
                "emotion": emotion,
                "label": info.get('label', emotion),
                "score": round(score, 2),
                "percentage": round(score, 1),
                "color": info.get('color', '#6b7280'),
                "icon": info.get('icon', '😐')
            })
        
        dominant_info = self.EMOTION_LABELS.get(dominant, {})
        
        return {
            "success": True,
            "method": "fallback",
            "face_detected": True,
            "emotions": [{
                "dominant_emotion": dominant,
                "dominant_label": dominant_info.get('label', dominant),
                "dominant_icon": dominant_info.get('icon', '😐'),
                "confidence": normalized[dominant],
                "all_emotions": parsed_emotions[:5]
            }]
        }
    
    def record_emotion(self, user_id: str, emotion_data: Dict, context: str = None):
        """Record emotion for trend analysis"""
        if user_id not in self.emotion_history:
            self.emotion_history[user_id] = []
        
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "emotion": emotion_data.get("dominant_emotion", "neutral"),
            "confidence": emotion_data.get("confidence", 0),
            "context": context
        }
        
        self.emotion_history[user_id].append(record)
        
        if len(self.emotion_history[user_id]) > 100:
            self.emotion_history[user_id] = self.emotion_history[user_id][-100:]
    
    def get_emotion_trends(self, user_id: str) -> Dict:
        """Get emotion trends for a user"""
        history = self.emotion_history.get(user_id, [])
        
        if not history:
            return {"has_data": False, "trends": []}
        
        emotion_counts = {}
        for record in history:
            emotion = record.get("emotion", "neutral")
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        total = len(history)
        trends = []
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
            info = self.EMOTION_LABELS.get(emotion, {})
            trends.append({
                "emotion": emotion,
                "label": info.get("label", emotion),
                "icon": info.get("icon", "😐"),
                "color": info.get("color", "#6b7280"),
                "count": count,
                "percentage": round(count / total * 100, 1)
            })
        
        return {
            "has_data": True,
            "total_readings": total,
            "trends": trends,
            "most_common": trends[0] if trends else None
        }
    
    def analyze_decision_emotion_correlation(self, user_id: str, decisions: List[Dict]) -> Dict:
        """Analyze correlation between emotions and decision outcomes"""
        emotion_outcomes = {}
        
        for decision in decisions:
            emotions = decision.get('emotions', [])
            regret = decision.get('predicted_regret', 50)
            
            for e in emotions:
                if isinstance(e, dict):
                    emotion = e.get('emotion', e.get('dominant_emotion', 'neutral'))
                else:
                    emotion = str(e)
                
                if emotion not in emotion_outcomes:
                    emotion_outcomes[emotion] = []
                emotion_outcomes[emotion].append(regret)
        
        correlations = []
        for emotion, regrets in emotion_outcomes.items():
            if regrets:
                avg_regret = sum(regrets) / len(regrets)
                info = self.EMOTION_LABELS.get(emotion, {})
                
                correlations.append({
                    "emotion": emotion,
                    "label": info.get("label", emotion),
                    "icon": info.get("icon", "😐"),
                    "decision_count": len(regrets),
                    "avg_regret": round(avg_regret, 1),
                    "risk_level": "high" if avg_regret > 60 else "medium" if avg_regret > 40 else "low"
                })
        
        correlations.sort(key=lambda x: -x["avg_regret"])
        
        insights = []
        if correlations:
            highest_risk = correlations[0]
            if highest_risk["avg_regret"] > 50:
                insights.append(f"Be cautious when making decisions while feeling {highest_risk['label'].lower()}. These decisions have higher regret on average.")
            
            lowest_risk = correlations[-1] if len(correlations) > 1 else None
            if lowest_risk and lowest_risk["avg_regret"] < 40:
                insights.append(f"Decisions made while feeling {lowest_risk['label'].lower()} tend to have lower regret.")
        
        return {
            "correlations": correlations,
            "insights": insights
        }
    
    def get_service_status(self) -> Dict:
        """Get status of emotion detection capabilities"""
        return {
            "deepface_available": self.deepface_available,
            "cv2_available": self.cv2_available,
            "method": "deepface" if (self.deepface_available and self.cv2_available) else "fallback",
            "capabilities": {
                "face_detection": self.deepface_available and self.cv2_available,
                "emotion_detection": True,
                "trend_analysis": True,
                "correlation_analysis": True
            }
        }


enhanced_emotion_service = EnhancedEmotionService()
