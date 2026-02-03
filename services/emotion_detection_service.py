import base64
import io
import os
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import cv2

@dataclass
class EmotionResult:
    emotion: str
    confidence: float
    all_emotions: Dict[str, float]
    face_location: Dict[str, int]

@dataclass
class FaceDetectionResult:
    faces_detected: int
    emotions: List[EmotionResult]
    dominant_emotion: Optional[str]
    average_confidence: float
    timestamp: str
    analysis_time_ms: float
    recommendations: List[str] = field(default_factory=list)

class EmotionDetectionService:
    EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    EMOTION_COLORS = {
        'angry': (0, 0, 255),
        'disgust': (0, 140, 100),
        'fear': (128, 0, 128),
        'happy': (0, 255, 0),
        'sad': (255, 0, 0),
        'surprise': (0, 255, 255),
        'neutral': (128, 128, 128)
    }

    EMOTION_RECOMMENDATIONS = {
        'angry': [
            "Take a moment to breathe before making career decisions",
            "Consider waiting 24 hours before responding to situations",
            "Channel this energy into productive problem-solving"
        ],
        'disgust': [
            "Identify what specifically bothers you about the situation",
            "Consider if this relates to a values mismatch",
            "Document your concerns objectively"
        ],
        'fear': [
            "Fear often signals growth opportunities",
            "Break down the decision into smaller, manageable steps",
            "Consider what's the worst case and how you'd handle it"
        ],
        'happy': [
            "Great emotional state for brainstorming new opportunities",
            "Document what's contributing to this positive state",
            "Consider if this is a good time for important decisions"
        ],
        'sad': [
            "Take time for self-care before major decisions",
            "Consider speaking with a mentor or coach",
            "Postpone major career decisions if possible"
        ],
        'surprise': [
            "Give yourself time to process unexpected information",
            "Document your initial reactions for later reflection",
            "Consider both positive and negative implications"
        ],
        'neutral': [
            "This is often the best emotional state for objective decisions",
            "Good time for analytical thinking and planning",
            "Consider gathering more information to inform decisions"
        ]
    }

    def __init__(self, model_path: Optional[str] = None):
        self.initialized = False
        self.face_cascade = None
        self.emotion_model = None
        self.model_path = model_path
        self._detection_history: Dict[str, List[EmotionResult]] = {}
        self._user_baselines: Dict[str, Dict[str, float]] = {}
        self._consent_registry: Dict[str, Dict[str, bool]] = {}
        self._volatility_monitor: Dict[str, List[float]] = {}

    async def initialize(self):
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

            if self.face_cascade.empty():
                print("Warning: Could not load face cascade classifier")
                self.initialized = False
                return False

            self._load_emotion_model()

            self.initialized = True
            print("Emotion Detection Service initialized successfully")
            return True

        except Exception as e:
            print(f"Failed to initialize emotion detection: {e}")
            self.initialized = False
            return False

    def register_consent(self, user_id: str, permissions: Dict[str, bool]):
        """Explicit consent control for multi-modal inputs"""
        self._consent_registry[user_id] = {
            'video': permissions.get('video', False),
            'voice': permissions.get('voice', False),
            'biometric': permissions.get('biometric', False),
            'text': True
        }

    def check_signal_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """Assess signal quality for reliability"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            brightness = np.mean(gray)
            contrast = np.std(gray)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

            return {
                'is_good': brightness > 40 and brightness < 220 and blur_score > 100,
                'brightness': brightness,
                'contrast': contrast,
                'blur': blur_score,
                'issues': [
                    msg for condition, msg in [
                        (brightness < 40, "Image too dark"),
                        (brightness > 220, "Image too bright"),
                        (blur_score < 100, "Image too blurry")
                    ] if condition
                ]
            }
        except Exception:
            return {'is_good': True, 'issues': []}

    def update_baseline(self, user_id: str, emotion: str, confidence: float):
        """Learn individual emotional patterns continuously"""
        if user_id not in self._user_baselines:
            self._user_baselines[user_id] = defaultdict(float)

        current_baseline = self._user_baselines[user_id].get(emotion, 0.0)
        self._user_baselines[user_id][emotion] = (current_baseline * 0.9) + (confidence * 0.1)

    def detect_volatility(self, user_id: str, current_emotions: Dict[str, float]) -> float:
        """Detect emotional volatility over time"""
        if user_id not in self._volatility_monitor:
            self._volatility_monitor[user_id] = []

        emotion_vector = np.array([current_emotions.get(e, 0) for e in self.EMOTION_LABELS])
        self._volatility_monitor[user_id].append(emotion_vector)

        if len(self._volatility_monitor[user_id]) > 5:
            self._volatility_monitor[user_id] = self._volatility_monitor[user_id][-5:]

            changes = []
            for i in range(1, len(self._volatility_monitor[user_id])):
                v1 = self._volatility_monitor[user_id][i-1]
                v2 = self._volatility_monitor[user_id][i]
                changes.append(np.linalg.norm(v2 - v1))

            return np.mean(changes)
        return 0.0

    def get_calibrated_confidence(self, user_id: str, emotion: str, raw_confidence: float) -> float:
        """Improve accuracy over time using baselines"""
        baseline = self._user_baselines.get(user_id, {}).get(emotion, 0.5)
        if baseline > 0.6:
            return raw_confidence * 0.8
        return raw_confidence

    def _load_emotion_model(self):
        self.emotion_model = self._rule_based_emotion_model
        print("Using rule-based emotion detection (for demonstration)")

    def _rule_based_emotion_model(self, face_roi: np.ndarray) -> Dict[str, float]:
        height, width = face_roi.shape[:2]

        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_roi

        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)

        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)

        top_half = gray[:height//2, :]
        bottom_half = gray[height//2:, :]
        brightness_ratio = np.mean(top_half) / (np.mean(bottom_half) + 1e-6)

        np.random.seed(int(mean_brightness * 100) % 1000)
        base_scores = np.random.dirichlet([1.0] * 7)

        emotions = {}
        for i, emotion in enumerate(self.EMOTION_LABELS):
            score = base_scores[i]

            if emotion == 'happy' and mean_brightness > 120:
                score *= 1.5
            elif emotion == 'sad' and mean_brightness < 100:
                score *= 1.3
            elif emotion == 'angry' and std_brightness > 50:
                score *= 1.4
            elif emotion == 'surprise' and edge_density > 0.1:
                score *= 1.4
            elif emotion == 'neutral' and 100 < mean_brightness < 140:
                score *= 1.3

            emotions[emotion] = float(score)

        total = sum(emotions.values())
        emotions = {k: v/total for k, v in emotions.items()}

        return emotions

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if self.face_cascade is None:
            return []

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        gray = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        return [(x, y, w, h) for (x, y, w, h) in faces]

    def analyze_emotions(self, image: np.ndarray, user_id: Optional[str] = None) -> FaceDetectionResult:
        start_time = datetime.now()

        if user_id and user_id in self._consent_registry:
            if not self._consent_registry[user_id]['video']:
                return FaceDetectionResult(
                    faces_detected=0, emotions=[], dominant_emotion="privacy_restricted",
                    average_confidence=0.0, timestamp=datetime.utcnow().isoformat(),
                    analysis_time_ms=0.0, recommendations=["Video analysis disabled by user settings"]
                )

        if not self.initialized:
            return FaceDetectionResult(
                faces_detected=0, emotions=[], dominant_emotion=None,
                average_confidence=0.0, timestamp=datetime.utcnow().isoformat(),
                analysis_time_ms=0.0, recommendations=["Emotion detection service not initialized"]
            )

        quality = self.check_signal_quality(image)
        quality_warnings = quality.get('issues', [])

        faces = self.detect_faces(image)

        if not faces:
            if quality_warnings:
                recs = [f"Detection failed: {w}" for w in quality_warnings]
            else:
                recs = ["No face detected. Please ensure good lighting and face the camera directly."]

            return FaceDetectionResult(
                faces_detected=0, emotions=[], dominant_emotion=None,
                average_confidence=0.0, timestamp=datetime.utcnow().isoformat(),
                analysis_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                recommendations=recs
            )

        emotion_results = []

        for (x, y, w, h) in faces:
            padding = int(w * 0.1)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)

            face_roi = image[y1:y2, x1:x2]

            if face_roi.size == 0:
                continue

            face_resized = cv2.resize(face_roi, (48, 48))

            emotions = self.emotion_model(face_resized)

            dominant_emotion_tuple = max(emotions.items(), key=lambda x: x[1])
            dom_emotion = dominant_emotion_tuple[0]
            raw_confidence = dominant_emotion_tuple[1]

            if user_id:
                confidence = self.get_calibrated_confidence(user_id, dom_emotion, raw_confidence)
                self.update_baseline(user_id, dom_emotion, confidence)
                volatility = self.detect_volatility(user_id, emotions)
            else:
                confidence = raw_confidence

            result = EmotionResult(
                emotion=dom_emotion,
                confidence=confidence,
                all_emotions=emotions,
                face_location={'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
            )

            emotion_results.append(result)

        dominant_emotion = None
        avg_confidence = 0.0
        recommendations = []

        if emotion_results:
            emotion_counts = {}
            for result in emotion_results:
                emotion_counts[result.emotion] = emotion_counts.get(result.emotion, 0) + 1
            dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]

            avg_confidence = sum(r.confidence for r in emotion_results) / len(emotion_results)

            base_recs = self.EMOTION_RECOMMENDATIONS.get(dominant_emotion, [])
            recommendations.extend(base_recs)

            if user_id and self.detect_volatility(user_id, {}) > 0.3:
                recommendations.insert(0, "High emotional volatility detected. Consider pausing.")

            if quality_warnings:
                recommendations.extend([f"Note: {w}" for w in quality_warnings])

        analysis_time = (datetime.now() - start_time).total_seconds() * 1000

        return FaceDetectionResult(
            faces_detected=len(emotion_results),
            emotions=emotion_results,
            dominant_emotion=dominant_emotion,
            average_confidence=avg_confidence,
            timestamp=datetime.utcnow().isoformat(),
            analysis_time_ms=analysis_time,
            recommendations=recommendations
        )

    def decode_base64_image(self, base64_string: str) -> Optional[np.ndarray]:
        try:
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]

            image_data = base64.b64decode(base64_string)

            nparr = np.frombuffer(image_data, np.uint8)

            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            return image

        except Exception as e:
            print(f"Error decoding base64 image: {e}")
            return None

    def analyze_base64_image(self, base64_string: str) -> FaceDetectionResult:
        image = self.decode_base64_image(base64_string)

        if image is None:
            return FaceDetectionResult(
                faces_detected=0,
                emotions=[],
                dominant_emotion=None,
                average_confidence=0.0,
                timestamp=datetime.utcnow().isoformat(),
                analysis_time_ms=0.0,
                recommendations=["Failed to decode image. Please try again with a valid image."]
            )

        return self.analyze_emotions(image)

    def annotate_image(self, image: np.ndarray, result: FaceDetectionResult) -> np.ndarray:
        annotated = image.copy()

        for emotion_result in result.emotions:
            loc = emotion_result.face_location
            x, y, w, h = loc['x'], loc['y'], loc['width'], loc['height']

            color = self.EMOTION_COLORS.get(emotion_result.emotion, (255, 255, 255))

            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            label = f"{emotion_result.emotion}: {emotion_result.confidence:.1%}"

            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            cv2.rectangle(
                annotated,
                (x, y - text_height - 10),
                (x + text_width + 10, y),
                color,
                -1
            )

            cv2.putText(
                annotated,
                label,
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        return annotated

    def encode_image_to_base64(self, image: np.ndarray) -> str:
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')

    def record_emotion(self, user_id: str, result: EmotionResult):
        if user_id not in self._detection_history:
            self._detection_history[user_id] = []

        self._detection_history[user_id].append(result)

        if len(self._detection_history[user_id]) > 100:
            self._detection_history[user_id] = self._detection_history[user_id][-100:]

    def get_emotion_history(self, user_id: str) -> List[Dict[str, Any]]:
        if user_id not in self._detection_history:
            return []

        return [
            {
                'emotion': r.emotion,
                'confidence': r.confidence,
                'all_emotions': r.all_emotions
            }
            for r in self._detection_history[user_id]
        ]

    def get_emotion_trends(self, user_id: str) -> Dict[str, Any]:
        history = self._detection_history.get(user_id, [])

        if not history:
            return {
                'total_detections': 0,
                'emotion_distribution': {},
                'dominant_emotion': None,
                'average_confidence': 0.0
            }

        emotion_counts = {}
        total_confidence = 0.0

        for result in history:
            emotion_counts[result.emotion] = emotion_counts.get(result.emotion, 0) + 1
            total_confidence += result.confidence

        total = len(history)
        emotion_distribution = {k: v/total for k, v in emotion_counts.items()}

        return {
            'total_detections': total,
            'emotion_distribution': emotion_distribution,
            'dominant_emotion': max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else None,
            'average_confidence': total_confidence / total if total > 0 else 0.0
        }

    def to_dict(self, result: FaceDetectionResult) -> Dict[str, Any]:
        return {
            'faces_detected': result.faces_detected,
            'emotions': [
                {
                    'emotion': e.emotion,
                    'confidence': e.confidence,
                    'all_emotions': e.all_emotions,
                    'face_location': e.face_location
                }
                for e in result.emotions
            ],
            'dominant_emotion': result.dominant_emotion,
            'average_confidence': result.average_confidence,
            'timestamp': result.timestamp,
            'analysis_time_ms': result.analysis_time_ms,
            'recommendations': result.recommendations
        }

    def cleanup(self):
        self.face_cascade = None
        self.emotion_model = None
        self._detection_history.clear()
        self.initialized = False
