import asyncio
import uuid
import time
import json
import gc
import os
import warnings
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
from pathlib import Path

import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=False)
except RuntimeError:
    pass

warnings.filterwarnings('ignore', message='.*resource_tracker.*')
warnings.filterwarnings('ignore', message='.*leaked semaphore.*')
warnings.filterwarnings('ignore', category=UserWarning, module='multiprocessing')

from fastapi import FastAPI, HTTPException, Request, Depends, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field, field_validator

from config import settings
from models.ml_pipeline import EnhancedRegretPredictor
from models.graph_engine import AdvancedDecisionGraph
from services import (
    EnhancedOllamaService, OllamaService, OllamaConfig,
    AdvancedFeedbackLoop, FeedbackLoop,
    ResponseHumanizer, RAGService,
    AuthService, RateLimiter, CacheService, MonitoringService,
    AnalyticsService, NLPService,
    JournalService, SimulationService, CoachingService,
    MarketIntelligenceService, CommunityInsightsService,
    ExportService, GamificationService,
    EmotionDetectionService, OutcomeLearningService,
    DecisionTemplateService, FileUploadService,
    FutureSelfService, OpportunityScoutService,
    BiasInterceptorService, GlobalRegretDatabase,
    PersistenceService, persistence_service,
    MultiverseVisualizationService,
    mentor_matching_service, decision_sharing_service,
    multi_llm_service, fine_tuning_service,
    ab_testing_service, enterprise_integration_service,
    roadmap_service, knowledge_service, simulation_service,
    YouTubeRecommendationService, youtube_recommendation_service,
    LLMProvider
)
from services.migration_service import migration_service, PHASE_3_MIGRATIONS

try:
    migration_service.apply_migrations(PHASE_3_MIGRATIONS)
except Exception as e:
    print(f"Startup Migration Error: {e}")

from services.security import (
    SecurityConfig,
    SecurityHeaders,
    InputValidator,
    HardenedRateLimiter,
    HardenedAuthService,
    IPManager,
    AuditLogger,
    SecurityMiddlewareHelper,
    RequestValidator,
    get_security_helper,
    get_audit_logger,
    AuthService,
    RateLimiter,
    CacheService,
    MonitoringService
)

class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)

def safe_json_response(data: Any, status_code: int = 200) -> Response:
    content = json.dumps(data, cls=SafeJSONEncoder, ensure_ascii=False)
    body = content.encode('utf-8')
    return Response(
        content=body,
        status_code=status_code,
        media_type="application/json; charset=utf-8"
    )

class DecisionInput(BaseModel):
    decision_type: str = Field(..., max_length=50)
    description: str = Field(..., max_length=5000)
    emotions: List[str] = Field(default=[])
    years_experience: Optional[int] = Field(default=5, ge=0, le=50)
    age: Optional[int] = Field(default=30, ge=18, le=80)
    risk_tolerance: Optional[float] = Field(default=0.5, ge=0, le=1)
    financial_stability: Optional[float] = Field(default=0.5, ge=0, le=1)
    time_pressure: Optional[float] = Field(default=0.5, ge=0, le=1)
    industry: Optional[str] = Field(default="technology", max_length=100)
    user_id: Optional[str] = Field(default=None, max_length=100)

    @field_validator('decision_type', 'description', 'industry', 'user_id', mode='before')
    @classmethod
    def sanitize_strings(cls, v):
        if v is None:
            return v
        return InputValidator.sanitize_string(str(v))

    @field_validator('emotions', mode='before')
    @classmethod
    def validate_emotions(cls, v):
        if not v:
            return []
        return [InputValidator.sanitize_string(str(e))[:50] for e in v[:20]]

class ChatInput(BaseModel):
    message: str = Field(..., max_length=10000)
    user_id: Optional[str] = Field(default=None, max_length=100)

    @field_validator('message', mode='before')
    @classmethod
    def validate_message(cls, v):
        if not v:
            raise ValueError("Message cannot be empty")
        valid, error = InputValidator.validate_input(str(v), "message")
        if not valid:
            raise ValueError(error)
        return InputValidator.sanitize_string(str(v), max_length=10000)

    @field_validator('user_id', mode='before')
    @classmethod
    def sanitize_user_id(cls, v):
        if v is None:
            return v
        return InputValidator.sanitize_string(str(v), max_length=100)

class FeedbackInput(BaseModel):
    feedback_type: str = Field(..., max_length=50)
    content: Dict
    analysis_id: Optional[str] = Field(default=None, max_length=100)
    user_id: Optional[str] = Field(default=None, max_length=100)

    @field_validator('feedback_type', 'analysis_id', 'user_id', mode='before')
    @classmethod
    def sanitize_strings(cls, v):
        if v is None:
            return v
        return InputValidator.sanitize_string(str(v))

    @field_validator('content', mode='before')
    @classmethod
    def validate_content(cls, v):
        if not InputValidator.validate_json_depth(v):
            raise ValueError("Content structure too deeply nested")
        return v

class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=30)
    email: str = Field(..., max_length=254)
    password: str = Field(..., min_length=12, max_length=128)

    @field_validator('username')
    @classmethod
    def validate_username(cls, v):
        if not InputValidator.validate_username(v):
            raise ValueError("Username must be 3-30 alphanumeric characters or underscores")
        return v

    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        if not InputValidator.validate_email(v):
            raise ValueError("Invalid email format")
        return v.lower()

    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        valid, msg = InputValidator.validate_password_strength(v)
        if not valid:
            raise ValueError(msg)
        return v

class UserLogin(BaseModel):
    username: str = Field(..., min_length=3, max_length=30)
    password: str = Field(..., min_length=1, max_length=128)


class AppState:
    ml_predictor: EnhancedRegretPredictor = None
    decision_graph: AdvancedDecisionGraph = None
    ollama_service: EnhancedOllamaService = None
    feedback_loop: AdvancedFeedbackLoop = None
    humanizer: ResponseHumanizer = None
    rag_service: RAGService = None
    auth_service: AuthService = None
    rate_limiter: RateLimiter = None
    cache: CacheService = None
    monitoring: MonitoringService = None
    analytics: AnalyticsService = None
    nlp_service: NLPService = None
    journal_service: JournalService = None
    simulation_service: SimulationService = None
    coaching_service: CoachingService = None
    market_intelligence: MarketIntelligenceService = None
    community_insights: CommunityInsightsService = None
    export_service: ExportService = None
    gamification: GamificationService = None
    emotion_detection: EmotionDetectionService = None
    outcome_learning: OutcomeLearningService = None
    decision_templates: DecisionTemplateService = None
    upload_service: FileUploadService = None
    future_self: FutureSelfService = None
    opportunity_scout: OpportunityScoutService = None
    bias_interceptor: BiasInterceptorService = None
    global_regret_db: GlobalRegretDatabase = None
    persistence: PersistenceService = None
    multiverse_viz: MultiverseVisualizationService = None
    youtube_recommendation: YouTubeRecommendationService = None

app_state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Career Decision Regret System...")

    app_state.ml_predictor = EnhancedRegretPredictor(model_path=settings.MODEL_PATH)
    print("ML/DL Pipeline initialized")

    app_state.decision_graph = AdvancedDecisionGraph(decay_factor=settings.DECAY_FACTOR)
    print("Decision Graph initialized")

    app_state.rag_service = RAGService()
    await app_state.rag_service.initialize()
    print("RAG Service initialized")

    ollama_config = OllamaConfig(base_url=settings.OLLAMA_BASE_URL, model=settings.OLLAMA_MODEL)
    app_state.ollama_service = EnhancedOllamaService(config=ollama_config, rag_service=app_state.rag_service)
    await app_state.ollama_service.check_availability()
    status = "Connected" if app_state.ollama_service.is_available else "Fallback Mode"
    print(f"Ollama Service: {status}")

    app_state.feedback_loop = AdvancedFeedbackLoop(
        ml_predictor=app_state.ml_predictor, decision_graph=app_state.decision_graph
    )
    print("Feedback Loop initialized")

    app_state.humanizer = ResponseHumanizer()
    app_state.auth_service = AuthService()
    app_state.rate_limiter = RateLimiter(requests_per_minute=100, requests_per_hour=2000)
    app_state.cache = CacheService(max_size=500, ttl=300)
    app_state.monitoring = MonitoringService()
    app_state.analytics = AnalyticsService()

    app_state.nlp_service = NLPService()
    await app_state.nlp_service.initialize()
    print("NLP Service initialized")

    app_state.journal_service = JournalService()
    app_state.simulation_service = SimulationService()
    app_state.coaching_service = CoachingService()
    app_state.market_intelligence = MarketIntelligenceService()
    app_state.community_insights = CommunityInsightsService()
    app_state.export_service = ExportService()
    app_state.gamification = GamificationService()

    app_state.emotion_detection = EmotionDetectionService()
    await app_state.emotion_detection.initialize()
    print("Emotion Detection Service initialized")

    app_state.outcome_learning = OutcomeLearningService()
    print("Outcome Learning Service initialized")

    app_state.decision_templates = DecisionTemplateService()
    print("Decision Templates System initialized")

    app_state.upload_service = FileUploadService(upload_dir="uploads")
    print("File Upload Service initialized")

    app_state.future_self = FutureSelfService()
    print("Future Self Simulation Service initialized")

    app_state.opportunity_scout = OpportunityScoutService()
    print("Opportunity Scout Agent initialized")

    app_state.bias_interceptor = BiasInterceptorService()
    print("Real-Time Bias Interceptor initialized")

    app_state.global_regret_db = GlobalRegretDatabase()
    print("Global Regret Database initialized")

    app_state.persistence = PersistenceService()
    print("Persistence Service initialized")

    app_state.multiverse_viz = MultiverseVisualizationService()
    print("Multiverse 3D Visualization Service initialized")

    app_state.youtube_recommendation = youtube_recommendation_service
    print("YouTube Recommendation Engine initialized")

    print("Feature Services initialized (Journal, Simulation, Coaching, Market, Community, Export, Gamification, Emotion Detection, Outcome Learning, Decision Templates, Future Self, Opportunity Scout, Bias Interceptor, Global Regret DB, Persistence, Multiverse Viz, YouTube Recommendations)")


    print("All services initialized successfully")

    print("=" * 50)
    print("Career Decision Regret System Ready")
    print(f"API: http://{settings.HOST}:{settings.PORT}")
    print("=" * 50)

    yield

    print("Shutting down Career Decision Regret System...")

    try:
        if app_state.ollama_service:
            await app_state.ollama_service.close()

        if app_state.rag_service:
            app_state.rag_service.cleanup()

        app_state.ml_predictor = None
        app_state.decision_graph = None
        app_state.feedback_loop = None
        app_state.humanizer = None
        app_state.nlp_service = None
        app_state.journal_service = None
        app_state.simulation_service = None
        app_state.coaching_service = None
        app_state.market_intelligence = None
        app_state.community_insights = None
        app_state.export_service = None
        app_state.gamification = None

        if app_state.emotion_detection:
            app_state.emotion_detection.cleanup()
        app_state.emotion_detection = None

        gc.collect()

    except Exception as e:
        print(f"Warning during shutdown: {e}")

    print("Shutdown complete")

class HardenedSecurityMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.security_helper = get_security_helper()
        self.audit = get_audit_logger()

    async def dispatch(self, request: Request, call_next):
        direct_ip = request.client.host if request.client else "unknown"
        headers_dict = {k.lower(): v for k, v in request.headers.items()}
        client_ip = self.security_helper.get_client_ip(headers_dict, direct_ip)

        user_agent = headers_dict.get("user-agent", "")
        allowed, error_msg, status_code = self.security_helper.check_request(
            ip_address=client_ip,
            path=str(request.url.path),
            method=request.method,
            headers=headers_dict,
            user_agent=user_agent
        )

        if not allowed:
            return JSONResponse(
                status_code=status_code,
                content={"error": error_msg, "code": "SECURITY_BLOCK"}
            )

        content_length = headers_dict.get("content-length", "0")
        try:
            if int(content_length) > SecurityConfig.MAX_REQUEST_SIZE:
                return JSONResponse(
                    status_code=413,
                    content={"error": "Request too large", "code": "PAYLOAD_TOO_LARGE"}
                )
        except ValueError:
            pass

        try:
            response = await call_next(request)
        except Exception as e:
            self.audit.log(
                event_type="ERROR",
                ip_address=client_ip,
                resource=str(request.url.path),
                action=request.method,
                success=False,
                details={"error": str(e)}
            )
            raise

        is_html = "text/html" in response.headers.get("content-type", "")
        for header, value in SecurityHeaders.get_headers(is_html).items():
            if value:
                response.headers[header] = value

        return response


CORS_ORIGINS = os.getenv("CORS_ORIGINS", "").split(",")
CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS if origin.strip()]
if not CORS_ORIGINS:
    if settings.DEBUG:
        CORS_ORIGINS = ["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:3000"]
    else:
        CORS_ORIGINS = []

app = FastAPI(
    title="Career Decision Regret System",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

app.add_middleware(HardenedSecurityMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS else [],
    allow_credentials=True if CORS_ORIGINS else False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With", "X-CSRF-Token"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"],
    max_age=600
)

async def get_current_user(request: Request) -> str:
    """FastAPI dependency to validate session token and return user_id"""
    auth_header = request.headers.get("Authorization", "")
    
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        user_id = app_state.auth_service.validate_session(token)
        if user_id:
            return user_id
    
    client_ip = request.client.host if request.client else "unknown"
    
    is_loopback = (
        client_ip in ["127.0.0.1", "::1", "localhost", "testclient", "unknown"] or 
        client_ip.startswith("127.0.0.") or 
        client_ip.startswith("::ffff:127.0.0.1")
    )
    
    host_header = request.headers.get("host", "").split(":")[0]
    is_local_host = host_header in ["localhost", "127.0.0.1", "::1"]
    
    if is_loopback or is_local_host:
        path_user_id = request.path_params.get("user_id")
        if path_user_id:
            return path_user_id
        return "default_user"
    
    error_msg = f"Authentication required for access from {client_ip} (Host: {host_header})."
    print(f"SECURITY ALERT: Blocking remote access from {client_ip}. Path: {request.url.path}")
    raise HTTPException(status_code=401, detail=error_msg)

def verify_owner(requested_user_id: str, authenticated_user_id: str):
    """Enforce IDOR protection by verifying user ownership"""
    if requested_user_id != authenticated_user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this resource")

async def get_current_admin(user_id: str = Depends(get_current_user)) -> str:
    """Dependency to ensure the current authenticated user has admin privileges"""
    user = app_state.auth_service.get_user_by_id(user_id)
    if not user or not getattr(user, 'is_admin', False):
        get_audit_logger().log(
            event_type="UNAUTHORIZED",
            ip_address="internal",
            resource="admin_gate",
            action="access_denied",
            success=False,
            user_id=user_id
        )
        raise HTTPException(status_code=403, detail="Administrative privileges required")
    return user_id

@app.get("/apple-touch-icon.png")
@app.get("/apple-touch-icon-precomposed.png")
async def get_apple_touch_icon():
    return FileResponse("assets/apple-touch-icon.png", media_type="image/png")

DASHBOARD_HTML = '''<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="AI-powered career decision analysis">
    <title>Career Decision AI | Professional Guidance</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="manifest" href="/manifest.json">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        if ('serviceWorker' in navigator) {
            window.addEventListener('load', () => {
                navigator.serviceWorker.register('/service-worker.js?v=5').then(reg => {
                    console.log('SW registered');
                    reg.update();
                });
            });
        }
        // Background unregister any non-v3 workers
        navigator.serviceWorker.getRegistrations().then(regs => {
            regs.forEach(reg => { if(!reg.active || !reg.active.scriptURL.includes('v=5')) reg.unregister(); });
        });
        // Safe stubs for functions that may be defined later in the large script block.
        // These avoid uncaught ReferenceErrors when users click UI before parsing finishes.
        window.loadGraph = window.loadGraph || function() {
            console.warn('loadGraph() called before initialization');
        };
        window.resetGraphZoom = window.resetGraphZoom || function() {
            console.warn('resetGraphZoom() called before initialization');
        };
        window.zoomGraph = window.zoomGraph || function(factor) {
            console.warn('zoomGraph(' + factor + ') called before initialization');
        };
        window.focusOnNode = window.focusOnNode || function(id) {
            console.warn('focusOnNode(' + id + ') called before initialization');
        };
    </script>
    <style>
        :root {
            --bg-primary: #0a0a0a;
            --bg-secondary: rgba(15, 15, 15, 0.95);
            --bg-card: rgba(18, 18, 18, 0.9);
            --bg-elevated: rgba(24, 24, 24, 0.9);
            --bg-hover: rgba(40, 40, 40, 0.95);
            --text-primary: #ffffff;
            --text-secondary: #b0b0b0;
            --text-muted: #707070;
            --border: rgba(255, 255, 255, 0.1);
            --border-hover: rgba(255, 255, 255, 0.2);
            --accent: #ffffff;
            --accent-gradient: linear-gradient(135deg, #ffffff 0%, #d0d0d0 100%);
            --accent-dark: #1a1a1a;
            --success: #22c55e;
            --warning: #eab308;
            --danger: #ef4444;
            --radius-sm: 8px;
            --radius-md: 12px;
            --radius-lg: 16px;
            --radius-xl: 24px;
            --radius-full: 9999px;
            --glass-blur: blur(24px);
            --shadow-sm: 0 2px 8px rgba(0,0,0,0.3);
            --shadow-md: 0 4px 20px rgba(0,0,0,0.4);
            --shadow-lg: 0 8px 40px rgba(0,0,0,0.5);
            --transition-fast: 0.15s cubic-bezier(0.4, 0, 0.2, 1);
            --transition-normal: 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            --transition-slow: 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        }

        [data-theme="light"] {
            --bg-primary: #f8f9fa;
            --bg-secondary: rgba(255, 255, 255, 0.8);
            --bg-card: rgba(255, 255, 255, 0.7);
            --bg-elevated: rgba(241, 243, 245, 0.7);
            --bg-hover: #e9ecef;
            --text-primary: #1a1a1b;
            --text-secondary: #495057;
            --text-muted: #868e96;
            --border: rgba(0, 0, 0, 0.1);
            --border-hover: rgba(0, 0, 0, 0.2);
            --accent: #1a1a1b;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        /* Modern Animations */
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        @keyframes slideInLeft {
            from { opacity: 0; transform: translateX(-20px); }
            to { opacity: 1; transform: translateX(0); }
        }
        
        @keyframes slideInRight {
            from { opacity: 0; transform: translateX(20px); }
            to { opacity: 1; transform: translateX(0); }
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.05); opacity: 0.8; }
        }
        
        @keyframes shimmer {
            0% { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-5px); }
        }
        
        @keyframes recording {
            0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
            50% { box-shadow: 0 0 0 12px rgba(239, 68, 68, 0); }
        }
        
        @keyframes glow {
            0%, 100% { box-shadow: 0 0 20px rgba(99, 102, 241, 0.3); }
            50% { box-shadow: 0 0 40px rgba(99, 102, 241, 0.5); }
        }

        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            overflow: hidden;
        }

        /* Sidebar */
        .sidebar {
            width: 260px;
            background: var(--bg-secondary);
            backdrop-filter: var(--glass-blur);
            -webkit-backdrop-filter: var(--glass-blur);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            height: 100vh;
            position: fixed;
            left: 0;
            top: 0;
            z-index: 100;
            animation: slideInLeft 0.4s ease-out;
            overflow-y: auto;
            scrollbar-width: thin;
            scrollbar-color: rgba(255, 255, 255, 0.1) transparent;
        }

        .sidebar::-webkit-scrollbar {
            width: 6px;
        }

        .sidebar::-webkit-scrollbar-track {
            background: rgba(0, 0, 0, 0.05);
        }

        .sidebar::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.15);
            border-radius: 10px;
        }

        .sidebar::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.3);
        }
        
        .sidebar::after {
            content: '';
            position: absolute;
            top: 0;
            right: 0;
            width: 1px;
            height: 100%;
            background: linear-gradient(to bottom, transparent, rgba(255, 255, 255, 0.15), transparent);
            opacity: 0.5;
        }

        .sidebar-header {
            padding: 1.5rem;
            border-bottom: 1px solid var(--border);
        }

        .brand {
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 3px;
            text-transform: uppercase;
            color: var(--text-primary);
        }

        .brand-sub {
            font-size: 0.7rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }

        .new-conversation-btn {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            width: calc(100% - 2rem);
            margin: 1rem;
            padding: 0.875rem 1rem;
            background: var(--accent);
            color: var(--bg-primary);
            border: none;
            border-radius: var(--radius-md);
            font-weight: 600;
            font-size: 0.875rem;
            cursor: pointer;
            transition: all 0.2s;
        }

        .new-conversation-btn:hover {
            opacity: 0.9;
            transform: translateY(-1px);
        }

        .nav-section {
            padding: 0 0.75rem;
            margin-bottom: 1.5rem;
        }

        .nav-label {
            font-size: 0.65rem;
            font-weight: 600;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            color: var(--text-muted);
            padding: 0 0.75rem;
            margin-bottom: 0.5rem;
        }

        .nav-btn {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            width: 100%;
            padding: 0.75rem 1rem;
            background: transparent;
            border: none;
            border-radius: var(--radius-md);
            color: var(--text-secondary);
            font-size: 0.875rem;
            text-align: left;
            cursor: pointer;
            transition: all var(--transition-fast);
            margin-bottom: 4px;
            position: relative;
            overflow: hidden;
        }
        
        .nav-btn::before {
            content: '';
            position: absolute;
            left: 0;
            top: 50%;
            transform: translateY(-50%) scaleY(0);
            width: 3px;
            height: 60%;
            background: var(--accent);
            border-radius: 0 3px 3px 0;
            transition: transform var(--transition-fast);
        }

        .nav-btn:hover {
            background: var(--bg-hover);
            color: var(--text-primary);
            transform: translateX(4px);
        }
        
        .nav-btn:hover::before {
            transform: translateY(-50%) scaleY(1);
        }

        .nav-btn.active {
            background: var(--bg-hover);
            color: var(--text-primary);
            font-weight: 500;
        }
        
        .nav-btn.active::before {
            transform: translateY(-50%) scaleY(1);
        }

        .sidebar-footer {
            margin-top: auto;
            padding: 1rem;
            border-top: 1px solid var(--border);
        }

        .connection-status {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.75rem;
            background: var(--bg-elevated);
            border-radius: var(--radius-sm);
            margin-bottom: 0.75rem;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--success);
            animation: pulse 2s infinite;
        }

        .status-dot.offline {
            background: var(--warning);
            animation: none;
        }

        @keyframes pulse {

            0%,
            100% {
                opacity: 1
            }

            50% {
                opacity: 0.5
            }
        }

        .status-text {
            font-size: 0.75rem;
            color: var(--text-secondary);
        }

        .status-mode {
            font-weight: 600;
            color: var(--text-primary);
        }

        .settings-btn {
            width: 100%;
            padding: 0.75rem;
            background: transparent;
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            color: var(--text-secondary);
            font-size: 0.875rem;
            cursor: pointer;
            transition: all 0.15s;
        }

        .settings-btn:hover {
            background: var(--bg-hover);
            border-color: var(--border-hover);
        }

        /* Main Content */
        .main {
            flex: 1;
            margin-left: 260px;
            display: flex;
            flex-direction: column;
            height: 100vh;
            position: relative;
            z-index: 1;
        }

        /* Header */
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 1.5rem;
            border-bottom: 1px solid var(--border);
            background: var(--bg-card);
            backdrop-filter: var(--glass-blur);
            -webkit-backdrop-filter: var(--glass-blur);
        }

        .breadcrumb {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.875rem;
            color: var(--text-secondary);
        }

        .breadcrumb-current {
            color: var(--text-primary);
            font-weight: 500;
        }

        .header-actions {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .status-badge {
            padding: 0.375rem 0.75rem;
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            font-size: 0.75rem;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.15s;
        }

        .status-badge:hover {
            background: var(--bg-hover);
        }

        .status-badge.active {
            background: var(--success);
            color: white;
            border-color: var(--success);
        }

        /* Chat Container */
        .chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .message-group {
            display: flex;
            gap: 0.75rem;
            max-width: 85%;
            animation: fadeInUp 0.4s ease-out;
        }

        .message-group.user {
            align-self: flex-end;
            flex-direction: row-reverse;
            animation: slideInRight 0.4s ease-out;
        }

        .avatar {
            width: 40px;
            height: 40px;
            border-radius: var(--radius-md);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8rem;
            font-weight: 600;
            flex-shrink: 0;
            transition: transform var(--transition-fast);
        }
        
        .avatar:hover {
            transform: scale(1.05);
        }

        .avatar.ai {
            background: rgba(255, 255, 255, 0.08);
            color: var(--text-secondary);
            border: 1px solid rgba(255, 255, 255, 0.15);
        }

        .avatar.user {
            background: var(--accent);
            color: var(--bg-primary);
        }

        .message-content {
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }

        .message-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.75rem;
        }

        .message-name {
            font-weight: 600;
            color: var(--text-primary);
        }

        .message-time {
            color: var(--text-muted);
        }

        .message-bubble {
            padding: 1rem 1.25rem;
            border-radius: var(--radius-lg);
            font-size: 0.9rem;
            line-height: 1.6;
            transition: all var(--transition-fast);
        }
        
        .message-bubble:hover {
            transform: translateY(-1px);
        }

        .message-group.assistant .message-bubble {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-top-left-radius: 4px;
        }

        .message-group.user .message-bubble {
            background: var(--accent);
            color: var(--bg-primary);
            border-top-right-radius: 4px;
            box-shadow: 0 4px 15px rgba(255, 255, 255, 0.08);
        }

        /* Chat Input */
        .chat-input-area {
            padding: 1rem 1.5rem 1.5rem;
            border-top: 1px solid var(--border);
            background: var(--bg-card);
            backdrop-filter: var(--glass-blur);
            -webkit-backdrop-filter: var(--glass-blur);
        }

        .input-container {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            background: var(--bg-elevated);
            backdrop-filter: var(--glass-blur);
            -webkit-backdrop-filter: var(--glass-blur);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 0.75rem 1rem;
            transition: all 0.2s;
        }

        .input-container:focus-within {
            border-color: var(--border-hover);
        }

        .chat-input {
            flex: 1;
            background: transparent;
            border: none;
            color: var(--text-primary);
            font-size: 0.9rem;
            font-family: inherit;
            outline: none;
        }

        .chat-input::placeholder {
            color: var(--text-muted);
        }
        
        /* File Upload Styles */
        .file-preview {
            border-top: 1px solid var(--border);
            border-left: 1px solid var(--border);
            border-right: 1px solid var(--border);
            border-top-left-radius: var(--radius-lg);
            border-top-right-radius: var(--radius-lg);
            background: var(--bg-elevated);
            margin: 0;
        }
        
        .file-tag {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.25rem 0.5rem;
            background: var(--bg-hover);
            border: 1px solid var(--border);
            border-radius: 4px;
            font-size: 0.8rem;
            max-width: 200px;
        }
        
        .file-icon {
            font-size: 1.2rem;
        }
        
        .file-info {
            overflow: hidden;
        }
        
        .file-name {
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            font-weight: 500;
        }
        
        .file-status {
            font-size: 0.7rem;
            color: var(--text-muted);
        }
        
        .file-tag.success { border-color: rgba(34, 197, 94, 0.3); background: rgba(34, 197, 94, 0.1); }
        .file-tag.error { border-color: rgba(239, 68, 68, 0.3); background: rgba(239, 68, 68, 0.1); }

        .input-actions {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .input-btn {
            padding: 0.625rem 1rem;
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            color: var(--text-secondary);
            font-size: 0.85rem;
            cursor: pointer;
            transition: all var(--transition-fast);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }

        .input-btn:hover {
            background: var(--bg-hover);
            border-color: var(--border-hover);
            transform: translateY(-1px);
            color: var(--text-primary);
        }
        
        .input-btn:active {
            transform: translateY(0);
        }

        /* Modern Voice Button */
        .voice-btn {
            width: 42px;
            height: 42px;
            padding: 0;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: var(--radius-full);
            color: var(--text-secondary);
            cursor: pointer;
            transition: all var(--transition-normal);
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: hidden;
        }
        
        .voice-btn::before {
            content: '';
            position: absolute;
            inset: 0;
            background: rgba(255, 255, 255, 0.1);
            opacity: 0;
            transition: opacity var(--transition-fast);
            border-radius: inherit;
        }
        
        .voice-btn:hover {
            transform: scale(1.08);
            border-color: rgba(255, 255, 255, 0.4);
            color: var(--text-primary);
            box-shadow: 0 4px 20px rgba(255, 255, 255, 0.1);
        }
        
        .voice-btn:hover::before {
            opacity: 1;
        }
        
        .voice-btn.recording {
            background: rgba(239, 68, 68, 0.15);
            border-color: rgba(239, 68, 68, 0.5);
            color: #ef4444;
            animation: recording 1.5s ease-in-out infinite;
        }
        
        .voice-btn svg {
            width: 20px;
            height: 20px;
            position: relative;
            z-index: 1;
        }
        
        /* Attach Button */
        .attach-btn {
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            padding: 0.625rem 1rem;
            border-radius: var(--radius-md);
            color: var(--text-secondary);
            font-size: 0.85rem;
            cursor: pointer;
            transition: all var(--transition-fast);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .attach-btn:hover {
            background: var(--bg-hover);
            border-color: var(--border-hover);
            color: var(--text-primary);
            transform: translateY(-1px);
        }
        
        .attach-btn svg {
            width: 16px;
            height: 16px;
        }

        .send-btn {
            padding: 0.625rem 1.5rem;
            background: var(--accent);
            border: none;
            border-radius: var(--radius-md);
            color: var(--bg-primary);
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            transition: all var(--transition-fast);
            display: flex;
            align-items: center;
            gap: 0.5rem;
            box-shadow: 0 4px 15px rgba(255, 255, 255, 0.1);
        }

        .send-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 25px rgba(255, 255, 255, 0.15);
            opacity: 0.9;
        }
        
        .send-btn:active {
            transform: translateY(0);
        }
        
        .send-btn svg {
            width: 16px;
            height: 16px;
        }

        .input-hint {
            display: flex;
            justify-content: space-between;
            margin-top: 0.5rem;
            font-size: 0.7rem;
            color: var(--text-muted);
        }

        .char-count {
            font-family: monospace;
        }

        /* Tabs */
        .tab-content {
            display: none;
            flex: 1;
            overflow-y: auto;
        }

        .tab-content.active {
            display: flex;
            flex-direction: column;
        }

        /* Grid layouts for other tabs */
        .content-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            padding: 1.5rem;
            animation: fadeIn 0.5s ease-out;
        }

        .card {
            background: var(--bg-card);
            backdrop-filter: var(--glass-blur);
            -webkit-backdrop-filter: var(--glass-blur);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            overflow: hidden;
            transition: all var(--transition-normal);
            animation: fadeInUp 0.5s ease-out;
        }
        
        .card:hover {
            border-color: var(--border-hover);
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }

        .card-header {
            padding: 1rem 1.25rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .card-title {
            font-size: 0.7rem;
            font-weight: 600;
            letter-spacing: 1px;
            text-transform: uppercase;
            color: var(--text-muted);
        }

        .card-body {
            padding: 1.25rem;
        }

        /* Stats */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            padding: 1.5rem;
        }

        .stat-card {
            background: var(--bg-card);
            backdrop-filter: var(--glass-blur);
            -webkit-backdrop-filter: var(--glass-blur);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 1.5rem;
            text-align: center;
            transition: all var(--transition-normal);
            animation: fadeInUp 0.5s ease-out backwards;
            position: relative;
            overflow: hidden;
        }
        
        .stat-card::before {
            content: '';
            position: absolute;
            inset: 0;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.03) 0%, rgba(255, 255, 255, 0.01) 100%);
            opacity: 0;
            transition: opacity var(--transition-fast);
        }
        
        .stat-card:hover {
            border-color: rgba(255, 255, 255, 0.2);
            transform: translateY(-4px);
            box-shadow: 0 8px 30px rgba(255, 255, 255, 0.05);
        }
        
        .stat-card:hover::before {
            opacity: 1;
        }
        
        .stat-card:nth-child(1) { animation-delay: 0.1s; }
        .stat-card:nth-child(2) { animation-delay: 0.2s; }
        .stat-card:nth-child(3) { animation-delay: 0.3s; }
        .stat-card:nth-child(4) { animation-delay: 0.4s; }

        .stat-value {
            font-size: 2rem;
            font-weight: 700;
            font-family: monospace;
        }

        .stat-label {
            font-size: 0.7rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 0.5rem;
        }

        /* Forms */
        .form-group {
            margin-bottom: 1.5rem;
        }

        .form-label {
            display: block;
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.5rem;
        }

        .form-input,
        .form-select,
        .form-textarea {
            width: 100%;
            padding: 0.875rem 1rem;
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            color: var(--text-primary);
            font-size: 0.9rem;
            font-family: inherit;
        }

        .form-input:focus,
        .form-select:focus,
        .form-textarea:focus {
            outline: none;
            border-color: var(--border-hover);
        }

        .form-textarea {
            min-height: 100px;
            resize: vertical;
        }

        .btn {
            padding: 0.875rem 1.5rem;
            border: none;
            border-radius: var(--radius-md);
            font-weight: 600;
            font-size: 0.875rem;
            cursor: pointer;
            transition: all 0.2s;
        }

        .btn-primary {
            background: var(--accent);
            color: var(--bg-primary);
            width: 100%;
        }

        .btn-primary:hover {
            opacity: 0.9;
        }

        .btn-secondary {
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            color: var(--text-primary);
        }

        /* Empty state */
        .empty-state {
            text-align: center;
            padding: 3rem;
            color: var(--text-muted);
        }

        /* Loading */
        .loading {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.8);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }

        .loading.active {
            display: flex;
        }

        .spinner {
            width: 40px;
            height: 40px;
            border: 2px solid var(--border);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to {
                transform: rotate(360deg);
            }
        }

        /* Toast */
        .toast-container {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            z-index: 1001;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .toast {
            background: var(--bg-card);
            backdrop-filter: var(--glass-blur);
            -webkit-backdrop-filter: var(--glass-blur);
            border: 1px solid var(--border);
            border-left: 3px solid;
            border-image: linear-gradient(135deg, rgba(255, 255, 255, 0.8) 0%, rgba(255, 255, 255, 0.4) 100%) 1;
            border-radius: var(--radius-md);
            padding: 1rem 1.5rem;
            font-size: 0.875rem;
            box-shadow: var(--shadow-lg);
            animation: toastSlideIn 0.4s cubic-bezier(0.16, 1, 0.3, 1);
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        
        .toast.success {
            border-image: linear-gradient(135deg, #22c55e 0%, #10b981 100%) 1;
        }
        
        .toast.error {
            border-image: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) 1;
        }
        
        .toast.warning {
            border-image: linear-gradient(135deg, #eab308 0%, #f59e0b 100%) 1;
        }

        @keyframes toastSlideIn {
            from {
                opacity: 0;
                transform: translateX(100%) scale(0.9);
            }
            to {
                opacity: 1;
                transform: translateX(0) scale(1);
            }
        }

        /* Graph Layout */
        /* Premium Graph Aesthetics */
        .graph-layout {
            display: grid;
            grid-template-columns: 1fr 320px;
            gap: 1.5rem;
            padding: 1rem 1.5rem;
            height: calc(100vh - 120px);
            background: radial-gradient(circle at top right, rgba(15, 15, 25, 1) 0%, rgba(5, 5, 10, 1) 100%);
        }

        .graph-main {
            position: relative;
            background: rgba(10, 10, 15, 0.4);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: inset 0 0 40px rgba(0, 0, 0, 0.5);
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        .graph-viewport {
            flex: 1;
            width: 100%;
            height: 100%;
            cursor: crosshair;
        }

        .glass-panel {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 1.25rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            transition: transform 0.3s ease;
        }

        .detail-panel { min-height: 200px; }
        .stats-panel { margin-top: 1rem; }
        .legend-panel { margin-top: 1rem; }

        .panel-header h3 {
            margin: 0 0 1rem 0;
            font-size: 1rem;
            font-weight: 600;
            color: #fff;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .panel-content .empty-selection {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 150px;
            color: rgba(255, 255, 255, 0.3);
            text-align: center;
        }

        .empty-selection i { font-size: 2rem; margin-bottom: 1rem; opacity: 0.5; }

        .stat-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.75rem;
        }

        .stat-box {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 0.75rem;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .stat-box .label { font-size: 0.65rem; color: rgba(255, 255, 255, 0.5); text-transform: uppercase; }
        .stat-box .value { font-size: 1.1rem; font-weight: 700; color: #fff; margin-top: 0.25rem; }

        .legend-list { display: flex; flex-direction: column; gap: 0.75rem; }
        .legend-item { display: flex; align-items: center; gap: 0.75rem; border-bottom: 1px solid rgba(255, 255, 255, 0.05); padding-bottom: 0.5rem; font-size: 0.85rem; color: rgba(255, 255, 255, 0.8); }
        .legend-item:last-child { border-bottom: none; }

        .dot { width: 10px; height: 10px; border-radius: 50%; box-shadow: 0 0 10px currentColor; }
        .dot.decision { background: #ff0088; color: #ff0088; }
        .dot.outcome { background: #00d4ff; color: #00d4ff; }
        .dot.milestone { background: #00ffbb; color: #00ffbb; }
        .dot.factor { background: #ffcc00; color: #ffcc00; }

        .graph-overlay-controls {
            position: absolute;
            bottom: 1.5rem;
            left: 1.5rem;
            z-index: 100;
        }

        .zoom-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .glass-btn {
            width: 42px;
            height: 42px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            color: #fff;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            backdrop-filter: blur(8px);
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .glass-btn:hover { background: rgba(255, 255, 255, 0.15); transform: translateY(-2px); border-color: rgba(255, 255, 255, 0.2); }

        .graph-node-tooltip {
            position: absolute;
            padding: 1rem;
            background: rgba(15, 15, 25, 0.95);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 12px;
            color: #fff;
            font-size: 0.85rem;
            pointer-events: none;
            z-index: 1000;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.6);
            backdrop-filter: blur(4px);
            max-width: 240px;
        }

        .btn-neon {
            background: transparent;
            color: #fff;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 0.6rem 1.2rem;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 500;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.6rem;
            transition: all 0.3s;
        }

        .btn-neon:hover {
            border-color: #00ffbb;
            box-shadow: 0 0 15px rgba(0, 255, 187, 0.3);
            text-shadow: 0 0 5px rgba(255, 255, 255, 0.5);
        }
        
        .header-actions {
            display: flex;
            gap: 0.5rem;
        }
        
        .btn-secondary {
            padding: 0.5rem 1rem;
            border-radius: var(--radius-md);
            border: 1px solid var(--border);
            background: var(--bg-elevated);
            color: var(--text-primary);
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .btn-secondary:hover {
            background: var(--bg-hover);
            border-color: var(--border-hover);
        }
        
        @media (max-width: 900px) {
            .graph-layout {
                grid-template-columns: 1fr;
            }
            .graph-sidebar {
                display: none;
            }
        }

        /* Neural Pulse Animation */
        @keyframes neural-pulse {
            0% { opacity: 0.4; filter: blur(2px); }
            50% { opacity: 0.8; filter: blur(4px); }
            100% { opacity: 0.4; filter: blur(2px); }
        }

        .node-group:hover circle:first-child {
            animation: neural-pulse 2s infinite ease-in-out;
            stroke-width: 4px;
        }

        /* Graph nodes - no transitions to prevent shaking during physics */
        .node-group circle {
            transition: stroke 0.3s, stroke-width 0.3s, fill 0.3s;
        }

        /* Typing indicator */
        .typing-indicator {
            display: flex;
            gap: 4px;
            padding: 1rem;
            background: var(--bg-card);
            border-radius: var(--radius-lg);
            width: fit-content;
        }

        .typing-dot {
            width: 8px;
            height: 8px;
            background: var(--text-muted);
            border-radius: 50%;
            animation: typing 1.4s ease-in-out infinite;
        }

        .typing-dot:nth-child(2) {
            animation-delay: 0.2s;
        }

        .typing-dot:nth-child(3) {
            animation-delay: 0.4s;
        }

        @keyframes typing {

            0%,
            60%,
            100% {
                transform: translateY(0);
            }

            30% {
                transform: translateY(-6px);
            }
        }

        /* Journal items */
        .journal-item {
            padding: 1rem;
            background: var(--bg-elevated);
            border-radius: var(--radius-md);
            margin-bottom: 0.75rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
            border: 1px solid var(--border);
            transition: all 0.2s;
        }

        .journal-item:hover {
            transform: translateX(4px);
            border-color: var(--border-hover);
        }

        .journal-type {
            font-size: 0.7rem;
            padding: 0.25rem 0.5rem;
            background: var(--bg-hover);
            border-radius: 4px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .journal-regret {
            font-weight: 700;
            font-family: monospace;
        }

        /* Slider */
        .slider {
            width: 100%;
            height: 6px;
            border-radius: 3px;
            background: var(--bg-hover);
            -webkit-appearance: none;
            appearance: none;
            cursor: pointer;
        }

        .slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            background: var(--accent);
            border-radius: 50%;
            cursor: pointer;
        }

        .slider-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.5rem;
        }

        .slider-value {
            font-family: monospace;
            font-weight: 600;
        }

        /* Metrics */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
            margin: 1rem 0;
        }

        .metric-card {
            background: var(--bg-elevated);
            padding: 1rem;
            border-radius: var(--radius-md);
            text-align: center;
            border: 1px solid var(--border);
        }

        .metric-value {
            font-size: 1.5rem;
            font-weight: 700;
            font-family: monospace;
        }

        .metric-label {
            font-size: 0.65rem;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-top: 0.25rem;
        }

        .regret-meter {
            height: 8px;
            background: var(--bg-hover);
            border-radius: 4px;
            overflow: hidden;
            margin: 1rem 0;
        }

        .regret-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s;
        }

        .regret-fill.low {
            background: var(--success);
        }

        .regret-fill.moderate {
            background: var(--warning);
        }

        .regret-fill.high {
            background: var(--danger);
        }

        .recommendations h4 {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
            margin-bottom: 0.75rem;
        }

        .rec-item {
            padding: 0.75rem 0;
            border-bottom: 1px solid var(--border);
            font-size: 0.9rem;
            color: var(--text-secondary);
        }

        .rec-item:last-child {
            border-bottom: none;
        }

        /* Floating animations */
        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-8px); }
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes glow {
            0%, 100% { box-shadow: 0 0 5px rgba(255,255,255,0.1); }
            50% { box-shadow: 0 0 20px rgba(255,255,255,0.15); }
        }

        .card {
            animation: fadeInUp 0.5s ease-out;
        }

        .stat-card {
            animation: fadeInUp 0.5s ease-out;
            transition: all 0.3s ease;
        }

        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }

        .nav-btn {
            transition: all 0.2s ease;
        }

        .nav-btn:hover {
            transform: translateX(5px);
        }

        .new-conversation-btn {
            transition: all 0.3s ease;
            animation: glow 3s ease-in-out infinite;
        }

        .new-conversation-btn:hover {
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 8px 25px rgba(255,255,255,0.2);
        }

        .message-group {
            animation: fadeInUp 0.3s ease-out;
        }

        .avatar {
            transition: all 0.2s ease;
        }

        .avatar:hover {
            transform: scale(1.1);
        }

        .brand {
            animation: fadeInUp 0.5s ease-out;
        }

        .connection-status {
            animation: fadeInUp 0.5s ease-out;
        }

        /* Graph nodes - no transitions to prevent shaking */
        .graph-node {
            /* Static - no transitions */
        }

        @media (max-width: 768px) {
            .sidebar {
                display: none;
            }

            .main {
                margin-left: 0;
            }

            .stats-grid {
                grid-template-columns: repeat(2, 1fr);
            }

            .content-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>

<body>
    <div id="canvas-container" style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: 0; pointer-events: none; background: var(--bg-primary); transition: background 0.3s;"></div>
    <aside class="sidebar">
        <div class="sidebar-header">
            <div class="brand">CAREER DECISION AI</div>
            <div class="brand-sub">Professional Guidance Platform</div>
        </div>

        <button class="new-conversation-btn" onclick="clearChat()">
            <span>+</span> New Conversation
        </button>

        <div class="nav-section">
            <div class="nav-label">Workspace</div>
            <button class="nav-btn active" onclick="showTab('chat')" id="nav-chat">Chat</button>
            <button class="nav-btn" onclick="showTab('templates')" id="nav-templates">Templates</button>
            <button class="nav-btn" onclick="showTab('journal')" id="nav-journal">History</button>
        </div>

        <div class="nav-section">
            <div class="nav-label">Tools</div>
            <button class="nav-btn" onclick="showTab('graph')" id="nav-graph">Graph View</button>
            <button class="nav-btn" onclick="showTab('analytics')" id="nav-analytics">Analytics</button>
            <button class="nav-btn" onclick="showTab('resume')" id="nav-resume">Resume Analysis</button>
            <button class="nav-btn" onclick="showTab('simulate')" id="nav-simulate">Outcome Simulation</button>
            <button class="nav-btn" onclick="showTab('interview')" id="nav-interview">Interview Practice</button>
        </div>

        <div class="nav-section">
            <div class="nav-label">Advanced</div>
            <button class="nav-btn" onclick="showTab('goals')" id="nav-goals">Goals</button>
            <button class="nav-btn" onclick="showTab('opportunities')" id="nav-opportunities">Opportunities</button>
            <button class="nav-btn" onclick="showTab('mentor')" id="nav-mentor">Mentor Matching</button>
            <button class="nav-btn" onclick="showTab('privacy')" id="nav-privacy">Privacy</button>
        </div>

        <div class="nav-section">
            <div class="nav-label">Operations</div>
            <button class="nav-btn" onclick="showTab('monitoring')" id="nav-monitoring">System Status</button>
            <button class="nav-btn" onclick="showTab('calendar')" id="nav-calendar">Calendar</button>
            <button class="nav-btn" onclick="showTab('integrations')" id="nav-integrations">Integrations</button>
        </div>

        <div class="sidebar-footer">
            <div class="connection-status" style="justify-content: center;">
                <div class="status-dot" id="statusDot"></div>
                <div class="status-mode" id="statusMode" style="font-weight: 600;">Detecting...</div>
            </div>
            <button class="settings-btn" onclick="toggleSettings()">Settings</button>
        </div>
    </aside>

    <script>
        // Core Dashboard Navigation Logic
        window.userId = localStorage.getItem('userId') || ('user_' + Date.now());
        localStorage.setItem('userId', window.userId);
        window.journalDecisions = JSON.parse(localStorage.getItem('decisions') || '[]');

        function showTab(tabId) {
            console.log('Switching to tab:', tabId);
            if (tabId === 'analyze') tabId = 'templates';
            
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
            
            const tab = document.getElementById('tab-' + tabId);
            if (tab) tab.classList.add('active');
            
            const navBtn = document.getElementById('nav-' + tabId);
            if (navBtn) navBtn.classList.add('active');
            
            // Breadcrumb update
            const bc = document.querySelector('.breadcrumb-current');
            if (bc) {
                const labelMap = {
                    'chat': 'Career Counselor',
                    'templates': 'Decision Templates',
                    'journal': 'History',
                    'graph': 'Graph View',
                    'analytics': 'Analytics',
                    'resume': 'Resume Analysis',
                    'simulate': 'Outcome Simulation',
                    'interview': 'Interview Practice',
                    'goals': 'Career Goals',
                    'opportunities': 'Market Opportunities',
                    'mentor': 'Mentor Network',
                    'privacy': 'Privacy Settings',
                    'monitoring': 'System Status',
                    'calendar': 'Calendar',
                    'integrations': 'Integrations'
                };
                bc.textContent = labelMap[tabId] || tabId;
            }

            try {
                if (tabId === 'graph' && typeof loadGraph === 'function') loadGraph();
                if (tabId === 'analytics' && typeof loadAnalytics === 'function') loadAnalytics();
                if (tabId === 'journal' && typeof updateJournal === 'function') updateJournal();
                if (tabId === 'templates' && typeof loadTemplates === 'function') loadTemplates();
                if (tabId === 'goals' && typeof loadGoals === 'function') loadGoals();
                if (tabId === 'opportunities' && typeof loadOpportunities === 'function') {
                    loadOpportunities();
                    if (typeof loadMarketNews === 'function') loadMarketNews();
                }
                if (tabId === 'privacy' && typeof loadPrivacySettings === 'function') loadPrivacySettings();
                if (tabId === 'monitoring' && typeof loadMonitoring === 'function') setTimeout(loadMonitoring, 10);
                if (tabId === 'calendar' && typeof loadCalendar === 'function') setTimeout(loadCalendar, 10);
                if (tabId === 'mentor' && typeof loadMentors === 'function') { loadMentors(); loadConnectedMentors(); }
                if (tabId === 'integrations' && typeof loadKnowledge === 'function') loadKnowledge();
            } catch (e) {
                console.warn('Advanced loader failed for tab ' + tabId, e);
            }
        }

        function toggleSettings() {
            const modal = document.getElementById('settingsModal');
            if (modal) modal.style.display = modal.style.display === 'none' ? 'flex' : 'none';
        }

        async function clearChat() {
            if (!confirm('Clear this conversation?')) return;
            try {
                await fetch('/api/context/clear', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ user_id: userId })
                });
            } catch (e) {}
            showTab('chat');
            const messagesDiv = document.getElementById('chatMessages');
            if (messagesDiv) {
                messagesDiv.innerHTML = `<div class="message-group assistant"><div class="avatar ai">AI</div><div class="message-content"><div class="message-header"><span class="message-name">Career Counselor</span><span class="message-time">Just now</span></div><div class="message-bubble">Session cleared. How can I help you now?</div></div></div>`;
            }
        }
    </script>

    <main class="main">
        <!-- Chat Tab -->
        <div id="tab-chat" class="tab-content active">
            <div class="header">
                <div class="breadcrumb">
                    <span>Home</span>
                    <span>/</span>
                    <span class="breadcrumb-current">Career Counselor</span>
                </div>
            </div>

            <div class="chat-container">
                <div class="chat-messages" id="chatMessages">
                    <div class="message-group assistant">
                        <div class="avatar ai">AI</div>
                        <div class="message-content">
                            <div class="message-header">
                                <span class="message-name">Career Counselor</span>
                                <span class="message-time">Today</span>
                            </div>
                            <div class="message-bubble">Hello, I'm your career counselor. Share a decision you're
                                considering, or ask me anything about your career path.</div>
                        </div>
                    </div>
                </div>

                <div class="chat-input-area">
                    <!-- File Upload Input (Hidden) -->
                    <input type="file" id="fileInput" style="display: none;" onchange="handleFileSelect(event)">
                    <input type="file" id="videoInput" style="display: none;" accept=".mp4,.webm,.avi,.mov,.mkv,.flv,.wmv" onchange="handleVideoSelect(event)">
                    
                    <!-- File Preview Area -->
                    <div id="filePreview" class="file-preview" style="display: none; padding: 0.5rem 1.5rem; gap: 0.5rem; flex-wrap: wrap;"></div>
                    
                    <!-- URL Input Modal -->
                    <div id="urlInputModal" style="display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.7); z-index: 1000; align-items: center; justify-content: center;">
                        <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 1.5rem; max-width: 500px; width: 90%; box-shadow: var(--shadow-lg);">
                            <h3 style="margin-bottom: 1rem; color: var(--text-primary);">Add URL (YouTube, Articles, etc.)</h3>
                            <input type="text" id="urlInput" placeholder="https://youtube.com/watch?v=... or any URL" style="width: 100%; padding: 0.75rem; background: var(--bg-elevated); border: 1px solid var(--border); border-radius: var(--radius-sm); color: var(--text-primary); margin-bottom: 1rem; font-family: inherit;">
                            <div style="display: flex; gap: 0.75rem;">
                                <button class="btn btn-primary" onclick="submitUrlInput()" style="flex: 1; margin: 0;">Add</button>
                                <button class="btn btn-secondary" onclick="closeUrlModal()" style="flex: 1; margin: 0; background: var(--bg-elevated); border: 1px solid var(--border); color: var(--text-primary);">Cancel</button>
                            </div>
                        </div>
                    </div>
                    
                    <div class="input-container">
                        <input type="text" class="chat-input" id="chatInput" placeholder="Ask about your career path..."
                            maxlength="2000">
                        <div class="input-actions">
                            <button class="voice-btn" id="voiceBtn" onclick="toggleVoiceInput()" title="Voice input">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                    <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
                                    <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                                    <line x1="12" x2="12" y1="19" y2="22"/>
                                </svg>
                            </button>
                            
                            <!-- Attach Dropdown -->
                            <div style="position: relative; display: inline-block;">
                                <button class="attach-btn" onclick="toggleAttachMenu()" title="Attach file, video, or URL">
                                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                        <path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 8.57a2 2 0 0 1-2.83-2.83l8.49-8.48"/>
                                    </svg>
                                    <span>Attach</span>
                                </button>
                                <div id="attachMenu" style="display: none; position: absolute; bottom: 100%; right: 0; background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius-md); box-shadow: var(--shadow-lg); z-index: 100; min-width: 200px; overflow: hidden;">
                                    <button style="width: 100%; padding: 0.75rem 1rem; text-align: left; background: transparent; border: none; color: var(--text-primary); cursor: pointer; border-bottom: 1px solid var(--border); font-size: 0.9rem; transition: background 0.15s;" onmouseover="this.style.background='var(--bg-hover)'" onmouseout="this.style.background='transparent'" onclick="document.getElementById('fileInput').click(); closeAttachMenu()">
                                        <svg style="width: 16px; height: 16px; display: inline-block; margin-right: 0.5rem; vertical-align: -2px;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                                        Upload Files
                                    </button>
                                    <button style="width: 100%; padding: 0.75rem 1rem; text-align: left; background: transparent; border: none; color: var(--text-primary); cursor: pointer; border-bottom: 1px solid var(--border); font-size: 0.9rem; transition: background 0.15s;" onmouseover="this.style.background='var(--bg-hover)'" onmouseout="this.style.background='transparent'" onclick="document.getElementById('videoInput').click(); closeAttachMenu()">
                                        <svg style="width: 16px; height: 16px; display: inline-block; margin-right: 0.5rem; vertical-align: -2px;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>
                                        Upload Video
                                    </button>
                                    <button style="width: 100%; padding: 0.75rem 1rem; text-align: left; background: transparent; border: none; color: var(--text-primary); cursor: pointer; font-size: 0.9rem; transition: background 0.15s;" onmouseover="this.style.background='var(--bg-hover)'" onmouseout="this.style.background='transparent'" onclick="openUrlModal(); closeAttachMenu()">
                                        <svg style="width: 16px; height: 16px; display: inline-block; margin-right: 0.5rem; vertical-align: -2px;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
                                        Add URL/YouTube
                                    </button>
                                </div>
                            </div>
                            
                            <button class="send-btn" onclick="sendMessage()">
                                <span>Send</span>
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                    <line x1="22" x2="11" y1="2" y2="13"/>
                                    <polygon points="22 2 15 22 11 13 2 9 22 2"/>
                                </svg>
                            </button>
                        </div>

                    </div>
                    <div class="input-hint">
                        <span>Press Enter to send • Shift+Enter for new line</span>
                        <span class="char-count"><span id="charCount">0</span> / 2000</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Other tabs will be injected here by JavaScript -->
        <div id="tab-templates" class="tab-content">
            <div class="header">
                <div class="breadcrumb">
                    <span>Home</span><span>/</span><span class="breadcrumb-current">Templates</span>
                </div>
            </div>
            <div class="content-grid" style="grid-template-columns: 1fr 1fr; gap: 1.5rem;">
                <div>
                    <h2 style="padding: 1.5rem; font-size: 1.25rem;">Decision Templates</h2>
                    <p style="padding: 0 1.5rem; color: var(--text-secondary);">Choose a template for guided decision-making</p>
                    <div class="stats-grid" id="templateGrid" style="grid-template-columns: 1fr;"></div>
                </div>
                <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius-lg); display: none;" id="templateDetails"></div>
            </div>
        </div>

        <div id="tab-journal" class="tab-content">
            <div class="header">
                <div class="breadcrumb">
                    <span>Home</span><span>/</span><span class="breadcrumb-current">History</span>
                </div>
            </div>
            <div class="content-grid">
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Decision History</div>
                        <span style="color:var(--text-muted);font-size:0.75rem;" id="journalCount">0 entries</span>
                    </div>
                    <div class="card-body" id="journalList" style="max-height:500px;overflow-y:auto;">
                        <div class="empty-state">No decisions recorded yet</div>
                    </div>
                </div>
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Decision Details</div>
                    </div>
                    <div class="card-body" id="journalDetails">
                        <div class="empty-state">Select a decision to view details</div>
                    </div>
                </div>
            </div>
        </div>

        <div id="tab-graph" class="tab-content">
            <div class="header">
                <div class="breadcrumb">
                    <span>Home</span><span>/</span><span class="breadcrumb-current">Decision Network</span>
                </div>
                <div class="header-actions">
                    <button class="btn-neon" onclick="resetGraphZoom()">
                        <i class="fas fa-expand-arrows-alt"></i> Reset View
                    </button>
                    <button class="btn-neon" onclick="loadGraph()">
                        <i class="fas fa-sync-alt"></i> Refresh
                    </button>
                </div>
            </div>
            
            <div class="graph-layout">
                <div class="graph-main">
                    <div class="graph-viewport" id="graphContainer">
                        <div class="graph-loading">Initializing Neural Mapping...</div>
                    </div>
                    
                    <div class="graph-overlay-controls">
                        <div class="zoom-group">
                            <button class="glass-btn" onclick="zoomGraph(1.3)" title="Zoom In"><i class="fas fa-plus"></i></button>
                            <button class="glass-btn" onclick="zoomGraph(0.7)" title="Zoom Out"><i class="fas fa-minus"></i></button>
                            <button class="glass-btn" onclick="resetGraphZoom()" title="Reset View"><i class="fas fa-compress"></i></button>
                        </div>
                    </div>

                    <!-- Floating Tooltip -->
                    <div id="graphTooltip" class="graph-node-tooltip" style="display: none;"></div>
                </div>

                <div class="graph-sidebar">
                    <div class="glass-panel detail-panel" id="graphDetailPanel">
                        <div class="panel-header">
                            <h3><i class="fas fa-project-diagram"></i> Node Insights</h3>
                        </div>
                        <div id="nodeDetailContent" class="panel-content">
                            <div class="empty-selection">
                                <i class="fas fa-mouse-pointer"></i>
                                <p>Select a node to view its neural connections and career impact</p>
                            </div>
                        </div>
                    </div>

                    <div class="glass-panel stats-panel">
                        <h3><i class="fas fa-chart-network"></i> Statistics</h3>
                        <div class="stat-grid">
                            <div class="stat-box">
                                <span class="label">Nodes</span>
                                <span class="value" id="graphNodeCount">0</span>
                            </div>
                            <div class="stat-box">
                                <span class="label">Links</span>
                                <span class="value" id="graphEdgeCount">0</span>
                            </div>
                            <div class="stat-box">
                                <span class="label">Density</span>
                                <span class="value" id="graphDensity">0.0</span>
                            </div>
                        </div>
                    </div>

                    <div class="glass-panel legend-panel">
                        <h3><i class="fas fa-key"></i> Neural Legend</h3>
                        <div class="legend-list">
                            <div class="legend-item"><span class="dot decision"></span> Decision Point</div>
                            <div class="legend-item"><span class="dot outcome"></span> Career Outcome</div>
                            <div class="legend-item"><span class="dot milestone"></span> Milestone</div>
                            <div class="legend-item"><span class="dot factor"></span> Influence Factor</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div id="tab-analytics" class="tab-content">
            <div class="header">
                <div class="breadcrumb">
                    <span>Home</span><span>/</span><span class="breadcrumb-current">Analytics</span>
                </div>
            </div>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value" id="totalDecisions">0</div>
                    <div class="stat-label">Total Decisions</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="avgRegret">0%</div>
                    <div class="stat-label">Avg Regret</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="commonType">-</div>
                    <div class="stat-label">Most Common</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="riskProfile">-</div>
                    <div class="stat-label">Risk Profile</div>
                </div>
            </div>
            <div style="padding: 0 1.5rem;">
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Regret Over Time</div>
                    </div>
                    <div class="card-body">
                        <div id="regretChartContainer" style="height:300px;">
                            <div class="empty-state" id="regretChartEmpty">Make decisions to see trends</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Goals Tab -->
        <div id="tab-goals" class="tab-content">
            <div class="header">
                <div class="breadcrumb">
                    <span>Home</span><span>/</span><span class="breadcrumb-current">Career Goals</span>
                </div>
            </div>
            <div class="content-grid">
                <div class="card" style="grid-column: span 2;">
                    <div class="card-header">
                        <div class="card-title">Create New Goal</div>
                    </div>
                    <div class="card-body">
                        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1rem;">
                            <input type="text" id="goalTitle" placeholder="Goal Title (e.g., Become Tech Lead)" 
                                   style="padding:0.75rem;background:var(--bg-elevated);border:1px solid var(--border);border-radius:8px;color:var(--text-primary);">
                            <select id="goalCategory" style="padding:0.75rem;background:var(--bg-elevated);border:1px solid var(--border);border-radius:8px;color:var(--text-primary);">
                                <option value="leadership">Leadership</option>
                                <option value="technical">Technical Skills</option>
                                <option value="career_growth">Career Growth</option>
                                <option value="compensation">Compensation</option>
                                <option value="work_life_balance">Work-Life Balance</option>
                            </select>
                        </div>
                        <textarea id="goalDescription" rows="2" placeholder="Describe your goal..."
                                  style="width:100%;padding:0.75rem;background:var(--bg-elevated);border:1px solid var(--border);border-radius:8px;color:var(--text-primary);resize:none;margin-bottom:1rem;"></textarea>
                        <button onclick="createGoal()" style="padding:0.75rem 2rem;background:var(--accent);color:var(--bg-primary);border:none;border-radius:8px;font-weight:600;cursor:pointer;">Create Goal</button>
                    </div>
                </div>
                <div class="card" style="grid-column: span 2;">
                    <div class="card-header">
                        <div class="card-title">Your Goals</div>
                    </div>
                    <div class="card-body" id="goalsList" style="max-height:400px;overflow-y:auto;">
                        <div class="empty-state">Create your first career goal above</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Opportunities Tab -->
        <div id="tab-opportunities" class="tab-content">
            <div class="header">
                <div class="breadcrumb">
                    <span>Home</span><span>/</span><span class="breadcrumb-current">Opportunity Scout</span>
                </div>
            </div>
            <div class="content-grid">
                <div class="card" style="grid-column: span 2;">
                    <div class="card-header" style="display:flex;justify-content:space-between;align-items:center;">
                        <div class="card-title">Personalized Opportunities</div>
                        <button onclick="scanOpportunities()" style="padding:0.5rem 1rem;background:var(--accent);color:var(--bg-primary);border:none;border-radius:8px;font-weight:600;cursor:pointer;font-size:0.85rem;">Scan Now</button>
                    </div>
                    <div class="card-body" id="opportunitiesList" style="max-height:500px;overflow-y:auto;">
                        <div class="empty-state">Click "Scan Now" to discover opportunities matching your profile</div>
                    </div>
                </div>
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Quick Stats</div>
                    </div>
                    <div class="card-body" id="opportunityStats">
                        <div style="display:grid;gap:1rem;">
                            <div style="padding:1rem;background:var(--bg-elevated);border-radius:8px;">
                                <div style="font-size:2rem;font-weight:700;color:var(--accent);" id="totalOpportunities">0</div>
                                <div style="color:var(--text-secondary);font-size:0.85rem;">Total Opportunities</div>
                            </div>
                            <div style="padding:1rem;background:var(--bg-elevated);border-radius:8px;">
                                <div style="font-size:2rem;font-weight:700;color:var(--success);" id="highMatchCount">0</div>
                                <div style="color:var(--text-secondary);font-size:0.85rem;">High Match Score</div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Market Insights</div>
                    </div>
                    <div class="card-body" id="marketInsights">
                        <div id="newsContainer" style="display:grid;gap:0.75rem;">
                            <div class="empty-state" style="font-size:0.85rem;">Loading market news...</div>
                        </div>
                    </div>
                </div>
                <div class="card" style="grid-column: span 3;">
                    <div class="card-header">
                        <div class="card-title">Active Applications</div>
                    </div>
                    <div class="card-body">
                        <div id="applicationsList" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1rem;">
                            <div class="empty-state">No active applications track. Start by applying to an opportunity!</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Privacy Tab -->
        <div id="tab-privacy" class="tab-content">
            <div class="header">
                <div class="breadcrumb">
                    <span>Home</span><span>/</span><span class="breadcrumb-current">Privacy Center</span>
                </div>
            </div>
            <div class="content-grid">
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Data Consent</div>
                    </div>
                    <div class="card-body" id="consentPanel">
                        <div style="display:grid;gap:1rem;">
                            <label style="display:flex;align-items:center;gap:1rem;padding:1rem;background:var(--bg-elevated);border-radius:8px;cursor:pointer;">
                                <input type="checkbox" id="consentDataCollection" onchange="updateConsent('data_collection', this.checked)">
                                <div>
                                    <div style="font-weight:600;">Data Collection</div>
                                    <div style="color:var(--text-secondary);font-size:0.85rem;">Allow collection of usage data to improve the service</div>
                                </div>
                            </label>
                            <label style="display:flex;align-items:center;gap:1rem;padding:1rem;background:var(--bg-elevated);border-radius:8px;cursor:pointer;">
                                <input type="checkbox" id="consentAnalytics" onchange="updateConsent('analytics', this.checked)">
                                <div>
                                    <div style="font-weight:600;">Analytics</div>
                                    <div style="color:var(--text-secondary);font-size:0.85rem;">Help improve predictions through anonymized analytics</div>
                                </div>
                            </label>
                            <label style="display:flex;align-items:center;gap:1rem;padding:1rem;background:var(--bg-elevated);border-radius:8px;cursor:pointer;">
                                <input type="checkbox" id="consentGlobalInsights" onchange="updateConsent('global_insights', this.checked)">
                                <div>
                                    <div style="font-weight:600;">Global Insights</div>
                                    <div style="color:var(--text-secondary);font-size:0.85rem;">Contribute anonymized data to help others make better decisions</div>
                                </div>
                            </label>
                        </div>
                    </div>
                </div>
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Your Rights</div>
                    </div>
                    <div class="card-body">
                        <div style="display:grid;gap:0.75rem;">
                            <button onclick="requestDataExport()" style="padding:0.75rem 1rem;background:var(--bg-elevated);border:1px solid var(--border);border-radius:8px;color:var(--text-primary);cursor:pointer;text-align:left;">
                                <div style="font-weight:600;">Export My Data</div>
                                <div style="color:var(--text-secondary);font-size:0.8rem;">Download all your data in a portable format</div>
                            </button>
                            <button onclick="requestAccountDeletion()" style="padding:0.75rem 1rem;background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);border-radius:8px;color:#ef4444;cursor:pointer;text-align:left;">
                                <div style="font-weight:600;">Delete My Account</div>
                                <div style="color:rgba(239,68,68,0.8);font-size:0.8rem;">Permanently remove all your data</div>
                            </button>
                        </div>
                    </div>
                </div>
                <div class="card" style="grid-column: span 2;">
                    <div class="card-header">
                        <div class="card-title">Data Retention Policy</div>
                    </div>
                    <div class="card-body" id="retentionPolicy">
                        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1rem;">
                            <div style="padding:1rem;background:var(--bg-elevated);border-radius:8px;">
                                <div style="font-weight:600;">Profile Data</div>
                                <div style="color:var(--text-secondary);font-size:0.85rem;">Kept for 3 years</div>
                            </div>
                            <div style="padding:1rem;background:var(--bg-elevated);border-radius:8px;">
                                <div style="font-weight:600;">Decision History</div>
                                <div style="color:var(--text-secondary);font-size:0.85rem;">Kept for 5 years</div>
                            </div>
                            <div style="padding:1rem;background:var(--bg-elevated);border-radius:8px;">
                                <div style="font-weight:600;">Analytics Data</div>
                                <div style="color:var(--text-secondary);font-size:0.85rem;">Kept for 2 years</div>
                            </div>
                            <div style="padding:1rem;background:var(--bg-elevated);border-radius:8px;">
                                <div style="font-weight:600;">Bias Detection</div>
                                <div style="color:var(--text-secondary);font-size:0.85rem;">Kept for 2 years</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <!-- Resume Analysis Tab -->
        <div id="tab-resume" class="tab-content">
            <div class="header">
                <div class="breadcrumb">
                    <span>Home</span><span>/</span><span class="breadcrumb-current">Resume Analysis</span>
                </div>
            </div>
            <div class="content-grid">
                <div class="card" style="grid-column: span 2;">
                    <div class="card-header" style="display:flex;justify-content:space-between;align-items:center;">
                        <div class="card-title">Resume Parser</div>
                        <input type="file" id="resumeUpload" style="display:none;" accept=".pdf,.docx,.txt" onchange="uploadResume(event)">
                        <button onclick="document.getElementById('resumeUpload').click()" style="padding:0.5rem 1rem;background:var(--accent);color:var(--bg-primary);border:none;border-radius:8px;font-weight:600;cursor:pointer;">Upload Resume</button>
                    </div>
                    <div class="card-body">
                        <div id="resumeAnalysisResult" style="min-height:200px;display:flex;align-items:center;justify-content:center;flex-direction:column;text-align:center;">
                            <div class="empty-state">Upload your resume to analyze skills and experience</div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Skill Gaps</div>
                    </div>
                    <div class="card-body">
                        <div style="margin-bottom:1rem;">
                            <label style="display:block;font-size:0.85rem;color:var(--text-secondary);margin-bottom:0.5rem;">Target Role</label>
                            <select id="targetRoleSelect" onchange="updateSkillGaps()" style="width:100%;padding:0.75rem;background:var(--bg-elevated);border:1px solid var(--border);border-radius:8px;color:var(--text-primary);">
                                <option value="software engineer">Software Engineer</option>
                                <option value="product manager">Product Manager</option>
                                <option value="data scientist">Data Scientist</option>
                                <option value="frontend developer">Frontend Developer</option>
                                <option value="devops engineer">DevOps Engineer</option>
                            </select>
                        </div>
                        <div id="skillGapsList">
                            <div class="empty-state">Upload resume first</div>
                        </div>
                    </div>
                </div>
                <div class="card" style="grid-column: span 3;">
                    <div class="card-header" style="display:flex;justify-content:space-between;align-items:center;">
                        <div class="card-title">Personalized Career Roadmap</div>
                        <button onclick="generateRoadmap()" id="roadmapBtn" style="padding:0.4rem 0.8rem;background:var(--accent);color:var(--bg-primary);border:none;border-radius:6px;font-size:0.75rem;font-weight:700;display:none;">Generate Path</button>
                    </div>
                    <div class="card-body">
                        <div id="careerRoadmap" style="display:grid;grid-template-columns:repeat(3, 1fr);gap:1.5rem;">
                            <div class="empty-state" style="grid-column:span 3;">Run a skill gap analysis to generate your roadmap</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>


        <!-- Monitoring Tab -->
        <div id="tab-monitoring" class="tab-content">
            <div class="header">
                <div class="breadcrumb">
                    <span>Home</span><span>/</span><span class="breadcrumb-current">System Status</span>
                </div>
            </div>
            <div class="content-grid">
                <div class="card" style="grid-column: span 2;">
                    <div class="card-header">
                        <div class="card-title">System Health</div>
                    </div>
                    <div class="card-body">
                        <div class="stats-grid" id="systemHealthGrid" style="grid-template-columns: repeat(4, 1fr);">
                            <!-- Health stats will be injected here -->
                            <div class="stat-card">
                                <div class="stat-value" id="systemStatus">-</div>
                                <div class="stat-label">Overall Status</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value" id="uptime">-</div>
                                <div class="stat-label">Uptime</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value" id="errorRate">-</div>
                                <div class="stat-label">Error Rate</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value" id="avgLatency">-</div>
                                <div class="stat-label">Avg Latency</div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Active Alerts</div>
                    </div>
                    <div class="card-body" id="activeAlerts" style="max-height: 400px; overflow-y: auto;">
                        <div class="empty-state">No active alerts</div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Top Endpoints</div>
                    </div>
                    <div class="card-body" id="topEndpoints" style="max-height: 400px; overflow-y: auto;">
                        <div class="empty-state">Loading endpoint metrics...</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Calendar Tab -->
        <div id="tab-calendar" class="tab-content">
            <div class="header">
                <div class="breadcrumb">
                    <span>Home</span><span>/</span><span class="breadcrumb-current">Calendar</span>
                </div>
                <div style="display: flex; gap: 1rem; align-items: center;">
                    <button class="attach-btn" onclick="syncCalendar()">Sync to Google</button>
                    <button class="send-btn" onclick="showAddEventModal()">+ Add Event</button>
                </div>
            </div>
            <div class="content-grid">
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Today's Agenda</div>
                    </div>
                    <div class="card-body" id="todayAgenda">
                        <div class="empty-state">Loading agenda...</div>
                    </div>
                </div>

                <div class="card" style="grid-column: span 2; row-span: 2;">
                    <div class="card-header">
                        <div class="card-title">Upcoming Events</div>
                    </div>
                    <div class="card-body" id="upcomingEvents" style="max-height: 600px; overflow-y: auto;">
                        <div class="empty-state">Loading events...</div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Sync Status</div>
                    </div>
                    <div class="card-body" id="syncStatus">
                        <div class="empty-state">Checking status...</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Mentor Matching Tab -->
        <div id="tab-mentor" class="tab-content">
            <div class="header" style="display:flex;justify-content:space-between;align-items:center;gap:1rem;">
                    <div style="display:flex;align-items:center;gap:1rem;">
                        <div class="breadcrumb">
                            <span>Home</span><span>/</span><span class="breadcrumb-current">Mentor Matching</span>
                        </div>
                        <div style="display:flex;align-items:center;gap:0.5rem;">
                            <input id="youtubeQuery" type="search" placeholder="Search YouTube for mentoring..." style="padding:0.45rem 0.75rem;border-radius:8px;border:1px solid var(--border);background:var(--bg-elevated);color:var(--text-primary);width:320px;">
                            <button class="attach-btn" onclick="youtubeSearch()" title="Search YouTube">YouTube</button>
                        </div>
                    </div>
                    <div style="display:flex;align-items:center;gap:0.75rem;">
                        <button class="send-btn" onclick="loadMentors()">Find Matches</button>
                    </div>
                </div>
            <div class="content-grid">
                <div class="card" style="grid-column: span 2;">
                    <div class="card-header">
                        <div class="card-title">Recommended Mentors</div>
                    </div>
                    <div class="card-body">
                        <div id="mentorList" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1.25rem;">
                            <div class="empty-state">Finding current matches...</div>
                        </div>
                    </div>
                </div>
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">My Mentors</div>
                    </div>
                    <div class="card-body">
                        <div id="connectedMentors" style="display: flex; flex-direction: column; gap: 1rem;">
                            <div class="empty-state">No active connections</div>
                        </div>
                    </div>
                </div>
                
                <!-- YouTube Video Recommendations Section -->
                <div class="card" style="grid-column: span 2;">
                    <div class="card-header">
                        <div class="card-title" style="display: flex; align-items: center; gap: 0.5rem;">
                            <i class="fas fa-play-circle" style="color: #ff0000;"></i>
                            Learning Videos from Mentors
                        </div>
                    </div>
                    <div class="card-body">
                        <div style="display: grid; gap: 1rem; margin-bottom: 1rem; grid-template-columns: repeat(2, 1fr);">
                            <div>
                                <label class="form-label">Filter by Mentor</label>
                                <select id="videoMentorFilter" class="form-input" onchange="loadMentorVideos()">
                                    <option value="">All Mentors</option>
                                </select>
                            </div>
                            <div>
                                <label class="form-label">Video Category</label>
                                <select id="videoCategoryFilter" class="form-input" onchange="filterVideosByCategory()">
                                    <option value="">All Categories</option>
                                </select>
                            </div>
                        </div>
                        <div id="videoRecommendationsContainer" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 1rem; max-height: 500px; overflow-y: auto;">
                            <div class="empty-state" style="grid-column: span 2;">Connect with a mentor to see video recommendations</div>
                        </div>
                    </div>
                </div>
                
                <!-- Video Details Modal -->
                <div id="videoModal" class="modal" style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.85);z-index:2000;align-items:center;justify-content:center;">
                    <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:16px;width:90%;max-width:700px;display:flex;flex-direction:column;overflow:hidden;box-shadow:var(--shadow-lg);">
                        <div style="padding:1.25rem;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center;">
                            <div id="videoModalTitle" style="font-weight:700;font-size:1.1rem;">Video Details</div>
                            <button onclick="closeVideoModal()" style="background:none;border:none;color:var(--text-muted);font-size:1.5rem;cursor:pointer;">&times;</button>
                        </div>
                        <div style="flex:1;overflow-y:auto;padding:1.5rem;display:flex;flex-direction:column;gap:1rem;">
                            <div id="videoModalContent">
                                <div style="text-align:center;padding:2rem;color:var(--text-muted);">Loading video details...</div>
                            </div>
                        </div>
                        <div style="padding:1.25rem;border-top:1px solid var(--border);display:flex;gap:0.75rem;justify-content:flex-end;">
                            <button onclick="saveVideoForLater()" class="btn btn-secondary" style="width:auto;">
                                <i class="fas fa-bookmark" style="margin-right: 0.5rem;"></i>Save for Later
                            </button>
                            <button onclick="openVideoOnYouTube()" class="btn btn-primary" style="width:auto;background:linear-gradient(135deg, #ff0000, #ff6b6b);">
                                <i class="fab fa-youtube" style="margin-right: 0.5rem;"></i>Watch on YouTube
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Mentor Chat Modal -->
        <div id="mentorChatModal" class="modal" style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.85);z-index:2000;align-items:center;justify-content:center;">
            <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:16px;width:90%;max-width:600px;height:80vh;display:flex;flex-direction:column;overflow:hidden;box-shadow:var(--shadow-lg);">
                <div style="padding:1.25rem;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center;background:rgba(255,255,255,0.02);">
                    <div style="display:flex;align-items:center;gap:0.75rem;">
                        <div id="activeMentorAvatar" style="width:32px;height:32px;background:var(--accent);border-radius:50%;display:flex;align-items:center;justify-content:center;color:var(--bg-primary);font-weight:bold;font-size:0.8rem;"></div>
                        <div>
                            <div id="activeMentorName" style="font-weight:700;font-size:1.1rem;">Mentor Chat</div>
                            <div style="font-size:0.7rem;color:var(--success);">Online</div>
                        </div>
                    </div>
                    <button onclick="closeMentorChat()" style="background:none;border:none;color:var(--text-muted);font-size:1.5rem;cursor:pointer;">&times;</button>
                </div>
                <div id="mentorChatHistory" style="flex:1;overflow-y:auto;padding:1.5rem;display:flex;flex-direction:column;gap:1.1rem;"></div>
                <div style="padding:1.25rem;border-top:1px solid var(--border);background:rgba(255,255,255,0.02);">
                    <div style="display:flex;gap:0.75rem;">
                        <input type="text" id="mentorChatMessage" placeholder="Type your message..." style="flex:1;background:var(--bg-elevated);border:1px solid var(--border);border-radius:12px;padding:0.75rem 1rem;color:var(--text-primary);outline:none;">
                        <button onclick="sendMentorMessageToServer()" style="background:var(--accent);color:var(--bg-primary);border:none;border-radius:12px;padding:0 1.25rem;font-weight:700;cursor:pointer;">Send</button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Outcome Simulation Tab -->
        <div id="tab-simulate" class="tab-content">
            <div class="header">
                <div class="breadcrumb">
                    <span>Home</span><span>/</span><span class="breadcrumb-current">Outcome Simulation</span>
                </div>
            </div>
            <div class="content-grid">
                <div class="card" style="grid-column: span 2;">
                    <div class="card-header">
                        <div class="card-title">Run Monte Carlo Simulation</div>
                    </div>
                    <div class="card-body">
                        <div style="display:grid;gap:1.25rem;">
                            <div>
                                <label class="form-label">Decision Description</label>
                                <input type="text" id="simDecision" class="form-input" placeholder="e.g. Switching to AI Research role at a startup">
                            </div>
                            <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
                                <div>
                                    <label class="form-label">Base Salary ($)</label>
                                    <input type="number" id="simSalary" class="form-input" value="120000">
                                </div>
                                <div>
                                    <label class="form-label">Uncertainty Level (0-1)</label>
                                    <input type="range" id="simUncertainty" min="0.05" max="0.5" step="0.05" value="0.2" style="width:100%;accent-color:var(--accent);">
                                </div>
                            </div>
                            <button onclick="runSimulation()" class="send-btn" style="width:100%;">Run 5-Year Projection</button>
                        </div>
                    </div>
                </div>
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Simulation Statistics</div>
                    </div>
                    <div class="card-body" id="simStats">
                        <div class="empty-state">Run a simulation to see results</div>
                    </div>
                </div>
                <div class="card" style="grid-column: span 3;">
                    <div class="card-header">
                        <div class="card-title">Projected Outcomes</div>
                    </div>
                    <div class="card-body" id="simResults" style="display:grid;grid-template-columns:repeat(3, 1fr);gap:1.5rem;">
                        <div class="empty-state" style="grid-column: span 3;">No data available</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Interview Practice Tab -->
        <div id="tab-interview" class="tab-content">
            <div class="header">
                <div class="breadcrumb">
                    <span>Home</span><span>/</span><span class="breadcrumb-current">Interview Practice</span>
                </div>
            </div>
            <div class="content-grid">
                <div class="card" style="grid-column: span 2;">
                    <div class="card-header">
                        <div class="card-title">Voice Interview Simulator</div>
                    </div>
                    <div class="card-body" style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:3rem;gap:2rem;text-align:center;">
                        <div id="interviewStatus" style="font-size:1.2rem;font-weight:600;color:var(--text-secondary);">Ready to start your mock interview?</div>
                        <div style="position:relative;">
                            <div id="interviewVisualizer" style="display:flex;align-items:center;gap:4px;height:40px;">
                                <div style="width:4px;height:10px;background:var(--accent);border-radius:2px;"></div>
                                <div style="width:4px;height:20px;background:var(--accent);border-radius:2px;"></div>
                                <div style="width:4px;height:15px;background:var(--accent);border-radius:2px;"></div>
                                <div style="width:4px;height:25px;background:var(--accent);border-radius:2px;"></div>
                                <div style="width:4px;height:12px;background:var(--accent);border-radius:2px;"></div>
                            </div>
                        </div>
                        <button onclick="startInterview()" id="interviewBtn" class="send-btn" style="padding:1rem 2.5rem;font-size:1.1rem;border-radius:50px;box-shadow:0 0 20px rgba(var(--accent-rgb), 0.3);">Start Interview Session</button>
                        <div style="font-size:0.85rem;color:var(--text-muted);max-width:400px;">Our AI will ask you industry-specific questions. Respond via voice to receive real-time feedback.</div>
                    </div>
                </div>
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Session Settings</div>
                    </div>
                    <div class="card-body">
                        <div style="display:grid;gap:1rem;">
                            <div>
                                <label class="form-label">Target Role</label>
                                <select id="interviewRole" class="form-select">
                                    <option value="software_engineer">Software Engineer</option>
                                    <option value="product_manager">Product Manager</option>
                                    <option value="data_scientist">Data Scientist</option>
                                    <option value="designer">UX Designer</option>
                                </select>
                            </div>
                            <div>
                                <label class="form-label">Difficulty</label>
                                <select id="interviewDifficulty" class="form-select">
                                    <option value="junior">Junior / Associate</option>
                                    <option value="mid">Mid-Level / Senior</option>
                                    <option value="lead">Lead / Principal</option>
                                </select>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Integrations Tab -->
        <div id="tab-integrations" class="tab-content">
            <div class="header">
                <div class="breadcrumb">
                    <span>Home</span><span>/</span><span class="breadcrumb-current">Integrations</span>
                </div>
            </div>
            <div class="content-grid">
                <div class="card" style="grid-column: span 2;">
                    <div class="card-header" style="display:flex;justify-content:space-between;align-items:center;">
                        <div class="card-title">AI Knowledge Base (RAG)</div>
                        <input type="file" id="knowledgeUpload" style="display:none;" onchange="uploadKnowledge(event)">
                        <button onclick="document.getElementById('knowledgeUpload').click()" style="padding:0.4rem 0.8rem;background:var(--accent);color:var(--bg-primary);border:none;border-radius:6px;font-size:0.75rem;font-weight:700;">Add Document</button>
                    </div>
                    <div class="card-body">
                        <p style="color:var(--text-muted);font-size:0.85rem;margin-bottom:1rem;">Upload company policies, project specs, or personal notes to give the AI custom context.</p>
                        <div id="knowledgeList" style="display:grid;grid-template-columns:repeat(auto-fill, minmax(200px, 1fr));gap:1rem;">
                            <div class="empty-state">No specialized knowledge added.</div>
                        </div>
                    </div>
                </div>
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Zapier Webhooks</div>
                    </div>
                    <div class="card-body">
                        <p style="margin-bottom: 1rem; color: var(--text-secondary); font-size: 0.9rem;">Trigger automated workflows when decisions are made.</p>
                        <input type="text" id="zapierUrl" placeholder="https://hooks.zapier.com/..." style="width: 100%; padding: 0.75rem; background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 8px; color: var(--text-primary); margin-bottom: 1rem;">
                        <button class="attach-btn" style="width: 100%;" onclick="setupWebhook()">Save Webhook</button>
                    </div>
                </div>
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Enterprise Connect</div>
                    </div>
                    <div class="card-body">
                        <div style="display: flex; flex-direction: column; gap: 1rem;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span>Slack Bot</span>
                                <button class="nav-btn" style="width: auto; padding: 0.4rem 1rem;">Connect</button>
                            </div>
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span>Microsoft Teams</span>
                                <button class="nav-btn" style="width: auto; padding: 0.4rem 1rem;">Connect</button>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">API Access</div>
                    </div>
                    <div class="card-body">
                        <p style="margin-bottom: 1rem; color: var(--text-secondary); font-size: 0.9rem;">Use your token for custom integrations.</p>
                        <div id="apiKeyDisplay" style="padding: 0.75rem; background: #000; border-radius: 8px; font-family: monospace; font-size: 0.8rem; margin-bottom: 1rem; display: none;"></div>
                        <button class="nav-btn" style="width: 100%;" onclick="generateApiKey()">Generate Token</button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Check-in Modal -->
        <div id="checkInModal" class="modal" style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.8);z-index:1000;align-items:center;justify-content:center;">
            <div class="card" style="width:100%;max-width:500px;margin:20px;">
                <div class="card-header">
                    <div class="card-title">Weekly Reflection</div>
                    <button onclick="document.getElementById('checkInModal').style.display='none'" style="background:none;border:none;color:var(--text-secondary);cursor:pointer;font-size:1.5rem;">&times;</button>
                </div>
                <div class="card-body">
                    <div id="checkInQuestions" style="display:grid;gap:1rem;"></div>
                    <div style="margin-top:1.5rem;display:flex;justify-content:flex-end;gap:1rem;">
                        <button class="attach-btn" onclick="document.getElementById('checkInModal').style.display='none'">Cancel</button>
                        <button class="send-btn" onclick="submitCheckIn()">Complete</button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Add Event Modal -->
        <div id="addEventModal" class="modal" style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.8);z-index:1000;align-items:center;justify-content:center;">
            <div class="card" style="width:100%;max-width:500px;margin:20px;">
                <div class="card-header">
                    <div class="card-title">Add Calendar Event</div>
                    <button onclick="document.getElementById('addEventModal').style.display='none'" style="background:none;border:none;color:var(--text-secondary);cursor:pointer;font-size:1.5rem;">&times;</button>
                </div>
                <div class="card-body">
                    <div class="input-container" style="display:grid;gap:1rem;">
                        <input type="text" id="eventTitle" placeholder="Event Title" style="padding:0.75rem;background:var(--bg-elevated);border:1px solid var(--border);border-radius:8px;color:var(--text-primary);">
                        <select id="eventType" style="padding:0.75rem;background:var(--bg-elevated);border:1px solid var(--border);border-radius:8px;color:var(--text-primary);">
                            <option value="decision_deadline">Decision Deadline</option>
                            <option value="check_in">Check-in</option>
                            <option value="goal_milestone">Goal Milestone</option>
                            <option value="interview">Interview</option>
                            <option value="follow_up">Follow-up</option>
                        </select>
                        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
                            <input type="datetime-local" id="eventStart" style="padding:0.75rem;background:var(--bg-elevated);border:1px solid var(--border);border-radius:8px;color:var(--text-primary);">
                            <input type="datetime-local" id="eventEnd" style="padding:0.75rem;background:var(--bg-elevated);border:1px solid var(--border);border-radius:8px;color:var(--text-primary);">
                        </div>
                        <input type="text" id="eventLocation" placeholder="Location (optional)" style="padding:0.75rem;background:var(--bg-elevated);border:1px solid var(--border);border-radius:8px;color:var(--text-primary);">
                        <textarea id="eventDescription" rows="3" placeholder="Description..." style="padding:0.75rem;background:var(--bg-elevated);border:1px solid var(--border);border-radius:8px;color:var(--text-primary);resize:none;"></textarea>
                    </div>
                    <div style="margin-top:1.5rem;display:flex;justify-content:flex-end;gap:1rem;">
                        <button class="attach-btn" onclick="document.getElementById('addEventModal').style.display='none'">Cancel</button>
                        <button class="send-btn" onclick="submitEvent()">Create Event</button>
                    </div>
                </div>
            </div>
        </div>

        <!-- PWA Install Prompt -->
        <div id="pwaInstallPrompt" style="display:none;position:fixed;bottom:20px;left:20px;z-index:9999;background:var(--bg-card);border:1px solid var(--accent);border-radius:12px;padding:1rem;box-shadow:var(--shadow-lg);animation:slideInLeft 0.5s ease;">
            <div style="display:flex;align-items:center;gap:1rem;">
                <div style="background:var(--accent);color:var(--bg-primary);width:40px;height:40px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-weight:bold;">AI</div>
                <div>
                    <div style="font-weight:600;">Install App</div>
                    <div style="font-size:0.8rem;color:var(--text-secondary);">Add to home screen</div>
                </div>
                <button onclick="installPWA()" style="padding:0.5rem 1rem;background:var(--accent);color:var(--bg-primary);border:none;border-radius:6px;font-weight:600;cursor:pointer;margin-left:0.5rem;">Install</button>
                <button onclick="dismissInstall()" style="background:none;border:none;color:var(--text-secondary);cursor:pointer;font-size:1.2rem;">&times;</button>
            </div>
        </div>

    <div class="loading" id="loading">
        <div class="spinner"></div>
    </div>
    <div class="toast-container" id="toastContainer"></div>

    <!-- Settings Modal -->
    <div class="settings-modal" id="settingsModal"
        style="display:none;position:fixed;inset:0;background:rgba(0,0,0,0.8);z-index:1000;align-items:center;justify-content:center;">
        <div
            style="background:var(--bg-card);border:1px solid var(--border);border-radius:16px;width:100%;max-width:400px;overflow:hidden;">
            <div
                style="padding:1.25rem;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center;">
                <div class="card-title">Settings</div>
                <button onclick="toggleSettings()"
                    style="background:none;border:none;color:var(--text-muted);font-size:1.5rem;cursor:pointer;">&times;</button>
            </div>
            <div style="padding:1.5rem;display:grid;gap:1.5rem;">
                <div>
                    <div class="form-label" style="margin-bottom: 0.75rem;">Theme</div>
                    <div style="display:grid;grid-template-columns:repeat(3, 1fr);gap:0.75rem;">
                        <button class="nav-btn theme-btn" id="themeLight" onclick="setTheme('light')"
                            style="width:100%;font-size:0.8rem;padding:0.5rem 0.25rem;">Light</button>
                        <button class="nav-btn theme-btn" id="themeDark" onclick="setTheme('dark')"
                            style="width:100%;font-size:0.8rem;padding:0.5rem 0.25rem;">Dark</button>
                        <button class="nav-btn theme-btn" id="themeSystem" onclick="setTheme('system')"
                            style="width:100%;font-size:0.8rem;padding:0.5rem 0.25rem;">Auto</button>
                    </div>
                </div>

                <div>
                    <div class="form-label" style="margin-bottom: 0.75rem;">Multi-LLM Engine</div>
                    <select id="llmProvider" class="form-select" onchange="switchLLM(this.value)" style="margin-bottom: 0.5rem;">
                        <option value="mock">Mock Service</option>
                        <option value="ollama">Ollama (Local)</option>
                        <option value="openai">OpenAI Cloud</option>
                    </select>
                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-top: 0.5rem;">
                        <input type="checkbox" id="fineTuneToggle" onchange="toggleFineTuning(this.checked)">
                        <label for="fineTuneToggle" style="font-size: 0.8rem; color: var(--text-secondary);">Enable Fine-tuned Career Expert</label>
                    </div>
                </div>

                <div>
                    <div class="form-label" style="margin-bottom: 0.75rem;">Advanced Settings</div>
                    <div style="display:grid;gap:0.5rem;">
                        <button class="btn btn-secondary" onclick="showShortcutHelp()" style="width:100%;font-size:0.8rem;padding:0.5rem 0.75rem;">Keyboard Shortcuts</button>
                        <button class="btn btn-secondary" onclick="clearAllData()" style="width:100%;color:var(--danger);font-size:0.8rem;padding:0.5rem 0.75rem;">Clear All Local Data</button>
                    </div>
                </div>

                <div style="font-size:0.75rem;color:var(--text-muted);text-align:center;">
                    Version 3.0.0-growth • Connected: <span id="settingsConnectionStatus">Detecting...</span>
                </div>
            </div>
        </div>
    </div>

    <!-- Shortcuts Modal -->
    <div id="shortcutsModal" class="modal" style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.8);z-index:1100;align-items:center;justify-content:center;">
        <div class="card" style="width:100%;max-width:400px;margin:20px;">
            <div class="card-header">
                <div class="card-title">Keyboard Shortcuts</div>
                <button onclick="document.getElementById('shortcutsModal').style.display='none'" style="background:none;border:none;color:var(--text-secondary);cursor:pointer;font-size:1.5rem;">&times;</button>
            </div>
            <div class="card-body">
                <div style="display:grid;gap:1rem;font-size:0.9rem;">
                    <div style="display:flex;justify-content:space-between;"><span>New Conversation</span> <kbd>Alt + N</kbd></div>
                    <div style="display:flex;justify-content:space-between;"><span>Templates</span> <kbd>Alt + T</kbd></div>
                    <div style="display:flex;justify-content:space-between;"><span>Journal View</span> <kbd>Alt + J</kbd></div>
                    <div style="display:flex;justify-content:space-between;"><span>Toggle Settings</span> <kbd>Alt + S</kbd></div>
                    <div style="display:flex;justify-content:space-between;"><span>Focus Input</span> <kbd>/</kbd></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Initialize state (synced with core script)
        userId = window.userId;
        journalDecisions = window.journalDecisions;

        // Character counter
        if (document.getElementById('chatInput')) {
            document.getElementById('chatInput').addEventListener('input', (e) => {
                document.getElementById('charCount').textContent = e.target.value.length;
            });

            document.getElementById('chatInput').addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });
        }

        function formatTime(date) {
            const now = new Date();
            const d = new Date(date);
            if (d.toDateString() === now.toDateString()) {
                return 'Today at ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            }
            return d.toLocaleDateString();
        }

        let pendingUploads = [];

        async function handleFileSelect(event) {
            const files = event.target.files;
            if (!files.length) return;
            
            const file = files[0];
            if (file.size > 1024 * 1024 * 1024) { // 1GB limit
                showToast('File too large. Max size is 1GB.');
                return;
            }
            
            const previewId = 'upload-' + Date.now();
            const previewArea = document.getElementById('filePreview');
            previewArea.style.display = 'flex';
            
            const previewEl = document.createElement('div');
            previewEl.className = 'file-tag';
            previewEl.id = previewId;
            previewEl.innerHTML = `
                <div class="file-icon"></div>
                <div class="file-info">
                    <div class="file-name">${file.name}</div>
                    <div class="file-status">Uploading...</div>
                </div>
                <button class="remove-file-btn" onclick="removeFile('${previewId}')">✕</button>
            `;
            previewArea.appendChild(previewEl);
            
            try {
                const uploadedFile = await uploadFile(file);
                // uploadedFile.previewId = previewId; // Store content if needed
                pendingUploads.push(uploadedFile);
                previewEl.querySelector('.file-status').textContent = 'Uploaded';
                previewEl.classList.add('success');
                showToast('File uploaded successfully');
                
                // Add system message about upload
                const messagesDiv = document.getElementById('chatMessages');
                messagesDiv.innerHTML += `
                    <div class="message-group system" id="msg-${previewId}">
                        <div class="message-content" style="margin-left:0;text-align:center;color:var(--text-secondary);font-size:0.8rem;">
                            Uploaded: ${file.name}
                        </div>
                    </div>
                `;
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
                
            } catch (error) {
                previewEl.querySelector('.file-status').textContent = 'Failed';
                previewEl.classList.add('error');
                showToast('Upload failed: ' + error.message);
                setTimeout(() => previewEl.remove(), 3000);
            }
            
            // Clear input
            event.target.value = '';
        }

        function removeFile(previewId) {
            const el = document.getElementById(previewId);
            if (el) el.remove();
            
            const msgEl = document.getElementById('msg-' + previewId);
            if (msgEl) msgEl.remove();
            
            // Since we don't have a specific endpoint to remove just ONE file from context yet,
            // we'll rely on clearing the whole context or just hiding it from UI.
            // For a robust implementation, we'd need a delete endpoint. 
            // However, "cancel to upload" implies removing it before it's "used/sent" effectively.
            // Given the current architecture where all uploads go to context immediately,
            // we will simulate removal by just hiding it, and let Clear Chat handle the real cleanup.
            // Or we could implement a delete endpoint. For now, UI removal is the primary "cancel".
        }

        async function clearChat() {
            // Call backend to clear context
            try {
                await fetch('/api/context/clear', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ user_id: userId })
                });
            } catch (e) {
                console.error("Failed to clear backend context", e);
            }

            // Switch to chat tab first
            showTab('chat');
            
            const time = formatTime(new Date());
            document.getElementById('chatMessages').innerHTML = `
                <div class="message-group assistant">
                    <div class="avatar ai">AI</div>
                    <div class="message-content">
                        <div class="message-header">
                            <span class="message-name">Career Counselor</span>
                            <span class="message-time">${time}</span>
                        </div>
                        <div class="message-bubble">Hello, I'm your career counselor. Share a decision you're considering, or ask me anything about your career path.</div>
                    </div>
                </div>
            `;
            
            // Clear input field and reset character count
            const chatInput = document.getElementById('chatInput');
            if (chatInput) {
                chatInput.value = '';
                chatInput.focus();
            }
            document.getElementById('charCount').textContent = '0';
            
            // Clear file previews and pending uploads
            document.getElementById('filePreview').innerHTML = '';
            document.getElementById('filePreview').style.display = 'none';
            pendingUploads = [];
            
            showToast('New conversation started');
        }

        async function uploadFile(file) {
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch(`/api/upload?user_id=${userId}`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Upload failed');
            }
            
            return await response.json();
        }

        async function handleVideoSelect(event) {
            const files = event.target.files;
            if (!files.length) return;
            
            const file = files[0];
            const maxSize = 2 * 1024 * 1024 * 1024; // 2GB limit for videos
            if (file.size > maxSize) {
                showToast('Video file too large. Max size is 2GB.');
                return;
            }
            
            const previewId = 'upload-video-' + Date.now();
            const previewArea = document.getElementById('filePreview');
            previewArea.style.display = 'flex';
            
            const previewEl = document.createElement('div');
            previewEl.className = 'file-tag';
            previewEl.id = previewId;
            previewEl.innerHTML = `
                <div class="file-icon">🎥</div>
                <div class="file-info">
                    <div class="file-name">${file.name}</div>
                    <div class="file-status">Uploading...</div>
                </div>
                <button class="remove-file-btn" onclick="removeFile('${previewId}')">✕</button>
            `;
            previewArea.appendChild(previewEl);
            
            try {
                const uploadedFile = await uploadVideo(file);
                pendingUploads.push(uploadedFile);
                previewEl.querySelector('.file-status').textContent = 'Processing...';
                previewEl.classList.add('success');
                showToast('Video uploaded successfully. Processing...');
                
                // Add system message about upload
                const messagesDiv = document.getElementById('chatMessages');
                messagesDiv.innerHTML += `
                    <div class="message-group system" id="msg-${previewId}">
                        <div class="message-content" style="margin-left:0;text-align:center;color:var(--text-secondary);font-size:0.8rem;">
                            Video uploaded: ${file.name} - The system will extract transcripts and learn from it
                        </div>
                    </div>
                `;
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
                
            } catch (error) {
                previewEl.querySelector('.file-status').textContent = 'Failed';
                previewEl.classList.add('error');
                showToast('Video upload failed: ' + error.message);
                setTimeout(() => previewEl.remove(), 3000);
            }
            
            event.target.value = '';
        }

        async function uploadVideo(file) {
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch(`/api/upload/video?user_id=${userId}`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Video upload failed');
            }
            
            return await response.json();
        }

        function toggleAttachMenu() {
            const menu = document.getElementById('attachMenu');
            menu.style.display = menu.style.display === 'none' ? 'block' : 'none';
        }

        function closeAttachMenu() {
            document.getElementById('attachMenu').style.display = 'none';
        }

        function openUrlModal() {
            document.getElementById('urlInputModal').style.display = 'flex';
            document.getElementById('urlInput').focus();
        }

        function closeUrlModal() {
            document.getElementById('urlInputModal').style.display = 'none';
            document.getElementById('urlInput').value = '';
        }

        async function submitUrlInput() {
            const url = document.getElementById('urlInput').value.trim();
            
            if (!url) {
                showToast('Please enter a URL');
                return;
            }
            
            // Basic URL validation
            if (!url.startsWith('http://') && !url.startsWith('https://')) {
                showToast('URL must start with http:// or https://');
                return;
            }
            
            const previewId = 'upload-url-' + Date.now();
            const previewArea = document.getElementById('filePreview');
            previewArea.style.display = 'flex';
            
            const previewEl = document.createElement('div');
            previewEl.className = 'file-tag';
            previewEl.id = previewId;
            
            // Determine icon based on URL
            let icon = '🔗';
            if (url.includes('youtube.com') || url.includes('youtu.be')) {
                icon = '🎬';
            }
            
            previewEl.innerHTML = `
                <div class="file-icon">${icon}</div>
                <div class="file-info">
                    <div class="file-name">${new URL(url).hostname}</div>
                    <div class="file-status">Processing...</div>
                </div>
                <button class="remove-file-btn" onclick="removeFile('${previewId}')">✕</button>
            `;
            previewArea.appendChild(previewEl);
            
            closeUrlModal();
            
            try {
                const response = await fetch(`/api/upload/url?user_id=${userId}&url=${encodeURIComponent(url)}`, {
                    method: 'POST'
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'URL processing failed');
                }
                
                const result = await response.json();
                pendingUploads.push(result);
                previewEl.querySelector('.file-status').textContent = 'Ready';
                previewEl.classList.add('success');
                
                // Determine content type message
                let contentMsg = result.type === 'youtube' ? 'YouTube video' : 'web content';
                showToast(`${contentMsg} added successfully. The system will analyze and learn from this content.`);
                
                // Add system message
                const messagesDiv = document.getElementById('chatMessages');
                const contentType = result.type === 'youtube' ? '🎬 YouTube' : '📄 URL';
                messagesDiv.innerHTML += `
                    <div class="message-group system" id="msg-${previewId}">
                        <div class="message-content" style="margin-left:0;text-align:center;color:var(--text-secondary);font-size:0.8rem;">
                            ${contentType} added: ${result.title} - The system will extract and analyze this content for training
                        </div>
                    </div>
                `;
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
                
            } catch (error) {
                previewEl.querySelector('.file-status').textContent = 'Failed';
                previewEl.classList.add('error');
                showToast('URL processing failed: ' + error.message);
                setTimeout(() => previewEl.remove(), 3000);
            }
        }

        // Close menu when clicking outside
        document.addEventListener('click', function(event) {
            const attachMenu = document.getElementById('attachMenu');
            const attachBtn = document.querySelector('.attach-btn');
            if (attachMenu && attachBtn && !attachMenu.contains(event.target) && !attachBtn.contains(event.target)) {
                attachMenu.style.display = 'none';
            }
        });

        async function sendMessage() {
            const input = document.getElementById('chatInput');
            const msg = input.value.trim();
            if (!msg) return;

            const messagesDiv = document.getElementById('chatMessages');
            const time = formatTime(new Date());

            messagesDiv.innerHTML += `
                <div class="message-group user">
                    <div class="avatar user">U</div>
                    <div class="message-content">
                        <div class="message-header">
                            <span class="message-name">You</span>
                            <span class="message-time">${time}</span>
                        </div>
                        <div class="message-bubble">${msg}</div>
                    </div>
                </div>
            `;
            input.value = '';
            document.getElementById('charCount').textContent = '0';
            
            // Clear pending uploads UI (they are now part of context)
            document.getElementById('filePreview').innerHTML = '';
            document.getElementById('filePreview').style.display = 'none';
            pendingUploads = [];
            
            messagesDiv.scrollTop = messagesDiv.scrollHeight;

            const typingDiv = document.createElement('div');
            typingDiv.className = 'typing-indicator';
            typingDiv.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';
            messagesDiv.appendChild(typingDiv);

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: msg, user_id: userId })
                });
                typingDiv.remove();
                const data = await response.json();
                const responseTime = formatTime(new Date());

                messagesDiv.innerHTML += `
                    <div class="message-group assistant">
                        <div class="avatar ai">AI</div>
                        <div class="message-content">
                            <div class="message-header">
                                <span class="message-name">Career Counselor</span>
                                <span class="message-time">${responseTime}</span>
                            </div>
                            <div class="message-bubble">${data.response || 'I understand. Let me help you think through this.'}</div>
                        </div>
                    </div>
                `;
            } catch (error) {
                typingDiv.remove();
                messagesDiv.innerHTML += `
                    <div class="message-group assistant">
                        <div class="avatar ai">AI</div>
                        <div class="message-content">
                            <div class="message-header">
                                <span class="message-name">Career Counselor</span>
                                <span class="message-time">${formatTime(new Date())}</span>
                            </div>
                            <div class="message-bubble">I apologize, but I encountered an error. Please try again.</div>
                        </div>
                    </div>
                `;
            }
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }



        // Sliders (with null checks)
        const riskSlider = document.getElementById('riskTolerance');
        if (riskSlider) {
            riskSlider.addEventListener('input', (e) => {
                const riskValue = document.getElementById('riskValue');
                if (riskValue) riskValue.textContent = e.target.value + '%';
            });
        }
        
        const financialSlider = document.getElementById('financialStability');
        if (financialSlider) {
            financialSlider.addEventListener('input', (e) => {
                const financialValue = document.getElementById('financialValue');
                if (financialValue) financialValue.textContent = e.target.value + '%';
            });
        }

        // Form submission (with null check)
        const decisionForm = document.getElementById('decisionForm');
        if (decisionForm) {
            decisionForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                const loading = document.getElementById('loading');
                if (loading) loading.classList.add('active');

                const data = {
                    decision_type: document.getElementById('decisionType')?.value || '',
                    description: document.getElementById('description')?.value || '',
                    risk_tolerance: (document.getElementById('riskTolerance')?.value || 50) / 100,
                    financial_stability: (document.getElementById('financialStability')?.value || 50) / 100,
                    emotions: [],
                    user_id: userId
                };

                try {
                    const response = await fetch('/api/analyze', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(data)
                    });
                    const result = await response.json();
                    displayResults(result);
                    saveToJournal(data, result);
                } catch (error) {
                    showToast('Analysis failed');
                }
                if (loading) loading.classList.remove('active');
            });
        }

        function displayResults(result) {
            const pred = result.prediction || {};
            const regret = (pred.predicted_regret || 0) * 100;
            const conf = (pred.confidence || 0) * 100;
            const riskClass = regret < 30 ? 'low' : regret < 60 ? 'moderate' : 'high';

            let html = `
                <div class="regret-meter"><div class="regret-fill ${riskClass}" style="width:${regret}%"></div></div>
                <div class="metrics-grid">
                    <div class="metric-card"><div class="metric-value">${regret.toFixed(0)}%</div><div class="metric-label">Predicted Regret</div></div>
                    <div class="metric-card"><div class="metric-value">${conf.toFixed(0)}%</div><div class="metric-label">Confidence</div></div>
                </div>
            `;

            if (pred.recommendations && pred.recommendations.length > 0) {
                html += '<div class="recommendations"><h4>Recommendations</h4>';
                pred.recommendations.slice(0, 5).forEach(rec => {
                    html += `<div class="rec-item">${rec}</div>`;
                });
                html += '</div>';
            }

            const analysisContainer = document.getElementById('analysisResults');
            if (analysisContainer) analysisContainer.innerHTML = html;
        }

        function saveToJournal(data, result) {
            const entry = {
                id: Date.now(),
                type: data.decision_type,
                description: (data.description || '').substring(0, 100),
                regret: ((result.prediction?.predicted_regret || 0) * 100).toFixed(0),
                timestamp: new Date().toISOString()
            };
            journalDecisions.unshift(entry);
            localStorage.setItem('decisions', JSON.stringify(journalDecisions));
            updateJournal();
        }

        function updateJournal() {
            const container = document.getElementById('journalList');
            if (!journalDecisions.length) {
                container.innerHTML = '<div class="empty-state">No decisions recorded yet</div>';
                return;
            }
            container.innerHTML = journalDecisions.slice(0, 20).map((d, i) => `
                <div class="journal-item" onclick="showDecisionDetail(${i})">
                    <div>
                        <span class="journal-type">${(d.type || 'unknown').replace(/_/g, ' ')}</span>
                        <div style="margin-top:0.5rem;color:var(--text-secondary);font-size:0.85rem;">${(d.description || '').substring(0, 50)}...</div>
                    </div>
                    <div class="journal-regret">${d.regret || 0}%</div>
                </div>
            `).join('');
            document.getElementById('journalCount').textContent = journalDecisions.length + ' entries';
        }

        function showDecisionDetail(index) {
            const d = journalDecisions[index];
            document.getElementById('journalDetails').innerHTML = `
                <h3 style="margin-bottom:1rem;text-transform:capitalize;">${(d.type || '').replace(/_/g, ' ')}</h3>
                <div class="stat-card" style="margin-bottom:1rem;text-align:center;">
                    <div class="stat-value">${d.regret || 0}%</div>
                    <div class="stat-label">Predicted Regret</div>
                </div>
                <p style="color:var(--text-secondary);line-height:1.6;">${d.description || 'No description'}</p>
                <p style="margin-top:1rem;font-size:0.75rem;color:var(--text-muted);">${new Date(d.timestamp).toLocaleString()}</p>
            `;
        }

        let graphZoom = null;
        let graphSvg = null;
        let graphG = null;
        let graphSimulation = null;

        async function loadGraph() {
            const container = document.getElementById('graphContainer');
            if (container) container.innerHTML = '<div class="graph-loading" style="color:#00ffbb;">Analyzing Neural Pathways...</div>';
            
            try {
                const response = await fetch('/api/graph');
                const data = await response.json();
                renderGraph(data);
            } catch (error) { 
                console.error('Graph error:', error);
                if (container) container.innerHTML = '<div class="graph-loading" style="color:#ff0088;">Neural Mapping Offline</div>';
            }
        }

        function zoomGraph(factor) {
            if (graphZoom && graphSvg) {
                graphSvg.transition().duration(500).ease(d3.easeCubicOut).call(graphZoom.scaleBy, factor);
            }
        }

        function resetGraphZoom() {
            if (graphZoom && graphSvg) {
                graphSvg.transition().duration(750).ease(d3.easePolyOut).call(graphZoom.transform, d3.zoomIdentity);
            }
        }

        function renderGraph(data) {
            const container = document.getElementById('graphContainer');
            if (!container) return;
            container.innerHTML = '';
            
            const width = container.offsetWidth || 1000;
            const height = container.offsetHeight || 700;

            const colors = {
                decision: '#ff0088',
                outcome: '#00d4ff',
                milestone: '#00ffbb',
                factor: '#ffcc00',
                link: 'rgba(255, 255, 255, 0.1)',
                linkHighlight: 'rgba(0, 255, 187, 0.4)'
            };

            const graphData = data.graph || data;
            const nodes = graphData.nodes || [];
            const links = graphData.links || graphData.edges || [];

            document.getElementById('graphNodeCount').textContent = nodes.length;
            document.getElementById('graphEdgeCount').textContent = links.length;
            const density = nodes.length > 1 ? (2 * links.length) / (nodes.length * (nodes.length - 1)) : 0;
            document.getElementById('graphDensity').textContent = density.toFixed(3);

            graphSvg = d3.select(container).append('svg')
                .attr('width', '100%')
                .attr('height', '100%')
                .attr('viewBox', [0, 0, width, height])
                .style('cursor', 'grab');

            const defs = graphSvg.append('defs');
            const glowFilter = defs.append('filter')
                .attr('id', 'neonGlow')
                .attr('x', '-50%')
                .attr('y', '-50%')
                .attr('width', '200%')
                .attr('height', '200%');
            glowFilter.append('feGaussianBlur').attr('stdDeviation', '4').attr('result', 'coloredBlur');
            const feMerge = glowFilter.append('feMerge');
            feMerge.append('feMergeNode').attr('in', 'coloredBlur');
            feMerge.append('feMergeNode').attr('in', 'SourceGraphic');

            const gridG = graphSvg.append('g').attr('class', 'background-grid');
            const gridSize = 50;
            for (let x = 0; x <= width + gridSize; x += gridSize) {
                gridG.append('line').attr('x1', x).attr('y1', 0).attr('x2', x).attr('y2', height)
                    .attr('stroke', 'rgba(255,255,255,0.03)').attr('stroke-width', 1);
            }
            for (let y = 0; y <= height + gridSize; y += gridSize) {
                gridG.append('line').attr('x1', 0).attr('y1', y).attr('x2', width).attr('y2', y)
                    .attr('stroke', 'rgba(255,255,255,0.03)').attr('stroke-width', 1);
            }

            graphG = graphSvg.append('g');

            graphZoom = d3.zoom()
                .scaleExtent([0.1, 5])
                .on('zoom', (event) => {
                    graphG.attr('transform', event.transform);
                });
            graphSvg.call(graphZoom);

            graphSimulation = d3.forceSimulation(nodes)
                .force('link', d3.forceLink(links).id(d => d.id).distance(120).strength(0.4))
                .force('charge', d3.forceManyBody().strength(-400))
                .force('center', d3.forceCenter(width / 2, height / 2))
                .force('collision', d3.forceCollide().radius(40));

            const link = graphG.append('g')
                .attr('class', 'links')
                .selectAll('line')
                .data(links)
                .join('line')
                .attr('stroke', colors.link)
                .attr('stroke-width', 1.5)
                .style('transition', 'stroke 0.3s, stroke-width 0.3s');

            const node = graphG.append('g')
                .attr('class', 'nodes')
                .selectAll('g')
                .data(nodes)
                .join('g')
                .attr('class', 'node-group')
                .call(d3.drag()
                    .on('start', (event, d) => {
                        if (!event.active) graphSimulation.alphaTarget(0.3).restart();
                        d.fx = d.x; d.fy = d.y;
                    })
                    .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y; })
                    .on('end', (event, d) => {
                        if (!event.active) graphSimulation.alphaTarget(0);
                        d.fx = null; d.fy = null;
                    })
                );

            node.append('circle')
                .attr('r', 12)
                .attr('fill', d => colors[d.group] || colors.decision)
                .attr('filter', 'url(#neonGlow)')
                .attr('opacity', 0.6);

            node.append('circle')
                .attr('r', 8)
                .attr('fill', '#fff')
                .attr('stroke', d => colors[d.group] || colors.decision)
                .attr('stroke-width', 3);

            node.append('text')
                .attr('dy', 25)
                .attr('text-anchor', 'middle')
                .attr('fill', 'rgba(255,255,255,0.7)')
                .attr('font-size', '11px')
                .attr('font-weight', '500')
                .style('pointer-events', 'none')
                .text(d => (d.label || d.id).replace(/_/g, ' '));

            node.on('mouseenter', function(event, d) {
                node.style('opacity', n => (n === d || isConnected(d, n)) ? 1 : 0.15);
                link.attr('stroke', l => (l.source.id === d.id || l.target.id === d.id) ? colors.milestone : colors.link)
                    .attr('stroke-width', l => (l.source.id === d.id || l.target.id === d.id) ? 3 : 1.5)
                    .style('opacity', l => (l.source.id === d.id || l.target.id === d.id) ? 1 : 0.1);
                showTooltip(event, d);
            })
            .on('mouseleave', function() {
                node.style('opacity', 1);
                link.attr('stroke', colors.link).attr('stroke-width', 1.5).style('opacity', 1);
                hideTooltip();
            })
            .on('click', function(event, d) {
                updateDetailPanel(d);
            });

            function isConnected(a, b) {
                return links.some(l => 
                    (l.source.id === a.id && l.target.id === b.id) || 
                    (l.source.id === b.id && l.target.id === a.id)
                );
            }

            function showTooltip(event, d) {
                const tt = document.getElementById('graphTooltip');
                if (!tt) return;
                tt.style.display = 'block';
                tt.style.left = (event.pageX + 15) + 'px';
                tt.style.top = (event.pageY + 15) + 'px';
                tt.innerHTML = `
                    <div style="font-weight:700;color:${colors[d.group] || colors.decision};text-transform:uppercase;font-size:0.75rem;">${d.group || 'Node'}</div>
                    <div style="margin-top:4px;font-size:1rem;">${(d.label || d.id).replace(/_/g, ' ')}</div>
                    ${d.metadata?.description ? `<div style="margin-top:8px;color:rgba(255,255,255,0.6);font-size:0.8rem;">${d.metadata.description}</div>` : ''}
                `;
            }

            function hideTooltip() {
                const tt = document.getElementById('graphTooltip');
                if (tt) tt.style.display = 'none';
            }

            function updateDetailPanel(d) {
                const panel = document.getElementById('nodeDetailContent');
                if (!panel) return;
                const connectedCount = links.filter(l => l.source.id === d.id || l.target.id === d.id).length;
                panel.innerHTML = `
                    <div class="node-detail-header" style="border-left:4px solid ${colors[d.group] || colors.decision};padding-left:1rem;margin-bottom:1.5rem;">
                        <h2 style="font-size:1.25rem;margin:0;">${(d.label || d.id).replace(/_/g, ' ')}</h2>
                        <span style="font-size:0.75rem;color:${colors[d.group] || colors.decision};text-transform:uppercase;font-weight:700;">${d.group || 'Node type'}</span>
                    </div>
                    <div class="detail-section" style="margin-bottom:1.5rem;">
                        <label style="display:block;font-size:0.7rem;color:rgba(255,255,255,0.4);text-transform:uppercase;margin-bottom:0.5rem;">Summary</label>
                        <p style="font-size:0.9rem;line-height:1.6;color:rgba(255,255,255,0.8);">${d.metadata?.description || 'No detailed analysis available.'}</p>
                    </div>
                    <div class="detail-section" style="margin-bottom:1.5rem;">
                        <label style="display:block;font-size:0.7rem;color:rgba(255,255,255,0.4);text-transform:uppercase;margin-bottom:0.8rem;">Neural Impact</label>
                        <div style="display:flex;flex-direction:column;gap:0.75rem;">
                            <div style="display:flex;justify-content:space-between;align-items:center;background:rgba(255,255,255,0.03);padding:0.6rem 1rem;border-radius:8px;">
                                <span style="font-size:0.85rem;">Connectivity</span>
                                <span style="color:#00ffbb;font-weight:600;">${connectedCount} nodes</span>
                            </div>
                            <div style="display:flex;justify-content:space-between;align-items:center;background:rgba(255,255,255,0.03);padding:0.6rem 1rem;border-radius:8px;">
                                <span style="font-size:0.85rem;">Relative Weight</span>
                                <span style="color:#ffcc00;font-weight:600;">${(d.weight || 0.5).toFixed(2)}</span>
                            </div>
                        </div>
                    </div>
                    <button class="btn-neon" style="width:100%;justify-content:center;margin-top:1rem;" onclick="focusOnNode('${d.id}')">
                        <i class="fas fa-bullseye"></i> Focus Pathway
                    </button>
                `;
            }

            graphSimulation.on('tick', () => {
                link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
                node.attr('transform', d => `translate(${d.x},${d.y})`);
            });
        }

        function focusOnNode(nodeId) {
            if (!graphSimulation || !graphSvg || !graphZoom) return;
            const nodes = graphSimulation.nodes();
            const node = nodes.find(n => n.id === nodeId);
            if (node) {
                const container = document.getElementById('graphContainer');
                const width = container.offsetWidth;
                const height = container.offsetHeight;
                const transform = d3.zoomIdentity
                    .translate(width / 2, height / 2)
                    .scale(1.5)
                    .translate(-node.x, -node.y);
                graphSvg.transition().duration(1000).ease(d3.easeCubicInOut).call(graphZoom.transform, transform);
            }
        }


        async function loadAnalytics() {
            try {
                const response = await fetch('/api/analytics/' + userId);
                const analytics = await response.json();

                let total = journalDecisions.length || analytics.total_analyses || 0;
                let avgRegret = 0;
                if (journalDecisions.length) {
                    avgRegret = journalDecisions.reduce((a, d) => a + parseFloat(d.regret || 0), 0) / journalDecisions.length;
                }

                document.getElementById('totalDecisions').textContent = total;
                document.getElementById('avgRegret').textContent = avgRegret.toFixed(0) + '%';

                renderRegretChart();
            } catch (error) { console.error('Analytics error:', error); }
        }

        function renderRegretChart() {
            const container = document.getElementById('regretChartContainer');
            const emptyState = document.getElementById('regretChartEmpty');

            if (!journalDecisions.length) {
                emptyState.style.display = 'flex';
                return;
            }
            emptyState.style.display = 'none';
            d3.select('#regretChartContainer svg').remove();

            const data = journalDecisions.filter(d => d.timestamp).map((d, i) => ({
                date: new Date(d.timestamp),
                regret: parseFloat(d.regret) || 0
            })).sort((a, b) => a.date - b.date);

            if (!data.length) return;

            const margin = { top: 20, right: 20, bottom: 30, left: 40 };
            const width = container.offsetWidth - margin.left - margin.right;
            const height = 260 - margin.top - margin.bottom;

            const svg = d3.select('#regretChartContainer').append('svg')
                .attr('width', width + margin.left + margin.right)
                .attr('height', height + margin.top + margin.bottom)
                .append('g').attr('transform', `translate(${margin.left},${margin.top})`);

            const x = d3.scaleTime().domain(d3.extent(data, d => d.date)).range([0, width]);
            const y = d3.scaleLinear().domain([0, 100]).range([height, 0]);

            const area = d3.area().x(d => x(d.date)).y0(height).y1(d => y(d.regret)).curve(d3.curveMonotoneX);
            const line = d3.line().x(d => x(d.date)).y(d => y(d.regret)).curve(d3.curveMonotoneX);

            const chartDefs = svg.append('defs');
            const areaGradient = chartDefs.append('linearGradient').attr('id', 'areaGradient').attr('x1', '0%').attr('y1', '0%').attr('x2', '0%').attr('y2', '100%');
            areaGradient.append('stop').attr('offset', '0%').attr('stop-color', 'var(--accent)').attr('stop-opacity', 0.2);
            areaGradient.append('stop').attr('offset', '100%').attr('stop-color', 'var(--accent)').attr('stop-opacity', 0);

            svg.append('path').datum(data).attr('fill', 'url(#areaGradient)').attr('d', area);
            svg.append('path').datum(data).attr('fill', 'none').attr('stroke', 'var(--accent)').attr('stroke-width', 3).attr('d', line);
            svg.selectAll('.dot').data(data).join('circle').attr('cx', d => x(d.date)).attr('cy', d => y(d.regret)).attr('r', 4).attr('fill', 'var(--bg-card)').attr('stroke', 'var(--accent)').attr('stroke-width', 2);
        }

        async function loadTemplates() {
            try {
                console.log('Loading templates...');
                const response = await fetch('/api/templates');
                if (!response.ok) {
                    throw new Error('Failed to fetch templates: ' + response.status);
                }
                const data = await response.json();
                console.log('Templates received:', data);
                
                const grid = document.getElementById('templateGrid');
                if (!grid) {
                    console.error('Template grid element not found!');
                    return;
                }
                
                const templates = data.templates || [];
                console.log('Number of templates:', templates.length);
                
                if (templates.length === 0) {
                    grid.innerHTML = '<div style="padding:2rem;text-align:center;color:var(--text-muted);">No templates available</div>';
                    return;
                }
                
                grid.innerHTML = templates.map(t => `
                    <div class="stat-card" style="cursor:pointer;text-align:left;transition:all 0.2s;" 
                         onclick="selectTemplate(event, '${t.id}')" 
                         onmouseover="this.style.background='var(--bg-hover)'" 
                         onmouseout="this.style.background='var(--bg-card)'">
                        <div style="font-weight:600;margin-bottom:0.5rem;">${t.name}</div>
                        <div style="font-size:0.85rem;color:var(--text-secondary);">${t.description}</div>
                        <div style="font-size:0.75rem;color:var(--text-muted);margin-top:0.5rem;">${t.question_count} questions</div>
                    </div>
                `).join('');
                console.log('Templates rendered successfully');
            } catch (error) { 
                console.error('Templates error:', error);
                const grid = document.getElementById('templateGrid');
                if (grid) {
                    grid.innerHTML = '<div style="padding:2rem;text-align:center;color:var(--danger);">Error loading templates: ' + error.message + '</div>';
                }
            }
        }

        function selectTemplate(event, id) {
            if (event) {
                event.stopPropagation();
                event.preventDefault();
            }
            console.log('selectTemplate called with id:', id);
            if (!id) {
                showToast('No template selected');
                return false;
            }
            showTemplateDetails(id);
            return false;
        }

        async function showTemplateDetails(templateId) {
            try {
                console.log('showTemplateDetails called with templateId:', templateId);
                const response = await fetch('/api/templates/' + templateId);
                if (!response.ok) {
                    throw new Error('Failed to fetch template: ' + response.status);
                }
                const template = await response.json();
                console.log('Template loaded:', template);
                
                // Store template for later use
                window.currentTemplate = template;
                
                // Show template details in the same tab
                const detailsDiv = document.getElementById('templateDetails');
                if (!detailsDiv) {
                    console.error('Template details element not found!');
                    showToast('Error: Details panel not found');
                    return;
                }
                
                // Create interactive question form
                const questionsHtml = template.questions
                    .map((q, i) => `
                        <div style="padding: 1rem; border-bottom: 1px solid var(--border); animation: fadeInUp 0.3s ease-out ${i * 0.1}s both;">
                            <label style="display: block; font-size: 0.85rem; color: var(--text-primary); margin-bottom: 0.5rem; font-weight: 500;">
                                <span style="color: var(--accent); margin-right: 0.5rem;">${i+1}.</span>${q.question}
                            </label>
                            <textarea id="template-answer-${i}" 
                                placeholder="Type your answer here..." 
                                style="width: 100%; min-height: 60px; padding: 0.75rem; background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 8px; color: var(--text-primary); font-family: inherit; font-size: 0.85rem; resize: vertical; transition: all 0.2s;"
                                onfocus="this.style.borderColor='var(--accent)'"
                                onblur="this.style.borderColor='var(--border)'"
                            ></textarea>
                        </div>
                    `).join('');
                
                detailsDiv.style.display = 'block';
                detailsDiv.innerHTML = `
                    <div style="padding: 1.5rem; max-height: 80vh; overflow-y: auto;">
                        <button onclick="closeTemplateDetails()" 
                                style="background: none; border: none; color: var(--text-muted); font-size: 1.5rem; cursor: pointer; float: right; padding: 0; transition: all 0.2s;"
                                onmouseover="this.style.color='var(--text-primary)'"
                                onmouseout="this.style.color='var(--text-muted)'">
                            ×
                        </button>
                        <h3 style="font-size: 1.25rem; font-weight: 700; margin-bottom: 0.5rem; color: var(--text-primary);">${template.name}</h3>
                        <p style="color: var(--text-secondary); font-size: 0.95rem; margin-bottom: 1rem;">${template.description}</p>
                        
                        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.75rem; margin-bottom: 1.5rem;">
                            <div style="background: var(--bg-elevated); padding: 0.75rem; border-radius: 8px; border: 1px solid var(--border); text-align: center;">
                                <div style="font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase; margin-bottom: 0.25rem;">Category</div>
                                <div style="font-weight: 600; text-transform: capitalize; color: var(--text-primary); font-size: 0.85rem;">${template.category.replace('_', ' ')}</div>
                            </div>
                            <div style="background: var(--bg-elevated); padding: 0.75rem; border-radius: 8px; border: 1px solid var(--border); text-align: center;">
                                <div style="font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase; margin-bottom: 0.25rem;">Est. Time</div>
                                <div style="font-weight: 600; color: var(--text-primary); font-size: 0.85rem;">${template.estimated_time}</div>
                            </div>
                            <div style="background: var(--bg-elevated); padding: 0.75rem; border-radius: 8px; border: 1px solid var(--border); text-align: center;">
                                <div style="font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase; margin-bottom: 0.25rem;">Questions</div>
                                <div style="font-weight: 600; color: var(--accent); font-size: 0.85rem;">${template.questions.length}</div>
                            </div>
                        </div>
                        
                        <form id="templateQuestionsForm" onsubmit="submitTemplateAnswers(event)">
                            <div style="margin-bottom: 1.5rem;">
                                <div style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.75rem; font-weight: 600;">
                                    Answer the Questions
                                </div>
                                <div style="background: var(--bg-elevated); border-radius: 8px; border: 1px solid var(--border); overflow: hidden;">
                                    ${questionsHtml}
                                </div>
                            </div>
                            
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem;">
                                <div>
                                    <div style="font-size: 0.7rem; color: var(--success); text-transform: uppercase; margin-bottom: 0.5rem; font-weight: 600;">Pros</div>
                                    <div style="display: flex; flex-direction: column; gap: 0.25rem;">
                                        ${template.pros_factors.slice(0, 3).map(p => `<div style="font-size: 0.8rem; color: var(--text-secondary);">• ${p}</div>`).join('')}
                                    </div>
                                </div>
                                <div>
                                    <div style="font-size: 0.7rem; color: var(--danger); text-transform: uppercase; margin-bottom: 0.5rem; font-weight: 600;">Cons</div>
                                    <div style="display: flex; flex-direction: column; gap: 0.25rem;">
                                        ${template.cons_factors.slice(0, 3).map(p => `<div style="font-size: 0.8rem; color: var(--text-secondary);">• ${p}</div>`).join('')}
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Additional Details Section -->
                            <div style="margin-bottom: 1.5rem; padding: 1rem; background: var(--bg-elevated); border-radius: 8px; border: 1px solid var(--border);">
                                <div style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 1rem; font-weight: 600;">
                                    Additional Details
                                </div>
                                
                                <div style="margin-bottom: 1rem;">
                                    <label style="display: block; font-size: 0.8rem; color: var(--text-secondary); font-weight: 500; margin-bottom: 0.5rem;">Brief Description</label>
                                    <textarea id="templateDescription" 
                                        placeholder="Describe your decision situation in a few sentences..."
                                        style="width: 100%; min-height: 70px; padding: 0.75rem; background: var(--bg-primary); border: 1px solid var(--border); border-radius: 8px; color: var(--text-primary); font-family: inherit; font-size: 0.85rem; resize: vertical; transition: all 0.2s;"
                                        onfocus="this.style.borderColor='var(--accent)'"
                                        onblur="this.style.borderColor='var(--border)'"
                                    ></textarea>
                                </div>
                                
                                <div>
                                    <label style="display: block; font-size: 0.8rem; color: var(--text-secondary); font-weight: 500; margin-bottom: 0.5rem;">Relevant Skills (comma separated)</label>
                                    <input type="text" id="templateSkills" 
                                        placeholder="e.g., leadership, programming, marketing..."
                                        style="width: 100%; padding: 0.75rem; background: var(--bg-primary); border: 1px solid var(--border); border-radius: 8px; color: var(--text-primary); font-family: inherit; font-size: 0.85rem; transition: all 0.2s;"
                                        onfocus="this.style.borderColor='var(--accent)'"
                                        onblur="this.style.borderColor='var(--border)'"
                                    >
                                </div>
                            </div>
                            
                            <!-- Face Emotion Detection -->
                            <div style="margin-bottom: 1.5rem; padding: 1rem; background: var(--bg-elevated); border-radius: 8px; border: 1px solid var(--border);">
                                <div style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 1rem; font-weight: 600;">
                                    Emotion Detection (Optional)
                                </div>
                                
                                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                                    <div style="position: relative;">
                                        <div id="templateCameraPlaceholder" 
                                            style="background: var(--bg-primary); border-radius: 8px; height: 120px; display: flex; align-items: center; justify-content: center; border: 1px solid var(--border);">
                                            <span style="color: var(--text-muted); font-size: 0.8rem;">Camera Preview</span>
                                        </div>
                                        <video id="templateCameraVideo" autoplay playsinline style="display: none; width: 100%; height: 120px; object-fit: cover; border-radius: 8px;"></video>
                                        <canvas id="templateCameraCanvas" style="display: none;"></canvas>
                                    </div>
                                    <div style="display: flex; flex-direction: column; gap: 0.5rem;">
                                        <button type="button" id="templateStartCameraBtn" onclick="startTemplateCamera()" 
                                            style="padding: 0.5rem; background: var(--bg-primary); border: 1px solid var(--border); border-radius: 6px; color: var(--text-primary); font-size: 0.75rem; cursor: pointer;">
                                            Start Camera
                                        </button>
                                        <button type="button" id="templateCaptureFaceBtn" onclick="captureTemplateEmotion()" 
                                            style="display: none; padding: 0.5rem; background: var(--accent); border: none; border-radius: 6px; color: var(--bg-primary); font-size: 0.75rem; cursor: pointer; font-weight: 600;">
                                            Capture Emotion
                                        </button>
                                        <button type="button" id="templateStopCameraBtn" onclick="stopTemplateCamera()" 
                                            style="display: none; padding: 0.5rem; background: var(--bg-primary); border: 1px solid var(--danger); border-radius: 6px; color: var(--danger); font-size: 0.75rem; cursor: pointer;">
                                            Stop Camera
                                        </button>
                                        <div id="templateEmotionResults" style="margin-top: auto;">
                                            <div style="font-size: 0.75rem; color: var(--text-muted);">No emotion captured</div>
                                        </div>
                                        <input type="hidden" id="templateDetectedEmotion" value="">
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Decision Analyzer Sliders -->
                            <div style="margin-bottom: 1.5rem; padding: 1rem; background: var(--bg-elevated); border-radius: 8px; border: 1px solid var(--border);">
                                <div style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 1rem; font-weight: 600;">
                                    Decision Parameters
                                </div>
                                
                                <div style="margin-bottom: 1rem;">
                                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                                        <label style="font-size: 0.8rem; color: var(--text-secondary); font-weight: 500;">Risk Tolerance</label>
                                        <span id="templateRiskValue" style="font-size: 0.8rem; color: var(--accent); font-weight: 600; font-family: monospace;">50%</span>
                                    </div>
                                    <input type="range" id="templateRiskTolerance" min="0" max="100" value="50" 
                                        style="width: 100%; height: 6px; border-radius: 3px; background: var(--bg-hover); -webkit-appearance: none; appearance: none; cursor: pointer;"
                                        oninput="document.getElementById('templateRiskValue').textContent = this.value + '%'">
                                </div>
                                
                                <div>
                                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                                        <label style="font-size: 0.8rem; color: var(--text-secondary); font-weight: 500;">Financial Stability</label>
                                        <span id="templateFinancialValue" style="font-size: 0.8rem; color: var(--accent); font-weight: 600; font-family: monospace;">50%</span>
                                    </div>
                                    <input type="range" id="templateFinancialStability" min="0" max="100" value="50"
                                        style="width: 100%; height: 6px; border-radius: 3px; background: var(--bg-hover); -webkit-appearance: none; appearance: none; cursor: pointer;"
                                        oninput="document.getElementById('templateFinancialValue').textContent = this.value + '%'">
                                </div>
                            </div>
                            
                            <div style="display: flex; gap: 0.75rem;">
                                <button type="submit" 
                                        style="flex: 2; padding: 0.875rem; background: var(--accent); color: var(--bg-primary); border: none; border-radius: 8px; font-weight: 600; cursor: pointer; transition: all 0.2s;"
                                        onmouseover="this.style.opacity='0.9'; this.style.transform='translateY(-2px)'"
                                        onmouseout="this.style.opacity='1'; this.style.transform='translateY(0)'">
                                    Analyze Decision
                                </button>
                                <button type="button" onclick="closeTemplateDetails()" 
                                        style="flex: 1; padding: 0.875rem; background: var(--bg-elevated); color: var(--text-primary); border: 1px solid var(--border); border-radius: 8px; font-weight: 600; cursor: pointer; transition: all 0.2s;"
                                        onmouseover="this.style.background='var(--bg-hover)'"
                                        onmouseout="this.style.background='var(--bg-elevated)'">
                                    Cancel
                                </button>
                            </div>
                        </form>
                    </div>
                `;
                showToast('Fill out the template questions');
            } catch (error) {
                console.error('Failed to load template:', error);
                showToast('Error: ' + error.message);
            }
        }
        
        function closeTemplateDetails() {
            stopTemplateCamera();
            document.getElementById('templateDetails').style.display = 'none';
            document.getElementById('templateDetails').innerHTML = '';
        }
        
        // Template camera stream reference
        let templateCameraStream = null;
        
        async function startTemplateCamera() {
            try {
                const video = document.getElementById('templateCameraVideo');
                const placeholder = document.getElementById('templateCameraPlaceholder');
                
                templateCameraStream = await navigator.mediaDevices.getUserMedia({ video: true });
                video.srcObject = templateCameraStream;
                video.style.display = 'block';
                if (placeholder) placeholder.style.display = 'none';
                
                document.getElementById('templateStartCameraBtn').style.display = 'none';
                document.getElementById('templateCaptureFaceBtn').style.display = 'block';
                document.getElementById('templateStopCameraBtn').style.display = 'block';
                
                showToast('Camera started');
            } catch (error) {
                showToast('Camera access denied: ' + error.message);
            }
        }
        
        function stopTemplateCamera() {
            if (templateCameraStream) {
                templateCameraStream.getTracks().forEach(track => track.stop());
                templateCameraStream = null;
            }
            const video = document.getElementById('templateCameraVideo');
            const placeholder = document.getElementById('templateCameraPlaceholder');
            if (video) video.style.display = 'none';
            if (placeholder) placeholder.style.display = 'flex';
            
            const startBtn = document.getElementById('templateStartCameraBtn');
            const captureBtn = document.getElementById('templateCaptureFaceBtn');
            const stopBtn = document.getElementById('templateStopCameraBtn');
            if (startBtn) startBtn.style.display = 'block';
            if (captureBtn) captureBtn.style.display = 'none';
            if (stopBtn) stopBtn.style.display = 'none';
        }
        
        async function captureTemplateEmotion() {
            const video = document.getElementById('templateCameraVideo');
            const canvas = document.getElementById('templateCameraCanvas');
            const resultsDiv = document.getElementById('templateEmotionResults');
            const emotionInput = document.getElementById('templateDetectedEmotion');
            
            if (!video || !canvas) return;
            
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0);
            
            const imageData = canvas.toDataURL('image/jpeg');
            
            try {
                const response = await fetch('/api/emotion/detect', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: imageData })
                });
                const result = await response.json();
                
                if (result.emotions && result.emotions.length > 0) {
                    const emotion = result.emotions[0];
                    emotionInput.value = emotion.dominant_emotion || emotion.emotion || '';
                    const confidence = Math.round((emotion.confidence || 0.8) * 100);
                    resultsDiv.innerHTML = `
                        <div style="font-size: 0.8rem; color: var(--success); font-weight: 600;">
                            ${getEmotionLabel(emotionInput.value)}
                        </div>
                        <div style="font-size: 0.7rem; color: var(--text-muted);">${confidence}% confidence</div>
                    `;
                    showToast('Emotion captured: ' + emotionInput.value);
                } else {
                    resultsDiv.innerHTML = '<div style="font-size: 0.75rem; color: var(--warning);">No face detected</div>';
                    showToast('No face detected');
                }
            } catch (error) {
                resultsDiv.innerHTML = '<div style="font-size: 0.75rem; color: var(--danger);">Detection failed</div>';
                showToast('Emotion detection failed');
            }
        }
        
        function getEmotionLabel(emotion) {
            const labels = { happy: 'Happy', sad: 'Sad', angry: 'Angry', fear: 'Fear', surprise: 'Surprise', neutral: 'Neutral', disgust: 'Disgust' };
            return labels[emotion?.toLowerCase()] || emotion || 'Unknown';
        }
        
        async function submitTemplateAnswers(event) {
            event.preventDefault();
            
            const template = window.currentTemplate;
            if (!template) {
                showToast('No template selected');
                return;
            }
            
            // Collect all answers
            const answers = [];
            let allAnswered = true;
            
            template.questions.forEach((q, i) => {
                const input = document.getElementById('template-answer-' + i);
                const answer = input ? input.value.trim() : '';
                answers.push({
                    question: q.question,
                    answer: answer
                });
                if (!answer) allAnswered = false;
            });
            
            if (!allAnswered) {
                showToast('Please answer all questions');
                return;
            }
            
            // Create a summary from answers
            const summary = answers.map(a => a.answer).join(' ');
            
            // Submit for analysis
            document.getElementById('loading').classList.add('active');
            
            try {
                // Get slider values from template form
                const riskSlider = document.getElementById('templateRiskTolerance');
                const financialSlider = document.getElementById('templateFinancialStability');
                const riskTolerance = riskSlider ? riskSlider.value / 100 : 0.5;
                const financialStability = financialSlider ? financialSlider.value / 100 : 0.5;
                
                // Get additional details
                const descriptionEl = document.getElementById('templateDescription');
                const skillsEl = document.getElementById('templateSkills');
                const emotionEl = document.getElementById('templateDetectedEmotion');
                
                const additionalDescription = descriptionEl ? descriptionEl.value.trim() : '';
                const skills = skillsEl ? skillsEl.value.split(',').map(s => s.trim()).filter(s => s) : [];
                const detectedEmotion = emotionEl ? emotionEl.value : '';
                
                // Combine answers summary with additional description
                const fullDescription = additionalDescription ? `${additionalDescription}\n\nResponses: ${summary}` : summary;
                
                // Build emotions array
                const emotions = detectedEmotion ? [detectedEmotion] : [];
                
                const data = {
                    decision_type: template.category || 'general',
                    description: fullDescription,
                    risk_tolerance: riskTolerance,
                    financial_stability: financialStability,
                    emotions: emotions,
                    skills: skills,
                    user_id: userId,
                    template_id: template.id,
                    template_answers: answers
                };
                
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                const result = await response.json();
                
                // Save to journal
                saveToJournal(data, result);
                
                // Display results in template details panel
                displayTemplateResults(result, template.name);
                
                showToast('Analysis complete!');
            } catch (error) {
                showToast('Analysis failed: ' + error.message);
            }
            
            document.getElementById('loading').classList.remove('active');
        }
        
        function displayTemplateResults(result, templateName) {
            const detailsDiv = document.getElementById('templateDetails');
            if (!detailsDiv) return;
            
            const regret = Math.round((result.predicted_regret || result.regret || 0) * 100);
            const recommendation = result.recommendation || 'No recommendation available';
            const factors = result.factors || [];
            const pros = factors.filter(f => f.impact > 0 || f.type === 'pro').slice(0, 4);
            const cons = factors.filter(f => f.impact < 0 || f.type === 'con').slice(0, 4);
            
            detailsDiv.innerHTML = `
                <div style="padding: 1.5rem; max-height: 80vh; overflow-y: auto;">
                    <button onclick="closeTemplateDetails()" 
                            style="background: none; border: none; color: var(--text-muted); font-size: 1.5rem; cursor: pointer; float: right; padding: 0;">
                        ×
                    </button>
                    
                    <h3 style="font-size: 1.25rem; font-weight: 700; margin-bottom: 0.5rem; color: var(--text-primary);">
                        Analysis Results
                    </h3>
                    <p style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 1.5rem;">
                        Based on template: ${templateName}
                    </p>
                    
                    <!-- Regret Score -->
                    <div style="background: linear-gradient(135deg, ${regret < 30 ? '#22c55e33' : regret < 60 ? '#eab30833' : '#ef444433'}, transparent); padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; text-align: center;">
                        <div style="font-size: 3rem; font-weight: 700; color: ${regret < 30 ? 'var(--success)' : regret < 60 ? 'var(--warning)' : 'var(--danger)'};">
                            ${regret}%
                        </div>
                        <div style="font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px;">
                            Predicted Regret
                        </div>
                        <div style="margin-top: 0.5rem; font-size: 0.9rem; color: var(--text-secondary);">
                            ${regret < 30 ? 'Low risk of regret' : regret < 60 ? 'Moderate risk' : 'High risk - consider alternatives'}
                        </div>
                    </div>
                    
                    <!-- Recommendation -->
                    <div style="background: var(--bg-elevated); padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem; border-left: 3px solid var(--accent);">
                        <div style="font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; margin-bottom: 0.5rem;">Recommendation</div>
                        <div style="color: var(--text-primary); line-height: 1.6;">${recommendation}</div>
                    </div>
                    
                    <!-- Pros and Cons -->
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.5rem;">
                        <div style="background: var(--bg-elevated); padding: 1rem; border-radius: 8px;">
                            <div style="font-size: 0.7rem; color: var(--success); text-transform: uppercase; margin-bottom: 0.75rem; font-weight: 600;">Positive Factors</div>
                            ${pros.length ? pros.map(f => `<div style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.5rem;">• ${f.description || f.name || f}</div>`).join('') : '<div style="font-size: 0.85rem; color: var(--text-muted);">No major positives identified</div>'}
                        </div>
                        <div style="background: var(--bg-elevated); padding: 1rem; border-radius: 8px;">
                            <div style="font-size: 0.7rem; color: var(--danger); text-transform: uppercase; margin-bottom: 0.75rem; font-weight: 600;">Risk Factors</div>
                            ${cons.length ? cons.map(f => `<div style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.5rem;">• ${f.description || f.name || f}</div>`).join('') : '<div style="font-size: 0.85rem; color: var(--text-muted);">No major risks identified</div>'}
                        </div>
                    </div>
                    
                    <div style="display: flex; gap: 0.75rem;">
                        <button onclick="closeTemplateDetails()" 
                                style="flex: 1; padding: 0.875rem; background: var(--accent); color: var(--bg-primary); border: none; border-radius: 8px; font-weight: 600; cursor: pointer;">
                            Done
                        </button>
                        <button onclick="showTab('journal')" 
                                style="flex: 1; padding: 0.875rem; background: var(--bg-elevated); color: var(--text-primary); border: 1px solid var(--border); border-radius: 8px; font-weight: 600; cursor: pointer;">
                            View History
                        </button>
                    </div>
                </div>
            `;
        }


        function updateSettingsStatus() {
            const statusEl = document.getElementById('settingsConnectionStatus');
            const mode = document.getElementById('statusMode').textContent;
            statusEl.innerHTML = `<span style="color:var(--text-primary);">${mode}</span>`;
        }

        function setTheme(theme) {
            document.querySelectorAll('.theme-btn').forEach(btn => {
                btn.style.borderColor = 'var(--border)';
            });

            if (theme === 'light') {
                document.documentElement.setAttribute('data-theme', 'light');
                document.getElementById('themeLight').style.borderColor = 'var(--accent)';
            } else if (theme === 'dark') {
                document.documentElement.removeAttribute('data-theme');
                document.getElementById('themeDark').style.borderColor = 'var(--accent)';
            } else {
                const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
                if (prefersDark) {
                    document.documentElement.removeAttribute('data-theme');
                } else {
                    document.documentElement.setAttribute('data-theme', 'light');
                }
                document.getElementById('themeSystem').style.borderColor = 'var(--accent)';
            }
            localStorage.setItem('theme', theme);
            showToast('Theme changed to ' + theme);
        }

        function clearAllData() {
            if (confirm('Are you sure you want to clear all data? This cannot be undone.')) {
                localStorage.removeItem('decisions');
                localStorage.removeItem('quickDecisions');
                journalDecisions = [];
                updateJournal();
                toggleSettings();
                showToast('All data cleared');
            }
        }

        // Voice input with fallback
        let recognition = null;
        let isListening = false;
        let mediaRecorder = null;
        let audioChunks = [];

        function initVoiceRecognition() {
            try {
                const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                if (SpeechRecognition) {
                    recognition = new SpeechRecognition();
                    recognition.continuous = false;
                    recognition.interimResults = true;
                    recognition.lang = 'en-US';

                    recognition.onstart = () => {
                        isListening = true;
                        updateVoiceButton(true);
                        showToast('Listening...');
                    };

                    recognition.onend = () => {
                        isListening = false;
                        updateVoiceButton(false);
                    };

                    recognition.onresult = (event) => {
                        let transcript = '';
                        for (let i = event.resultIndex; i < event.results.length; i++) {
                            if (event.results[i].isFinal) {
                                transcript += event.results[i][0].transcript;
                            }
                        }
                        if (transcript) {
                            document.getElementById('chatInput').value = transcript;
                            document.getElementById('charCount').textContent = transcript.length;
                            showToast('Voice captured!');
                        }
                    };

                    recognition.onerror = (event) => {
                        console.log('Speech recognition error:', event.error);
                        isListening = false;
                        updateVoiceButton(false);
                        // Fall back to MediaRecorder for any error
                        recognition = null;
                        showToast('Switching to local recording...');
                        setTimeout(() => startMediaRecorder(), 300);
                    };
                }
            } catch (e) {
                console.log('Web Speech API not supported, will use MediaRecorder');
            }
        }

        function updateVoiceButton(active) {
            const btn = document.getElementById('voiceBtn');
            if (btn) {
                if (active) {
                    btn.classList.add('recording');
                    btn.innerHTML = `<svg viewBox="0 0 24 24" fill="currentColor" stroke="none">
                        <rect x="6" y="4" width="12" height="16" rx="2"/>
                    </svg>`;
                } else {
                    btn.classList.remove('recording');
                    btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
                        <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                        <line x1="12" x2="12" y1="19" y2="22"/>
                    </svg>`;
                }
            }
        }

        async function startMediaRecorder() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                
                mediaRecorder.ondataavailable = (event) => {
                    audioChunks.push(event.data);
                };
                
                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    stream.getTracks().forEach(track => track.stop());
                    updateVoiceButton(false);
                    isListening = false;
                    
                    // Send to backend for transcription
                    showToast('Processing audio...');
                    try {
                        const formData = new FormData();
                        formData.append('audio', audioBlob, 'recording.webm');
                        
                        const response = await fetch('/api/speech-to-text', {
                            method: 'POST',
                            body: formData
                        });
                        
                        if (response.ok) {
                            const data = await response.json();
                            if (data.text) {
                                document.getElementById('chatInput').value = data.text;
                                document.getElementById('charCount').textContent = data.text.length;
                                showToast('Voice captured!');
                            } else {
                                showToast('No speech detected. Try again.');
                            }
                        } else {
                            showToast('Transcription failed. Try typing instead.');
                        }
                    } catch (e) {
                        console.error('Transcription error:', e);
                        showToast('Voice processing unavailable. Try typing.');
                    }
                };
                
                mediaRecorder.start();
                isListening = true;
                updateVoiceButton(true);
                showToast('Recording... Click again to stop');
                
            } catch (e) {
                console.error('MediaRecorder error:', e);
                showToast('Microphone access denied');
            }
        }

        function toggleVoiceInput() {
            // If using MediaRecorder
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
                return;
            }
            
            // If no recognition yet, try to initialize
            if (!recognition) {
                initVoiceRecognition();
            }

            if (recognition) {
                if (isListening) {
                    recognition.stop();
                } else {
                    try {
                        recognition.start();
                    } catch (e) {
                        // Fall back to MediaRecorder
                        startMediaRecorder();
                    }
                }
            } else {
                // No Web Speech API, use MediaRecorder directly
                startMediaRecorder();
            }
        }

        // Check system status
        async function checkSystemStatus() {
            const dot = document.getElementById('statusDot');
            const mode = document.getElementById('statusMode');
            
            if (!dot || !mode) {
                console.log('Status elements not found, retrying...');
                return;
            }
            
            try {
                const response = await fetch('/api/health');
                if (!response.ok) throw new Error('Health check failed');
                const data = await response.json();
                
                console.log('Health check:', data.ollama ? 'Ollama connected' : 'Offline');
                
                if (data.ollama === true) {
                    dot.className = 'status-dot';
                    mode.textContent = 'Online (AI)';
                } else {
                    dot.className = 'status-dot offline';
                    mode.textContent = 'Offline Mode';
                }
            } catch (error) {
                console.log('Health check error:', error.message);
                if (dot) dot.className = 'status-dot offline';
                if (mode) mode.textContent = 'Disconnected';
            }
        }

        // Load saved theme
        function loadSavedTheme() {
            const saved = localStorage.getItem('theme') || 'dark';
            setTheme(saved);
        }

        // Camera and Emotion Detection
        let cameraStream = null;

        async function startCamera() {
            try {
                cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
                const video = document.getElementById('cameraVideo');
                video.srcObject = cameraStream;
                document.getElementById('cameraPlaceholder').style.display = 'none';
                document.getElementById('startCameraBtn').style.display = 'none';
                document.getElementById('captureFaceBtn').style.display = 'block';
                document.getElementById('stopCameraBtn').style.display = 'block';
                showToast('Camera started');
            } catch (error) {
                showToast('Camera access denied');
            }
        }

        function stopCamera() {
            if (cameraStream) {
                cameraStream.getTracks().forEach(track => track.stop());
                cameraStream = null;
            }
            document.getElementById('cameraVideo').srcObject = null;
            document.getElementById('cameraPlaceholder').style.display = 'flex';
            document.getElementById('startCameraBtn').style.display = 'block';
            document.getElementById('captureFaceBtn').style.display = 'none';
            document.getElementById('stopCameraBtn').style.display = 'none';
        }

        async function captureAndAnalyze() {
            const video = document.getElementById('cameraVideo');
            const canvas = document.getElementById('cameraCanvas');
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 480;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0);
            const imageData = canvas.toDataURL('image/jpeg', 0.8);

            document.getElementById('emotionResults').innerHTML = '<div class="empty-state" style="font-size:0.8rem;">Analyzing...</div>';

            try {
                const response = await fetch('/api/emotion/detect', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: imageData, user_id: userId })
                });
                const data = await response.json();

                if (data.emotions && data.emotions.length > 0) {
                    const topEmotion = data.emotions[0];
                    document.getElementById('detectedEmotion').value = topEmotion.emotion;
                    document.getElementById('detectedEmotionLabel').textContent = topEmotion.emotion;
                    document.getElementById('detectedEmotionDisplay').style.display = 'block';

                    let html = '<div style="font-size:0.75rem;color:var(--text-muted);margin-bottom:0.5rem;">Detected Emotions:</div>';
                    data.emotions.slice(0, 3).forEach(e => {
                        html += `<div style="display:flex;justify-content:space-between;padding:0.25rem 0;font-size:0.8rem;"><span>${e.emotion}</span><span>${(e.confidence * 100).toFixed(0)}%</span></div>`;
                    });
                    document.getElementById('emotionResults').innerHTML = html;
                    showToast('Emotion detected: ' + topEmotion.emotion);
                } else {
                    document.getElementById('emotionResults').innerHTML = '<div class="empty-state" style="font-size:0.8rem;">No face detected</div>';
                }
            } catch (error) {
                document.getElementById('emotionResults').innerHTML = '<div class="empty-state" style="font-size:0.8rem;">Detection failed</div>';
            }
        }

        // Initialize
        loadSavedTheme();
        initVoiceRecognition();
        
        // Check status immediately and a few times quickly, then every 10 seconds
        checkSystemStatus();
        setTimeout(checkSystemStatus, 500);
        setTimeout(checkSystemStatus, 1500);
        setInterval(checkSystemStatus, 10000);
        
        updateJournal();

        // 3D Background Animation
        function init3DBackground() {
            const container = document.getElementById('canvas-container');
            if (!container) return;
            
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
            
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            container.appendChild(renderer.domElement);
            
            // Particles
            const particlesGeometry = new THREE.BufferGeometry();
            const particlesCount = 700;
            const posArray = new Float32Array(particlesCount * 3);
            
            for(let i = 0; i < particlesCount * 3; i++) {
                posArray[i] = (Math.random() - 0.5) * 15;
            }
            
            particlesGeometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
            
            // Material
            const material = new THREE.PointsMaterial({
                size: 0.02,
                color: 0x444444,
                transparent: true,
                opacity: 0.4
            });
            
            // Theme adaptation
            const updateMaterialColor = () => {
                const isDark = !document.documentElement.getAttribute('data-theme');
                material.color.setHex(isDark ? 0xaaaaaa : 0x222222);
                material.opacity = isDark ? 0.3 : 0.2;
            };
            
            // Hook into existing setTheme function
            const originalSetTheme = window.setTheme;
            window.setTheme = function(theme) {
                if (originalSetTheme) originalSetTheme(theme);
                updateMaterialColor();
            };
            
            // Initial call
            updateMaterialColor();
            
            const particlesMesh = new THREE.Points(particlesGeometry, material);
            scene.add(particlesMesh);
            
            camera.position.z = 3;
            
            // Mouse interaction
            let mouseX = 0;
            let mouseY = 0;
            
            document.addEventListener('mousemove', (event) => {
                mouseX = event.clientX / window.innerWidth - 0.5;
                mouseY = event.clientY / window.innerHeight - 0.5;
            });
            
            const clock = new THREE.Clock();
            
            function animate() {
                requestAnimationFrame(animate);
                const elapsedTime = clock.getElapsedTime();
                
                particlesMesh.rotation.y = elapsedTime * 0.05;
                particlesMesh.rotation.x = mouseY * 0.1;
                particlesMesh.rotation.y += mouseX * 0.1;
                
                renderer.render(scene, camera);
            }
            
            animate();
            
            // Resize handler
            window.addEventListener('resize', () => {
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            });
        }
        
        // Defer initialization to ensure Three.js is loaded
        setTimeout(init3DBackground, 500);

        // ============= ADVANCED FEATURES =============
        
        // Goals Management
        async function createGoal() {
            const title = document.getElementById('goalTitle').value.trim();
            const category = document.getElementById('goalCategory').value;
            const description = document.getElementById('goalDescription').value.trim();
            
            if (!title) {
                showToast('Please enter a goal title', 'warning');
                return;
            }
            
            try {
                    const response = await fetch('/api/analytics/goals', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ user_id: userId, title, category, description })
                });
                const data = await response.json();
                
                if (data.goal_id) {
                    showToast('Goal created successfully!', 'success');
                    document.getElementById('goalTitle').value = '';
                    document.getElementById('goalDescription').value = '';
                    loadGoals();
                }
            } catch (error) {
                showToast('Failed to create goal', 'error');
            }
        }
        
        async function loadGoals() {
            try {
                const response = await fetch(`/api/analytics/goals/${userId}`);
                const data = await response.json();
                const goals = data.goals || [];
                
                const container = document.getElementById('goalsList');
                if (!goals || goals.length === 0) {
                    container.innerHTML = '<div class="empty-state">Create your first career goal above</div>';
                    return;
                }
                
                container.innerHTML = goals.map(goal => `
                    <div style="padding:1rem;background:var(--bg-elevated);border-radius:8px;margin-bottom:0.75rem;border-left:4px solid ${goal.on_track ? 'var(--success)' : 'var(--warning)'};">
                        <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                            <div>
                                <div style="font-weight:600;">${goal.title}</div>
                                <div style="color:var(--text-secondary);font-size:0.85rem;margin-top:0.25rem;">${goal.description || 'No description'}</div>
                                <div style="display:flex;gap:1rem;margin-top:0.5rem;">
                                    <span style="font-size:0.75rem;color:var(--text-muted);">Category: ${goal.category}</span>
                                    <span style="font-size:0.75rem;color:${goal.on_track ? 'var(--success)' : 'var(--warning)'};">${goal.on_track ? '✓ On Track' : '⚠ Needs Attention'}</span>
                                </div>
                            </div>
                            <div style="text-align:right;">
                                <div style="font-size:1.5rem;font-weight:700;">${goal.progress || 0}%</div>
                                <div style="color:var(--text-muted);font-size:0.75rem;">Progress</div>
                            </div>
                        </div>
                    </div>
                `).join('');
            } catch (error) {
                console.error('Failed to load goals:', error);
            }
        }
        
        // Opportunity Scout
        async function scanOpportunities() {
            showToast('Scanning for opportunities...', 'info');
            
            try {
                await fetch(`/api/scout/register`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        user_id: userId,
                        current_role: 'Software Engineer',
                        industry: 'technology',
                        skills: ['Python', 'JavaScript', 'AI/ML'],
                        salary_target: 150000
                    })
                });
                
                const response = await fetch(`/api/scout/scan/${userId}`, { method: 'POST' });
                const data = await response.json();
                
                displayOpportunities(data.opportunities || []);
                loadMarketNews();
                loadApplications();
            } catch (error) {
                showToast('Failed to scan opportunities', 'error');
            }
        }

        async function loadOpportunities() {
            try {
                const [oppsRes, appsRes] = await Promise.all([
                    fetch(`/api/scout/opportunities/${userId}`),
                    fetch(`/api/scout/applications/${userId}`)
                ]);
                
                const oppsData = await oppsRes.json();
                const appsData = await appsRes.json();
                
                displayOpportunities(oppsData.opportunities || []);
                displayApplications(appsData.applications || []);
                loadMarketNews();
            } catch (e) {
                console.error('Failed to load scout data', e);
            }
        }
        
        function displayOpportunities(opportunities) {
            const container = document.getElementById('opportunitiesList');
            if (!container) return;

            if (!opportunities || opportunities.length === 0) {
                container.innerHTML = '<div class="empty-state">No opportunities found. Click "Scan Now".</div>';
                document.getElementById('totalOpportunities').textContent = '0';
                document.getElementById('highMatchCount').textContent = '0';
                return;
            }
            
            const highMatch = opportunities.filter(o => (o.match_score || 0) >= 0.75).length;
            document.getElementById('totalOpportunities').textContent = opportunities.length;
            document.getElementById('highMatchCount').textContent = highMatch;
            
            container.innerHTML = opportunities.map(opp => `
                <div style="padding:1.25rem;background:var(--bg-elevated);border-radius:12px;margin-bottom:1rem;border:1px solid var(--border);transition:all 0.3s ease;">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.75rem;">
                        <div>
                            <div style="font-weight:700;font-size:1.1rem;color:var(--text-primary);">${opp.title || 'Opportunity'}</div>
                            <div style="font-size:0.75rem;color:var(--text-muted);margin-top:0.25rem;">Source: ${opp.source}</div>
                        </div>
                        <div style="padding:0.3rem 0.8rem;background:${(opp.match_score || 0) >= 0.75 ? 'rgba(34, 197, 94, 0.15)' : 'rgba(234, 179, 8, 0.15)'};color:${(opp.match_score || 0) >= 0.75 ? '#4ade80' : '#facc15'};border-radius:20px;font-size:0.75rem;font-weight:700;border:1px solid currentColor;">
                            ${Math.round((opp.match_score || 0) * 100)}% Match
                        </div>
                    </div>
                    <div style="color:var(--text-secondary);font-size:0.9rem;margin-bottom:1rem;line-height:1.5;">${opp.description || ''}</div>
                    <div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:1.25rem;">
                        ${(opp.match_reasons || []).slice(0, 3).map(r => `
                            <span style="padding:0.25rem 0.6rem;background:rgba(255,255,255,0.05);border-radius:6px;font-size:0.7rem;color:var(--text-muted);">${r}</span>
                        `).join('')}
                    </div>
                    <div style="display:flex;gap:0.75rem;">
                        <button onclick="applyToOpportunity('${opp.id}')" style="flex:1;padding:0.6rem;background:var(--accent);color:var(--bg-primary);border:none;border-radius:8px;font-weight:700;cursor:pointer;font-size:0.8rem;transition:transform 0.2s;">Apply Now</button>
                        <button onclick="markOpportunity('${opp.id}', 'save')" style="padding:0.6rem 1rem;background:rgba(255,255,255,0.05);color:var(--text-primary);border:1px solid var(--border);border-radius:8px;font-weight:600;cursor:pointer;font-size:0.8rem;">Save</button>
                        <button onclick="markOpportunity('${opp.id}', 'dismiss')" style="padding:0.6rem;background:none;border:none;color:var(--danger);cursor:pointer;opacity:0.7;"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6L6 18M6 6l12 12"/></svg></button>
                    </div>
                </div>
            `).join('');
        }

        async function applyToOpportunity(oppId) {
            const notes = prompt("Add a note to your application (optional):");
            if (notes === null) return;

            try {
                const res = await fetch(`/api/scout/apply/${userId}/${oppId}`, { 
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ notes })
                });
                const data = await res.json();
                if (data.success) {
                    showToast('Application sent!', 'success');
                    loadOpportunities();
                }
            } catch (e) {
                showToast('Failed to apply', 'error');
            }
        }

        async function markOpportunity(oppId, action) {
            try {
                await fetch(`/api/scout/opportunity/${userId}/${oppId}/${action}`, { method: 'POST' });
                showToast(`Opportunity ${action}d`, 'info');
                loadOpportunities();
            } catch (e) {
                showToast('Action failed', 'error');
            }
        }

        async function loadApplications() {
            try {
                const response = await fetch(`/api/scout/applications/${userId}`);
                const data = await response.json();
                displayApplications(data.applications || []);
            } catch (e) {
                console.error('Failed to load applications', e);
            }
        }

        function displayApplications(applications) {
            const container = document.getElementById('applicationsList');
            if (!container) return;

            if (!applications || applications.length === 0) {
                container.innerHTML = '<div class="empty-state">No active applications</div>';
                return;
            }

            container.innerHTML = applications.map(app => `
                <div style="padding:1rem;background:var(--bg-elevated);border-radius:12px;border:1px solid var(--border);">
                    <div style="font-weight:700;margin-bottom:0.25rem;">${app.title}</div>
                    <div style="font-size:0.75rem;color:var(--text-muted);margin-bottom:0.75rem;">Applied on ${new Date(app.applied_at).toLocaleDateString()}</div>
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span style="padding:0.2rem 0.5rem;background:rgba(59, 130, 246, 0.1);color:#60a5fa;border-radius:4px;font-size:0.7rem;font-weight:700;text-transform:uppercase;">${app.status}</span>
                        <div style="font-size:0.7rem;color:var(--text-secondary);">Next: ${app.next_step}</div>
                    </div>
                </div>
            `).join('');
        }
        
        async function loadMarketNews() {
            try {
                const response = await fetch('/api/integrations/news?industry=technology&limit=3');
                const data = await response.json();
                
                const container = document.getElementById('newsContainer');
                if (!data.articles || data.articles.length === 0) {
                    container.innerHTML = '<div class="empty-state" style="font-size:0.85rem;">No news available</div>';
                    return;
                }
                
                container.innerHTML = data.articles.map(article => `
                    <div style="padding:0.75rem;background:var(--bg-elevated);border-radius:6px;border-left:3px solid var(--accent);">
                        <div style="font-weight:500;font-size:0.85rem;">${article.title}</div>
                        <div style="color:var(--text-muted);font-size:0.75rem;margin-top:0.25rem;">${article.source}</div>
                    </div>
                `).join('');
            } catch (error) {
                console.error('Failed to load news:', error);
            }
        }
        
        // Privacy Management
        async function updateConsent(consentType, granted) {
            try {
                await fetch('/api/privacy/consent', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ user_id: userId, consent_type: consentType, granted })
                });
                showToast(`Consent ${granted ? 'granted' : 'revoked'}`, 'success');
            } catch (error) {
                showToast('Failed to update consent', 'error');
            }
        }
        
        async function loadPrivacySettings() {
            try {
                const response = await fetch(`/api/privacy/dashboard/${userId}`);
                const data = await response.json();
                
                if (data.consents) {
                    document.getElementById('consentDataCollection').checked = data.consents.data_collection || false;
                    document.getElementById('consentAnalytics').checked = data.consents.analytics || false;
                    document.getElementById('consentGlobalInsights').checked = data.consents.global_insights || false;
                }
            } catch (error) {
                console.error('Failed to load privacy settings:', error);
            }
        }
        
        async function requestDataExport() {
            try {
                const response = await fetch('/api/privacy/export', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ user_id: userId })
                });
                const data = await response.json();
                showToast('Data export requested. You will be notified when ready.', 'success');
            } catch (error) {
                showToast('Failed to request export', 'error');
            }
        }
        
        async function requestAccountDeletion() {
            if (!confirm('Are you sure you want to delete your account? This action cannot be undone.')) {
                return;
            }
            
            try {
                const response = await fetch('/api/privacy/delete', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ user_id: userId, reason: 'User requested deletion' })
                });
                const data = await response.json();
                showToast('Account deletion scheduled. You have 30 days to cancel.', 'warning');
            } catch (error) {
                showToast('Failed to request deletion', 'error');
            }
        }
        

        // ============= PHASE 2 FEATURES =============

        // Enhanced Toast
        function showToast(msg, type = 'info') {
            const container = document.getElementById('toastContainer');
            if (!container) return;
            
            const toast = document.createElement('div');
            toast.style.padding = '1rem';
            toast.style.borderRadius = '8px';
            toast.style.background = 'var(--bg-elevated)';
            toast.style.color = 'var(--text-primary)';
            toast.style.boxShadow = 'var(--shadow-lg)';
            toast.style.border = '1px solid var(--border)';
            toast.style.borderLeft = `4px solid ${type === 'success' ? 'var(--success)' : type === 'error' ? 'var(--danger)' : type === 'warning' ? 'var(--warning)' : 'var(--accent)'}`;
            toast.style.animation = 'slideInRight 0.3s ease';
            toast.style.minWidth = '250px';
            toast.style.display = 'flex';
            toast.style.alignItems = 'center';
            toast.style.justifyContent = 'space-between';
            toast.style.zIndex = '10000';
            
            toast.innerHTML = `
                <div>${msg}</div>
                <button onclick="this.parentElement.remove()" style="background:none;border:none;color:var(--text-secondary);cursor:pointer;margin-left:1rem;">&times;</button>
            `;
            
            container.appendChild(toast);
            
            setTimeout(() => {
                toast.style.animation = 'slideInRight 0.3s ease reverse';
                setTimeout(() => toast.remove(), 300);
            }, 5000);
        }

        // Monitoring Dashboard
        window.DASHBOARD_VERSION = "v4-final";
        async function loadMonitoring() {
            console.log("[Monitoring] Starting load...");
            try {
                // Fetch all monitoring data in parallel
                const [healthRes, appRes, alertsRes, endpointsRes] = await Promise.all([
                    fetch('/api/monitoring/health').catch(e => ({error: e})),
                    fetch('/api/monitoring/metrics/application').catch(e => ({error: e})),
                    fetch('/api/monitoring/alerts').catch(e => ({error: e})),
                    fetch('/api/monitoring/metrics/endpoints').catch(e => ({error: e}))
                ]);

                // 1. Health Status
                if (healthRes.ok) {
                    const healthData = await healthRes.json();
                    console.log("[Monitoring] Health data:", healthData);
                    const statusEl = document.getElementById('systemStatus');
                    if (statusEl) {
                        statusEl.textContent = (healthData.status || 'healthy').toUpperCase();
                        statusEl.style.color = healthData.status === 'unhealthy' ? 'var(--danger)' : 
                                              healthData.status === 'degraded' ? 'var(--warning)' : 'var(--success)';
                    }
                }

                // 2. App Metrics
                if (appRes.ok) {
                    const appData = await appRes.json();
                    console.log("[Monitoring] App data:", appData);
                    const sets = [
                        ['uptime', appData.uptime?.formatted || '0s'],
                        ['errorRate', (appData.requests?.error_rate_percent || 0).toFixed(2) + '%'],
                        ['avgLatency', (appData.latency?.average_ms || 0).toFixed(0) + 'ms']
                    ];
                    sets.forEach(([id, val]) => {
                        const el = document.getElementById(id);
                        if (el) el.textContent = val;
                    });
                }

                // 3. Alerts
                if (alertsRes.ok) {
                    const alertsData = await alertsRes.json();
                    const container = document.getElementById('activeAlerts');
                    if (container) {
                        if (alertsData.alerts && alertsData.alerts.length > 0) {
                            container.innerHTML = alertsData.alerts.map(a => `
                                <div style="padding:0.75rem;background:rgba(239,68,68,0.1);border:1px solid var(--danger);border-radius:6px;margin-bottom:0.5rem;display:flex;justify-content:space-between;align-items:center;">
                                    <div>
                                        <div style="font-weight:600;color:var(--danger);">${a.title || 'Alert'}</div>
                                        <div style="font-size:0.8rem;color:var(--text-primary);">${a.message || ''}</div>
                                    </div>
                                    <button onclick="acknowledgeAlert('${a.id}')" style="padding:0.25rem 0.5rem;background:var(--bg-elevated);border:1px solid var(--border);border-radius:4px;cursor:pointer;">Ack</button>
                                </div>
                            `).join('');
                        } else {
                            container.innerHTML = '<div class="empty-state">No active alerts. System healthy.</div>';
                        }
                    }
                }

                // 4. Endpoints
                if (endpointsRes.ok) {
                    const endData = await endpointsRes.json();
                    const container = document.getElementById('topEndpoints');
                    if (container) {
                        const endpoints = endData.endpoints || [];
                        if (endpoints.length > 0) {
                            container.innerHTML = endpoints.map(e => `
                                <div style="display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid var(--border);">
                                    <div style="font-family:monospace;font-size:0.85rem;">${e.method || ''} ${e.endpoint || '/'}</div>
                                    <div style="display:flex;gap:1rem;font-size:0.85rem;">
                                        <span>${e.request_count || 0} req</span>
                                        <span>${(e.avg_latency_ms || 0).toFixed(0)}ms</span>
                                    </div>
                                </div>
                            `).join('');
                        } else {
                            container.innerHTML = '<div class="empty-state">No request data yet</div>';
                        }
                    }
                }
            } catch (error) {
                console.error('[Monitoring] Critical failure:', error);
                showToast('Monitoring update failed', 'error');
            }
        }

        async function acknowledgeAlert(alertId) {
            try {
                await fetch(`/api/monitoring/alerts/acknowledge/${alertId}`, { method: 'POST' });
                loadMonitoring();
                showToast('Alert acknowledged', 'success');
            } catch (e) {
                showToast('Failed to acknowledge alert', 'error');
            }
        }

        // Calendar System
        async function loadCalendar() {
            console.log("[Calendar] Loading for user:", userId);
            
            try {
                // Fetch all data in parallel
                const [agendaRes, eventsRes, statusRes] = await Promise.all([
                    fetch(`/api/calendar/today/${userId}`).catch(e => ({error: e})),
                    fetch(`/api/calendar/events/${userId}`).catch(e => ({error: e})),
                    fetch(`/api/calendar/status/${userId}`).catch(e => ({error: e}))
                ]);

                // Today's Agenda
                if (agendaRes.ok) {
                    const agendaData = await agendaRes.json();
                    const container = document.getElementById('todayAgenda');
                    if (container) {
                        const events = agendaData.events || [];
                        if (events.length > 0) {
                            container.innerHTML = events.map(evt => `
                                <div style="padding:0.75rem;background:var(--bg-elevated);border-left:3px solid var(--accent);border-radius:4px;margin-bottom:0.5rem;">
                                    <div style="font-weight:600;">${evt.title || 'Untitled'}</div>
                                    <div style="font-size:0.8rem;color:var(--text-secondary);">
                                        ${evt.start ? new Date(evt.start).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}) : 'All day'}
                                        ${evt.end ? ' - ' + new Date(evt.end).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}) : ''}
                                    </div>
                                </div>
                            `).join('');
                        } else {
                            container.innerHTML = '<div class="empty-state">No events scheduled for today</div>';
                        }
                    }
                }

                // Upcoming Events
                if (eventsRes.ok) {
                    const eventsData = await eventsRes.json();
                    const container = document.getElementById('upcomingEvents');
                    if (container) {
                        const events = eventsData.events || [];
                        if (events.length > 0) {
                            container.innerHTML = events.map(evt => `
                                <div style="padding:1rem;background:var(--bg-elevated);border-radius:8px;margin-bottom:0.75rem;border:1px solid var(--border);">
                                    <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                                        <div style="font-weight:600;">${evt.title || 'Untitled'}</div>
                                        <div style="padding:0.25rem 0.5rem;background:var(--bg-card);border-radius:4px;font-size:0.75rem;">
                                            ${evt.start ? new Date(evt.start).toLocaleDateString() : ''}
                                        </div>
                                    </div>
                                    <div style="margin-top:0.5rem;font-size:0.85rem;color:var(--text-secondary);">${evt.description || ''}</div>
                                    <div style="margin-top:0.5rem;display:flex;gap:0.5rem;">
                                        <span style="font-size:0.75rem;color:var(--text-muted);padding:2px 6px;background:rgba(255,255,255,0.05);border-radius:4px;">${evt.type || 'event'}</span>
                                    </div>
                                </div>
                            `).join('');
                        } else {
                            container.innerHTML = '<div class="empty-state">No upcoming events found</div>';
                        }
                    }
                }

                // Sync Status
                if (statusRes.ok) {
                    const statusData = await statusRes.json();
                    const container = document.getElementById('syncStatus');
                    if (container) {
                        if (statusData.connected) {
                            container.innerHTML = `
                                <div style="color:var(--success);font-weight:600;display:flex;align-items:center;gap:0.5rem;">
                                    <span>●</span> Connected to Google Calendar
                                </div>
                                <div style="font-size:0.8rem;color:var(--text-secondary);margin-top:0.5rem;">Last sync: Just now</div>
                            `;
                        } else {
                            container.innerHTML = `
                                <div style="color:var(--text-secondary);margin-bottom:1rem;">Not connected</div>
                                <button onclick="window.open('${statusData.oauth_url}', '_blank')" style="width:100%;padding:0.5rem;background:var(--bg-elevated);border:1px solid var(--border);border-radius:6px;cursor:pointer;">Connect Google Calendar</button>
                            `;
                        }
                    }
                }
                
            } catch (e) {
                console.error('[Calendar] Critical failure:', e);
                showToast('Error loading calendar', 'error');
            }
        }

        async function syncCalendar() {
            showToast('Syncing calendar...', 'info');
            try {
                const res = await fetch(`/api/calendar/sync/${userId}`, { method: 'POST' });
                const data = await res.json();
                if (data.synced !== undefined) {
                    showToast(`Synced ${data.synced} events`, 'success');
                    loadCalendar();
                } else {
                    showToast('Sync failed. Check connection.', 'error');
                }
            } catch (e) {
                showToast('Sync error', 'error');
            }
        }
        
        // Modal Handling
        function showAddEventModal() {
            document.getElementById('addEventModal').style.display = 'flex';
        }
        
        async function submitEvent() {
            const title = document.getElementById('eventTitle').value;
            const type = document.getElementById('eventType').value;
            const start = document.getElementById('eventStart').value;
            const end = document.getElementById('eventEnd').value;
            const desc = document.getElementById('eventDescription').value;
            
            if (!title || !start) {
                showToast('Title and Start Date required', 'warning');
                return;
            }
            
            try {
                const res = await fetch('/api/calendar/events', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        user_id: userId,
                        title: title,
                        event_type: type,
                        start_time: new Date(start).toISOString(),
                        end_time: end ? new Date(end).toISOString() : null,
                        description: desc
                    })
                });
                
                if (res.ok) {
                    showToast('Event created successfully', 'success');
                    document.getElementById('addEventModal').style.display = 'none';
                    loadCalendar();
                } else {
                    showToast('Failed to create event', 'error');
                }
            } catch (e) {
                showToast('Error creating event', 'error');
            }
        }

        // PWA Installation
        let deferredPrompt;
        window.addEventListener('beforeinstallprompt', (e) => {
            e.preventDefault();
            deferredPrompt = e;
            document.getElementById('pwaInstallPrompt').style.display = 'block';
        });

        async function installPWA() {
            if (!deferredPrompt) return;
            deferredPrompt.prompt();
            const { outcome } = await deferredPrompt.userChoice;
            if (outcome === 'accepted') {
                showToast('Installing app...', 'success');
            }
            deferredPrompt = null;
            document.getElementById('pwaInstallPrompt').style.display = 'none';
        }

        function dismissInstall() {
            document.getElementById('pwaInstallPrompt').style.display = 'none';
        }

        // Resume Analysis
        let currentResumeId = null;

        async function uploadResume(event) {
            const file = event.target.files[0];
            if (!file) return;
            
            showToast('Uploading and analyzing...', 'info');
            const formData = new FormData();
            formData.append('file', file);
            formData.append('user_id', userId);
            
            try {
                const response = await fetch('/api/upload', { method: 'POST', body: formData });
                const data = await response.json();
                
                if (data.is_resume && data.resume_id) {
                    currentResumeId = data.resume_id;
                    await loadResumeData(currentResumeId);
                    showToast('Resume analyzed successfully', 'success');
                } else {
                    showToast('Uploaded, but not recognized as a resume', 'warning');
                }
            } catch (error) {
                showToast('Upload failed', 'error');
                console.error(error);
            }
        }
        
        async function loadResumeData(resumeId) {
            try {
                const response = await fetch(`/api/resume/${resumeId}`);
                const resume = await response.json();
                
                const container = document.getElementById('resumeAnalysisResult');
                container.style.textAlign = 'left';
                container.style.alignItems = 'flex-start';
                container.style.display = 'block';
                
                container.innerHTML = `
                    <div style="display:flex;justify-content:space-between;margin-bottom:1rem;">
                        <div>
                            <div style="font-size:1.25rem;font-weight:700;">${resume.name || 'Candidate'}</div>
                            <div style="color:var(--text-secondary);">${resume.email || ''} • ${resume.location || ''}</div>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-size:1.5rem;font-weight:700;color:var(--accent);">${Math.round(resume.confidence_score * 100)}%</div>
                            <div style="font-size:0.75rem;color:var(--text-muted);">Confidence</div>
                        </div>
                    </div>
                    
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;">
                        <div>
                            <div style="font-weight:600;margin-bottom:0.5rem;border-bottom:1px solid var(--border);">Experience</div>
                            <div style="font-size:0.9rem;">${resume.experience_count} roles identified</div>
                            <div style="font-size:0.9rem;">${resume.years_of_experience} years total</div>
                            <div style="font-size:0.9rem;">Level: ${resume.seniority_level}</div>
                        </div>
                        <div>
                            <div style="font-weight:600;margin-bottom:0.5rem;border-bottom:1px solid var(--border);">Skills</div>
                            <div style="display:flex;flex-wrap:wrap;gap:0.5rem;">
                                ${resume.skills.slice(0, 10).map(s => `<span style="padding:2px 8px;background:var(--bg-elevated);border-radius:4px;font-size:0.8rem;">${s}</span>`).join('')}
                                ${resume.skills.length > 10 ? `<span style="font-size:0.8rem;color:var(--text-muted);">+${resume.skills.length - 10} more</span>` : ''}
                            </div>
                        </div>
                    </div>
                `;
                
                updateSkillGaps();
                document.getElementById('roadmapBtn').style.display = 'block';
            } catch (e) {
                console.error(e);
            }
        }
        
        async function updateSkillGaps() {
            if (!currentResumeId) return;
            
            const role = document.getElementById('targetRoleSelect').value;
            const container = document.getElementById('skillGapsList');
            container.innerHTML = '<div class="empty-state">Analyzing gaps...</div>';
            
            try {
                const response = await fetch(`/api/resume/${currentResumeId}/skill-gaps?target_role=${encodeURIComponent(role)}`);
                const data = await response.json();
                
                container.innerHTML = `
                    <div style="margin-bottom:1rem;">
                        <div style="display:flex;justify-content:space-between;margin-bottom:0.5rem;">
                            <span style="font-weight:600;">Match Score</span>
                            <span style="font-weight:700;color:${data.match_percentage > 70 ? 'var(--success)' : 'var(--warning)'};">${data.match_percentage}%</span>
                        </div>
                        <div style="height:8px;background:var(--bg-elevated);border-radius:4px;overflow:hidden;">
                            <div style="height:100%;width:${data.match_percentage}%;background:${data.match_percentage > 70 ? 'var(--success)' : 'var(--warning)'};"></div>
                        </div>
                    </div>
                    
                    <div style="display:grid;gap:1rem;">
                        <div>
                            <div style="font-size:0.85rem;color:var(--success);margin-bottom:0.5rem;">Matching Skills</div>
                            <div style="display:flex;flex-wrap:wrap;gap:0.5rem;">
                                ${data.matching_skills.map(s => `<span style="padding:2px 8px;background:rgba(34,197,94,0.1);color:var(--success);border-radius:4px;font-size:0.8rem;">${s}</span>`).join('')}
                            </div>
                        </div>
                        <div>
                            <div style="font-size:0.85rem;color:var(--danger);margin-bottom:0.5rem;">Missing Skills</div>
                            <div style="display:flex;flex-wrap:wrap;gap:0.5rem;">
                                ${data.missing_skills.map(s => `<span style="padding:2px 8px;background:rgba(239,68,68,0.1);color:var(--danger);border-radius:4px;font-size:0.8rem;">${s}</span>`).join('')}
                            </div>
                        </div>
                    </div>
                    
                    <div style="margin-top:1rem;padding:1rem;background:var(--bg-elevated);border-radius:8px;">
                        <div style="font-weight:600;margin-bottom:0.5rem;">Recommendations</div>
                        <ul style="padding-left:1.5rem;margin:0;font-size:0.9rem;color:var(--text-secondary);">
                            ${data.recommendations.map(r => `<li>${r}</li>`).join('')}
                        </ul>
                    </div>
                `;
            } catch (e) {
                container.innerHTML = '<div class="empty-state">Failed to analyze gaps</div>';
            }
        }

        async function generateRoadmap() {
            if (!currentResumeId) return;
            const role = document.getElementById('targetRoleSelect').value;
            const roadmapContainer = document.getElementById('careerRoadmap');
            
            showToast('Generating your career roadmap...', 'info');
            roadmapContainer.innerHTML = '<div class="empty-state" style="grid-column: span 3;">Synthesizing your path...</div>';
            
            try {
                const gapRes = await fetch(`/api/resume/${currentResumeId}/skill-gaps?target_role=${encodeURIComponent(role)}`);
                const gapData = await gapRes.json();
                
                const response = await fetch(`/api/roadmap/generate?user_id=${userId}&target_role=${encodeURIComponent(role)}&gap_skills=${gapData.missing_skills.join(',')}`, { method: 'POST' });
                const data = await response.json();
                
                roadmapContainer.innerHTML = data.milestones.map((m, idx) => `
                    <div style="padding:1.5rem;background:var(--bg-elevated);border:1px solid var(--border);border-radius:12px;display:flex;flex-direction:column;gap:1rem;position:relative;overflow:hidden;border-left:4px solid var(--accent);">
                        <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                            <div>
                                <div style="font-size:0.75rem;color:var(--accent);font-weight:700;text-transform:uppercase;">Phase ${idx + 1}</div>
                                <div style="font-weight:700;font-size:1.1rem;">${m.phase}</div>
                            </div>
                            <div style="font-size:0.75rem;color:var(--text-muted);">${m.weeks}</div>
                        </div>
                        <div style="display:flex;flex-direction:column;gap:0.75rem;">
                            <div style="font-size:0.85rem;color:var(--text-secondary);">
                                <div style="font-weight:600;color:var(--text-primary);margin-bottom:0.25rem;">Key Tasks:</div>
                                <ul style="padding-left:1.25rem;margin:0;">
                                    ${m.tasks.map(t => `<li>${t}</li>`).join('')}
                                </ul>
                            </div>
                            <div style="font-size:0.85rem;color:var(--text-secondary);">
                                <div style="font-weight:600;color:var(--text-primary);margin-bottom:0.25rem;">Resources:</div>
                                <div style="display:flex;flex-wrap:wrap;gap:0.4rem;">
                                    ${m.resources.map(r => `<span style="padding:2px 6px;background:rgba(255,255,255,0.05);border-radius:4px;font-size:0.7rem;">${r}</span>`).join('')}
                                </div>
                            </div>
                        </div>
                    </div>
                `).join('');
                
                showToast('Roadmap generated!', 'success');
            } catch (e) {
                showToast('Failed to generate roadmap', 'error');
                roadmapContainer.innerHTML = '<div class="empty-state" style="grid-column: span 3;">Roadmap generation failed.</div>';
            }
        }
        
        // ============= PHASE 3 FEATURES =============

        let currentMentorChat = null;

        async function loadMentors() {
            const list = document.getElementById('mentorList');
            if (!list) return;
            list.innerHTML = '<div class="empty-state">Finding your matches...</div>';
            
            try {
                const res = await fetch(`/api/mentor/matches/${userId}?industry=Technology`);
                const data = await res.json();
                
                if (data.matches && data.matches.length > 0) {
                    list.innerHTML = data.matches.map(m => `
                        <div style="background:var(--bg-elevated);border:1px solid var(--border);border-radius:12px;padding:1.25rem;display:flex;flex-direction:column;gap:0.75rem;transition:all 0.3s ease;">
                            <div style="display:flex;justify-content:space-between;align-items:center;">
                                <div style="font-weight:700;font-size:1rem;">${m.name}</div>
                                <div style="font-size:0.7rem;color:var(--success);font-weight:700;">${Math.round(m.match_score * 100)}% Match</div>
                            </div>
                            <div style="font-size:0.8rem;color:var(--text-secondary);line-height:1.4;">${m.bio}</div>
                            <div style="display:flex;flex-wrap:wrap;gap:0.4rem;">
                                ${m.expertise.map(e => `<span style="padding:2px 6px;background:rgba(255,255,255,0.05);border-radius:4px;font-size:0.65rem;color:var(--text-muted);">${e}</span>`).join('')}
                            </div>
                            <button class="send-btn" style="margin-top:0.5rem;padding:0.4rem;font-size:0.8rem;" onclick="requestMentor('${m.id}')">Connect</button>
                        </div>
                    `).join('');
                } else {
                    list.innerHTML = '<div class="empty-state">No matches found.</div>';
                }
                loadConnectedMentors();
            } catch (e) {
                showToast('Failed to load mentors', 'error');
            }
        }

        async function loadConnectedMentors() {
            const container = document.getElementById('connectedMentors');
            if (!container) return;
            
            try {
                const res = await fetch(`/api/mentor/connected/${userId}`);
                const data = await res.json();
                
                if (data.mentors && data.mentors.length > 0) {
                    container.innerHTML = data.mentors.map(m => `
                        <div onclick="openMentorChat('${m.mentor.id}', '${m.mentor.name}')" style="padding:0.85rem;background:var(--bg-elevated);border:1px solid var(--border);border-radius:10px;cursor:pointer;display:flex;align-items:center;gap:0.75rem;transition:transform 0.2s;">
                            <div style="width:36px;height:36px;background:var(--accent);border-radius:50%;display:flex;align-items:center;justify-content:center;color:var(--bg-primary);font-weight:bold;font-size:0.9rem;">${m.mentor.name[0]}</div>
                            <div style="flex:1;">
                                <div style="font-weight:600;font-size:0.9rem;">${m.mentor.name}</div>
                                <div style="font-size:0.7rem;color:var(--text-muted);">${m.messages[m.messages.length-1]?.text.substring(0, 30)}...</div>
                            </div>
                        </div>
                    `).join('');
                    
                    // Load mentor filter options
                    updateMentorFilterOptions(data.mentors);
                    
                    // Load video recommendations
                    loadMentorVideos();
                } else {
                    container.innerHTML = '<div class="empty-state">No mentors connected</div>';
                }
            } catch (e) {
                console.error('Failed to load connected mentors', e);
            }
        }

        // YouTube Video Recommendations Functions
        async function loadMentorVideos() {
            const mentorFilter = document.getElementById('videoMentorFilter')?.value;
            const container = document.getElementById('videoRecommendationsContainer');
            
            if (!container) return;
            
            try {
                let videos = [];
                
                if (mentorFilter) {
                    // Load videos for specific mentor
                    const res = await fetch(`/api/mentor/videos/specialty/${mentorFilter}`);
                    const data = await res.json();
                    videos = data.videos || [];
                } else {
                    // Load personalized recommendations
                    const res = await fetch(`/api/mentor/videos/personalized/${userId}?limit=12`);
                    const data = await res.json();
                    videos = data.videos || [];
                }
                
                if (videos.length > 0) {
                    container.innerHTML = videos.map(v => `
                        <div onclick="showVideoDetails('${v.id}', '${escapeHtml(v.title)}', '${escapeHtml(v.channel)}', ${v.duration_minutes}, '${escapeHtml(v.description)}', '${v.url}')" style="background:var(--bg-elevated);border:1px solid var(--border);border-radius:12px;padding:1rem;cursor:pointer;display:flex;flex-direction:column;gap:0.75rem;transition:all 0.3s ease;position:relative;overflow:hidden;">
                            <div style="position:absolute;top:0;left:0;width:100%;height:100%;background:linear-gradient(135deg, rgba(255,0,0,0.1) 0%, transparent 100%);opacity:0;transition:opacity 0.3s;" class="video-hover-overlay"></div>
                            <div style="display:flex;align-items:center;gap:0.5rem;position:relative;z-index:1;">
                                <i class="fas fa-play-circle" style="color:#ff0000;font-size:1.2rem;"></i>
                                <span style="font-weight:600;font-size:0.85rem;color:var(--text-muted);">${v.duration_minutes}m</span>
                            </div>
                            <div style="font-weight:600;font-size:0.9rem;line-height:1.3;position:relative;z-index:1;">${v.title}</div>
                            <div style="font-size:0.75rem;color:var(--text-muted);position:relative;z-index:1;">${v.channel}</div>
                            <div style="font-size:0.7rem;color:rgba(255,255,255,0.5);position:relative;z-index:1;">👁 ${(v.view_count/1000).toFixed(0)}K views</div>
                            <div style="display:flex;gap:0.5rem;margin-top:0.5rem;position:relative;z-index:1;">
                                <span style="padding:2px 8px;background:rgba(255,0,0,0.2);border:1px solid rgba(255,0,0,0.4);border-radius:4px;font-size:0.65rem;color:#ff6b6b;">${v.category.replace(/_/g, ' ')}</span>
                            </div>
                        </div>
                    `).join('');
                } else {
                    container.innerHTML = '<div class="empty-state">No videos available</div>';
                }
            } catch (e) {
                console.error('Failed to load videos', e);
                container.innerHTML = '<div class="empty-state">Failed to load videos</div>';
            }
        }

        function updateMentorFilterOptions(mentors) {
            const select = document.getElementById('videoMentorFilter');
            if (!select) return;
            
            const currentValue = select.value;
            const options = mentors.map(m => `<option value="${m.mentor.id}">${m.mentor.name}</option>`).join('');
            select.innerHTML = '<option value="">All Mentors</option>' + options;
            select.value = currentValue;
        }

        function filterVideosByCategory() {
            // This function can be expanded to filter videos by category
            loadMentorVideos();
        }

        function showVideoDetails(videoId, title, channel, duration, description, url) {
            const modal = document.getElementById('videoModal');
            if (!modal) return;
            
            const content = document.getElementById('videoModalContent');
            content.innerHTML = `
                <div style="display:grid;gap:1rem;">
                    <div>
                        <div style="font-weight:600;font-size:1.1rem;margin-bottom:0.5rem;">${title}</div>
                        <div style="font-size:0.9rem;color:var(--text-secondary);">Channel: ${channel}</div>
                    </div>
                    <div style="display:grid;grid-template-columns:repeat(3, 1fr);gap:1rem;padding:1rem;background:rgba(255,255,255,0.05);border-radius:8px;">
                        <div style="text-align:center;">
                            <div style="font-size:1.5rem;font-weight:700;">${duration}</div>
                            <div style="font-size:0.75rem;color:var(--text-muted);">Minutes</div>
                        </div>
                        <div style="text-align:center;">
                            <div style="font-size:1.5rem;"></div>
                            <div style="font-size:0.75rem;color:var(--text-muted);">Video</div>
                        </div>
                        <div style="text-align:center;">
                            <div style="font-size:1.5rem;"></div>
                            <div style="font-size:0.75rem;color:var(--text-muted);">Rate it</div>
                        </div>
                    </div>
                    <div>
                        <div style="font-weight:600;font-size:0.9rem;margin-bottom:0.5rem;">Description</div>
                        <div style="font-size:0.85rem;color:var(--text-secondary);line-height:1.5;">${description}</div>
                    </div>
                    <div>
                        <button onclick="markVideoWatched('${videoId}')" class="btn btn-secondary" style="width:100%;">
                            <i class="fas fa-check" style="margin-right: 0.5rem;"></i>Mark as Watched
                        </button>
                    </div>
                </div>
            `;
            
            document.getElementById('videoModalTitle').textContent = title;
            modal.style.display = 'flex';
            
            // Store current video ID for modal actions
            window.currentVideoId = videoId;
            window.currentVideoUrl = url;
        }

        function closeVideoModal() {
            const modal = document.getElementById('videoModal');
            if (modal) modal.style.display = 'none';
        }

        async function saveVideoForLater() {
            if (!window.currentVideoId) return;
            
            try {
                const res = await fetch(`/api/mentor/videos/save?user_id=${userId}&video_id=${window.currentVideoId}`, { method: 'POST' });
                const data = await res.json();
                if (data.success) {
                    showToast('Video saved for later!', 'success');
                }
            } catch (e) {
                showToast('Failed to save video', 'error');
            }
        }

        async function markVideoWatched(videoId) {
            try {
                const res = await fetch(`/api/mentor/videos/watched?user_id=${userId}&video_id=${videoId}`, { method: 'POST' });
                const data = await res.json();
                if (data.success) {
                    showToast('Great! Video marked as watched', 'success');
                    setTimeout(() => closeVideoModal(), 500);
                }
            } catch (e) {
                showToast('Failed to mark video', 'error');
            }
        }

        function openVideoOnYouTube() {
            if (window.currentVideoUrl) {
                window.open(window.currentVideoUrl, '_blank');
                showToast('Opening YouTube...', 'info');
            }
        }

        function escapeHtml(text) {
            const map = {
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                '"': '&quot;',
                "'": '&#039;'
            };
            return text.replace(/[&<>"']/g, m => map[m]);
        }

        function youtubeSearch() {
            const qEl = document.getElementById('youtubeQuery');
            const q = qEl ? qEl.value.trim() : '';
            if (!q) {
                showToast('Enter a search query for YouTube', 'warning');
                if (qEl) qEl.focus();
                return;
            }
            const url = 'https://www.youtube.com/results?search_query=' + encodeURIComponent(q);
            showToast('Opening YouTube search...', 'info');
            window.open(url, '_blank');
        }

        async function requestMentor(mentorId) {
            try {
                const res = await fetch(`/api/mentor/request?user_id=${userId}&mentor_id=${mentorId}`, { method: 'POST' });
                const data = await res.json();
                if (data.success) {
                    showToast('Mentor connected!', 'success');
                    loadMentors();
                }
            } catch (e) {
                showToast('Failed to send request', 'error');
            }
        }

        async function openMentorChat(mentorId, mentorName) {
            currentMentorChat = mentorId;
            document.getElementById('activeMentorName').textContent = mentorName;
            document.getElementById('activeMentorAvatar').textContent = mentorName[0];
            document.getElementById('mentorChatModal').style.display = 'flex';
            renderMentorChat();
        }

        async function renderMentorChat() {
            if (!currentMentorChat) return;
            const container = document.getElementById('mentorChatHistory');
            
            try {
                const res = await fetch(`/api/mentor/connected/${userId}`);
                const data = await res.json();
                const current = data.mentors.find(m => m.mentor.id === currentMentorChat);
                
                if (current) {
                    container.innerHTML = current.messages.map(msg => `
                        <div style="display:flex;justify-content:${msg.sender === 'user' ? 'flex-end' : 'flex-start'};">
                            <div style="max-width:80%;padding:0.75rem 1rem;border-radius:12px;font-size:0.9rem;line-height:1.4;background:${msg.sender === 'user' ? 'var(--accent)' : 'var(--bg-elevated)'};color:${msg.sender === 'user' ? 'var(--bg-primary)' : 'var(--text-primary)'};border:${msg.sender === 'user' ? 'none' : '1px solid var(--border)'};">
                                ${msg.text}
                                <div style="font-size:0.65rem;margin-top:0.4rem;opacity:0.7;text-align:right;">${new Date(msg.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</div>
                            </div>
                        </div>
                    `).join('');
                    container.scrollTop = container.scrollHeight;
                }
            } catch (e) {
                console.error('Chat render failed', e);
            }
        }

        async function sendMentorMessageToServer() {
            const input = document.getElementById('mentorChatMessage');
            const text = input.value.trim();
            if (!text || !currentMentorChat) return;

            try {
                const res = await fetch(`/api/mentor/message?user_id=${userId}&mentor_id=${currentMentorChat}&text=${encodeURIComponent(text)}`, { method: 'POST' });
                const data = await res.json();
                if (data.success) {
                    input.value = '';
                    renderMentorChat();
                }
            } catch (e) {
                showToast('Failed to send message', 'error');
            }
        }

        function closeMentorChat() {
            document.getElementById('mentorChatModal').style.display = 'none';
            currentMentorChat = null;
        }

        async function runSimulation() {
            const desc = document.getElementById('simDecision').value;
            const salary = document.getElementById('simSalary').value;
            const uncertainty = document.getElementById('simUncertainty').value;
            
            if (!desc) return showToast('Enter a decision description', 'warning');
            
            showToast('Running Monte Carlo simulation...', 'info');
            try {
                const res = await fetch(`/api/simulate/run?decision_desc=${encodeURIComponent(desc)}&base_salary=${salary}&uncertainty=${uncertainty}`, { method: 'POST' });
                const data = await res.json();
                
                // Display Stats
                document.getElementById('simStats').innerHTML = `
                    <div style="display:grid;gap:1rem;">
                        <div style="padding:0.75rem;background:rgba(255,255,255,0.02);border-radius:8px;border:1px solid var(--border);">
                            <div style="font-size:0.75rem;color:var(--text-muted);">Expected 5-Year Value</div>
                            <div style="font-size:1.4rem;font-weight:700;color:var(--accent);">$${data.stats.expected_value.toLocaleString()}</div>
                        </div>
                        <div style="padding:0.75rem;background:rgba(255,255,255,0.02);border-radius:8px;border:1px solid var(--border);">
                            <div style="font-size:0.75rem;color:var(--text-muted);">Upside Potential</div>
                            <div style="font-size:1.4rem;font-weight:700;color:var(--success);">$${data.stats.max_upside.toLocaleString()}</div>
                        </div>
                    </div>
                `;
                
                // Display Results
                document.getElementById('simResults').innerHTML = data.results.map(r => `
                    <div style="padding:1.5rem;background:var(--bg-elevated);border:1px solid var(--border);border-radius:12px;display:flex;flex-direction:column;gap:0.75rem;">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <div style="font-weight:700;">${r.scenario_name}</div>
                            <div style="font-size:0.7rem;padding:2px 8px;background:rgba(255,255,255,0.1);border-radius:20px;">${Math.round(r.probability*100)}% Prob.</div>
                        </div>
                        <div style="font-size:0.8rem;color:var(--text-secondary);line-height:1.4;">${r.description}</div>
                        <div style="margin-top:auto;padding-top:1rem;border-top:1px solid rgba(255,255,255,0.05);">
                            <div style="font-size:0.7rem;color:var(--text-muted);">Potential Regret</div>
                            <div style="width:100%;height:4px;background:rgba(255,255,255,0.05);border-radius:2px;margin-top:4px;">
                                <div style="width:${r.regret_potential}%;height:100%;background:${r.regret_potential > 50 ? 'var(--danger)' : 'var(--success)'};border-radius:2px;"></div>
                            </div>
                        </div>
                    </div>
                `).join('');
                
            } catch (e) {
                showToast('Simulation failed', 'error');
            }
        }

        let isInterviewing = false;

        async function uploadKnowledge(event) {
            const file = event.target.files[0];
            if (!file) return;
            
            showToast('Processing document for knowledge base...', 'info');
            try {
                // Mocking the text extraction and upload
                const res = await fetch(`/api/knowledge/add?user_id=${userId}&filename=${encodeURIComponent(file.name)}&content=${encodeURIComponent("Custom knowledge from " + file.name)}`, { method: 'POST' });
                const data = await res.json();
                showToast(`Document "${file.name}" added to KB`, 'success');
                loadKnowledge();
            } catch (e) {
                showToast('Failed to upload document', 'error');
            }
        }

        async function loadKnowledge() {
            const container = document.getElementById('knowledgeList');
            if (!container) return;
            
            try {
                const res = await fetch(`/api/knowledge/list/${userId}`);
                const data = await res.json();
                
                if (data.documents && data.documents.length > 0) {
                    container.innerHTML = data.documents.map(doc => `
                        <div style="padding:1rem;background:var(--bg-elevated);border:1px solid var(--border);border-radius:10px;display:flex;flex-direction:column;gap:0.5rem;">
                            <div style="display:flex;justify-content:space-between;align-items:center;">
                                <div style="font-weight:700;font-size:0.85rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${doc.filename}</div>
                                <button onclick="deleteKnowledge('${doc.id}')" style="background:none;border:none;color:var(--danger);cursor:pointer;padding:2px;">&times;</button>
                            </div>
                            <div style="font-size:0.75rem;color:var(--text-muted);">${doc.chars} chars • ${new Date(doc.added_at).toLocaleDateString()}</div>
                            <div style="font-size:0.7rem;color:var(--text-secondary);line-height:1.3;margin-top:0.25rem;">${doc.summary}</div>
                        </div>
                    `).join('');
                } else {
                    container.innerHTML = '<div class="empty-state">No knowledge added</div>';
                }
            } catch (e) {
                console.error('KB load failed', e);
            }
        }

        async function deleteKnowledge(docId) {
            try {
                await fetch(`/api/knowledge/${userId}/${docId}`, { method: 'DELETE' });
                showToast('Document removed', 'info');
                loadKnowledge();
            } catch (e) {
                showToast('Delete failed', 'error');
            }
        }
        let interviewHistory = [];
        let interviewMediaRecorder = null;
        let interviewAudioChunks = [];

        function speakText(text, callback) {
            if (!('speechSynthesis' in window)) {
                if (callback) callback();
                return;
            }
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            utterance.volume = 1.0;
            utterance.onend = () => {
                if (callback) callback();
            };
            window.speechSynthesis.speak(utterance);
        }

        async function startInterviewRecording(callback) {
            try {
                const status = document.getElementById('interviewStatus');
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                interviewMediaRecorder = new MediaRecorder(stream);
                interviewAudioChunks = [];
                
                interviewMediaRecorder.ondataavailable = (event) => {
                    interviewAudioChunks.push(event.data);
                };
                
                interviewMediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(interviewAudioChunks, { type: 'audio/webm' });
                    stream.getTracks().forEach(track => track.stop());
                    
                    if (!isInterviewing) return;
                    
                    status.textContent = "Processing answer...";
                    showToast('Transcribing...', 'info');
                    
                    try {
                        const formData = new FormData();
                        formData.append('audio', audioBlob, 'interview.webm');
                        
                        const response = await fetch('/api/speech-to-text', {
                            method: 'POST',
                            body: formData
                        });
                        
                        if (response.ok) {
                            const data = await response.json();
                            if (data.text && data.text.length > 3) {
                                callback(data.text);
                            } else {
                                showToast("I didn't catch that. Could you repeat?", 'warning');
                                if (isInterviewing) setTimeout(() => startInterviewRecording(callback), 1000);
                            }
                        } else {
                            showToast('STT connection failed.', 'error');
                        }
                    } catch (e) {
                        console.error('Transcription error:', e);
                    }
                };
                
                interviewMediaRecorder.start();
                status.textContent = "Your turn (Speak now)...";
                // Max 15 seconds per answer for mock
                setTimeout(() => {
                    if (interviewMediaRecorder && interviewMediaRecorder.state === 'recording') {
                        interviewMediaRecorder.stop();
                    }
                }, 15000);
            } catch (e) {
                console.error('Interview recording error:', e);
                showToast('Microphone required', 'error');
                isInterviewing = false;
            }
        }

        async function startInterview() {
            const btn = document.getElementById('interviewBtn');
            const status = document.getElementById('interviewStatus');
            const role = document.getElementById('interviewRole').value;
            const difficulty = document.getElementById('interviewDifficulty').value;
            
            if (!isInterviewing) {
                isInterviewing = true;
                interviewHistory = [];
                btn.textContent = 'End Interview';
                btn.style.background = 'var(--danger)';
                status.textContent = 'Contacting interviewer...';
                
                showToast(`Starting ${role} interview`, 'info');
                
                try {
                    const response = await fetch('/api/interview/next', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            user_id: userId,
                            role: role,
                            difficulty: difficulty,
                            user_response: "",
                            history: []
                        })
                    });
                    const data = await response.json();
                    if (data.question) {
                        handleInterviewQuestion(data.question);
                    }
                } catch (e) {
                    showToast('Failed to start interview', 'error');
                    stopInterviewUI();
                }
            } else {
                stopInterviewUI();
            }
        }

        function stopInterviewUI() {
            isInterviewing = false;
            if (interviewMediaRecorder && interviewMediaRecorder.state === 'recording') {
                interviewMediaRecorder.stop();
            }
            const btn = document.getElementById('interviewBtn');
            const status = document.getElementById('interviewStatus');
            btn.textContent = 'Start Interview Session';
            btn.style.background = 'var(--accent)';
            status.textContent = 'Session ended.';
            window.speechSynthesis.cancel();
        }

        function handleInterviewQuestion(question) {
            if (!isInterviewing) return;
            const status = document.getElementById('interviewStatus');
            status.textContent = "Interviewer speaking...";
            
            speakText(question, () => {
                if (isInterviewing) {
                    startInterviewRecording((userText) => {
                        interviewHistory.push(question);
                        interviewHistory.push(userText);
                        getNextInterviewQuestion(userText);
                    });
                }
            });
        }

        async function getNextInterviewQuestion(userText) {
            if (!isInterviewing) return;
            const role = document.getElementById('interviewRole').value;
            const difficulty = document.getElementById('interviewDifficulty').value;
            const status = document.getElementById('interviewStatus');
            status.textContent = "Interviewer thinking...";
            
            try {
                const response = await fetch('/api/interview/next', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        user_id: userId,
                        role: role,
                        difficulty: difficulty,
                        user_response: userText,
                        history: interviewHistory.slice(-4)
                    })
                });
                const data = await response.json();
                if (data.question) {
                    handleInterviewQuestion(data.question);
                }
            } catch (e) {
                showToast('Interviewer connection lost', 'error');
                stopInterviewUI();
            }
        }

        async function createShareLink(decisionId) {
            try {
                const res = await fetch(`/api/share/create?decision_id=${decisionId}&user_id=${userId}`, { method: 'POST' });
                const data = await res.json();
                if (data.url) {
                    const fullUrl = window.location.origin + data.url;
                    navigator.clipboard.writeText(fullUrl);
                    showToast('Share link copied to clipboard!', 'success');
                }
            } catch (e) {
                showToast('Failed to create share link', 'error');
            }
        }

        async function switchLLM(provider) {
            try {
                const res = await fetch(`/api/llm/switch?provider=${provider}`, { method: 'POST' });
                const data = await res.json();
                if (data.success) {
                    showToast(`Switched to ${provider} engine`, 'success');
                }
            } catch (e) {
                showToast('Failed to switch LLM', 'error');
            }
        }

        async function toggleFineTuning(active) {
            try {
                showToast(active ? 'Activating specialized model...' : 'Reverting to base model...');
                const res = await fetch(`/api/finetune/toggle?active=${active}`, { method: 'POST' });
                const data = await res.json();
                showToast(`Active model: ${data.model}`, 'success');
            } catch (e) {
                showToast('Fine-tuning toggle failed', 'error');
            }
        }

        async function setupWebhook() {
            const url = document.getElementById('zapierUrl').value;
            if (!url) return showToast('Enter a valid URL', 'warning');
            
            try {
                const res = await fetch(`/api/integrations/webhook/setup?user_id=${userId}&url=${encodeURIComponent(url)}`, { method: 'POST' });
                const data = await res.json();
                if (data.success) {
                    showToast('Zapier webhook configured!', 'success');
                }
            } catch (e) {
                showToast('Webhook setup failed', 'error');
            }
        }

        async function generateApiKey() {
            const display = document.getElementById('apiKeyDisplay');
            display.style.display = 'block';
            display.textContent = 'Generating...';
            
            // Mocking the generation since it's usually handled by the enterprise service
            setTimeout(() => {
                const key = 'sk_career_' + Math.random().toString(36).substr(2, 16);
                display.textContent = key;
                showToast('API token generated!', 'success');
            }, 800);
        }

        function showShortcutHelp() {
            document.getElementById('shortcutsModal').style.display = 'flex';
        }

        // Global Keyboard Listeners
        window.addEventListener('keydown', (e) => {
            // Alt + N: New Conversation
            if (e.altKey && e.key === 'n') {
                e.preventDefault();
                clearChat();
                showTab('chat');
            }
            // Alt + T: Templates
            if (e.altKey && e.key === 't') {
                e.preventDefault();
                showTab('templates');
            }
            // Alt + J: Journal
            if (e.altKey && e.key === 'j') {
                e.preventDefault();
                showTab('journal');
            }
            // Alt + S: Settings
            if (e.altKey && e.key === 's') {
                e.preventDefault();
                toggleSettings();
            }
            // /: Focus input
            if (e.key === '/' && document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'TEXTAREA') {
                e.preventDefault();
                document.getElementById('chatInput').focus();
            }
            // Escape: Close modals
            if (e.key === 'Escape') {
                document.querySelectorAll('.modal').forEach(m => m.style.display = 'none');
                document.getElementById('settingsModal').style.display = 'none';
            }
        });

        // Initialize Phase 3
        async function initPhase3() {
            try {
                const res = await fetch('/api/llm/config');
                const data = await res.json();
                const select = document.getElementById('llmProvider');
                if (select) select.value = data.active_provider;
                
                const fRes = await fetch('/api/finetune/status');
                const fData = await fRes.json();
                const check = document.getElementById('fineTuneToggle');
                if (check) check.checked = fData.is_finetuned;
            } catch (e) {}
        }
        
        initPhase3();

    </script>
    </main>
</body>

</html>'''

@app.post("/api/interview/next")
async def interview_next(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "default")
    role = data.get("role", "software_engineer")
    difficulty = data.get("difficulty", "junior")
    user_response = data.get("user_response", "")
    history = data.get("history", [])

    role_guidance = {
        "software_engineer": "Focus on algorithms, system design, and coding best practices.",
        "product_manager": "Focus on product-market fit, prioritization, and roadmapping.",
        "data_scientist": "Focus on statistical modeling, data analysis, and machine learning.",
        "designer": "Focus on user experience, visual design, and prototyping."
    }.get(role, "")

    system_prompt = f"""You are a professional technical interviewer for the role of {role.replace('_', ' ')} at a {difficulty} level.
Your goal is to conduct a realistic, high-quality mock interview. {role_guidance}
Rules:
1. Ask exactly ONE question at a time.
2. Keep your responses brief and conversational (max 2-3 sentences).
3. Be professional but encouraging.
4. If this is the start (no user response yet), introduce yourself and ask a relevant opening question.
5. If the user has responded, acknowledge their response briefly and ask the next logical interview question.
6. Cover both technical skills and behavioral/situational aspects.
7. Focus on specific {role} related scenarios.
"""

    full_prompt = user_response if user_response else "START_INTERVIEW"
    if history:
        history_str = "\n".join([f"{'User' if i%2==0 else 'Interviewer'}: {msg}" for i, msg in enumerate(history)])
        full_prompt = f"Interview History:\n{history_str}\n\nUser's latest response: {user_response}\n\nNext interviewer question:"

    response = await app_state.ollama_service.generate(
        prompt=full_prompt,
        system_prompt=system_prompt,
        user_id=user_id
    )

    return safe_json_response({
        "question": response,
        "success": True
    })

@app.get("/", response_class=HTMLResponse)
async def root():
    app_state.monitoring.record("/")
    body = DASHBOARD_HTML.encode('utf-8')
    return Response(
        content=body,
        status_code=200,
        media_type="text/html; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

@app.get("/.well-known/{path:path}")
async def well_known(path: str):
    """Handle Chrome DevTools and other .well-known requests silently"""
    return Response(status_code=204)

@app.get("/camera-test", response_class=HTMLResponse)
async def camera_test():
    try:
        with open("camera_test.html", "r") as f:
            return Response(content=f.read(), media_type="text/html; charset=utf-8")
    except:
        return Response(content="<h1>camera_test.html not found</h1>", media_type="text/html")

@app.get("/api/health")
async def health():
    return safe_json_response({
        "status": "healthy",
        "ollama": app_state.ollama_service.is_available,
        "ollama_mode": "connected" if app_state.ollama_service.is_available else "fallback",
        "nlp": app_state.nlp_service.is_initialized if app_state.nlp_service else False,
        "graph_nodes": app_state.decision_graph.graph.number_of_nodes(),
        "rag_docs": app_state.rag_service.get_statistics() if app_state.rag_service else None,
        "services": {
            "journal": app_state.journal_service is not None,
            "simulation": app_state.simulation_service is not None,
            "coaching": app_state.coaching_service is not None,
            "market_intelligence": app_state.market_intelligence is not None,
            "community_insights": app_state.community_insights is not None,
            "gamification": app_state.gamification is not None,
            "emotion_detection": app_state.emotion_detection.initialized if app_state.emotion_detection else False
        }
    })

@app.get("/api/status")
async def get_system_status():

    return safe_json_response({
        "online": True,
        "ai_available": app_state.ollama_service.is_available,
        "mode": "online" if app_state.ollama_service.is_available else "fallback",
        "services": {
            "ollama": {
                "available": app_state.ollama_service.is_available,
                "model": settings.OLLAMA_MODEL if app_state.ollama_service.is_available else None
            },
            "nlp": {
                "available": app_state.nlp_service.is_initialized if app_state.nlp_service else False,
                "mode": "rule-based"
            },
            "rag": {
                "available": app_state.rag_service.initialized if app_state.rag_service else False,
                "documents": len(app_state.rag_service.documents) if app_state.rag_service else 0
            }
        },
        "features": {
            "chat": True,
            "analyze": True,
            "journal": True,
            "simulation": True,
            "coaching": True,
            "market_data": True,
            "community": True,
            "gamification": True
        }
    })

@app.post("/api/analyze")
async def analyze(decision: DecisionInput, request: Request):
    app_state.monitoring.record("/api/analyze")
    client_ip = request.client.host if request.client else "unknown"
    allowed, info = app_state.rate_limiter.is_allowed(decision.user_id or client_ip)
    if not allowed:
        raise HTTPException(status_code=429, detail=info)

    analysis_id = f"analysis_{uuid.uuid4().hex[:8]}"
    decision_data = decision.model_dump()

    nlp_analysis = app_state.nlp_service.analyze(decision.description, include_summary=True)
    nlp_dict = app_state.nlp_service.to_dict(nlp_analysis)

    if not decision.emotions and nlp_analysis.sentiment.emotions:
        decision_data['emotions'] = nlp_analysis.sentiment.emotions[:5]
        decision_data['auto_detected_emotions'] = True

    decision_data['nlp_confidence'] = nlp_analysis.text_analysis.confidence_level
    decision_data['detected_entities'] = {
        'job_titles': nlp_analysis.entities.job_titles,
        'skills': nlp_analysis.entities.skills,
        'industries': nlp_analysis.entities.industries
    }

    prediction = app_state.ml_predictor.predict(decision_data, include_explanation=True)

    decision_node = f"decision_{analysis_id}"
    app_state.decision_graph.add_decision(
        decision_id=decision_node, decision_type=decision.decision_type,
        description=decision.description, user_factors=decision_data
    )
    graph_analysis = app_state.decision_graph.analyze_decision(decision_node, decision_data)

    app_state.analytics.record_decision(
        user_id=decision.user_id or "anonymous", decision_id=analysis_id,
        decision_type=decision.decision_type, description=decision.description,
        predicted_regret=float(prediction["predicted_regret"])
    )

    humanized = app_state.humanizer.humanize_regret_analysis(prediction, decision_data)

    emotional_insights = app_state.nlp_service.get_emotional_insights(
        decision_data.get('emotions', [])
    )

    response_data = {
        "analysis_id": analysis_id,
        "prediction": prediction,
        "graph_analysis": graph_analysis,
        "humanized": {"main_message": humanized.main_message},
        "nlp_analysis": {
            "sentiment": nlp_dict['sentiment'],
            "intent": nlp_dict['intent'],
            "entities": nlp_dict['entities'],
            "keywords": nlp_dict['keywords'],
            "text_metrics": {
                "confidence_level": nlp_analysis.text_analysis.confidence_level,
                "uncertainty_markers": nlp_analysis.text_analysis.uncertainty_markers,
                "complexity": nlp_analysis.text_analysis.complexity_score
            },
            "summary": nlp_dict['summary']
        },
        "emotional_insights": emotional_insights,
        "timestamp": datetime.utcnow().isoformat()
    }

    return safe_json_response(response_data)

@app.post("/api/context/clear")
async def clear_context(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "default")

    app_state.ollama_service.clear_conversation(user_id)

    app_state.upload_service.clear_user_context(user_id)

    return safe_json_response({"status": "cleared", "message": "Context and files cleared"})

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), user_id: str = "default", current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/upload")
    try:
        content = await file.read()
        uploaded_file = await app_state.upload_service.process_file(
            content, file.filename, user_id
        )
        response_data = {
            "id": uploaded_file.id,
            "filename": uploaded_file.original_name,
            "status": "success",
            "extracted_content_preview": uploaded_file.extracted_content[:200] + "..." if uploaded_file.extracted_content else ""
        }

        if uploaded_file.metadata.get("is_resume"):
            response_data["is_resume"] = True
            if "resume_data" in uploaded_file.metadata:
                response_data["resume_id"] = uploaded_file.metadata["resume_data"].get("id")

        return safe_json_response(response_data)
    except Exception as e:
        print(f"Upload error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/upload/url")
async def upload_from_url(user_id: str = "default", url: str = None, current_user: str = Depends(get_current_user)):
    """Upload and process content from a URL (YouTube, articles, etc.)"""
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/upload/url")
    
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")
    
    try:
        from services.media_ingestion_service import media_ingestion_service
        
        media_source = await media_ingestion_service.process_url(url, user_id)
        
        await app_state.upload_service.train_system_with_media(user_id, media_source)
        
        return safe_json_response({
            "status": "success",
            "media_id": media_source.id,
            "title": media_source.title,
            "type": media_source.source_type,
            "processing_status": media_source.processing_status,
            "message": f"Successfully added {media_source.source_type}: {media_source.title}. The system is now training with this content."
        })
            
    except Exception as e:
        print(f"URL upload error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to process URL: {str(e)}")

@app.post("/api/upload/video")
async def upload_video_file(file: UploadFile = File(...), user_id: str = "default", current_user: str = Depends(get_current_user)):
    """Upload and process a video file for training"""
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/upload/video")
    
    allowed_extensions = {'.mp4', '.webm', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        from services.media_ingestion_service import media_ingestion_service
        
        content = await file.read()
        media_source = await media_ingestion_service.process_video_file(
            content, file.filename, user_id
        )

        await app_state.upload_service.train_system_with_media(user_id, media_source)
        
        return safe_json_response({
            "status": "success",
            "media_id": media_source.id,
            "title": media_source.title,
            "type": "video",
            "processing_status": media_source.processing_status,
            "message": f"Video uploaded and queued for processing: {media_source.title}. The system is now training with this video content."
        })
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Video upload error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to process video: {str(e)}")

@app.get("/api/media/{user_id}")
async def get_user_media(user_id: str, current_user: str = Depends(get_current_user)):
    """Get all media sources (videos, URLs) uploaded by user"""
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/media/{user_id}")
    
    try:
        from services.media_ingestion_service import media_ingestion_service
        
        media_sources = media_ingestion_service.get_user_media(user_id)
        
        media_list = [
            {
                "id": m.id,
                "type": m.source_type,
                "title": m.title,
                "url": m.original_url,
                "status": m.processing_status,
                "upload_time": m.upload_time.isoformat(),
                "duration": m.duration,
                "size": m.size,
                "error": m.error_message if m.processing_status == "failed" else None
            }
            for m in media_sources
        ]
        
        return safe_json_response({
            "media": media_list,
            "count": len(media_list)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve media: {str(e)}")

@app.get("/api/media/info/{media_id}")
async def get_media_info(media_id: str, user_id: str = "default", current_user: str = Depends(get_current_user)):
    """Get detailed information about a specific media source"""
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/media/info/{media_id}")
    
    try:
        from services.media_ingestion_service import media_ingestion_service
        
        media = media_ingestion_service.get_media_info(media_id)
        
        if not media:
            raise HTTPException(status_code=404, detail="Media not found")
        
        if media.user_id != user_id:
            raise HTTPException(status_code=403, detail="Not authorized to access this media")
        
        return safe_json_response({
            "id": media.id,
            "type": media.source_type,
            "title": media.title,
            "description": media.description,
            "url": media.original_url,
            "status": media.processing_status,
            "upload_time": media.upload_time.isoformat(),
            "duration": media.duration,
            "size": media.size,
            "extracted_content": media.extracted_content[:1000] if media.extracted_content else "",
            "transcript_preview": media.transcript[:500] if media.transcript else "",
            "metadata": media.metadata,
            "error": media.error_message if media.processing_status == "failed" else None
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve media info: {str(e)}")

@app.delete("/api/media/{media_id}")
async def delete_media(media_id: str, user_id: str = "default", current_user: str = Depends(get_current_user)):
    """Delete a media source"""
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/media/{media_id}")
    
    try:
        from services.media_ingestion_service import media_ingestion_service
        
        media = media_ingestion_service.get_media_info(media_id)
        
        if not media:
            raise HTTPException(status_code=404, detail="Media not found")
        
        if media.user_id != user_id:
            raise HTTPException(status_code=403, detail="Not authorized to delete this media")
        
        if media.file_path and Path(media.file_path).exists():
            try:
                Path(media.file_path).unlink()
            except:
                pass
        
        del media_ingestion_service.media_db[media_id]
        if user_id in media_ingestion_service.user_media:
            media_ingestion_service.user_media[user_id] = [
                m for m in media_ingestion_service.user_media[user_id] if m.id != media_id
            ]
        
        return safe_json_response({"status": "deleted", "media_id": media_id})
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete media: {str(e)}")

@app.get("/api/media-stats/{user_id}")
async def get_media_stats(user_id: str, current_user: str = Depends(get_current_user)):
    """Get statistics about user's uploaded media"""
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/media-stats/{user_id}")
    
    try:
        from services.media_ingestion_service import media_ingestion_service
        
        user_media = media_ingestion_service.get_user_media(user_id)
        
        stats = {
            "total_media": len(user_media),
            "by_type": {},
            "total_size": 0,
            "total_duration": 0,
            "processing_status": {}
        }
        
        for media in user_media:
            stats["by_type"][media.source_type] = stats["by_type"].get(media.source_type, 0) + 1
            
            stats["total_size"] += media.size
            
            stats["total_duration"] += media.duration
            
            stats["processing_status"][media.processing_status] = stats["processing_status"].get(media.processing_status, 0) + 1
        
        return safe_json_response(stats)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@app.post("/api/chat")
async def chat(chat_input: ChatInput):
    app_state.monitoring.record("/api/chat")

    intent_result = app_state.nlp_service.classify_intent(chat_input.message)
    sentiment_result = app_state.nlp_service.analyze_sentiment(chat_input.message)
    user_id = chat_input.user_id or "default"
    user_context = app_state.upload_service.get_user_context(user_id)
    file_context_str = user_context.get_context_for_ai()

    nlp_context = {
        'intent': intent_result.intent,
        'intent_confidence': intent_result.confidence,
        'sentiment': sentiment_result.sentiment,
        'emotions': sentiment_result.emotions,
        'file_context': file_context_str,
        'decision_type': 'career_decision'
    }

    response = await app_state.ollama_service.chat(
        chat_input.message, user_id=user_id,
        context=nlp_context
    )

    return safe_json_response({
        "response": response,
        "nlp": {
            "detected_intent": intent_result.intent,
            "sentiment": sentiment_result.sentiment,
            "emotions": sentiment_result.emotions[:3] if sentiment_result.emotions else []
        },
        "timestamp": datetime.utcnow().isoformat()
    })

@app.post("/api/feedback")
async def feedback(fb: FeedbackInput):
    entry = app_state.feedback_loop.add_feedback(
        user_id=fb.user_id or "anonymous", feedback_type=fb.feedback_type,
        content=fb.content, analysis_id=fb.analysis_id
    )
    return safe_json_response({"feedback_id": entry.id, "status": "received"})

@app.get("/api/graph")
async def get_graph():
    app_state.monitoring.record("/api/graph")
    return safe_json_response({
        "graph": app_state.decision_graph.export_graph(),
        "stats": app_state.decision_graph.get_graph_statistics()
    })

@app.get("/api/insights")
async def get_insights():
    return safe_json_response(app_state.feedback_loop.get_improvement_insights())

@app.get("/api/analytics/{user_id}")
async def get_analytics(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/analytics/{user_id}")
    analytics = app_state.analytics.get_user_analytics(user_id)
    trends = app_state.analytics.get_trends(user_id)
    return safe_json_response({**analytics, "trends": trends})

@app.get("/api/report/{user_id}")
async def get_report(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    report = app_state.analytics.generate_report(user_id)
    return safe_json_response(report)

@app.get("/api/metrics")
async def get_metrics():
    return safe_json_response(app_state.monitoring.get_metrics())

class NLPInput(BaseModel):
    text: str
    include_summary: bool = False

@app.post("/api/nlp/analyze")
async def analyze_text(nlp_input: NLPInput):

    app_state.monitoring.record("/api/nlp/analyze")

    analysis = app_state.nlp_service.analyze(nlp_input.text, include_summary=nlp_input.include_summary)
    result = app_state.nlp_service.to_dict(analysis)

    if analysis.sentiment.emotions:
        result['emotional_insights'] = app_state.nlp_service.get_emotional_insights(
            analysis.sentiment.emotions
        )

    return safe_json_response(result)

@app.post("/api/nlp/sentiment")
async def analyze_sentiment(nlp_input: NLPInput):

    result = app_state.nlp_service.analyze_sentiment(nlp_input.text)
    return safe_json_response({
        "sentiment": result.sentiment,
        "confidence": result.confidence,
        "emotions": result.emotions,
        "emotion_scores": result.emotion_scores
    })

@app.post("/api/nlp/intent")
async def classify_intent(nlp_input: NLPInput):

    result = app_state.nlp_service.classify_intent(nlp_input.text)
    return safe_json_response({
        "intent": result.intent,
        "confidence": result.confidence,
        "sub_intents": result.sub_intents
    })

@app.post("/api/nlp/entities")
async def extract_entities(nlp_input: NLPInput):

    result = app_state.nlp_service.extract_entities(nlp_input.text)
    return safe_json_response({
        "job_titles": result.job_titles,
        "companies": result.companies,
        "skills": result.skills,
        "industries": result.industries,
        "locations": result.locations,
        "time_references": result.time_references,
        "monetary_values": result.monetary_values
    })

@app.post("/api/nlp/keywords")
async def extract_keywords(nlp_input: NLPInput):

    keywords = app_state.nlp_service.extract_keywords(nlp_input.text, top_n=15)
    return safe_json_response({"keywords": keywords})

@app.post("/api/nlp/summarize")
async def summarize_text(nlp_input: NLPInput):

    summary = app_state.nlp_service.summarize(nlp_input.text, max_sentences=3)
    return safe_json_response({
        "original_length": len(nlp_input.text),
        "summary_length": len(summary),
        "summary": summary
    })

class JournalInput(BaseModel):
    decision_type: str
    title: str
    description: str
    predicted_regret: float = 0.5
    predicted_confidence: float = 0.5
    emotions: List[str] = []
    alternatives: List[str] = []
    tags: List[str] = []

@app.post("/api/journal/create")
@app.post("/api/journal/create")
async def create_journal_entry(entry: JournalInput, user_id: str = "default", current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)

    app_state.monitoring.record("/api/journal/create")

    result = app_state.journal_service.create_entry(
        user_id=user_id,
        decision_type=entry.decision_type,
        title=entry.title,
        description=entry.description,
        predicted_regret=entry.predicted_regret,
        predicted_confidence=entry.predicted_confidence,
        emotions=entry.emotions,
        alternatives=entry.alternatives,
        tags=entry.tags
    )

    app_state.gamification.record_activity(user_id, 'decision_analyzed')

    return safe_json_response(app_state.journal_service.to_dict(result))

@app.get("/api/journal/{user_id}")
async def get_journal_entries(user_id: str, current_user: str = Depends(get_current_user), limit: int = 50):
    verify_owner(user_id, current_user)
    entries = app_state.journal_service.get_user_entries(user_id, limit=limit)
    return safe_json_response({
        "entries": [app_state.journal_service.to_dict(e) for e in entries],
        "total": len(entries)
    })

@app.get("/api/journal/entry/{entry_id}")
async def get_journal_entry(entry_id: str, current_user: str = Depends(get_current_user)):
    entry = app_state.journal_service.get_entry(entry_id)
    if not entry:
        raise HTTPException(404, "Entry not found")
    verify_owner(entry.user_id, current_user)
    return safe_json_response(app_state.journal_service.to_dict(entry))

@app.post("/api/journal/{entry_id}/record-decision")
async def record_decision(entry_id: str, chosen_option: str, notes: str = "", current_user: str = Depends(get_current_user)):
    entry = app_state.journal_service.get_entry(entry_id)
    if not entry:
        raise HTTPException(404, "Entry not found")
    verify_owner(entry.user_id, current_user)
    
    result = app_state.journal_service.record_decision(entry_id, chosen_option, notes)
    if not result:
        raise HTTPException(404, "Entry not found")
    return safe_json_response(app_state.journal_service.to_dict(result))

class JournalOutcomeInput(BaseModel):
    actual_regret: float
    satisfaction: float
    would_decide_same: bool
    lessons_learned: str = ""
    unexpected_outcomes: List[str] = []

@app.post("/api/journal/{entry_id}/record-outcome")
async def record_outcome(entry_id: str, outcome: JournalOutcomeInput, current_user: str = Depends(get_current_user)):
    entry = app_state.journal_service.get_entry(entry_id)
    if not entry:
        raise HTTPException(404, "Entry not found")
    verify_owner(entry.user_id, current_user)

    result = app_state.journal_service.record_outcome(
        entry_id=entry_id,
        actual_regret=outcome.actual_regret,
        satisfaction=outcome.satisfaction,
        would_decide_same=outcome.would_decide_same,
        lessons_learned=outcome.lessons_learned,
        unexpected_outcomes=outcome.unexpected_outcomes
    )

    app_state.gamification.record_activity(entry.user_id, 'outcome_recorded')

    return safe_json_response(app_state.journal_service.to_dict(result))

@app.get("/api/journal/{user_id}/followups")
async def get_pending_followups(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    followups = app_state.journal_service.get_pending_followups(user_id)
    return safe_json_response({"follow_ups": followups})

@app.get("/api/journal/{user_id}/accuracy")
async def get_journal_accuracy(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    metrics = app_state.journal_service.get_accuracy_metrics(user_id)
    return safe_json_response(metrics)

@app.get("/api/journal/{user_id}/timeline")
async def get_decision_timeline(user_id: str, current_user: str = Depends(get_current_user), days: int = 365):
    verify_owner(user_id, current_user)
    timeline = app_state.journal_service.get_timeline(user_id, days)
    return safe_json_response({"timeline": timeline})

@app.get("/api/journal/{user_id}/statistics")
async def get_journal_statistics(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    stats = app_state.journal_service.get_statistics(user_id)
    accuracy = app_state.journal_service.get_accuracy_metrics(user_id)
    return safe_json_response({
        "statistics": stats,
        "accuracy_metrics": accuracy
    })

@app.get("/api/journal/{user_id}/insights")
async def get_journal_insights(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)

    entries = app_state.journal_service.get_user_entries(user_id, limit=100)
    accuracy = app_state.journal_service.get_accuracy_metrics(user_id)
    stats = app_state.journal_service.get_statistics(user_id)

    insights = []

    if accuracy.get('accuracy') is not None:
        acc = accuracy['accuracy']
        if acc > 0.8:
            insights.append("Your predictions have been highly accurate - trust your judgment!")
        elif acc > 0.6:
            insights.append("Your prediction accuracy is good - continue tracking for improvement.")
        else:
            insights.append("Consider gathering more information before making predictions.")

    if accuracy.get('repeat_decision_rate') is not None:
        rate = accuracy['repeat_decision_rate']
        if rate > 0.7:
            insights.append(f"{rate * 100:.0f}% of your decisions you would make again - great decision-making!")
        elif rate < 0.4:
            insights.append("Consider what factors led to regretted decisions.")

    if stats.get('by_type'):
        most_common = max(stats['by_type'].items(), key=lambda x: x[1], default=None)
        if most_common:
            insights.append(f"Your most common decision type is {most_common[0].replace('_', ' ').title()}")

    if stats.get('most_common_emotions'):
        top_emotion = stats['most_common_emotions'][0] if stats['most_common_emotions'] else None
        if top_emotion:
            insights.append(f"You often feel {top_emotion['emotion']} when making decisions")

    if not insights:
        insights.append("Continue tracking decisions to generate personalized insights.")

    return safe_json_response({
        "insights": insights,
        "total_decisions": stats.get('total_entries', 0),
        "avg_regret": accuracy.get('avg_prediction_error', 0),
        "accuracy_rate": accuracy.get('accuracy')
    })


class SimulationInput(BaseModel):
    decision_type: str
    years: int = 5
    initial_salary: float = 80000
    initial_satisfaction: float = 0.6
    risk_tolerance: float = 0.5
    current_career_level: int = 3

class ScenarioInput(BaseModel):
    name: str
    decision_type: str
    salary: float = 80000
    satisfaction: float = 0.6
    risk_tolerance: float = 0.5

@app.post("/api/simulation/run")
@app.post("/api/simulate/run")
async def run_simulation(sim_input: SimulationInput, user_id: str = "default", current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)

    app_state.monitoring.record("/api/simulation/run")

    result = app_state.simulation_service.monte_carlo_simulation(
        decision_type=sim_input.decision_type,
        years=sim_input.years,
        num_simulations=500,
        initial_salary=sim_input.initial_salary,
        initial_satisfaction=sim_input.initial_satisfaction,
        risk_tolerance=sim_input.risk_tolerance,
        current_career_level=sim_input.current_career_level
    )

    app_state.gamification.record_activity(user_id, 'simulation_run')

    return safe_json_response(result)

@app.post("/api/simulation/compare")
async def compare_scenarios(scenario_a: ScenarioInput, scenario_b: ScenarioInput, years: int = 5):

    comparison = app_state.simulation_service.compare_scenarios(
        scenario_a=scenario_a.model_dump(),
        scenario_b=scenario_b.model_dump(),
        years=years
    )
    return safe_json_response(app_state.simulation_service.to_dict(comparison))

@app.post("/api/simulation/projections")
async def get_projections(sim_input: SimulationInput):

    projections = app_state.simulation_service.generate_year_by_year_projection(
        decision_type=sim_input.decision_type,
        years=sim_input.years,
        initial_salary=sim_input.initial_salary
    )
    return safe_json_response({"projections": projections})

class CoachingInput(BaseModel):
    session_type: str = "general"
    current_text: str = ""

@app.post("/api/coaching/session")
@app.post("/api/coaching/session")
async def create_coaching_session(coaching_input: CoachingInput, user_id: str = "default", current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)

    app_state.monitoring.record("/api/coaching/session")

    recent_entries = app_state.journal_service.get_user_entries(user_id, limit=10)
    recent_decisions = [app_state.journal_service.to_dict(e) for e in recent_entries]

    session = app_state.coaching_service.create_coaching_session(
        user_id=user_id,
        session_type=coaching_input.session_type,
        recent_decisions=recent_decisions,
        current_text=coaching_input.current_text
    )

    return safe_json_response(app_state.coaching_service.to_dict(session))

@app.get("/api/coaching/profile/{user_id}")
async def get_coaching_profile(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)

    profile = app_state.coaching_service.get_or_create_profile(user_id)
    return safe_json_response({
        "user_id": profile.user_id,
        "decision_style": profile.decision_style,
        "risk_profile": profile.risk_profile,
        "primary_biases": [b.value for b in profile.primary_biases],
        "strengths": profile.strengths,
        "growth_areas": profile.growth_areas,
        "total_sessions": profile.total_sessions,
        "progress_score": profile.progress_score
    })

@app.get("/api/coaching/{user_id}/actions")
async def get_action_items(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)

    items = app_state.coaching_service.get_user_action_items(user_id)
    return safe_json_response({
        "action_items": [
            {
                "id": a.id,
                "title": a.title,
                "description": a.description,
                "priority": a.priority,
                "category": a.category,
                "due_date": a.due_date.isoformat() if a.due_date else None
            }
            for a in items
        ]
    })

@app.post("/api/coaching/actions/{action_id}/complete")
@app.post("/api/coaching/complete-action/{action_id}")
async def complete_action(action_id: str, user_id: str = "default", current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)

    success = app_state.coaching_service.complete_action_item(user_id, action_id)
    if success:
        app_state.gamification.record_activity(user_id, 'action_completed')
    return safe_json_response({"success": success})

@app.get("/api/coaching/{user_id}/weekly-checkin")
async def weekly_checkin(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)

    checkin = app_state.coaching_service.get_weekly_checkin(user_id)
    return safe_json_response(checkin)

@app.post("/api/coaching/detect-biases")
async def detect_biases(text: str):

    biases = app_state.coaching_service.detect_biases(text)
    return safe_json_response({
        "biases": [
            {
                "type": b.bias_type.value,
                "confidence": b.confidence,
                "evidence": b.evidence,
                "mitigation_tips": b.mitigation_tips
            }
            for b in biases
        ]
    })

@app.get("/api/market/salary")
async def get_salary_benchmark(role: str, location: str = "remote_us", experience_years: int = 5):

    salary = app_state.market_intelligence.get_salary_benchmark(role, location, experience_years)
    return safe_json_response(app_state.market_intelligence.to_dict(salary))

@app.get("/api/market/salary/compare")
async def compare_salaries(role: str, current_location: str, target_location: str, experience_years: int = 5):

    comparison = app_state.market_intelligence.compare_salaries(
        role, current_location, target_location, experience_years
    )
    return safe_json_response(comparison)

@app.get("/api/market/industry/{industry}")
async def get_industry_trend(industry: str):

    trend = app_state.market_intelligence.get_industry_trend(industry)
    return safe_json_response({
        "industry": trend.industry,
        "trend_direction": trend.trend_direction,
        "growth_rate": trend.growth_rate,
        "hiring_activity": trend.hiring_activity,
        "emerging_roles": trend.emerging_roles,
        "declining_roles": trend.declining_roles,
        "key_skills": trend.key_skills
    })

@app.get("/api/market/health")
async def get_market_health(industry: str = "technology", location: str = "remote_us"):

    health = app_state.market_intelligence.get_job_market_health(industry, location)
    return safe_json_response({
        "industry": health.industry,
        "location": health.location,
        "demand_score": health.demand_score,
        "supply_score": health.supply_score,
        "competition_level": health.competition_level,
        "avg_time_to_hire_days": health.avg_time_to_hire_days,
        "remote_friendly_pct": health.remote_friendly_pct,
        "salary_trend": health.salary_trend
    })

@app.get("/api/market/skill/{skill}")
async def get_skill_demand(skill: str):

    demand = app_state.market_intelligence.get_skill_demand(skill)
    return safe_json_response({
        "skill": demand.skill,
        "demand_score": demand.demand_score,
        "trend": demand.trend,
        "salary_premium": demand.avg_salary_premium,
        "related_roles": demand.related_roles
    })

@app.post("/api/market/skills-gap")
async def analyze_skills_gap(current_skills: List[str], target_role: str):

    analysis = app_state.market_intelligence.get_skills_gap_analysis(current_skills, target_role)
    return safe_json_response(analysis)

@app.get("/api/market/summary")
async def get_market_summary(industry: str = "technology", location: str = "remote_us"):

    summary = app_state.market_intelligence.get_market_summary(industry, location)
    return safe_json_response(summary)

@app.get("/api/community/social-proof/{decision_type}")
async def get_social_proof(decision_type: str):

    proof = app_state.community_insights.get_social_proof(decision_type)
    return safe_json_response(app_state.community_insights.to_dict(proof))

@app.post("/api/community/compare")
async def compare_to_community(decision_type: str, user_data: Dict[str, Any]):

    comparison = app_state.community_insights.get_pattern_comparison(decision_type, user_data)
    return safe_json_response(comparison)

@app.get("/api/community/stats/{decision_type}")
async def get_community_stats(decision_type: str):

    stats = app_state.community_insights.get_similar_decisions_stats(decision_type)
    return safe_json_response(stats)

@app.post("/api/community/contribute")
@app.post("/api/outcome/contribute")
async def contribute_outcome(decision_type: str, outcome_data: Dict[str, Any], user_id: str = "default", current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)

    result = app_state.community_insights.contribute_outcome(user_id, decision_type, outcome_data)

    app_state.gamification.record_activity(user_id, 'story_shared')

    return safe_json_response(result)

@app.get("/api/community/wisdom/{decision_type}")
async def get_community_wisdom(decision_type: str, count: int = 3):

    nuggets = app_state.community_insights.get_wisdom_nuggets(decision_type, count)
    return safe_json_response({"wisdom": nuggets})

@app.post("/api/export/report")
async def generate_report(user_id: str = "default", decision_data: Dict = None, analysis_result: Dict = None):

    app_state.monitoring.record("/api/export/report")

    report = app_state.export_service.generate_decision_report(
        user_id=user_id,
        decision_data=decision_data or {},
        analysis_result=analysis_result or {}
    )

    app_state.gamification.record_activity(user_id, 'report_generated')

    return safe_json_response(app_state.export_service.to_dict(report))

@app.get("/api/export/report/{report_id}")
async def get_report(report_id: str, format: str = "json"):

    report = app_state.export_service.generated_reports.get(report_id)
    if not report:
        raise HTTPException(404, "Report not found")

    if format == "markdown":
        content = app_state.export_service.export_to_markdown(report)
        return Response(content=content, media_type="text/markdown")
    elif format == "json":
        return safe_json_response(app_state.export_service.to_dict(report))
    else:
        return safe_json_response(app_state.export_service.to_dict(report))

@app.get("/api/export/journal/{user_id}")
async def export_journal(user_id: str, current_user: str = Depends(get_current_user), format: str = "json"):
    verify_owner(user_id, current_user)

    entries = app_state.journal_service.get_user_entries(user_id, limit=1000)
    entries_dict = [app_state.journal_service.to_dict(e) for e in entries]

    if format == "csv":
        csv_content = app_state.export_service.export_to_csv(entries_dict)
        return Response(content=csv_content, media_type="text/csv")
    else:
        return safe_json_response({"entries": entries_dict, "count": len(entries_dict)})

@app.get("/api/export/calendar/{user_id}")
async def export_calendar(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)

    followups = app_state.journal_service.get_pending_followups(user_id)
    entries = app_state.journal_service.get_user_entries(user_id)

    events = app_state.export_service.get_calendar_events(
        [app_state.journal_service.to_dict(e) for e in entries],
        followups
    )

    ical = app_state.export_service.generate_ical(events)
    return Response(content=ical, media_type="text/calendar")

@app.get("/api/gamification/{user_id}")
async def get_gamification_status(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)

    user = app_state.gamification.get_or_create_user(user_id)
    stats = app_state.gamification.get_user_stats(user_id)
    return safe_json_response(stats)

@app.get("/api/gamification/{user_id}/achievements")
async def get_achievements(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)

    user = app_state.gamification.get_or_create_user(user_id)
    return safe_json_response({
        "achievements": [
            {
                "id": a.id,
                "name": a.name,
                "description": a.description,
                "icon": a.icon,
                "level": a.level.value,
                "points": a.points,
                "unlocked": a.unlocked,
                "unlocked_at": a.unlocked_at.isoformat() if a.unlocked_at else None
            }
            for a in user.achievements
        ]
    })

@app.get("/api/gamification/{user_id}/challenges")
async def get_daily_challenges(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)

    challenges = app_state.gamification.get_daily_challenges(user_id)
    return safe_json_response({
        "challenges": [
            {
                "id": c.id,
                "title": c.title,
                "description": c.description,
                "points": c.reward_points,
                "completed": c.completed,
                "expires_at": c.expires_at.isoformat()
            }
            for c in challenges
        ]
    })

@app.post("/api/gamification/challenges/{challenge_id}/complete")
@app.post("/api/gamification/challenge/complete/{challenge_id}")
async def complete_challenge(challenge_id: str, user_id: str = "default", current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)

    result = app_state.gamification.complete_challenge(user_id, challenge_id)
    return safe_json_response(result)

@app.get("/api/gamification/leaderboard")
async def get_leaderboard(limit: int = 10):

    leaderboard = app_state.gamification.get_leaderboard(limit)
    return safe_json_response({"leaderboard": leaderboard})

@app.get("/api/gamification/{user_id}/motivation")
async def get_motivation(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)

    message = app_state.gamification.get_motivational_message(user_id)
    prompts = app_state.gamification.get_reflection_prompts()
    return safe_json_response({
        "message": message,
        "reflection_prompts": prompts
    })

@app.post("/api/auth/register")
async def register(user: UserRegister, request: Request):
    """Register a new user with strict validation"""
    client_ip = request.client.host if request.client else "unknown"

    security_helper = get_security_helper()
    allowed, info = security_helper.rate_limiter.is_allowed(f"register:{client_ip}")
    if not allowed:
        raise HTTPException(429, f"Too many registration attempts. Retry after {info.get('retry_after', 60)} seconds.")

    result, message = app_state.auth_service.register(user.username, user.email, user.password)

    if result is None:
        get_audit_logger().log(
            event_type="AUTH",
            ip_address=client_ip,
            resource="user",
            action="register_failed",
            success=False,
            details={"username": user.username[:3] + "***"}
        )
        raise HTTPException(400, message)

    return safe_json_response({
        "success": True,
        "message": "Registration successful. Please login.",
        "user_id": result.id
    })

@app.post("/api/auth/login")
async def login(user: UserLogin, request: Request):
    """Authenticate user with brute force protection"""
    client_ip = request.client.host if request.client else "unknown"

    result, message = app_state.auth_service.authenticate(user.username, user.password)

    if result is None:
        raise HTTPException(401, "Invalid credentials")

    session_token = app_state.auth_service.create_session(result.id) if hasattr(app_state.auth_service, 'create_session') else None

    return safe_json_response({
        "success": True,
        "user_id": result.id,
        "username": result.username,
        "email": result.email,
        "api_key": result.api_key if os.getenv("ENABLE_API_KEYS", "false").lower() == "true" else None,
        "session_token": session_token
    })

@app.post("/api/auth/logout")
async def logout(request: Request):
    """Invalidate user session"""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        session_token = auth_header[7:]
        if hasattr(app_state.auth_service, 'logout'):
            app_state.auth_service.logout(session_token)

    return safe_json_response({"success": True, "message": "Logged out successfully"})

@app.get("/api/security/audit")
async def get_security_audit(limit: int = 100, admin_user: str = Depends(get_current_admin)):
    """Get security audit log (admin only)"""
    audit = get_audit_logger()
    events = audit.get_events(limit=min(limit, 1000))

    return safe_json_response({
        "events": [
            {
                "timestamp": e.timestamp.isoformat(),
                "event_type": e.event_type,
                "user_id": e.user_id,
                "ip_address": e.ip_address,
                "resource": e.resource,
                "action": e.action,
                "success": e.success
            }
            for e in events
        ],
        "total": len(events)
    })

@app.get("/api/security/suspicious")
async def get_suspicious_activity(request: Request, hours: int = 24):
    """Get suspicious activity report (admin only in production)"""
    if not settings.DEBUG:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(401, "Authentication required")

    audit = get_audit_logger()
    suspicious = audit.get_suspicious_activity(hours=min(hours, 168))

    return safe_json_response({
        "suspicious_events": [
            {
                "timestamp": e.timestamp.isoformat(),
                "event_type": e.event_type,
                "ip_address": e.ip_address,
                "resource": e.resource,
                "action": e.action,
                "details": e.details
            }
            for e in suspicious
        ],
        "total": len(suspicious),
        "period_hours": hours
    })


class EmotionDetectionInput(BaseModel):
    image: str = Field(..., description="Base64 encoded image")
    user_id: Optional[str] = Field(default=None, max_length=100)
    annotate: bool = Field(default=False, description="Return annotated image with face boxes")

@app.post("/api/emotion/detect")
async def detect_emotions(input_data: EmotionDetectionInput):
    app_state.monitoring.record("/api/emotion/detect")

    if not app_state.emotion_detection.initialized:
        raise HTTPException(503, "Emotion detection service not available")

    result = app_state.emotion_detection.analyze_base64_image(input_data.image)

    response_data = app_state.emotion_detection.to_dict(result)

    if input_data.user_id and result.emotions:
        for emotion_result in result.emotions:
            app_state.emotion_detection.record_emotion(input_data.user_id, emotion_result)

    if input_data.annotate and result.faces_detected > 0:
        image = app_state.emotion_detection.decode_base64_image(input_data.image)
        if image is not None:
            annotated = app_state.emotion_detection.annotate_image(image, result)
            response_data['annotated_image'] = app_state.emotion_detection.encode_image_to_base64(annotated)

    return safe_json_response(response_data)

@app.get("/api/emotion/history/{user_id}")
async def get_emotion_history(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/emotion/history")

    history = app_state.emotion_detection.get_emotion_history(user_id)
    trends = app_state.emotion_detection.get_emotion_trends(user_id)

    return safe_json_response({
        "user_id": user_id,
        "history": history[-50:],
        "trends": trends
    })

@app.get("/api/emotion/trends/{user_id}")
async def get_emotion_trends(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/emotion/trends")

    trends = app_state.emotion_detection.get_emotion_trends(user_id)

    recommendations = []
    if trends.get('dominant_emotion'):
        from services.emotion_detection_service import EmotionDetectionService
        recommendations = EmotionDetectionService.EMOTION_RECOMMENDATIONS.get(
            trends['dominant_emotion'], []
        )

    return safe_json_response({
        "user_id": user_id,
        "trends": trends,
        "recommendations": recommendations
    })

@app.get("/api/emotion/status")
async def get_emotion_service_status():
    return safe_json_response({
        "initialized": app_state.emotion_detection.initialized,
        "available": app_state.emotion_detection.initialized,
        "supported_emotions": app_state.emotion_detection.EMOTION_LABELS if app_state.emotion_detection else [],
        "model_type": "rule-based"
    })

@app.post("/api/speech-to-text")
async def speech_to_text(request: Request, audio: UploadFile = File(None)):
    app_state.monitoring.record("/api/speech-to-text")

    try:
        import whisper
        import tempfile
        import base64
        import os

        audio_bytes = None

        if audio:
            audio_bytes = await audio.read()
        else:
            try:
                body = await request.json()
                audio_data = body.get("audio", "")

                if not audio_data:
                    return safe_json_response({"error": "No audio data provided", "text": ""})

                if "," in audio_data:
                    audio_data = audio_data.split(",")[1]

                audio_bytes = base64.b64decode(audio_data)
            except:
                return safe_json_response({"error": "Invalid audio data", "text": ""})

        if not audio_bytes:
            return safe_json_response({"error": "No audio data provided", "text": ""})

        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        try:
            model = whisper.load_model("base")
            result = model.transcribe(temp_path)
            text = result.get("text", "").strip()

            return safe_json_response({
                "text": text,
                "success": True
            })
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except ImportError:
        return safe_json_response({
            "error": "Whisper not installed. Run: pip install openai-whisper",
            "text": "",
            "success": False
        })
    except Exception as e:
        print(f"Speech-to-text error: {e}")
        return safe_json_response({
            "error": str(e),
            "text": "",
            "success": False
        })

class OutcomeInput(BaseModel):
    decision_id: str = Field(..., max_length=100)
    user_id: str = Field(..., max_length=100)
    predicted_regret: float = Field(..., ge=0, le=100)
    actual_regret: float = Field(..., ge=0, le=100)
    satisfaction_score: float = Field(..., ge=0, le=100)
    outcome_notes: Optional[str] = Field(default="", max_length=2000)
    surprises: Optional[List[str]] = Field(default=[])
    lessons_learned: Optional[str] = Field(default="", max_length=2000)
    decision_date: Optional[str] = Field(default=None)

@app.post("/api/outcome/record")
async def record_decision_outcome(input_data: OutcomeInput):
    app_state.monitoring.record("/api/outcome/record")

    if not app_state.outcome_learning:
        raise HTTPException(503, "Outcome learning service not available")

    decision_date = None
    if input_data.decision_date:
        try:
            decision_date = datetime.fromisoformat(input_data.decision_date.replace('Z', '+00:00'))
        except:
            pass

    result = app_state.outcome_learning.record_outcome(
        decision_id=input_data.decision_id,
        user_id=input_data.user_id,
        predicted_regret=input_data.predicted_regret,
        actual_regret=input_data.actual_regret,
        satisfaction_score=input_data.satisfaction_score,
        outcome_notes=input_data.outcome_notes,
        surprises=input_data.surprises,
        lessons_learned=input_data.lessons_learned,
        decision_date=decision_date
    )

    return safe_json_response(result)

@app.get("/api/outcome/profile/{user_id}")
async def get_learning_profile(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/outcome/profile")

    if not app_state.outcome_learning:
        raise HTTPException(503, "Outcome learning service not available")

    profile = app_state.outcome_learning.get_learning_profile(user_id)
    return safe_json_response(profile)

@app.get("/api/outcome/history/{user_id}")
async def get_outcome_history(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/outcome/history")

    if not app_state.outcome_learning:
        raise HTTPException(503, "Outcome learning service not available")

    history = app_state.outcome_learning.get_outcome_history(user_id)
    return safe_json_response({"outcomes": history, "total": len(history)})

@app.get("/api/outcome/prediction-reality/{user_id}")
async def get_prediction_vs_reality(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/outcome/prediction-reality")

    if not app_state.outcome_learning:
        raise HTTPException(503, "Outcome learning service not available")

    data = app_state.outcome_learning.get_prediction_vs_reality_data(user_id)
    return safe_json_response(data)

@app.get("/api/outcome/adjusted-prediction/{user_id}/{base_prediction}")
async def get_adjusted_prediction(user_id: str, base_prediction: float, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/outcome/adjusted-prediction")

    if not app_state.outcome_learning:
        return safe_json_response({"adjusted_prediction": base_prediction, "adjustment": 0})

    adjusted = app_state.outcome_learning.get_adjusted_prediction(user_id, base_prediction)
    return safe_json_response({
        "base_prediction": base_prediction,
        "adjusted_prediction": adjusted,
        "adjustment": adjusted - base_prediction
    })

@app.get("/api/templates")
async def get_all_templates():
    app_state.monitoring.record("/api/templates")

    if not app_state.decision_templates:
        raise HTTPException(503, "Decision template service not available")

    templates = app_state.decision_templates.get_all_templates()
    return safe_json_response({"templates": templates})

@app.get("/api/templates/{template_id}")
async def get_template(template_id: str):
    app_state.monitoring.record("/api/templates/detail")

    if not app_state.decision_templates:
        raise HTTPException(503, "Decision template service not available")

    template = app_state.decision_templates.get_template(template_id)
    if not template:
        raise HTTPException(404, "Template not found")

    return safe_json_response(template)

class TemplateAnalysisInput(BaseModel):
    template_id: str = Field(..., max_length=50)
    answers: Dict[str, Any]
    user_id: Optional[str] = Field(default=None, max_length=100)

@app.post("/api/templates/analyze")
async def analyze_template(input_data: TemplateAnalysisInput):
    app_state.monitoring.record("/api/templates/analyze")

    if not app_state.decision_templates:
        raise HTTPException(503, "Decision template service not available")

    result = app_state.decision_templates.analyze_template(
        template_id=input_data.template_id,
        answers=input_data.answers,
        user_id=input_data.user_id
    )

    if "error" in result:
        raise HTTPException(404, result["error"])

    return safe_json_response(result)

@app.get("/api/templates/history/{user_id}")
async def get_template_history(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/templates/history")

    if not app_state.decision_templates:
        raise HTTPException(503, "Decision template service not available")

    history = app_state.decision_templates.get_user_history(user_id)
    return safe_json_response({"history": history})

class FutureSelfStartInput(BaseModel):
    decision_type: str = Field(..., max_length=50)
    decision_description: str = Field(..., max_length=2000)
    timeframe: str = Field(default="5_years")
    scenario: str = Field(default="realistic")
    user_id: Optional[str] = Field(default=None, max_length=100)

@app.post("/api/future-self/start")
async def start_future_self_conversation(input_data: FutureSelfStartInput):
    app_state.monitoring.record("/api/future-self/start")

    if not app_state.future_self:
        raise HTTPException(503, "Future Self service not available")

    result = app_state.future_self.start_conversation(
        user_id=input_data.user_id or "anonymous",
        decision_type=input_data.decision_type,
        decision_desc=input_data.decision_description,
        timeframe=input_data.timeframe,
        scenario=input_data.scenario
    )
    return safe_json_response(result)

class FutureSelfMessageInput(BaseModel):
    session_id: str = Field(..., max_length=100)
    message: str = Field(..., max_length=2000)

@app.post("/api/future-self/message")
async def send_future_self_message(input_data: FutureSelfMessageInput):
    app_state.monitoring.record("/api/future-self/message")

    if not app_state.future_self:
        raise HTTPException(503, "Future Self service not available")

    result = app_state.future_self.send_message(
        session_id=input_data.session_id,
        message=input_data.message
    )

    if "error" in result:
        raise HTTPException(404, result["error"])

    return safe_json_response(result)

@app.post("/api/future-self/end/{session_id}")
async def end_future_self_conversation(session_id: str):
    app_state.monitoring.record("/api/future-self/end")

    if not app_state.future_self:
        raise HTTPException(503, "Future Self service not available")

    result = app_state.future_self.end_conversation(session_id)

    if "error" in result:
        raise HTTPException(404, result["error"])

    return safe_json_response(result)

@app.get("/api/future-self/session/{session_id}")
async def get_future_self_session(session_id: str):
    app_state.monitoring.record("/api/future-self/session")

    if not app_state.future_self:
        raise HTTPException(503, "Future Self service not available")

    result = app_state.future_self.get_session(session_id)

    if not result:
        raise HTTPException(404, "Session not found")

    return safe_json_response(result)


class ScoutProfileInput(BaseModel):
    current_role: str = Field(default="Software Engineer", max_length=100)
    industry: str = Field(default="technology", max_length=50)
    skills: List[str] = Field(default=[])
    interests: List[str] = Field(default=[])
    risk_tolerance: float = Field(default=0.5, ge=0, le=1)
    salary_target: float = Field(default=150000)
    locations: List[str] = Field(default=["Remote"])
    career_goals: List[str] = Field(default=[])
    user_id: Optional[str] = Field(default=None, max_length=100)

@app.post("/api/scout/register")
async def register_scout_profile(input_data: ScoutProfileInput):
    app_state.monitoring.record("/api/scout/register")

    if not app_state.opportunity_scout:
        raise HTTPException(503, "Opportunity Scout service not available")

    user_id = input_data.user_id or "anonymous"
    profile = app_state.opportunity_scout.register_user_profile(user_id, input_data.model_dump())

    if app_state.persistence:
        app_state.persistence.save_scout_profile(user_id, input_data.model_dump())

    return safe_json_response({"success": True, "user_id": user_id})

@app.post("/api/scout/scan/{user_id}")
async def scan_opportunities(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/scout/scan")

    if not app_state.opportunity_scout:
        raise HTTPException(503, "Opportunity Scout service not available")

    opportunities = await app_state.opportunity_scout.scan_opportunities(user_id)

    return safe_json_response({
        "opportunities": [app_state.opportunity_scout._opp_to_dict(o) for o in opportunities],
        "count": len(opportunities)
    })

@app.get("/api/scout/opportunities/{user_id}")
async def get_opportunities(user_id: str, type_filter: Optional[str] = None, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/scout/opportunities")

    if not app_state.opportunity_scout:
        raise HTTPException(503, "Opportunity Scout service not available")

    opportunities = app_state.opportunity_scout.get_opportunities(user_id, type_filter)
    return safe_json_response({"opportunities": opportunities})

@app.get("/api/scout/alerts/{user_id}")
async def get_scout_alerts(user_id: str, unread_only: bool = True, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/scout/alerts")

    if not app_state.opportunity_scout:
        raise HTTPException(503, "Opportunity Scout service not available")

    alerts = app_state.opportunity_scout.get_alerts(user_id, unread_only)
    return safe_json_response({"alerts": alerts})

@app.get("/api/scout/summary/{user_id}")
async def get_scout_summary(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/scout/summary")

    if not app_state.opportunity_scout:
        raise HTTPException(503, "Opportunity Scout service not available")

    summary = app_state.opportunity_scout.get_scout_summary(user_id)
    return safe_json_response(summary)

@app.get("/api/scout/saved/{user_id}")
async def get_saved_opportunities(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/scout/saved")
    if not app_state.opportunity_scout:
        raise HTTPException(503, "Opportunity Scout service not available")
    return safe_json_response({"opportunities": app_state.opportunity_scout.get_saved_opportunities(user_id)})

@app.post("/api/scout/apply/{user_id}/{opp_id}")
async def apply_opportunity(user_id: str, opp_id: str, notes: Optional[str] = None, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/scout/apply")
    if not app_state.opportunity_scout:
        raise HTTPException(503, "Opportunity Scout service not available")
    result = app_state.opportunity_scout.apply_for_opportunity(user_id, opp_id, notes or "")
    return safe_json_response(result)

@app.get("/api/scout/applications/{user_id}")
async def get_applications(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/scout/applications")
    if not app_state.opportunity_scout:
        raise HTTPException(503, "Opportunity Scout service not available")
    return safe_json_response({"applications": app_state.opportunity_scout.get_applications(user_id)})

@app.post("/api/scout/opportunity/{user_id}/{opp_id}/{action}")
async def mark_opportunity(user_id: str, opp_id: str, action: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/scout/opportunity/action")

    if not app_state.opportunity_scout:
        raise HTTPException(503, "Opportunity Scout service not available")

    if action not in ["save", "dismiss", "view"]:
        raise HTTPException(400, "Invalid action. Use: save, dismiss, view")

    success = app_state.opportunity_scout.mark_opportunity(user_id, opp_id, action)
    return safe_json_response({"success": success})


class BiasAnalysisInput(BaseModel):
    text: str = Field(..., max_length=5000)
    user_id: Optional[str] = Field(default=None, max_length=100)

@app.post("/api/bias/analyze")
async def analyze_bias(input_data: BiasAnalysisInput):
    app_state.monitoring.record("/api/bias/analyze")

    if not app_state.bias_interceptor:
        raise HTTPException(503, "Bias Interceptor service not available")

    detections = app_state.bias_interceptor.analyze_text(
        input_data.text,
        input_data.user_id
    )
    return safe_json_response({"detections": detections})

@app.post("/api/bias/realtime")
async def get_realtime_bias_feedback(input_data: BiasAnalysisInput):
    app_state.monitoring.record("/api/bias/realtime")

    if not app_state.bias_interceptor:
        raise HTTPException(503, "Bias Interceptor service not available")

    feedback = app_state.bias_interceptor.get_real_time_feedback(
        input_data.text,
        input_data.user_id
    )
    return safe_json_response(feedback)

@app.get("/api/bias/profile/{user_id}")
async def get_bias_profile(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/bias/profile")

    if not app_state.bias_interceptor:
        raise HTTPException(503, "Bias Interceptor service not available")

    profile = app_state.bias_interceptor.get_user_bias_profile(user_id)
    return safe_json_response(profile)

@app.get("/api/bias/explain/{bias_type}")
async def explain_bias(bias_type: str):
    app_state.monitoring.record("/api/bias/explain")

    if not app_state.bias_interceptor:
        raise HTTPException(503, "Bias Interceptor service not available")

    explanation = app_state.bias_interceptor.get_bias_explanation(bias_type)
    return safe_json_response(explanation)

class GlobalOutcomeInput(BaseModel):
    decision_type: str = Field(..., max_length=50)
    industry: str = Field(default="technology", max_length=50)
    years_experience: int = Field(default=5, ge=0, le=50)
    predicted_regret: float = Field(..., ge=0, le=100)
    actual_regret: float = Field(..., ge=0, le=100)
    satisfaction_score: float = Field(..., ge=0, le=100)
    factors: List[str] = Field(default=[])
    user_id: Optional[str] = Field(default=None, max_length=100)

@app.post("/api/global-regret/contribute")
async def contribute_to_global_db(input_data: GlobalOutcomeInput):
    app_state.monitoring.record("/api/global-regret/contribute")

    if not app_state.global_regret_db:
        raise HTTPException(503, "Global Regret Database not available")

    result = app_state.global_regret_db.contribute_outcome(
        user_id=input_data.user_id or "anonymous",
        decision_type=input_data.decision_type,
        industry=input_data.industry,
        years_experience=input_data.years_experience,
        predicted_regret=input_data.predicted_regret,
        actual_regret=input_data.actual_regret,
        satisfaction_score=input_data.satisfaction_score,
        decision_date=datetime.utcnow(),
        factors=input_data.factors
    )
    return safe_json_response(result)

@app.get("/api/global-regret/insights/{decision_type}")
async def get_global_insights(decision_type: str):
    app_state.monitoring.record("/api/global-regret/insights")

    if not app_state.global_regret_db:
        raise HTTPException(503, "Global Regret Database not available")

    insights = app_state.global_regret_db.get_global_insights(decision_type)
    return safe_json_response(insights)

@app.get("/api/global-regret/adjusted-prediction")
async def get_globally_adjusted_prediction(
    base_prediction: float,
    decision_type: str,
    industry: str = "technology",
    years_experience: int = 5
):
    app_state.monitoring.record("/api/global-regret/adjusted-prediction")

    if not app_state.global_regret_db:
        return safe_json_response({"adjusted_prediction": base_prediction})

    result = app_state.global_regret_db.get_adjusted_prediction(
        base_prediction, decision_type, industry, years_experience
    )
    return safe_json_response(result)

@app.get("/api/global-regret/compare")
async def compare_decision_types():
    app_state.monitoring.record("/api/global-regret/compare")

    if not app_state.global_regret_db:
        raise HTTPException(503, "Global Regret Database not available")

    comparisons = app_state.global_regret_db.compare_decision_types()
    return safe_json_response({"comparisons": comparisons})

@app.get("/api/global-regret/stats")
async def get_global_stats():
    app_state.monitoring.record("/api/global-regret/stats")

    if not app_state.global_regret_db:
        raise HTTPException(503, "Global Regret Database not available")

    stats = app_state.global_regret_db.get_database_stats()
    return safe_json_response(stats)

class MultiverseInput(BaseModel):
    decision_type: str = Field(..., max_length=50)
    description: str = Field(..., max_length=2000)
    user_id: Optional[str] = Field(default=None, max_length=100)

@app.post("/api/multiverse/generate")
async def generate_multiverse(input_data: MultiverseInput):
    app_state.monitoring.record("/api/multiverse/generate")

    if not app_state.multiverse_viz:
        raise HTTPException(503, "Multiverse Visualization service not available")

    user_id = input_data.user_id or "anonymous"
    forest_data = app_state.multiverse_viz.generate_decision_forest(
        user_id=user_id,
        current_decision={
            "decision_type": input_data.decision_type,
            "description": input_data.description
        }
    )
    return safe_json_response(forest_data)

@app.get("/api/multiverse/timeline/{user_id}/{timeline_id}")
async def get_timeline_details(user_id: str, timeline_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/multiverse/timeline")

    if not app_state.multiverse_viz:
        raise HTTPException(503, "Multiverse Visualization service not available")

    details = app_state.multiverse_viz.get_timeline_details(user_id, timeline_id)

    if not details:
        raise HTTPException(404, "Timeline not found")

    return safe_json_response(details)

@app.get("/api/persistence/profile/{user_id}")
async def get_persisted_profile(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/persistence/profile")

    if not app_state.persistence:
        raise HTTPException(503, "Persistence service not available")

    profile = app_state.persistence.get_learning_profile(user_id)
    if not profile:
        return safe_json_response({"user_id": user_id, "exists": False})

    return safe_json_response({**profile, "exists": True})

@app.get("/api/persistence/outcomes/{user_id}")
async def get_persisted_outcomes(user_id: str, current_user: str = Depends(get_current_user), limit: int = 50):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/persistence/outcomes")

    if not app_state.persistence:
        raise HTTPException(503, "Persistence service not available")

    outcomes = app_state.persistence.get_user_outcomes(user_id, limit)
    return safe_json_response({"outcomes": outcomes, "count": len(outcomes)})

@app.get("/api/persistence/bias-stats/{user_id}")
async def get_persisted_bias_stats(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/persistence/bias-stats")

    if not app_state.persistence:
        raise HTTPException(503, "Persistence service not available")

    stats = app_state.persistence.get_user_bias_stats(user_id)
    return safe_json_response(stats)


from services.websocket_service import connection_manager, realtime_service, collaboration_service

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    client = await connection_manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "typing":
                if app_state.bias_interceptor:
                    await realtime_service.process_typing_input(
                        user_id, data.get("text", ""), app_state.bias_interceptor
                    )

            elif data.get("type") == "heartbeat":
                await websocket.send_json({"type": "heartbeat", "status": "ok"})

            elif data.get("type") == "subscribe":
                client.subscriptions.add(data.get("channel", ""))

    except WebSocketDisconnect:
        connection_manager.disconnect(websocket, user_id)

@app.get("/api/ws/status")
async def get_websocket_status():
    return safe_json_response({
        "active_connections": connection_manager.get_connection_count(),
        "active_rooms": len(connection_manager.collaboration_rooms)
    })


class CollaborationCreateInput(BaseModel):
    decision_id: str = Field(..., max_length=100)
    decision_data: Dict[str, Any] = Field(default={})
    user_id: Optional[str] = Field(default=None, max_length=100)

@app.post("/api/collaboration/create")
async def create_shared_decision(input_data: CollaborationCreateInput):
    app_state.monitoring.record("/api/collaboration/create")

    user_id = input_data.user_id or "anonymous"
    room_id = collaboration_service.create_shared_decision(
        input_data.decision_id, user_id, input_data.decision_data
    )

    return safe_json_response({"room_id": room_id, "decision_id": input_data.decision_id})

class CollaborationInviteInput(BaseModel):
    decision_id: str = Field(..., max_length=100)
    inviter_id: str = Field(..., max_length=100)
    invitee_id: str = Field(..., max_length=100)

@app.post("/api/collaboration/invite")
async def invite_to_decision(input_data: CollaborationInviteInput):
    app_state.monitoring.record("/api/collaboration/invite")

    result = await collaboration_service.invite_collaborator(
        input_data.decision_id, input_data.inviter_id, input_data.invitee_id
    )

    return safe_json_response(result)

class CollaborationCommentInput(BaseModel):
    decision_id: str = Field(..., max_length=100)
    user_id: str = Field(..., max_length=100)
    comment: str = Field(..., max_length=2000)

@app.post("/api/collaboration/comment")
async def add_comment(input_data: CollaborationCommentInput):
    app_state.monitoring.record("/api/collaboration/comment")

    comment = await collaboration_service.add_comment(
        input_data.decision_id, input_data.user_id, input_data.comment
    )

    return safe_json_response(comment)

class CollaborationVoteInput(BaseModel):
    decision_id: str = Field(..., max_length=100)
    user_id: str = Field(..., max_length=100)
    option: str = Field(..., max_length=100)
    vote: str = Field(..., pattern="^(support|oppose|neutral)$")

@app.post("/api/collaboration/vote")
async def vote_on_option(input_data: CollaborationVoteInput):
    app_state.monitoring.record("/api/collaboration/vote")

    result = await collaboration_service.vote_on_option(
        input_data.decision_id, input_data.user_id, input_data.option, input_data.vote
    )

    return safe_json_response(result)

@app.get("/api/collaboration/{decision_id}")
async def get_collaboration_details(decision_id: str):
    app_state.monitoring.record("/api/collaboration/details")

    details = collaboration_service.get_decision_details(decision_id)

    if not details:
        raise HTTPException(404, "Decision not found")

    return safe_json_response(details)


from services.voice_speech_service import voice_speech_service, voice_journal_service, future_self_voice_service

@app.post("/api/voice/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    app_state.monitoring.record("/api/voice/transcribe")

    audio_data = await file.read()
    audio_format = file.filename.split(".")[-1] if file.filename else "webm"

    result = await voice_speech_service.transcribe_audio(audio_data, audio_format)

    return safe_json_response({
        "text": result.text,
        "confidence": result.confidence,
        "duration": result.duration_seconds,
        "provider": result.provider.value
    })

class TTSInput(BaseModel):
    text: str = Field(..., max_length=5000)
    persona: str = Field(default="future_self_calm")

@app.post("/api/voice/synthesize")
async def synthesize_speech(input_data: TTSInput):
    app_state.monitoring.record("/api/voice/synthesize")

    result = await voice_speech_service.text_to_speech(input_data.text, input_data.persona)

    if result.audio_data:
        import base64
        audio_base64 = base64.b64encode(result.audio_data).decode()
        return safe_json_response({
            "audio": audio_base64,
            "format": result.format,
            "duration": result.duration_seconds
        })

    return safe_json_response({"error": "TTS not available", "use_browser": True})

@app.get("/api/voice/personas")
async def get_voice_personas():
    app_state.monitoring.record("/api/voice/personas")

    personas = voice_speech_service.get_available_personas()
    return safe_json_response({"personas": personas})

@app.get("/api/voice/status")
async def get_voice_status():
    app_state.monitoring.record("/api/voice/status")

    status = voice_speech_service.get_service_status()
    return safe_json_response(status)

@app.post("/api/voice/journal/{user_id}")
async def create_voice_journal_entry(user_id: str, file: UploadFile = File(...), current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/voice/journal")

    audio_data = await file.read()
    audio_format = file.filename.split(".")[-1] if file.filename else "webm"

    entry = await voice_journal_service.process_voice_entry(user_id, audio_data, audio_format)

    return safe_json_response(entry)

@app.get("/api/voice/journal/{user_id}")
async def get_voice_journal_entries(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/voice/journal/list")

    entries = voice_journal_service.get_voice_entries(user_id)
    return safe_json_response({"entries": entries})


from services.external_integration_service import external_integration_service, IntegrationType

@app.get("/api/integrations/status/{user_id}")
async def get_integration_status(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/integrations/status")

    status = external_integration_service.get_integration_status(user_id)
    return safe_json_response(status)

@app.get("/api/integrations/oauth/{integration_type}/{user_id}")
async def get_oauth_url(integration_type: str, user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/integrations/oauth")

    try:
        it = IntegrationType(integration_type)
    except ValueError:
        raise HTTPException(400, f"Invalid integration type: {integration_type}")

    url = external_integration_service.get_oauth_url(it, user_id)
    return safe_json_response({"oauth_url": url})

@app.post("/api/integrations/callback/{integration_type}/{user_id}")
async def handle_oauth_callback(integration_type: str, user_id: str, code: str):
    app_state.monitoring.record("/api/integrations/callback")

    try:
        it = IntegrationType(integration_type)
    except ValueError:
        raise HTTPException(400, f"Invalid integration type: {integration_type}")

    result = await external_integration_service.handle_oauth_callback(it, user_id, code)
    return safe_json_response(result)

@app.get("/api/integrations/linkedin/{user_id}")
async def get_linkedin_profile(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/integrations/linkedin")

    profile = await external_integration_service.fetch_linkedin_profile(user_id)

    if not profile:
        raise HTTPException(404, "Profile not found")

    return safe_json_response({
        "name": profile.name,
        "headline": profile.headline,
        "current_role": profile.current_role,
        "industry": profile.industry,
        "skills": profile.skills,
        "experience_years": profile.experience_years
    })

@app.get("/api/integrations/jobs/search")
async def search_jobs(
    query: str = "",
    location: str = "",
    salary_min: int = 0,
    limit: int = 10,
    user_id: str = "anonymous"
):
    app_state.monitoring.record("/api/integrations/jobs/search")

    jobs = await external_integration_service.search_jobs(user_id, query, location, salary_min, limit)

    return safe_json_response({
        "jobs": [
            {
                "id": job.id,
                "title": job.title,
                "company": job.company,
                "location": job.location,
                "salary_range": job.salary_range,
                "match_score": job.match_score,
                "posted_date": job.posted_date.isoformat()
            }
            for job in jobs
        ]
    })

@app.get("/api/integrations/news")
async def get_industry_news(
    industry: str = "technology",
    limit: int = 10
):
    app_state.monitoring.record("/api/integrations/news")

    articles = await external_integration_service.fetch_industry_news(industry, limit=limit)

    return safe_json_response({
        "articles": [
            {
                "id": article.id,
                "title": article.title,
                "summary": article.summary,
                "source": article.source,
                "category": article.category,
                "published_at": article.published_at.isoformat()
            }
            for article in articles
        ]
    })

@app.delete("/api/integrations/{integration_type}/{user_id}")
async def disconnect_integration(integration_type: str, user_id: str):
    app_state.monitoring.record("/api/integrations/disconnect")

    try:
        it = IntegrationType(integration_type)
    except ValueError:
        raise HTTPException(400, f"Invalid integration type: {integration_type}")

    success = external_integration_service.disconnect_integration(user_id, it)
    return safe_json_response({"success": success})


from services.advanced_analytics_service import advanced_analytics_service

@app.get("/api/analytics/advanced/{user_id}")
async def get_advanced_analytics(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/analytics/advanced")

    dashboard = advanced_analytics_service.get_analytics_dashboard(user_id)
    return safe_json_response(dashboard)

@app.get("/api/analytics/predictions/{user_id}")
async def get_prediction_accuracy(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/analytics/predictions")

    metrics = advanced_analytics_service.get_prediction_accuracy(user_id)
    return safe_json_response({
        "total_predictions": metrics.total_predictions,
        "accuracy_percentage": metrics.accuracy_percentage,
        "average_error": metrics.average_error,
        "error_trend": metrics.error_trend,
        "by_decision_type": metrics.by_decision_type
    })

@app.get("/api/analytics/bias-patterns/{user_id}")
async def get_bias_patterns(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/analytics/bias-patterns")

    analysis = advanced_analytics_service.get_bias_pattern_analysis(user_id)
    return safe_json_response({
        "most_common_biases": analysis.most_common_biases,
        "frequency_over_time": analysis.bias_frequency_over_time,
        "improvement_rate": analysis.improvement_rate,
        "recommendations": analysis.recommendations
    })

@app.get("/api/analytics/timeline/{user_id}")
async def get_decision_timeline(user_id: str, limit: int = 50, decision_type: Optional[str] = None, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/analytics/timeline")

    timeline = advanced_analytics_service.get_decision_timeline(user_id, limit, decision_type)
    return safe_json_response({"timeline": timeline})

class GoalInput(BaseModel):
    title: str = Field(..., max_length=200)
    description: str = Field(..., max_length=2000)
    category: str = Field(..., max_length=50)
    target_date: Optional[str] = None
    milestones: List[Dict] = Field(default=[])
    user_id: Optional[str] = Field(default=None, max_length=100)

@app.post("/api/analytics/goals")
async def create_goal(input_data: GoalInput):
    app_state.monitoring.record("/api/analytics/goals/create")

    target_date = None
    if input_data.target_date:
        target_date = datetime.fromisoformat(input_data.target_date)

    goal = advanced_analytics_service.create_career_goal(
        user_id=input_data.user_id or "anonymous",
        title=input_data.title,
        description=input_data.description,
        category=input_data.category,
        target_date=target_date,
        milestones=input_data.milestones
    )

    return safe_json_response({"goal_id": goal.id, "created": True})

@app.get("/api/analytics/goals/{user_id}")
async def get_goals(user_id: str, category: Optional[str] = None, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/analytics/goals")

    goals = advanced_analytics_service.get_user_goals(user_id, category)
    return safe_json_response({"goals": goals})

class GoalProgressInput(BaseModel):
    progress: Optional[float] = None
    milestone_completed: Optional[str] = None

@app.put("/api/analytics/goals/{user_id}/{goal_id}")
async def update_goal_progress(user_id: str, goal_id: str, input_data: GoalProgressInput):
    app_state.monitoring.record("/api/analytics/goals/update")

    result = advanced_analytics_service.update_goal_progress(
        user_id, goal_id, input_data.progress, input_data.milestone_completed
    )

    if not result:
        raise HTTPException(404, "Goal not found")

    return safe_json_response({
        "goal_id": result.goal_id,
        "progress": result.current_progress,
        "on_track": result.on_track,
        "next_action": result.next_action
    })

@app.get("/api/analytics/export/{user_id}")
async def export_analytics_data(user_id: str):
    app_state.monitoring.record("/api/analytics/export")

    data = advanced_analytics_service.export_analytics_data(user_id)
    return safe_json_response(data)


from services.data_privacy_service import data_privacy_service, ConsentType, DataCategory

@app.get("/api/privacy/dashboard/{user_id}")
async def get_privacy_dashboard(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/privacy/dashboard")

    dashboard = data_privacy_service.get_privacy_dashboard(user_id)
    return safe_json_response(dashboard)

class ConsentInput(BaseModel):
    consent_type: str = Field(...)
    granted: bool = Field(...)
    user_id: Optional[str] = Field(default=None, max_length=100)

@app.post("/api/privacy/consent")
async def update_consent(input_data: ConsentInput, request: Request):
    app_state.monitoring.record("/api/privacy/consent")

    try:
        ct = ConsentType(input_data.consent_type)
    except ValueError:
        raise HTTPException(400, f"Invalid consent type: {input_data.consent_type}")

    ip_address = request.client.host if request.client else ""

    consent = data_privacy_service.record_consent(
        user_id=input_data.user_id or "anonymous",
        consent_type=ct,
        granted=input_data.granted,
        ip_address=ip_address
    )

    return safe_json_response({
        "consent_type": consent.consent_type.value,
        "granted": consent.granted,
        "timestamp": consent.granted_at.isoformat() if consent.granted_at else None
    })

@app.get("/api/privacy/consents/{user_id}")
async def get_consents(user_id: str):
    app_state.monitoring.record("/api/privacy/consents")

    consents = data_privacy_service.get_user_consents(user_id)
    return safe_json_response({"consents": consents})

class ExportRequestInput(BaseModel):
    categories: List[str] = Field(default=[])
    user_id: Optional[str] = Field(default=None, max_length=100)

@app.post("/api/privacy/export")
async def request_data_export(input_data: ExportRequestInput):
    app_state.monitoring.record("/api/privacy/export")

    categories = []
    for cat in input_data.categories:
        try:
            categories.append(DataCategory(cat))
        except ValueError:
            pass

    if not categories:
        categories = list(DataCategory)

    request = data_privacy_service.request_data_export(
        user_id=input_data.user_id or "anonymous",
        categories=categories
    )

    return safe_json_response({
        "request_id": request.id,
        "status": request.status,
        "requested_at": request.requested_at.isoformat()
    })

class DeletionRequestInput(BaseModel):
    reason: Optional[str] = Field(default=None, max_length=1000)
    user_id: Optional[str] = Field(default=None, max_length=100)

@app.post("/api/privacy/delete")
async def request_account_deletion(input_data: DeletionRequestInput):
    app_state.monitoring.record("/api/privacy/delete")

    request = data_privacy_service.request_account_deletion(
        user_id=input_data.user_id or "anonymous",
        reason=input_data.reason
    )

    return safe_json_response({
        "request_id": request.id,
        "status": request.status,
        "scheduled_at": request.scheduled_at.isoformat(),
        "message": "Your account is scheduled for deletion in 30 days. You can cancel this request anytime before then."
    })

@app.delete("/api/privacy/delete/{request_id}/{user_id}")
async def cancel_deletion_request(request_id: str, user_id: str):
    app_state.monitoring.record("/api/privacy/delete/cancel")

    success = data_privacy_service.cancel_deletion_request(request_id, user_id)

    if not success:
        raise HTTPException(404, "Deletion request not found or already processed")

    return safe_json_response({"cancelled": True})

@app.get("/api/privacy/access-log/{user_id}")
async def get_access_log(user_id: str, current_user: str = Depends(get_current_user), limit: int = 100):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/privacy/access-log")

    logs = data_privacy_service.get_access_log(user_id, limit)
    return safe_json_response({"logs": logs})

from services.push_notification_service import push_notification_service, NotificationType, NotificationPriority

class PushSubscriptionInput(BaseModel):
    endpoint: str = Field(..., max_length=500)
    p256dh_key: str = Field(..., max_length=200)
    auth_key: str = Field(..., max_length=100)
    preferences: Optional[Dict[str, bool]] = None
    user_id: Optional[str] = Field(default=None, max_length=100)

@app.post("/api/notifications/subscribe")
async def subscribe_push_notifications(input_data: PushSubscriptionInput):
    app_state.monitoring.record("/api/notifications/subscribe")

    result = push_notification_service.subscribe(
        user_id=input_data.user_id or "anonymous",
        endpoint=input_data.endpoint,
        p256dh_key=input_data.p256dh_key,
        auth_key=input_data.auth_key,
        preferences=input_data.preferences
    )
    return safe_json_response(result)

@app.delete("/api/notifications/unsubscribe/{user_id}")
async def unsubscribe_push_notifications(user_id: str):
    app_state.monitoring.record("/api/notifications/unsubscribe")

    result = push_notification_service.unsubscribe(user_id)
    return safe_json_response(result)

@app.get("/api/notifications/preferences/{user_id}")
async def get_notification_preferences(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/notifications/preferences")

    result = push_notification_service.get_preferences(user_id)
    return safe_json_response(result)

@app.get("/api/notifications/{user_id}")
async def get_notifications(user_id: str, current_user: str = Depends(get_current_user), unread_only: bool = False, limit: int = 20):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/notifications")

    notifications = push_notification_service.get_notifications(user_id, unread_only, limit)
    return safe_json_response({
        "notifications": notifications,
        "unread_count": push_notification_service.get_unread_count(user_id)
    })

@app.post("/api/notifications/read/{user_id}/{notification_id}")
async def mark_notification_read(user_id: str, notification_id: str):
    app_state.monitoring.record("/api/notifications/read")

    success = push_notification_service.mark_as_read(user_id, notification_id)
    return safe_json_response({"marked_read": success})

@app.post("/api/notifications/read-all/{user_id}")
async def mark_all_notifications_read(user_id: str):
    app_state.monitoring.record("/api/notifications/read-all")

    count = push_notification_service.mark_all_read(user_id)
    return safe_json_response({"marked_read": count})

from services.scheduled_checkin_service import scheduled_checkin_service, CheckInType, CheckInFrequency

class CheckInInput(BaseModel):
    check_in_type: str = Field(default="weekly_reflection", max_length=50)
    title: str = Field(..., max_length=200)
    description: Optional[str] = Field(default="", max_length=500)
    frequency: str = Field(default="weekly", max_length=20)
    related_decision_id: Optional[str] = None
    related_goal_id: Optional[str] = None
    user_id: Optional[str] = Field(default=None, max_length=100)

@app.post("/api/checkins/create")
async def create_check_in(input_data: CheckInInput):
    app_state.monitoring.record("/api/checkins/create")

    try:
        check_in_type = CheckInType(input_data.check_in_type)
    except ValueError:
        check_in_type = CheckInType.WEEKLY_REFLECTION

    try:
        frequency = CheckInFrequency(input_data.frequency)
    except ValueError:
        frequency = CheckInFrequency.WEEKLY

    check_in = scheduled_checkin_service.create_check_in(
        user_id=input_data.user_id or "anonymous",
        check_in_type=check_in_type,
        title=input_data.title,
        description=input_data.description,
        frequency=frequency,
        related_decision_id=input_data.related_decision_id,
        related_goal_id=input_data.related_goal_id
    )

    return safe_json_response({
        "check_in_id": check_in.id,
        "title": check_in.title,
        "frequency": check_in.frequency.value,
        "next_due": check_in.next_due.isoformat()
    })

@app.get("/api/checkins/due/{user_id}")
async def get_due_check_ins(user_id: str):
    app_state.monitoring.record("/api/checkins/due")

    due = scheduled_checkin_service.get_due_check_ins(user_id)
    return safe_json_response({"due_check_ins": due})

@app.get("/api/checkins/{user_id}")
async def get_all_check_ins(user_id: str):
    app_state.monitoring.record("/api/checkins")

    check_ins = scheduled_checkin_service.get_all_check_ins(user_id)
    return safe_json_response({"check_ins": check_ins})

class CheckInResponseInput(BaseModel):
    responses: Dict[str, Any] = Field(default_factory=dict)
    mood_score: int = Field(default=5, ge=1, le=10)
    user_id: Optional[str] = Field(default=None, max_length=100)

@app.post("/api/checkins/complete/{check_in_id}")
async def complete_check_in(check_in_id: str, input_data: CheckInResponseInput):
    app_state.monitoring.record("/api/checkins/complete")

    result = scheduled_checkin_service.complete_check_in(
        user_id=input_data.user_id or "anonymous",
        check_in_id=check_in_id,
        responses=input_data.responses,
        mood_score=input_data.mood_score
    )
    return safe_json_response(result)

@app.get("/api/checkins/stats/{user_id}")
async def get_check_in_stats(user_id: str):
    app_state.monitoring.record("/api/checkins/stats")

    stats = scheduled_checkin_service.get_check_in_stats(user_id)
    return safe_json_response(stats)

@app.post("/api/checkins/setup-defaults/{user_id}")
async def setup_default_check_ins(user_id: str):
    app_state.monitoring.record("/api/checkins/setup-defaults")

    created = scheduled_checkin_service.setup_default_check_ins(user_id)
    return safe_json_response({"created": created})

from services.resume_parser_service import resume_parser_service

class ResumeTextInput(BaseModel):
    text_content: str = Field(..., max_length=50000)
    filename: Optional[str] = Field(default="", max_length=200)
    user_id: Optional[str] = Field(default=None, max_length=100)

@app.post("/api/resume/parse")
async def parse_resume(input_data: ResumeTextInput):
    app_state.monitoring.record("/api/resume/parse")

    result = resume_parser_service.parse_resume(
        user_id=input_data.user_id or "anonymous",
        text_content=input_data.text_content,
        filename=input_data.filename
    )
    return safe_json_response(result)

@app.get("/api/resume/{resume_id}")
async def get_parsed_resume(resume_id: str):
    app_state.monitoring.record("/api/resume/get")

    result = resume_parser_service.get_resume(resume_id)
    if not result:
        raise HTTPException(404, "Resume not found")
    return safe_json_response(result)

@app.get("/api/resume/{resume_id}/skill-gaps")
async def get_skill_gaps(resume_id: str, target_role: str = "software engineer"):
    app_state.monitoring.record("/api/resume/skill-gaps")

    result = resume_parser_service.get_skill_gaps(resume_id, target_role)
    return safe_json_response(result)

from services.proactive_suggestion_service import proactive_suggestion_service

class UserContextInput(BaseModel):
    current_role: Optional[str] = Field(default="", max_length=100)
    target_role: Optional[str] = Field(default="", max_length=100)
    skills: Optional[List[str]] = Field(default_factory=list)
    goals: Optional[List[str]] = Field(default_factory=list)
    recent_decisions: Optional[int] = 0
    pending_decisions: Optional[int] = 0
    bias_patterns: Optional[List[str]] = Field(default_factory=list)
    user_id: Optional[str] = Field(default=None, max_length=100)

@app.post("/api/suggestions/context")
async def update_user_context(input_data: UserContextInput):
    app_state.monitoring.record("/api/suggestions/context")

    context = proactive_suggestion_service.update_user_context(
        user_id=input_data.user_id or "anonymous",
        current_role=input_data.current_role,
        target_role=input_data.target_role,
        skills=input_data.skills,
        goals=input_data.goals,
        recent_decisions=input_data.recent_decisions,
        pending_decisions=input_data.pending_decisions,
        bias_patterns=input_data.bias_patterns
    )

    return safe_json_response({"context_updated": True})

@app.get("/api/suggestions/{user_id}")
async def get_suggestions(user_id: str, current_user: str = Depends(get_current_user), max_suggestions: int = 3):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/suggestions")

    suggestions = proactive_suggestion_service.get_active_suggestions(user_id)
    return safe_json_response({"suggestions": suggestions})

@app.post("/api/suggestions/dismiss/{user_id}/{suggestion_id}")
async def dismiss_suggestion(user_id: str, suggestion_id: str, feedback: str = None):
    app_state.monitoring.record("/api/suggestions/dismiss")

    success = proactive_suggestion_service.dismiss_suggestion(user_id, suggestion_id, feedback)
    return safe_json_response({"dismissed": success})

@app.post("/api/suggestions/act/{user_id}/{suggestion_id}")
async def act_on_suggestion(user_id: str, suggestion_id: str, action: str = "default"):
    app_state.monitoring.record("/api/suggestions/act")

    result = proactive_suggestion_service.act_on_suggestion(user_id, suggestion_id, action)
    return safe_json_response(result)

@app.get("/api/suggestions/stats/{user_id}")
async def get_suggestion_stats(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/suggestions/stats")

    stats = proactive_suggestion_service.get_suggestion_stats(user_id)
    return safe_json_response(stats)

from services.monitoring_dashboard_service import monitoring_dashboard_service

@app.get("/api/monitoring/health")
async def get_health_status():
    health = monitoring_dashboard_service.check_health()
    return safe_json_response(health)

@app.get("/api/monitoring/metrics/system")
async def get_system_metrics():
    monitoring_dashboard_service.record_request("/api/monitoring/metrics/system", "GET", 0, 200)

    metrics = monitoring_dashboard_service.get_system_metrics()
    return safe_json_response(metrics)

@app.get("/api/monitoring/metrics/application")
async def get_application_metrics():
    monitoring_dashboard_service.record_request("/api/monitoring/metrics/application", "GET", 0, 200)

    metrics = monitoring_dashboard_service.get_application_metrics()
    return safe_json_response(metrics)

@app.get("/api/monitoring/metrics/endpoints")
async def get_endpoint_metrics(limit: int = 20):
    monitoring_dashboard_service.record_request("/api/monitoring/metrics/endpoints", "GET", 0, 200)

    metrics = monitoring_dashboard_service.get_endpoint_metrics(limit)
    return safe_json_response({"endpoints": metrics})

@app.get("/api/monitoring/dashboard")
async def get_monitoring_dashboard():
    monitoring_dashboard_service.record_request("/api/monitoring/dashboard", "GET", 0, 200)

    dashboard = monitoring_dashboard_service.get_dashboard_summary()
    return safe_json_response(dashboard)

@app.get("/api/monitoring/alerts")
async def get_active_alerts(limit: int = 10):
    alerts = monitoring_dashboard_service.get_active_alerts(limit)
    return safe_json_response({"alerts": alerts})

@app.post("/api/monitoring/alerts/acknowledge/{alert_id}")
async def acknowledge_alert(alert_id: str):
    success = monitoring_dashboard_service.acknowledge_alert(alert_id)
    return safe_json_response({"acknowledged": success})

from services.calendar_sync_service import google_calendar_service, CalendarEventType

@app.get("/api/calendar/status/{user_id}")
async def get_calendar_status(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/calendar/status")

    status = google_calendar_service.get_connection_status(user_id)
    return safe_json_response(status)

class CalendarEventInput(BaseModel):
    event_type: str = Field(default="decision_deadline", max_length=50)
    title: str = Field(..., max_length=200)
    start_time: str = Field(..., max_length=50)
    end_time: Optional[str] = Field(default=None, max_length=50)
    description: Optional[str] = Field(default="", max_length=1000)
    location: Optional[str] = Field(default="", max_length=200)
    user_id: Optional[str] = Field(default=None, max_length=100)

@app.post("/api/calendar/events")
async def create_calendar_event(input_data: CalendarEventInput):
    app_state.monitoring.record("/api/calendar/events")
    from datetime import datetime

    try:
        event_type = CalendarEventType(input_data.event_type)
    except ValueError:
        event_type = CalendarEventType.DECISION_DEADLINE

    start_time = datetime.fromisoformat(input_data.start_time.replace("Z", "+00:00"))
    end_time = None
    if input_data.end_time:
        end_time = datetime.fromisoformat(input_data.end_time.replace("Z", "+00:00"))

    result = google_calendar_service.create_event(
        user_id=input_data.user_id or "anonymous",
        event_type=event_type,
        title=input_data.title,
        start_time=start_time,
        end_time=end_time,
        description=input_data.description,
        location=input_data.location
    )
    return safe_json_response(result)

@app.get("/api/calendar/events/{user_id}")
async def get_calendar_events(user_id: str, current_user: str = Depends(get_current_user), days: int = 30):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/calendar/events")

    events = google_calendar_service.get_upcoming_events(user_id, days)
    return safe_json_response({"events": events})

@app.get("/api/calendar/today/{user_id}")
async def get_today_agenda(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/calendar/today")

    agenda = google_calendar_service.get_today_agenda(user_id)
    return safe_json_response(agenda)

@app.post("/api/calendar/sync/{user_id}")
async def sync_calendar(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/calendar/sync")

    result = google_calendar_service.sync_to_google(user_id)
    return safe_json_response(result)

@app.get("/manifest.json")
async def get_manifest():
    manifest_path = os.path.join(os.path.dirname(__file__), "assets", "manifest.json")
    try:
        with open(manifest_path, "r") as f:
            import json
            manifest = json.load(f)
        return safe_json_response(manifest)
    except FileNotFoundError:
        return safe_json_response({
            "name": "Career Decision AI",
            "short_name": "CareerAI",
            "start_url": "/",
            "display": "standalone",
            "background_color": "#000000",
            "theme_color": "#000000"
        })

@app.get("/service-worker.js")
async def get_service_worker():
    sw_path = os.path.join(os.path.dirname(__file__), "assets", "service-worker.js")
    try:
        with open(sw_path, "r") as f:
            content = f.read()
        return Response(
            content=content,
            media_type="application/javascript",
            headers={"Service-Worker-Allowed": "/"}
        )
    except FileNotFoundError:
        return Response(content="// Service worker not found", media_type="application/javascript")

@app.get("/api/pwa/icon-{size}.png")
async def get_pwa_icon(size: int):
    svg_icon = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 100 100">
        <rect width="100" height="100" rx="20" fill="#000000" />
        <text x="50" y="65" font-family="Arial" font-size="50" fill="white" text-anchor="middle" font-weight="bold">AI</text>
    </svg>'''
    return Response(content=svg_icon, media_type="image/svg+xml")


@app.get("/api/mentor/matches/{user_id}")
async def get_mentor_matches(user_id: str, industry: str = "", expertise: str = "", current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/mentor/matches")
    requirements = {
        "industry": industry,
        "expertise": expertise.split(",") if expertise else []
    }
    matches = mentor_matching_service.get_matches(requirements)
    return safe_json_response({"matches": [m.__dict__ for m in matches]})

@app.post("/api/mentor/request")
async def request_mentor(user_id: str, mentor_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/mentor/request")
    success = mentor_matching_service.request_match(user_id, mentor_id)
    return safe_json_response({"success": success})

@app.get("/api/mentor/connected/{user_id}")
async def get_connected_mentors(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/mentor/connected")
    mentors = mentor_matching_service.get_user_mentors(user_id)
    return safe_json_response({"mentors": mentors})

@app.post("/api/mentor/message")
async def send_mentor_message(user_id: str, mentor_id: str, text: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/mentor/message")
    success = mentor_matching_service.send_message(user_id, mentor_id, text)
    return safe_json_response({"success": success})

@app.get("/api/mentor/videos/personalized/{user_id}")
async def get_personalized_video_recommendations(
    user_id: str,
    mentor_expertise: str = "",
    user_goals: str = "",
    limit: int = 8,
    current_user: str = Depends(get_current_user)
):
    """Get personalized YouTube video recommendations based on mentor's expertise and user's goals"""
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/mentor/videos/personalized")
    
    mentor_skills = mentor_expertise.split(",") if mentor_expertise else []
    goals = user_goals.split(",") if user_goals else None
    
    videos = app_state.youtube_recommendation.get_personalized_recommendations(
        user_id=user_id,
        mentor_expertise=mentor_skills,
        user_goals=goals,
        limit=limit
    )
    
    return safe_json_response({
        "videos": [
            {
                "id": v.video_id,
                "title": v.title,
                "channel": v.channel,
                "duration_minutes": v.duration_minutes,
                "category": v.category.value,
                "keywords": v.keywords,
                "description": v.description,
                "view_count": v.view_count,
                "relevance_score": v.relevance_score,
                "url": f"https://www.youtube.com/watch?v={v.video_id}"
            }
            for v in videos
        ],
        "total": len(videos)
    })


@app.get("/api/mentor/videos/specialty/{mentor_id}")
async def get_mentor_specialty_videos(
    mentor_id: str,
    limit: int = 6,
    current_user: str = Depends(get_current_user)
):
    """Get videos specifically related to mentor's expertise and industry"""
    app_state.monitoring.record("/api/mentor/videos/specialty")
    
    mentor = mentor_matching_service.mentors.get(mentor_id)
    if not mentor:
        raise HTTPException(status_code=404, detail="Mentor not found")
    
    videos = app_state.youtube_recommendation.get_mentor_specialty_videos(
        mentor_expertise=mentor.expertise,
        mentor_industry=mentor.industry,
        limit=limit
    )
    
    return safe_json_response({
        "mentor_name": mentor.name,
        "mentor_expertise": mentor.expertise,
        "videos": [
            {
                "id": v.video_id,
                "title": v.title,
                "channel": v.channel,
                "duration_minutes": v.duration_minutes,
                "category": v.category.value,
                "description": v.description,
                "view_count": v.view_count,
                "url": f"https://www.youtube.com/watch?v={v.video_id}"
            }
            for v in videos
        ]
    })


@app.get("/api/mentor/videos/skill-gaps")
async def get_skill_gap_videos(
    current_skills: str = "",
    target_skills: str = "",
    limit: int = 6,
    current_user: str = Depends(get_current_user)
):
    """Get videos to help bridge identified skill gaps"""
    app_state.monitoring.record("/api/mentor/videos/skill-gaps")
    
    current = current_skills.split(",") if current_skills else []
    target = target_skills.split(",") if target_skills else []
    
    videos = app_state.youtube_recommendation.get_skill_gap_videos(
        current_skills=current,
        target_skills=target,
        limit=limit
    )
    
    return safe_json_response({
        "skill_gaps": list(set(target) - set(current)),
        "videos": [
            {
                "id": v.video_id,
                "title": v.title,
                "channel": v.channel,
                "duration_minutes": v.duration_minutes,
                "category": v.category.value,
                "description": v.description,
                "keywords": v.keywords,
                "view_count": v.view_count,
                "url": f"https://www.youtube.com/watch?v={v.video_id}"
            }
            for v in videos
        ]
    })


@app.get("/api/mentor/videos/industry-trends/{industry}")
async def get_industry_trending_videos(
    industry: str,
    limit: int = 6,
    current_user: str = Depends(get_current_user)
):
    """Get trending videos in a specific industry"""
    app_state.monitoring.record("/api/mentor/videos/industry-trends")
    
    videos = app_state.youtube_recommendation.get_industry_trending_videos(
        industry=industry,
        limit=limit
    )
    
    return safe_json_response({
        "industry": industry,
        "trending_videos": [
            {
                "id": v.video_id,
                "title": v.title,
                "channel": v.channel,
                "duration_minutes": v.duration_minutes,
                "category": v.category.value,
                "description": v.description,
                "view_count": v.view_count,
                "url": f"https://www.youtube.com/watch?v={v.video_id}"
            }
            for v in videos
        ]
    })


@app.get("/api/mentor/videos/learning-path")
async def get_learning_path_videos(
    career_goal: str,
    current_level: str = "beginner",
    limit: int = 10,
    current_user: str = Depends(get_current_user)
):
    """Get structured learning path videos for a specific career goal"""
    app_state.monitoring.record("/api/mentor/videos/learning-path")
    
    if current_level not in ["beginner", "intermediate", "advanced"]:
        current_level = "beginner"
    
    videos = app_state.youtube_recommendation.get_learning_path_videos(
        career_goal=career_goal,
        current_level=current_level,
        limit=limit
    )
    
    return safe_json_response({
        "career_goal": career_goal,
        "current_level": current_level,
        "learning_path": [
            {
                "id": v.video_id,
                "title": v.title,
                "channel": v.channel,
                "duration_minutes": v.duration_minutes,
                "category": v.category.value,
                "description": v.description,
                "keywords": v.keywords,
                "url": f"https://www.youtube.com/watch?v={v.video_id}"
            }
            for v in videos
        ]
    })


@app.post("/api/mentor/videos/watched")
async def mark_video_watched(
    user_id: str,
    video_id: str,
    current_user: str = Depends(get_current_user)
):
    """Mark a video as watched by the user"""
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/mentor/videos/watched")
    
    success = app_state.youtube_recommendation.mark_video_watched(user_id, video_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return safe_json_response({"success": True, "message": "Video marked as watched"})


@app.post("/api/mentor/videos/save")
async def save_video_for_later(
    user_id: str,
    video_id: str,
    current_user: str = Depends(get_current_user)
):
    """Save a video to watch later list"""
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/mentor/videos/save")
    
    success = app_state.youtube_recommendation.save_video_for_later(user_id, video_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return safe_json_response({"success": True, "message": "Video saved for later"})


@app.post("/api/mentor/videos/rate")
async def rate_video(
    user_id: str,
    video_id: str,
    rating: float,
    current_user: str = Depends(get_current_user)
):
    """Rate a video (1-5 stars)"""
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/mentor/videos/rate")
    
    if not (1.0 <= rating <= 5.0):
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
    
    success = app_state.youtube_recommendation.rate_video(user_id, video_id, rating)
    
    if not success:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return safe_json_response({"success": True, "rating": rating, "message": "Video rated successfully"})


@app.get("/api/mentor/videos/search")
async def search_videos(
    query: str,
    limit: int = 10,
    current_user: str = Depends(get_current_user)
):
    """Search for videos by keyword or title"""
    app_state.monitoring.record("/api/mentor/videos/search")
    
    if not query or len(query) < 2:
        raise HTTPException(status_code=400, detail="Search query must be at least 2 characters")
    
    videos = app_state.youtube_recommendation.search_videos(query, limit)
    
    return safe_json_response({
        "query": query,
        "results": [
            {
                "id": v.video_id,
                "title": v.title,
                "channel": v.channel,
                "duration_minutes": v.duration_minutes,
                "category": v.category.value,
                "description": v.description,
                "view_count": v.view_count,
                "url": f"https://www.youtube.com/watch?v={v.video_id}"
            }
            for v in videos
        ],
        "total_results": len(videos)
    })


@app.get("/api/mentor/videos/watch-history/{user_id}")
async def get_watch_history(
    user_id: str,
    limit: int = 10,
    current_user: str = Depends(get_current_user)
):
    """Get user's video watch history"""
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/mentor/videos/watch-history")
    
    videos = app_state.youtube_recommendation.get_watch_history(user_id, limit)
    
    return safe_json_response({
        "watch_history": [
            {
                "id": v.video_id,
                "title": v.title,
                "channel": v.channel,
                "duration_minutes": v.duration_minutes,
                "category": v.category.value,
                "watched_at": v.created_at.isoformat(),
                "url": f"https://www.youtube.com/watch?v={v.video_id}"
            }
            for v in videos
        ],
        "total": len(videos)
    })


@app.get("/api/mentor/videos/categories")
async def get_video_categories(current_user: str = Depends(get_current_user)):
    """Get all available video categories"""
    app_state.monitoring.record("/api/mentor/videos/categories")
    
    categories = app_state.youtube_recommendation.get_all_categories()
    
    return safe_json_response({
        "categories": categories,
        "total": len(categories)
    })


@app.post("/api/mentor/setup-videos")
async def setup_mentor_with_videos(
    user_id: str,
    mentor_id: str,
    current_user: str = Depends(get_current_user)
):
    """Setup recommended videos for a mentor match"""
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/mentor/setup-videos")
    
    mentor = mentor_matching_service.mentors.get(mentor_id)
    if not mentor:
        raise HTTPException(status_code=404, detail="Mentor not found")
    
    videos = app_state.youtube_recommendation.get_personalized_recommendations(
        user_id=user_id,
        mentor_expertise=mentor.expertise,
        user_goals=None,
        limit=8
    )
    
    specialty_videos = app_state.youtube_recommendation.get_mentor_specialty_videos(
        mentor_expertise=mentor.expertise,
        mentor_industry=mentor.industry,
        limit=4
    )
    
    video_data = [
        {
            "id": v.video_id,
            "title": v.title,
            "channel": v.channel,
            "duration_minutes": v.duration_minutes,
            "category": v.category.value,
            "url": f"https://www.youtube.com/watch?v={v.video_id}"
        }
        for v in videos
    ]
    
    specialty_data = [
        {
            "id": v.video_id,
            "title": v.title,
            "channel": v.channel,
            "duration_minutes": v.duration_minutes,
            "category": v.category.value,
            "url": f"https://www.youtube.com/watch?v={v.video_id}"
        }
        for v in specialty_videos
    ]
    
    mentor_matching_service.add_recommended_videos_to_match(user_id, mentor_id, video_data)
    
    return safe_json_response({
        "success": True,
        "mentor_name": mentor.name,
        "recommended_videos": video_data,
        "specialty_videos": specialty_data,
        "total_videos": len(video_data) + len(specialty_data)
    })


@app.post("/api/simulate/run")
async def run_simulation(decision_desc: str, base_salary: float, uncertainty: float = 0.2):
    app_state.monitoring.record("/api/simulate/run")
    result = simulation_service.run_monte_carlo(decision_desc, base_salary, uncertainty)
    return safe_json_response(result)

@app.post("/api/roadmap/generate")
async def generate_career_roadmap(user_id: str, target_role: str, gap_skills: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/roadmap/generate")
    skills_list = gap_skills.split(",") if gap_skills else []
    result = roadmap_service.generate_roadmap(user_id, target_role, skills_list)
    return safe_json_response(result)



@app.post("/api/knowledge/add")
async def add_knowledge(user_id: str, filename: str, content: str = "", current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/knowledge/add")
    doc = knowledge_service.add_document(user_id, filename, content)
    return safe_json_response(doc)

@app.get("/api/knowledge/list/{user_id}")
async def list_knowledge(user_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/knowledge/list")
    docs = knowledge_service.get_documents(user_id)
    return safe_json_response({"documents": docs})

@app.delete("/api/knowledge/{user_id}/{doc_id}")
async def delete_knowledge(user_id: str, doc_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    app_state.monitoring.record("/api/knowledge/delete")
    success = knowledge_service.delete_document(user_id, doc_id)
    return safe_json_response({"success": success})

@app.post("/api/share/create")
async def create_shared_link(decision_id: str, user_id: str):
    app_state.monitoring.record("/api/share/create")
    entry = app_state.journal_service.get_entry(user_id, decision_id)
    if not entry:
        raise HTTPException(404, "Decision not found")

    short_code = decision_sharing_service.share_decision(entry.__dict__)
    return safe_json_response({"short_code": short_code, "url": f"/shared/{short_code}"})

@app.get("/api/share/{short_code}")
async def get_shared_decision(short_code: str):
    app_state.monitoring.record("/api/share/view")
    data = decision_sharing_service.get_shared_decision(short_code)
    if not data:
        raise HTTPException(404, "Shared decision not found or expired")
    return safe_json_response(data)

@app.get("/api/llm/config")
async def get_llm_config():
    return safe_json_response({
        "active_provider": multi_llm_service.active_provider,
        "available_providers": [p.value for p in LLMProvider]
    })

@app.post("/api/llm/switch")
async def switch_llm_provider(provider: str):
    success = multi_llm_service.set_active_provider(provider)
    return safe_json_response({"success": success, "provider": provider})

@app.get("/api/finetune/status")
async def get_finetune_status():
    return safe_json_response(fine_tuning_service.get_stats())

@app.post("/api/finetune/toggle")
async def toggle_finetune(active: bool):
    model = fine_tuning_service.toggle_finetuned_model(active)
    return safe_json_response({"active": active, "model": model})

@app.get("/api/abtest/variant/{user_id}/{experiment_id}")
async def get_ab_variant(user_id: str, experiment_id: str, current_user: str = Depends(get_current_user)):
    verify_owner(user_id, current_user)
    variant = ab_testing_service.get_variant(user_id, experiment_id)
    return safe_json_response({"variant": variant})

@app.post("/api/integrations/webhook/setup")
async def setup_webhook(user_id: str, url: str):
    enterprise_integration_service.setup_zapier_webhook(user_id, url)
    return safe_json_response({"success": True})

@app.post("/api/integrations/slack/webhook")
async def slack_webhook(request: Request):
    data = await request.form()
    signature = request.headers.get("X-Slack-Signature", "")
    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    response = enterprise_integration_service.handle_slack_command(dict(data), signature, timestamp)
    return safe_json_response(response)

if __name__ == "__main__":
    import uvicorn

    if not settings.DEBUG:
        if not os.getenv("JWT_SECRET_KEY"):
            print("CRITICAL: Set JWT_SECRET_KEY environment variable for production!")
        if os.getenv("CORS_ORIGINS", "") == "":
            print("WARNING: CORS_ORIGINS not set. API may be inaccessible.")

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        access_log=True,
        proxy_headers=True,
        forwarded_allow_ips="*" if settings.DEBUG else os.getenv("TRUSTED_PROXIES", "127.0.0.1")
    )
