# Career Decision Regret AI - API Documentation

## Base URL

```
http://localhost:8000
```

## Authentication

Most endpoints require authentication via JWT tokens. Include the token in the Authorization header:

```
Authorization: Bearer <access_token>
```

---

## Health & Status

### Health Check

```
GET /health
```

Returns basic health status of the application.

**Response:**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "features": ["chat", "analytics", "simulation", "templates"],
  "services": {
    "ollama": true,
    "database": true,
    "redis": false
  }
}
```

### API Status

```
GET /api/status
```

Returns detailed service status including uptime and request counts.

---

## Authentication Endpoints

### Register User

```
POST /api/auth/register
```

**Request Body:**
```json
{
  "email": "user@example.com",
  "username": "username",
  "password": "password123",
  "full_name": "John Doe"
}
```

**Response:**
```json
{
  "user": {
    "id": "user_123",
    "email": "user@example.com",
    "username": "username",
    "full_name": "John Doe"
  },
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

### Login

```
POST /api/auth/login
```

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "password123"
}
```

### Logout

```
POST /api/auth/logout
```

**Query Parameters:**
- `session_token` (required): Session token to invalidate

### Refresh Token

```
POST /api/auth/refresh
```

**Request Body:**
```json
{
  "refresh_token": "eyJ..."
}
```

### Get Current User

```
GET /api/auth/me
```

**Query Parameters:**
- `token` (required): Access token

### Change Password

```
POST /api/auth/change-password
```

**Request Body:**
```json
{
  "user_id": "user_123",
  "old_password": "oldpass",
  "new_password": "newpass123"
}
```

---

## Notifications

### Get User Notifications

```
GET /api/notifications/{user_id}
```

**Query Parameters:**
- `limit` (optional): Number of notifications to return (default: 50)
- `unread_only` (optional): Return only unread notifications (default: false)

**Response:**
```json
{
  "notifications": [
    {
      "id": "notif_123",
      "title": "Decision Reminder",
      "message": "You have a pending decision to review",
      "type": "REMINDER",
      "is_read": false,
      "created_at": "2024-01-15T10:00:00Z"
    }
  ],
  "total": 10,
  "unread_count": 3
}
```

### Get Unread Notifications

```
GET /api/notifications/{user_id}/unread
```

### Get Pending Notifications

```
GET /api/notifications/{user_id}/pending
```

### Mark Notification as Read

```
POST /api/notifications/{user_id}/read/{notification_id}
```

### Mark All Notifications as Read

```
POST /api/notifications/{user_id}/read-all
```

---

## Data Export & Import

### Export All Data

```
GET /api/export/{user_id}/all
```

**Query Parameters:**
- `format` (optional): Export format - `json`, `csv`, or `zip` (default: json)

### Export Decisions

```
GET /api/export/{user_id}/decisions
```

**Query Parameters:**
- `format` (optional): Export format - `json` or `csv` (default: json)

### Export Calendar

```
GET /api/export/{user_id}/calendar
```

**Query Parameters:**
- `format` (optional): Export format - `json` or `ics` (default: json)

### Import Data

```
POST /api/import/{user_id}
```

**Request Body:**
```json
{
  "content": "base64_encoded_content",
  "format": "json"
}
```

### Create Backup

```
POST /api/backup/{user_id}/create
```

### List Backups

```
GET /api/backup/{user_id}/list
```

---

## Enhanced Analytics

### Get Dashboard Analytics

```
GET /api/analytics/{user_id}/dashboard
```

**Response:**
```json
{
  "overview": {
    "total_decisions": 45,
    "average_regret": 32.5,
    "decisions_this_month": 8,
    "pending_outcomes": 5
  },
  "patterns": {
    "most_common_category": "career",
    "best_outcomes_time": "morning",
    "regret_trend": "decreasing"
  },
  "emotions": {
    "dominant_emotion": "hopeful",
    "emotional_volatility": 0.35
  },
  "recommendations": [
    {
      "type": "timing",
      "message": "You make better decisions in the morning"
    }
  ]
}
```

### Get Decision Patterns

```
GET /api/analytics/{user_id}/patterns
```

**Query Parameters:**
- `days` (optional): Number of days to analyze (default: 90)

### Generate Report

```
GET /api/analytics/{user_id}/report
```

**Query Parameters:**
- `format` (optional): Report format - `json`, `pdf`, or `html`
- `period` (optional): Report period - `week`, `month`, `quarter`, `year`

### Get Activity Heatmap

```
GET /api/analytics/{user_id}/heatmap
```

---

## Decision Comparison

### Create Comparison

```
POST /api/compare/create
```

**Request Body:**
```json
{
  "title": "Job Offer Comparison",
  "options": ["Option A - Startup", "Option B - Enterprise"],
  "criteria": [
    {"name": "Salary", "weight": 0.3},
    {"name": "Growth", "weight": 0.25},
    {"name": "Work-Life Balance", "weight": 0.25},
    {"name": "Location", "weight": 0.2}
  ]
}
```

**Response:**
```json
{
  "comparison_id": "comp_123",
  "title": "Job Offer Comparison",
  "options": [...],
  "criteria": [...],
  "created_at": "2024-01-15T10:00:00Z"
}
```

### Evaluate Options

```
POST /api/compare/evaluate
```

**Request Body:**
```json
{
  "comparison_id": "comp_123",
  "scores": {
    "Option A": {
      "Salary": 7,
      "Growth": 9,
      "Work-Life Balance": 6,
      "Location": 8
    },
    "Option B": {...}
  }
}
```

### What-If Analysis

```
POST /api/compare/what-if
```

**Request Body:**
```json
{
  "comparison_id": "comp_123",
  "scenario": "If salary was 20% higher for Option A"
}
```

### Sensitivity Analysis

```
POST /api/compare/sensitivity
```

---

## Emotion Detection

### Detect Emotion from Image

```
POST /api/emotion/detect
```

**Request Body:**
```json
{
  "image_data": "base64_encoded_image"
}
```

**Response:**
```json
{
  "dominant_emotion": "neutral",
  "confidence": 0.87,
  "all_emotions": {
    "happy": 0.05,
    "sad": 0.02,
    "angry": 0.01,
    "fearful": 0.02,
    "surprised": 0.03,
    "neutral": 0.87
  },
  "face_detected": true
}
```

### Record Emotion

```
POST /api/emotion/record/{user_id}
```

**Request Body:**
```json
{
  "emotion": "hopeful",
  "confidence": 0.85,
  "decision_id": "dec_123"
}
```

### Get Emotion Trends

```
GET /api/emotion/trends/{user_id}
```

**Query Parameters:**
- `days` (optional): Number of days to analyze (default: 30)

### Check Emotion Service Status

```
GET /api/emotion/status
```

---

## AI Personalization

### Get User AI Context

```
GET /api/ai/context/{user_id}
```

**Response:**
```json
{
  "user_profile": {
    "decision_style": "analytical",
    "risk_tolerance": "moderate",
    "preferred_timeframe": "long-term"
  },
  "learning_insights": {
    "prediction_accuracy": 0.78,
    "common_biases": ["optimism bias"],
    "improvement_areas": ["considering alternatives"]
  },
  "personalized_prompts": [
    "Consider the long-term implications",
    "What would your future self think?"
  ]
}
```

### Get Personalized Suggestions

```
GET /api/ai/suggestions/{user_id}
```

**Query Parameters:**
- `context` (optional): Current decision context

### Submit AI Feedback

```
POST /api/ai/feedback
```

**Query Parameters:**
- `user_id` (required): User identifier
- `message_id` (required): Message being rated
- `feedback_type` (required): Type of feedback - `helpful`, `unhelpful`, `accurate`, `inaccurate`
- `rating` (optional): Rating from 1-5

### Get AI Learning Insights

```
GET /api/ai/insights/{user_id}
```

---

## Decision Outcomes

### Record Decision Outcome

```
POST /api/decisions/{user_id}/{decision_id}/outcome
```

**Request Body:**
```json
{
  "actual_regret": 25,
  "satisfaction": 8,
  "outcome_notes": "The decision turned out better than expected",
  "would_do_again": true
}
```

### Search Decisions

```
GET /api/decisions/{user_id}/search
```

**Query Parameters:**
- `search` (optional): Search query
- `status` (optional): Filter by status - `pending`, `completed`, `archived`
- `decision_type` (optional): Filter by type - `career`, `personal`, `financial`
- `limit` (optional): Results limit (default: 50)
- `offset` (optional): Results offset (default: 0)

---

## User Preferences

### Get User Preferences

```
GET /api/user/{user_id}/preferences
```

### Update User Preferences

```
PUT /api/user/{user_id}/preferences
```

**Request Body:**
```json
{
  "preferences": {
    "default_view": "analytics",
    "auto_save": true,
    "ai_suggestions": true
  }
}
```

### Update Theme

```
PUT /api/user/{user_id}/theme
```

**Request Body:**
```json
{
  "theme": "dark"
}
```

### Update Notification Settings

```
PUT /api/user/{user_id}/notifications
```

**Request Body:**
```json
{
  "settings": {
    "email_notifications": true,
    "decision_reminders": true,
    "weekly_digest": false
  }
}
```

---

## Chat & Analysis

### Send Chat Message

```
POST /api/chat
```

**Request Body:**
```json
{
  "message": "I'm considering changing careers to data science",
  "user_id": "user_123",
  "session_id": "session_456",
  "context": {}
}
```

### Analyze Decision

```
POST /api/analyze
```

**Request Body:**
```json
{
  "decision_text": "Should I accept the job offer?",
  "decision_type": "career",
  "context": {
    "current_situation": "Employed, but underpaid",
    "options": ["Accept", "Negotiate", "Decline"]
  }
}
```

---

## Simulation

### Run Outcome Simulation

```
POST /api/simulate
```

**Request Body:**
```json
{
  "decision_id": "dec_123",
  "scenarios": [
    "Best case scenario",
    "Worst case scenario",
    "Most likely scenario"
  ]
}
```

---

## Error Responses

All endpoints may return the following error responses:

### 400 Bad Request

```json
{
  "error": "Invalid request parameters",
  "detail": "email is required"
}
```

### 401 Unauthorized

```json
{
  "error": "Invalid or expired token"
}
```

### 404 Not Found

```json
{
  "error": "Resource not found"
}
```

### 429 Too Many Requests

```json
{
  "error": "Rate limit exceeded",
  "retry_after": 60
}
```

### 500 Internal Server Error

```json
{
  "error": "Internal server error",
  "detail": "An unexpected error occurred"
}
```

---

## Rate Limiting

API endpoints are rate-limited to prevent abuse:

- **Default limit**: 100 requests per minute per user
- **Chat endpoints**: 30 requests per minute
- **Export endpoints**: 10 requests per hour

Rate limit headers are included in all responses:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1705312800
```

---

## Versioning

The API follows semantic versioning. The current version is **v2.0.0**.

For backward compatibility, deprecated endpoints will remain available for at least 6 months after deprecation notice.

---

## WebSocket Endpoints

### Real-time Updates

```
WS /ws/{user_id}
```

Provides real-time updates for:
- New notifications
- Decision analysis results
- Collaborative editing (future feature)

**Message Format:**
```json
{
  "type": "notification",
  "data": {
    "id": "notif_123",
    "title": "New Insight",
    "message": "Your decision pattern analysis is ready"
  }
}
```
