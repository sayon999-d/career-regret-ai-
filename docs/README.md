# Documentation

This folder contains comprehensive documentation for the StepWise AI system.

## Documentation Map

```mermaid
graph TD
    ROOT["README.md<br/>Project Overview"]
    ROOT --> DOCS["docs/README.md<br/>API Documentation Index"]
    ROOT --> MODELS["models/README.md<br/>ML Pipeline, Graph Engine, Database"]
    ROOT --> SERVICES["services/README.md<br/>50+ Microservices Architecture"]
    ROOT --> TESTS["tests/README.md<br/>Testing Framework"]
    ROOT --> SECURITY["SECURITY.md<br/>Security Configuration"]

    DOCS --> API["docs/API.md<br/>Full REST API Reference"]
```

## Contents

### API Documentation

**[API.md](API.md)** -- Complete REST API reference including:

- Authentication endpoints (register, login, logout, GitHub OAuth)
- Decision analysis and AI chat
- Resume upload and parsing
- Journal and history management
- Analytics and simulation
- Data export and import (JSON, PDF, Text)
- Emotion detection endpoints
- WebSocket real-time updates
- Rate limiting information
- Error response formats

## Authentication Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant AG as Auth Guard (JS)
    participant API as FastAPI
    participant AUTH as HardenedAuthService

    Note over C,AUTH: Login Flow
    C->>API: POST /api/auth/login {username, password}
    API->>AUTH: authenticate(username, password, ip)
    AUTH->>AUTH: Verify password (PBKDF2-SHA256)
    AUTH->>AUTH: Check brute force lockout
    AUTH-->>API: session_token, user_id
    API-->>C: 200 {session_token, user_id, username}
    C->>C: localStorage.setItem("session_token", token)

    Note over C,AUTH: Dashboard Access
    C->>AG: Navigate to /
    AG->>AG: Check localStorage.session_token
    alt Token exists
        AG-->>C: Load dashboard
    else No token
        AG-->>C: Redirect to /login
    end

    Note over C,AUTH: GitHub OAuth Flow
    C->>API: GET /api/auth/github
    API-->>C: Redirect to github.com/login/oauth/authorize
    C->>API: GET /api/auth/github/callback?code=xxx
    API->>AUTH: Register or find gh_username
    AUTH-->>API: session_token
    API-->>C: Redirect to /?auth_token=xxx
    AG->>AG: Capture token from URL, store in localStorage
```

## Quick Start

### Base URL

All API endpoints are available at:

```
http://localhost:8000
```

### Authentication

Most endpoints require a session token:

```bash
# Login
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "myuser", "password": "MyPassword123!"}'

# Response: {"session_token": "xxx", "user_id": "user_abc", "username": "myuser"}
```

Use the returned `session_token` in subsequent requests:

```bash
curl http://localhost:8000/api/analytics/user_abc \
  -H "Authorization: Bearer <session_token>"
```

### Common Operations

Analyze a Decision:
```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"decision_type": "job_change", "description": "Should I accept this offer?"}'
```

AI Chat:
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I am considering switching careers", "user_id": "user_abc"}'
```

Export Data:
```bash
curl http://localhost:8000/api/export/user_abc/json \
  -H "Authorization: Bearer <session_token>" \
  -o export.json
```

## Additional Resources

- **[Main README](../README.md)** -- Project overview, architecture, and setup
- **[Services README](../services/README.md)** -- Microservices architecture and API patterns
- **[Models README](../models/README.md)** -- ML pipeline and database documentation
- **[Tests README](../tests/README.md)** -- Testing framework and guides
- **[SECURITY.md](../SECURITY.md)** -- Security features and configuration

## API Versioning

The current API version is v3.0.0.

All endpoints are backward compatible. Breaking changes are documented in release notes.
