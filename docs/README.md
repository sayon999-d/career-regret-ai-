# Documentation

This folder contains comprehensive documentation for the Career Decision Regret AI system.

## Contents

### API Documentation

**[API.md](API.md)** - Complete REST API reference including:

- Authentication endpoints (register, login, logout, refresh)
- Notification management
- Data export and import (JSON, CSV, ICS, ZIP)
- Enhanced analytics and dashboard
- Decision comparison and what-if analysis
- Emotion detection endpoints
- AI personalization and feedback
- User preferences and settings
- Health check and status endpoints
- Rate limiting information
- Error response formats
- WebSocket real-time updates

## Quick Start

### Base URL

All API endpoints are available at:

```
http://localhost:8000
```

### Authentication

Most endpoints require JWT authentication:

```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password123"}'
```

Use the returned `access_token` in subsequent requests:

```bash
curl http://localhost:8000/api/notifications/user_123 \
  -H "Authorization: Bearer <access_token>"
```

### Common Operations

**Analyze a Decision:**
```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"decision_text": "Should I accept this job offer?"}'
```

**Export Data:**
```bash
curl http://localhost:8000/api/export/user_123/all?format=zip \
  -o backup.zip
```

**Get Dashboard Analytics:**
```bash
curl http://localhost:8000/api/analytics/user_123/dashboard
```

## Additional Resources

- **[Main README](../README.md)** - Project overview and setup
- **[Services README](../services/README.md)** - Microservices documentation
- **[Models README](../models/README.md)** - ML pipeline documentation
- **[Tests README](../tests/README.md)** - Testing framework
- **[SECURITY.md](../SECURITY.md)** - Security documentation

## API Versioning

The current API version is **v2.0.0**.

All endpoints are backward compatible. Deprecated endpoints will remain available for at least 6 months after deprecation notice.

## Support

For API issues or questions, please open an issue on GitHub or contact the development team.
