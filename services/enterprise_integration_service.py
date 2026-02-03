import hmac
import hashlib
import json
import httpx
import secrets
from typing import Dict, Any, Optional
from datetime import datetime

class EnterpriseIntegrationService:
    def __init__(self):
        self.webhook_configs: Dict[str, Dict] = {}
        self.verified_tokens: Dict[str, str] = {}

    def setup_zapier_webhook(self, user_id: str, webhook_url: str):
        self.webhook_configs[user_id] = {
            "zapier_url": webhook_url,
            "created_at": datetime.utcnow().isoformat()
        }

    async def trigger_webhook(self, user_id: str, payload: Dict[str, Any]):
        """Send data to Zapier/Custom endpoint."""
        config = self.webhook_configs.get(user_id)
        if not config or "zapier_url" not in config:
            return False

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(config["zapier_url"], json=payload)
                return response.status_code < 300
            except Exception as e:
                print(f"Webhook error for {user_id}: {e}")
                return False

    def handle_slack_command(self, payload: Dict[str, Any], signature: str, timestamp: str) -> Dict[str, Any]:
        """Mock handler for Slack slash commands."""
        text = payload.get("text", "")
        return {
            "response_type": "in_channel",
            "text": f"CareerAI received your request: '{text}'. Analyzing now..."
        }

    def generate_api_key(self, user_id: str) -> str:
        """Generate a token for external tools (Zapier/Custom)."""
        token = hashlib.sha256(f"{user_id}{datetime.utcnow()}{secrets.token_hex(8)}".encode()).hexdigest()
        self.verified_tokens[token] = user_id
        return token

enterprise_integration_service = EnterpriseIntegrationService()
