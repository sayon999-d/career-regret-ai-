import secrets
import string
from typing import Dict, Optional, Any
from datetime import datetime, timedelta

class DecisionSharingService:
    def __init__(self):
        self.shared_decisions: Dict[str, Dict] = {}
        self.expirations: Dict[str, datetime] = {}

    def share_decision(self, decision_data: Dict[str, Any], expiry_days: int = 7) -> str:
        """Create a shared link for a decision."""
        alphabet = string.ascii_letters + string.digits
        short_code = ''.join(secrets.choice(alphabet) for _ in range(8))

        self.shared_decisions[short_code] = decision_data
        self.expirations[short_code] = datetime.utcnow() + timedelta(days=expiry_days)

        return short_code

    def get_shared_decision(self, short_code: str) -> Optional[Dict]:
        """Retrieve a shared decision if not expired."""
        if short_code not in self.shared_decisions:
            return None

        if datetime.utcnow() > self.expirations[short_code]:
            del self.shared_decisions[short_code]
            del self.expirations[short_code]
            return None

        return self.shared_decisions[short_code]

    def revoke_share(self, short_code: str):
        if short_code in self.shared_decisions:
            del self.shared_decisions[short_code]
            del self.expirations[short_code]

decision_sharing_service = DecisionSharingService()
