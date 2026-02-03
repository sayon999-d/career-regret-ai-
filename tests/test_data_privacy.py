import pytest
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.data_privacy_service import (
    DataPrivacyService,
    ConsentType,
    DataCategory,
    EncryptionService
)

class TestEncryptionService:
    @pytest.fixture
    def service(self):
        return EncryptionService("test_master_key")

    def test_encrypt_decrypt(self, service):
        original_text = "This is sensitive information"

        encrypted = service.encrypt_data(original_text, "user_salt")

        assert encrypted != original_text

        decrypted = service.decrypt_data(encrypted, "user_salt")

        assert decrypted == original_text

    def test_different_salts_different_results(self, service):
        text = "Same text to encrypt"

        encrypted1 = service.encrypt_data(text, "salt1")
        encrypted2 = service.encrypt_data(text, "salt2")

        assert encrypted1 != encrypted2

    def test_hash_pii(self, service):
        email = "user@example.com"

        hashed = service.hash_pii(email)

        assert email not in hashed
        assert len(hashed) == 64

    def test_hash_is_consistent(self, service):
        email = "user@example.com"

        hash1 = service.hash_pii(email)
        hash2 = service.hash_pii(email)

        assert hash1 == hash2

    def test_anonymize_user_id(self, service):
        user_id = "user_123"

        anonymized = service.anonymize_user_id(user_id)

        assert anonymized.startswith("anon_")
        assert user_id not in anonymized


class TestDataPrivacyService:
    @pytest.fixture
    def service(self):
        return DataPrivacyService()

    def test_record_consent(self, service, sample_user_id):
        consent = service.record_consent(
            user_id=sample_user_id,
            consent_type=ConsentType.DATA_COLLECTION,
            granted=True,
            ip_address="127.0.0.1"
        )

        assert consent.granted == True
        assert consent.user_id == sample_user_id

    def test_check_consent(self, service, sample_user_id):
        service.record_consent(sample_user_id, ConsentType.ANALYTICS, True)

        assert service.check_consent(sample_user_id, ConsentType.ANALYTICS) == True
        assert service.check_consent(sample_user_id, ConsentType.MARKETING) == False

    def test_revoke_consent(self, service, sample_user_id):
        service.record_consent(sample_user_id, ConsentType.GLOBAL_INSIGHTS, True)

        assert service.check_consent(sample_user_id, ConsentType.GLOBAL_INSIGHTS) == True

        service.record_consent(sample_user_id, ConsentType.GLOBAL_INSIGHTS, False)

        assert service.check_consent(sample_user_id, ConsentType.GLOBAL_INSIGHTS) == False

    def test_get_user_consents(self, service, sample_user_id):
        service.record_consent(sample_user_id, ConsentType.DATA_COLLECTION, True)
        service.record_consent(sample_user_id, ConsentType.ANALYTICS, True)
        service.record_consent(sample_user_id, ConsentType.MARKETING, False)

        consents = service.get_user_consents(sample_user_id)

        assert consents["data_collection"] == True
        assert consents["analytics"] == True
        assert consents["marketing"] == False

    def test_request_data_export(self, service, sample_user_id):
        request = service.request_data_export(
            user_id=sample_user_id,
            categories=[DataCategory.PROFILE, DataCategory.DECISIONS]
        )

        assert request.user_id == sample_user_id
        assert request.status == "pending"
        assert len(request.categories) == 2

    def test_request_full_export(self, service, sample_user_id):
        request = service.request_data_export(user_id=sample_user_id)

        assert len(request.categories) == len(DataCategory)

    def test_request_deletion(self, service, sample_user_id):
        request = service.request_account_deletion(
            user_id=sample_user_id,
            reason="No longer using the service"
        )

        assert request.user_id == sample_user_id
        assert request.status == "pending"
        assert request.scheduled_at > datetime.utcnow()

    def test_cancel_deletion(self, service, sample_user_id):
        request = service.request_account_deletion(sample_user_id)

        success = service.cancel_deletion_request(request.id, sample_user_id)

        assert success == True

    def test_anonymize_data(self, service):
        user_data = {
            "user_id": "user_123",
            "name": "John Doe",
            "email": "john@example.com",
            "decision_type": "job_change",
            "regret_score": 25
        }

        anonymized = service.anonymize_data_for_global_db(user_data)

        assert "John Doe" not in str(anonymized)
        assert "john@example.com" not in str(anonymized)
        assert anonymized["decision_type"] == "job_change"

    def test_encrypt_sensitive_entry(self, service, sample_user_id):
        entry = {
            "id": "entry_1",
            "journal_content": "This is my private journal entry",
            "decision_type": "career_switch"
        }

        encrypted = service.encrypt_sensitive_entry(sample_user_id, entry)

        assert encrypted["journal_content_encrypted"] == True
        assert encrypted["journal_content"] != "This is my private journal entry"

    def test_decrypt_sensitive_entry(self, service, sample_user_id):
        original_content = "This is my private journal entry"
        entry = {
            "id": "entry_1",
            "journal_content": original_content,
            "decision_type": "career_switch"
        }

        encrypted = service.encrypt_sensitive_entry(sample_user_id, entry)
        decrypted = service.decrypt_sensitive_entry(sample_user_id, encrypted)

        assert decrypted["journal_content"] == original_content

    def test_privacy_dashboard(self, service, sample_user_id):
        service.record_consent(sample_user_id, ConsentType.DATA_COLLECTION, True)

        dashboard = service.get_privacy_dashboard(sample_user_id)

        assert "consents" in dashboard
        assert "data_retention" in dashboard
        assert "rights" in dashboard

    def test_access_log(self, service, sample_user_id):
        service.record_consent(sample_user_id, ConsentType.ANALYTICS, True)
        service.request_data_export(sample_user_id)

        logs = service.get_access_log(sample_user_id)

        assert len(logs) >= 2


class TestDataRetention:
    @pytest.fixture
    def service(self):
        return DataPrivacyService()

    def test_retention_periods_defined(self, service):
        for category in DataCategory:
            assert category in service.RETENTION_PERIODS

    def test_retention_periods_reasonable(self, service):
        for category, days in service.RETENTION_PERIODS.items():
            assert days >= 365
            assert days <= 365 * 7
