import pytest
import asyncio
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.opportunity_scout_service import OpportunityScoutService

class TestOpportunityScoutService:
    @pytest.fixture
    def service(self):
        return OpportunityScoutService()

    def test_register_profile(self, service, sample_user_id, sample_profile):
        profile = service.register_user_profile(sample_user_id, sample_profile)

        assert profile is not None
        assert profile.user_id == sample_user_id
        assert profile.current_role == sample_profile["current_role"]

    @pytest.mark.asyncio
    async def test_scan_opportunities(self, service, sample_user_id, sample_profile):
        service.register_user_profile(sample_user_id, sample_profile)

        opportunities = await service.scan_opportunities(sample_user_id)

        assert isinstance(opportunities, list)

    @pytest.mark.asyncio
    async def test_opportunity_types(self, service, sample_user_id, sample_profile):
        service.register_user_profile(sample_user_id, sample_profile)

        opportunities = await service.scan_opportunities(sample_user_id)

        if opportunities:
            types = set(opp.type.value for opp in opportunities)
            assert len(types) >= 1

    def test_get_opportunities(self, service, sample_user_id, sample_profile):
        service.register_user_profile(sample_user_id, sample_profile)

        asyncio.get_event_loop().run_until_complete(
            service.scan_opportunities(sample_user_id)
        )

        opportunities = service.get_opportunities(sample_user_id)

        assert isinstance(opportunities, list)

    def test_filter_by_type(self, service, sample_user_id, sample_profile):
        service.register_user_profile(sample_user_id, sample_profile)

        asyncio.get_event_loop().run_until_complete(
            service.scan_opportunities(sample_user_id)
        )

        job_opps = service.get_opportunities(sample_user_id, type_filter="job")

        assert all(opp.get("type") == "job" for opp in job_opps)

    def test_mark_opportunity_saved(self, service, sample_user_id, sample_profile):
        service.register_user_profile(sample_user_id, sample_profile)

        asyncio.get_event_loop().run_until_complete(
            service.scan_opportunities(sample_user_id)
        )

        opportunities = service.get_opportunities(sample_user_id)

        if opportunities:
            opp_id = opportunities[0].get("id")
            success = service.mark_opportunity(sample_user_id, opp_id, "save")
            assert success == True

    def test_mark_opportunity_dismissed(self, service, sample_user_id, sample_profile):
        service.register_user_profile(sample_user_id, sample_profile)

        asyncio.get_event_loop().run_until_complete(
            service.scan_opportunities(sample_user_id)
        )

        opportunities = service.get_opportunities(sample_user_id)

        if opportunities:
            opp_id = opportunities[0].get("id")
            success = service.mark_opportunity(sample_user_id, opp_id, "dismiss")
            assert success == True

    def test_get_alerts(self, service, sample_user_id, sample_profile):
        service.register_user_profile(sample_user_id, sample_profile)

        asyncio.get_event_loop().run_until_complete(
            service.scan_opportunities(sample_user_id)
        )

        alerts = service.get_alerts(sample_user_id)

        assert isinstance(alerts, list)

    def test_scout_summary(self, service, sample_user_id, sample_profile):
        service.register_user_profile(sample_user_id, sample_profile)

        asyncio.get_event_loop().run_until_complete(
            service.scan_opportunities(sample_user_id)
        )

        summary = service.get_scout_summary(sample_user_id)

        assert "total_opportunities" in summary
        assert "profile_active" in summary

    def test_unregistered_user_scan(self, service):
        result = asyncio.get_event_loop().run_until_complete(
            service.scan_opportunities("unregistered_user")
        )

        assert result == []


class TestOpportunityMatching:
    @pytest.fixture
    def service(self):
        return OpportunityScoutService()

    def test_match_score_range(self, service, sample_user_id, sample_profile):
        service.register_user_profile(sample_user_id, sample_profile)

        opportunities = asyncio.get_event_loop().run_until_complete(
            service.scan_opportunities(sample_user_id)
        )

        for opp in opportunities:
            assert 0 <= opp.match_score <= 1

    def test_potential_regret_score_range(self, service, sample_user_id, sample_profile):
        service.register_user_profile(sample_user_id, sample_profile)

        opportunities = asyncio.get_event_loop().run_until_complete(
            service.scan_opportunities(sample_user_id)
        )

        for opp in opportunities:
            assert 0 <= opp.potential_regret_score <= 100

    def test_high_match_opportunities_first(self, service, sample_user_id, sample_profile):
        service.register_user_profile(sample_user_id, sample_profile)

        opportunities = asyncio.get_event_loop().run_until_complete(
            service.scan_opportunities(sample_user_id)
        )

        if len(opportunities) > 1:
            match_scores = [opp.match_score for opp in opportunities]
            assert match_scores == sorted(match_scores, reverse=True)
