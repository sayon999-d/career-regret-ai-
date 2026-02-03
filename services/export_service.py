import io
import base64
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

class ExportFormat(str, Enum):
    PDF = "pdf"
    JSON = "json"
    CSV = "csv"
    MARKDOWN = "markdown"

@dataclass
class ReportSection:

    title: str
    content: str
    data: Optional[Dict] = None
    chart_type: Optional[str] = None

@dataclass
class GeneratedReport:

    id: str
    user_id: str
    report_type: str
    title: str
    sections: List[ReportSection]
    summary: str
    generated_at: datetime = field(default_factory=datetime.utcnow)
    format: ExportFormat = ExportFormat.JSON

class ExportService:


    def __init__(self):
        self.generated_reports: Dict[str, GeneratedReport] = {}

    def generate_decision_report(
        self,
        user_id: str,
        decision_data: Dict[str, Any],
        analysis_result: Dict[str, Any],
        nlp_analysis: Dict[str, Any] = None,
        simulation_result: Dict[str, Any] = None
    ) -> GeneratedReport:

        report_id = f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        sections = []

        sections.append(ReportSection(
            title="Executive Summary",
            content=self._generate_executive_summary(decision_data, analysis_result)
        ))

        sections.append(ReportSection(
            title="Decision Overview",
            content="Detailed overview of the decision and its context.".strip(),
            data=decision_data
        ))

        prediction = analysis_result.get('prediction', {})
        sections.append(ReportSection(
            title="Regret Prediction Analysis",
            content="Analysis of predicted regret and risk levels based on ML models.".strip(),
            data=prediction,
            chart_type='gauge'
        ))

        if nlp_analysis:
            sections.append(ReportSection(
                title="Natural Language Analysis",
                content="Deep dive into the emotional and linguistic patterns detected in your decision description.".strip(),
                data=nlp_analysis
            ))

        if simulation_result:
            salary_proj = simulation_result.get('salary_projections', {})
            sections.append(ReportSection(
                title="Career Path Simulation",
                content="Projections of potential career outcomes and salary trends over a 5-year period.".strip(),
                data=simulation_result,
                chart_type='line'
            ))

        sections.append(ReportSection(
            title="Recommended Next Steps",
            content=self._generate_action_items(decision_data, analysis_result)
        ))

        summary = self._generate_report_summary(decision_data, analysis_result)

        report = GeneratedReport(
            id=report_id,
            user_id=user_id,
            report_type="decision_analysis",
            title=f"Decision Analysis: {decision_data.get('decision_type', 'Unknown').replace('_', ' ').title()}",
            sections=sections,
            summary=summary
        )

        self.generated_reports[report_id] = report
        return report

    def _generate_executive_summary(self, decision_data: Dict, analysis: Dict) -> str:

        prediction = analysis.get('prediction', {})
        regret = prediction.get('predicted_regret', 0.5)
        risk = prediction.get('risk_level', 'moderate')

        if regret < 0.3:
            outlook = "This decision shows low regret potential and appears well-aligned with your goals."
        elif regret < 0.5:
            outlook = "This decision carries moderate risk but has reasonable prospects for success."
        else:
            outlook = "This decision may carry significant risk. Careful consideration is recommended."

        return f"Executive summary for decision: {decision_data.get('decision_type')}".strip()

    def _format_factors(self, factors: List) -> str:

        if not factors:
            return "- No significant factors identified"

        lines = []
        for factor in factors[:5]:
            if isinstance(factor, (list, tuple)) and len(factor) >= 2:
                name = factor[0].replace('_', ' ').title()
                score = factor[1]
                lines.append(f"- {name}: {score:.2%}")
            else:
                lines.append(f"- {factor}")

        return '\n'.join(lines)

    def _format_list(self, items: List[str]) -> str:

        if not items:
            return "- No specific recommendations"
        return '\n'.join(f"- {item}" for item in items[:7])

    def _generate_action_items(self, decision_data: Dict, analysis: Dict) -> str:

        decision_type = decision_data.get('decision_type', 'job_change')
        risk = analysis.get('prediction', {}).get('risk_level', 'moderate')

        actions = {
            'job_change': [
                "Research the company culture through Glassdoor and LinkedIn",
                "Connect with current or former employees for insights",
                "Prepare negotiation points for salary and benefits",
                "Update your resume and LinkedIn profile",
                "Plan your transition timeline"
            ],
            'career_switch': [
                "Identify and address skill gaps through courses or projects",
                "Build a portfolio demonstrating relevant work",
                "Network with professionals in your target field",
                "Consider informational interviews",
                "Create a 6-month financial runway"
            ],
            'startup': [
                "Verify the company's funding and runway",
                "Research the founders' backgrounds and track record",
                "Understand the cap table and equity terms fully",
                "Assess the market opportunity independently",
                "Talk to current employees about culture and workload"
            ]
        }

        items = actions.get(decision_type, actions['job_change'])

        if risk == 'high':
            items.insert(0, "**CRITICAL:** Given the high risk level, proceed with extra caution")

        return '\n'.join(f"{i+1}. {item}" for i, item in enumerate(items))

    def _generate_report_summary(self, decision_data: Dict, analysis: Dict) -> str:

        prediction = analysis.get('prediction', {})
        regret = prediction.get('predicted_regret', 0.5)

        if regret < 0.3:
            verdict = "proceed with confidence"
        elif regret < 0.5:
            verdict = "proceed with preparation"
        elif regret < 0.7:
            verdict = "carefully consider all factors before deciding"
        else:
            verdict = "reconsider or seek additional information"

        return f"Based on comprehensive analysis, our recommendation is to {verdict}."

    def generate_journal_summary_report(
        self,
        user_id: str,
        entries: List[Dict],
        accuracy_metrics: Dict
    ) -> GeneratedReport:

        report_id = f"journal_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        sections = []

        sections.append(ReportSection(
            title="Journal Overview",
            content="High-level overview of your decision journal history and tracking performance.".strip(),
            data=accuracy_metrics
        ))

        type_counts = {}
        for entry in entries:
            dt = entry.get('decision_type', 'unknown')
            type_counts[dt] = type_counts.get(dt, 0) + 1

        type_breakdown = '\n'.join(f"- {dt.replace('_', ' ').title()}: {count}" for dt, count in type_counts.items())

        sections.append(ReportSection(
            title="Decisions by Type",
            content=type_breakdown or "No decisions recorded yet",
            chart_type='pie',
            data=type_counts
        ))

        sections.append(ReportSection(
            title="Key Insights",
            content=self._generate_journal_insights(entries, accuracy_metrics)
        ))

        report = GeneratedReport(
            id=report_id,
            user_id=user_id,
            report_type="journal_summary",
            title="Decision Journal Summary Report",
            sections=sections,
            summary=f"Summary of {len(entries)} decisions tracked with {accuracy_metrics.get('accuracy', 0) * 100:.0f}% prediction accuracy."
        )

        self.generated_reports[report_id] = report
        return report

    def _generate_journal_insights(self, entries: List[Dict], metrics: Dict) -> str:

        insights = []

        if metrics.get('accuracy'):
            acc = metrics['accuracy']
            if acc > 0.8:
                insights.append("Your predictions have been highly accurate - trust your judgment!")
            elif acc > 0.6:
                insights.append("Your prediction accuracy is good - continue tracking for improvement.")
            else:
                insights.append("Consider gathering more information before making predictions.")

        if metrics.get('repeat_decision_rate'):
            rate = metrics['repeat_decision_rate']
            if rate > 0.7:
                insights.append(f"{rate * 100:.0f}% of your decisions you would make again - great decision-making!")
            elif rate < 0.4:
                insights.append("Consider what factors led to regretted decisions.")

        if not insights:
            insights.append("Continue tracking decisions to generate personalized insights.")

        return '\n'.join(f"• {insight}" for insight in insights)

    def export_to_json(self, report: GeneratedReport) -> str:

        return json.dumps(self.to_dict(report), indent=2)

    def export_to_markdown(self, report: GeneratedReport) -> str:

        md = []
        md.append(f"# {report.title}")
        md.append(f"\n*Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')} UTC*\n")
        md.append(f"**Report ID:** {report.id}\n")

        for section in report.sections:
            md.append(f"## {section.title}")
            md.append(section.content)
            md.append("")

        md.append("\n---")
        md.append(f"\n**Summary:** {report.summary}")

        return '\n'.join(md)

    def export_to_csv(self, entries: List[Dict]) -> str:

        if not entries:
            return "No data to export"

        all_keys = set()
        for entry in entries:
            all_keys.update(entry.keys())

        keys = sorted([k for k in all_keys if k not in ['nlp_analysis', 'factors']])

        lines = [','.join(keys)]

        for entry in entries:
            values = []
            for key in keys:
                val = entry.get(key, '')
                if isinstance(val, (list, dict)):
                    val = str(val).replace(',', ';')
                elif isinstance(val, datetime):
                    val = val.isoformat()
                values.append(f'"{val}"')
            lines.append(','.join(values))

        return '\n'.join(lines)

    def get_calendar_events(
        self,
        entries: List[Dict],
        follow_ups: List[Dict]
    ) -> List[Dict[str, Any]]:

        events = []

        for fu in follow_ups:
            events.append({
                'title': f"Decision Follow-up: {fu.get('title', 'Review Decision')[:30]}",
                'description': f"Time to review your decision: {fu.get('title', 'No title')}",
                'date': fu.get('scheduled_date'),
                'type': 'follow_up',
                'decision_id': fu.get('decision_id')
            })

        return events

    def generate_ical(self, events: List[Dict]) -> str:

        ical = []
        ical.append("BEGIN:VCALENDAR")
        ical.append("VERSION:2.0")
        ical.append("PRODID:-//Career Decision AI//EN")

        for event in events:
            ical.append("BEGIN:VEVENT")
            date = event.get('date', datetime.utcnow())
            if isinstance(date, str):
                date_str = date.replace('-', '').replace(':', '').split('.')[0]
            else:
                date_str = date.strftime('%Y%m%dT%H%M%SZ')

            ical.append(f"DTSTART:{date_str}")
            ical.append(f"SUMMARY:{event.get('title', 'Decision Review')}")
            ical.append(f"DESCRIPTION:{event.get('description', '')}")
            ical.append("END:VEVENT")

        ical.append("END:VCALENDAR")
        return '\n'.join(ical)

    def to_dict(self, report: GeneratedReport) -> Dict[str, Any]:

        return {
            'id': report.id,
            'user_id': report.user_id,
            'report_type': report.report_type,
            'title': report.title,
            'sections': [
                {
                    'title': s.title,
                    'content': s.content,
                    'chart_type': s.chart_type,
                    'data': s.data
                }
                for s in report.sections
            ],
            'summary': report.summary,
            'generated_at': report.generated_at.isoformat(),
            'format': report.format.value
        }
