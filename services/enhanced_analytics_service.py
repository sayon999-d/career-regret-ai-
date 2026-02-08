import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import statistics

from .database_service import db_service


class EnhancedAnalyticsService:
    """Comprehensive analytics and insights service"""
    
    def __init__(self):
        pass
    
    def get_dashboard_analytics(self, user_id: str) -> Dict:
        """Get comprehensive dashboard analytics"""
        return {
            "overview": self.get_overview_stats(user_id),
            "decision_patterns": self.get_decision_patterns(user_id),
            "emotion_analysis": self.get_emotion_analysis(user_id),
            "regret_trends": self.get_regret_trends(user_id),
            "activity_heatmap": self.get_activity_heatmap(user_id),
            "recommendations": self.get_personalized_recommendations(user_id)
        }
    
    def get_overview_stats(self, user_id: str) -> Dict:
        """Get high-level overview statistics"""
        decisions, total = db_service.get_decisions(user_id, limit=10000)
        
        if not decisions:
            return {
                "total_decisions": 0,
                "decisions_this_month": 0,
                "decisions_this_week": 0,
                "avg_predicted_regret": 0,
                "avg_actual_regret": 0,
                "prediction_accuracy": 0,
                "completed_decisions": 0,
                "pending_decisions": 0
            }
        
        now = datetime.utcnow()
        month_ago = now - timedelta(days=30)
        week_ago = now - timedelta(days=7)
        
        decisions_this_month = 0
        decisions_this_week = 0
        predicted_regrets = []
        actual_regrets = []
        completed = 0
        pending = 0
        
        for d in decisions:
            created = datetime.fromisoformat(d['created_at'].replace('Z', '+00:00')) if d.get('created_at') else now
            
            if created >= month_ago:
                decisions_this_month += 1
            if created >= week_ago:
                decisions_this_week += 1
            
            if d.get('predicted_regret') is not None:
                predicted_regrets.append(d['predicted_regret'])
            if d.get('actual_regret') is not None:
                actual_regrets.append(d['actual_regret'])
            
            if d.get('status') == 'completed':
                completed += 1
            elif d.get('status') in ['pending', 'in_progress']:
                pending += 1
        
        prediction_accuracy = 0
        if predicted_regrets and actual_regrets:
            paired = [(p, a) for p, a in zip(predicted_regrets, actual_regrets) if p is not None and a is not None]
            if paired:
                errors = [abs(p - a) for p, a in paired]
                prediction_accuracy = max(0, 100 - statistics.mean(errors))
        
        return {
            "total_decisions": len(decisions),
            "decisions_this_month": decisions_this_month,
            "decisions_this_week": decisions_this_week,
            "avg_predicted_regret": round(statistics.mean(predicted_regrets), 1) if predicted_regrets else 0,
            "avg_actual_regret": round(statistics.mean(actual_regrets), 1) if actual_regrets else 0,
            "prediction_accuracy": round(prediction_accuracy, 1),
            "completed_decisions": completed,
            "pending_decisions": pending
        }
    
    def get_decision_patterns(self, user_id: str) -> Dict:
        """Analyze decision-making patterns"""
        decisions, _ = db_service.get_decisions(user_id, limit=10000)
        
        if not decisions:
            return {"types": [], "time_of_day": [], "day_of_week": [], "monthly_trend": []}
        
        type_counts = defaultdict(int)
        for d in decisions:
            dtype = d.get('decision_type', 'general')
            type_counts[dtype] += 1
        
        types = [{"type": k, "count": v} for k, v in sorted(type_counts.items(), key=lambda x: -x[1])]
        
        hour_counts = defaultdict(int)
        for d in decisions:
            if d.get('created_at'):
                try:
                    dt = datetime.fromisoformat(d['created_at'].replace('Z', '+00:00'))
                    hour_counts[dt.hour] += 1
                except:
                    pass
        
        time_of_day = [{"hour": h, "count": hour_counts.get(h, 0)} for h in range(24)]
        
        day_counts = defaultdict(int)
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for d in decisions:
            if d.get('created_at'):
                try:
                    dt = datetime.fromisoformat(d['created_at'].replace('Z', '+00:00'))
                    day_counts[dt.weekday()] += 1
                except:
                    pass
        
        day_of_week = [{"day": day_names[i], "count": day_counts.get(i, 0)} for i in range(7)]
        
        monthly_counts = defaultdict(int)
        for d in decisions:
            if d.get('created_at'):
                try:
                    dt = datetime.fromisoformat(d['created_at'].replace('Z', '+00:00'))
                    month_key = dt.strftime('%Y-%m')
                    monthly_counts[month_key] += 1
                except:
                    pass
        
        monthly_trend = [
            {"month": k, "count": v} 
            for k, v in sorted(monthly_counts.items())[-12:]
        ]
        
        return {
            "types": types,
            "time_of_day": time_of_day,
            "day_of_week": day_of_week,
            "monthly_trend": monthly_trend
        }
    
    def get_emotion_analysis(self, user_id: str) -> Dict:
        """Analyze emotions associated with decisions"""
        decisions, _ = db_service.get_decisions(user_id, limit=10000)
        
        emotion_counts = defaultdict(int)
        emotion_regret = defaultdict(list)
        
        for d in decisions:
            emotions = d.get('emotions', [])
            if isinstance(emotions, str):
                try:
                    emotions = json.loads(emotions)
                except:
                    emotions = []
            
            regret = d.get('predicted_regret', 50)
            
            for e in emotions:
                if isinstance(e, dict):
                    emotion = e.get('emotion', 'neutral')
                else:
                    emotion = str(e)
                emotion_counts[emotion] += 1
                emotion_regret[emotion].append(regret)
        
        emotion_data = []
        for emotion, count in emotion_counts.items():
            avg_regret = statistics.mean(emotion_regret[emotion]) if emotion_regret[emotion] else 50
            emotion_data.append({
                "emotion": emotion,
                "count": count,
                "avg_regret": round(avg_regret, 1)
            })
        
        emotion_data.sort(key=lambda x: -x['count'])
        
        biases = []
        if emotion_data:
            most_common = emotion_data[0]['emotion']
            if emotion_data[0]['count'] > len(decisions) * 0.3:
                biases.append({
                    "type": "emotion_bias",
                    "description": f"You frequently make decisions while feeling {most_common}",
                    "recommendation": f"Consider decisions made when feeling {most_common} more carefully"
                })
        
        return {
            "emotions": emotion_data[:10],
            "biases": biases
        }
    
    def get_regret_trends(self, user_id: str) -> Dict:
        """Analyze regret prediction vs actual over time"""
        decisions, _ = db_service.get_decisions(user_id, limit=10000)
        
        weekly_data = defaultdict(lambda: {"predicted": [], "actual": []})
        
        for d in decisions:
            if d.get('created_at'):
                try:
                    dt = datetime.fromisoformat(d['created_at'].replace('Z', '+00:00'))
                    week_key = dt.strftime('%Y-W%W')
                    
                    if d.get('predicted_regret') is not None:
                        weekly_data[week_key]["predicted"].append(d['predicted_regret'])
                    if d.get('actual_regret') is not None:
                        weekly_data[week_key]["actual"].append(d['actual_regret'])
                except:
                    pass
        
        trends = []
        for week, data in sorted(weekly_data.items())[-12:]:
            trends.append({
                "week": week,
                "avg_predicted": round(statistics.mean(data["predicted"]), 1) if data["predicted"] else None,
                "avg_actual": round(statistics.mean(data["actual"]), 1) if data["actual"] else None,
                "count": len(data["predicted"])
            })
        
        improvement = 0
        if len(trends) >= 4:
            early_regrets = [t['avg_predicted'] for t in trends[:len(trends)//2] if t['avg_predicted'] is not None]
            late_regrets = [t['avg_predicted'] for t in trends[len(trends)//2:] if t['avg_predicted'] is not None]
            if early_regrets and late_regrets:
                improvement = statistics.mean(early_regrets) - statistics.mean(late_regrets)
        
        return {
            "weekly_trends": trends,
            "improvement": round(improvement, 1),
            "improving": improvement > 0
        }
    
    def get_activity_heatmap(self, user_id: str) -> Dict:
        """Generate activity heatmap data"""
        decisions, _ = db_service.get_decisions(user_id, limit=10000)
        events = db_service.get_calendar_events(user_id)
        
        activity = {}
        now = datetime.utcnow()
        
        for i in range(90):
            day = (now - timedelta(days=i)).strftime('%Y-%m-%d')
            activity[day] = 0
        
        for d in decisions:
            if d.get('created_at'):
                try:
                    day = d['created_at'][:10]
                    if day in activity:
                        activity[day] += 1
                except:
                    pass
        
        for e in events:
            if e.get('start_time'):
                try:
                    day = e['start_time'][:10]
                    if day in activity:
                        activity[day] += 1
                except:
                    pass
        
        heatmap = [{"date": k, "count": v} for k, v in sorted(activity.items())]
        
        return {
            "heatmap": heatmap,
            "max_count": max(activity.values()) if activity else 0
        }
    
    def get_personalized_recommendations(self, user_id: str) -> List[Dict]:
        """Generate personalized recommendations based on user patterns"""
        recommendations = []
        
        decisions, _ = db_service.get_decisions(user_id, limit=10000)
        
        if not decisions:
            recommendations.append({
                "type": "getting_started",
                "title": "Start Tracking Decisions",
                "description": "Begin by recording your first career decision to get personalized insights.",
                "priority": "high"
            })
            return recommendations
        
        pending = [d for d in decisions if d.get('status') == 'pending']
        if len(pending) > 5:
            recommendations.append({
                "type": "pending_overload",
                "title": "Too Many Pending Decisions",
                "description": f"You have {len(pending)} pending decisions. Consider addressing some of them.",
                "priority": "high"
            })
        
        with_outcomes = [d for d in decisions if d.get('actual_regret') is not None]
        if len(decisions) > 10 and len(with_outcomes) < len(decisions) * 0.3:
            recommendations.append({
                "type": "outcome_tracking",
                "title": "Track More Outcomes",
                "description": "Recording actual outcomes helps improve future regret predictions.",
                "priority": "medium"
            })
        
        with_emotions = [d for d in decisions if d.get('emotions') and len(d['emotions']) > 0]
        if len(decisions) > 5 and len(with_emotions) < len(decisions) * 0.5:
            recommendations.append({
                "type": "emotion_detection",
                "title": "Use Emotion Detection",
                "description": "Capturing your emotional state leads to better decision insights.",
                "priority": "low"
            })
        
        now = datetime.utcnow()
        week_ago = now - timedelta(days=7)
        recent = [d for d in decisions if d.get('created_at') and 
                  datetime.fromisoformat(d['created_at'].replace('Z', '+00:00')) >= week_ago]
        
        if len(recent) == 0 and len(decisions) > 0:
            recommendations.append({
                "type": "inactivity",
                "title": "Stay Consistent",
                "description": "You haven't recorded a decision in a week. Regular tracking improves insights.",
                "priority": "medium"
            })
        
        return recommendations[:5]
    
    def generate_report(self, user_id: str, report_type: str = "monthly") -> Dict:
        """Generate comprehensive report"""
        analytics = self.get_dashboard_analytics(user_id)
        
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "report_type": report_type,
            "period": self._get_report_period(report_type),
            "summary": analytics["overview"],
            "patterns": analytics["decision_patterns"],
            "emotions": analytics["emotion_analysis"],
            "trends": analytics["regret_trends"],
            "recommendations": analytics["recommendations"]
        }
        
        report["narrative"] = self._generate_narrative(report)
        
        return report
    
    def _get_report_period(self, report_type: str) -> Dict:
        """Get report period based on type"""
        now = datetime.utcnow()
        
        if report_type == "weekly":
            start = now - timedelta(days=7)
        elif report_type == "monthly":
            start = now - timedelta(days=30)
        elif report_type == "quarterly":
            start = now - timedelta(days=90)
        else:
            start = now - timedelta(days=30)
        
        return {
            "start": start.isoformat(),
            "end": now.isoformat(),
            "days": (now - start).days
        }
    
    def _generate_narrative(self, report: Dict) -> str:
        """Generate human-readable narrative summary"""
        summary = report.get("summary", {})
        trends = report.get("trends", {})
        
        total = summary.get("total_decisions", 0)
        avg_regret = summary.get("avg_predicted_regret", 0)
        accuracy = summary.get("prediction_accuracy", 0)
        improving = trends.get("improving", False)
        
        narrative = f"""

During this period, you tracked **{total} decisions** with an average predicted regret of **{avg_regret}%**.

"""
        
        if accuracy > 70:
            narrative += f"\n- ✅ Excellent prediction accuracy at {accuracy}%! You have good self-awareness."
        elif accuracy > 50:
            narrative += f"\n- 📊 Moderate prediction accuracy at {accuracy}%. Consider tracking more outcomes."
        else:
            narrative += f"\n- ⚠️ Prediction accuracy is {accuracy}%. More outcome tracking will help calibrate predictions."
        
        if improving:
            narrative += f"\n- 📈 Your regret predictions are improving over time. Great progress!"
        else:
            narrative += f"\n- 📉 Consider reviewing past decisions to improve future predictions."
        
        recommendations = report.get("recommendations", [])
        if recommendations:
            narrative += "\n\n### Recommendations\n"
            for rec in recommendations[:3]:
                narrative += f"\n- **{rec['title']}**: {rec['description']}"
        
        return narrative


# Global instance
enhanced_analytics_service = EnhancedAnalyticsService()
