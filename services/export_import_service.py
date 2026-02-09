import os
import json
import csv
import io
import secrets
from datetime import datetime
from typing import Dict, List, Optional, BinaryIO
from dataclasses import dataclass
import zipfile
import tempfile

from .database_service import db_service


class ExportFormat:
    JSON = "json"
    CSV = "csv"
    ICS = "ics"
    ZIP = "zip"


class ExportImportService:
    """Handles all data export and import functionality"""
    
    def __init__(self):
        default_export_dir = os.path.join(tempfile.gettempdir(), "exports")
        self.export_dir = os.getenv("EXPORT_DIR", default_export_dir)
        os.makedirs(self.export_dir, exist_ok=True)
    
    
    def export_all_data(self, user_id: str, format: str = ExportFormat.JSON) -> Dict:
        """Export all user data"""
        data = db_service.export_user_data(user_id)
        
        if format == ExportFormat.JSON:
            return self._export_json(data, f"career_data_{user_id}")
        elif format == ExportFormat.ZIP:
            return self._export_zip(user_id, data)
        else:
            return self._export_json(data, f"career_data_{user_id}")
    
    def export_decisions(
        self,
        user_id: str,
        format: str = ExportFormat.JSON,
        status: str = None,
        decision_type: str = None
    ) -> Dict:
        """Export decisions to specified format"""
        decisions, _ = db_service.get_decisions(
            user_id,
            status=status,
            decision_type=decision_type,
            limit=10000
        )
        
        if format == ExportFormat.CSV:
            return self._export_decisions_csv(decisions, user_id)
        else:
            return self._export_json({"decisions": decisions, "exported_at": datetime.utcnow().isoformat()}, 
                                    f"decisions_{user_id}")
    
    def export_calendar_events(
        self,
        user_id: str,
        format: str = ExportFormat.ICS,
        start_date: str = None,
        end_date: str = None
    ) -> Dict:
        """Export calendar events to specified format"""
        events = db_service.get_calendar_events(user_id, start_date, end_date)
        
        if format == ExportFormat.ICS:
            return self._export_events_ics(events, user_id)
        elif format == ExportFormat.CSV:
            return self._export_events_csv(events, user_id)
        else:
            return self._export_json({"events": events, "exported_at": datetime.utcnow().isoformat()}, 
                                    f"events_{user_id}")
    
    def export_conversations(self, user_id: str) -> Dict:
        """Export all conversations"""
        conversations = db_service.get_conversations(user_id, limit=10000)
        full_conversations = []
        
        for conv in conversations:
            full_conv = db_service.get_conversation(user_id, conv['id'])
            if full_conv:
                full_conversations.append(full_conv)
        
        return self._export_json({
            "conversations": full_conversations,
            "exported_at": datetime.utcnow().isoformat()
        }, f"conversations_{user_id}")
    
    def _export_json(self, data: Dict, filename: str) -> Dict:
        """Export data as JSON"""
        json_str = json.dumps(data, indent=2, default=str)
        filepath = os.path.join(self.export_dir, f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(filepath, 'w') as f:
            f.write(json_str)
        
        return {
            "format": "json",
            "filepath": filepath,
            "filename": os.path.basename(filepath),
            "size_bytes": len(json_str.encode('utf-8')),
            "content": json_str,
            "content_type": "application/json"
        }
    
    def _export_decisions_csv(self, decisions: List[Dict], user_id: str) -> Dict:
        """Export decisions as CSV"""
        output = io.StringIO()
        
        fieldnames = [
            'id', 'title', 'description', 'decision_type', 'status',
            'predicted_regret', 'actual_regret', 'confidence',
            'chosen_option', 'notes', 'created_at', 'updated_at', 'decided_at'
        ]
        
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        
        for decision in decisions:
            row = {k: decision.get(k, '') for k in fieldnames}
            for field in ['emotions', 'factors', 'pros', 'cons', 'alternatives', 'tags']:
                if field in decision and isinstance(decision[field], list):
                    row[field] = '; '.join(str(x) for x in decision[field])
            writer.writerow(row)
        
        csv_content = output.getvalue()
        filepath = os.path.join(self.export_dir, f"decisions_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        
        with open(filepath, 'w') as f:
            f.write(csv_content)
        
        return {
            "format": "csv",
            "filepath": filepath,
            "filename": os.path.basename(filepath),
            "size_bytes": len(csv_content.encode('utf-8')),
            "content": csv_content,
            "content_type": "text/csv"
        }
    
    def _export_events_ics(self, events: List[Dict], user_id: str) -> Dict:
        """Export events as ICS (iCalendar) format"""
        ics_lines = [
            "BEGIN:VCALENDAR",
            "VERSION:2.0",
            "PRODID:-//Career Decision Regret AI//EN",
            "CALSCALE:GREGORIAN",
            "METHOD:PUBLISH"
        ]
        
        for event in events:
            start_time = event.get('start_time', '')
            end_time = event.get('end_time', start_time)
            
            def format_ics_date(date_str):
                if date_str:
                    try:
                        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        return dt.strftime('%Y%m%dT%H%M%SZ')
                    except:
                        return datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
                return datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
            
            ics_lines.extend([
                "BEGIN:VEVENT",
                f"UID:{event.get('id', secrets.token_hex(8))}@careerdecisionai.local",
                f"DTSTART:{format_ics_date(start_time)}",
                f"DTEND:{format_ics_date(end_time)}",
                f"SUMMARY:{event.get('title', 'Untitled Event')}",
                f"DESCRIPTION:{event.get('description', '')}",
                f"LOCATION:{event.get('location', '')}",
                f"CATEGORIES:{event.get('event_type', 'general')}",
                "STATUS:CONFIRMED",
                "END:VEVENT"
            ])
        
        ics_lines.append("END:VCALENDAR")
        ics_content = '\r\n'.join(ics_lines)
        
        filepath = os.path.join(self.export_dir, f"calendar_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ics")
        
        with open(filepath, 'w') as f:
            f.write(ics_content)
        
        return {
            "format": "ics",
            "filepath": filepath,
            "filename": os.path.basename(filepath),
            "size_bytes": len(ics_content.encode('utf-8')),
            "content": ics_content,
            "content_type": "text/calendar"
        }
    
    def _export_events_csv(self, events: List[Dict], user_id: str) -> Dict:
        """Export events as CSV"""
        output = io.StringIO()
        
        fieldnames = ['id', 'title', 'description', 'event_type', 'start_time', 'end_time', 'location', 'synced']
        
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        
        for event in events:
            writer.writerow({k: event.get(k, '') for k in fieldnames})
        
        csv_content = output.getvalue()
        filepath = os.path.join(self.export_dir, f"events_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        
        with open(filepath, 'w') as f:
            f.write(csv_content)
        
        return {
            "format": "csv",
            "filepath": filepath,
            "filename": os.path.basename(filepath),
            "size_bytes": len(csv_content.encode('utf-8')),
            "content": csv_content,
            "content_type": "text/csv"
        }
    
    def _export_zip(self, user_id: str, data: Dict) -> Dict:
        """Export all data as a ZIP archive"""
        zip_path = os.path.join(self.export_dir, f"career_backup_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr("data.json", json.dumps(data, indent=2, default=str))
            
            if data.get('decisions'):
                csv_result = self._export_decisions_csv(data['decisions'], user_id)
                zipf.writestr("decisions.csv", csv_result['content'])
            
            if data.get('calendar_events'):
                ics_result = self._export_events_ics(data['calendar_events'], user_id)
                zipf.writestr("calendar.ics", ics_result['content'])
            
            readme = f"""Career Decision Regret AI - Data Export
========================================

Exported: {datetime.utcnow().isoformat()}
User ID: {user_id}

Contents:
- data.json: Complete data export in JSON format
- decisions.csv: Decisions in CSV format (spreadsheet compatible)
- calendar.ics: Calendar events in iCalendar format

To import this data, use the Import feature in the application.
"""
            zipf.writestr("README.txt", readme)
        
        with open(zip_path, 'rb') as f:
            zip_content = f.read()
        
        return {
            "format": "zip",
            "filepath": zip_path,
            "filename": os.path.basename(zip_path),
            "size_bytes": len(zip_content),
            "content_type": "application/zip"
        }
    
    # ============ IMPORT FUNCTIONS ============
    
    def import_data(self, user_id: str, data: Dict) -> Dict:
        """Import data from JSON"""
        return db_service.import_user_data(user_id, data)
    
    def import_from_json(self, user_id: str, json_content: str) -> Dict:
        """Import data from JSON string"""
        try:
            data = json.loads(json_content)
            return self.import_data(user_id, data)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON format"}
    
    def import_from_file(self, user_id: str, file_path: str) -> Dict:
        """Import data from file"""
        if file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                return self.import_from_json(user_id, f.read())
        elif file_path.endswith('.zip'):
            return self._import_from_zip(user_id, file_path)
        else:
            return {"error": "Unsupported file format"}
    
    def _import_from_zip(self, user_id: str, zip_path: str) -> Dict:
        """Import data from ZIP archive"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                if 'data.json' in zipf.namelist():
                    data = json.loads(zipf.read('data.json').decode('utf-8'))
                    return self.import_data(user_id, data)
                else:
                    return {"error": "No data.json found in archive"}
        except Exception as e:
            return {"error": str(e)}
    
    def import_decisions_csv(self, user_id: str, csv_content: str) -> Dict:
        """Import decisions from CSV"""
        imported = 0
        errors = []
        
        reader = csv.DictReader(io.StringIO(csv_content))
        
        for row in reader:
            try:
                decision_data = {
                    'title': row.get('title', 'Imported Decision'),
                    'description': row.get('description', ''),
                    'decision_type': row.get('decision_type', row.get('type', 'general')),
                    'status': row.get('status', 'pending'),
                    'notes': row.get('notes', f"Imported from CSV")
                }
                
                if row.get('predicted_regret'):
                    try:
                        decision_data['predicted_regret'] = float(row['predicted_regret'])
                    except:
                        pass
                
                db_service.create_decision(user_id, decision_data)
                imported += 1
            except Exception as e:
                errors.append(str(e))
        
        return {
            "imported": imported,
            "errors": errors[:10]
        }
    
    def import_calendar_ics(self, user_id: str, ics_content: str) -> Dict:
        """Import calendar events from ICS format"""
        imported = 0
        errors = []
        
        current_event = {}
        in_event = False
        
        for line in ics_content.split('\n'):
            line = line.strip()
            
            if line == 'BEGIN:VEVENT':
                in_event = True
                current_event = {}
            elif line == 'END:VEVENT':
                in_event = False
                try:
                    event_data = {
                        'title': current_event.get('SUMMARY', 'Imported Event'),
                        'description': current_event.get('DESCRIPTION', ''),
                        'location': current_event.get('LOCATION', ''),
                        'event_type': current_event.get('CATEGORIES', 'general'),
                        'start_time': self._parse_ics_date(current_event.get('DTSTART', '')),
                        'end_time': self._parse_ics_date(current_event.get('DTEND', ''))
                    }
                    db_service.create_calendar_event(user_id, event_data)
                    imported += 1
                except Exception as e:
                    errors.append(str(e))
            elif in_event and ':' in line:
                key, value = line.split(':', 1)
                if ';' in key:
                    key = key.split(';')[0]
                current_event[key] = value
        
        return {
            "imported": imported,
            "errors": errors[:10]
        }
    
    def _parse_ics_date(self, date_str: str) -> str:
        """Parse ICS date format to ISO format"""
        if not date_str:
            return datetime.utcnow().isoformat()
        
        try:
            date_str = date_str.replace('Z', '')
            if len(date_str) == 8:
                dt = datetime.strptime(date_str, '%Y%m%d')
            elif len(date_str) == 15:
                dt = datetime.strptime(date_str, '%Y%m%dT%H%M%S')
            else:
                return datetime.utcnow().isoformat()
            return dt.isoformat()
        except:
            return datetime.utcnow().isoformat()
    
    
    def create_backup(self, user_id: str = None) -> Dict:
        """Create a full backup"""
        if user_id:
            data = db_service.export_user_data(user_id)
            return self._export_zip(user_id, data)
        else:
            return {"error": "Full system backup not implemented"}
    
    def list_backups(self, user_id: str = None) -> List[Dict]:
        """List available backups"""
        backups = []
        
        for filename in os.listdir(self.export_dir):
            if user_id and user_id not in filename:
                continue
            
            filepath = os.path.join(self.export_dir, filename)
            stat = os.stat(filepath)
            
            backups.append({
                "filename": filename,
                "filepath": filepath,
                "size_bytes": stat.st_size,
                "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat()
            })
        
        return sorted(backups, key=lambda x: x['created_at'], reverse=True)


export_import_service = ExportImportService()
