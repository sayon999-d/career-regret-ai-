from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import json
from fastapi import WebSocket, WebSocketDisconnect

class MessageType(str, Enum):
    BIAS_ALERT = "bias_alert"
    OPPORTUNITY_ALERT = "opportunity_alert"
    TYPING_ANALYSIS = "typing_analysis"
    COLLABORATION_UPDATE = "collaboration_update"
    NOTIFICATION = "notification"
    HEARTBEAT = "heartbeat"
    SESSION_UPDATE = "session_update"

@dataclass
class WebSocketClient:
    websocket: WebSocket
    user_id: str
    connected_at: datetime = field(default_factory=datetime.utcnow)
    subscriptions: Set[str] = field(default_factory=set)
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocketClient]] = {}
        self.user_sessions: Dict[str, WebSocketClient] = {}
        self.collaboration_rooms: Dict[str, Set[str]] = {}

    async def connect(self, websocket: WebSocket, user_id: str) -> WebSocketClient:
        await websocket.accept()
        client = WebSocketClient(websocket=websocket, user_id=user_id)

        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        self.active_connections[user_id].append(client)
        self.user_sessions[f"{user_id}_{id(websocket)}"] = client

        await self.send_personal_message({
            "type": MessageType.NOTIFICATION.value,
            "message": "Connected to real-time updates",
            "timestamp": datetime.utcnow().isoformat()
        }, client)

        return client

    def disconnect(self, websocket: WebSocket, user_id: str):
        if user_id in self.active_connections:
            self.active_connections[user_id] = [
                c for c in self.active_connections[user_id]
                if c.websocket != websocket
            ]
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]

        session_key = f"{user_id}_{id(websocket)}"
        if session_key in self.user_sessions:
            del self.user_sessions[session_key]

    async def send_personal_message(self, message: Dict, client: WebSocketClient):
        try:
            await client.websocket.send_json(message)
        except Exception:
            pass

    async def broadcast_to_user(self, user_id: str, message: Dict):
        if user_id in self.active_connections:
            for client in self.active_connections[user_id]:
                await self.send_personal_message(message, client)

    async def broadcast_to_room(self, room_id: str, message: Dict, exclude_user: str = None):
        if room_id in self.collaboration_rooms:
            for user_id in self.collaboration_rooms[room_id]:
                if user_id != exclude_user:
                    await self.broadcast_to_user(user_id, message)

    async def broadcast_to_all(self, message: Dict):
        for user_id in self.active_connections:
            await self.broadcast_to_user(user_id, message)

    def join_room(self, user_id: str, room_id: str):
        if room_id not in self.collaboration_rooms:
            self.collaboration_rooms[room_id] = set()
        self.collaboration_rooms[room_id].add(user_id)

    def leave_room(self, user_id: str, room_id: str):
        if room_id in self.collaboration_rooms:
            self.collaboration_rooms[room_id].discard(user_id)
            if not self.collaboration_rooms[room_id]:
                del self.collaboration_rooms[room_id]

    def get_room_members(self, room_id: str) -> List[str]:
        return list(self.collaboration_rooms.get(room_id, set()))

    def get_connection_count(self) -> int:
        return sum(len(clients) for clients in self.active_connections.values())

    def get_user_connection_count(self, user_id: str) -> int:
        return len(self.active_connections.get(user_id, []))

class RealTimeService:
    def __init__(self, connection_manager: ConnectionManager):
        self.manager = connection_manager
        self.typing_buffers: Dict[str, str] = {}
        self.analysis_throttle: Dict[str, datetime] = {}

    async def send_bias_alert(self, user_id: str, bias_data: Dict):
        message = {
            "type": MessageType.BIAS_ALERT.value,
            "data": bias_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.manager.broadcast_to_user(user_id, message)

    async def send_opportunity_alert(self, user_id: str, opportunity_data: Dict):
        message = {
            "type": MessageType.OPPORTUNITY_ALERT.value,
            "data": opportunity_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.manager.broadcast_to_user(user_id, message)

    async def send_typing_analysis(self, user_id: str, analysis: Dict):
        now = datetime.utcnow()
        last_analysis = self.analysis_throttle.get(user_id)

        if last_analysis and (now - last_analysis).total_seconds() < 1:
            return

        self.analysis_throttle[user_id] = now

        message = {
            "type": MessageType.TYPING_ANALYSIS.value,
            "data": analysis,
            "timestamp": now.isoformat()
        }
        await self.manager.broadcast_to_user(user_id, message)

    async def send_collaboration_update(self, room_id: str, update_data: Dict, sender_id: str):
        message = {
            "type": MessageType.COLLABORATION_UPDATE.value,
            "sender": sender_id,
            "data": update_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.manager.broadcast_to_room(room_id, message, exclude_user=sender_id)

    async def send_notification(self, user_id: str, title: str, body: str, priority: str = "normal"):
        message = {
            "type": MessageType.NOTIFICATION.value,
            "title": title,
            "body": body,
            "priority": priority,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.manager.broadcast_to_user(user_id, message)

    async def process_typing_input(self, user_id: str, text: str, bias_interceptor) -> Optional[Dict]:
        if not text or len(text) < 20:
            return None

        self.typing_buffers[user_id] = text

        feedback = bias_interceptor.get_real_time_feedback(text, user_id)

        if feedback.get("has_bias"):
            await self.send_bias_alert(user_id, feedback)
            return feedback

        return None

class CollaborationService:
    def __init__(self, connection_manager: ConnectionManager, realtime_service: RealTimeService):
        self.manager = connection_manager
        self.realtime = realtime_service
        self.shared_decisions: Dict[str, Dict] = {}
        self.decision_comments: Dict[str, List[Dict]] = {}
        self.decision_votes: Dict[str, Dict[str, str]] = {}

    def create_shared_decision(self, decision_id: str, owner_id: str, decision_data: Dict) -> str:
        room_id = f"decision_{decision_id}"

        self.shared_decisions[decision_id] = {
            "owner_id": owner_id,
            "data": decision_data,
            "collaborators": [owner_id],
            "created_at": datetime.utcnow().isoformat(),
            "last_updated": datetime.utcnow().isoformat()
        }

        self.manager.join_room(owner_id, room_id)
        return room_id

    async def invite_collaborator(self, decision_id: str, inviter_id: str, invitee_id: str):
        if decision_id not in self.shared_decisions:
            return {"error": "Decision not found"}

        decision = self.shared_decisions[decision_id]
        if inviter_id != decision["owner_id"] and inviter_id not in decision["collaborators"]:
            return {"error": "Not authorized"}

        room_id = f"decision_{decision_id}"
        decision["collaborators"].append(invitee_id)
        self.manager.join_room(invitee_id, room_id)

        await self.realtime.send_notification(
            invitee_id,
            "Collaboration Invite",
            f"You've been invited to collaborate on a decision",
            priority="high"
        )

        return {"success": True, "room_id": room_id}

    async def add_comment(self, decision_id: str, user_id: str, comment: str):
        if decision_id not in self.decision_comments:
            self.decision_comments[decision_id] = []

        comment_data = {
            "id": f"comment_{len(self.decision_comments[decision_id])}",
            "user_id": user_id,
            "comment": comment,
            "timestamp": datetime.utcnow().isoformat()
        }

        self.decision_comments[decision_id].append(comment_data)

        room_id = f"decision_{decision_id}"
        await self.realtime.send_collaboration_update(
            room_id,
            {"action": "new_comment", "comment": comment_data},
            user_id
        )

        return comment_data

    async def vote_on_option(self, decision_id: str, user_id: str, option: str, vote: str):
        if decision_id not in self.decision_votes:
            self.decision_votes[decision_id] = {}

        self.decision_votes[decision_id][f"{user_id}_{option}"] = vote

        room_id = f"decision_{decision_id}"
        await self.realtime.send_collaboration_update(
            room_id,
            {"action": "vote", "user_id": user_id, "option": option, "vote": vote},
            user_id
        )

        return self.get_vote_summary(decision_id)

    def get_vote_summary(self, decision_id: str) -> Dict:
        if decision_id not in self.decision_votes:
            return {}

        summary = {}
        for key, vote in self.decision_votes[decision_id].items():
            _, option = key.rsplit("_", 1)
            if option not in summary:
                summary[option] = {"support": 0, "oppose": 0, "neutral": 0}
            summary[option][vote] = summary[option].get(vote, 0) + 1

        return summary

    def get_decision_details(self, decision_id: str) -> Optional[Dict]:
        if decision_id not in self.shared_decisions:
            return None

        return {
            **self.shared_decisions[decision_id],
            "comments": self.decision_comments.get(decision_id, []),
            "votes": self.get_vote_summary(decision_id)
        }

connection_manager = ConnectionManager()
realtime_service = RealTimeService(connection_manager)
collaboration_service = CollaborationService(connection_manager, realtime_service)
