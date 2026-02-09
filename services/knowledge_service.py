import hashlib
from typing import List, Dict, Any
from datetime import datetime

class KnowledgeService:
    def __init__(self):
        self.documents: Dict[str, List[Dict]] = {}

    def add_document(self, user_id: str, filename: str, content: str, doc_type: str = "general") -> Dict:
        """Saves a document to the user's knowledge base."""
        doc_id = hashlib.sha256(f"{filename}{datetime.utcnow()}".encode()).hexdigest()[:10]

        doc = {
            "id": doc_id,
            "filename": filename,
            "type": doc_type,
            "chars": len(content),
            "added_at": datetime.utcnow().isoformat(),
            "summary": f"Document about {filename} with {len(content)} characters of specialized knowledge."
        }

        if user_id not in self.documents:
            self.documents[user_id] = []

        self.documents[user_id].append(doc)
        return doc

    def get_documents(self, user_id: str) -> List[Dict]:
        return self.documents.get(user_id, [])

    def delete_document(self, user_id: str, doc_id: str) -> bool:
        if user_id not in self.documents:
            return False

        initial_len = len(self.documents[user_id])
        self.documents[user_id] = [d for d in self.documents[user_id] if d["id"] != doc_id]
        return len(self.documents[user_id]) < initial_len

knowledge_service = KnowledgeService()
