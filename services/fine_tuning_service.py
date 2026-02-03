import json
import os
from typing import List, Dict, Any
from datetime import datetime

class FineTuningService:
    def __init__(self, data_path: str = "data/fine_tuning"):
        self.data_path = data_path
        os.makedirs(data_path, exist_ok=True)
        self.current_model = "base-career-model"
        self.is_finetuned_active = False

    def collect_training_pair(self, user_id: str, prompt: str, completion: str, feedback_score: int):
        """Collect high-quality interactions for future fine-tuning."""
        if feedback_score < 4:
            return

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "instruction": prompt,
            "response": completion,
            "quality_score": feedback_score
        }

        file_path = os.path.join(self.data_path, f"user_interactions.jsonl")
        with open(file_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def toggle_finetuned_model(self, active: bool):
        """Mock method to toggle between base and fine-tuned logic."""
        self.is_finetuned_active = active
        self.current_model = "finetuned-career-expert-v1" if active else "base-career-model"
        return self.current_model

    def get_stats(self) -> Dict[str, Any]:
        count = 0
        file_path = os.path.join(self.data_path, "user_interactions.jsonl")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                count = sum(1 for _ in f)

        return {
            "dataset_size": count,
            "active_model": self.current_model,
            "is_finetuned": self.is_finetuned_active
        }

fine_tuning_service = FineTuningService()
