import json
import os


class LocalSeenStore:
    def __init__(self, path: str = "seen.json"):
        self.path = path

    def load(self) -> set[str]:
        if not os.path.exists(self.path):
            return set()
        with open(self.path, encoding="utf-8") as f:
            return set(json.load(f))

    def save(self, ids: set[str]) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(sorted(ids), f)
