"""Support tickets for handing a conversation to a human agent.
Demo version: tickets are saved as lines in a file. A real bank would send them to its support system."""
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path


class TicketStore:
    def __init__(self, path="data/tickets.jsonl"):
        self.path = Path(path)

    def create(self, summary: str, lang: str) -> dict:
        ticket = {
            "id": f"T-{secrets.randbelow(900000) + 100000}",
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "lang": lang,
            "summary": summary,
            "status": "waiting_for_agent",
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ticket, ensure_ascii=False) + "\n")
        return ticket
