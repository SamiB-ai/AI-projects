import json
import os
import urllib.request


def send(text: str) -> None:
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        print(text)
        return
    req = urllib.request.Request(
        url,
        data=json.dumps({"content": text[:1900]}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "agent-veille"},
    )
    urllib.request.urlopen(req, timeout=10)
