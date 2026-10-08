import os

from openai import OpenAI

_client = None


def ask(prompt: str, max_tokens: int = 500) -> str:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.environ.get("LLM_API_KEY", "ollama"),
        )
    extra = {}
    effort = os.environ.get("LLM_REASONING_EFFORT")
    if effort:
        extra["extra_body"] = {"reasoning_effort": effort}
    resp = _client.chat.completions.create(
        model=os.environ.get("LLM_MODEL", "gemma3:12b"),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.2,
        **extra,
    )
    return (resp.choices[0].message.content or "").strip()