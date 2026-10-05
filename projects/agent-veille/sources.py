import re
from urllib.parse import quote_plus

import feedparser

import config


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text or "")).strip()


def _entries(url: str, source: str) -> list[dict]:
    feed = feedparser.parse(url)
    return [
        {
            "id": e.get("id") or e.get("link"),
            "title": _clean(e.get("title", "")),
            "summary": _clean(e.get("summary", ""))[:1500],
            "link": e.get("link", ""),
            "source": source,
        }
        for e in feed.entries
        if (e.get("id") or e.get("link"))
    ]


def fetch_all() -> list[dict]:
    articles = []
    for q in config.ARXIV_QUERIES:
        url = (
            "https://export.arxiv.org/api/query?search_query="
            f"{quote_plus(q)}&sortBy=submittedDate&sortOrder=descending"
            f"&max_results={config.ARXIV_MAX_PER_QUERY}"
        )
        articles += _entries(url, "arXiv")
    for url in config.RSS_FEEDS:
        articles += _entries(url, "RSS")
    return list({a["id"]: a for a in articles}.values())
