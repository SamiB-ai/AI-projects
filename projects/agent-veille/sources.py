import html
import re
import urllib.request
from urllib.parse import quote_plus, urlparse

import feedparser

import config


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text or "")).strip()


def _text(entry) -> str:
    text = entry.get("summary") or ""
    if not text and entry.get("content"):
        text = entry["content"][0].get("value", "")
    return _clean(text)[:1500]


def _entries(url: str, source: str) -> list[dict]:
    feed = feedparser.parse(url)
    articles = []
    for e in feed.entries:
        article_id = e.get("id") or e.get("link")
        if not article_id:
            continue
        articles.append(
            {
                "id": article_id,
                "title": _clean(e.get("title", "")),
                "summary": _text(e),
                "link": e.get("link", ""),
                "source": source,
            }
        )
    return articles


def fetch_page_text(url: str) -> str:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "agent-veille"})
        page = urllib.request.urlopen(req, timeout=10).read().decode("utf-8", errors="ignore")
    except Exception:
        return ""
    parts = []
    for tag in re.findall(r"<meta[^>]+>", page):
        if 'og:description' in tag or 'name="description"' in tag:
            m = re.search(r'content="([^"]*)"', tag)
            if m:
                parts.append(m.group(1))
                break
    parts += re.findall(r"<p[^>]*>(.*?)</p>", page, re.S)
    return _clean(html.unescape(" ".join(parts)))[:1500]


def ensure_text(articles: list[dict]) -> list[dict]:
    out = []
    for a in articles:
        if len(a["summary"]) < config.MIN_TEXT_LENGTH and a["link"]:
            a = {**a, "summary": fetch_page_text(a["link"])}
        if len(a["summary"]) >= config.MIN_TEXT_LENGTH:
            out.append(a)
    return out


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
        articles += _entries(url, urlparse(url).netloc)
    return list({a["id"]: a for a in articles}.values())