import json
import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

import config
import llm
import notify
import sources
from store import LocalSeenStore

store = LocalSeenStore()


class State(TypedDict, total=False):
    articles: list[dict]
    new: list[dict]
    selected: list[dict]
    digest: str


def collect(state: State) -> State:
    return {"articles": sources.fetch_all()}


def dedupe(state: State) -> State:
    seen = store.load()
    new = [a for a in state["articles"] if a["id"] not in seen]
    return {"new": new[: config.MAX_ARTICLES]}


def _parse_score(text: str) -> tuple[int, str]:
    try:
        data = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        return max(0, min(10, int(data["score"]))), str(data.get("reason", ""))
    except Exception:
        m = re.search(r"\b(10|[0-9])\b", text)
        return (int(m.group(1)) if m else 0), ""


def score(state: State) -> State:
    selected = []
    for a in state["new"]:
        prompt = (
            f"Centres d'intérêt de l'utilisateur : {config.INTERESTS}\n\n"
            f"Titre : {a['title']}\nRésumé : {a['summary']}\n\n"
            "Note la pertinence de cet article de 0 à 10. "
            'Réponds uniquement en JSON : {"score": <entier>, "reason": "<courte raison>"}'
        )
        s, reason = _parse_score(llm.ask(prompt, max_tokens=120))
        if s >= config.MIN_SCORE:
            selected.append({**a, "score": s, "reason": reason})
    selected.sort(key=lambda a: a["score"], reverse=True)
    return {"selected": selected[: config.MAX_DIGEST]}


def summarize(state: State) -> State:
    out = []
    for a in state["selected"]:
        prompt = (
            "Résume en français, en 2 phrases maximum, l'intérêt de cet article "
            f"pour un ingénieur IA.\n\nTitre : {a['title']}\nContenu : {a['summary']}"
        )
        out.append({**a, "short": llm.ask(prompt, max_tokens=200)})
    return {"selected": out}


def digest(state: State) -> State:
    lines = ["**Veille IA du jour**", ""]
    for a in state["selected"]:
        lines += [f"**{a['title']}** ({a['score']}/10)", a["short"], a["link"], ""]
    return {"digest": "\n".join(lines)}


def send(state: State) -> State:
    notify.send(state["digest"])
    return {}


def save_seen(state: State) -> State:
    store.save(store.load() | {a["id"] for a in state["new"]})
    return {}


def after_dedupe(state: State) -> str:
    return "score" if state["new"] else END


def after_score(state: State) -> str:
    return "summarize" if state["selected"] else "save_seen"


def build():
    g = StateGraph(State)
    for fn in (collect, dedupe, score, summarize, digest, save_seen):
        g.add_node(fn.__name__, fn)
    g.add_node("notify", send)
    g.add_edge(START, "collect")
    g.add_edge("collect", "dedupe")
    g.add_conditional_edges("dedupe", after_dedupe, ["score", END])
    g.add_conditional_edges("score", after_score, ["summarize", "save_seen"])
    g.add_edge("summarize", "digest")
    g.add_edge("digest", "notify")
    g.add_edge("notify", "save_seen")
    g.add_edge("save_seen", END)
    return g.compile()
