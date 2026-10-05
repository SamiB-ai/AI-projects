import graph
from store import LocalSeenStore

ARTICLES = [
    {"id": "1", "title": "RAG en prod", "summary": "x", "link": "http://a", "source": "RSS"},
    {"id": "2", "title": "Recette de cuisine", "summary": "y", "link": "http://b", "source": "RSS"},
]


def fake_llm(prompt, max_tokens=0):
    if "Note la pertinence" in prompt:
        return '{"score": 9, "reason": "ok"}' if "Titre : RAG" in prompt else '{"score": 1, "reason": "hors sujet"}'
    return "Résumé court."


def setup(monkeypatch, tmp_path, articles=ARTICLES):
    monkeypatch.setattr(graph, "store", LocalSeenStore(str(tmp_path / "seen.json")))
    monkeypatch.setattr(graph.sources, "fetch_all", lambda: articles)
    monkeypatch.setattr(graph.llm, "ask", fake_llm)
    sent = []
    monkeypatch.setattr(graph.notify, "send", sent.append)
    return sent


def test_digest_contient_seulement_le_pertinent(monkeypatch, tmp_path):
    sent = setup(monkeypatch, tmp_path)
    graph.build().invoke({})
    assert len(sent) == 1 and "RAG en prod" in sent[0] and "cuisine" not in sent[0]


def test_deuxieme_run_ne_renvoie_rien(monkeypatch, tmp_path):
    sent = setup(monkeypatch, tmp_path)
    graph.build().invoke({})
    graph.build().invoke({})
    assert len(sent) == 1  # rien de nouveau au 2e passage


def test_rien_de_pertinent_pas_d_envoi(monkeypatch, tmp_path):
    sent = setup(monkeypatch, tmp_path, [ARTICLES[1]])
    graph.build().invoke({})
    assert sent == []
    assert graph.store.load() == {"2"}  # mais l'article est marqué comme vu


def test_parse_score_json_casse():
    assert graph._parse_score("score : 8 sur 10")[0] == 8
