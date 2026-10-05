# Agent de veille IA (LangGraph)

Agent qui collecte des articles (arXiv + flux RSS), filtre ceux qui sont pertinents
avec un LLM, les résume et envoie un digest quotidien.

```
START -> collect -> dedupe -->(rien de nouveau)--> END
                      |
                      v
                    score -->(rien de pertinent)--> save_seen -> END
                      |
                      v
                  summarize -> digest -> notify -> save_seen -> END
```

## Lancer en local

```bash
pip install -r requirements.txt
cp .env.example .env      
python main.py
```

Par défaut le LLM est Ollama en local (`gemma3:12b`). Pour utiliser une API
avec quota gratuit (Groq, Gemini...), change `LLM_BASE_URL`, `LLM_API_KEY` et
`LLM_MODEL` dans `.env`.

Les sujets et sources se règlent dans `config.py`.

## Tests

```bash
python -m pytest
```

Les tests utilisent un faux LLM et de fausses données : aucun appel réseau.


