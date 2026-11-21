"""LangGraph workflow:  plan (LLM) -> [web search + index] -> local vector search."""
from datetime import date
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from . import embeddings, llm, sources, store


class State(TypedDict, total=False):
    question: str
    use_web: bool
    plan: dict
    fresh_ids: list[str]
    results: list[dict]


PLAN_PROMPT = f"""You plan searches for a research-paper search engine. Today's year is {date.today().year}.
Reply with JSON only, no other text, with these keys:
"semantic_query": one clear sentence describing the ideal paper (used for embedding search),
"keywords": 3 to 5 short search phrases (1-4 words) using the technical terms, synonyms and acronyms papers use,
"year_from": an integer year only if the user implies recency or a date range, otherwise null."""


def plan_node(s: State) -> State:
    q = s["question"]
    raw = llm.chat_json([{"role": "system", "content": PLAN_PROMPT},
                         {"role": "user", "content": q}], max_tokens=300, temperature=0.1)
    kws = [k.strip() for k in raw.get("keywords", []) if isinstance(k, str) and k.strip()][:5] or [q]
    yf = raw.get("year_from")
    yf = yf if isinstance(yf, int) and 1950 <= yf <= date.today().year else None
    return {"plan": {"semantic_query": raw.get("semantic_query") or q, "keywords": kws, "year_from": yf}}


def web_node(s: State) -> State:
    plan = s["plan"]
    papers = sources.web_search(plan["keywords"], plan["year_from"])
    new = [p for p in papers if not store.has_paper(p["paper_id"])]
    for p in new:
        store.upsert_paper(p)
    store.add_chunks([{"paper_id": p["paper_id"], "year": p["year"], "kind": "abstract", "idx": 0,
                       "text": f"{p['title']}. {p['abstract']}"} for p in new])
    return {"fresh_ids": [p["paper_id"] for p in new]}


def local_node(s: State) -> State:
    plan = s["plan"]
    vec = embeddings.embed([plan["semantic_query"]])[0]
    hits = store.search_chunks(vec, k=40, year_from=plan["year_from"])
    best: dict[str, dict] = {}
    for h in hits:  # hits are sorted by score, so the first hit per paper is its best chunk
        best.setdefault(h["paper_id"], h)
    meta = {p["paper_id"]: p for p in store.get_papers(list(best))}
    fresh = set(s.get("fresh_ids", []))
    results = [{**meta[pid], "score": round(h["score"], 3), "snippet": h["text"][:300],
                "fresh": pid in fresh} for pid, h in best.items() if pid in meta][:10]
    return {"results": results}


def route(s: State) -> str:
    return "web" if s.get("use_web", True) else "local"


_g = StateGraph(State)
_g.add_node("plan", plan_node)
_g.add_node("web", web_node)
_g.add_node("local", local_node)
_g.add_edge(START, "plan")
_g.add_conditional_edges("plan", route, {"web": "web", "local": "local"})
_g.add_edge("web", "local")
_g.add_edge("local", END)
search_graph = _g.compile()


def run_search(question: str, use_web: bool = True) -> dict:
    out = search_graph.invoke({"question": question, "use_web": use_web})
    return {"plan": out["plan"], "results": out.get("results", [])}
