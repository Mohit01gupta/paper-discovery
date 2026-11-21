"""Paper Analysis/Compare Agent: retrieve evidence per paper and aspect, then one LLM call.

Budget: the endpoint allows 8192 tokens input+output, so evidence is capped at ~12k chars
(~3k tokens) and the answer at 1024 tokens.
"""
from . import embeddings, ingest, llm, store

ASPECTS = {
    "objective": "research objective and problem addressed",
    "method": "methodology, approach, model architecture",
    "data": "dataset, data used, embeddings or features",
    "evaluation": "evaluation metrics, experiments and results",
    "limitations": "limitations, weaknesses and future work",
}
MAX_PAPERS = 3
EVIDENCE_CHARS = 12000

SYSTEM = """You compare research papers using ONLY the evidence provided. If the evidence does not
say something, write "Not stated". Reply with JSON only:
{"papers":[{"id":"...","title":"...","objective":"","method":"","data":"","evaluation":"","limitations":""}],
 "summary":"2-3 sentences on the key differences"}
Keep every field under 30 words."""


def compare_papers(paper_ids: list[str], focus: str | None = None) -> dict:
    papers = store.get_papers(paper_ids[:MAX_PAPERS])
    if len(papers) < 2:
        raise ValueError("Select at least two papers from the library")

    for p in papers:
        ingest.ensure_fulltext(p)

    vecs = embeddings.embed([f"{q}. {focus}" if focus else q for q in ASPECTS.values()])
    per_cell = EVIDENCE_CHARS // (len(papers) * len(ASPECTS))
    blocks = []
    for p in papers:
        lines = [f"### PAPER id={p['paper_id']} title={p['title']}"]
        for (aspect, _), vec in zip(ASPECTS.items(), vecs):
            hits = store.search_chunks(vec, k=2, paper_ids=[p["paper_id"]])
            lines.append(f"[{aspect}] " + " ".join(h["text"] for h in hits)[:per_cell])
        blocks.append("\n".join(lines))

    user = (f"Focus: {focus}\n\n" if focus else "") + "\n\n".join(blocks)
    text = llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                    max_tokens=1024, temperature=0.1)
    import json, re
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else {"error": "No JSON returned", "raw": text}
    except json.JSONDecodeError:
        return {"error": "Could not parse model output", "raw": text}
