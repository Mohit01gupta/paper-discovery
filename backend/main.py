"""FastAPI backend.  Run:  uvicorn backend.main:app --port 8000"""
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from qdrant_client import QdrantClient
import qdrant_client.models
from . import compare, graph, ingest, store

app = FastAPI(title="Research Paper Discovery")


class SearchReq(BaseModel):
    question: str
    use_web: bool = True


class CompareReq(BaseModel):
    paper_ids: list[str]
    focus: str | None = None


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/search")
def search(req: SearchReq):
    try:
        return graph.run_search(req.question, req.use_web)
    except Exception as e:
        raise HTTPException(502, f"Search failed: {e}")


@app.get("/papers")
def papers():
    return [{k: p[k] for k in ("paper_id", "title", "year", "source")} for p in store.get_papers()]


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF")
    data = file.file.read()
    if len(data) > 25 * 1024 * 1024:
        raise HTTPException(400, "PDF is larger than 25 MB")
    try:
        p = ingest.ingest_upload(file.filename, data)
    except Exception as e:
        raise HTTPException(400, str(e))
    return {"paper_id": p["paper_id"], "title": p["title"]}


@app.post("/compare")
def compare_endpoint(req: CompareReq):
    try:
        return compare.compare_papers(req.paper_ids, req.focus)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"Comparison failed: {e}")
@app.get("/debug/chunks")
def debug_chunks(paper_id: str | None = None, limit: int = 30):
    from qdrant_client.models import FieldCondition, Filter, MatchValue
    flt = Filter(must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]) if paper_id else None
    pts, _ = store._qd.scroll(store.COLL, scroll_filter=flt, limit=limit, with_payload=True, with_vectors=True)
    return [{"id": str(p.id), "paper_id": p.payload["paper_id"], "kind": p.payload["kind"],
             "text": p.payload["text"], "vector_size": len(p.vector),
             "vector_first_8": [round(x, 4) for x in p.vector[:8]]} for p in pts]