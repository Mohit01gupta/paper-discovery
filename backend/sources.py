"""Web Search Agent tools: arXiv, Semantic Scholar, and OpenAlex public APIs."""
import logging
import random
import re
import time
import xml.etree.ElementTree as ET

import httpx

from . import config

log = logging.getLogger(__name__)
NS = {"a": "http://www.w3.org/2005/Atom"}
MAX_KEYWORDS = 3

# Shared client with a descriptive User-Agent (critical to avoid arXiv 429s)
_CLIENT = httpx.Client(
    timeout=45.0,
    follow_redirects=True,
    headers={
        "User-Agent": "ResearchSearchAgent/1.0 (mailto:researcher@example.com)",
        "Accept": "application/json, application/atom+xml, */*",
    },
)

_LAST_CALL: dict[str, float] = {}


def _request_with_retry(
    url: str,
    params: dict,
    source: str,
    min_interval: float,
    headers: dict | None = None,
    max_retries: int = 3,
) -> httpx.Response:
    """Enforces per-source pacing and retries 429/5xx/timeouts with exponential backoff."""
    for attempt in range(max_retries):
        elapsed = time.monotonic() - _LAST_CALL.get(source, 0.0)
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        try:
            _LAST_CALL[source] = time.monotonic()
            r = _CLIENT.get(url, params=params, headers=headers or {})

            if r.status_code in (429, 500, 502, 503, 504):
                retry_after = r.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait = min(float(retry_after), 30.0)
                else:
                    wait = (3.0 * (2 ** attempt)) + random.uniform(0.5, 1.5)
                log.info("%s rate-limited (%s), retrying in %.1fs...", source, r.status_code, wait)
                time.sleep(wait)
                continue

            r.raise_for_status()
            return r

        except httpx.TimeoutException:
            if attempt == max_retries - 1:
                raise
            wait = 3.0 * (2 ** attempt)
            log.info("%s timed out, retrying in %.1fs...", source, wait)
            time.sleep(wait)

    raise httpx.HTTPStatusError(
        f"{source} failed after {max_retries} retries (429 Too Many Requests)",
        request=httpx.Request("GET", url),
        response=httpx.Response(429),
    )


def _arxiv(query: str, n: int) -> list[dict]:
    r = _request_with_retry(
        url="https://export.arxiv.org/api/query",
        params={"search_query": f'all:"{query}"', "max_results": n, "sortBy": "relevance"},
        source="arxiv",
        min_interval=3.2,  # arXiv strictly enforces >= 3s between calls
    )
    out = []
    for e in ET.fromstring(r.text).findall("a:entry", NS):
        aid = re.sub(r"v\d+$", "", (e.findtext("a:id", "", NS) or "").rsplit("/abs/", 1)[-1])
        abstract = " ".join((e.findtext("a:summary", "", NS) or "").split())
        if not aid or not abstract:
            continue
        out.append({
            "paper_id": f"arxiv:{aid}",
            "title": " ".join((e.findtext("a:title", "", NS) or "").split()),
            "authors": ", ".join(a.findtext("a:name", "", NS) for a in e.findall("a:author", NS)),
            "year": int((e.findtext("a:published", "0000", NS) or "0000")[:4]) or None,
            "abstract": abstract,
            "url": f"https://arxiv.org/abs/{aid}",
            "pdf_url": f"https://arxiv.org/pdf/{aid}",
            "source": "arxiv",
            "citations": None,
        })
    return out


def _s2(query: str, n: int) -> list[dict]:
    api_key = getattr(config, "S2_API_KEY", None)
    headers = {"x-api-key": api_key} if api_key else {}
    r = _request_with_retry(
        url="https://api.semanticscholar.org/graph/v1/paper/search",
        params={
            "query": query,
            "limit": n,
            "fields": "title,abstract,year,authors,citationCount,externalIds,url,openAccessPdf",
        },
        source="semantic_scholar",
        min_interval=1.2 if api_key else 3.0,
        headers=headers,
    )
    out = []
    for d in r.json().get("data", []):
        if not d.get("abstract"):
            continue
        ext = d.get("externalIds") or {}
        if ext.get("ArXiv"):
            pid = f"arxiv:{ext['ArXiv']}"
        elif ext.get("DOI"):
            pid = f"doi:{ext['DOI'].lower()}"
        else:
            pid = f"s2:{d['paperId']}"
        out.append({
            "paper_id": pid,
            "title": d.get("title") or "",
            "authors": ", ".join(a.get("name", "") for a in d.get("authors", [])),
            "year": d.get("year"),
            "abstract": d["abstract"],
            "url": d.get("url") or "",
            "pdf_url": (d.get("openAccessPdf") or {}).get("url") or "",
            "source": "semantic_scholar",
            "citations": d.get("citationCount"),
        })
    return out


def _openalex(query: str, n: int) -> list[dict]:
    """Reliable open-access fallback (10 req/sec free limit, no API key needed)."""
    r = _request_with_retry(
        url="https://openalex.org",
        params={
            "search": query,
            "per-page": n,
            "mailto": "researcher@example.com",
            "select": "id,doi,title,publication_year,authorships,cited_by_count,abstract_inverted_index,primary_location,open_access",
        },
        source="openalex",
        min_interval=0.2,
    )
    out = []
    for d in r.json().get("results", []):
        inv = d.get("abstract_inverted_index")
        if not inv:
            continue
        # Reconstruct abstract from OpenAlex inverted index
        words = sorted((pos, word) for word, positions in inv.items() for pos in positions)
        abstract = " ".join(word for _, word in words)
        doi = (d.get("doi") or "").replace("https://doi.org", "").lower()
        pid = f"doi:{doi}" if doi else f"openalex:{d['id'].rsplit('/', 1)[-1]}"
        loc = d.get("primary_location") or {}
        oa = d.get("open_access") or {}
        out.append({
            "paper_id": pid,
            "title": d.get("title") or "",
            "authors": ", ".join(
                (a.get("author") or {}).get("display_name", "") for a in d.get("authorships", [])
            ),
            "year": d.get("publication_year"),
            "abstract": abstract,
            "url": loc.get("landing_page_url") or d.get("id") or "",
            "pdf_url": oa.get("oa_url") or loc.get("pdf_url") or "",
            "source": "openalex",
            "citations": d.get("cited_by_count"),
        })
    return out


FETCHERS = {"arxiv": _arxiv, "semantic_scholar": _s2, "openalex": _openalex}


def web_search(keywords: list[str], year_from: int | None = None, per_query: int = 6) -> list[dict]:
    found: dict[str, dict] = {}
    kws = keywords[:MAX_KEYWORDS]
    for kw in kws:
        for name in config.SOURCES:
            fn = FETCHERS.get(name)
            if not fn:
                continue
            try:
                papers = fn(kw, per_query)
            except Exception as e:
                log.warning("%s failed for %r: %s", name, kw, e)
                continue
            for p in papers:
                if year_from and p["year"] and p["year"] < year_from:
                    continue
                old = found.setdefault(p["paper_id"], p)
                if old is not p and old.get("citations") is None and p.get("citations") is not None:
                    old["citations"] = p["citations"]
    return list(found.values())
