import os

import pandas as pd
import requests
import streamlit as st

API = os.getenv("BACKEND_URL", "http://localhost:8000")
st.set_page_config(page_title="Research Paper Discovery", layout="wide")
st.title("Research Paper Discovery")


def call(method, path, **kw):
    try:
        r = requests.request(method, f"{API}{path}", timeout=300, **kw)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        st.error(f"{e.response.status_code}: {e.response.text[:300]}")
    except requests.RequestException as e:
        st.error(f"Backend not reachable: {e}")
    return None


tab_search, tab_upload, tab_compare = st.tabs(["Search", "Upload PDF", "Compare"])

with tab_search:
    q = st.text_input("Ask a research question", placeholder="How can small language models run on phones?")
    use_web = st.checkbox("Also search arXiv / Semantic Scholar", value=True)
    if st.button("Search", type="primary") and q.strip():
        with st.spinner("Planning query, searching, ranking..."):
            data = call("post", "/search", json={"question": q, "use_web": use_web})
        if data:
            plan = data["plan"]
            st.caption(f"Query: {plan['semantic_query']}  |  Keywords: {', '.join(plan['keywords'])}"
                       + (f"  |  From {plan['year_from']}" if plan["year_from"] else ""))
            for p in data["results"]:
                with st.expander(f"{p['title']} ({p['year'] or 'n/a'})  -  score {p['score']}"
                                 + ("  [new]" if p["fresh"] else "")):
                    st.write(p["authors"])
                    st.write(p["abstract"])
                    st.markdown(f"[Open paper]({p['url']})" if p["url"] else "")

with tab_upload:
    f = st.file_uploader("PDF", type="pdf")
    if f and st.button("Add to library"):
        with st.spinner("Extracting and indexing..."):
            res = call("post", "/upload", files={"file": (f.name, f.getvalue(), "application/pdf")})
        if res:
            st.success(f"Added: {res['title']}")

with tab_compare:
    lib = call("get", "/papers") or []
    labels = {p["paper_id"]: f"{p['title']} ({p['year'] or p['source']})" for p in lib}
    chosen = st.multiselect("Select 2-3 papers", list(labels), format_func=labels.get, max_selections=3)
    focus = st.text_input("Optional focus", placeholder="e.g. memory footprint")
    if st.button("Compare", type="primary") and len(chosen) >= 2:
        with st.spinner("Reading papers and comparing (may take a minute)..."):
            res = call("post", "/compare", json={"paper_ids": chosen, "focus": focus or None})
        if res and "papers" in res:
            df = pd.DataFrame(res["papers"]).set_index("title").drop(columns=["id"], errors="ignore").T
            st.table(df)
            st.info(res.get("summary", ""))
        elif res:
            st.warning(res.get("error", "Unexpected response"))
            st.code(res.get("raw", ""))
