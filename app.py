import hashlib
import pandas as pd
import streamlit as st
import scoring as S
import llm

def get_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:          # no secrets file / key not set -> llm.py falls back to env var
        return None


st.set_page_config(page_title="Vendor Advisor", page_icon="📊", layout="wide")
st.title("📊 Vendor Selection Advisor")
st.caption("Score vendors on cost, quality, lead time and reliability with weights you control. "
           "Scores are computed by transparent rules; Gemini only explains the result.")

with st.sidebar:
    st.header("1. Data")
    up = st.file_uploader("Upload vendor CSV", type="csv")
    st.download_button("Download sample CSV", open("sample_vendors.csv", "rb").read(),
                       "sample_vendors.csv", "text/csv")
    st.header("2. Criteria weights")
    w = {k: st.slider(k, 0, 100, v) for k, v in
         {"Cost": 30, "Quality": 35, "Lead time": 15, "Reliability": 20}.items()}
    st.header("3. Hard constraints (optional)")
    use_c = st.checkbox("Max cost per complete ($)")
    max_cost = st.number_input("Max cost", 1, 10000, 50, disabled=not use_c) if use_c else None
    use_q = st.checkbox("Min quality score")
    min_q = st.slider("Min quality", 0.0, 10.0, 7.0, 0.1, disabled=not use_q) if use_q else None
    use_l = st.checkbox("Max lead time (days)")
    max_l = st.number_input("Max lead days", 1, 365, 15, disabled=not use_l) if use_l else None
    st.header("4. Ask the AI (optional)")
    q = st.text_input("Question about the result", placeholder="e.g. Why not FastTrack?")
    st.info("🔒 Privacy: only the vendor table you load and your question are sent to Google's "
            "Gemini API. Don't upload confidential contract terms or personal data.")
    run = st.button("Rank vendors", type="primary", use_container_width=True)

# ---- load + validate -------------------------------------------------------
try:
    raw = pd.read_csv(up) if up else pd.read_csv("sample_vendors.csv")
except Exception:
    st.error("Could not read that file as CSV. Check the format and try again.")
    st.stop()
df, errors, warns = S.validate(raw)
for e in errors:
    st.error(e)
for m in warns:
    st.warning(m)
if errors:
    st.stop()
if sum(w.values()) == 0:
    st.error("Set at least one weight above zero.")
    st.stop()

with st.expander(f"Vendor data ({len(df)} valid vendors)", expanded=False):
    st.dataframe(df, use_container_width=True, hide_index=True)

# ---- session state: results survive refresh-of-widgets and double clicks ----
constraints = dict(max_cost=max_cost, min_quality=min_q, max_lead=max_l)
sig = hashlib.md5(repr((df.to_dict("records"), w, constraints, q)).encode()).hexdigest()
if run or "res" not in st.session_state:
    if st.session_state.get("sig") != sig:          # identical resubmit = no new API call
        ranked = S.score(df, w, **constraints)
        with st.spinner("Asking Gemini to explain the ranking..."):
            text, src = llm.explain(ranked, w, {k: v for k, v in constraints.items() if v is not None}, q,
                                    api_key=get_key())
        st.session_state.update(res=ranked, text=text, src=src, sig=sig, w=dict(w), cons=constraints)

ranked, wts, cons = st.session_state["res"], st.session_state["w"], st.session_state["cons"]
if st.session_state["sig"] != sig:
    st.info("Inputs changed since the last ranking - click **Rank vendors** to refresh.")

# ---- output ------------------------------------------------------------------
el = ranked[ranked["eligible"]]
c1, c2 = st.columns([3, 2])
with c1:
    st.subheader("Ranked shortlist")
    show = ranked[["rank", "vendor", "score", "cost_per_complete_usd", "quality_score",
                   "lead_time_days", "on_time_delivery_pct", "excluded_reason"]]
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.download_button("Export ranking (CSV)", show.to_csv(index=False), "vendor_ranking.csv", "text/csv")
with c2:
    st.subheader("AI rationale")
    st.markdown(st.session_state["text"])
    st.caption(f"Source: {st.session_state['src']} · AI-generated explanation - verify against the table.")

if not el.empty:
    st.subheader("Why each vendor scored what it did")
    contrib = el.set_index("vendor")[[f"c_{k}" for k in S.CRITERIA]]
    contrib.columns = list(S.CRITERIA)
    st.bar_chart(contrib)

    st.subheader("Sanity checks")
    flags = S.expert_flags(ranked)
    for f in flags:
        st.warning(f)
    if not flags:
        st.success("No rule-of-thumb red flags on the top pick.")
    top, same, total, flips = S.stability(df, wts, **cons)
    st.write(f"**Stability:** #1 stays **{top}** in {same} of {total} weight nudges (±10 points each).")
    if flips:
        st.caption("Flips: " + "; ".join(f"{k} {d:+d} → {v}" for k, d, v in flips))
