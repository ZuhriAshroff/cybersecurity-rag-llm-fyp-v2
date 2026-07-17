import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from rag_pipeline import (
    load_corpus, build_retriever, build_chain,
    score_hallucination, query_pipeline, query_finetuned,
)

st.set_page_config(
    page_title="CyberRAG",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

:root {
  --bg:           #0d0f14;
  --bg-card:      rgba(255, 255, 255, 0.04);
  --accent:       #64ffda;
  --accent-glow:  rgba(100, 255, 218, 0.12);
  --text:         #e2e8f0;
  --text-muted:   #8892a4;
  --border:       rgba(255, 255, 255, 0.08);
  --green:        #4ade80;
  --yellow:       #fbbf24;
  --red:          #f87171;
  --ease:         cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Global ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section[data-testid="stMain"] > div {
  background-color: var(--bg) !important;
  font-family: 'Space Grotesk', sans-serif !important;
  color: var(--text) !important;
}
[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed;
  inset: 0;
  background: radial-gradient(ellipse 80% 55% at 50% -5%,
    rgba(100, 255, 218, 0.07) 0%, transparent 70%);
  pointer-events: none;
  z-index: 0;
}
[data-testid="stHeader"]  { background: transparent !important; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
.block-container { background: transparent !important; }
.block-container { max-width: 1100px !important; padding-top: 0 !important; }
*, *::before, *::after { font-family: 'Space Grotesk', sans-serif !important; }

/* ── Tabs ── */
[data-testid="stTabs"] { margin-top: 0.5rem; }
[data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid rgba(255,255,255,0.08) !important;
  gap: 0 !important;
}
[data-baseweb="tab"] {
  background: transparent !important;
  color: var(--text-muted) !important;
  font-weight: 600 !important;
  font-size: 0.85rem !important;
  letter-spacing: 0.04em !important;
  padding: 0.75rem 1.5rem !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  transition: color 0.3s var(--ease), border-color 0.3s var(--ease) !important;
}
[aria-selected="true"][data-baseweb="tab"] {
  color: var(--accent) !important;
  border-bottom-color: var(--accent) !important;
}
[data-baseweb="tab-panel"] { padding: 2rem 0 0 !important; background: transparent !important; }
[data-baseweb="tab-highlight"] { display: none !important; }

/* ── Input ── */
[data-testid="stTextInput"] input {
  background:    rgba(255,255,255,0.04) !important;
  border:        1px solid var(--border) !important;
  border-radius: 8px !important;
  color:         var(--text) !important;
  font-size:     1rem !important;
  padding:       0.75rem 1rem !important;
  transition:    border-color 0.3s var(--ease), box-shadow 0.3s var(--ease) !important;
}
[data-testid="stTextInput"] input:focus {
  border-color: var(--accent) !important;
  box-shadow:   0 0 0 3px var(--accent-glow) !important;
  outline:      none !important;
}
[data-testid="stTextInput"] input::placeholder { color: var(--text-muted) !important; }

/* ── Button ── */
[data-testid="stButton"] > button {
  background:    var(--accent) !important;
  color:         #0d0f14 !important;
  border:        none !important;
  border-radius: 8px !important;
  font-weight:   700 !important;
  font-size:     1.1rem !important;
  padding:       0.72rem 1.4rem !important;
  width:         100% !important;
  cursor:        pointer !important;
  transition:    opacity 0.3s var(--ease), transform 0.3s var(--ease) !important;
}
[data-testid="stButton"] > button:hover {
  opacity:   0.85 !important;
  transform: translateY(-1px) !important;
}

/* ── Spinner / alerts ── */
[data-testid="stSpinner"] p { color: var(--accent) !important; }
[data-testid="stAlert"] {
  background:   rgba(251,191,36,0.08) !important;
  border-color: #fbbf24 !important;
}
</style>
""", unsafe_allow_html=True)


# ── HTML helpers ──────────────────────────────────────────────────────────────
def eyebrow(label: str) -> str:
    return (
        '<div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;">'
        '  <div style="width:28px;height:2px;background:#64ffda;flex-shrink:0;"></div>'
        f' <span style="font-size:0.7rem;font-weight:700;letter-spacing:0.11em;'
        f'text-transform:uppercase;color:#64ffda;">{label}</span>'
        '</div>'
    )


def card(inner_html: str, extra: str = "") -> str:
    return (
        f'<div style="background:rgba(255,255,255,0.04);'
        f'border:1px solid rgba(255,255,255,0.08);'
        f'border-left:3px solid #64ffda;'
        f'border-radius:12px;padding:1.5rem;margin-bottom:1.5rem;'
        f'transition:all 0.3s cubic-bezier(0.4,0,0.2,1);{extra}">'
        f'  {inner_html}'
        f'</div>'
    )


def card_muted(inner_html: str) -> str:
    """Dimmed card used for placeholder / unavailable content."""
    return (
        '<div style="background:rgba(255,255,255,0.02);'
        'border:1px solid rgba(255,255,255,0.05);'
        'border-left:3px solid rgba(136,146,164,0.4);'
        'border-radius:12px;padding:1.5rem;margin-bottom:1.5rem;'
        'opacity:0.6;">'
        f'  {inner_html}'
        '</div>'
    )


def confidence_bar_html(pct: float, dim: bool = False) -> str:
    color = "#4ade80" if pct >= 70 else ("#fbbf24" if pct >= 40 else "#f87171")
    if dim:
        color = "#8892a4"
    return (
        f'<div style="margin-top:0.25rem;">'
        f'  <div style="display:flex;align-items:baseline;gap:0.5rem;margin-bottom:0.75rem;">'
        f'    <span style="font-size:3rem;font-weight:700;color:{color};line-height:1;">{pct:.0f}%</span>'
        f'    <span style="color:#8892a4;font-size:0.9rem;">confidence</span>'
        f'  </div>'
        f'  <div style="background:rgba(255,255,255,0.06);border-radius:999px;height:8px;overflow:hidden;">'
        f'    <div style="width:{pct:.1f}%;height:100%;background:{color};border-radius:999px;'
        f'transition:width 0.6s cubic-bezier(0.4,0,0.2,1);"></div>'
        f'  </div>'
        f'</div>'
    )


def sentence_row_html(sentence: str, sim: float, supported: bool) -> str:
    if supported:
        icon, color, bg = "✓", "#4ade80", "rgba(74,222,128,0.06)"
    else:
        icon, color, bg = "✗", "#f87171", "rgba(248,113,113,0.06)"
    return (
        f'<div style="display:flex;align-items:flex-start;gap:12px;'
        f'padding:0.75rem 1rem;background:{bg};border-radius:8px;margin-bottom:8px;'
        f'transition:all 0.3s cubic-bezier(0.4,0,0.2,1);">'
        f'  <span style="color:{color};font-weight:700;font-size:0.95rem;'
        f'flex-shrink:0;margin-top:2px;">{icon}</span>'
        f'  <div style="flex:1;">'
        f'    <p style="margin:0 0 3px;color:#e2e8f0;font-size:0.9rem;line-height:1.55;">{sentence}</p>'
        f'    <span style="font-size:0.73rem;color:#8892a4;">similarity · {sim:.2f}</span>'
        f'  </div>'
        f'</div>'
    )


def delta_badge(baseline: float, finetuned: float) -> str:
    diff  = finetuned - baseline
    color = "#4ade80" if diff >= 0 else "#f87171"
    sign  = "+" if diff >= 0 else ""
    return (
        f'<span style="display:inline-block;padding:0.25rem 0.75rem;'
        f'background:rgba(100,255,218,0.08);border:1px solid rgba(100,255,218,0.2);'
        f'border-radius:999px;font-size:0.78rem;font-weight:700;color:{color};">'
        f'{sign}{diff:.0f}% vs baseline</span>'
    )


def col_header(title: str, badge: str = "") -> str:
    return (
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'margin-bottom:1.5rem;padding-bottom:0.75rem;'
        f'border-bottom:1px solid rgba(255,255,255,0.08);">'
        f'  <span style="font-size:0.85rem;font-weight:700;color:#e2e8f0;">{title}</span>'
        f'  {badge}'
        f'</div>'
    )


def render_pipeline_result(result: dict) -> None:
    st.markdown(eyebrow("Answer"), unsafe_allow_html=True)
    st.markdown(
        card(f'<p style="margin:0;color:#e2e8f0;font-size:0.95rem;line-height:1.75;">'
             f'{result["answer"]}</p>'),
        unsafe_allow_html=True,
    )
    st.markdown(eyebrow("Confidence Score"), unsafe_allow_html=True)
    st.markdown(
        card(
            confidence_bar_html(result["confidence"])
            + f'<p style="margin:0.9rem 0 0;color:#8892a4;font-size:0.85rem;">'
            f'{result["n_supported"]} of {len(result["sentence_scores"])} sentences '
            f'supported by retrieved context</p>'
        ),
        unsafe_allow_html=True,
    )
    if result["sentence_scores"]:
        st.markdown(eyebrow("Sentence Analysis"), unsafe_allow_html=True)
        rows = "".join(
            sentence_row_html(s, sim, ok) for s, sim, ok in result["sentence_scores"]
        )
        st.markdown(f'<div style="margin-bottom:1.5rem;">{rows}</div>', unsafe_allow_html=True)
    st.markdown(eyebrow("Cited Sources"), unsafe_allow_html=True)
    for i, (src, excerpt) in enumerate(result["sources"], 1):
        st.markdown(
            card(
                f'<p style="margin:0 0 0.5rem;font-weight:600;color:#64ffda;font-size:0.9rem;">'
                f'[{i}] {src}</p>'
                f'<p style="margin:0;color:#8892a4;font-size:0.85rem;line-height:1.65;font-style:italic;">'
                f'"{excerpt.strip()}…"</p>'
            ),
            unsafe_allow_html=True,
        )


# ── Pipeline (cached once per server session) ─────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    docs                  = load_corpus()
    retriever, embeddings = build_retriever(docs)
    chain                 = build_chain(retriever)
    return chain, retriever, embeddings


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:3rem 0 2rem;">
  <h1 style="font-size:2.5rem;font-weight:700;color:#e2e8f0;margin:0 0 0.4rem;">
    Cyber<span style="color:#64ffda;">RAG</span>
  </h1>
  <p style="color:#8892a4;font-size:1rem;margin:0;">
    Retrieval-augmented cybersecurity Q&amp;A &nbsp;·&nbsp;
    Llama 3.1 via Groq &nbsp;·&nbsp;
    Semantic chunking + cross-encoder reranking + hallucination scoring
  </p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading knowledge base…"):
    chain, retriever, embeddings = load_pipeline()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_rag, tab_eval = st.tabs(["RAG Assistant", "Evaluation Dashboard"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — RAG Assistant
# ════════════════════════════════════════════════════════════════════════════
with tab_rag:
    col_q, col_btn = st.columns([5, 1])
    with col_q:
        question = st.text_input(
            label="question",
            placeholder="e.g. What is a man-in-the-middle attack?",
            label_visibility="collapsed",
            key="rag_q",
        )
    with col_btn:
        submitted = st.button("→", key="rag_submit")

    st.markdown("<div style='margin-bottom:2rem;'></div>", unsafe_allow_html=True)

    if submitted and not question.strip():
        st.warning("Please enter a question first.")
    elif submitted and question.strip():
        with st.spinner("Retrieving, reranking, and generating…"):
            result = query_pipeline(chain, question.strip(), embeddings)
        render_pipeline_result(result)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Evaluation Dashboard
# ════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown(eyebrow("Evaluation Dashboard"), unsafe_allow_html=True)
    st.markdown("""
    <p style="color:#8892a4;font-size:0.9rem;margin:0 0 2rem;">
      Run the same question through both pipelines and compare confidence scores side by side.
      The fine-tuned model column will be populated once Step 5 (domain fine-tuning) is complete.
    </p>
    """, unsafe_allow_html=True)

    col_eq, col_ebtn = st.columns([5, 1])
    with col_eq:
        eval_q = st.text_input(
            label="eval question",
            placeholder="e.g. What are common ransomware attack vectors?",
            label_visibility="collapsed",
            key="eval_q",
        )
    with col_ebtn:
        eval_submitted = st.button("→", key="eval_submit")

    st.markdown("<div style='margin-bottom:2rem;'></div>", unsafe_allow_html=True)

    if eval_submitted and not eval_q.strip():
        st.warning("Please enter a question first.")

    elif eval_submitted and eval_q.strip():
        with st.spinner("Running baseline pipeline…"):
            baseline_result = query_pipeline(chain, eval_q.strip(), embeddings)
        st.session_state["eval_baseline"] = baseline_result
        st.session_state["eval_question"]  = eval_q.strip()
        st.session_state.pop("eval_finetuned", None)
        st.session_state.pop("eval_ft_error", None)
        try:
            with st.spinner("Running fine-tuned pipeline…"):
                ft_result = query_finetuned(eval_q.strip(), retriever, embeddings)
            st.session_state["eval_finetuned"] = ft_result
        except Exception as exc:
            st.session_state["eval_ft_error"] = str(exc)

    if "eval_baseline" in st.session_state:
        br       = st.session_state["eval_baseline"]
        ft       = st.session_state.get("eval_finetuned")
        ft_error = st.session_state.get("eval_ft_error")

        # ── Confidence comparison bar ──────────────────────────────────────
        st.markdown(eyebrow("Confidence Comparison"), unsafe_allow_html=True)
        b_color  = "#4ade80" if br["confidence"] >= 70 else ("#fbbf24" if br["confidence"] >= 40 else "#f87171")
        conf_pct = f"{br['confidence']:.0f}"
        conf_w   = f"{br['confidence']:.1f}"

        if ft:
            ft_color   = "#4ade80" if ft["confidence"] >= 70 else ("#fbbf24" if ft["confidence"] >= 40 else "#f87171")
            ft_right   = (
                f'<p style="margin:0 0 0.5rem;font-size:0.75rem;font-weight:700;'
                f'letter-spacing:0.08em;text-transform:uppercase;color:#64ffda;">Fine-tuned</p>'
                f'<div style="display:flex;align-items:center;gap:1rem;">'
                f'<span style="font-size:2.2rem;font-weight:700;color:{ft_color};">{ft["confidence"]:.0f}%</span>'
                f'<div style="flex:1;background:rgba(255,255,255,0.06);border-radius:999px;height:8px;overflow:hidden;">'
                f'<div style="width:{ft["confidence"]:.1f}%;height:100%;background:{ft_color};'
                f'border-radius:999px;transition:width 0.6s cubic-bezier(0.4,0,0.2,1);"></div>'
                f'</div></div>'
            )
        else:
            ft_right = (
                f'<p style="margin:0 0 0.5rem;font-size:0.75rem;font-weight:700;'
                f'letter-spacing:0.08em;text-transform:uppercase;color:#8892a4;">'
                f'Fine-tuned &nbsp;'
                f'<span style="padding:0.15rem 0.5rem;background:rgba(255,255,255,0.06);'
                f'border-radius:4px;font-size:0.65rem;letter-spacing:0.05em;">UNAVAILABLE</span></p>'
                f'<div style="display:flex;align-items:center;gap:1rem;">'
                f'<span style="font-size:2.2rem;font-weight:700;color:#8892a4;">—</span>'
                f'<div style="flex:1;background:rgba(255,255,255,0.06);border-radius:999px;height:8px;">'
                f'<div style="width:0%;height:100%;background:#8892a4;border-radius:999px;"></div>'
                f'</div></div>'
            )

        cmp_html = (
            f'<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);'
            f'border-radius:12px;padding:1.5rem;margin-bottom:2rem;">'
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:2rem;">'
            f'<div>'
            f'<p style="margin:0 0 0.5rem;font-size:0.75rem;font-weight:700;letter-spacing:0.08em;'
            f'text-transform:uppercase;color:#64ffda;">Baseline</p>'
            f'<div style="display:flex;align-items:center;gap:1rem;">'
            f'<span style="font-size:2.2rem;font-weight:700;color:{b_color};">{conf_pct}%</span>'
            f'<div style="flex:1;background:rgba(255,255,255,0.06);border-radius:999px;height:8px;overflow:hidden;">'
            f'<div style="width:{conf_w}%;height:100%;background:{b_color};border-radius:999px;'
            f'transition:width 0.6s cubic-bezier(0.4,0,0.2,1);"></div>'
            f'</div></div></div>'
            f'<div>{ft_right}</div>'
            f'</div></div>'
        )
        st.markdown(cmp_html, unsafe_allow_html=True)

        # ── Side-by-side results ───────────────────────────────────────────
        col_base, col_ft = st.columns(2, gap="large")

        with col_base:
            st.markdown(
                col_header("Baseline RAG",
                           f'<span style="font-size:0.72rem;color:{b_color};font-weight:700;">'
                           f'{br["confidence"]:.0f}% confidence</span>'),
                unsafe_allow_html=True,
            )
            render_pipeline_result(br)

        with col_ft:
            if ft:
                ft_color = "#4ade80" if ft["confidence"] >= 70 else ("#fbbf24" if ft["confidence"] >= 40 else "#f87171")
                st.markdown(
                    col_header(
                        "Fine-tuned Model (Phi-2 + QLoRA)",
                        delta_badge(br["confidence"], ft["confidence"]),
                    ),
                    unsafe_allow_html=True,
                )
                render_pipeline_result(ft)
            else:
                st.markdown(
                    col_header("Fine-tuned Model (Phi-2 + QLoRA)",
                               '<span style="padding:0.2rem 0.6rem;background:rgba(255,255,255,0.06);'
                               'border-radius:4px;font-size:0.72rem;color:#8892a4;">Unavailable</span>'),
                    unsafe_allow_html=True,
                )
                error_body = ft_error if ft_error else "Adapter directory not found."
                st.markdown(
                    card_muted(
                        '<div style="padding:1.5rem 1rem;">'
                        '  <p style="font-weight:600;color:#f87171;margin:0 0 0.5rem;">Adapter not loaded</p>'
                        f' <p style="color:#8892a4;font-size:0.85rem;margin:0;line-height:1.6;">{error_body}</p>'
                        '  <p style="color:#8892a4;font-size:0.85rem;margin:0.75rem 0 0;line-height:1.6;">'
                        '    Place the trained adapter at <code style="color:#64ffda;">models/lora_adapter/</code>'
                        '    and resubmit.'
                        '  </p>'
                        '</div>'
                    ),
                    unsafe_allow_html=True,
                )
