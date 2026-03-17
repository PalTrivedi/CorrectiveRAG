import logging
import os
import time

import requests
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

API_URL = os.getenv("CRAG_API_URL", "http://127.0.0.1:8000/query")
REQUEST_TIMEOUT = int(os.getenv("CRAG_TIMEOUT_SECONDS", "90"))
LOG_LEVEL = os.getenv("CRAG_LOG_LEVEL", "INFO").upper()

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("crag.frontend")

st.set_page_config(
    page_title="Chronicles of Westeros",
    page_icon="🐉",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ========================
# SESSION STATE
# ========================
if "last_payload" not in st.session_state:
    st.session_state.last_payload = None
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "current_query" not in st.session_state:
    st.session_state.current_query = ""
if "log_events" not in st.session_state:
    st.session_state.log_events = []


def log_event(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.log_events.append(f"[{timestamp}] {message}")
    st.session_state.log_events = st.session_state.log_events[-50:]


def escape_html(text: str) -> str:
    return text.replace("<", "&lt;").replace(">", "&gt;")


def score_class(score: float) -> str:
    if score >= 0.60:
        return "pill-green"
    if score >= 0.40:
        return "pill-amber"
    return "pill-red"


# ========================
# GLOBAL STYLES
# ========================
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Cinzel+Decorative:wght@400;700&family=Cinzel:wght@400;500;600;700&family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;1,400;1,500&display=swap');

        :root {
            --bg-deep:    #0b0e17;
            --bg-mid:     #12162b;
            --bg-card:    rgba(18, 22, 43, 0.85);
            --gold:       #c9a84c;
            --gold-light: #e8d48b;
            --gold-dim:   rgba(201, 168, 76, 0.12);
            --gold-glow:  rgba(201, 168, 76, 0.25);
            --crimson:    #8b1a2b;
            --crimson-l:  #b22234;
            --ice:        #a8c8e8;
            --ice-dim:    rgba(168, 200, 232, 0.6);
            --parch:      #ece4d4;
            --parch-dim:  rgba(236, 228, 212, 0.75);
            --smoke:      rgba(236, 228, 212, 0.35);
            --line:       rgba(201, 168, 76, 0.18);
            --shadow-lg:  0 30px 80px rgba(0,0,0,0.55);
            --shadow-md:  0 16px 48px rgba(0,0,0,0.4);
            --shadow-sm:  0 8px 24px rgba(0,0,0,0.3);
            --radius-lg:  28px;
            --radius-md:  20px;
            --radius-sm:  14px;
        }

        /* Hide Streamlit chrome */
        [data-testid="stSidebar"],
        [data-testid="collapsedControl"],
        #MainMenu,
        header,
        footer {
            display: none !important;
        }

        /* App background */
        .stApp {
            background:
                radial-gradient(ellipse at 20% 0%, rgba(139, 26, 43, 0.15) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 100%, rgba(201, 168, 76, 0.08) 0%, transparent 50%),
                radial-gradient(ellipse at 50% 50%, rgba(18, 22, 43, 1) 0%, rgba(11, 14, 23, 1) 100%);
            background-attachment: fixed;
            color: var(--parch);
            font-family: "Cormorant Garamond", serif;
            min-height: 100vh;
        }

        .block-container {
            padding-top: 2.5rem !important;
            padding-bottom: 4rem !important;
            max-width: 820px !important;
        }

        /* Floating particles */
        .particle {
            position: fixed;
            width: 3px;
            height: 3px;
            border-radius: 50%;
            background: rgba(201, 168, 76, 0.25);
            opacity: 0.2;
            animation: float 12s linear infinite;
            z-index: 0;
        }
        .p1 { left: 10%; bottom: -10px; animation-delay: 0s; }
        .p2 { left: 35%; bottom: -20px; animation-delay: 2s; }
        .p3 { left: 60%; bottom: -15px; animation-delay: 4s; }
        .p4 { left: 80%; bottom: -25px; animation-delay: 6s; }
        .p5 { left: 90%; bottom: -30px; animation-delay: 8s; }

        /* Hero */
        .got-hero {
            text-align: center;
            padding: 3.5rem 2rem 2.5rem;
            position: relative;
            margin-bottom: 2rem;
            z-index: 1;
        }

        .got-hero::before {
            content: "";
            position: absolute;
            bottom: 0;
            left: 10%;
            right: 10%;
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--gold), transparent);
            opacity: 0.6;
        }

        .got-hero::after {
            content: "";
            position: absolute;
            bottom: -1px;
            left: 10%;
            width: 80px;
            height: 3px;
            background: radial-gradient(circle, rgba(232, 212, 139, 0.9), transparent 70%);
            animation: heroGlow 6s linear infinite;
        }

        .got-sigil {
            font-size: 3.2rem;
            margin-bottom: 0.6rem;
            display: block;
            animation: sigilPulse 3s ease-in-out infinite;
        }

        .got-title {
            font-family: "Cinzel Decorative", "Cinzel", serif;
            font-size: 2.6rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            background: linear-gradient(120deg, var(--gold-light), var(--gold), #a07830, var(--gold-light));
            background-size: 200% 200%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
            line-height: 1.1;
            animation: titleShimmer 5s linear infinite;
        }

        .got-quote {
            font-family: "Cormorant Garamond", serif;
            font-style: italic;
            font-size: 1.05rem;
            color: var(--ice-dim);
            margin-top: 0.9rem;
            letter-spacing: 0.02em;
        }

        .got-tech {
            font-family: "Cinzel", serif;
            font-size: 0.7rem;
            color: var(--smoke);
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-top: 1.2rem;
        }

        /* Suggestions */
        .suggestions-label {
            font-family: "Cinzel", serif;
            font-size: 0.75rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--gold);
            text-align: center;
            margin-bottom: 0.8rem;
            opacity: 0.7;
        }

        /* Textarea */
        .stTextArea textarea {
            background: var(--bg-card) !important;
            color: var(--parch) !important;
            border: 1.5px solid var(--line) !important;
            border-radius: var(--radius-md) !important;
            font-family: "Cormorant Garamond", serif !important;
            font-size: 1.15rem !important;
            line-height: 1.6 !important;
            padding: 1.2rem 1.4rem !important;
            transition: border-color 0.3s, box-shadow 0.3s !important;
            resize: none !important;
        }

        .stTextArea textarea:focus {
            border-color: var(--gold) !important;
            box-shadow: 0 0 0 1px var(--gold), 0 0 30px var(--gold-dim) !important;
            animation: focusPulse 0.6s ease-out;
        }

        .stTextArea textarea::placeholder {
            color: var(--smoke) !important;
            font-style: italic !important;
        }

        .stTextArea label {
            display: none !important;
        }

        /* Buttons */
        div[data-testid="stButton"] > button[kind="primary"] {
            background: linear-gradient(135deg, var(--crimson) 0%, var(--crimson-l) 50%, var(--crimson) 100%) !important;
            background-size: 200% 200% !important;
            border: 1.5px solid var(--gold) !important;
            color: var(--gold-light) !important;
            font-family: "Cinzel", serif !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            letter-spacing: 0.1em !important;
            text-transform: uppercase !important;
            border-radius: var(--radius-sm) !important;
            padding: 0.85rem 2rem !important;
            box-shadow: var(--shadow-sm), inset 0 1px 0 rgba(255,255,255,0.06) !important;
            transition: all 0.3s ease !important;
            animation: buttonShimmer 5s linear infinite;
        }

        div[data-testid="stButton"] > button[kind="primary"]:hover {
            transform: translateY(-3px) !important;
            box-shadow: var(--shadow-md), 0 0 35px rgba(139, 26, 43, 0.5) !important;
            border-color: var(--gold-light) !important;
        }

        div[data-testid="stButton"] > button[kind="primary"]:active {
            transform: translateY(0) !important;
        }

        div[data-testid="stButton"] > button[kind="secondary"],
        div[data-testid="stButton"] > button:not([kind="primary"]) {
            background: transparent !important;
            border: 1px solid var(--line) !important;
            color: var(--parch-dim) !important;
            font-family: "Cormorant Garamond", serif !important;
            font-size: 0.92rem !important;
            font-weight: 500 !important;
            border-radius: 30px !important;
            padding: 0.45rem 1.1rem !important;
            transition: all 0.3s ease !important;
            box-shadow: none !important;
        }

        div[data-testid="stButton"] > button[kind="secondary"]:hover,
        div[data-testid="stButton"] > button:not([kind="primary"]):hover {
            background: var(--gold-dim) !important;
            border-color: var(--gold) !important;
            color: var(--gold-light) !important;
            box-shadow: 0 0 20px var(--gold-dim) !important;
            transform: translateY(-1px) !important;
        }

        /* Answer card */
        .answer-card {
            background: linear-gradient(135deg, rgba(201,168,76,0.04) 0%, rgba(18,22,43,0.6) 100%);
            border: 1.5px solid var(--line);
            border-radius: var(--radius-lg);
            padding: 2.4rem 2.6rem;
            margin: 2rem 0;
            box-shadow: var(--shadow-lg);
            position: relative;
            overflow: hidden;
            animation: fadeSlideUp 0.6s ease-out;
            z-index: 1;
        }

        .answer-card::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, var(--gold), transparent);
            transform: scaleX(0);
            transform-origin: center;
            animation: expandLine 0.6s ease-out forwards;
        }

        .answer-heading {
            font-family: "Cinzel", serif;
            font-size: 0.75rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--gold);
            margin-bottom: 1.2rem;
            opacity: 0.8;
        }

        .answer-body {
            font-family: "Cormorant Garamond", serif;
            font-size: 1.18rem;
            line-height: 1.9;
            color: var(--parch);
            white-space: pre-wrap;
            animation: fadeIn 0.8s ease 0.3s both;
        }

        /* Metadata pills */
        .meta-strip {
            display: flex;
            gap: 0.6rem;
            flex-wrap: wrap;
            justify-content: center;
            margin-top: 1.6rem;
            padding-top: 1.4rem;
            border-top: 1px solid var(--line);
        }

        .meta-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            background: rgba(201, 168, 76, 0.08);
            border: 1px solid var(--line);
            border-radius: 30px;
            padding: 0.3rem 0.85rem;
            font-family: "Cinzel", serif;
            font-size: 0.68rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: var(--ice-dim);
            animation: pillIn 0.5s ease-out both;
        }

        .meta-pill:nth-child(1) { animation-delay: 0s; }
        .meta-pill:nth-child(2) { animation-delay: 0.15s; }
        .meta-pill:nth-child(3) { animation-delay: 0.3s; }

        .meta-pill .pill-val { color: var(--gold); font-weight: 600; }
        .pill-green .pill-val  { color: #6bcb77; }
        .pill-red .pill-val    { color: #e8655a; }
        .pill-amber .pill-val  { color: var(--gold-light); }

        /* History */
        .history-section {
            margin-top: 3rem;
            padding-top: 1.5rem;
            border-top: 1px solid var(--line);
        }

        .history-label {
            font-family: "Cinzel", serif;
            font-size: 0.7rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--gold);
            opacity: 0.5;
            margin-bottom: 0.8rem;
        }

        .history-item {
            display: flex;
            align-items: baseline;
            gap: 0.6rem;
            padding: 0.4rem 0;
            font-size: 0.9rem;
            color: var(--smoke);
            border-bottom: 1px solid rgba(201,168,76,0.06);
            animation: slideInLeft 0.5s ease-out both;
        }

        .history-time {
            font-family: "Cinzel", serif;
            font-size: 0.65rem;
            color: rgba(201,168,76,0.4);
            letter-spacing: 0.04em;
            flex-shrink: 0;
        }

        .history-query {
            font-family: "Cormorant Garamond", serif;
            font-style: italic;
        }

        /* Expander */
        .streamlit-expanderHeader {
            background: transparent !important;
            border: 1px solid var(--line) !important;
            border-radius: var(--radius-sm) !important;
            color: var(--smoke) !important;
            font-family: "Cinzel", serif !important;
            font-size: 0.72rem !important;
            letter-spacing: 0.08em !important;
            text-transform: uppercase !important;
        }

        details[open] .streamlit-expanderHeader {
            border-color: var(--gold) !important;
            color: var(--gold) !important;
        }

        .streamlit-expanderContent {
            border: 1px solid var(--line) !important;
            border-top: none !important;
            border-radius: 0 0 var(--radius-sm) var(--radius-sm) !important;
            background: var(--bg-card) !important;
        }

        /* Spinner */
        .stSpinner > div {
            border-top-color: var(--gold) !important;
        }

        .stSpinner > div > span {
            color: var(--ice-dim) !important;
            font-family: "Cormorant Garamond", serif !important;
            font-style: italic !important;
        }

        /* Footer */
        .got-footer {
            text-align: center;
            padding: 3rem 0 1rem;
            position: relative;
        }

        .got-footer::before {
            content: "";
            position: absolute;
            top: 0;
            left: 20%;
            right: 20%;
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--line), transparent);
        }

        .footer-text {
            font-family: "Cinzel", serif;
            font-size: 0.6rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: rgba(201,168,76,0.4);
        }

        .footer-quote {
            font-family: "Cormorant Garamond", serif;
            font-style: italic;
            font-size: 0.85rem;
            color: rgba(168,200,232,0.45);
            margin-top: 0.4rem;
        }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: var(--bg-deep); }
        ::-webkit-scrollbar-thumb { background: rgba(201,168,76,0.25); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(201,168,76,0.45); }

        /* Keyframes */
        @keyframes titleShimmer {
            0% { background-position: 0% 50%; }
            100% { background-position: 200% 50%; }
        }

        @keyframes sigilPulse {
            0%, 100% { transform: scale(1.0); filter: drop-shadow(0 0 12px rgba(201,168,76,0.25)); }
            50% { transform: scale(1.08); filter: drop-shadow(0 0 24px rgba(201,168,76,0.45)); }
        }

        @keyframes heroGlow {
            0% { transform: translateX(0); opacity: 0.3; }
            50% { opacity: 0.8; }
            100% { transform: translateX(520px); opacity: 0.3; }
        }

        @keyframes buttonShimmer {
            0% { background-position: 0% 50%; }
            100% { background-position: 200% 50%; }
        }

        @keyframes fadeSlideUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes expandLine {
            from { transform: scaleX(0); }
            to { transform: scaleX(1); }
        }

        @keyframes pillIn {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes focusPulse {
            0% { box-shadow: 0 0 0 rgba(201,168,76,0); }
            100% { box-shadow: 0 0 20px rgba(201,168,76,0.35); }
        }

        @keyframes slideInLeft {
            from { opacity: 0; transform: translateX(-12px); }
            to { opacity: 1; transform: translateX(0); }
        }

        @keyframes float {
            0% { transform: translateY(0); opacity: 0; }
            15% { opacity: 0.2; }
            100% { transform: translateY(-120vh); opacity: 0; }
        }

        @media (prefers-reduced-motion: reduce) {
            * { animation: none !important; transition: none !important; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Floating particles markup
st.markdown(
    """
    <div class="particle p1"></div>
    <div class="particle p2"></div>
    <div class="particle p3"></div>
    <div class="particle p4"></div>
    <div class="particle p5"></div>
    """,
    unsafe_allow_html=True,
)


# ========================
# HERO
# ========================
st.markdown(
    """
    <div class="got-hero">
        <span class="got-sigil">🐉</span>
        <h1 class="got-title">Chronicles of Westeros</h1>
        <div class="got-quote">"A reader lives a thousand lives before he dies. The man who never reads lives only one."</div>
        <div class="got-tech">Corrective RAG - NVIDIA - Pinecone - Hugging Face</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ========================
# SUGGESTED QUESTIONS
# ========================
st.markdown(
    '<div class="suggestions-label">Suggested Questions</div>',
    unsafe_allow_html=True,
)

suggestions = [
    "Who killed Jon Arryn and why?",
    "What happened at the Red Wedding?",
    "Tell me about the Targaryen dynasty",
    "Describe the Night's Watch",
    "What is the story of Tyrion?",
    "Who is Azor Ahai?",
]

row1 = st.columns(3)
row2 = st.columns(3)
rows = [row1, row2]

for idx, s in enumerate(suggestions):
    r = idx // 3
    c = idx % 3
    if rows[r][c].button(s, key=f"sug_{idx}", use_container_width=True):
        st.session_state.current_query = s
        log_event(f"Suggestion clicked: {s}")


# ========================
# QUERY INPUT
# ========================
st.markdown("")

query = st.text_area(
    "question",
    value=st.session_state.current_query,
    placeholder="What would you ask the Maester?",
    height=110,
    label_visibility="collapsed",
)

st.markdown("")

if st.button(
    "CONSULT THE ARCHIVES",
    use_container_width=True,
    type="primary",
):
    if not query.strip():
        st.warning("A raven must carry a message - please write your question.")
        log_event("Blocked empty query.")
    else:
        with st.spinner("The Maester consults the ancient scrolls..."):
            try:
                log_event(f"POST {API_URL} (timeout {REQUEST_TIMEOUT}s)")
                resp = requests.post(
                    API_URL,
                    json={"query": query},
                    timeout=REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
                st.session_state.last_payload = resp.json()
                st.session_state.query_history.append(
                    {"q": query, "ts": time.strftime("%H:%M:%S")}
                )
                st.session_state.current_query = ""
                log_event("Response received.")
            except requests.exceptions.Timeout:
                st.error(
                    f"The ravens did not return in time ({REQUEST_TIMEOUT}s). Is the API server running?"
                )
                log_event("Timeout error.")
                st.session_state.last_payload = None
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the Citadel - make sure the FastAPI server is running.")
                log_event("Connection error.")
                st.session_state.last_payload = None
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")
                log_event(f"Unhandled error: {exc}")
                st.session_state.last_payload = None


# ========================
# ANSWER
# ========================
payload = st.session_state.last_payload
if payload:
    answer_raw = payload.get("answer", "")
    answer_safe = escape_html(answer_raw)

    rel = float(payload.get("relevance_score", 0.0))
    web = bool(payload.get("used_web_search", False))
    sources = payload.get("sources", [])
    source_count = len(sources)
    score_cls = score_class(rel)

    web_label = "Yes" if web else "No"
    web_cls = "pill-green" if web else "pill-red"
    rel_label = "NA" if web else f"{rel:.2f}"

    st.markdown(
        f"""
        <div class="answer-card">
            <div class="answer-heading">The Maester's Answer</div>
            <div class="answer-body">{answer_safe}</div>
            <div class="meta-strip">
                <span class="meta-pill {score_cls}">Relevance <span class="pill-val">{rel_label}</span></span>
                <span class="meta-pill {web_cls}">Web Search <span class="pill-val">{web_label}</span></span>
                <span class="meta-pill pill-amber">Sources <span class="pill-val">{source_count}</span></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ========================
# FOOTER
# ========================
st.markdown(
    """
    <div class="got-footer">
        <div class="footer-text">Corrective RAG - NVIDIA gpt-oss-120b - Pinecone - Hugging Face</div>
        <div class="footer-quote">"Valar Morghulis" - All men must debug</div>
    </div>
    """,
    unsafe_allow_html=True,
)
