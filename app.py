import hashlib
import io
import re
from datetime import datetime
from html import escape

import easyocr
import spacy
import streamlit as st
from dotenv import load_dotenv

from auth import AuthError, authenticate, register
from db import DatabaseUnavailable, clear_history, init_db, list_history, record_analysis
from pipeline import (
    CLAUSE_READ_LIMIT,
    AnalysisError,
    analyze_comparison,
    analyze_single,
    extract_text_digital,
    generate_pdf_report,
    redline_html,
    slugify,
)
from rag import answer_question

load_dotenv()


@st.cache_resource
def load_ocr_model():
    return easyocr.Reader(['en'], gpu=False)


@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")


@st.cache_resource
def ensure_schema():
    init_db()
    return True


# ---- STYLE ----

STYLE = """<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Caslon+Text:ital,wght@0,400;0,700;1,400&family=Mukta:wght@400;500;600;700&display=swap');
:root{
--paper:#EDEFE8; --sheet:#FBFBF8; --well:#F3F4EF;
--ink:#1D2331; --ink-2:#4E5566; --ink-3:#666C7D;
--rule:#D6DACF; --rule-2:#BFC5B6;
--violet:#5A3D9A; --violet-deep:#47307C; --violet-wash:#EFEBF7;
--red:#A8382B; --red-wash:#F7E8E5;
--ochre:#8F6213; --ochre-wash:#F5EEDE;
--green:#2E6E52; --green-wash:#E4EFE8;
--serif:'Libre Caslon Text','Iowan Old Style',Georgia,serif;
--sans:'Mukta','Segoe UI',system-ui,sans-serif;
--lift:0 1px 0 rgba(29,35,49,.05),0 12px 30px -20px rgba(29,35,49,.32);
}
html,body,.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:var(--paper)!important;color:var(--ink);}
:where(.stApp) :where(p,li,label,input,button,textarea){font-family:var(--sans);}
.block-container{max-width:1180px!important;padding:2.4rem 2rem 3rem!important;}
[data-testid="stHeader"]{background:transparent!important;pointer-events:none;}
[data-testid="stHeader"] button{pointer-events:auto;}
[data-testid="stDecoration"],[data-testid="stStatusWidget"],[data-testid="stMainMenu"],
[data-testid="stToolbarActions"],[data-testid="stAppDeployButton"],#MainMenu,footer{display:none!important;}
[data-testid="stExpandSidebarButton"] svg{fill:var(--violet)!important;}
::selection{background:#DCD3EE;}
.stApp :focus-visible{outline:2px solid var(--violet)!important;outline-offset:2px;}
/* type */
.t-display{font-family:var(--serif);font-size:2.6rem;line-height:1.14;color:var(--ink);margin:0 0 1.1rem;max-width:18ch;}
.t-h1{font-family:var(--serif);font-size:1.85rem;line-height:1.22;color:var(--ink);margin:.2rem 0 .7rem;overflow-wrap:anywhere;}
.t-h2{font-family:var(--serif);font-size:1.45rem;line-height:1.3;color:var(--ink);margin:2.6rem 0 .35rem;}
.t-h3{font-family:var(--serif);font-size:1.12rem;line-height:1.35;color:var(--ink);margin:0;}
.t-body{font-size:1.04rem;line-height:1.62;color:var(--ink-2);margin:0;max-width:62ch;}
.t-small{font-size:.9rem;line-height:1.5;color:var(--ink-3);margin:0;}
.t-lede{margin:0 0 1.3rem;}
/* brand */
.brand{display:flex;align-items:center;gap:.7rem;}
.seal{width:2.1rem;height:2.1rem;flex:none;border-radius:50%;display:grid;place-items:center;color:var(--violet);
border:1.5px solid var(--violet);box-shadow:inset 0 0 0 2.5px var(--paper),inset 0 0 0 3.5px var(--violet);
font-family:var(--serif);font-style:italic;font-size:.95rem;}
.brand-name{font-family:var(--serif);font-size:1.35rem;color:var(--ink);}
.brand-name small{font-family:var(--sans);font-size:.85rem;color:var(--ink-3);margin-left:.35rem;}
/* mode switch */
[data-testid="stButtonGroup"]{margin:1.8rem 0 .2rem;border-bottom:1px solid var(--rule);}
[data-testid="stButtonGroup"] [role="radiogroup"]{gap:1.75rem!important;flex-wrap:wrap;}
[data-testid="stButtonGroup"] button[data-variant="segmented_control"]{background:transparent!important;border:none!important;
border-bottom:2px solid transparent!important;border-radius:0!important;box-shadow:none!important;padding:.5rem 0!important;
margin-bottom:-1px;min-height:0!important;color:var(--ink-3)!important;}
[data-testid="stButtonGroup"] button[data-variant="segmented_control"] p{font-size:1.02rem!important;font-weight:500!important;color:inherit!important;}
[data-testid="stButtonGroup"] button[data-variant="segmented_control"]:hover{color:var(--ink)!important;}
[data-testid="stButtonGroup"] button[data-variant="segmented_control"][aria-checked="true"]{color:var(--ink)!important;border-bottom-color:var(--violet)!important;}
/* sheets */
.sheet{background:var(--sheet);border:1px solid var(--rule);border-radius:4px;box-shadow:var(--lift);padding:1.8rem 2rem;}
.dochead{display:flex;justify-content:space-between;align-items:flex-start;gap:2rem;margin:1.4rem 0 1.1rem;}
.stMarkdown ul.reviews,.stMarkdown ul.asks{padding-left:0!important;margin-left:0!important;}
.stMarkdown ul.reviews li,.stMarkdown ul.asks li{margin-left:0!important;padding-left:0;list-style:none;}
.dochead-main{min-width:0;flex:1;}
.facts{display:flex;flex-wrap:wrap;gap:.35rem 1.4rem;margin-top:1.1rem;padding-top:.9rem;border-top:1px solid var(--rule);}
.facts > span{font-size:.9rem;color:var(--ink-3);}
.facts b{font-weight:600;color:var(--ink-2);}
/* the stamp */
.stamp{flex:none;color:var(--violet);border:2px solid currentColor;border-radius:11px;padding:4px;opacity:.9;
transform:rotate(-4deg);mix-blend-mode:multiply;margin:.4rem .3rem 0 0;animation:stamp-in .42s cubic-bezier(.2,.9,.25,1.15) both;}
.stamp-ring{display:flex;flex-direction:column;align-items:center;gap:.12rem;border:1px solid currentColor;border-radius:7px;
padding:.55rem 1.15rem .55rem;min-width:11.5rem;text-align:center;text-transform:uppercase;font-family:var(--serif);}
.stamp-top{font-size:.6rem;letter-spacing:.22em;}
.stamp-word{font-size:1.22rem;font-weight:700;letter-spacing:.06em;line-height:1.15;}
.stamp-score{font-size:.92rem;letter-spacing:.08em;}
.stamp-date{font-size:.6rem;letter-spacing:.18em;border-top:1px solid currentColor;padding-top:.22rem;margin-top:.22rem;align-self:stretch;}
@keyframes stamp-in{from{opacity:0;transform:rotate(-4deg) scale(1.22);}to{opacity:.9;transform:rotate(-4deg) scale(1);}}
@media (prefers-reduced-motion:reduce){.stamp{animation:none;}}
/* risk marks */
.mark{display:inline-flex;align-items:center;gap:.4rem;font-size:.9rem;font-weight:600;white-space:nowrap;}
.mark::before{content:"";width:.5rem;height:.5rem;border-radius:50%;background:currentColor;flex:none;}
.mark--high{color:var(--red);}
.mark--medium{color:var(--ochre);}
.mark--low{color:var(--green);}
.mark--new{color:var(--violet);}
.mark--muted{color:var(--ink-3);}
/* clauses */
.clause{background:var(--sheet);border:1px solid var(--rule);border-left:3px solid var(--rule-2);border-radius:2px 4px 4px 2px;
padding:1.2rem 1.4rem 1.25rem;margin-bottom:.85rem;}
.clause--high{border-left-color:var(--red);}
.clause--medium{border-left-color:var(--ochre);}
.clause--low{border-left-color:var(--green);}
.clause--new{border-left-color:var(--violet);}
.clause-head{display:flex;justify-content:space-between;align-items:baseline;gap:.4rem 1rem;flex-wrap:wrap;margin-bottom:.6rem;}
.clause-text{font-family:var(--serif);font-size:1rem;line-height:1.78;color:var(--ink);margin:0;max-width:70ch;}
.note{margin-top:.9rem;padding:.7rem .95rem .75rem;background:var(--well);border-radius:4px;max-width:70ch;}
.note-label{font-weight:600;font-size:.88rem;color:var(--violet-deep);margin:0 0 .1rem;}
.note-body{font-size:.96rem;line-height:1.55;color:var(--ink-2);margin:0;}
.shift{font-size:.95rem;color:var(--ink-2);margin:-.2rem 0 .7rem;display:flex;flex-wrap:wrap;gap:.3rem .55rem;align-items:center;}
.redline{font-family:var(--serif);font-size:1rem;line-height:1.85;color:var(--ink);max-width:70ch;}
.redline del{color:var(--red);background:var(--red-wash);text-decoration:line-through;text-decoration-color:rgba(168,56,43,.6);padding:0 .12rem;}
.redline ins{color:var(--green);background:var(--green-wash);text-decoration:none;border-bottom:1px solid rgba(46,110,82,.5);padding:0 .12rem;}
.termdiff{display:flex;flex-wrap:wrap;gap:.4rem 1.6rem;margin:0 0 .8rem;font-size:.92rem;color:var(--ink-3);}
.tok{display:inline-block;font-size:.88rem;padding:.05rem .45rem;border-radius:3px;background:var(--well);color:var(--ink);margin:.1rem .25rem .1rem 0;}
.tok--out{background:var(--red-wash);color:var(--red);text-decoration:line-through;}
.tok--in{background:var(--green-wash);color:var(--green);}
/* aside */
.aside-block{padding-top:1.4rem;}
.aside-block + .aside-block{margin-top:1.6rem;border-top:1px solid var(--rule);}
.asks{list-style:none;margin:.8rem 0 0;padding:0;}
.asks li{padding:.75rem 0;border-bottom:1px solid var(--rule);}
.asks li:last-child{border-bottom:none;}
.asks p{margin:.15rem 0 0;font-size:.98rem;line-height:1.55;color:var(--ink);}
.riskbar{display:flex;height:.6rem;border-radius:999px;overflow:hidden;background:var(--rule);margin:.9rem 0 .7rem;gap:2px;}
.riskbar i{display:block;}
.rb-high{background:var(--red);} .rb-medium{background:var(--ochre);} .rb-low{background:var(--green);}
.legend{display:flex;flex-wrap:wrap;gap:.3rem 1.2rem;font-size:.9rem;color:var(--ink-3);}
.terms{margin:.8rem 0 0;}
.terms dt{font-size:.9rem;color:var(--ink-3);margin-top:.7rem;}
.terms dd{margin:.15rem 0 0;}
/* intake */
.intake-hint{margin:.1rem 0 1rem;}
.version-label{font-family:var(--serif);font-size:1.05rem;color:var(--ink);margin:.4rem 0 .45rem;}
[data-testid="stFileUploaderDropzone"]{background:var(--sheet)!important;border:1.5px dashed var(--rule-2)!important;
border-radius:6px!important;padding:1.5rem 1.4rem!important;}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--violet)!important;}
[data-testid="stFileUploaderFile"]{background:var(--sheet)!important;}
/* callouts */
.callout{background:var(--sheet);border:1px solid var(--rule);border-left:3px solid var(--ochre);border-radius:2px 4px 4px 2px;
padding:.8rem 1.05rem .85rem;margin:.9rem 0;}
.callout--error{border-left-color:var(--red);}
.callout--info{border-left-color:var(--violet);}
.callout-title{font-weight:600;color:var(--ink);margin:0;font-size:.98rem;}
.callout-body{color:var(--ink-2);margin:.1rem 0 0;font-size:.95rem;line-height:1.55;max-width:72ch;}
/* widgets */
.stButton button,[data-testid="stDownloadButton"] button,[data-testid="stFormSubmitButton"] button{font-weight:600!important;}
[data-testid="stForm"]{background:var(--sheet)!important;border:1px solid var(--rule)!important;border-radius:6px!important;
box-shadow:var(--lift);padding:1.4rem 1.5rem 1.5rem!important;}
[data-testid="stExpander"] details{background:var(--sheet)!important;border:1px solid var(--rule)!important;border-radius:4px!important;}
[data-testid="stExpander"] summary p{font-weight:500!important;}
[data-testid="stTabs"] [role="tablist"]{gap:1.5rem;}
[data-testid="stTabs"] [role="tab"] p{font-size:1.02rem!important;font-weight:500!important;}
/* sidebar */
[data-testid="stSidebar"]{border-right:1px solid var(--rule);}
[data-testid="stSidebarContent"]{padding-top:.4rem;}
.who-name{font-family:var(--serif);font-size:1.12rem;color:var(--ink);margin:0;}
.reviews{list-style:none;margin:.7rem 0 1rem;padding:0;}
.rev{padding:.75rem 0;border-bottom:1px solid var(--rule);}
.rev:first-child{border-top:1px solid var(--rule);}
.rev-title{font-weight:600;font-size:.95rem;color:var(--ink);line-height:1.35;overflow-wrap:anywhere;margin:0;}
.rev-line{display:flex;flex-wrap:wrap;gap:.2rem .8rem;align-items:center;margin:.25rem 0 .1rem;}
.rev-line .mark{font-size:.85rem;}
/* document questions */
.qa-list{margin-top:.9rem;}
.qa{background:var(--sheet);border:1px solid var(--rule);border-radius:4px;padding:1.05rem 1.3rem 1.15rem;margin-bottom:.75rem;}
.qa-q{font-weight:600;font-size:1.02rem;color:var(--ink);margin:0 0 .35rem;}
.qa-a{font-size:1rem;line-height:1.62;color:var(--ink);margin:0;max-width:72ch;}
.qa-cite{font-size:.75rem;font-weight:600;color:var(--violet);margin-left:.12rem;}
.qa-sources{margin-top:.9rem;padding-top:.7rem;border-top:1px dashed var(--rule-2);}
.stMarkdown blockquote.qa-src{margin:.45rem 0 0!important;padding:.55rem .85rem .6rem!important;background:var(--well);
border-left:2px solid var(--rule-2)!important;border-radius:0 4px 4px 0;font-family:var(--serif);font-size:.95rem;
line-height:1.68;color:var(--ink-2);max-width:72ch;opacity:1!important;}
.qa-src b{font-weight:700;color:var(--ink);}
/* auth */
.specimen{margin-top:1.8rem;padding:1.4rem 1.6rem;max-width:34rem;}
.specimen mark{background:transparent;color:inherit;text-decoration:underline;text-decoration-color:var(--red);
text-decoration-thickness:2px;text-underline-offset:3px;}
.specimen-note{display:flex;gap:.7rem;align-items:baseline;margin-top:1rem;padding-top:.85rem;border-top:1px dashed var(--rule-2);
font-size:.95rem;line-height:1.5;color:var(--ink-2);}
.footer{margin-top:3.2rem;padding-top:1.1rem;border-top:1px solid var(--rule);}
/* responsive */
@media (max-width:900px){
.block-container{padding:2rem 1.2rem 2.4rem!important;}
[data-testid="stHorizontalBlock"]{flex-wrap:wrap!important;row-gap:1rem!important;}
[data-testid="stHorizontalBlock"] [data-testid="stColumn"]{min-width:100%!important;flex:1 1 100%!important;}
.t-display{font-size:2.1rem;}
.dochead{flex-direction:column;gap:1.4rem;}
.stamp{margin:0 0 0 .4rem;}
}
@media (max-width:600px){
.block-container{padding:1.6rem .9rem 2rem!important;}
.sheet{padding:1.3rem 1.15rem;}
.clause{padding:1rem 1rem 1.05rem;}
.t-display{font-size:1.8rem;}
.t-h1{font-size:1.5rem;}
.stamp-ring{min-width:9.5rem;padding:.45rem .8rem;}
.stamp-word{font-size:1.05rem;}
[data-testid="stButtonGroup"] [role="radiogroup"]{gap:1.1rem!important;}
.specimen{display:none;}
}
</style>"""

st.set_page_config(
    page_title="Legaldocs AI",
    page_icon=":material/balance:",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(STYLE, unsafe_allow_html=True)


# ---- SMALL BUILDERS ----

RISK_LABEL = {"high": "High risk", "medium": "Medium risk", "low": "Low risk"}
RISK_ORDER = {"high": 0, "medium": 1, "low": 2}
EXPOSURE_WORD = {"high": "High exposure", "medium": "Moderate exposure", "low": "Low exposure"}
VERSION_STATUSES = ("added", "removed", "modified", "unchanged")


def text(value):
    """Escape for HTML and collapse whitespace so model output can't break a markdown HTML block."""
    return escape(" ".join(str(value).split()))


def html(markup):
    st.markdown(markup, unsafe_allow_html=True)


def titleize(value):
    return str(value).replace("_", " ").title()


def doc_type_label(doc_type):
    label = str(doc_type).replace("_", " ")
    return {"nda": "Non-disclosure agreement", "other": "Agreement"}.get(doc_type, label.capitalize())


def stem(filename):
    return filename[:-4] if filename.lower().endswith(".pdf") else filename


def plural(count, singular, plural_form=None):
    word = singular if count == 1 else (plural_form or singular + "s")
    return f"{count} {word}"


def mark(level, label=None):
    return f'<span class="mark mark--{level}">{text(label or RISK_LABEL.get(level, "Unrated"))}</span>'


def callout(title, body, tone="warn"):
    html(
        f'<div class="callout callout--{tone}"><p class="callout-title">{text(title)}</p>'
        f'<p class="callout-body">{text(body)}</p></div>'
    )


def stamp(top, word, score, when, label):
    return (
        f'<div class="stamp" role="img" aria-label="{text(label)}"><div class="stamp-ring">'
        f'<span class="stamp-top">{text(top)}</span><span class="stamp-word">{text(word)}</span>'
        f'<span class="stamp-score">{text(score)}</span>'
        f'<span class="stamp-date">{when:%d %b %Y}</span></div></div>'
    )


def law_note(note):
    if not note or note.startswith("No specific rules"):
        return ""
    return (
        '<div class="note"><p class="note-label">Under Indian law</p>'
        f'<p class="note-body">{text(note)}</p></div>'
    )


def brand(subtitle=True):
    small = "<small>for Indian contracts</small>" if subtitle else ""
    html(f'<div class="brand"><span class="seal" aria-hidden="true">Ld</span>'
         f'<span class="brand-name">Legaldocs{small}</span></div>')


def footer():
    html(
        '<div class="footer"><p class="t-small">Legaldocs flags risks automatically and is not legal advice. '
        'If anything in an agreement worries you, speak to an advocate before you sign.</p></div>'
    )


def file_digest(data):
    return hashlib.sha256(data).hexdigest()


# ---- ANALYSIS PLUMBING ----

def run_analysis(task):
    """Run task(on_step) under a live status line; returns None if it failed."""
    holder = st.empty()
    error = None
    with holder.status("Starting the review") as status:
        try:
            result = task(lambda message: status.update(label=message))
        except AnalysisError as exc:
            error, result = str(exc), None
    holder.empty()
    if error:
        callout("The review stopped", error, "error")
        return None
    result["reviewed_at"] = datetime.now()
    return result


def risk_counts(clauses):
    return {level: sum(1 for c in clauses if c["risk_level"] == level) for level in ("high", "medium", "low")}


def save_to_history(kind, title, doc_type, risk_score, overall_risk, clauses, details):
    try:
        record_analysis(
            st.session_state["user"]["id"], kind, title, doc_type,
            risk_score, overall_risk, risk_counts(clauses), details,
        )
    except DatabaseUnavailable:
        callout("Not saved to your history",
                "The review is shown below, but the database could not be reached to save it.")


def save_single(result):
    save_to_history(
        "single", result["filename"], result["doc_type"], result["risk_score"], result["overall_risk"],
        result["clauses"], {"clauses": len(result["clauses"]), "scanned": result["scanned"]},
    )


def save_comparison(result):
    earlier, later, rows = result["earlier"], result["later"], result["rows"]
    save_to_history(
        "comparison", f'{earlier["name"]} → {later["name"]}', result["shared_type"],
        later["risk_score"], later["overall_risk"], later["clauses"],
        {
            "earlier": earlier["name"],
            "later": later["name"],
            "earlier_score": earlier["risk_score"],
            "later_score": later["risk_score"],
            "changes": {status: sum(1 for r in rows if r["status"] == status) for status in VERSION_STATUSES},
        },
    )


def showing_previous(label, state_key):
    cols = st.columns([5, 1], gap="small", vertical_alignment="center")
    with cols[0]:
        html(f'<p class="t-small">You are looking at your last review of {text(label)}. '
             'Upload a different file to review it instead.</p>')
    with cols[1]:
        if st.button("Clear", key=f"clear_{state_key}", use_container_width=True):
            del st.session_state[state_key]
            st.rerun()


# ---- SINGLE REVIEW ----

def render_single_document():
    html('<div class="t-h2" role="heading" aria-level="2">Upload an agreement</div>'
         '<p class="t-small intake-hint">A PDF, typed or scanned. Loan, rental, employment, non-disclosure, '
         'insurance and credit card agreements are read best.</p>')
    uploaded = st.file_uploader(
        "Agreement PDF", type=["pdf"], key="single_upload", label_visibility="collapsed"
    )

    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        digest = file_digest(file_bytes)
        stored = st.session_state.get("single")
        if not stored or stored["digest"] != digest:
            result = run_analysis(lambda on_step: analyze_single(
                file_bytes, uploaded.name, load_ocr_model, load_spacy_model(), on_step
            ))
            if result is not None:
                result["digest"] = digest
                # Questions read the raw text: it keeps line breaks (clause headings) and the
                # quotes, apostrophes and currency symbols that clean_text strips.
                result["qa_text"] = (
                    result["text"] if result["scanned"] else extract_text_digital(io.BytesIO(file_bytes))
                )
                st.session_state["single"] = result
                if result["clauses"]:
                    save_single(result)

    result = st.session_state.get("single")
    if result is None:
        return
    if uploaded is None:
        showing_previous(result["filename"], "single")
    render_single_result(result)


def single_summary(clauses):
    counts = risk_counts(clauses)
    reviewed = f'{plural(len(clauses), "clause")} reviewed.'
    if counts["high"]:
        verb = "needs" if counts["high"] == 1 else "need"
        return f'{reviewed} {counts["high"]} {verb} your attention before you sign.'
    if counts["medium"]:
        return f'{reviewed} None is high risk, but {counts["medium"]} deserve a careful read.'
    return f"{reviewed} Nothing stands out as risky."


def render_single_result(result):
    clauses = sorted(result["clauses"], key=lambda c: RISK_ORDER.get(c["risk_level"], 1))
    source = "Scanned PDF, read with OCR" if result["scanned"] else "Digital PDF"

    stamp_html = ""
    if clauses:
        stamp_html = stamp(
            "Legaldocs reviewed", EXPOSURE_WORD[result["overall_risk"]],
            f'{result["risk_score"]} / 100', result["reviewed_at"],
            f'Verdict: {EXPOSURE_WORD[result["overall_risk"]].lower()}, {result["risk_score"]} out of 100',
        )
    summary = single_summary(clauses) if clauses else (
        "No clauses could be picked out of this document. Try a clearer scan or a different agreement."
    )
    html(
        f'<div class="sheet dochead"><div class="dochead-main">'
        f'<p class="t-small">{text(doc_type_label(result["doc_type"]))}</p>'
        f'<div class="t-h1" role="heading" aria-level="1">{text(stem(result["filename"]))}</div>'
        f'<p class="t-body">{text(summary)}</p>'
        f'<div class="facts"><span><b>{result["size"] / 1024:,.0f} KB</b></span><span>{source}</span>'
        f'<span>{len(result["text"]):,} characters read</span></div>'
        f'</div>{stamp_html}</div>'
    )

    action_cols = st.columns([1, 2.4], gap="medium", vertical_alignment="top")
    with action_cols[0]:
        if clauses:
            if "report_pdf" not in result:
                result["report_pdf"] = generate_pdf_report(
                    result["doc_type"], result["risk_score"], result["overall_risk"],
                    clauses, result["questions"], result["entities"],
                ).getvalue()
            st.download_button(
                "Download report (PDF)", data=result["report_pdf"], type="primary",
                file_name=f'{slugify(stem(result["filename"]))}_review.pdf', mime="application/pdf",
                use_container_width=True,
            )
    with action_cols[1]:
        with st.expander("Text read from the document"):
            html(f'<p class="clause-text">{text(result["text"])}</p>')

    render_document_qa(result)

    if not clauses:
        return

    main_col, side_col = st.columns([1.75, 1], gap="large")
    with main_col:
        html('<div class="t-h2" role="heading" aria-level="2">Clause by clause</div>'
             '<p class="t-small t-lede">Ordered from the highest risk down.</p>')
        html("".join(
            f'<article class="clause clause--{c["risk_level"]}"><div class="clause-head">'
            f'<span class="t-h3" role="heading" aria-level="3">{text(titleize(c["clause_name"]))}</span>'
            f'{mark(c["risk_level"])}</div>'
            f'<p class="clause-text">{text(c["clause_text"])}</p>{law_note(c["india_note"])}</article>'
            for c in clauses
        ))
    with side_col:
        render_single_aside(result, clauses)


CITATION = re.compile(r"\[(\d{1,2})\]")


def qa_entry(entry):
    answer = CITATION.sub(r'<sup class="qa-cite">\1</sup>', text(entry["answer"]))
    if not entry["answered"]:
        return (
            f'<article class="qa qa--none"><p class="qa-q">{text(entry["question"])}</p>'
            f'<p class="qa-a">{mark("muted", "Not covered in this agreement")}</p>'
            f'<p class="t-small" style="margin-top:.3rem">{answer}</p></article>'
        )
    sources = "".join(
        f'<blockquote class="qa-src"><sup class="qa-cite">{p["n"]}</sup> '
        f'{f"<b>{text(p["heading"])}.</b> " if p["heading"] else ""}{text(p["text"])}</blockquote>'
        for p in entry["passages"]
    )
    return (
        f'<article class="qa"><p class="qa-q">{text(entry["question"])}</p><p class="qa-a">{answer}</p>'
        f'<div class="qa-sources"><p class="t-small">From the agreement</p>{sources}</div></article>'
    )


def render_document_qa(result):
    html('<div class="t-h2" role="heading" aria-level="2">Ask about this agreement</div>'
         '<p class="t-small t-lede">Answers come only from the text of this document, with the passages '
         'they rely on. It reads the whole document, including parts the clause review does not cover.</p>')

    # A fresh key after each answer gives an empty box; a failed question keeps its key and its text.
    round_ = st.session_state.get("qa_round", 0)
    with st.form("qa_form", border=False):
        cols = st.columns([5, 1], gap="small", vertical_alignment="bottom")
        question = cols[0].text_input(
            "Your question", key=f"qa_question_{round_}", max_chars=300, label_visibility="collapsed",
            placeholder="For example: what happens if I miss a payment?",
        )
        asked = cols[1].form_submit_button("Ask", type="primary", use_container_width=True)

    if asked and question.strip():
        try:
            with st.spinner("Searching the agreement. The first question takes a little longer."):
                entry = answer_question(question.strip(), result["digest"], result.get("qa_text") or result["text"])
        except AnalysisError as exc:
            callout("The question could not be answered", str(exc), "error")
        else:
            result.setdefault("qa", []).insert(0, entry)
            st.session_state.pop(f"qa_question_{round_}", None)
            st.session_state["qa_round"] = round_ + 1
            st.rerun()

    if result.get("qa"):
        html(f'<div class="qa-list">{"".join(qa_entry(e) for e in result["qa"])}</div>')


def render_single_aside(result, clauses):
    blocks = []

    if result["questions"]:
        items = "".join(
            f'<li><p class="t-small">{text(titleize(q["clause_name"]))}</p><p>{text(q["question"])}</p></li>'
            for q in result["questions"]
        )
        blocks.append('<section class="aside-block"><div class="t-h3" role="heading" aria-level="2">'
                      f'Ask before you sign</div><ul class="asks">{items}</ul></section>')
    else:
        reason = result["questions_error"] or "No clause here is risky enough to need a question."
        blocks.append('<section class="aside-block"><div class="t-h3" role="heading" aria-level="2">'
                      f'Ask before you sign</div><p class="t-small" style="margin-top:.6rem">{text(reason)}</p></section>')

    counts = risk_counts(clauses)
    segments = "".join(
        f'<i class="rb-{level}" style="flex:{counts[level]}"></i>' for level in ("high", "medium", "low") if counts[level]
    )
    legend = "".join(
        f'<span>{mark(level, level.capitalize())} {counts[level]}</span>' for level in ("high", "medium", "low")
    )
    blocks.append(
        '<section class="aside-block"><div class="t-h3" role="heading" aria-level="2">Where the risk sits</div>'
        f'<div class="riskbar" role="img" aria-label="{counts["high"]} high, {counts["medium"]} medium, '
        f'{counts["low"]} low risk clauses">{segments}</div><div class="legend">{legend}</div>'
        f'<p class="t-small" style="margin-top:.7rem">Exposure score {result["risk_score"]} out of 100. '
        'Higher means the agreement leans further against you.</p></section>'
    )

    entities = result["entities"]
    groups = [
        ("Amounts", [m for m in entities["money"] if "billion" not in m.lower() and "million" not in m.lower()]),
        ("Rates", entities["percentages"]),
        ("Dates and periods", entities["dates"]),
        ("Parties named", entities["organizations"] + entities["persons"]),
    ]
    rows = "".join(
        f'<dt>{label}</dt><dd>{"".join(f"<span class=tok>{text(v)}</span>" for v in values[:12])}</dd>'
        for label, values in groups if values
    )
    if rows:
        blocks.append('<section class="aside-block"><div class="t-h3" role="heading" aria-level="2">'
                      f'Terms found in the text</div><dl class="terms">{rows}</dl></section>')

    html("".join(blocks))


# ---- VERSION COMPARISON ----

def render_version_comparison():
    html('<div class="t-h2" role="heading" aria-level="2">Compare two versions</div>'
         '<p class="t-small intake-hint">Upload an earlier and a later version of the same agreement, '
         "such as last year's loan terms and this year's, to see what changed for you.</p>")

    cols = st.columns(2, gap="medium")
    with cols[0]:
        html('<p class="version-label">Earlier version</p>')
        earlier = st.file_uploader("Earlier version", type=["pdf"], key="version_earlier",
                                   label_visibility="collapsed")
    with cols[1]:
        html('<p class="version-label">Later version</p>')
        later = st.file_uploader("Later version", type=["pdf"], key="version_later",
                                 label_visibility="collapsed")

    if earlier is not None and later is not None:
        earlier_bytes, later_bytes = earlier.getvalue(), later.getvalue()
        digest = (file_digest(earlier_bytes), file_digest(later_bytes))
        stored = st.session_state.get("comparison")
        if not stored or stored["digest"] != digest:
            result = run_analysis(lambda on_step: analyze_comparison(
                (earlier_bytes, earlier.name), (later_bytes, later.name), load_ocr_model, on_step
            ))
            if result is not None:
                result["digest"] = digest
                st.session_state["comparison"] = result
                if result["rows"]:
                    save_comparison(result)

    result = st.session_state.get("comparison")
    if result is None:
        return
    if earlier is None or later is None:
        showing_previous(f'{result["earlier"]["name"]} and {result["later"]["name"]}', "comparison")
    render_comparison_result(result)


def comparison_headline(delta):
    if delta > 0:
        return "The later version carries more risk", "Less favourable"
    if delta < 0:
        return "The later version is better for you", "More favourable"
    return "Overall risk is the same in both versions", "No net change"


def comparison_summary(rows):
    counts = {status: sum(1 for r in rows if r["status"] == status) for status in VERSION_STATUSES}
    riskier = sum(1 for r in rows if r["risk_shift"] > 0)
    safer = sum(1 for r in rows if r["risk_shift"] < 0)
    parts = []
    if counts["modified"]:
        detail = []
        if riskier:
            detail.append(f"{riskier} now riskier")
        if safer:
            detail.append(f"{safer} now safer")
        changed = plural(counts["modified"], "clause") + " changed"
        parts.append(changed + (f' ({", ".join(detail)})' if detail else ""))
    if counts["added"]:
        parts.append(plural(counts["added"], "clause") + " added")
    if counts["removed"]:
        parts.append(plural(counts["removed"], "clause") + " removed")
    if counts["unchanged"]:
        parts.append(f'{counts["unchanged"]} unchanged')
    if not parts:
        return "No differences were found."
    sentence = "; ".join(parts)
    return sentence[0].upper() + sentence[1:] + "."


def version_line(label, doc):
    return (
        f'<div><p class="t-small">{label}</p><p class="rev-title" style="font-size:1rem">{text(doc["name"])}</p>'
        f'<div class="rev-line">{mark(doc["overall_risk"])}'
        f'<span class="t-small">Exposure {doc["risk_score"]}</span></div></div>'
    )


def render_comparison_result(result):
    earlier, later, rows = result["earlier"], result["later"], result["rows"]
    shared_type = result["shared_type"]

    if result["type_mismatch"]:
        callout(
            "These may be different kinds of agreement",
            f'The earlier version reads as a {doc_type_label(shared_type).lower()} and the later one as a '
            f'{doc_type_label(later["doc_type"]).lower()}. Both were read as a {doc_type_label(shared_type).lower()} '
            "so their clauses could be lined up, so treat the comparison with care.",
        )
    if result["truncated"]:
        which = " and ".join(f'the {slot} version ({result[slot]["text_length"]:,} characters)'
                             for slot in result["truncated"])
        callout(
            "Only the opening of each document was compared",
            f"Clauses are read from the first {CLAUSE_READ_LIMIT:,} characters of each document, and {which} "
            "runs longer. Clauses after that point may show up as added or removed.",
            "info",
        )

    if not rows:
        missing = " and ".join(slot for slot in ("earlier", "later") if not result[slot]["clauses"])
        callout("Nothing to compare",
                f"No clauses could be picked out of the {missing} version. Try a clearer scan or a different file.",
                "error")
        return

    delta = later["risk_score"] - earlier["risk_score"]
    headline, verdict = comparison_headline(delta)
    shown_delta = f"{delta:+d} points" if delta else "0 points"
    html(
        f'<div class="sheet dochead"><div class="dochead-main">'
        f'<p class="t-small">{text(doc_type_label(shared_type))}, two versions</p>'
        f'<div class="t-h1" role="heading" aria-level="1">{text(headline)}</div>'
        f'<p class="t-body">{text(comparison_summary(rows))}</p>'
        f'<div class="facts" style="gap:1rem 2.6rem">{version_line("Earlier", earlier)}{version_line("Later", later)}</div>'
        f'</div>{stamp("Legaldocs compared", verdict, shown_delta, result["reviewed_at"], f"Verdict: {verdict.lower()}, {shown_delta}")}'
        f'</div>'
    )

    changed = [r for r in rows if r["status"] != "unchanged"]
    unchanged = [r for r in rows if r["status"] == "unchanged"]

    if changed:
        html('<div class="t-h2" role="heading" aria-level="2">What changed</div>'
             '<p class="t-small t-lede">Changes that raise your risk come first.</p>')
        html("".join(change_article(row, result["summaries"].get(slugify(row["name"]), "")) for row in changed))
    if unchanged:
        with st.expander(f"Clauses that did not change ({len(unchanged)})"):
            html("".join(
                f'<article class="clause clause--{r["new"]["risk_level"]}"><div class="clause-head">'
                f'<span class="t-h3">{text(titleize(r["name"]))}</span>{mark(r["new"]["risk_level"])}</div>'
                f'<p class="clause-text">{text(r["new"]["clause_text"])}</p></article>'
                for r in unchanged
            ))


def change_article(row, summary):
    status, old, new, shift = row["status"], row["old"], row["new"], row["risk_shift"]

    if status == "added":
        tone, badge = "new", mark("new", "Added in the later version")
        shift_html = f'<div class="shift">{mark(new["risk_level"])}</div>'
        body = f'<p class="clause-text">{text(new["clause_text"])}</p>'
    elif status == "removed":
        tone, badge = "muted", mark("muted", "Removed from the later version")
        shift_html = f'<div class="shift"><span class="t-small">Was</span> {mark(old["risk_level"])}</div>'
        body = f'<p class="clause-text" style="color:var(--ink-3)">{text(old["clause_text"])}</p>'
    else:
        tone = "high" if shift > 0 else ("low" if shift < 0 else "muted")
        direction = "Riskier" if shift > 0 else ("Safer" if shift < 0 else "Reworded")
        badge = mark(tone if shift else "muted", direction)
        if shift:
            shift_html = (f'<div class="shift">{mark(old["risk_level"])}'
                          f'<span class="t-small">became</span>{mark(new["risk_level"])}</div>')
        else:
            shift_html = f'<div class="shift"><span class="t-small">Still</span> {mark(new["risk_level"])}</div>'
        body = f'<div class="redline">{redline_html(" ".join(old["clause_text"].split()), " ".join(new["clause_text"].split()))}</div>'

    terms = ""
    if row["terms_removed"] or row["terms_added"]:
        out = "".join(f'<span class="tok tok--out">{text(t)}</span>' for t in row["terms_removed"])
        into = "".join(f'<span class="tok tok--in">{text(t)}</span>' for t in row["terms_added"])
        terms = ('<div class="termdiff">'
                 + (f'<span>Dropped {out}</span>' if out else "")
                 + (f'<span>New {into}</span>' if into else "")
                 + '</div>')

    what = ""
    if summary:
        what = f'<div class="note" style="margin:0 0 .85rem"><p class="note-label">In plain words</p><p class="note-body">{text(summary)}</p></div>'

    note = law_note((new or old)["india_note"])
    return (
        f'<article class="clause clause--{tone}"><div class="clause-head">'
        f'<span class="t-h3" role="heading" aria-level="3">{text(titleize(row["name"]))}</span>{badge}</div>'
        f'{shift_html}{what}{terms}{body}{note}</article>'
    )


# ---- ACCOUNT ----

def sign_in(user):
    # Start every account from a clean session so nothing carries over between users on one tab.
    st.session_state.clear()
    st.session_state["user"] = user
    st.rerun()


def render_auth():
    intro, forms = st.columns([1.15, 0.85], gap="large")
    with intro:
        brand(subtitle=False)
        html(
            '<div style="height:2.2rem"></div>'
            '<div class="t-display" role="heading" aria-level="1">Know what an agreement asks of you before you sign it.</div>'
            '<p class="t-body">Upload a loan, rental or employment agreement. Legaldocs reads each clause, '
            'checks it against Indian law and RBI rules, and tells you what to ask before you sign.</p>'
            '<div class="sheet specimen"><p class="clause-text">4. Prepayment. The Borrower may prepay this '
            'floating-rate loan after twelve months, <mark>subject to a prepayment charge of 4% of the '
            'outstanding principal</mark>.</p>'
            f'<div class="specimen-note">{mark("high")}<span>An RBI circular mandates no prepayment penalty '
            'on floating-rate loans.</span></div></div>'
        )
    with forms:
        html('<div style="height:3.6rem"></div>')
        sign_in_tab, register_tab = st.tabs(["Sign in", "Create an account"])
        with sign_in_tab:
            with st.form("sign_in_form", border=False):
                email = st.text_input("Email", autocomplete="email")
                password = st.text_input("Password", type="password", autocomplete="current-password")
                submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
            if submitted:
                try:
                    sign_in(authenticate(email, password))
                except AuthError as exc:
                    callout("Could not sign you in", str(exc), "error")
        with register_tab:
            with st.form("register_form", border=False):
                full_name = st.text_input("Your name", autocomplete="name")
                new_email = st.text_input("Email", autocomplete="email", key="register_email")
                new_password = st.text_input("Password", type="password", autocomplete="new-password",
                                             key="register_password", help="At least 8 characters.")
                confirm = st.text_input("Type the password again", type="password", autocomplete="new-password")
                created = st.form_submit_button("Create account", type="primary", use_container_width=True)
            if created:
                try:
                    sign_in(register(full_name, new_email, new_password, confirm))
                except AuthError as exc:
                    callout("Could not create the account", str(exc), "error")
    footer()


def review_entry(entry):
    when = entry["created_at"].astimezone()
    stamp_text = f'{when:%d %b %Y}, {when.strftime("%I:%M %p").lstrip("0").lower()}'
    if entry["kind"] == "comparison":
        details = entry["details"]
        title = stem(details.get("later", entry["title"]))
        sub = f'Compared with {stem(details.get("earlier", ""))}'
        score = f'Exposure {details.get("earlier_score", "?")} to {details.get("later_score", "?")}'
    else:
        title = stem(entry["title"])
        sub = doc_type_label(entry["doc_type"])
        score = f'Exposure {entry["risk_score"]}'
    return (
        f'<li class="rev"><p class="rev-title">{text(title)}</p>'
        f'<p class="t-small">{text(sub)}</p>'
        f'<div class="rev-line">{mark(entry["overall_risk"])}<span class="t-small">{text(score)}</span></div>'
        f'<p class="t-small">{stamp_text}</p></li>'
    )


def render_sidebar(user):
    with st.sidebar:
        html(f'<p class="who-name">{text(user["full_name"])}</p><p class="t-small">{text(user["email"])}</p>')
        if st.button("Sign out", key="sign_out"):
            st.session_state.clear()
            st.rerun()

        html('<div style="height:1.4rem"></div><div class="t-h3" role="heading" aria-level="2">Your reviews</div>')
        history = list_history(user["id"])
        if not history:
            html('<p class="t-small" style="margin-top:.5rem">Reviews you run are saved here, '
                 'visible only to you.</p>')
            return
        html(f'<ul class="reviews">{"".join(review_entry(e) for e in history)}</ul>')

        if st.session_state.get("confirm_clear"):
            html('<p class="t-small">Delete your whole review history? This cannot be undone.</p>')
            yes, no = st.columns(2)
            if yes.button("Delete", type="primary", key="clear_yes", use_container_width=True):
                clear_history(user["id"])
                del st.session_state["confirm_clear"]
                st.rerun()
            if no.button("Keep", key="clear_no", use_container_width=True):
                del st.session_state["confirm_clear"]
                st.rerun()
        elif st.button("Clear history", type="tertiary", key="clear_history"):
            st.session_state["confirm_clear"] = True
            st.rerun()


# ---- PAGE ----

def database_down(exc):
    callout("The database is not reachable",
            f"{exc} Start the PostgreSQL server, then reload this page.", "error")


def main():
    try:
        ensure_schema()
    except DatabaseUnavailable as exc:
        brand()
        database_down(exc)
        return

    user = st.session_state.get("user")
    if user is None:
        render_auth()
        return

    brand()
    mode = st.segmented_control(
        "Review mode",
        ["Review an agreement", "Compare two versions"],
        default="Review an agreement",
        required=True,
        key="review_mode",
        label_visibility="collapsed"
    )
    if mode == "Compare two versions":
        render_version_comparison()
    else:
        render_single_document()
    footer()
    # Drawn last so a review saved during this run already appears in the history.
    render_sidebar(user)


try:
    main()
except DatabaseUnavailable as exc:
    database_down(exc)
