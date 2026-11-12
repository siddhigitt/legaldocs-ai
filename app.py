import streamlit as st
from groq import Groq
import json
import spacy
import pandas as pd
import pdfplumber
from pdf2image import convert_from_bytes
from PIL import Image
from dotenv import load_dotenv
import easyocr
import numpy as np
import os
import re
import sqlite3
from datetime import datetime
from html import escape as esc

def init_database():
    conn = sqlite3.connect("lexscan_history.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            doc_type TEXT,
            risk_score INTEGER,
            overall_risk TEXT,
            high_risk_count INTEGER,
            medium_risk_count INTEGER,
            low_risk_count INTEGER,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_analysis(filename, doc_type, risk_score, overall_risk, scored_clauses):
    conn = sqlite3.connect("lexscan_history.db")
    cursor = conn.cursor()

    high_count = sum(1 for c in scored_clauses if c["risk_level"] == "high")
    medium_count = sum(1 for c in scored_clauses if c["risk_level"] == "medium")
    low_count = sum(1 for c in scored_clauses if c["risk_level"] == "low")

    cursor.execute("""
        INSERT INTO analyses
        (filename, doc_type, risk_score, overall_risk, high_risk_count, medium_risk_count, low_risk_count, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        filename,
        doc_type,
        risk_score,
        overall_risk,
        high_count,
        medium_count,
        low_count,
        datetime.now().strftime("%d %B %Y, %I:%M %p")
    ))

    conn.commit()
    conn.close()

def get_history():
    conn = sqlite3.connect("lexscan_history.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM analyses ORDER BY id DESC LIMIT 10")
    rows = cursor.fetchall()
    conn.close()
    return rows

# Load API key
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# ---- LOAD MODELS ----
@st.cache_resource
def load_ocr_model():
    reader = easyocr.Reader(['en'], gpu=False)
    return reader

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

# ---- FUNCTIONS ----

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'[^\w\s.,;:!?()%\-/+]', '', text)
    return text.strip()

def extract_text_digital(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_easyocr(pdf_file, reader):
    text = ""
    images = convert_from_bytes(
        pdf_file.read(),
        poppler_path=r"C:\Program Files\poppler\Library\bin"
    )
    for image in images:
        image_np = np.array(image)
        results = reader.readtext(image_np)
        page_text = " ".join([result[1] for result in results])
        text += page_text + "\n"
    return text

def is_scanned_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                return False
    return True

def detect_document_type(text, api_key):
    client = Groq(api_key=api_key)
    sample_text = text[:1000]
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": "You are a legal document classifier for Indian contracts. Reply with ONLY one of these exact words: loan_agreement, rent_agreement, employment_offer, nda, insurance, credit_card, other"
            },
            {
                "role": "user",
                "content": f"What type of legal document is this?\n\n{sample_text}"
            }
        ]
    )
    return response.choices[0].message.content.strip().lower()

def extract_clauses(text, doc_type, api_key):
    client = Groq(api_key=api_key)

    prompt = f"""You are a legal clause extractor for Indian contracts.

Extract all important clauses from this {doc_type.replace('_', ' ')} document.

Return ONLY a JSON object — no explanation, no markdown, no extra text.
Just the raw JSON starting with {{ and ending with }}.

Format:
{{
    "clauses": [
        {{
            "clause_name": "name of clause in snake_case",
            "clause_text": "exact relevant text from document"
        }}
    ]
}}

For a loan agreement extract: interest_rate, late_penalty, prepayment, tenure, insurance, arbitration
For a rent agreement extract: monthly_rent, security_deposit, notice_period, maintenance, lock_in_period
For an employment offer extract: salary, notice_period, probation, non_compete, intellectual_property, termination
For an NDA extract: confidentiality_period, scope, exceptions, penalties
For other documents extract: whatever important terms exist

Important: Extract the COMPLETE clause text including all sentences
in that clause. Do not truncate or summarize — copy the full clause.

Document text:
{text[:3000]}

JSON output:"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": "You are a legal clause extractor. You return ONLY valid JSON. Never include explanations or markdown formatting."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1
    )

    raw_response = response.choices[0].message.content.strip()

    if "```json" in raw_response:
        raw_response = raw_response.split("```json")[1].split("```")[0].strip()
    elif "```" in raw_response:
        raw_response = raw_response.split("```")[1].split("```")[0].strip()

    try:
        parsed = json.loads(raw_response)
        return parsed.get("clauses", [])
    except json.JSONDecodeError:
        st.warning("Could not parse clauses — try uploading again")
        return []

def extract_entities(text, nlp):
    from spacy.matcher import Matcher

    doc = nlp(text)

    entities = {
        "money": [],
        "dates": [],
        "percentages": [],
        "organizations": [],
        "persons": []
    }

    ignore_list = [
        "register", "lpa", "nda", "internship", "ppo", "nb",
        "duration", "characteristics", "organization", "cum"
    ]

    for ent in doc.ents:
        text_clean = ent.text.strip()

        if len(text_clean) < 2:
            continue
        if text_clean.lower() in ignore_list:
            continue
        if "/" in text_clean or "+" in text_clean:
            continue
        if text_clean.isupper() and len(text_clean) > 8:
            continue
        if ent.label_ == "ORG" and len(text_clean.split()) > 3:
            continue
        false_org_phrases = [
            "internship cum", "personal characteristics", "internship duration",
            "sw development", "swdevelopment", "based ppo", "recruitment drive",
            "school of computer", "the organization", "royal philips"
        ]
        if any(phrase in text_clean.lower() for phrase in false_org_phrases):
            continue
        if "billion" in text_clean.lower() or "million" in text_clean.lower():
            continue

        if ent.label_ == "MONEY" and text_clean not in entities["money"]:
            entities["money"].append(text_clean)
        elif ent.label_ == "DATE" and text_clean not in entities["dates"]:
            entities["dates"].append(text_clean)
        elif ent.label_ == "PERCENT" and text_clean not in entities["percentages"]:
            entities["percentages"].append(text_clean)
        elif ent.label_ == "ORG" and text_clean not in entities["organizations"]:
            entities["organizations"].append(text_clean)
        elif ent.label_ == "PERSON" and text_clean not in entities["persons"]:
            entities["persons"].append(text_clean)

    indian_money_patterns = [
        r'Rs\.?\s*[\d,]+(?:\.\d+)?(?:\s*(?:LPA|lpa|per month|/month|lakhs?|crores?))?',
        r'₹\s*[\d,]+(?:\.\d+)?(?:\s*(?:LPA|lpa|per month|/month|lakhs?|crores?))?',
        r'INR\s*[\d,]+(?:\.\d+)?',
        r'[\d,]+(?:\.\d+)?\s*LPA',
        r'[\d,]+(?:\.\d+)?\s*(?:lakhs?|crores?)',
    ]

    for pattern in indian_money_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            match_clean = match.strip()
            if match_clean and match_clean not in entities["money"]:
                entities["money"].append(match_clean)

    return entities

def score_clause_risk(clause_name, clause_text, doc_type):
    with open("baseline_rules.json", "r") as f:
        rules = json.load(f)

    if doc_type not in rules:
        return "medium", "No specific rules for this document type"

    doc_rules = rules[doc_type]

    clause_key = clause_name.lower().replace(" ", "_")

    matched_key = None
    if clause_key in doc_rules:
        matched_key = clause_key
    else:
        for rule_key in doc_rules.keys():
            if rule_key in clause_key or clause_key in rule_key:
                matched_key = rule_key
                break

    if not matched_key:
        return "medium", "No specific rules for this clause"

    clause_rules = doc_rules[matched_key]
    india_note = clause_rules.get("india_note", "")
    clause_lower = clause_text.lower()

    red_keywords = [
        "mandatory insurance", "compulsory insurance", "mandatory purchase",
        "mandatory arbitration", "mandatory deduction", "non refundable deposit",
        "non-refundable deposit", "unlimited liability", "no upper cap",
        "no maximum cap", "no cap on penalty", "no maximum limit",
        "no maximum limit or upper cap", "mandatory and bundled",
        "shall be mandatory", "insurance shall be mandatory",
        "lender appointed arbitrator", "lender shall appoint",
        "appointed exclusively by the bank", "appointed exclusively by",
        "sole arbitrator shall be appointed", "exclusively by the lender",
        "exclusively by the bank", "perpetual confidentiality",
        "perpetual obligation", "all intellectual property", "assign all ip",
        "including personal projects", "outside scope of employment",
        "no prepayment allowed", "prepayment not permitted"
    ]

    green_keywords = [
        "insurance is optional", "insurance optional", "may opt out",
        "can opt out", "prepayment without penalty", "no prepayment charge",
        "prepayment charge nil", "neutral arbitrator",
        "mutually appointed arbitrator", "limited to work performed",
        "limited to employment scope", "refundable deposit",
        "deposit shall be refunded", "waived on request"
    ]

    for keyword in red_keywords:
        if keyword in clause_lower:
            return "high", india_note

    for keyword in green_keywords:
        if keyword in clause_lower:
            return "low", india_note

    numbers = re.findall(r'\d+\.?\d*', clause_text)

    if numbers:
        value = float(numbers[0])

        if "interest" in clause_key:
            if value > 18:
                return "high", india_note
            elif value > 12:
                return "medium", india_note
            else:
                return "low", india_note

        if "notice" in clause_key:
            if value > 90:
                return "high", india_note
            elif value > 30:
                return "medium", india_note
            else:
                return "low", india_note

        if "security" in clause_key or "deposit" in clause_key:
            if value > 3:
                return "high", india_note
            elif value > 2:
                return "medium", india_note
            else:
                return "low", india_note

        if "probation" in clause_key:
            if value > 6:
                return "high", india_note
            elif value > 3:
                return "medium", india_note
            else:
                return "low", india_note

        if "confidentiality" in clause_key:
            if value > 5:
                return "high", india_note
            elif value > 2:
                return "medium", india_note
            else:
                return "low", india_note

        if "prepay" in clause_key or "prepayment" in clause_key or "foreclosure" in clause_key:
            if value > 2:
                return "high", india_note
            elif value > 1:
                return "medium", india_note
            else:
                return "low", india_note

    return "medium", india_note

def calculate_overall_risk(scored_clauses):
    if not scored_clauses:
        return 100, "low"

    score = 100

    for clause in scored_clauses:
        if clause["risk_level"] == "high":
            score -= 15
        elif clause["risk_level"] == "medium":
            score -= 7

    score = max(0, score)

    if score >= 75:
        overall_risk = "low"
    elif score >= 50:
        overall_risk = "medium"
    else:
        overall_risk = "high"

    return score, overall_risk

def generate_questions(scored_clauses, doc_type, api_key):
    client = Groq(api_key=api_key)

    risky_clauses = [
        c for c in scored_clauses
        if c["risk_level"] in ["high", "medium"]
    ]

    if not risky_clauses:
        return []

    clause_summary = ""
    for clause in risky_clauses:
        clause_summary += f"- {clause['clause_name']}: {clause['clause_text'][:200]}\n"

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": "You are a legal advisor helping Indians understand contracts. Generate specific, practical questions they should ask before signing. Return ONLY a JSON array of objects with clause_name and question fields. No markdown, no explanation."
            },
            {
                "role": "user",
                "content": f"""For this {doc_type.replace('_', ' ')} document, generate one specific question
to ask for each risky clause before signing.

Risky clauses found:
{clause_summary}

Return ONLY this JSON format:
[
    {{"clause_name": "clause name", "question": "specific question to ask"}}
]

JSON:"""
            }
        ],
        temperature=0.3
    )

    raw_response = response.choices[0].message.content.strip()

    if "```json" in raw_response:
        raw_response = raw_response.split("```json")[1].split("```")[0].strip()
    elif "```" in raw_response:
        raw_response = raw_response.split("```")[1].split("```")[0].strip()

    try:
        questions = json.loads(raw_response)
        return questions
    except json.JSONDecodeError:
        return []

def generate_pdf_report(doc_type, risk_score, overall_risk, scored_clauses, questions, entities):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    import io
    from datetime import datetime

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        textColor=colors.HexColor('#1a1a2e'),
        spaceAfter=6
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#1a1a2e'),
        spaceBefore=12,
        spaceAfter=6
    )

    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=4
    )

    small_style = ParagraphStyle(
        'CustomSmall',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#666666'),
        spaceAfter=4
    )

    elements = []

    elements.append(Paragraph("Legaldocs AI", title_style))
    elements.append(Paragraph("Legal Document Risk Analysis Report", styles['Heading2']))
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%d %B %Y, %I:%M %p')}", small_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cccccc')))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Document Overview", heading_style))
    elements.append(Paragraph(f"<b>Document Type:</b> {doc_type.replace('_', ' ').title()}", normal_style))
    elements.append(Paragraph(f"<b>Risk Score:</b> {risk_score}/100", normal_style))
    elements.append(Paragraph(f"<b>Overall Risk:</b> {overall_risk.upper()}", normal_style))
    elements.append(Spacer(1, 12))

    high_count = sum(1 for c in scored_clauses if c["risk_level"] == "high")
    medium_count = sum(1 for c in scored_clauses if c["risk_level"] == "medium")
    low_count = sum(1 for c in scored_clauses if c["risk_level"] == "low")

    summary_data = [
        ["Risk Level", "Count"],
        ["High Risk", str(high_count)],
        ["Medium Risk", str(medium_count)],
        ["Low Risk", str(low_count)],
        ["Total Clauses", str(len(scored_clauses))]
    ]

    summary_table = Table(summary_data, colWidths=[3*inch, 1.5*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f5f5f5'), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 16))

    red_flags = [c for c in scored_clauses if c["risk_level"] == "high"]
    if red_flags:
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cccccc')))
        elements.append(Paragraph("Red Flags", heading_style))
        for flag in red_flags:
            elements.append(Paragraph(
                f"<b>{flag['clause_name'].replace('_', ' ').title()}</b> — "
                f"{flag['india_note'] if flag['india_note'] else 'Review this clause carefully'}",
                normal_style
            ))
        elements.append(Spacer(1, 12))

    if questions:
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cccccc')))
        elements.append(Paragraph("Questions To Ask Before Signing", heading_style))
        for i, q in enumerate(questions, 1):
            clause_name = q.get("clause_name", "").replace("_", " ").title()
            question = q.get("question", "")
            elements.append(Paragraph(f"<b>{i}. {clause_name}</b>", normal_style))
            elements.append(Paragraph(f"-> {question}", normal_style))
            elements.append(Spacer(1, 4))
        elements.append(Spacer(1, 12))

    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cccccc')))
    elements.append(Paragraph("Clause by Clause Analysis", heading_style))

    clause_data = [["Clause", "Risk Level", "Indian Law Note"]]
    for clause in scored_clauses:
        risk_label = clause["risk_level"].upper()
        clause_data.append([
            clause["clause_name"].replace("_", " ").title(),
            risk_label,
            clause["india_note"][:80] + "..." if len(clause.get("india_note", "")) > 80 else clause.get("india_note", "N/A")
        ])

    clause_table = Table(clause_data, colWidths=[1.8*inch, 1*inch, 3.7*inch])
    clause_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f5f5f5'), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    elements.append(clause_table)
    elements.append(Spacer(1, 16))

    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cccccc')))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(
        "Disclaimer: This report is generated by AI and is for informational purposes only. "
        "It does not constitute legal advice. Please consult a qualified lawyer before making any legal decisions.",
        small_style
    ))
    elements.append(Paragraph("Generated by Legaldocs AI — Legal Document Risk platform for Indian Contracts", small_style))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ---- UI ----

THEME_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&display=swap');
:root{
--bg:#090d12; --surface:#111822; --surface-2:#161f2b;
--border:#1f2935; --border-2:#2b3744;
--text:#e6eaf0; --muted:#98a3b2; --dim:#687486;
--accent:#c3a76d; --accent-dim:#8a7446;
--high:#bd5c53; --med:#b8903f; --low:#5d8a6b;
}
/* --- base --- */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"]{
background:var(--bg) !important; color:var(--text);
}
[data-testid="stDecoration"], [data-testid="stStatusWidget"],
[data-testid="stMainMenu"], [data-testid="stToolbarActions"],
[data-testid="stAppDeployButton"], #MainMenu, footer{
display:none !important; visibility:hidden !important; height:0 !important;
}
/* keep the header shell so the collapsed-sidebar reopen control stays reachable */
[data-testid="stHeader"]{
background:var(--bg) !important; box-shadow:none !important;
}
[data-testid="stToolbar"]{background:transparent !important;}
[data-testid="stExpandSidebarButton"]{
visibility:visible !important; opacity:1 !important;
}
[data-testid="stExpandSidebarButton"] svg{fill:var(--accent) !important;}
[data-testid="stExpandSidebarButton"]:hover{background:rgba(195,167,109,.12) !important;}
.block-container{max-width:1340px !important; padding:2.4rem 2.6rem 3rem !important;}
body, .stApp, p, span, div, label, li, input, button, textarea{
font-family:'Inter','Segoe UI',-apple-system,sans-serif;
}
h1,h2,h3,h4,h5{font-family:'Source Serif 4',Georgia,'Times New Roman',serif !important;
color:var(--text) !important; letter-spacing:-.01em;}
a{color:var(--accent) !important;}
hr{border-color:var(--border) !important;}
::selection{background:rgba(195,167,109,.28);}
/* --- masthead --- */
.masthead{display:flex;align-items:flex-end;justify-content:space-between;gap:2.5rem;
padding-bottom:1.15rem;border-bottom:1px solid var(--border-2);margin-bottom:.35rem;flex-wrap:wrap;}
.mark{display:flex;align-items:center;gap:.95rem;}
.mark-sigil{width:44px;height:44px;flex:none;border:1px solid var(--accent-dim);color:var(--accent);
display:grid;place-items:center;font-family:'Source Serif 4',Georgia,serif;font-size:1.05rem;
font-weight:600;letter-spacing:.06em;background:linear-gradient(160deg,rgba(195,167,109,.09),transparent);}
.mark-name{font-family:'Source Serif 4',Georgia,serif;font-size:1.58rem;font-weight:600;line-height:1.1;}
.mark-sub{font-size:.76rem;color:var(--dim);letter-spacing:.09em;text-transform:uppercase;margin-top:.22rem;}
.mark-meta{text-align:right;}
.mark-meta-v{font-size:.8rem;color:var(--muted);margin-top:.3rem;line-height:1.5;}
/* --- section headers --- */
.eyebrow{font-size:.66rem;letter-spacing:.19em;text-transform:uppercase;color:var(--dim);font-weight:600;}
.sec{display:flex;align-items:center;gap:1.1rem;margin:2.3rem 0 1.05rem;}
.sec-t{font-family:'Source Serif 4',Georgia,serif;font-size:1.2rem;font-weight:600;white-space:nowrap;}
.sec-rule{flex:1;height:1px;background:linear-gradient(90deg,var(--border-2),rgba(43,55,68,.15));}
.sec-n{font-size:.74rem;color:var(--dim);white-space:nowrap;letter-spacing:.04em;}
/* --- panels --- */
.panel{background:var(--surface);border:1px solid var(--border);padding:1.2rem 1.3rem;}
.panel-h{font-size:.66rem;letter-spacing:.17em;text-transform:uppercase;color:var(--dim);
font-weight:600;margin-bottom:.85rem;}
.panel-note{font-size:.78rem;color:var(--dim);line-height:1.6;margin-top:.9rem;
padding-top:.85rem;border-top:1px solid var(--border);}
.panel--score{min-height:214px;display:flex;flex-direction:column;}
.panel--dist{min-height:214px;}
.empty{font-size:.82rem;color:var(--dim);font-style:italic;}
/* --- terms register --- */
.terms-register{padding:0 1.3rem;}
.term-row{display:grid;grid-template-columns:11rem 2.4rem 1fr;gap:1.1rem;align-items:start;
padding:1rem 0;border-bottom:1px solid var(--border);}
.term-row:last-child{border-bottom:none;}
.term-label{font-size:.68rem;letter-spacing:.14em;text-transform:uppercase;color:var(--dim);
font-weight:600;padding-top:.3rem;}
.term-count{font-family:'Source Serif 4',Georgia,serif;font-size:.92rem;color:var(--accent);
font-variant-numeric:tabular-nums;padding-top:.22rem;}
.term-values{display:flex;flex-wrap:wrap;gap:.4rem;}
/* --- stat tiles --- */
.stat{background:var(--surface);border:1px solid var(--border);padding:1rem 1.1rem;min-height:112px;
display:flex;flex-direction:column;justify-content:space-between;}
.stat-l{font-size:.64rem;letter-spacing:.17em;text-transform:uppercase;color:var(--dim);font-weight:600;}
.stat-v{font-family:'Source Serif 4',Georgia,serif;font-size:1.42rem;font-weight:600;line-height:1.2;
margin:.5rem 0 .35rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.stat-v--num{font-size:1.75rem;font-variant-numeric:tabular-nums;}
.stat-s{font-size:.75rem;color:var(--muted);}
/* --- score --- */
.score{display:flex;align-items:baseline;gap:.55rem;flex-wrap:wrap;margin-bottom:1.05rem;}
.score-v{font-family:'Source Serif 4',Georgia,serif;font-size:3.5rem;font-weight:600;line-height:1;
font-variant-numeric:tabular-nums;}
.score-d{font-size:1rem;color:var(--dim);}
.score-p{margin-left:auto;}
.meter{height:6px;background:#1a2330;border:1px solid var(--border);}
.meter>i{display:block;height:100%;}
.meter--high>i,i.meter--high{background:var(--high);}
.meter--medium>i,i.meter--medium{background:var(--med);}
.meter--low>i,i.meter--low{background:var(--low);}
/* --- distribution --- */
.dist{display:grid;grid-template-columns:6.1rem 1fr 1.9rem;align-items:center;gap:.85rem;padding:.52rem 0;}
.dist-l{font-size:.78rem;color:var(--muted);}
.dist-track{height:6px;background:#1a2330;}
.dist-track>i{display:block;height:100%;}
.dist-c{font-size:.86rem;text-align:right;font-variant-numeric:tabular-nums;color:var(--text);}
i.bar--high{background:var(--high);}
i.bar--medium{background:var(--med);}
i.bar--low{background:var(--low);}
/* --- pills --- */
.pill{display:inline-flex;align-items:center;gap:.45rem;font-size:.64rem;font-weight:700;
letter-spacing:.14em;text-transform:uppercase;padding:.32rem .62rem;border:1px solid;white-space:nowrap;}
.pill::before{content:"";width:5px;height:5px;flex:none;}
.pill--high{color:#dc9a93;border-color:#55302c;background:#241715;}
.pill--high::before{background:var(--high);}
.pill--medium{color:#d6b474;border-color:#4e3f24;background:#221c12;}
.pill--medium::before{background:var(--med);}
.pill--low{color:#8fb79c;border-color:#2c4435;background:#131f18;}
.pill--low::before{background:var(--low);}
/* --- flags --- */
.flag{background:var(--surface);border:1px solid var(--border);border-left:2px solid var(--high);
padding:.92rem 1.1rem;margin-bottom:.55rem;}
.flag-n{font-size:.91rem;font-weight:600;letter-spacing:.01em;}
.flag-b{font-size:.82rem;color:var(--muted);line-height:1.6;margin-top:.35rem;}
/* --- questions --- */
.q{display:flex;gap:1.05rem;padding:.82rem 0;border-bottom:1px solid var(--border);}
.q:last-child{border-bottom:none;padding-bottom:0;}
.q-i{font-family:'Source Serif 4',Georgia,serif;color:var(--accent);font-size:.95rem;
min-width:1.5rem;flex:none;padding-top:.08rem;}
.q-n{font-size:.68rem;letter-spacing:.14em;text-transform:uppercase;color:var(--dim);font-weight:600;}
.q-t{font-size:.86rem;color:var(--text);line-height:1.6;margin-top:.3rem;}
/* --- chips --- */
.chips{display:flex;flex-wrap:wrap;gap:.4rem;}
.chip{font-size:.79rem;padding:.3rem .58rem;background:var(--surface-2);border:1px solid var(--border-2);
color:var(--text);font-variant-numeric:tabular-nums;}
.step-b{font-size:.81rem;color:var(--muted);line-height:1.65;}
/* --- clause body --- */
.clause-h{font-size:.64rem;letter-spacing:.17em;text-transform:uppercase;color:var(--dim);
font-weight:600;margin-bottom:.5rem;}
.clause-t{font-size:.86rem;color:var(--text);line-height:1.72;}
.law{margin-top:1.05rem;padding:.85rem 1rem;background:var(--surface-2);
border:1px solid var(--border);border-left:2px solid var(--accent-dim);}
.law-h{font-size:.64rem;letter-spacing:.17em;text-transform:uppercase;color:var(--accent);
font-weight:600;margin-bottom:.4rem;}
.law-b{font-size:.82rem;color:var(--muted);line-height:1.65;}
/* --- sidebar --- */
[data-testid="stSidebar"]{background:#070a0e !important;border-right:1px solid var(--border);}
[data-testid="stSidebar"] .block-container,[data-testid="stSidebarContent"]{padding:1.9rem 1.25rem !important;}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2{font-size:1.12rem !important;}
.rec{padding:.8rem 0;border-bottom:1px solid var(--border);}
.rec-f{font-size:.82rem;font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.rec-m{font-size:.71rem;color:var(--dim);margin-top:.22rem;letter-spacing:.03em;}
.rec-r{display:flex;justify-content:space-between;align-items:center;gap:.6rem;margin-top:.5rem;}
.rec-s{font-family:'Source Serif 4',Georgia,serif;font-size:.92rem;color:var(--muted);
font-variant-numeric:tabular-nums;}
.rec-c{font-size:.71rem;color:var(--dim);margin-top:.35rem;font-variant-numeric:tabular-nums;}
/* --- widgets --- */
[data-testid="stFileUploaderDropzone"],[data-testid="stFileUploadDropzone"],
section[data-testid="stFileUploaderDropzone"]{background:var(--surface) !important;
border:1px dashed var(--border-2) !important;border-radius:0 !important;padding:1.9rem !important;}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--accent-dim) !important;}
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small{color:var(--muted) !important;}
[data-testid="stFileUploaderFile"]{background:var(--surface-2) !important;border:1px solid var(--border) !important;}
.stButton>button,[data-testid="stDownloadButton"]>button,[data-testid="stBaseButton-secondary"],
[data-testid="baseButton-secondary"]{background:transparent !important;color:var(--accent) !important;
border:1px solid var(--accent-dim) !important;border-radius:0 !important;font-size:.7rem !important;
letter-spacing:.15em !important;text-transform:uppercase !important;font-weight:600 !important;
padding:.68rem 1.4rem !important;box-shadow:none !important;transition:background .15s ease;}
.stButton>button:hover,[data-testid="stDownloadButton"]>button:hover{
background:rgba(195,167,109,.1) !important;border-color:var(--accent) !important;color:var(--accent) !important;}
[data-testid="stExpander"]{background:var(--surface) !important;border:1px solid var(--border) !important;
border-radius:0 !important;margin-bottom:.5rem !important;box-shadow:none !important;}
[data-testid="stExpander"] details{background:transparent !important;border:none !important;}
[data-testid="stExpander"] summary{padding:.85rem 1.1rem !important;font-size:.85rem !important;
font-weight:500 !important;color:var(--text) !important;}
[data-testid="stExpander"] summary:hover{color:var(--accent) !important;}
[data-testid="stExpander"] summary p{font-size:.85rem !important;font-weight:500 !important;}
[data-testid="stExpanderDetails"]{padding:0 1.1rem 1.1rem !important;}
[data-testid="stMetric"]{background:var(--surface) !important;border:1px solid var(--border) !important;
border-radius:0 !important;padding:1rem 1.1rem !important;}
[data-testid="stProgressBar"]>div>div>div>div{background:var(--accent) !important;}
.stSpinner>div{border-top-color:var(--accent) !important;}
[data-testid="stSpinner"] p,.stSpinner p{color:var(--muted) !important;font-size:.82rem !important;}
[data-testid="stAlert"]{background:var(--surface) !important;border:1px solid var(--border) !important;
border-radius:0 !important;color:var(--text) !important;}
[data-testid="stCaptionContainer"],.stCaption{color:var(--dim) !important;}
/* --- footer --- */
.foot{margin-top:2.8rem;padding-top:1.15rem;border-top:1px solid var(--border);
display:flex;justify-content:space-between;gap:2.5rem;flex-wrap:wrap;}
.foot-d{font-size:.74rem;color:var(--dim);line-height:1.7;max-width:62ch;}
.foot-b{font-size:.7rem;color:var(--dim);letter-spacing:.13em;text-transform:uppercase;white-space:nowrap;}
/* --- responsive --- */
html, body{overflow-x:hidden;}
[data-testid="stAppViewContainer"], [data-testid="stMain"]{overflow-x:hidden;}
@media (max-width: 900px){
  .block-container{padding:1.6rem 1.2rem 2.4rem !important;}
  [data-testid="stHorizontalBlock"]{flex-wrap:wrap !important;row-gap:.9rem !important;}
  [data-testid="stHorizontalBlock"] [data-testid="stColumn"]{
    min-width:220px !important; flex:1 1 calc(50% - .6rem) !important;}
  .masthead{align-items:flex-start;gap:1rem;}
  .mark-meta{text-align:left;}
  .score-v{font-size:2.8rem;}
  .term-row{grid-template-columns:1fr;gap:.4rem;}
  .term-count{padding-top:0;}
}
@media (max-width: 600px){
  .block-container{padding:1.2rem .9rem 2rem !important;}
  .mark-name{font-size:1.28rem;}
  .mark-sigil{width:38px;height:38px;font-size:.92rem;}
  .mark-sub{font-size:.68rem;}
  [data-testid="stHorizontalBlock"] [data-testid="stColumn"]{
    min-width:100% !important; flex:1 1 100% !important;}
  .sec{gap:.75rem;margin:1.7rem 0 .85rem;flex-wrap:wrap;}
  .sec-t{font-size:1.05rem;}
  .score{gap:.4rem;}
  .score-v{font-size:2.3rem;}
  .score-p{margin-left:0;}
  .panel{padding:1rem 1.05rem;}
  .terms-register{padding:0 1.05rem;}
  .q{flex-direction:column;gap:.3rem;}
  .q-i{min-width:0;}
  .foot{flex-direction:column;gap:1rem;}
}
</style>"""

st.set_page_config(
    page_title="Legaldocs AI — Contract Risk Review",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(THEME_CSS, unsafe_allow_html=True)

RISK_LABEL = {"high": "High Risk", "medium": "Medium Risk", "low": "Low Risk"}
RISK_ORDER = {"high": 0, "medium": 1, "low": 2}


def pill(level):
    return f'<span class="pill pill--{level}">{RISK_LABEL.get(level, "Unclassified")}</span>'


def section(title, note=""):
    note_html = f'<span class="sec-n">{esc(note)}</span>' if note else ""
    st.markdown(
        f'<div class="sec"><span class="sec-t">{esc(title)}</span>'
        f'<span class="sec-rule"></span>{note_html}</div>',
        unsafe_allow_html=True
    )


def stat_tile(label, value, sub="", numeric=False):
    cls = "stat-v stat-v--num" if numeric else "stat-v"
    sub_html = f'<div class="stat-s">{esc(sub)}</div>' if sub else '<div class="stat-s">&nbsp;</div>'
    return (
        f'<div class="stat"><div class="stat-l">{esc(label)}</div>'
        f'<div class="{cls}" title="{esc(value)}">{esc(value)}</div>{sub_html}</div>'
    )


def terms_register(categories):
    rows = ""
    for label, items in categories:
        count = len(items)
        if items:
            values = '<div class="term-values">' + "".join(
                f'<span class="chip">{esc(item)}</span>' for item in items
            ) + '</div>'
        else:
            values = '<div class="term-values"><span class="empty">None detected</span></div>'
        rows += (
            '<div class="term-row">'
            f'<div class="term-label">{esc(label)}</div>'
            f'<div class="term-count">{count:02d}</div>'
            f'{values}</div>'
        )
    return f'<div class="panel terms-register">{rows}</div>'


def dist_row(label, count, total, level):
    width = (count / total * 100) if total else 0
    return (
        f'<div class="dist"><div class="dist-l">{esc(label)}</div>'
        f'<div class="dist-track"><i class="bar--{level}" style="width:{width:.0f}%"></i></div>'
        f'<div class="dist-c">{count}</div></div>'
    )


def titleize(value):
    return str(value).replace("_", " ").title()


init_database()

# ---- SIDEBAR : ANALYSIS REGISTER ----
with st.sidebar:
    st.markdown(
        '<div class="eyebrow">Register</div>'
        '<div style="font-family:\'Source Serif 4\',Georgia,serif;font-size:1.22rem;'
        'font-weight:600;margin:.3rem 0 1.1rem;">Previous Reviews</div>',
        unsafe_allow_html=True
    )

    history = get_history()

    if history:
        records = ""
        for row in history:
            _id, filename, h_doc_type, h_risk_score, h_overall, high, medium, low, timestamp = row
            records += (
                f'<div class="rec"><div class="rec-f" title="{esc(filename)}">{esc(filename)}</div>'
                f'<div class="rec-m">{esc(titleize(h_doc_type))}</div>'
                f'<div class="rec-r">{pill(h_overall)}'
                f'<span class="rec-s">{h_risk_score}/100</span></div>'
                f'<div class="rec-c">{high} high &nbsp;·&nbsp; {medium} medium &nbsp;·&nbsp; {low} low</div>'
                f'<div class="rec-c">{esc(timestamp)}</div></div>'
            )
        st.markdown(records, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="empty">No reviews on record. Submit a document to begin.</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div style="height:1.4rem"></div>', unsafe_allow_html=True)

    if st.button("Clear Register", use_container_width=True):
        conn = sqlite3.connect("lexscan_history.db")
        cursor = conn.cursor()
        cursor.execute("DELETE FROM analyses")
        conn.commit()
        conn.close()
        st.rerun()

# ---- MASTHEAD ----
st.markdown(
    '<div class="masthead">'
    '<div class="mark"><div class="mark-sigil">LD</div>'
    '<div><div class="mark-name">Legaldocs AI</div>'
    '<div class="mark-sub">Contract Risk Review &middot; Indian Jurisdiction</div></div></div>'
    '<div class="mark-meta"><div class="eyebrow">Statutory Baseline</div>'
    '<div class="mark-meta-v">Indian Contract Act, 1872 &middot; RBI Directions<br>'
    'Model Tenancy Act, 2021 &middot; Labour Codes</div></div>'
    '</div>',
    unsafe_allow_html=True
)

reader = load_ocr_model()
nlp = load_spacy_model()

# ---- INTAKE ----
section("Document Intake", "PDF · digital or scanned")

uploaded_file = st.file_uploader(
    "Submit a contract for review",
    type=["pdf"],
    label_visibility="collapsed",
    help="Loan agreements, rent agreements, employment offers, NDAs and insurance documents"
)

if uploaded_file is not None:

    # ---- PROCESSING ----
    with st.spinner("Inspecting document structure"):
        scanned = is_scanned_pdf(uploaded_file)

    uploaded_file.seek(0)

    if scanned:
        extraction_mode = "Optical recognition"
        extraction_note = "Scanned document detected"
        with st.spinner("Reading scanned pages through optical character recognition"):
            raw_text = extract_text_easyocr(uploaded_file, reader)
    else:
        extraction_mode = "Direct extraction"
        extraction_note = "Digital document detected"
        raw_text = extract_text_digital(uploaded_file)

    cleaned_text = clean_text(raw_text)

    with st.spinner("Classifying document type"):
        doc_type = detect_document_type(cleaned_text, api_key)

    entities = extract_entities(cleaned_text, nlp)

    with st.spinner("Isolating and scoring clauses"):
        clauses = extract_clauses(cleaned_text, doc_type, api_key)

    scored_clauses = []
    for clause in clauses:
        risk_level, india_note = score_clause_risk(
            clause["clause_name"],
            clause["clause_text"],
            doc_type
        )
        scored_clauses.append({
            "clause_name": clause["clause_name"],
            "clause_text": clause["clause_text"],
            "risk_level": risk_level,
            "india_note": india_note
        })

    questions = []
    if scored_clauses:
        score, overall_risk = calculate_overall_risk(scored_clauses)
        risk_score = 100 - score

        save_analysis(
            uploaded_file.name,
            doc_type,
            risk_score,
            overall_risk,
            scored_clauses
        )

        with st.spinner("Preparing questions to raise before signing"):
            questions = generate_questions(scored_clauses, doc_type, api_key)

    # ---- DOCUMENT IDENTITY ----
    section("Document", extraction_note)

    id_cols = st.columns(4, gap="medium")
    identity = [
        ("Classification", titleize(doc_type), "Determined from document content", False),
        ("Source File", uploaded_file.name, f"{uploaded_file.size / 1024:,.0f} KB", False),
        ("Extraction", extraction_mode, "Scanned" if scanned else "Digital", False),
        ("Text Volume", f"{len(cleaned_text):,}", "Characters after normalisation", True),
    ]
    for col, (label, value, sub, numeric) in zip(id_cols, identity):
        with col:
            st.markdown(stat_tile(label, value, sub, numeric), unsafe_allow_html=True)

    with st.expander("View normalised source text"):
        st.markdown(
            f'<div class="clause-h">Extracted Text</div>'
            f'<div class="clause-t">{esc(cleaned_text)}</div>',
            unsafe_allow_html=True
        )

    if not scored_clauses:
        section("Review")
        st.markdown(
            '<div class="panel"><div class="panel-h">No Clauses Isolated</div>'
            '<div class="empty">No recognisable clauses could be extracted from this document. '
            'Try a clearer scan or a different contract.</div></div>',
            unsafe_allow_html=True
        )
    else:
        high_clauses = [c for c in scored_clauses if c["risk_level"] == "high"]
        medium_clauses = [c for c in scored_clauses if c["risk_level"] == "medium"]
        low_clauses = [c for c in scored_clauses if c["risk_level"] == "low"]
        total_clauses = len(scored_clauses)

        # ---- RISK ASSESSMENT ----
        section("Risk Assessment", f"{total_clauses} clauses reviewed")

        assess_cols = st.columns([1.05, 1], gap="medium")

        with assess_cols[0]:
            st.markdown(
                '<div class="panel panel--score"><div class="panel-h">Overall Exposure</div>'
                f'<div class="score"><span class="score-v">{risk_score}</span>'
                f'<span class="score-d">/ 100</span>'
                f'<span class="score-p">{pill(overall_risk)}</span></div>'
                f'<div class="meter"><i class="meter--{overall_risk}" '
                f'style="width:{risk_score}%"></i></div>'
                '<div class="panel-note">A higher score indicates greater contractual exposure. '
                'Scoring applies statutory thresholds and keyword tests to each isolated clause.</div>'
                '</div>',
                unsafe_allow_html=True
            )

        with assess_cols[1]:
            st.markdown(
                '<div class="panel panel--dist"><div class="panel-h">Clause Distribution</div>'
                + dist_row("High risk", len(high_clauses), total_clauses, "high")
                + dist_row("Medium risk", len(medium_clauses), total_clauses, "medium")
                + dist_row("Low risk", len(low_clauses), total_clauses, "low")
                + f'<div class="panel-note">{total_clauses} clauses isolated from the document. '
                  f'{len(high_clauses)} require attention before signing.</div>'
                  '</div>',
                unsafe_allow_html=True
            )

        # ---- FINDINGS : FLAGS AND QUESTIONS ----
        section("Findings")

        finding_cols = st.columns(2, gap="medium")

        with finding_cols[0]:
            if high_clauses:
                flags_html = ""
                for flag in high_clauses:
                    note = flag["india_note"] or "Review this clause carefully before signing."
                    flags_html += (
                        f'<div class="flag"><div class="flag-n">{esc(titleize(flag["clause_name"]))}</div>'
                        f'<div class="flag-b">{esc(note)}</div></div>'
                    )
                st.markdown(
                    f'<div class="panel-h">Red Flags &nbsp;·&nbsp; {len(high_clauses)}</div>'
                    + flags_html,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="panel"><div class="panel-h">Red Flags</div>'
                    '<div class="empty">No high-risk clauses were identified in this document.</div>'
                    '</div>',
                    unsafe_allow_html=True
                )

        with finding_cols[1]:
            if questions:
                questions_html = ""
                for index, item in enumerate(questions, 1):
                    clause_name = titleize(item.get("clause_name", ""))
                    question_text = item.get("question", "")
                    questions_html += (
                        f'<div class="q"><div class="q-i">{index:02d}</div>'
                        f'<div><div class="q-n">{esc(clause_name)}</div>'
                        f'<div class="q-t">{esc(question_text)}</div></div></div>'
                    )
                st.markdown(
                    '<div class="panel"><div class="panel-h">Questions To Raise Before Signing</div>'
                    + questions_html + '</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="panel"><div class="panel-h">Questions To Raise Before Signing</div>'
                    '<div class="empty">No questions generated for this document.</div></div>',
                    unsafe_allow_html=True
                )

        # ---- EXTRACTED TERMS ----
        section("Extracted Terms", "Named entities and financial values")

        relevant_money = [
            m for m in entities["money"]
            if "billion" not in m.lower() and "million" not in m.lower()
        ]

        term_rows = [
            ("Monetary Values", relevant_money),
            ("Rates", entities["percentages"]),
            ("Dates & Durations", entities["dates"]),
            ("Parties & Entities", entities["organizations"] + entities["persons"]),
        ]
        st.markdown(terms_register(term_rows), unsafe_allow_html=True)

        # ---- CLAUSE REGISTER ----
        section("Clause Register", "Ordered by severity")

        ordered_clauses = sorted(
            scored_clauses,
            key=lambda c: RISK_ORDER.get(c["risk_level"], 1)
        )

        for clause in ordered_clauses:
            label = (
                f'{titleize(clause["clause_name"])}'
                f'   —   {RISK_LABEL.get(clause["risk_level"], "Unclassified").upper()}'
            )
            with st.expander(label):
                law_html = ""
                if clause["india_note"]:
                    law_html = (
                        '<div class="law"><div class="law-h">Indian Law Reference</div>'
                        f'<div class="law-b">{esc(clause["india_note"])}</div></div>'
                    )
                st.markdown(
                    f'<div class="clause-h">Clause Text</div>'
                    f'<div class="clause-t">{esc(clause["clause_text"])}</div>{law_html}',
                    unsafe_allow_html=True
                )

        # ---- EXPORT ----
        section("Export")

        export_cols = st.columns([1, 1.6], gap="medium")

        with export_cols[0]:
            st.markdown(
                '<div class="panel"><div class="panel-h">Written Report</div>'
                '<div class="step-b">A formatted PDF containing the risk summary, red flags, '
                'questions to raise and the full clause register.</div></div>',
                unsafe_allow_html=True
            )

        with export_cols[1]:
            if st.button("Generate PDF Report"):
                with st.spinner("Composing report"):
                    pdf_buffer = generate_pdf_report(
                        doc_type,
                        risk_score,
                        overall_risk,
                        scored_clauses,
                        questions,
                        entities
                    )
                st.download_button(
                    label="Download Report",
                    data=pdf_buffer,
                    file_name=f"legaldocs_report_{doc_type}.pdf",
                    mime="application/pdf"
                )

# ---- FOOTER ----
st.markdown(
    '<div class="foot">'
    '<div class="foot-d">Legaldocs AI produces automated analysis for informational purposes only. '
    'It does not constitute legal advice and is not a substitute for a qualified advocate. '
    'Consult a legal professional before executing any agreement.</div>'
    '<div class="foot-b">Legaldocs AI</div>'
    '</div>',
    unsafe_allow_html=True
)
