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
        model="llama-3.1-8b-instant",
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
        model="llama-3.1-8b-instant",
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
        "mandatory insurance",
        "compulsory insurance",
        "mandatory purchase",
        "mandatory arbitration",
        "mandatory deduction",
        "non refundable deposit",
        "non-refundable deposit",
        "unlimited liability",
        "no upper cap",
        "no maximum cap",
        "no cap on penalty",
        "no maximum limit",
        "no maximum limit or upper cap",
        "mandatory and bundled",
        "shall be mandatory",
        "insurance shall be mandatory",
        "lender appointed arbitrator",
        "lender shall appoint",
        "appointed exclusively by the bank",
        "appointed exclusively by",
        "sole arbitrator shall be appointed",
        "exclusively by the lender",
        "exclusively by the bank",
        "perpetual confidentiality",
        "perpetual obligation",
        "all intellectual property",
        "assign all ip",
        "including personal projects",
        "outside scope of employment",
        "no prepayment allowed",
        "prepayment not permitted"
    ]

    green_keywords = [
        "insurance is optional",
        "insurance optional",
        "may opt out",
        "can opt out",
        "prepayment without penalty",
        "no prepayment charge",
        "prepayment charge nil",
        "neutral arbitrator",
        "mutually appointed arbitrator",
        "limited to work performed",
        "limited to employment scope",
        "refundable deposit",
        "deposit shall be refunded",
        "waived on request"
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
        model="llama-3.1-8b-instant",
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

# ---- UI ----

st.title("LegalDocs 🏛️")
st.subheader("Legal Document Risk Analyzer for Indian Contracts")

with st.spinner("Loading OCR model... (first time takes 1-2 mins)"):
    reader = load_ocr_model()

st.success("OCR model ready!")

uploaded_file = st.file_uploader("Upload your legal document", type=["pdf"])

if uploaded_file is not None:
    st.success("File uploaded successfully!")

    with st.spinner("Detecting PDF type..."):
        scanned = is_scanned_pdf(uploaded_file)

    if scanned:
        st.info("📷 Scanned PDF detected — using EasyOCR")
        uploaded_file.seek(0)
        with st.spinner("Reading document with EasyOCR..."):
            raw_text = extract_text_easyocr(uploaded_file, reader)
    else:
        st.info("📄 Digital PDF detected — extracting text directly")
        uploaded_file.seek(0)
        raw_text = extract_text_digital(uploaded_file)

    cleaned_text = clean_text(raw_text)

    with st.spinner("Identifying document type..."):
        doc_type = detect_document_type(cleaned_text, api_key)

    st.subheader("Document Analysis:")
    st.info(f"📋 Document Type: **{doc_type.replace('_', ' ').title()}**")

    with st.expander("📄 View Raw Extracted Text"):
        st.write(cleaned_text)

    nlp = load_spacy_model()
    entities = extract_entities(cleaned_text, nlp)

    st.subheader("🔍 Key Information Found:")
    col1, col2 = st.columns(2)

    with col1:
        relevant_money = [m for m in entities["money"]
                         if "billion" not in m.lower()
                         and "million" not in m.lower()]
        if relevant_money:
            st.markdown("**💰 Money Amounts**")
            for m in relevant_money:
                st.write(f"• {m}")

        if entities["percentages"]:
            st.markdown("**📊 Percentages**")
            for p in entities["percentages"]:
                st.write(f"• {p}")

    with col2:
        if entities["dates"]:
            st.markdown("**📅 Dates & Durations**")
            for d in entities["dates"]:
                st.write(f"• {d}")

        if entities["organizations"]:
            st.markdown("**🏢 Organizations**")
            for o in entities["organizations"]:
                st.write(f"• {o}")

    st.divider()

    with st.spinner("Extracting and analyzing clauses..."):
        clauses = extract_clauses(cleaned_text, doc_type, api_key)

    if clauses:
        # Score every clause
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

        # Calculate overall score
        score, overall_risk = calculate_overall_risk(scored_clauses)

        # Display risk score
        st.subheader("⚖️ Overall Risk Assessment:")

        col1, col2, col3 = st.columns(3)

        with col1:
            risk_score = 100 - score
            st.metric("Risk Score", f"{risk_score}/100")

        with col2:
            if overall_risk == "high":
                st.error("🔴 HIGH RISK")
            elif overall_risk == "medium":
                st.warning("🟡 MEDIUM RISK")
            else:
                st.success("🟢 LOW RISK")

        with col3:
            high_count = sum(1 for c in scored_clauses if c["risk_level"] == "high")
            st.metric("🔴 High Risk Clauses", high_count)

        st.progress(risk_score / 100)

        st.divider()

        # Red flags summary
        red_flags = [c for c in scored_clauses if c["risk_level"] == "high"]
        if red_flags:
            st.subheader("⚠️ Red Flags Summary:")
            for flag in red_flags:
                st.error(
                    f"**{flag['clause_name'].replace('_', ' ').title()}** — "
                    f"{flag['india_note'] if flag['india_note'] else 'Review this clause carefully'}"
                )

        st.divider()

        # Generate questions
        with st.spinner("Generating questions to ask..."):
            questions = generate_questions(scored_clauses, doc_type, api_key)

        if questions:
            st.subheader("❓ Questions To Ask Before Signing:")
            for i, q in enumerate(questions, 1):
                clause_name = q.get("clause_name", "").replace("_", " ").title()
                question = q.get("question", "")
                st.markdown(f"**{i}. {clause_name}**")
                st.write(f"→ {question}")

        st.divider()

        # Clause by clause analysis
        st.subheader("📋 Clause Risk Analysis:")
        for clause in scored_clauses:
            if clause["risk_level"] == "high":
                color = "🔴"
                badge = "HIGH RISK"
            elif clause["risk_level"] == "medium":
                color = "🟡"
                badge = "MEDIUM RISK"
            else:
                color = "🟢"
                badge = "LOW RISK"

            with st.expander(f"{color} {clause['clause_name'].replace('_', ' ').title()} — {badge}"):
                st.write(f"**Clause Text:** {clause['clause_text']}")
                if clause["india_note"]:
                    st.info(f"📜 **Indian Law:** {clause['india_note']}")

        st.success(f"Analyzed {len(scored_clauses)} clauses")

    else:
        st.warning("No clauses extracted — try a different document")