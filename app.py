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

    # Standard NER extraction
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
        # Skip if more than 3 words — real org names are short
        if ent.label_ == "ORG" and len(text_clean.split()) > 3:
            continue
        # Skip common false positive phrases
        false_org_phrases = [
            "internship cum", "personal characteristics", "internship duration",
            "sw development", "swdevelopment", "based ppo", "recruitment drive",
            "school of computer", "the organization", "royal philips"
        ]
        if any(phrase in text_clean.lower() for phrase in false_org_phrases):
            continue
        # Skip billion/million — company stats not contract terms
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

    # Custom Indian currency pattern matcher
    import re
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

# ---- UI ----

st.title("LexScan AI 🏛️")
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

    # Extract entities
    nlp = load_spacy_model()
    entities = extract_entities(cleaned_text, nlp)

    # Display entities
    st.subheader("🔍 Key Information Found:")
    col1, col2 = st.columns(2)

    with col1:
        # Filter out billion/million — not contract terms
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

    with st.spinner("Extracting clauses..."):
        clauses = extract_clauses(cleaned_text, doc_type, api_key)

    #clause display
    if clauses:
        st.subheader("📋 Extracted Clauses:")
        df = pd.DataFrame(clauses)
        df.columns = ["Clause Name", "Clause Text"]
        df["Clause Name"] = df["Clause Name"].str.replace("_", " ").str.title()
        df.index = range(1, len(df) + 1)
        st.table(df)
        st.success(f"Found {len(clauses)} clauses")
    else:
        st.warning("No clauses extracted — try a different document")