import difflib
import io
import json
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pdfplumber
import streamlit as st
from groq import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    Groq,
    RateLimitError,
)
from html import escape as esc
from pdf2image import convert_from_bytes
from pdf2image.exceptions import PDFInfoNotInstalledError, PopplerNotInstalledError

MODEL = "openai/gpt-oss-20b"
RULES_PATH = Path(__file__).with_name("baseline_rules.json")
DOC_TYPES = {"loan_agreement", "rent_agreement", "employment_offer", "nda", "insurance", "credit_card", "other"}


class AnalysisError(Exception):
    """A failure whose message is fit to show the user."""


@lru_cache(maxsize=1)
def _client():
    # The free Groq tier allows 8,000 tokens a minute; extra retries let the SDK wait out a 429.
    return Groq(api_key=os.getenv("GROQ_API_KEY"), max_retries=4)


def _complete(messages, temperature=None, json_mode=False):
    kwargs = {"model": MODEL, "messages": messages}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        response = _client().chat.completions.create(**kwargs)
    except RateLimitError as exc:
        raise AnalysisError(
            "The analysis service has reached its usage limit for the minute. Wait a moment and try again."
        ) from exc
    except AuthenticationError as exc:
        raise AnalysisError("The Groq API key was rejected. Check GROQ_API_KEY in the .env file.") from exc
    except APIConnectionError as exc:
        raise AnalysisError("The analysis service could not be reached. Check the internet connection.") from exc
    except BadRequestError as exc:
        if json_mode:
            raise
        raise AnalysisError("The analysis service rejected the request for this document.") from exc
    except APIStatusError as exc:
        raise AnalysisError("The analysis service returned an error. Please try again shortly.") from exc
    return (response.choices[0].message.content or "").strip()


def _complete_json(messages, temperature, attempts=2):
    for _ in range(attempts):
        try:
            parsed = json.loads(_complete(messages, temperature, json_mode=True))
        except (BadRequestError, json.JSONDecodeError):
            # BadRequestError here is Groq's json_validate_failed: the model broke JSON mode.
            continue
        if isinstance(parsed, dict):
            return parsed
    raise AnalysisError("The document could not be read into clauses. Please try again.")


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'[^\w\s.,;:!?()%\-/+]', '', text)
    return text.strip()

def extract_text_digital(pdf_file):
    # Some PDFs (e.g. ones exported without real space glyphs) place words ~2.9pt apart with
    # no space character at all; pdfplumber's default x_tolerance of 3 merges them into one
    # word. A tighter tolerance still sits well above normal intra-word glyph gaps (<0.3pt).
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text(x_tolerance=1)
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

def detect_document_type(text):
    sample_text = text[:1000]
    answer = _complete([
        {
            "role": "system",
            "content": "You are a legal document classifier for Indian contracts. Reply with ONLY one of these exact words: loan_agreement, rent_agreement, employment_offer, nda, insurance, credit_card, other"
        },
        {
            "role": "user",
            "content": f"What type of legal document is this?\n\n{sample_text}"
        }
    ])
    doc_type = slugify(answer)
    return doc_type if doc_type in DOC_TYPES else "other"

def extract_clauses(text, doc_type):
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

    parsed = _complete_json([
        {
            "role": "system",
            "content": "You are a legal clause extractor. You return ONLY valid JSON. Never include explanations or markdown formatting."
        },
        {
            "role": "user",
            "content": prompt
        }
    ], temperature=0.1)

    return [
        {"clause_name": str(c["clause_name"]), "clause_text": str(c["clause_text"])}
        for c in parsed.get("clauses", [])
        if isinstance(c, dict) and c.get("clause_name") and c.get("clause_text")
    ]

def extract_entities(text, nlp):
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
    with open(RULES_PATH, "r") as f:
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

def generate_questions(scored_clauses, doc_type):
    risky_clauses = [
        c for c in scored_clauses
        if c["risk_level"] in ["high", "medium"]
    ]

    if not risky_clauses:
        return []

    clause_summary = ""
    for clause in risky_clauses:
        clause_summary += f"- {clause['clause_name']}: {clause['clause_text'][:200]}\n"

    parsed = _complete_json([
        {
            "role": "system",
            "content": "You are a legal advisor helping Indians understand contracts. Generate specific, practical questions they should ask before signing. Return ONLY a JSON object with a questions array of objects with clause_name and question fields. No markdown, no explanation."
        },
        {
            "role": "user",
            "content": f"""For this {doc_type.replace('_', ' ')} document, generate one specific question
to ask for each risky clause before signing.

Risky clauses found:
{clause_summary}

Return ONLY this JSON format:
{{"questions": [
    {{"clause_name": "clause name", "question": "specific question to ask"}}
]}}

JSON:"""
        }
    ], temperature=0.3)

    return [
        {"clause_name": str(q.get("clause_name", "")), "question": str(q["question"])}
        for q in parsed.get("questions", [])
        if isinstance(q, dict) and q.get("question")
    ]

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

# ---- VERSION COMPARISON ----

RISK_RANK = {"low": 0, "medium": 1, "high": 2}
NAME_FILLER = {"clause", "the", "of", "and", "a", "an", "for", "on", "to", "in"}
TERM_PATTERN = re.compile(
    r'(?:rs\.?|inr)\s*\d[\d,]*(?:\.\d+)?(?:\s*(?:lakhs?|crores?|lpa))?'
    r'|\d[\d,]*(?:\.\d+)?\s*(?:%|(?:percent|per cent|months?|days?|weeks?|years?|lakhs?|crores?|lpa)\b)'
    r'|\d[\d,]*\.\d+'
    r'|\d{1,3}(?:,\d{2,3})+'
    r'|\d{2,}',
    re.IGNORECASE
)
CLAUSE_ENUMERATOR = re.compile(r'^\s*(?:clause\s+)?\d+(?:\.\d+)*[.)]\s+', re.IGNORECASE)
# Mirrors the text[:3000] slice inside extract_clauses.
CLAUSE_READ_LIMIT = 3000


def slugify(value):
    return re.sub(r'[^a-z0-9]+', '_', str(value).lower()).strip('_')


def term_key(term):
    return re.sub(r'\s+', '', term.lower())


def baseline_rule_key(name, doc_type, rules):
    key = slugify(name)
    if not key:
        return None
    doc_rules = rules.get(doc_type, {})
    if key in doc_rules:
        return key
    for rule_key in doc_rules:
        if rule_key in key or key in rule_key:
            return rule_key
    return None


def word_similarity(text_a, text_b):
    return difflib.SequenceMatcher(
        None, text_a.lower().split(), text_b.lower().split(), autojunk=False
    ).ratio()


def clause_name_similarity(name_a, name_b, doc_type, rules):
    norm_a, norm_b = slugify(name_a), slugify(name_b)
    if norm_a == norm_b:
        return 1.0
    rule_a = baseline_rule_key(name_a, doc_type, rules)
    if rule_a and rule_a == baseline_rule_key(name_b, doc_type, rules):
        return 0.95
    tokens_a = set(norm_a.split("_")) - NAME_FILLER
    tokens_b = set(norm_b.split("_")) - NAME_FILLER
    union = tokens_a | tokens_b
    jaccard = len(tokens_a & tokens_b) / len(union) if union else 0.0
    return max(jaccard, difflib.SequenceMatcher(None, norm_a, norm_b).ratio())


def resolve_rule_key(name, doc_type, rules):
    key = baseline_rule_key(name, doc_type, rules)
    if key:
        return key
    best_key, best_sim = None, 0.7
    for rule_key in rules.get(doc_type, {}):
        similarity = clause_name_similarity(name, rule_key, "", {})
        if similarity >= best_sim:
            best_key, best_sim = rule_key, similarity
    return best_key


def clause_body(text, name):
    # The extractor inconsistently keeps "4. Tenure:" style prefixes; they would skew diffs and
    # the scoring engine, which reads the first number in the text.
    body = CLAUSE_ENUMERATOR.sub("", " ".join(str(text).split()), count=1)
    label, sep, rest = body.partition(":")
    if sep and rest.strip() and len(label) <= 60 and clause_name_similarity(label, name, "", {}) >= 0.6:
        body = rest
    return body.strip()


def prepare_clauses(clauses):
    return [
        {"clause_name": c["clause_name"], "clause_text": clause_body(c["clause_text"], c["clause_name"])}
        for c in clauses
    ]


def score_clause(clause, scoring_name, doc_type):
    risk_level, india_note = score_clause_risk(scoring_name, clause["clause_text"], doc_type)
    return {**clause, "risk_level": risk_level, "india_note": india_note}


def match_clauses(old_clauses, new_clauses, doc_type, rules):
    candidates = []
    for i, old in enumerate(old_clauses):
        for j, new in enumerate(new_clauses):
            name_sim = clause_name_similarity(old["clause_name"], new["clause_name"], doc_type, rules)
            text_sim = word_similarity(old["clause_text"], new["clause_text"])
            if name_sim >= 0.7 or (name_sim >= 0.5 and text_sim >= 0.6) or text_sim >= 0.8:
                candidates.append((0.7 * name_sim + 0.3 * text_sim, i, j))

    candidates.sort(reverse=True)
    used_old, used_new, pairs = set(), set(), []
    for _score, i, j in candidates:
        if i not in used_old and j not in used_new:
            used_old.add(i)
            used_new.add(j)
            pairs.append((i, j))

    removed = [i for i in range(len(old_clauses)) if i not in used_old]
    added = [j for j in range(len(new_clauses)) if j not in used_new]
    return pairs, removed, added


def extract_terms(text):
    seen, terms = set(), []
    for match in TERM_PATTERN.findall(text):
        term = re.sub(r'\s+', ' ', match).strip()
        if term_key(term) not in seen:
            seen.add(term_key(term))
            terms.append(term)
    return terms


def redline_html(old_text, new_text):
    old_words, new_words = old_text.split(), new_text.split()
    matcher = difflib.SequenceMatcher(None, old_words, new_words, autojunk=False)
    parts = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            parts.append(esc(" ".join(old_words[i1:i2])))
        if op in ("delete", "replace"):
            parts.append(f'<del>{esc(" ".join(old_words[i1:i2]))}</del>')
        if op in ("insert", "replace"):
            parts.append(f'<ins>{esc(" ".join(new_words[j1:j2]))}</ins>')
    return " ".join(parts)


def compare_versions(old_clauses, new_clauses, doc_type, rules):
    pairs, removed, added = match_clauses(old_clauses, new_clauses, doc_type, rules)
    rows = []

    for i, j in pairs:
        # Score both sides under one name so extractor naming drift can't fake a risk shift.
        scoring_name = (
            resolve_rule_key(old_clauses[i]["clause_name"], doc_type, rules)
            or resolve_rule_key(new_clauses[j]["clause_name"], doc_type, rules)
            or old_clauses[i]["clause_name"]
        )
        old = score_clause(old_clauses[i], scoring_name, doc_type)
        new = score_clause(new_clauses[j], scoring_name, doc_type)
        old_terms = extract_terms(old["clause_text"])
        new_terms = extract_terms(new["clause_text"])
        old_keys = {term_key(t) for t in old_terms}
        new_keys = {term_key(t) for t in new_terms}
        terms_removed = [t for t in old_terms if term_key(t) not in new_keys]
        terms_added = [t for t in new_terms if term_key(t) not in old_keys]
        similarity = word_similarity(old["clause_text"], new["clause_text"])
        risk_shift = RISK_RANK.get(new["risk_level"], 1) - RISK_RANK.get(old["risk_level"], 1)
        unchanged = (
            risk_shift == 0 and not terms_removed and not terms_added and similarity >= 0.98
        )
        rows.append({
            "status": "unchanged" if unchanged else "modified",
            "name": new["clause_name"],
            "old": old,
            "new": new,
            "similarity": similarity,
            "terms_removed": terms_removed,
            "terms_added": terms_added,
            "risk_shift": risk_shift,
        })

    for status, indices, source, side in (
        ("removed", removed, old_clauses, "old"),
        ("added", added, new_clauses, "new"),
    ):
        for index in indices:
            name = source[index]["clause_name"]
            clause = score_clause(source[index], resolve_rule_key(name, doc_type, rules) or name, doc_type)
            rows.append({
                "status": status,
                "name": clause["clause_name"],
                "old": clause if side == "old" else None,
                "new": clause if side == "new" else None,
                "similarity": 0.0,
                "terms_removed": [],
                "terms_added": [],
                "risk_shift": 0,
            })

    return rows


def change_rank(row):
    if row["status"] == "modified":
        if row["risk_shift"] > 0:
            return 0
        return 3 if row["risk_shift"] < 0 else 4
    return {"added": 1, "removed": 2, "unchanged": 5}[row["status"]]


class _NothingToCache(Exception):
    """Raised inside a cached function so an empty LLM result is retried next time."""


@st.cache_data(show_spinner=False)
def read_document(file_bytes, _get_reader):
    buffer = io.BytesIO(file_bytes)
    scanned = is_scanned_pdf(buffer)
    buffer.seek(0)
    if scanned:
        try:
            raw_text = extract_text_easyocr(buffer, _get_reader())
        except (PDFInfoNotInstalledError, PopplerNotInstalledError) as exc:
            raise AnalysisError(
                "This looks like a scanned PDF. Reading scans needs Poppler, which is not installed "
                "on this computer, so only PDFs with selectable text can be reviewed for now."
            ) from exc
    else:
        raw_text = extract_text_digital(buffer)
    return scanned, clean_text(raw_text)


@st.cache_data(show_spinner=False)
def classify_document(text):
    return detect_document_type(text)


@st.cache_data(show_spinner=False)
def _isolate_clauses(text, doc_type):
    clauses = extract_clauses(text, doc_type)
    if not clauses:
        raise _NothingToCache
    return clauses


def isolate_clauses(text, doc_type):
    try:
        return _isolate_clauses(text, doc_type)
    except _NothingToCache:
        return []


@st.cache_data(show_spinner=False)
def find_questions(scored_clauses, doc_type):
    return generate_questions(scored_clauses, doc_type)


@st.cache_data(show_spinner=False)
def summarize_clause_changes(changes, doc_type):
    listing = "\n\n".join(
        f"Clause: {name}\nEARLIER: {old_text[:700]}\nLATER: {new_text[:700]}"
        for name, old_text, new_text in changes
    )

    parsed = _complete_json([
        {
            "role": "system",
            "content": "You compare two versions of clauses from Indian contracts. For each clause, state in one plain-English sentence what substantively changed for the person signing. Return ONLY a JSON object with a changes array of objects with clause_name and summary fields. No markdown, no explanation."
        },
        {
            "role": "user",
            "content": f"""Document type: {doc_type.replace('_', ' ')}

{listing}

Return ONLY this JSON format:
{{"changes": [
    {{"clause_name": "clause name exactly as given", "summary": "one sentence"}}
]}}

JSON:"""
        }
    ], temperature=0.1)

    return {
        slugify(item.get("clause_name", "")): str(item["summary"])
        for item in parsed.get("changes", [])
        if isinstance(item, dict) and item.get("summary")
    }


def load_rules():
    with open(RULES_PATH, "r") as f:
        return json.load(f)


def score_document(clauses, doc_type, rules):
    return [
        score_clause(clause, resolve_rule_key(clause["clause_name"], doc_type, rules) or clause["clause_name"], doc_type)
        for clause in clauses
    ]


def risk_summary(scored_clauses):
    score, overall_risk = calculate_overall_risk(scored_clauses)
    return 100 - score, overall_risk


def _no_progress(_message):
    pass


def analyze_single(file_bytes, filename, get_reader, nlp, on_step=_no_progress):
    on_step("Reading the document")
    scanned, text = read_document(file_bytes, get_reader)
    on_step("Identifying the document type")
    doc_type = classify_document(text)
    entities = extract_entities(text, nlp)

    on_step("Isolating and scoring clauses")
    clauses = score_document(prepare_clauses(isolate_clauses(text, doc_type)), doc_type, load_rules())

    result = {
        "filename": filename,
        "size": len(file_bytes),
        "scanned": scanned,
        "text": text,
        "doc_type": doc_type,
        "entities": entities,
        "clauses": clauses,
        "questions": [],
        "questions_error": "",
        "risk_score": None,
        "overall_risk": None,
    }
    if clauses:
        result["risk_score"], result["overall_risk"] = risk_summary(clauses)
        on_step("Preparing questions to raise before signing")
        try:
            result["questions"] = find_questions(clauses, doc_type)
        except AnalysisError as exc:
            result["questions_error"] = str(exc)
    return result


def analyze_comparison(earlier, later, get_reader, on_step=_no_progress):
    """earlier and later are (file_bytes, filename) pairs."""
    docs = {}
    for slot, (file_bytes, filename) in (("earlier", earlier), ("later", later)):
        on_step(f"Reading the {slot} version")
        scanned, text = read_document(file_bytes, get_reader)
        on_step(f"Classifying the {slot} version")
        docs[slot] = {
            "name": filename,
            "scanned": scanned,
            "text": text,
            "text_length": len(text),
            "doc_type": classify_document(text),
        }

    # Both versions are read as the earlier version's type so their clause names line up.
    shared_type = docs["earlier"]["doc_type"]
    for slot, doc in docs.items():
        on_step(f"Isolating clauses in the {slot} version")
        doc["clauses"] = prepare_clauses(isolate_clauses(doc.pop("text"), shared_type))

    result = {
        "earlier": docs["earlier"],
        "later": docs["later"],
        "shared_type": shared_type,
        "type_mismatch": docs["later"]["doc_type"] != shared_type,
        "truncated": [slot for slot, doc in docs.items() if doc["text_length"] > CLAUSE_READ_LIMIT],
        "rows": [],
        "summaries": {},
    }
    if not docs["earlier"]["clauses"] or not docs["later"]["clauses"]:
        return result

    rows = sorted(
        compare_versions(docs["earlier"]["clauses"], docs["later"]["clauses"], shared_type, load_rules()),
        key=lambda r: (change_rank(r), -RISK_RANK.get((r["new"] or r["old"])["risk_level"], 1))
    )
    for doc, side in ((docs["earlier"], "old"), (docs["later"], "new")):
        doc["clauses"] = [row[side] for row in rows if row[side]]
        doc["risk_score"], doc["overall_risk"] = risk_summary(doc["clauses"])
    result["rows"] = rows

    reworded = tuple(
        (row["name"], row["old"]["clause_text"], row["new"]["clause_text"])
        for row in rows
        if row["status"] == "modified" and row["similarity"] < 0.85
    )
    if reworded:
        on_step("Summarising substantive rewording")
        try:
            result["summaries"] = summarize_clause_changes(reworded, shared_type)
        except AnalysisError:
            pass
    return result
