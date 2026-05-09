import streamlit as st
import pdfplumber
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
from dotenv import load_dotenv
import os
import re

# Load API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Tell pytesseract where Tesseract is installed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---- FUNCTIONS ----

def clean_text(text):
    # Remove extra whitespace and fix spacing
    text = re.sub(r'\s+', ' ', text)
    # Fix words stuck together by adding space before capitals
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Remove special characters except basic punctuation
    text = re.sub(r'[^\w\s.,;:!?()%-]', '', text)
    return text.strip()

def extract_text_digital(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_scanned(pdf_file):
    text = ""
    images = convert_from_bytes(pdf_file.read())
    for image in images:
        text += pytesseract.image_to_string(image) + "\n"
    return text

def is_scanned_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                return False
    return True

# ---- UI ----

st.title("LexScan AI 🏛️")
st.subheader("Legal Document Risk Analyzer for Indian Contracts")

uploaded_file = st.file_uploader("Upload your legal document", type=["pdf"])

if uploaded_file is not None:
    st.success("File uploaded successfully!")

    # Detect PDF type
    with st.spinner("Detecting PDF type..."):
        scanned = is_scanned_pdf(uploaded_file)

    if scanned:
        st.info("📷 Scanned PDF detected — using OCR")
        uploaded_file.seek(0)
        raw_text = extract_text_scanned(uploaded_file)
    else:
        st.info("📄 Digital PDF detected — extracting text directly")
        uploaded_file.seek(0)
        raw_text = extract_text_digital(uploaded_file)

    # Clean the text
    cleaned_text = clean_text(raw_text)

    # Display
    st.subheader("Extracted Text:")
    st.write(cleaned_text)