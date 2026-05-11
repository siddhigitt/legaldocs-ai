import streamlit as st
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
api_key = os.getenv("GEMINI_API_KEY")

# ---- LOAD EasyOCR MODEL ----
@st.cache_resource
def load_ocr_model():
    # English only for now — we'll add Hindi later
    reader = easyocr.Reader(['en'], gpu=False)
    return reader

# ---- FUNCTIONS ----

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
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

def extract_text_easyocr(pdf_file, reader):
    text = ""
    images = convert_from_bytes(
        pdf_file.read(),
        poppler_path=r"C:\Program Files\poppler\Library\bin"
    )
    for image in images:
        # Convert PIL image to numpy array — EasyOCR needs numpy
        image_np = np.array(image)
        # Run OCR
        results = reader.readtext(image_np)
        # Extract just the text from results
        page_text = " ".join([result[1] for result in results])
        text += page_text + "\n"
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

# Load OCR model once
with st.spinner("Loading OCR model... (first time takes 1-2 mins)"):
    reader = load_ocr_model()

st.success("OCR model ready!")

uploaded_file = st.file_uploader("Upload your legal document", type=["pdf"])

if uploaded_file is not None:
    st.success("File uploaded successfully!")

    # Detect PDF type
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

    # Clean the text
    cleaned_text = clean_text(raw_text)

    # Display
    st.subheader("Extracted Text:")
    st.write(cleaned_text)