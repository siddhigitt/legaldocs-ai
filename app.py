import streamlit as st
import pdfplumber
from dotenv import load_dotenv
import os

# Load API key from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Page title
st.title("LegalDocs AI 🏛️")
st.subheader("Legal Document Risk Analyzer for Indian Contracts")

# File upload button
uploaded_file = st.file_uploader("Upload your legal document", type=["pdf"])

# What happens when a file is uploaded
if uploaded_file is not None:
    st.success("File uploaded successfully!")
    
    # Extract text from PDF
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    
    # Show extracted text
    st.subheader("Extracted Text:")
    st.write(text)