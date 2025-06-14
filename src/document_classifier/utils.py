# src/document_classifier/utils.py
# Handles PDF download, text extraction, and cleaning.

import os
import string
import nltk
import fitz #PyMuPDF
import requests
from config import PDF_DIR
from nltk.corpus import stopwords

# ensure stpwords are downloaded
nltk.download('stopwords')


# Downloads a PDF only if it doesn't already exist.
def download_pdf(url, filename):
    path = os.path.join(PDF_DIR, filename)

    if not os.path.exists(path):
        response = requests.get(url)
        with open(path, "wb") as f:
            f.write(response.content)

        return path
    
# Reads a PDF and extracts text using PyMuPDF
def extract_text_from_pdf(filepath):
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# Basic NLP text cleaning — lowercasing, removing punctuation and stopwords.
def clean_text(text):
    stops = set(stopwords.words("english"))

    text = text.lower()
    text = text.translate(str.maketrans("","", string.punctuation))
    tokens = [word for word in text.split() if word not in stops]
    return " ".join(tokens)
