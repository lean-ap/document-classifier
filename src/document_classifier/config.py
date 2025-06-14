# src/document_classifier/config.py
# Centralized configuration for file paths and folders.

import os

# Get absolute path of the current file (config.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Set the folder to store downloaded PDFs
PDF_DIR = os.path.join(BASE_DIR, "../../data/pdfs")

# Set the path where the trained model will be saved/loaded
MODEL_PATH = os.path.join(BASE_DIR, "models/model.pkl")

# Make sure these directories exist at runtime
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)