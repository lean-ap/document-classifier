# Document Classifier

A Flask-based document classification application that uses a machine learning pipeline to classify documents into categories based on their content. The app downloads datasets, extracts and cleans text from PDFs, trains a logistic regression model using TF-IDF features, and serves classification predictions via a REST API.

---

## Features

- PDF text extraction using PyMuPDF
- Text cleaning with NLTK (stopword removal, punctuation stripping)
- Machine learning pipeline using scikit-learn (TF-IDF + Logistic Regression)
- Dataset loading from Hugging Face datasets library
- REST API built with Flask for document classification
- Poetry-managed Python environment and dependencies

---

## Project Structure
```
.
├── pyproject.toml # Poetry project config
├── poetry.lock # Poetry lock file
├── README.md # Project documentation
├── src
│ └── document_classifier # Source code
│ ├── app.py # Flask app
│ ├── config.py # Config paths
│ ├── extractor.py # ML pipeline
│ ├── utils.py # PDF/text utilities
│ └── models
│ └── model.pkl # Trained ML model (generated)
└── tests
└── test_extractor.py # Unit tests
```

## Prerequisites

- Python 3.12 or higher
- Poetry package manager (https://python-poetry.org/docs/)

---

## Setup & Installation

1. **Clone the repository**

   ```bash
   git clone <repo-url>
   cd document-classifier

2. **Install dependencies**
  ```bash
   poetry install
   
3. **Activate the virtual environment**
  ```bash
   poetry shell

## Usage

1. **Running the Flask API**
  ```bash
   poetry run python src/document_classifier/app.py 

or 
  ```bash
   poetry shell
   python src/document_classifier/app.py


2. **Classify a document** 
Send a POST request to /classify endpoint with a PDF file:
  ```bash
   curl -X POST -F "file=@/path/to/your/document.pdf" http://127.0.0.1:5000/classify

Sample response:
{
  "predicted label": "invoice"
}

## Notes
The project creates a data/pdfs directory to store downloaded PDFs.

The trained model is saved at src/document_classifier/models/model.pkl.

The app cleans and processes text before classification, so PDF quality affects performance.

For production deployment, consider more robust file handling and security.

