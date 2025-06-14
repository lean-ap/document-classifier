# src/document_classifier/extractor.py
# ML pipeline logic – training, loading, classifying.

import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from datasets import load_dataset
from utils import download_pdf, extract_text_from_pdf, clean_text
from config import MODEL_PATH

# Downloads dataset from Hugging Face.
# Downloads PDFs, extracts & cleans text.
# Returns clean samples and labels for training.

def load_and_prepare_data():
    dataset = load_dataset("AyoubChLin/CompanyDocuments", split="train")
    texts, labels = [], []
    # print(dataset[0])

    for item in dataset:
        # url = item["document"]
        # label = item["label"]
        # filename = url.split("/")[-1]
        # filepath = download_pdf(url, filename)
        # raw_text = extract_text_from_pdf(filepath)
        # cleaned_text = clean_text(raw_text)

        # if cleaned_text.strip():
        #     texts.append(cleaned_text)
        #     labels.append(label)
        raw_text = item.get("file_content", "")
        label = item.get("document_type", "Unknown")
        
        #clean and process text
        cleaned_text = clean_text(raw_text)

        if cleaned_text.strip():
            texts.append(cleaned_text)
            labels.append(label)

    return texts, labels
    
# Trains a Logistic Regression classifier using a TF-IDF vectorizer.
# Saves the model to disk with joblib.

def train_model():
    texts, labels = load_and_prepare_data()

    x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)), # Convert text to vector
        ("clf", LogisticRegression(max_iter=200)) # Train classifier
    ])

    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    print(classification_report(y_test, y_pred))

    joblib.dump(pipeline, MODEL_PATH) # Save model

    return pipeline


def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        return train_model()