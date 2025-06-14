# src/document_classifier/app.py
# Flask web server to handle document classification requests.

from flask import Flask, request, jsonify
from extractor import load_model
from utils import extract_text_from_pdf, clean_text
import os

# Starts the Flask app and loads the trained model into memory
app = Flask(__name__)
model = load_model()

@app.route("/classify", methods=["POST"])
def classify():
    if "file" not in request.files:
        return jsonify({"error": "no file uoloaded"}), 400
    
    file = request.files["file"]
    filepath = f"temp_{file.filename}"
    file.save(filepath)

    text = extract_text_from_pdf(filepath)
    cleaned = clean_text(text)
    prediction = model.predict([cleaned])[0]

    os.remove(filepath)
    return jsonify({"predicted label": prediction})

if __name__ == "__main__":
    app.run(debug=True)

