import os
import pickle
from flask import Flask, render_template, request
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained model and tokenizer
with open("bert_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("bert_tokenizer.pkl", "rb") as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Load the TfidfVectorizer (if used in the previous model)
with open("tfidf_vectorizer.pkl", "rb") as tfidf_file:
    tfidf_vectorizer = pickle.load(tfidf_file)

# Helper function to clean the resume text
def clean_resume(text):
    # Remove unwanted characters and extra spaces
    text = re.sub(r'\n+', ' ', text)  # Replace newline characters with spaces
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.strip()

# Prediction function for BERT model
def predict_category_with_bert(resume_text):
    # Tokenize the text
    inputs = tokenizer(resume_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    prediction = outputs.logits.argmax(dim=-1).item()  # Get the predicted category (index)
    return prediction

# Home route (renders the index page)
@app.route('/')
def home():
    return render_template('index.html')

# Resume upload and prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'resume' not in request.files:
        return "No file part"
    file = request.files['resume']
    if file.filename == '':
        return 'No selected file'

    # Read the resume content
    resume_text = file.read().decode('utf-8')
    resume_text = clean_resume(resume_text)  # Clean the resume text

    # Predict category (you can switch to BERT or another model here)
    predicted_category = predict_category_with_bert(resume_text)

    return render_template('result.html', prediction=predicted_category)

if __name__ == "__main__":
    app.run(debug=True)
