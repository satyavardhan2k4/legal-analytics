import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

import streamlit as st
import json
import re

# Gemini API
import google.generativeai as genai
from dotenv import load_dotenv
import os
load_dotenv()

# Access the GEMINI_API_KEY
api_key = os.getenv('GEMINI_API_KEY')


# Load the pre-processed dataset
df = pd.read_pickle(r'C:\Users\satya\OneDrive\Documents\projects\legal analytics\task1_data[1].pkl')
df.rename(columns={'Facts': 'facts'}, inplace=True)
df.drop(columns=['index'], inplace=True)
df.reset_index(inplace=True)

df_list = df.values.tolist()
result = []
for row in df_list:
    result.append(row[1:])
    mirrored_row = row.copy()
    mirrored_row[4] = row[5]
    mirrored_row[5] = row[4]
    mirrored_row[7] = 1 - mirrored_row[7]
    result.append(mirrored_row[1:])
df2 = pd.DataFrame(result)
df2.rename(columns={0: 'ID', 1: 'name', 2: 'href', 3: 'first_party', 4: 'second_party',
                        5: 'winning_party', 6: 'winner_index', 7: 'facts'}, inplace=True)
df = df2
df.reset_index(inplace=True)

# Vectorization
vectorizer_facts = TfidfVectorizer()

# Train-test split
X_train_facts_text, X_test_facts_text, y_train, y_test = train_test_split(
    df['facts'], df['winner_index'], test_size=0.2, stratify=df['winner_index'])

# Vectorize the facts
X_train_facts = vectorizer_facts.fit_transform(X_train_facts_text)
X_test_facts = vectorizer_facts.transform(X_test_facts_text)

# Train SVM Model
model_svm = LinearSVC(max_iter=1000, C=0.1, loss='squared_hinge', penalty='l2', tol=1e-4)
model_svm.fit(X_train_facts, y_train)

# Train KNN Model
model_knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
model_knn.fit(X_train_facts, y_train)

# Function to predict
def predict(facts):
    facts_vec = vectorizer_facts.transform([facts])
    case_vec = facts_vec.toarray()
    svm_pred = model_svm.predict(case_vec)[0]
    knn_pred = model_knn.predict(case_vec)[0]
    return svm_pred, knn_pred

# Configure Gemini API
genai.configure(api_key=api_key) # Replace with your actual Gemini API key
model_gemini = genai.GenerativeModel('gemini-2.0-flash')

def gget_first_part(facts_text):
    prompt = f"Identify and extract the First Party and the Second Party involved in the following legal case facts. Return the result as a JSON object with keys 'first_party' and 'second_party'. If only one party is clearly mentioned, set the other to null or an empty string. If no parties are clearly identifiable, set both to null or empty strings:\n\n{facts_text}"
    try:
        response = model_gemini.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        error_message = f"Error from Gemini: {str(e)}"
        st.error(error_message)
        return ""

# Function to get summary from Gemini
def get_summary_from_gemini(facts_text):
    prompt = f"Summarize the following legal case facts in 100-200 words:\n\n{facts_text}"
    try:
        response = model_gemini.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return ""

# Function to explain prediction with Gemini
def explain_prediction_with_llm(facts_text, predicted_party):
    prompt = f"The following legal case facts were given:\n\n{facts_text}\n\nThe predicted winning party is: {predicted_party}.\nPlease explain why this party might be more likely to win based on the facts."
    try:
        response = model_gemini.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Error generating explanation: {str(e)}")
        return ""

def safe_json_loads(json_str):
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        st.warning(f"JSONDecodeError (Initial): {e}")
        cleaned_str = json_str.strip()
        cleaned_str = cleaned_str.replace("'", '"')
        cleaned_str = re.sub(r',\s*}', '}', cleaned_str)
        cleaned_str = re.sub(r',\s*]', ']', cleaned_str)
        cleaned_str = cleaned_str.lstrip('\ufeff')
        cleaned_str = cleaned_str.replace('\r', '')

        # Remove the "json" prefix if present
        if cleaned_str.lower().startswith('json'):
            cleaned_str = cleaned_str[4:].lstrip()

        try:
            return json.loads(cleaned_str)
        except json.JSONDecodeError as e2:
            st.error(f"Could not parse JSON after aggressive cleaning: {e2}")
            return None

# Streamlit UI
st.title("🧑‍⚖ Legal Case Prediction App")
st.write("Enter the case details to predict the winning party and view an AI-generated summary and reasoning.")

# Input fields
extracted_party1=st.text_input("First Party")
extracted_party2= st.text_input("Second Party")

facts = st.text_area("Case Facts", "")

show_explanation = st.checkbox("Include AI Explanation", value=True)

if st.button("Predict Outcome"):
    if facts:
        # Predict
        svm_pred, knn_pred = predict(facts)
        svm_winner_index = svm_pred
        knn_winner_index = knn_pred

        # Determine winner names based on extracted parties
        svm_winner = extracted_party1 if svm_winner_index == 0 else extracted_party2
        knn_winner = extracted_party1 if knn_winner_index == 0 else extracted_party2
       

        # Summary
        summary = get_summary_from_gemini(facts)

        # Optional explanation
        explanation = ""
        if show_explanation:
            explanation = explain_prediction_with_llm(facts, svm_winner)

        # Output
        st.subheader("📄 Case Summary:")
        st.write(summary)

        st.subheader("🏆 Prediction:")
        st.write(f"SVM Predicted Winner: *{svm_winner}*")
        # st.write(f"KNN Predicted Winner: *{knn_winner}* (Index: {knn_winner_index})") # Removed KNN prediction

        if show_explanation:
            st.subheader("🧠 Explanation:")
            st.write(explanation)
    else:
        st.warning("Please enter the case facts before predicting.")

# Save models and vectorizer
joblib.dump(model_svm, "svm_model.pkl")
joblib.dump(model_knn, "knn_model.pkl")
joblib.dump(vectorizer_facts, "tfidf_vectorizer.pkl")