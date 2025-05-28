# 🧠 Legal Analytics

Legal Analytics is an AI-powered tool that predicts the winner of legal cases using Support Vector Machines (SVM). It integrates Google’s Gemini API to summarize case facts and provide insight into why a particular party is likely to win. The application features an intuitive Streamlit interface that makes legal data analysis accessible, insightful, and actionable.

---

## 🚀 Features

- 🔍 **Predict Legal Case Outcomes** — Uses an SVM classifier trained on real court data.
- 🧾 **Summarize Case Facts** — Integrates the Gemini API to generate concise summaries.
- 📊 **Explain Predictions** — Outputs human-readable insights on why the model favors a particular party.
- 🌐 **Streamlit UI** — Clean and interactive interface for end users.

---
## 🧠 How It Works

1. Users paste the **case facts** into a text field.
2. The **Gemini API** summarizes the case and highlights the parties.
3. The **SVM model** predicts which party is likely to win.
4. A natural-language **insight** explains the prediction.

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/legal-analytics.git
cd legal-analytics
pip install -r requirements.txt
streamlit run app.py
