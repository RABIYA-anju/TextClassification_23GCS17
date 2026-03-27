# 📌 Text Classification System using LLM and Traditional NLP

## 📖 Project Description

This project implements a text classification system using both:

* Traditional NLP (TF-IDF + Logistic Regression)
* Large Language Model (LLM) using Groq API

The system compares the performance of both approaches.

---

## 🎯 Problem Statement

Traditional NLP methods rely on feature extraction and machine learning algorithms, whereas LLMs understand context better. This project analyzes their effectiveness.

---

## 🛠 Tools & Technologies

* Python
* Scikit-learn
* Pandas
* Groq API (LLM)
* dotenv

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

```bash
python3 main_app.py
```

---

## 📊 Sample Input

```text
AI is changing the future
```

---

## 📈 Output

```text
Traditional NLP : Technology
LLM Model      : Technology
```

---

## 📁 Project Structure

```
text_classification_terminal/
│
├── main_app.py
├── README.md
├── requirements.txt
├── .gitignore
├── .env (not uploaded)
```

---

## 🔐 Security Note

API keys are stored securely using environment variables and are not exposed in the repository.

---

## ✅ Conclusion

LLM-based classification provides better contextual understanding compared to traditional NLP methods.

---

## 🚀 Future Scope

* Use larger datasets
* Improve accuracy with fine-tuning
* Deploy as web application
