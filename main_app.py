
import os
from groq import Groq
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from dotenv import load_dotenv
load_dotenv()

# ==============================
#  SET YOUR GROQ API KEY
# ==============================
# Replace with your actual key OR use environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)



# ==============================
#  LOAD DATASET
# ==============================
data = {
    "text": [
        "AI is transforming technology",
        "New smartphone released",
        "Football match was exciting",
        "India won the cricket match",
        "Government passed new law",
        "Election results announced",
        "New vaccine developed",
        "Health benefits of yoga",
        "New movie released in theatres",
        "Actor won award"
    ],
    "label": [
        "Technology",
        "Technology",
        "Sports",
        "Sports",
        "Politics",
        "Politics",
        "Health",
        "Health",
        "Entertainment",
        "Entertainment"
    ]
}

df = pd.DataFrame(data)

# ==============================
#  TRADITIONAL NLP MODEL
# ==============================
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

model = LogisticRegression()
model.fit(X,y)
def classify_traditional(text):
    text_vec = vectorizer.transform([text])
    return model.predict(text_vec)[0]

# ==============================
#  LLM MODEL (GROQ API)
# ==============================
def classify_llm(text):
    try:
        prompt = f"""
        Classify the following text into one category:
        Technology, Sports, Politics, Health, Entertainment.

        Text: "{text}"

        Answer only the category.
        """

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error: {str(e)}"

# ==============================
#  MAIN APPLICATION
# ==============================
if __name__ == "__main__":
    print("\n📌 TEXT CLASSIFICATION SYSTEM")
    print("----------------------------------")

    user_input = input("Enter text: ")

    # Traditional NLP
    traditional_result = classify_traditional(user_input)

    # LLM
    llm_result = classify_llm(user_input)

    print("\n🔍 RESULTS")
    print("----------------------------------")
    print(f"Traditional NLP Prediction : {traditional_result}")
    print(f"LLM Prediction             : {llm_result}")

