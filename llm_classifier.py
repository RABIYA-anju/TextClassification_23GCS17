import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def classify_text(text):

    prompt = f"""
    Classify the following text into one category:

    Technology
    Sports
    Politics
    Health
    Entertainment

    Text: {text}

    Return only the category name.
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",   # updated model
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    user_input = input("Enter text: ")
    category = classify_text(user_input)
    print("Predicted Category:", category)
