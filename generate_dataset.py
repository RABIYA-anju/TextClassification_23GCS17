import random
import pandas as pd

categories = {
    "Technology": [
        "AI", "software", "hardware", "app", "cloud", "data", "machine learning"
    ],
    "Sports": [
        "cricket", "football", "tennis", "basketball", "match", "tournament"
    ],
    "Politics": [
        "government", "election", "policy", "minister", "law", "parliament"
    ],
    "Health": [
        "doctor", "hospital", "medicine", "diet", "exercise", "mental health"
    ],
    "Entertainment": [
        "movie", "music", "actor", "show", "festival", "concert"
    ]
}

data = []

for label, words in categories.items():
    for _ in range(100):
        text = f"New {random.choice(words)} update released"
        data.append([text, label])

df = pd.DataFrame(data, columns=["text", "label"])
df.to_csv("dataset.csv", index=False)

print("Dataset generated successfully!")
