# Practical 4D – CBOW for Recommendation System

```python

# Step 1: Import libraries
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# Step 2: Load the faculty-provided Twitter dataset
data = pd.read_csv("twitter_dataset.csv")
tweets = data["Text"].dropna().astype(str)

# Step 3: Preprocess and tokenize tweets
stop_words = set(ENGLISH_STOP_WORDS)

def preprocess_text(text):
    text = text.lower()

    # Remove URLs, mentions and special characters
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)

    # Tokenize and remove stop words
    words = text.split()
    words = [word for word in words if word not in stop_words]

    return words

# Convert tweets into lists of words
sentences = tweets.apply(preprocess_text).tolist()

# Remove empty tweets
sentences = [words for words in sentences if len(words) > 0]

# Step 4: Train CBOW Word2Vec model
cbow_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=0,
    workers=4,
    epochs=10
)

print("CBOW Model Trained Successfully!")

def document_vector(sentence):
    vectors = []

    for word in sentence:
        if word in cbow_model.wv:
            vectors.append(cbow_model.wv[word])

    # Average all word embeddings
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)

    return np.zeros(100)

document_vectors = np.array([
    document_vector(sentence) for sentence in sentences
])

# Step 6: Take user input
user_input = input("Enter your search text: ")

input_words = preprocess_text(user_input)

# Check whether words exist
if not any(word in cbow_model.wv for word in input_words):
    print("No matching words found!")

else:
    input_vector = document_vector(input_words)

    # Step 7: Calculate cosine similarity
    similarity = cosine_similarity(
        input_vector.reshape(1, -1),
        document_vectors
    )[0]

    # Step 8: Recommend top 5 similar tweets
    top_indices = np.argsort(similarity)[::-1][:5]

    print("\nTop 5 Recommended Tweets:")

    for i in top_indices:
        print("\nTweet:", tweets.iloc[i])
        print("Similarity:", round(similarity[i], 4))

    # Step 9: Plot similarity graph
    plt.bar(range(1, len(top_indices) + 1), similarity[top_indices])
    plt.title("Top 5 Recommended Tweets")
    plt.xlabel("Recommended Tweets")
    plt.ylabel("Cosine Similarity")
    plt.show()
```
