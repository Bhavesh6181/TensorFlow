# Practical 4A – CBOW for Document Classification

```python

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from gensim.models import Word2Vec

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense

# Step 2: Load the faculty-provided IMDb dataset
data = pd.read_csv("IMDB Dataset.csv")

# Step 3: Separate reviews and sentiment labels
X = data["review"].astype(str)
y = data["sentiment"].str.lower().map({
    "positive": 1,
    "negative": 0
})

# Remove rows with missing or invalid labels
valid_rows = y.notna()
X = X[valid_rows].reset_index(drop=True)
y = y[valid_rows].astype("int32").reset_index(drop=True)

# Step 4: Preprocess the text
# Convert to lowercase, remove special characters and stop words
stop_words = set(ENGLISH_STOP_WORDS)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)

    words = text.split()

    # Remove stop words
    words = [word for word in words if word not in stop_words]

    return words

# Convert reviews into lists of words
sentences = X.apply(preprocess_text).tolist()

# Step 5: Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    sentences, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 6: Create vocabulary and train CBOW Word2Vec model
# Word2Vec creates vocabulary from training sentences
# sg=0 selects CBOW architecture

cbow_model = Word2Vec(
    sentences=X_train,
    vector_size=100,
    window=5,
    min_count=1,
    sg=0,
    workers=4,
    epochs=10
)

# Step 7: Convert documents into average word embeddings
def document_vector(sentence):
    vectors = []

    for word in sentence:
        if word in cbow_model.wv:
            vectors.append(cbow_model.wv[word])

    # Average all word embeddings
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)

    return np.zeros(100)

# Convert training and testing documents into vectors
X_train_vectors = np.array([
    document_vector(sentence) for sentence in X_train
])

X_test_vectors = np.array([
    document_vector(sentence) for sentence in X_test
])

# Step 8: Build feed-forward neural network
model = Sequential([
    Input(shape=(100,)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Step 9: Compile and train the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train_vectors,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1
)

# Step 10: Evaluate the model on test data
loss, accuracy = model.evaluate(X_test_vectors, y_test)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Plot training and validation accuracy
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("CBOW Document Classification Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
```
