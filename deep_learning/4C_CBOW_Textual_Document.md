# Practical 4C – CBOW for Textual Document

```python

# Step 1: Import libraries
import re
import numpy as np
import matplotlib.pyplot as plt

from gensim.models import Word2Vec

# Step 2: Load the given Document 3
with open("document3.txt", "r", encoding="utf-8") as file:
    document = file.read()

# Step 3: Preprocess the document
# Convert text into lowercase
document = document.lower()

# Split document into sentences
sentences = re.split(r'(?<=[.!?])\s+', document)

# Tokenize sentences into words
def preprocess_text(sentence):
    words = re.findall(r"[a-z]+(?:-[a-z]+)*", sentence)
    return words

# Convert sentences into lists of words
tokenized_sentences = [preprocess_text(sentence) for sentence in sentences]

# Remove empty sentences
tokenized_sentences = [words for words in tokenized_sentences if len(words) > 0]

print("Tokenized Document:")
print(tokenized_sentences)

# Step 4: Train Word2Vec CBOW model
# sg=0 selects CBOW architecture
cbow_model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=0,
    workers=1,
    epochs=100
)

# Step 5: Display vocabulary
print("\nVocabulary Size:", len(cbow_model.wv))

print("\nVocabulary Words:")
print(list(cbow_model.wv.index_to_key))

# Step 6: Display word embeddings
word = "influenza"

print("\nSelected Word:", word)
print("Word Embedding:")
print(cbow_model.wv[word])

# Step 7: Find similar words
print("\nMost Similar Words:")
similar_words = cbow_model.wv.most_similar(word, topn=5)

for w, score in similar_words:
    print(w, ":", score)

# Step 8: Plot word embedding values
plt.plot(cbow_model.wv[word])
plt.title("Word Embedding Values")
plt.xlabel("Embedding Dimensions")
plt.ylabel("Values")
plt.show()

print("\nCBOW Model Training Completed Successfully!")
```
