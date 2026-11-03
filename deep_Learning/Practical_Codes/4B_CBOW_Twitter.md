# Practical 4B – CBOW for Twitter

```python

# Step 1: Import libraries
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from gensim.models import Word2Vec

# Step 2: Load the faculty-provided Twitter dataset
data = pd.read_csv("twitter_dataset.csv")
tweets = data["Text"].dropna().astype(str)


stop_words = set(ENGLISH_STOP_WORDS)

def preprocess_text(text):
    text = text.lower()

    # Remove URLs, mentions and special characters
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)

    # Tokenize into words
    words = text.split()

    # Remove stop words
    words = [word for word in words if word not in stop_words]

    return words

# Convert tweets into lists of words
sentences = tweets.apply(preprocess_text).tolist()

# Remove empty tweets after preprocessing
sentences = [words for words in sentences if len(words) > 0]

# Step 5: Train CBOW Word2Vec model
# sg=0 selects CBOW architecture
cbow_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=0,
    workers=4,
    epochs=10
)

# Step 6: Display vocabulary
print("Vocabulary Size:", len(cbow_model.wv))

# Display word embeddings
word = sentences[0][0]

print("\nSelected Word:", word)
print("Word Embedding:")
print(cbow_model.wv[word])

# Step 7: Find similar words
print("\nMost Similar Words:")
similar_words = cbow_model.wv.most_similar(word, topn=5)

for w, score in similar_words:
    print(w, ":", score)

# Step 8: Evaluate and display results
print("\nCBOW Model Training Completed Successfully!")

# Plot word embedding values for the selected word
plt.plot(cbow_model.wv[word])
plt.title("Word Embedding Values")
plt.xlabel("Embedding Dimensions")
plt.ylabel("Values")
plt.show()

import pandas as pd 
import numpy as np 
import re 
import matplotlib.pyplot as plt 

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS 
from gensim.models import Word2Vec 

data = pd.read_csv("twitter_dataset.csv")
tweets = data["Text"].dropna().astype(str)

stop_words = set(ENGLISH_STOP_WORDS) 

def preprocess_text(text):

    text= text.lower()
    text = re.sub(r"http\S+|www\S+","",text)
    text = re.sub(r"@\w+|#\w+","",text)
    text = re.sub(r"[^a-z\s]"," ",text)

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return words

sentences = tweets.apply(preprocess_text).tolist() 

sentences = [words for words in sentences if len(words) >0]

cbow_model = Word2Vec(
    sentences= sentences, 
    vector_size=100, 
    window=5, 
    min_count=1, 
    sg=0, 
    workers=4, 
    epochs=10
)

print("Vocabulary size: ", len(cbow_model.wv))

word = sentences[0][0]
print("\nselected word: ", word)
print("word embeddings: ")
print(cbow_model.wv[word]) 

print("Most similar words: ")
similar_word = cbow_model.wv.most_similar(word, topn=5)

for w, score in similar_word:
    print(w,": ", score)


print("\nCBOW model training competed successfully ")

plt.plot(cbow_model.wv[word])
plt.title("Word Embedding values")
plt.xlabel("word embedding ")
plt.ylabel("values")
plt.show()


    
```
