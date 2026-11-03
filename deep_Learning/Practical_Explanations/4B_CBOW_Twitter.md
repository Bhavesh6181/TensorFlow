# Practical 4B – CBOW for Twitter

## 1. Objective
The goal is to train a Continuous Bag of Words (CBOW) Word2Vec model on unstructured Twitter text data, learn continuous dense word embeddings, and explore semantic relationships and word similarities.

## 2. Dataset
- **Name:** Twitter Dataset (`twitter_dataset.csv`)
- **Key Column:** `Text` containing raw, noisy social media tweets (with user mentions, hashtags, URLs, and abbreviations)

## 3. Libraries Used
| Library | Why We Use It |
|---|---|
| `pandas` | For reading the CSV tweet dataset and extracting the `Text` column. |
| `numpy` | For numerical array operations. |
| `re` (Regular Expressions) | For cleaning Twitter-specific text artifacts (URLs, @user mentions, #hashtags, punctuation). |
| `matplotlib.pyplot` | For visualizing embedding vector values across latent dimensions. |
| `ENGLISH_STOP_WORDS` (sklearn) | Standard stop word collection to filter out non-informative words. |
| `Word2Vec` (gensim.models) | For training the CBOW word embedding model. |

## 4. Important Functions and Code
- `re.sub(r"http\S+|www\S+", "", text)`: Removes web hyperlinks and URLs from tweet text.
- `re.sub(r"@\w+|#\w+", "", text)`: Strips Twitter mentions (`@username`) and hashtags (`#topic`), removing social media noise.
- `re.sub(r"[^a-z\s]", " ", text)`: Retains only lowercase letters and spaces.
- `sentences = [words for words in sentences if len(words) > 0]`: Drops tweets that become completely empty after text cleaning.
- `Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0, workers=4, epochs=10)`: Trains CBOW embeddings (`sg=0`) with 100 dimensions over 10 epochs.
- `cbow_model.wv[word]`: Retrieves the learned 100-dimensional numerical embedding vector for a given word.
- `cbow_model.wv.most_similar(word, topn=5)`: Computes cosine similarity between the target word's vector and all other words in vocabulary, returning the 5 most semantically similar words and their similarity scores.
- `plt.plot(cbow_model.wv[word])`: Visualizes the 100 dimensional values for an extracted word.

## 5. Important Parameters
- **Architecture (`sg=0`):** Continuous Bag of Words (CBOW) model.
- **Vector Dimension (`vector_size=100`):** Length of dense vector representing each word.
- **Context Window (`window=5`):** 5 words before and after target word.
- **Minimum Count (`min_count=1`):** Every word appearing at least once is included in the vocabulary.
- **Workers (`workers=4`):** Uses 4 CPU threads for parallel processing.
- **Epochs (`epochs=10`):** Number of passes over the tokenized tweet corpus.

## 6. Model Architecture / Algorithm
- **Input:** Tokenized words from preprocessed tweets.
- **CBOW Principle:** Context words within the window size of 5 act as inputs to predict the middle target word.
- **Network Structure:** Input projection layer (averaging context word embeddings) -> Output Softmax / Hierarchical Softmax predicting the target word.
- **Output:** Learned word vectors stored in `cbow_model.wv`.

## 7. Preprocessing Steps
- **Lowercasing:** Normalizes letter casing across tweets.
- **URL Removal:** Eliminates web addresses (`http://...`, `www...`).
- **Handle & Hashtag Removal:** Strips `@username` handles and `#hashtags`.
- **Special Character Removal:** Strips emojis, symbols, and punctuation.
- **Stop Word Removal:** Filters out words like "the", "is", and "at".
- **Empty Sentence Filtering:** Drops any tweets left with zero words after preprocessing.

## 8. Expected Output
- Total vocabulary size printed (number of unique learned words).
- Selected word name and its full 100-dimensional floating point vector.
- Top 5 most similar words along with their cosine similarity scores (values between 0.0 and 1.0).
- Matplotlib line plot showing the 100-dimensional values of the selected word vector.

## 9. Viva Questions and Answers
**Q1. What makes Twitter data preprocessing different from standard text data?**
**Ans:** Twitter data contains social media noise like URLs (`http://...`), user handles (`@user`), hashtags (`#topic`), emojis, and slang that must be cleaned using regex.

**Q2. What is the difference between CBOW and one-hot encoding?**
**Ans:** One-hot vectors are high-dimensional, sparse, and contain zero semantic information. CBOW vectors are low-dimensional, dense, and capture semantic relationships where similar words have similar vectors.

**Q3. How does `most_similar()` determine similar words?**
**Ans:** It calculates the cosine similarity between the query word's vector and all other vectors in the vocabulary, ranking words by highest cosine score.

**Q4. What is the range of cosine similarity?**
**Ans:** Cosine similarity ranges from -1 to +1, where +1 indicates identical direction (highest semantic similarity) and 0 indicates orthogonality (no relationship).

**Q5. Why is `min_count=1` used here?**
**Ans:** It ensures that every single unique word found in the dataset is included in the vocabulary without discarding rare terms.

**Q6. What does `workers=4` specify in Gensim?**
**Ans:** It allows multi-core parallel processing using 4 CPU worker threads to speed up model training.

**Q7. Is there a neural network classifier trained in this practical?**
**Ans:** No. This practical focuses strictly on unsupervised representation learning — building and evaluating the Word2Vec CBOW embedding space itself.

**Q8. What does `cbow_model.wv` store?**
**Ans:** It is the `KeyedVectors` object in Gensim that stores the vocabulary mapping and learned embedding vectors for all words.

**Q9. What does the Matplotlib line plot display?**
**Ans:** It plots the 100 continuous numerical weights of the learned vector for a selected word across dimensions 0 to 99.

**Q10. Can Word2Vec handle out-of-vocabulary (OOV) words at test time?**
**Ans:** No. Standard Word2Vec cannot generate embeddings for words not seen during training; attempting to access an unseen word raises a `KeyError`.

## 10. Quick Revision Flow
`Load twitter_dataset.csv -> Extract Text column -> Clean (remove URLs, @mentions, #hashtags, punctuation, stop words) -> Tokenize tweets -> Train Word2Vec CBOW (sg=0, vector_size=100, window=5) -> Print vocabulary size -> Retrieve word vector -> Find top 5 similar words via most_similar() -> Plot embedding vector`
