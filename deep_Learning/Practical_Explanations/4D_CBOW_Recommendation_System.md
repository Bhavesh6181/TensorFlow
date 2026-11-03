# Practical 4D – CBOW for Recommendation System

## 1. Objective
The goal is to build a content-based text recommendation system by training a CBOW Word2Vec model on Twitter data, vectorizing tweets, taking a user search query, and recommending the top 5 most semantically similar tweets using Cosine Similarity.

## 2. Dataset
- **Name:** Twitter Dataset (`twitter_dataset.csv`)
- **Key Column:** `Text` containing raw user tweets that form the search corpus

## 3. Libraries Used
| Library | Why We Use It |
|---|---|
| `pandas` | For loading the CSV dataset and accessing the `Text` column. |
| `numpy` | For computing average document vectors and ranking similarity scores (`np.argsort`). |
| `re` (Regular Expressions) | For cleaning URLs, user handles, hashtags, and non-alphabetic characters. |
| `matplotlib.pyplot` | For plotting a bar chart of cosine similarity scores for top recommendations. |
| `ENGLISH_STOP_WORDS` (sklearn) | For removing non-informative stop words. |
| `Word2Vec` (gensim.models) | For learning word embeddings using the CBOW architecture. |
| `cosine_similarity` (sklearn.metrics.pairwise) | For calculating angular similarity between the user query vector and all tweet vectors. |

## 4. Important Functions and Code
- `preprocess_text(text)`: Lowercases text, strips URLs (`http\S+`), handles (`@\w+`), hashtags (`#\w+`), punctuation, and stop words.
- `Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0, workers=4, epochs=10)`: Trains CBOW embeddings (`sg=0`) on preprocessed tweets.
- `document_vector(sentence)`: Computes the 100-dimensional representation of a tweet by taking the element-wise average of the embeddings of all its constituent words.
- `user_input = input("Enter your search text: ")`: Accepts an interactive search query from the user.
- `input_vector = document_vector(input_words)`: Transforms the user's query into the same 100-dimensional embedding space.
- `cosine_similarity(input_vector.reshape(1, -1), document_vectors)[0]`: Measures the cosine of the angle between the query vector and every tweet vector in the dataset.
- `np.argsort(similarity)[::-1][:5]`: Sorts similarity scores in descending order and extracts the indices of the top 5 highest matching tweets.
- `plt.bar(...)`: Generates a bar graph visualizing similarity scores of the top 5 recommendations.

## 5. Important Parameters
- **Architecture (`sg=0`):** Continuous Bag of Words (CBOW).
- **Embedding Dimension (`vector_size=100`):** 100-dimensional semantic space.
- **Context Window (`window=5`):** 5 context words on each side of the target word.
- **Top-N Recommendations (5):** The system returns the top 5 most relevant tweets.
- **Similarity Metric:** Cosine similarity, ranging from -1 to 1 (with higher values indicating greater relevance).

## 6. Model Architecture / Algorithm
- **Step 1:** Clean and tokenize all corpus tweets.
- **Step 2:** Train CBOW Word2Vec model on the tokenized tweets.
- **Step 3:** Convert every tweet in the corpus into an average 100-dimensional vector.
- **Step 4:** User enters a search query -> Query is cleaned and converted to a 100-dimensional vector.
- **Step 5:** Compute Cosine Similarity between the query vector and all tweet vectors.
- **Step 6:** Rank tweets by similarity and display the top 5 matches with a bar chart.

## 7. Preprocessing Steps
- **Cleaning:** Strips URLs, user mentions (`@`), hashtags (`#`), and special characters using regular expressions.
- **Stop Word Removal:** Filters out generic words to ensure recommendations match substantive search terms.
- **Query Preprocessing:** The user's input query undergoes the exact same cleaning pipeline as the dataset tweets to ensure compatibility.
- **Vocabulary Check:** Checks whether any query words exist in the learned vocabulary before computing similarity.

## 8. Expected Output
- Confirmation: `"CBOW Model Trained Successfully!"`.
- Interactive prompt: `Enter your search text: `.
- Printed text of the top 5 recommended tweets along with their cosine similarity scores (e.g., `Similarity: 0.8412`).
- Matplotlib bar chart showing cosine similarity values for the top 5 recommended tweets.

## 9. Viva Questions and Answers
**Q1. How does this recommendation system work?**
**Ans:** It is a content-based recommendation system. Both the tweets and the user's search query are converted into 100-dimensional semantic vectors using CBOW embeddings. Tweets with the highest cosine similarity to the query vector are recommended.

**Q2. What is Cosine Similarity?**
**Ans:** It is a metric that measures the cosine of the angle between two vectors in a multi-dimensional space:
$$	ext{Cosine Similarity} = rac{A \cdot B}{\|A\| \|B\|}$$
It measures directional alignment rather than vector magnitude.

**Q3. Why is Cosine Similarity preferred over Euclidean distance for text vectors?**
**Ans:** Cosine similarity evaluates the angle between vectors, making it independent of document length. Euclidean distance is biased by document length (longer documents have larger vectors even if they share the same topic).

**Q4. What happens if none of the words in the user query exist in the vocabulary?**
**Ans:** The system detects this condition (`if not any(...)`) and prints `"No matching words found!"` without attempting invalid calculations.

**Q5. How is a tweet vector calculated from word vectors?**
**Ans:** By averaging all individual 100-dimensional word vectors contained in that tweet (`np.mean(vectors, axis=0)`).

**Q6. What does `np.argsort(similarity)[::-1][:5]` do?**
**Ans:** `np.argsort` sorts indices in ascending order; `[::-1]` reverses it to descending order (highest similarity first); and `[:5]` slices the top 5 index positions.

**Q7. What is the difference between collaborative filtering and content-based recommendation?**
**Ans:** Collaborative filtering recommends items based on other users' past behaviors/ratings. Content-based recommendation (used here) recommends items based on the similarity between item text features and the user's query.

**Q8. Why use CBOW embeddings instead of keyword matching?**
**Ans:** Keyword matching requires exact word matches. CBOW embeddings capture semantic meaning, allowing the system to recommend relevant tweets even if they use synonyms or related terms not explicitly in the query.

**Q9. What are the dimensions of the document vectors matrix?**
**Ans:** `(N, 100)`, where $N$ is the total number of valid tweets and 100 is the embedding dimension.

**Q10. What does the final Matplotlib bar plot show?**
**Ans:** It displays the cosine similarity scores of the top 5 recommended tweets as vertical bars, providing visual comparison of match quality.

## 10. Quick Revision Flow
`Load tweets -> Preprocess text (regex, stop words) -> Train Word2Vec CBOW (sg=0, vector_size=100) -> Compute average vectors for all tweets -> Take user query input -> Preprocess query & compute query vector -> Calculate cosine_similarity against all tweet vectors -> Get top 5 indices using np.argsort -> Print top 5 tweets and plot bar chart`
