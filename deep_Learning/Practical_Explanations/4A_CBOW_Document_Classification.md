# Practical 4A – CBOW for Document Classification

## 1. Objective
The goal is to perform document-level sentiment classification on IMDb movie reviews by training a Continuous Bag of Words (CBOW) Word2Vec model to learn word embeddings, averaging word vectors to create document representations, and classifying them as positive or negative using a Feedforward Neural Network.

## 2. Dataset
- **Name:** IMDb Movie Reviews dataset (`IMDB Dataset.csv`)
- **Columns:** `review` (raw text review) and `sentiment` (target class: `"positive"` or `"negative"`)
- **Target Encoding:** Binary mapping: `positive = 1`, `negative = 0`

## 3. Libraries Used
| Library | Why We Use It |
|---|---|
| `pandas` | For loading tabular review data and mapping sentiment labels. |
| `numpy` | For computing document embedding averages and array transformations. |
| `matplotlib.pyplot` | For plotting training and validation accuracy over training epochs. |
| `re` (Regular Expressions) | For text cleaning (removing non-alphabetic characters and punctuation). |
| `ENGLISH_STOP_WORDS` (sklearn) | Predefined set of common English words (e.g., "is", "the", "and") removed to retain semantic content. |
| `train_test_split` (sklearn) | Partitions documents and labels into 80% train and 20% test sets with stratification. |
| `Word2Vec` (gensim.models) | For training the CBOW word embedding model on preprocessed text tokens. |
| `Sequential`, `Input`, `Dense` (tensorflow.keras) | For building and training the binary classification neural network. |

## 4. Important Functions and Code
- `re.sub(r"[^a-z\s]", " ", text)`: Strips numbers, HTML tags, and punctuation, retaining only lowercase alphabetic characters and spaces.
- `words = [w for w in words if w not in stop_words]`: Filters out frequent, low-information stop words.
- `Word2Vec(sentences=X_train, vector_size=100, window=5, min_count=1, sg=0, ...)`: Trains word embeddings. `sg=0` specifically selects the **Continuous Bag of Words (CBOW)** architecture (predicting target word from surrounding context).
- `document_vector(sentence)`: Computes the document-level embedding by looking up each word's 100-dimensional vector in `cbow_model.wv` and taking the element-wise mean (`np.mean(vectors, axis=0)`). If no recognized words exist, returns a 100-dimensional zero vector.
- `Dense(64, relu) -> Dense(32, relu) -> Dense(1, sigmoid)`: Feedforward classification network that classifies the 100-dimensional document vector as positive or negative sentiment.
- `model.compile(optimizer="adam", loss="binary_crossentropy", ...)`: Standard binary classification setup.

## 5. Important Parameters
- **Architecture (`sg=0`):** `sg=0` denotes Continuous Bag of Words (CBOW); `sg=1` would denote Skip-gram.
- **Embedding Dimension (`vector_size=100`):** Each word is mapped to a 100-dimensional dense continuous vector.
- **Context Window (`window=5`):** Considers up to 5 words before and 5 words after the target word.
- **Minimum Count (`min_count=1`):** Retains all words that appear at least once in the training vocabulary.
- **Classification Epochs (10) & Batch Size (32):** Standard hyperparameters for the classifier.
- **Loss (`binary_crossentropy`):** Appropriate for binary sentiment classification (positive vs negative).

## 6. Model Architecture / Algorithm
- **Stage 1 (CBOW Word Embedding via Gensim):**
  - Surrounding context words -> Continuous Bag of Words projection -> Predict center word.
  - Generates 100-dimensional dense vectors for all vocabulary terms.
- **Stage 2 (Document Vectorization):**
  - Each review is transformed into a single 100-dimensional vector by averaging all constituent word vectors.
- **Stage 3 (Classifier via Keras):**
  - Input(100) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(1, Sigmoid).
- **Data Flow:** `Text Review -> Clean/Tokenize -> CBOW Embeddings -> Average Vector(100) -> Dense(64) -> Dense(32) -> Dense(1) -> Sentiment Score`

## 7. Preprocessing Steps
- **Lowercasing:** Unifies vocabulary so words like "Great" and "great" share the same embedding.
- **Regex Cleaning:** Removes HTML markup, digits, and special characters.
- **Stop Word Removal:** Eliminates semantically empty words ("in", "at", "the") that do not indicate sentiment.
- **Stratified Splitting:** Maintains balanced positive and negative review proportions in train and test sets.

## 8. Expected Output
- Word2Vec training completion log.
- Shapes of vectorized training and test sets `(N, 100)`.
- 10 epochs of classifier training displaying loss and accuracy.
- Test loss and test classification accuracy printed.
- Plot of Training vs. Validation Accuracy over 10 epochs.

## 9. Viva Questions and Answers
**Q1. What is CBOW (Continuous Bag of Words)?**
**Ans:** CBOW is a Word2Vec architecture that predicts a target word given its surrounding context words.

**Q2. How does CBOW differ from Skip-gram?**
**Ans:** CBOW predicts the target word from context words (`Context -> Target`), while Skip-gram predicts the surrounding context words given a target word (`Target -> Context`). CBOW is faster to train and performs well on frequent words.

**Q3. How is a full review converted into a single vector?**
**Ans:** By computing the average (mean) of the 100-dimensional embedding vectors of all valid words present in that review.

**Q4. What does the parameter `sg=0` signify in Gensim?**
**Ans:** It explicitly specifies the CBOW architecture (`sg=0` for CBOW, `sg=1` for Skip-gram).

**Q5. What does `vector_size=100` mean?**
**Ans:** Each word in the vocabulary is represented by a dense continuous vector of 100 real numbers.

**Q6. What does `window=5` mean?**
**Ans:** It considers a symmetric window of 5 words to the left and 5 words to the right of the target word during embedding training.

**Q7. Why is `binary_crossentropy` used as the loss function?**
**Ans:** The task is two-class sentiment classification (positive or negative) with a single output neuron using Sigmoid activation.

**Q8. What happens if a review contains no vocabulary words?**
**Ans:** The function safely returns a 100-dimensional zero vector (`np.zeros(100)`).

**Q9. What is a word embedding?**
**Ans:** A dense vector representation of words where semantically similar or related words are mapped to nearby points in a continuous vector space.

**Q10. Why is stop word removal important before training Word2Vec for classification?**
**Ans:** Stop words like "the", "a", and "is" appear frequently across all documents without carrying sentiment information; removing them helps the model focus on sentiment-bearing words.

## 10. Quick Revision Flow
`Load IMDb CSV -> Map sentiment to 0/1 -> Clean text (lowercase, regex, stop words) -> Train Word2Vec CBOW (sg=0, vector_size=100, window=5) -> Compute average document vectors -> Build FNN (100 -> 64 -> 32 -> 1) -> Compile (binary_crossentropy, Adam) -> Fit (10 epochs) -> Evaluate -> Plot accuracy`
