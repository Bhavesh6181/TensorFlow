# Practical 4C – CBOW for Textual Document

## 1. Objective
The goal is to train a Continuous Bag of Words (CBOW) Word2Vec model on a specific text document (`document3.txt`), learn domain-specific word embeddings, and inspect the semantic vector space around key terms (specifically the term `"influenza"`).

## 2. Dataset
- **Name:** Plain text document (`document3.txt`)
- **Content:** An informational medical text document discussing viral diseases and epidemiology, featuring terms like `"influenza"`.

## 3. Libraries Used
| Library | Why We Use It |
|---|---|
| `re` (Regular Expressions) | For sentence splitting on punctuation boundaries and tokenizing words (including hyphenated medical terms). |
| `numpy` | For numerical operations on vector dimensions. |
| `matplotlib.pyplot` | For plotting the 100-dimensional embedding curve of the target word. |
| `Word2Vec` (gensim.models) | For training the CBOW word embedding model on the tokenized document text. |

## 4. Important Functions and Code
- `with open("document3.txt", "r", encoding="utf-8") as file: ...`: Opens and reads the raw document with UTF-8 encoding.
- `re.split(r'(?<=[.!?])\s+', document)`: Uses a regex lookbehind to split the document into distinct sentences at punctuation marks (`.`, `!`, `?`) followed by whitespace.
- `re.findall(r"[a-z]+(?:-[a-z]+)*", sentence)`: Tokenizes sentences into alphabetic words while preserving hyphenated compound terms (e.g., "virus-like", "air-borne").
- `Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, sg=0, workers=1, epochs=100)`: Trains the CBOW model over **100 epochs**. Higher epochs are used here because the single document is small, so the model needs multiple passes to learn meaningful vector relationships.
- `cbow_model.wv.index_to_key`: Returns the complete list of unique words present in the vocabulary.
- `cbow_model.wv["influenza"]`: Retrieves the 100-dimensional dense embedding for the target keyword `"influenza"`.
- `cbow_model.wv.most_similar("influenza", topn=5)`: Finds the 5 most contextually and semantically related words to `"influenza"` based on cosine similarity.

## 5. Important Parameters
- **Architecture (`sg=0`):** Continuous Bag of Words (CBOW).
- **Embedding Dimension (`vector_size=100`):** 100 dimensions per word vector.
- **Window Size (`window=5`):** 5 context words on either side of the target word.
- **Minimum Count (`min_count=1`):** Keeps all words in vocabulary.
- **Epochs (`epochs=100`):** Set to 100 (unlike 10 in 4A/4B) to ensure sufficient gradient updates on a small single-document corpus.
- **Workers (`workers=1`):** Single thread to ensure deterministic reproducible results on a small text file.

## 6. Model Architecture / Algorithm
- **Input:** Tokenized sentences from `document3.txt`.
- **CBOW Flow:** For every word in each sentence, the model averages the vectors of its neighboring context words within the window of 5, then updates weights to maximize the probability of predicting the center target word.
- **Output:** Dense 100-dimensional vector space tailored to the vocabulary of the document.

## 7. Preprocessing Steps
- **Lowercasing:** Normalizes capitalization.
- **Sentence Splitting:** Segments text into grammatical sentences using punctuation lookbehinds (`[.!?]`). Word2Vec windows should not bridge across unrelated sentences.
- **Regex Word Extraction:** Extracts clean lowercase words and hyphenated compounds while removing extraneous punctuation.
- **Empty Sentence Removal:** Drops empty token lists.

## 8. Expected Output
- Tokenized sentence structure printed.
- Vocabulary size and complete list of vocabulary words printed.
- Selected word (`"influenza"`) confirmation and its 100-dimensional vector printed.
- Top 5 most similar words to `"influenza"` with similarity scores.
- Matplotlib plot displaying the embedding curve across 100 dimensions.
- Confirmation message: `"CBOW Model Training Completed Successfully!"`.

## 9. Viva Questions and Answers
**Q1. What is the source of text data in this practical?**
**Ans:** A local text file named `document3.txt`, containing domain-specific text about viral infections and influenza.

**Q2. Why are 100 epochs used in this practical compared to 10 epochs in 4A and 4B?**
**Ans:** A single document has a very small corpus size. Training for 100 epochs ensures the model iterates over the limited sentences enough times to learn meaningful vector representations.

**Q3. Why split the document into sentences before training Word2Vec?**
**Ans:** Context windows should only consider words within the same sentence; words at the end of one paragraph should not artificially serve as context for the start of another.

**Q4. What does the regex `r"[a-z]+(?:-[a-z]+)*"` match?**
**Ans:** It matches standard alphabetic words as well as hyphenated words (e.g., "illness-related"), preventing hyphens from splitting compound terms.

**Q5. What is the target word inspected in this code?**
**Ans:** The word `"influenza"`.

**Q6. What does `cbow_model.wv.index_to_key` contain?**
**Ans:** An ordered list of all unique vocabulary words learned by the model.

**Q7. What does a cosine similarity score of 0.85 between two words indicate?**
**Ans:** It indicates strong semantic similarity; the two words frequently appear in similar sentence contexts within the document.

**Q8. Why is `workers=1` used here?**
**Ans:** For very small datasets, multi-threading overhead is unnecessary, and a single worker ensures completely deterministic, reproducible word embeddings.

**Q9. What are the dimensions of the vector for `"influenza"`?**
**Ans:** A 1D array of length 100 (`vector_size=100`).

**Q10. How is Word2Vec different from traditional CountVectorizer / TF-IDF?**
**Ans:** TF-IDF represents documents as sparse bags of words with no semantic awareness. Word2Vec produces dense continuous vectors that capture contextual similarity and semantic relationships between words.

## 10. Quick Revision Flow
`Read document3.txt -> Lowercase -> Split into sentences with regex -> Tokenize words (re.findall) -> Train Word2Vec CBOW (sg=0, vector_size=100, epochs=100) -> Print vocab size & words -> Inspect vector for "influenza" -> Find top 5 similar words -> Plot embedding values`
