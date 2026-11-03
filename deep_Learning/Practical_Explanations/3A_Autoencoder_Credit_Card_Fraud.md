# Practical 3A – Autoencoder for Credit Card Fraud Detection

## 1. Objective
The goal is to detect fraudulent credit card transactions by training an Autoencoder on normal transactions only, identifying fraud cases as anomalies with high reconstruction error.

## 2. Dataset
- **Name:** Credit Card Fraud Detection dataset (`creditcard.csv`)
- **Features:** 30 numerical features (Time, Amount, and V1 to V28 derived from PCA transformations to protect privacy)
- **Target:** Column `Class` where `0 = Normal transaction` and `1 = Fraudulent transaction` (highly imbalanced dataset)

## 3. Libraries Used
| Library | Why We Use It |
|---|---|
| `pandas` | For loading tabular CSV transaction records and separating features/labels. |
| `numpy` | For numerical array manipulation and computing mean squared reconstruction errors. |
| `matplotlib.pyplot` | For plotting training and validation loss curves. |
| `tensorflow` | Deep learning framework to construct and train the Autoencoder network. |
| `train_test_split` (sklearn) | For stratified dataset splitting (80% train, 20% test). |
| `MinMaxScaler` (sklearn) | For scaling feature values into the range `[0, 1]` to match sigmoid output activations. |
| `accuracy_score`, `confusion_matrix`, `classification_report` (sklearn) | Evaluation metrics to quantify anomaly detection performance against true labels. |
| `Input`, `Dense`, `Model` | Functional API components to build the encoder-decoder architecture. |

## 4. Important Functions and Code
- `x_train = x_train[y_train == 0]`: Critical filtering step. Autoencoder is trained **exclusively** on normal transactions so it only learns patterns of legitimate behavior.
- `MinMaxScaler()`: Normalizes input features into `[0, 1]`, matching the range of the final Sigmoid layer.
- `Dense(16, activation="relu")` -> `Dense(8, activation="relu")`: **Encoder** compresses the 30-dimensional input into an 8-dimensional bottleneck code.
- `Dense(16, activation="relu")` -> `Dense(n_features, activation="sigmoid")`: **Decoder** reconstructs the original 30 features from the 8-dimensional code.
- `model.compile(optimizer="adam", loss="mse", metrics=["mae"])`: Uses Mean Squared Error (MSE) loss to minimize reconstruction discrepancy.
- `np.mean((x_train - predictions) ** 2, axis=1)`: Calculates the MSE reconstruction error for each sample.
- `threshold = np.mean(train_errors) + np.std(train_errors)`: Sets statistical anomaly cutoff. Test samples with errors above this threshold are classified as fraud (`1`).

## 5. Important Parameters
- **Epochs (20):** Sufficient passes for the network to reconstruct normal transactions accurately.
- **Batch Size (512):** Large batch size suitable for large tabular datasets, providing smooth gradient updates.
- **Bottleneck Dimension (8):** Compressed latent space forcing the model to learn only essential patterns.
- **Optimizer (Adam):** Efficient gradient-based optimizer with adaptive learning rates.
- **Loss (MSE):** Mean Squared Error directly measures reconstruction quality: `(input - output)^2`.
- **Threshold Formula:** `Mean + 1 Standard Deviation` of normal training errors.

## 6. Model Architecture
- **Input:** 30 numerical features.
- **Encoder:**
  - Dense(16, ReLU)
  - Dense(8, ReLU) — Latent Bottleneck Representation
- **Decoder:**
  - Dense(16, ReLU)
  - Dense(30, Sigmoid) — Reconstruction Layer
- **Data Flow:** `30 -> 16 -> 8 (Bottleneck) -> 16 -> 30 (Reconstruction)`

## 7. Preprocessing Steps
- **Stratified Split:** Preserves the rare fraud ratio in both train and test splits.
- **Normal Class Filtering:** Only legitimate transactions (`Class == 0`) are used for training.
- **Min-Max Scaling:** Fits scaler on normal training data and applies it to test data, mapping all features to `[0, 1]`.

## 8. Expected Output
- Initial dataset shape and head inspection.
- 20 epochs of Autoencoder training showing steadily decreasing MSE loss.
- Calculated anomaly threshold printed (e.g., ~0.003 to 0.005).
- Total number of detected fraudulent transactions.
- Accuracy score, Confusion Matrix, and Classification Report (Precision, Recall, F1-score).
- Plot of training vs. validation MSE loss over 20 epochs.

## 9. Viva Questions and Answers
**Q1. What is an Autoencoder?**
**Ans:** An unsupervised neural network designed to compress input data into a low-dimensional latent space (Encoder) and then reconstruct it back to the original input (Decoder).

**Q2. Why is the Autoencoder trained ONLY on normal transactions?**
**Ans:** If the model only learns how normal transactions look, it fails to reconstruct unfamiliar fraudulent transactions accurately, resulting in a high reconstruction error that flags them as anomalies.

**Q3. How is a transaction classified as fraud?**
**Ans:** If its reconstruction error `MSE = mean((input - output)^2)` exceeds the predetermined threshold `(mean + std)`.

**Q4. What is the role of the bottleneck layer?**
**Ans:** It forces the network to learn a compressed representation, preventing it from simply memorizing or copying the identity mapping.

**Q5. Why is MinMaxScaler used instead of StandardScaler?**
**Ans:** The output layer uses a Sigmoid activation which produces values strictly in `[0, 1]`. Min-Max scaling matches this output range.

**Q6. Why is this an unsupervised approach even though labels exist?**
**Ans:** Labels are not used during model training; the model learns to reconstruct inputs `X -> X`. Labels are only used post-training to evaluate detection performance.

**Q7. What is the loss function used to measure reconstruction error?**
**Ans:** Mean Squared Error (MSE).

**Q8. What is the bottleneck dimension in this code?**
**Ans:** 8 neurons (compressed from 30 input features).

**Q9. What batch size is used and why?**
**Ans:** Batch size is 512, which speeds up training on tabular datasets with hundreds of thousands of transactions.

**Q10. What does the threshold `mean + std` represent statistically?**
**Ans:** It establishes an upper confidence boundary; errors above one standard deviation beyond the mean normal error are considered statistically abnormal.

## 10. Quick Revision Flow
`Load creditcard.csv -> Split train/test -> Filter normal (Class == 0) -> Scale with MinMaxScaler -> Build Autoencoder (30 -> 16 -> 8 -> 16 -> 30) -> Fit on normal data -> Compute threshold (mean + std) -> Predict on test -> Flag errors > threshold as fraud -> Evaluate metrics`
