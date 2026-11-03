# Practical 3B – Autoencoder for MVTec AD

> ⚠️ **Syllabus–Code Mismatch:** The syllabus title refers to "MVTec AD" (an industrial image anomaly detection benchmark). However, the faculty-provided notebook code implements **ECG (Electrocardiogram) Anomaly Detection** using `ecg.csv`. The explanation below documents the actual code implementation.

## 1. Objective
The goal is to detect abnormal heartbeats (cardiac anomalies) from Electrocardiogram (ECG) time-series signals using an Autoencoder trained exclusively on normal ECG waveforms.

## 2. Dataset
- **Name:** ECG5000 / ECG Dataset (`ecg.csv`, loaded without header)
- **Features:** 140 continuous numerical readings representing an ECG heartbeat waveform
- **Target:** The final column contains class labels where `1 = Normal heartbeat` and other values (`0` or `2-5`) represent abnormal heartbeats (arrhythmias)

## 3. Libraries Used
| Library | Why We Use It |
|---|---|
| `pandas` | For reading headerless CSV heartbeat records into a DataFrame. |
| `numpy` | For array manipulations and calculating mean squared reconstruction errors. |
| `matplotlib.pyplot` | For plotting training and validation loss curves across epochs. |
| `train_test_split` (sklearn) | To partition data into 80% training and 20% testing with stratification. |
| `MinMaxScaler` (sklearn) | To normalize ECG signal amplitudes into `[0, 1]` to match sigmoid output activations. |
| `accuracy_score`, `confusion_matrix`, `classification_report` (sklearn) | For evaluating anomaly classification metrics against true labels. |
| `layers`, `models` (tensorflow.keras) | For defining the functional Autoencoder architecture. |

## 4. Important Functions and Code
- `df = pd.read_csv("ecg.csv", header=None)`: Reads the dataset where each row is an unlabelled sequence of 140 signal points followed by the class label.
- `x_train = x_train[y_train == 1]`: Filters the training set to retain **only normal ECG beats** (`label == 1`).
- `MinMaxScaler()`: Rescales ECG signal amplitudes to `[0, 1]`.
- `layers.Dense(32, activation="relu")` -> `layers.Dense(16, activation="relu")`: **Encoder** stages that compress 140 signal points into a 16-dimensional bottleneck.
- `layers.Dense(32, activation="relu")` -> `layers.Dense(n_features, activation="sigmoid")`: **Decoder** stages that reconstruct the 140 ECG signal points.
- `threshold = np.mean(train_errors) + np.std(train_errors)`: Threshold separating normal from abnormal heartbeats.
- `y_actual = (y_test != 1).astype(int)`: Converts ground truth so that `1 = Abnormal` (anomaly) and `0 = Normal`, matching the anomaly detection convention.

## 5. Important Parameters
- **Epochs (20):** Training cycles to learn the normal heartbeat waveform.
- **Batch Size (512):** Large mini-batch size for smooth optimization.
- **Bottleneck Size (16):** 16 latent features capturing the core morphological properties of a normal heartbeat.
- **Optimizer (Adam):** Adaptive gradient descent optimizer.
- **Loss (MSE):** Mean Squared Error to measure waveform reconstruction fidelity.

## 6. Model Architecture
- **Input:** 140 numerical ECG signal values.
- **Encoder:**
  - Dense(32, ReLU)
  - Dense(16, ReLU) — Bottleneck representation
- **Decoder:**
  - Dense(32, ReLU)
  - Dense(140, Sigmoid) — Reconstructed ECG signal
- **Data Flow:** `140 -> 32 -> 16 (Bottleneck) -> 32 -> 140 (Output)`

## 7. Preprocessing Steps
- **Feature/Label Separation:** `X = df.iloc[:, :-1]` (features), `y = df.iloc[:, -1]` (labels).
- **Stratified Split:** Preserves the proportion of normal and abnormal heartbeats in train and test sets.
- **Normal Signal Filtering:** Only healthy heartbeats (`y == 1`) are retained for training.
- **Min-Max Scaling:** Scales ECG amplitudes into `[0, 1]` to align with the sigmoid reconstruction layer.

## 8. Expected Output
- Dataset dimensions printed (`shape: (N, 141)`).
- 20 epochs showing decreasing reconstruction loss.
- Anomaly threshold printed.
- Total detected cardiac anomalies.
- Accuracy score, Confusion Matrix, and Classification Report.
- Training vs. Validation Loss plot.

## 9. Viva Questions and Answers
**Q1. What is the discrepancy between the syllabus and the practical code for 3B?**
**Ans:** The syllabus specifies "MVTec AD" (an image anomaly dataset), but the actual code implements ECG anomaly detection using time-series heartbeats from `ecg.csv`.

**Q2. Why is an Autoencoder suitable for ECG anomaly detection?**
**Ans:** Normal heartbeats follow consistent periodic shapes (P-Q-R-S-T waves). An Autoencoder trained only on normal beats reconstructs them accurately, but fails to reconstruct abnormal arrhythmias, producing a high error that flags anomalies.

**Q3. What does each row in `ecg.csv` represent?**
**Ans:** A single recorded heartbeat consisting of 140 sequential signal voltage readings and a class label.

**Q4. Why do we filter `x_train[y_train == 1]`?**
**Ans:** In this dataset, class 1 represents normal healthy heartbeats. The autoencoder must only learn normal cardiac behavior.

**Q5. Why is `y_actual = (y_test != 1).astype(int)` used during evaluation?**
**Ans:** To invert labels so that anomalous/abnormal beats become class 1 and normal beats become class 0, aligning with standard anomaly detection evaluation.

**Q6. What is the bottleneck dimension?**
**Ans:** 16 neurons.

**Q7. How is the decision threshold computed?**
**Ans:** `Threshold = Mean(train reconstruction errors) + Standard Deviation(train reconstruction errors)`.

**Q8. What activation is used in the final layer and why?**
**Ans:** Sigmoid, because it outputs values in `[0, 1]`, matching the Min-Max scaled input signal range.

**Q9. What loss function guides the autoencoder training?**
**Ans:** Mean Squared Error (MSE), measuring the point-by-point difference between the input and reconstructed ECG waveforms.

**Q10. What metrics evaluate the final performance?**
**Ans:** Accuracy, Confusion Matrix, Precision, Recall, and F1-score.

## 10. Quick Revision Flow
`Load ecg.csv (header=None) -> Split train/test -> Filter normal (y == 1) -> Scale with MinMaxScaler -> Build Autoencoder (140 -> 32 -> 16 -> 32 -> 140) -> Fit (20 epochs) -> Calculate threshold (mean + std) -> Flag errors > threshold as anomaly -> Invert test labels -> Evaluate metrics`
