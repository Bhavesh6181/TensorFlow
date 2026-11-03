# Practical 3E – Autoencoder for NSL-KDD

## 1. Objective
The goal is to build an Autoencoder using the Keras Functional API to detect network intrusions and cyber attacks from network connection records in the NSL-KDD dataset.

## 2. Dataset
- **Name:** NSL-KDD Network Intrusion Dataset (`KDDTrain+.TXT` and `KDDTest+.TXT`)
- **Columns (43):** 41 connection features, 1 `label` (e.g., normal, dos, probe, r2l, u2r), and 1 `difficulty` score
- **Feature Types:**
  - **Categorical (3):** `protocol_type` (tcp, udp, icmp), `service` (http, smtp, ftp, etc.), `flag` (SF, S0, REJ, etc.)
  - **Numerical (38):** `duration`, `src_bytes`, `dst_bytes`, `count`, error rates, etc.
- **Target:** Converted to binary: `0 = Normal traffic`, `1 = Network attack`

## 3. Libraries Used
| Library | Why We Use It |
|---|---|
| `pandas` | For reading structured text datasets with custom assigned column names. |
| `numpy` | For numerical operations and percentile-based threshold calculation. |
| `matplotlib.pyplot` | For plotting training loss curves and reconstruction error scatter/line plots. |
| `StandardScaler` (sklearn) | Standardizes numerical features by removing the mean and scaling to unit variance. |
| `OneHotEncoder` (sklearn) | Converts categorical text features (`protocol_type`, `service`, `flag`) into binary one-hot vectors. |
| `ColumnTransformer` (sklearn) | Applies `StandardScaler` to numerical columns and `OneHotEncoder` to categorical columns in a single pipeline. |
| `train_test_split` (sklearn) | Splits normal training data into training and validation sets. |
| `confusion_matrix`, `classification_report` (sklearn) | Measures intrusion detection precision, recall, and F1-score on test traffic. |
| `Input`, `Dense`, `Model` | Keras Functional API for creating deep symmetric autoencoder architectures. |

## 4. Important Functions and Code
- `pd.read_csv("...", names=columns)`: NSL-KDD files do not have header rows, so explicit column names are provided during loading.
- `y_test = (y_test != "normal").astype(int)`: Binary mapping where attacks are labelled `1` and normal traffic is labelled `0`.
- `ColumnTransformer([("num", StandardScaler(), numerical), ("cat", OneHotEncoder(sparse_output=False), categorical)])`: Simultaneously standardizes continuous variables and one-hot encodes categorical protocol attributes.
- `preprocessor.fit_transform(X_train)`: Fits scalers/encoders strictly on normal training data and transforms validation and test data.
- **Functional API Autoencoder:**
  - `encoded = Dense(64, relu) -> Dense(32, relu) -> Dense(16, relu)` (**Encoder**)
  - `decoded = Dense(32, relu) -> Dense(64, relu) -> Dense(n_features, linear)` (**Decoder**)
- `threshold = np.percentile(val_errors, 95)`: Percentile-based threshold. Flags the top 5% of validation errors as the cutoff boundary for attack detection.
- `y_pred = (test_errors > threshold).astype(int)`: Identifies test connections with reconstruction errors above the 95th percentile as network attacks.

## 5. Important Parameters
- **Epochs (10):** Training passes.
- **Batch Size (32):** Mini-batch size.
- **Latent Bottleneck (16):** Highly compressed representation of normal network traffic.
- **Output Activation (`linear`):** Linear activation because numerical features are standardized with `StandardScaler` (can be negative or greater than 1).
- **Threshold (95th Percentile):** Statistically separates the top 5% highest normal reconstruction errors.
- **Optimizer (Adam) & Loss (MSE):** Standard configuration for regression-based autoencoders.

## 6. Model Architecture
- **Input:** Standardized & one-hot encoded network connection features.
- **Encoder:**
  - Dense(64, ReLU)
  - Dense(32, ReLU)
  - Dense(16, ReLU) — Bottleneck
- **Decoder:**
  - Dense(32, ReLU)
  - Dense(64, ReLU)
  - Dense(n_features, Linear) — Output
- **Data Flow:** `Input -> Dense(64) -> Dense(32) -> Dense(16) -> Dense(32) -> Dense(64) -> Output(Linear)`

## 7. Preprocessing Steps
- **Label Binarization:** Strips trailing dots/whitespace and converts multi-class attacks into binary: `0 = Normal`, `1 = Attack`.
- **Filtering Normal Traffic:** Training set retains only connections where `label == "normal"`.
- **ColumnTransformer:**
  - `StandardScaler` applied to 38 continuous numerical features.
  - `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` applied to 3 categorical columns (`protocol_type`, `service`, `flag`).

## 8. Expected Output
- Initial training data head and shape.
- 10 training epochs logging training and validation MSE loss.
- 95th percentile threshold value printed.
- Confusion Matrix and Classification Report (Normal vs Attack).
- Plot 1: Autoencoder training and validation loss curve.
- Plot 2: Test sample reconstruction error plot with a red dashed horizontal threshold line.

## 9. Viva Questions and Answers
**Q1. What is the NSL-KDD dataset used for?**
**Ans:** It is a standard cybersecurity benchmark dataset used to evaluate network intrusion detection systems (NIDS).

**Q2. Why is ColumnTransformer used?**
**Ans:** To handle mixed data types in a single pipeline: applying `StandardScaler` to numerical columns and `OneHotEncoder` to categorical text columns simultaneously.

**Q3. What are the 3 categorical features in NSL-KDD?**
**Ans:** `protocol_type` (e.g., tcp, udp), `service` (e.g., http, ftp), and `flag` (connection status flag).

**Q4. Why does the output layer use a `linear` activation instead of `sigmoid`?**
**Ans:** Because features were scaled using `StandardScaler` (mean=0, variance=1), which produces both positive and negative values. A sigmoid function would restrict outputs to `[0, 1]`, failing to reconstruct negative standardized values.

**Q5. Why is the threshold chosen as the 95th percentile of validation errors?**
**Ans:** The 95th percentile sets a 5% false-alarm tolerance on normal validation traffic, providing a statistically sound cutoff boundary to flag unusual connection behaviors.

**Q6. What happens when an attack connection passes through this autoencoder?**
**Ans:** Attack traffic possesses unusual byte counts, connection rates, or flag patterns. The model fails to reconstruct these unfamiliar patterns, resulting in a large MSE error exceeding the threshold.

**Q7. What is the bottleneck dimension?**
**Ans:** 16 neurons.

**Q8. Why do we drop the `difficulty` column?**
**Ans:** The difficulty column is a metadata rating included in NSL-KDD indicating how difficult the record is for classifiers; it is not a network feature.

**Q9. What are the four main types of network attacks in NSL-KDD?**
**Ans:** DoS (Denial of Service), Probe (surveillance/scanning), R2L (Remote to Local unauthorized access), and U2R (User to Root privilege escalation).

**Q10. How does this model handle unseen categorical categories in test data?**
**Ans:** `handle_unknown="ignore"` in `OneHotEncoder` creates all-zero vectors for unseen categories, preventing runtime crashes.

## 10. Quick Revision Flow
`Define column names -> Load KDDTrain+ and KDDTest+ -> Binarize labels (normal vs attack) -> Filter normal training data -> ColumnTransformer (StandardScaler + OneHotEncoder) -> Build Functional Autoencoder (64 -> 32 -> 16 -> 32 -> 64 -> Linear) -> Fit on normal data -> Threshold at 95th percentile -> Flag errors > threshold as attack -> Evaluate`
