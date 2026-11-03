# Practical 3F – Autoencoder for Sign Language MNIST

## 1. Objective
The goal is to implement an Autoencoder for anomaly detection on Sign Language MNIST images by training the model on 24 normal hand gesture classes and detecting gestures from an unseen class (Class 24) as anomalies.

## 2. Dataset
- **Name:** Sign Language MNIST (`sign_mnist_train.csv` and `sign_mnist_test.csv`)
- **Features:** 784 pixel values representing 28x28 grayscale images of hand gestures
- **Target Setup:**
  - Classes 0 to 23 are treated as **Normal** gesture classes.
  - Class 24 is designated as the **Anomaly** class.

## 3. Libraries Used
| Library | Why We Use It |
|---|---|
| `pandas` | For loading CSV train and test gesture data into DataFrames. |
| `numpy` | For array manipulations and computing mean squared reconstruction errors. |
| `matplotlib.pyplot` | For plotting training and validation loss curves over epochs. |
| `accuracy_score`, `confusion_matrix`, `classification_report` (sklearn) | For evaluating anomaly detection performance against ground truth labels. |
| `Input`, `Dense`, `Model` (tensorflow.keras) | For defining the functional Autoencoder network. |

## 4. Important Functions and Code
- `X_train = X_train[y_train != 24]`: Filters the training set to exclude Class 24 entirely. The Autoencoder only learns the visual patterns of normal classes (0–23).
- `X_train / 255.0`: Normalizes pixel values into `[0.0, 1.0]`.
- `Dense(16, activation="relu")` -> `Dense(8, activation="relu")`: **Encoder** compresses 784 pixels into a compact 8-dimensional bottleneck.
- `Dense(16, activation="relu")` -> `Dense(784, activation="sigmoid")`: **Decoder** reconstructs the 784 image pixels from the 8-dimensional code.
- `threshold = np.mean(train_errors) + np.std(train_errors)`: Threshold cutoff based on training reconstruction error distribution.
- `y_actual = (y_test == 24).astype(int)`: Converts test labels so that Class 24 is marked as the anomaly (`1`) and all other classes are marked as normal (`0`).
- `anomalies = (errors > threshold).astype(int)`: Test images with reconstruction error above the threshold are classified as anomalies.

## 5. Important Parameters
- **Epochs (20):** Training passes.
- **Batch Size (512):** Large mini-batch size for smooth gradient updates.
- **Bottleneck Size (8):** 8 latent units compressing the 784-pixel image by nearly 99%, forcing the model to capture only primary gesture shapes.
- **Optimizer (Adam):** Efficient adaptive learning rate optimizer.
- **Loss (MSE):** Mean Squared Error measuring pixel-wise reconstruction discrepancy.
- **Output Activation (Sigmoid):** Constrains reconstructed pixel outputs to `[0, 1]`, matching normalized input pixels.

## 6. Model Architecture
- **Input:** 784 numerical pixel inputs (flattened 28x28 image).
- **Encoder:**
  - Dense(16, ReLU)
  - Dense(8, ReLU) — Bottleneck representation
- **Decoder:**
  - Dense(16, ReLU)
  - Dense(784, Sigmoid) — Reconstructed image
- **Data Flow:** `784 -> 16 -> 8 (Bottleneck) -> 16 -> 784 (Reconstruction)`

## 7. Preprocessing Steps
- **Feature/Label Separation:** Drops the label column to extract 784-pixel vectors.
- **Normalization (/255.0):** Rescales pixel values from `[0, 255]` to `[0, 1]`.
- **Anomaly Exclusion:** Drops all samples belonging to Class 24 from the training set, reserving them exclusively for test-time anomaly evaluation.

## 8. Expected Output
- Initial training data inspection (head and shape).
- 20 epochs logging decreasing MSE reconstruction loss.
- Anomaly threshold value printed.
- Number of detected gesture anomalies.
- Classification accuracy, Confusion Matrix, and Classification Report.
- Training vs. Validation Loss plot.

## 9. Viva Questions and Answers
**Q1. How is anomaly detection formulated in this practical?**
**Ans:** One specific class (Class 24) is held out as an "unseen anomaly." The Autoencoder is trained only on the other 23 classes (0–23). At test time, samples from Class 24 produce high reconstruction errors and are flagged as anomalies.

**Q2. Why does the model produce high reconstruction error for Class 24?**
**Ans:** The Autoencoder never saw the hand orientation, finger configuration, or shape of Class 24 during training. Its compressed latent space cannot represent or reconstruct this unfamiliar pattern.

**Q3. What is the bottleneck dimension?**
**Ans:** 8 neurons.

**Q4. What is the compression ratio of this autoencoder?**
**Ans:** From 784 input pixels down to 8 bottleneck neurons — a compression ratio of nearly 98:1.

**Q5. Why is Sigmoid used in the output layer?**
**Ans:** Pixel intensities are normalized to `[0, 1]`. Sigmoid naturally constrains reconstructed values to the exact same range `[0, 1]`.

**Q6. What does `y_actual = (y_test == 24).astype(int)` accomplish?**
**Ans:** It creates binary evaluation labels where Class 24 is assigned label 1 (anomaly) and all other classes are assigned label 0 (normal).

**Q7. What threshold formula is used?**
**Ans:** `Threshold = Mean(train errors) + Standard Deviation(train errors)`.

**Q8. Why is batch size set to 512?**
**Ans:** It accelerates training on tens of thousands of image samples while providing stable gradient estimates.

**Q9. How does an Autoencoder differ from a standard classifier?**
**Ans:** A classifier learns decision boundaries between known labeled classes ($X ightarrow Y$). An Autoencoder learns the underlying distribution of normal data ($X ightarrow X$) without needing target class labels during training.

**Q10. What loss function is minimized?**
**Ans:** Mean Squared Error (MSE), measuring the average squared difference between original and reconstructed pixel values.

## 10. Quick Revision Flow
`Load train & test CSVs -> Separate X and y -> Normalize (/255.0) -> Filter out Class 24 from training set -> Build Autoencoder (784 -> 16 -> 8 -> 16 -> 784) -> Train on normal classes (20 epochs) -> Calculate threshold (mean + std) -> Evaluate on test data (flag error > threshold as Class 24 anomaly) -> Print metrics & plot loss`
