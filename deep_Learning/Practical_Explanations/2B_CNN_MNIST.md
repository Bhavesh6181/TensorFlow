# Practical 2B – CNN using MNIST

## 1. Objective
The goal is to implement and train a Convolutional Neural Network (CNN) to classify 28x28 grayscale handwritten digits (0 to 9) from the MNIST dataset.

## 2. Dataset
- **Name:** MNIST Handwritten Digits
- **Files Used:** `mnist_train.csv` and `mnist_test.csv`
- **Features:** 784 pixel values representing 28x28 grayscale images
- **Target:** Column `label` with values 0 through 9 (10 digit classes)

## 3. Libraries Used
| Library | Why We Use It |
|---|---|
| `tensorflow` | Deep learning framework to construct, compile, and fit the CNN model. |
| `pandas` | To read the train and test CSV datasets into tabular dataframes. |
| `numpy` | For numerical array manipulation. |
| `matplotlib.pyplot` | To plot training and validation accuracy and loss over training epochs. |
| `Sequential` | Container to stack layers sequentially. |
| `Input`, `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense` | Core CNN layers: `Conv2D` applies convolution filters, `MaxPooling2D` downsamples, `Flatten` converts to 1D, `Dense` performs classification. |

## 4. Important Functions and Code
- `.reshape(-1, 28, 28, 1)`: Reshapes flat 784-element arrays into 4D image tensors `(batch_size, height=28, width=28, channels=1)`. The trailing 1 explicitly specifies a single grayscale channel required by Conv2D.
- `.astype("float32") / 255.0`: Scales pixel intensities from `[0, 255]` to `[0.0, 1.0]`.
- `Conv2D(32, (3, 3), activation="relu")`: First convolutional layer with 32 filters of size 3x3 to detect initial low-level edges. Output size becomes 26x26 (without padding).
- `MaxPooling2D((2, 2))`: Downsamples 26x26 feature maps to 13x13 by retaining maximum values in 2x2 windows.
- `Conv2D(64, (3, 3), activation="relu")`: Second convolutional layer with 64 filters of size 3x3, producing 11x11 feature maps.
- Second `MaxPooling2D((2, 2))`: Downsamples 11x11 feature maps to 5x5.
- `Flatten()`: Unrolls 5x5x64 tensor into 1600 linear features.
- `Dense(64, activation="relu")`: Fully connected layer learning high-level representations.
- `Dense(10, activation="softmax")`: Final layer outputting class probability scores for digits 0 to 9.

## 5. Important Parameters
- **Epochs (5):** 5 epochs are sufficient for MNIST because digits are simple and CNNs achieve 98%+ accuracy quickly.
- **Batch Size (32):** Standard mini-batch size.
- **Validation Split (0.1):** 10% of training data reserved to monitor generalization.
- **Input Shape (28, 28, 1):** Specifies height, width, and single grayscale channel.
- **Optimizer (Adam):** Adaptive gradient descent optimizer.
- **Loss (`sparse_categorical_crossentropy`):** Standard loss for integer class labels.

## 6. Model Architecture
- **Input:** (28, 28, 1) grayscale image.
- **Conv Block 1:** Conv2D (32 filters, 3x3, valid padding) + ReLU -> MaxPooling2D (2x2) [output: 13x13x32]
- **Conv Block 2:** Conv2D (64 filters, 3x3, valid padding) + ReLU -> MaxPooling2D (2x2) [output: 5x5x64]
- **Flatten Layer:** Reshapes (5, 5, 64) -> 1600 elements.
- **Dense Layer:** 64 neurons, ReLU activation.
- **Output Layer:** 10 neurons, Softmax activation.
- **Data Flow:** `(28, 28, 1) -> Conv2D(32) -> MaxPool(13, 13, 32) -> Conv2D(64) -> MaxPool(5, 5, 64) -> Flatten(1600) -> Dense(64) -> Dense(10)`

## 7. Preprocessing Steps
- **Reshaping with Channel Dimension (`reshape(-1, 28, 28, 1)`):** Keras `Conv2D` requires a 4D tensor `(samples, height, width, channels)`. The 1 indicates grayscale.
- **Normalization (/255.0):** Rescales pixel values to `[0, 1]` for numerical stability and faster gradient descent convergence.

## 8. Expected Output
- Model training output over 5 epochs with test accuracy typically exceeding 98%.
- Printed test loss and test accuracy.
- Two visual graphs: Training vs. Validation Accuracy, and Training vs. Validation Loss.

## 9. Viva Questions and Answers
**Q1. Why do we reshape to `(-1, 28, 28, 1)` instead of `(-1, 28, 28)`?**
**Ans:** `Conv2D` layers strictly require a 4D input tensor including the channel dimension. For grayscale images, the channel dimension is 1.

**Q2. Why does the feature map shrink from 28x28 to 26x26 in the first Conv2D?**
**Ans:** Because default padding is `"valid"` (no padding). When a 3x3 filter slides over 28x28 without padding: `28 - 3 + 1 = 26`.

**Q3. Why are only 5 epochs needed for MNIST CNN?**
**Ans:** MNIST images are simple high-contrast grayscale patterns; a CNN achieves high accuracy (98%+) within 3 to 5 epochs.

**Q4. What is the output shape before the Flatten layer?**
**Ans:** `(5, 5, 64)`, which flattens to `5 * 5 * 64 = 1600` features.

**Q5. Why is Max Pooling used instead of Average Pooling?**
**Ans:** Max pooling extracts the most prominent features (edges, peaks) and is less sensitive to background noise compared to average pooling.

**Q6. What optimizer and loss function are used?**
**Ans:** Adam optimizer and `sparse_categorical_crossentropy` loss.

**Q7. What does the final Dense layer with Softmax output?**
**Ans:** A probability distribution of 10 values summing to 1, indicating the likelihood of each digit (0 to 9).

**Q8. How does a CNN avoid overfitting compared to an FNN with equal parameters?**
**Ans:** CNNs use weight sharing (same filter applied across all spatial positions) and local receptive fields, dramatically reducing total parameter count.

**Q9. What is the batch size in this practical?**
**Ans:** 32.

**Q10. How does CNN performance on MNIST compare to FNN performance on MNIST (Practical 1A)?**
**Ans:** CNN achieves higher accuracy (~98-99% vs ~95-97% for FNN) because it recognizes local digit strokes and curves regardless of minor spatial shifts.

## 10. Quick Revision Flow
`Load MNIST CSVs -> Separate X and y -> Reshape to (28, 28, 1) -> Normalize (/255.0) -> Sequential (Conv2D 32 -> MaxPool -> Conv2D 64 -> MaxPool -> Flatten -> Dense 64 -> Dense 10) -> Compile (Adam) -> Fit (5 epochs) -> Evaluate -> Plot graphs`
