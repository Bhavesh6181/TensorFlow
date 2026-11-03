# Practical 1E – FNN using Sign Language MNIST

## 1. Objective
The goal is to build, train, and evaluate a Feedforward Neural Network to recognize American Sign Language (ASL) hand gestures representing 24 alphabetical letters from 28x28 grayscale images.

## 2. Dataset
- **Name:** Sign Language MNIST
- **Files Used:** `sign_mnist_train.csv` and `sign_mnist_test.csv`
- **Features:** 784 pixel values representing 28x28 images of hand gestures
- **Target:** Hand gestures for letters A to Y. Letters **J (label 9)** and **Z (label 25)** require motion and are excluded from the static image dataset, leaving **24 classes**.

## 3. Libraries Used
| Library | Why We Use It |
|---|---|
| `tensorflow` | Primary framework to build, compile, and train the deep learning model. |
| `pandas` | To load the train and test CSV datasets into dataframes. |
| `matplotlib.pyplot` | For plotting training vs. validation loss and accuracy curves. |
| `Sequential`, `Input`, `Flatten`, `Dense` | Model construction layers for building feedforward architecture. |

## 4. Important Functions and Code
- `train_data.drop("label", axis=1).values`: Extracts pixel matrices from class labels.
- `y_train[y_train > 9] -= 1`: Critical label correction step. Since letter J (index 9) is skipped, original labels go up to 24 with a gap. Subtracting 1 from labels greater than 9 creates a contiguous range from 0 to 23.
- `.astype("float32") / 255.0`: Normalizes grayscale pixel brightness to `[0.0, 1.0]`.
- `.reshape(-1, 28, 28)`: Reconstructs 2D image matrices from 784-element arrays.
- `Flatten()`: Unrolls 2D image inputs into 1D vectors for fully connected layers.
- `Dense(128, activation="relu")` & `Dense(64, activation="relu")`: Hidden layers extracting hand shape features.
- `Dense(24, activation="softmax")`: Output layer with 24 neurons corresponding to the 24 static sign alphabet classes.
- `model.compile(optimizer="adam", ...)`: Configures Adam optimizer and sparse categorical cross-entropy loss.

## 5. Important Parameters
- **Epochs (10):** Number of full training iterations.
- **Batch Size (32):** Mini-batch size for gradient descent updates.
- **Validation Split (0.1):** 10% of training data set aside to measure generalization.
- **Output Classes (24):** 24 distinct sign language gestures.
- **Optimizer (Adam):** Adaptive gradient descent optimizer.
- **Loss (`sparse_categorical_crossentropy`):** Loss calculation for integer targets (0 to 23).

## 6. Model Architecture
- **Input:** 28x28 grayscale image of hand gesture.
- **Flatten Layer:** Reshapes (28, 28) into 784 inputs.
- **Hidden Layer 1:** Dense layer with 128 neurons and ReLU activation.
- **Hidden Layer 2:** Dense layer with 64 neurons and ReLU activation.
- **Output Layer:** Dense layer with 24 neurons and Softmax activation.
- **Data Flow:** `(28, 28) -> Flatten(784) -> Dense(128) -> Dense(64) -> Dense(24)`

## 7. Preprocessing Steps
- **Label Remapping (`y[y > 9] -= 1`):** Essential fix. Since gesture J (9) is missing, labels originally spanned 0 to 24 with a missing index. Shifting labels creates contiguous integers 0–23 matching the 24 output units.
- **Normalization (/255.0):** Scales values to `[0, 1]` for numerical stability during gradient calculations.
- **Reshaping:** Transforms 784 flat values into 28x28 matrices.

## 8. Expected Output
- Printed training/test shapes and label range confirmation (`0 to 23`).
- 10 epochs of training showing convergence of loss and accuracy.
- Test loss and test accuracy metrics.
- Side-by-side Matplotlib plots of loss and accuracy curves.

## 9. Viva Questions and Answers
**Q1. Why does the dataset have 24 classes instead of 26?**
**Ans:** The letters J and Z require hand motion/gestures, so they cannot be represented in static images and are excluded from the dataset.

**Q2. Why is `y_train[y_train > 9] -= 1` applied to the labels?**
**Ans:** Because letter J (label 9) is absent, leaving a gap. Subtracting 1 from labels above 9 maps them into contiguous indices from 0 to 23 for proper cross-entropy calculation.

**Q3. What would happen if we set the final Dense layer to 26 neurons without label remapping?**
**Ans:** The model would have unused output nodes for missing classes, wasting parameters and degrading probability calibration.

**Q4. What is the dimension of each image in Sign Language MNIST?**
**Ans:** 28x28 grayscale image (784 total pixels).

**Q5. Why is Adam chosen as the optimizer here?**
**Ans:** Adam adapts the learning rate per parameter, enabling faster and smoother training on diverse hand shape variations.

**Q6. What does `Flatten()` do?**
**Ans:** Transforms the 2D (28, 28) hand image matrix into a 1D vector of 784 features for Dense layers.

**Q7. What is the final layer activation and why?**
**Ans:** Softmax, because this is a multi-class single-label classification problem with 24 mutually exclusive classes.

**Q8. What batch size and number of epochs are used in this code?**
**Ans:** Batch size is 32 and epochs is 10.

**Q9. Why do we normalize features by dividing by 255.0?**
**Ans:** To rescale 8-bit integers to `[0, 1]`, preventing saturated neurons and gradient explosion.

**Q10. How is this practical different from MNIST digit classification?**
**Ans:** It classifies human hand gestures into 24 alphabet classes with label remapping, rather than 10 digits.

## 10. Quick Revision Flow
`Load train & test CSVs -> Separate X and y -> Remap labels (y > 9 -= 1) -> Normalize (/255.0) -> Reshape to 28x28 -> Define FNN (Flatten -> Dense 128 -> Dense 64 -> Dense 24) -> Compile (Adam) -> Fit (10 epochs) -> Evaluate -> Plot curves`
