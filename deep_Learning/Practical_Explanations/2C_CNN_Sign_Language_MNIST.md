# Practical 2C – CNN using Sign Language MNIST

## 1. Objective
The goal is to design, implement, and train a Convolutional Neural Network (CNN) to recognize 24 American Sign Language (ASL) alphabet hand gestures from 28x28 grayscale images.

## 2. Dataset
- **Name:** Sign Language MNIST
- **Files Used:** `sign_mnist_train.csv` and `sign_mnist_test.csv`
- **Features:** 784 pixel values representing 28x28 grayscale images of hand gestures
- **Target:** Column `label` representing alphabet letters. Letters **J (label 9)** and **Z (label 25)** require motion and are excluded, leaving **24 classes**.

## 3. Libraries Used
| Library | Why We Use It |
|---|---|
| `tensorflow` | Deep learning framework to build and train the convolutional neural network. |
| `pandas` | For reading the train and test CSV files into tabular structures. |
| `matplotlib.pyplot` | To plot training and validation accuracy and loss curves over epochs. |
| `Sequential` | Container to stack layers sequentially. |
| `Input`, `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense` | Fundamental CNN modules: convolution for feature maps, max pooling for downsampling, flatten to convert to 1D, and dense for classification. |

## 4. Important Functions and Code
- `y_train[y_train > 9] -= 1`: Remaps labels above 9 by subtracting 1, closing the gap caused by the missing letter J and mapping labels into a continuous range from 0 to 23.
- `.reshape(-1, 28, 28, 1)`: Reshapes flat 784 pixels into a 4D tensor with 1 grayscale channel for `Conv2D`.
- `.astype("float32") / 255.0`: Scales pixel brightness from `[0, 255]` to `[0.0, 1.0]`.
- `Conv2D(32, (3, 3), activation="relu")` + `MaxPooling2D((2, 2))`: First convolutional stage detecting edges and finger contours, downsampling feature maps to 13x13.
- `Conv2D(64, (3, 3), activation="relu")` + `MaxPooling2D((2, 2))`: Second convolutional stage capturing complex hand configurations, downsampling feature maps to 5x5.
- `Flatten()`: Unrolls 5x5x64 tensor into 1600 linear features.
- `Dense(64, activation="relu")`: Dense feature layer.
- `Dense(24, activation="softmax")`: Classification layer with 24 neurons corresponding to the 24 sign language gestures.
- `model.summary()`: Prints layer-by-layer output shapes and parameter counts.

## 5. Important Parameters
- **Epochs (10):** Total training cycles.
- **Batch Size (32):** Mini-batch size.
- **Validation Split (0.1):** 10% held-out training data to validate generalization.
- **Output Units (24):** 24 distinct sign language categories.
- **Optimizer (Adam):** Efficient adaptive learning rate optimizer.
- **Loss (`sparse_categorical_crossentropy`):** Multi-class cross-entropy for integer labels.

## 6. Model Architecture
- **Input:** (28, 28, 1) grayscale image of hand gesture.
- **Conv Block 1:** Conv2D (32 filters, 3x3, valid padding) + ReLU -> MaxPooling2D (2x2) [output: 13x13x32]
- **Conv Block 2:** Conv2D (64 filters, 3x3, valid padding) + ReLU -> MaxPooling2D (2x2) [output: 5x5x64]
- **Flatten Layer:** Reshapes (5, 5, 64) -> 1600 elements.
- **Dense Layer:** 64 neurons, ReLU activation.
- **Output Layer:** 24 neurons, Softmax activation.
- **Data Flow:** `(28, 28, 1) -> Conv2D(32) -> MaxPool(13, 13, 32) -> Conv2D(64) -> MaxPool(5, 5, 64) -> Flatten(1600) -> Dense(64) -> Dense(24)`

## 7. Preprocessing Steps
- **Label Remapping (`y[y > 9] -= 1`):** Removes the index gap from the missing letter J so that target classes are strictly integers from 0 to 23.
- **Adding Channel Dimension (`reshape(-1, 28, 28, 1)`):** Creates the 4D input tensor required by Keras `Conv2D`.
- **Normalization (/255.0):** Rescales pixel values to `[0, 1]` to ensure stable gradients during backpropagation.

## 8. Expected Output
- Model architecture summary showing layers and trainable parameters.
- Training progression across 10 epochs showing loss reduction and accuracy improvement.
- Final test loss and test accuracy.
- Two plots: Training vs. Validation Accuracy, and Training vs. Validation Loss.

## 9. Viva Questions and Answers
**Q1. Why are there 24 output units instead of 26?**
**Ans:** The letters J and Z involve motion and are excluded from this static image dataset.

**Q2. Why is label remapping (`y[y > 9] -= 1`) necessary?**
**Ans:** Letter J (label 9) is absent. Without remapping, labels would skip 9 and run from 0 to 24 (25 indices), causing errors with 24 output units.

**Q3. Why is CNN preferred over FNN for Sign Language recognition?**
**Ans:** Hand gestures depend heavily on spatial finger positions, orientations, and contours. CNNs detect these spatial features effectively, whereas FNNs flatten the image and lose spatial relationships.

**Q4. What is the receptive field of a 3x3 filter?**
**Ans:** It inspects a 3x3 local patch of pixels at a time to identify patterns like edges, curves, and corners.

**Q5. What is the role of `MaxPooling2D((2, 2))`?**
**Ans:** It halves the spatial dimensions (height and width), reducing computation and giving slight translation invariance to hand position variations.

**Q6. What does `model.summary()` display?**
**Ans:** Layer names, layer types, output shapes at each step, and total/trainable parameter counts.

**Q7. What optimizer is used in this practical?**
**Ans:** Adam optimizer (`optimizer="adam"`).

**Q8. What is the shape of the tensor entering the Flatten layer?**
**Ans:** `(5, 5, 64)`, which flattens into 1600 inputs for the Dense layer.

**Q9. What loss function is used?**
**Ans:** `sparse_categorical_crossentropy` because class labels are integers (0 to 23).

**Q10. How does CNN performance in 2C compare to FNN in 1E on the same dataset?**
**Ans:** CNN achieves substantially higher accuracy because convolutions preserve hand anatomy and finger contours that FNN loses upon flattening.

## 10. Quick Revision Flow
`Load Sign MNIST CSVs -> Separate features & labels -> Remap labels (y > 9 -= 1) -> Normalize (/255.0) -> Reshape to (28, 28, 1) -> Sequential (Conv2D 32 -> MaxPool -> Conv2D 64 -> MaxPool -> Flatten -> Dense 64 -> Dense 24) -> Compile (Adam) -> Fit (10 epochs) -> Evaluate -> Plot results`
