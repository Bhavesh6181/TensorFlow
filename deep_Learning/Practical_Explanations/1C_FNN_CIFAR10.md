# Practical 1C – FNN using CIFAR-10

## 1. Objective
The goal is to implement and evaluate a Feedforward Neural Network to classify 32x32 color (RGB) images into 10 real-world object categories from the CIFAR-10 dataset.

## 2. Dataset
- **Name:** CIFAR-10 (Canadian Institute For Advanced Research)
- **File Used:** `cifar-10.csv`
- **Features:** 3072 pixel values per image (32 width x 32 height x 3 color channels: Red, Green, Blue)
- **Target:** Column `label` containing integers 0 to 9 representing 10 object classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

## 3. Libraries Used
| Library | Why We Use It |
|---|---|
| `pandas` | For loading tabular CSV data and inspecting shapes. |
| `numpy` | For numerical operations and array handling. |
| `tensorflow` | Deep learning library used to construct and train the neural network. |
| `matplotlib.pyplot` | For plotting training and validation accuracy/loss trends. |
| `train_test_split` (sklearn) | Splits single CSV dataset into 80% training and 20% testing sets with label stratification. |
| `Sequential`, `Input`, `Flatten`, `Dense` | Keras components to build sequential dense networks. |

## 4. Important Functions and Code
- `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`: Splits data while preserving the equal proportion of each class in both sets.
- `x_train.astype("float32") / 255.0`: Normalizes RGB pixel intensity values from `[0, 255]` to `[0.0, 1.0]`.
- `.reshape(-1, 32, 32, 3)`: Reshapes flat 3072 features into 3D image format (32 height, 32 width, 3 color channels).
- `Flatten()`: Flattens 32x32x3 (3072 values) into a 1D vector for the first dense layer.
- `Dense(512, activation="relu")` & `Dense(256, activation="relu")`: Large dense layers with 512 and 256 units to handle higher dimensionality of color images.
- `Dense(10, activation="softmax")`: Computes classification probabilities for the 10 object classes.
- `model.compile(optimizer="adam", ...)`: Uses the adaptive learning rate Adam optimizer.

## 5. Important Parameters
- **Epochs (10):** Number of complete training cycles.
- **Batch Size (64):** Larger batch size suitable for larger input dimensions (3072).
- **Validation Split (0.1):** 10% of training data reserved for validation.
- **Optimizer (Adam):** Adaptive Moment Estimation, adjusts learning rates dynamically per parameter.
- **Loss (`sparse_categorical_crossentropy`):** Evaluates error against integer class labels (0 to 9).
- **Input Shape (32, 32, 3):** Corresponds to 32x32 pixels with 3 RGB color channels.

## 6. Model Architecture
- **Input Layer:** (32, 32, 3) color image.
- **Flatten Layer:** Unrolls (32, 32, 3) into 3072 linear inputs.
- **Hidden Layer 1:** Dense layer with 512 neurons, ReLU activation.
- **Hidden Layer 2:** Dense layer with 256 neurons, ReLU activation.
- **Output Layer:** Dense layer with 10 neurons, Softmax activation.
- **Data Flow:** `(32, 32, 3) -> Flatten(3072) -> Dense(512) -> Dense(256) -> Dense(10)`

## 7. Preprocessing Steps
- **Stratified Train-Test Split:** Ensures train and test sets have balanced distributions of all 10 classes.
- **Normalization (/255.0):** Rescales pixel values to `[0, 1]` to stabilize weight adjustments.
- **Reshaping:** Converts flat 3072 columns into 32x32x3 color images.

## 8. Expected Output
- Data shapes printed before and after reshaping (e.g., training shape `(-1, 32, 32, 3)`).
- 10 epoch training progression displaying loss and accuracy.
- Test loss and test accuracy printed.
- Dual plots displaying training/validation loss and training/validation accuracy across epochs.

## 9. Viva Questions and Answers
**Q1. Why does CIFAR-10 have 3072 features per image?**
**Ans:** Because each image is 32 pixels wide, 32 pixels high, and has 3 color channels (Red, Green, Blue): `32 * 32 * 3 = 3072`.

**Q2. Why is Adam optimizer preferred over SGD here?**
**Ans:** Adam adapts individual learning rates for each parameter using momentum and squared gradients, which converges faster on complex multidimensional data like CIFAR-10.

**Q3. What does `stratify=y` do in `train_test_split`?**
**Ans:** It ensures the train and test splits contain the exact same percentage of each class as the original dataset.

**Q4. Why are the hidden layers larger (512 and 256 neurons) compared to MNIST?**
**Ans:** CIFAR-10 is a color image dataset with 3072 inputs and complex real-world objects, requiring more network capacity to learn patterns.

**Q5. Why does an FNN achieve lower accuracy on CIFAR-10 compared to MNIST?**
**Ans:** FNN ignores spatial pixel hierarchy and color geometry. Real objects have variations in angle, lighting, and position that CNNs handle far better.

**Q6. What does `Input(shape=(32, 32, 3))` specify?**
**Ans:** It explicitly defines the height (32), width (32), and color channels (3) of incoming samples.

**Q7. What is the batch size used in this practical?**
**Ans:** 64.

**Q8. What loss function is used and why?**
**Ans:** `sparse_categorical_crossentropy` because target labels are integer encoded (0 through 9).

**Q9. What activation function is in the final layer?**
**Ans:** Softmax, to give normalized probabilities for the 10 classes.

**Q10. What are the 10 classes in CIFAR-10?**
**Ans:** Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.

## 10. Quick Revision Flow
`Load cifar-10.csv -> Split features & label -> train_test_split (stratify) -> Normalize (/255.0) -> Reshape to (32, 32, 3) -> Sequential (Flatten -> Dense 512 -> Dense 256 -> Dense 10) -> Compile (Adam) -> Fit (10 epochs, batch 64) -> Evaluate -> Plot results`
